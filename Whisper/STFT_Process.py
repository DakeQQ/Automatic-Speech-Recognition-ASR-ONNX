#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production ONNX-exportable STFT / ISTFT module used by the Whisper exporter.

The implementation precomputes its numerical kernels as PyTorch buffers and
includes the packed-power path required by Whisper feature extraction.
"""

import torch

NFFT         = 400           # FFT size (number of frequency bins before folding)
WIN_LENGTH   = 400           # Analysis window length in samples (≤ NFFT)
HOP_LENGTH   = 160           # Hop (stride) between successive frames
WINDOW_TYPE  = 'hann'        # Window function: bartlett | blackman | hamming | hann | kaiser
CENTER_PAD   = True          # True  → pad signal so frame centres align with sample indices
                             # False → no padding, first frame starts at sample 0
PAD_MODE     = 'constant'    # Padding style when CENTER_PAD is True: 'reflect' | 'constant'
STFT_SIGNAL_LENGTH = 101

# -- Window function registry ----------------------------------------------
WINDOW_FUNCTIONS = {
    'bartlett':  lambda L: torch.bartlett_window(L, periodic=True),
    'blackman':  lambda L: torch.blackman_window(L, periodic=True),
    'hamming':   lambda L: torch.hamming_window(L,  periodic=True),
    'hann':      lambda L: torch.hann_window(L,     periodic=True),
    'hann_sqrt': lambda L: torch.hann_window(L,     periodic=False).pow(0.5),
    'povey':     lambda L: torch.hann_window(L,     periodic=False).pow(0.85),
    'kaiser':    lambda L: torch.kaiser_window(L,   periodic=True, beta=12.0)
}
DEFAULT_WINDOW_FN = lambda L: torch.hann_window(L, periodic=True)


def create_padded_window(win_length: int, n_fft: int, window_type: str) -> torch.Tensor:
    """Create a window of length *n_fft*, center-padding or cropping as needed."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    win = win_fn(win_length).float()

    if win_length == n_fft:
        return win
    if win_length < n_fft:
        pad_total = n_fft - win_length
        pad_left  = pad_total // 2
        pad_right = pad_total - pad_left
        return torch.cat([torch.zeros(pad_left), win, torch.zeros(pad_right)])
    start = (win_length - n_fft) // 2
    return win[start : start + n_fft]


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Optimized STFT / ISTFT Models (Static Graph)
# ═════════════════════════════════════════════════════════════════════════════

class STFT_Process(torch.nn.Module):
    """
    Static-graph Conv1d STFT / ConvTranspose1d ISTFT for ONNX export.

    All constants precomputed in __init__() as registered buffers.
    Forward path is pure tensor ops — no dispatch, no branching, no shape queries.

    Variants
    --------
    stft_A   → Conv1d producing real part only.
    stft_B   → Conv1d producing real + imag (split after convolution).
    stft_B_power → Conv1d producing power directly from packed real/imag channels.
    istft_A  → (magnitude, phase) → ConvTranspose1d reconstruction.
    istft_B  → (real, imag) → ConvTranspose1d reconstruction.
    """

    def __init__(
        self,
        model_type: str,
        n_fft: int       = NFFT,
        win_length: int  = WIN_LENGTH,
        hop_len: int     = HOP_LENGTH,
        max_frames: int  = STFT_SIGNAL_LENGTH,
        window_type: str = WINDOW_TYPE,
        center_pad: bool = CENTER_PAD,
        pad_mode: str    = PAD_MODE,
        input_scale: float = 1.0,
        drop_last_frame: bool = False,
    ):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.n_frames   = max_frames
        self.input_scale = float(input_scale)

        f_bins = self.half_n_fft + 1
        self.f_bins = f_bins
        window = create_padded_window(win_length, n_fft, window_type)

        if drop_last_frame:
            # A centered STFT emits floor(audio_len / hop) + 1 frames. Whisper always
            # discards the last frame. Reducing only the right reflection pad by one hop
            # makes Conv1d emit floor(audio_len / hop) frames directly for every length.
            self._right_pad = self.half_n_fft - hop_len
        else:
            self._right_pad = self.half_n_fft

        # ── Precompute static output slice bounds for ISTFT ───────────────
        raw_len = n_fft + hop_len * (max_frames - 1)
        if center_pad:
            self._out_start = self.half_n_fft
            self._out_end   = raw_len - self.half_n_fft
        else:
            self._out_start = 0
            self._out_end   = raw_len

        # ── Bind forward to the correct variant (no dispatch overhead) ────
        self.forward = getattr(self, f'_{model_type}_forward')

        # ── STFT: constant zero-padding buffer ────────────────────────────
        if model_type in ('stft_A', 'stft_B', 'stft_B_power'):
            kernel_type = 'stft_A' if model_type == 'stft_A' else 'stft_B'
            self._build_stft_kernels(n_fft, f_bins, window, kernel_type, input_scale)
            if center_pad and pad_mode == 'constant':
                self.register_buffer(
                    'padding_zero_left',
                    torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32)
                )
                self.register_buffer(
                    'padding_zero_right',
                    torch.zeros(1, 1, self._right_pad, dtype=torch.float32)
                )
            self._center_pad = center_pad
            self._pad_mode   = pad_mode

        # ── ISTFT: inverse kernel + pre-sliced normalization ──────────────
        if model_type in ('istft_A', 'istft_B'):
            self._build_istft_kernels(n_fft, f_bins, window, hop_len, max_frames)

    def _build_stft_kernels(self, n_fft, f_bins, window, model_type, input_scale):
        """Precompute windowed DFT basis as Conv1d kernel weights."""
        omega_factor = 2.0 * torch.pi / n_fft
        t = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        f = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        omega = omega_factor * f * t

        scaled_window = window * float(input_scale)
        windowed_cos = ( torch.cos(omega) * scaled_window.unsqueeze(0)).unsqueeze(1)
        windowed_sin = (-torch.sin(omega) * scaled_window.unsqueeze(0)).unsqueeze(1)

        if model_type == 'stft_A':
            self.register_buffer('stft_kernel', windowed_cos)
        else:
            self.register_buffer('stft_kernel', torch.cat([windowed_cos, windowed_sin], dim=0))

    def _build_istft_kernels(self, n_fft, f_bins, window, hop_len, n_frames):
        """Precompute inverse-DFT kernel and window² kernel for COLA normalization."""
        omega_factor = 2.0 * torch.pi / n_fft
        k = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        omega = omega_factor * k * n

        cos_basis = torch.cos(omega)
        sin_basis = torch.sin(omega)

        scale = 2.0 * torch.ones(f_bins, 1)
        scale[0] = 1.0
        if n_fft % 2 == 0:
            scale[f_bins - 1] = 1.0

        inv_n     = 1.0 / n_fft
        ifft_real = (scale *  cos_basis * inv_n) * window.unsqueeze(0)
        ifft_imag = (scale * -sin_basis * inv_n) * window.unsqueeze(0)

        self.register_buffer(
            'inverse_kernel',
            torch.cat([ifft_real, ifft_imag], dim=0).unsqueeze(1)
        )

        # Store window² kernel for dynamic COLA normalization in forward.
        self.register_buffer('win_sq_kernel', window.square().reshape(1, 1, -1))

    # --------------------------------------------------------------------- #
    #  STFT forward variants (no branching, static tensor ops only)         #
    # --------------------------------------------------------------------- #

    def _stft_A_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT producing real part only (cosine projection)."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                if self._right_pad:
                    right = x[..., -(self._right_pad + 1): -1].flip(2)
                    x = torch.cat([left, x, right], dim=2)
                else:
                    x = torch.cat([left, x], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero_left = torch.cat([self.padding_zero_left] * x.shape[0], dim=0)
                    padding_zero_right = torch.cat([self.padding_zero_right] * x.shape[0], dim=0)
                else:
                    padding_zero_left = self.padding_zero_left
                    padding_zero_right = self.padding_zero_right
                x = torch.cat([padding_zero_left, x, padding_zero_right], dim=2)
        return torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)

    def _stft_B_forward(self, x: torch.Tensor):
        """STFT producing (real, imag) via a single Conv1d + channel Split."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                if self._right_pad:
                    right = x[..., -(self._right_pad + 1): -1].flip(2)
                    x = torch.cat([left, x, right], dim=2)
                else:
                    x = torch.cat([left, x], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero_left = torch.cat([self.padding_zero_left] * x.shape[0], dim=0)
                    padding_zero_right = torch.cat([self.padding_zero_right] * x.shape[0], dim=0)
                else:
                    padding_zero_left = self.padding_zero_left
                    padding_zero_right = self.padding_zero_right
                x = torch.cat([padding_zero_left, x, padding_zero_right], dim=2)
        out = torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)
        return torch.split(out, self.half_n_fft + 1, dim=1)

    def _stft_B_power_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT power from one packed real/imag Conv1d without Split or duplicate Mul nodes."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left = x[..., 1: self.half_n_fft + 1].flip(2)
                if self._right_pad:
                    right = x[..., -(self._right_pad + 1): -1].flip(2)
                    x = torch.cat([left, x, right], dim=2)
                else:
                    x = torch.cat([left, x], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero_left = torch.cat([self.padding_zero_left] * x.shape[0], dim=0)
                    padding_zero_right = torch.cat([self.padding_zero_right] * x.shape[0], dim=0)
                else:
                    padding_zero_left = self.padding_zero_left
                    padding_zero_right = self.padding_zero_right
                x = torch.cat([padding_zero_left, x, padding_zero_right], dim=2)
        packed = torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)
        # Whisper's encoder batch is statically one; using fixed leading dimensions avoids
        # introducing Shape/Gather nodes solely to expose the packed complex channel axis.
        packed = packed.reshape(1, 2, self.f_bins, -1)
        return torch.sum(packed * packed, dim=1)

    # --------------------------------------------------------------------- #
    #  ISTFT forward variants (static slicing, no Shape/Gather ops)         #
    # --------------------------------------------------------------------- #

    def _istft_B_forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """ISTFT from rectangular form. Dynamic-length compatible."""
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)
        # Compute COLA normalization dynamically based on input n_frames.
        ones = torch.ones(1, 1, real.shape[2], dtype=real.dtype, device=real.device)
        win_sum = torch.nn.functional.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)
        inv = inv[..., self._out_start:self._out_end] / win_sum[..., self._out_start:self._out_end]
        return inv

    def _istft_A_forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """ISTFT from polar form. Dynamic-length compatible."""
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_kernel, stride=self.hop_len)
        # Compute COLA normalization dynamically based on input n_frames.
        ones = torch.ones(1, 1, magnitude.shape[2], dtype=magnitude.dtype, device=magnitude.device)
        win_sum = torch.nn.functional.conv_transpose1d(ones, self.win_sq_kernel, stride=self.hop_len)
        inv = inv[..., self._out_start:self._out_end] / win_sum[..., self._out_start:self._out_end]
        return inv

