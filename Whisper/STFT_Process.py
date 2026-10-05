#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production ONNX-exportable STFT / ISTFT module used by the Whisper exporter.

The implementation precomputes its numerical kernels as PyTorch buffers and
includes the packed-power path required by Whisper feature extraction.
"""

import torch
from torch.onnx import symbolic_helper

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

class ONNX_REFLECT_PAD_1D(torch.autograd.Function):
    """Emit one standard ONNX reflect Pad for a rank-3 waveform."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, left: int, right: int) -> torch.Tensor:
        return torch.nn.functional.pad(x, (left, right), mode="reflect")

    @staticmethod
    def symbolic(g, x, left, right):
        left_value = symbolic_helper._get_const(left, "i", "left")
        right_value = symbolic_helper._get_const(right, "i", "right")
        pads = g.op(
            "Constant",
            value_t=torch.tensor(
                [0, 0, left_value, 0, 0, right_value], dtype=torch.int64
            ),
        )
        return g.op("Pad", x, pads, mode_s="reflect")



class STFT_Process(torch.nn.Module):
    """
    Conv1d STFT / ConvTranspose1d ISTFT for ONNX export.

    Kernels are precomputed in __init__() as registered buffers.
    Set dynamic_frames=False to precompute ISTFT normalization for max_frames.

    Variants
    --------
    stft_A   → Conv1d producing real part only.
    stft_B   → Conv1d producing real + imag (split after convolution).
    stft_B_power -> Power from packed real/imag channels.
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
        dynamic_frames: bool = True,
    ):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.f_bins     = self.half_n_fft + 1
        self.n_frames   = max_frames
        self.drop_last_frame = bool(drop_last_frame)
        self.input_scale = float(input_scale)

        f_bins = self.f_bins
        window = create_padded_window(win_length, n_fft, window_type)

        self._out_start = self.half_n_fft if center_pad else 0

        if model_type == 'stft_A':
            self.forward = self._stft_A_forward
        elif model_type == 'stft_B':
            self.forward = self._stft_B_forward
        elif model_type == 'stft_B_power':
            self.forward = self._stft_B_power_forward
        elif model_type == 'istft_A':
            self.forward = self._istft_A_forward
        elif model_type == 'istft_B':
            self.forward = self._istft_B_forward
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_type in ('stft_A', 'stft_B', 'stft_B_power'):
            self._build_stft_kernels(
                n_fft, f_bins, window, model_type, float(input_scale)
            )
            self._reflect_pad = center_pad and pad_mode == 'reflect'
            self._conv_padding = self.half_n_fft if center_pad and not self._reflect_pad else 0
            self._pad_left = self.half_n_fft
            self._pad_right = self.half_n_fft
            self._drop_after_conv = self.drop_last_frame
            if self._reflect_pad and self.drop_last_frame:
                reduced_right = self.half_n_fft - self.hop_len
                if reduced_right >= 0:
                    self._pad_right = reduced_right
                    self._drop_after_conv = False

        if model_type in ('istft_A', 'istft_B'):
            self._build_istft_kernels(n_fft, f_bins, window, hop_len, max_frames, dynamic_frames)

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

    def _build_istft_kernels(self, n_fft, f_bins, window, hop_len, n_frames, dynamic_frames):
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

        win_sq_kernel = window.square().reshape(1, 1, -1)
        self._dynamic_frames = dynamic_frames
        if dynamic_frames:
            self.register_buffer('win_sq_kernel', win_sq_kernel)
        else:
            win_sum = torch.nn.functional.conv_transpose1d(
                torch.ones(1, 1, n_frames), win_sq_kernel, stride=hop_len, padding=self._out_start
            )
            self.register_buffer(
                'inv_win_sum', win_sum.clamp_min(1e-11).reciprocal()
            )

    # --------------------------------------------------------------------- #
    #  STFT forward variants (no branching, static tensor ops only)         #
    # --------------------------------------------------------------------- #

    def _stft_A_forward(self, x: torch.Tensor) -> torch.Tensor:
        """STFT producing real part only (cosine projection)."""
        if self._reflect_pad:
            x = ONNX_REFLECT_PAD_1D.apply(
                x, self._pad_left, self._pad_right
            )
        out = torch.nn.functional.conv1d(
            x, self.stft_kernel, stride=self.hop_len, padding=self._conv_padding
        )
        return out[..., :-1] if self._drop_after_conv else out

    def _stft_B_forward(self, x: torch.Tensor):
        """STFT producing (real, imag) via a single Conv1d + channel Split."""
        if self._reflect_pad:
            x = ONNX_REFLECT_PAD_1D.apply(
                x, self._pad_left, self._pad_right
            )
        out = torch.nn.functional.conv1d(
            x, self.stft_kernel, stride=self.hop_len, padding=self._conv_padding
        )
        if self._drop_after_conv:
            out = out[..., :-1]
        return torch.split(out, self.f_bins, dim=1)

    # --------------------------------------------------------------------- #
    #  ISTFT forward variants (static slicing, no Shape/Gather ops)         #
    # --------------------------------------------------------------------- #

    def _istft_B_forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """ISTFT from rectangular form. Dynamic-length compatible."""
        inp = torch.cat((real, imag), dim=1)
        inv = torch.nn.functional.conv_transpose1d(
            inp, self.inverse_kernel, stride=self.hop_len, padding=self._out_start
        )
        return self._normalize_istft(inv, real)

    def _istft_A_forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """ISTFT from polar form. Dynamic-length compatible."""
        trig = torch.cat((torch.cos(phase), torch.sin(phase)), dim=1)
        mag_cat = torch.cat((magnitude, magnitude), dim=1)
        inp = trig * mag_cat
        inv = torch.nn.functional.conv_transpose1d(
            inp, self.inverse_kernel, stride=self.hop_len, padding=self._out_start
        )
        return self._normalize_istft(inv, magnitude)

    def _normalize_istft(self, inv: torch.Tensor, spectrum: torch.Tensor) -> torch.Tensor:
        if not self._dynamic_frames:
            return inv * self.inv_win_sum
        n_frames = spectrum.shape[2]
        ones = torch.ones(1, 1, n_frames, dtype=spectrum.dtype, device=spectrum.device)
        win_sum = torch.nn.functional.conv_transpose1d(
            ones, self.win_sq_kernel, stride=self.hop_len, padding=self._out_start
        )
        return inv / win_sum.clamp_min(1e-11)


    def _stft_B_power_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return power without splitting the packed complex channels."""
        if self._reflect_pad:
            x = ONNX_REFLECT_PAD_1D.apply(x, self._pad_left, self._pad_right)
        packed = torch.nn.functional.conv1d(
            x, self.stft_kernel, stride=self.hop_len, padding=self._conv_padding
        )
        if self._drop_after_conv:
            packed = packed[..., :-1]
        packed = packed.reshape(1, 2, self.f_bins, -1)
        return torch.sum(packed * packed, dim=1)

