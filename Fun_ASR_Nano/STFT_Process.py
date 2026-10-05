#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production Conv1d STFT / ConvTranspose1d ISTFT module used by the exporter.

The implementation includes the Kaldi-compatible fused fbank kernel required by
Fun-ASR-Nano. Constants are precomputed as registered buffers for ONNX tracing.
"""

import torch
from torch.onnx import symbolic_helper

# Default constructor parameters.
NFFT         = 400           # FFT size (number of frequency bins before folding)
WIN_LENGTH   = 400           # Analysis window length in samples (≤ NFFT)
HOP_LENGTH   = 160           # Hop (stride) between successive frames
WINDOW_TYPE  = 'hann'        # Window function: bartlett | blackman | hamming | hann | kaiser
CENTER_PAD   = True          # True  → pad signal so frame centres align with sample indices
                             # False → no padding, first frame starts at sample 0
PAD_MODE     = 'constant'    # Padding style when CENTER_PAD is True: 'reflect' | 'constant'
INPUT_AUDIO_LENGTH = 16000   # Reference length used only to preserve defaults

NFFT       = min(NFFT, INPUT_AUDIO_LENGTH)
WIN_LENGTH = min(WIN_LENGTH, NFFT)
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)

if CENTER_PAD:
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
else:
    STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - NFFT) // HOP_LENGTH + 1

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
    istft_A  → (magnitude, phase) → ConvTranspose1d reconstruction.
    istft_B  → (real, imag) → ConvTranspose1d reconstruction.
    """

    def __init__(self, model_type: str, n_fft: int=NFFT, win_length: int=WIN_LENGTH, hop_len: int=HOP_LENGTH, max_frames: int=STFT_SIGNAL_LENGTH, window_type: str=WINDOW_TYPE, center_pad: bool=CENTER_PAD, pad_mode: str=PAD_MODE, dft_size: int=0, pre_emphasis: float=0.0, remove_dc: bool=False, window_periodic: bool=True, input_scale: float=1.0, drop_last_frame: bool=False, dynamic_frames: bool=True):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.f_bins     = self.half_n_fft + 1
        self.n_frames   = max_frames
        self.drop_last_frame = bool(drop_last_frame)

        if dft_size > 0:
            self.half_n_fft = dft_size // 2
            self.f_bins = self.half_n_fft + 1
            self._reflect_pad = False
            self._conv_padding = 0
            self._drop_after_conv = self.drop_last_frame
            self.forward = self._stft_B_forward
            self._build_kaldi_fbank_kernel(
                win_length, dft_size, window_type, window_periodic, pre_emphasis, remove_dc
            )
            if input_scale != 1.0:
                self.stft_kernel.mul_(float(input_scale))
            return

        f_bins = self.f_bins
        window = create_padded_window(win_length, n_fft, window_type)

        self._out_start = self.half_n_fft if center_pad else 0

        if model_type == 'stft_A':
            self.forward = self._stft_A_forward
        elif model_type == 'stft_B':
            self.forward = self._stft_B_forward
        elif model_type == 'istft_A':
            self.forward = self._istft_A_forward
        elif model_type == 'istft_B':
            self.forward = self._istft_B_forward
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if model_type in ('stft_A', 'stft_B'):
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

    def _build_stft_kernels(
        self, n_fft, f_bins, window, model_type, input_scale
    ):
        """Precompute windowed DFT basis as Conv1d kernel weights."""
        omega_factor = 2.0 * torch.pi / n_fft
        t = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        f = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        omega = omega_factor * f * t

        windowed_cos = (
            torch.cos(omega) * window.unsqueeze(0) * input_scale
        ).unsqueeze(1)
        windowed_sin = (
            -torch.sin(omega) * window.unsqueeze(0) * input_scale
        ).unsqueeze(1)

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


    def _build_kaldi_fbank_kernel(self, win_length, dft_size, window_type, window_periodic, pre_emphasis, remove_dc):
        """
        Build one Conv1d kernel that reproduces torchaudio.compliance.kaldi.fbank
        framing for a single ``win_length``-sample frame:

            per-frame DC removal -> pre-emphasis (replicate pad) -> window
            -> zero-pad to ``dft_size`` -> real DFT (dft_size // 2 + 1 bins)

        All operations are linear, so they collapse into one
        (2 * f_bins, 1, win_length) kernel applied with stride ``hop_len`` and no
        centering (Kaldi snip_edges=True). Built in float64, stored as float32.
        """
        L      = win_length
        f_bins = dft_size // 2 + 1

        # Symmetric (periodic=False) window matches Kaldi's a = 2*pi/(N-1) form.
        window_builders = {
            'bartlett': torch.bartlett_window,
            'blackman': torch.blackman_window,
            'hamming':  torch.hamming_window,
            'hann':     torch.hann_window,
        }
        if window_type in window_builders:
            window = window_builders[window_type](L, periodic=window_periodic, dtype=torch.float64)
        elif window_type == 'kaiser':
            window = torch.kaiser_window(L, periodic=window_periodic, beta=12.0, dtype=torch.float64)
        else:
            window = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(L).to(torch.float64)

        # DFT basis over the first L samples (samples L..dft_size-1 are zero -> omitted).
        n = torch.arange(L,      dtype=torch.float64).unsqueeze(0)   # (1, L)
        k = torch.arange(f_bins, dtype=torch.float64).unsqueeze(1)   # (f_bins, 1)
        omega     =  (2.0 * torch.pi / dft_size) * k * n            # (f_bins, L)
        cos_basis =  torch.cos(omega)                              # (f_bins, L)
        sin_basis = -torch.sin(omega)                              # (f_bins, L)  (matches stft_B imag sign)

        # Per-frame pre-processing matrix M (L, L): M = diag(window) @ Preemph @ DCremove.
        eye = torch.eye(L, dtype=torch.float64)
        dc  = eye - (1.0 / L) if remove_dc else eye                # x - mean(x) = (I - J/L) x

        preemph = eye.clone()
        if pre_emphasis and pre_emphasis > 0.0:
            rows = torch.arange(1, L)
            preemph[rows, rows - 1] = -pre_emphasis                # f[i] -= a * f[i-1]
            preemph[0, 0]           = 1.0 - pre_emphasis           # replicate pad on f[0]

        M = torch.diag(window) @ preemph @ dc                      # (L, L)

        real_kernel = cos_basis @ M                                # (f_bins, L)
        imag_kernel = sin_basis @ M                                # (f_bins, L)
        kernel = torch.cat([real_kernel, imag_kernel], dim=0).unsqueeze(1).to(torch.float32)  # (2*f_bins, 1, L)
        self.register_buffer('stft_kernel', kernel)
