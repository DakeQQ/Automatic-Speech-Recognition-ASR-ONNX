#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Production Conv1d STFT / ConvTranspose1d ISTFT module used by the exporter.

The implementation includes the Kaldi-compatible fused fbank kernel required by
Fun-ASR-Nano. Constants are precomputed as registered buffers for ONNX tracing.
"""

import torch

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


class STFT_Process(torch.nn.Module):
    """
    Static-graph Conv1d STFT / ConvTranspose1d ISTFT for ONNX export.

    All constants precomputed in __init__() as registered buffers.
    Forward path is pure tensor ops — no dispatch, no branching, no shape queries.

    Variants
    --------
    stft_A   → Conv1d producing real part only.
    stft_B   → Conv1d producing real + imag (split after convolution).
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
        dft_size: int        = 0,     # >0 -> Kaldi fbank mode: zero-pad win_length -> dft_size before the DFT
        pre_emphasis: float  = 0.0,   # Kaldi per-frame pre-emphasis coefficient (replicate-padded)
        remove_dc: bool      = False, # Kaldi per-frame DC-offset removal
        window_periodic: bool = True  # False -> symmetric window (Kaldi's 2*pi/(N-1) convention)
    ):
        super().__init__()

        self.model_type = model_type
        self.n_fft      = n_fft
        self.hop_len    = hop_len
        self.half_n_fft = n_fft // 2
        self.n_frames   = max_frames

        # ── Kaldi-compatible fbank STFT ───────────────────────────────────
        # Reproduces torchaudio.compliance.kaldi.fbank framing: snip_edges (no
        # centering), per-frame DC removal + pre-emphasis, symmetric window, and
        # zero-padding win_length -> dft_size before the DFT. Every step is linear,
        # so they all fold into a single Conv1d kernel built once here.
        if dft_size > 0:
            self.half_n_fft  = dft_size // 2          # channel split -> dft_size // 2 + 1 bins
            self._center_pad = False                  # Kaldi snip_edges=True -> no centering
            self.forward     = self._stft_B_forward
            self._build_kaldi_fbank_kernel(win_length, dft_size, window_type, window_periodic, pre_emphasis, remove_dc)
            return

        f_bins = self.half_n_fft + 1
        window = create_padded_window(win_length, n_fft, window_type)

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
        if model_type in ('stft_A', 'stft_B'):
            self._build_stft_kernels(n_fft, f_bins, window, model_type)
            if center_pad and pad_mode == 'constant':
                self.register_buffer(
                    'padding_zero',
                    torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32)
                )
            self._center_pad = center_pad
            self._pad_mode   = pad_mode

        # ── ISTFT: inverse kernel + pre-sliced normalization ──────────────
        if model_type in ('istft_A', 'istft_B'):
            self._build_istft_kernels(n_fft, f_bins, window, hop_len, max_frames)

    def _build_stft_kernels(self, n_fft, f_bins, window, model_type):
        """Precompute windowed DFT basis as Conv1d kernel weights."""
        omega_factor = 2.0 * torch.pi / n_fft
        t = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        f = torch.arange(f_bins, dtype=torch.float32).unsqueeze(1)
        omega = omega_factor * f * t

        windowed_cos = ( torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
        windowed_sin = (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)

        if model_type == 'stft_A':
            self.register_buffer('stft_kernel', windowed_cos)
        else:
            self.register_buffer('stft_kernel', torch.cat([windowed_cos, windowed_sin], dim=0))

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
                right = x[..., -(self.half_n_fft + 1): -1].flip(2)
                x = torch.cat([left, x, right], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
                else:
                    padding_zero = self.padding_zero
                x = torch.cat([padding_zero, x, padding_zero], dim=2)
        return torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)

    def _stft_B_forward(self, x: torch.Tensor):
        """STFT producing (real, imag) via a single Conv1d + channel Split."""
        if self._center_pad:
            if self._pad_mode == 'reflect':
                left  = x[..., 1: self.half_n_fft + 1].flip(2)
                right = x[..., -(self.half_n_fft + 1): -1].flip(2)
                x = torch.cat([left, x, right], dim=2)
            else:
                if x.shape[0] != 1:
                    padding_zero = torch.cat([self.padding_zero] * x.shape[0], dim=0)
                else:
                    padding_zero = self.padding_zero
                x = torch.cat([padding_zero, x, padding_zero], dim=2)
        out = torch.nn.functional.conv1d(x, self.stft_kernel, stride=self.hop_len)
        return torch.split(out, self.half_n_fft + 1, dim=1)

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
