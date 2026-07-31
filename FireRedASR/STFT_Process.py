#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Production STFT/ISTFT module used by FireRedASR exporter integrations."""

import torch

# Default constructor and window parameters.
NFFT = 512                           # Number of FFT components for the STFT process
WIN_LENGTH = 400                     # Length of the window function (can be different from NFFT)
HOP_LENGTH = 160                     # Number of samples between successive frames in the STFT
INPUT_AUDIO_LENGTH  = 16000          # dummy length for export / test
MAX_SIGNAL_LENGTH   = 2048           # Maximum number of frames for the audio length after STFT processed. Set a appropriate larger value for long audio input, such as 4096.
WINDOW_TYPE         = 'hann'         # bartlett | blackman | hamming | hann | kaiser
PAD_MODE            = 'constant'     # reflect | constant

# clip parameters to sensible ranges
NFFT       = min(NFFT, INPUT_AUDIO_LENGTH)
WIN_LENGTH = min(WIN_LENGTH, NFFT)
HOP_LENGTH = min(HOP_LENGTH, INPUT_AUDIO_LENGTH)

WINDOW_FUNCTIONS = {
    'bartlett': torch.bartlett_window,
    'blackman': torch.blackman_window,
    'hamming' : torch.hamming_window,
    'hann'    : torch.hann_window,
    'kaiser'  : lambda L: torch.kaiser_window(L, periodic=True, beta=12.0)
}
DEFAULT_WINDOW_FN = torch.hann_window


def create_padded_window(win_length, n_fft, window_type):
    """Return length-n_fft window (centre-padded / cropped if needed)."""
    win_fn = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)
    win    = win_fn(win_length).float()
    if win_length == n_fft:
        return win
    if win_length < n_fft:                                   # pad
        pl = (n_fft - win_length) // 2
        pr = n_fft - win_length - pl
        return torch.nn.functional.pad(win, (pl, pr))
    # truncate (shouldn’t occur given sanity checks)
    start = (win_length - n_fft) // 2
    return win[start:start + n_fft]


class STFT_Process(torch.nn.Module):
    def __init__(self,
                 model_type,
                 n_fft=NFFT,
                 win_length=WIN_LENGTH,
                 hop_len=HOP_LENGTH,
                 max_frames=MAX_SIGNAL_LENGTH,
                 window_type=WINDOW_TYPE):
        super().__init__()
        self.model_type  = model_type
        self.n_fft       = n_fft
        self.hop_len     = hop_len
        self.half_n_fft  = n_fft // 2

        window = create_padded_window(win_length, n_fft, window_type)

        # constant-pad buffer (for 'constant' pad mode)
        self.register_buffer('padding_zero', torch.zeros(1, 1, self.half_n_fft, dtype=torch.float32))

        # ─── kernels for STFT_A / STFT_B ───────────────────────────────────
        if model_type in ('stft_A', 'stft_B'):
            t  = torch.arange(n_fft).float().unsqueeze(0)
            f  = torch.arange(self.half_n_fft + 1).float().unsqueeze(1)
            omega = 2 * torch.pi * f * t / n_fft
            self.register_buffer(
                'cos_kernel',
                (torch.cos(omega) * window.unsqueeze(0)).unsqueeze(1)
            )
            self.register_buffer(
                'sin_kernel',
                (-torch.sin(omega) * window.unsqueeze(0)).unsqueeze(1)
            )

        # ─── kernels for ISTFT_A / ISTFT_B ─────────────────────────────────
        if model_type in ('istft_A', 'istft_B'):
            fourier_basis = torch.fft.fft(torch.eye(n_fft, dtype=torch.float32))
            fourier_basis = torch.vstack([
                torch.real(fourier_basis[:self.half_n_fft + 1]),
                torch.imag(fourier_basis[:self.half_n_fft + 1])
            ]).float()

            forward_basis = window * fourier_basis.unsqueeze(1)
            inverse_basis = window * torch.linalg.pinv(
                (fourier_basis * n_fft) / hop_len
            ).T.unsqueeze(1)

            # overlap-add weighting
            n          = n_fft + hop_len * (max_frames - 1)
            window_sum = torch.zeros(n, dtype=torch.float32)

            orig_win = WINDOW_FUNCTIONS.get(window_type, DEFAULT_WINDOW_FN)(win_length).float()
            wn = orig_win / orig_win.abs().max()

            if win_length < n_fft:
                pl = (n_fft - win_length) // 2
                pr = n_fft - win_length - pl
                win_sq = torch.nn.functional.pad(wn ** 2, (pl, pr))
            else:
                win_sq = wn ** 2

            for i in range(max_frames):
                s = i * hop_len
                window_sum[s:s + n_fft] += win_sq[:max(0, min(n_fft, n - s))]

            self.register_buffer('forward_basis', forward_basis)
            self.register_buffer('inverse_basis', inverse_basis)
            self.register_buffer('window_sum_inv', n_fft / (window_sum * hop_len + 1e-7))

    # ───── dispatcher ──────────────────────────────────────────────────────
    def forward(self, *args):
        return getattr(self, f'{self.model_type}_forward')(*args)

    # ───── STFT (A & B) ────────────────────────────────────────────────────
    def _pad_input(self, x, mode):
        if mode == 'reflect':
            return torch.nn.functional.pad(x, (self.half_n_fft, self.half_n_fft), mode='reflect')
        return torch.cat((self.padding_zero, x, self.padding_zero), dim=-1)

    def stft_A_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        x_padded = self._pad_input(x, pad_mode)
        return torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)

    def stft_B_forward(self, x, pad_mode='reflect' if PAD_MODE == 'reflect' else 'constant'):
        x_padded = self._pad_input(x, pad_mode)
        real = torch.nn.functional.conv1d(x_padded, self.cos_kernel, stride=self.hop_len)
        imag = torch.nn.functional.conv1d(x_padded, self.sin_kernel, stride=self.hop_len)
        return real, imag

    # ───── ISTFT_A (magnitude, phase) ──────────────────────────────────────
    def istft_A_forward(self, magnitude, phase):
        cos_p = torch.cos(phase)
        sin_p = torch.sin(phase)
        inp   = torch.cat((magnitude * cos_p, magnitude * sin_p), dim=1)
        inv   = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        s, e  = self.half_n_fft, inv.size(-1) - self.half_n_fft
        return inv[:, :, s:e] * self.window_sum_inv[s:e]

    # ───── ISTFT_B (real, imag) — updated as requested ────────────────────
    def istft_B_forward(self, real, imag):
        inp = torch.cat((real, imag), dim=1)  # == cat(real, imag)
        inv = torch.nn.functional.conv_transpose1d(inp, self.inverse_basis, stride=self.hop_len)
        s, e = self.half_n_fft, inv.size(-1) - self.half_n_fft
        return inv[:, :, s:e] * self.window_sum_inv[s:e]


