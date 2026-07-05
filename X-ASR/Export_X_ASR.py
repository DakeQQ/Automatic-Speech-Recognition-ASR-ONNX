import gc
import subprocess
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn


# ============================================================================================
#                                       User configuration
# ============================================================================================
# Directory of the icefall Zipformer recipe that holds the pure-torch model modules
# (zipformer.py, scaling.py, subsampling.py, decoder.py, joiner.py, scaling_converter.py).
ZIPFORMER_DIR = "/home/DakeQQ/Downloads/X-ASR-main/X-ASR-zh-en/zipformer"
CHECKPOINT    = ZIPFORMER_DIR + "/checkpoint/fintuned_with_punctuation.pt"
TOKENS_TXT    = ZIPFORMER_DIR + "/data/lang_5000_with_punctuation/tokens.txt"
WEIGHT_KEY    = "model_avg"         # Which weight tensor set to export: "model_avg" (icefall's averaged weights, recommended) or "model".


DYNAMIC_AXES  = False               # Dynamic batch N (streaming still runs batch 1).
CHUNK_MS      = 160                 # Streaming chunk latency in ms; picks the (chunk_size, left_context_frames) pair below.
SAMPLE_RATE   = 16000               # Model parameter, do not edit.
INPUT_AUDIO_DTYPE = "INT16"         # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. "INT16" feeds raw PCM (÷32768 in-graph); "F32"/"F16" feed audio pre-normalised to [-1, 1].
OPSET         = 18

NFFT          = 512                 # kaldi rounds the 400-sample (25 ms) frame up to the next power of two before the FFT.
FEATURE_DIM   = 80                  # kaldi fbank mel bins, do not edit.
WINDOW_LENGTH = 400                 # kaldi frame length (25 ms @ 16 kHz), do not edit.
HOP_LENGTH    = 160                 # kaldi frame shift (10 ms @ 16 kHz), do not edit.
PRE_EMPHASIS  = 0.97                # kaldi pre-emphasis coefficient, do not edit.

onnx_folder = Path(__file__).resolve().parent / "X_ASR_ONNX"
onnx_folder.mkdir(parents=True, exist_ok=True)
onnx_model_Metadata = str(onnx_folder / "X_ASR_Metadata.onnx")
onnx_encoder = str(onnx_folder / "X_ASR_Encoder.onnx")
onnx_decoder = str(onnx_folder / "X_ASR_Decoder.onnx")
onnx_joiner = str(onnx_folder / "X_ASR_Joiner.onnx")

_audio_torch_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]  # export dummy / self-test dtype


# ============================================================================================
#              Standalone shims: import the pure-torch model modules without k2 / icefall
# ============================================================================================
@contextmanager
def _torch_autocast(*_a, **_k):
    # Every inference call site uses torch_autocast(enabled=False); a no-op passthrough is exact.
    yield


def _install_shims():
    _ic = types.ModuleType("icefall")
    _icu = types.ModuleType("icefall.utils")
    _icu.torch_autocast = _torch_autocast
    _ic.utils = _icu
    sys.modules.setdefault("icefall", _ic)
    sys.modules.setdefault("icefall.utils", _icu)

    def _swoosh_l(x):
        return torch.logaddexp(torch.zeros((), dtype=x.dtype, device=x.device), x - 4.0) - 0.08 * x - 0.035

    def _swoosh_r(x):
        return torch.logaddexp(torch.zeros((), dtype=x.dtype, device=x.device), x - 1.0) - 0.08 * x - 0.313261687

    _k2 = types.ModuleType("k2")
    # Exact SwooshL/R (same formulas as scaling.py's tracing branch); *_and_deriv are backward-only stubs.
    _k2.swoosh_l = _swoosh_l
    _k2.swoosh_r = _swoosh_r
    _k2.swoosh_l_forward = _swoosh_l
    _k2.swoosh_r_forward = _swoosh_r
    _k2.swoosh_l_forward_and_deriv = lambda x: (_swoosh_l(x), torch.ones_like(x))
    _k2.swoosh_r_forward_and_deriv = lambda x: (_swoosh_r(x), torch.ones_like(x))
    sys.modules.setdefault("k2", _k2)

_install_shims()
sys.path.insert(0, ZIPFORMER_DIR)

from subsampling import Conv2dSubsampling          
from zipformer import Zipformer2                   
from decoder import Decoder                         
from joiner import Joiner                           
from scaling import ScheduledFloat                      
from scaling_converter import convert_scaled_to_non_scaled  

# ============================================================================================
#                    Exact X-ASR-zh-en architecture (verified against the checkpoint)
# ============================================================================================
# chunk_ms -> (chunk_size @ 50Hz, left_context_frames @ 50Hz). Verified from the released onnx metadata.
CHUNK_TABLE = {160: (8, 96), 480: (24, 256), 960: (48, 256), 1920: (96, 256)}
CHUNK_SIZE, LEFT_CONTEXT_FRAMES = CHUNK_TABLE[CHUNK_MS]

ARCH = dict(
    output_downsampling_factor=2,
    downsampling_factor=(1, 2, 4, 8, 4, 2),
    num_encoder_layers=(2, 2, 4, 5, 4, 2),           # 19 layers total
    encoder_dim=(192, 256, 512, 768, 512, 256),
    encoder_unmasked_dim=(192, 192, 256, 256, 256, 192),   # training-only (no params); default kept
    query_head_dim=(32,) * 6,
    pos_head_dim=(4,) * 6,
    value_head_dim=(12,) * 6,
    pos_dim=48,
    num_heads=(4, 4, 4, 8, 4, 4),
    feedforward_dim=(512, 768, 1536, 2048, 1536, 768),     # = feed_forward1.in_proj_out * 4 // 3
    cnn_module_kernel=(31, 31, 15, 15, 15, 31),
)
VOCAB_SIZE = 5000
DECODER_DIM = 512
JOINER_DIM = 512
CONTEXT_SIZE = 2
BLANK_ID = 0
PAD_LENGTH = 7 + 2 * 3               # encoder_embed subsample (7) + ConvNeXt right pad (2*3)
NUM_LAYERS_TOTAL = sum(ARCH["num_encoder_layers"])   # 19


def build_model():
    """Construct the four sub-modules exactly as train.py.get_model does, then load the
    checkpoint's encoder_embed. / encoder. / decoder. / joiner. sub-state-dicts."""
    sched = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))
    encoder_embed = Conv2dSubsampling(
        in_channels=FEATURE_DIM, out_channels=ARCH["encoder_dim"][0], dropout=sched
    )
    encoder = Zipformer2(
        output_downsampling_factor=ARCH["output_downsampling_factor"],
        downsampling_factor=ARCH["downsampling_factor"],
        num_encoder_layers=ARCH["num_encoder_layers"],
        encoder_dim=ARCH["encoder_dim"],
        encoder_unmasked_dim=ARCH["encoder_unmasked_dim"],
        query_head_dim=ARCH["query_head_dim"],
        pos_head_dim=ARCH["pos_head_dim"],
        value_head_dim=ARCH["value_head_dim"],
        pos_dim=ARCH["pos_dim"],
        num_heads=ARCH["num_heads"],
        feedforward_dim=ARCH["feedforward_dim"],
        cnn_module_kernel=ARCH["cnn_module_kernel"],
        dropout=sched,
        warmup_batches=4000.0,
        causal=True,
        chunk_size=(CHUNK_SIZE,),
        left_context_frames=(LEFT_CONTEXT_FRAMES,),
    )
    decoder = Decoder(vocab_size=VOCAB_SIZE, decoder_dim=DECODER_DIM, blank_id=BLANK_ID, context_size=CONTEXT_SIZE)
    joiner = Joiner(encoder_dim=max(ARCH["encoder_dim"]), decoder_dim=DECODER_DIM, joiner_dim=JOINER_DIM, vocab_size=VOCAB_SIZE)

    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    sd = ckpt[WEIGHT_KEY] if WEIGHT_KEY in ckpt else ckpt.get("model", ckpt)

    def _sub(prefix):
        n = len(prefix) + 1
        return {k[n:]: v for k, v in sd.items() if k.startswith(prefix + ".")}

    encoder_embed.load_state_dict(_sub("encoder_embed"))
    encoder.load_state_dict(_sub("encoder"))
    decoder.load_state_dict(_sub("decoder"))
    joiner.load_state_dict(_sub("joiner"))
    del ckpt, sd
    gc.collect()
    return encoder_embed.eval(), encoder.eval(), decoder.eval(), joiner.eval()


# ============================================================================================
#                                 ONNX-friendly wrapper modules
# ============================================================================================
class XasrEncoder(nn.Module):
    """Zipformer2 streaming encoder + encoder_embed, with the joiner.encoder_proj FUSED so the
    output is already in joiner space. Threads the 6-per-layer sliding caches (+ ConvNeXt left
    pad + processed_lens) as flat graph inputs/outputs."""

    def __init__(self, encoder, encoder_embed, encoder_proj):
        super().__init__()
        self.encoder = encoder
        self.encoder_embed = encoder_embed
        self.encoder_proj = encoder_proj
        self.chunk_size = encoder.chunk_size[0]
        self.left_context_len = encoder.left_context_frames[0]
        self.pad_length = PAD_LENGTH
        # Pre-compute the static per-chunk frame count and the constant length vector (no runtime shape reads).
        self.T = self.chunk_size * 2 + self.pad_length
        self.register_buffer("_x_lens", torch.tensor([self.T], dtype=torch.int64), persistent=False)
        # Pre-compute the [0..left_context_len) ramp used to build the initial-state mask (int64, reused).
        self.register_buffer("_ctx_ramp", torch.arange(self.left_context_len, dtype=torch.int64).unsqueeze(0), persistent=False)
        self.register_buffer("_ctx_ramp_reversed", torch.arange(self.left_context_len - 1, -1, -1, dtype=torch.int64).unsqueeze(0), persistent=False)
        self.register_buffer("_current_mask", torch.zeros(1, self.chunk_size, dtype=torch.bool), persistent=False)
        # int16 audio is raw PCM (normalised to [-1, 1] in _fbank via ÷32768); f32/f16 audio is assumed
        # pre-normalised to [-1, 1], so the in-graph division is skipped.
        self.inv_int16 = float(1.0 / 32768.0)
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")
        self._build_fbank_frontend()
        # Subsampled chunk length after Conv2dSubsampling: (T - 7) // 2 - 3 (== CHUNK_SIZE). Added as a
        # constant so new_processed_lens never depends on a runtime shape read.
        self.register_buffer("_x_lens_sub", torch.tensor([(self.T - 7) // 2 - 3], dtype=torch.int64), persistent=False)
        # Fold every geometry-dependent constant of the inlined streaming encoder into reusable buffers.
        self._precompute_encoder_constants()

    def _build_fbank_frontend(self):
        # Kaldi-faithful log-Mel front-end folded into the graph (SenseVoice template). ONE Conv1d
        # kernel folds, per frame: DC-offset removal, pre-emphasis (replicate boundary), the povey
        # analysis window and the one-sided windowed DFT (NFFT-point basis over WINDOW_LENGTH samples).
        # Conv stride = HOP_LENGTH with no padding == Kaldi snip_edges=True framing; the runner
        # reflection-pads the waveform so this exactly reproduces the training snip_edges=False fbank.
        import torchaudio.compliance.kaldi as kaldi
        self.hop_length = HOP_LENGTH
        self.fbank_freq = NFFT // 2 + 1                                         # one-sided FFT bins (257)
        self.audio_chunk_samples = (self.T - 1) * HOP_LENGTH + WINDOW_LENGTH    # waveform samples per encoder chunk
        window = torch.hann_window(WINDOW_LENGTH, periodic=False, dtype=torch.float32).pow(0.85)   # kaldi 'povey' window
        freqs = torch.arange(self.fbank_freq, dtype=torch.float32).unsqueeze(1)
        samples = torch.arange(WINDOW_LENGTH, dtype=torch.float32).unsqueeze(0)
        omega = (2.0 * torch.pi / NFFT) * freqs * samples
        cos_basis = torch.cos(omega) * window
        sin_basis = -torch.sin(omega) * window

        def fold_frontend(basis):
            shifted = torch.cat([basis[:, 1:], torch.zeros_like(basis[:, :1])], dim=1)   # basis[:, n + 1], zero past the frame edge
            folded = basis - PRE_EMPHASIS * shifted                                       # pre-emphasis: pf[n] = s[n] - c * s[n - 1]
            folded[:, 0] = folded[:, 0] - PRE_EMPHASIS * basis[:, 0]                       # replicate boundary: pf[0] = (1 - c) * s[0]
            return folded - folded.mean(dim=1, keepdim=True)                              # per-frame DC removal (subtract the frame mean)

        fbank_kernel = torch.cat([fold_frontend(cos_basis), fold_frontend(sin_basis)], dim=0).unsqueeze(1)
        self.register_buffer("fbank_kernel", fbank_kernel.contiguous(), persistent=False)   # (2 * fbank_freq, 1, WINDOW_LENGTH)
        self.log_eps = float(torch.finfo(torch.float32).eps)                              # Kaldi's log floor (FLT_EPSILON)
        # Kaldi triangular mel filterbank (low_freq .. high_freq) over the NFFT spectrum, padded with a
        # zero Nyquist column and baked as a constant of shape (1, fbank_freq, FEATURE_DIM).
        mel_banks, _ = kaldi.get_mel_banks(FEATURE_DIM, NFFT, float(SAMPLE_RATE), 20.0, -400.0, 100.0, -500.0, 1.0)
        self.register_buffer("mel_filters", torch.nn.functional.pad(mel_banks, (0, 1), value=0.0).transpose(0, 1).unsqueeze(0).contiguous(), persistent=False)

    def _fbank(self, audio):
        # audio: (N, 1, audio_chunk_samples) waveform. INT16 raw PCM is normalised to [-1, 1] here (÷32768);
        # F32/F16 audio is assumed pre-normalised to [-1, 1], so the division is skipped. Returns (N, T, FEATURE_DIM).
        if self.input_audio_is_int16:
            audio = audio.float() * self.inv_int16
        else:
            audio = audio.float()
        spectrum = torch.nn.functional.conv1d(audio, self.fbank_kernel, stride=self.hop_length)
        real_power, imag_power = torch.split(spectrum * spectrum, self.fbank_freq, dim=1)   # one square over 2*fbank_freq channels, then split (== real^2 / imag^2)
        power = (real_power + imag_power).transpose(1, 2)                                   # (N, T, fbank_freq)
        return torch.matmul(power, self.mel_filters).clamp(min=self.log_eps).log()          # (N, T, FEATURE_DIM)

    def get_init_states(self, batch_size: int = 1):
        states = self.encoder.get_init_states(batch_size)
        states.append(self.encoder_embed.get_init_states(batch_size))
        states.append(torch.zeros(batch_size, dtype=torch.int64))
        return states

    def _precompute_encoder_constants(self):
        """Fold every geometry-dependent constant of the inlined streaming encoder into reusable buffers.

        Everything here depends only on the FIXED chunk / left-context geometry, so it is computed once
        and folds to graph constants:
          * per-layer projected relative-position tables  (linear_pos(pos_emb) reshaped for the score matmul),
          * per-stack relative->absolute gather indices    (int32, batch-independent),
          * per-conv chunk-scale tables                    (ChunkCausalDepthwiseConv1d._get_chunk_scale),
          * per-stack SimpleDownsample softmax weights,
          * the channel-conversion / full-dim-combine / output-downsample / subsampling geometry plans.
        All source sub-modules used here (encoder_pos, linear_pos, depthwise_conv, downsample.bias) are
        invariant under convert_scaled_to_non_scaled, so precomputing in __init__ stays valid post-convert."""
        enc = self.encoder
        ds_factors = enc.downsampling_factor
        self._num_stacks = len(ds_factors)
        self._stack_meta = []
        self._layer_refs = []
        self._stack_gl_offset = []
        with torch.no_grad():
            gl = 0
            for s, module in enumerate(enc.encoders):
                ds = int(ds_factors[s])
                is_downsampled = ds != 1
                inner = module.encoder if is_downsampled else module          # DownsampledZipformer2Encoder wraps the stack
                n_layers = inner.num_layers
                embed_dim = int(enc.encoder_dim[s])
                seq = self.chunk_size // ds                                    # inner (post-downsample) chunk length
                left = self.left_context_len // ds                            # inner left-context frames
                k_len = seq + left
                seq_len2 = 2 * seq - 1 + left                                  # relative-position axis length
                saw0 = inner.layers[0].self_attn_weights
                heads, qhd, phd = saw0.num_heads, saw0.query_head_dim, saw0.pos_head_dim
                hidden = inner.layers[0].nonlin_attention.hidden_channels
                conv_left_pad = inner.layers[0].conv_module1.depthwise_conv.kernel_size // 2
                # Constant relative-position embedding for this stack (only x.size(0)==seq matters).
                pos_emb = inner.encoder_pos(torch.zeros(seq, 1, embed_dim), left)   # (1, seq_len2, pos_dim)
                # Relative->absolute gather index (int32), batch-independent: base[t, c] = (seq - 1 - t) + c.
                t_idx = torch.arange(seq, dtype=torch.int64).unsqueeze(1)
                c_idx = torch.arange(k_len, dtype=torch.int64).unsqueeze(0)
                gidx = ((seq - 1 - t_idx) + c_idx).to(torch.int32).reshape(1, 1, seq, k_len)
                self.register_buffer(f"_gidx_{s}", gidx.contiguous(), persistent=False)
                if ds != 1:
                    mask_idx = torch.arange(0, self.left_context_len + self.chunk_size, ds, dtype=torch.int32)
                    self.register_buffer(f"_kpm_idx_{s}", mask_idx.contiguous(), persistent=False)
                if is_downsampled:
                    self.register_buffer(f"_dsw_{s}", module.downsample.bias.detach().softmax(dim=0).contiguous(), persistent=False)
                self._stack_gl_offset.append(gl)
                layer_refs = []
                for l in range(n_layers):
                    layer = inner.layers[l]
                    # linear_pos(pos_emb) -> (heads, 1, phd, seq_len2): removes 19x linear_pos + reshape + permute.
                    pp = layer.self_attn_weights.linear_pos(pos_emb).reshape(-1, seq_len2, heads, phd).permute(2, 0, 3, 1)
                    self.register_buffer(f"_pp_{gl}", pp.detach().contiguous(), persistent=False)
                    self.register_buffer(f"_cs1_{gl}", layer.conv_module1.depthwise_conv._get_chunk_scale(seq).detach().contiguous(), persistent=False)
                    self.register_buffer(f"_cs2_{gl}", layer.conv_module2.depthwise_conv._get_chunk_scale(seq).detach().contiguous(), persistent=False)
                    layer_refs.append(layer)
                    gl += 1
                self._layer_refs.append(layer_refs)
                vdim = inner.layers[0].self_attn1.in_proj.out_features            # heads * value_head_dim
                self._stack_meta.append({
                    "s": s, "ds": ds, "is_downsampled": is_downsampled, "module": module,
                    "n_layers": n_layers, "embed_dim": embed_dim, "seq": seq, "left": left,
                    "k_len": k_len, "seq_len2": seq_len2, "heads": heads, "qhd": qhd, "phd": phd,
                    "vhd": vdim // heads, "vdim": vdim, "hidden": hidden, "conv_left_pad": conv_left_pad,
                })
            # Channel-conversion plan between stacks (convert_num_channels): 'none' / 'slice' / 'pad'.
            self._cnc_plan = []
            for s in range(self._num_stacks):
                enter = int(enc.encoder_dim[0]) if s == 0 else int(enc.encoder_dim[s - 1])
                nc = int(enc.encoder_dim[s])
                if nc == enter:
                    self._cnc_plan.append(("none", 0))
                elif nc < enter:
                    self._cnc_plan.append(("slice", nc))
                else:
                    pad_channels = nc - enter
                    self._cnc_plan.append(("pad", pad_channels))
                    self.register_buffer(f"_cnc_pad_{s}", torch.zeros(self.chunk_size, 1, pad_channels), persistent=False)
            # Full-dim combine plan (_get_full_dim_output): take each dim from the most-recent stack that has it.
            dims = [int(d) for d in enc.encoder_dim]
            plan = [(self._num_stacks - 1, 0, dims[-1], True)]
            cur = dims[-1]
            for i in range(self._num_stacks - 2, -1, -1):
                if dims[i] > cur:
                    plan.append((i, cur, dims[i], False))
                    cur = dims[i]
            self._full_plan = plan
            # Output SimpleDownsample (softmax bias weights) + its static reshape geometry.
            self.register_buffer("_dso_w", enc.downsample_output.bias.detach().softmax(dim=0).contiguous(), persistent=False)
            self._output_ds = int(enc.output_downsampling_factor)
            self._output_dim = max(dims)
            self._output_dseq = self.chunk_size // self._output_ds
            # Conv2dSubsampling / ConvNeXt static geometry.
            cx = self.encoder_embed.convnext
            self._sub_pad0 = cx.padding[0]
            self._sub_pad1 = cx.padding[1]
            self._sub_groups = cx.depthwise_conv.groups
            self._sub_conv_T = (self.T - 7) // 2 - self._sub_pad0
            self._sub_feat = self.encoder_embed.out_width * self.encoder_embed.layer3_channels

    def _subsample(self, x, cached_left_pad):
        # Inlined Conv2dSubsampling.streaming_forward: the front Conv2d stack is a plain Sequential (called
        # as-is), then the ConvNeXt left-pad cache handling / depthwise conv / bypass is inlined, and the
        # flatten + out Linear + out BiasNorm follow. x: (N, T=29, 80) -> (N, 8, 192).
        ee = self.encoder_embed
        cx = ee.convnext
        x = x.unsqueeze(1)                                              # (N, 1, 29, 80)
        x = ee.conv(x)                                                  # (N, 128, 11, 19)
        T = self._sub_conv_T                                           # 8
        bypass = x[:, :, :T, :]
        x = torch.cat([cached_left_pad, x], dim=2)                     # prepend cached left pad
        new_cached_left_pad = x[:, :, T:T + self._sub_pad0, :]
        x = torch.nn.functional.conv2d(x, cx.depthwise_conv.weight, cx.depthwise_conv.bias, padding=(0, self._sub_pad1), groups=self._sub_groups)
        x = cx.pointwise_conv1(x)
        x = cx.activation(x)                                           # SwooshL
        x = cx.pointwise_conv2(x)
        x = bypass + x                                                 # (N, 128, 8, 19)
        x = x.transpose(1, 2).reshape(-1, T, self._sub_feat)          # (N, 8, 2432); -1 keeps N dynamic
        x = ee.out(x)                                                 # (N, 8, 192)
        return ee.out_norm(x), new_cached_left_pad

    def _build_mask(self, N, processed_lens):
        # Padding mask over [left_context | current_chunk]; the left part masks not-yet-seen context. The
        # current-chunk part is always all-valid (zeros), so the conv-module mask is a no-op and dropped.
        current = self._current_mask.expand(N, self.chunk_size)
        ramp = self._ctx_ramp_reversed.expand(N, self.left_context_len)
        processed_mask = processed_lens.unsqueeze(1) <= ramp
        new_processed_lens = processed_lens + self._x_lens_sub
        return torch.cat([processed_mask, current], dim=1), new_processed_lens

    def _attn_weights(self, layer, gl, meta, src, cached_key, kpm, N):
        # RelPositionMultiheadAttentionWeights.streaming_forward, inlined: one fused in_proj -> (q, k, p)
        # via split; the projected relative-position table is precomputed, and rel->absolute uses a
        # precomputed int32 gather. The head_dim**-0.25 scale is already baked into in_proj (ScaledLinear).
        # Reshapes use -1 for the batch axis (no shape reads); N is threaded in only for the gather expand.
        saw = layer.self_attn_weights
        heads, qhd, phd = meta["heads"], meta["qhd"], meta["phd"]
        seq, left, k_len = meta["seq"], meta["left"], meta["k_len"]
        qdim = qhd * heads
        q, k, p = torch.split(saw.in_proj(src), [qdim, qdim, heads * phd], dim=-1)
        k = torch.cat([cached_key, k], dim=0)                          # (k_len, N, qdim)
        new_cached_key = k[-left:]
        q = q.reshape(seq, -1, heads, qhd).permute(2, 1, 0, 3)         # (h, N, seq, qhd)
        p = p.reshape(seq, -1, heads, phd).permute(2, 1, 0, 3)         # (h, N, seq, phd)
        k = k.reshape(k_len, -1, heads, qhd).permute(2, 1, 3, 0)       # (h, N, qhd, k_len)
        attn_scores = torch.matmul(q, k)                              # (h, N, seq, k_len)
        pos_scores = torch.matmul(p, getattr(self, f"_pp_{gl}"))      # (h, N, seq, seq_len2)
        idx = getattr(self, f"_gidx_{meta['s']}").expand(heads, N, seq, k_len)
        pos_scores = torch.gather(pos_scores, 3, idx)                 # relative -> absolute (h, N, seq, k_len)
        attn_scores = attn_scores + pos_scores
        attn_scores = attn_scores.masked_fill(kpm.unsqueeze(1), -1000.0)
        return attn_scores.softmax(dim=-1), new_cached_key

    def _self_attn(self, sa, meta, src, attn_weights, cached_val):
        # SelfAttention.streaming_forward, inlined: value in_proj -> cache concat -> attn-weighted sum.
        # Value-head and merged value dims are precomputed, so every reshape uses -1 for the batch axis.
        heads, seq, left, k_len, vhd, vdim = meta["heads"], meta["seq"], meta["left"], meta["k_len"], meta["vhd"], meta["vdim"]
        x = torch.cat([cached_val, sa.in_proj(src)], dim=0)            # (k_len, N, heads*vhd)
        new_cached_val = x[-left:]
        x = x.reshape(k_len, -1, heads, vhd).permute(2, 1, 0, 3)      # (h, N, k_len, vhd)
        x = torch.matmul(attn_weights, x)                            # (h, N, seq, vhd)
        x = x.permute(2, 1, 0, 3).reshape(seq, -1, vdim)             # (seq, N, heads*vhd)
        return sa.out_proj(x), new_cached_val

    def _nonlin_attn(self, na, meta, src, attn_weights0, cached_x):
        # NonlinAttention.streaming_forward, inlined (single head over the whole hidden dim). The no-op
        # unsqueeze/reshape around tanh(s) in the original is dropped; reshapes use -1 for the batch axis.
        hidden, seq, left = meta["hidden"], meta["seq"], meta["left"]
        s, x, y = torch.split(na.in_proj(src), hidden, dim=2)          # each (seq, N, hidden)
        x = x * na.tanh(s)
        x = x.reshape(seq, -1, 1, hidden).permute(2, 1, 0, 3)         # (1, N, seq, hidden)
        x_pad = torch.cat([cached_x, x], dim=2)                        # (1, N, left+seq, hidden)
        new_cached_x = x_pad[:, :, -left:, :]
        x = torch.matmul(attn_weights0, x_pad)                       # (1, N, seq, hidden)
        x = x.permute(2, 1, 0, 3).reshape(seq, -1, hidden) * y        # (seq, N, hidden)
        return na.out_proj(x), new_cached_x

    def _conv_module(self, conv, chunk_scale, meta, src, cache):
        # ConvolutionModule.streaming_forward, inlined: GLU gate, then the ChunkCausalDepthwiseConv1d
        # streaming (causal branch + chunk-scaled chunkwise branch). The src_key_padding_mask masked_fill
        # is dropped: the current-chunk mask is always all-False.
        ed, left_pad = meta["embed_dim"], meta["conv_left_pad"]
        dc = conv.depthwise_conv
        x, s = torch.split(conv.in_proj(src), ed, dim=2)              # (seq, N, ed) each
        x = (x * conv.sigmoid(s)).permute(1, 2, 0)                    # (N, ed, seq)
        x = torch.cat([cache, x], dim=2)                              # prepend cached left pad
        new_cache = x[..., -left_pad:]
        x_chunk = dc.chunkwise_conv(x[..., left_pad:]) * chunk_scale
        x = (x_chunk + dc.causal_conv(x)).permute(2, 0, 1)           # (seq, N, ed)
        return conv.out_proj(x), new_cache

    def _simple_downsample(self, meta, src):
        # SimpleDownsample.forward, inlined (pad is always 0 for a streaming chunk): weighted sum over ds.
        ds, C = meta["ds"], meta["embed_dim"]
        src = src.reshape(meta["seq"], ds, -1, C)                     # (d_seq, ds, N, C); -1 keeps N dynamic
        return (src * getattr(self, f"_dsw_{meta['s']}").reshape(1, ds, 1, 1)).sum(dim=1)

    def _simple_upsample(self, meta, src):
        # SimpleUpsample.forward, inlined: repeat each frame `ds` times along the time axis.
        ds, C = meta["ds"], meta["embed_dim"]
        return src.unsqueeze(1).expand(-1, ds, -1, -1).reshape(meta["seq"] * ds, -1, C)

    @staticmethod
    def _bypass(bypass_module, src_orig, src):
        # BypassModule.forward, inlined: learnable per-channel bypass on the non-residual term.
        return src_orig + (src - src_orig) * bypass_module.bypass_scale

    def _convert_channels(self, x, s, N):
        # convert_num_channels, inlined via a static per-stack plan (slice to shrink / zero-pad to grow).
        mode, param = self._cnc_plan[s]
        if mode == "none":
            return x
        if mode == "slice":
            return x[..., :param]
        return torch.cat([x, getattr(self, f"_cnc_pad_{s}").expand(self.chunk_size, N, param)], dim=-1)

    def _full_dim(self, outputs):
        # _get_full_dim_output, inlined via a static plan of (stack, lo, hi) slices concatenated on the channel axis.
        pieces = [outputs[st] if is_full else outputs[st][..., lo:hi] for (st, lo, hi, is_full) in self._full_plan]
        return torch.cat(pieces, dim=-1)

    def _downsample_output(self, x):
        # Final SimpleDownsample (output_downsampling_factor), inlined with precomputed softmax weights.
        ds = self._output_ds
        x = x.reshape(self._output_dseq, ds, -1, self._output_dim)   # (d_seq, ds, N, C); -1 keeps N dynamic
        return (x * self._dso_w.reshape(1, -1, 1, 1)).sum(dim=1)

    def _encoder_layer(self, meta, gl, layer, src, layer_states, kpm, N):
        # Zipformer2EncoderLayer.streaming_forward, inlined. Feedforward / BiasNorm / bypass leaf modules
        # are called directly (their tracing branch is already minimal and exact).
        cached_key, cached_nonlin, cached_val1, cached_val2, cached_conv1, cached_conv2 = layer_states
        src_orig = src
        attn_weights, new_key = self._attn_weights(layer, gl, meta, src, cached_key, kpm, N)
        src = src + layer.feed_forward1(src)
        na, new_nonlin = self._nonlin_attn(layer.nonlin_attention, meta, src, attn_weights[0:1], cached_nonlin)
        src = src + na
        sa, new_val1 = self._self_attn(layer.self_attn1, meta, src, attn_weights, cached_val1)
        src = src + sa
        sc, new_conv1 = self._conv_module(layer.conv_module1, getattr(self, f"_cs1_{gl}"), meta, src, cached_conv1)
        src = src + sc
        src = src + layer.feed_forward2(src)
        src = self._bypass(layer.bypass_mid, src_orig, src)
        sa, new_val2 = self._self_attn(layer.self_attn2, meta, src, attn_weights, cached_val2)
        src = src + sa
        sc, new_conv2 = self._conv_module(layer.conv_module2, getattr(self, f"_cs2_{gl}"), meta, src, cached_conv2)
        src = src + sc
        src = src + layer.feed_forward3(src)
        src = layer.norm(src)                                         # BiasNorm (tracing branch is exact)
        src = self._bypass(layer.bypass, src_orig, src)
        return src, [new_key, new_nonlin, new_val1, new_val2, new_conv1, new_conv2]

    def _stack_forward(self, s, x, stack_states, kpm, N):
        # One encoder stack: plain Zipformer2Encoder (ds == 1) or DownsampledZipformer2Encoder
        # (downsample -> layers -> upsample -> bypass combine). The mask is subsampled by ds per stack.
        # The full-length axis is a constant chunk_size, so the upsample trim needs no shape read.
        meta = self._stack_meta[s]
        kpm_s = kpm if meta["ds"] == 1 else torch.index_select(kpm, 1, getattr(self, f"_kpm_idx_{s}"))
        if meta["is_downsampled"]:
            src_orig = x
            x = self._simple_downsample(meta, x)
            x, new_states = self._run_layers(s, meta, x, stack_states, kpm_s, N)
            x = self._simple_upsample(meta, x)[:self.chunk_size]
            return self._bypass(meta["module"].out_combiner, src_orig, x), new_states
        return self._run_layers(s, meta, x, stack_states, kpm_s, N)

    def _run_layers(self, s, meta, x, stack_states, kpm_s, N):
        gl0 = self._stack_gl_offset[s]
        new_states = []
        for l, layer in enumerate(self._layer_refs[s]):
            x, layer_new = self._encoder_layer(meta, gl0 + l, layer, x, stack_states[l * 6:(l + 1) * 6], kpm_s, N)
            new_states += layer_new
        return x, new_states

    def forward(self, audio, states):
        # audio: (N, 1, audio_chunk_samples) waveform chunk.  states: 6*19 caches + [embed_states, processed_lens].
        # Fully inlined streaming encoder: fbank -> subsampling -> 6 Zipformer2 stacks -> full-dim combine ->
        # output downsample -> encoder_proj (fused into joiner space).
        x = self._fbank(audio)                                        # (N, T=29, 80)
        N = x.shape[0]
        x, new_cached_embed_left_pad = self._subsample(x, states[-2])  # (N, 8, 192)
        kpm, new_processed_lens = self._build_mask(N, states[-1])     # (N, 104)
        x = x.transpose(0, 1)                                        # (T, N, C) time-major
        outputs = []
        new_states = []
        offset = 0
        for s in range(self._num_stacks):
            n6 = self._stack_meta[s]["n_layers"] * 6
            x = self._convert_channels(x, s, N)
            x, stack_new = self._stack_forward(s, x, states[offset:offset + n6], kpm, N)
            outputs.append(x)
            new_states += stack_new
            offset += n6
        x = self._downsample_output(self._full_dim(outputs))         # (T', N, 768)
        encoder_out = self.encoder_proj(x.transpose(0, 1))          # (N, T', 512) already in joiner space
        new_states += [new_cached_embed_left_pad, new_processed_lens]
        return encoder_out, new_states


class XasrDecoder(nn.Module):
    """Stateless predictor + the joiner.decoder_proj FUSED so the output is already in joiner space."""

    def __init__(self, decoder, decoder_proj):
        super().__init__()
        self.context_size = decoder.context_size
        self.decoder_dim = decoder.embedding.embedding_dim
        self._fold_embedding_conv(decoder)
        self.decoder_proj = decoder_proj

    def _fold_embedding_conv(self, decoder):
        assert self.context_size == CONTEXT_SIZE, (self.context_size, CONTEXT_SIZE)
        assert self.context_size > 1, self.context_size
        conv = decoder.conv
        assert conv.bias is None
        groups = conv.groups
        in_per_group = conv.in_channels // groups
        out_per_group = conv.out_channels // groups
        assert conv.in_channels == conv.out_channels == self.decoder_dim
        assert conv.kernel_size[0] == self.context_size
        embed = decoder.embedding.weight.detach().reshape(decoder.vocab_size, groups, in_per_group)
        weight = conv.weight.detach().reshape(groups, out_per_group, in_per_group, self.context_size)
        tables = torch.einsum("vgi,goik->kvgo", embed, weight).reshape(
            self.context_size, decoder.vocab_size, self.decoder_dim
        )
        self.register_buffer("context_table0", tables[0].contiguous(), persistent=False)
        self.register_buffer("context_table1", tables[1].contiguous(), persistent=False)

    def forward(self, y):
        # y: (N, context_size) int32 token ids.  context_size is fixed, so embedding+Conv1d is
        # pre-folded into one contribution table per context position.
        y0, y1 = torch.split(y, 1, dim=1)
        decoder_out = self.context_table0.index_select(0, y0.reshape(-1))
        decoder_out = decoder_out + self.context_table1.index_select(0, y1.reshape(-1))
        decoder_out = torch.relu(decoder_out)
        return self.decoder_proj(decoder_out)                          # (N, joiner_dim)


class XasrJoiner(nn.Module):
    """tanh(encoder_out + decoder_out) -> output_linear, with the greedy ARGMAX folded in so the
    graph emits an int32 token id per frame (no per-frame 5000-way logit transfer)."""

    def __init__(self, output_linear):
        super().__init__()
        self.output_linear = output_linear

    def forward(self, encoder_out, decoder_out):
        logit = self.output_linear(torch.tanh(encoder_out + decoder_out))   # (N, vocab)
        return logit.argmax(dim=-1).to(torch.int32)                         # (N,) int32 token id


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


# ============================================================================================
#                                          Export
# ============================================================================================
def _encoder_io_spec(init_states):
    """Build (input_names, output_names, dynamic_axes) for the 6-per-layer cache contract,
    matching the released sherpa naming (cached_key_i / cached_nonlin_attn_i / cached_val1_i /
    cached_val2_i / cached_conv1_i / cached_conv2_i, then embed_states / processed_lens)."""
    input_names = ["audio"]
    output_names = ["encoder_out"]
    dyn = {"audio": {0: "N"}, "encoder_out": {0: "N"}} if DYNAMIC_AXES else None
    # time-major caches carry batch on dim 1, conv caches carry batch on dim 0, nonlin on dim 1.
    per_layer = [("cached_key", 1), ("cached_nonlin_attn", 1), ("cached_val1", 1),
                 ("cached_val2", 1), ("cached_conv1", 0), ("cached_conv2", 0)]
    for i in range(NUM_LAYERS_TOTAL):
        for base, bdim in per_layer:
            name = f"{base}_{i}"
            input_names.append(name)
            output_names.append(f"new_{name}")
            if dyn is not None:
                dyn[name] = {bdim: "N"}
                dyn[f"new_{name}"] = {bdim: "N"}
    for name in ("embed_states", "processed_lens"):
        input_names.append(name)
        output_names.append(f"new_{name}")
        if dyn is not None:
            dyn[name] = {0: "N"}
            dyn[f"new_{name}"] = {0: "N"}
    return input_names, output_names, dyn


def _add_meta(path, meta):
    m = onnx.load(path, load_external_data=False)
    existing = {prop.key: prop for prop in m.metadata_props}
    for k, v in meta.items():
        key, value = str(k), str(v)
        if key in existing:
            existing[key].value = value
        else:
            m.metadata_props.add(key=key, value=value)
    onnx.save(m, path)


def export_all():
    encoder_embed, encoder, decoder, joiner = build_model()

    enc = XasrEncoder(encoder, encoder_embed, joiner.encoder_proj)
    dec = XasrDecoder(decoder, joiner.decoder_proj)
    joi = XasrJoiner(joiner.output_linear)
    # Fold trained scales into weights + drop every training-only op (Balancer / Whiten / Dropout /
    # ScaleGrad -> Identity).  is_onnx is intentionally OFF: for a FIXED chunk the relative-position
    # encoding is traced to constants (it depends only on the static chunk / left-context lengths),
    # which avoids a scripted submodule and pre-computes the whole Fourier position table.
    enc = convert_scaled_to_non_scaled(enc, inplace=True).eval()
    dec = convert_scaled_to_non_scaled(dec, inplace=True).eval()
    joi = convert_scaled_to_non_scaled(joi, inplace=True).eval()

    init_states = enc.get_init_states(1)
    in_names, out_names, dyn = _encoder_io_spec(init_states)

    left_context_len = [LEFT_CONTEXT_FRAMES // k for k in ARCH["downsampling_factor"]]
    common_meta = {
        "x_asr_metadata_version": 1,
        "producer": "Export_X_ASR.py",
        "sample_rate": SAMPLE_RATE,
        "chunk_ms": CHUNK_MS,
        "chunk_size": CHUNK_SIZE,
        "left_context_frames": LEFT_CONTEXT_FRAMES,
        "decode_chunk_len": CHUNK_SIZE * 2,
        "audio_chunk_samples": enc.audio_chunk_samples,
        "input_audio_dtype": INPUT_AUDIO_DTYPE,
        "num_mels": FEATURE_DIM,
        "nfft_stft": NFFT,
        "window_length": WINDOW_LENGTH,
        "hop_length": HOP_LENGTH,
        "pre_emphasis": PRE_EMPHASIS,
        "vocab_size": VOCAB_SIZE,
        "context_size": CONTEXT_SIZE,
        "decoder_dim": DECODER_DIM,
        "joiner_dim": JOINER_DIM,
        "blank_id": BLANK_ID,
        "num_layers_total": NUM_LAYERS_TOTAL,
    }
    enc_meta = {
        **common_meta,
        "model_type": "zipformer2", "version": "1", "model_author": "k2-fsa",
        "comment": "X-ASR streaming zipformer2 (encoder_proj fused)",
        "decode_chunk_len": CHUNK_SIZE * 2, "T": enc.T, "audio_chunk_samples": enc.audio_chunk_samples,
        "num_encoder_layers": ",".join(map(str, ARCH["num_encoder_layers"])),
        "encoder_dims": ",".join(map(str, ARCH["encoder_dim"])),
        "cnn_module_kernels": ",".join(map(str, ARCH["cnn_module_kernel"])),
        "left_context_len": ",".join(map(str, left_context_len)),
        "query_head_dims": ",".join(map(str, ARCH["query_head_dim"])),
        "value_head_dims": ",".join(map(str, ARCH["value_head_dim"])),
        "num_heads": ",".join(map(str, ARCH["num_heads"])),
    }

    with torch.inference_mode():
        metadata_marker = torch.zeros((1,), dtype=torch.int64)
        torch.onnx.export(
            METADATA_CARRIER(), (metadata_marker,), onnx_model_Metadata,
            input_names=["metadata_marker"], output_names=["metadata_marker_out"],
            dynamic_axes=None, opset_version=OPSET, dynamo=False,
        )
        _add_meta(onnx_model_Metadata, common_meta)
        del metadata_marker

        if INPUT_AUDIO_DTYPE == "INT16":
            audio = torch.randint(-32768, 32768, (1, 1, enc.audio_chunk_samples), dtype=torch.int16)
        else:
            audio = torch.randn(1, 1, enc.audio_chunk_samples, dtype=_audio_torch_dtype)
        print(f"\nExporting encoder ({CHUNK_MS}ms, audio_samples={enc.audio_chunk_samples}, T={enc.T}, {len(in_names)} inputs) ...")
        torch.onnx.export(
            enc, (audio, init_states), onnx_encoder,
            input_names=in_names, output_names=out_names, dynamic_axes=dyn,
            do_constant_folding=True, opset_version=OPSET, dynamo=False,
        )
        _add_meta(onnx_encoder, enc_meta)

        print("Exporting decoder ...")
        y = torch.zeros(1, CONTEXT_SIZE, dtype=torch.int32)
        torch.onnx.export(
            dec, (y,), onnx_decoder,
            input_names=["y"], output_names=["decoder_out"],
            dynamic_axes={"y": {0: "N"}, "decoder_out": {0: "N"}} if DYNAMIC_AXES else None,
            do_constant_folding=True, opset_version=OPSET, dynamo=False,
        )
        _add_meta(onnx_decoder, common_meta)

        print("Exporting joiner (argmax folded) ...")
        eo = torch.randn(1, JOINER_DIM, dtype=torch.float32)
        do = torch.randn(1, JOINER_DIM, dtype=torch.float32)
        torch.onnx.export(
            joi, (eo, do), onnx_joiner,
            input_names=["encoder_out", "decoder_out"], output_names=["max_token_id"],
            dynamic_axes={"encoder_out": {0: "N"}, "decoder_out": {0: "N"}, "max_token_id": {0: "N"}} if DYNAMIC_AXES else None,
            do_constant_folding=True, opset_version=OPSET, dynamo=False,
        )
        _add_meta(onnx_joiner, common_meta)

    print(f"\n[Metadata] Stamped {len(common_meta)} common keys into 4 ONNX graph(s):")
    for _key in sorted(common_meta):
        print(f"    {_key} = {common_meta[_key]}")

    del encoder_embed, encoder, decoder, joiner, enc, dec, joi
    gc.collect()
    return init_states


def run_exported_inference():
    print("\nExport done. Running ONNX Runtime demo via Inference_X_ASR_ONNX.py ...")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "Inference_X_ASR_ONNX.py"), "--onnx-folder", str(onnx_folder)],
        check=True,
    )


if __name__ == "__main__":
    print("\n===== X-ASR ONNX export =====")
    export_all()
    run_exported_inference()
