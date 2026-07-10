"""
Export NVIDIA Parakeet TDT 0.6B v3 ASR (Token-and-Duration Transducer, Fast-Conformer) to ONNX -- OFFLINE.

Standalone and NeMo-free: reads the HuggingFace `ParakeetForTDT` weights straight from ``model.safetensors``
and fuses the whole pipeline into a single-pass, ONNX Runtime-friendly graph set. Everything is precomputed
in ``__init__`` -- LayerNorm affines are folded into the following linears, Q/K/V are fused into one GEMM,
BatchNorm is folded to a per-channel affine, the relative-position projection is baked per layer, and the
mel front-end (pre-emphasis + STFT + librosa mel + per-feature normalization) lives inside the encoder.

Three graphs are produced (mirroring the Nemotron ASR export layout):

  ASR_Matadata.onnx           marker -> marker                (carries all runtime metadata)
  Parakeet_ASR_Encoder.onnx   audio -> enc_proj               (mel + Fast-Conformer + encoder projector)
  Parakeet_ASR_Decoder.onnx   enc_proj + frame_idx + token + state -> next_token, is_blank, duration, state

The exported decoder/joint reproduces the HuggingFace TDT greedy step exactly: it always runs the LSTM and
keeps the previous state on a blank emission (equivalent to the reference decoder cache-skip), and emits a
per-step duration so the reused greedy loop can advance the encoder frame pointer.
"""

import gc
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

# Configuration
_SCRIPT_DIR    = Path(__file__).resolve().parent
_DOWNLOADS     = Path("/home/DakeQQ/Downloads")

_MODEL_DIR_NAME = "parakeet-tdt-0.6b-v3"              # HuggingFace ParakeetForTDT snapshot folder name.


def _resolve_model_path() -> Path:
    """Find the model snapshot folder across the likely sibling/parent locations."""
    candidates = [
        _DOWNLOADS / _MODEL_DIR_NAME,
        _DOWNLOADS.parent / _MODEL_DIR_NAME,
        _DOWNLOADS.parent.parent / _MODEL_DIR_NAME,
        _SCRIPT_DIR / _MODEL_DIR_NAME,
    ]
    for cand in candidates:
        if (cand / "config.json").exists():
            return cand
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not locate '{_MODEL_DIR_NAME}'. Looked in:\n  {tried}")


MODEL_PATH = _resolve_model_path()

OPSET             = 20             # >=17 for fused LayerNormalization.
INPUT_AUDIO_DTYPE = "INT16"        # "INT16" (raw PCM, graph divides by 32768) | "F32" | "F16".
DO_EXPORT         = True           # Run the ONNX export.
DYNAMIC_AXES      = True           # True keeps audio length dynamic; False bakes a fixed length.
FIXED_INPUT_AUDIO_SECONDS = 10.0   # Used only when DYNAMIC_AXES is False.
# Longest utterance (in encoder frames) the baked relative-position table supports. One encoder frame is
# subsampling_factor * hop / sample_rate = 80 ms, so 1536 frames ~= 123 s. Raise for longer offline audio.
PE_MAX_LEN        = 1536

ONNX_FOLDER   = _SCRIPT_DIR / "Parakeet_ASR_ONNX"
METADATA_NAME = "ASR_Matadata.onnx"
ENCODER_NAME  = "Parakeet_ASR_Encoder.onnx"
DECODER_NAME  = "Parakeet_ASR_Decoder.onnx"

INFERENCE_SCRIPT = _SCRIPT_DIR / "Inference_Parakeet_ASR_ONNX.py"

# Tokenizer / config side files copied next to the graphs so inference is self-contained.
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# Front-end constants (processor_config.json / feature_extraction_parakeet.py).
LOG_GUARD = 2.0 ** -24            # log_zero_guard_value
NORM_EPS  = 1e-5                  # per-feature normalization epsilon (EPSILON)
LN_EPS    = 1e-5                  # Conformer LayerNorm epsilon
BN_EPS    = 1e-5                  # BatchNorm1d epsilon (nn.BatchNorm1d default)


# Model geometry (config.json)
def _read_config() -> dict:
    return json.loads((MODEL_PATH / "config.json").read_text())


def _read_processor_config() -> dict:
    return json.loads((MODEL_PATH / "processor_config.json").read_text())


_CFG      = _read_config()
_ENC_CFG  = _CFG["encoder_config"]
_FEAT_CFG = _read_processor_config()["feature_extractor"]

SAMPLE_RATE = int(_FEAT_CFG["sampling_rate"])
N_MELS      = int(_FEAT_CFG["feature_size"])
N_FFT       = int(_FEAT_CFG["n_fft"])
WIN_LENGTH  = int(_FEAT_CFG["win_length"])
HOP_LENGTH  = int(_FEAT_CFG["hop_length"])
PREEMPH     = float(_FEAT_CFG["preemphasis"])
FMIN        = 0.0
FMAX        = SAMPLE_RATE / 2.0

D_MODEL      = int(_ENC_CFG["hidden_size"])
N_LAYERS     = int(_ENC_CFG["num_hidden_layers"])
N_HEADS      = int(_ENC_CFG["num_attention_heads"])
HEAD_DIM     = D_MODEL // N_HEADS
D_FF         = int(_ENC_CFG["intermediate_size"])
CONV_KERNEL  = int(_ENC_CFG["conv_kernel_size"])
CONV_PAD     = (CONV_KERNEL - 1) // 2
SUB_FACTOR   = int(_ENC_CFG["subsampling_factor"])
SUB_CHANNELS = int(_ENC_CFG["subsampling_conv_channels"])
SUB_KERNEL   = int(_ENC_CFG["subsampling_conv_kernel_size"])
SUB_STRIDE   = int(_ENC_CFG["subsampling_conv_stride"])
SUB_PAD      = (SUB_KERNEL - 1) // 2
SUB_LAYERS   = int(round(math.log2(SUB_FACTOR)))
ENC_ACT      = _ENC_CFG.get("hidden_act", "silu")

DEC_HIDDEN   = int(_CFG["decoder_hidden_size"])
LSTM_LAYERS  = int(_CFG["num_decoder_layers"])
VOCAB_SIZE   = int(_CFG["vocab_size"])
BLANK_ID     = int(_CFG["blank_token_id"])
DURATIONS    = list(_CFG["durations"])
NUM_DURATION = len(DURATIONS)
LOGITS_SIZE  = VOCAB_SIZE + NUM_DURATION
MAX_SYMBOLS  = int(_CFG["max_symbols_per_step"])
JOINT_ACT    = _CFG.get("hidden_act", "relu")

_FRAME_MS    = SUB_FACTOR * HOP_LENGTH / SAMPLE_RATE * 1000.0

_AUDIO_TORCH_DTYPE = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
FIXED_INPUT_AUDIO_LENGTH = int(round(FIXED_INPUT_AUDIO_SECONDS * SAMPLE_RATE))


# Metadata helpers
def _compact_json(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def build_model_metadata(*sections):
    metadata = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (dict, list, tuple)):
                metadata[str(key)] = _compact_json(value)
            else:
                metadata[str(key)] = str(value)
    return metadata


def finalize_graph(onnx_path: Path, metadata: dict | None = None) -> None:
    """Merge torch external sidecars into one data file and optionally stamp metadata."""
    import onnx

    model = onnx.load(str(onnx_path))
    if metadata:
        existing = {prop.key: prop for prop in model.metadata_props}
        for key, value in metadata.items():
            if key in existing:
                existing[key].value = value
            else:
                model.metadata_props.add(key=key, value=value)
    for sidecar in onnx_path.parent.glob("*Constant_*_attr__value"):
        sidecar.unlink()
    data_name = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save(model, str(onnx_path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=data_name, size_threshold=1024, convert_attribute=True)


# Fused LayerNormalization op.
class _LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xf, scale, bias, epsilon, axis):
        mean = xf.mean(dim=axis, keepdim=True)
        xc = xf - mean
        var = xc.pow(2).mean(dim=axis, keepdim=True)
        y = xc * torch.rsqrt(var + epsilon) * scale
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def symbolic(g, x, scale, bias, epsilon, axis):
        if bias is None:
            return g.op("LayerNormalization", x, scale, axis_i=axis, epsilon_f=epsilon, stash_type_i=1)
        return g.op("LayerNormalization", x, scale, bias, axis_i=axis, epsilon_f=epsilon, stash_type_i=1)


def layer_norm(x, scale, bias=None, epsilon=LN_EPS, axis=-1):
    return _LAYER_NORM.apply(x, scale, bias, float(epsilon), axis)


def silu(x):
    return x * torch.sigmoid(x)


class MetadataCarrier(nn.Module):
    def forward(self, marker: Tensor) -> Tensor:
        return marker


# Single-pass encoder graph: mel front-end + Fast-Conformer + encoder projector.
class ParakeetEncoder(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        g = lambda k: sd[k].float()

        # ---- mel front-end (pre-emphasis + STFT + librosa mel + log) ----
        import librosa

        window = torch.hann_window(WIN_LENGTH, periodic=False)
        pad_total = N_FFT - WIN_LENGTH
        pad_l = pad_total // 2
        win = torch.cat([torch.zeros(pad_l), window, torch.zeros(pad_total - pad_l)])
        f = torch.arange(N_FFT // 2 + 1, dtype=torch.float32).unsqueeze(1)
        t = torch.arange(N_FFT, dtype=torch.float32).unsqueeze(0)
        omega = (2.0 * math.pi / N_FFT) * f * t
        cos_k = (torch.cos(omega) * win.unsqueeze(0)).unsqueeze(1)
        sin_k = (-torch.sin(omega) * win.unsqueeze(0)).unsqueeze(1)
        self.register_buffer("stft_kernel", torch.cat([cos_k, sin_k], dim=0), persistent=False)
        mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, norm="slaney")
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).to(torch.float32).contiguous(), persistent=False)
        self.register_buffer("inv_int16", torch.tensor(1.0 / 32768.0, dtype=torch.float32), persistent=False)
        self.register_buffer("preemph", torch.tensor(PREEMPH, dtype=torch.float32), persistent=False)
        self.register_buffer("log_guard", torch.tensor(LOG_GUARD, dtype=torch.float32), persistent=False)
        self.register_buffer("norm_eps", torch.tensor(NORM_EPS, dtype=torch.float32), persistent=False)
        self.f_bins = N_FFT // 2 + 1

        # ---- subsampling (Conv2D, symmetric padding, factor SUB_FACTOR) ----
        self.register_buffer("sub0_w", g("encoder.subsampling.layers.0.weight"), persistent=False)
        self.register_buffer("sub0_b", g("encoder.subsampling.layers.0.bias"), persistent=False)
        self.register_buffer("sub2_w", g("encoder.subsampling.layers.2.weight"), persistent=False)
        self.register_buffer("sub2_b", g("encoder.subsampling.layers.2.bias"), persistent=False)
        self.register_buffer("sub3_w", g("encoder.subsampling.layers.3.weight"), persistent=False)
        self.register_buffer("sub3_b", g("encoder.subsampling.layers.3.bias"), persistent=False)
        self.register_buffer("sub5_w", g("encoder.subsampling.layers.5.weight"), persistent=False)
        self.register_buffer("sub5_b", g("encoder.subsampling.layers.5.bias"), persistent=False)
        self.register_buffer("sub6_w", g("encoder.subsampling.layers.6.weight"), persistent=False)
        self.register_buffer("sub6_b", g("encoder.subsampling.layers.6.bias"), persistent=False)
        self.register_buffer("sub_lin_w", g("encoder.subsampling.linear.weight"), persistent=False)
        self.register_buffer("sub_lin_b", g("encoder.subsampling.linear.bias"), persistent=False)

        self.register_buffer("ln_ones", torch.ones(D_MODEL), persistent=False)
        inv_sqrt_dk = HEAD_DIM ** -0.5

        # ---- relative positional projection, precomputed per layer (Parakeet inv_freq interleaved) ----
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D_MODEL, 2, dtype=torch.float32) / D_MODEL))
        positions = torch.arange(PE_MAX_LEN - 1, -PE_MAX_LEN, -1, dtype=torch.float32)
        freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)                      # (2P-1, D/2)
        pos_embed = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(freqs.shape[0], D_MODEL)  # interleaved
        self.pe_center = PE_MAX_LEN

        for l in range(N_LAYERS):
            p = f"encoder.layers.{l}."

            def ln(name):
                return g(p + name + ".weight"), g(p + name + ".bias")

            # feed_forward1: fold norm affine into linear1, fold 0.5 residual into linear2 (no bias in FF).
            gw, gb = ln("norm_feed_forward1")
            l1w = g(p + "feed_forward1.linear1.weight")
            l2w = g(p + "feed_forward1.linear2.weight")
            self.register_buffer(f"ff1_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous(), persistent=False)
            self.register_buffer(f"ff1_l1b_{l}", torch.matmul(l1w, gb), persistent=False)
            self.register_buffer(f"ff1_l2w_{l}", (l2w * 0.5).contiguous(), persistent=False)

            # self_attn: fuse QKV, fold norm affine, fold attention scale into q + bias_u/bias_v.
            gw, gb = ln("norm_self_att")
            qw = g(p + "self_attn.q_proj.weight")
            kw = g(p + "self_attn.k_proj.weight")
            vw = g(p + "self_attn.v_proj.weight")
            qkv_w_orig = torch.cat([qw, kw, vw], dim=0)
            qkv_b = torch.matmul(qkv_w_orig, gb).clone()
            qkv_w = (qkv_w_orig * gw.unsqueeze(0)).clone()
            qkv_w[:D_MODEL] *= inv_sqrt_dk
            qkv_b[:D_MODEL] *= inv_sqrt_dk
            self.register_buffer(f"qkv_w_{l}", qkv_w.contiguous(), persistent=False)
            self.register_buffer(f"qkv_b_{l}", qkv_b.contiguous(), persistent=False)
            # Bake relative_k_proj over the whole position table -> (1, H, HEAD_DIM, 2P-1).
            rel_k = F.linear(pos_embed, g(p + "self_attn.relative_k_proj.weight"))
            rel_k = rel_k.reshape(-1, N_HEADS, HEAD_DIM).permute(1, 2, 0).contiguous()
            self.register_buffer(f"pos_proj_{l}", rel_k.unsqueeze(0).half(), persistent=False)
            self.register_buffer(f"bias_u_{l}", (g(p + "self_attn.bias_u") * inv_sqrt_dk).unsqueeze(1).contiguous(), persistent=False)
            self.register_buffer(f"bias_v_{l}", (g(p + "self_attn.bias_v") * inv_sqrt_dk).unsqueeze(1).contiguous(), persistent=False)
            self.register_buffer(f"out_w_{l}", g(p + "self_attn.o_proj.weight").contiguous(), persistent=False)

            # conv: fold norm_conv into pointwise_conv1; fold BatchNorm (running stats) into a per-channel affine.
            gw, gb = ln("norm_conv")
            pw1 = g(p + "conv.pointwise_conv1.weight").squeeze(-1)
            self.register_buffer(f"pw1_w_{l}", (pw1 * gw.unsqueeze(0)).contiguous(), persistent=False)
            self.register_buffer(f"pw1_b_{l}", torch.matmul(pw1, gb), persistent=False)
            self.register_buffer(f"dw_w_{l}", g(p + "conv.depthwise_conv.weight").contiguous(), persistent=False)
            bn_w = g(p + "conv.norm.weight")
            bn_b = g(p + "conv.norm.bias")
            bn_rm = g(p + "conv.norm.running_mean")
            bn_rv = g(p + "conv.norm.running_var")
            bn_scale = bn_w / torch.sqrt(bn_rv + BN_EPS)
            bn_shift = bn_b - bn_rm * bn_scale
            self.register_buffer(f"bn_scale_{l}", bn_scale.contiguous(), persistent=False)
            self.register_buffer(f"bn_shift_{l}", bn_shift.contiguous(), persistent=False)
            self.register_buffer(f"pw2_w_{l}", g(p + "conv.pointwise_conv2.weight").squeeze(-1).contiguous(), persistent=False)

            # feed_forward2: same folding as feed_forward1.
            gw, gb = ln("norm_feed_forward2")
            l1w = g(p + "feed_forward2.linear1.weight")
            l2w = g(p + "feed_forward2.linear2.weight")
            self.register_buffer(f"ff2_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous(), persistent=False)
            self.register_buffer(f"ff2_l1b_{l}", torch.matmul(l1w, gb), persistent=False)
            self.register_buffer(f"ff2_l2w_{l}", (l2w * 0.5).contiguous(), persistent=False)

            ow, ob = ln("norm_out")
            self.register_buffer(f"no_w_{l}", ow, persistent=False)
            self.register_buffer(f"no_b_{l}", ob, persistent=False)

        # encoder projector -> enc_proj (joint "enc" side, projects D_MODEL -> DEC_HIDDEN).
        self.register_buffer("enc_proj_w", g("encoder_projector.weight").contiguous(), persistent=False)
        self.register_buffer("enc_proj_b", g("encoder_projector.bias").contiguous(), persistent=False)

    def _preprocess(self, audio):
        # Float inputs are assumed normalized; INT16 is scaled to [-1, 1).
        if INPUT_AUDIO_DTYPE == "INT16":
            x = audio.float() * self.inv_int16
        else:
            x = audio.float()
        # Pre-emphasis over the full signal (keep x[0]).
        x = torch.cat([x[:, :, :1], x[:, :, 1:] - self.preemph * x[:, :, :-1]], dim=2)
        # torch.stft(center=True) -> pad n_fft//2 each side with zeros, then framed DFT.
        x = F.pad(x, (N_FFT // 2, N_FFT // 2))
        stft = F.conv1d(x, self.stft_kernel, stride=HOP_LENGTH)
        real, imag = torch.split(stft, self.f_bins, dim=1)
        power = real * real + imag * imag
        # Keep mel_fb on the left; Optimize_ONNX.py skips FusionGemm to avoid onnxslim's bad const@var rewrite.
        mel = torch.matmul(self.mel_fb, power)
        feats = torch.log(mel + self.log_guard)              # (B, N_MELS, T_full)
        # Drop the trailing STFT frame (valid = floor(L/hop) = T_full - 1) and per-feature normalize.
        valid = feats[:, :, :-1]
        n = valid.shape[2]
        mean = valid.mean(dim=2, keepdim=True)
        var = (valid - mean).pow(2).sum(dim=2, keepdim=True) / (n - 1)
        std = torch.sqrt(var)
        normed = (valid - mean) / (std + self.norm_eps)
        return normed.transpose(1, 2)                        # (B, N_valid, N_MELS)

    def _subsample(self, feats):
        x = feats.unsqueeze(1)                               # (B, 1, T, N_MELS)
        x = F.relu(F.conv2d(x, self.sub0_w, self.sub0_b, stride=SUB_STRIDE, padding=SUB_PAD))
        x = F.conv2d(x, self.sub2_w, self.sub2_b, stride=SUB_STRIDE, padding=SUB_PAD, groups=SUB_CHANNELS)
        x = F.relu(F.conv2d(x, self.sub3_w, self.sub3_b))
        x = F.conv2d(x, self.sub5_w, self.sub5_b, stride=SUB_STRIDE, padding=SUB_PAD, groups=SUB_CHANNELS)
        x = F.relu(F.conv2d(x, self.sub6_w, self.sub6_b))
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return F.linear(x, self.sub_lin_w, self.sub_lin_b)   # (B, S, D_MODEL)

    @staticmethod
    def _rel_shift(x, S):
        # Transformer-XL skew; input (B, H, S, 2S-1) -> (B, H, S, 2S-1), caller slices [..., :S].
        b, h = x.shape[0], x.shape[1]
        x = F.pad(x, (1, 0))
        x = x.reshape(b, h, -1, S)
        x = x[:, :, 1:]
        x = x.reshape(b, h, S, 2 * S - 1)
        return x

    def forward(self, audio):
        x = self._subsample(self._preprocess(audio))
        batch_size = x.shape[0]
        S = x.shape[1]

        for l in range(N_LAYERS):
            residual = x
            m = layer_norm(x, self.ln_ones)
            m = F.linear(m, getattr(self, f"ff1_l1w_{l}"), getattr(self, f"ff1_l1b_{l}"))
            m = silu(m)
            m = F.linear(m, getattr(self, f"ff1_l2w_{l}"))
            residual = residual + m

            m = layer_norm(residual, self.ln_ones)
            qkv = F.linear(m, getattr(self, f"qkv_w_{l}"), getattr(self, f"qkv_b_{l}"))
            qkv = qkv.reshape(batch_size, -1, 3 * N_HEADS, HEAD_DIM).transpose(1, 2)
            q, k, v = torch.split(qkv, N_HEADS, dim=1)
            q_u = q + getattr(self, f"bias_u_{l}")
            q_v = q + getattr(self, f"bias_v_{l}")
            k_t = k.transpose(2, 3)
            p_t = getattr(self, f"pos_proj_{l}")[..., self.pe_center - S: self.pe_center + S - 1].float()
            ac = torch.matmul(q_u, k_t)
            bd = torch.matmul(q_v, p_t)
            bd = self._rel_shift(bd, S)[..., :S]
            scores = ac + bd
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)
            ctx = ctx.transpose(1, 2).reshape(batch_size, -1, D_MODEL)
            m = F.linear(ctx, getattr(self, f"out_w_{l}"))
            residual = residual + m

            m = layer_norm(residual, self.ln_ones)
            xc = F.linear(m, getattr(self, f"pw1_w_{l}"), getattr(self, f"pw1_b_{l}"))
            xc = xc.transpose(1, 2)
            xc = F.glu(xc, dim=1)
            xc = F.conv1d(xc, getattr(self, f"dw_w_{l}"), padding=CONV_PAD, groups=D_MODEL)
            xc = xc.transpose(1, 2)
            xc = xc * getattr(self, f"bn_scale_{l}") + getattr(self, f"bn_shift_{l}")
            xc = silu(xc)
            xc = F.linear(xc, getattr(self, f"pw2_w_{l}"))
            residual = residual + xc

            m = layer_norm(residual, self.ln_ones)
            m = F.linear(m, getattr(self, f"ff2_l1w_{l}"), getattr(self, f"ff2_l1b_{l}"))
            m = silu(m)
            m = F.linear(m, getattr(self, f"ff2_l2w_{l}"))
            residual = residual + m

            x = layer_norm(residual, getattr(self, f"no_w_{l}"), getattr(self, f"no_b_{l}"))

        enc_proj = F.linear(x, self.enc_proj_w, self.enc_proj_b)
        return enc_proj


# Folded TDT decoder + joint + greedy step.
class ParakeetDecoderJoint(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        self.blank_id = BLANK_ID
        self.vocab_size = VOCAB_SIZE
        self.embed = nn.Embedding(VOCAB_SIZE, DEC_HIDDEN)
        self.lstm = nn.LSTM(DEC_HIDDEN, DEC_HIDDEN, LSTM_LAYERS, batch_first=True)
        with torch.no_grad():
            self.embed.weight.copy_(sd["decoder.embedding.weight"].float())
            for li in range(LSTM_LAYERS):
                getattr(self.lstm, f"weight_ih_l{li}").copy_(sd[f"decoder.lstm.weight_ih_l{li}"].float())
                getattr(self.lstm, f"weight_hh_l{li}").copy_(sd[f"decoder.lstm.weight_hh_l{li}"].float())
                getattr(self.lstm, f"bias_ih_l{li}").copy_(sd[f"decoder.lstm.bias_ih_l{li}"].float())
                getattr(self.lstm, f"bias_hh_l{li}").copy_(sd[f"decoder.lstm.bias_hh_l{li}"].float())
        self.register_buffer("dec_proj_w", sd["decoder.decoder_projector.weight"].float().contiguous(), persistent=False)
        self.register_buffer("dec_proj_b", sd["decoder.decoder_projector.bias"].float().contiguous(), persistent=False)
        self.register_buffer("head_w", sd["joint.head.weight"].float().contiguous(), persistent=False)
        self.register_buffer("head_b", sd["joint.head.bias"].float().contiguous(), persistent=False)
        self.register_buffer("durations", torch.tensor(DURATIONS, dtype=torch.int32), persistent=False)

    def forward(self, enc_proj, frame_idx, token, state_h, state_c):
        enc_frame = torch.index_select(enc_proj, 1, frame_idx)
        emb = self.embed(token)
        out, (h, c) = self.lstm(emb, (state_h, state_c))
        dec_out = F.linear(out, self.dec_proj_w, self.dec_proj_b)
        z = torch.relu(enc_frame + dec_out)
        logits = F.linear(z, self.head_w, self.head_b)
        token_logits, dur_logits = torch.split(logits, [self.vocab_size, NUM_DURATION], dim=-1)
        argmax = torch.argmax(token_logits, dim=-1).to(torch.int32)
        dur_idx = torch.argmax(dur_logits, dim=-1)
        duration = torch.index_select(self.durations, 0, dur_idx.reshape(-1)).reshape(dur_idx.shape)
        is_blank = argmax == self.blank_id
        # Blank steps keep token/state unchanged (matches the reference decoder cache-skip); force forward
        # progress by advancing at least one frame on a blank that predicted duration 0.
        next_token = torch.where(is_blank, token, argmax)
        duration = torch.where(is_blank & (duration == 0), torch.ones_like(duration), duration)
        h = torch.where(is_blank, state_h, h)
        c = torch.where(is_blank, state_c, c)
        return next_token, is_blank.to(torch.int32), duration.to(torch.int32), h, c


# Metadata
def make_metadata() -> dict:
    decoder_state_specs = [
        {"name": "state_h", "shape": [LSTM_LAYERS, "B", DEC_HIDDEN], "dtype": "float32"},
        {"name": "state_c", "shape": [LSTM_LAYERS, "B", DEC_HIDDEN], "dtype": "float32"},
    ]
    header = {
        "parakeet_asr_metadata_version": 1,
        "producer": Path(__file__).name,
        "model_type": _CFG.get("model_type", "parakeet_tdt"),
        "sample_rate": SAMPLE_RATE,
        "input_audio_dtype": INPUT_AUDIO_DTYPE,
        "opset": OPSET,
        "dynamic_axes": DYNAMIC_AXES,
        "fixed_input_audio_length": 0 if DYNAMIC_AXES else FIXED_INPUT_AUDIO_LENGTH,
        "tokenizer_files": list(TOKENIZER_FILES),
        "frame_ms": round(_FRAME_MS, 4),
    }
    feature_section = {
        "num_mels": N_MELS, "n_fft": N_FFT, "window_length": WIN_LENGTH, "hop_length": HOP_LENGTH,
        "preemph": PREEMPH, "log_guard": LOG_GUARD, "norm_eps": NORM_EPS,
        "normalize": "per_feature", "mag_power": 2.0, "mel_norm": "slaney", "fmin": FMIN, "fmax": FMAX,
    }
    encoder_section = {
        "encoder_layers": N_LAYERS, "encoder_d_model": D_MODEL, "encoder_heads": N_HEADS,
        "head_dim": HEAD_DIM, "d_ff": D_FF, "conv_kernel_size": CONV_KERNEL,
        "subsampling_factor": SUB_FACTOR, "subsampling_conv_channels": SUB_CHANNELS,
        "subsampling_conv_kernel_size": SUB_KERNEL, "subsampling_conv_stride": SUB_STRIDE,
        "encoder_activation": ENC_ACT, "pe_max_len": PE_MAX_LEN, "enc_proj_dim": DEC_HIDDEN,
    }
    decoder_section = {
        "decoder_pred_hidden": DEC_HIDDEN, "decoder_layers": LSTM_LAYERS, "joint_activation": JOINT_ACT,
        "vocab_size": VOCAB_SIZE, "logits_vocab_size": LOGITS_SIZE, "blank_id": BLANK_ID,
        "durations": DURATIONS, "num_durations": NUM_DURATION, "max_symbols": MAX_SYMBOLS,
        "decoder_uses_frame_idx": True, "decoder_state_specs": decoder_state_specs,
    }
    return build_model_metadata(header, feature_section, encoder_section, decoder_section)


# Export driver
def _copy_side_files() -> None:
    for name in TOKENIZER_FILES:
        src = MODEL_PATH / name
        if src.exists():
            (ONNX_FOLDER / name).write_bytes(src.read_bytes())


def export_all():
    ONNX_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Parakeet TDT ASR export -> {ONNX_FOLDER}")
    encoder = decjoint = sd = None
    try:
        sd = load_file(str(MODEL_PATH / "model.safetensors"))
        metadata = make_metadata()

        encoder = ParakeetEncoder(sd).eval()
        decjoint = ParakeetDecoderJoint(sd).eval()
        _copy_side_files()

        with torch.inference_mode():
            if DO_EXPORT:
                p = ONNX_FOLDER / METADATA_NAME
                torch.onnx.export(MetadataCarrier().eval(), (torch.zeros(1, dtype=torch.int64),), str(p),
                                  input_names=["metadata_marker"], output_names=["metadata_marker_out"],
                                  opset_version=OPSET, dynamo=False)
                finalize_graph(p, metadata)

                p = ONNX_FOLDER / ENCODER_NAME
                audio = torch.zeros(1, 1, FIXED_INPUT_AUDIO_LENGTH, dtype=_AUDIO_TORCH_DTYPE)
                enc_axes = {"audio": {0: "batch", 2: "num_samples"},
                            "enc_proj": {0: "batch", 1: "enc_frames"}} if DYNAMIC_AXES else None
                torch.onnx.export(encoder, (audio,), str(p),
                                  input_names=["audio"], output_names=["enc_proj"],
                                  dynamic_axes=enc_axes, opset_version=OPSET, dynamo=False)
                finalize_graph(p)

                p = ONNX_FOLDER / DECODER_NAME
                if DYNAMIC_AXES:
                    ep = torch.randn(1, 8, DEC_HIDDEN)
                else:
                    ep = encoder(audio)
                frame_idx = torch.zeros(1, dtype=torch.int32)
                tok = torch.zeros(1, 1, dtype=torch.int32)
                sh = torch.zeros(LSTM_LAYERS, 1, DEC_HIDDEN)
                sc = torch.zeros(LSTM_LAYERS, 1, DEC_HIDDEN)
                dec_axes = {"enc_proj": {0: "batch", 1: "enc_frames"},
                            "token": {0: "batch"}, "state_h": {1: "batch"}, "state_c": {1: "batch"},
                            "next_token": {0: "batch"}, "is_blank": {0: "batch"}, "duration": {0: "batch"},
                            "state_h_next": {1: "batch"}, "state_c_next": {1: "batch"}} if DYNAMIC_AXES else None
                torch.onnx.export(decjoint, (ep, frame_idx, tok, sh, sc), str(p),
                                  input_names=["enc_proj", "frame_idx", "token", "state_h", "state_c"],
                                  output_names=["next_token", "is_blank", "duration", "state_h_next", "state_c_next"],
                                  dynamic_axes=dec_axes, opset_version=OPSET, dynamo=False)
                finalize_graph(p)

                print(f"Stamped {len(metadata)} metadata keys. Export complete.")

        print("\n" + "-" * 70)
        print(f"Running ONNX Runtime demo via {INFERENCE_SCRIPT.name} ...\n")
        subprocess.run([sys.executable, str(INFERENCE_SCRIPT), "--onnx-folder", str(ONNX_FOLDER)], check=True)
    finally:
        del encoder, decjoint, sd
        gc.collect()


if __name__ == "__main__":
    export_all()
