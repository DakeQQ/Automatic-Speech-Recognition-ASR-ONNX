"""
Export NVIDIA Parakeet TDT 0.6B v3 ASR (Token-and-Duration Transducer, Fast-Conformer) to ONNX -- OFFLINE.

Standalone and NeMo-free: reads the HuggingFace `ParakeetForTDT` weights straight from ``model.safetensors``
and fuses the whole pipeline into a single-pass, ONNX Runtime-friendly graph set. Everything is precomputed
in ``__init__`` -- LayerNorm affines are folded into the following linears, Q/K/V are fused into one GEMM,
BatchNorm is folded into depthwise convolution, the relative-position projection is baked per layer, and the
mel front-end (pre-emphasis + STFT + librosa mel + per-feature normalization) lives inside the encoder.

Three graphs are produced (mirroring the Nemotron ASR export layout):

  ASR_Metadata.onnx           marker -> marker                (carries all runtime metadata)
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

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import Tensor, nn

# Configuration
_SCRIPT_DIR    = Path(__file__).resolve().parent
_DOWNLOADS     = Path.home() / "Downloads"

_MODEL_DIR_NAME = "parakeet-tdt-0.6b-v3"              # HuggingFace ParakeetForTDT snapshot folder name.


MODEL_PATH = _DOWNLOADS / _MODEL_DIR_NAME

OPSET             = 20             # >=17 for fused LayerNormalization.
INPUT_AUDIO_DTYPE = "F32"          # "INT16" (raw PCM, graph divides by 32768) | "F32" | "F16".
DYNAMIC_AXES      = True           # True keeps audio length dynamic; False bakes a fixed length.
FIXED_INPUT_AUDIO_SECONDS = 10.0   # Used only when DYNAMIC_AXES is False.
# Longest utterance (in encoder frames) the baked relative-position table supports. One encoder frame is
# subsampling_factor * hop / sample_rate = 80 ms, so 1536 frames ~= 123 s. Raise for longer offline audio.
PE_MAX_LEN        = 1536

ONNX_FOLDER   = _SCRIPT_DIR / "Parakeet_ASR_ONNX"
METADATA_NAME = "ASR_Metadata.onnx"
ENCODER_NAME  = "Parakeet_ASR_Encoder.onnx"
DECODER_NAME  = "Parakeet_ASR_Decoder.onnx"

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
JOINT_ACT    = _CFG.get("hidden_act", "relu")


_AUDIO_TORCH_DTYPE = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
FIXED_INPUT_AUDIO_LENGTH = int(round(FIXED_INPUT_AUDIO_SECONDS * SAMPLE_RATE))
AUDIO_PCM_SCALE = 32768
MAX_SYMBOLS_PER_STEP = int(_CFG["max_symbols_per_step"])
MAX_AUDIO_SAMPLES = (
    FIXED_INPUT_AUDIO_LENGTH
    if not DYNAMIC_AXES
    else PE_MAX_LEN * SUB_FACTOR * HOP_LENGTH
)


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


def write_metadata_carrier(onnx_path, metadata):
    """Replace the ASR metadata carrier properties."""
    import onnx

    expected = {str(key): str(value) for key, value in metadata.items()}
    model = onnx.load(str(onnx_path), load_external_data=False)
    del model.metadata_props[:]
    for key in sorted(expected):
        model.metadata_props.add(key=key, value=expected[key])
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)


def _iter_graph_tensors(graph):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _iter_graph_tensors(attr.g)
            for nested_graph in attr.graphs:
                yield from _iter_graph_tensors(nested_graph)


def finalize_graph(
    onnx_path: Path,
    metadata: dict | None = None,
    *,
    replace_metadata: bool = False,
) -> None:
    """Merge torch external sidecars into one data file and optionally stamp metadata."""
    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    raw_sidecars = {
        entry.value
        for tensor in _iter_graph_tensors(model.graph)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
    onnx.load_external_data_for_model(model, str(onnx_path.parent))
    if replace_metadata:
        del model.metadata_props[:]
    if metadata:
        existing = {prop.key: prop for prop in model.metadata_props}
        for key, value in metadata.items():
            if key in existing:
                existing[key].value = value
            else:
                model.metadata_props.add(key=key, value=value)
    data_name = onnx_path.name + ".data"
    data_path = onnx_path.parent / data_name
    if data_path.exists():
        data_path.unlink()
    onnx.save(model, str(onnx_path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=data_name, size_threshold=1024, convert_attribute=True)
    merged_data_path = data_path.resolve()
    for location in raw_sidecars:
        sidecar = (onnx_path.parent / location).resolve()
        if sidecar != merged_data_path and sidecar.is_file():
            sidecar.unlink()


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


class _PAD_LAST_DIM_LEFT_ONE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pads):
        return F.pad(x, (1, 0))

    @staticmethod
    def symbolic(g, x, pads):
        return g.op("Pad", x, pads, mode_s="constant")


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
        self.register_buffer("stft_kernel", torch.cat([cos_k, sin_k], dim=0))
        preemph_scale = 1.0 / 32768.0 if INPUT_AUDIO_DTYPE == "INT16" else 1.0
        preemph_kernel = torch.tensor([-PREEMPH, 1.0, 0.0], dtype=torch.float32) * preemph_scale
        self.register_buffer("preemph_kernel", preemph_kernel.reshape(1, 1, 3))
        mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, norm="slaney")
        self.register_buffer("mel_fb", torch.from_numpy(mel_fb).to(torch.float32).contiguous())
        self.register_buffer("log_guard", torch.tensor(LOG_GUARD, dtype=torch.float32))
        self.register_buffer("norm_eps", torch.tensor(NORM_EPS, dtype=torch.float32))
        self.f_bins = N_FFT // 2 + 1

        # ---- subsampling (Conv2D, symmetric padding, factor SUB_FACTOR) ----
        self.register_buffer("sub0_w", g("encoder.subsampling.layers.0.weight"))
        self.register_buffer("sub0_b", g("encoder.subsampling.layers.0.bias"))
        sub2_w = g("encoder.subsampling.layers.2.weight")
        sub2_b = g("encoder.subsampling.layers.2.bias")
        sub3_w = g("encoder.subsampling.layers.3.weight")
        sub3_b = g("encoder.subsampling.layers.3.bias") + torch.einsum("oihw,i->o", sub3_w, sub2_b)
        self.register_buffer("sub2_w", sub2_w)
        self.register_buffer("sub3_w", sub3_w)
        self.register_buffer("sub3_b", sub3_b)
        sub5_w = g("encoder.subsampling.layers.5.weight")
        sub5_b = g("encoder.subsampling.layers.5.bias")
        sub6_w = g("encoder.subsampling.layers.6.weight")
        sub6_b = g("encoder.subsampling.layers.6.bias") + torch.einsum("oihw,i->o", sub6_w, sub5_b)
        self.register_buffer("sub5_w", sub5_w)
        self.register_buffer("sub6_w", sub6_w)
        self.register_buffer("sub6_b", sub6_b)
        self.register_buffer("sub_lin_w", g("encoder.subsampling.linear.weight"))
        self.register_buffer("sub_lin_b", g("encoder.subsampling.linear.bias"))

        self.register_buffer("ln_ones", torch.ones(D_MODEL))
        self.register_buffer("rel_shift_pads", torch.tensor([0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.int64))
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
            self.register_buffer(f"ff1_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous())
            self.register_buffer(f"ff1_l1b_{l}", torch.matmul(l1w, gb))
            self.register_buffer(f"ff1_l2w_{l}", (l2w * 0.5).contiguous())

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
            self.register_buffer(f"qkv_w_{l}", qkv_w.contiguous())
            self.register_buffer(f"qkv_b_{l}", qkv_b.contiguous())
            # Bake relative_k_proj over the whole position table -> (1, H, HEAD_DIM, 2P-1).
            rel_k = F.linear(pos_embed, g(p + "self_attn.relative_k_proj.weight"))
            rel_k = rel_k.reshape(-1, N_HEADS, HEAD_DIM).permute(1, 2, 0).contiguous()
            self.register_buffer(f"pos_proj_{l}", rel_k.unsqueeze(0).half())
            self.register_buffer(f"bias_u_{l}", (g(p + "self_attn.bias_u") * inv_sqrt_dk).unsqueeze(1).contiguous())
            self.register_buffer(f"bias_v_{l}", (g(p + "self_attn.bias_v") * inv_sqrt_dk).unsqueeze(1).contiguous())
            self.register_buffer(f"out_w_{l}", g(p + "self_attn.o_proj.weight").contiguous())

            # conv: fold norm_conv into pointwise_conv1; fold BatchNorm (running stats) into a per-channel affine.
            gw, gb = ln("norm_conv")
            pw1 = g(p + "conv.pointwise_conv1.weight").squeeze(-1)
            self.register_buffer(f"pw1_w_{l}", (pw1 * gw.unsqueeze(0)).contiguous())
            self.register_buffer(f"pw1_b_{l}", torch.matmul(pw1, gb))
            dw_w = g(p + "conv.depthwise_conv.weight")
            bn_w = g(p + "conv.norm.weight")
            bn_b = g(p + "conv.norm.bias")
            bn_rm = g(p + "conv.norm.running_mean")
            bn_rv = g(p + "conv.norm.running_var")
            bn_scale = bn_w / torch.sqrt(bn_rv + BN_EPS)
            bn_shift = bn_b - bn_rm * bn_scale
            self.register_buffer(f"dw_w_{l}", (dw_w * bn_scale[:, None, None]).contiguous())
            self.register_buffer(f"dw_b_{l}", bn_shift.contiguous())
            self.register_buffer(f"pw2_w_{l}", g(p + "conv.pointwise_conv2.weight").squeeze(-1).contiguous())

            # feed_forward2: same folding as feed_forward1.
            gw, gb = ln("norm_feed_forward2")
            l1w = g(p + "feed_forward2.linear1.weight")
            l2w = g(p + "feed_forward2.linear2.weight")
            self.register_buffer(f"ff2_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous())
            self.register_buffer(f"ff2_l1b_{l}", torch.matmul(l1w, gb))
            self.register_buffer(f"ff2_l2w_{l}", (l2w * 0.5).contiguous())

            ow, ob = ln("norm_out")
            self.register_buffer(f"no_w_{l}", ow)
            self.register_buffer(f"no_b_{l}", ob)

        # encoder projector -> enc_proj (joint "enc" side, projects D_MODEL -> DEC_HIDDEN).
        self.register_buffer("enc_proj_w", g("encoder_projector.weight").contiguous())
        self.register_buffer("enc_proj_b", g("encoder_projector.bias").contiguous())

    def _preprocess(self, audio):
        # The fixed Conv implements x[t] - preemph*x[t-1]; INT16 scaling is folded into its weights.
        x = F.conv1d(audio.float(), self.preemph_kernel, padding=1)
        stft = F.conv1d(x, self.stft_kernel, stride=HOP_LENGTH, padding=N_FFT // 2)
        real, imag = torch.split(stft, self.f_bins, dim=1)
        power = real * real + imag * imag
        # Keep mel_fb on the left; Optimize_ONNX.py skips FusionGemm to avoid onnxslim's bad const@var rewrite.
        mel = torch.matmul(self.mel_fb, power)
        mel = torch.clamp_min(mel,self.log_guard)
        feats = torch.log(mel)              # (B, N_MELS, T_full)
        # Drop the trailing STFT frame (valid = floor(L/hop) = T_full - 1) and per-feature normalize.
        valid = feats[:, :, :-1]
        n = valid.shape[2]
        mean = valid.mean(dim=2, keepdim=True)
        centered = valid - mean
        var = (centered * centered).sum(dim=2, keepdim=True) / (n - 1)
        std = torch.sqrt(var)
        normed = centered / (std + self.norm_eps)
        return normed.transpose(1, 2)                        # (B, N_valid, N_MELS)

    def _subsample(self, feats):
        x = feats.unsqueeze(1)                               # (B, 1, T, N_MELS)
        x = F.relu(F.conv2d(x, self.sub0_w, self.sub0_b, stride=SUB_STRIDE, padding=SUB_PAD))
        x = F.conv2d(x, self.sub2_w, stride=SUB_STRIDE, padding=SUB_PAD, groups=SUB_CHANNELS)
        x = F.relu(F.conv2d(x, self.sub3_w, self.sub3_b))
        x = F.conv2d(x, self.sub5_w, stride=SUB_STRIDE, padding=SUB_PAD, groups=SUB_CHANNELS)
        x = F.relu(F.conv2d(x, self.sub6_w, self.sub6_b))
        x = x.transpose(1, 2).flatten(2)
        return F.linear(x, self.sub_lin_w, self.sub_lin_b)   # (B, S, D_MODEL)

    def _rel_shift(self, x, batch_size, seq_len, relative_width):
        # Transformer-XL skew; input (B, H, S, 2S-1) -> (B, H, S, 2S-1), caller slices [..., :S].
        x = _PAD_LAST_DIM_LEFT_ONE.apply(x, self.rel_shift_pads)
        x = x.reshape(batch_size, N_HEADS, -1, seq_len)
        x = x[:, :, 1:]
        x = x.reshape(batch_size, N_HEADS, seq_len, relative_width)
        return x

    def forward(self, audio):
        x = self._subsample(self._preprocess(audio))
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        relative_width = 2 * seq_len - 1
        pe_start = self.pe_center - seq_len
        pe_end = self.pe_center + seq_len - 1

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
            p_t = getattr(self, f"pos_proj_{l}")[..., pe_start:pe_end].float()
            ac = torch.matmul(q_u, k_t)
            bd = torch.matmul(q_v, p_t)
            bd = self._rel_shift(bd, batch_size, seq_len, relative_width)[..., :seq_len]
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
            xc = F.conv1d(xc, getattr(self, f"dw_w_{l}"), getattr(self, f"dw_b_{l}"),
                          padding=CONV_PAD, groups=D_MODEL)
            xc = xc.transpose(1, 2)
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
        self.register_buffer("dec_proj_w", sd["decoder.decoder_projector.weight"].float().contiguous())
        self.register_buffer("dec_proj_b", sd["decoder.decoder_projector.bias"].float().contiguous())
        self.register_buffer("head_w", sd["joint.head.weight"].float().contiguous())
        self.register_buffer("head_b", sd["joint.head.bias"].float().contiguous())
        self.duration_is_index = DURATIONS == list(range(NUM_DURATION))
        if not self.duration_is_index:
            self.register_buffer("durations", torch.tensor(DURATIONS, dtype=torch.int32))

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
        if self.duration_is_index:
            duration = dur_idx.to(torch.int32)
        else:
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
def _load_special_token_ids() -> dict:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
    token_config = json.loads((MODEL_PATH / "tokenizer_config.json").read_text())
    generation_config_path = MODEL_PATH / "generation_config.json"
    generation_config = (
        json.loads(generation_config_path.read_text())
        if generation_config_path.is_file()
        else {}
    )

    special_token_ids = {
        "blank": BLANK_ID,
        "unknown": int(tokenizer.token_to_id(token_config["unk_token"])),
        "pad": int(tokenizer.token_to_id(token_config["pad_token"])),
        "eos": int(tokenizer.token_to_id(token_config["eos_token"])),
    }
    optional_tokens = {
        "no_speech": "<|nospeech|>",
        "start_of_transcript": "<|startoftranscript|>",
        "start_of_context": "<|startofcontext|>",
    }
    for role, token in optional_tokens.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            special_token_ids[role] = int(token_id)
    decoder_start = generation_config.get("decoder_start_token_id")
    if decoder_start is not None:
        special_token_ids["decoder_start"] = int(decoder_start)

    return special_token_ids


def make_metadata() -> dict:
    special_token_ids = _load_special_token_ids()
    return build_model_metadata({
        "sample_rate": SAMPLE_RATE,
        "audio_pcm_scale": AUDIO_PCM_SCALE,
        "max_symbols_per_step": MAX_SYMBOLS_PER_STEP,
        "special_token_ids": special_token_ids,
    })


# Export driver
def _copy_side_files() -> None:
    for name in TOKENIZER_FILES:
        src = MODEL_PATH / name
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
            p = ONNX_FOLDER / METADATA_NAME
            torch.onnx.export(MetadataCarrier().eval(), (torch.zeros(1, dtype=torch.int64),), str(p),
                              input_names=["metadata_marker"], output_names=["metadata_marker_out"],
                              opset_version=OPSET, dynamo=False)
            finalize_graph(p)
            write_metadata_carrier(p, metadata)

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

    finally:
        del encoder, decjoint, sd
        gc.collect()


if __name__ == "__main__":
    export_all()
    if subprocess.call(
        [
            sys.executable,
            str(_SCRIPT_DIR / "Inference_Parakeet_ASR_ONNX.py"),
            "--onnx-folder",
            str(ONNX_FOLDER),
        ],
        cwd=str(_SCRIPT_DIR),
    ) != 0:
        raise RuntimeError("Parakeet ASR inference failed after export.")
