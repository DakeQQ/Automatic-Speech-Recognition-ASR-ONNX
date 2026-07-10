"""
Export NVIDIA Nemotron 3.5 ASR (RNN-T, cache-aware Conformer) to ONNX -- OFFLINE or STREAMING.

Standalone and NeMo-free: reads weights/config from the .nemo archive. The CHUNK_MS knob selects
the export mode (see the Configuration block):

  CHUNK_MS = 0  -> OFFLINE. One full-sequence graph folding mel, Conformer, and prompt projection:
      ASR_Matadata.onnx   marker -> marker
      Nemotron_ASR_Encoder.onnx    audio + prompt_id -> enc_proj
      Nemotron_ASR_Decoder.onnx    enc_proj + frame_idx + token + state -> next_token + is_blank + state

  CHUNK_MS > 0  -> STREAMING. A cache-aware encoder consuming one fixed audio window per step while
      threading NeMo's Conformer caches; each step emits VALID_OUT_LEN frames that are bit-for-bit
      equal to the offline graph, so the reused RNN-T greedy decoder keeps offline quality:
      ASR_Matadata.onnx / _Encoder.onnx / _Decoder.onnx

Offline graphs go to Nemotron_ASR_ONNX/; streaming graphs go to Streaming/Nemotron_ASR_Streaming_ONNX/.
"""

import gc
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

# Configuration
_SCRIPT_DIR    = Path(__file__).resolve().parent
_REPO_ROOT     = _SCRIPT_DIR.parent
_DOWNLOADS     = _REPO_ROOT.parent

NEMO_PATH     = _DOWNLOADS / "nemotron-3.5-asr-streaming-0.6b" / "nemotron-3.5-asr-streaming-0.6b.nemo"

OPSET             = 20            # >=17 for fused LayerNormalization.
INPUT_AUDIO_DTYPE = "INT16"       # "INT16" (raw PCM, graph divides by 32768) | "F32" | "F16".
# CHUNK_MS selects the export mode:
#   CHUNK_MS = 0  -> OFFLINE full-sequence encoder (mel + Conformer + prompt fused into one graph).
#   CHUNK_MS > 0  -> STREAMING cache-aware encoder; the value picks one of the model's trained
#                    att_context look-ahead modes. This model emits one output frame per 80 ms, so:
#                       CHUNK_MS =   80 -> att_context [56,  0]  ( 1 frame / chunk, lowest latency)
#                       CHUNK_MS =  320 -> att_context [56,  3]  ( 4 frames/ chunk)
#                       CHUNK_MS =  560 -> att_context [56,  6]  ( 7 frames/ chunk)
#                       CHUNK_MS = 1120 -> att_context [56, 13]  (14 frames/ chunk, highest accuracy)
#                    The nearest supported mode is picked automatically.
CHUNK_MS          = 560             # 0 = offline export; >0 = streaming chunk size in milliseconds.
STREAMING         = CHUNK_MS != 0   # Derived export-mode flag.
DO_EXPORT         = True            # Run the ONNX export.
DYNAMIC_AXES      = True            # Offline only: True keeps audio length dynamic; False bakes a fixed length.
FIXED_INPUT_AUDIO_SECONDS = 10.0    # Offline only: used when DYNAMIC_AXES is False.

if STREAMING:
    ONNX_FOLDER        = _SCRIPT_DIR / "Nemotron_ASR_Streaming_ONNX"
    METADATA_NAME      = "ASR_Matadata.onnx"
    ENCODER_NAME       = "Nemotron_ASR_Streaming_Encoder.onnx"
    DECODER_JOINT_NAME = "Nemotron_ASR_Streaming_Decoder.onnx"
else:
    ONNX_FOLDER        = _SCRIPT_DIR / "Nemotron_ASR_ONNX"
    METADATA_NAME      = "ASR_Matadata.onnx"
    ENCODER_NAME       = "Nemotron_ASR_Encoder.onnx"
    DECODER_JOINT_NAME = "Nemotron_ASR_Decoder.onnx"

# The fused inference script auto-detects offline vs streaming graphs, so both modes reuse it.
INFERENCE_SCRIPT = _SCRIPT_DIR / "Inference_Nemotron_ASR_ONNX.py"

TOKENIZER_MODEL_NAME = "tokenizer.model"
TOKENIZER_VOCAB_NAME = "vocab.txt"

# Constants absent from model_config.yaml, plus export-only settings.
PREEMPH       = 0.97          # NeMo mel pre-emphasis default (absent from config)
LOG_GUARD     = 2.0 ** -24    # NeMo log_zero_guard_value default (absent from config)
LN_EPS        = 1e-5          # Conformer LayerNorm epsilon
DROP_EXTRA    = 2             # drop_extra_pre_encoded frames (cache-aware streaming)
MAX_SYMBOLS   = 10            # greedy RNN-T max emitted symbols per frame
# Must cover the whole fused utterance; 1536 encoder frames is roughly 120 seconds.
PE_MAX_LEN    = 1536


# Model geometry from model_config.yaml.
def _read_model_config() -> dict:
    """Load model_config.yaml without extracting the full checkpoint."""
    for cached in (ONNX_FOLDER / "model_config.yaml",
                   _SCRIPT_DIR / "Nemotron_ASR_ONNX" / "model_config.yaml",
                   _SCRIPT_DIR / "Nemotron_ASR_Streaming_ONNX" / "model_config.yaml"):
        if cached.exists():
            return yaml.safe_load(cached.read_text())
    with tarfile.open(NEMO_PATH, "r:*") as tar:
        member = next(m for m in tar.getmembers()
                      if m.isfile() and Path(m.name).name == "model_config.yaml")
        with tar.extractfile(member) as src:
            return yaml.safe_load(src.read())


def _select_att_context(att_context, chunk_ms, frame_ms) -> list:
    """Pick the trained att_context [left, right] for the requested mode.

    ``chunk_ms == 0`` (offline) selects the widest right-context pair (whole-utterance look-ahead).
    ``chunk_ms > 0`` (streaming) selects the pair whose chunk duration is closest to ``chunk_ms``:
    each streaming step advances (right + 1) output frames = (right + 1) * frame_ms of audio.
    """
    if att_context and isinstance(att_context[0], (list, tuple)):
        pairs = [list(pair) for pair in att_context]
    else:
        pairs = [list(att_context)]
    if chunk_ms <= 0:
        left, right = max(pairs, key=lambda pair: pair[1])
    else:
        left, right = min(pairs, key=lambda pair: abs((int(pair[1]) + 1) * frame_ms - chunk_ms))
    return [int(left), int(right)]


_CFG          = _read_model_config()
_PRE_CFG      = _CFG["preprocessor"]
_ENC_CFG      = _CFG["encoder"]
_DEC_CFG      = _CFG["decoder"]
_JOINT_CFG    = _CFG["joint"]
_DEFAULTS_CFG = _CFG.get("model_defaults", {})

SAMPLE_RATE   = int(_PRE_CFG["sample_rate"])
N_MELS        = int(_PRE_CFG["features"])
N_FFT         = int(_PRE_CFG["n_fft"])
WIN_LENGTH    = int(round(float(_PRE_CFG["window_size"]) * SAMPLE_RATE))
HOP_LENGTH    = int(round(float(_PRE_CFG["window_stride"]) * SAMPLE_RATE))

D_MODEL       = int(_ENC_CFG["d_model"])
N_LAYERS      = int(_ENC_CFG["n_layers"])
N_HEADS       = int(_ENC_CFG["n_heads"])
HEAD_DIM      = D_MODEL // N_HEADS
D_FF          = int(round(D_MODEL * float(_ENC_CFG["ff_expansion_factor"])))
CONV_KERNEL   = int(_ENC_CFG["conv_kernel_size"])
CONV_CACHE    = CONV_KERNEL - 1
SUB_FACTOR    = int(_ENC_CFG["subsampling_factor"])
SUB_CHANNELS  = int(_ENC_CFG["subsampling_conv_channels"])
_FRAME_MS     = SUB_FACTOR * HOP_LENGTH / SAMPLE_RATE * 1000.0          # duration of one encoder output frame
ATT_CONTEXT_SIZE = _select_att_context(_ENC_CFG["att_context_size"], CHUNK_MS, _FRAME_MS)
LEFT_CONTEXT  = ATT_CONTEXT_SIZE[0]

PRED_HIDDEN   = int(_DEC_CFG["prednet"]["pred_hidden"])
LSTM_LAYERS   = int(_DEC_CFG["prednet"]["pred_rnn_layers"])
JOINT_HIDDEN  = int(_JOINT_CFG["jointnet"]["joint_hidden"])
VOCAB_SIZE    = int(_DEC_CFG.get("vocab_size", _JOINT_CFG.get("num_classes")))
LOGITS_SIZE   = VOCAB_SIZE + 1
BLANK_ID      = VOCAB_SIZE
NUM_PROMPTS   = int(_DEFAULTS_CFG.get("num_prompts", 128))

VALID_OUT_LEN             = ATT_CONTEXT_SIZE[1] + 1
CHUNK_FEATURE_FRAMES      = VALID_OUT_LEN * SUB_FACTOR
PRE_ENCODE_CACHE_FRAMES   = SUB_FACTOR + 1

# Cache-aware streaming geometry (used when STREAMING; matches NeMo setup_streaming_params).
STREAM_CHUNK_MS        = int(round(VALID_OUT_LEN * _FRAME_MS))  # actual chunk duration picked from CHUNK_MS (ms)
STREAM_KV_LEN          = LEFT_CONTEXT + VALID_OUT_LEN        # attention key length = 56 + 14 = 70 per chunk
STREAM_MEL_CHUNK       = VALID_OUT_LEN * SUB_FACTOR          # new mel frames per chunk (112 = shift_size)
STREAM_MEL_CACHE       = PRE_ENCODE_CACHE_FRAMES            # pre-encode left-context mel frames (9)
STREAM_MEL_WINDOW      = STREAM_MEL_CACHE + STREAM_MEL_CHUNK # mel frames fed to pre_encode (121)
STREAM_STRIDE_SAMPLES  = STREAM_MEL_CHUNK * HOP_LENGTH       # audio advanced per chunk (17920 = 1.12 s @ 16 kHz)
STREAM_LEFT_OVERLAP    = (N_FFT // 2) + 1                    # left audio overlap for STFT/pre-emphasis continuity
# One extra left sample feeds pre-emphasis history; it is dropped before the snip-edges STFT.
STREAM_WINDOW_SAMPLES  = (STREAM_MEL_CHUNK - 1) * HOP_LENGTH + N_FFT + 1  # fixed encoder audio window (18273)

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

# Asset loading
def ensure_assets(nemo_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = {}
    ckpt_path = out_dir / "model_weights.ckpt"
    need = not ckpt_path.exists() or not (out_dir / "model_config.yaml").exists()
    with tarfile.open(nemo_path, "r:*") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        if need:
            for want, dst in (("model_weights.ckpt", ckpt_path),
                              ("model_config.yaml", out_dir / "model_config.yaml")):
                m = members.get(want)
                if m is not None:
                    with tar.extractfile(m) as src:
                        dst.write_bytes(src.read())
        tok = next((m for n, m in members.items() if n.endswith("_tokenizer.model") or n == "tokenizer.model"), None)
        voc = next((m for n, m in members.items() if n.endswith("_vocab.txt") or n == "vocab.txt"), None)
        if tok is not None and not (out_dir / TOKENIZER_MODEL_NAME).exists():
            with tar.extractfile(tok) as src:
                (out_dir / TOKENIZER_MODEL_NAME).write_bytes(src.read())
        if voc is not None and not (out_dir / TOKENIZER_VOCAB_NAME).exists():
            with tar.extractfile(voc) as src:
                (out_dir / TOKENIZER_VOCAB_NAME).write_bytes(src.read())
    assets["ckpt"] = ckpt_path
    assets["config"] = out_dir / "model_config.yaml"
    assets["tokenizer_model"] = out_dir / TOKENIZER_MODEL_NAME
    assets["tokenizer_vocab"] = out_dir / TOKENIZER_VOCAB_NAME
    return assets


def remove_extracted_checkpoint(assets: dict) -> None:
    ckpt_path = assets.get("ckpt")
    if ckpt_path is None:
        return
    if ckpt_path.exists():
        try:
            ckpt_path.unlink()
            print(f"Removed temporary checkpoint: {ckpt_path}")
        except OSError as exc:
            print(f"Warning: could not remove temporary checkpoint {ckpt_path}: {exc}", file=sys.stderr)

# Fused LayerNormalization op.
class _LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, epsilon, axis):
        xf = x.float()
        mean = xf.mean(dim=axis, keepdim=True)
        xc = xf - mean
        var = xc.pow(2).mean(dim=axis, keepdim=True)
        y = xc * torch.rsqrt(var + epsilon) * scale
        if bias is not None:
            y = y + bias
        return y.to(x.dtype)

    @staticmethod
    def symbolic(g, x, scale, bias, epsilon, axis):
        if bias is None:
            return g.op("LayerNormalization", x, scale, axis_i=axis, epsilon_f=epsilon, stash_type_i=1)
        return g.op("LayerNormalization", x, scale, bias, axis_i=axis, epsilon_f=epsilon, stash_type_i=1)


def layer_norm(x, scale, bias=None, epsilon=LN_EPS, axis=-1):
    return _LAYER_NORM.apply(x, scale, bias, float(epsilon), axis)


def swish(x):
    return x * torch.sigmoid(x)

class MetadataCarrier(nn.Module):
    def forward(self, marker: Tensor) -> Tensor:
        return marker

# Single-pass encoder graph.
class NemotronEncoder(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        g = lambda k: sd[k].float()
        self.register_buffer("c0_w", g("encoder.pre_encode.conv.0.weight"), persistent=False)
        self.register_buffer("c0_b", g("encoder.pre_encode.conv.0.bias"), persistent=False)
        self.register_buffer("c2_w", g("encoder.pre_encode.conv.2.weight"), persistent=False)
        self.register_buffer("c2_b", g("encoder.pre_encode.conv.2.bias"), persistent=False)
        self.register_buffer("c3_w", g("encoder.pre_encode.conv.3.weight"), persistent=False)
        self.register_buffer("c3_b", g("encoder.pre_encode.conv.3.bias"), persistent=False)
        self.register_buffer("c5_w", g("encoder.pre_encode.conv.5.weight"), persistent=False)
        self.register_buffer("c5_b", g("encoder.pre_encode.conv.5.bias"), persistent=False)
        self.register_buffer("c6_w", g("encoder.pre_encode.conv.6.weight"), persistent=False)
        self.register_buffer("c6_b", g("encoder.pre_encode.conv.6.bias"), persistent=False)
        self.register_buffer("out_w", g("encoder.pre_encode.out.weight"), persistent=False)
        self.register_buffer("out_b", g("encoder.pre_encode.out.bias"), persistent=False)

        window = sd["preprocessor.featurizer.window"].float()
        fb = sd["preprocessor.featurizer.fb"].float()
        pad_total = N_FFT - WIN_LENGTH
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        win = torch.cat([torch.zeros(pad_l), window, torch.zeros(pad_r)])
        f = torch.arange(N_FFT // 2 + 1, dtype=torch.float32).unsqueeze(1)
        t = torch.arange(N_FFT, dtype=torch.float32).unsqueeze(0)
        omega = (2.0 * math.pi / N_FFT) * f * t
        cos_k = (torch.cos(omega) * win.unsqueeze(0)).unsqueeze(1)
        sin_k = (-torch.sin(omega) * win.unsqueeze(0)).unsqueeze(1)
        self.register_buffer("stft_kernel", torch.cat([cos_k, sin_k], dim=0), persistent=False)
        self.register_buffer("pad_zero", torch.zeros(1, 1, N_FFT // 2, dtype=torch.float32), persistent=False)
        self.register_buffer("fb", fb.squeeze(0).contiguous(), persistent=False)
        self.register_buffer("inv_int16", torch.tensor(1.0 / 32768.0, dtype=torch.float32), persistent=False)
        self.register_buffer("preemph", torch.tensor(PREEMPH, dtype=torch.float32), persistent=False)
        self.register_buffer("log_guard", torch.tensor(LOG_GUARD, dtype=torch.float32), persistent=False)
        self.f_bins = N_FFT // 2 + 1
        # Fixed pads replace streaming caches in the offline graph.
        self.register_buffer("pre_encode_pad", torch.zeros(1, N_MELS, PRE_ENCODE_CACHE_FRAMES, dtype=torch.float32), persistent=False)
        self.register_buffer("conv_pad", torch.zeros(1, D_MODEL, CONV_CACHE, dtype=torch.float32), persistent=False)

        positions = torch.arange(PE_MAX_LEN - 1, -PE_MAX_LEN, -1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2, dtype=torch.float32) * -(math.log(10000.0) / D_MODEL))
        pe = torch.zeros(positions.shape[0], D_MODEL)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)
        # Precomputed block mask reproduces chunked attention; int8 -128 becomes softmax zero.
        frame_index = torch.arange(PE_MAX_LEN, dtype=torch.int16)
        chunk_start = torch.div(frame_index, VALID_OUT_LEN, rounding_mode="floor") * VALID_OUT_LEN
        lo = (chunk_start - LEFT_CONTEXT).unsqueeze(1)
        hi = (chunk_start + ATT_CONTEXT_SIZE[1]).unsqueeze(1)
        valid = (frame_index.unsqueeze(0) >= lo) & (frame_index.unsqueeze(0) <= hi)
        attention_mask = torch.zeros(PE_MAX_LEN, PE_MAX_LEN, dtype=torch.int8)
        attention_mask.masked_fill_(~valid, -128)
        self.register_buffer("attention_mask", attention_mask[None, None].contiguous(), persistent=False)
        self.pe_center = PE_MAX_LEN

        self.register_buffer("ln_ones", torch.ones(D_MODEL), persistent=False)
        inv_sqrt_dk = HEAD_DIM ** -0.5

        for l in range(N_LAYERS):
            p = f"encoder.layers.{l}."

            def ln(name):
                return g(p + name + ".weight"), g(p + name + ".bias")

            # Fold LN affine into FF1; fold residual half-step into linear2.
            gw, gb = ln("norm_feed_forward1")
            l1w = g(p + "feed_forward1.linear1.weight")
            l2w = g(p + "feed_forward1.linear2.weight")
            self.register_buffer(f"ff1_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous(), persistent=False)
            self.register_buffer(f"ff1_l1b_{l}", torch.matmul(l1w, gb), persistent=False)
            self.register_buffer(f"ff1_l2w_{l}", (l2w * 0.5).contiguous(), persistent=False)

            # Fuse QKV, LN affine, and attention scale.
            gw, gb = ln("norm_self_att")
            qw = g(p + "self_attn.linear_q.weight")
            kw = g(p + "self_attn.linear_k.weight")
            vw = g(p + "self_attn.linear_v.weight")
            qkv_w_orig = torch.cat([qw, kw, vw], dim=0)
            qkv_b = torch.matmul(qkv_w_orig, gb)
            qkv_w = (qkv_w_orig * gw.unsqueeze(0)).clone()
            qkv_b = qkv_b.clone()
            qkv_w[:D_MODEL] *= inv_sqrt_dk
            qkv_b[:D_MODEL] *= inv_sqrt_dk
            self.register_buffer(f"qkv_w_{l}", qkv_w.contiguous(), persistent=False)
            self.register_buffer(f"qkv_b_{l}", qkv_b.contiguous(), persistent=False)
            # Linear projection commutes with the centered rel-pos slice; store attention matmul layout.
            pos_proj = F.linear(pe, g(p + "self_attn.linear_pos.weight")).reshape(1, -1, N_HEADS, HEAD_DIM)
            pos_proj = pos_proj.permute(0, 2, 3, 1).contiguous()
            self.register_buffer(f"pos_proj_{l}", pos_proj.half(), persistent=False)
            # Pre-transposed to (N_HEADS, 1, HEAD_DIM) so q is transposed once before the bias add.
            self.register_buffer(f"bias_u_{l}", (g(p + "self_attn.pos_bias_u") * inv_sqrt_dk).unsqueeze(1).contiguous(), persistent=False)
            self.register_buffer(f"bias_v_{l}", (g(p + "self_attn.pos_bias_v") * inv_sqrt_dk).unsqueeze(1).contiguous(), persistent=False)
            self.register_buffer(f"out_w_{l}", g(p + "self_attn.linear_out.weight").contiguous(), persistent=False)

            # Fold norm_conv into pointwise_conv1; keep batch_norm affine.
            gw, gb = ln("norm_conv")
            pw1 = g(p + "conv.pointwise_conv1.weight").squeeze(-1)
            self.register_buffer(f"pw1_w_{l}", (pw1 * gw.unsqueeze(0)).contiguous(), persistent=False)
            self.register_buffer(f"pw1_b_{l}", torch.matmul(pw1, gb), persistent=False)
            self.register_buffer(f"dw_w_{l}", g(p + "conv.depthwise_conv.weight").contiguous(), persistent=False)
            bnw, bnb = ln("conv.batch_norm")
            self.register_buffer(f"bn_w_{l}", bnw, persistent=False)
            self.register_buffer(f"bn_b_{l}", bnb, persistent=False)
            self.register_buffer(f"pw2_w_{l}", g(p + "conv.pointwise_conv2.weight").squeeze(-1).contiguous(), persistent=False)

            gw, gb = ln("norm_feed_forward2")
            l1w = g(p + "feed_forward2.linear1.weight")
            l2w = g(p + "feed_forward2.linear2.weight")
            self.register_buffer(f"ff2_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous(), persistent=False)
            self.register_buffer(f"ff2_l1b_{l}", torch.matmul(l1w, gb), persistent=False)
            self.register_buffer(f"ff2_l2w_{l}", (l2w * 0.5).contiguous(), persistent=False)

            ow, ob = ln("norm_out")
            self.register_buffer(f"no_w_{l}", ow, persistent=False)
            self.register_buffer(f"no_b_{l}", ob, persistent=False)

        # Fold prompt bias and joint.enc into a frame-wise projection.
        W0 = sd["prompt_kernel.0.weight"].float()
        b0 = sd["prompt_kernel.0.bias"].float()
        W2 = sd["prompt_kernel.2.weight"].float()
        b2 = sd["prompt_kernel.2.bias"].float()
        je_w = sd["joint.enc.weight"].float()
        je_b = sd["joint.enc.bias"].float()
        W0_enc = W0[:, :D_MODEL]
        W0_prompt = W0[:, D_MODEL:]
        lang_bias = b0.unsqueeze(0) + W0_prompt.t()
        Wc = torch.matmul(je_w, W2)
        bc = torch.matmul(je_w, b2) + je_b
        self.register_buffer("W0_enc", W0_enc.contiguous(), persistent=False)
        self.register_buffer("lang_bias", lang_bias.contiguous(), persistent=False)
        self.register_buffer("Wc", Wc.contiguous(), persistent=False)
        self.register_buffer("bc", bc.contiguous(), persistent=False)

    def _causal_conv2d(self, x, w, b, groups):
        x = F.pad(x, (2, 1, 2, 1))
        return F.conv2d(x, w, b, stride=2, padding=0, groups=groups)

    def _pre_encode(self, feats):
        x = feats.transpose(1, 2).unsqueeze(1)
        x = F.relu(self._causal_conv2d(x, self.c0_w, self.c0_b, 1))
        x = self._causal_conv2d(x, self.c2_w, self.c2_b, SUB_CHANNELS)
        x = F.conv2d(x, self.c3_w, self.c3_b)
        x = F.relu(x)
        x = self._causal_conv2d(x, self.c5_w, self.c5_b, SUB_CHANNELS)
        x = F.conv2d(x, self.c6_w, self.c6_b)
        x = F.relu(x)
        x = x.transpose(1, 2).flatten(2)
        return F.linear(x, self.out_w, self.out_b)

    @staticmethod
    def _rel_shift_full(x, L, b):
        # Transformer-XL skew: out[i, j] reads centered relative distance i - j.
        x = F.pad(x, (1, 0))
        x = x.reshape(b, -1, 2 * L, L)
        x = x[:, :, 1:, :]
        x = x.reshape(b, -1, L, 2 * L - 1)
        return x[:, :, :, :L]

    def _preprocess(self, audio):
        # Float inputs are assumed normalized; INT16 is scaled to PCM range.
        if INPUT_AUDIO_DTYPE == "INT16":
            x = audio.float() * self.inv_int16
        else:
            x = audio.float()
        x = torch.cat([x[:, :, :1], x[:, :, 1:] - self.preemph * x[:, :, :-1]], dim=2)
        x = torch.cat([self.pad_zero, x, self.pad_zero], dim=2)
        stft = F.conv1d(x, self.stft_kernel, stride=HOP_LENGTH).square()
        real_sq, imag_sq = torch.split(stft, self.f_bins, dim=1)
        power = real_sq + imag_sq
        # Keep fb on the left; Optimize_ONNX.py skips FusionGemm to avoid onnxslim's bad const@var rewrite.
        mel = torch.matmul(self.fb, power)
        features = torch.log(mel + self.log_guard)
        # Match the streaming path: keep floor(S / HOP) frames.
        return features[:, :, :-1]

    def forward(self, audio, prompt_id):
        # Full-sequence graph; block mask reproduces cache-aware chunk attention.
        features = self._preprocess(audio)
        features = torch.cat([self.pre_encode_pad, features], dim=2)
        x = self._pre_encode(features)
        x = x[:, DROP_EXTRA:, :]
        batch_size = x.shape[0]
        L = x.shape[1]

        # Leading slice handles startup and partial final chunks.
        neg = self.attention_mask[..., :L, :L].float()

        for l in range(N_LAYERS):
            residual = x
            m = layer_norm(x, self.ln_ones)
            m = F.linear(m, getattr(self, f"ff1_l1w_{l}"), getattr(self, f"ff1_l1b_{l}"))
            m = swish(m)
            m = F.linear(m, getattr(self, f"ff1_l2w_{l}"))
            residual = residual + m

            m = layer_norm(residual, self.ln_ones)
            qkv = F.linear(m, getattr(self, f"qkv_w_{l}"), getattr(self, f"qkv_b_{l}"))
            qkv = qkv.reshape(batch_size, -1, 3 * N_HEADS, HEAD_DIM).transpose(1, 2)
            q, k, v = torch.split(qkv, N_HEADS, dim=1)
            q_u = q + getattr(self, f"bias_u_{l}")
            q_v = q + getattr(self, f"bias_v_{l}")
            k_t = k.transpose(2, 3)
            p_t = getattr(self, f"pos_proj_{l}")[..., self.pe_center - L: self.pe_center + L - 1].float()
            ac = torch.matmul(q_u, k_t)
            bd = torch.matmul(q_v, p_t)
            bd = self._rel_shift_full(bd, L, batch_size)
            scores = ac + bd + neg
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)
            ctx = ctx.transpose(1, 2).reshape(batch_size, -1, D_MODEL)
            m = F.linear(ctx, getattr(self, f"out_w_{l}"))
            residual = residual + m

            m = layer_norm(residual, self.ln_ones)
            xc = F.linear(m, getattr(self, f"pw1_w_{l}"), getattr(self, f"pw1_b_{l}"))
            xc = xc.transpose(1, 2)
            xc = F.glu(xc, dim=1)
            dw_in = torch.cat([self.conv_pad, xc], dim=2)
            xc = F.conv1d(dw_in, getattr(self, f"dw_w_{l}"), groups=D_MODEL)
            xc = xc.transpose(1, 2)
            xc = layer_norm(xc, getattr(self, f"bn_w_{l}"), getattr(self, f"bn_b_{l}"))
            xc = swish(xc)
            xc = F.linear(xc, getattr(self, f"pw2_w_{l}"))
            residual = residual + xc

            m = layer_norm(residual, self.ln_ones)
            m = F.linear(m, getattr(self, f"ff2_l1w_{l}"), getattr(self, f"ff2_l1b_{l}"))
            m = swish(m)
            m = F.linear(m, getattr(self, f"ff2_l2w_{l}"))
            residual = residual + m

            x = layer_norm(residual, getattr(self, f"no_w_{l}"), getattr(self, f"no_b_{l}"))

        # Prompt projection runs after all encoder layers.
        bias = torch.index_select(self.lang_bias, 0, prompt_id).unsqueeze(1)
        h0 = F.relu(F.linear(x, self.W0_enc) + bias)
        enc_proj = F.linear(h0, self.Wc, self.bc)
        return enc_proj


class NemotronStreamingEncoder(nn.Module):
    """Cache-aware streaming encoder.

    Consumes ONE fixed-length audio window plus NeMo's Conformer streaming caches and emits exactly
    VALID_OUT_LEN encoder frames. Reuses every fused buffer built by :class:`NemotronEncoder`; only the
    front-end (snip-edges mel + mel cache) and the per-layer attention/conv caching differ so that the
    output matches the offline graph frame-for-frame.

    Inputs
        audio               (1, 1, STREAM_WINDOW_SAMPLES)      one overlapping audio window
        mel_cache           (1, N_MELS, STREAM_MEL_CACHE)      previous 9 mel frames (pre-encode context)
        cache_last_channel  (N_LAYERS, 1, LEFT_CONTEXT, D)     per-layer pre-attention context (56 frames)
        cache_last_time     (N_LAYERS, 1, D, CONV_CACHE)       per-layer depthwise-conv context (8 frames)
        cache_len           (1,)                               valid frames currently in cache_last_channel
        prompt_id           (1,)                               language/prompt selector

    Outputs
        enc_proj            (1, VALID_OUT_LEN, JOINT_HIDDEN)
        mel_cache_next, cache_last_channel_next, cache_last_time_next, cache_len_next
    """

    def __init__(self, enc: "NemotronEncoder"):
        super().__init__()
        self.enc = enc
        # Reuse the offline block mask instead of recomputing the startup mask at runtime.
        # A steady-state row of enc.attention_mask is the fully-valid key window (kv_len zeros);
        # left-padding it with LEFT_CONTEXT masked slots represents the cache-warmup frames that
        # sit before absolute frame 0. Gathering this row at `cache_len` then reproduces the
        # per-chunk mask for every startup state (first LEFT_CONTEXT - cache_len keys masked).
        anchor = ((LEFT_CONTEXT + VALID_OUT_LEN - 1) // VALID_OUT_LEN) * VALID_OUT_LEN
        col0 = anchor - LEFT_CONTEXT
        window = enc.attention_mask[0, 0, anchor, col0:col0 + STREAM_KV_LEN].float()
        stream_mask = F.pad(window, (LEFT_CONTEXT, 0), value=-128.0)
        # Pre-shape to [1, 1, 1, N] so the runtime gather yields the broadcast mask with no reshape,
        # and pre-compute the constant key offsets reused every step.
        self.register_buffer("stream_mask", stream_mask.reshape(1, 1, 1, -1).contiguous(), persistent=False)
        self.register_buffer("key_arange", torch.arange(STREAM_KV_LEN, dtype=torch.int64), persistent=False)
        # Pre-slice / cast the relative-position projection for the fixed streaming key
        # window so each layer's attention reuses a ready [1, N_HEADS, HEAD_DIM, 2*kv_len-1] buffer.
        pos_lo = enc.pe_center - STREAM_KV_LEN
        pos_hi = enc.pe_center + STREAM_KV_LEN - 1
        for l in range(N_LAYERS):
            pos_t = getattr(enc, f"pos_proj_{l}")[..., pos_lo:pos_hi].float().contiguous()
            self.register_buffer(f"pos_t_{l}", pos_t, persistent=False)

    def _stream_mel(self, audio):
        # Snip-edges STFT over the window (no centre padding); the leading sample only seeds pre-emphasis.
        enc = self.enc
        if INPUT_AUDIO_DTYPE == "INT16":
            x = audio.float() * enc.inv_int16
        else:
            x = audio.float()
        x = x[:, :, 1:] - enc.preemph * x[:, :, :-1]
        stft = F.conv1d(x, enc.stft_kernel, stride=HOP_LENGTH)
        real, imag = torch.split(stft, enc.f_bins, dim=1)
        power = real * real + imag * imag
        mel = torch.matmul(enc.fb, power)
        return torch.log(mel + enc.log_guard)

    @staticmethod
    def _rel_shift(x, klen):
        # NeMo Transformer-XL relative shift for cache-aware attention (query length != key length).
        b, h, qlen, pos_len = x.shape
        x = F.pad(x, (1, 0))
        x = x.reshape(b, h, pos_len + 1, qlen)
        x = x[:, :, 1:]
        x = x.reshape(b, h, qlen, pos_len)
        return x[:, :, :, :klen]

    def forward(self, audio, mel_cache, cache_last_channel, cache_last_time, cache_len, prompt_id):
        enc = self.enc
        mel_new = self._stream_mel(audio)                       # (1, N_MELS, STREAM_MEL_CHUNK)
        mel_full = torch.cat([mel_cache, mel_new], dim=2)       # (1, N_MELS, STREAM_MEL_WINDOW)
        mel_cache_next = mel_full[:, :, -STREAM_MEL_CACHE:]
        x = enc._pre_encode(mel_full)
        x = x[:, DROP_EXTRA:, :]                                # (1, VALID_OUT_LEN, D_MODEL)
        batch_size = x.shape[0]
        C = VALID_OUT_LEN
        kv_len = STREAM_KV_LEN

        # Startup mask: gather the pre-shaped offline block mask at cache_len so the first
        # (LEFT_CONTEXT - cache_len) cache-warmup keys are masked out (all queries share it).
        key_pos = self.key_arange + cache_len
        neg = self.stream_mask.index_select(3, key_pos)

        channel_next = []
        time_next = []
        for l in range(N_LAYERS):
            residual = x
            m = layer_norm(x, enc.ln_ones)
            m = F.linear(m, getattr(enc, f"ff1_l1w_{l}"), getattr(enc, f"ff1_l1b_{l}"))
            m = swish(m)
            m = F.linear(m, getattr(enc, f"ff1_l2w_{l}"))
            residual = residual + m

            m = layer_norm(residual, enc.ln_ones)                       # norm_self_att == cache_last_channel content
            m_full = torch.cat([cache_last_channel[l], m], dim=1)        # (1, kv_len, D)
            channel_next.append(m_full[:, -LEFT_CONTEXT:, :])
            qkv = F.linear(m_full, getattr(enc, f"qkv_w_{l}"), getattr(enc, f"qkv_b_{l}"))
            qkv = qkv.reshape(batch_size, -1, 3 * N_HEADS, HEAD_DIM).transpose(1, 2)
            q, k, v = torch.split(qkv, N_HEADS, dim=1)
            q_t = q[:, :, -C:, :]                                        # queries = current chunk only
            q_u = q_t + getattr(enc, f"bias_u_{l}")
            q_v = q_t + getattr(enc, f"bias_v_{l}")
            k_t = k.transpose(2, 3)
            ac = torch.matmul(q_u, k_t)
            bd = torch.matmul(q_v, getattr(self, f"pos_t_{l}"))          # pre-sliced/cast/permuted rel-pos
            bd = self._rel_shift(bd, kv_len)
            scores = ac + bd + neg
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)
            ctx = ctx.transpose(1, 2).reshape(batch_size, -1, D_MODEL)
            m = F.linear(ctx, getattr(enc, f"out_w_{l}"))
            residual = residual + m

            m = layer_norm(residual, enc.ln_ones)
            xc = F.linear(m, getattr(enc, f"pw1_w_{l}"), getattr(enc, f"pw1_b_{l}"))
            xc = xc.transpose(1, 2)
            xc = F.glu(xc, dim=1)
            dw_in = torch.cat([cache_last_time[l], xc], dim=2)           # (1, D, CONV_CACHE + C)
            time_next.append(dw_in[:, :, -CONV_CACHE:])
            xc = F.conv1d(dw_in, getattr(enc, f"dw_w_{l}"), groups=D_MODEL)
            xc = xc.transpose(1, 2)
            xc = layer_norm(xc, getattr(enc, f"bn_w_{l}"), getattr(enc, f"bn_b_{l}"))
            xc = swish(xc)
            xc = F.linear(xc, getattr(enc, f"pw2_w_{l}"))
            residual = residual + xc

            m = layer_norm(residual, enc.ln_ones)
            m = F.linear(m, getattr(enc, f"ff2_l1w_{l}"), getattr(enc, f"ff2_l1b_{l}"))
            m = swish(m)
            m = F.linear(m, getattr(enc, f"ff2_l2w_{l}"))
            residual = residual + m

            x = layer_norm(residual, getattr(enc, f"no_w_{l}"), getattr(enc, f"no_b_{l}"))

        bias = torch.index_select(enc.lang_bias, 0, prompt_id).unsqueeze(1)
        h0 = F.relu(F.linear(x, enc.W0_enc) + bias)
        enc_proj = F.linear(h0, enc.Wc, enc.bc)

        cache_last_channel_next = torch.stack(channel_next, dim=0)
        cache_last_time_next = torch.stack(time_next, dim=0)
        cache_len_next = torch.clamp(cache_len + C, max=LEFT_CONTEXT)
        return enc_proj, mel_cache_next, cache_last_channel_next, cache_last_time_next, cache_len_next


class NemotronDecoderJoint(nn.Module):
    def __init__(self, sd: dict, blank_id: int):
        super().__init__()
        self.blank_id = int(blank_id)
        self.embed = nn.Embedding(LOGITS_SIZE, PRED_HIDDEN)
        self.lstm = nn.LSTM(PRED_HIDDEN, PRED_HIDDEN, LSTM_LAYERS, batch_first=True)
        with torch.no_grad():
            self.embed.weight.copy_(sd["decoder.prediction.embed.weight"].half())
            for li in range(LSTM_LAYERS):
                getattr(self.lstm, f"weight_ih_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{li}"].float())
                getattr(self.lstm, f"weight_hh_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{li}"].float())
                getattr(self.lstm, f"bias_ih_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{li}"].float())
                getattr(self.lstm, f"bias_hh_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{li}"].float())
        self.register_buffer("pred_w", sd["joint.pred.weight"].float().contiguous(), persistent=False)
        self.register_buffer("pred_b", sd["joint.pred.bias"].float().contiguous(), persistent=False)
        self.register_buffer("jnet_w", sd["joint.joint_net.2.weight"].float().contiguous(), persistent=False)
        self.register_buffer("jnet_b", sd["joint.joint_net.2.bias"].float().contiguous(), persistent=False)

    def forward(self, enc_proj, frame_idx, token, state_h, state_c):
        enc_proj_frame = torch.index_select(enc_proj, 1, frame_idx)
        emb = self.embed(token).float()
        out, (h, c) = self.lstm(emb, (state_h, state_c))
        pred = F.linear(out, self.pred_w, self.pred_b)
        z = torch.relu(enc_proj_frame + pred)
        logits = F.linear(z, self.jnet_w, self.jnet_b)
        argmax = torch.argmax(logits, dim=-1).to(torch.int32)
        is_blank = argmax == self.blank_id
        # Blank steps keep token/state unchanged for in-place IOBinding. is_blank is
        # (batch, 1); it broadcasts against the (layers, batch, hidden) LSTM state through an
        # implicit leading axis, so the batch dimension aligns without an unsqueeze.
        next_token = torch.where(is_blank, token, argmax)
        h = torch.where(is_blank, state_h, h)
        c = torch.where(is_blank, state_c, c)
        return next_token, is_blank.to(torch.int32), h, c


# Metadata
def make_metadata(cfg: dict, assets: dict) -> dict:
    model_defaults = cfg.get("model_defaults", {})
    prompt_dictionary = model_defaults.get("prompt_dictionary", {})
    cache_specs = [
        {"name": "cache_last_channel", "shape": ["B", N_LAYERS, LEFT_CONTEXT, D_MODEL], "dtype": "float32"},
        {"name": "cache_last_time", "shape": ["B", N_LAYERS, D_MODEL, CONV_CACHE], "dtype": "float32"},
        {"name": "cache_last_channel_len", "shape": ["B"], "dtype": "int64"},
    ]
    decoder_state_specs = [
        {"name": "state_h", "shape": [LSTM_LAYERS, "B", PRED_HIDDEN], "dtype": "float32"},
        {"name": "state_c", "shape": [LSTM_LAYERS, "B", PRED_HIDDEN], "dtype": "float32"},
    ]
    header = {
        "nemotron_asr_metadata_version": 1,
        "producer": Path(__file__).name,
        "sample_rate": SAMPLE_RATE,
        "input_audio_dtype": INPUT_AUDIO_DTYPE,
        "opset": OPSET,
        "tokenizer_model": assets["tokenizer_model"].name,
        "tokenizer_vocab": assets["tokenizer_vocab"].name,
    }
    if STREAMING:
        header["streaming"] = True
    else:
        header["dynamic_axes"] = DYNAMIC_AXES
        header["fixed_input_audio_length"] = 0 if DYNAMIC_AXES else FIXED_INPUT_AUDIO_LENGTH
    feature_section = {
        "num_mels": N_MELS, "n_fft": N_FFT, "window_length": WIN_LENGTH, "hop_length": HOP_LENGTH,
        "preemph": PREEMPH, "log_guard": LOG_GUARD,
        "normalize": cfg.get("preprocessor", {}).get("normalize", "NA"), "mag_power": 2.0,
    }
    encoder_section = {
        "encoder_layers": N_LAYERS, "encoder_d_model": D_MODEL, "encoder_heads": N_HEADS,
        "head_dim": HEAD_DIM, "d_ff": D_FF, "conv_kernel_size": CONV_KERNEL,
        "subsampling_factor": SUB_FACTOR, "subsampling_conv_channels": SUB_CHANNELS,
        "drop_extra_pre_encoded": DROP_EXTRA, "att_context_size": ATT_CONTEXT_SIZE,
        "left_context": LEFT_CONTEXT, "conv_cache": CONV_CACHE,
        "valid_out_len": VALID_OUT_LEN,
        "chunk_feature_frames": CHUNK_FEATURE_FRAMES,
        "pre_encode_cache_frames": PRE_ENCODE_CACHE_FRAMES,
        "cache_tensor_specs": cache_specs,
    }
    decoder_section = {
        "decoder_pred_hidden": PRED_HIDDEN, "decoder_layers": LSTM_LAYERS, "joint_hidden": JOINT_HIDDEN,
        "vocab_size": VOCAB_SIZE, "logits_vocab_size": LOGITS_SIZE, "blank_id": BLANK_ID,
        "max_symbols": MAX_SYMBOLS, "decoder_uses_frame_idx": True, "decoder_state_specs": decoder_state_specs,
    }
    prompt_section = {
        "prompt_num_prompts": NUM_PROMPTS, "prompt_dictionary": prompt_dictionary,
        "default_target_lang": "en-US", "auto_prompt_id": prompt_dictionary.get("auto", 101),
    }
    sections = [header, feature_section, encoder_section]
    if STREAMING:
        sections.append({
            "stream_chunk_ms": STREAM_CHUNK_MS,
            "stream_kv_len": STREAM_KV_LEN,
            "stream_mel_chunk": STREAM_MEL_CHUNK,
            "stream_mel_cache": STREAM_MEL_CACHE,
            "stream_mel_window": STREAM_MEL_WINDOW,
            "stream_window_samples": STREAM_WINDOW_SAMPLES,
            "stream_stride_samples": STREAM_STRIDE_SAMPLES,
            "stream_left_overlap": STREAM_LEFT_OVERLAP,
        })
    sections.append(decoder_section)
    sections.append(prompt_section)
    return build_model_metadata(*sections)

# Export driver
def export_all():
    ONNX_FOLDER.mkdir(parents=True, exist_ok=True)
    if STREAMING:
        print(f"Nemotron ASR streaming export -> {ONNX_FOLDER}")
        print(f"  CHUNK_MS={CHUNK_MS} -> att_context={ATT_CONTEXT_SIZE}  "
              f"(chunk {STREAM_CHUNK_MS} ms = {VALID_OUT_LEN} frames; "
              f"window={STREAM_WINDOW_SAMPLES}, stride={STREAM_STRIDE_SAMPLES} samples)")
    else:
        print(f"Nemotron ASR export -> {ONNX_FOLDER}")
    assets = {"ckpt": ONNX_FOLDER / "model_weights.ckpt"}
    sd = offline_encoder = encoder = decjoint = None
    try:
        assets = ensure_assets(NEMO_PATH, ONNX_FOLDER)
        cfg = _CFG
        sd = torch.load(str(assets["ckpt"]), map_location="cpu", weights_only=True, mmap=True)
        metadata = make_metadata(cfg, assets)

        offline_encoder = NemotronEncoder(sd).eval()
        encoder = NemotronStreamingEncoder(offline_encoder).eval() if STREAMING else offline_encoder
        decjoint = NemotronDecoderJoint(sd, BLANK_ID).eval()

        with torch.inference_mode():
            if DO_EXPORT:
                p = ONNX_FOLDER / METADATA_NAME
                torch.onnx.export(MetadataCarrier().eval(), (torch.zeros(1, dtype=torch.int64),), str(p),
                                  input_names=["metadata_marker"], output_names=["metadata_marker_out"],
                                  opset_version=OPSET, dynamo=False)
                finalize_graph(p, metadata)

                p = ONNX_FOLDER / ENCODER_NAME
                if STREAMING:
                    # One fixed audio window + streaming caches (all static shapes).
                    audio = torch.zeros(1, 1, STREAM_WINDOW_SAMPLES, dtype=_AUDIO_TORCH_DTYPE)
                    mel_cache = torch.zeros(1, N_MELS, STREAM_MEL_CACHE)
                    chan = torch.zeros(N_LAYERS, 1, LEFT_CONTEXT, D_MODEL)
                    time_c = torch.zeros(N_LAYERS, 1, D_MODEL, CONV_CACHE)
                    clen = torch.zeros(1, dtype=torch.int64)
                    pid = torch.zeros(1, dtype=torch.int32)
                    torch.onnx.export(encoder, (audio, mel_cache, chan, time_c, clen, pid), str(p),
                                      input_names=["audio", "mel_cache", "cache_last_channel", "cache_last_time",
                                                   "cache_len", "prompt_id"],
                                      output_names=["enc_proj", "mel_cache_next", "cache_last_channel_next",
                                                    "cache_last_time_next", "cache_len_next"],
                                      dynamic_axes=None,
                                      opset_version=OPSET, dynamo=False)
                else:
                    audio = torch.zeros(1, 1, FIXED_INPUT_AUDIO_LENGTH, dtype=_AUDIO_TORCH_DTYPE)
                    pid = torch.zeros(1, dtype=torch.int32)
                    enc_axes = {"audio": {0: "batch", 2: "num_samples"},
                                "prompt_id": {0: "batch"},
                                "enc_proj": {0: "batch", 1: "enc_frames"}} if DYNAMIC_AXES else None
                    torch.onnx.export(encoder, (audio, pid), str(p),
                                      input_names=["audio", "prompt_id"],
                                      output_names=["enc_proj"],
                                      dynamic_axes=enc_axes,
                                      opset_version=OPSET, dynamo=False)
                finalize_graph(p)

                p = ONNX_FOLDER / DECODER_JOINT_NAME
                if STREAMING:
                    # Streaming reuses the decoder across chunks, so enc_frames stays dynamic.
                    ep = torch.randn(1, VALID_OUT_LEN, JOINT_HIDDEN)
                    dec_dynamic = True
                else:
                    # Static export runs the encoder once so decoder enc_frames is baked in too.
                    ep = torch.randn(1, 4, JOINT_HIDDEN) if DYNAMIC_AXES else encoder(audio, pid)
                    dec_dynamic = DYNAMIC_AXES
                frame_idx = torch.zeros(1, dtype=torch.int32)
                tok = torch.zeros(1, 1, dtype=torch.int32)
                sh = torch.zeros(LSTM_LAYERS, 1, PRED_HIDDEN)
                sc = torch.zeros(LSTM_LAYERS, 1, PRED_HIDDEN)
                dec_axes = {"enc_proj": {0: "batch", 1: "enc_frames"},
                            "token": {0: "batch"},
                            "state_h": {1: "batch"}, "state_c": {1: "batch"},
                            "next_token": {0: "batch"}, "is_blank": {0: "batch"},
                            "state_h_next": {1: "batch"}, "state_c_next": {1: "batch"}} if dec_dynamic else None
                torch.onnx.export(decjoint, (ep, frame_idx, tok, sh, sc), str(p),
                                  input_names=["enc_proj", "frame_idx", "token", "state_h", "state_c"],
                                  output_names=["next_token", "is_blank", "state_h_next", "state_c_next"],
                                  dynamic_axes=dec_axes,
                                  opset_version=OPSET, dynamo=False)
                finalize_graph(p)

                print(f"Stamped {len(metadata)} metadata keys. Export complete.")

        print("\n" + "─" * 70)
        print(f"Running ONNX Runtime demo via {INFERENCE_SCRIPT.name} ...\n")
        subprocess.run(
            [sys.executable, str(INFERENCE_SCRIPT), "--onnx-folder", str(ONNX_FOLDER)],
            check=True,
        )
    finally:
        del encoder, decjoint, sd, offline_encoder
        gc.collect()
        remove_extracted_checkpoint(assets)


if __name__ == "__main__":
    export_all()
