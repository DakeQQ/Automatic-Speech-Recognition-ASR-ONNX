"""
Export NVIDIA Nemotron 3.5 ASR (RNN-T, cache-aware Conformer) to ONNX -- OFFLINE or STREAMING.

Standalone and NeMo-free: reads weights/config from the .nemo archive. The CHUNK_MS knob selects
the export mode (see the Configuration block):

  CHUNK_MS = 0  -> OFFLINE. One full-sequence graph folding mel, Conformer, and prompt projection:
      ASR_Metadata.onnx   marker -> marker
      Nemotron_ASR_Encoder.onnx    audio + prompt_id -> enc_proj
      Nemotron_ASR_Decoder.onnx    enc_proj + frame_idx + token + state -> next_token + is_blank + state

  CHUNK_MS > 0  -> STREAMING. A cache-aware encoder consuming one fixed audio window per step while
      threading NeMo's Conformer caches; each step emits VALID_OUT_LEN frames that are bit-for-bit
      equal to the offline graph, so the reused RNN-T greedy decoder keeps offline quality:
      ASR_Metadata.onnx / _Encoder.onnx / _Decoder.onnx

Offline graphs go to Nemotron_ASR_ONNX/; streaming graphs go to Streaming/Nemotron_ASR_Streaming_ONNX/.
"""

import gc
import json
import math
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

# Configuration
_SCRIPT_DIR    = Path(__file__).resolve().parent
_DOWNLOADS     = Path.home() / "Downloads"

NEMO_PATH     = _DOWNLOADS / "nemotron-3.5-asr-streaming-0.6b" / "nemotron-3.5-asr-streaming-0.6b.nemo"

OPSET             = 20            # >=17 for fused LayerNormalization.
INPUT_AUDIO_DTYPE = "F32"         # "INT16" (raw PCM, graph divides by 32768) | "F32" | "F16".
USE_FP16_STORAGE  = True          # False exports strict-F32 encoder positions and decoder embeddings; True retains compact FP16 storage.
FLOAT_STORAGE_DTYPE = torch.float16 if USE_FP16_STORAGE else torch.float32
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
DYNAMIC_AXES      = True            # Offline only: True keeps audio length dynamic; False bakes a fixed length.
FIXED_INPUT_AUDIO_SECONDS = 10.0    # Offline only: used when DYNAMIC_AXES is False.

if STREAMING:
    ONNX_FOLDER        = _SCRIPT_DIR / "Nemotron_ASR_Streaming_ONNX"
    METADATA_NAME      = "ASR_Metadata.onnx"
    ENCODER_NAME       = "Nemotron_ASR_Streaming_Encoder.onnx"
    DECODER_JOINT_NAME = "Nemotron_ASR_Streaming_Decoder.onnx"
else:
    ONNX_FOLDER        = _SCRIPT_DIR / "Nemotron_ASR_ONNX"
    METADATA_NAME      = "ASR_Metadata.onnx"
    ENCODER_NAME       = "Nemotron_ASR_Encoder.onnx"
    DECODER_JOINT_NAME = "Nemotron_ASR_Decoder.onnx"

TOKENIZER_MODEL_NAME = "tokenizer.model"
TOKENIZER_VOCAB_NAME = "vocab.txt"

# Constants absent from model_config.yaml, plus export-only settings.
PREEMPH       = 0.97          # NeMo mel pre-emphasis default (absent from config)
LOG_GUARD     = 2.0 ** -24    # NeMo log_zero_guard_value default (absent from config)
LN_EPS        = 1e-5          # Conformer LayerNorm epsilon
DROP_EXTRA    = 2             # drop_extra_pre_encoded frames (cache-aware streaming)
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
MAX_SYMBOLS_PER_FRAME = int(_CFG.get("decoding", {}).get("greedy", {}).get("max_symbols", 10))
AUDIO_PCM_SCALE = 32768
AUTO_PROMPT_ID = int(_DEFAULTS_CFG.get("prompt_dictionary", {}).get("auto", 101))
_TOKEN_LABELS = list(_CFG.get("labels", _JOINT_CFG.get("vocabulary", ())))
UNKNOWN_ID = _TOKEN_LABELS.index("<unk>")

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
MAX_AUDIO_SAMPLES = (
    PE_MAX_LEN * SUB_FACTOR * HOP_LENGTH
    if DYNAMIC_AXES
    else FIXED_INPUT_AUDIO_LENGTH
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
    """Consolidate external sidecars one tensor at a time and stamp metadata."""
    import onnx
    from onnx import external_data_helper, numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    raw_sidecars = {
        entry.value
        for tensor in _iter_graph_tensors(model.graph)
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
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
    sidecar_paths = []
    for location in raw_sidecars:
        sidecar = (onnx_path.parent / location).resolve()
        sidecar_paths.append(sidecar)
    tensor_fields = ("float_data", "int32_data", "string_data", "int64_data",
                     "double_data", "uint64_data")

    def tensor_payload(tensor):
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            external_data_helper.load_external_data_for_tensor(tensor, str(onnx_path.parent))
        if tensor.raw_data:
            return bytes(tensor.raw_data)
        if tensor.data_type == onnx.TensorProto.STRING:
            return None
        packed = numpy_helper.from_array(numpy_helper.to_array(tensor), name=tensor.name)
        return bytes(packed.raw_data)

    def clear_inline_data(tensor):
        tensor.ClearField("raw_data")
        for field in tensor_fields:
            del getattr(tensor, field)[:]

    data_tmp = model_tmp = None
    wrote_external = False
    try:
        with tempfile.NamedTemporaryFile(prefix=data_name + ".", suffix=".tmp",
                                         dir=onnx_path.parent, delete=False) as data_out:
            data_tmp = Path(data_out.name)
            for tensor in _iter_graph_tensors(model.graph):
                was_external = tensor.data_location == onnx.TensorProto.EXTERNAL
                payload = tensor_payload(tensor)
                if payload is None:
                    if was_external:
                        raise RuntimeError(f"Cannot consolidate external string tensor {tensor.name!r}")
                    continue
                if len(payload) >= 1024:
                    offset = data_out.tell()
                    data_out.write(payload)
                    clear_inline_data(tensor)
                    tensor.raw_data = payload
                    external_data_helper.set_external_data(
                        tensor, location=data_name, offset=offset, length=len(payload))
                    # An empty-but-present raw_data field makes onnx.save_model()
                    # rewrite this already-written external range with zero bytes.
                    tensor.ClearField("raw_data")
                    wrote_external = True
                elif was_external:
                    clear_inline_data(tensor)
                    tensor.raw_data = payload
                    tensor.data_location = onnx.TensorProto.DEFAULT
                    del tensor.external_data[:]

        with tempfile.NamedTemporaryFile(prefix=onnx_path.name + ".", suffix=".tmp",
                                         dir=onnx_path.parent, delete=False) as model_out:
            model_tmp = Path(model_out.name)
        onnx.save_model(model, str(model_tmp), save_as_external_data=False)

        if wrote_external:
            data_tmp.replace(data_path)
        else:
            data_tmp.unlink(missing_ok=True)
            data_path.unlink(missing_ok=True)
        model_tmp.replace(onnx_path)

        merged_data_path = data_path.resolve()
        for sidecar in sidecar_paths:
            if sidecar != merged_data_path and sidecar.is_file():
                sidecar.unlink()
    finally:
        if data_tmp is not None:
            data_tmp.unlink(missing_ok=True)
        if model_tmp is not None:
            model_tmp.unlink(missing_ok=True)

# Asset loading
def ensure_assets(nemo_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = {}
    ckpt_path = out_dir / "model_weights.ckpt"
    need = not ckpt_path.exists() or not (out_dir / "model_config.yaml").exists()
    with tarfile.open(nemo_path, "r:*") as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}

        def extract_member(member, destination):
            tmp_path = None
            try:
                with tar.extractfile(member) as src, tempfile.NamedTemporaryFile(
                        prefix=destination.name + ".", suffix=".tmp",
                        dir=out_dir, delete=False) as out:
                    tmp_path = Path(out.name)
                    shutil.copyfileobj(src, out, length=8 << 20)
                tmp_path.replace(destination)
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)

        if need:
            for want, dst in (("model_weights.ckpt", ckpt_path),
                              ("model_config.yaml", out_dir / "model_config.yaml")):
                m = members.get(want)
                if m is not None:
                    extract_member(m, dst)
        tok = next((m for n, m in members.items() if n.endswith("_tokenizer.model") or n == "tokenizer.model"), None)
        voc = next((m for n, m in members.items() if n.endswith("_vocab.txt") or n == "vocab.txt"), None)
        if tok is not None and not (out_dir / TOKENIZER_MODEL_NAME).exists():
            extract_member(tok, out_dir / TOKENIZER_MODEL_NAME)
        if voc is not None and not (out_dir / TOKENIZER_VOCAB_NAME).exists():
            extract_member(voc, out_dir / TOKENIZER_VOCAB_NAME)
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
        ckpt_path.unlink()
        print(f"Removed temporary checkpoint: {ckpt_path}")

# Fused LayerNormalization op.
class _LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, epsilon, axis):
        mean = x.mean(dim=axis, keepdim=True)
        xc = x - mean
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


class _ASYMMETRIC_CONV_2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, pad_top, groups):
        return F.conv2d(F.pad(x, (2, 1, pad_top, 1)), weight, bias, stride=2, groups=groups)

    @staticmethod
    def symbolic(g, x, weight, bias, pad_top, groups):
        return g.op("Conv", x, weight, bias, dilations_i=[1, 1], group_i=groups,
                    kernel_shape_i=[3, 3], pads_i=[pad_top, 2, 1, 1], strides_i=[2, 2])


class _LEFT_PAD_CONV_1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, pad_left, groups):
        return F.conv1d(F.pad(x, (pad_left, 0)), weight, groups=groups)

    @staticmethod
    def symbolic(g, x, weight, pad_left, groups):
        return g.op("Conv", x, weight, dilations_i=[1], group_i=groups,
                    kernel_shape_i=[CONV_KERNEL], pads_i=[pad_left, 0], strides_i=[1])


class _STATIC_PAD_4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pads):
        return F.pad(x, (1, 0))

    @staticmethod
    def symbolic(g, x, pads):
        return g.op("Pad", x, pads, mode_s="constant")


class _GEMM_RESIDUAL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, residual):
        return F.linear(x, weight) + residual

    @staticmethod
    def symbolic(g, x, weight, residual):
        return g.op("Gemm", x, weight, residual, alpha_f=1.0, beta_f=1.0, transB_i=1)


def gemm_residual(x, weight, residual):
    return _GEMM_RESIDUAL.apply(x, weight, residual)


def swish(x):
    return x * torch.sigmoid(x)

class MetadataCarrier(nn.Module):
    def forward(self, marker: Tensor) -> Tensor:
        return marker

# Single-pass encoder graph.
class NemotronEncoder(nn.Module):
    def __init__(self, sd: dict, *, relative_max_len=PE_MAX_LEN,
                 enable_offline_attention=True, position_storage_dtype=FLOAT_STORAGE_DTYPE):
        super().__init__()
        relative_max_len = int(relative_max_len)
        g = lambda k: sd[k].float()
        self.register_buffer("c0_w", g("encoder.pre_encode.conv.0.weight"), persistent=True)
        self.register_buffer("c0_b", g("encoder.pre_encode.conv.0.bias"), persistent=True)
        self.register_buffer("c2_w", g("encoder.pre_encode.conv.2.weight"), persistent=True)
        self.register_buffer("c2_b", g("encoder.pre_encode.conv.2.bias"), persistent=True)
        self.register_buffer("c3_w", g("encoder.pre_encode.conv.3.weight"), persistent=True)
        self.register_buffer("c3_b", g("encoder.pre_encode.conv.3.bias"), persistent=True)
        self.register_buffer("c5_w", g("encoder.pre_encode.conv.5.weight"), persistent=True)
        self.register_buffer("c5_b", g("encoder.pre_encode.conv.5.bias"), persistent=True)
        self.register_buffer("c6_w", g("encoder.pre_encode.conv.6.weight"), persistent=True)
        self.register_buffer("c6_b", g("encoder.pre_encode.conv.6.bias"), persistent=True)
        self.register_buffer("out_w", g("encoder.pre_encode.out.weight"), persistent=True)
        self.register_buffer("out_b", g("encoder.pre_encode.out.bias"), persistent=True)

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
        stft_kernel = torch.cat([cos_k, sin_k], dim=0)
        if INPUT_AUDIO_DTYPE == "INT16":
            # 2^-15 is exact in fp32, so moving PCM scaling into this immutable kernel is bit-exact.
            stft_kernel = stft_kernel * (1.0 / 32768.0)
        self.register_buffer("stft_kernel", stft_kernel, persistent=True)
        self.register_buffer("fb", fb.squeeze(0).contiguous(), persistent=True)
        self.register_buffer("preemph", torch.tensor(PREEMPH, dtype=torch.float32), persistent=True)
        self.register_buffer("log_guard", torch.tensor(LOG_GUARD, dtype=torch.float32), persistent=True)
        self.f_bins = N_FFT // 2 + 1

        positions = torch.arange(relative_max_len - 1, -relative_max_len, -1, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D_MODEL, 2, dtype=torch.float32) * -(math.log(10000.0) / D_MODEL))
        pe = torch.zeros(positions.shape[0], D_MODEL)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        if enable_offline_attention:
            # Precomputed block mask reproduces chunked attention; int8 -128 becomes softmax zero.
            frame_index = torch.arange(relative_max_len, dtype=torch.int16)
            chunk_start = torch.div(frame_index, VALID_OUT_LEN, rounding_mode="floor") * VALID_OUT_LEN
            lo = (chunk_start - LEFT_CONTEXT).unsqueeze(1)
            hi = (chunk_start + ATT_CONTEXT_SIZE[1]).unsqueeze(1)
            valid = (frame_index.unsqueeze(0) >= lo) & (frame_index.unsqueeze(0) <= hi)
            attention_mask = torch.zeros(relative_max_len, relative_max_len, dtype=torch.int8)
            attention_mask.masked_fill_(~valid, -128)
            self.register_buffer("attention_mask", attention_mask[None, None].contiguous(), persistent=True)
            self.register_buffer("rel_shift_pad", torch.tensor(
                [0, 0, 0, 1, 0, 0, 0, 0], dtype=torch.int64), persistent=True)
        self.pe_center = relative_max_len

        self.register_buffer("ln_ones", torch.ones(D_MODEL), persistent=True)
        inv_sqrt_dk = HEAD_DIM ** -0.5

        for l in range(N_LAYERS):
            p = f"encoder.layers.{l}."

            def ln(name):
                return g(p + name + ".weight"), g(p + name + ".bias")

            # Fold LN affine into FF1; fold residual half-step into linear2.
            gw, gb = ln("norm_feed_forward1")
            l1w = g(p + "feed_forward1.linear1.weight")
            l2w = g(p + "feed_forward1.linear2.weight")
            self.register_buffer(f"ff1_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous(), persistent=True)
            self.register_buffer(f"ff1_l1b_{l}", torch.matmul(l1w, gb), persistent=True)
            self.register_buffer(f"ff1_l2w_{l}", (l2w * 0.5).contiguous(), persistent=True)

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
            self.register_buffer(f"qkv_w_{l}", qkv_w.contiguous(), persistent=True)
            self.register_buffer(f"qkv_b_{l}", qkv_b.contiguous(), persistent=True)
            # Linear projection commutes with the centered rel-pos slice; store attention matmul layout.
            pos_proj = F.linear(pe, g(p + "self_attn.linear_pos.weight")).reshape(-1, N_HEADS, HEAD_DIM)
            pos_proj = pos_proj.permute(1, 2, 0).contiguous().to(position_storage_dtype)
            self.register_buffer(f"pos_proj_{l}", pos_proj, persistent=True)
            # Pre-transposed to (N_HEADS, 1, HEAD_DIM) so q is transposed once before the bias add.
            self.register_buffer(f"bias_u_{l}", (g(p + "self_attn.pos_bias_u") * inv_sqrt_dk).unsqueeze(1).contiguous(), persistent=True)
            self.register_buffer(f"bias_v_{l}", (g(p + "self_attn.pos_bias_v") * inv_sqrt_dk).unsqueeze(1).contiguous(), persistent=True)
            self.register_buffer(f"out_w_{l}", g(p + "self_attn.linear_out.weight").contiguous(), persistent=True)

            # Fold norm_conv into pointwise_conv1; keep batch_norm affine.
            gw, gb = ln("norm_conv")
            pw1 = g(p + "conv.pointwise_conv1.weight").squeeze(-1)
            self.register_buffer(f"pw1_w_{l}", (pw1 * gw.unsqueeze(0)).contiguous(), persistent=True)
            self.register_buffer(f"pw1_b_{l}", torch.matmul(pw1, gb), persistent=True)
            self.register_buffer(f"dw_w_{l}", g(p + "conv.depthwise_conv.weight").contiguous(), persistent=True)
            bnw, bnb = ln("conv.batch_norm")
            self.register_buffer(f"bn_w_{l}", bnw, persistent=True)
            self.register_buffer(f"bn_b_{l}", bnb, persistent=True)
            self.register_buffer(f"pw2_w_{l}", g(p + "conv.pointwise_conv2.weight").squeeze(-1).contiguous(), persistent=True)

            gw, gb = ln("norm_feed_forward2")
            l1w = g(p + "feed_forward2.linear1.weight")
            l2w = g(p + "feed_forward2.linear2.weight")
            self.register_buffer(f"ff2_l1w_{l}", (l1w * gw.unsqueeze(0)).contiguous(), persistent=True)
            self.register_buffer(f"ff2_l1b_{l}", torch.matmul(l1w, gb), persistent=True)
            self.register_buffer(f"ff2_l2w_{l}", (l2w * 0.5).contiguous(), persistent=True)

            ow, ob = ln("norm_out")
            self.register_buffer(f"no_w_{l}", ow, persistent=True)
            self.register_buffer(f"no_b_{l}", ob, persistent=True)

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
        self.register_buffer("W0_enc", W0_enc.contiguous(), persistent=True)
        self.register_buffer("lang_bias", lang_bias.contiguous(), persistent=True)
        self.register_buffer("Wc", Wc.contiguous(), persistent=True)
        self.register_buffer("bc", bc.contiguous(), persistent=True)

    def _causal_conv2d(self, x, w, b, groups, pad_top=2):
        return _ASYMMETRIC_CONV_2D.apply(x, w, b, pad_top, groups)

    def _pre_encode(self, feats, cache_frames=0, output_frames=None):
        x = feats.transpose(1, 2).unsqueeze(1)
        x = F.relu(self._causal_conv2d(x, self.c0_w, self.c0_b, 1, 2 + cache_frames))
        x = self._causal_conv2d(x, self.c2_w, self.c2_b, SUB_CHANNELS)
        x = F.conv2d(x, self.c3_w, self.c3_b)
        x = F.relu(x)
        x = self._causal_conv2d(x, self.c5_w, self.c5_b, SUB_CHANNELS)
        x = F.conv2d(x, self.c6_w, self.c6_b)
        x = F.relu(x)
        x = x.transpose(1, 2)
        if output_frames is None:
            x = x.flatten(2)
        else:
            x = x.reshape(1, output_frames, self.out_w.shape[1])
        return F.linear(x, self.out_w, self.out_b)

    def _rel_shift_full(self, x, L, b):
        # Transformer-XL skew: out[i, j] reads centered relative distance i - j.
        x = _STATIC_PAD_4D.apply(x, self.rel_shift_pad)
        x = x.reshape(b, -1, 2 * L, L)
        x = x[:, :, 1:, :]
        x = x.reshape(b, -1, L, 2 * L - 1)
        return x[:, :, :, :L]

    def _preprocess(self, audio):
        # Float inputs are assumed normalized; INT16 scaling is folded into stft_kernel.
        x = audio.float()
        x = torch.cat([x[:, :, :1], x[:, :, 1:] - self.preemph * x[:, :, :-1]], dim=2)
        stft = F.conv1d(x, self.stft_kernel, stride=HOP_LENGTH, padding=N_FFT // 2).square()
        real_sq, imag_sq = torch.split(stft, self.f_bins, dim=1)
        power = real_sq + imag_sq
        # Keep fb on the left; Optimize_ONNX.py skips FusionGemm to avoid onnxslim's bad const@var rewrite.
        mel = torch.matmul(self.fb, power)
        mel = torch.clamp_min(mel, self.log_guard)
        features = torch.log(mel)
        # Match the streaming path: keep floor(S / HOP) frames.
        return features[:, :, :-1]

    def forward(self, audio, prompt_id):
        # Full-sequence graph; block mask reproduces cache-aware chunk attention.
        features = self._preprocess(audio)
        x = self._pre_encode(features, PRE_ENCODE_CACHE_FRAMES)
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
            xc = _LEFT_PAD_CONV_1D.apply(xc, getattr(self, f"dw_w_{l}"), CONV_CACHE, D_MODEL)
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
        key_index = torch.arange(STREAM_KV_LEN, dtype=torch.int16).unsqueeze(0)
        cache_lengths = torch.arange(LEFT_CONTEXT + 1, dtype=torch.int16).unsqueeze(1)
        valid = key_index >= (LEFT_CONTEXT - cache_lengths)
        stream_masks = torch.zeros(LEFT_CONTEXT + 1, 1, STREAM_KV_LEN)
        stream_masks.masked_fill_(~valid.unsqueeze(1), -128.0)
        self.register_buffer("stream_masks", stream_masks.contiguous(), persistent=True)

        shift_index = ((VALID_OUT_LEN - 1 - torch.arange(VALID_OUT_LEN, dtype=torch.int32)).unsqueeze(1)
                       + torch.arange(STREAM_KV_LEN, dtype=torch.int32).unsqueeze(0))
        shift_index = shift_index.reshape(1, VALID_OUT_LEN, STREAM_KV_LEN)
        self.register_buffer("rel_shift_index", shift_index.expand(
            N_HEADS, VALID_OUT_LEN, STREAM_KV_LEN).contiguous(), persistent=True)

        pos_lo = enc.pe_center - STREAM_KV_LEN
        pos_hi = enc.pe_center + STREAM_KV_LEN - 1
        for l in range(N_LAYERS):
            pos_name = f"pos_proj_{l}"
            pos_t = getattr(enc, pos_name)[..., pos_lo:pos_hi].float().contiguous()
            self.register_buffer(f"pos_t_{l}", pos_t, persistent=True)
            delattr(enc, pos_name)
            qkv_w = getattr(enc, f"qkv_w_{l}")
            qkv_b = getattr(enc, f"qkv_b_{l}")
            self.register_buffer(f"q_w_{l}", qkv_w[:D_MODEL], persistent=True)
            self.register_buffer(f"q_b_{l}", qkv_b[:D_MODEL], persistent=True)
            self.register_buffer(f"kv_w_{l}", qkv_w[D_MODEL:], persistent=True)
            self.register_buffer(f"kv_b_{l}", qkv_b[D_MODEL:], persistent=True)
            delattr(enc, f"qkv_w_{l}")
            delattr(enc, f"qkv_b_{l}")

    def _stream_mel(self, audio):
        # Snip-edges STFT over the window (no centre padding); the leading sample only seeds pre-emphasis.
        enc = self.enc
        x = audio.float()
        x = x[:, :, 1:] - enc.preemph * x[:, :, :-1]
        stft = F.conv1d(x, enc.stft_kernel, stride=HOP_LENGTH).square()
        real_sq, imag_sq = torch.split(stft, enc.f_bins, dim=1)
        power = real_sq + imag_sq
        mel = torch.matmul(enc.fb, power)
        mel = torch.clamp_min(mel, enc.log_guard)
        return torch.log(mel)

    def forward(self, audio, mel_cache, cache_last_channel, cache_last_time, cache_len, prompt_id):
        enc = self.enc
        mel_new = self._stream_mel(audio)                       # (1, N_MELS, STREAM_MEL_CHUNK)
        mel_full = torch.cat([mel_cache, mel_new], dim=2)       # (1, N_MELS, STREAM_MEL_WINDOW)
        mel_cache_next = mel_full[:, :, -STREAM_MEL_CACHE:]
        x = enc._pre_encode(mel_full, output_frames=VALID_OUT_LEN + DROP_EXTRA)
        x = x.squeeze(0)[DROP_EXTRA:]                            # (VALID_OUT_LEN, D_MODEL)

        # Startup mask: one precomputed row per cache_len; all queries share the broadcast row.
        neg = torch.index_select(self.stream_masks, 0, cache_len)
        cache_last_channel = cache_last_channel.squeeze(1)

        channel_next = []
        time_next = []
        for l in range(N_LAYERS):
            residual = x
            m = layer_norm(x, enc.ln_ones)
            m = F.linear(m, getattr(enc, f"ff1_l1w_{l}"), getattr(enc, f"ff1_l1b_{l}"))
            m = swish(m)
            residual = gemm_residual(m, getattr(enc, f"ff1_l2w_{l}"), residual)

            m = layer_norm(residual, enc.ln_ones)                  # norm_self_att == cache content
            m_full = torch.cat([cache_last_channel[l], m], dim=0)  # (STREAM_KV_LEN, D_MODEL)
            channel_next.append(m_full[-LEFT_CONTEXT:])
            q = F.linear(m, getattr(self, f"q_w_{l}"), getattr(self, f"q_b_{l}"))
            q = q.reshape(VALID_OUT_LEN, N_HEADS, HEAD_DIM).transpose(0, 1)
            kv = F.linear(m_full, getattr(self, f"kv_w_{l}"), getattr(self, f"kv_b_{l}"))
            kv = kv.reshape(STREAM_KV_LEN, 2 * N_HEADS, HEAD_DIM).transpose(0, 1)
            k, v = torch.split(kv, N_HEADS, dim=0)
            q_u = q + getattr(enc, f"bias_u_{l}")
            q_v = q + getattr(enc, f"bias_v_{l}")
            k_t = k.transpose(1, 2)
            ac = torch.matmul(q_u, k_t)
            bd = torch.matmul(q_v, getattr(self, f"pos_t_{l}"))
            bd = torch.gather(bd, 2, self.rel_shift_index)
            scores = ac + bd + neg
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v)
            ctx = ctx.transpose(0, 1).reshape(VALID_OUT_LEN, D_MODEL)
            residual = gemm_residual(ctx, getattr(enc, f"out_w_{l}"), residual)

            m = layer_norm(residual, enc.ln_ones)
            xc = F.linear(m, getattr(enc, f"pw1_w_{l}"), getattr(enc, f"pw1_b_{l}"))
            xc = F.glu(xc, dim=1)
            xc = xc.transpose(0, 1).unsqueeze(0)
            dw_in = torch.cat([cache_last_time[l], xc], dim=2)
            time_next.append(dw_in[:, :, -CONV_CACHE:])
            xc = F.conv1d(dw_in, getattr(enc, f"dw_w_{l}"), groups=D_MODEL)
            xc = xc.squeeze(0).transpose(0, 1)
            xc = layer_norm(xc, getattr(enc, f"bn_w_{l}"), getattr(enc, f"bn_b_{l}"))
            xc = swish(xc)
            residual = gemm_residual(xc, getattr(enc, f"pw2_w_{l}"), residual)

            m = layer_norm(residual, enc.ln_ones)
            m = F.linear(m, getattr(enc, f"ff2_l1w_{l}"), getattr(enc, f"ff2_l1b_{l}"))
            m = swish(m)
            residual = gemm_residual(m, getattr(enc, f"ff2_l2w_{l}"), residual)

            x = layer_norm(residual, getattr(enc, f"no_w_{l}"), getattr(enc, f"no_b_{l}"))

        bias = torch.index_select(enc.lang_bias, 0, prompt_id).squeeze(0)
        h0 = F.relu(F.linear(x, enc.W0_enc, bias))
        enc_proj = F.linear(h0, enc.Wc, enc.bc).unsqueeze(0)

        cache_last_channel_next = torch.cat(channel_next, dim=0).reshape(
            N_LAYERS, 1, LEFT_CONTEXT, D_MODEL)
        cache_last_time_next = torch.cat(time_next, dim=0).unsqueeze(1)
        cache_len_next = torch.clamp(cache_len + VALID_OUT_LEN, max=LEFT_CONTEXT)
        return enc_proj, mel_cache_next, cache_last_channel_next, cache_last_time_next, cache_len_next


class NemotronDecoderJoint(nn.Module):
    def __init__(self, sd: dict, blank_id: int):
        super().__init__()
        self.blank_id = int(blank_id)
        self.embed = nn.Embedding(LOGITS_SIZE, PRED_HIDDEN, dtype=FLOAT_STORAGE_DTYPE)
        self.lstm = nn.LSTM(PRED_HIDDEN, PRED_HIDDEN, LSTM_LAYERS, batch_first=True)
        with torch.no_grad():
            self.embed.weight.copy_(sd["decoder.prediction.embed.weight"].to(FLOAT_STORAGE_DTYPE))
            for li in range(LSTM_LAYERS):
                getattr(self.lstm, f"weight_ih_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.weight_ih_l{li}"].float())
                getattr(self.lstm, f"weight_hh_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.weight_hh_l{li}"].float())
                getattr(self.lstm, f"bias_ih_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.bias_ih_l{li}"].float())
                getattr(self.lstm, f"bias_hh_l{li}").copy_(sd[f"decoder.prediction.dec_rnn.lstm.bias_hh_l{li}"].float())
        self.register_buffer("pred_w", sd["joint.pred.weight"].float().contiguous(), persistent=True)
        self.register_buffer("pred_b", sd["joint.pred.bias"].float().contiguous(), persistent=True)
        self.register_buffer("jnet_w", sd["joint.joint_net.2.weight"].float().contiguous(), persistent=True)
        self.register_buffer("jnet_b", sd["joint.joint_net.2.bias"].float().contiguous(), persistent=True)

    def forward(self, enc_proj, frame_idx, token, state_h, state_c):
        enc_proj_frame = torch.flatten(torch.index_select(enc_proj, 1, frame_idx), start_dim=1)
        emb = self.embed(token).float()
        out, (h, c) = self.lstm(emb, (state_h, state_c))
        pred = F.linear(torch.flatten(out, start_dim=1), self.pred_w, self.pred_b)
        z = torch.relu(enc_proj_frame + pred)
        logits = F.linear(z, self.jnet_w, self.jnet_b)
        argmax = torch.argmax(logits, dim=-1, keepdim=True)
        is_blank = argmax == self.blank_id
        argmax = argmax.to(torch.int32)
        # Blank steps keep token/state unchanged for in-place IOBinding. is_blank is
        # (batch, 1); it broadcasts against the (layers, batch, hidden) LSTM state through an
        # implicit leading axis, so the batch dimension aligns without an unsqueeze.
        next_token = torch.where(is_blank, token, argmax)
        h = torch.where(is_blank, state_h, h)
        c = torch.where(is_blank, state_c, c)
        return next_token, is_blank.to(torch.int32), h, c


# Metadata
_LANGUAGE_NAMES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "ay": "Aymara",
    "az": "Azerbaijani", "bg": "Bulgarian", "bn": "Bengali", "cs": "Czech",
    "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "fa": "Persian", "fi": "Finnish",
    "fr": "French", "gn": "Guarani", "gu": "Gujarati", "ha": "Hausa",
    "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi", "hr": "Croatian",
    "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian", "ig": "Igbo",
    "it": "Italian", "ja": "Japanese", "ka": "Georgian", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "ku": "Kurdish", "ky": "Kyrgyz",
    "ln": "Lingala", "lt": "Lithuanian", "lv": "Latvian", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mr": "Marathi", "ms": "Malay",
    "mt": "Maltese", "nah": "Nahuatl", "nb": "Norwegian Bokmal",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian",
    "ny": "Chichewa", "or": "Odia", "pl": "Polish", "pt": "Portuguese",
    "qu": "Quechua", "ro": "Romanian", "ru": "Russian", "rw": "Kinyarwanda",
    "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sm": "Samoan",
    "so": "Somali", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "to": "Tongan",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yo": "Yoruba", "zh": "Chinese", "zu": "Zulu",
}


def _build_supported_languages(prompt_dictionary: dict) -> dict:
    """Group the authoritative prompt dictionary by prompt ID without losing aliases."""
    grouped = {}
    for spelling, raw_prompt_id in prompt_dictionary.items():
        prompt_id = int(raw_prompt_id)
        grouped.setdefault(prompt_id, []).append(spelling)

    catalog = {}
    for prompt_id, spellings in sorted(grouped.items()):
        canonical = next(
            (item for item in spellings if item == "auto"),
            next((item for item in spellings if "-" in item), spellings[0]),
        )
        base = canonical.split("-", 1)[0].lower()
        entry = {
            "name": "Automatic language detection" if canonical == "auto" else _LANGUAGE_NAMES.get(base, canonical),
            "aliases": sorted(item for item in spellings if item != canonical),
            "prompt_id": prompt_id,
            "status": "automatic" if canonical == "auto" else "supported",
        }
        catalog[canonical] = entry
    return catalog


def make_metadata(cfg: dict, assets: dict) -> dict:
    del assets
    model_defaults = cfg.get("model_defaults", {})
    prompt_dictionary = model_defaults.get("prompt_dictionary", {})
    metadata = {
        "sample_rate": SAMPLE_RATE,
        "audio_pcm_scale": AUDIO_PCM_SCALE,
        "max_symbols_per_frame": MAX_SYMBOLS_PER_FRAME,
        "special_token_ids": {"blank": BLANK_ID, "unknown": UNKNOWN_ID},
        "supported_languages": _build_supported_languages(prompt_dictionary),
    }
    if STREAMING:
        metadata.update({
            "stream_stride_samples": STREAM_STRIDE_SAMPLES,
            "stream_left_overlap": STREAM_LEFT_OVERLAP,
        })
    return build_model_metadata(metadata)


def _offline_encoder_frames(num_samples: int) -> int:
    """Exact static encoder-frame count for the three stride-2 pre-encode convolutions."""
    frames = int(num_samples) // HOP_LENGTH
    frames = (frames + (2 + PRE_ENCODE_CACHE_FRAMES) + 1 - 3) // 2 + 1
    for _ in range(2):
        frames = (frames + 2 + 1 - 3) // 2 + 1
    return frames - DROP_EXTRA


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

        if STREAMING:
            relative_max_len = STREAM_KV_LEN
        elif DYNAMIC_AXES:
            relative_max_len = PE_MAX_LEN
        else:
            relative_max_len = _offline_encoder_frames(FIXED_INPUT_AUDIO_LENGTH)
        offline_encoder = NemotronEncoder(
            sd,
            relative_max_len=relative_max_len,
            enable_offline_attention=not STREAMING,
            position_storage_dtype=FLOAT_STORAGE_DTYPE,
        ).eval()
        encoder = NemotronStreamingEncoder(offline_encoder).eval() if STREAMING else offline_encoder

        with torch.inference_mode():
            p = ONNX_FOLDER / METADATA_NAME
            torch.onnx.export(MetadataCarrier().eval(), (torch.zeros(1, dtype=torch.int64),), str(p),
                              input_names=["metadata_marker"], output_names=["metadata_marker_out"],
                              opset_version=OPSET, dynamo=False)
            finalize_graph(p)
            write_metadata_carrier(p, metadata)

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

            # Decoder construction/export does not need the encoder; release its large immutable state first.
            encoder = None
            offline_encoder = None
            gc.collect()
            decjoint = NemotronDecoderJoint(sd, BLANK_ID).eval()
            sd = None
            gc.collect()

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

    finally:
        del encoder, decjoint, sd, offline_encoder
        gc.collect()
        remove_extracted_checkpoint(assets)


if __name__ == "__main__":
    export_all()
    subprocess.run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "Inference_Nemotron_ASR_ONNX.py"),
            "--onnx-folder",
            str(ONNX_FOLDER),
        ],
        cwd=str(_SCRIPT_DIR),
        check=True,
    )
