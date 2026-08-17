import atexit
import ctypes
import errno
import gc
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import uuid
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn
from torch.onnx import symbolic_helper
from transformers import AutoTokenizer

from STFT_Process import STFT_Process



# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
download_path                  = str(Path.home() / "Downloads" / "ARK-ASR-0.6B")         # Set the path where the ARK-ASR-[0.6B, 1.7B] model downloaded.
script_dir                     = Path(__file__).resolve().parent
final_onnx_folder              = script_dir / "ARK_ASR_ONNX"                             # Final merged deployment folder.
_bundle_export_temp            = tempfile.TemporaryDirectory(
    prefix=".ark_asr_export_", dir=script_dir
)
onnx_folder                    = Path(_bundle_export_temp.name) / final_onnx_folder.name
_split_export_temp             = tempfile.TemporaryDirectory(prefix="ark_asr_split_")     # Auto-cleaned staging area; never retained in the workspace.
split_export_folder            = Path(_split_export_temp.name)
atexit.register(_bundle_export_temp.cleanup)
atexit.register(_split_export_temp.cleanup)

MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "ARK_ASR_Encoder.onnx",
    "embed": "ARK_ASR_Decoder_Embed.onnx",
    "main": "ARK_ASR_Decoder_Main.onnx",
    "rotary_prefill": "ARK_ASR_Rotary_Mask_Text_Prefill.onnx",
    "rotary_decode": "ARK_ASR_Rotary_Mask_Text_Decode.onnx",
    # Functional roles: plain greedy is Argmax; history-tracking greedy is used
    # after Apply_Penalty.  The source artifact names are intentionally inverted.
    "greedy": "ARK_ASR_Argmax.onnx",
    "penalty_greedy": "ARK_ASR_Greedy_Search.onnx",
    "penalty": "ARK_ASR_Apply_Penalty.onnx",
    "sampling": "ARK_ASR_TopKTopPSampling.onnx",
    "prefill_greedy": "ARK_ASR_Prefill_Greedy.onnx",
    "prefill_penalty_greedy": "ARK_ASR_Prefill_Penalty_Greedy.onnx",
    "prefill_sampling": "ARK_ASR_PrefillSampling.onnx",
    "decode_greedy": "ARK_ASR_Decode_Greedy.onnx",
    "decode_penalty_greedy": "ARK_ASR_Decode_Penalty_Greedy.onnx",
    "decode_sampling": "ARK_ASR_DecodeSampling.onnx",
    "shared_initializers": "ARK_ASR_SharedInitializers.onnx",
}
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"

onnx_model_Metadata            = str(split_export_folder / MODEL_FILE_NAMES["metadata"])
onnx_model_Encoder             = str(split_export_folder / MODEL_FILE_NAMES["encoder"])
onnx_model_Embed               = str(split_export_folder / MODEL_FILE_NAMES["embed"])
onnx_model_Main                = str(split_export_folder / MODEL_FILE_NAMES["main"])
onnx_model_Rotary_Mask_Prefill = str(split_export_folder / MODEL_FILE_NAMES["rotary_prefill"])
onnx_model_Rotary_Mask_Decode  = str(split_export_folder / MODEL_FILE_NAMES["rotary_decode"])
onnx_model_Greedy              = str(split_export_folder / MODEL_FILE_NAMES["penalty_greedy"])
onnx_model_TopKTopP_Sampling   = str(split_export_folder / MODEL_FILE_NAMES["sampling"])
onnx_model_Penalty             = str(split_export_folder / MODEL_FILE_NAMES["penalty"])
onnx_model_Argmax              = str(split_export_folder / MODEL_FILE_NAMES["greedy"])


# ============================== USER CONFIG ==============================
MAX_INPUT_AUDIO_LENGTH         = 480000                        # Maximum deployment audio length (30 s at the model's fixed 16 kHz sample rate).
MAX_SEQ_LEN                    = 1024                          # Maximum context length, including prompt + audio + decode tokens.
USE_FP16_KV                    = True                          # Use FP16 KV cache for normal deployment exports.
COMPUTE_IN_F32                 = False                         # F16-KV compute precision. False = minimum-cast f16 attention (Q@K/mask/softmax/attn@V all run in f16 on the f16 KV cache; storage AND compute f16). True = keep the f16 KV *storage* (cache I/O dtype unchanged) but upcast K/V to f32 at the matmul use points and keep Q/mask/softmax in f32 (f16 storage, f32 compute). No effect when USE_FP16_KV=False.
INPUT_AUDIO_DTYPE              = "F32"                         # Model audio input dtype: "INT16", "F32", or "F16". "INT16" feeds raw PCM (÷32768 inside the graph). "F32"/"F16" feed audio already normalised to [-1, 1] (the in-graph ÷32768 is skipped); "F16" is cast up to f32 for compute.

# Weight-quantization-friendly reorder (exact and absorbed into the weights).
REORDER_DOWNPROJ_FOR_QUANT     = True                          # Reorder MLP intermediate channels so down_proj block-quant groups are magnitude-homogeneous.
REORDER_OPROJ_FOR_QUANT        = True                          # Reorder each head's head_dim so o_proj sub-head groups are homogeneous. Pure win for f16 KV.
REORDER_KEY                    = "absmean"                     # "absmean" (best at group=32) | "L4" (best at group=128) | "rms" | "std".

OPSET                          = 20                            # ONNX Runtime opset version.
# ========================================================================

# Fixed ARK-ASR model constants and metadata defaults; these are not user tunables.
_MODEL_SAMPLE_RATE             = 16000
_MODEL_WINDOW_TYPE             = "hann"
_MODEL_NUM_MELS                = 128
_MODEL_NFFT_STFT               = 400
_MODEL_WINDOW_LENGTH           = 400
_MODEL_HOP_LENGTH              = 160
_MODEL_AUDIO_PCM_SCALE         = 32768

ROTARY_STORAGE_DTYPE           = torch.float16 if USE_FP16_KV else torch.float32


def build_model_metadata(*sections):
    metadata = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (dict, list, tuple)):
                metadata[str(key)] = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            else:
                metadata[str(key)] = str(value)
    return metadata


def replace_onnx_metadata(onnx_path, metadata):
    import onnx

    expected = {str(key): str(value) for key, value in metadata.items()}
    model = onnx.load(str(onnx_path), load_external_data=False)
    model.producer_name = ""
    model.producer_version = ""
    del model.metadata_props[:]
    for key in sorted(expected):
        model.metadata_props.add(key=key, value=expected[key])
    onnx.save_model(model, str(onnx_path), save_as_external_data=False)


def _try_atomic_directory_exchange(left: Path, right: Path) -> bool:
    """Atomically exchange two same-filesystem directories when Linux supports it."""
    if os.name != "posix":
        return False
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        return False
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        0x2,
    )
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
        return False
    raise OSError(error, os.strerror(error), str(left))


def publish_deployment_bundle(staging_folder: Path, destination_folder: Path) -> None:
    """Publish a complete staged bundle without exposing a partial destination."""
    if not staging_folder.is_dir():
        raise RuntimeError(f"Missing staged deployment bundle: {staging_folder}")
    if not destination_folder.exists():
        os.replace(staging_folder, destination_folder)
        return
    if not destination_folder.is_dir():
        raise RuntimeError(
            f"Deployment destination is not a directory: {destination_folder}"
        )
    if _try_atomic_directory_exchange(staging_folder, destination_folder):
        shutil.rmtree(staging_folder)
        return

    backup_folder = destination_folder.with_name(
        f".{destination_folder.name}.previous-{uuid.uuid4().hex}"
    )
    os.replace(destination_folder, backup_folder)
    try:
        os.replace(staging_folder, destination_folder)
    except BaseException:
        os.replace(backup_folder, destination_folder)
        raise
    shutil.rmtree(backup_folder)


def load_local_ark_model(model_path: str):
    """Load ARK's local remote-code modules without relying on its folder name."""
    model_dir = Path(model_path).resolve()
    package_name = "_ark_asr_local_model"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    package.__package__ = package_name
    sys.modules[package_name] = package

    def load_module(module_name: str):
        qualified_name = f"{package_name}.{module_name}"
        spec = importlib.util.spec_from_file_location(
            qualified_name, model_dir / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified_name] = module
        spec.loader.exec_module(module)
        return module

    config_module = load_module("configuration_arkasr")
    model_module = load_module("modeling_arkasr")
    config = config_module.ArkasrConfig.from_pretrained(str(model_dir))
    model = model_module.ArkasrForConditionalGeneration.from_pretrained(
        str(model_dir), config=config, torch_dtype=torch.float32
    )
    return config, model


# ══════════════════════════════════════════════════════════════════════════════
def get_kv_io(
    tensors_dict:  Dict[str, Tensor],
    kv_specs:      Sequence[Tuple[str, int]],
    num_layers:    int,
    batch_axis:    str = "batch",
    seq_axis:      str = "history_len",
    out_seq_axis:  str = "kv_seq_len",
) -> Tuple[List[Tensor], List[str], List[str], Dict[str, Dict[int, str]]]:
    """Build KV I/O tensor lists, name lists, and dynamic-axes dict for onnx.export."""
    inputs:       List[Tensor]              = []
    input_names:  List[str]                 = []
    output_names: List[str]                 = []
    dynamic_axes: Dict[str, Dict[int, str]] = {}
    for name, dim in kv_specs:
        tensor = tensors_dict[name]
        for i in range(num_layers):
            in_name  = f"past_{name}_{i}"
            out_name = f"present_{name}_{i}"
            inputs.append(tensor)
            input_names.append(in_name)
            output_names.append(out_name)
            dynamic_axes[in_name]  = {0: batch_axis, dim: seq_axis}
            dynamic_axes[out_name] = {0: batch_axis, dim: out_seq_axis}
    return inputs, input_names, output_names, dynamic_axes


class POSITIVE_CEIL_DIV(torch.autograd.Function):
    """Positive int64 ceil-div without legacy-exporter cast chains."""

    @staticmethod
    def forward(ctx, value: Tensor, divisor: int) -> Tensor:
        return (value + divisor - 1) // divisor

    @staticmethod
    def symbolic(g, value, divisor):
        divisor_value = symbolic_helper._get_const(divisor, "i", "divisor")
        offset = g.op(
            "Constant",
            value_t=torch.tensor([divisor_value - 1], dtype=torch.int64),
        )
        denominator = g.op(
            "Constant",
            value_t=torch.tensor([divisor_value], dtype=torch.int64),
        )
        return g.op("Div", g.op("Add", value, offset), denominator)


class ONNX_SHAPE_DIM(torch.autograd.Function):
    """Return one dimension as an int64 vector via ONNX Shape start/end."""

    @staticmethod
    def forward(ctx, x: Tensor, axis: int) -> Tensor:
        return torch._shape_as_tensor(x)[axis:axis + 1]

    @staticmethod
    def symbolic(g, x, axis):
        axis_value = symbolic_helper._get_const(axis, "i", "axis")
        return g.op("Shape", x, start_i=axis_value, end_i=axis_value + 1)


class ONNX_STATIC_RESHAPE(torch.autograd.Function):
    """Emit Reshape with a constant target; zero copies the matching input dim."""

    @staticmethod
    def forward(ctx, x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        eager_shape = tuple(
            x.shape[index] if dim == 0 else dim
            for index, dim in enumerate(shape)
        )
        return x.reshape(eager_shape)

    @staticmethod
    def symbolic(g, x, shape):
        shape_const = g.op(
            "Constant", value_t=torch.tensor(shape, dtype=torch.int64)
        )
        return g.op("Reshape", x, shape_const)


def onnx_reshape_batch(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return ONNX_STATIC_RESHAPE.apply(x, (0,) + tuple(shape))


class ONNX_WHISPER_ROPE(torch.autograd.Function):
    """Emit Whisper's pairwise RoPE as a compact ONNX subgraph."""

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        rotary_dim: int,
        num_heads: int,
    ) -> Tensor:
        rotary = x[..., :rotary_dim]
        paired = rotary.reshape(*rotary.shape[:-1], rotary_dim // 2, 2)
        first, second = torch.split(paired, 1, dim=-1)
        flipped = torch.cat([second, first], dim=-1).flatten(-2)
        rotated = rotary * cos + flipped * sin
        return torch.cat([rotated, x[..., rotary_dim:]], dim=-1)

    @staticmethod
    def symbolic(g, x, cos, sin, rotary_dim, num_heads):
        rotary_dim_value = symbolic_helper._get_const(
            rotary_dim, "i", "rotary_dim"
        )
        num_heads_value = symbolic_helper._get_const(
            num_heads, "i", "num_heads"
        )

        def constant(values):
            return g.op(
                "Constant", value_t=torch.tensor(values, dtype=torch.int64)
            )

        starts = constant([0])
        ends = constant([rotary_dim_value])
        axes = constant([4])
        steps = constant([1])
        rotary = g.op("Slice", x, starts, ends, axes, steps)
        paired = g.op(
            "Reshape",
            rotary,
            constant([0, 2, num_heads_value, -1, rotary_dim_value // 2, 2]),
        )
        first, second = g.op(
            "Split", paired, constant([1, 1]), axis_i=5, outputs=2
        )
        flipped = g.op("Concat", second, first, axis_i=5)
        flipped = g.op(
            "Reshape",
            flipped,
            constant([0, 2, num_heads_value, -1, rotary_dim_value]),
        )
        rotated = g.op(
            "Add",
            g.op("Mul", rotary, cos),
            g.op("Mul", flipped, sin),
        )
        tail = g.op(
            "Slice",
            x,
            ends,
            constant([torch.iinfo(torch.int64).max]),
            axes,
            steps,
        )
        return g.op("Concat", rotated, tail, axis_i=4)


class PENALIZE_LOGITS(torch.autograd.Function):
    """Keep history indices int32 in exported GatherElements/ScatterElements."""

    @staticmethod
    def forward(
        ctx, logits: Tensor, target_indices: Tensor, penalty_value: Tensor
    ) -> Tensor:
        indices = target_indices.long()
        penalized = logits.gather(1, indices) * penalty_value
        return logits.scatter(1, indices, penalized)

    @staticmethod
    def symbolic(g, logits, target_indices, penalty_value):
        selected = g.op("GatherElements", logits, target_indices, axis_i=1)
        penalized = g.op("Mul", selected, penalty_value)
        return g.op(
            "ScatterElements",
            logits,
            target_indices,
            penalized,
            axis_i=1,
            reduction_s="none",
        )


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Emit ORT's fused mean-based RMS normalization in the default ONNX domain."""

    @staticmethod
    def forward(ctx, x: Tensor, scale: Tensor, epsilon: float, axis: int) -> Tensor:
        variance = x.pow(2).mean(dim=axis, keepdim=True)
        return x * torch.rsqrt(variance + epsilon) * scale

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x,
            scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(
    x: Tensor, scale: Tensor, epsilon: float, axis: int = -1
) -> Tensor:
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


# ══════════════════════════════════════════════════════════════════════════════
# ── ARK-ASR audio + Qwen2 decoder modules ────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class ARK_ASR_ENCODER(torch.nn.Module):
    """WhisperFeatureExtractor + WhisperSpecialEncoder + AudioMLPAdapter.

    The static prompt prefix is ``<|user|><|begin_of_audio|>``.  Each adapter
    output replaces exactly one dynamic ``<|audio|>`` placeholder, and the
    runtime supplies the remaining ``<|end_of_audio|>task<|assistant|>`` tail.
    """

    def __init__(
        self,
        audio_encoder: torch.nn.Module,
        embed_tokens: torch.nn.Embedding,
        prefix_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.whisper = audio_encoder.whisper
        self.adapter_norm = audio_encoder.layer_norm
        self.adapter = audio_encoder.adapting
        self.merge_factor = int(audio_encoder.merge_factor)
        self.model_dim = int(self.whisper.config.d_model)
        self.num_heads = int(self.whisper.config.encoder_attention_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.rotary_dim = int(self.whisper.rotary_embedding.dim)
        self.attention_scale = self.head_dim ** -0.5
        # Scaling before RoPE is bit-exact for a finite binary power-of-two.
        self.fold_q_attention_scale = (
            math.isfinite(self.attention_scale)
            and math.frexp(self.attention_scale)[0] == 0.5
        )
        self._fuse_whisper_qkv_weights()

        self.input_audio_is_int16 = INPUT_AUDIO_DTYPE == "INT16"
        stft_input_scale = (
            1.0 / _MODEL_AUDIO_PCM_SCALE if self.input_audio_is_int16 else 1.0
        )
        self.stft = STFT_Process(
            model_type="stft_B",
            n_fft=_MODEL_NFFT_STFT,
            win_length=_MODEL_WINDOW_LENGTH,
            hop_len=_MODEL_HOP_LENGTH,
            max_frames=0,
            window_type=_MODEL_WINDOW_TYPE,
            center_pad=True,
            pad_mode="reflect",
            input_scale=stft_input_scale,
            drop_last_frame=True,
        ).eval()
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=(_MODEL_NFFT_STFT // 2) + 1,
            f_min=0.0,
            f_max=_MODEL_SAMPLE_RATE / 2,
            n_mels=_MODEL_NUM_MELS,
            sample_rate=_MODEL_SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1).unsqueeze(0).contiguous()
        self.register_buffer(
            "mel_filters", mel_filters.to(dtype=torch.float32), persistent=True
        )

        max_encoder_frames = (MAX_INPUT_AUDIO_LENGTH // _MODEL_HOP_LENGTH + 1) // 2
        rotary_positions = torch.arange(max_encoder_frames, dtype=torch.float32).unsqueeze(-1)
        rotary_inv_freq = 1.0 / (
            10000.0
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )
        rotary_angles = rotary_positions * rotary_inv_freq
        rotary_cos = torch.cos(rotary_angles).view(1, 1, max_encoder_frames, -1)
        rotary_sin = torch.sin(rotary_angles).view(1, 1, max_encoder_frames, -1)
        self.register_buffer(
            "whisper_rotary_cos",
            torch.stack([rotary_cos, rotary_cos], dim=-1).flatten(-2).half(),
            persistent=True,
        )
        self.register_buffer(
            "whisper_rotary_sin",
            torch.stack([-rotary_sin, rotary_sin], dim=-1).flatten(-2).half(),
            persistent=True,
        )
        self.register_buffer(
            "merge_pad",
            torch.zeros((1, self.merge_factor - 1, self.model_dim), dtype=torch.float16),
            persistent=True,
        )
        with torch.no_grad():
            prefix = embed_tokens(
                torch.tensor([list(prefix_ids)], dtype=torch.long)
            ).detach().float()
        self.register_buffer("prompt_prefix_embed", prefix, persistent=True)

    def _fuse_whisper_qkv_weights(self) -> None:
        """Pack each encoder attention projection into one immutable Linear."""
        with torch.no_grad():
            for layer_index, layer in enumerate(self.whisper.layers):
                attn = layer.self_attn
                projections = (attn.q_proj, attn.k_proj, attn.v_proj)
                reference = projections[0]
                if any(
                    projection.in_features != reference.in_features
                    or projection.out_features != reference.out_features
                    for projection in projections[1:]
                ):
                    raise RuntimeError(
                        f"Whisper encoder layer {layer_index} has incompatible Q/K/V projections."
                    )
                qkv = torch.nn.Linear(
                    reference.in_features,
                    sum(projection.out_features for projection in projections),
                    bias=True,
                    device=reference.weight.device,
                    dtype=reference.weight.dtype,
                )
                qkv.weight.copy_(
                    torch.cat([projection.weight for projection in projections], dim=0)
                )
                qkv.bias.copy_(
                    torch.cat(
                        [
                            projection.bias
                            if projection.bias is not None
                            else torch.zeros(
                                projection.out_features,
                                device=reference.weight.device,
                                dtype=reference.weight.dtype,
                            )
                            for projection in projections
                        ],
                        dim=0,
                    )
                )
                if self.fold_q_attention_scale:
                    qkv.weight[:reference.out_features].mul_(self.attention_scale)
                    qkv.bias[:reference.out_features].mul_(self.attention_scale)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj

    def _apply_whisper_rope(
        self, x: Tensor, cos: Tensor, sin: Tensor
    ) -> Tensor:
        return ONNX_WHISPER_ROPE.apply(
            x, cos, sin, self.rotary_dim, self.num_heads
        )

    def forward(self, audio: Tensor, prompt_tail_embed: Tensor) -> Tuple[Tensor, Tensor]:
        # This is WhisperFeatureExtractor's centered, reflect-padded log-mel
        # path.  STFT_Process removes the trailing unused centered frame, giving
        # ``floor(raw_samples / hop_length)`` dynamic mel frames.
        audio = audio.float()
        real, imag = self.stft(audio)
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_filters, power)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        input_features = mel * 0.25 + 1.0

        hidden_states = F.gelu(self.whisper.conv1(input_features))
        hidden_states = F.gelu(self.whisper.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)
        encoder_len = ONNX_SHAPE_DIM.apply(hidden_states, 1)
        rotary_cos = self.whisper_rotary_cos[:, :, :encoder_len].float()
        rotary_sin = self.whisper_rotary_sin[:, :, :encoder_len].float()

        for layer in self.whisper.layers:
            residual = hidden_states
            normed = layer.self_attn_layer_norm(hidden_states)
            attn = layer.self_attn
            qkv = attn.qkv(normed)
            qkv = onnx_reshape_batch(
                qkv, (-1, 3, self.num_heads, self.head_dim)
            ).permute(0, 2, 3, 1, 4)
            qk, v = torch.split(qkv, [2, 1], dim=1)
            v = v.squeeze(1)
            qk = self._apply_whisper_rope(qk, rotary_cos, rotary_sin)
            q, k = torch.unbind(qk, dim=1)
            weights = torch.matmul(q, k.transpose(-1, -2))
            if not self.fold_q_attention_scale:
                weights = weights * self.attention_scale
            weights = torch.softmax(weights, dim=-1)
            attended = torch.matmul(weights, v).transpose(1, 2)
            attended = onnx_reshape_batch(attended, (-1, self.model_dim))
            hidden_states = residual + attn.out_proj(attended)

            residual = hidden_states
            normed = layer.final_layer_norm(hidden_states)
            hidden_states = residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))

        encoded = self.adapter_norm(hidden_states)
        encoded_len = ONNX_SHAPE_DIM.apply(encoded, 1)
        pad_len = torch.clamp(self.merge_factor - encoded_len, min=0)
        encoded = torch.cat([encoded, self.merge_pad[:, :pad_len].float()], dim=1)
        merge_len = torch.maximum(
            encoded_len,
            torch.tensor([self.merge_factor], dtype=torch.int64),
        )
        merge_len = (merge_len // self.merge_factor) * self.merge_factor
        encoded = encoded[:, :merge_len]
        encoded = onnx_reshape_batch(
            encoded, (-1, self.model_dim * self.merge_factor)
        )
        audio_hidden = self.adapter(encoded)
        prefill_embed = torch.cat(
            [self.prompt_prefix_embed, audio_hidden, prompt_tail_embed], dim=1
        )
        return prefill_embed, ONNX_SHAPE_DIM.apply(prefill_embed, 1)


class ARK_ASR_ROTARY_MASK_PREFILL(torch.nn.Module):
    def __init__(self, rope_theta: float, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.mask_dtype = (
            torch.float32
            if USE_FP16_KV and COMPUTE_IN_F32
            else (torch.float16 if USE_FP16_KV else torch.float32)
        )
        mask_positions = torch.arange(max_seq_len, dtype=torch.int32)
        self.register_buffer(
            "causal_mask",
            torch.where(
                mask_positions.view(1, 1, 1, max_seq_len, 1)
                >= mask_positions.view(1, 1, 1, 1, max_seq_len),
                torch.zeros((), dtype=self.mask_dtype),
                torch.full((), -128.0, dtype=self.mask_dtype),
            ),
            persistent=True,
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        theta = (positions * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.rotary_dim = head_dim
        self.register_buffer(
            "rotary_pos_emb",
            torch.cat(
                [torch.cat([cos, cos], dim=-1), torch.cat([-sin, sin], dim=-1)],
                dim=-1,
            ).to(ROTARY_STORAGE_DTYPE),
            persistent=True,
        )

    def forward(
        self, ids_len: Tensor, history_len: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        kv_seq_len = ids_len + history_len
        rotary = self.rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_cos, rotary_sin = torch.split(
            rotary, [self.rotary_dim, self.rotary_dim], dim=-1
        )
        attention_mask = self.causal_mask[..., :ids_len, :kv_seq_len]
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ARK_ASR_ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, rope_theta: float, head_dim: int, max_seq_len: int) -> None:
        super().__init__()
        positions = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        theta = (positions * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos, sin = torch.cos(theta), torch.sin(theta)
        self.rotary_dim = head_dim
        self.register_buffer(
            "rotary_pos_emb",
            torch.cat(
                [torch.cat([cos, cos], dim=-1), torch.cat([-sin, sin], dim=-1)],
                dim=-1,
            ).to(ROTARY_STORAGE_DTYPE),
            persistent=True,
        )

    def forward(self, kv_seq_len: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        kv_seq_len_next = kv_seq_len + 1
        rotary = self.rotary_pos_emb[:, kv_seq_len].float()
        rotary_cos, rotary_sin = torch.split(
            rotary, [self.rotary_dim, self.rotary_dim], dim=-1
        )
        return rotary_cos, rotary_sin, kv_seq_len_next


class ARK_ASR_DECODER_EMBED(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embed_tokens = model.model.embed_tokens

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)


class ARK_ASR_DECODER_MAIN(torch.nn.Module):
    """Fused Qwen2 decoder with native ``[B, KVH, S, D]`` K/V cache tensors."""

    def __init__(
        self,
        model: torch.nn.Module,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.llm = model.model
        self.lm_head = model.lm_head
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.num_layers = num_layers
        self.qk_heads = num_heads + num_kv_heads
        self.total_qkv_heads = self.qk_heads + num_kv_heads
        self.qkv_split_sizes = (self.qk_heads, num_kv_heads)
        self.qk_split_sizes = (num_heads, num_kv_heads)
        self.attention_output_size = int(self.llm.layers[0].self_attn.o_proj.in_features)
        self.intermediate_size = int(self.llm.layers[0].mlp.down_proj.in_features)
        self.mlp_split_sizes = (self.intermediate_size, self.intermediate_size)
        self.use_fp16_kv = USE_FP16_KV
        self.compute_in_f32 = COMPUTE_IN_F32
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _fuse_weights(self) -> None:
        with torch.no_grad():
            for layer in self.llm.layers:
                attn = layer.self_attn
                projections = (attn.q_proj, attn.k_proj, attn.v_proj)
                has_bias = any(projection.bias is not None for projection in projections)
                qkv = torch.nn.Linear(
                    attn.q_proj.in_features,
                    attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features,
                    bias=has_bias,
                )
                qkv.weight.copy_(torch.cat([projection.weight for projection in projections], dim=0))
                if has_bias:
                    biases = [
                        projection.bias
                        if projection.bias is not None
                        else torch.zeros(projection.out_features, dtype=qkv.weight.dtype)
                        for projection in projections
                    ]
                    qkv.bias.copy_(torch.cat(biases, dim=0))
                q_out = attn.q_proj.out_features
                qkv.weight[:q_out].mul_(self.head_dim ** -0.5)
                if qkv.bias is not None:
                    qkv.bias[:q_out].mul_(self.head_dim ** -0.5)
                attn.qkv = qkv

                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                gate_up = torch.nn.Linear(
                    gate.in_features,
                    gate.out_features + up.out_features,
                    bias=gate.bias is not None or up.bias is not None,
                )
                gate_up.weight.copy_(torch.cat([gate.weight, up.weight], dim=0))
                if gate_up.bias is not None:
                    gate_bias = gate.bias if gate.bias is not None else torch.zeros(gate.out_features, dtype=gate_up.weight.dtype)
                    up_bias = up.bias if up.bias is not None else torch.zeros(up.out_features, dtype=gate_up.weight.dtype)
                    gate_up.bias.copy_(torch.cat([gate_bias, up_bias], dim=0))
                layer.mlp.gate_up_proj = gate_up
                del attn.q_proj, attn.k_proj, attn.v_proj
                del layer.mlp.gate_proj, layer.mlp.up_proj

    def _reorder_downproj_for_quant(self, key: str) -> None:
        with torch.no_grad():
            for layer in self.llm.layers:
                weights = layer.mlp.down_proj.weight
                absolute = weights.abs()
                if key == "rms":
                    statistic = (weights * weights).mean(0).sqrt()
                elif key == "L4":
                    statistic = absolute.pow(4).mean(0).pow(0.25)
                elif key == "std":
                    statistic = weights.std(0)
                else:
                    statistic = absolute.mean(0)
                permutation = torch.argsort(statistic)
                intermediate = layer.mlp.down_proj.in_features
                gate_up = layer.mlp.gate_up_proj.weight
                layer.mlp.gate_up_proj.weight.copy_(
                    torch.cat(
                        [gate_up[:intermediate][permutation], gate_up[intermediate:][permutation]],
                        dim=0,
                    )
                )
                layer.mlp.down_proj.weight.copy_(weights[:, permutation])

    def _reorder_oproj_for_quant(self, key: str) -> None:
        groups = self.num_kv_groups
        with torch.no_grad():
            for layer in self.llm.layers:
                weights = layer.self_attn.o_proj.weight
                by_head = weights.view(weights.shape[0], self.num_heads, self.head_dim)
                permutations = []
                for kv_head in range(self.num_kv_heads):
                    columns = by_head[:, kv_head * groups:(kv_head + 1) * groups, :]
                    absolute = columns.abs()
                    if key == "rms":
                        statistic = (columns * columns).mean(dim=(0, 1)).sqrt()
                    elif key == "std":
                        statistic = columns.reshape(-1, self.head_dim).std(0)
                    elif key == "L4":
                        statistic = absolute.pow(4).mean(dim=(0, 1)).pow(0.25)
                    else:
                        statistic = absolute.mean(dim=(0, 1))
                    permutations.append(torch.argsort(statistic))
                reordered = by_head.clone()
                for head in range(self.num_heads):
                    reordered[:, head, :] = reordered[:, head, permutations[head // groups]]
                weights.copy_(reordered.reshape(weights.shape[0], -1))

                qkv_weights = layer.self_attn.qkv.weight
                qkv_by_head = qkv_weights.view(-1, self.head_dim, qkv_weights.shape[1]).clone()
                value_offset = self.num_heads + self.num_kv_heads
                for kv_head in range(self.num_kv_heads):
                    qkv_by_head[value_offset + kv_head] = qkv_by_head[value_offset + kv_head][permutations[kv_head]]
                qkv_weights.copy_(qkv_by_head.reshape(qkv_weights.shape[0], qkv_weights.shape[1]))
                if layer.self_attn.qkv.bias is not None:
                    qkv_bias = layer.self_attn.qkv.bias
                    qkv_bias_by_head = qkv_bias.view(-1, self.head_dim).clone()
                    for kv_head in range(self.num_kv_heads):
                        qkv_bias_by_head[value_offset + kv_head] = qkv_bias_by_head[
                            value_offset + kv_head
                        ][permutations[kv_head]]
                    qkv_bias.copy_(qkv_bias_by_head.reshape(qkv_bias.shape[0]))

    def _rms_norm(self, x: Tensor, scale: Tensor, eps: float) -> Tensor:
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x: Tensor, head_count: int) -> Tensor:
        x = onnx_reshape_batch(
            x, (-1, 1, head_count, 2, self.head_dim_half)
        )
        x = x.flip(-2)
        return onnx_reshape_batch(x, (-1, 1, head_count, self.head_dim))

    def forward(self, *all_inputs: Tensor) -> Tuple[Tensor, ...]:
        hidden_states = all_inputs[-4]
        rotary_cos = all_inputs[-3]
        rotary_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]
        for layer_index, layer in enumerate(self.llm.layers):
            residual = hidden_states
            norm = layer.input_layernorm
            hidden_states = self._rms_norm(
                hidden_states,
                norm.weight,
                float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))),
            )
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = onnx_reshape_batch(
                qkv, (-1, 1, self.total_qkv_heads, self.head_dim)
            )
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)
            qk = qk * rotary_cos + self._rotate_half(qk, self.qk_heads) * rotary_sin
            q, k = torch.split(qk, self.qk_split_sizes, dim=-2)
            if self.use_fp16_kv:
                if not self.compute_in_f32:
                    q = q.half()
                k = k.half()
                v = v.half()

            q = onnx_reshape_batch(
                q, (-1, self.num_kv_heads, self.num_kv_groups, self.head_dim)
            ).permute(0, 2, 3, 1, 4)
            k = onnx_reshape_batch(k, (-1, self.num_kv_heads, self.head_dim)).permute(0, 2, 1, 3)
            v = onnx_reshape_batch(v, (-1, self.num_kv_heads, self.head_dim)).permute(0, 2, 1, 3)
            k = torch.cat([all_inputs[layer_index], k], dim=2)
            v = torch.cat([all_inputs[layer_index + self.num_layers], v], dim=2)
            self.save_key[layer_index] = k
            self.save_value[layer_index] = v
            if self.use_fp16_kv and self.compute_in_f32:
                attention = torch.matmul(
                    q, k.float().unsqueeze(2).transpose(-1, -2)
                ) + attention_mask
                attention = torch.softmax(attention, dim=-1)
                attention = torch.matmul(attention, v.float().unsqueeze(2))
            else:
                attention = torch.matmul(
                    q, k.unsqueeze(2).transpose(-1, -2)
                ) + attention_mask
                attention = torch.softmax(attention, dim=-1)
                attention = torch.matmul(attention, v.unsqueeze(2))
            attention = onnx_reshape_batch(
                attention.permute(0, 3, 1, 2, 4),
                (-1, self.attention_output_size),
            )
            if self.use_fp16_kv and not self.compute_in_f32:
                attention = attention.float()
            hidden_states = residual + layer.self_attn.o_proj(attention)

            residual = hidden_states
            norm = layer.post_attention_layernorm
            hidden_states = self._rms_norm(
                hidden_states,
                norm.weight,
                float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))),
            )
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, self.mlp_split_sizes, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        norm = self.llm.norm
        hidden_states = self._rms_norm(
            hidden_states[:, -1],
            norm.weight,
            float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))),
        )
        logits = self.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoding Strategy Modules (mirrors Export_Fun_ASR_Nano.py) ────────────────
# ══════════════════════════════════════════════════════════════════════════════
class ARK_SPECIAL_TOKEN_FILTER(torch.nn.Module):
    """Mirror ARK's ``bad_words_ids`` policy inside the split search graphs."""

    def __init__(self, blocked_token_ids: Sequence[int]) -> None:
        super().__init__()
        self.register_buffer(
            "blocked_token_ids",
            torch.tensor([list(blocked_token_ids)], dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "blocked_logits",
            torch.full((1, len(blocked_token_ids)), float("-inf"), dtype=torch.float32),
            persistent=False,
        )

    def suppress_special_tokens(self, logits: Tensor) -> Tensor:
        indices = self.blocked_token_ids.repeat(logits.size(0), 1)
        values = self.blocked_logits.repeat(logits.size(0), 1).to(logits.dtype)
        return logits.scatter(1, indices, values)


class GREEDY_SEARCH(ARK_SPECIAL_TOKEN_FILTER):
    def __init__(self, blocked_token_ids: Sequence[int]) -> None:
        super().__init__(blocked_token_ids)

    def forward(self, logits: Tensor, save_id: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.suppress_special_tokens(logits)
        max_idx = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)
        return max_idx, torch.cat([save_id, max_idx], dim=-1)


class TOPK_TOPP_SAMPLING(ARK_SPECIAL_TOKEN_FILTER):
    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self, blocked_token_ids: Sequence[int]):
        super().__init__(blocked_token_ids)
        self.register_buffer(
            "neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_min", torch.tensor(self.GUMBEL_EPS, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_max",
            torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        logits: Tensor,
        temperature: Tensor,
        top_k: Tensor,
        top_p: Tensor,
        repetition_penalty: Tensor,
        previous_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        logits = self.suppress_special_tokens(logits)
        inv_penalty = torch.reciprocal(repetition_penalty)
        prev_logits = torch.gather(logits, 1, previous_ids)
        prev_scores = torch.where(
            prev_logits < 0.0,
            prev_logits * repetition_penalty,
            prev_logits * inv_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids, prev_scores)
        scores = scores * torch.reciprocal(temperature)

        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep_topp = (sorted_cumsum - sorted_probs) <= top_p
        sorted_scores = torch.where(keep_topp, sorted_scores, self.neg_inf)

        noise = torch.clamp(
            torch.rand_like(sorted_scores), self.gumbel_min, self.gumbel_max
        )
        gumbel = -torch.log(-torch.log(noise))
        winner = torch.argmax(sorted_scores + gumbel, dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        save_id = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_id


class APPLY_PENALTY(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits:        Tensor,
        save_id:       Tensor,
        penalty_value: Tensor,
        penalty_range: Tensor,
    ) -> Tensor:
        target_indices = save_id[:, -penalty_range:]
        return PENALIZE_LOGITS.apply(logits, target_indices, penalty_value)


class ARGMAX(ARK_SPECIAL_TOKEN_FILTER):
    def __init__(self, blocked_token_ids: Sequence[int]) -> None:
        super().__init__(blocked_token_ids)

    def forward(self, logits: Tensor) -> Tensor:
        logits = self.suppress_special_tokens(logits)
        return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


# ══════════════════════════════════════════════════════════════════════════════
# ── Export Loop ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
onnx_folder.mkdir(parents=True)

print("\nExport start …\n")

with torch.inference_mode():

    # ── Load model ────────────────────────────────────────────────────────────
    config, model = load_local_ark_model(download_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict) and "rope_theta" in rope_parameters:
        rope_theta = float(rope_parameters["rope_theta"])
    elif hasattr(config, "rope_theta"):
        rope_theta = float(config.rope_theta)
    else:
        raise RuntimeError(
            "ARK model configuration is missing the decoder rope_theta value."
        )

    whisper_cfg = config.whisper_config
    num_layers = int(config.num_hidden_layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    hidden_size = int(config.hidden_size)
    head_dim = hidden_size // num_heads
    vocab_size = int(config.vocab_size)
    tokenizer_vocabulary = tokenizer.get_vocab()
    user_token_id = int(tokenizer_vocabulary["<|user|>"])
    begin_audio_token_id = int(tokenizer_vocabulary["<|begin_of_audio|>"])
    end_audio_token_id = int(tokenizer_vocabulary["<|end_of_audio|>"])
    assistant_token_id = int(tokenizer_vocabulary["<|assistant|>"])
    eos_token_id = int(config.eos_token_id)
    pad_token_id = int(config.pad_token_id)
    output_special_token_ids = {
        int(token_id) for token_id in tokenizer.all_special_ids
    }
    output_special_token_ids.update(
        int(token_id)
        for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">")
    )
    blocked_output_token_ids = sorted(output_special_token_ids - {eos_token_id})
    special_token_ids = {
        "stop": [eos_token_id],
        "pad": pad_token_id,
        "user": user_token_id,
        "begin_audio": begin_audio_token_id,
        "audio": int(config.audio_token_id),
        "end_audio": end_audio_token_id,
        "assistant": assistant_token_id,
        "eos": eos_token_id,
        "blocked_output_token_ids": blocked_output_token_ids,
        "remove_from_output": sorted(output_special_token_ids),
    }
    kv_dtype = torch.float16 if USE_FP16_KV else torch.float32
    # ARK follows Qwen2's native cache representation for both K and V.
    kv_specs = [("key", 2), ("value", 2)]

    print(
        f"  Encoder : layers={whisper_cfg.encoder_layers}, "
        f"d_model={whisper_cfg.d_model}, merge_factor={config.merge_factor}"
    )
    print(f"  Decoder : layers={num_layers}, heads={num_heads}/{num_kv_heads} GQA, head_dim={head_dim}")
    print(f"  KV dtype: {'float16' if USE_FP16_KV else 'float32'}")

    prompt_prefix_ids = [
        special_token_ids["user"],
        special_token_ids["begin_audio"],
    ]

    # Trace-only values establish rank/type examples; every corresponding runtime
    # dimension or decoding value is an ONNX input and remains dynamic.
    _dummy_seq_len = 16
    _dummy_batch_size = 10
    _dummy_history_len = 10
    _dummy_penalty_value = 1.0
    _dummy_penalty_range = _dummy_history_len
    dummy_prompt_tail_ids = torch.tensor(
        [[special_token_ids["end_audio"], special_token_ids["assistant"]]],
        dtype=torch.int32,
    )
    dummy_prompt_tail_embed = model.model.embed_tokens(dummy_prompt_tail_ids)

    ids_len     = torch.tensor([_dummy_seq_len], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    kv_seq_len  = ids_len + history_len
    logits      = torch.ones((_dummy_batch_size, vocab_size), dtype=torch.float32)
    save_id     = torch.zeros((_dummy_batch_size, 0), dtype=torch.int32)

    kv_tensors  = {
        "key":   torch.zeros((_dummy_batch_size, num_kv_heads, 0, head_dim), dtype=kv_dtype),
        "value": torch.zeros((_dummy_batch_size, num_kv_heads, 0, head_dim), dtype=kv_dtype),
    }

    # ── Fused Audio Encoder (pre-process + encoder in one graph) ─────────────
    encoder = ARK_ASR_ENCODER(
        model.audio_encoder,
        model.model.embed_tokens,
        prompt_prefix_ids,
    ).eval()
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    _dummy_audio_length = _MODEL_SAMPLE_RATE
    dummy_audio = torch.ones((1, 1, _dummy_audio_length), dtype=_audio_export_dtype)
    torch.onnx.export(
        encoder,
        (dummy_audio, dummy_prompt_tail_embed),
        onnx_model_Encoder,
        input_names=["audio", "prompt_tail_embed"],
        output_names=["hidden_states", "ids_len"],
        dynamic_axes={
            "audio": {2: "audio_len"},
            "prompt_tail_embed": {1: "prompt_tail_len"},
            "hidden_states": {1: "total_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del encoder, dummy_audio, dummy_prompt_tail_ids, dummy_prompt_tail_embed
    gc.collect()

    # ── Decoder Embed ─────────────────────────────────────────────────────────
    embed_mod = ARK_ASR_DECODER_EMBED(model).eval()
    dummy_ids  = torch.ones((1, _dummy_seq_len), dtype=torch.int32)
    torch.onnx.export(
        embed_mod,
        (dummy_ids,),
        onnx_model_Embed,
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids":     {0: "batch", 1: "ids_len"},
            "hidden_states": {0: "batch", 1: "ids_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del embed_mod
    gc.collect()

    # ── Rotary + Mask — Prefill ────────────────────────────────────────────────
    rotary_prefill = ARK_ASR_ROTARY_MASK_PREFILL(
        rope_theta, head_dim, MAX_SEQ_LEN
    ).eval()
    torch.onnx.export(
        rotary_prefill,
        (ids_len, history_len),
        onnx_model_Rotary_Mask_Prefill,
        input_names=["ids_len", "history_len"],
        output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        dynamic_axes={
            "rotary_cos":     {1: "ids_len"},
            "rotary_sin":     {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del rotary_prefill
    gc.collect()

    # ── Rotary — Decode (no attention_mask output) ─────────────────────────────
    rotary_decode = ARK_ASR_ROTARY_MASK_DECODE(
        rope_theta, head_dim, MAX_SEQ_LEN
    ).eval()
    torch.onnx.export(
        rotary_decode,
        (kv_seq_len,),
        onnx_model_Rotary_Mask_Decode,
        input_names=["kv_seq_len"],
        output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
        dynamic_axes={},
        opset_version=OPSET,
        dynamo=False,
    )
    del rotary_decode
    gc.collect()

    # ── Decoder Main ──────────────────────────────────────────────────────────
    kv_inputs, kv_input_names, kv_output_names, kv_axes = get_kv_io(kv_tensors, kv_specs, num_layers)

    hidden_states  = torch.ones((_dummy_batch_size, _dummy_seq_len, hidden_size), dtype=torch.float32)
    rotary_cos     = torch.ones((1, _dummy_seq_len, 1, 1, head_dim),             dtype=torch.float32)
    rotary_sin     = torch.zeros((1, _dummy_seq_len, 1, 1, head_dim),            dtype=torch.float32)
    attention_mask_dtype = torch.float32 if (USE_FP16_KV and COMPUTE_IN_F32) else kv_dtype
    # Main receives the mask in its attention-compute dtype; f16 KV storage is unchanged.
    attention_mask = torch.zeros((1, 1, 1, _dummy_seq_len, _dummy_seq_len),      dtype=attention_mask_dtype)

    all_inputs   = kv_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]
    input_names  = kv_input_names + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"]
    output_names = kv_output_names + ["logits"]
    dynamic_axes = {
        **kv_axes,
        "hidden_states":  {0: "batch", 1: "ids_len"},
        "rotary_cos":     {1: "ids_len"},
        "rotary_sin":     {1: "ids_len"},
        "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        "logits":         {0: "batch"},
    }

    # ── Decoder Main ──────────────────────────────────────────────────────────
    decoder_main = ARK_ASR_DECODER_MAIN(
        model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size
    ).eval()
    del model
    gc.collect()

    torch.onnx.export(
        decoder_main,
        tuple(all_inputs),
        onnx_model_Main,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )
    del decoder_main
    gc.collect()

    # ── Greedy Search ─────────────────────────────────────────────────────────
    torch.onnx.export(
        GREEDY_SEARCH(blocked_output_token_ids).eval(),
        (logits[[0]], save_id[[0]]),
        onnx_model_Greedy,
        input_names=["logits", "save_id_in"],
        output_names=["max_logits_idx", "save_id_out"],
        dynamic_axes={
            "logits":         {0: "batch"},
            "save_id_in":     {0: "batch", 1: "history_len"},
            "max_logits_idx": {0: "batch"},
            "save_id_out":    {0: "batch", 1: "history_len_out"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── Apply Penalty ─────────────────────────────────────────────────────────
    dummy_save_id = torch.zeros(
        (_dummy_batch_size, _dummy_history_len), dtype=torch.int32
    )
    penalty_value = torch.tensor([_dummy_penalty_value], dtype=torch.float32)
    penalty_range = torch.tensor([_dummy_penalty_range], dtype=torch.int64)
    torch.onnx.export(
        APPLY_PENALTY().eval(),
        (logits, dummy_save_id, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=["logits_in", "save_id_in", "penalty_value", "penalty_range"],
        output_names=["logits_out"],
        dynamic_axes={
            "logits_in":  {0: "batch"},
            "save_id_in": {0: "batch", 1: "history_len"},
            "logits_out": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── Argmax ────────────────────────────────────────────────────────────────
    torch.onnx.export(
        ARGMAX(blocked_output_token_ids).eval(),
        (logits,),
        onnx_model_Argmax,
        input_names=["logits"],
        output_names=["max_logits_idx"],
        dynamic_axes={
            "logits":         {0: "batch"},
            "max_logits_idx": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── Top-K / Top-P Sampling ────────────────────────────────────────────────
    sampling_temperature = torch.tensor([0.8], dtype=torch.float32)
    sampling_top_k = torch.tensor([50], dtype=torch.int32)
    sampling_top_p = torch.tensor([0.95], dtype=torch.float32)
    sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    sampling_previous_ids = torch.zeros((1, _dummy_history_len), dtype=torch.int32)
    torch.onnx.export(
        TOPK_TOPP_SAMPLING(blocked_output_token_ids).eval(),
        (
            logits[[0]],
            sampling_temperature,
            sampling_top_k,
            sampling_top_p,
            sampling_repetition_penalty,
            sampling_previous_ids,
        ),
        onnx_model_TopKTopP_Sampling,
        input_names=[
            "logits",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "previous_ids",
        ],
        output_names=["sampled_id", "save_id_out"],
        dynamic_axes={
            "previous_ids": {1: "history_len"},
            "save_id_out": {1: "history_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del sampling_temperature, sampling_top_k, sampling_top_p
    del sampling_repetition_penalty, sampling_previous_ids
    del kv_inputs, kv_tensors, logits, save_id
    gc.collect()

    onnx_metadata = build_model_metadata(
        {
            "audio_pcm_scale": _MODEL_AUDIO_PCM_SCALE,
            "max_audio_samples": MAX_INPUT_AUDIO_LENGTH,
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": _MODEL_SAMPLE_RATE,
            "max_mel_frames": MAX_INPUT_AUDIO_LENGTH // _MODEL_HOP_LENGTH,
            "padding_side": "right",
            "merge_factor": int(config.merge_factor),
            "audio_token_count_formula": "max(((mel_frames + 1) // 2) // merge_factor, 1)",
            "prompt_template": "<|user|><|begin_of_audio|><|audio|>*N<|end_of_audio|>{task}<|assistant|>",
            "special_token_ids": special_token_ids,
        },
    )
    metadata_marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        onnx_model_Metadata,
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False,
    )
    del metadata_marker

    replace_onnx_metadata(onnx_model_Metadata, onnx_metadata)
    # ── Save the tokenizer into the ONNX folder so the exported folder runs inference ──
    # stand-alone (no external ARK-ASR model path needed at inference time).
    _tokenizer_dir = onnx_folder / "tokenizer"
    tokenizer.save_pretrained(str(_tokenizer_dir))
    print(f"[Tokenizer] Saved tokenizer -> {_tokenizer_dir}")

    # ── Compose the deployment graphs around one data-less Main and stream all
    # large Main initializers into one mmap-friendly external-data bundle. ─────
    import Shared_Merged

    print("\n[SharedMerged] Building ASR prefill/decode strategy graphs ...")
    _bundle = Shared_Merged.build_shared_merged_bundle(
        split_export_folder,
        out_folder=onnx_folder,
        model_file_names=MODEL_FILE_NAMES,
    )
    # Encoder and Main optimization donors were already saved as data-light graphs
    # referencing the shared blob. When the token-embedding table was shared, Embed
    # was likewise saved into onnx_folder. Do not overwrite any of those outputs
    # with their fat split-export copies.
    _embed_dedup = _bundle.get("embed_dedup")
    _embed_consolidated = _bundle.get("embed_consolidated")
    _embed_shared = _embed_dedup or _embed_consolidated
    _skip_standalone = ("encoder", "embed") if _embed_shared else ("encoder",)
    _copied_standalones = Shared_Merged.copy_runtime_standalones(
        split_export_folder,
        onnx_folder,
        MODEL_FILE_NAMES,
        skip_roles=_skip_standalone,
    )

    replace_onnx_metadata(
        str(onnx_folder / MODEL_FILE_NAMES["metadata"]),
        onnx_metadata,
    )

    for _name, _path in _bundle["graphs"].items():
        print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
    print(
        f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
        f"({Path(_bundle['shared_data']).stat().st_size} bytes)"
    )
    if _embed_dedup:
        print(
            f"    {MODEL_FILE_NAMES['embed']} shares the tied lm_head table "
            f"({Path(onnx_folder / MODEL_FILE_NAMES['embed']).stat().st_size} bytes; "
            "reads the [hidden, vocab] weight from the shared bundle)"
        )
    elif _embed_consolidated:
        print(
            f"    {MODEL_FILE_NAMES['embed']} reads its embedding table from the shared "
            f"bundle ({Path(onnx_folder / MODEL_FILE_NAMES['embed']).stat().st_size} bytes; "
            "untied [vocab, hidden] table streamed once into the shared blob)"
        )
    print(f"    Standalone ASR graphs copied: {len(_copied_standalones)}")

    _split_export_temp.cleanup()
    print("[SharedMerged] Removed automatic split-graph staging directory.")

try:
    publish_deployment_bundle(onnx_folder, final_onnx_folder)
finally:
    _bundle_export_temp.cleanup()
print("\nExport complete.\n")
subprocess.run(
    [
        sys.executable,
        str(script_dir / "Inference_ARK_ASR_ONNX.py"),
        "--onnx-folder",
        str(final_onnx_folder),
    ],
    cwd=str(script_dir),
    check=True,
)
