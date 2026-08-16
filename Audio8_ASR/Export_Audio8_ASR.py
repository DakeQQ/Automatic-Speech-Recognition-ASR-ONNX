import atexit
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.onnx import symbolic_helper
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
# Works across transformers 4.5x and 5.x (only the config + module-tree closure is used).

from STFT_Process import STFT_Process



# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
download_path                  = str(Path.home() / "Downloads" / "Audio8-ASR-0.1B")         # Official local Audio8-ASR checkpoint folder.
script_dir                     = Path(__file__).resolve().parent
final_onnx_folder              = script_dir / "Audio8_ASR_ONNX"                             # Final merged deployment folder.
_output_export_temp            = tempfile.TemporaryDirectory(prefix="audio8_asr_output_")   # Auto-cleaned final-bundle staging area.
onnx_folder                    = Path(_output_export_temp.name) / final_onnx_folder.name
_split_export_temp             = tempfile.TemporaryDirectory(prefix="audio8_asr_split_")    # Auto-cleaned staging area; never retained in the workspace.
split_export_folder            = Path(_split_export_temp.name)
atexit.register(_output_export_temp.cleanup)
atexit.register(_split_export_temp.cleanup)

MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Audio8_ASR_Encoder.onnx",
    "embed": "Audio8_ASR_Decoder_Embed.onnx",
    "concat_embed": "Audio8_ASR_Concat_Embed.onnx",
    "main": "Audio8_ASR_Decoder_Main.onnx",
    "rotary_prefill": "Audio8_ASR_Rotary_Mask_Text_Prefill.onnx",
    "rotary_decode": "Audio8_ASR_Rotary_Mask_Text_Decode.onnx",
    # Functional roles: plain greedy is Argmax; history-tracking greedy is used
    # after Apply_Penalty.  The source artifact names are intentionally inverted.
    "greedy": "Audio8_ASR_Argmax.onnx",
    "penalty_greedy": "Audio8_ASR_Greedy_Search.onnx",
    "penalty": "Audio8_ASR_Apply_Penalty.onnx",
    "sampling": "Audio8_ASR_TopKTopPSampling.onnx",
    "prefill_greedy": "Audio8_ASR_Prefill_Greedy.onnx",
    "prefill_penalty_greedy": "Audio8_ASR_Prefill_Penalty_Greedy.onnx",
    "prefill_sampling": "Audio8_ASR_PrefillSampling.onnx",
    "decode_greedy": "Audio8_ASR_Decode_Greedy.onnx",
    "decode_penalty_greedy": "Audio8_ASR_Decode_Penalty_Greedy.onnx",
    "decode_sampling": "Audio8_ASR_DecodeSampling.onnx",
    "shared_initializers": "Audio8_ASR_SharedInitializers.onnx",
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
onnx_model_Concat_Embed        = str(split_export_folder / MODEL_FILE_NAMES["concat_embed"])


# ============================== USER CONFIG ==============================
MAX_INPUT_AUDIO_LENGTH         = 480000                        # Maximum deployment audio length (30 s at the model's fixed 16 kHz sample rate).
MAX_SEQ_LEN                    = 1024                          # Maximum context length, including prompt + audio + decode tokens.
USE_FP16_KV                    = True                          # Use FP16 KV cache for normal deployment exports.
COMPUTE_IN_F32                 = False                         # F16-KV compute precision. False = minimum-cast f16 attention (Q@K/mask/softmax/attn@V all run in f16 on the f16 KV cache; storage AND compute f16). True = keep the f16 KV *storage* (cache I/O dtype unchanged) but upcast K/V to f32 at the matmul use points and keep Q/mask/softmax in f32 (f16 storage, f32 compute). No effect when USE_FP16_KV=False.
INPUT_AUDIO_DTYPE              = "F32"                         # Model audio input dtype: "INT16", "F32", or "F16". "INT16" feeds raw PCM (÷32768 inside the graph). "F32"/"F16" feed audio already normalised to [-1, 1] (the in-graph ÷32768 is skipped); "F16" is cast up to f32 for compute.
TRANSCRIBE_PROMPT              = "Please transcribe this audio." # Official example user text appended after the audio span.


# Weight-quantization-friendly reorder (exact and absorbed into the weights).
REORDER_DOWNPROJ_FOR_QUANT     = True                          # Reorder MLP intermediate channels so down_proj block-quant groups are magnitude-homogeneous.
REORDER_OPROJ_FOR_QUANT        = True                          # Reorder each head's head_dim so o_proj sub-head groups are homogeneous. Pure win for f16 KV.
REORDER_KEY                    = "absmean"                     # "absmean" (best at group=32) | "L4" (best at group=128) | "rms" | "std".

OPSET                          = 20                            # ONNX Runtime opset version.
# ========================================================================

# Fixed Audio8-ASR model constants and metadata defaults; these are not user tunables.
_MODEL_SAMPLE_RATE             = 16000
_MODEL_WINDOW_TYPE             = "hann"
_MODEL_NUM_MELS                = 128
_MODEL_NFFT_STFT               = 400
_MODEL_WINDOW_LENGTH           = 400
_MODEL_HOP_LENGTH              = 160
_MODEL_AUDIO_PCM_SCALE         = 32768
ROTARY_STORAGE_DTYPE           = torch.float16 if USE_FP16_KV else torch.float32
ATTENTION_MASK_DTYPE           = (
    torch.float32 if USE_FP16_KV and COMPUTE_IN_F32 else ROTARY_STORAGE_DTYPE
)


# Official Audio8 languages: Chinese, English, French, German, Japanese,
# Korean, and Cantonese. Audio8 detects the spoken language automatically;
# unlike language-conditioned ASR models, it has no language prompt tokens.
SUPPORTED_LANGUAGES = {
    "zh": {
        "name": "Chinese",
        "aliases": ["chinese", "mandarin", "cn"],
        "prompt_token_ids": [],
    },
    "en": {
        "name": "English",
        "aliases": ["english", "eng"],
        "prompt_token_ids": [],
    },
    "fr": {
        "name": "French",
        "aliases": ["french", "fra", "fre"],
        "prompt_token_ids": [],
    },
    "de": {
        "name": "German",
        "aliases": ["german", "deu", "ger"],
        "prompt_token_ids": [],
    },
    "ja": {
        "name": "Japanese",
        "aliases": ["japanese", "jpn"],
        "prompt_token_ids": [],
    },
    "ko": {
        "name": "Korean",
        "aliases": ["korean", "kor"],
        "prompt_token_ids": [],
    },
    "yue": {
        "name": "Cantonese",
        "aliases": ["cantonese"],
        "prompt_token_ids": [],
    },
}

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


def commit_staged_export(staged_folder: Path, final_folder: Path) -> None:
    """Replace the final bundle only after staged validation has succeeded."""
    staged_folder = Path(staged_folder)
    final_folder = Path(final_folder)
    final_parent = final_folder.parent
    final_parent.mkdir(parents=True, exist_ok=True)
    if not staged_folder.is_dir():
        raise RuntimeError(f"Missing staged Audio8 export folder: {staged_folder}")
    if os.stat(staged_folder).st_dev != os.stat(final_parent).st_dev:
        raise RuntimeError(
            "Audio8 staged export and final destination must share a filesystem "
            "for an atomic directory commit."
        )
    if final_folder.exists() and not final_folder.is_dir():
        raise RuntimeError(f"Audio8 final export path is not a directory: {final_folder}")

    backup_root: Path | None = None
    backup_folder: Path | None = None
    preserve_backup = False
    try:
        if final_folder.exists() or final_folder.is_symlink():
            backup_root = Path(
                tempfile.mkdtemp(
                    prefix=f".{final_folder.name}_previous_", dir=final_parent
                )
            )
            backup_folder = backup_root / final_folder.name
            os.replace(final_folder, backup_folder)
        os.replace(staged_folder, final_folder)
    except BaseException:
        if (
            backup_folder is not None
            and (backup_folder.exists() or backup_folder.is_symlink())
            and not (final_folder.exists() or final_folder.is_symlink())
        ):
            try:
                os.replace(backup_folder, final_folder)
            except OSError:
                preserve_backup = True
        raise
    finally:
        if backup_root is not None and not preserve_backup:
            shutil.rmtree(backup_root, ignore_errors=True)


def _load_local_audio8_remote_code(checkpoint: Path) -> dict[str, types.ModuleType]:
    """Load the official local modules with a valid relative-import package name."""
    package_name = "_audio8_asr_local_remote_code"
    package = types.ModuleType(package_name)
    package.__path__ = [str(checkpoint)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    modules: dict[str, types.ModuleType] = {}
    for stem in (
        "qwen3_asr_audio_config",
        "qwen3_asr_audio_model",
        "configuration_arkasr",
        "modeling_arkasr",
        "processing_arkasr",
    ):
        module_name = f"{package_name}.{stem}"
        spec = importlib.util.spec_from_file_location(
            module_name, checkpoint / f"{stem}.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load official Audio8 module {stem!r}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        modules[stem] = module
    return modules


def restore_tied_lm_head(model: nn.Module, config) -> None:
    """Restore Audio8's checkpoint-declared embedding/LM-head alias."""
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return
    try:
        embedding = model.language_model.model.embed_tokens.weight
        lm_head = model.language_model.lm_head
    except AttributeError as exc:
        raise RuntimeError(
            "Audio8 checkpoint declares tied embeddings but lacks the Qwen LM-head path."
        ) from exc
    if lm_head.weight.shape != embedding.shape:
        raise RuntimeError(
            "Audio8 tied embedding and LM-head weights have incompatible shapes: "
            f"{tuple(embedding.shape)} versus {tuple(lm_head.weight.shape)}."
        )
    if lm_head.weight.dtype != embedding.dtype or lm_head.weight.device != embedding.device:
        raise RuntimeError("Audio8 tied embedding and LM-head weights differ in dtype or device.")
    lm_head.weight = embedding
    if lm_head.weight.data_ptr() != embedding.data_ptr():
        raise RuntimeError("Audio8 failed to restore the tied embedding/LM-head storage.")


def load_audio8_checkpoint(checkpoint: Path):
    """Use the official Auto* contract, with a narrow local-loader fallback.

    The installed Transformers dynamic-module loader derives an invalid Python
    package from this checkpoint's hyphenated directory name. The fallback only
    supplies a valid package shell for the unchanged official local modules.
    """
    try:
        config = AutoConfig.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("transformers_modules.Audio8-ASR-0"):
            raise
        modules = _load_local_audio8_remote_code(checkpoint)
        config_class = modules["configuration_arkasr"].ArkasrConfig
        model_class = modules["modeling_arkasr"].ArkasrForConditionalGeneration
        processor_class = modules["processing_arkasr"].ArkasrProcessor
        config = config_class.from_pretrained(checkpoint, local_files_only=True)
        model = model_class.from_pretrained(
            checkpoint,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        processor = processor_class.from_pretrained(
            checkpoint, local_files_only=True
        )
    model = model.cpu().float().eval()
    restore_tied_lm_head(model, config)
    return config, model, processor


def _get_feat_extract_output_lengths(input_lengths: Tensor) -> Tensor:
    input_lengths_leave = input_lengths % 100
    feature_lengths = torch.clamp(input_lengths_leave - 1, min=0) // 2 + 1
    feature_lengths = feature_lengths * (input_lengths_leave > 0).to(feature_lengths.dtype)
    feature_lengths_2 = torch.clamp(feature_lengths - 1, min=0) // 2 + 1
    feature_lengths_2 = feature_lengths_2 * (feature_lengths > 0).to(feature_lengths_2.dtype)
    feature_lengths_3 = torch.clamp(feature_lengths_2 - 1, min=0) // 2 + 1
    feature_lengths_3 = feature_lengths_3 * (feature_lengths_2 > 0).to(feature_lengths_3.dtype)
    return feature_lengths_3 + (input_lengths // 100) * 13


def absorb_layer_norm_affine(norm: nn.LayerNorm, linear: nn.Linear) -> None:
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = nn.Parameter(
                torch.zeros(linear.out_features, dtype=linear.weight.dtype)
            )
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias = None


def get_kv_io(
    tensors_dict: Dict[str, Tensor],
    kv_specs: Sequence[Tuple[str, int]],
    num_layers: int,
    batch_axis: str = "batch",
    seq_axis: str = "history_len",
    out_seq_axis: str = "kv_seq_len",
) -> Tuple[List[Tensor], List[str], List[str], Dict[str, Dict[int, str]]]:
    inputs: List[Tensor] = []
    input_names: List[str] = []
    output_names: List[str] = []
    dynamic_axes: Dict[str, Dict[int, str]] = {}
    for name, dim in kv_specs:
        tensor = tensors_dict[name]
        for index in range(num_layers):
            input_name = f"past_{name}_{index}"
            output_name = f"present_{name}_{index}"
            inputs.append(tensor)
            input_names.append(input_name)
            output_names.append(output_name)
            dynamic_axes[input_name] = {0: batch_axis, dim: seq_axis}
            dynamic_axes[output_name] = {0: batch_axis, dim: out_seq_axis}
    return inputs, input_names, output_names, dynamic_axes


class POSITIVE_CEIL_DIV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, divisor: int) -> Tensor:
        return (value + divisor - 1) // divisor

    @staticmethod
    def symbolic(g, value, divisor):
        divisor_value = symbolic_helper._get_const(divisor, "i", "divisor")
        offset = g.op(
            "Constant", value_t=torch.tensor([divisor_value - 1], dtype=torch.int64)
        )
        denominator = g.op(
            "Constant", value_t=torch.tensor([divisor_value], dtype=torch.int64)
        )
        return g.op("Div", g.op("Add", value, offset), denominator)


class ONNX_SHAPE_DIM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, axis: int) -> Tensor:
        return torch._shape_as_tensor(x)[axis:axis + 1]

    @staticmethod
    def symbolic(g, x, axis):
        axis_value = symbolic_helper._get_const(axis, "i", "axis")
        return g.op("Shape", x, start_i=axis_value, end_i=axis_value + 1)


class ONNX_STATIC_RESHAPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        eager_shape = tuple(
            x.shape[index] if dim == 0 else dim
            for index, dim in enumerate(shape)
        )
        return x.reshape(eager_shape)

    @staticmethod
    def symbolic(g, x, shape):
        target = g.op("Constant", value_t=torch.tensor(shape, dtype=torch.int64))
        return g.op("Reshape", x, target)


def onnx_reshape_batch(x: Tensor, shape: Tuple[int, ...]) -> Tensor:
    return ONNX_STATIC_RESHAPE.apply(x, (0,) + tuple(shape))


class ONNX_RANGE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, limit: Tensor) -> Tensor:
        return torch.arange(int(limit.item()), dtype=torch.int64, device=limit.device)

    @staticmethod
    def symbolic(g, limit):
        zero = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
        one = g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64))
        return g.op("Range", zero, limit, one)


class PENALIZE_LOGITS(torch.autograd.Function):
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


class AUDIO8_ASR_ENCODER(nn.Module):
    """Official Whisper log-mel frontend plus Audio8 encoder, bridge, and projector."""

    def __init__(
        self,
        audio_encoder: nn.Module,
        audio_mlp_tower: nn.Module,
        audio_projector: nn.Module,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        num_mels: int,
        mel_filters: np.ndarray,
        audio_token_id: int,
    ) -> None:
        super().__init__()
        self.audio_tower = audio_encoder.float()
        self.audio_mlp_tower = audio_mlp_tower.float()
        self.audio_projector = audio_projector.float()
        self.input_audio_is_int16 = INPUT_AUDIO_DTYPE == "INT16"
        self.input_audio_requires_f32_cast = INPUT_AUDIO_DTYPE != "F32"
        self.n_fft = int(n_fft)
        self._fuse_encoder_weights()

        if int(self.audio_tower.config.num_mel_bins) != num_mels:
            raise RuntimeError(
                "Audio8 processor and audio encoder disagree on mel-bin count."
            )
        self.num_mels = int(num_mels)
        self.hop_length = int(hop_length)
        self.audio_token_id = int(audio_token_id)
        input_scale = (
            1.0 / _MODEL_AUDIO_PCM_SCALE if self.input_audio_is_int16 else 1.0
        )
        self.stft = STFT_Process(
            model_type="stft_B",
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_len=self.hop_length,
            max_frames=0,
            window_type=_MODEL_WINDOW_TYPE,
            center_pad=True,
            pad_mode="reflect",
            input_scale=input_scale,
            drop_last_frame=True,
        ).eval()
        with torch.no_grad():
            self.stft.stft_kernel.copy_(
                self._build_reference_stft_kernel(self.n_fft, input_scale)
            )

        mel_filters = torch.as_tensor(mel_filters, dtype=torch.float32)
        expected_mel_shape = ((self.n_fft // 2) + 1, self.num_mels)
        if tuple(mel_filters.shape) != expected_mel_shape:
            raise RuntimeError(
                "Audio8 processor mel filters have shape "
                f"{tuple(mel_filters.shape)}, expected {expected_mel_shape}."
            )
        self.register_buffer(
            "mel_filters",
            mel_filters.transpose(0, 1).unsqueeze(0).contiguous(),
            persistent=False,
        )

        self.chunk_size = int(self.audio_tower.n_window * 2)
        self.chunk_size_minus = self.chunk_size - 1
        self.chunk_aftercnn = int(
            _get_feat_extract_output_lengths(torch.tensor([self.chunk_size])).item()
        )
        self.chunks_per_window = int(
            self.audio_tower.n_window_infer // self.chunk_size
        )
        self.chunks_per_window_minus = self.chunks_per_window - 1
        self.tokens_per_window = self.chunks_per_window * self.chunk_aftercnn
        self.model_dim = int(self.audio_tower.config.d_model)
        self.output_dim = int(self.audio_tower.config.output_dim)
        self.conv_out_features = int(self.audio_tower.conv_out.in_features)
        self.attn_out_features = int(
            self.audio_tower.layers[0].self_attn.out_proj.in_features
        )
        self.num_heads = int(self.audio_tower.layers[0].self_attn.num_heads)
        self.head_dim = self.model_dim // self.num_heads

        max_mel_frames = MAX_INPUT_AUDIO_LENGTH // self.hop_length
        self.max_chunks = (max_mel_frames + self.chunk_size_minus) // self.chunk_size
        key_mask_lookup = torch.zeros(
            1,
            self.tokens_per_window + 1,
            1,
            1,
            self.tokens_per_window,
            dtype=torch.float32,
        )
        for count in range(self.tokens_per_window + 1):
            key_mask_lookup[0, count, 0, 0, count:] = -128.0
        self.register_buffer(
            "pos",
            self.audio_tower.positional_embedding.positional_embedding[
                : self.chunk_aftercnn
            ].unsqueeze(0).float(),
            persistent=False,
        )
        self.register_buffer(
            "chunk_starts_full",
            torch.arange(self.max_chunks + 1, dtype=torch.int64) * self.chunk_size,
            persistent=False,
        )
        self.register_buffer(
            "aftercnn_lens_lookup",
            _get_feat_extract_output_lengths(
                torch.arange(self.chunk_size + 1, dtype=torch.int64)
            ).long(),
            persistent=False,
        )
        self.register_buffer(
            "chunk_pad_zeros",
            torch.zeros(
                (
                    self.chunks_per_window_minus,
                    self.chunk_aftercnn,
                    self.model_dim,
                ),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "aftercnn_pad_zeros",
            torch.zeros(self.chunks_per_window_minus, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("key_mask_lookup", key_mask_lookup, persistent=False)
        self.register_buffer(
            "mel_pad_zeros",
            torch.zeros((1, self.num_mels, self.chunk_size), dtype=torch.float32),
            persistent=False,
        )

    @staticmethod
    def _build_reference_stft_kernel(n_fft: int, input_scale: float) -> Tensor:
        frequency_bins = (n_fft // 2) + 1
        window = np.hanning(n_fft + 1)[:-1]
        frequencies = np.arange(frequency_bins, dtype=np.float64)[:, None]
        samples = np.arange(n_fft, dtype=np.float64)[None, :]
        phase = (2.0 * np.pi / n_fft) * frequencies * samples
        kernel = np.concatenate(
            (
                np.cos(phase) * window,
                -np.sin(phase) * window,
            ),
            axis=0,
        )
        return torch.from_numpy(
            (kernel * float(input_scale)).astype(np.float32, copy=False)
        ).unsqueeze(1)

    def _fuse_encoder_weights(self) -> None:
        with torch.no_grad():
            for layer in self.audio_tower.layers:
                attn = layer.self_attn
                qkv = nn.Linear(
                    attn.q_proj.in_features,
                    attn.q_proj.out_features
                    + attn.k_proj.out_features
                    + attn.v_proj.out_features,
                    bias=True,
                )
                qkv.weight.copy_(
                    torch.cat(
                        [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight],
                        dim=0,
                    )
                )
                qkv.bias.copy_(
                    torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0)
                )
                absorb_layer_norm_affine(layer.self_attn_layer_norm, qkv)
                scale_sqrt = float(attn.scaling) ** 0.5
                q_size = attn.q_proj.out_features
                k_size = attn.k_proj.out_features
                qkv.weight.data[:q_size].mul_(scale_sqrt)
                qkv.weight.data[q_size : q_size + k_size].mul_(scale_sqrt)
                qkv.bias.data[:q_size].mul_(scale_sqrt)
                qkv.bias.data[q_size : q_size + k_size].mul_(scale_sqrt)
                absorb_layer_norm_affine(layer.final_layer_norm, layer.fc1)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj
            absorb_layer_norm_affine(self.audio_tower.ln_post, self.audio_tower.proj1)

    def forward(self, audio: Tensor, input_ids: Tensor) -> Tensor:
        if self.input_audio_requires_f32_cast:
            audio = audio.float()
        real, imag = self.stft(audio)
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_filters, power)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        input_features = mel * 0.25 + 1.0

        feature_len = ONNX_SHAPE_DIM.apply(input_features, 2)
        num_chunks = POSITIVE_CEIL_DIV.apply(feature_len, self.chunk_size)
        pad_frames = self.chunk_starts_full[num_chunks] - feature_len
        padded_features = torch.cat(
            [input_features, self.mel_pad_zeros[..., :pad_frames]], dim=-1
        )
        chunks = padded_features.reshape(
            1, self.num_mels, -1, self.chunk_size
        ).permute(2, 0, 1, 3)
        chunk_starts = self.chunk_starts_full[:num_chunks]
        raw_chunk_lens = torch.clamp(
            feature_len - chunk_starts, min=0, max=self.chunk_size
        )
        aftercnn_lens = self.aftercnn_lens_lookup[raw_chunk_lens]

        hidden_states = F.gelu(self.audio_tower.conv2d1(chunks))
        hidden_states = F.gelu(self.audio_tower.conv2d2(hidden_states))
        hidden_states = F.gelu(self.audio_tower.conv2d3(hidden_states))
        hidden_states = self.audio_tower.conv_out(
            hidden_states.permute(0, 3, 1, 2).contiguous().view(
                -1, self.chunk_aftercnn, self.conv_out_features
            )
        )
        hidden_states = hidden_states + self.pos

        num_windows = POSITIVE_CEIL_DIV.apply(num_chunks, self.chunks_per_window)
        total_chunks_padded = num_windows * self.chunks_per_window
        pad_chunks = total_chunks_padded - num_chunks
        hidden_states = torch.cat(
            [hidden_states, self.chunk_pad_zeros[:pad_chunks]], dim=0
        )
        aftercnn_lens = torch.cat(
            [aftercnn_lens, self.aftercnn_pad_zeros[:pad_chunks]]
        )
        hidden_states = hidden_states.reshape(-1, self.tokens_per_window, self.model_dim)
        valid_counts = aftercnn_lens.reshape(-1, self.chunks_per_window).sum(dim=1)
        key_mask = self.key_mask_lookup[:, valid_counts]
        for layer in self.audio_tower.layers:
            residual = hidden_states
            normed = layer.self_attn_layer_norm(hidden_states)
            qkv = layer.self_attn.qkv(normed).reshape(
                -1,
                self.tokens_per_window,
                3,
                self.num_heads,
                self.head_dim,
            ).permute(2, 0, 3, 1, 4)
            query, key, value = qkv.split(1, dim=0)
            attention = torch.matmul(query, key.transpose(-1, -2))
            attention = torch.softmax(attention + key_mask, dim=-1)
            attention = torch.matmul(attention, value).transpose(2, 3).reshape(
                -1, self.tokens_per_window, self.attn_out_features
            )
            hidden_states = residual + layer.self_attn.out_proj(attention)
            residual = hidden_states
            normed = layer.final_layer_norm(hidden_states)
            hidden_states = residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))

        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj2(
            self.audio_tower.act(self.audio_tower.proj1(hidden_states))
        )
        hidden_states = hidden_states.reshape(1, -1, self.output_dim)
        encoded_len = aftercnn_lens.sum(dim=0, keepdim=True)
        hidden_states = hidden_states[:, :encoded_len]
        hidden_states = self.audio_mlp_tower(hidden_states)
        audio_mask = input_ids[0].eq(self.audio_token_id)
        audio_token_count = audio_mask.to(torch.int64).sum()
        audio_time = ONNX_SHAPE_DIM.apply(hidden_states, 1)
        output_positions = ONNX_RANGE.apply(audio_token_count)
        input_positions = ONNX_RANGE.apply(audio_time)
        starts = torch.div(
            output_positions * audio_time, audio_token_count, rounding_mode="floor"
        )
        ends = torch.div(
            (output_positions + 1) * audio_time + audio_token_count - 1,
            audio_token_count,
            rounding_mode="floor",
        )
        pool_weights = (
            (input_positions.unsqueeze(0) >= starts.unsqueeze(1))
            & (input_positions.unsqueeze(0) < ends.unsqueeze(1))
        ).to(hidden_states.dtype)
        pooled = torch.matmul(pool_weights, hidden_states.squeeze(0))
        pooled = pooled / pool_weights.sum(dim=1, keepdim=True)
        return self.audio_projector(pooled).unsqueeze(0)


class AUDIO8_ASR_ROTARY_MASK_PREFILL(nn.Module):
    def __init__(self, llm: nn.Module, max_seq_len: int) -> None:
        super().__init__()
        self.mask_dtype = ATTENTION_MASK_DTYPE
        mask_row_pos = torch.arange(max_seq_len, dtype=torch.int32).view(
            1, 1, 1, max_seq_len, 1
        )
        mask_col_pos = torch.arange(max_seq_len, dtype=torch.int32).view(
            1, 1, 1, 1, max_seq_len
        )
        # A non-trainable parameter exports as an initializer, allowing the
        # merger to store this shared prefill table only once across strategies.
        self.causal_mask = nn.Parameter(
            torch.where(
                mask_col_pos <= mask_row_pos,
                torch.tensor(0.0, dtype=self.mask_dtype),
                torch.tensor(-128.0, dtype=self.mask_dtype),
            ).to(torch.int8),
            requires_grad=False,
        )
        cosine, sine = self._build_rotary_table(llm, max_seq_len)
        self.rotary_dim = int(cosine.shape[-1] * 2)
        self.register_buffer(
            "rotary_pos_emb",
            torch.cat(
                [torch.cat([cosine, cosine], dim=-1), torch.cat([-sine, sine], dim=-1)],
                dim=-1,
            ).to(ROTARY_STORAGE_DTYPE),
            persistent=False,
        )

    @staticmethod
    def _build_rotary_table(llm: nn.Module, max_seq_len: int) -> Tuple[Tensor, Tensor]:
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.rotary_emb.inv_freq.float()
        theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(theta), torch.sin(theta)

    def forward(self, ids_len: Tensor, history_len: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        kv_seq_len = ids_len + history_len
        rotary = self.rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_cos, rotary_sin = torch.split(
            rotary, [self.rotary_dim, self.rotary_dim], dim=-1
        )
        attention_mask = self.causal_mask[..., :ids_len, :kv_seq_len].to(self.mask_dtype)
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class AUDIO8_ASR_ROTARY_MASK_DECODE(nn.Module):
    def __init__(self, llm: nn.Module, max_seq_len: int) -> None:
        super().__init__()
        cosine, sine = AUDIO8_ASR_ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.rotary_dim = int(cosine.shape[-1] * 2)
        self.register_buffer(
            "rotary_pos_emb",
            torch.cat(
                [torch.cat([cosine, cosine], dim=-1), torch.cat([-sine, sine], dim=-1)],
                dim=-1,
            ).to(ROTARY_STORAGE_DTYPE),
            persistent=False,
        )

    def forward(self, kv_seq_len: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        kv_seq_len_next = kv_seq_len + 1
        rotary = self.rotary_pos_emb[:, kv_seq_len].float()
        rotary_cos, rotary_sin = torch.split(
            rotary, [self.rotary_dim, self.rotary_dim], dim=-1
        )
        return rotary_cos, rotary_sin, kv_seq_len_next


class AUDIO8_ASR_DECODER_EMBED(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = model.language_model.model.embed_tokens.float()

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)


class AUDIO8_ASR_CONCAT_EMBED(nn.Module):
    """Inject Encoder-projected Audio8 states into official audio slots only."""

    def __init__(self, audio_token_id: int) -> None:
        super().__init__()
        self.register_buffer(
            "audio_token_id",
            torch.tensor(int(audio_token_id), dtype=torch.int32),
            persistent=False,
        )

    def forward(
        self, base_embed: Tensor, audio_hidden: Tensor, input_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        audio_mask = input_ids[0].eq(self.audio_token_id)
        audio_order = torch.clamp(
            audio_mask.to(torch.int32).cumsum(dim=0) - 1, min=0
        )
        audio_by_position = torch.index_select(
            audio_hidden.squeeze(0), 0, audio_order
        ).unsqueeze(0)
        concat_embed = torch.where(
            audio_mask.view(1, -1, 1), audio_by_position, base_embed
        )
        return concat_embed, ONNX_SHAPE_DIM.apply(concat_embed, 1)


class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
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


def simplified_layer_norm(x: Tensor, scale: Tensor, epsilon: float, axis: int = -1) -> Tensor:
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


class AUDIO8_ASR_DECODER_MAIN(nn.Module):
    """Qwen2 decoder donor with Qwen-ASR's cache and graph ABI."""

    def __init__(
        self,
        model: nn.Module,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_layers: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.llm = model.language_model.model.float()
        self.lm_head = model.language_model.lm_head.float()
        first_layer = self.llm.layers[0]
        if hasattr(first_layer.self_attn, "q_norm") or hasattr(first_layer.self_attn, "k_norm"):
            raise RuntimeError("Audio8 Qwen2 decoder unexpectedly exposes Qwen3 q/k norms.")

        self.head_dim = int(head_dim)
        self.head_dim_half = self.head_dim // 2
        self.num_heads = int(num_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.qk_heads = self.num_heads + self.num_kv_heads
        self.num_layers = int(num_layers)
        self.use_fp16_kv = USE_FP16_KV
        self.compute_in_f32 = COMPUTE_IN_F32
        self.total_qkv_heads = self.qk_heads + self.num_kv_heads
        self.qkv_split_sizes = (
            self.num_heads,
            self.num_kv_heads,
            self.num_kv_heads,
        )
        self.attention_output_size = int(first_layer.self_attn.o_proj.in_features)
        self.intermediate_size = int(first_layer.mlp.down_proj.in_features)
        self.mlp_split_sizes = (self.intermediate_size, self.intermediate_size)
        hidden_norm = first_layer.input_layernorm
        self.hidden_rms_norm_eps = float(
            getattr(hidden_norm, "variance_epsilon", getattr(hidden_norm, "eps", 1e-6))
        )
        self.register_buffer(
            "hidden_norm_scale",
            torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32),
            persistent=False,
        )
        self.save_key = [None] * self.num_layers
        self.save_value = [None] * self.num_layers
        self._fuse_weights(hidden_size)
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _fuse_weights(self, hidden_size: int) -> None:
        norm_factor = hidden_size ** 0.5
        with torch.no_grad():
            for layer in self.llm.layers:
                attn = layer.self_attn
                projections = (attn.q_proj, attn.k_proj, attn.v_proj)
                has_qkv_bias = any(projection.bias is not None for projection in projections)
                qkv = nn.Linear(
                    attn.q_proj.in_features,
                    attn.q_proj.out_features
                    + attn.k_proj.out_features
                    + attn.v_proj.out_features,
                    bias=has_qkv_bias,
                )
                qkv.weight.copy_(torch.cat([projection.weight for projection in projections], dim=0))
                if has_qkv_bias:
                    qkv.bias.copy_(
                        torch.cat(
                            [
                                projection.bias
                                if projection.bias is not None
                                else torch.zeros(
                                    projection.out_features,
                                    dtype=qkv.weight.dtype,
                                    device=qkv.weight.device,
                                )
                                for projection in projections
                            ],
                            dim=0,
                        )
                    )
                qkv.weight.mul_(
                    layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                )
                q_size = attn.q_proj.out_features
                qkv.weight[:q_size].mul_(attn.scaling)
                if qkv.bias is not None:
                    qkv.bias[:q_size].mul_(attn.scaling)
                attn.qkv = qkv

                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                mlp_bias = gate.bias is not None or up.bias is not None
                gate_up = nn.Linear(
                    gate.in_features,
                    gate.out_features + up.out_features,
                    bias=mlp_bias,
                )
                post_norm_weight = (
                    layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                )
                gate_up.weight.copy_(
                    torch.cat(
                        [gate.weight * post_norm_weight, up.weight * post_norm_weight],
                        dim=0,
                    )
                )
                if mlp_bias:
                    gate_up.bias.copy_(
                        torch.cat(
                            [
                                gate.bias
                                if gate.bias is not None
                                else torch.zeros(
                                    gate.out_features,
                                    dtype=gate_up.weight.dtype,
                                    device=gate_up.weight.device,
                                ),
                                up.bias
                                if up.bias is not None
                                else torch.zeros(
                                    up.out_features,
                                    dtype=gate_up.weight.dtype,
                                    device=gate_up.weight.device,
                                ),
                            ],
                            dim=0,
                        )
                    )
                layer.mlp.gate_up_proj = gate_up

                del attn.q_proj, attn.k_proj, attn.v_proj
                del layer.input_layernorm, layer.post_attention_layernorm
                del layer.mlp.gate_proj, layer.mlp.up_proj

            self.register_buffer(
                "final_norm_scale",
                self.llm.norm.weight.detach().clone().float(),
                persistent=False,
            )
            del self.llm.norm

    @staticmethod
    def _channel_statistic(weights: Tensor, key: str, dims) -> Tensor:
        absolute = weights.abs()
        if key == "rms":
            return (weights * weights).mean(dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dims).pow(0.25)
        if key == "std":
            return weights.reshape(-1, weights.shape[-1]).std(0)
        return absolute.mean(dims)

    def _reorder_downproj_for_quant(self, key: str) -> None:
        with torch.no_grad():
            for layer in self.llm.layers:
                down_proj = layer.mlp.down_proj
                permutation = torch.argsort(
                    self._channel_statistic(down_proj.weight, key, 0)
                )
                intermediate = down_proj.in_features
                gate_up = layer.mlp.gate_up_proj
                weights = gate_up.weight
                gate_up.weight.copy_(
                    torch.cat(
                        [weights[:intermediate][permutation], weights[intermediate:][permutation]],
                        dim=0,
                    )
                )
                if gate_up.bias is not None:
                    bias = gate_up.bias
                    gate_up.bias.copy_(
                        torch.cat(
                            [bias[:intermediate][permutation], bias[intermediate:][permutation]],
                            dim=0,
                        )
                    )
                down_proj.weight.copy_(down_proj.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key: str) -> None:
        heads = self.num_heads
        kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        groups = heads // kv_heads
        with torch.no_grad():
            for layer in self.llm.layers:
                output_weight = layer.self_attn.o_proj.weight
                output_by_head = output_weight.view(
                    output_weight.shape[0], heads, head_dim
                )
                permutations = []
                for kv_head in range(kv_heads):
                    columns = output_by_head[:, kv_head * groups : (kv_head + 1) * groups, :]
                    permutations.append(
                        torch.argsort(self._channel_statistic(columns, key, (0, 1)))
                    )
                reordered_output = output_by_head.clone()
                for head in range(heads):
                    reordered_output[:, head, :] = reordered_output[
                        :, head, permutations[head // groups]
                    ]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv = layer.self_attn.qkv
                qkv_weight = qkv.weight.view(-1, head_dim, qkv.weight.shape[1]).clone()
                for kv_head in range(kv_heads):
                    value_head = self.qk_heads + kv_head
                    qkv_weight[value_head] = qkv_weight[value_head][permutations[kv_head]]
                qkv.weight.copy_(qkv_weight.reshape_as(qkv.weight))
                if qkv.bias is not None:
                    qkv_bias = qkv.bias.view(-1, head_dim).clone()
                    for kv_head in range(kv_heads):
                        value_head = self.qk_heads + kv_head
                        qkv_bias[value_head] = qkv_bias[value_head][permutations[kv_head]]
                    qkv.bias.copy_(qkv_bias.reshape_as(qkv.bias))

    def _rms_norm(self, x: Tensor, scale: Tensor, epsilon: float) -> Tensor:
        return simplified_layer_norm(x, scale, epsilon)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x = onnx_reshape_batch(
            x, (-1, 1, self.qk_heads, 2, self.head_dim_half)
        )
        x = x.flip(-2)
        return onnx_reshape_batch(x, (-1, 1, self.qk_heads, self.head_dim))

    def forward(self, *all_inputs: Tensor) -> Tuple[Tensor, ...]:
        hidden_states = all_inputs[-4]
        rotary_cos = all_inputs[-3]
        rotary_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]
        attn_mask = attention_mask
        for index, layer in enumerate(self.llm.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps
            )
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = onnx_reshape_batch(
                qkv, (-1, 1, self.total_qkv_heads, self.head_dim)
            )
            query_key, value = torch.split(
                qkv, (self.qk_heads, self.num_kv_heads), dim=-2
            )
            query_key = (
                query_key * rotary_cos
                + self._rotate_half(query_key) * rotary_sin
            )
            if self.use_fp16_kv and not self.compute_in_f32:
                query_key = query_key.half()
                value = value.half()
            qkv = torch.cat((query_key, value), dim=-2)
            qkv = qkv.permute(0, 3, 2, 1, 4)
            query, key, value = torch.split(qkv, self.qkv_split_sizes, dim=1)
            if self.use_fp16_kv:
                if self.compute_in_f32:
                    key = key.half()
                value = value.half()
            query = onnx_reshape_batch(
                query,
                (
                    self.num_kv_heads,
                    self.num_kv_groups,
                    -1,
                    self.head_dim,
                ),
            )
            key = key.transpose(-1, -2)
            key = torch.cat((all_inputs[index], key), dim=-1)
            value = torch.cat((all_inputs[index + self.num_layers], value), dim=-2)
            self.save_key[index] = key
            self.save_value[index] = value
            if self.use_fp16_kv and self.compute_in_f32:
                attention = torch.matmul(query, key.float()) + attn_mask
                attention = torch.softmax(attention, dim=-1)
                attention = torch.matmul(attention, value.float())
            else:
                attention = torch.matmul(query, key) + attn_mask
                attention = torch.softmax(attention, dim=-1)
                attention = torch.matmul(attention, value)
            attention = onnx_reshape_batch(
                attention.permute(0, 3, 1, 2, 4),
                (-1, self.attention_output_size),
            )
            if self.use_fp16_kv and not self.compute_in_f32:
                attention = attention.float()
            hidden_states = residual + layer.self_attn.o_proj(attention)
            residual = hidden_states
            hidden_states = self._rms_norm(
                hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps
            )
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, self.mlp_split_sizes, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        hidden_states = self._rms_norm(
            hidden_states[:, -1], self.final_norm_scale, self.hidden_rms_norm_eps
        )
        logits = self.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits


class GREEDY_SEARCH(nn.Module):
    def forward(self, logits: Tensor, save_id: Tensor) -> Tuple[Tensor, Tensor]:
        max_idx = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)
        return max_idx, torch.cat([save_id, max_idx], dim=-1)


class TOPK_TOPP_SAMPLING(nn.Module):
    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False)
        self.register_buffer("gumbel_min", torch.tensor(self.GUMBEL_EPS, dtype=torch.float32), persistent=False)
        self.register_buffer("gumbel_max", torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32), persistent=False)

    def forward(
        self,
        logits: Tensor,
        temperature: Tensor,
        top_k: Tensor,
        top_p: Tensor,
        repetition_penalty: Tensor,
        previous_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        inverse_penalty = torch.reciprocal(repetition_penalty)
        previous_logits = torch.gather(logits, 1, previous_ids)
        previous_scores = torch.where(
            previous_logits < 0.0,
            previous_logits * repetition_penalty,
            previous_logits * inverse_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids, previous_scores)
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
        winner = torch.argmax(sorted_scores - torch.log(-torch.log(noise)), dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        return sampled_id, torch.cat([previous_ids, sampled_id], dim=-1)


class APPLY_PENALTY(nn.Module):
    def forward(
        self,
        logits: Tensor,
        save_id: Tensor,
        penalty_value: Tensor,
        penalty_range: Tensor,
    ) -> Tensor:
        target_indices = save_id[:, -penalty_range:]
        return PENALIZE_LOGITS.apply(logits, target_indices, penalty_value)


class ARGMAX(nn.Module):
    def forward(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)


class METADATA_CARRIER(nn.Module):
    def forward(self, marker: Tensor) -> Tensor:
        return marker


def audio_token_count(sample_count: int, hop_length: int, merge_factor: int) -> int:
    mel_frames = int(sample_count) // int(hop_length)
    downsampled = (mel_frames + 1) // 2
    return max(downsampled // max(int(merge_factor), 1), 1)


def build_audio8_prompt_ids(
    special_token_ids: Dict[str, int | List[int]],
    prompt_token_ids: Sequence[int],
    audio_slots: int,
) -> List[int]:
    return [
        int(special_token_ids["user"]),
        int(special_token_ids["audio_begin"]),
        *([int(special_token_ids["audio"])] * int(audio_slots)),
        int(special_token_ids["audio_end"]),
        *[int(token_id) for token_id in prompt_token_ids],
        int(special_token_ids["assistant"]),
    ]


onnx_folder.mkdir(parents=True)

print("\nExport start ...\n")

with torch.inference_mode():
    checkpoint = Path(download_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Official Audio8 checkpoint is not a directory: {checkpoint}")
    config, model, processor = load_audio8_checkpoint(checkpoint)
    tokenizer = processor.tokenizer
    feature_extractor = processor.feature_extractor
    language_model = model.language_model
    llm = language_model.model
    first_layer = llm.layers[0]
    audio_cfg = model.audio_encoder.config

    sample_rate = int(feature_extractor.sampling_rate)
    n_fft = int(feature_extractor.n_fft)
    hop_length = int(feature_extractor.hop_length)
    processor_mels = int(feature_extractor.feature_size)
    processor_max_samples = int(feature_extractor.n_samples)
    if (
        sample_rate,
        n_fft,
        hop_length,
        processor_mels,
        processor_max_samples,
    ) != (16000, 400, 160, 128, 480000):
        raise RuntimeError(
            "Audio8 processor no longer matches the verified 30-second Whisper frontend contract."
        )
    if MAX_INPUT_AUDIO_LENGTH != processor_max_samples:
        raise RuntimeError(
            "MAX_INPUT_AUDIO_LENGTH must equal the official Audio8 processor sample limit."
        )
    if int(audio_cfg.num_mel_bins) != processor_mels:
        raise RuntimeError("Audio8 audio encoder and processor mel dimensions differ.")
    if int(model.audio_mlp_tower.hidden_size) != int(audio_cfg.output_dim):
        raise RuntimeError("Audio8 MLP tower input does not match audio encoder output_dim.")
    if int(model.audio_projector[1].in_features) != int(audio_cfg.output_dim):
        raise RuntimeError("Audio8 projector input does not match audio encoder output_dim.")
    if int(model.audio_projector[1].out_features) != int(config.hidden_size):
        raise RuntimeError("Audio8 projector output does not match the Qwen2 hidden size.")
    if hasattr(first_layer.self_attn, "q_norm") or hasattr(first_layer.self_attn, "k_norm"):
        raise RuntimeError("Audio8 uses an unexpected Qwen decoder with q/k norms.")

    num_layers = len(llm.layers)
    num_heads = int(config.num_attention_heads)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(first_layer.self_attn.head_dim)
    hidden_size = int(config.hidden_size)
    vocab_size = int(config.vocab_size)
    if num_heads * head_dim != hidden_size:
        raise RuntimeError("Audio8 Qwen2 attention dimensions are inconsistent.")

    def token_id(token: str) -> int:
        value = int(tokenizer.convert_tokens_to_ids(token))
        if value < 0:
            raise RuntimeError(f"Audio8 tokenizer has no token {token!r}.")
        return value

    special_token_ids: Dict[str, int | List[int]] = {
        "stop": [token_id(processor.assistant_end_token), token_id("<|endoftext|>")],
        "audio": token_id(processor.audio_token),
        "user": token_id(processor.user_token),
        "audio_begin": token_id(processor.bos_audio_token),
        "audio_end": token_id(processor.eos_audio_token),
        "assistant": token_id(processor.assistant_token),
    }
    if int(special_token_ids["audio"]) != int(model.audio_token_id):
        raise RuntimeError("Audio8 processor audio token does not match model.audio_token_id.")
    merge_factor = int(processor.merge_factor)
    prompt_token_ids = [
        int(token_id) for token_id in tokenizer.encode(TRANSCRIBE_PROMPT, add_special_tokens=False)
    ]

    # Let the official processor prove the runtime prompt construction that the
    # ONNX frontend later recreates from metadata and actual post-resample length.
    probe_samples = hop_length * 100 + 73
    probe = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "array": np.zeros(probe_samples, dtype=np.float32)},
                    {"type": "text", "text": TRANSCRIBE_PROMPT},
                ],
            }
        ],
        return_tensors="pt",
        sampling_rate=sample_rate,
        audio_padding="longest",
        add_generation_prompt=True,
        audio_max_length=MAX_INPUT_AUDIO_LENGTH,
        text_kwargs={"padding": "longest", "truncation": True, "max_length": 1000},
    )
    expected_probe_ids = build_audio8_prompt_ids(
        special_token_ids,
        prompt_token_ids,
        audio_token_count(probe_samples, hop_length, merge_factor),
    )
    if probe["input_ids"][0].tolist() != expected_probe_ids:
        raise RuntimeError("Audio8 processor prompt contract differs from the verified export frontend.")
    del probe

    kv_dtype = torch.float16 if USE_FP16_KV else torch.float32
    kv_specs = [("key", 4), ("value", 3)]
    print(
        "  Encoder : "
        f"layers={audio_cfg.encoder_layers}, d_model={audio_cfg.d_model}, "
        f"output_dim={audio_cfg.output_dim}, bridge={model.audio_mlp_tower.num_layers}x"
        f"{model.audio_mlp_tower.intermediate_size}->{model.audio_projector[1].out_features}"
    )
    print(
        f"  Decoder : layers={num_layers}, heads={num_heads}/{num_kv_heads} "
        f"Qwen2, head_dim={head_dim}"
    )
    print(f"  KV dtype: {'float16' if USE_FP16_KV else 'float32'}")

    dummy_seq_len = 16
    dummy_batch_size = 10
    dummy_history_len = 10
    dummy_penalty_value = 1.0
    dummy_penalty_range = dummy_history_len
    ids_len = torch.tensor([dummy_seq_len], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    kv_seq_len = ids_len + history_len
    logits = torch.ones((dummy_batch_size, vocab_size), dtype=torch.float32)
    save_id = torch.zeros((dummy_batch_size, 0), dtype=torch.int32)
    kv_tensors = {
        "key": torch.zeros(
            (dummy_batch_size, num_kv_heads, 1, head_dim, 0), dtype=kv_dtype
        ),
        "value": torch.zeros(
            (dummy_batch_size, num_kv_heads, 1, 0, head_dim), dtype=kv_dtype
        ),
    }
    dummy_audio_slots = audio_token_count(probe_samples, hop_length, merge_factor)
    dummy_prompt_ids = torch.tensor(
        [
            build_audio8_prompt_ids(
                special_token_ids, prompt_token_ids, dummy_audio_slots
            )
        ],
        dtype=torch.int32,
    )

    encoder = AUDIO8_ASR_ENCODER(
        model.audio_encoder,
        model.audio_mlp_tower,
        model.audio_projector,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        num_mels=processor_mels,
        mel_filters=feature_extractor.mel_filters,
        audio_token_id=int(special_token_ids["audio"]),
    ).eval()
    audio_export_dtype = {
        "INT16": torch.int16,
        "F32": torch.float32,
        "F16": torch.float16,
    }[INPUT_AUDIO_DTYPE]
    dummy_audio = torch.ones(
        (1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=audio_export_dtype
    )
    torch.onnx.export(
        encoder,
        (dummy_audio, dummy_prompt_ids),
        onnx_model_Encoder,
        input_names=["audio", "input_ids"],
        output_names=["audio_hidden"],
        dynamic_axes={
            "audio": {2: "audio_len"},
            "input_ids": {1: "prompt_len"},
            "audio_hidden": {1: "audio_encoded_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del encoder, dummy_audio
    gc.collect()

    embed_mod = AUDIO8_ASR_DECODER_EMBED(model).eval()
    dummy_ids = torch.ones((1, dummy_seq_len), dtype=torch.int32)
    torch.onnx.export(
        embed_mod,
        (dummy_ids,),
        onnx_model_Embed,
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "ids_len"},
            "hidden_states": {0: "batch", 1: "ids_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del embed_mod
    gc.collect()

    rotary_prefill = AUDIO8_ASR_ROTARY_MASK_PREFILL(llm, MAX_SEQ_LEN).eval()
    torch.onnx.export(
        rotary_prefill,
        (ids_len, history_len),
        onnx_model_Rotary_Mask_Prefill,
        input_names=["ids_len", "history_len"],
        output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        dynamic_axes={
            "rotary_cos": {1: "ids_len"},
            "rotary_sin": {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del rotary_prefill
    gc.collect()

    rotary_decode = AUDIO8_ASR_ROTARY_MASK_DECODE(llm, MAX_SEQ_LEN).eval()
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

    concat_embed = AUDIO8_ASR_CONCAT_EMBED(int(special_token_ids["audio"])).eval()
    dummy_base_embed = torch.ones(
        (1, dummy_prompt_ids.shape[1], hidden_size), dtype=torch.float32
    )
    dummy_audio_hidden = torch.ones(
        (1, dummy_audio_slots, hidden_size), dtype=torch.float32
    )
    torch.onnx.export(
        concat_embed,
        (dummy_base_embed, dummy_audio_hidden, dummy_prompt_ids),
        onnx_model_Concat_Embed,
        input_names=["base_embed", "audio_hidden", "input_ids"],
        output_names=["concat_embed", "concat_len"],
        dynamic_axes={
            "base_embed": {1: "prompt_len"},
            "audio_hidden": {1: "audio_encoded_len"},
            "input_ids": {1: "prompt_len"},
            "concat_embed": {1: "prompt_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del concat_embed, dummy_prompt_ids, dummy_base_embed, dummy_audio_hidden
    gc.collect()

    kv_inputs, kv_input_names, kv_output_names, kv_axes = get_kv_io(
        kv_tensors, kv_specs, num_layers
    )
    hidden_states = torch.ones(
        (dummy_batch_size, dummy_seq_len, hidden_size), dtype=torch.float32
    )
    rotary_cos = torch.ones(
        (1, dummy_seq_len, 1, 1, head_dim), dtype=torch.float32
    )
    rotary_sin = torch.zeros(
        (1, dummy_seq_len, 1, 1, head_dim), dtype=torch.float32
    )
    attention_mask = torch.zeros(
        (1, 1, 1, dummy_seq_len, dummy_seq_len), dtype=ATTENTION_MASK_DTYPE
    )
    all_inputs = kv_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]
    input_names = kv_input_names + [
        "hidden_states",
        "rotary_cos",
        "rotary_sin",
        "attention_mask",
    ]
    output_names = kv_output_names + ["logits"]
    dynamic_axes = {
        **kv_axes,
        "hidden_states": {0: "batch", 1: "ids_len"},
        "rotary_cos": {1: "ids_len"},
        "rotary_sin": {1: "ids_len"},
        "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        "logits": {0: "batch"},
    }
    decoder_main = AUDIO8_ASR_DECODER_MAIN(
        model,
        num_heads,
        num_kv_heads,
        head_dim,
        num_layers,
        hidden_size,
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

    torch.onnx.export(
        GREEDY_SEARCH().eval(),
        (logits[[0]], save_id[[0]]),
        onnx_model_Greedy,
        input_names=["logits", "save_id_in"],
        output_names=["max_logits_idx", "save_id_out"],
        dynamic_axes={
            "logits": {0: "batch"},
            "save_id_in": {0: "batch", 1: "history_len"},
            "max_logits_idx": {0: "batch"},
            "save_id_out": {0: "batch", 1: "history_len_out"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    dummy_save_id = torch.zeros(
        (dummy_batch_size, dummy_history_len), dtype=torch.int32
    )
    penalty_value = torch.tensor([dummy_penalty_value], dtype=torch.float32)
    penalty_range = torch.tensor([dummy_penalty_range], dtype=torch.int64)
    torch.onnx.export(
        APPLY_PENALTY().eval(),
        (logits, dummy_save_id, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=["logits_in", "save_id_in", "penalty_value", "penalty_range"],
        output_names=["logits_out"],
        dynamic_axes={
            "logits_in": {0: "batch"},
            "save_id_in": {0: "batch", 1: "history_len"},
            "logits_out": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    torch.onnx.export(
        ARGMAX().eval(),
        (logits,),
        onnx_model_Argmax,
        input_names=["logits"],
        output_names=["max_logits_idx"],
        dynamic_axes={"logits": {0: "batch"}, "max_logits_idx": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )

    sampling_temperature = torch.tensor([0.8], dtype=torch.float32)
    sampling_top_k = torch.tensor([50], dtype=torch.int32)
    sampling_top_p = torch.tensor([0.95], dtype=torch.float32)
    sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    sampling_previous_ids = torch.zeros((1, dummy_history_len), dtype=torch.int32)
    torch.onnx.export(
        TOPK_TOPP_SAMPLING().eval(),
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
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": sample_rate,
            "max_audio_samples": MAX_INPUT_AUDIO_LENGTH,
            "audio_hop_length": hop_length,
            "merge_factor": merge_factor,
            "prompt_token_ids": prompt_token_ids,
            "special_token_ids": special_token_ids,
            "supported_languages": SUPPORTED_LANGUAGES,
        }
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

    tokenizer_dir = onnx_folder / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))
    print(f"[Tokenizer] Prepared tokenizer -> {final_onnx_folder / 'tokenizer'}")

    import Shared_Merged

    print("\n[SharedMerged] Building ASR prefill/decode strategy graphs ...")
    bundle = Shared_Merged.build_shared_merged_bundle(
        split_export_folder,
        out_folder=onnx_folder,
        model_file_names=MODEL_FILE_NAMES,
    )
    embed_dedup = bundle.get("embed_dedup")
    embed_consolidated = bundle.get("embed_consolidated")
    embed_shared = embed_dedup or embed_consolidated
    skip_standalone = ("encoder", "embed") if embed_shared else ("encoder",)
    copied_standalones = Shared_Merged.copy_runtime_standalones(
        split_export_folder,
        onnx_folder,
        MODEL_FILE_NAMES,
        skip_roles=skip_standalone,
    )
    replace_onnx_metadata(
        str(onnx_folder / MODEL_FILE_NAMES["metadata"]), onnx_metadata
    )

    for name, path in bundle["graphs"].items():
        print(f"    {name} ({Path(path).stat().st_size} bytes)")
    print(
        f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
        f"({Path(bundle['shared_data']).stat().st_size} bytes)"
    )
    if embed_dedup:
        print(
            f"    {MODEL_FILE_NAMES['embed']} shares the verified tied lm_head table "
            f"({Path(onnx_folder / MODEL_FILE_NAMES['embed']).stat().st_size} bytes)"
        )
    elif embed_consolidated:
        print(
            f"    {MODEL_FILE_NAMES['embed']} reads its embedding table from the shared bundle "
            f"({Path(onnx_folder / MODEL_FILE_NAMES['embed']).stat().st_size} bytes)"
        )
    print(f"    Standalone ASR graphs copied: {len(copied_standalones)}")
    _split_export_temp.cleanup()
    print("[SharedMerged] Removed automatic split-graph staging directory.")

print("\nStaged export complete. Validating ONNX bundle ...\n")
subprocess.run(
    [
        sys.executable,
        str(script_dir / "Inference_Audio8_ASR_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ],
    cwd=str(script_dir),
    check=True,
)
commit_staged_export(onnx_folder, final_onnx_folder)
_output_export_temp.cleanup()
print(f"\nExport complete -> {final_onnx_folder}\n")


# ══════════════════════════════════════════════════════════════════════════════
