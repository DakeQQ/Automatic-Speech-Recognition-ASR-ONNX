import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import safe_open
from torch import Tensor, nn
from torch.onnx import symbolic_helper
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
# Works across transformers 4.5x and 5.x (only the config + module-tree closure is used).

from STFT_Process import STFT_Process
from Shared_Merged import (
    METADATA_MODEL_NAME,
    ROUTER_DEGRADED_PROBABILITY_OUTPUT,
    ROUTER_MODEL_NAME,
    VARIANT_MODEL_FILE_NAMES,
    VARIANT_NAMES,
)



# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
maga_path                      = Path.home() / "Downloads" / "Mega-ASR"
base_model_path                = maga_path / "Qwen3-ASR-1.7B"
mega_lora_path                 = maga_path / "mega-asr-merged"
router_checkpoint_path         = maga_path / "audio_quality_router" / "best_acc_model.safetensors"
script_dir                     = Path(__file__).resolve().parent
onnx_folder                    = script_dir / "Mega_ASR_ONNX"
_EXPORT_VARIANT                = os.environ.get("MEGA_ASR_EXPORT_VARIANT")
if _EXPORT_VARIANT not in (None, *VARIANT_NAMES):
    raise ValueError(
        "MEGA_ASR_EXPORT_VARIANT must be one of "
        f"{VARIANT_NAMES}, got {_EXPORT_VARIANT!r}."
    )
_artifact_variant              = _EXPORT_VARIANT or "mega"
download_path                  = str(base_model_path)
_split_export_temp             = tempfile.TemporaryDirectory(prefix="mega_asr_split_")
split_export_folder            = Path(_split_export_temp.name)

MODEL_FILE_NAMES = dict(VARIANT_MODEL_FILE_NAMES[_artifact_variant])

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

# Weight-quantization-friendly reorder (exact and absorbed into the weights).
REORDER_DOWNPROJ_FOR_QUANT     = True                          # Reorder MLP intermediate channels so down_proj block-quant groups are magnitude-homogeneous.
REORDER_OPROJ_FOR_QUANT        = True                          # Reorder each head's head_dim so o_proj sub-head groups are homogeneous. Pure win for f16 KV.
REORDER_KEY                    = "absmean"                     # "absmean" (best at group=32) | "L4" (best at group=128) | "rms" | "std".

OPSET                          = 20                            # ONNX Runtime opset version.
# ========================================================================

# Fixed Qwen3-ASR model constants and metadata defaults; these are not user tunables.
_MODEL_SAMPLE_RATE             = 16000
_MODEL_WINDOW_TYPE             = "hann"
_MODEL_NUM_MELS                = 128
_MODEL_NFFT_STFT               = 400
_MODEL_WINDOW_LENGTH           = 400
_MODEL_HOP_LENGTH              = 160
_LANG_PREFIX                   = "language "
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


class ADDITIVE_LORA_LINEAR(nn.Module):
    """Apply a low-rank LoRA update without materializing its dense weight delta."""

    def __init__(
        self,
        a_matrix: Tensor,
        b_matrix: Tensor,
        bias: Tensor | None = None,
    ) -> None:
        super().__init__()
        if a_matrix.ndim != 2 or b_matrix.ndim != 2:
            raise ValueError("LoRA factors must be rank-2 matrices.")
        if b_matrix.shape[1] != a_matrix.shape[0]:
            raise ValueError(
                "LoRA factor rank mismatch: "
                f"A={tuple(a_matrix.shape)}, B={tuple(b_matrix.shape)}."
            )
        if bias is not None and bias.numel() != b_matrix.shape[0]:
            raise ValueError(
                "LoRA bias width does not match its output factor: "
                f"bias={tuple(bias.shape)}, B={tuple(b_matrix.shape)}."
            )
        # Frozen parameters, rather than non-persistent buffers, are emitted as
        # ONNX initializers by the legacy exporter and can therefore be streamed
        # once through Shared_Merged's external-data bundle.
        self.a = nn.Parameter(
            a_matrix.detach().to(dtype=torch.float32).contiguous().clone(),
            requires_grad=False,
        )
        self.b = nn.Parameter(
            b_matrix.detach().to(dtype=torch.float32).contiguous().clone(),
            requires_grad=False,
        )
        self.has_bias = bias is not None
        if bias is not None:
            self.bias = nn.Parameter(
                bias.detach().to(dtype=torch.float32).reshape(-1).contiguous().clone(),
                requires_grad=False,
            )

    def forward(self, hidden_states: Tensor) -> Tensor:
        output = F.linear(F.linear(hidden_states, self.a), self.b)
        if self.has_bias:
            output = output + self.bias
        return output

    @torch.no_grad()
    def reorder_input_channels(self, permutation: Tensor) -> None:
        self.a.copy_(self.a[:, permutation])

    @torch.no_grad()
    def reorder_output_channels(self, permutation: Tensor) -> None:
        self.b.copy_(self.b[permutation])
        if self.has_bias:
            self.bias.copy_(self.bias[permutation])


class FUSED_ADDITIVE_LORA(nn.Module):
    """Concatenate independent low-rank updates for a fused output projection."""

    def __init__(self, parts: Sequence[ADDITIVE_LORA_LINEAR]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("A fused LoRA projection needs at least one part.")
        self.parts = nn.ModuleList(parts)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return torch.cat(
            [part(hidden_states) for part in self.parts],
            dim=-1,
        )


class MegaLoRAFactors:
    """Load the Mega-ASR adapter as scaled A/B factors keyed by model module name.

    The legacy exporter formed a dense ``B @ A`` tensor and added it to every
    base weight. Keeping the factors separate lets the exported Mega graphs use
    ``base_projection(x) + B(A(x))`` while Base and Mega share the unchanged
    base initializers. Some decoder targets occur under both historical module
    namespaces in the checkpoint; their factors are rank-concatenated so both
    legacy updates remain part of one additive projection.
    """

    def __init__(self, factors: dict[str, tuple[Tensor, Tensor]]) -> None:
        self._factors = factors
        self._used: set[str] = set()

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        while True:
            for prefix in ("base_model.model.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            else:
                break
        if name.startswith("thinker.layers."):
            name = name.replace("thinker.layers.", "thinker.model.layers.", 1)
        return name

    @classmethod
    def _split_lora_key(
        cls, key: str
    ) -> tuple[str | None, str | None, str | None]:
        for marker, kind in ((".lora_A.", "A"), (".lora_B.", "B")):
            if marker in key:
                raw_name = key.split(marker)[0]
                while True:
                    for prefix in ("base_model.model.", "model."):
                        if raw_name.startswith(prefix):
                            raw_name = raw_name[len(prefix):]
                            break
                    else:
                        break
                return cls._normalize_module_name(raw_name), raw_name, kind
        return None, None, None

    @staticmethod
    def _load_adapter_state(adapter_dir: Path) -> dict[str, Tensor]:
        safetensors_path = adapter_dir / "adapter_model.safetensors"
        if safetensors_path.exists():
            return safe_load_file(str(safetensors_path))
        return torch.load(adapter_dir / "adapter_model.bin", map_location="cpu")

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def from_adapter(
        cls,
        parent_module: nn.Module,
        adapter_dir: str | os.PathLike[str],
    ) -> "MegaLoRAFactors":
        adapter_path = Path(adapter_dir)
        config = cls._load_json(adapter_path / "adapter_config.json")
        blocks_path = adapter_path / "mega_lora_blocks.json"
        blocks = cls._load_json(blocks_path) if blocks_path.exists() else {}
        if bool(config.get("fan_in_fan_out", False)):
            raise ValueError(
                "Mega-ASR additive LoRA export expects fan_in_fan_out=False."
            )

        alpha_pattern = config.get("alpha_pattern") or {}
        rank_pattern = config.get("rank_pattern") or {}
        module_dict = dict(parent_module.named_modules())
        grouped: dict[str, dict[str, dict[str, Tensor]]] = {}
        for key, tensor in cls._load_adapter_state(adapter_path).items():
            module_name, raw_module_name, factor_kind = cls._split_lora_key(key)
            if (
                module_name is None
                or raw_module_name is None
                or factor_kind is None
            ):
                continue
            raw_pair = grouped.setdefault(module_name, {}).setdefault(
                raw_module_name, {}
            )
            previous = raw_pair.get(factor_kind)
            if previous is not None:
                raise ValueError(
                    f"Duplicate LoRA {factor_kind} factor for {raw_module_name}."
                )
            raw_pair[factor_kind] = tensor.detach().cpu()

        lora_alpha = float(config.get("lora_alpha", 1))
        default_rank = config.get("r")
        factors: dict[str, tuple[Tensor, Tensor]] = {}
        missing_pairs: list[str] = []
        unresolved_modules: list[str] = []
        for module_name, raw_pairs in grouped.items():
            module = module_dict.get(module_name)
            if module is None or not hasattr(module, "weight"):
                unresolved_modules.append(module_name)
                continue
            merged_a_parts: list[Tensor] = []
            merged_b_parts: list[Tensor] = []
            for raw_module_name, pair in raw_pairs.items():
                if set(pair) != {"A", "B"}:
                    missing_pairs.append(raw_module_name)
                    continue
                a_matrix = pair["A"].to(dtype=torch.float32).contiguous()
                b_matrix = pair["B"].to(dtype=torch.float32).contiguous()
                if a_matrix.ndim != 2 or b_matrix.ndim != 2:
                    raise ValueError(
                        f"LoRA factors for {raw_module_name} must be rank-2: "
                        f"A={tuple(a_matrix.shape)}, B={tuple(b_matrix.shape)}."
                    )
                if b_matrix.shape[1] != a_matrix.shape[0]:
                    raise ValueError(
                        f"LoRA factor rank mismatch for {raw_module_name}: "
                        f"A={tuple(a_matrix.shape)}, B={tuple(b_matrix.shape)}."
                    )

                module_blocks = blocks.get(raw_module_name) or blocks.get(module_name)
                if module_blocks:
                    a_parts: list[Tensor] = []
                    b_parts: list[Tensor] = []
                    for block in module_blocks:
                        start = int(block["start"])
                        end = int(block["end"])
                        block_rank = int(block.get("rank", end - start))
                        block_alpha = float(block.get("alpha", block_rank))
                        if not 0 <= start < end <= a_matrix.shape[0]:
                            raise ValueError(
                                f"Invalid LoRA block [{start}, {end}) for "
                                f"{raw_module_name} with rank {a_matrix.shape[0]}."
                            )
                        if block_rank <= 0:
                            raise ValueError(
                                f"LoRA block rank must be positive for {raw_module_name}."
                            )
                        a_parts.append(a_matrix[start:end])
                        b_parts.append(
                            b_matrix[:, start:end] * (block_alpha / block_rank)
                        )
                    a_matrix = torch.cat(a_parts, dim=0).contiguous()
                    b_matrix = torch.cat(b_parts, dim=1).contiguous()
                else:
                    adapter_rank = rank_pattern.get(
                        raw_module_name, rank_pattern.get(module_name, default_rank)
                    )
                    if adapter_rank is None:
                        adapter_rank = a_matrix.shape[0]
                    adapter_rank = int(adapter_rank)
                    if adapter_rank <= 0:
                        raise ValueError(
                            f"LoRA rank must be positive for {raw_module_name}."
                        )
                    adapter_alpha = float(
                        alpha_pattern.get(
                            raw_module_name,
                            alpha_pattern.get(module_name, lora_alpha),
                        )
                    )
                    b_matrix = (
                        b_matrix * (adapter_alpha / adapter_rank)
                    ).contiguous()

                if tuple(module.weight.shape) != (
                    b_matrix.shape[0],
                    a_matrix.shape[1],
                ):
                    raise ValueError(
                        f"LoRA factors for {raw_module_name} resolve to "
                        f"{(b_matrix.shape[0], a_matrix.shape[1])}, but the base "
                        f"weight shape is {tuple(module.weight.shape)}."
                    )
                merged_a_parts.append(a_matrix)
                merged_b_parts.append(b_matrix)
            if merged_a_parts:
                factors[module_name] = (
                    torch.cat(merged_a_parts, dim=0).contiguous(),
                    torch.cat(merged_b_parts, dim=1).contiguous(),
                )

        if missing_pairs or unresolved_modules:
            details = []
            if missing_pairs:
                details.append(f"missing A/B pairs: {missing_pairs[:5]}")
            if unresolved_modules:
                details.append(f"unresolved modules: {unresolved_modules[:5]}")
            raise RuntimeError("Could not load Mega-ASR LoRA factors; " + "; ".join(details))
        if not factors:
            raise RuntimeError("Mega-ASR adapter did not resolve any target module.")
        return cls(factors)

    def require(self, module_name: str) -> tuple[Tensor, Tensor]:
        normalized_name = self._normalize_module_name(module_name)
        try:
            factors = self._factors[normalized_name]
        except KeyError as error:
            raise KeyError(f"Mega-ASR LoRA is missing {normalized_name}.") from error
        self._used.add(normalized_name)
        return factors

    def assert_all_used(self) -> None:
        unused = sorted(set(self._factors) - self._used)
        if unused:
            raise RuntimeError(
                "Mega-ASR LoRA targets were loaded but never exported: "
                f"{unused[:10]}"
            )

    def __len__(self) -> int:
        return len(self._factors)


def make_additive_lora(
    lora_factors: MegaLoRAFactors,
    module_name: str,
    input_scale: Tensor | None = None,
    input_bias: Tensor | None = None,
    output_scale: float | Tensor | None = None,
) -> ADDITIVE_LORA_LINEAR:
    """Transform one LoRA update through export-time affine fusions.

    When a LayerNorm affine is folded into a following linear, the LoRA A
    factor receives the scale and its beta contribution becomes a small output
    bias. Q/K score scaling is folded into B for the audio encoder.
    """
    a_matrix, b_matrix = lora_factors.require(module_name)
    lora_bias = None
    if input_bias is not None:
        input_bias = input_bias.detach().to(dtype=torch.float32).reshape(1, -1)
        if input_bias.shape[1] != a_matrix.shape[1]:
            raise ValueError(
                f"LoRA input bias width mismatch for {module_name}: "
                f"{input_bias.shape[1]} != {a_matrix.shape[1]}."
            )
        lora_bias = F.linear(F.linear(input_bias, a_matrix), b_matrix).reshape(-1)
    if input_scale is not None:
        input_scale = input_scale.detach().to(dtype=torch.float32).reshape(-1)
        if input_scale.numel() != a_matrix.shape[1]:
            raise ValueError(
                f"LoRA input scale width mismatch for {module_name}: "
                f"{input_scale.numel()} != {a_matrix.shape[1]}."
            )
        a_matrix = a_matrix * input_scale.unsqueeze(0)
    if output_scale is not None:
        output_scale = torch.as_tensor(output_scale, dtype=torch.float32).reshape(-1)
        if output_scale.numel() not in (1, b_matrix.shape[0]):
            raise ValueError(
                f"LoRA output scale width mismatch for {module_name}: "
                f"{output_scale.numel()} not in (1, {b_matrix.shape[0]})."
            )
        b_matrix = b_matrix * output_scale.reshape(-1, 1)
        if lora_bias is not None:
            lora_bias = lora_bias * output_scale
    return ADDITIVE_LORA_LINEAR(a_matrix, b_matrix, lora_bias)


class ONNXLogMelSpectrogram(nn.Module):
    """ONNX-exportable realization of the official LogMelSpectrogram frontend."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> None:
        super().__init__()
        self.stft = STFT_Process(
            model_type="stft_B",
            n_fft=n_fft,
            win_length=win_length,
            hop_len=hop_length,
            max_frames=0,
            window_type="hann",
            center_pad=True,
            pad_mode="reflect",
            drop_last_frame=False,
        ).eval()
        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=(n_fft // 2) + 1,
            f_min=0.0,
            f_max=sample_rate / 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1).unsqueeze(0).contiguous()
        self.register_buffer("mel_filters", mel_filters, persistent=False)

    def forward(self, waveform: Tensor) -> Tensor:
        real, imag = self.stft(waveform.unsqueeze(1))
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_filters, power)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        return (log_mel + 4.0) * 0.25


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Identity()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        weights = self.query(x).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class ConvFrontend(nn.Module):
    def __init__(self, n_mels: int, d_model: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Identity(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Identity(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class AudioQualityClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 192,
        nhead: int = 4,
        dim_feedforward: int = 512,
        max_len: int = 3000,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.downsample_rate = 4
        self.frontend = ConvFrontend(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len // 4 + 100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        encoder_layer.dropout = nn.Identity()
        encoder_layer.dropout1 = nn.Identity()
        encoder_layer.dropout2 = nn.Identity()
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            norm=nn.LayerNorm(d_model),
        )
        self.pooling = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Identity(),
            nn.Linear(d_model // 2, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def forward(self, mels: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.frontend(mels)
        time_steps = x.shape[1]
        if mask is not None:
            mask = mask[:, :: self.downsample_rate]
            if mask.shape[1] > time_steps:
                mask = mask[:, :time_steps]
            elif mask.shape[1] < time_steps:
                pad = torch.ones(
                    mask.shape[0],
                    time_steps - mask.shape[1],
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask = torch.cat([mask, pad], dim=1)
        x = self.pos_encoder(x)
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pooling(x, mask)
        return self.classifier(x)


def create_audio_quality_model(config: dict[str, Any]) -> nn.Module:
    return AudioQualityClassifier(
        n_mels=config.get("n_mels", 80),
        d_model=config.get("d_model", 192),
        nhead=config.get("nhead", 4),
        dim_feedforward=config.get("dim_feedforward", 512),
        max_len=config.get("max_len", 3000),
        num_classes=config.get("num_classes", 2),
    )


class ONNX_MEGA_ASR_AUDIO_QUALITY_ROUTER(nn.Module):
    """Dynamic-length ONNX realization of AudioQualityClassifier.forward()."""

    def __init__(self, model: AudioQualityClassifier, mel_extractor: nn.Module) -> None:
        super().__init__()
        if len(model.transformer.layers) != 1:
            raise RuntimeError(
                "The official Mega-ASR router export expects its one encoder layer."
            )
        self.frontend = model.frontend
        self.pos_encoder = model.pos_encoder
        self.encoder_layer = model.transformer.layers[0]
        self.transformer_norm = model.transformer.norm
        self.pooling = model.pooling
        self.classifier = model.classifier
        self.mel_extractor = mel_extractor
        self.num_heads = int(self.encoder_layer.self_attn.num_heads)
        self.embed_dim = int(self.encoder_layer.self_attn.embed_dim)
        self.head_dim = self.embed_dim // self.num_heads
        if self.embed_dim % self.num_heads:
            raise RuntimeError("Router attention embed_dim must divide num_heads.")
        self.attention_scale = self.head_dim ** -0.5

    def _self_attention(self, hidden_states: Tensor) -> Tensor:
        attention = self.encoder_layer.self_attn
        qkv = F.linear(
            hidden_states,
            attention.in_proj_weight,
            attention.in_proj_bias,
        )
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = query.reshape(1, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        key = key.reshape(1, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        value = value.reshape(1, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        weights = torch.matmul(query * self.attention_scale, key.transpose(-1, -2))
        weights = torch.softmax(weights, dim=-1)
        attended = torch.matmul(weights, value)
        attended = attended.permute(0, 2, 1, 3).reshape(1, -1, self.embed_dim)
        return attention.out_proj(attended)

    def forward(self, waveform: Tensor) -> Tensor:
        mel = self.mel_extractor(waveform.squeeze(1))
        hidden_states = self.frontend(mel.transpose(1, 2))
        hidden_states = self.pos_encoder(hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_layer.norm1(hidden_states)
        hidden_states = residual + self._self_attention(hidden_states)
        residual = hidden_states
        hidden_states = self.encoder_layer.norm2(hidden_states)
        hidden_states = residual + self.encoder_layer.linear2(
            self.encoder_layer.activation(self.encoder_layer.linear1(hidden_states))
            )
        hidden_states = self.transformer_norm(hidden_states)
        hidden_states = self.pooling(hidden_states, mask=None)
        logits = self.classifier(hidden_states)
        return F.softmax(logits, dim=-1)[:, 1].reshape(1)


def export_audio_quality_router() -> Path:
    if not router_checkpoint_path.is_file():
        raise FileNotFoundError(router_checkpoint_path)
    with safe_open(str(router_checkpoint_path), framework="pt", device="cpu") as handle:
        checkpoint_config = json.loads(handle.metadata().get("config", "{}"))
    router_config = checkpoint_config.get("model", {})
    router_model = create_audio_quality_model(router_config)
    router_model.load_state_dict(safe_load_file(str(router_checkpoint_path), device="cpu"))
    router = ONNX_MEGA_ASR_AUDIO_QUALITY_ROUTER(
        router_model.eval(),
        ONNXLogMelSpectrogram(
            sample_rate=16000,
            n_mels=router_config.get("n_mels", 80),
        ).eval(),
    ).eval()
    output_path = onnx_folder / ROUTER_MODEL_NAME
    torch.onnx.export(
        router,
        (torch.zeros((1, 1, 16000), dtype=torch.float32),),
        str(output_path),
        input_names=["waveform"],
        output_names=[ROUTER_DEGRADED_PROBABILITY_OUTPUT],
        dynamic_axes={"waveform": {2: "audio_samples"}},
        opset_version=OPSET,
        dynamo=False,
    )
    replace_onnx_metadata(
        output_path,
        build_model_metadata(
            {
                "mega_asr_router": {
                    "input": "waveform",
                    "output": ROUTER_DEGRADED_PROBABILITY_OUTPUT,
                    "sample_rate": 16000,
                    "threshold": 0.5,
                }
            }
        ),
    )
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ── Vendored Qwen3-ASR configuration + modeling (transformers==4.57.6) ─────────
#
# The QwenASR-specific configuration and modeling classes are inlined here (they
# previously lived in ./modeling_modified/inference_asr_standalone.py) so this
# export script is fully standalone. Only the closure required to load
# `Qwen3ASRForConditionalGeneration` through AutoModel.from_pretrained is kept; the
# processor, audio/text I/O helpers and the high-level inference wrapper were
# dropped because every forward() consumed by the export is re-implemented below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────────────
class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRTextConfig(PretrainedConfig):
    model_type = "qwen3_asr_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"
    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        # Build the sub-configs BEFORE `super().__init__()`: transformers>=5 runs
        # `@strict` validators (e.g. `validate_token_ids` -> `get_text_config()`)
        # inside `super().__init__()`, so `self.text_config` must already exist.
        # On the transformers 4.x path this ordering is harmless.
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config
        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        # `getattr` guard stays safe even if a validator calls this before
        # `__init__` has finished assigning `text_config`.
        return getattr(self, "text_config", self)


class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {"thinker_config": Qwen3ASRThinkerConfig}

    def __init__(self, thinker_config=None, support_languages=None, **kwargs):
        # Build the sub-config BEFORE `super().__init__()`: transformers>=5 runs
        # `@strict` validators inside `super().__init__()` that call
        # `get_text_config()` -> `self.thinker_config`, which would otherwise not
        # exist yet (AttributeError on newer transformers).
        if thinker_config is None:
            thinker_config = {}
        if isinstance(thinker_config, dict):
            thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.thinker_config = thinker_config
        self.support_languages = support_languages
        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        thinker_config = getattr(self, "thinker_config", None)
        if thinker_config is None:
            return self
        return thinker_config.get_text_config(decoder=decoder, encoder=encoder)


# ── Modeling ──────────────────────────────────────────────────────────────────
# Only the ``__init__`` closure that builds the module tree is kept (so
# AutoModel.from_pretrained can load the checkpoint and expose the weights/buffers
# the ONNX graphs read). Every forward()/generate() + their helpers were removed —
# the export re-implements all computation in the QWEN3_ASR_* wrappers below.
class Qwen3ASRTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class Qwen3ASRTextAttention(nn.Module):
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)


class Qwen3ASRTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class Qwen3ASRThinkerTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3ASRTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRTextMLP(config)
        self.input_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3ASRPreTrainedModel(PreTrainedModel):
    config: Qwen3ASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"attentions": Qwen3ASRTextAttention}


class Qwen3ASRPreTrainedModelForConditionalGeneration(Qwen3ASRPreTrainedModel):
    pass


class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1
        self.config = config
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


class Qwen3ASRAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer("positional_embedding", torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1), persistent=False)


class Qwen3ASRAudioEncoder(Qwen3ASRPreTrainedModel):
    config: Qwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__(config)
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.post_init()


class Qwen3ASRThinkerTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3ASRConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling.get("rope_type", "default") if getattr(config, "rope_scaling", None) else "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if self.rope_init_fn is not None:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        else:
            # transformers>=5 removed the "default" entry from ROPE_INIT_FUNCTIONS
            # (plain RoPE is no longer registry-based). Compute the standard,
            # non-scaled inverse frequencies directly so this module stays agnostic
            # to the installed transformers version.
            base = getattr(config, "rope_theta", None)
            if base is None and getattr(config, "rope_scaling", None):
                base = config.rope_scaling.get("rope_theta")
            base = 10000.0 if base is None else base
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
            self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class Qwen3ASRThinkerTextModel(Qwen3ASRPreTrainedModel):
    config: Qwen3ASRConfig
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3ASRThinkerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3ASRThinkerTextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()


class Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen3ASRThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer", "Qwen3ASRThinkerTextDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        # transformers>=5 pops generation token-ids (e.g. pad_token_id) out of the
        # config, so it may no longer be an attribute; fall back to -1 defensively.
        pad_token_id = getattr(self.config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)


class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.config = config
        self.thinker = Qwen3ASRThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()


AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)


def _get_feat_extract_output_lengths(input_lengths: Tensor) -> Tensor:
    input_lengths_leave = input_lengths % 100
    feat_lengths = torch.clamp(input_lengths_leave - 1, min=0) // 2 + 1
    feat_lengths = feat_lengths * (input_lengths_leave > 0).to(feat_lengths.dtype)
    feat_lengths_2 = torch.clamp(feat_lengths - 1, min=0) // 2 + 1
    feat_lengths_2 = feat_lengths_2 * (feat_lengths > 0).to(feat_lengths_2.dtype)
    feat_lengths_3 = torch.clamp(feat_lengths_2 - 1, min=0) // 2 + 1
    feat_lengths_3 = feat_lengths_3 * (feat_lengths_2 > 0).to(feat_lengths_3.dtype)
    return feat_lengths_3 + (input_lengths // 100) * 13


def replace_gelu_with_tanh(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh(child)


def absorb_layer_norm_affine(norm: torch.nn.LayerNorm, linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(
                torch.zeros(linear.out_features, dtype=linear.weight.dtype)
            )
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias   = None


def refresh_non_persistent_buffers(model: torch.nn.Module, text_cfg) -> None:
    """Re-materialize the ``__init__``-computed, non-persistent buffers this export
    reads directly. transformers>=5 overwrites every non-persistent buffer with
    ``torch.empty_like`` during ``from_pretrained`` and only re-initializes a known
    set of module types afterwards (a RotaryEmbedding is only refilled when it owns
    an ``original_inv_freq`` attribute, and the Whisper-style sinusoidal position
    embedding is not handled at all) — leaving them as uninitialized memory. That
    silently corrupts the RoPE table and audio positions, producing garbage output.
    On transformers 4.x these buffers already hold the correct values, so this is a
    harmless no-op there."""
    # ── Default/plain RoPE inverse frequencies ────────────────────────────────
    llm = getattr(model.thinker, "model", None)
    rotary = getattr(llm, "rotary_emb", None)
    if rotary is not None and getattr(rotary, "inv_freq", None) is not None:
        base = getattr(text_cfg, "rope_theta", None)
        if base is None and getattr(text_cfg, "rope_scaling", None):
            base = text_cfg.rope_scaling.get("rope_theta")
        base = 10000.0 if base is None else float(base)
        head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
        dim = int(head_dim * getattr(text_cfg, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        rotary.inv_freq = inv_freq.to(rotary.inv_freq.dtype)
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = rotary.inv_freq
    # ── Whisper-style sinusoidal audio positional embedding ───────────────────
    sinu = getattr(model.thinker.audio_tower, "positional_embedding", None)
    if sinu is not None and getattr(sinu, "positional_embedding", None) is not None:
        length, channels = int(sinu.positional_embedding.shape[0]), int(sinu.positional_embedding.shape[1])
        log_timescale_increment = np.log(10000) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        sinu.positional_embedding = pe.to(sinu.positional_embedding.dtype)


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


# ══════════════════════════════════════════════════════════════════════════════
# ── Audio Encoder (fused mel + encoder + prompt concat) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_ENCODER(torch.nn.Module):
    def __init__(
        self,
        audio_tower: torch.nn.Module,
        embed_tokens: torch.nn.Embedding,
        head_ids: Sequence[int],
        tail_ids: Sequence[int],
        query_suffix_ids: Sequence[int],
        lora_factors: MegaLoRAFactors | None = None,
    ) -> None:
        super().__init__()
        self.audio_tower = audio_tower.float()
        self.has_lora = lora_factors is not None
        self.lora_qkv = nn.ModuleList()
        self.lora_o = nn.ModuleList()
        self.lora_fc1 = nn.ModuleList()
        self.lora_fc2 = nn.ModuleList()
        self.lora_conv_out = None
        self.lora_proj1 = None
        self.lora_proj2 = None
        # Raw int16 PCM uses an exact power-of-two scale. Fold it into the
        # immutable DFT kernel so the runtime path needs only int16 -> float32.
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")
        stft_input_scale = (
            (1.0 / _MODEL_AUDIO_PCM_SCALE)
            if self.input_audio_is_int16
            else 1.0
        )
        replace_gelu_with_tanh(self.audio_tower)
        self.audio_tower.act = torch.nn.GELU(approximate="tanh")
        for layer in self.audio_tower.layers:
            layer.activation_fn = torch.nn.GELU(approximate="tanh")
        self._fuse_encoder_weights(lora_factors)

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
        self.register_buffer("mel_filters", mel_filters.to(dtype=torch.float32), persistent=False)

        chunk_size           = int(self.audio_tower.n_window * 2)          # 100
        self.chunk_size      = chunk_size
        self.chunk_size_minus = chunk_size - 1
        chunk_aftercnn       = int(_get_feat_extract_output_lengths(torch.tensor([chunk_size])).item()) # 13
        self.chunk_aftercnn  = chunk_aftercnn
        chunks_per_window    = int(self.audio_tower.n_window_infer // chunk_size)  # 8
        self.chunks_per_window = chunks_per_window
        self.chunks_per_window_minus = chunks_per_window - 1
        tokens_per_window    = chunks_per_window * chunk_aftercnn           # 104
        self.tokens_per_window = tokens_per_window
        self.model_dim       = int(self.audio_tower.config.d_model)
        self.output_dim      = int(self.audio_tower.config.output_dim)
        self.conv_out_features = int(self.audio_tower.conv_out.in_features)
        self.attn_out_features = int(
            self.audio_tower.layers[0].self_attn.out_proj.in_features
        )
        self.num_heads       = int(self.audio_tower.layers[0].self_attn.num_heads)
        self.head_dim        = self.model_dim // self.num_heads

        max_mel_frames = MAX_INPUT_AUDIO_LENGTH // _MODEL_HOP_LENGTH
        self.max_chunks = (max_mel_frames + self.chunk_size_minus) // chunk_size
        key_mask_lookup = torch.zeros(
            1,
            tokens_per_window + 1,
            1,
            1,
            tokens_per_window,
            dtype=torch.float32,
        )
        for n in range(tokens_per_window + 1):
            key_mask_lookup[0, n, 0, 0, n:] = -128

        self.register_buffer(
            "pos",
            self.audio_tower.positional_embedding.positional_embedding[
                :self.chunk_aftercnn
            ].unsqueeze(0).float(),
            persistent=False,
        )
        self.register_buffer(
            "chunk_starts_full",
            torch.arange(self.max_chunks + 1, dtype=torch.int64) * chunk_size,
            persistent=False,
        )
        self.register_buffer(
            "aftercnn_lens_lookup",
            _get_feat_extract_output_lengths(
                torch.arange(chunk_size + 1, dtype=torch.int64)
            ).long(),
            persistent=False,
        )
        self.register_buffer(
            "chunk_pad_zeros",
            torch.zeros(
                (self.chunks_per_window_minus, chunk_aftercnn, self.model_dim),
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
            torch.zeros((1, _MODEL_NUM_MELS, chunk_size), dtype=torch.float32),
            persistent=False,
        )

        with torch.no_grad():
            head_ids_tensor = torch.tensor([list(head_ids)], dtype=torch.int32)
            tail_ids_tensor = torch.tensor([list(tail_ids)], dtype=torch.int32)
            query_suffix_tensor = torch.tensor([list(query_suffix_ids)], dtype=torch.int32)
            self.register_buffer("head_embed", embed_tokens(head_ids_tensor).float(), persistent=False)
            self.register_buffer("tail_embed", embed_tokens(tail_ids_tensor).float(), persistent=False)
            self.register_buffer("query_suffix_embed", embed_tokens(query_suffix_tensor).float(), persistent=False)
        self.head_len = self.head_embed.shape[1]
        self.tail_len = self.tail_embed.shape[1]

    def _fuse_encoder_weights(
        self, lora_factors: MegaLoRAFactors | None
    ) -> None:
        with torch.no_grad():
            for layer_index, layer in enumerate(self.audio_tower.layers):
                attn = layer.self_attn
                qkv  = torch.nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features, bias=True)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                qkv.bias.copy_(torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0))

                scale_sqrt = attn.scaling ** 0.5
                if lora_factors is not None:
                    layer_prefix = f"thinker.audio_tower.layers.{layer_index}"
                    attention_norm = layer.self_attn_layer_norm
                    ffn_norm = layer.final_layer_norm
                    self.lora_qkv.append(
                        FUSED_ADDITIVE_LORA(
                            [
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.self_attn.q_proj",
                                    input_scale=attention_norm.weight,
                                    input_bias=attention_norm.bias,
                                    output_scale=scale_sqrt,
                                ),
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.self_attn.k_proj",
                                    input_scale=attention_norm.weight,
                                    input_bias=attention_norm.bias,
                                    output_scale=scale_sqrt,
                                ),
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.self_attn.v_proj",
                                    input_scale=attention_norm.weight,
                                    input_bias=attention_norm.bias,
                                ),
                            ]
                        )
                    )
                    self.lora_o.append(
                        make_additive_lora(
                            lora_factors,
                            f"{layer_prefix}.self_attn.out_proj",
                        )
                    )
                    self.lora_fc1.append(
                        make_additive_lora(
                            lora_factors,
                            f"{layer_prefix}.fc1",
                            input_scale=ffn_norm.weight,
                            input_bias=ffn_norm.bias,
                        )
                    )
                    self.lora_fc2.append(
                        make_additive_lora(
                            lora_factors,
                            f"{layer_prefix}.fc2",
                        )
                    )
                absorb_layer_norm_affine(layer.self_attn_layer_norm, qkv)

                q_out = attn.q_proj.out_features
                k_out = attn.k_proj.out_features
                qkv.weight.data[:q_out].mul_(scale_sqrt)
                qkv.weight.data[q_out:q_out + k_out].mul_(scale_sqrt)
                qkv.bias.data[:q_out].mul_(scale_sqrt)
                qkv.bias.data[q_out:q_out + k_out].mul_(scale_sqrt)

                absorb_layer_norm_affine(layer.final_layer_norm, layer.fc1)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj

            if lora_factors is not None:
                output_norm = self.audio_tower.ln_post
                self.lora_conv_out = make_additive_lora(
                    lora_factors,
                    "thinker.audio_tower.conv_out",
                )
                self.lora_proj1 = make_additive_lora(
                    lora_factors,
                    "thinker.audio_tower.proj1",
                    input_scale=output_norm.weight,
                    input_bias=output_norm.bias,
                )
                self.lora_proj2 = make_additive_lora(
                    lora_factors,
                    "thinker.audio_tower.proj2",
                )
            absorb_layer_norm_affine(self.audio_tower.ln_post, self.audio_tower.proj1)

    def forward(self, audio: Tensor, query_embed: Tensor) -> Tuple[Tensor, Tensor]:
        # Matches WhisperFeatureExtractor exactly: no pre-emphasis, no mean/DC removal,
        # reflect-padded STFT with the unused trailing frame removed in framing,
        # then Slaney log-mel. The int16 scale is folded into the DFT kernel.
        if INPUT_AUDIO_DTYPE != "F32":
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
            1, _MODEL_NUM_MELS, -1, self.chunk_size
        ).permute(2, 0, 1, 3)
        chunk_starts = self.chunk_starts_full[:num_chunks]
        raw_chunk_lens = torch.clamp(feature_len - chunk_starts, min=0, max=self.chunk_size)
        aftercnn_lens = self.aftercnn_lens_lookup[raw_chunk_lens]
        x = F.gelu(self.audio_tower.conv2d1(chunks), approximate="tanh")
        x = F.gelu(self.audio_tower.conv2d2(x), approximate="tanh")
        x = F.gelu(self.audio_tower.conv2d3(x), approximate="tanh")
        conv_features = x.permute(0, 3, 1, 2).contiguous().view(
            -1, self.chunk_aftercnn, self.conv_out_features
        )
        x = self.audio_tower.conv_out(conv_features)
        if self.has_lora:
            x = x + self.lora_conv_out(conv_features)
        hidden_states = x + self.pos
        num_windows = POSITIVE_CEIL_DIV.apply(
            num_chunks, self.chunks_per_window
        )
        total_chunks_padded = num_windows * self.chunks_per_window
        pad_chunks = total_chunks_padded - num_chunks
        hidden_states = torch.cat(
            [hidden_states, self.chunk_pad_zeros[:pad_chunks]], dim=0
        )
        aftercnn_lens = torch.cat(
            [aftercnn_lens, self.aftercnn_pad_zeros[:pad_chunks]]
        )
        hidden_states = hidden_states.reshape(
            -1, self.tokens_per_window, self.model_dim
        )
        valid_counts = aftercnn_lens.reshape(
            -1, self.chunks_per_window
        ).sum(dim=1)
        key_mask = self.key_mask_lookup[:, valid_counts]
        for layer_index, layer in enumerate(self.audio_tower.layers):
            residual = hidden_states
            normed = layer.self_attn_layer_norm(hidden_states)
            qkv = layer.self_attn.qkv(normed)
            if self.has_lora:
                qkv = qkv + self.lora_qkv[layer_index](normed)
            qkv = qkv.reshape(
                -1,
                self.tokens_per_window,
                3,
                self.num_heads,
                self.head_dim,
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.split(1, dim=0)
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(attn + key_mask, dim=-1)
            attn = torch.matmul(attn, v).transpose(2, 3).reshape(
                -1, self.tokens_per_window, self.attn_out_features
            )
            attention_output = layer.self_attn.out_proj(attn)
            if self.has_lora:
                attention_output = attention_output + self.lora_o[layer_index](attn)
            hidden_states = residual + attention_output
            residual = hidden_states
            normed = layer.final_layer_norm(hidden_states)
            ffn_hidden = layer.fc1(normed)
            if self.has_lora:
                ffn_hidden = ffn_hidden + self.lora_fc1[layer_index](normed)
            ffn_hidden = layer.activation_fn(ffn_hidden)
            ffn_output = layer.fc2(ffn_hidden)
            if self.has_lora:
                ffn_output = ffn_output + self.lora_fc2[layer_index](ffn_hidden)
            hidden_states = residual + ffn_output
        hidden_states = self.audio_tower.ln_post(hidden_states)
        projection_hidden = self.audio_tower.proj1(hidden_states)
        if self.has_lora:
            projection_hidden = projection_hidden + self.lora_proj1(hidden_states)
        projection_hidden = self.audio_tower.act(projection_hidden)
        hidden_states = self.audio_tower.proj2(projection_hidden)
        if self.has_lora:
            hidden_states = hidden_states + self.lora_proj2(projection_hidden)
        hidden_states = hidden_states.reshape(1, -1, self.output_dim)
        encoded_len = aftercnn_lens.sum(dim=0, keepdim=True)
        audio_hidden = hidden_states[:, :encoded_len]
        concat_embed = torch.cat([self.head_embed, query_embed, self.query_suffix_embed, audio_hidden, self.tail_embed], dim=1)
        ids_len = ONNX_SHAPE_DIM.apply(concat_embed, 1)
        return concat_embed, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# ── Rotary + Mask — Prefill Phase ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_ROTARY_MASK_PREFILL(torch.nn.Module):
    def __init__(self, llm: torch.nn.Module, max_seq_len: int) -> None:
        super().__init__()
        # Mask dtype = the KV-cache dtype, so the exported mask I/O dtype is stable: float16 with the f16 KV
        # cache, float32 with a float32 KV cache. It is deliberately NOT gated on COMPUTE_IN_F32 -- in the
        # f16-storage / f32-compute path the decoder upcasts this f16 mask to f32 INTERNALLY (the mask I/O
        # dtype, and the inference runtime, stay unchanged). Added to the attention scores in QWEN3_ASR_DECODER_MAIN.
        self.mask_dtype = torch.float16 if USE_FP16_KV else torch.float32
        positions = torch.arange(max_seq_len, dtype=torch.int32)
        self.register_buffer(
            "causal_mask",
            torch.where(
                positions.view(1, 1, 1, 1, max_seq_len)
                <= positions.view(1, 1, 1, max_seq_len, 1),
                torch.tensor(0, dtype=torch.int8),
                torch.tensor(-128, dtype=torch.int8),
            ),
            persistent=False,
        )
        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.rotary_dim = int(cos.shape[-1] * 2)
        self.register_buffer(
            "rotary_pos_emb",
            torch.cat(
                [torch.cat([cos, cos], dim=-1), torch.cat([-sin, sin], dim=-1)],
                dim=-1,
            ).to(ROTARY_STORAGE_DTYPE),
            persistent=False,
        )

    @staticmethod
    def _build_rotary_table(
        llm: torch.nn.Module, max_seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = llm.rotary_emb.inv_freq.float()
        theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(theta), torch.sin(theta)

    def forward(
        self, ids_len: Tensor, history_len: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        kv_seq_len = ids_len + history_len
        rotary = self.rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_cos, rotary_sin = torch.split(
            rotary, [self.rotary_dim, self.rotary_dim], dim=-1
        )
        attention_mask = self.causal_mask[..., :ids_len, :kv_seq_len].to(
            self.mask_dtype
        )
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


# ══════════════════════════════════════════════════════════════════════════════
# ── Rotary — Decode Phase (single new token, no mask output) ──────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, llm: torch.nn.Module, max_seq_len: int) -> None:
        super().__init__()
        cos, sin = QWEN3_ASR_ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.rotary_dim = int(cos.shape[-1] * 2)
        self.register_buffer(
            "rotary_pos_emb",
            torch.cat(
                [torch.cat([cos, cos], dim=-1), torch.cat([-sin, sin], dim=-1)],
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


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoder Embedding ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_DECODER_EMBED(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embed_tokens = model.thinker.model.embed_tokens.float()

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# ── Simplified RMS LayerNorm (ORT fused SimplifiedLayerNormalization) ──────────
# ══════════════════════════════════════════════════════════════════════════════
class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Emit ONNX Runtime's fused ``SimplifiedLayerNormalization`` (RMS norm) in place of the
    Mul -> ReduceSum -> Add -> Sqrt -> Div chain.

    ORT registers this op in the DEFAULT ONNX domain (``kOnnxDomain``) with SinceVersion 1, so it is
    available at any opset. It computes ``Y = X * rsqrt(mean(X^2, axis) + epsilon) * scale`` and, with
    ``stash_type = 1``, performs the mean/rsqrt reduction in float32 regardless of the I/O dtype (so no
    manual float16-overflow guard is needed). ``forward`` runs only during export tracing; the exported
    graph uses ``symbolic`` (the whole Function collapses into one node).
    """

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        # Decoder activations and norm scales are float32. Avoid trace-only
        # no-op casts; symbolic() replaces this whole body with one fused node.
        variance   = x.pow(2).mean(dim=axis, keepdim=True)
        normalized = x * torch.rsqrt(variance + epsilon)
        return normalized * scale

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x, scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    """Return ``SimplifiedLayerNormalization(x, scale, axis, epsilon)`` -- a single fused RMS-norm node."""
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoder Main (deeply fused transformer) ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_DECODER_MAIN(torch.nn.Module):
    def __init__(
        self,
        model:        torch.nn.Module,
        num_heads:    int,
        num_kv_heads: int,
        head_dim:     int,
        num_layers:   int,
        hidden_size:  int,
        lora_factors: MegaLoRAFactors | None = None,
    ) -> None:
        super().__init__()
        self.llm      = model.thinker.model.float()
        self.lm_head  = model.thinker.lm_head.float()
        self.head_dim       = head_dim
        self.head_dim_half  = head_dim // 2
        self.num_heads      = num_heads
        self.num_kv_heads   = num_kv_heads
        self.num_kv_groups  = num_heads // num_kv_heads
        self.qk_heads       = num_heads + num_kv_heads   # total heads before split
        self.num_layers     = num_layers
        self.use_fp16_kv    = USE_FP16_KV
        self.compute_in_f32 = COMPUTE_IN_F32
        self.total_qkv_heads = self.qk_heads + self.num_kv_heads
        self.qkv_split_sizes = (self.qk_heads, self.num_kv_heads)
        self.qk_split_sizes = (self.num_heads, self.num_kv_heads)
        self.attention_output_size = int(
            self.llm.layers[0].self_attn.o_proj.in_features
        )
        self.intermediate_size = int(
            self.llm.layers[0].mlp.down_proj.in_features
        )
        self.mlp_split_sizes = (
            self.intermediate_size,
            self.intermediate_size,
        )

        # RMS norm is emitted as ORT's fused SimplifiedLayerNormalization (default ONNX domain): it computes
        # y = x * rsqrt(mean(x^2) + eps) * scale and reduces in float32 (stash_type=1). Feeding scale = 1/sqrt(N)
        # (N = normalized size) and the model's per-element eps reproduces this file's sum-based
        # r = x * rsqrt(sum(x^2) + N*eps) EXACTLY, while the float32 reduction avoids f16 reduction
        # overflow. The g*sqrt(N) norm weight stays absorbed into the following linear.
        hidden_rms_norm = self.llm.layers[0].input_layernorm
        qk_rms_norm     = self.llm.layers[0].self_attn.q_norm
        self.hidden_rms_norm_eps = float(getattr(hidden_rms_norm, "variance_epsilon", getattr(hidden_rms_norm, "eps", 1e-6)))
        self.qk_rms_norm_eps     = float(getattr(qk_rms_norm, "variance_epsilon", getattr(qk_rms_norm, "eps", self.hidden_rms_norm_eps)))
        self.register_buffer("hidden_norm_scale", torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32))
        self.register_buffer("qk_norm_scale",     torch.full((head_dim,),    head_dim ** -0.5,    dtype=torch.float32))

        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        self.has_lora = lora_factors is not None
        self.lora_qkv = nn.ModuleList()
        self.lora_o = nn.ModuleList()
        self.lora_gate_up = nn.ModuleList()
        self.lora_down = nn.ModuleList()

        self._fuse_weights(hidden_size, lora_factors)

        # Quantization-friendly weight reorder (exact, zero runtime cost; absorbed into the fused weights).
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    # ── Weight fusion (runs once at init) ──────────────────────────────────────
    def _fuse_weights(
        self,
        hidden_size: int,
        lora_factors: MegaLoRAFactors | None,
    ) -> None:
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer_index, layer in enumerate(self.llm.layers):
                attn = layer.self_attn
                has_bias = any(proj.bias is not None for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
                qkv = torch.nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                if has_bias:
                    def _bias_or_zero(proj: torch.nn.Linear) -> Tensor:
                        return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
                    qkv.bias.copy_(torch.cat([_bias_or_zero(attn.q_proj), _bias_or_zero(attn.k_proj), _bias_or_zero(attn.v_proj)], dim=0))

                input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(input_norm_weight)
                attn.qkv = qkv

                combined_scale = scale_factor * norm_factor_qk
                attn.q_norm.weight.mul_(combined_scale)
                attn.k_norm.weight.mul_(combined_scale)
                q_norm_rep = attn.q_norm.weight.repeat(self.num_heads)
                k_norm_rep = attn.k_norm.weight.repeat(self.num_kv_heads)
                attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_rep, k_norm_rep], dim=0).view(1, 1, 1, self.qk_heads, self.head_dim))

                gate = layer.mlp.gate_proj
                up   = layer.mlp.up_proj
                post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
                layer.mlp.gate_up_proj = gate_up

                if lora_factors is not None:
                    layer_prefix = f"thinker.model.layers.{layer_index}"
                    input_scale = input_norm_weight.reshape(-1)
                    post_input_scale = post_norm_weight.reshape(-1)
                    self.lora_qkv.append(
                        FUSED_ADDITIVE_LORA(
                            [
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.self_attn.q_proj",
                                    input_scale=input_scale,
                                ),
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.self_attn.k_proj",
                                    input_scale=input_scale,
                                ),
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.self_attn.v_proj",
                                    input_scale=input_scale,
                                ),
                            ]
                        )
                    )
                    self.lora_o.append(
                        make_additive_lora(
                            lora_factors,
                            f"{layer_prefix}.self_attn.o_proj",
                        )
                    )
                    self.lora_gate_up.append(
                        FUSED_ADDITIVE_LORA(
                            [
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.mlp.gate_proj",
                                    input_scale=post_input_scale,
                                ),
                                make_additive_lora(
                                    lora_factors,
                                    f"{layer_prefix}.mlp.up_proj",
                                    input_scale=post_input_scale,
                                ),
                            ]
                        )
                    )
                    self.lora_down.append(
                        make_additive_lora(
                            lora_factors,
                            f"{layer_prefix}.mlp.down_proj",
                        )
                    )

                del attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm
                del layer.input_layernorm, layer.post_attention_layernorm
                del layer.mlp.gate_proj, layer.mlp.up_proj

            # Keep lm_head PRISTINE (tied to embed_tokens): instead of folding the
            # final RMSNorm's learned weight into lm_head (which breaks the
            # embed_tokens <-> lm_head tie), apply it explicitly on the last token
            # via the final-norm scale.  Matches the Qwen-v3 reference and makes the
            # exported lm_head initializer exactly transpose(embed_tokens.weight), so
            # the shared bundle can store the tied table once.
            self.register_buffer(
                "final_norm_scale",
                self.llm.norm.weight.detach().clone().float(),
                persistent=False,
            )
            del self.llm.norm

    # ── Quantization-friendly weight reorder (exact, zero runtime cost) ─────────
    def _reorder_downproj_for_quant(self, key: str) -> None:
        """Reorder each layer's MLP intermediate channels so contiguous block-quant groups over down_proj's
        INPUT axis hold magnitude-homogeneous channels (smaller per-group scale -> lower weight-quant error).
        The SAME permutation is applied to gate_up's OUTPUT rows (gate + up halves) and down_proj's INPUT
        columns, so act(gate)*up @ down_proj is unchanged -- fully absorbed, no runtime de-permutation.
        """
        with torch.no_grad():
            for layer_index, layer in enumerate(self.llm.layers):
                W = layer.mlp.down_proj.weight              # (hidden, intermediate)
                a = W.abs()
                if key == "rms":
                    stat = (W * W).mean(0).sqrt()
                elif key == "L4":
                    stat = a.pow(4).mean(0).pow(0.25)
                elif key == "std":
                    stat = W.std(0)
                else:                                       # "absmean" (default / fallback)
                    stat = a.mean(0)
                perm  = torch.argsort(stat)
                inter = layer.mlp.down_proj.in_features
                gu    = layer.mlp.gate_up_proj.weight       # (2*intermediate, hidden): [gate; up]
                layer.mlp.gate_up_proj.weight.copy_(torch.cat([gu[:inter][perm], gu[inter:][perm]], dim=0))
                layer.mlp.down_proj.weight.copy_(W[:, perm])
                if self.has_lora:
                    lora_gate, lora_up = self.lora_gate_up[layer_index].parts
                    lora_gate.reorder_output_channels(perm)
                    lora_up.reorder_output_channels(perm)
                    self.lora_down[layer_index].reorder_input_channels(perm)

    def _reorder_oproj_for_quant(self, key: str) -> None:
        """Reorder each head's head_dim so contiguous SUB-head o_proj quant groups (group_size < head_dim)
        are magnitude-homogeneous. ONE head_dim permutation per kv head (shared by its GQA query heads) is
        applied to o_proj's INPUT columns and to the v-rows of the fused qkv; q/k/RoPE/qk_norm untouched.
        Fully absorbed into the weights. Pure win for the f16 KV cache (exact V-cache round-trip).
        """
        H, KVH, Dh, qk_heads = self.num_heads, self.num_kv_heads, self.head_dim, self.qk_heads
        G = H // KVH
        with torch.no_grad():
            for layer_index, layer in enumerate(self.llm.layers):
                Wo  = layer.self_attn.o_proj.weight                 # (hidden, H*head_dim)
                Woc = Wo.view(Wo.shape[0], H, Dh)                   # (hidden, H, head_dim)
                perms = []
                for kvh in range(KVH):                              # one order per kv head
                    cols = Woc[:, kvh * G:(kvh + 1) * G, :]         # its G query heads combined
                    a = cols.abs()
                    if key == "rms":
                        stat = (cols * cols).mean(dim=(0, 1)).sqrt()
                    elif key == "std":
                        stat = cols.reshape(-1, Dh).std(0)
                    elif key == "L4":
                        stat = a.pow(4).mean(dim=(0, 1)).pow(0.25)
                    else:                                           # "absmean" (default / fallback)
                        stat = a.mean(dim=(0, 1))
                    perms.append(torch.argsort(stat))
                # o_proj input columns: permute each head's head_dim by its kv-head order
                Woc2 = Woc.clone()
                for h in range(H):
                    Woc2[:, h, :] = Woc2[:, h, perms[h // G]]
                Wo.copy_(Woc2.reshape(Wo.shape[0], H * Dh))
                # compensate on the v-rows of the fused qkv (head_dim); q/k rows untouched
                Wq  = layer.self_attn.qkv.weight                    # (total_heads*head_dim, hidden)
                Wqr = Wq.view(-1, Dh, Wq.shape[1]).clone()          # (total_heads, head_dim, hidden)
                for kvh in range(KVH):
                    Wqr[qk_heads + kvh] = Wqr[qk_heads + kvh][perms[kvh]]
                Wq.copy_(Wqr.reshape(Wq.shape[0], Wq.shape[1]))
                if self.has_lora:
                    o_input_permutation = torch.cat(
                        [
                            perms[head_index // G] + head_index * Dh
                            for head_index in range(H)
                        ]
                    )
                    value_output_permutation = torch.cat(
                        [
                            perms[kv_head_index] + kv_head_index * Dh
                            for kv_head_index in range(KVH)
                        ]
                    )
                    self.lora_o[layer_index].reorder_input_channels(
                        o_input_permutation
                    )
                    self.lora_qkv[layer_index].parts[2].reorder_output_channels(
                        value_output_permutation
                    )

    def _rms_norm(self, x: Tensor, scale: Tensor, eps: float) -> Tensor:
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x: Tensor) -> Tensor:
        # The sign is pre-applied to the sine table.
        x = onnx_reshape_batch(
            x, (-1, 1, self.qk_heads, 2, self.head_dim_half)
        )
        x = x.flip(-2)
        return onnx_reshape_batch(x, (-1, 1, self.qk_heads, self.head_dim))

    def forward(self, *all_inputs: Tensor) -> Tuple[Tensor, ...]:
        hidden_states  = all_inputs[-4]
        rotary_cos     = all_inputs[-3]
        rotary_sin     = all_inputs[-2]
        attention_mask = all_inputs[-1]
        # f16-storage / f32-compute (COMPUTE_IN_F32): the causal mask is kept f16 at the graph boundary (I/O
        # dtype unchanged) and upcast to f32 ONCE here, shared by every layer (principle: cast loop-invariant
        # constants once). In every other mode it is used as-is (f16 minimum-cast, or f32 for a float32 cache).
        attn_mask = attention_mask.float() if (self.use_fp16_kv and self.compute_in_f32) else attention_mask
        for i, layer in enumerate(self.llm.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)
            qkv = layer.self_attn.qkv(hidden_states)
            if self.has_lora:
                qkv = qkv + self.lora_qkv[i](hidden_states)
            qkv = onnx_reshape_batch(
                qkv, (-1, 1, self.total_qkv_heads, self.head_dim)
            )
            qk, v = torch.split(qkv, self.qkv_split_sizes, dim=-2)
            qk = self._rms_norm(qk, self.qk_norm_scale, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_cos + self._rotate_half(qk) * rotary_sin
            # Minimum-cast float16 KV attention: cast qk_rot (and V) DOWN to f16 before the split (the K/V
            # cache is f16), so Q@K, the mask Add, Softmax and attn@V all run in float16; only the context is
            # cast back to f32 for o_proj (after the dtype-agnostic Transpose/Reshape). A float32 KV cache
            # keeps the plain float32 attention.
            # COMPUTE_IN_F32: keep the f16 KV *storage* (K/V are still cast to f16 before the cache concat, so
            # the exported cache I/O dtype is unchanged) but upcast the f16 K/V to f32 at the matmul use points
            # and keep Q/mask/softmax in f32 -- i.e. f16 storage, f32 compute. Q is never downcast.
            if self.use_fp16_kv and not self.compute_in_f32:
                qk_rot = qk_rot.half()
            q, k = torch.split(qk_rot, self.qk_split_sizes, dim=-2)
            if self.use_fp16_kv:
                if self.compute_in_f32:
                    k = k.half()
                v = v.half()
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            q = onnx_reshape_batch(
                q,
                (
                    -1,
                    self.num_kv_heads,
                    self.num_kv_groups,
                    self.head_dim,
                ),
            ).permute(0, 2, 3, 1, 4)
            k = torch.cat((all_inputs[i],                   k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v
            if self.use_fp16_kv and self.compute_in_f32:
                attn = torch.matmul(q, k.float()) + attn_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v.float())
            else:
                attn = torch.matmul(q, k) + attn_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v)
            attn = onnx_reshape_batch(
                attn.permute(0, 3, 1, 2, 4),
                (-1, self.attention_output_size),
            )
            if self.use_fp16_kv and not self.compute_in_f32:
                attn = attn.float()
            attention_output = layer.self_attn.o_proj(attn)
            if self.has_lora:
                attention_output = attention_output + self.lora_o[i](attn)
            hidden_states = residual + attention_output
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            if self.has_lora:
                gate_up = gate_up + self.lora_gate_up[i](hidden_states)
            gate, up = torch.split(gate_up, self.mlp_split_sizes, dim=-1)
            mlp_hidden = layer.mlp.act_fn(gate) * up
            mlp_output = layer.mlp.down_proj(mlp_hidden)
            if self.has_lora:
                mlp_output = mlp_output + self.lora_down[i](mlp_hidden)
            hidden_states = residual + mlp_output
        # Final RMSNorm uses the learned norm weight directly (mean-based RMSNorm),
        # then the pristine tied lm_head projects to logits.
        hidden_states = self._rms_norm(hidden_states[:, -1], self.final_norm_scale, self.hidden_rms_norm_eps)
        logits = self.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoding Strategy Modules (mirrors Export_Fun_ASR_Nano.py) ────────────────
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    def forward(self, logits: Tensor, save_id: Tensor) -> Tuple[Tensor, Tensor]:
        max_idx = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)
        return max_idx, torch.cat([save_id, max_idx], dim=-1)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self):
        super().__init__()
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


class ARGMAX(torch.nn.Module):
    def forward(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


class CONCAT_EMBED(torch.nn.Module):
    def forward(self, base_embed: Tensor, language_tail_embed: Tensor) -> Tensor:
        # Empty tail: automatic language detection. Non-empty tail: the runtime
        # embeds ``language + <asr_text>`` once and supplies it here. Keeping the
        # optional tail as an input lets Encoder + Concat + Prefill form one graph
        # without an ONNX If or a separate forced-language session.
        concat_embed = torch.cat([base_embed, language_tail_embed], dim=1)
        return concat_embed, ONNX_SHAPE_DIM.apply(concat_embed, 1)


# ══════════════════════════════════════════════════════════════════════════════
# ── Export Loop ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _export_all_variants() -> None:
    if onnx_folder.exists():
        shutil.rmtree(onnx_folder)
    onnx_folder.mkdir(parents=True)
    router_path = export_audio_quality_router()
    print(f"[Router] Exported official audio-quality router -> {router_path}")
    for variant in VARIANT_NAMES:
        environment = dict(os.environ)
        environment["MEGA_ASR_EXPORT_VARIANT"] = variant
        subprocess.check_call(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(script_dir),
            env=environment,
        )
    import Shared_Merged

    deduplication = Shared_Merged.deduplicate_variant_shared_initializers(
        onnx_folder,
        VARIANT_MODEL_FILE_NAMES["base"],
        VARIANT_MODEL_FILE_NAMES["mega"],
    )
    print(
        "[SharedMerged] Reused "
        f"{deduplication['shared_initializer_count']} unchanged Mega initializer(s) "
        f"from Base ({deduplication['shared_data_bytes'] / 2**30:.2f} GiB); "
        f"saved {deduplication['saved_data_bytes'] / 2**30:.2f} GiB."
    )
    print("\nAll Mega-ASR variants exported. Running ONNX Runtime inference ...\n")
    subprocess.run(
        [
            sys.executable,
            str(script_dir / "Inference_Mega_ASR_ONNX.py"),
            "--onnx-folder",
            str(onnx_folder),
        ],
        cwd=str(script_dir),
        check=True,
    )


if _EXPORT_VARIANT is None:
    _export_all_variants()
    raise SystemExit(0)


onnx_folder.mkdir(parents=True, exist_ok=True)

print(f"\nExport start ({_EXPORT_VARIANT}) ...\n")

with torch.inference_mode():

    # ── Load model ────────────────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(download_path, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(
            download_path,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    except TypeError as error:
        if "unexpected keyword argument 'dtype'" not in str(error):
            raise
        model = AutoModel.from_pretrained(
            download_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    mega_lora_factors: MegaLoRAFactors | None = None
    if _EXPORT_VARIANT == "mega":
        print("  Loading official Mega-ASR LoRA factors for additive export ...")
        mega_lora_factors = MegaLoRAFactors.from_adapter(model, mega_lora_path)
        print(
            f"  Loaded {len(mega_lora_factors)} LoRA target module(s); "
            "base weights remain unchanged."
        )

    audio_cfg  = config.thinker_config.audio_config
    text_cfg   = config.thinker_config.text_config
    # transformers>=5 leaves __init__-computed non-persistent buffers (RoPE inv_freq,
    # sinusoidal audio positions) as uninitialized memory after from_pretrained;
    # recompute them before any graph reads their values. No-op on transformers 4.x.
    refresh_non_persistent_buffers(model, text_cfg)
    num_layers   = text_cfg.num_hidden_layers
    num_heads    = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim     = text_cfg.head_dim
    hidden_size  = text_cfg.hidden_size
    vocab_size   = text_cfg.vocab_size
    language_codes = {
        "Chinese": ("zh", ["chinese", "mandarin", "cn", "中文"]),
        "English": ("en", ["english", "eng"]),
        "Cantonese": ("yue", ["cantonese", "粤语", "廣東話", "广东话"]),
        "Arabic": ("ar", ["arabic"]),
        "German": ("de", ["german"]),
        "French": ("fr", ["french"]),
        "Spanish": ("es", ["spanish"]),
        "Portuguese": ("pt", ["portuguese"]),
        "Indonesian": ("id", ["indonesian"]),
        "Italian": ("it", ["italian"]),
        "Korean": ("ko", ["korean", "한국어"]),
        "Russian": ("ru", ["russian"]),
        "Thai": ("th", ["thai"]),
        "Vietnamese": ("vi", ["vietnamese"]),
        "Japanese": ("ja", ["japanese", "日本語"]),
        "Turkish": ("tr", ["turkish"]),
        "Hindi": ("hi", ["hindi"]),
        "Malay": ("ms", ["malay"]),
        "Dutch": ("nl", ["dutch"]),
        "Swedish": ("sv", ["swedish"]),
        "Danish": ("da", ["danish"]),
        "Finnish": ("fi", ["finnish"]),
        "Polish": ("pl", ["polish"]),
        "Czech": ("cs", ["czech"]),
        "Filipino": ("fil", ["filipino", "tagalog"]),
        "Persian": ("fa", ["persian", "farsi"]),
        "Greek": ("el", ["greek"]),
        "Romanian": ("ro", ["romanian"]),
        "Hungarian": ("hu", ["hungarian"]),
        "Macedonian": ("mk", ["macedonian"]),
    }
    configured_languages = list(config.support_languages or ())
    tokenizer_vocabulary = tokenizer.get_vocab()
    end_of_text_id = int(tokenizer_vocabulary["<|endoftext|>"])
    im_end_id = int(tokenizer_vocabulary["<|im_end|>"])
    special_token_ids = {
        "stop": [end_of_text_id, im_end_id],
        "asr_text": [int(tokenizer_vocabulary["<asr_text>"])],
        "audio_start": int(tokenizer_vocabulary["<|audio_start|>"]),
        "audio_end": int(tokenizer_vocabulary["<|audio_end|>"]),
        "audio_pad": int(tokenizer_vocabulary["<|audio_pad|>"]),
        "im_start": int(tokenizer_vocabulary["<|im_start|>"]),
        "im_end": im_end_id,
        "system": int(tokenizer.encode("system", add_special_tokens=False)[0]),
        "user": int(tokenizer.encode("user", add_special_tokens=False)[0]),
        "assistant": int(tokenizer.encode("assistant", add_special_tokens=False)[0]),
        "newline": int(tokenizer.encode("\n", add_special_tokens=False)[0]),
        "language_prefix": [
            int(token_id)
            for token_id in tokenizer.encode(_LANG_PREFIX, add_special_tokens=False)
        ],
    }
    asr_text_ids = list(special_token_ids["asr_text"])
    supported_languages = {}
    for language_name in configured_languages:
        code, aliases = language_codes[language_name]
        prompt_ids = [
            int(token_id)
            for token_id in tokenizer.encode(language_name, add_special_tokens=False)
        ] + asr_text_ids
        supported_languages[code] = {
            "name": language_name,
            "aliases": aliases,
            "prompt_token_ids": prompt_ids,
        }
    kv_dtype     = torch.float16 if USE_FP16_KV else torch.float32
    kv_specs     = [("key", 4), ("value", 3)]

    print(f"  Encoder : layers={audio_cfg.encoder_layers}, d_model={audio_cfg.d_model}, output_dim={audio_cfg.output_dim}")
    print(f"  Decoder : layers={num_layers}, heads={num_heads}/{num_kv_heads} GQA, head_dim={head_dim}")
    print(f"  KV dtype: {'float16' if USE_FP16_KV else 'float32'}")

    head_ids = [
        special_token_ids["im_start"],
        special_token_ids["system"],
        special_token_ids["newline"],
    ]
    # The assistant turn is trained to always begin with the "language " prefix.
    # Baking it into the encoder tail primes the decoder so that even in auto-detect
    # mode it reliably emits the detected language; a bare "assistant\n" context makes
    # the quantized decoder degenerate into garbage on the very first token.
    tail_ids = [
        special_token_ids["audio_end"],
        special_token_ids["im_end"],
        special_token_ids["newline"],
        special_token_ids["im_start"],
        special_token_ids["assistant"],
        special_token_ids["newline"],
    ] + list(special_token_ids["language_prefix"])

    # Trace-only values establish rank/type examples; every corresponding runtime
    # dimension or decoding value is an ONNX input and remains dynamic.
    _dummy_seq_len = 16
    _dummy_batch_size = 10
    _dummy_history_len = 10
    _dummy_penalty_value = 1.0
    _dummy_penalty_range = _dummy_history_len
    dummy_query_ids = torch.empty((1, 0), dtype=torch.int32)
    dummy_query_embed = model.thinker.model.embed_tokens(dummy_query_ids).float()

    ids_len     = torch.tensor([_dummy_seq_len], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    kv_seq_len  = ids_len + history_len
    logits      = torch.ones((_dummy_batch_size, vocab_size), dtype=torch.float32)
    save_id     = torch.zeros((_dummy_batch_size, 0), dtype=torch.int32)

    kv_tensors  = {
        "key":   torch.zeros((_dummy_batch_size, num_kv_heads, 1, head_dim, 0), dtype=kv_dtype),
        "value": torch.zeros((_dummy_batch_size, num_kv_heads, 1, 0, head_dim), dtype=kv_dtype),
    }

    query_suffix_ids = [
        special_token_ids["im_end"],
        special_token_ids["newline"],
        special_token_ids["im_start"],
        special_token_ids["user"],
        special_token_ids["newline"],
        special_token_ids["audio_start"],
    ]

    # ── Fused Audio Encoder (pre-process + encoder in one graph) ─────────────
    encoder     = QWEN3_ASR_ENCODER(
        model.thinker.audio_tower,
        model.thinker.model.embed_tokens.float(),
        head_ids,
        tail_ids,
        query_suffix_ids,
        lora_factors=mega_lora_factors,
    ).eval()
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    dummy_audio = torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)
    torch.onnx.export(
        encoder,
        (dummy_audio, dummy_query_embed),
        onnx_model_Encoder,
        input_names=["audio", "query_embed"],
        output_names=["concat_embed", "ids_len"],
        dynamic_axes={
            "audio": {2: "audio_len"},
            "query_embed": {1: "num_token"},
            "concat_embed": {1: "total_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del encoder, dummy_audio, dummy_query_ids, dummy_query_embed
    gc.collect()

    # ── Decoder Embed ─────────────────────────────────────────────────────────
    embed_mod  = QWEN3_ASR_DECODER_EMBED(model).eval()
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
    rotary_prefill = QWEN3_ASR_ROTARY_MASK_PREFILL(model.thinker.model, MAX_SEQ_LEN).eval()
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
    rotary_decode = QWEN3_ASR_ROTARY_MASK_DECODE(model.thinker.model, MAX_SEQ_LEN).eval()
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
    # The mask is added to the (minimum-cast) f16 / f32 attention scores, so it must match the KV dtype.
    attention_mask = torch.zeros((1, 1, 1, _dummy_seq_len, _dummy_seq_len),      dtype=kv_dtype)

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

    # ── Concat Embed ──────────────────────────────────────────────────────────
    concat_embed = CONCAT_EMBED().eval()
    dummy_embed_0 = torch.ones((1, 5, hidden_size), dtype=torch.float32)
    dummy_embed_1 = torch.ones((1, 3, hidden_size), dtype=torch.float32)
    torch.onnx.export(
        concat_embed,
        (dummy_embed_0, dummy_embed_1),
        onnx_model_Concat_Embed,
        input_names=["base_embed", "language_tail_embed"],
        output_names=["concat_embed", "concat_len"],
        dynamic_axes={
            "base_embed":          {1: "base_seq_len"},
            "language_tail_embed": {1: "language_tail_len"},
            "concat_embed":  {1: "total_seq_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del concat_embed, dummy_embed_0, dummy_embed_1
    gc.collect()

    # ── Decoder Main ──────────────────────────────────────────────────────────
    decoder_main = QWEN3_ASR_DECODER_MAIN(
        model,
        num_heads,
        num_kv_heads,
        head_dim,
        num_layers,
        hidden_size,
        lora_factors=mega_lora_factors,
    ).eval()
    if mega_lora_factors is not None:
        mega_lora_factors.assert_all_used()
        del mega_lora_factors
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
        GREEDY_SEARCH().eval(),
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
        ARGMAX().eval(),
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
            "sample_rate": _MODEL_SAMPLE_RATE,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
            "mega_asr_routing": {
                "base_variant": "base",
                "default_variant_when_routing_disabled": "mega",
                "mega_variant": "mega",
                "router_model": ROUTER_MODEL_NAME,
                "router_degraded_probability_output": ROUTER_DEGRADED_PROBABILITY_OUTPUT,
                "router_sample_rate": 16000,
                "router_threshold": 0.5,
                "router_waveform_input": "waveform",
                "variant_model_files": VARIANT_MODEL_FILE_NAMES,
            },
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
    # stand-alone (no external Qwen3-ASR model path needed at inference time).
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

print(f"\nExport complete ({_EXPORT_VARIANT}).\n")
