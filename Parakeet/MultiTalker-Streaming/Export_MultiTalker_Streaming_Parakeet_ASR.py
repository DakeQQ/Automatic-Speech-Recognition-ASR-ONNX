#!/usr/bin/env python
"""Export MultiTalker Parakeet and Streaming Sortformer ONNX models.

The Parakeet export is standalone and NeMo-free. When this file is run as a
script, it also exports the sibling Streaming Sortformer model and launches the
combined Sortformer-to-Parakeet ONNX Runtime workflow.
"""

import gc
import json
import math
import os
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
_SCRIPT_DIR = Path(__file__).resolve().parent
_DOWNLOADS = Path.home() / "Downloads"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from STFT_Process import STFT_Process

_MODEL_DIR_NAME = "multitalker-parakeet-streaming-0.6b-v1"
NEMO_PATH = _DOWNLOADS / _MODEL_DIR_NAME / f"{_MODEL_DIR_NAME}.nemo"
_SORTFORMER_MODEL_DIR_NAME = "diar_streaming_sortformer_4spk-v2.1"
SORTFORMER_NEMO_PATH = (
    _DOWNLOADS
    / _SORTFORMER_MODEL_DIR_NAME
    / f"{_SORTFORMER_MODEL_DIR_NAME}.nemo"
)

OPSET = 20
INPUT_AUDIO_DTYPE = "F32"
USE_FP16_STORAGE = True
SORTFORMER_INPUT_AUDIO_DTYPE = "F32"
_SORTFORMER_INPUT_AUDIO_DTYPES = {
    "F16": torch.float16,
    "F32": torch.float32,
    "INT16": torch.int16,
}
SORTFORMER_INPUT_AUDIO_DTYPE = SORTFORMER_INPUT_AUDIO_DTYPE.upper()
if SORTFORMER_INPUT_AUDIO_DTYPE not in _SORTFORMER_INPUT_AUDIO_DTYPES:
    raise ValueError(
        f"Unsupported SORTFORMER_INPUT_AUDIO_DTYPE={SORTFORMER_INPUT_AUDIO_DTYPE!r}; "
        f"expected one of {tuple(_SORTFORMER_INPUT_AUDIO_DTYPES)}."
    )
SORTFORMER_TORCH_DTYPE = _SORTFORMER_INPUT_AUDIO_DTYPES[
    SORTFORMER_INPUT_AUDIO_DTYPE
]
SORTFORMER_LENGTH_DTYPE = torch.int32
SORTFORMER_INDEX_DTYPE = torch.int32
SORTFORMER_FLAG_DTYPE = torch.int8

ONNX_FOLDER = _SCRIPT_DIR / "MultiTalker_Streaming_Parakeet_ASR_ONNX"
METADATA_NAME = "ASR_Metadata.onnx"
ENCODER_NAME = "MultiTalker_Streaming_Parakeet_ASR_Encoder.onnx"
DECODER_NAME = "MultiTalker_Streaming_Parakeet_ASR_Decoder.onnx"
SORTFORMER_ONNX_FOLDER = ONNX_FOLDER / "NVIDIA_Streaming_Sortformer_4spk"
SORTFORMER_MODEL_NAME = "NVIDIA_Streaming_Sortformer_4spk.onnx"
SORTFORMER_METADATA_NAME = "NVIDIA_Streaming_Sortformer_4spk_Metadata.onnx"
SORTFORMER_ONNX_MODEL = SORTFORMER_ONNX_FOLDER / SORTFORMER_MODEL_NAME
SORTFORMER_METADATA_MODEL = SORTFORMER_ONNX_FOLDER / SORTFORMER_METADATA_NAME

TOKENIZER_MODEL_NAME = "tokenizer.model"
TOKENIZER_VOCAB_NAME = "vocab.txt"
TOKENIZER_SPECIAL_VOCAB_NAME = "tokenizer.vocab"

# NeMo AudioToMelSpectrogramPreprocessor defaults used by this checkpoint.
PREEMPH = 0.97
LOG_GUARD = 2.0 ** -24
LN_EPS = 1e-5
AUDIO_PCM_SCALE = 32768

FLOAT_STORAGE_DTYPE = torch.float16 if USE_FP16_STORAGE else torch.float32


def _read_model_config() -> dict:
    with tarfile.open(NEMO_PATH, "r:*") as archive:
        member = next(
            item
            for item in archive.getmembers()
            if item.isfile() and Path(item.name).name == "model_config.yaml"
        )
        with archive.extractfile(member) as source:
            return yaml.safe_load(source.read())


_CFG = _read_model_config()
_PRE_CFG = _CFG["preprocessor"]
_ENC_CFG = _CFG["encoder"]
_DEC_CFG = _CFG["decoder"]
_JOINT_CFG = _CFG["joint"]

SAMPLE_RATE = int(_PRE_CFG["sample_rate"])
N_MELS = int(_PRE_CFG["features"])
N_FFT = int(_PRE_CFG["n_fft"])
WIN_LENGTH = int(round(float(_PRE_CFG["window_size"]) * SAMPLE_RATE))
HOP_LENGTH = int(round(float(_PRE_CFG["window_stride"]) * SAMPLE_RATE))

D_MODEL = int(_ENC_CFG["d_model"])
N_LAYERS = int(_ENC_CFG["n_layers"])
N_HEADS = int(_ENC_CFG["n_heads"])
HEAD_DIM = D_MODEL // N_HEADS
CONV_KERNEL = int(_ENC_CFG["conv_kernel_size"])
CONV_CACHE = CONV_KERNEL - 1
SUB_FACTOR = int(_ENC_CFG["subsampling_factor"])
SUB_CHANNELS = int(_ENC_CFG["subsampling_conv_channels"])

# The checkpoint's first configured context is its default inference mode.
ATT_CONTEXT_SIZE = tuple(int(value) for value in _ENC_CFG["att_context_size"][0])
LEFT_CONTEXT, RIGHT_CONTEXT = ATT_CONTEXT_SIZE
VALID_OUT_LEN = RIGHT_CONTEXT + 1

PRED_HIDDEN = int(_DEC_CFG["prednet"]["pred_hidden"])
LSTM_LAYERS = int(_DEC_CFG["prednet"]["pred_rnn_layers"])
JOINT_HIDDEN = int(_JOINT_CFG["jointnet"]["joint_hidden"])
VOCAB_SIZE = int(_DEC_CFG["vocab_size"])
BLANK_ID = VOCAB_SIZE
LOGITS_SIZE = VOCAB_SIZE + 1
MAX_SYMBOLS_PER_FRAME = int(_CFG["decoding"]["greedy"]["max_symbols"])

SPK_KERNEL_LAYER = int(_CFG["spk_kernel_layers"][0])
NUM_SPEAKERS = int(_CFG["model_defaults"]["num_speakers"])

# CacheAwareStreamingConfig values derived from the target causal dw-striding frontend.
PRE_ENCODE_CACHE_FRAMES = SUB_FACTOR + 1
DROP_EXTRA = 1 + (PRE_ENCODE_CACHE_FRAMES - 1) // SUB_FACTOR
STREAM_MEL_CHUNK = VALID_OUT_LEN * SUB_FACTOR
STREAM_MEL_CACHE = PRE_ENCODE_CACHE_FRAMES
STREAM_MEL_WINDOW = STREAM_MEL_CACHE + STREAM_MEL_CHUNK
STREAM_KV_LEN = LEFT_CONTEXT + VALID_OUT_LEN
STREAM_STRIDE_SAMPLES = STREAM_MEL_CHUNK * HOP_LENGTH
STREAM_LEFT_OVERLAP = (N_FFT // 2) + 1
STREAM_WINDOW_SAMPLES = (STREAM_MEL_CHUNK - 1) * HOP_LENGTH + N_FFT + 1

_AUDIO_TORCH_DTYPE = {
    "INT16": torch.int16,
    "F32": torch.float32,
    "F16": torch.float16,
}[INPUT_AUDIO_DTYPE]


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


def finalize_graph(
    raw_path: Path,
    final_path: Path,
    metadata: dict | None = None,
    *,
    replace_metadata: bool = False,
) -> None:
    """Consolidate a temporary torch export and atomically replace the final graph."""
    import onnx

    model = onnx.load(str(raw_path), load_external_data=False)
    onnx.load_external_data_for_model(model, str(raw_path.parent))
    if replace_metadata:
        del model.metadata_props[:]
    if metadata:
        existing = {prop.key: prop for prop in model.metadata_props}
        for key, value in metadata.items():
            if key in existing:
                existing[key].value = value
            else:
                model.metadata_props.add(key=key, value=value)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    data_path = final_path.with_name(final_path.name + ".data")
    with tempfile.TemporaryDirectory(
        prefix=f".{final_path.name}.",
        dir=final_path.parent,
    ) as staging_name:
        staging_dir = Path(staging_name)
        staged_model_path = staging_dir / final_path.name
        staged_data_path = staging_dir / data_path.name
        onnx.save(
            model,
            str(staged_model_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path.name,
            size_threshold=1024,
            convert_attribute=True,
        )
        onnx.checker.check_model(str(staged_model_path))

        previous_data_path = staging_dir / (data_path.name + ".previous")
        installed_data = False
        try:
            if staged_data_path.exists():
                if data_path.exists():
                    os.replace(data_path, previous_data_path)
                os.replace(staged_data_path, data_path)
                installed_data = True
            os.replace(staged_model_path, final_path)
        except Exception:
            if installed_data:
                data_path.unlink(missing_ok=True)
            if previous_data_path.exists():
                os.replace(previous_data_path, data_path)
            raise

        if not installed_data:
            data_path.unlink(missing_ok=True)


# Checkpoint asset loading
def ensure_assets(nemo_path: Path, out_dir: Path, checkpoint_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / "model_weights.ckpt"
    config_path = out_dir / "model_config.yaml"

    with tarfile.open(nemo_path, "r:*") as archive:
        members = {
            Path(member.name).name: member
            for member in archive.getmembers()
            if member.isfile()
        }

        def extract_member(member, destination):
            temporary_path = None
            try:
                with archive.extractfile(member) as source, tempfile.NamedTemporaryFile(
                    prefix=destination.name + ".",
                    suffix=".tmp",
                    dir=destination.parent,
                    delete=False,
                ) as destination_file:
                    temporary_path = Path(destination_file.name)
                    shutil.copyfileobj(source, destination_file, length=8 << 20)
                temporary_path.replace(destination)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        for source_name, destination in (
            ("model_weights.ckpt", ckpt_path),
            ("model_config.yaml", config_path),
        ):
            if not destination.exists():
                extract_member(members[source_name], destination)

        tokenizer_assets = (
            (
                next(name for name in members if name.endswith("_tokenizer.model")),
                out_dir / TOKENIZER_MODEL_NAME,
            ),
            (
                next(name for name in members if name.endswith("_vocab.txt")),
                out_dir / TOKENIZER_VOCAB_NAME,
            ),
            (
                next(name for name in members if name.endswith("_tokenizer.vocab")),
                out_dir / TOKENIZER_SPECIAL_VOCAB_NAME,
            ),
        )
        for source_name, destination in tokenizer_assets:
            if not destination.exists():
                extract_member(members[source_name], destination)

    return {
        "ckpt": ckpt_path,
        "config": config_path,
        "tokenizer_model": out_dir / TOKENIZER_MODEL_NAME,
        "tokenizer_vocab": out_dir / TOKENIZER_VOCAB_NAME,
        "tokenizer_special_vocab": out_dir / TOKENIZER_SPECIAL_VOCAB_NAME,
    }


def remove_extracted_checkpoint(assets: dict) -> None:
    ckpt_path = assets.get("ckpt")
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()
        print("Removed temporary checkpoint.")


# ONNX-friendly fused primitives
class _LAYER_NORM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, epsilon, axis):
        mean = x.mean(dim=axis, keepdim=True)
        centered = x - mean
        variance = centered.pow(2).mean(dim=axis, keepdim=True)
        output = centered * torch.rsqrt(variance + epsilon) * scale
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def symbolic(g, x, scale, bias, epsilon, axis):
        if bias is None:
            return g.op(
                "LayerNormalization",
                x,
                scale,
                axis_i=axis,
                epsilon_f=epsilon,
                stash_type_i=1,
            )
        return g.op(
            "LayerNormalization",
            x,
            scale,
            bias,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def layer_norm(x, scale, bias=None, epsilon=LN_EPS, axis=-1):
    return _LAYER_NORM.apply(x, scale, bias, float(epsilon), axis)


class _ASYMMETRIC_CONV_2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, pad_top, groups):
        return F.conv2d(
            F.pad(x, (2, 1, pad_top, 1)),
            weight,
            bias,
            stride=2,
            groups=groups,
        )

    @staticmethod
    def symbolic(g, x, weight, bias, pad_top, groups):
        return g.op(
            "Conv",
            x,
            weight,
            bias,
            dilations_i=[1, 1],
            group_i=groups,
            kernel_shape_i=[3, 3],
            pads_i=[pad_top, 2, 1, 1],
            strides_i=[2, 2],
        )


class _GEMM_RESIDUAL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, residual):
        return F.linear(x, weight) + residual

    @staticmethod
    def symbolic(g, x, weight, residual):
        return g.op(
            "Gemm",
            x,
            weight,
            residual,
            alpha_f=1.0,
            beta_f=1.0,
            transB_i=1,
        )


def gemm_residual(x, weight, residual):
    return _GEMM_RESIDUAL.apply(x, weight, residual)


class _RELATIVE_SHIFT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, index, flatten_shape, output_shape):
        return torch.index_select(value.reshape(N_HEADS, -1), 1, index).reshape(
            N_HEADS,
            VALID_OUT_LEN,
            STREAM_KV_LEN,
        )

    @staticmethod
    def symbolic(g, value, index, flatten_shape, output_shape):
        flattened = g.op("Reshape", value, flatten_shape)
        selected = g.op("Gather", flattened, index, axis_i=1)
        return g.op("Reshape", selected, output_shape)


def relative_shift(value, index, flatten_shape, output_shape):
    return _RELATIVE_SHIFT.apply(value, index, flatten_shape, output_shape)


def swish(x):
    return x * torch.sigmoid(x)


class MetadataCarrier(nn.Module):
    def forward(self, marker: Tensor) -> Tensor:
        return marker


class _SortformerSTFTPower(nn.Module):
    """Sortformer-compatible centered power-STFT without a VAD workspace import."""

    def __init__(self, n_fft: int, win_length: int, hop_len: int) -> None:
        super().__init__()
        window = torch.hann_window(win_length, periodic=False).float()
        if win_length < n_fft:
            pad_total = n_fft - win_length
            window = torch.cat(
                [
                    torch.zeros(pad_total // 2),
                    window,
                    torch.zeros(pad_total - pad_total // 2),
                ]
            )
        elif win_length > n_fft:
            start = (win_length - n_fft) // 2
            window = window[start : start + n_fft]

        half_n_fft = n_fft // 2
        support = torch.nonzero(window, as_tuple=False).flatten()
        if support.numel() == 0:
            raise ValueError("The Sortformer STFT window must contain a nonzero value.")
        kernel_start = int(support[0])
        kernel_end = int(support[-1]) + 1
        self.pad_left = half_n_fft - kernel_start
        self.pad_right = kernel_end - half_n_fft

        samples = torch.arange(kernel_start, kernel_end, dtype=torch.float32)
        frequencies = torch.arange(half_n_fft + 1, dtype=torch.float32).unsqueeze(1)
        phase = (2.0 * math.pi / n_fft) * frequencies * samples.unsqueeze(0)
        window = window[kernel_start:kernel_end].unsqueeze(0)
        self.register_buffer(
            "kernel",
            torch.cat(
                [
                    (torch.cos(phase) * window).unsqueeze(1),
                    (-torch.sin(phase) * window).unsqueeze(1),
                ],
                dim=0,
            ),
            persistent=True,
        )
        self.register_buffer(
            "padding_left",
            torch.zeros(1, 1, self.pad_left),
            persistent=True,
        )
        self.register_buffer(
            "padding_right",
            torch.zeros(1, 1, self.pad_right),
            persistent=True,
        )
        self.half_n_fft = half_n_fft
        self.hop_len = hop_len

    def forward(self, audio: Tensor) -> Tensor:
        padding_left = self.padding_left
        padding_right = self.padding_right
        if audio.shape[0] != 1:
            padding_left = torch.cat([padding_left] * audio.shape[0], dim=0)
            padding_right = torch.cat([padding_right] * audio.shape[0], dim=0)
        packed = F.conv1d(
            torch.cat([padding_left, audio, padding_right], dim=2),
            self.kernel,
            stride=self.hop_len,
        )
        packed = packed * packed
        real_power, imaginary_power = torch.split(
            packed,
            self.half_n_fft + 1,
            dim=1,
        )
        return real_power + imaginary_power


def _load_sortformer_model(path: Path):
    """Load the Sortformer checkpoint with the compatibility shims it requires."""
    import lightning.pytorch.loggers as lightning_loggers

    if not hasattr(lightning_loggers, "NeptuneLogger"):

        class _UnavailableNeptuneLogger:
            pass

        lightning_loggers.NeptuneLogger = _UnavailableNeptuneLogger

    from lightning.pytorch import Trainer as LightningTrainer

    if LightningTrainer.save_checkpoint.__annotations__.get("weights_only") == bool | None:
        LightningTrainer.save_checkpoint.__annotations__["weights_only"] = bool

    from nemo.collections.asr.models import SortformerEncLabelModel

    return SortformerEncLabelModel.restore_from(
        str(path),
        map_location="cpu",
        strict=False,
    ).float().eval()


class SortformerStreamingAudioFrontend(nn.Module):
    """Checkpoint-matched waveform frontend for the embedded Sortformer export."""

    def __init__(self, preprocessor, output_feature_frames: int) -> None:
        super().__init__()
        featurizer = preprocessor.featurizer
        self.stft = _SortformerSTFTPower(
            n_fft=int(featurizer.n_fft),
            win_length=int(featurizer.win_length),
            hop_len=int(featurizer.hop_length),
        )
        self.register_buffer(
            "mel_kernel",
            featurizer.fb.detach().squeeze(0).float().unsqueeze(2),
            persistent=True,
        )
        self.register_buffer(
            "preemphasis",
            torch.tensor(float(featurizer.preemph), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "log_guard",
            torch.tensor(
                float(featurizer.log_zero_guard_value),
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "input_scale",
            torch.tensor(
                1.0 / 32768.0
                if SORTFORMER_INPUT_AUDIO_DTYPE == "INT16"
                else 1.0,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.hop_length = int(featurizer.hop_length)
        self.output_feature_frames = output_feature_frames
        self.register_buffer(
            "frame_positions",
            torch.arange(
                output_feature_frames,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )

    def forward(
        self,
        audio_with_context: Tensor,
        audio_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        audio_with_context = audio_with_context.to(torch.float32) * self.input_scale
        previous_sample = audio_with_context[:, :, :1]
        audio = audio_with_context[:, :, 1:]
        preemphasized = torch.cat(
            (
                audio[:, :, :1] - self.preemphasis * previous_sample,
                audio[:, :, 1:] - self.preemphasis * audio[:, :, :-1],
            ),
            dim=2,
        )
        power = self.stft(preemphasized)
        features = torch.log(
            F.conv1d(power, self.mel_kernel) + self.log_guard
        )[:, :, : self.output_feature_frames]
        feature_lengths = torch.div(
            audio_lengths,
            self.hop_length,
            rounding_mode="floor",
        ).to(SORTFORMER_LENGTH_DTYPE).clamp(max=self.output_feature_frames)
        valid = self.frame_positions < feature_lengths.unsqueeze(1)
        features = features * valid.unsqueeze(1).to(features.dtype)
        return features, feature_lengths


class SortformerPackedStreamingNeural(nn.Module):
    """Neural Sortformer step with the checkpoint's packed cache layout."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        modules = model.sortformer_modules
        self.speaker_cache_capacity = int(modules.spkcache_len)
        self.fifo_capacity = int(modules.fifo_len)
        self.output_activity_frames = int(modules.chunk_len)
        feature_cache_frames = int(modules.subsampling_factor) + 1
        self.pre_encode_drop_frames = (
            feature_cache_frames + int(modules.subsampling_factor) - 1
        ) // int(modules.subsampling_factor)
        self.register_buffer(
            "packed_positions",
            torch.arange(
                self.speaker_cache_capacity
                + self.fifo_capacity
                + self.output_activity_frames,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )

    @staticmethod
    def _gather_rows(values: Tensor, indices: Tensor) -> Tensor:
        return values[0].index_select(0, indices[0]).unsqueeze(0)

    def forward(
        self,
        chunk: Tensor,
        chunk_lengths: Tensor,
        spkcache: Tensor,
        spkcache_lengths: Tensor,
        fifo: Tensor,
        fifo_lengths: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        chunk_embs, chunk_lengths = self.model.encoder.pre_encode(
            x=chunk,
            lengths=chunk_lengths,
        )
        chunk_embs = chunk_embs[:, self.pre_encode_drop_frames :, :]
        chunk_lengths = (
            chunk_lengths - self.pre_encode_drop_frames
        ).clamp(min=0).to(SORTFORMER_LENGTH_DTYPE)

        source = torch.cat((spkcache, fifo, chunk_embs), dim=1)
        positions = self.packed_positions
        cache_lengths = spkcache_lengths.unsqueeze(1)
        fifo_lengths_expanded = fifo_lengths.unsqueeze(1)
        source_indices = torch.where(
            positions < cache_lengths,
            positions,
            torch.where(
                positions < cache_lengths + fifo_lengths_expanded,
                self.speaker_cache_capacity + positions - cache_lengths,
                self.speaker_cache_capacity
                + self.fifo_capacity
                + positions
                - cache_lengths
                - fifo_lengths_expanded,
            ),
        ).clamp(min=0, max=source.shape[1] - 1)
        packed = self._gather_rows(source, source_indices)
        packed_lengths = spkcache_lengths + fifo_lengths + chunk_lengths
        packed = packed * (
            positions < packed_lengths.unsqueeze(1)
        ).unsqueeze(2).to(packed.dtype)

        frontend_embs, frontend_lengths = self.model.frontend_encoder(
            processed_signal=packed,
            processed_signal_length=packed_lengths,
            bypass_pre_encode=True,
        )
        probabilities = self.model.forward_infer(frontend_embs, frontend_lengths)
        return probabilities, chunk_embs, chunk_lengths


class SortformerFixedStreamingState(nn.Module):
    """Tensor-only fixed-capacity equivalent of Sortformer's cache update."""

    def __init__(self, modules) -> None:
        super().__init__()
        self.speaker_cache_capacity = int(modules.spkcache_len)
        self.fifo_capacity = int(modules.fifo_len)
        self.output_activity_frames = int(modules.chunk_len)
        self.speaker_cache_update_period = int(modules.spkcache_update_period)
        self.speaker_cache_silence_frames = int(
            modules.spkcache_sil_frames_per_spk
        )
        self.speaker_count = int(modules.n_spk)
        self.export_batch_size = 1
        self.prediction_score_threshold = float(modules.pred_score_threshold)
        self.silence_threshold = float(modules.sil_threshold)
        self.fifo_candidate_capacity = (
            self.fifo_capacity + self.output_activity_frames
        )
        self.cache_candidate_capacity = (
            self.speaker_cache_capacity + self.speaker_cache_update_period
        )
        self.cache_score_capacity = (
            self.cache_candidate_capacity + self.speaker_cache_silence_frames
        )
        cache_rows_per_speaker = (
            self.speaker_cache_capacity // self.speaker_count
            - self.speaker_cache_silence_frames
        )
        self.strong_boost_frames = int(
            cache_rows_per_speaker * float(modules.strong_boost_rate)
        )
        self.weak_boost_frames = int(
            cache_rows_per_speaker * float(modules.weak_boost_rate)
        )
        self.min_positive_score_frames = int(
            cache_rows_per_speaker * float(modules.min_pos_scores_rate)
        )
        self.register_buffer(
            "activity_positions",
            torch.arange(
                self.output_activity_frames,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "fifo_positions",
            torch.arange(
                self.fifo_candidate_capacity,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "fifo_state_positions",
            torch.arange(
                self.fifo_capacity,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "cache_positions",
            torch.arange(
                self.cache_candidate_capacity,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "speaker_cache_positions",
            torch.arange(
                self.speaker_cache_capacity,
                dtype=SORTFORMER_INDEX_DTYPE,
            ).unsqueeze(0),
            persistent=True,
        )
        self.register_buffer(
            "latest_cache_positions",
            (
                torch.arange(
                    self.cache_candidate_capacity,
                    dtype=SORTFORMER_INDEX_DTYPE,
                )
                >= self.speaker_cache_capacity
            ).view(1, -1, 1),
            persistent=True,
        )
        self.register_buffer(
            "silence_scores",
            torch.full(
                (
                    1,
                    self.speaker_cache_silence_frames,
                    self.speaker_count,
                ),
                torch.inf,
            ),
            persistent=True,
        )
        self.register_buffer(
            "log_two",
            torch.tensor(0.6931471805599453, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "latest_score_boost",
            torch.tensor(float(modules.scores_boost_latest), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "strong_boost_values",
            torch.full(
                (
                    self.export_batch_size,
                    self.strong_boost_frames,
                    self.speaker_count,
                ),
                1.3862943611198906,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "weak_boost_values",
            torch.full(
                (
                    self.export_batch_size,
                    self.weak_boost_frames,
                    self.speaker_count,
                ),
                0.6931471805599453,
                dtype=torch.float32,
            ),
            persistent=True,
        )
        self.register_buffer(
            "placeholder_indices",
            torch.full(
                (self.export_batch_size, self.speaker_cache_capacity),
                99_999,
                dtype=SORTFORMER_INDEX_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "zero_indices",
            torch.zeros(
                self.export_batch_size,
                self.speaker_cache_capacity,
                dtype=SORTFORMER_INDEX_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "zero_speaker_predictions",
            torch.zeros(
                self.export_batch_size,
                self.speaker_cache_capacity,
                self.speaker_count,
            ),
            persistent=True,
        )
        self.register_buffer(
            "scheduled_pop_lengths",
            torch.full(
                (self.export_batch_size,),
                self.speaker_cache_update_period,
                dtype=SORTFORMER_LENGTH_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "zero_lengths",
            torch.zeros(
                self.export_batch_size,
                dtype=SORTFORMER_LENGTH_DTYPE,
            ),
            persistent=True,
        )
        self.register_buffer(
            "one_has_preds",
            torch.ones(
                self.export_batch_size,
                dtype=SORTFORMER_LENGTH_DTYPE,
            ),
            persistent=True,
        )

    @staticmethod
    def _gather_rows(values: Tensor, indices: Tensor) -> Tensor:
        return values[0].index_select(0, indices[0]).unsqueeze(0)

    def _boost_topk(
        self,
        scores: Tensor,
        frame_count: int,
        boost_values: Tensor,
    ) -> Tensor:
        _, indices = torch.topk(
            scores,
            frame_count,
            dim=1,
            largest=True,
            sorted=False,
        )
        indices = indices.to(SORTFORMER_INDEX_DTYPE)
        return scores.scatter_add(1, indices, boost_values)

    def _compress_cache(
        self,
        embeddings: Tensor,
        predictions: Tensor,
        cache_lengths: Tensor,
        mean_sil_emb: Tensor,
    ) -> tuple[Tensor, Tensor]:
        log_probs = torch.log(
            torch.clamp(predictions, min=self.prediction_score_threshold)
        )
        log_inverse_probs = torch.log(
            torch.clamp(
                1.0 - predictions,
                min=self.prediction_score_threshold,
            )
        )
        scores = (
            log_probs
            - log_inverse_probs
            + log_inverse_probs.sum(dim=2, keepdim=True)
            + self.log_two
        )
        speech = predictions > 0.5
        scores = torch.where(speech, scores, torch.full_like(scores, -torch.inf))
        positive = scores > 0.0
        replace_nonpositive = (
            (~positive)
            & speech
            & (
                positive.sum(dim=1, keepdim=True)
                >= self.min_positive_score_frames
            )
        )
        scores = torch.where(
            replace_nonpositive,
            torch.full_like(scores, -torch.inf),
            scores,
        )
        valid_cache = self.cache_positions < cache_lengths.unsqueeze(1)
        scores = torch.where(
            valid_cache.unsqueeze(2),
            scores,
            torch.full_like(scores, -torch.inf),
        )
        scores = torch.where(
            self.latest_cache_positions,
            scores + self.latest_score_boost,
            scores,
        )
        scores = self._boost_topk(
            scores,
            self.strong_boost_frames,
            self.strong_boost_values,
        )
        scores = self._boost_topk(
            scores,
            self.weak_boost_frames,
            self.weak_boost_values,
        )
        scores = torch.cat((scores, self.silence_scores), dim=1)
        flattened = scores.permute(0, 2, 1).reshape(self.export_batch_size, -1)
        values, indices = torch.topk(
            flattened,
            self.speaker_cache_capacity,
            dim=1,
            largest=True,
            sorted=False,
        )
        indices = indices.to(SORTFORMER_INDEX_DTYPE)
        indices = torch.where(
            values == -torch.inf,
            self.placeholder_indices,
            indices,
        )
        indices, _ = torch.sort(indices, dim=1)
        is_disabled = indices == 99_999
        indices = torch.remainder(indices, self.cache_score_capacity)
        is_disabled = is_disabled | (indices >= self.cache_candidate_capacity)
        indices = torch.where(is_disabled, self.zero_indices, indices)

        compressed_embeddings = self._gather_rows(embeddings, indices)
        compressed_embeddings = torch.where(
            is_disabled.unsqueeze(2),
            mean_sil_emb.unsqueeze(1),
            compressed_embeddings,
        )
        compressed_predictions = self._gather_rows(predictions, indices)
        compressed_predictions = torch.where(
            is_disabled.unsqueeze(2),
            self.zero_speaker_predictions,
            compressed_predictions,
        )
        return compressed_embeddings, compressed_predictions

    def forward(
        self,
        probabilities: Tensor,
        chunk_embeddings: Tensor,
        chunk_lengths: Tensor,
        spkcache: Tensor,
        spkcache_lengths: Tensor,
        spkcache_preds: Tensor,
        spkcache_has_preds: Tensor,
        fifo: Tensor,
        fifo_lengths: Tensor,
        mean_sil_emb: Tensor,
        n_sil_frames: Tensor,
    ) -> tuple[Tensor, ...]:
        activity_indices = (
            spkcache_lengths.unsqueeze(1)
            + fifo_lengths.unsqueeze(1)
            + self.activity_positions
        ).clamp(
            max=(
                self.speaker_cache_capacity
                + self.fifo_capacity
                + self.output_activity_frames
                - 1
            )
        )
        activity = self._gather_rows(probabilities, activity_indices)
        activity = activity * (
            self.activity_positions < chunk_lengths.unsqueeze(1)
        ).unsqueeze(2).to(activity.dtype)

        fifo_indices = torch.where(
            self.fifo_positions < fifo_lengths.unsqueeze(1),
            self.fifo_positions,
            self.fifo_capacity
            + self.fifo_positions
            - fifo_lengths.unsqueeze(1),
        ).clamp(max=self.fifo_candidate_capacity - 1)
        fifo_candidate = self._gather_rows(
            torch.cat((fifo, chunk_embeddings), dim=1),
            fifo_indices,
        )
        total_fifo_lengths = fifo_lengths + chunk_lengths
        fifo_candidate = fifo_candidate * (
            self.fifo_positions < total_fifo_lengths.unsqueeze(1)
        ).unsqueeze(2).to(fifo_candidate.dtype)
        fifo_prediction_indices = (
            spkcache_lengths.unsqueeze(1) + self.fifo_positions
        ).clamp(
            max=(
                self.speaker_cache_capacity
                + self.fifo_capacity
                + self.output_activity_frames
                - 1
            )
        )
        fifo_predictions = self._gather_rows(probabilities, fifo_prediction_indices)
        fifo_predictions = fifo_predictions * (
            self.fifo_positions < total_fifo_lengths.unsqueeze(1)
        ).unsqueeze(2).to(fifo_predictions.dtype)

        cache_predictions_from_model = self._gather_rows(
            probabilities,
            self.speaker_cache_positions,
        )
        cache_predictions_from_model = cache_predictions_from_model * (
            self.speaker_cache_positions < spkcache_lengths.unsqueeze(1)
        ).unsqueeze(2).to(cache_predictions_from_model.dtype)
        cache_predictions = torch.where(
            (spkcache_has_preds > 0).reshape(1, 1, 1),
            spkcache_preds,
            cache_predictions_from_model,
        )

        overflow = total_fifo_lengths - self.fifo_capacity
        scheduled_pop_lengths = torch.maximum(
            self.scheduled_pop_lengths,
            overflow,
        )
        pop_lengths = torch.minimum(scheduled_pop_lengths, total_fifo_lengths)
        pop_lengths = torch.where(
            total_fifo_lengths > self.fifo_capacity,
            pop_lengths,
            self.zero_lengths,
        )
        next_fifo_lengths = total_fifo_lengths - pop_lengths
        next_fifo_indices = (
            self.fifo_state_positions + pop_lengths.unsqueeze(1)
        ).clamp(max=self.fifo_candidate_capacity - 1)
        next_fifo = self._gather_rows(fifo_candidate, next_fifo_indices)
        next_fifo = next_fifo * (
            self.fifo_state_positions < next_fifo_lengths.unsqueeze(1)
        ).unsqueeze(2).to(next_fifo.dtype)

        cache_candidate_indices = torch.where(
            self.cache_positions < spkcache_lengths.unsqueeze(1),
            self.cache_positions,
            self.speaker_cache_capacity
            + self.cache_positions
            - spkcache_lengths.unsqueeze(1),
        ).clamp(
            max=self.speaker_cache_capacity + self.fifo_candidate_capacity - 1
        )
        cache_candidate = self._gather_rows(
            torch.cat((spkcache, fifo_candidate), dim=1),
            cache_candidate_indices,
        )
        cache_prediction_candidate = self._gather_rows(
            torch.cat((cache_predictions, fifo_predictions), dim=1),
            cache_candidate_indices,
        )
        cache_candidate_lengths = spkcache_lengths + pop_lengths
        cache_candidate = cache_candidate * (
            self.cache_positions < cache_candidate_lengths.unsqueeze(1)
        ).unsqueeze(2).to(cache_candidate.dtype)
        cache_prediction_candidate = cache_prediction_candidate * (
            self.cache_positions < cache_candidate_lengths.unsqueeze(1)
        ).unsqueeze(2).to(cache_prediction_candidate.dtype)

        silence = (
            fifo_predictions.sum(dim=2) < self.silence_threshold
        ) & (self.fifo_positions < pop_lengths.unsqueeze(1))
        silence_count = silence.sum(dim=1).to(SORTFORMER_LENGTH_DTYPE)
        silence_embedding_sum = (
            fifo_candidate * silence.unsqueeze(2).to(fifo_candidate.dtype)
        ).sum(dim=1)
        next_n_sil_frames = n_sil_frames + silence_count
        updated_mean_sil_emb = (
            mean_sil_emb * n_sil_frames.unsqueeze(1) + silence_embedding_sum
        ) / torch.clamp(next_n_sil_frames.unsqueeze(1), min=1)
        next_mean_sil_emb = torch.where(
            (silence_count > 0).unsqueeze(1),
            updated_mean_sil_emb,
            mean_sil_emb,
        )

        needs_compression = cache_candidate_lengths > self.speaker_cache_capacity
        compressed_cache, compressed_cache_preds = self._compress_cache(
            cache_candidate,
            cache_prediction_candidate,
            cache_candidate_lengths,
            next_mean_sil_emb,
        )
        next_spkcache = torch.where(
            needs_compression.reshape(1, 1, 1),
            compressed_cache,
            cache_candidate[:, : self.speaker_cache_capacity, :],
        )
        next_spkcache_preds = torch.where(
            needs_compression.reshape(1, 1, 1),
            compressed_cache_preds,
            cache_prediction_candidate[:, : self.speaker_cache_capacity, :],
        )
        next_spkcache_lengths = cache_candidate_lengths.clamp(
            max=self.speaker_cache_capacity
        )
        next_spkcache_has_preds = torch.where(
            needs_compression,
            self.one_has_preds,
            spkcache_has_preds.to(SORTFORMER_LENGTH_DTYPE),
        ).to(SORTFORMER_FLAG_DTYPE)
        return (
            activity,
            next_spkcache,
            next_spkcache_lengths,
            next_spkcache_preds,
            next_spkcache_has_preds,
            next_fifo,
            next_fifo_lengths,
            next_mean_sil_emb,
            next_n_sil_frames,
        )


class SortformerStreamingPipeline(nn.Module):
    """Static waveform-to-activity Sortformer step with ONNX-visible state."""

    def __init__(self, model) -> None:
        super().__init__()
        modules = model.sortformer_modules
        self.sample_rate = int(model.preprocessor.featurizer.sample_rate)
        self.feature_cache_frames = int(modules.subsampling_factor) + 1
        self.new_feature_frames = (
            int(modules.chunk_len) * int(modules.subsampling_factor)
        )
        self.input_feature_frames = (
            self.feature_cache_frames + self.new_feature_frames
        )
        hop_length = int(model.preprocessor.featurizer.hop_length)
        self.audio_cache_samples = self.feature_cache_frames * hop_length
        self.new_audio_samples = self.new_feature_frames * hop_length
        self.audio_chunk_samples = self.audio_cache_samples + self.new_audio_samples
        self.audio_input_samples = self.audio_chunk_samples + 1
        self.speaker_cache_capacity = int(modules.spkcache_len)
        self.fifo_capacity = int(modules.fifo_len)
        self.embedding_dimension = int(modules.fc_d_model)
        self.speaker_count = int(modules.n_spk)
        self.output_activity_frames = int(modules.chunk_len)
        self.frontend = SortformerStreamingAudioFrontend(
            model.preprocessor,
            self.input_feature_frames,
        )
        self.neural = SortformerPackedStreamingNeural(model)
        self.state = SortformerFixedStreamingState(modules)

    def forward(
        self,
        audio: Tensor,
        audio_lengths: Tensor,
        activity_threshold: Tensor,
        spkcache: Tensor,
        spkcache_lengths: Tensor,
        spkcache_preds: Tensor,
        spkcache_has_preds: Tensor,
        fifo: Tensor,
        fifo_lengths: Tensor,
        mean_sil_emb: Tensor,
        n_sil_frames: Tensor,
        active_frame_counts: Tensor,
        overlap_frame_count: Tensor,
    ) -> tuple[Tensor, ...]:
        features, chunk_lengths = self.frontend(audio, audio_lengths)
        chunk = features.transpose(1, 2)
        probabilities, chunk_embeddings, chunk_lengths = self.neural(
            chunk,
            chunk_lengths,
            spkcache,
            spkcache_lengths,
            fifo,
            fifo_lengths,
        )
        (
            activity,
            next_spkcache,
            next_spkcache_lengths,
            next_spkcache_preds,
            next_spkcache_has_preds,
            next_fifo,
            next_fifo_lengths,
            next_mean_sil_emb,
            next_n_sil_frames,
        ) = self.state(
            probabilities,
            chunk_embeddings,
            chunk_lengths,
            spkcache,
            spkcache_lengths,
            spkcache_preds,
            spkcache_has_preds,
            fifo,
            fifo_lengths,
            mean_sil_emb,
            n_sil_frames,
        )
        active_frames = activity > activity_threshold
        next_active_frame_counts = active_frame_counts + active_frames.to(
            SORTFORMER_LENGTH_DTYPE
        ).sum(dim=1, dtype=SORTFORMER_LENGTH_DTYPE)
        overlap_frames = active_frames.to(SORTFORMER_LENGTH_DTYPE).sum(
            dim=2,
            dtype=SORTFORMER_LENGTH_DTYPE,
        ) > 1
        next_overlap_frame_count = overlap_frame_count + overlap_frames.to(
            SORTFORMER_LENGTH_DTYPE
        ).sum(dim=1, dtype=SORTFORMER_LENGTH_DTYPE)
        return (
            activity,
            next_spkcache,
            next_spkcache_lengths,
            next_spkcache_preds,
            next_spkcache_has_preds,
            next_fifo,
            next_fifo_lengths,
            next_mean_sil_emb,
            next_n_sil_frames,
            next_active_frame_counts,
            next_overlap_frame_count,
        )


class MultiTalkerStreamingParakeetBackbone(nn.Module):
    """Checkpoint-derived FastConformer weights and speaker-kernel modules."""

    def __init__(self, state_dict: dict):
        super().__init__()
        get = lambda key: state_dict[key].float()

        self.register_buffer("c0_w", get("encoder.pre_encode.conv.0.weight"), persistent=True)
        self.register_buffer("c0_b", get("encoder.pre_encode.conv.0.bias"), persistent=True)
        self.register_buffer("c2_w", get("encoder.pre_encode.conv.2.weight"), persistent=True)
        self.register_buffer("c2_b", get("encoder.pre_encode.conv.2.bias"), persistent=True)
        self.register_buffer("c3_w", get("encoder.pre_encode.conv.3.weight"), persistent=True)
        self.register_buffer("c3_b", get("encoder.pre_encode.conv.3.bias"), persistent=True)
        self.register_buffer("c5_w", get("encoder.pre_encode.conv.5.weight"), persistent=True)
        self.register_buffer("c5_b", get("encoder.pre_encode.conv.5.bias"), persistent=True)
        self.register_buffer("c6_w", get("encoder.pre_encode.conv.6.weight"), persistent=True)
        self.register_buffer("c6_b", get("encoder.pre_encode.conv.6.bias"), persistent=True)
        self.register_buffer(
            "pre_encode_out_w",
            get("encoder.pre_encode.out.weight"),
            persistent=True,
        )
        self.register_buffer(
            "pre_encode_out_b",
            get("encoder.pre_encode.out.bias"),
            persistent=True,
        )

        window = get("preprocessor.featurizer.window")
        filterbank = get("preprocessor.featurizer.fb").squeeze(0).contiguous()
        self.stft = STFT_Process(
            "stft_B",
            n_fft=N_FFT,
            win_length=WIN_LENGTH,
            hop_len=HOP_LENGTH,
            max_frames=STREAM_MEL_CHUNK,
            window_type=str(_PRE_CFG["window"]),
            center_pad=False,
        )
        pad_total = N_FFT - WIN_LENGTH
        pad_left = pad_total // 2
        padded_window = torch.cat(
            [torch.zeros(pad_left), window, torch.zeros(pad_total - pad_left)]
        )
        frequencies = torch.arange(N_FFT // 2 + 1, dtype=torch.float32).unsqueeze(1)
        samples = torch.arange(N_FFT, dtype=torch.float32).unsqueeze(0)
        omega = (2.0 * math.pi / N_FFT) * frequencies * samples
        self.stft.stft_kernel = torch.cat(
            [
                (torch.cos(omega) * padded_window.unsqueeze(0)).unsqueeze(1),
                (-torch.sin(omega) * padded_window.unsqueeze(0)).unsqueeze(1),
            ],
            dim=0,
        )
        if INPUT_AUDIO_DTYPE == "INT16":
            self.stft.stft_kernel.mul_(1.0 / AUDIO_PCM_SCALE)
        self.register_buffer("filterbank", filterbank, persistent=True)
        self.register_buffer(
            "preemph",
            torch.tensor(PREEMPH, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "log_guard",
            torch.tensor(LOG_GUARD, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer("ln_ones", torch.ones(D_MODEL), persistent=True)
        positions = torch.arange(
            STREAM_KV_LEN - 1,
            -STREAM_KV_LEN,
            -1,
            dtype=torch.float32,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, D_MODEL, 2, dtype=torch.float32)
            * -(math.log(10000.0) / D_MODEL)
        )
        positional_encoding = torch.zeros(positions.shape[0], D_MODEL)
        positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        positional_encoding[:, 1::2] = torch.cos(positions * div_term)
        self.pe_center = STREAM_KV_LEN

        inv_sqrt_head = HEAD_DIM ** -0.5
        for layer_index in range(N_LAYERS):
            prefix = f"encoder.layers.{layer_index}."

            def norm(name):
                return get(prefix + name + ".weight"), get(prefix + name + ".bias")

            norm_weight, norm_bias = norm("norm_feed_forward1")
            linear1_weight = get(prefix + "feed_forward1.linear1.weight")
            linear2_weight = get(prefix + "feed_forward1.linear2.weight")
            self.register_buffer(
                f"ff1_l1w_{layer_index}",
                (linear1_weight * norm_weight.unsqueeze(0)).contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"ff1_l1b_{layer_index}",
                torch.matmul(linear1_weight, norm_bias),
                persistent=True,
            )
            self.register_buffer(
                f"ff1_l2w_{layer_index}",
                (linear2_weight * 0.5).contiguous(),
                persistent=True,
            )

            norm_weight, norm_bias = norm("norm_self_att")
            q_weight = get(prefix + "self_attn.linear_q.weight")
            k_weight = get(prefix + "self_attn.linear_k.weight")
            v_weight = get(prefix + "self_attn.linear_v.weight")
            qkv_weight_original = torch.cat([q_weight, k_weight, v_weight], dim=0)
            qkv_bias = torch.matmul(qkv_weight_original, norm_bias).clone()
            qkv_weight = (qkv_weight_original * norm_weight.unsqueeze(0)).clone()
            qkv_weight[:D_MODEL] *= inv_sqrt_head
            qkv_bias[:D_MODEL] *= inv_sqrt_head
            self.register_buffer(
                f"qkv_w_{layer_index}",
                qkv_weight.contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"qkv_b_{layer_index}",
                qkv_bias.contiguous(),
                persistent=True,
            )
            position_projection = F.linear(
                positional_encoding,
                get(prefix + "self_attn.linear_pos.weight"),
            ).reshape(-1, N_HEADS, HEAD_DIM)
            position_projection = (
                position_projection.permute(1, 2, 0)
                .contiguous()
                .to(FLOAT_STORAGE_DTYPE)
            )
            self.register_buffer(
                f"pos_proj_{layer_index}",
                position_projection,
                persistent=True,
            )
            self.register_buffer(
                f"bias_u_{layer_index}",
                (
                    get(prefix + "self_attn.pos_bias_u") * inv_sqrt_head
                ).unsqueeze(1).contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"bias_v_{layer_index}",
                (
                    get(prefix + "self_attn.pos_bias_v") * inv_sqrt_head
                ).unsqueeze(1).contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"attn_out_w_{layer_index}",
                get(prefix + "self_attn.linear_out.weight").contiguous(),
                persistent=True,
            )

            norm_weight, norm_bias = norm("norm_conv")
            pointwise1_weight = get(prefix + "conv.pointwise_conv1.weight").squeeze(-1)
            self.register_buffer(
                f"pw1_w_{layer_index}",
                (pointwise1_weight * norm_weight.unsqueeze(0)).contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"pw1_b_{layer_index}",
                torch.matmul(pointwise1_weight, norm_bias),
                persistent=True,
            )
            self.register_buffer(
                f"dw_w_{layer_index}",
                get(prefix + "conv.depthwise_conv.weight").contiguous(),
                persistent=True,
            )
            batch_norm_weight, batch_norm_bias = norm("conv.batch_norm")
            self.register_buffer(
                f"conv_norm_w_{layer_index}",
                batch_norm_weight,
                persistent=True,
            )
            self.register_buffer(
                f"conv_norm_b_{layer_index}",
                batch_norm_bias,
                persistent=True,
            )
            self.register_buffer(
                f"pw2_w_{layer_index}",
                get(prefix + "conv.pointwise_conv2.weight").squeeze(-1).contiguous(),
                persistent=True,
            )

            norm_weight, norm_bias = norm("norm_feed_forward2")
            linear1_weight = get(prefix + "feed_forward2.linear1.weight")
            linear2_weight = get(prefix + "feed_forward2.linear2.weight")
            self.register_buffer(
                f"ff2_l1w_{layer_index}",
                (linear1_weight * norm_weight.unsqueeze(0)).contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"ff2_l1b_{layer_index}",
                torch.matmul(linear1_weight, norm_bias),
                persistent=True,
            )
            self.register_buffer(
                f"ff2_l2w_{layer_index}",
                (linear2_weight * 0.5).contiguous(),
                persistent=True,
            )

            output_weight, output_bias = norm("norm_out")
            self.register_buffer(
                f"norm_out_w_{layer_index}",
                output_weight,
                persistent=True,
            )
            self.register_buffer(
                f"norm_out_b_{layer_index}",
                output_bias,
                persistent=True,
            )

        kernel_prefix = f"spk_kernels.{SPK_KERNEL_LAYER}"
        background_kernel_prefix = f"bg_spk_kernels.{SPK_KERNEL_LAYER}"
        for name, prefix in (("spk", kernel_prefix), ("bg_spk", background_kernel_prefix)):
            self.register_buffer(
                f"{name}_w0",
                get(prefix + ".0.weight").contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"{name}_b0",
                get(prefix + ".0.bias").contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"{name}_w3",
                get(prefix + ".3.weight").contiguous(),
                persistent=True,
            )
            self.register_buffer(
                f"{name}_b3",
                get(prefix + ".3.bias").contiguous(),
                persistent=True,
            )

        self.register_buffer(
            "enc_proj_w",
            get("joint.enc.weight").contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "enc_proj_b",
            get("joint.enc.bias").contiguous(),
            persistent=True,
        )

    def _causal_conv2d(self, x, weight, bias, groups, pad_top=2):
        return _ASYMMETRIC_CONV_2D.apply(x, weight, bias, pad_top, groups)

    def pre_encode(self, features, output_frames):
        x = features.transpose(1, 2).unsqueeze(1)
        x = F.relu(self._causal_conv2d(x, self.c0_w, self.c0_b, 1))
        x = self._causal_conv2d(x, self.c2_w, self.c2_b, SUB_CHANNELS)
        x = F.relu(F.conv2d(x, self.c3_w, self.c3_b))
        x = self._causal_conv2d(x, self.c5_w, self.c5_b, SUB_CHANNELS)
        x = F.relu(F.conv2d(x, self.c6_w, self.c6_b))
        x = x.transpose(1, 2).reshape(
            1,
            output_frames,
            self.pre_encode_out_w.shape[1],
        )
        return F.linear(x, self.pre_encode_out_w, self.pre_encode_out_b)

    def speaker_kernel(self, x, prefix):
        return F.linear(
            F.relu(
                F.linear(
                    x,
                    getattr(self, f"{prefix}_w0"),
                    getattr(self, f"{prefix}_b0"),
                )
            ),
            getattr(self, f"{prefix}_w3"),
            getattr(self, f"{prefix}_b3"),
        )


class MultiTalkerStreamingParakeetEncoder(nn.Module):
    """Fixed-window cache-aware FastConformer encoder for one target speaker."""

    def __init__(self, backbone: MultiTalkerStreamingParakeetBackbone):
        super().__init__()
        self.backbone = backbone

        key_index = torch.arange(STREAM_KV_LEN, dtype=torch.int16).unsqueeze(0)
        cache_lengths = torch.arange(LEFT_CONTEXT + 1, dtype=torch.int16).unsqueeze(1)
        valid = key_index >= (LEFT_CONTEXT - cache_lengths)
        stream_masks = torch.zeros(LEFT_CONTEXT + 1, 1, STREAM_KV_LEN)
        stream_masks.masked_fill_(~valid.unsqueeze(1), -128.0)
        self.register_buffer("stream_masks", stream_masks.contiguous(), persistent=True)

        shift_index = (
            (VALID_OUT_LEN - 1 - torch.arange(VALID_OUT_LEN, dtype=torch.int32)).unsqueeze(1)
            + torch.arange(STREAM_KV_LEN, dtype=torch.int32).unsqueeze(0)
        )
        self.register_buffer(
            "rel_shift_flat_index",
            (
                torch.arange(VALID_OUT_LEN, dtype=torch.int32).unsqueeze(1)
                * (2 * STREAM_KV_LEN - 1)
                + shift_index
            ).reshape(-1).contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "rel_shift_flat_shape",
            torch.tensor(
                [N_HEADS, VALID_OUT_LEN * (2 * STREAM_KV_LEN - 1)],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "rel_shift_output_shape",
            torch.tensor(
                [N_HEADS, VALID_OUT_LEN, STREAM_KV_LEN],
                dtype=torch.int64,
            ),
            persistent=True,
        )

        position_low = backbone.pe_center - STREAM_KV_LEN
        position_high = backbone.pe_center + STREAM_KV_LEN - 1
        for layer_index in range(N_LAYERS):
            position = getattr(backbone, f"pos_proj_{layer_index}")[
                ...,
                position_low:position_high,
            ].float().contiguous()
            self.register_buffer(f"pos_t_{layer_index}", position, persistent=True)
            delattr(backbone, f"pos_proj_{layer_index}")

            qkv_weight = getattr(backbone, f"qkv_w_{layer_index}")
            qkv_bias = getattr(backbone, f"qkv_b_{layer_index}")
            self.register_buffer(
                f"q_w_{layer_index}",
                qkv_weight[:D_MODEL],
                persistent=True,
            )
            self.register_buffer(
                f"q_b_{layer_index}",
                qkv_bias[:D_MODEL],
                persistent=True,
            )
            self.register_buffer(
                f"kv_w_{layer_index}",
                qkv_weight[D_MODEL:],
                persistent=True,
            )
            self.register_buffer(
                f"kv_b_{layer_index}",
                qkv_bias[D_MODEL:],
                persistent=True,
            )
            delattr(backbone, f"qkv_w_{layer_index}")
            delattr(backbone, f"qkv_b_{layer_index}")

    def _stream_mel(self, audio):
        x = audio.float()
        x = x[:, :, 1:] - self.backbone.preemph * x[:, :, :-1]
        real, imaginary = self.backbone.stft(x)
        power = real.square() + imaginary.square()
        mel = torch.matmul(self.backbone.filterbank, power)
        return torch.log(mel + self.backbone.log_guard)

    def forward(
        self,
        audio,
        mel_cache,
        cache_last_channel,
        cache_last_time,
        cache_len,
        speaker_activity,
        background_activity,
    ):
        backbone = self.backbone
        mel_new = self._stream_mel(audio)
        mel_full = torch.cat([mel_cache, mel_new], dim=2)
        mel_cache_next = mel_full[:, :, -STREAM_MEL_CACHE:]
        x = backbone.pre_encode(mel_full, VALID_OUT_LEN + DROP_EXTRA).squeeze(0)
        x = x[DROP_EXTRA:]

        # SpeakerKernelMixin registers this pre-hook at encoder.layers[0]. The
        # background branch intentionally consumes the target-updated residual.
        speaker_mask = speaker_activity.squeeze(0).unsqueeze(1)
        background_mask = background_activity.squeeze(0).unsqueeze(1)
        x = x + backbone.speaker_kernel(x * speaker_mask, "spk")
        x = x + backbone.speaker_kernel(x * background_mask, "bg_spk")

        neg = torch.index_select(self.stream_masks, 0, cache_len)
        cache_last_channel = cache_last_channel.squeeze(1)
        channel_next = []
        time_next = []

        for layer_index in range(N_LAYERS):
            residual = x
            m = layer_norm(x, backbone.ln_ones)
            m = F.linear(
                m,
                getattr(backbone, f"ff1_l1w_{layer_index}"),
                getattr(backbone, f"ff1_l1b_{layer_index}"),
            )
            m = swish(m)
            residual = gemm_residual(
                m,
                getattr(backbone, f"ff1_l2w_{layer_index}"),
                residual,
            )

            m = layer_norm(residual, backbone.ln_ones)
            m_full = torch.cat([cache_last_channel[layer_index], m], dim=0)
            channel_next.append(m_full[-LEFT_CONTEXT:])
            q = F.linear(
                m,
                getattr(self, f"q_w_{layer_index}"),
                getattr(self, f"q_b_{layer_index}"),
            )
            q = q.reshape(VALID_OUT_LEN, N_HEADS, HEAD_DIM).transpose(0, 1)
            kv = F.linear(
                m_full,
                getattr(self, f"kv_w_{layer_index}"),
                getattr(self, f"kv_b_{layer_index}"),
            )
            kv = kv.reshape(STREAM_KV_LEN, 2 * N_HEADS, HEAD_DIM).transpose(0, 1)
            k, v = torch.split(kv, N_HEADS, dim=0)
            q_u = q + getattr(backbone, f"bias_u_{layer_index}")
            q_v = q + getattr(backbone, f"bias_v_{layer_index}")
            content_attention = torch.matmul(q_u, k.transpose(1, 2))
            position_attention = torch.matmul(q_v, getattr(self, f"pos_t_{layer_index}"))
            position_attention = relative_shift(
                position_attention,
                self.rel_shift_flat_index,
                self.rel_shift_flat_shape,
                self.rel_shift_output_shape,
            )
            attention = torch.softmax(content_attention + position_attention + neg, dim=-1)
            context = torch.matmul(attention, v).transpose(0, 1).reshape(
                VALID_OUT_LEN,
                D_MODEL,
            )
            residual = gemm_residual(
                context,
                getattr(backbone, f"attn_out_w_{layer_index}"),
                residual,
            )

            m = layer_norm(residual, backbone.ln_ones)
            convolution = F.linear(
                m,
                getattr(backbone, f"pw1_w_{layer_index}"),
                getattr(backbone, f"pw1_b_{layer_index}"),
            )
            convolution = F.glu(convolution, dim=1).transpose(0, 1).unsqueeze(0)
            convolution_input = torch.cat(
                [cache_last_time[layer_index], convolution],
                dim=2,
            )
            time_next.append(convolution_input[:, :, -CONV_CACHE:])
            convolution = F.conv1d(
                convolution_input,
                getattr(backbone, f"dw_w_{layer_index}"),
                groups=D_MODEL,
            )
            convolution = convolution.squeeze(0).transpose(0, 1)
            convolution = layer_norm(
                convolution,
                getattr(backbone, f"conv_norm_w_{layer_index}"),
                getattr(backbone, f"conv_norm_b_{layer_index}"),
            )
            convolution = swish(convolution)
            residual = gemm_residual(
                convolution,
                getattr(backbone, f"pw2_w_{layer_index}"),
                residual,
            )

            m = layer_norm(residual, backbone.ln_ones)
            m = F.linear(
                m,
                getattr(backbone, f"ff2_l1w_{layer_index}"),
                getattr(backbone, f"ff2_l1b_{layer_index}"),
            )
            m = swish(m)
            residual = gemm_residual(
                m,
                getattr(backbone, f"ff2_l2w_{layer_index}"),
                residual,
            )
            x = layer_norm(
                residual,
                getattr(backbone, f"norm_out_w_{layer_index}"),
                getattr(backbone, f"norm_out_b_{layer_index}"),
            )

        enc_proj = F.linear(x, backbone.enc_proj_w, backbone.enc_proj_b).unsqueeze(0)
        cache_last_channel_next = torch.cat(channel_next, dim=0).reshape(
            N_LAYERS,
            1,
            LEFT_CONTEXT,
            D_MODEL,
        )
        cache_last_time_next = torch.cat(time_next, dim=0).unsqueeze(1)
        cache_len_next = torch.clamp(cache_len + VALID_OUT_LEN, max=LEFT_CONTEXT)
        return (
            enc_proj,
            mel_cache_next,
            cache_last_channel_next,
            cache_last_time_next,
            cache_len_next,
        )


class MultiTalkerStreamingParakeetDecoderJoint(nn.Module):
    """One source-equivalent RNN-T prediction/joint step."""

    def __init__(self, state_dict: dict):
        super().__init__()
        self.blank_id = BLANK_ID
        self.embed = nn.Embedding(
            LOGITS_SIZE,
            PRED_HIDDEN,
            dtype=FLOAT_STORAGE_DTYPE,
        )
        self.lstm = nn.LSTM(
            PRED_HIDDEN,
            PRED_HIDDEN,
            LSTM_LAYERS,
            batch_first=True,
        )
        with torch.no_grad():
            self.embed.weight.copy_(
                state_dict["decoder.prediction.embed.weight"].to(FLOAT_STORAGE_DTYPE)
            )
            for layer_index in range(LSTM_LAYERS):
                for parameter in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(self.lstm, f"{parameter}_l{layer_index}").copy_(
                        state_dict[
                            f"decoder.prediction.dec_rnn.lstm.{parameter}_l{layer_index}"
                        ].float()
                    )
        self.register_buffer(
            "pred_w",
            state_dict["joint.pred.weight"].float().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "pred_b",
            state_dict["joint.pred.bias"].float().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "joint_w",
            state_dict["joint.joint_net.2.weight"].float().contiguous(),
            persistent=True,
        )
        self.register_buffer(
            "joint_b",
            state_dict["joint.joint_net.2.bias"].float().contiguous(),
            persistent=True,
        )

    def forward(self, enc_proj, frame_idx, token, state_h, state_c):
        enc_frame = torch.flatten(torch.index_select(enc_proj, 1, frame_idx), start_dim=1)
        embedding = self.embed(token).float()
        prediction, (state_h_next, state_c_next) = self.lstm(
            embedding,
            (state_h, state_c),
        )
        prediction = F.linear(
            torch.flatten(prediction, start_dim=1),
            self.pred_w,
            self.pred_b,
        )
        logits = F.linear(
            torch.relu(enc_frame + prediction),
            self.joint_w,
            self.joint_b,
        )
        next_token = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)
        is_blank = next_token == self.blank_id
        next_token = torch.where(is_blank, token, next_token)
        state_h_next = torch.where(is_blank, state_h, state_h_next)
        state_c_next = torch.where(is_blank, state_c, state_c_next)
        return next_token, is_blank.to(torch.int32), state_h_next, state_c_next


def make_metadata() -> dict:
    labels = list(_CFG["labels"])
    return build_model_metadata(
        {
            "sample_rate": SAMPLE_RATE,
            "audio_pcm_scale": AUDIO_PCM_SCALE,
            "max_symbols_per_frame": MAX_SYMBOLS_PER_FRAME,
            "special_token_ids": {
                "blank": BLANK_ID,
                "unknown": labels.index("<unk>"),
            },
            "stream_stride_samples": STREAM_STRIDE_SAMPLES,
            "stream_left_overlap": STREAM_LEFT_OVERLAP,
            "stream_att_context_size": ATT_CONTEXT_SIZE,
            "stream_valid_output_frames": VALID_OUT_LEN,
            "speaker_kernel_layers": [SPK_KERNEL_LAYER],
            "speaker_activity_input": {
                "name": "speaker_activity",
                "dtype": "float32",
                "frames": VALID_OUT_LEN,
            },
            "background_activity_input": {
                "name": "background_activity",
                "dtype": "bool",
                "frames": VALID_OUT_LEN,
            },
            "num_speakers": NUM_SPEAKERS,
        }
    )


def export_all() -> None:
    ONNX_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"MultiTalker Streaming Parakeet ASR export -> {ONNX_FOLDER}")
    print(
        f"  att_context={list(ATT_CONTEXT_SIZE)} chunk={VALID_OUT_LEN} frames "
        f"window={STREAM_WINDOW_SAMPLES} stride={STREAM_STRIDE_SAMPLES} samples"
    )

    assets = {"ckpt": None}
    export_temp = tempfile.TemporaryDirectory(prefix="multitalker-parakeet-export-")
    raw_dir = Path(export_temp.name)
    state_dict = backbone = encoder = decoder = None
    try:
        assets = ensure_assets(NEMO_PATH, ONNX_FOLDER, raw_dir)
        state_dict = torch.load(
            str(assets["ckpt"]),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        state_dict = state_dict.get("state_dict", state_dict)
        metadata = make_metadata()

        backbone = MultiTalkerStreamingParakeetBackbone(state_dict).eval()
        encoder = MultiTalkerStreamingParakeetEncoder(backbone).eval()

        with torch.inference_mode():
            metadata_path = raw_dir / METADATA_NAME
            torch.onnx.export(
                MetadataCarrier().eval(),
                (torch.zeros(1, dtype=torch.int64),),
                str(metadata_path),
                input_names=["metadata_marker"],
                output_names=["metadata_marker_out"],
                opset_version=OPSET,
                dynamo=False,
            )
            finalize_graph(
                metadata_path,
                ONNX_FOLDER / METADATA_NAME,
                metadata,
                replace_metadata=True,
            )

            encoder_path = raw_dir / ENCODER_NAME
            audio = torch.zeros(
                1,
                1,
                STREAM_WINDOW_SAMPLES,
                dtype=_AUDIO_TORCH_DTYPE,
            )
            mel_cache = torch.zeros(1, N_MELS, STREAM_MEL_CACHE)
            channel_cache = torch.zeros(N_LAYERS, 1, LEFT_CONTEXT, D_MODEL)
            time_cache = torch.zeros(N_LAYERS, 1, D_MODEL, CONV_CACHE)
            cache_len = torch.zeros(1, dtype=torch.int64)
            speaker_activity = torch.zeros(1, VALID_OUT_LEN)
            background_activity = torch.zeros(
                1,
                VALID_OUT_LEN,
                dtype=torch.bool,
            )
            torch.onnx.export(
                encoder,
                (
                    audio,
                    mel_cache,
                    channel_cache,
                    time_cache,
                    cache_len,
                    speaker_activity,
                    background_activity,
                ),
                str(encoder_path),
                input_names=[
                    "audio",
                    "mel_cache",
                    "cache_last_channel",
                    "cache_last_time",
                    "cache_len",
                    "speaker_activity",
                    "background_activity",
                ],
                output_names=[
                    "enc_proj",
                    "mel_cache_next",
                    "cache_last_channel_next",
                    "cache_last_time_next",
                    "cache_len_next",
                ],
                dynamic_axes=None,
                opset_version=OPSET,
                dynamo=False,
            )
            finalize_graph(encoder_path, ONNX_FOLDER / ENCODER_NAME)

            encoder = None
            backbone = None
            gc.collect()
            decoder = MultiTalkerStreamingParakeetDecoderJoint(state_dict).eval()
            state_dict = None
            gc.collect()

            decoder_path = raw_dir / DECODER_NAME
            enc_proj = torch.randn(1, VALID_OUT_LEN, JOINT_HIDDEN)
            frame_idx = torch.zeros(1, dtype=torch.int32)
            token = torch.full((1, 1), BLANK_ID, dtype=torch.int32)
            state_h = torch.zeros(LSTM_LAYERS, 1, PRED_HIDDEN)
            state_c = torch.zeros(LSTM_LAYERS, 1, PRED_HIDDEN)
            dynamic_axes = {
                "enc_proj": {0: "batch", 1: "enc_frames"},
                "token": {0: "batch"},
                "state_h": {1: "batch"},
                "state_c": {1: "batch"},
                "next_token": {0: "batch"},
                "is_blank": {0: "batch"},
                "state_h_next": {1: "batch"},
                "state_c_next": {1: "batch"},
            }
            torch.onnx.export(
                decoder,
                (enc_proj, frame_idx, token, state_h, state_c),
                str(decoder_path),
                input_names=["enc_proj", "frame_idx", "token", "state_h", "state_c"],
                output_names=["next_token", "is_blank", "state_h_next", "state_c_next"],
                dynamic_axes=dynamic_axes,
                opset_version=OPSET,
                dynamo=False,
            )
            finalize_graph(decoder_path, ONNX_FOLDER / DECODER_NAME)
    finally:
        del state_dict, backbone, encoder, decoder
        gc.collect()
        try:
            remove_extracted_checkpoint(assets)
        finally:
            export_temp.cleanup()


def export_sortformer() -> None:
    """Export the matching Streaming Sortformer graph beside the ASR graphs."""
    if not SORTFORMER_NEMO_PATH.is_file():
        raise FileNotFoundError(
            f"Streaming Sortformer checkpoint not found: {SORTFORMER_NEMO_PATH}"
        )

    print("\nNVIDIA Streaming Sortformer 4spk v2.1 ONNX Export")
    model = pipeline = None
    try:
        model = _load_sortformer_model(SORTFORMER_NEMO_PATH)
        modules = model.sortformer_modules
        # Match the 14-frame MultiTalker Parakeet streaming deployment profile.
        modules.spkcache_len = 188
        modules.fifo_len = 188
        modules.chunk_len = VALID_OUT_LEN
        modules.chunk_left_context = 0
        modules.chunk_right_context = 0
        modules.spkcache_update_period = 144

        pipeline = SortformerStreamingPipeline(model).eval()
        SORTFORMER_ONNX_FOLDER.mkdir(parents=True, exist_ok=True)
        for stale_model in (
            SORTFORMER_ONNX_FOLDER / "NVIDIA_Streaming_Sortformer_4spk_Neural.onnx",
            SORTFORMER_ONNX_FOLDER / "NVIDIA_Streaming_Sortformer_4spk_Frontend.onnx",
        ):
            stale_model.unlink(missing_ok=True)
            stale_model.with_name(stale_model.stem + "_Metadata.onnx").unlink(
                missing_ok=True
            )
        for stale_model in SORTFORMER_ONNX_FOLDER.glob("*_raw.onnx"):
            stale_model.unlink(missing_ok=True)
            stale_model.with_name(stale_model.stem + "_Metadata.onnx").unlink(
                missing_ok=True
            )

        print(f"Checkpoint: {SORTFORMER_NEMO_PATH}")
        print(f"Export folder: {SORTFORMER_ONNX_FOLDER}")
        print(f"Audio input dtype: {SORTFORMER_INPUT_AUDIO_DTYPE}")
        print("Exporting static waveform-to-activity step ...")
        with tempfile.TemporaryDirectory(
            prefix="sortformer-export-",
            dir=SORTFORMER_ONNX_FOLDER,
        ) as temp_name:
            raw_dir = Path(temp_name)
            raw_model_path = raw_dir / SORTFORMER_MODEL_NAME
            torch.onnx.export(
                pipeline,
                (
                    torch.zeros(
                        1,
                        1,
                        pipeline.audio_input_samples,
                        dtype=SORTFORMER_TORCH_DTYPE,
                    ),
                    torch.full(
                        (1,),
                        pipeline.audio_chunk_samples,
                        dtype=SORTFORMER_LENGTH_DTYPE,
                    ),
                    torch.full((1,), 0.5, dtype=torch.float32),
                    torch.zeros(
                        1,
                        pipeline.speaker_cache_capacity,
                        pipeline.embedding_dimension,
                    ),
                    torch.zeros(1, dtype=SORTFORMER_LENGTH_DTYPE),
                    torch.zeros(
                        1,
                        pipeline.speaker_cache_capacity,
                        pipeline.speaker_count,
                    ),
                    torch.zeros(1, dtype=SORTFORMER_FLAG_DTYPE),
                    torch.zeros(
                        1,
                        pipeline.fifo_capacity,
                        pipeline.embedding_dimension,
                    ),
                    torch.zeros(1, dtype=SORTFORMER_LENGTH_DTYPE),
                    torch.zeros(1, pipeline.embedding_dimension),
                    torch.zeros(1, dtype=SORTFORMER_LENGTH_DTYPE),
                    torch.zeros(
                        1,
                        pipeline.speaker_count,
                        dtype=SORTFORMER_LENGTH_DTYPE,
                    ),
                    torch.zeros(1, dtype=SORTFORMER_LENGTH_DTYPE),
                ),
                str(raw_model_path),
                export_params=True,
                input_names=[
                    "audio",
                    "audio_lengths",
                    "activity_threshold",
                    "spkcache",
                    "spkcache_lengths",
                    "spkcache_preds",
                    "spkcache_has_preds",
                    "fifo",
                    "fifo_lengths",
                    "mean_sil_emb",
                    "n_sil_frames",
                    "active_frame_counts",
                    "overlap_frame_count",
                ],
                output_names=[
                    "activity",
                    "next_spkcache",
                    "next_spkcache_lengths",
                    "next_spkcache_preds",
                    "next_spkcache_has_preds",
                    "next_fifo",
                    "next_fifo_lengths",
                    "next_mean_sil_emb",
                    "next_n_sil_frames",
                    "next_active_frame_counts",
                    "next_overlap_frame_count",
                ],
                opset_version=OPSET,
                do_constant_folding=True,
                keep_initializers_as_inputs=False,
                training=torch.onnx.TrainingMode.EVAL,
                dynamo=False,
            )
            finalize_graph(raw_model_path, SORTFORMER_ONNX_MODEL)

            raw_metadata_path = raw_dir / SORTFORMER_METADATA_NAME
            torch.onnx.export(
                MetadataCarrier().eval(),
                (torch.zeros(1, dtype=torch.int64),),
                str(raw_metadata_path),
                input_names=["metadata_marker"],
                output_names=["metadata_marker_out"],
                opset_version=OPSET,
                dynamo=False,
            )
            finalize_graph(
                raw_metadata_path,
                SORTFORMER_METADATA_MODEL,
                build_model_metadata(
                    {
                        "sample_rate": pipeline.sample_rate,
                        "audio_cache_samples": pipeline.audio_cache_samples,
                        "new_audio_samples": pipeline.new_audio_samples,
                        "output_activity_frames": pipeline.output_activity_frames,
                        "num_speakers": pipeline.speaker_count,
                    }
                ),
                replace_metadata=True,
            )
        print(f"Published model: {SORTFORMER_ONNX_MODEL}")
        print(f"Metadata: {SORTFORMER_METADATA_MODEL}")
    finally:
        del model, pipeline
        gc.collect()


def _sortformer_numpy_dtype(onnx_type: str):
    import numpy as np

    dtypes = {
        "tensor(bool)": np.bool_,
        "tensor(float16)": np.float16,
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(int16)": np.int16,
        "tensor(uint16)": np.uint16,
        "tensor(int32)": np.int32,
        "tensor(uint32)": np.uint32,
        "tensor(int64)": np.int64,
        "tensor(uint64)": np.uint64,
    }
    try:
        return np.dtype(dtypes[onnx_type])
    except KeyError as error:
        raise TypeError(f"Unsupported Sortformer ONNX tensor type: {onnx_type!r}") from error


def _sortformer_static_shape(value_info) -> tuple[int, ...]:
    try:
        shape = tuple(int(dimension) for dimension in value_info.shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Sortformer ONNX tensor {value_info.name!r} must have a static shape; "
            f"got {value_info.shape!r}."
        ) from error
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(
            f"Sortformer ONNX tensor {value_info.name!r} has an invalid shape: "
            f"{value_info.shape!r}."
        )
    return shape


def run_sortformer_activity(
    audio_path: Path,
    activity_path: Path,
    activity_threshold: float = 0.5,
) -> Path:
    """Generate the soft `[chunks, frames, speakers]` activity sidecar locally."""
    import numpy as np
    import onnxruntime
    from pydub import AudioSegment

    if not 0.0 <= activity_threshold <= 1.0:
        raise ValueError("Sortformer activity_threshold must be between 0 and 1.")
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not SORTFORMER_ONNX_MODEL.is_file():
        raise FileNotFoundError(
            f"Sortformer ONNX model not found: {SORTFORMER_ONNX_MODEL}"
        )
    if not SORTFORMER_METADATA_MODEL.is_file():
        raise FileNotFoundError(
            f"Sortformer metadata model not found: {SORTFORMER_METADATA_MODEL}"
        )

    session_options = onnxruntime.SessionOptions()
    session_options.log_severity_level = 4
    session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session_options.add_session_config_entry("session.set_denormal_as_zero", "1")
    session_options.add_session_config_entry("session.intra_op.allow_spinning", "1")
    session_options.add_session_config_entry("session.inter_op.allow_spinning", "1")
    session = onnxruntime.InferenceSession(
        str(SORTFORMER_ONNX_MODEL),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    metadata_session = onnxruntime.InferenceSession(
        str(SORTFORMER_METADATA_MODEL),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    metadata = metadata_session.get_modelmeta().custom_metadata_map or {}
    try:
        sample_rate = int(metadata["sample_rate"])
        audio_cache_samples = int(metadata["audio_cache_samples"])
        new_audio_samples = int(metadata["new_audio_samples"])
        output_activity_frames = int(metadata["output_activity_frames"])
        speaker_count = int(metadata["num_speakers"])
    except KeyError as error:
        raise ValueError(
            f"Sortformer metadata is missing {error.args[0]!r}. Re-export the model."
        ) from error

    input_meta = {value.name: value for value in session.get_inputs()}
    output_names = [value.name for value in session.get_outputs()]
    output_meta = {value.name: value for value in session.get_outputs()}
    required_inputs = (
        "audio",
        "audio_lengths",
        "activity_threshold",
        "spkcache",
        "spkcache_lengths",
        "spkcache_preds",
        "spkcache_has_preds",
        "fifo",
        "fifo_lengths",
        "mean_sil_emb",
        "n_sil_frames",
        "active_frame_counts",
        "overlap_frame_count",
    )
    missing_inputs = [name for name in required_inputs if name not in input_meta]
    if missing_inputs:
        raise ValueError(
            "Sortformer ONNX model is missing required inputs: "
            f"{', '.join(missing_inputs)}."
        )
    state_input_names = required_inputs[3:]
    missing_outputs = [
        name
        for name in ("activity", *(f"next_{state}" for state in state_input_names))
        if name not in output_meta
    ]
    if missing_outputs:
        raise ValueError(
            "Sortformer ONNX model is missing required outputs: "
            f"{', '.join(missing_outputs)}."
        )

    audio_shape = _sortformer_static_shape(input_meta["audio"])
    audio_dtype = _sortformer_numpy_dtype(input_meta["audio"].type)
    if len(audio_shape) != 3 or np.prod(audio_shape[:-1], dtype=np.int64) != 1:
        raise ValueError(
            f"Sortformer audio input must describe one audio stream; got {audio_shape}."
        )
    audio_input_samples = audio_shape[-1]
    if audio_input_samples != audio_cache_samples + new_audio_samples + 1:
        raise ValueError(
            "Sortformer audio input length does not match metadata: "
            f"{audio_input_samples} != {audio_cache_samples} + "
            f"{new_audio_samples} + 1."
        )
    activity_shape = _sortformer_static_shape(output_meta["activity"])
    expected_activity_shape = (1, output_activity_frames, speaker_count)
    if activity_shape != expected_activity_shape:
        raise ValueError(
            "Sortformer activity output does not match metadata: "
            f"{activity_shape} != {expected_activity_shape}."
        )

    segment = (
        AudioSegment.from_file(audio_path)
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
    )
    audio_int16 = np.array(segment.get_array_of_samples(), dtype=np.int16)
    if audio_int16.size == 0:
        raise ValueError("Audio input contains no samples.")
    if audio_dtype == np.dtype(np.int16):
        model_audio = audio_int16
    elif np.issubdtype(audio_dtype, np.floating):
        model_audio = audio_int16.astype(audio_dtype)
        model_audio /= np.asarray(32768.0, dtype=audio_dtype)
    else:
        raise TypeError(
            f"Unsupported Sortformer waveform dtype {audio_dtype.name!r}."
        )

    inputs = {
        name: np.zeros(
            _sortformer_static_shape(input_meta[name]),
            dtype=_sortformer_numpy_dtype(input_meta[name].type),
        )
        for name in required_inputs
    }
    inputs["activity_threshold"].fill(activity_threshold)
    audio_window = inputs["audio"]
    audio_samples = audio_window.reshape(-1, audio_input_samples)[0]
    audio_lengths = inputs["audio_lengths"]
    num_chunks = (model_audio.size + new_audio_samples - 1) // new_audio_samples
    activity_history = np.empty(
        (num_chunks, output_activity_frames, speaker_count),
        dtype=np.float32,
    )

    for chunk_index in range(num_chunks):
        new_start = chunk_index * new_audio_samples
        window_start = new_start - audio_cache_samples
        source_start = max(window_start, 0)
        source_end = min(window_start + audio_cache_samples + new_audio_samples, model_audio.size)
        destination_start = source_start - window_start
        audio_window.fill(0)
        if window_start > 0:
            audio_samples[0] = model_audio[window_start - 1]
        audio_samples[
            1 + destination_start : 1 + destination_start + source_end - source_start
        ] = model_audio[source_start:source_end]
        audio_lengths.fill(
            audio_cache_samples + min(new_audio_samples, model_audio.size - new_start)
        )
        outputs = dict(zip(output_names, session.run(output_names, inputs)))
        activity_history[chunk_index] = outputs["activity"][0].astype(
            np.float32,
            copy=False,
        )
        for state_name in state_input_names:
            inputs[state_name] = outputs[f"next_{state_name}"]

    expected_asr_chunks = (
        model_audio.size + STREAM_STRIDE_SAMPLES - 1
    ) // STREAM_STRIDE_SAMPLES
    if num_chunks != expected_asr_chunks:
        raise ValueError(
            "Sortformer and Parakeet streaming strides produce different chunk "
            f"counts ({num_chunks} != {expected_asr_chunks})."
        )
    if activity_history.shape[1:] != (VALID_OUT_LEN, NUM_SPEAKERS):
        raise ValueError(
            "Sortformer activity does not match the Parakeet speaker contract: "
            f"{activity_history.shape[1:]} != {(VALID_OUT_LEN, NUM_SPEAKERS)}."
        )

    activity_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = activity_path.with_name(activity_path.name + ".tmp")
    try:
        with staging_path.open("wb") as destination:
            np.save(destination, activity_history)
        staging_path.replace(activity_path)
    finally:
        staging_path.unlink(missing_ok=True)
    print(f"Generated Sortformer activity: {activity_path}")
    return activity_path


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def export_sortformer_and_run_pipeline() -> None:
    """Export both models, generate activity, and run local speaker-tagged ASR."""
    export_sortformer()
    audio_path = _require_file(
        _SCRIPT_DIR.parents[1]
        / "Test_Examples"
        / "en"
        / "test_sample_multitalker_overlap.wav",
        "MultiTalker test audio",
    )
    activity_path = SORTFORMER_ONNX_FOLDER / f"{audio_path.stem}_sortformer.npy"
    run_sortformer_activity(audio_path, activity_path)

    asr_inference = _require_file(
        _SCRIPT_DIR / "Inference_MultiTalker_Streaming_Parakeet_ASR_ONNX.py",
        "MultiTalker Parakeet inference script",
    )
    print("\nRunning Streaming Sortformer and MultiTalker Parakeet together ...")
    subprocess.run(
        [
            sys.executable,
            str(asr_inference),
            "--onnx-folder",
            str(ONNX_FOLDER),
            "--audio-path",
            str(audio_path),
            "--diarization-activity-path",
            str(activity_path),
        ],
        cwd=str(_SCRIPT_DIR),
        check=True,
    )


if __name__ == "__main__":
    export_all()
    export_sortformer_and_run_pipeline()