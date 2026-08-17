#!/usr/bin/env python
"""Run exported MultiTalker Streaming Parakeet ONNX graphs with IOBinding.

The diarization source remains external to this ASR-only pipeline. Its source
output is read as one [14, 4] activity matrix per streaming chunk; each column
drives one independent speaker-conditioned ASR state.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
import sentencepiece as spm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ORT_IO import (
    array_for,
    filled_for,
    load_special_token_ids,
    metadata_by_name,
    numpy_dtype,
)


# User configuration and path-only CLI
_SORTFORMER_FOLDER_NAME = "NVIDIA_Streaming_Sortformer_4spk"
_SORTFORMER_MODEL_NAME = f"{_SORTFORMER_FOLDER_NAME}.onnx"
_SORTFORMER_METADATA_NAME = f"{_SORTFORMER_FOLDER_NAME}_Metadata.onnx"
_DEFAULT_ACTIVITY_THRESHOLD = 0.5


_GRAPH_NAMES = (
    "ASR_Metadata.onnx",
    "MultiTalker_Streaming_Parakeet_ASR_Encoder.onnx",
    "MultiTalker_Streaming_Parakeet_ASR_Decoder.onnx",
)
_DEFAULT_CANDIDATES = (
    _SCRIPT_DIR / "MultiTalker_Streaming_Parakeet_ASR_Optimized",
    _SCRIPT_DIR / "MultiTalker_Streaming_Parakeet_ASR_ONNX",
)


# Apply RMS gain normalization before preparing the model input; off preserves source loudness.
USE_NORMALISE_AUDIO = False
# Print each speaker's accumulated transcript after every streaming chunk.
PRINT_PARTIALS = True
# Advance only speaker states with recent diarization activity above the gating threshold.
CACHE_GATING = True
# Retain this many activity chunks when deciding which speaker states to advance.
CACHE_GATING_BUFFER_SIZE = 2

ORT_Accelerate_Providers = []
ORT_LOG = False
MAX_THREADS = 0
DEVICE_ID = 0


def _activity_threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "activity threshold must be a number from 0 to 1"
        ) from error
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "activity threshold must be between 0 and 1"
        )
    return threshold


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone Streaming Sortformer and MultiTalker Parakeet ONNX inference."
    )
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        "--asr-folder",
        dest="onnx_folder",
        type=Path,
        default=None,
        help="Folder with MultiTalker Streaming Parakeet ONNX graphs.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional SentencePiece model; defaults to tokenizer.model in the model folder.",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        help="Audio to transcribe; defaults to the bundled overlap fixture.",
    )
    parser.add_argument(
        "--diarization-activity-path",
        type=Path,
        default=None,
        help="Optional existing [chunks, frames, speakers] NumPy activity sidecar.",
    )
    parser.add_argument(
        "--sortformer-model-folder",
        type=Path,
        default=None,
        help=(
            "Folder with the embedded Sortformer graph; defaults to "
            "<onnx-folder>/NVIDIA_Streaming_Sortformer_4spk."
        ),
    )
    parser.add_argument(
        "--sortformer-activity-path",
        "--activity-path",
        dest="sortformer_activity_path",
        type=Path,
        default=None,
        help="Output path for generated Sortformer activity.",
    )
    parser.add_argument(
        "--activity-threshold",
        type=_activity_threshold,
        default=_DEFAULT_ACTIVITY_THRESHOLD,
        metavar="PROBABILITY",
        help=(
            "Count a Sortformer activity score above this probability as active "
            f"(default: {_DEFAULT_ACTIVITY_THRESHOLD})."
        ),
    )
    parser.add_argument(
        "--activity-only",
        action="store_true",
        help="Generate Sortformer activity without launching ASR.",
    )
    return parser.parse_args()


def _resolve_graphs(onnx_folder: Path | None) -> Path:
    candidates = []
    if onnx_folder is not None:
        candidates.append(onnx_folder.expanduser().resolve())
    candidates.extend(candidate.resolve() for candidate in _DEFAULT_CANDIDATES)
    for folder in candidates:
        if all((folder / name).exists() for name in _GRAPH_NAMES):
            return folder
    return candidates[0]


METADATA_NAME, ENCODER_NAME, DECODER_NAME = _GRAPH_NAMES

_DEFAULT_TEST_AUDIO = (
    _REPO_ROOT / "Test_Examples" / "en" / "test_sample_multitalker_overlap.wav"
)


# ONNX Runtime helpers
def _build_session_opts() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 0 if ORT_LOG else 4
    opts.inter_op_num_threads = MAX_THREADS
    opts.intra_op_num_threads = MAX_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.use_device_allocator_for_initializers": "1",
        "optimization.enable_cast_chain_elimination": "1",
    }.items():
        opts.add_session_config_entry(key, value)
    return opts


if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    _ort_device_type = C.OrtDevice.cuda()
    device_type = "cuda"
    provider_options = [{"device_id": DEVICE_ID}]
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    _ort_device_type = C.OrtDevice.dml()
    device_type = "dml"
    provider_options = [{"device_id": DEVICE_ID}]
else:
    _ort_device_type = C.OrtDevice.cpu()
    device_type = "cpu"
    provider_options = None

_ort_device_obj = C.OrtDevice(
    _ort_device_type,
    C.OrtDevice.default_memory(),
    DEVICE_ID,
)
_session_opts = _build_session_opts()
_packed = {
    "sess_options": _session_opts,
    "providers": ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    "provider_options": provider_options,
}


def _make_session(path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(path), **_packed)


_SORTFORMER_ONNX_DTYPE_MAP = {
    "tensor(bool)": np.dtype(np.bool_),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint64)": np.dtype(np.uint64),
}


@dataclass(frozen=True)
class _SortformerTensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True)
class _SortformerStateTensorSpec:
    input: _SortformerTensorSpec
    output: _SortformerTensorSpec


@dataclass(frozen=True)
class _SortformerModelIO:
    audio_input: _SortformerTensorSpec
    audio_lengths_input: _SortformerTensorSpec
    activity_threshold_input: _SortformerTensorSpec
    activity_output: _SortformerTensorSpec
    active_frame_counts_state: _SortformerStateTensorSpec
    overlap_frame_count_state: _SortformerStateTensorSpec
    state_tensors: tuple[_SortformerStateTensorSpec, ...]


def _build_sortformer_session_opts() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = 0
    options.intra_op_num_threads = 0
    options.enable_cpu_mem_arena = True
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "4",
        "optimization.enable_gelu_approximation": "1",
        "disable_synchronize_execution_providers": "0",
        "optimization.minimal_build_optimizations": "",
        "session.use_device_allocator_for_initializers": "1",
    }.items():
        options.add_session_config_entry(key, value)
    return options


def _make_sortformer_session(path: Path) -> onnxruntime.InferenceSession:
    is_cpu_only = not ORT_Accelerate_Providers or set(ORT_Accelerate_Providers) == {
        "CPUExecutionProvider"
    }
    disabled_optimizers = []
    if is_cpu_only:
        disabled_optimizers.extend(
            (
                "MatMulAddFusion",
                "NchwcTransformer",
                "ConvAddActivationFusion",
                "MatmulTransposeFusion",
            )
        )
    kwargs = {
        "sess_options": _build_sortformer_session_opts(),
        "providers": ORT_Accelerate_Providers or ["CPUExecutionProvider"],
        "disabled_optimizers": disabled_optimizers or None,
    }
    if provider_options is not None:
        kwargs["provider_options"] = provider_options
    return onnxruntime.InferenceSession(str(path), **kwargs)


def _sortformer_ortvalue_device_type(session: onnxruntime.InferenceSession) -> str:
    primary_provider = session.get_providers()[0]
    if primary_provider in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}:
        return "cuda"
    if primary_provider == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def _create_sortformer_ort_buffer(
    shape: tuple[int, ...],
    dtype: np.dtype,
    device_type: str,
) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype),
        device_type,
        DEVICE_ID,
    )


def _update_sortformer_ort_buffer(
    buffer: onnxruntime.OrtValue,
    value: np.ndarray,
) -> None:
    if hasattr(buffer, "update_inplace"):
        buffer.update_inplace(np.ascontiguousarray(value))
    else:
        np.copyto(buffer.numpy(), value)


def _sortformer_tensor_spec(node: object, direction: str) -> _SortformerTensorSpec:
    name = getattr(node, "name")
    node_type = getattr(node, "type")
    node_shape = getattr(node, "shape")
    try:
        dtype = _SORTFORMER_ONNX_DTYPE_MAP[node_type]
    except KeyError as error:
        raise TypeError(
            f"Unsupported Sortformer ONNX {direction} tensor type {node_type!r} "
            f"for {name!r}."
        ) from error
    if not node_shape:
        raise ValueError(
            f"Sortformer ONNX {direction} tensor {name!r} must have a static, "
            "non-empty shape."
        )
    try:
        shape = tuple(int(dimension) for dimension in node_shape)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Sortformer ONNX {direction} tensor {name!r} must have a static "
            f"shape; got {node_shape!r}."
        ) from error
    if any(dimension <= 0 for dimension in shape):
        raise ValueError(
            f"Sortformer ONNX {direction} tensor {name!r} must have only "
            f"positive dimensions; got {node_shape!r}."
        )
    return _SortformerTensorSpec(name=name, shape=shape, dtype=dtype)


def _sortformer_model_io(session: onnxruntime.InferenceSession) -> _SortformerModelIO:
    inputs = tuple(
        _sortformer_tensor_spec(node, "input") for node in session.get_inputs()
    )
    outputs = tuple(
        _sortformer_tensor_spec(node, "output") for node in session.get_outputs()
    )
    state_tensors = []
    for input_spec in inputs:
        matching_outputs = tuple(
            output_spec
            for output_spec in outputs
            if output_spec.name != input_spec.name
            and output_spec.name.endswith(input_spec.name)
            and output_spec.shape == input_spec.shape
            and output_spec.dtype == input_spec.dtype
        )
        if len(matching_outputs) > 1:
            raise ValueError(
                "Could not uniquely pair Sortformer ONNX state input "
                f"{input_spec.name!r} with an output."
            )
        if matching_outputs:
            state_tensors.append(
                _SortformerStateTensorSpec(input_spec, matching_outputs[0])
            )
    state_tensors = tuple(state_tensors)
    state_input_names = {state.input.name for state in state_tensors}
    state_output_names = {state.output.name for state in state_tensors}
    non_state_inputs = tuple(
        input_spec for input_spec in inputs if input_spec.name not in state_input_names
    )
    non_state_outputs = tuple(
        output_spec
        for output_spec in outputs
        if output_spec.name not in state_output_names
    )

    if not state_tensors:
        raise ValueError("The Sortformer ONNX model has no streaming state pairs.")
    if len(non_state_inputs) != 3 or len(non_state_outputs) != 1:
        raise ValueError(
            "Expected audio, audio-length, and activity-threshold inputs plus one "
            "non-state activity output in the Sortformer ONNX model. Re-export "
            "Sortformer with the current exporter."
        )

    audio_candidates = tuple(
        input_spec
        for input_spec in non_state_inputs
        if len(input_spec.shape) >= 2 and np.issubdtype(input_spec.dtype, np.number)
    )
    if len(audio_candidates) != 1:
        raise ValueError("Could not identify exactly one Sortformer waveform input.")
    audio_input = audio_candidates[0]
    lengths_candidates = tuple(
        input_spec
        for input_spec in non_state_inputs
        if input_spec.name != audio_input.name
        and input_spec.shape == (audio_input.shape[0],)
        and np.issubdtype(input_spec.dtype, np.integer)
    )
    if len(lengths_candidates) != 1:
        raise ValueError(
            "Could not identify exactly one Sortformer waveform-length input."
        )
    threshold_candidates = tuple(
        input_spec
        for input_spec in non_state_inputs
        if input_spec.name == "activity_threshold"
        and input_spec.shape == (1,)
        and np.issubdtype(input_spec.dtype, np.floating)
    )
    if len(threshold_candidates) != 1:
        raise ValueError(
            "Expected one float activity_threshold input with shape (1,) in the "
            "Sortformer ONNX model."
        )
    activity_candidates = tuple(
        output_spec
        for output_spec in non_state_outputs
        if output_spec.name == "activity"
        and len(output_spec.shape) == 3
        and output_spec.shape[0] == 1
        and np.issubdtype(output_spec.dtype, np.floating)
    )
    if len(activity_candidates) != 1:
        raise ValueError(
            "Expected one float activity output with shape [1, frames, speakers] "
            "in the Sortformer ONNX model."
        )

    states_by_input_name = {state.input.name: state for state in state_tensors}
    try:
        active_frame_counts_state = states_by_input_name["active_frame_counts"]
        overlap_frame_count_state = states_by_input_name["overlap_frame_count"]
    except KeyError as error:
        raise ValueError(
            "The Sortformer ONNX model has no cumulative activity-count states."
        ) from error
    if (
        len(active_frame_counts_state.input.shape) != 2
        or active_frame_counts_state.input.shape[0] != 1
        or not np.issubdtype(active_frame_counts_state.input.dtype, np.integer)
    ):
        raise ValueError("Invalid active_frame_counts state tensor in Sortformer.")
    if (
        overlap_frame_count_state.input.shape != (1,)
        or not np.issubdtype(overlap_frame_count_state.input.dtype, np.integer)
    ):
        raise ValueError("Invalid overlap_frame_count state tensor in Sortformer.")

    return _SortformerModelIO(
        audio_input=audio_input,
        audio_lengths_input=lengths_candidates[0],
        activity_threshold_input=threshold_candidates[0],
        activity_output=activity_candidates[0],
        active_frame_counts_state=active_frame_counts_state,
        overlap_frame_count_state=overlap_frame_count_state,
        state_tensors=state_tensors,
    )


def _sortformer_state_buffers(
    state_tensors: tuple[_SortformerStateTensorSpec, ...],
    device_type: str,
) -> dict[str, onnxruntime.OrtValue]:
    return {
        state.input.name: _create_sortformer_ort_buffer(
            state.input.shape,
            state.input.dtype,
            device_type,
        )
        for state in state_tensors
    }


def _bind_sortformer_step(
    session: onnxruntime.InferenceSession,
    model_io: _SortformerModelIO,
    audio: onnxruntime.OrtValue,
    audio_lengths: onnxruntime.OrtValue,
    activity_threshold: onnxruntime.OrtValue,
    activity_output: onnxruntime.OrtValue,
    current_state: dict[str, onnxruntime.OrtValue],
    next_state: dict[str, onnxruntime.OrtValue],
) -> onnxruntime.IOBinding:
    binding = session.io_binding()
    binding.bind_ortvalue_input(model_io.audio_input.name, audio)
    binding.bind_ortvalue_input(model_io.audio_lengths_input.name, audio_lengths)
    binding.bind_ortvalue_input(
        model_io.activity_threshold_input.name,
        activity_threshold,
    )
    binding.bind_ortvalue_output(model_io.activity_output.name, activity_output)
    for state in model_io.state_tensors:
        binding.bind_ortvalue_input(state.input.name, current_state[state.input.name])
        binding.bind_ortvalue_output(state.output.name, next_state[state.input.name])
    return binding


def _ort_from_numpy(array: np.ndarray) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array),
        device_type,
        DEVICE_ID,
    )


def _ort_shape(value: onnxruntime.OrtValue) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if callable(shape):
        return tuple(int(dim) for dim in shape())
    return tuple(int(dim) for dim in value.numpy().shape)


def _array_for_runtime_shape(value_meta, value) -> np.ndarray:
    array = np.asarray(value)
    return array_for(
        value_meta,
        array,
        axes={axis: int(size) for axis, size in enumerate(array.shape)},
    )


def _bind_inputs(binding, names, values) -> None:
    for name, value in zip(names, values):
        binding.bind_ortvalue_input(name, value)


def _bind_device_outputs(binding, names) -> None:
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _in_names(session):
    return [item.name for item in session.get_inputs()]


def _out_names(session):
    return [item.name for item in session.get_outputs()]


# Audio and diarization helpers
def load_audio_int16(path, sample_rate: int) -> np.ndarray:
    segment = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
    return np.array(segment.get_array_of_samples(), dtype=np.int16)


def _convert_sortformer_waveform_dtype(
    waveform: np.ndarray,
    audio_dtype: np.dtype,
) -> np.ndarray:
    if audio_dtype == np.dtype(np.int16):
        return np.clip(
            waveform * np.float32(32768.0),
            -32768.0,
            32767.0,
        ).astype(np.int16)
    if np.issubdtype(audio_dtype, np.floating):
        return np.ascontiguousarray(waveform.astype(audio_dtype))
    raise TypeError(
        f"Unsupported Sortformer waveform dtype {audio_dtype.name!r}; expected "
        "a floating-point type or int16."
    )


def run_sortformer_activity(
    audio_path: Path,
    model_folder: Path,
    activity_path: Path,
    activity_threshold: float,
) -> Path:
    """Generate the raw Sortformer `[chunks, frames, speakers]` sidecar locally."""
    if not 0.0 <= activity_threshold <= 1.0:
        raise ValueError("Sortformer activity_threshold must be between 0 and 1.")

    model_folder = model_folder.expanduser().resolve()
    model_path = model_folder / _SORTFORMER_MODEL_NAME
    metadata_path = model_folder / _SORTFORMER_METADATA_NAME
    if not model_path.is_file():
        raise FileNotFoundError(f"Sortformer ONNX model not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Sortformer metadata model not found: {metadata_path}"
        )

    session = _make_sortformer_session(model_path)
    metadata_session = _make_sortformer_session(metadata_path)
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

    model_io = _sortformer_model_io(session)
    if model_io.audio_input.shape[-1] != audio_cache_samples + new_audio_samples + 1:
        raise ValueError(
            "Sortformer waveform input length does not match streaming metadata: "
            f"{model_io.audio_input.shape[-1]} != {audio_cache_samples} + "
            f"{new_audio_samples} + 1."
        )
    if np.prod(model_io.audio_input.shape[:-1], dtype=np.int64) != 1:
        raise ValueError(
            "Sortformer waveform input must describe one audio stream; got "
            f"shape {model_io.audio_input.shape}."
        )
    if np.prod(model_io.audio_lengths_input.shape, dtype=np.int64) != 1:
        raise ValueError(
            "Sortformer waveform-length input must describe one audio stream; got "
            f"shape {model_io.audio_lengths_input.shape}."
        )
    if model_io.active_frame_counts_state.input.shape[-1] != speaker_count:
        raise ValueError(
            "Sortformer activity-count state does not match speaker metadata: "
            f"{model_io.active_frame_counts_state.input.shape[-1]} != {speaker_count}."
        )
    expected_activity_shape = (1, output_activity_frames, speaker_count)
    if model_io.activity_output.shape != expected_activity_shape:
        raise ValueError(
            "Sortformer activity output does not match streaming metadata: "
            f"{model_io.activity_output.shape} != {expected_activity_shape}."
        )

    audio_int16 = load_audio_int16(audio_path, sample_rate)
    if audio_int16.size == 0:
        raise ValueError("Audio input contains no samples.")
    waveform = audio_int16.astype(np.float32) * np.float32(1.0 / 32768.0)
    model_waveform = _convert_sortformer_waveform_dtype(
        waveform,
        model_io.audio_input.dtype,
    )
    num_chunks = (waveform.size + new_audio_samples - 1) // new_audio_samples
    device_type = _sortformer_ortvalue_device_type(session)
    audio_window = np.zeros(
        model_io.audio_input.shape,
        dtype=model_io.audio_input.dtype,
    )
    audio_samples = audio_window.reshape(-1, model_io.audio_input.shape[-1])[0]
    audio_lengths = np.zeros(
        model_io.audio_lengths_input.shape,
        dtype=model_io.audio_lengths_input.dtype,
    )
    audio_buffer = _create_sortformer_ort_buffer(
        model_io.audio_input.shape,
        model_io.audio_input.dtype,
        device_type,
    )
    audio_lengths_buffer = _create_sortformer_ort_buffer(
        model_io.audio_lengths_input.shape,
        model_io.audio_lengths_input.dtype,
        device_type,
    )
    threshold = np.full(
        model_io.activity_threshold_input.shape,
        activity_threshold,
        dtype=model_io.activity_threshold_input.dtype,
    )
    threshold_buffer = _create_sortformer_ort_buffer(
        model_io.activity_threshold_input.shape,
        model_io.activity_threshold_input.dtype,
        device_type,
    )
    _update_sortformer_ort_buffer(threshold_buffer, threshold)
    activity_buffers = tuple(
        _create_sortformer_ort_buffer(
            model_io.activity_output.shape,
            model_io.activity_output.dtype,
            device_type,
        )
        for _ in range(2)
    )
    activity_history = np.empty(
        (num_chunks, *model_io.activity_output.shape[1:]),
        dtype=np.float32,
    )
    state_a = _sortformer_state_buffers(model_io.state_tensors, device_type)
    state_b = _sortformer_state_buffers(model_io.state_tensors, device_type)
    bindings = (
        _bind_sortformer_step(
            session,
            model_io,
            audio_buffer,
            audio_lengths_buffer,
            threshold_buffer,
            activity_buffers[0],
            state_a,
            state_b,
        ),
        _bind_sortformer_step(
            session,
            model_io,
            audio_buffer,
            audio_lengths_buffer,
            threshold_buffer,
            activity_buffers[1],
            state_b,
            state_a,
        ),
    )
    run_options = onnxruntime.RunOptions()
    run_options.log_severity_level = 4
    run_options.log_verbosity_level = 4
    run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")

    start_time = time.perf_counter()
    for chunk_index in range(num_chunks):
        new_start = chunk_index * new_audio_samples
        window_start = new_start - audio_cache_samples
        source_start = max(window_start, 0)
        source_end = min(window_start + audio_cache_samples + new_audio_samples, waveform.size)
        destination_start = source_start - window_start
        audio_window.fill(0)
        if window_start > 0:
            audio_samples[0] = model_waveform[window_start - 1]
        audio_samples[
            1 + destination_start : 1 + destination_start + source_end - source_start
        ] = model_waveform[source_start:source_end]
        audio_lengths.fill(
            audio_cache_samples + min(new_audio_samples, waveform.size - new_start)
        )
        _update_sortformer_ort_buffer(audio_buffer, audio_window)
        _update_sortformer_ort_buffer(audio_lengths_buffer, audio_lengths)
        session.run_with_iobinding(bindings[chunk_index & 1], run_options=run_options)
        activity_history[chunk_index] = activity_buffers[chunk_index & 1].numpy()[0]

    activity_path = activity_path.expanduser().resolve()
    activity_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = activity_path.with_name(activity_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as destination:
            np.save(destination, activity_history)
        temporary_path.replace(activity_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    elapsed = time.perf_counter() - start_time
    final_state = state_b if num_chunks % 2 else state_a
    active_frame_counts = final_state[
        model_io.active_frame_counts_state.input.name
    ].numpy().reshape(-1)
    overlap_frame_count = int(
        final_state[model_io.overlap_frame_count_state.input.name]
        .numpy()
        .reshape(-1)[0]
    )
    audio_seconds = waveform.size / sample_rate
    print(f"  Sortformer providers: {session.get_providers()}")
    print(f"  Sortformer activity: {activity_path}")
    print(
        f"  Sortformer active frames: {active_frame_counts.tolist()}  "
        f"overlap_frames={overlap_frame_count}"
    )
    print(f"  Sortformer elapsed: {elapsed:.3f}s  RTF: {elapsed / audio_seconds:.3f}")
    return activity_path


def prepare_audio_input(
    audio_int16: np.ndarray,
    target_dtype: np.dtype,
    audio_pcm_scale: int,
) -> np.ndarray:
    target_dtype = np.dtype(target_dtype)
    if not USE_NORMALISE_AUDIO and target_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio_int16, dtype=target_dtype)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= 8192.0 / (rms + 1e-7)
            np.clip(audio, -32768.0, 32767.0, out=audio)
    if target_dtype == np.dtype(np.int16):
        return audio.astype(target_dtype)
    audio *= np.float32(1.0 / audio_pcm_scale)
    return audio.astype(target_dtype)


def stream_window(signal, base, left_overlap, window_samples):
    total = signal.shape[2]
    start = base - left_overlap
    end = start + window_samples
    segment = signal[:, :, max(0, start):min(total, end)]
    left_pad = max(0, -start)
    right_pad = max(0, end - total)
    if left_pad or right_pad:
        segment = np.pad(segment, ((0, 0), (0, 0), (left_pad, right_pad)))
    return np.ascontiguousarray(segment)


def load_diarization_activity(
    path: Path,
    audio_path: Path,
    num_chunks: int,
    valid_out_len: int,
    num_speakers: int,
) -> tuple[np.ndarray, str]:
    expected_tail_shape = (valid_out_len, num_speakers)
    if not path.exists():
        activity = np.zeros((num_chunks, *expected_tail_shape), dtype=np.float32)
        activity[:, :, 0] = 1.0
        return activity, f"{path.name} missing; using all-active speaker_0 fallback"

    activity = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    if activity.ndim != 3 or activity.shape[1:] != expected_tail_shape:
        raise ValueError(
            f"Invalid diarization activity shape {activity.shape} in {path}; expected "
            f"[chunks, {valid_out_len}, {num_speakers}]."
        )
    if activity.shape[0] < num_chunks:
        raise ValueError(
            f"Diarization activity in {path} has {activity.shape[0]} chunks, but "
            f"{num_chunks} are required for {audio_path.name}."
        )
    return activity[:num_chunks], path.name


def main() -> None:
    args = _parse_args()
    onnx_folder = _resolve_graphs(args.onnx_folder)
    audio_path = (
        args.audio_path.expanduser().resolve()
        if args.audio_path is not None
        else _DEFAULT_TEST_AUDIO.resolve()
    )
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    if args.diarization_activity_path is not None:
        if args.activity_only:
            raise ValueError(
                "--activity-only cannot be combined with --diarization-activity-path; "
                "use --sortformer-activity-path for the generated sidecar."
            )
        diarization_activity_path = args.diarization_activity_path.expanduser().resolve()
    else:
        sortformer_model_folder = (
            args.sortformer_model_folder.expanduser().resolve()
            if args.sortformer_model_folder is not None
            else onnx_folder / _SORTFORMER_FOLDER_NAME
        )
        generated_activity_path = (
            args.sortformer_activity_path.expanduser().resolve()
            if args.sortformer_activity_path is not None
            else audio_path.with_name(audio_path.stem + "_sortformer.npy")
        )
        print("Generating Streaming Sortformer activity with ONNX Runtime ...")
        diarization_activity_path = run_sortformer_activity(
            audio_path,
            sortformer_model_folder,
            generated_activity_path,
            args.activity_threshold,
        )
    if args.activity_only:
        return

    print(f"Loading MultiTalker Streaming Parakeet ONNX sessions from {onnx_folder} ...")
    sess_meta = _make_session(onnx_folder / METADATA_NAME)
    sess_enc = _make_session(onnx_folder / ENCODER_NAME)
    sess_dj = _make_session(onnx_folder / DECODER_NAME)
    print(f"  Providers: {sess_enc.get_providers()}")

    metadata = sess_meta.get_modelmeta().custom_metadata_map or {}
    special_token_ids = load_special_token_ids(metadata)
    blank_id = special_token_ids["blank"]
    sample_rate = int(metadata["sample_rate"])
    audio_pcm_scale = int(metadata["audio_pcm_scale"])
    max_symbols_per_frame = int(metadata["max_symbols_per_frame"])
    num_speakers = int(metadata["num_speakers"])
    valid_out_len = int(metadata["stream_valid_output_frames"])
    stride_samples = int(metadata["stream_stride_samples"])
    left_overlap = int(metadata["stream_left_overlap"])

    tokenizer_path = (
        args.tokenizer_path.expanduser().resolve()
        if args.tokenizer_path is not None
        else onnx_folder / "tokenizer.model"
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(tokenizer_path))

    enc_names_in = _in_names(sess_enc)
    enc_names_out = _out_names(sess_enc)
    (
        enc_audio_in,
        enc_mel_cache_in,
        enc_channel_cache_in,
        enc_time_cache_in,
        enc_cache_len_in,
        enc_speaker_activity_in,
        enc_background_activity_in,
    ) = enc_names_in
    (
        enc_proj_out,
        enc_mel_cache_next_out,
        enc_channel_cache_next_out,
        enc_time_cache_next_out,
        enc_cache_len_next_out,
    ) = enc_names_out
    enc_input_specs = metadata_by_name(sess_enc.get_inputs())
    enc_output_specs = metadata_by_name(sess_enc.get_outputs())

    dj_names_in = _in_names(sess_dj)
    dj_names_out = _out_names(sess_dj)
    (
        dj_enc_proj_in,
        dj_frame_idx_in,
        dj_token_in,
        dj_state_h_in,
        dj_state_c_in,
    ) = dj_names_in
    (
        dj_next_token_out,
        dj_is_blank_out,
        dj_state_h_next_out,
        dj_state_c_next_out,
    ) = dj_names_out
    dj_input_specs = metadata_by_name(sess_dj.get_inputs())
    dj_output_specs = metadata_by_name(sess_dj.get_outputs())

    audio_int16 = load_audio_int16(audio_path, sample_rate)
    audio = prepare_audio_input(
        audio_int16,
        numpy_dtype(enc_input_specs[enc_audio_in]),
        audio_pcm_scale,
    ).reshape(1, 1, -1)

    window_samples = int(enc_input_specs[enc_audio_in].shape[2])
    audio_buffer = _ort_from_numpy(
        filled_for(
            enc_input_specs[enc_audio_in],
            axes={0: 1, 1: 1, 2: window_samples},
        )
    )

    feedback_inputs = (
        enc_mel_cache_in,
        enc_channel_cache_in,
        enc_time_cache_in,
        enc_cache_len_in,
    )
    feedback_outputs = (
        enc_mel_cache_next_out,
        enc_channel_cache_next_out,
        enc_time_cache_next_out,
        enc_cache_len_next_out,
    )
    feedback_output_indices = [enc_names_out.index(name) for name in feedback_outputs]
    enc_proj_index = enc_names_out.index(enc_proj_out)

    class SpeakerRuntimeState:
        def __init__(self, speaker_id: int):
            self.speaker_id = speaker_id
            self.tokens = []
            self.mel_cache = _ort_from_numpy(filled_for(enc_input_specs[enc_mel_cache_in]))
            self.channel_cache = _ort_from_numpy(
                filled_for(enc_input_specs[enc_channel_cache_in])
            )
            self.time_cache = _ort_from_numpy(
                filled_for(enc_input_specs[enc_time_cache_in])
            )
            self.cache_len = _ort_from_numpy(filled_for(enc_input_specs[enc_cache_len_in]))
            self.speaker_activity = _ort_from_numpy(
                filled_for(enc_input_specs[enc_speaker_activity_in])
            )
            self.background_activity = _ort_from_numpy(
                filled_for(enc_input_specs[enc_background_activity_in])
            )
            self.encoder_binding = sess_enc.io_binding()
            _bind_inputs(
                self.encoder_binding,
                enc_names_in,
                (
                    audio_buffer,
                    self.mel_cache,
                    self.channel_cache,
                    self.time_cache,
                    self.cache_len,
                    self.speaker_activity,
                    self.background_activity,
                ),
            )
            _bind_device_outputs(self.encoder_binding, enc_names_out)

            self.frame_idx_array = filled_for(
                dj_input_specs[dj_frame_idx_in],
                axes={0: 1},
            )
            self.frame_idx = _ort_from_numpy(self.frame_idx_array)
            blank_token = filled_for(
                dj_input_specs[dj_token_in],
                blank_id,
                axes={0: 1, 1: 1},
            )
            self.token = _ort_from_numpy(blank_token)
            self.state_h = _ort_from_numpy(
                filled_for(dj_input_specs[dj_state_h_in], axes={1: 1})
            )
            self.state_c = _ort_from_numpy(
                filled_for(dj_input_specs[dj_state_c_in], axes={1: 1})
            )
            self.next_token = _ort_from_numpy(
                filled_for(dj_output_specs[dj_next_token_out], axes={0: 1, 1: 1})
            )
            self.is_blank = _ort_from_numpy(
                filled_for(dj_output_specs[dj_is_blank_out], axes={0: 1, 1: 1})
            )
            self.state_h_next = _ort_from_numpy(
                filled_for(dj_output_specs[dj_state_h_next_out], axes={1: 1})
            )
            self.state_c_next = _ort_from_numpy(
                filled_for(dj_output_specs[dj_state_c_next_out], axes={1: 1})
            )
            self.decoder_binding = sess_dj.io_binding()
            self.decoder_binding.bind_ortvalue_input(dj_frame_idx_in, self.frame_idx)
            self.decoder_binding.bind_ortvalue_input(dj_token_in, self.token)
            self.decoder_binding.bind_ortvalue_input(dj_state_h_in, self.state_h)
            self.decoder_binding.bind_ortvalue_input(dj_state_c_in, self.state_c)
            self.decoder_binding.bind_ortvalue_output(dj_next_token_out, self.next_token)
            self.decoder_binding.bind_ortvalue_output(dj_is_blank_out, self.is_blank)
            self.decoder_binding.bind_ortvalue_output(dj_state_h_next_out, self.state_h_next)
            self.decoder_binding.bind_ortvalue_output(dj_state_c_next_out, self.state_c_next)

    def decode_segment(state: SpeakerRuntimeState, enc_proj: onnxruntime.OrtValue) -> None:
        state.decoder_binding.bind_ortvalue_input(dj_enc_proj_in, enc_proj)
        for frame_index in range(_ort_shape(enc_proj)[1]):
            state.frame_idx_array[0] = frame_index
            state.frame_idx.update_inplace(state.frame_idx_array)
            emitted = 0
            while emitted < max_symbols_per_frame:
                sess_dj.run_with_iobinding(state.decoder_binding)
                state.token.update_inplace(state.next_token)
                state.state_h.update_inplace(state.state_h_next)
                state.state_c.update_inplace(state.state_c_next)
                if int(state.is_blank.numpy().flat[0]) != 0:
                    break
                state.tokens.append(int(state.token.numpy().flat[0]))
                emitted += 1

    states = [SpeakerRuntimeState(speaker_id) for speaker_id in range(num_speakers)]
    total_samples = audio.shape[2]
    num_chunks = (total_samples + stride_samples - 1) // stride_samples
    diarization_activity, diarization_source = load_diarization_activity(
        diarization_activity_path,
        audio_path,
        num_chunks,
        valid_out_len,
        num_speakers,
    )
    audio_seconds = total_samples / sample_rate
    cache_gating_activity = np.zeros((0, num_speakers), dtype=np.float32)

    print(
        f"  window={window_samples} stride={stride_samples} valid_out_len={valid_out_len} "
        f"speakers={num_speakers} chunks={num_chunks}"
    )
    print(f"  Audio: {audio_path.name}  ({audio_seconds:.2f}s)")
    print(f"  Diarization activity: {diarization_source}\n")

    start_time = time.time()
    base = 0
    for chunk_index in range(num_chunks):
        audio_value = _array_for_runtime_shape(
            enc_input_specs[enc_audio_in],
            stream_window(audio, base, left_overlap, window_samples),
        )
        audio_buffer.update_inplace(audio_value)
        activity = diarization_activity[chunk_index]
        cache_gating_activity = np.concatenate(
            [cache_gating_activity, activity],
            axis=0,
        )[-valid_out_len * CACHE_GATING_BUFFER_SIZE:]
        if CACHE_GATING:
            active_speakers = np.flatnonzero(
                np.max(cache_gating_activity, axis=0) > 0.5
            ).tolist()
        else:
            active_speakers = list(range(num_speakers))

        for speaker_id in active_speakers:
            state = states[speaker_id]
            inactive_speaker_ids = [
                other_speaker_id
                for other_speaker_id in active_speakers
                if other_speaker_id != speaker_id
            ]
            background_activity = (
                (activity[:, inactive_speaker_ids] > 0.5).sum(axis=-1) > 0
            ).reshape(1, -1)
            speaker_activity = activity[:, speaker_id].reshape(1, -1)
            state.speaker_activity.update_inplace(
                _array_for_runtime_shape(
                    enc_input_specs[enc_speaker_activity_in],
                    speaker_activity,
                )
            )
            state.background_activity.update_inplace(
                _array_for_runtime_shape(
                    enc_input_specs[enc_background_activity_in],
                    background_activity,
                )
            )
            sess_enc.run_with_iobinding(state.encoder_binding)
            outputs = state.encoder_binding.get_outputs()
            decode_segment(state, outputs[enc_proj_index])
            _bind_inputs(
                state.encoder_binding,
                feedback_inputs,
                [outputs[index] for index in feedback_output_indices],
            )
            _bind_device_outputs(state.encoder_binding, enc_names_out)

        if PRINT_PARTIALS:
            partials = [
                f"speaker_{state.speaker_id}: {tokenizer.decode(state.tokens)}"
                for state in states
            ]
            print(f"  [chunk {chunk_index + 1:2d}/{num_chunks}] " + " | ".join(partials))
        base += stride_samples

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Audio      : {audio_path.name}  ({audio_seconds:.2f}s)")
    for state in states:
        print(f"speaker_{state.speaker_id}  : {tokenizer.decode(state.tokens)}")
    print(f"Elapsed    : {elapsed:.3f}s   RTF: {elapsed / audio_seconds:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()