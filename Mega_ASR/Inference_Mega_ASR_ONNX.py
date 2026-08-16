"""Run Qwen3-ASR with merged prefill/decode ONNX graphs and shared weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
import soundfile as sf
from scipy.signal import resample_poly
from transformers import AutoTokenizer


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths
from ORT_IO import (
    array_for,
    filled_for,
    is_dynamic_dim,
    metadata_json_object,
    load_special_token_ids,
    load_supported_languages,
    numpy_dtype,
    resolve_supported_language,
    scalar_for,
)
from Shared_Merged import (
    METADATA_MODEL_NAME,
    ROUTER_MODEL_NAME,
    VARIANT_MODEL_FILE_NAMES,
    attach_shared_initializers,
    references_shared_bundle,
)


# ============================================================================
# Paths and demo inputs
# ============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merged Qwen3-ASR ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=_SCRIPT_DIR / "Mega_ASR_Optimized",
        help="Folder containing merged ONNX graphs and shared initializers.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional tokenizer directory; defaults to tokenizer inside the model folder.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
TOKENIZER_PATH = (
    _ARGS.tokenizer_path.expanduser().resolve()
    if _ARGS.tokenizer_path is not None
    else onnx_folder / "tokenizer"
)
_METADATA_MODEL_NAME = METADATA_MODEL_NAME

# Test audio for inference validation.
test_audio = model_audio_paths("qwen_asr")
LANGUAGE_PROMPTS = ["", "", "yue", "ja", "ko"]
TASK_PROMPTS = ["", "tribal chieftain", "", "", ""]


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally limited to model/tokenizer paths.
# Edit this section for all decoding, audio, demo, and runtime behavior.
USE_NORMALISE_AUDIO = False
USE_AUDIO_QUALITY_ROUTING = True
TEST_NOISE_SNR_DB = -3.0            # Relative to the clean recording; None disables the test mix.
TEST_SPEECH_GAIN_DB = -12.0         # Test-only attenuation before reflections and ambience.
TEST_NOISE_SEED = 9527

USE_SAMPLING = False                # Sampling takes precedence over deterministic decoding.
TEMPERATURE = 0.8
TOP_K = 10
TOP_P = 0.95
SAMPLING_REPETITION_PENALTY = 1.0
PENALTY_RANGE = 10
REPEAT_PENALTY = 0.8                # 1.0 selects greedy; another value selects penalty-greedy.

ORT_Accelerate_Providers = []       # ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider"]
ORT_LOG = False
ORT_FP16 = False
MAX_THREADS = 0
DEVICE_ID = 0

_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


# ============================================================================
# Metadata and utility helpers
# ============================================================================
def parse_asr_output(raw: str, user_language: Optional[str] = None) -> Tuple[str, str]:
    if not raw:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""
    if user_language:
        return user_language, text
    if _ASR_TEXT_TAG not in text:
        return "", text
    meta_part, text_part = text.split(_ASR_TEXT_TAG, 1)
    language = meta_part.strip()
    if language.lower().startswith(_LANG_PREFIX):
        language = language[len(_LANG_PREFIX) :].strip()
    if language:
        language = language[:1].upper() + language[1:].lower()
    return language, text_part.strip()


def build_query_prompt_ids(tokenizer: AutoTokenizer, system_prompt: str) -> List[int]:
    return tokenizer.encode(system_prompt, add_special_tokens=False) if system_prompt else []


def _spectral_noise(
    rng: np.random.Generator,
    sample_count: int,
    sample_rate: int,
    spectral_slope: float,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate)
    spectrum_scale = np.zeros_like(frequencies, dtype=np.float32)
    spectrum_scale[1:] = np.power(
        frequencies[1:], -spectral_slope / 2.0
    ).astype(np.float32)
    spectrum = (
        rng.standard_normal(frequencies.size, dtype=np.float32)
        + 1j * rng.standard_normal(frequencies.size, dtype=np.float32)
    ) * spectrum_scale
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=sample_count).astype(np.float32)


def _distant_chirps(
    rng: np.random.Generator,
    sample_count: int,
    sample_rate: int,
) -> np.ndarray:
    chirps = np.zeros(sample_count, dtype=np.float32)
    duration = sample_count / sample_rate
    chirp_count = max(3, int(math.ceil(duration * 1.6)))
    min_chirp_samples = max(8, int(sample_rate * 0.035))
    max_chirp_samples = max(min_chirp_samples + 1, int(sample_rate * 0.28))
    start_frequency_max = sample_rate * 0.24
    end_frequency_max = sample_rate * 0.42
    for _ in range(chirp_count):
        chirp_length = int(rng.integers(min_chirp_samples, max_chirp_samples))
        chirp_length = min(chirp_length, sample_count)
        chirp_start = int(rng.integers(0, sample_count - chirp_length + 1))
        time_axis = np.arange(chirp_length, dtype=np.float32) / sample_rate
        start_frequency = rng.uniform(sample_rate * 0.06, start_frequency_max)
        end_frequency = rng.uniform(sample_rate * 0.12, end_frequency_max)
        sweep_rate = (end_frequency - start_frequency) / (chirp_length / sample_rate)
        phase = 2.0 * np.pi * (
            start_frequency * time_axis + 0.5 * sweep_rate * time_axis * time_axis
        )
        envelope = np.hanning(chirp_length).astype(np.float32)
        chirps[chirp_start : chirp_start + chirp_length] += np.float32(
            rng.uniform(0.25, 0.80)
        ) * np.sin(phase).astype(np.float32) * envelope
    return chirps


def _nature_challenge_ambience(
    rng: np.random.Generator,
    sample_count: int,
    sample_rate: int,
) -> np.ndarray:
    duration = sample_count / sample_rate
    gust_points = max(4, int(math.ceil(duration / 0.35)) + 2)
    gust_positions = np.linspace(0, sample_count - 1, gust_points)
    gust_envelope = np.interp(
        np.arange(sample_count),
        gust_positions,
        rng.uniform(0.15, 1.0, size=gust_points),
    ).astype(np.float32)

    wind = _spectral_noise(rng, sample_count, sample_rate, spectral_slope=2.0)
    wind *= gust_envelope
    rumble = _spectral_noise(rng, sample_count, sample_rate, spectral_slope=3.2)
    rumble_points = max(3, int(math.ceil(duration / 1.2)) + 2)
    rumble *= np.interp(
        np.arange(sample_count),
        np.linspace(0, sample_count - 1, rumble_points),
        rng.uniform(0.10, 1.0, size=rumble_points),
    ).astype(np.float32)
    air = _spectral_noise(rng, sample_count, sample_rate, spectral_slope=0.8)
    rustle = _spectral_noise(rng, sample_count, sample_rate, spectral_slope=-0.4)
    rustle_points = max(4, int(math.ceil(duration / 0.18)) + 2)
    rustle_envelope = np.interp(
        np.arange(sample_count),
        np.linspace(0, sample_count - 1, rustle_points),
        rng.uniform(0.0, 1.0, size=rustle_points),
    ).astype(np.float32)
    rustle *= rustle_envelope * rustle_envelope

    rain_taps = np.zeros(sample_count, dtype=np.float32)
    tap_count = max(1, int(math.ceil(duration * 18.0)))
    tap_indices = rng.integers(0, sample_count, size=tap_count)
    np.add.at(
        rain_taps,
        tap_indices,
        rng.uniform(-1.0, 1.0, size=tap_count).astype(np.float32),
    )
    tap_kernel_size = max(8, int(sample_rate * 0.012))
    tap_decay = np.exp(
        -np.arange(tap_kernel_size, dtype=np.float32) / (sample_rate * 0.0025)
    )
    rain_taps = np.convolve(
        rain_taps,
        rng.standard_normal(tap_kernel_size, dtype=np.float32) * tap_decay,
        mode="same",
    ).astype(np.float32)

    turbulence_bursts = np.zeros(sample_count, dtype=np.float32)
    burst_count = max(2, int(math.ceil(duration * 0.8)))
    min_burst_samples = max(8, int(sample_rate * 0.04))
    max_burst_samples = max(min_burst_samples + 1, int(sample_rate * 0.20))
    for _ in range(burst_count):
        burst_length = int(rng.integers(min_burst_samples, max_burst_samples))
        burst_length = min(burst_length, sample_count)
        burst_start = int(rng.integers(0, sample_count - burst_length + 1))
        burst = rng.standard_normal(burst_length, dtype=np.float32)
        burst[1:] -= np.float32(0.78) * burst[:-1]
        burst *= np.hanning(burst_length).astype(np.float32)
        turbulence_bursts[burst_start : burst_start + burst_length] += (
            rng.uniform(0.25, 0.75) * burst
        )
    chirps = _distant_chirps(rng, sample_count, sample_rate)

    ambience = (
        np.float32(0.28) * wind
        + np.float32(0.20) * rumble
        + np.float32(0.17) * air
        + np.float32(0.12) * rustle
        + np.float32(0.10) * rain_taps
        + np.float32(0.07) * turbulence_bursts
        + np.float32(0.06) * chirps
    )
    ambience_rms = np.sqrt(
        np.mean(ambience * ambience, dtype=np.float32), dtype=np.float32
    )
    return ambience / max(ambience_rms, np.finfo(np.float32).eps)


def _add_early_reflections(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    reflected = signal.copy()
    for delay_seconds, gain in ((0.017, 0.36), (0.043, 0.24), (0.089, 0.14), (0.151, 0.08)):
        delay = max(1, int(sample_rate * delay_seconds))
        if delay < signal.size:
            reflected[delay:] += np.float32(gain) * signal[:-delay]
    return reflected


def _add_test_noise(
    waveform: np.ndarray,
    *,
    seed: int,
    lower_bound: float,
    upper_bound: float,
    sample_rate: int = 16_000,
) -> np.ndarray:
    if TEST_NOISE_SNR_DB is None:
        return np.ascontiguousarray(waveform)
    if not math.isfinite(TEST_NOISE_SNR_DB):
        raise ValueError("TEST_NOISE_SNR_DB must be finite or None.")
    if not math.isfinite(TEST_SPEECH_GAIN_DB):
        raise ValueError("TEST_SPEECH_GAIN_DB must be finite.")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    signal = np.asarray(waveform, dtype=np.float32)
    if signal.size == 0:
        return np.ascontiguousarray(signal)
    signal = np.array(signal, dtype=np.float32, copy=True).reshape(-1)
    reference_rms = np.sqrt(
        np.mean(signal * signal, dtype=np.float32), dtype=np.float32
    )
    if reference_rms <= np.finfo(np.float32).eps:
        return np.ascontiguousarray(signal.reshape(waveform.shape))

    rng = np.random.default_rng(seed)
    ambience = _nature_challenge_ambience(rng, signal.size, sample_rate)
    ambience *= reference_rms / np.float32(
        10.0 ** (TEST_NOISE_SNR_DB / 20.0)
    )
    speech_gain = np.float32(10.0 ** (TEST_SPEECH_GAIN_DB / 20.0))
    noisy = _add_early_reflections(signal * speech_gain, sample_rate) + ambience
    ceiling = np.float32(min(abs(lower_bound), abs(upper_bound)) * 0.995)
    peak = np.max(np.abs(noisy))
    if peak > ceiling:
        noisy *= ceiling / peak
    return np.ascontiguousarray(noisy.reshape(waveform.shape))


def prepare_audio_input(
    audio_int16: np.ndarray,
    target_dtype: np.dtype,
    audio_pcm_scale: int,
    target_rms: float = 4096.0,
) -> np.ndarray:
    if not USE_NORMALISE_AUDIO and target_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio_int16, dtype=target_dtype)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= target_rms / (rms + 1e-7)
            np.clip(
                audio,
                -float(audio_pcm_scale),
                float(audio_pcm_scale) - 1.0,
                out=audio,
            )
    if target_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio, dtype=target_dtype)
    audio *= np.float32(1.0 / audio_pcm_scale)
    return np.ascontiguousarray(audio, dtype=target_dtype)


def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
    options = onnxruntime.RunOptions()
    options.log_severity_level = 4 if silent else 0
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def _build_session_options() -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    entries = {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.enable_quant_qdq_cleanup": "1",
        "session.qdq_matmulnbits_accuracy_level": "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level": "2",
        "optimization.enable_gelu_approximation": "1",
        "optimization.minimal_build_optimizations": "",
        "optimization.enable_cast_chain_elimination": "1",
        "optimization.disable_specified_optimizers": (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if ORT_FP16
            else ""
        ),
    }
    for key, value in entries.items():
        options.add_session_config_entry(key, value)
    return options


def _resolve_execution_provider():
    if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
        return (
            "cpu",
            C.OrtDevice.cpu(),
            [{
                "device_type": "CPU",
                "precision": "ACCURACY",
                "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
                "num_streams": 1,
                "enable_opencl_throttling": False,
                "enable_qdq_optimizer": False,
                "disable_dynamic_shapes": False,
            }],
        )
    if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
        return (
            "cuda",
            C.OrtDevice.cuda(),
            [{
                "device_id": DEVICE_ID,
                "gpu_mem_limit": 24 * 1024**3,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "sdpa_kernel": "2",
                "use_tf32": "1",
                "fuse_conv_bias": "0",
                "cudnn_conv_use_max_workspace": "1",
                "cudnn_conv1d_pad_to_nc1d": "0",
                "tunable_op_enable": "0",
                "tunable_op_tuning_enable": "0",
                "tunable_op_max_tuning_duration_ms": 10,
                "do_copy_in_default_stream": "0",
                "enable_cuda_graph": "0",
                "prefer_nhwc": "0",
                "enable_skip_layer_norm_strict_mode": "0",
                "use_ep_level_unified_stream": "0",
            }],
        )
    if "DmlExecutionProvider" in ORT_Accelerate_Providers:
        return (
            "dml",
            C.OrtDevice.dml(),
            [{
                "device_id": DEVICE_ID,
                "performance_preference": "high_performance",
                "device_filter": "gpu",
                "disable_metacommands": "false",
                "enable_graph_capture": "false",
                "enable_graph_serialization": "false",
            }],
        )
    return "cpu", C.OrtDevice.cpu(), None


run_options = _build_run_options(silent=not ORT_LOG)
device_type, _ort_device_type, provider_options = _resolve_execution_provider()
_ort_device_obj = C.OrtDevice(
    _ort_device_type,
    C.OrtDevice.default_memory(),
    DEVICE_ID,
)


def _make_session(path: Path, shared_path: Path | None = None) -> onnxruntime.InferenceSession:
    options = _build_session_options()
    shared_refs = None
    if shared_path is not None:
        shared_refs = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
        provider_options=provider_options,
        disabled_optimizers=(
            ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
            if ORT_FP16
            else None
        ),
    )
    if shared_refs is not None:
        # SessionOptions.add_initializer does not own the numpy mmap.  Keep both
        # the mmap arrays and OrtValues alive for the complete session lifetime.
        session._native_llm_shared_initializers = shared_refs
    return session


def _load_metadata(folder: Path) -> dict[str, str]:
    path = folder / _METADATA_MODEL_NAME
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 4
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session.get_modelmeta().custom_metadata_map or {}


def _ort_from_numpy(array: np.ndarray, target_device: str | None = None) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array),
        target_device or device_type,
        DEVICE_ID,
    )


def _bind_device_outputs(binding, names: List[str]) -> None:
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _bind_ortvalue_input(
    binding,
    name: str,
    value: onnxruntime.OrtValue,
) -> None:
    if device_type != "cpu":
        binding.bind_ortvalue_input(name, value)
        return
    # Raw CPU binding borrows this address instead of retaining the OrtValue.
    input_values = getattr(binding, "_native_llm_input_values", None)
    if input_values is None:
        input_values = {}
        binding._native_llm_input_values = input_values
    input_values[name] = value
    binding.bind_input(
        name,
        value.device_name(),
        DEVICE_ID,
        value.element_type(),
        value.shape(),
        value.data_ptr(),
    )


def _value_rank(value_meta) -> int:
    return len(value_meta.shape)


def _non_batch_dynamic_axes(value_meta) -> list[int]:
    return [
        axis
        for axis, dim in enumerate(value_meta.shape)
        if axis != 0 and is_dynamic_dim(dim)
    ]


def _sequence_length_dimension(value_meta):
    dynamic_axes = _non_batch_dynamic_axes(value_meta)
    if len(dynamic_axes) > 1:
        raise RuntimeError(
            f"ONNX value {value_meta.name!r} has multiple non-batch dynamic "
            f"dimensions: {value_meta.shape!r}."
        )
    if dynamic_axes:
        return value_meta.shape[dynamic_axes[0]]
    if not value_meta.shape:
        raise RuntimeError(f"ONNX value {value_meta.name!r} has no sequence axis.")
    return value_meta.shape[-1]


def _empty_sequence(value_meta) -> np.ndarray:
    return filled_for(value_meta, axes=_sequence_axes(value_meta, 0))


def _leading_cache_values(values, *, role: str) -> list[object]:
    cache_values = []
    for value in values:
        if _value_rank(value) != 5:
            break
        cache_values.append(value)
    if not cache_values:
        raise RuntimeError(f"Mega-ASR {role} graph has no leading rank-5 cache I/O.")
    return cache_values


def _plan_merged_io(
    session: onnxruntime.InferenceSession,
    strategy: str,
    is_decode: bool,
) -> dict[str, object]:
    role = "decode" if is_decode else "prefill"
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    state_inputs = _leading_cache_values(inputs, role=role)
    state_count = len(state_inputs)
    state_outputs = outputs[:state_count]
    if len(state_outputs) != state_count or any(
        _value_rank(value) != 5 for value in state_outputs
    ):
        raise RuntimeError(
            f"Mega-ASR {role} graph cache outputs do not match its leading "
            "rank-5 cache inputs."
        )

    tail_inputs = inputs[state_count:]
    tail_outputs = outputs[state_count:]
    sequence_inputs = [
        value for value in tail_inputs if _value_rank(value) == 3
    ]
    history_inputs = [
        value for value in tail_inputs if _value_rank(value) == 2
    ]
    scalar_inputs = [
        value for value in tail_inputs if _value_rank(value) <= 1
    ]
    if len(sequence_inputs) + len(history_inputs) + len(scalar_inputs) != len(
        tail_inputs
    ):
        raise RuntimeError(
            f"Mega-ASR {role} graph has unsupported non-cache input ranks: "
            f"{[(value.name, _value_rank(value)) for value in tail_inputs]!r}."
        )

    if is_decode:
        if (
            len(tail_inputs) < 2
            or _value_rank(tail_inputs[0]) > 1
            or _value_rank(tail_inputs[1]) != 3
            or len(sequence_inputs) != 1
        ):
            raise RuntimeError(
                "Mega-ASR decode graph must expose a scalar sequence length "
                "followed by one rank-3 hidden-state input."
            )
        sequence_length_input = tail_inputs[0]
        hidden_states_input = tail_inputs[1]
    else:
        if (
            len(tail_inputs) < 4
            or any(_value_rank(value) != 3 for value in tail_inputs[:3])
            or _value_rank(tail_inputs[3]) > 1
            or len(sequence_inputs) != 3
        ):
            raise RuntimeError(
                "Mega-ASR prefill graph must expose three rank-3 inputs "
                "followed by a scalar sequence length."
            )
        audio_input, query_input, language_tail_input = tail_inputs[:3]
        sequence_length_input = tail_inputs[3]

    control_inputs = [
        value for value in scalar_inputs if value is not sequence_length_input
    ]
    if strategy == "greedy":
        expected_history_inputs, expected_control_inputs = 0, 0
    elif strategy == "sampling":
        expected_history_inputs, expected_control_inputs = 1, 4
    elif is_decode:
        expected_history_inputs, expected_control_inputs = 2, 2
    else:
        expected_history_inputs, expected_control_inputs = 1, 0
    if (
        len(history_inputs) != expected_history_inputs
        or len(control_inputs) != expected_control_inputs
    ):
        raise RuntimeError(
            f"Mega-ASR {role} {strategy} graph has an unexpected non-cache "
            f"I/O contract: history={len(history_inputs)}, "
            f"controls={len(control_inputs)}."
        )

    expected_tail_outputs = 2 if strategy == "greedy" else 3
    if len(tail_outputs) != expected_tail_outputs:
        raise RuntimeError(
            f"Mega-ASR {role} {strategy} graph has {len(tail_outputs)} "
            f"non-cache outputs; expected {expected_tail_outputs}."
        )
    max_output = tail_outputs[0]
    save_output = None if strategy == "greedy" else tail_outputs[1]
    sequence_length_output = tail_outputs[-1]
    if _value_rank(sequence_length_output) > 1:
        raise RuntimeError(
            f"Mega-ASR {role} graph sequence-length output must be scalar-like, "
            f"got {sequence_length_output.shape!r}."
        )
    if save_output is not None and _value_rank(save_output) != 2:
        raise RuntimeError(
            f"Mega-ASR {role} graph history output must be rank 2, got "
            f"{save_output.shape!r}."
        )

    plan = {
        "control_inputs": control_inputs,
        "max_output": max_output,
        "output_names": [value.name for value in outputs],
        "save_inputs": history_inputs,
        "save_output": save_output,
        "sequence_length_input": sequence_length_input,
        "sequence_length_output": sequence_length_output,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
    }
    if is_decode:
        plan["hidden_states_input"] = hidden_states_input
    else:
        plan.update(
            {
                "audio_input": audio_input,
                "language_tail_input": language_tail_input,
                "query_input": query_input,
            }
        )
    return plan


def _resolve_strategy() -> tuple[str, bool]:
    if USE_SAMPLING:
        strategy = "sampling"
        use_direct_penalty = False
    else:
        use_direct_penalty = REPEAT_PENALTY != 1.0
        strategy = "penalty_greedy" if use_direct_penalty else "greedy"
    return strategy, use_direct_penalty


def _scalar_control_values(
    input_metas: list[object],
    values: tuple[float | int, ...],
) -> list[tuple[object, onnxruntime.OrtValue]]:
    if len(input_metas) != len(values):
        raise RuntimeError(
            "Mega-ASR ONNX control-input count does not match the selected "
            f"runtime strategy: expected {len(values)}, got {len(input_metas)}."
        )
    return [
        (input_meta, _ort_from_numpy(scalar_for(input_meta, value)))
        for input_meta, value in zip(input_metas, values)
    ]


def _sampling_scalar_values(
    input_metas: list[object],
) -> list[tuple[object, onnxruntime.OrtValue]]:
    return _scalar_control_values(
        input_metas,
        (TEMPERATURE, TOP_K, TOP_P, SAMPLING_REPETITION_PENALTY),
    )


def _persistent_embed(
    session: onnxruntime.InferenceSession,
    token_ids,
    input_meta,
    output_meta,
) -> onnxruntime.OrtValue:
    token_count = len(token_ids)
    token_ids_value = _ort_from_numpy(array_for(
        input_meta,
        token_ids,
        axes=_sequence_axes(input_meta, token_count),
    ))
    binding = session.io_binding()
    _bind_ortvalue_input(binding, input_meta.name, token_ids_value)
    _bind_device_outputs(binding, [output_meta.name])
    _run(session, binding)
    return binding.get_outputs()[0]


def _sequence_axes(value_meta, sequence_length: int) -> dict[int, int]:
    dynamic_axes = [
        axis for axis, dim in enumerate(value_meta.shape) if is_dynamic_dim(dim)
    ]
    non_batch_axes = _non_batch_dynamic_axes(value_meta)
    if len(non_batch_axes) > 1:
        raise RuntimeError(
            f"ONNX input {value_meta.name!r} has multiple non-batch dynamic "
            f"dimensions: {value_meta.shape!r}."
        )
    axes = {0: 1} if 0 in dynamic_axes else {}
    if non_batch_axes:
        axes[non_batch_axes[0]] = sequence_length
    return axes


def _load_router_waveform(
    audio_path: str | Path,
    sample_rate: int,
    noise_seed: int,
) -> np.ndarray:
    audio_np, source_rate = sf.read(str(audio_path), always_2d=True)
    audio_np = audio_np.mean(axis=1)
    if source_rate != sample_rate:
        divisor = math.gcd(source_rate, sample_rate)
        audio_np = resample_poly(
            audio_np,
            sample_rate // divisor,
            source_rate // divisor,
        )
    waveform = np.ascontiguousarray(audio_np.astype(np.float32, copy=False))
    return _add_test_noise(
        waveform,
        seed=noise_seed,
        lower_bound=-1.0,
        upper_bound=1.0,
        sample_rate=sample_rate,
    )


def _build_router_runtime(
    router_path: Path,
) -> dict[str, object]:
    session = _make_session(router_path)
    inputs = list(session.get_inputs())
    outputs = list(session.get_outputs())
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(
            "Mega-ASR router must expose exactly one input and one output, got "
            f"{[value.name for value in inputs]!r} and "
            f"{[value.name for value in outputs]!r}."
        )
    input_meta = inputs[0]
    if len(input_meta.shape) != 3:
        raise RuntimeError(
            "Mega-ASR router input must be rank 3, got "
            f"{input_meta.shape!r}."
        )
    output_meta = outputs[0]
    if any(is_dynamic_dim(dim) for dim in output_meta.shape) or math.prod(
        int(dim) for dim in output_meta.shape
    ) != 1:
        raise RuntimeError(
            "Mega-ASR router output must contain one static probability, got "
            f"{output_meta.shape!r}."
        )

    probability_buffer = _ort_from_numpy(filled_for(output_meta))
    binding = session.io_binding()
    binding.bind_ortvalue_output(output_meta.name, probability_buffer)
    return {
        "binding": binding,
        "input_meta": input_meta,
        "probability_buffer": probability_buffer,
        "session": session,
    }


def _route_audio(
    router_runtime: dict[str, object] | None,
    audio_path: str | Path,
    sample_rate: int,
    threshold: float,
    noise_seed: int,
) -> tuple[bool, float | None, str]:
    if router_runtime is None:
        return True, None, "default"
    router_input_meta = router_runtime["input_meta"]
    waveform = _load_router_waveform(audio_path, sample_rate, noise_seed)
    waveform = array_for(
        router_input_meta,
        waveform,
        axes=_sequence_axes(router_input_meta, waveform.size),
    )
    waveform_buffer = _ort_from_numpy(waveform)
    binding = router_runtime["binding"]
    binding.clear_binding_inputs()
    _bind_ortvalue_input(binding, router_input_meta.name, waveform_buffer)
    _run(router_runtime["session"], binding)
    probability_values = router_runtime["probability_buffer"].numpy()
    if probability_values.size != 1:
        raise RuntimeError(
            "Mega-ASR router output changed after session initialization: "
            f"got {probability_values.shape!r}."
        )
    degraded_probability = float(probability_values.reshape(-1)[0])
    if not math.isfinite(degraded_probability) or not 0.0 <= degraded_probability <= 1.0:
        raise RuntimeError(
            "Mega-ASR router must return a finite degraded probability in [0, 1], "
            f"got {degraded_probability!r}."
        )
    return degraded_probability >= threshold, degraded_probability, "router"


def _build_variant_runtime(
    model_files: dict[str, str],
    *,
    strategy: str,
    use_direct_penalty: bool,
    is_sampling: bool,
    tokenizer: AutoTokenizer,
    task_prompts: list[str],
    language_prompts: list[str],
    supported_languages: dict[str, dict[str, object]],
) -> dict[str, object]:
    shared_path = onnx_folder / model_files["shared_initializers"]
    embed_shared_path = (
        shared_path
        if references_shared_bundle(
            onnx_folder / model_files["embed"],
            shared_path,
        )
        else None
    )
    embed_session = _make_session(
        onnx_folder / model_files["embed"], embed_shared_path
    )
    graph_pair = {
        "greedy": ("prefill_greedy", "decode_greedy"),
        "penalty_greedy": ("prefill_penalty_greedy", "decode_penalty_greedy"),
        "sampling": ("prefill_sampling", "decode_sampling"),
    }
    prefill_role, decode_role = graph_pair[strategy]
    prefill_session = _make_session(
        onnx_folder / model_files[prefill_role], shared_path
    )
    decode_session = _make_session(
        onnx_folder / model_files[decode_role], shared_path
    )
    prefill_plan = _plan_merged_io(prefill_session, strategy, False)
    decode_plan = _plan_merged_io(decode_session, strategy, True)
    kv_num_tensors = len(prefill_plan["state_inputs"])
    if kv_num_tensors % 2 or len(decode_plan["state_inputs"]) != kv_num_tensors:
        raise RuntimeError(
            "Mega-ASR prefill and decode graphs must expose matching key/value "
            "cache tensors."
        )
    num_layers = kv_num_tensors // 2
    embed_inputs = list(embed_session.get_inputs())
    embed_outputs = list(embed_session.get_outputs())
    if len(embed_inputs) != 1 or len(embed_outputs) != 1:
        raise RuntimeError(
            "Mega-ASR embedding graph must expose exactly one input and one "
            "output."
        )
    embed_input_meta = embed_inputs[0]
    embed_output_meta = embed_outputs[0]
    audio_meta = prefill_plan["audio_input"]
    audio_dtype = numpy_dtype(audio_meta)
    audio_sample_dim = _sequence_length_dimension(audio_meta)

    prompt_embeddings: list[onnxruntime.OrtValue] = []
    for prompt in task_prompts:
        if prompt:
            prompt_embeddings.append(
                _persistent_embed(
                    embed_session,
                    build_query_prompt_ids(tokenizer, prompt),
                    embed_input_meta,
                    embed_output_meta,
                )
            )
        else:
            prompt_embeddings.append(
                _ort_from_numpy(
                    filled_for(
                        prefill_plan["query_input"],
                        axes=_sequence_axes(prefill_plan["query_input"], 0),
                    )
                )
            )

    language_tail_embeddings: list[onnxruntime.OrtValue] = []
    for language in language_prompts:
        if language:
            _, language_entry = resolve_supported_language(
                supported_languages, language
            )
            language_tail_embeddings.append(
                _persistent_embed(
                    embed_session,
                    language_entry["prompt_token_ids"],
                    embed_input_meta,
                    embed_output_meta,
                )
            )
        else:
            language_tail_embeddings.append(
                _ort_from_numpy(
                    filled_for(
                        prefill_plan["language_tail_input"],
                        axes=_sequence_axes(
                            prefill_plan["language_tail_input"], 0
                        ),
                    )
                )
            )

    history_len_zero = _ort_from_numpy(
        scalar_for(prefill_plan["sequence_length_input"], 0)
    )
    hidden_states_buffer = _ort_from_numpy(
        filled_for(embed_output_meta, axes=_sequence_axes(embed_output_meta, 1))
    )
    decode_embed_binding = embed_session.io_binding()
    decode_embed_binding.bind_ortvalue_output(
        embed_output_meta.name, hidden_states_buffer
    )
    decode_bindings = [decode_session.io_binding(), decode_session.io_binding()]
    direct_penalty_controls = (
        _scalar_control_values(
            decode_plan["control_inputs"],
            (REPEAT_PENALTY, PENALTY_RANGE),
        )
        if use_direct_penalty
        else []
    )
    decode_sampling_scalars = (
        _sampling_scalar_values(decode_plan["control_inputs"])
        if is_sampling
        else []
    )
    for binding in decode_bindings:
        _bind_ortvalue_input(
            binding,
            decode_plan["hidden_states_input"].name,
            hidden_states_buffer,
        )
        for input_meta, value in direct_penalty_controls + decode_sampling_scalars:
            _bind_ortvalue_input(binding, input_meta.name, value)

    return {
        "audio_dtype": audio_dtype,
        "audio_meta": audio_meta,
        "audio_sample_dim": audio_sample_dim,
        "decode_bindings": decode_bindings,
        "decode_embed_binding": decode_embed_binding,
        "decode_plan": decode_plan,
        "decode_session": decode_session,
        "embed_input_meta": embed_input_meta,
        "embed_output_meta": embed_output_meta,
        "embed_session": embed_session,
        "history_len_zero": history_len_zero,
        "kv_num_tensors": kv_num_tensors,
        "language_tail_embeddings": language_tail_embeddings,
        "num_layers": num_layers,
        "prefill_plan": prefill_plan,
        "prefill_session": prefill_session,
        "prompt_embeddings": prompt_embeddings,
    }


# ============================================================================
# Merged runtime
# ============================================================================
def main() -> None:
    print("Starting merged ONNX Runtime inference ...\n")
    metadata = _load_metadata(onnx_folder)
    routing_metadata = metadata_json_object(metadata, "mega_asr_routing")
    variant_model_files = routing_metadata["variant_model_files"]
    if set(variant_model_files) != set(VARIANT_MODEL_FILE_NAMES):
        raise RuntimeError(
            "Mega-ASR metadata must describe exactly the base and mega variants."
        )
    for variant, expected_files in VARIANT_MODEL_FILE_NAMES.items():
        actual_files = variant_model_files[variant]
        if actual_files != expected_files:
            raise RuntimeError(
                f"Mega-ASR metadata has an incompatible {variant!r} artifact registry."
            )
    base_variant = str(routing_metadata["base_variant"])
    mega_variant = str(routing_metadata["mega_variant"])
    if base_variant not in variant_model_files or mega_variant not in variant_model_files:
        raise RuntimeError("Mega-ASR routing metadata references an unknown variant.")
    router_model_name = str(routing_metadata["router_model"])
    router_sample_rate = int(routing_metadata["router_sample_rate"])
    router_threshold = float(routing_metadata["router_threshold"])
    if router_model_name != ROUTER_MODEL_NAME:
        raise RuntimeError(f"Unexpected Mega-ASR router artifact: {router_model_name!r}.")

    audio_pcm_scale = int(metadata["audio_pcm_scale"])
    sample_rate = int(metadata["sample_rate"])
    max_seq_len = int(metadata["max_seq_len"])
    special_token_ids = load_special_token_ids(metadata)
    supported_languages = load_supported_languages(metadata)
    special_ids_by_role = {
        role: value if isinstance(value, list) else [value]
        for role, value in special_token_ids.items()
    }
    stop_token_set = set(special_ids_by_role["stop"])

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), trust_remote_code=True
    )
    strategy, use_direct_penalty = _resolve_strategy()
    is_sampling = strategy == "sampling"
    print(f"  Strategy         : {strategy}")
    if is_sampling:
        print(
            "  Sampling         : "
            f"temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, "
            f"repetition_penalty={SAMPLING_REPETITION_PENALTY}"
        )

    test_audio_list = [test_audio] if isinstance(test_audio, str) else list(test_audio)
    configured_task_prompts = TASK_PROMPTS
    configured_language_prompts = LANGUAGE_PROMPTS
    if len(configured_task_prompts) == 1:
        task_prompts = configured_task_prompts * len(test_audio_list)
    elif len(configured_task_prompts) == len(test_audio_list):
        task_prompts = configured_task_prompts
    else:
        task_prompts = configured_task_prompts
    if len(configured_language_prompts) == 1:
        language_prompts = configured_language_prompts * len(test_audio_list)
    elif len(configured_language_prompts) == len(test_audio_list):
        language_prompts = configured_language_prompts
    else:
        language_prompts = configured_language_prompts
    configured_language_names = [
        (
            str(resolve_supported_language(supported_languages, language)[1]["name"])
            if language.strip()
            else None
        )
        for language in language_prompts
    ]

    router_runtime = None
    if USE_AUDIO_QUALITY_ROUTING:
        router_runtime = _build_router_runtime(onnx_folder / router_model_name)
        print("  Audio router     : enabled")
    else:
        print("  Audio router     : disabled; Mega/LoRA is the official default")

    runtime_by_variant: dict[str, dict[str, object]] = {}

    def _runtime_for_variant(variant: str) -> dict[str, object]:
        runtime = runtime_by_variant.get(variant)
        if runtime is not None:
            return runtime
        print(f"Loading {variant} ASR sessions ...")
        runtime = _build_variant_runtime(
            dict(variant_model_files[variant]),
            strategy=strategy,
            use_direct_penalty=use_direct_penalty,
            is_sampling=is_sampling,
            tokenizer=tokenizer,
            task_prompts=task_prompts,
            language_prompts=language_prompts,
            supported_languages=supported_languages,
        )
        runtime_by_variant[variant] = runtime
        print(
            f"  {variant.title()} KV layout : {runtime['num_layers']} layers, "
            f"{runtime['kv_num_tensors']} leading tensors"
        )
        print(
            f"  {variant.title()} Providers : "
            f"{runtime['decode_session'].get_providers()}"
        )
        return runtime

    for input_index, (
        system_prompt,
        language_prompt,
        configured_language_name,
        test_path,
    ) in enumerate(zip(
        task_prompts,
        language_prompts,
        configured_language_names,
        test_audio_list,
    )):
        noise_seed = TEST_NOISE_SEED + input_index
        use_lora, degraded_prob, route_source = _route_audio(
            router_runtime,
            test_path,
            router_sample_rate,
            router_threshold,
            noise_seed,
        )
        selected_variant = mega_variant if use_lora else base_variant
        runtime = _runtime_for_variant(selected_variant)
        audio_dtype = runtime["audio_dtype"]
        audio_meta = runtime["audio_meta"]
        audio_sample_dim = runtime["audio_sample_dim"]
        decode_bindings = runtime["decode_bindings"]
        decode_embed_binding = runtime["decode_embed_binding"]
        decode_plan = runtime["decode_plan"]
        decode_session = runtime["decode_session"]
        embed_input_meta = runtime["embed_input_meta"]
        embed_session = runtime["embed_session"]
        history_len_zero = runtime["history_len_zero"]
        kv_num_tensors = runtime["kv_num_tensors"]
        language_tail_embed = runtime["language_tail_embeddings"][input_index]
        prefill_plan = runtime["prefill_plan"]
        prefill_session = runtime["prefill_session"]
        prompt_embed = runtime["prompt_embeddings"][input_index]
        audio_segment = AudioSegment.from_file(test_path)
        audio_pcm = np.asarray(
            audio_segment.set_channels(1)
            .set_frame_rate(sample_rate)
            .get_array_of_samples(),
            dtype=np.int16,
        )
        if TEST_NOISE_SNR_DB is not None:
            audio_pcm = np.rint(_add_test_noise(
                audio_pcm,
                seed=noise_seed,
                lower_bound=float(np.iinfo(np.int16).min),
                upper_bound=float(np.iinfo(np.int16).max),
                sample_rate=sample_rate,
            )).astype(np.int16)

        original_audio_len = len(audio_pcm)
        if not is_dynamic_dim(audio_sample_dim):
            audio_pcm = audio_pcm[:int(audio_sample_dim)]
        audio = prepare_audio_input(
            audio_pcm,
            audio_dtype,
            audio_pcm_scale,
        )
        audio = array_for(
            audio_meta,
            audio,
            axes=_sequence_axes(audio_meta, audio.size),
        )
        audio_value = _ort_from_numpy(audio)

        print(
            f"\nTest audio : {test_path}   "
            f"({original_audio_len / sample_rate:.2f} s)"
        )
        print(
            "  Route           : "
            f"{'Mega/LoRA' if use_lora else 'Base'} ({route_source})"
        )
        if degraded_prob is not None:
            print(f"  Degraded prob   : {degraded_prob:.6f}")
        if system_prompt:
            print(f"  System prompt   : {system_prompt}")
        if language_prompt:
            print(f"  Language prompt : {language_prompt}")
        if TEST_NOISE_SNR_DB is not None:
            print(f"  Test noise      : {TEST_NOISE_SNR_DB:.1f} dB SNR")
            print(f"  Test speech     : {TEST_SPEECH_GAIN_DB:.1f} dB gain")
        print("-" * 70)

        # One launch owns audio encoding, the optional language tail, rotary/mask,
        # transformer prefill, and first-token selection for the chosen strategy.
        start_time = time.time()
        prefill_binding = prefill_session.io_binding()
        prefill_state_values = []
        for input_meta in prefill_plan["state_inputs"]:
            value = _ort_from_numpy(_empty_sequence(input_meta))
            prefill_state_values.append(value)
            _bind_ortvalue_input(prefill_binding, input_meta.name, value)
        _bind_ortvalue_input(
            prefill_binding, prefill_plan["audio_input"].name, audio_value
        )
        _bind_ortvalue_input(
            prefill_binding,
            prefill_plan["query_input"].name,
            prompt_embed,
        )
        _bind_ortvalue_input(
            prefill_binding,
            prefill_plan["language_tail_input"].name,
            language_tail_embed,
        )
        _bind_ortvalue_input(
            prefill_binding,
            prefill_plan["sequence_length_input"].name,
            history_len_zero,
        )
        empty_history_values = []
        if prefill_plan["save_inputs"]:
            for input_meta in prefill_plan["save_inputs"]:
                empty_history = _ort_from_numpy(_empty_sequence(input_meta))
                empty_history_values.append(empty_history)
                _bind_ortvalue_input(
                    prefill_binding,
                    input_meta.name,
                    empty_history,
                )
        prefill_sampling_scalars = (
            _sampling_scalar_values(prefill_plan["control_inputs"])
            if is_sampling
            else []
        )
        for input_meta, value in prefill_sampling_scalars:
            _bind_ortvalue_input(prefill_binding, input_meta.name, value)
        _bind_device_outputs(prefill_binding, prefill_plan["output_names"])

        prefill_start = time.time()
        _run(prefill_session, prefill_binding)
        prefill_elapsed = time.time() - prefill_start
        prefill_outputs = prefill_binding.get_outputs()
        prefill_positions = {
            name: index
            for index, name in enumerate(prefill_plan["output_names"])
        }

        state_values = prefill_outputs[:kv_num_tensors]
        kv_seq_len = prefill_outputs[
            prefill_positions[prefill_plan["sequence_length_output"].name]
        ]
        ids_len_value = int(kv_seq_len.numpy().flat[0])
        generation_limit = max(max_seq_len - 10 - ids_len_value, 0)
        print(
            f"  Encoder+prefill done ({prefill_elapsed:.3f}s), "
            f"prompt tokens={ids_len_value}"
        )
        if generation_limit == 0:
            print("  No decoder context remains; skipping generation.")
            continue
        selected_token = int(
            prefill_outputs[
                prefill_positions[prefill_plan["max_output"].name]
            ].numpy().flat[0]
        )
        next_token = prefill_outputs[
            prefill_positions[prefill_plan["max_output"].name]
        ]
        if strategy in ("penalty_greedy", "sampling"):
            save_id = prefill_outputs[
                prefill_positions[prefill_plan["save_output"].name]
            ]
        else:
            save_id = None

        generated_tokens: list[int] = []
        generated_count = 0
        final_save_id = save_id
        if selected_token not in stop_token_set:
            generated_count = 1
            if strategy == "greedy":
                generated_tokens.append(selected_token)

        decode_positions = {
            name: index for index, name in enumerate(decode_plan["output_names"])
        }
        decode_steps = 0
        decode_start = time.time()

        while (
            generated_count < generation_limit
            and selected_token not in stop_token_set
        ):
            # Standalone Embed is retained because it also serves prompt/language
            # embedding; only the transformer stage is one merged run per token.
            _bind_ortvalue_input(
                decode_embed_binding,
                embed_input_meta.name,
                next_token,
            )
            _run(embed_session, decode_embed_binding)

            binding = decode_bindings[decode_steps & 1]
            for input_meta, value in zip(decode_plan["state_inputs"], state_values):
                _bind_ortvalue_input(binding, input_meta.name, value)
            _bind_ortvalue_input(
                binding,
                decode_plan["sequence_length_input"].name,
                kv_seq_len,
            )
            for input_meta in decode_plan["save_inputs"]:
                _bind_ortvalue_input(binding, input_meta.name, save_id)

            # Device-bound outputs are fresh for this binding invocation.  They
            # become the peer binding's inputs on the next step (ping-pong).
            binding.clear_binding_outputs()
            _bind_device_outputs(binding, decode_plan["output_names"])
            _run(decode_session, binding)
            outputs = binding.get_outputs()

            state_values = outputs[:kv_num_tensors]
            kv_seq_len = outputs[
                decode_positions[decode_plan["sequence_length_output"].name]
            ]
            selected_token = int(
                outputs[
                    decode_positions[decode_plan["max_output"].name]
                ].numpy().flat[0]
            )
            next_token = outputs[
                decode_positions[decode_plan["max_output"].name]
            ]
            if strategy in ("penalty_greedy", "sampling"):
                save_id = outputs[
                    decode_positions[decode_plan["save_output"].name]
                ]
                final_save_id = save_id

            if selected_token not in stop_token_set:
                generated_count += 1
                if strategy == "greedy":
                    generated_tokens.append(selected_token)
            decode_steps += 1

        decode_elapsed = time.time() - decode_start

        if strategy in ("penalty_greedy", "sampling"):
            generated_tokens = []
            if final_save_id is not None:
                for token in final_save_id.numpy().reshape(-1):
                    token = int(token)
                    if token in stop_token_set:
                        break
                    generated_tokens.append(token)

        raw_result = tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        ).strip()
        selected_language, asr_result = parse_asr_output(
            raw_result, configured_language_name
        )
        route_result = {
            "text": asr_result,
            "use_lora": use_lora,
            "degraded_prob": degraded_prob,
            "route_source": route_source,
        }

        total_elapsed = time.time() - start_time
        rtf = total_elapsed / (original_audio_len / sample_rate)
        decode_rate = decode_steps / decode_elapsed
        if selected_language:
            print(f"\nLanguage          : {selected_language}")
        print(f"\nRoute result:\n  {route_result}")
        print(f"\nTranscription:\n  {asr_result}")
        print(
            f"\nEncoder+prefill: {prefill_elapsed:.3f}s (1 merged launch)"
            f"\nMerged decode  : {decode_rate:.2f} token/s "
            f"({decode_steps} transformer launches)"
            f"\nRTF            : {rtf:.3f}   total {total_elapsed:.2f}s"
        )
        print("-" * 70)


if __name__ == "__main__":
    main()
