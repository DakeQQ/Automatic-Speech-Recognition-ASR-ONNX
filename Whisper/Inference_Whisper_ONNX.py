"""Run the merged/shared-initializer Whisper ONNX pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Example_Audio import model_audio_cases
from ORT_IO import (
    array_for,
    filled_for,
    is_dynamic_dim,
    metadata_by_name,
    load_special_token_ids,
    load_supported_languages,
    numpy_dtype,
    resolve_supported_language,
    scalar_for,
)
from Shared_Merged import DEFAULT_MODEL_FILE_NAMES, attach_shared_initializers


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the merged Whisper ONNX pipeline.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "Whisper_Optimized",
        help="Folder containing merged graphs and Whisper_SharedInitializers.onnx(.data).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional tokenizer/config directory; defaults to tokenizer inside the model folder.",
    )
    return parser.parse_args()


ARGS = _parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
TOKENIZER_DIR = (
    ARGS.tokenizer_path.expanduser().resolve()
    if ARGS.tokenizer_path is not None
    else ONNX_FOLDER / "tokenizer"
)
METADATA_PATH = ONNX_FOLDER / "ASR_Metadata.onnx"


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally limited to model/tokenizer paths.
# Edit this section for all decoding, language, audio, demo, and runtime behavior.
USE_SAMPLING = False
TEMPERATURE = 0.8
TOP_K = 10
TOP_P = 0.95
SAMPLING_REPETITION_PENALTY = 1.0

# Penalty-greedy uses this legacy direct multiplier in (0, 1].
REPEAT_PENALTY = 0.8  # 1.0 selects greedy; another value selects penalty-greedy.
PENALTY_RANGE = 20
REMOVE_REPEATED_PARTS = False

TARGET_LANGUAGE = "en"
TASK = "transcribe"
DETECT_LANGUAGE = True

NO_SPEECH_DETECTION = True
NO_SPEECH_THRESHOLD = 0.6

SLIDING_WINDOW = 0
USE_NORMALISE_AUDIO = False


# ============================================================================
# ONNX Runtime configuration
# ============================================================================
ORT_Accelerate_Providers = []    # ["CUDAExecutionProvider", OpenVINOExecutionProvider", "DmlExecutionProvider"]
ORT_LOG = False
ORT_FP16 = False
MAX_THREADS = 0
DEVICE_ID = 0


def prepare_audio_input(
    audio_int16: np.ndarray,
    target_dtype: np.dtype,
    *,
    audio_pcm_scale: int,
    target_rms: float = 4096.0,
):
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


def remove_repeated_parts(ids, repeat_words_threshold, ids_len):
    if ids_len <= repeat_words_threshold:
        return ids
    side_left = repeat_words_threshold // 2
    side_right = side_left + 1
    boundary = ids_len - side_left
    for i in range(side_left, boundary):
        for j in range(i + repeat_words_threshold, boundary):
            if all(ids[j + k] == ids[i + k] for k in range(-side_left, side_right)):
                return ids[:j - side_left]
    return ids


def _build_run_options(silent):
    options = onnxruntime.RunOptions()
    options.log_severity_level = 4 if silent else 0
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def _build_session_options():
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
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else ""
        ),
    }
    for key, value in entries.items():
        options.add_session_config_entry(key, value)
    return options


def _resolve_provider():
    if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
        return "cpu", C.OrtDevice.cpu(), [{
            "device_type": "CPU",
            "precision": "ACCURACY",
            "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }]
    if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
        return "cuda", C.OrtDevice.cuda(), [{
            "device_id": DEVICE_ID,
            "gpu_mem_limit": 24 * (1024 ** 3),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "sdpa_kernel": "2",
            "use_tf32": "1",
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": "0",
            "enable_cuda_graph": "0",
        }]
    if "DmlExecutionProvider" in ORT_Accelerate_Providers:
        return "dml", C.OrtDevice.dml(), [{
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
            "enable_graph_serialization": "false",
        }]
    return "cpu", C.OrtDevice.cpu(), None


RUN_OPTIONS = _build_run_options(silent=not ORT_LOG)
DEVICE_TYPE, ORT_DEVICE_TYPE, PROVIDER_OPTIONS = _resolve_provider()
ORT_DEVICE = C.OrtDevice(ORT_DEVICE_TYPE, C.OrtDevice.default_memory(), DEVICE_ID)
PROVIDERS = ORT_Accelerate_Providers or ["CPUExecutionProvider"]
DISABLED_OPTIMIZERS = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None
)


def _make_session(path: Path):
    return onnxruntime.InferenceSession(
        str(path),
        sess_options=_build_session_options(),
        providers=PROVIDERS,
        provider_options=PROVIDER_OPTIONS,
        disabled_optimizers=DISABLED_OPTIMIZERS,
    )


def _make_merged_session(path: Path, shared_path: Path):
    options = _build_session_options()
    shared_refs = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=PROVIDERS,
        provider_options=PROVIDER_OPTIONS,
        disabled_optimizers=DISABLED_OPTIMIZERS,
    )
    # The memmaps and OrtValues back add_initializer() and must outlive the session.
    session._native_llm_shared_initializers = shared_refs
    return session


def _run(session, binding):
    session.run_with_iobinding(binding, run_options=RUN_OPTIONS)


def _in_names(session):
    return [meta.name for meta in session.get_inputs()]


def _out_names(session):
    return [meta.name for meta in session.get_outputs()]


def _ort_value(array, device=None):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device or DEVICE_TYPE, DEVICE_ID
    )


def _bind_device_outputs(binding, names):
    for name in names:
        binding._iobinding.bind_output(name, ORT_DEVICE)


def _load_metadata(path: Path):
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.log_severity_level = 4
    session = onnxruntime.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session.get_modelmeta().custom_metadata_map or {}


MODEL_META = _load_metadata(METADATA_PATH)


MODEL_FILES = DEFAULT_MODEL_FILE_NAMES
AUDIO_PCM_SCALE = int(MODEL_META["audio_pcm_scale"])
MAX_SEQ_LEN = int(MODEL_META["max_seq_len"])
SAMPLE_RATE = int(MODEL_META["sample_rate"])
SPECIAL_TOKEN_IDS = load_special_token_ids(MODEL_META)
SUPPORTED_LANGUAGES = load_supported_languages(MODEL_META)

if USE_SAMPLING:
    STRATEGY = "sampling"
    USE_PENALTY = False
else:
    USE_PENALTY = REPEAT_PENALTY != 1.0
    STRATEGY = "penalty_greedy" if USE_PENALTY else "greedy"
GRAPH_PAIRS = {
    "greedy": (MODEL_FILES["probe_prefill_greedy"], MODEL_FILES["prefill_greedy"], MODEL_FILES["decode_greedy"]),
    "penalty_greedy": (
        MODEL_FILES["probe_prefill_penalty_greedy"], MODEL_FILES["prefill_penalty_greedy"], MODEL_FILES["decode_penalty_greedy"]
    ),
    "sampling": (MODEL_FILES["probe_prefill_sampling"], MODEL_FILES["prefill_sampling"], MODEL_FILES["decode_sampling"]),
}

PROBE_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][0]
PREFILL_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][1]
DECODE_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][2]
SHARED_PATH = ONNX_FOLDER / MODEL_FILES["shared_initializers"]
NO_SPEECH_PATH = ONNX_FOLDER / MODEL_FILES["no_speech"]


print("\nLoading merged Whisper sessions and one shared initializer mmap ...")
PROBE_SESSION = _make_merged_session(PROBE_PATH, SHARED_PATH)
PREFILL_SESSION = _make_merged_session(PREFILL_PATH, SHARED_PATH)
DECODE_SESSION = _make_merged_session(DECODE_PATH, SHARED_PATH)
NO_SPEECH_SESSION = _make_session(NO_SPEECH_PATH) if NO_SPEECH_DETECTION else None
print(f"Usable Providers: {DECODE_SESSION.get_providers()}")
print(
    f"Decoder strategy: {STRATEGY}; sessions=3 (Encoder+probe + cached prefill + decode); "
    "decode launches/token=1; shared initializer blob=1."
)


def _plan_merged_io(session, strategy, is_decode):
    input_names = _in_names(session)
    output_names = _out_names(session)
    state_inputs = []
    for name in input_names:
        if not name.startswith("in_de_"):
            break
        state_inputs.append(name)
    state_outputs = []
    for name in output_names:
        if not name.startswith("out_de_"):
            break
        state_outputs.append(name)

    if strategy == "sampling":
        max_output = "sampling_sampled_id"
        save_output = "sampling_save_id_out"
    elif strategy == "penalty_greedy":
        max_output = "greedy_max_logits_idx"
        save_output = "greedy_save_id_out"
    else:
        max_output = "argmax_max_logits_idx"
        save_output = None
    next_token_output = max_output

    kv_seq_output = "decode_kv_seq_len_next" if is_decode else "prefill_kv_seq_len"
    cross_outputs = [name for name in output_names if name.startswith(("encoder_en_key_", "encoder_en_value_"))]

    cross_inputs = [name for name in input_names if name.startswith(("en_key_", "en_value_"))]
    probe = "audio" in input_names
    cross_source = cross_outputs if probe else cross_inputs
    key_cross = [name for name in cross_source if "en_key_" in name]
    num_layers = len(key_cross)

    if strategy == "sampling":
        save_inputs = ["sampling_previous_ids"]
    elif strategy == "penalty_greedy":
        save_inputs = ["greedy_save_id_in"]
        if is_decode:
            save_inputs.insert(0, "penalty_save_id_in")
    else:
        save_inputs = []

    sampling_inputs = [
        "sampling_temperature",
        "sampling_top_k",
        "sampling_top_p",
        "sampling_repetition_penalty",
    ]
    if strategy != "sampling":
        sampling_inputs = []
    return {
        "inputs": input_names,
        "outputs": output_names,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "cross_inputs": cross_inputs,
        "cross_outputs": cross_outputs,
        "probe": probe,
        "token_input": "embed_input_ids",
        "kv_seq_input": "decode_kv_seq_len" if is_decode else None,
        "kv_seq_output": kv_seq_output,
        "raw_logits_output": "logits" if not is_decode else None,
        "max_output": max_output,
        "next_token_output": next_token_output,
        "save_inputs": save_inputs,
        "save_output": save_output,
        "sampling_inputs": sampling_inputs,
        "num_layers": num_layers,
    }


PROBE_PLAN = _plan_merged_io(PROBE_SESSION, STRATEGY, is_decode=False)
PREFILL_PLAN = _plan_merged_io(PREFILL_SESSION, STRATEGY, is_decode=False)
DECODE_PLAN = _plan_merged_io(DECODE_SESSION, STRATEGY, is_decode=True)
KV_NUM_TENSORS = len(DECODE_PLAN["state_inputs"])


PREFILL_INPUT_META = metadata_by_name(PREFILL_SESSION.get_inputs())
PROBE_INPUT_META = metadata_by_name(PROBE_SESSION.get_inputs())
DECODE_INPUT_META = metadata_by_name(DECODE_SESSION.get_inputs())
DECODE_OUTPUT_META = metadata_by_name(DECODE_SESSION.get_outputs())
DECODE_OUTPUT_INDEX = {name: index for index, name in enumerate(DECODE_PLAN["outputs"])}
PREFILL_OUTPUT_INDEX = {name: index for index, name in enumerate(PREFILL_PLAN["outputs"])}
PROBE_OUTPUT_INDEX = {name: index for index, name in enumerate(PROBE_PLAN["outputs"])}


def _self_kv_sequence_axis(meta):
    candidates = [
        axis
        for axis, dim in enumerate(meta.shape)
        if axis != 0 and is_dynamic_dim(dim)
    ]
    return candidates[-1]


if NO_SPEECH_SESSION is not None:
    NO_SPEECH_INPUT_META = NO_SPEECH_SESSION.get_inputs()[0]
else:
    NO_SPEECH_INPUT_META = None


def _empty_self_kv(meta):
    sequence_axis = _self_kv_sequence_axis(meta)
    return filled_for(meta, axes={0: 1, sequence_axis: 0})


def _bind_typed(binding, name, value, input_meta, keepalive, device=None, axes=None):
    ort_value = _ort_value(array_for(input_meta[name], value, axes=axes), device)
    keepalive.append(ort_value)
    binding.bind_ortvalue_input(name, ort_value)
    return ort_value


def _prefill(input_ids, cross_kv_by_name):
    binding = PREFILL_SESSION.io_binding()
    keepalive = []
    kv_device = "cpu" if DEVICE_TYPE == "dml" else DEVICE_TYPE
    for name in PREFILL_PLAN["state_inputs"]:
        sequence_axis = _self_kv_sequence_axis(PREFILL_INPUT_META[name])
        _bind_typed(
            binding,
            name,
            _empty_self_kv(PREFILL_INPUT_META[name]),
            PREFILL_INPUT_META,
            keepalive,
            kv_device,
            axes={0: 1, sequence_axis: 0},
        )
    for name in PREFILL_PLAN["cross_inputs"]:
        binding.bind_ortvalue_input(name, cross_kv_by_name[name])
    _bind_typed(
        binding,
        PREFILL_PLAN["token_input"],
        input_ids,
        PREFILL_INPUT_META,
        keepalive,
        axes={0: 1, 1: input_ids.shape[-1]},
    )
    _bind_typed(
        binding,
        "prefill_ids_len",
        scalar_for(PREFILL_INPUT_META["prefill_ids_len"], input_ids.shape[-1]),
        PREFILL_INPUT_META,
        keepalive,
        axes={0: 1},
    )
    _bind_typed(
        binding,
        "prefill_history_len",
        scalar_for(PREFILL_INPUT_META["prefill_history_len"], 0),
        PREFILL_INPUT_META,
        keepalive,
        axes={0: 1},
    )
    for name in PREFILL_PLAN["save_inputs"]:
        _bind_typed(
            binding,
            name,
            filled_for(PREFILL_INPUT_META[name], axes={0: 1, 1: 0}),
            PREFILL_INPUT_META,
            keepalive,
            axes={0: 1, 1: 0},
        )
    _bind_sampling_controls(binding, PREFILL_PLAN, PREFILL_INPUT_META, keepalive)
    _bind_device_outputs(binding, PREFILL_PLAN["outputs"])
    _run(PREFILL_SESSION, binding)
    return binding.get_outputs()


def _probe_prefill(audio_buffer, audio_window, input_ids):
    binding = PROBE_SESSION.io_binding()
    keepalive = []
    audio_buffer.update_inplace(array_for(
        PROBE_INPUT_META["audio"],
        audio_window,
        axes={0: 1, 1: 1, 2: audio_window.shape[2]},
    ))
    binding.bind_ortvalue_input("audio", audio_buffer)
    kv_device = "cpu" if DEVICE_TYPE == "dml" else DEVICE_TYPE
    for name in PROBE_PLAN["state_inputs"]:
        sequence_axis = _self_kv_sequence_axis(PROBE_INPUT_META[name])
        _bind_typed(
            binding,
            name,
            _empty_self_kv(PROBE_INPUT_META[name]),
            PROBE_INPUT_META,
            keepalive,
            kv_device,
            axes={0: 1, sequence_axis: 0},
        )
    _bind_typed(
        binding,
        PROBE_PLAN["token_input"],
        input_ids,
        PROBE_INPUT_META,
        keepalive,
        axes={0: 1, 1: input_ids.shape[-1]},
    )
    _bind_typed(
        binding,
        "prefill_ids_len",
        scalar_for(PROBE_INPUT_META["prefill_ids_len"], input_ids.shape[-1]),
        PROBE_INPUT_META,
        keepalive,
        axes={0: 1},
    )
    _bind_typed(
        binding,
        "prefill_history_len",
        scalar_for(PROBE_INPUT_META["prefill_history_len"], 0),
        PROBE_INPUT_META,
        keepalive,
        axes={0: 1},
    )
    for name in PROBE_PLAN["save_inputs"]:
        _bind_typed(
            binding,
            name,
            filled_for(PROBE_INPUT_META[name], axes={0: 1, 1: 0}),
            PROBE_INPUT_META,
            keepalive,
            axes={0: 1, 1: 0},
        )
    _bind_sampling_controls(binding, PROBE_PLAN, PROBE_INPUT_META, keepalive)
    _bind_device_outputs(binding, PROBE_PLAN["outputs"])
    _run(PROBE_SESSION, binding)
    return binding.get_outputs()


def _decode_static_inputs(binding, keepalive):
    if "penalty_penalty_range" in DECODE_PLAN["inputs"]:
        _bind_typed(
            binding,
            "penalty_penalty_range",
            scalar_for(DECODE_INPUT_META["penalty_penalty_range"], PENALTY_RANGE),
            DECODE_INPUT_META,
            keepalive,
            axes={0: 1},
        )
    _bind_sampling_controls(binding, DECODE_PLAN, DECODE_INPUT_META, keepalive)


def _bind_sampling_controls(binding, plan, input_meta, keepalive):
    values = {
        "sampling_temperature": TEMPERATURE,
        "sampling_top_k": TOP_K,
        "sampling_top_p": TOP_P,
        "sampling_repetition_penalty": SAMPLING_REPETITION_PENALTY,
    }
    for name in plan["sampling_inputs"]:
        _bind_typed(
            binding,
            name,
            scalar_for(input_meta[name], values[name]),
            input_meta,
            keepalive,
            axes={0: 1},
        )


def _decode_tokens(prefill_outputs, cross_kv_by_name, generate_limit, stop_tokens):
    state = prefill_outputs[:KV_NUM_TENSORS]
    next_token = prefill_outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["next_token_output"]]]
    kv_seq_len = prefill_outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["kv_seq_output"]]]
    selected = int(prefill_outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["max_output"]]].numpy().reshape(-1)[0])
    saved_ids = (
        prefill_outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["save_output"]]]
        if PREFILL_PLAN["save_output"] is not None else None
    )
    host_tokens = []
    generated_count = 0
    if selected not in stop_tokens and generate_limit > 0:
        generated_count = 1
        if saved_ids is None:
            host_tokens.append(selected)

    bindings = [DECODE_SESSION.io_binding(), DECODE_SESSION.io_binding()]
    static_keepalive = [[], []]
    for binding, keepalive in zip(bindings, static_keepalive):
        for name in DECODE_PLAN["cross_inputs"]:
            binding.bind_ortvalue_input(name, cross_kv_by_name[name])
        _decode_static_inputs(binding, keepalive)

    penalty_input = "penalty_penalty_value"
    penalty_off = penalty_on = None
    if penalty_input in DECODE_PLAN["inputs"]:
        penalty_off = _ort_value(
            scalar_for(DECODE_INPUT_META[penalty_input], 1.0)
        )
        penalty_on = _ort_value(
            scalar_for(DECODE_INPUT_META[penalty_input], REPEAT_PENALTY)
        )
        for keepalive in static_keepalive:
            keepalive.extend((penalty_off, penalty_on))

    decode_steps = 0
    start_time = time.time()
    while generated_count < generate_limit and selected not in stop_tokens:
        binding = bindings[decode_steps & 1]
        binding.bind_ortvalue_input(DECODE_PLAN["token_input"], next_token)
        binding.bind_ortvalue_input(DECODE_PLAN["kv_seq_input"], kv_seq_len)
        for name, value in zip(DECODE_PLAN["state_inputs"], state):
            binding.bind_ortvalue_input(name, value)
        for name in DECODE_PLAN["save_inputs"]:
            binding.bind_ortvalue_input(name, saved_ids)
        if penalty_on is not None:
            binding.bind_ortvalue_input(
                penalty_input,
                penalty_on if generated_count >= PENALTY_RANGE else penalty_off,
            )

        # KV and save_id outputs grow each token. Rebind every graph output to device-auto
        # allocation before reusing either side of the ping-pong; fixed scalar controls remain
        # device-resident and are fed back through their returned OrtValues.
        binding.clear_binding_outputs()
        _bind_device_outputs(binding, DECODE_PLAN["outputs"])
        _run(DECODE_SESSION, binding)
        outputs = binding.get_outputs()
        state = outputs[:KV_NUM_TENSORS]
        next_token = outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["next_token_output"]]]
        kv_seq_len = outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["kv_seq_output"]]]
        selected = int(outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["max_output"]]].numpy().reshape(-1)[0])
        if DECODE_PLAN["save_output"] is not None:
            saved_ids = outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["save_output"]]]
        if selected not in stop_tokens:
            generated_count += 1
            if saved_ids is None:
                host_tokens.append(selected)
        decode_steps += 1

    elapsed = time.time() - start_time
    if saved_ids is not None:
        token_array = saved_ids.numpy()[0]
        host_tokens = []
        for token in token_array:
            token = int(token)
            if token in stop_tokens or len(host_tokens) >= generate_limit:
                break
            host_tokens.append(token)
    return host_tokens, decode_steps, elapsed


# ============================================================================
# Tokenizer, language/task helpers, and probe Encoder setup
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)


ENCODER_INPUT_META = PROBE_INPUT_META["audio"]
AUDIO_NP_DTYPE = numpy_dtype(ENCODER_INPUT_META)

LANGUAGE_TOKEN_TO_CODE = {
    int(entry["token_id"]): code
    for code, entry in SUPPORTED_LANGUAGES.items()
}
LANGUAGE_TOKEN_IDS = np.asarray(list(LANGUAGE_TOKEN_TO_CODE), dtype=np.int64)
tasks = SPECIAL_TOKEN_IDS["tasks"]
START_TOKEN = SPECIAL_TOKEN_IDS["decoder_start"]
TASK_TOKEN = tasks[TASK]
STOP_TOKENS = set(
    SPECIAL_TOKEN_IDS["stop"]
    if isinstance(SPECIAL_TOKEN_IDS["stop"], list)
    else [SPECIAL_TOKEN_IDS["stop"]]
)
NO_TIMESTAMPS_TOKEN = SPECIAL_TOKEN_IDS["no_timestamps"]


def _run_no_speech(logits_value):
    if NO_SPEECH_SESSION is None:
        return None
    binding = NO_SPEECH_SESSION.io_binding()
    binding.bind_ortvalue_input(NO_SPEECH_INPUT_META.name, logits_value)
    _bind_device_outputs(binding, _out_names(NO_SPEECH_SESSION))
    _run(NO_SPEECH_SESSION, binding)
    probability = binding.get_outputs()[0].numpy()
    return float(probability.reshape(-1)[0])


def _decode_asr_tokens(tokens):
    token_array = np.asarray(
        tokens,
        dtype=numpy_dtype(DECODE_OUTPUT_META[DECODE_PLAN["next_token_output"]]),
    )
    if REMOVE_REPEATED_PARTS:
        token_array = remove_repeated_parts(token_array, 3, token_array.shape[-1])
    text, _ = tokenizer._decode_asr(
        [{"tokens": token_array.reshape(1, -1)}],
        return_timestamps=None,
        return_language=None,
        time_precision=0,
    )
    return text


# ============================================================================
# Inference
# ============================================================================
test_audio_cases = model_audio_cases("whisper")
for test_path, demo_language in test_audio_cases:
    print("-" * 106)
    print(f"\nTest Input Audio: {test_path}")
    language = demo_language or TARGET_LANGUAGE
    language, language_entry = resolve_supported_language(
        SUPPORTED_LANGUAGES, language
    )
    language_id = language_entry["token_id"]

    segment = AudioSegment.from_file(test_path).set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    raw_audio = np.asarray(segment.get_array_of_samples(), dtype=np.int16)
    audio_len = raw_audio.size
    audio = prepare_audio_input(
        raw_audio.reshape(1, 1, -1),
        AUDIO_NP_DTYPE,
        audio_pcm_scale=AUDIO_PCM_SCALE,
    )
    audio_sample_dim = ENCODER_INPUT_META.shape[2]
    input_audio_length = (
        audio_len if is_dynamic_dim(audio_sample_dim) else int(audio_sample_dim)
    )
    audio_buffer = _ort_value(filled_for(
        ENCODER_INPUT_META, axes={0: 1, 1: 1, 2: input_audio_length}
    ))
    stride = input_audio_length if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    if audio_len <= input_audio_length:
        windows = 1
    else:
        windows = int(np.ceil((audio_len - input_audio_length) / stride)) + 1
    aligned_length = (windows - 1) * stride + input_audio_length
    if audio.shape[-1] < aligned_length:
        padded_audio = filled_for(
            ENCODER_INPUT_META,
            axes={0: 1, 1: 1, 2: aligned_length},
        )
        padded_audio[..., :audio.shape[-1]] = audio
        audio = padded_audio

    all_tokens = []
    total_decode_steps = 0
    total_decode_time = 0.0
    start_time = time.time()
    no_speech = False

    for window_index in range(windows):
        start_sample = window_index * stride
        audio_window = audio[:, :, start_sample:start_sample + input_audio_length]
        needs_probe = window_index == 0 and (DETECT_LANGUAGE or NO_SPEECH_DETECTION)
        probe_token_values = (
            [[START_TOKEN]]
            if needs_probe
            else [[START_TOKEN, language_id, TASK_TOKEN, NO_TIMESTAMPS_TOKEN]]
        )
        probe_tokens = array_for(
            PROBE_INPUT_META[PROBE_PLAN["token_input"]],
            probe_token_values,
            axes={0: 1, 1: len(probe_token_values[0])},
        )
        probe_outputs = _probe_prefill(audio_buffer, audio_window, probe_tokens)
        cross_kv = {
            decode_name: probe_outputs[PROBE_OUTPUT_INDEX[probe_name]]
            for probe_name, decode_name in zip(PROBE_PLAN["cross_outputs"], PREFILL_PLAN["cross_inputs"])
        }

        # Language/no-speech use the raw SOT logits exposed by the merged prefill graph.  The
        # prefill begin-suppression/head outputs are ignored, so the probability distribution is
        # identical to the split Main output.
        if needs_probe:
            detection_logits = probe_outputs[
                PROBE_OUTPUT_INDEX[PROBE_PLAN["raw_logits_output"]]
            ]
            if DETECT_LANGUAGE:
                logits = detection_logits.numpy().reshape(-1)
                detected_token = int(LANGUAGE_TOKEN_IDS[np.argmax(logits[LANGUAGE_TOKEN_IDS])])
                language = LANGUAGE_TOKEN_TO_CODE.get(detected_token, language)
                language_id = detected_token
                print(f"Detected Language: {language}")
            if NO_SPEECH_DETECTION:
                probability = _run_no_speech(detection_logits)
                print(f"No-Speech Probability: {probability:.3f}")
                if probability >= NO_SPEECH_THRESHOLD:
                    no_speech = True
                    print("Audio classified as silence / non-speech; skipping transcription.")
                    break

        prompt_values = [[START_TOKEN, language_id, TASK_TOKEN, NO_TIMESTAMPS_TOKEN]]
        prompt = array_for(
            PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
            prompt_values,
            axes={0: 1, 1: len(prompt_values[0])},
        )
        prefill_outputs = (
            _prefill(prompt, cross_kv)
            if needs_probe
            else [
                probe_outputs[PROBE_OUTPUT_INDEX[name]]
                for name in PREFILL_PLAN["outputs"]
            ]
        )
        generate_limit = max(0, MAX_SEQ_LEN - prompt.shape[-1])
        window_tokens, decode_steps, decode_time = _decode_tokens(
            prefill_outputs, cross_kv, generate_limit, STOP_TOKENS
        )
        all_tokens.extend(window_tokens)
        total_decode_steps += decode_steps
        total_decode_time += decode_time

    elapsed = time.time() - start_time
    rtf = elapsed / (audio_len / SAMPLE_RATE)
    if no_speech or not all_tokens:
        text = "[no speech detected]" if no_speech else ""
    else:
        text = _decode_asr_tokens(all_tokens)
    decode_rate = total_decode_steps / total_decode_time
    print(
        f"\nASR Result:\n{text}\n\n"
        f"RTF: {rtf:.3f}   ({elapsed:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, "
        f"{len(all_tokens)} tokens; merged decode {decode_rate:.2f} token/s; "
        "1 graph launch/token)"
    )
    print("-" * 106)