"""Run Dolphin v1 with merged prefill/decode graphs and shared weights."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
import sentencepiece as spm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Example_Audio import model_audio_paths
from ORT_IO import (
    array_for,
    filled_for,
    is_dynamic_dim,
    metadata_by_name,
    load_special_token_ids,
    load_supported_languages,
    numpy_dtype,
    resolve_shape,
    resolve_supported_language,
    scalar_for,
)
from Shared_Merged import DEFAULT_MODEL_FILE_NAMES, attach_shared_initializers


VOCAB_FILE_NAME = "vocab_Dolphin.txt"
BPE_MODEL_FILE_NAME = "bpe.model"


def default_onnx_folder():
    """Prefer an optimized v1 bundle, then fall back to the exported v1 bundle."""
    candidates = (
        SCRIPT_DIR / "Dolphin_Optimized",
        SCRIPT_DIR / "Dolphin_ONNX",
    )
    for candidate in candidates:
        if (candidate / "ASR_Metadata.onnx").is_file():
            return candidate
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run merged Dolphin-v1 ONNX inference."
    )
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=default_onnx_folder(),
        help=(
            "Dolphin-v1 folder containing merged graphs and "
            "Dolphin_SharedInitializers.onnx(.data)."
        ),
    )
    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=None,
        help=f"Optional vocabulary text file; defaults to {VOCAB_FILE_NAME} in the model folder.",
    )
    parser.add_argument(
        "--tokenizer-path",
        "--bpe-model-path",
        dest="tokenizer_path",
        type=Path,
        default=None,
        help=f"Optional SentencePiece model; defaults to {BPE_MODEL_FILE_NAME} in the model folder.",
    )
    return parser.parse_args()


ARGS = parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
METADATA_PATH = ONNX_FOLDER / "ASR_Metadata.onnx"


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally limited to model/vocabulary/tokenizer
# paths. Edit this section for all decoding, audio, demo, and runtime behavior.

# Sampling takes precedence; otherwise REPEAT_PENALTY=1.0 selects greedy and
# another valid value selects penalty-greedy.
USE_SAMPLING = False
TEMPERATURE = 0.8
TOP_K = 10
TOP_P = 0.95
REPEAT_PENALTY = 0.8
SAMPLING_REPETITION_PENALTY = 1.0
PENALTY_RANGE = 20

USE_NORMALISE_AUDIO = False
SLIDING_WINDOW = 0
LANGUAGE = "auto-auto"


# ============================================================================
# ONNX Runtime configuration
# ============================================================================
ORT_ACCELERATE_PROVIDERS = []
ORT_LOG = False
ORT_FP16 = False
MAX_THREADS = 0
DEVICE_ID = 0


def prepare_audio_input(
    audio_int16: np.ndarray,
    input_audio_dtype: np.dtype,
    target_rms_pcm: float = 4096.0,
) -> np.ndarray:
    """Convert decoded PCM to the exact input representation used at export."""
    if input_audio_dtype == np.dtype(np.int16):
        if not USE_NORMALISE_AUDIO:
            return np.ascontiguousarray(audio_int16, dtype=input_audio_dtype)
        audio = audio_int16.astype(np.float32)
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= target_rms_pcm / (rms + 1e-7)
            np.clip(audio, -float(AUDIO_PCM_SCALE), float(AUDIO_PCM_SCALE - 1), out=audio)
        return audio.astype(input_audio_dtype)

    audio = audio_int16.astype(np.float32) * np.float32(1.0 / AUDIO_PCM_SCALE)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= np.float32(target_rms_pcm / AUDIO_PCM_SCALE) / (rms + 1e-7)
            np.clip(audio, -1.0, 1.0 - 1.0 / AUDIO_PCM_SCALE, out=audio)
    return audio.astype(input_audio_dtype)


class Tokenizer:
    def __init__(self, filename, bpe_model=None):
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.sp = None
        with open(filename, "r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                token = line.rstrip("\n")
                self.str_to_idx[token] = index
                self.idx_to_str[index] = token
        if bpe_model is not None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(str(bpe_model))

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, index):
        return self.idx_to_str.get(int(index))

    def decode_ids(self, ids):
        tokens = [self.decode(index) for index in ids]
        tokens = [token for token in tokens if token is not None]
        if self.sp is not None:
            return self.sp.DecodePieces(tokens).strip()
        return "".join(tokens).replace("▁", " ").strip()


def build_run_options(silent):
    options = onnxruntime.RunOptions()
    options.log_severity_level = 4 if silent else 0
    options.log_verbosity_level = 4
    options.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return options


def build_session_options():
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


def select_providers():
    available = set(onnxruntime.get_available_providers())
    selected = [
        provider
        for provider in ORT_ACCELERATE_PROVIDERS
        if provider in available
    ]
    return selected or ["CPUExecutionProvider"]


def resolve_provider(providers):
    if "OpenVINOExecutionProvider" in providers:
        return "cpu", C.OrtDevice.cpu(), [{
            "device_type": "CPU",
            "precision": "ACCURACY",
            "num_of_threads": MAX_THREADS if MAX_THREADS else 8,
            "num_streams": 1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer": False,
            "disable_dynamic_shapes": False,
        }]
    if "CUDAExecutionProvider" in providers:
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
    if "DmlExecutionProvider" in providers:
        return "dml", C.OrtDevice.dml(), [{
            "device_id": DEVICE_ID,
            "performance_preference": "high_performance",
            "device_filter": "gpu",
            "disable_metacommands": "false",
            "enable_graph_capture": "false",
            "enable_graph_serialization": "false",
        }]
    return "cpu", C.OrtDevice.cpu(), None


RUN_OPTIONS = build_run_options(silent=not ORT_LOG)
PROVIDERS = select_providers()
DEVICE_TYPE, ORT_DEVICE_TYPE, PROVIDER_OPTIONS = resolve_provider(PROVIDERS)
ORT_DEVICE = C.OrtDevice(
    ORT_DEVICE_TYPE, C.OrtDevice.default_memory(), DEVICE_ID
)
def make_merged_session(path: Path, shared_path: Path):
    options = build_session_options()
    references = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=PROVIDERS,
        provider_options=PROVIDER_OPTIONS,
        disabled_optimizers=(
            ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
            if ORT_FP16 else None
        ),
    )
    session._native_llm_shared_initializers = references
    return session


def run(session, binding):
    session.run_with_iobinding(binding, run_options=RUN_OPTIONS)


def input_names(session):
    return [meta.name for meta in session.get_inputs()]


def output_names(session):
    return [meta.name for meta in session.get_outputs()]


def ort_value(array, device=None):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), device or DEVICE_TYPE, DEVICE_ID
    )


def bind_device_outputs(binding, names):
    for name in names:
        binding._iobinding.bind_output(name, ORT_DEVICE)


def load_metadata(path: Path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    del model
    return metadata


MODEL_META = load_metadata(METADATA_PATH)


MODEL_FILE_ROLES = (
    "probe_prefill_greedy",
    "probe_prefill_penalty_greedy",
    "probe_prefill_sampling",
    "prefill_greedy",
    "prefill_penalty_greedy",
    "prefill_sampling",
    "decode_greedy",
    "decode_penalty_greedy",
    "decode_sampling",
    "shared_initializers",
    "shared_initializers_data",
)
MODEL_FILES = {role: DEFAULT_MODEL_FILE_NAMES[role] for role in MODEL_FILE_ROLES}

MAX_SEQ_LEN = int(MODEL_META["max_seq_len"])
SAMPLE_RATE = int(MODEL_META["sample_rate"])
AUDIO_PCM_SCALE = int(MODEL_META["audio_pcm_scale"])
PROMPT_CONTROL_TOKEN_COUNT = int(MODEL_META["prompt_control_token_count"])
SPECIAL_TOKEN_IDS = load_special_token_ids(MODEL_META)
SUPPORTED_LANGUAGES = load_supported_languages(MODEL_META)
LANGUAGE_CODE, LANGUAGE_ENTRY = resolve_supported_language(
    SUPPORTED_LANGUAGES, LANGUAGE
)
LANGUAGE_TOKEN_START = int(MODEL_META["language_token_start"])
LANGUAGE_TOKEN_END = int(MODEL_META["language_token_end"])
REGION_TOKEN_START = int(MODEL_META["region_token_start"])
REGION_TOKEN_END = int(MODEL_META["region_token_end"])
SOS_TOKEN = SPECIAL_TOKEN_IDS["sos"]
ASR_TOKEN = SPECIAL_TOKEN_IDS["asr"]
NOTIMESTAMP = SPECIAL_TOKEN_IDS["notimestamp"]
VOCAB_PATH = (
    ARGS.vocab_path.expanduser().resolve()
    if ARGS.vocab_path is not None
    else ONNX_FOLDER / VOCAB_FILE_NAME
)
BPE_MODEL_PATH = (
    ARGS.tokenizer_path.expanduser().resolve()
    if ARGS.tokenizer_path is not None
    else ONNX_FOLDER / BPE_MODEL_FILE_NAME
)
DEMO_AUDIO_KEY = "dolphin"
STOP_TOKENS = {SPECIAL_TOKEN_IDS["stop"]}

USE_PENALTY = not USE_SAMPLING and REPEAT_PENALTY != 1.0
if USE_SAMPLING:
    STRATEGY = "sampling"
elif USE_PENALTY:
    STRATEGY = "penalty_greedy"
else:
    STRATEGY = "greedy"

GRAPH_PAIRS = {
    "greedy": (MODEL_FILES["probe_prefill_greedy"], MODEL_FILES["prefill_greedy"], MODEL_FILES["decode_greedy"]),
    "penalty_greedy": (
        MODEL_FILES["probe_prefill_penalty_greedy"],
        MODEL_FILES["prefill_penalty_greedy"],
        MODEL_FILES["decode_penalty_greedy"],
    ),
    "sampling": (
        MODEL_FILES["probe_prefill_sampling"],
        MODEL_FILES["prefill_sampling"],
        MODEL_FILES["decode_sampling"],
    ),
}
PROBE_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][0]
PREFILL_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][1]
DECODE_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][2]
SHARED_PATH = ONNX_FOLDER / MODEL_FILES["shared_initializers"]

print(f"\nLoading Dolphin-v1 bundle: {ONNX_FOLDER}")
print("Loading merged decoder sessions and one shared initializer mmap ...")
PROBE_SESSION = make_merged_session(PROBE_PATH, SHARED_PATH)
PREFILL_SESSION = make_merged_session(PREFILL_PATH, SHARED_PATH)
DECODE_SESSION = make_merged_session(DECODE_PATH, SHARED_PATH)
print(f"Usable Providers: {DECODE_SESSION.get_providers()}")
print(
    f"Decoder strategy: {STRATEGY}; sessions=3 (Encoder+probe + cached prefill + decode); "
    "decode launches/token=1; shared initializer blob=1."
)


def indexed_layer_names(names, prefix):
    indexed = []
    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        indexed.append((int(suffix), name))
    indexed.sort()
    return [name for _, name in indexed]


_decode_inputs = input_names(DECODE_SESSION)
_state_key_inputs = indexed_layer_names(_decode_inputs, "in_de_key_layer_")
NUM_LAYERS = len(_state_key_inputs)
KV_NUM_TENSORS = NUM_LAYERS * 2


def plan_merged_io(session, strategy, is_decode):
    inputs = input_names(session)
    outputs = output_names(session)
    state_inputs = inputs[:KV_NUM_TENSORS]
    state_outputs = outputs[:KV_NUM_TENSORS]

    if strategy == "sampling":
        max_output = "sampling_sampled_id"
        next_token_output = max_output
        save_output = "sampling_save_id_out"
    elif strategy == "penalty_greedy":
        max_output = "greedy_max_logits_idx"
        next_token_output = max_output
        save_output = "greedy_save_id_out"
    else:
        max_output = "argmax_max_logits_idx"
        next_token_output = max_output
        save_output = None

    kv_sequence_outputs = [
        name
        for name in outputs
        if name.startswith("decode_kv_seq_len_next")
        or name.startswith("prefill_kv_seq_len")
    ]
    kv_sequence_output = kv_sequence_outputs[0]
    cross_inputs = [
        name for name in inputs if name.startswith(("en_key_", "en_value_"))
    ]
    cross_outputs = [
        name for name in outputs if name.startswith(("encoder_en_key_", "encoder_en_value_"))
    ]
    probe = "audio" in inputs
    if strategy == "greedy":
        save_inputs = []
    elif strategy == "sampling":
        save_inputs = ["sampling_previous_ids"]
    elif is_decode:
        save_inputs = ["penalty_save_id_in", "greedy_save_id_in"]
    else:
        save_inputs = ["greedy_save_id_in"]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "cross_inputs": cross_inputs,
        "cross_outputs": cross_outputs,
        "probe": probe,
        "token_input": "embed_input_ids",
        "kv_seq_input": "decode_kv_seq_len" if is_decode else None,
        "kv_seq_output": kv_sequence_output,
        "max_output": max_output,
        "logits_output": (
            "logits" if not is_decode and "logits" in outputs else None
        ),
        "next_token_output": next_token_output,
        "save_inputs": save_inputs,
        "save_output": save_output,
    }


PROBE_PLAN = plan_merged_io(PROBE_SESSION, STRATEGY, is_decode=False)
PREFILL_PLAN = plan_merged_io(PREFILL_SESSION, STRATEGY, is_decode=False)
DECODE_PLAN = plan_merged_io(DECODE_SESSION, STRATEGY, is_decode=True)
PROBE_INPUT_META = metadata_by_name(PROBE_SESSION.get_inputs())
PREFILL_INPUT_META = metadata_by_name(PREFILL_SESSION.get_inputs())
DECODE_INPUT_META = metadata_by_name(DECODE_SESSION.get_inputs())
PREFILL_OUTPUT_INDEX = {
    name: index for index, name in enumerate(PREFILL_PLAN["outputs"])
}
PROBE_OUTPUT_INDEX = {name: index for index, name in enumerate(PROBE_PLAN["outputs"])}
DECODE_OUTPUT_INDEX = {
    name: index for index, name in enumerate(DECODE_PLAN["outputs"])
}


def self_kv_sequence_axis(name):
    meta = DECODE_INPUT_META[name]
    candidates = [
        axis
        for axis, dim in enumerate(meta.shape)
        if axis != 0 and is_dynamic_dim(dim)
    ]
    return candidates[-1]


def empty_self_kv(meta):
    sequence_axis = self_kv_sequence_axis(meta.name)
    return filled_for(meta, axes={0: 1, sequence_axis: 0})


def bind_array(binding, meta, value, keepalive, *, axes, device=None):
    value_ort = ort_value(array_for(meta, value, axes=axes), device)
    keepalive.append(value_ort)
    binding.bind_ortvalue_input(meta.name, value_ort)
    return value_ort


def bind_scalar(binding, meta, value, keepalive, device=None):
    value_ort = ort_value(scalar_for(meta, value), device)
    keepalive.append(value_ort)
    binding.bind_ortvalue_input(meta.name, value_ort)
    return value_ort


def bind_feedback(binding, meta, value):
    binding.bind_ortvalue_input(meta.name, value)


def bind_sampling_controls(binding, input_meta, keepalive):
    controls = (
        ("sampling_temperature", TEMPERATURE),
        ("sampling_top_k", TOP_K),
        ("sampling_top_p", TOP_P),
        ("sampling_repetition_penalty", SAMPLING_REPETITION_PENALTY),
    )
    for name, value in controls:
        bind_scalar(binding, input_meta[name], value, keepalive)


def prefill(input_ids, cross_kv_by_name, sampling_history=None):
    binding = PREFILL_SESSION.io_binding()
    keepalive = []
    kv_device = "cpu" if DEVICE_TYPE == "dml" else DEVICE_TYPE
    for name in PREFILL_PLAN["state_inputs"]:
        meta = PREFILL_INPUT_META[name]
        bind_array(
            binding,
            meta,
            empty_self_kv(meta),
            keepalive,
            axes={0: 1, self_kv_sequence_axis(name): 0},
            device=kv_device,
        )
    for name in PREFILL_PLAN["cross_inputs"]:
        bind_feedback(
            binding,
            PREFILL_INPUT_META[name],
            cross_kv_by_name[name],
        )
    token_count = input_ids.shape[1]
    bind_array(
        binding,
        PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
        input_ids,
        keepalive,
        axes={0: 1, 1: token_count},
    )
    bind_scalar(
        binding, PREFILL_INPUT_META["prefill_ids_len"], token_count, keepalive
    )
    bind_scalar(
        binding, PREFILL_INPUT_META["prefill_history_len"], 0, keepalive
    )
    for name in PREFILL_PLAN["save_inputs"]:
        meta = PREFILL_INPUT_META[name]
        history = filled_for(meta, axes={0: 1, 1: 0}) if (
            sampling_history is None or name != "sampling_previous_ids"
        ) else array_for(
            meta,
            sampling_history,
            axes={0: 1, 1: np.asarray(sampling_history).shape[1]},
        )
        bind_array(
            binding, meta, history, keepalive, axes={0: 1, 1: history.shape[1]}
        )
    if STRATEGY == "sampling":
        bind_sampling_controls(binding, PREFILL_INPUT_META, keepalive)
    bind_device_outputs(binding, PREFILL_PLAN["outputs"])
    run(PREFILL_SESSION, binding)
    return binding.get_outputs()


def probe_prefill(audio_buffer, audio_window, input_ids):
    binding = PROBE_SESSION.io_binding()
    keepalive = []
    audio_meta = PROBE_INPUT_META["audio"]
    audio_buffer.update_inplace(
        array_for(
            audio_meta,
            audio_window,
            axes={0: 1, 1: 1, 2: audio_window.shape[2]},
        )
    )
    bind_feedback(binding, audio_meta, audio_buffer)
    kv_device = "cpu" if DEVICE_TYPE == "dml" else DEVICE_TYPE
    for name in PROBE_PLAN["state_inputs"]:
        meta = PROBE_INPUT_META[name]
        bind_array(
            binding,
            meta,
            empty_self_kv(meta),
            keepalive,
            axes={0: 1, self_kv_sequence_axis(name): 0},
            device=kv_device,
        )
    token_count = input_ids.shape[1]
    bind_array(
        binding,
        PROBE_INPUT_META[PROBE_PLAN["token_input"]],
        input_ids,
        keepalive,
        axes={0: 1, 1: token_count},
    )
    bind_scalar(binding, PROBE_INPUT_META["prefill_ids_len"], token_count, keepalive)
    bind_scalar(binding, PROBE_INPUT_META["prefill_history_len"], 0, keepalive)
    for name in PROBE_PLAN["save_inputs"]:
        meta = PROBE_INPUT_META[name]
        bind_array(
            binding,
            meta,
            filled_for(meta, axes={0: 1, 1: 0}),
            keepalive,
            axes={0: 1, 1: 0},
        )
    if STRATEGY == "sampling":
        bind_sampling_controls(binding, PROBE_INPUT_META, keepalive)
    bind_device_outputs(binding, PROBE_PLAN["outputs"])
    run(PROBE_SESSION, binding)
    return binding.get_outputs()


def decode_static_inputs(binding, keepalive):
    candidates = {
        "penalty_penalty_range": [PENALTY_RANGE],
    }
    for name, value in candidates.items():
        if name in DECODE_PLAN["inputs"]:
            bind_scalar(binding, DECODE_INPUT_META[name], value[0], keepalive)
    if STRATEGY == "sampling":
        bind_sampling_controls(binding, DECODE_INPUT_META, keepalive)


def decode_tokens(prefill_outputs, cross_kv_by_name, generate_limit):
    state = prefill_outputs[:KV_NUM_TENSORS]
    next_token = prefill_outputs[
        PREFILL_OUTPUT_INDEX[PREFILL_PLAN["next_token_output"]]
    ]
    kv_sequence_length = prefill_outputs[
        PREFILL_OUTPUT_INDEX[PREFILL_PLAN["kv_seq_output"]]
    ]
    selected = int(
        prefill_outputs[
            PREFILL_OUTPUT_INDEX[PREFILL_PLAN["max_output"]]
        ].numpy().reshape(-1)[0]
    )
    saved_ids = (
        prefill_outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["save_output"]]]
        if PREFILL_PLAN["save_output"] is not None
        else None
    )
    host_tokens = []
    generated_count = 0
    if selected not in STOP_TOKENS and generate_limit > 0:
        generated_count = 1
        if saved_ids is None:
            host_tokens.append(selected)

    bindings = [DECODE_SESSION.io_binding(), DECODE_SESSION.io_binding()]
    static_keepalive = [[], []]
    for binding, keepalive in zip(bindings, static_keepalive):
        for name in DECODE_PLAN["cross_inputs"]:
            bind_feedback(
                binding,
                DECODE_INPUT_META[name],
                cross_kv_by_name[name],
            )
        decode_static_inputs(binding, keepalive)
        bind_device_outputs(binding, DECODE_PLAN["outputs"])

    penalty_input = "penalty_penalty_value"
    penalty_off = penalty_on = None
    if penalty_input in DECODE_PLAN["inputs"]:
        penalty_meta = DECODE_INPUT_META[penalty_input]
        penalty_off = ort_value(scalar_for(penalty_meta, 1.0))
        penalty_on = ort_value(scalar_for(penalty_meta, REPEAT_PENALTY))
        for keepalive in static_keepalive:
            keepalive.extend((penalty_off, penalty_on))

    decode_step = 0
    start_time = time.time()
    while generated_count < generate_limit and selected not in STOP_TOKENS:
        binding_index = decode_step & 1
        binding = bindings[binding_index]
        bind_feedback(
            binding,
            DECODE_INPUT_META[DECODE_PLAN["token_input"]],
            next_token,
        )
        bind_feedback(
            binding,
            DECODE_INPUT_META[DECODE_PLAN["kv_seq_input"]],
            kv_sequence_length,
        )
        for name, value in zip(DECODE_PLAN["state_inputs"], state):
            bind_feedback(
                binding,
                DECODE_INPUT_META[name],
                value,
            )
        for name in DECODE_PLAN["save_inputs"]:
            bind_feedback(binding, DECODE_INPUT_META[name], saved_ids)
        if penalty_on is not None:
            bind_feedback(
                binding,
                DECODE_INPUT_META[penalty_input],
                penalty_on if generated_count >= PENALTY_RANGE else penalty_off,
            )

        binding.clear_binding_outputs()
        bind_device_outputs(binding, DECODE_PLAN["outputs"])
        run(DECODE_SESSION, binding)
        outputs = binding.get_outputs()
        state = outputs[:KV_NUM_TENSORS]
        selected = int(
            outputs[
                DECODE_OUTPUT_INDEX[DECODE_PLAN["max_output"]]
            ].numpy().reshape(-1)[0]
        )
        if DECODE_PLAN["save_output"] is not None:
            saved_ids = outputs[
                DECODE_OUTPUT_INDEX[DECODE_PLAN["save_output"]]
            ]
        next_token = outputs[
            DECODE_OUTPUT_INDEX[DECODE_PLAN["next_token_output"]]
        ]
        kv_sequence_length = outputs[
            DECODE_OUTPUT_INDEX[DECODE_PLAN["kv_seq_output"]]
        ]
        if selected not in STOP_TOKENS:
            generated_count += 1
            if saved_ids is None:
                host_tokens.append(selected)
        decode_step += 1

    elapsed = time.time() - start_time
    if saved_ids is not None:
        host_tokens = []
        for token in saved_ids.numpy()[0]:
            token = int(token)
            if token in STOP_TOKENS or len(host_tokens) >= generate_limit:
                break
            host_tokens.append(token)
    return host_tokens, decode_step, elapsed


# ============================================================================
# Dolphin-v1 encoder and language/region prompt setup
# ============================================================================
TOKENIZER = Tokenizer(VOCAB_PATH, BPE_MODEL_PATH)
ENCODER_INPUT_META = PROBE_INPUT_META["audio"]
ENCODER_INPUT_LENGTH = ENCODER_INPUT_META.shape[-1]
AUDIO_NP_DTYPE = numpy_dtype(ENCODER_INPUT_META)

language_prefix = list(LANGUAGE_ENTRY["prompt_token_ids"])
PARTIAL_LANGUAGE_AUTO = (
    LANGUAGE_CODE.endswith("-auto") and len(language_prefix) == 1
)
SPECIFY_LANGUAGE = len(language_prefix) == PROMPT_CONTROL_TOKEN_COUNT
prompt_values = [SOS_TOKEN, *language_prefix]
PROMPT_IDS = array_for(
    PROBE_INPUT_META[PROBE_PLAN["token_input"]],
    [prompt_values],
    axes={0: 1, 1: len(prompt_values)},
)


def resolve_prompt(probe_outputs, cross_kv):
    """Build Dolphin-v1's five-token ASR prompt, auto-detecting labels if needed."""
    if SPECIFY_LANGUAGE:
        language, _, region = LANGUAGE_CODE.partition("-")
        return PROMPT_IDS, language, region

    if PARTIAL_LANGUAGE_AUTO:
        detected_language_id = language_prefix[0]
    else:
        language_logits = probe_outputs[
            PROBE_OUTPUT_INDEX[PROBE_PLAN["logits_output"]]
        ].numpy()[0]
        detected_language_id = int(
            np.argmax(
                language_logits[LANGUAGE_TOKEN_START:LANGUAGE_TOKEN_END]
            )
            + LANGUAGE_TOKEN_START
        )

    region_probe = array_for(
        PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
        [[SOS_TOKEN, detected_language_id]],
        axes={0: 1, 1: 2},
    )
    region_outputs = prefill(
        region_probe,
        cross_kv,
        sampling_history=None,
    )
    region_logits = region_outputs[
        PREFILL_OUTPUT_INDEX[PREFILL_PLAN["logits_output"]]
    ].numpy()[0]
    detected_region_id = int(
        np.argmax(region_logits[REGION_TOKEN_START:REGION_TOKEN_END])
        + REGION_TOKEN_START
    )
    prompt_ids = array_for(
        PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
        [[SOS_TOKEN, detected_language_id, detected_region_id, ASR_TOKEN, NOTIMESTAMP]],
        axes={0: 1, 1: 5},
    )
    language_piece = TOKENIZER.decode(detected_language_id) or "?"
    region_piece = TOKENIZER.decode(detected_region_id) or "?"
    return prompt_ids, language_piece.strip("<>"), region_piece.strip("<>")


# ============================================================================
# Inference
# ============================================================================
TEST_AUDIO = list(model_audio_paths(DEMO_AUDIO_KEY))
for test_path in TEST_AUDIO:
    print("-" * 106)
    print(f"\nTest Input Audio: {test_path}")
    segment = (
        AudioSegment.from_file(test_path)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .set_sample_width(2)
    )
    raw_audio = np.asarray(segment.get_array_of_samples(), dtype=np.int16)
    audio_length = raw_audio.size
    audio_prefix = resolve_shape(
        ENCODER_INPUT_META, axes={0: 1, 1: 1, 2: audio_length}
    )[:2]
    audio = prepare_audio_input(
        raw_audio.reshape(*audio_prefix, audio_length), AUDIO_NP_DTYPE
    )
    if is_dynamic_dim(ENCODER_INPUT_LENGTH):
        input_audio_length = audio_length
    else:
        input_audio_length = int(ENCODER_INPUT_LENGTH)
    audio_shape = resolve_shape(
        ENCODER_INPUT_META, axes={0: 1, 1: 1, 2: input_audio_length}
    )
    stride = input_audio_length if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    windows = (
        1
        if audio_length <= input_audio_length
        else int(np.ceil((audio_length - input_audio_length) / stride)) + 1
    )
    aligned_length = (windows - 1) * stride + input_audio_length
    if audio.shape[-1] < aligned_length:
        audio = np.concatenate(
            [
                audio,
                np.zeros(
                    (*audio_shape[:2], aligned_length - audio.shape[-1]),
                    dtype=numpy_dtype(ENCODER_INPUT_META),
                ),
            ],
            axis=-1,
        )

    all_tokens = []
    configured_language, _, configured_region = LANGUAGE_CODE.partition("-")
    detected_language = configured_language if SPECIFY_LANGUAGE else "?"
    detected_region = configured_region if SPECIFY_LANGUAGE else "?"
    total_decode_steps = 0
    total_decode_time = 0.0
    start_time = time.time()
    audio_buffer = ort_value(
        filled_for(
            ENCODER_INPUT_META, axes={0: 1, 1: 1, 2: input_audio_length}
        )
    )
    for window_index in range(windows):
        start_sample = window_index * stride
        audio_window = array_for(
            ENCODER_INPUT_META,
            audio[:, :, start_sample:start_sample + input_audio_length],
            axes={0: audio_shape[0], 1: audio_shape[1], 2: input_audio_length},
        )
        probe_tokens = (
            PROMPT_IDS
            if SPECIFY_LANGUAGE
            else array_for(
                PROBE_INPUT_META[PROBE_PLAN["token_input"]],
                [[SOS_TOKEN]],
                axes={0: 1, 1: 1},
            )
        )
        probe_outputs = probe_prefill(audio_buffer, audio_window, probe_tokens)
        cross_kv = {
            decode_name: probe_outputs[PROBE_OUTPUT_INDEX[probe_name]]
            for probe_name, decode_name in zip(PROBE_PLAN["cross_outputs"], PREFILL_PLAN["cross_inputs"])
        }
        prompt_ids, prompt_language, prompt_region = resolve_prompt(probe_outputs, cross_kv)
        detected_language = prompt_language
        detected_region = prompt_region
        prefill_outputs = (
            [
                probe_outputs[PROBE_OUTPUT_INDEX[name]]
                for name in PREFILL_PLAN["outputs"]
            ]
            if SPECIFY_LANGUAGE
            else prefill(prompt_ids, cross_kv, sampling_history=None)
        )
        generate_limit = max(0, MAX_SEQ_LEN - prompt_ids.shape[-1])
        tokens, decode_steps, decode_time = decode_tokens(
            prefill_outputs, cross_kv, generate_limit
        )
        all_tokens.extend(tokens)
        total_decode_steps += decode_steps
        total_decode_time += decode_time

    elapsed = time.time() - start_time
    rtf = elapsed / (audio_length / SAMPLE_RATE)
    text = TOKENIZER.decode_ids(all_tokens)
    decode_rate = total_decode_steps / total_decode_time
    print(f"\nDetected: {detected_language}-{detected_region}")
    print(
        f"\nASR Result:\n{text}\n\n"
        f"RTF: {rtf:.3f}   ({elapsed:.3f}s for "
        f"{audio_length / SAMPLE_RATE:.2f}s audio, {len(all_tokens)} text tokens; "
        f"merged decode {decode_rate:.2f} token/s; 1 graph launch/token)"
    )
    print("-" * 106)
