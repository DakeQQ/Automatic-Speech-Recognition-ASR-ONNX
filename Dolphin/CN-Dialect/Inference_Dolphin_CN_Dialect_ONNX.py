"""Run Dolphin CN-Dialect with merged prefill/decode graphs and shared weights."""

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


VOCAB_FILE_NAME = "vocab_Dolphin_CN_Dialect.txt"
BPE_MODEL_FILE_NAME = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run merged Dolphin CN-Dialect ONNX inference."
    )
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "Dolphin_CN_Dialect_Optimized",
        help="Folder containing merged graphs and Dolphin_SharedInitializers.onnx(.data).",
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
        help="Optional SentencePiece model; this character-tokenizer bundle has no default BPE file.",
    )
    return parser.parse_args()


ARGS = parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
METADATA_PATH = ONNX_FOLDER / "ASR_Metadata.onnx"


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally limited to model/vocabulary/tokenizer
# paths. Edit the constants in this section for decoding, audio, demo, and runtime
# behavior; no command-line flags are provided for these settings.

# Decode strategy. Sampling takes precedence; otherwise REPEAT_PENALITY=1.0
# selects plain greedy and any other valid value selects penalty-greedy.
USE_SAMPLING = False
TEMPERATURE = 0.8
TOP_K = 10
TOP_P = 0.95
REPEAT_PENALITY = 0.8
SAMPLING_REPETITION_PENALTY = 1.0
PENALITY_RANGE = 20

# Audio/demo controls. None runs every bundled demo clip.
USE_NORMALISE_AUDIO = False
SLIDING_WINDOW = 0
LANGUAGE = "auto-auto"
HOTWORDS = ["开饭时间"]


# ============================================================================
# ONNX Runtime configuration
# ============================================================================
ORT_Accelerate_Providers = []   # [CUDAExecutionProvider, OpenVINOExecutionProvider, DmlExecutionProvider]
ORT_LOG = False
ORT_FP16 = False
MAX_THREADS = 0
DEVICE_ID = 0


def prepare_audio_input(
    audio_int16: np.ndarray,
    input_audio_dtype: np.dtype,
    target_rms: float = 4096.0,
) -> np.ndarray:
    if not USE_NORMALISE_AUDIO and input_audio_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio_int16, dtype=input_audio_dtype)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= target_rms / (rms + 1e-7)
            np.clip(audio, -32768.0, 32767.0, out=audio)
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


def resolve_provider():
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


RUN_OPTIONS = build_run_options(silent=not ORT_LOG)
DEVICE_TYPE, ORT_DEVICE_TYPE, PROVIDER_OPTIONS = resolve_provider()
ORT_DEVICE = C.OrtDevice(
    ORT_DEVICE_TYPE, C.OrtDevice.default_memory(), DEVICE_ID
)
PROVIDERS = ORT_Accelerate_Providers or ["CPUExecutionProvider"]
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
            if ORT_FP16
            else None
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
PROMPT_CONTROL_TOKEN_COUNT = int(MODEL_META["prompt_control_token_count"])
SPECIAL_TOKEN_IDS = load_special_token_ids(MODEL_META)
SUPPORTED_LANGUAGES = load_supported_languages(MODEL_META)
LANGUAGE_CODE, LANGUAGE_ENTRY = resolve_supported_language(
    SUPPORTED_LANGUAGES, LANGUAGE
)
SOS_TOKEN = SPECIAL_TOKEN_IDS["sos"]
PROMPT_START = SPECIAL_TOKEN_IDS["prompt_start"]
PROMPT_END = SPECIAL_TOKEN_IDS["prompt_end"]
VOCAB_PATH = (
    ARGS.vocab_path.expanduser().resolve()
    if ARGS.vocab_path is not None
    else ONNX_FOLDER / VOCAB_FILE_NAME
)
BPE_MODEL_PATH = (
    ARGS.tokenizer_path.expanduser().resolve()
    if ARGS.tokenizer_path is not None
    else ONNX_FOLDER / BPE_MODEL_FILE_NAME
    if BPE_MODEL_FILE_NAME
    else None
)
DEMO_AUDIO_KEY = "dolphin_cn_dialect"
STOP_TOKENS = {SPECIAL_TOKEN_IDS["stop"]}

USE_PENALTY = not USE_SAMPLING and REPEAT_PENALITY != 1.0
if USE_SAMPLING:
    STRATEGY = "sampling"
elif USE_PENALTY:
    STRATEGY = "penalty_greedy"
else:
    STRATEGY = "greedy"

GRAPH_PAIRS = {
    "greedy": (MODEL_FILES["prefill_greedy"], MODEL_FILES["decode_greedy"]),
    "penalty_greedy": (
        MODEL_FILES["prefill_penalty_greedy"],
        MODEL_FILES["decode_penalty_greedy"],
    ),
    "sampling": (
        MODEL_FILES["prefill_sampling"],
        MODEL_FILES["decode_sampling"],
    ),
}
PREFILL_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][0]
DECODE_PATH = ONNX_FOLDER / GRAPH_PAIRS[STRATEGY][1]
SHARED_PATH = ONNX_FOLDER / MODEL_FILES["shared_initializers"]

print("\nLoading merged Dolphin sessions and one shared initializer mmap ...")
PREFILL_SESSION = make_merged_session(PREFILL_PATH, SHARED_PATH)
DECODE_SESSION = make_merged_session(DECODE_PATH, SHARED_PATH)
print(f"Usable Providers: {DECODE_SESSION.get_providers()}")
print(
    f"Decoder strategy: {STRATEGY}; sessions=2 (Encoder+prefill + decode); "
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
        name
        for name in outputs
        if name.startswith(("encoder_en_key_", "encoder_en_value_"))
    ]
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


PREFILL_PLAN = plan_merged_io(PREFILL_SESSION, STRATEGY, is_decode=False)
DECODE_PLAN = plan_merged_io(DECODE_SESSION, STRATEGY, is_decode=True)
PREFILL_INPUT_META = metadata_by_name(PREFILL_SESSION.get_inputs())
DECODE_INPUT_META = metadata_by_name(DECODE_SESSION.get_inputs())


def self_kv_sequence_axis(name):
    meta = DECODE_INPUT_META[name]
    candidates = [
        axis
        for axis, dim in enumerate(meta.shape)
        if axis != 0 and is_dynamic_dim(dim)
    ]
    return candidates[-1]

PREFILL_OUTPUT_INDEX = {
    name: index for index, name in enumerate(PREFILL_PLAN["outputs"])
}
DECODE_OUTPUT_INDEX = {
    name: index for index, name in enumerate(DECODE_PLAN["outputs"])
}


def empty_self_kv(meta):
    sequence_axis = self_kv_sequence_axis(meta.name)
    return filled_for(meta, axes={0: 1, sequence_axis: 0})


def bind_array(binding, meta, value, keepalive, *, axes, device=None):
    ort = ort_value(array_for(meta, value, axes=axes), device)
    keepalive.append(ort)
    binding.bind_ortvalue_input(meta.name, ort)
    return ort


def bind_scalar(binding, meta, value, keepalive, device=None):
    ort = ort_value(scalar_for(meta, value), device)
    keepalive.append(ort)
    binding.bind_ortvalue_input(meta.name, ort)
    return ort


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


def prefill(audio_buffer, audio_window, input_ids, sampling_history=None):
    binding = PREFILL_SESSION.io_binding()
    keepalive = []
    audio_meta = PREFILL_INPUT_META["audio"]
    audio_axes = {0: 1, 1: 1, 2: audio_window.shape[2]}
    audio_buffer.update_inplace(array_for(audio_meta, audio_window, axes=audio_axes))
    bind_feedback(binding, audio_meta, audio_buffer)
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


def decode_static_inputs(binding, keepalive):
    candidates = {
        "penalty_penalty_range": [PENALITY_RANGE],
    }
    for name, value in candidates.items():
        if name in DECODE_PLAN["inputs"]:
            bind_scalar(binding, DECODE_INPUT_META[name], value[0], keepalive)
    if STRATEGY == "sampling":
        bind_sampling_controls(binding, DECODE_INPUT_META, keepalive)


def decode_tokens(prefill_outputs, generate_limit):
    state = prefill_outputs[:KV_NUM_TENSORS]
    cross_kv_by_name = {
        decode_name: prefill_outputs[PREFILL_OUTPUT_INDEX[prefill_name]]
        for prefill_name, decode_name in zip(
            PREFILL_PLAN["cross_outputs"],
            DECODE_PLAN["cross_inputs"],
        )
    }
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
        penalty_on = ort_value(scalar_for(penalty_meta, REPEAT_PENALITY))
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
                penalty_on if generated_count >= PENALITY_RANGE else penalty_off,
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
# Direct Encoder+prefill audio and prompt setup
# ============================================================================
TOKENIZER = Tokenizer(VOCAB_PATH, BPE_MODEL_PATH)
ENCODER_INPUT_META = PREFILL_INPUT_META["audio"]
ENCODER_INPUT_LENGTH = ENCODER_INPUT_META.shape[-1]
AUDIO_NP_DTYPE = numpy_dtype(ENCODER_INPUT_META)

hotword_ids = [
    token_id
    for word in HOTWORDS
    for character in word
    for token_id in (TOKENIZER.encode(character),)
    if token_id is not None
]
language_prefix = list(LANGUAGE_ENTRY["prompt_token_ids"])
SPECIFY_LANGUAGE = len(language_prefix) == PROMPT_CONTROL_TOKEN_COUNT
PARTIAL_LANGUAGE_AUTO = (
    LANGUAGE_CODE.endswith("-auto") and len(language_prefix) == 1
)
prompt_values = [SOS_TOKEN, PROMPT_START, *hotword_ids, PROMPT_END, *language_prefix]
GENERATED_CONTROL_COUNT = PROMPT_CONTROL_TOKEN_COUNT - len(language_prefix)
PROMPT_IDS = array_for(
    PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
    [prompt_values],
    axes={0: 1, 1: len(prompt_values)},
)
if hotword_ids:
    print(f"Prompt hotwords: {HOTWORDS} -> {PROMPT_IDS.tolist()[0]}")


def resolve_prompt():
    """Return the model-family-specific final prompt and detected labels."""
    language, _, region = LANGUAGE_CODE.partition("-")
    return (
        PROMPT_IDS,
        language if SPECIFY_LANGUAGE or PARTIAL_LANGUAGE_AUTO else None,
        region if SPECIFY_LANGUAGE else None,
    )


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
    detected_language = (
        configured_language
        if SPECIFY_LANGUAGE or PARTIAL_LANGUAGE_AUTO
        else "?"
    )
    detected_region = configured_region if SPECIFY_LANGUAGE else "?"
    total_decode_steps = 0
    total_decode_time = 0.0
    start_time = time.time()
    audio_buffer = ort_value(
        filled_for(
            ENCODER_INPUT_META,
            axes={0: 1, 1: 1, 2: input_audio_length},
        )
    )
    for window_index in range(windows):
        start_sample = window_index * stride
        audio_window = array_for(
            ENCODER_INPUT_META,
            audio[:, :, start_sample:start_sample + input_audio_length],
            axes={0: audio_shape[0], 1: audio_shape[1], 2: input_audio_length},
        )
        prompt_ids, prompt_language, prompt_region = resolve_prompt()
        if prompt_language is not None:
            detected_language = prompt_language
        if prompt_region is not None:
            detected_region = prompt_region
        prefill_outputs = prefill(
            audio_buffer,
            audio_window,
            prompt_ids,
            sampling_history=None,
        )
        generate_limit = max(0, MAX_SEQ_LEN - prompt_ids.shape[-1])
        tokens, decode_steps, decode_time = decode_tokens(
            prefill_outputs, generate_limit
        )
        if not SPECIFY_LANGUAGE:
            label_offset = 0 if PARTIAL_LANGUAGE_AUTO else 1
            if not PARTIAL_LANGUAGE_AUTO:
                detected_language = (
                    (TOKENIZER.decode(tokens[0]) or "?").strip("<>")
                )
            detected_region = (
                (TOKENIZER.decode(tokens[label_offset]) or "?").strip("<>")
            )
            tokens = tokens[GENERATED_CONTROL_COUNT:]
        all_tokens.extend(tokens)
        total_decode_steps += decode_steps
        total_decode_time += decode_time

    elapsed = time.time() - start_time
    rtf = elapsed / (audio_length / SAMPLE_RATE)
    text_ids = all_tokens
    text = TOKENIZER.decode_ids(text_ids)
    decode_rate = total_decode_steps / total_decode_time
    print(f"\nDetected: {detected_language}-{detected_region}")
    print(
        f"\nASR Result:\n{text}\n\n"
        f"RTF: {rtf:.3f}   ({elapsed:.3f}s for "
        f"{audio_length / SAMPLE_RATE:.2f}s audio, {len(text_ids)} text tokens; "
        f"merged decode {decode_rate:.2f} token/s; 1 graph launch/token)"
    )
    print("-" * 106)
