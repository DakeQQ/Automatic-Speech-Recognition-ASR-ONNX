"""Run the shared-initializer, merged-graph Dolphin streaming ONNX pipeline."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Dolphin-CN-Dialect streaming merged ONNX inference."
    )
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=_SCRIPT_DIR / "Dolphin_CN_Dialect_Streaming_Optimized",
        help=(
            "Folder containing the merged Dolphin graphs and shared initializer "
            "blob, for example Dolphin_CN_Dialect_Streaming_Optimized or "
            "Dolphin_CN_Dialect_Streaming_ONNX."
        ),
    )
    parser.add_argument(
        "--vocab-path",
        "--tokenizer-path",
        dest="vocab_path",
        type=Path,
        default=None,
        help=f"Optional vocabulary text file; defaults to {VOCAB_FILE_NAME} in the model folder.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
ONNX_FOLDER = _ARGS.onnx_folder.expanduser().resolve()
METADATA_PATH = ONNX_FOLDER / "ASR_Metadata.onnx"


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally limited to model/vocabulary paths.
# Edit this section for demo, language, audio, and ONNX Runtime behavior.
test_audio = model_audio_paths("dolphin_cn_dialect")
# The encoder input dtype is read from ONNX. Float variants carry int16-range
# values because Dolphin's Kaldi fbank intentionally does not divide by 32768.
USE_NORMALISE_AUDIO = False
LANGUAGE = "auto-auto"


# ============================================================================
# ONNX Runtime configuration
# ============================================================================
ORT_Accelerate_Providers = []
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
    def __init__(self, filename):
        self.str_to_idx = {}
        self.idx_to_str = {}
        with open(filename, "r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                token = line.rstrip("\n")
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def decode_ids(self, ids):
        tokens = [self.decode(int(idx)) for idx in ids]
        tokens = [token for token in tokens if token is not None]
        return "".join(tokens).replace("▁", " ").strip()


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
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if ORT_FP16
            else ""
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
ORT_DEVICE = C.OrtDevice(
    ORT_DEVICE_TYPE, C.OrtDevice.default_memory(), DEVICE_ID
)
PROVIDERS = ORT_Accelerate_Providers or ["CPUExecutionProvider"]


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
    references = attach_shared_initializers(options, shared_path)
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=PROVIDERS,
        provider_options=PROVIDER_OPTIONS,
        disabled_optimizers=DISABLED_OPTIMIZERS,
    )
    # Both the memmaps and OrtValues back add_initializer() and must outlive ORT.
    session._native_llm_shared_initializers = references
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
DISABLED_OPTIMIZERS = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16
    else None
)


MODEL_FILE_ROLES = (
    "metadata",
    "encoder",
    "prefill_greedy",
    "decode_greedy",
    "shared_initializers",
    "shared_initializers_data",
    "vocab",
)
MODEL_FILES = {
    role: (VOCAB_FILE_NAME if role == "vocab" else DEFAULT_MODEL_FILE_NAMES[role])
    for role in MODEL_FILE_ROLES
}
ENCODER_PATH = ONNX_FOLDER / MODEL_FILES["encoder"]
PREFILL_PATH = ONNX_FOLDER / MODEL_FILES["prefill_greedy"]
DECODE_PATH = ONNX_FOLDER / MODEL_FILES["decode_greedy"]
SHARED_PATH = ONNX_FOLDER / MODEL_FILES["shared_initializers"]
VOCAB_PATH = (
    _ARGS.vocab_path.expanduser().resolve()
    if _ARGS.vocab_path is not None
    else ONNX_FOLDER / MODEL_FILES["vocab"]
)


SAMPLE_RATE = int(MODEL_META["sample_rate"])
MAX_SEQ_LEN = int(MODEL_META["max_seq_len"])
stream_stride_samples = int(MODEL_META["stream_stride_samples"])
PROMPT_CONTROL_TOKEN_COUNT = int(MODEL_META["prompt_control_token_count"])
SPECIAL_TOKEN_IDS = load_special_token_ids(MODEL_META)
SUPPORTED_LANGUAGES = load_supported_languages(MODEL_META)
LANGUAGE_CODE, LANGUAGE_ENTRY = resolve_supported_language(
    SUPPORTED_LANGUAGES, LANGUAGE
)
SOS_TOKEN = SPECIAL_TOKEN_IDS["sos"]
NOTIMESTAMP = SPECIAL_TOKEN_IDS["notimestamp"]
BLANK_TOKEN = SPECIAL_TOKEN_IDS["blank"]
STOP_TOKEN = {SPECIAL_TOKEN_IDS["stop"]}


print("\nLoading merged Dolphin streaming sessions and shared initializer mmap ...")
ENCODER_SESSION = _make_session(ENCODER_PATH)
PREFILL_SESSION = _make_merged_session(PREFILL_PATH, SHARED_PATH)
DECODE_SESSION = _make_merged_session(DECODE_PATH, SHARED_PATH)
print(f"Usable Providers: {ENCODER_SESSION.get_providers()}")
print(
    "Decoder strategy: greedy; sessions=2 (prefill+decode); "
    "decode launches/token=1; shared initializer blob=1."
)
# ============================================================================
# Streaming encoder contract
# ============================================================================
ENCODER_INPUT_NAMES = _in_names(ENCODER_SESSION)
ENCODER_OUTPUT_NAMES = _out_names(ENCODER_SESSION)
ENCODER_INPUT_META = metadata_by_name(ENCODER_SESSION.get_inputs())
ENCODER_OUTPUT_META = metadata_by_name(ENCODER_SESSION.get_outputs())
ENCODER_AUDIO_INPUT = ENCODER_INPUT_NAMES[0]
ENCODER_AUDIO_META = ENCODER_INPUT_META[ENCODER_AUDIO_INPUT]
ENCODER_AUDIO_LEN_META = ENCODER_INPUT_META["audio_len"]
ENCODER_BINDING = ENCODER_SESSION.io_binding()
shape_value_in = ENCODER_AUDIO_META.shape[-1]
stream_window_samples = int(shape_value_in)
audio_input_dtype = numpy_dtype(ENCODER_AUDIO_META)
print(
    f"Model metadata: sample_rate={SAMPLE_RATE}, "
    f"stream_stride_samples={stream_stride_samples}, "
    f"encoder_window_samples={stream_window_samples}."
)

num_layer_en = (len(ENCODER_INPUT_NAMES) - 2) // 3
en_att_k_in = ENCODER_INPUT_NAMES[2:2 + num_layer_en]
en_att_v_in = ENCODER_INPUT_NAMES[2 + num_layer_en:2 + 2 * num_layer_en]
en_cnn_in = ENCODER_INPUT_NAMES[2 + 2 * num_layer_en:2 + 3 * num_layer_en]
num_cross = len(ENCODER_OUTPUT_NAMES) - 3 * num_layer_en - 1
num_cross_layers = num_cross // 2


# ============================================================================
# Merged decoder contract
# ============================================================================
def _indexed_layer_names(names, prefix):
    indexed = []
    for name in names:
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        indexed.append((int(suffix), name))
    indexed.sort()
    return [name for _, name in indexed]


_decode_inputs = _in_names(DECODE_SESSION)
_state_key_inputs = _indexed_layer_names(_decode_inputs, "in_de_key_layer_")
NUM_LAYERS = len(_state_key_inputs)
KV_NUM_TENSORS = NUM_LAYERS * 2


def _plan_merged(session, is_decode):
    inputs = _in_names(session)
    outputs = _out_names(session)
    state_inputs = inputs[:KV_NUM_TENSORS]
    state_outputs = outputs[:KV_NUM_TENSORS]
    cross_inputs = [
        name for name in inputs if name.startswith(("en_key_", "en_value_"))
    ]
    token_input = "embed_input_ids"
    counter_input = "decode_kv_seq_len" if is_decode else None
    counter_output = (
        "decode_kv_seq_len_next" if is_decode else "prefill_kv_seq_len"
    )
    return {
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "cross_inputs": cross_inputs,
        "token_input": token_input,
        "counter_input": counter_input,
        "counter_output": counter_output,
        "token_output": "argmax_max_logits_idx",
    }


PREFILL_PLAN = _plan_merged(PREFILL_SESSION, is_decode=False)
DECODE_PLAN = _plan_merged(DECODE_SESSION, is_decode=True)
PREFILL_INPUT_META = metadata_by_name(PREFILL_SESSION.get_inputs())
DECODE_INPUT_META = metadata_by_name(DECODE_SESSION.get_inputs())
PREFILL_OUTPUT_INDEX = {
    name: index for index, name in enumerate(PREFILL_PLAN["outputs"])
}
DECODE_OUTPUT_INDEX = {
    name: index for index, name in enumerate(DECODE_PLAN["outputs"])
}

def _self_kv_sequence_axis(name):
    meta = DECODE_INPUT_META[name]
    candidates = [
        axis
        for axis, dim in enumerate(meta.shape)
        if axis != 0 and is_dynamic_dim(dim)
    ]
    return candidates[-1]


def _cross_sequence_axis(meta, head_dim_axis):
    candidates = [
        axis
        for axis, dim in enumerate(meta.shape)
        if axis not in (0, head_dim_axis) and is_dynamic_dim(dim)
    ]
    return candidates[-1]


CROSS_KEY_SEQ_AXIS = _cross_sequence_axis(
    PREFILL_INPUT_META[PREFILL_PLAN["cross_inputs"][0]],
    head_dim_axis=1,
)
CROSS_VALUE_SEQ_AXIS = _cross_sequence_axis(
    PREFILL_INPUT_META[PREFILL_PLAN["cross_inputs"][NUM_LAYERS]],
    head_dim_axis=2,
)

def _empty_self_kv(meta):
    sequence_axis = _self_kv_sequence_axis(meta.name)
    return _ort_value(
        filled_for(meta, axes={0: 1, sequence_axis: 0}),
        "cpu" if DEVICE_TYPE == "dml" else DEVICE_TYPE,
    )


def _bind_feedback(binding, meta, value):
    binding.bind_ortvalue_input(meta.name, value)


EMPTY_SELF_KV = [
    _empty_self_kv(PREFILL_INPUT_META[name])
    for name in PREFILL_PLAN["state_inputs"]
]
HISTORY_LEN = _ort_value(scalar_for(PREFILL_INPUT_META["prefill_history_len"], 0))
TOKENIZER = Tokenizer(VOCAB_PATH)


lang_prefix = list(LANGUAGE_ENTRY["prompt_token_ids"])
specify_language = len(lang_prefix) == PROMPT_CONTROL_TOKEN_COUNT
partial_language_auto = (
    LANGUAGE_CODE.endswith("-auto") and len(lang_prefix) == 1
)
prompt_values = [SOS_TOKEN, *lang_prefix]
GENERATED_CONTROL_COUNT = PROMPT_CONTROL_TOKEN_COUNT - len(lang_prefix)
prompt_ids_np = array_for(
    PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
    [prompt_values],
    axes={0: 1, 1: len(prompt_values)},
)
if specify_language:
    language_name, _, region_name = LANGUAGE_CODE.partition("-")
    print(
        f"\nLanguage: forced {language_name}-{region_name} -> "
        f"prefix ids {prompt_ids_np.tolist()[0]}"
    )
PROMPT_IDS = _ort_value(prompt_ids_np)
PROMPT_IDS_LEN = _ort_value(
    scalar_for(PREFILL_INPUT_META["prefill_ids_len"], prompt_ids_np.shape[1])
)


def build_text(save_token_array):
    tokens = []
    for token in save_token_array:
        token = int(token)
        if token in STOP_TOKEN:
            break
        tokens.append(token)
    if specify_language:
        lang_s, _, region_s = LANGUAGE_CODE.partition("-")
        text_ids = tokens
    else:
        if partial_language_auto:
            lang_s, _, _ = LANGUAGE_CODE.partition("-")
            region_offset = 0
        else:
            lang_token = TOKENIZER.decode(tokens[0]) if len(tokens) > 0 else None
            lang_s = lang_token.strip("<>") if lang_token else "?"
            region_offset = 1
        region_token = (
            TOKENIZER.decode(tokens[region_offset])
            if len(tokens) > region_offset
            else None
        )
        region_s = region_token.strip("<>") if region_token else "?"
        text_ids = tokens[GENERATED_CONTROL_COUNT:]
    return lang_s, region_s, TOKENIZER.decode_ids(text_ids), len(text_ids)


def ctc_collapse(ids, blank_id=BLANK_TOKEN):
    output, previous = [], None
    for token in ids:
        token = int(token)
        if token != previous and token != blank_id and token > NOTIMESTAMP:
            output.append(token)
        previous = token
    return output


def _run_merged_attention_prefix(cross_kv_by_name, generate_limit=4):
    """Run one merged prefill and a two-binding zero-copy decode ping-pong."""
    prefill_binding = PREFILL_SESSION.io_binding()
    for name, value in zip(PREFILL_PLAN["state_inputs"], EMPTY_SELF_KV):
        _bind_feedback(
            prefill_binding,
            PREFILL_INPUT_META[name],
            value,
        )
    for name in PREFILL_PLAN["cross_inputs"]:
        _bind_feedback(
            prefill_binding,
            PREFILL_INPUT_META[name],
            cross_kv_by_name[name],
        )
    _bind_feedback(
        prefill_binding,
        PREFILL_INPUT_META[PREFILL_PLAN["token_input"]],
        PROMPT_IDS,
    )
    _bind_feedback(
        prefill_binding,
        PREFILL_INPUT_META["prefill_ids_len"],
        PROMPT_IDS_LEN,
    )
    _bind_feedback(
        prefill_binding,
        PREFILL_INPUT_META["prefill_history_len"],
        HISTORY_LEN,
    )
    _bind_device_outputs(prefill_binding, PREFILL_PLAN["outputs"])
    _run(PREFILL_SESSION, prefill_binding)
    outputs = prefill_binding.get_outputs()

    state = outputs[:KV_NUM_TENSORS]
    next_token = outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["token_output"]]]
    counter = outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["counter_output"]]]
    selected = int(next_token.numpy().reshape(-1)[0])
    generated = []

    # Cross-KV is input-only and bound once to both reusable decode bindings.
    bindings = [DECODE_SESSION.io_binding(), DECODE_SESSION.io_binding()]
    for binding in bindings:
        for name in DECODE_PLAN["cross_inputs"]:
            _bind_feedback(
                binding,
                DECODE_INPUT_META[name],
                cross_kv_by_name[name],
            )

    decode_steps = 0
    decode_start = time.perf_counter()
    while len(generated) < generate_limit and selected not in STOP_TOKEN:
        generated.append(selected)
        if len(generated) >= generate_limit:
            break
        binding = bindings[decode_steps & 1]
        _bind_feedback(
            binding,
            DECODE_INPUT_META[DECODE_PLAN["token_input"]],
            next_token,
        )
        _bind_feedback(
            binding,
            DECODE_INPUT_META[DECODE_PLAN["counter_input"]],
            counter,
        )
        for name, value in zip(DECODE_PLAN["state_inputs"], state):
            _bind_feedback(
                binding,
                DECODE_INPUT_META[name],
                value,
            )

        # Device-auto outputs are always distinct from their inputs, avoiding the
        # CUDA scalar-counter self-alias crash and preserving zero-copy KV growth.
        binding.clear_binding_outputs()
        _bind_device_outputs(binding, DECODE_PLAN["outputs"])
        _run(DECODE_SESSION, binding)
        outputs = binding.get_outputs()
        state = outputs[:KV_NUM_TENSORS]
        next_token = outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["token_output"]]]
        counter = outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["counter_output"]]]
        selected = int(next_token.numpy().reshape(-1)[0])
        decode_steps += 1

    return generated, decode_steps, time.perf_counter() - decode_start


# ============================================================================
# Streaming inference
# ============================================================================
for test in test_audio:
    print("-" * 106)
    print(f"\nTest Input Audio: {test}")
    audio = np.asarray(
        AudioSegment.from_file(test)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .get_array_of_samples(),
        dtype=np.int16,
    )
    audio_len = len(audio)
    audio_prefix = resolve_shape(
        ENCODER_AUDIO_META, axes={0: 1, 1: 1, 2: audio_len}
    )[:2]
    audio = prepare_audio_input(
        audio.reshape(*audio_prefix, audio_len), audio_input_dtype
    )
    input_audio_length = audio_len
    input_audio_shape = resolve_shape(
        ENCODER_AUDIO_META, axes={0: 1, 1: 1, 2: input_audio_length}
    )
    audio = audio[:, :, :input_audio_length]
    if audio.shape[2] < input_audio_length:
        audio = np.concatenate(
            (
                audio,
                np.zeros(
                    (*input_audio_shape[:2], input_audio_length - audio.shape[2]),
                    dtype=numpy_dtype(ENCODER_AUDIO_META),
                ),
            ),
            axis=2,
        )

    text = ""
    lang_str = region_str = "?"
    n_text = 0
    total_decode_steps = 0
    total_decode_time = 0.0

    att_k_ort = [
        _ort_value(
            filled_for(
                ENCODER_INPUT_META[name],
                axes={1: 0},
            )
        )
        for name in en_att_k_in
    ]
    att_v_ort = [
        _ort_value(
            filled_for(
                ENCODER_INPUT_META[name],
                axes={1: 0},
            )
        )
        for name in en_att_v_in
    ]
    cnn_ort = [
        _ort_value(
            filled_for(ENCODER_INPUT_META[input_name])
        )
        for input_name in en_cnn_in
    ]
    cross_k = [None] * num_cross_layers
    cross_v = [None] * num_cross_layers
    ctc_ids_all = []
    chunk_ends = list(
        range(stream_window_samples, input_audio_length, stream_stride_samples)
    ) + [input_audio_length]
    win_start = 0
    en_audio_buffer = _ort_value(
        filled_for(
            ENCODER_AUDIO_META,
            axes={0: 1, 1: 1, 2: stream_window_samples},
        )
    )
    en_audio_len_buffer = _ort_value(
        scalar_for(ENCODER_AUDIO_LEN_META, stream_window_samples)
    )

    start_time = time.time()
    for chunk_no, slice_end in enumerate(chunk_ends):
        is_final_chunk = chunk_no == len(chunk_ends) - 1
        en_chunk = np.ascontiguousarray(audio[:, :, win_start:slice_end])
        win_start += stream_stride_samples
        valid_audio_len = en_chunk.shape[-1]
        padded_chunk = filled_for(ENCODER_AUDIO_META)
        padded_chunk[..., :valid_audio_len] = en_chunk
        en_audio_buffer.update_inplace(padded_chunk)
        en_audio_len_buffer.update_inplace(
            scalar_for(ENCODER_AUDIO_LEN_META, valid_audio_len)
        )

        _bind_feedback(ENCODER_BINDING, ENCODER_AUDIO_META, en_audio_buffer)
        _bind_feedback(
            ENCODER_BINDING,
            ENCODER_AUDIO_LEN_META,
            en_audio_len_buffer,
        )
        for name, value in zip(en_att_k_in, att_k_ort):
            _bind_feedback(
                ENCODER_BINDING,
                ENCODER_INPUT_META[name],
                value,
            )
        for name, value in zip(en_att_v_in, att_v_ort):
            _bind_feedback(
                ENCODER_BINDING,
                ENCODER_INPUT_META[name],
                value,
            )
        for name, value in zip(en_cnn_in, cnn_ort):
            _bind_feedback(
                ENCODER_BINDING,
                ENCODER_INPUT_META[name],
                value,
            )
        ENCODER_BINDING.clear_binding_outputs()
        _bind_device_outputs(ENCODER_BINDING, ENCODER_OUTPUT_NAMES)
        _run(ENCODER_SESSION, ENCODER_BINDING)
        encoder_outputs = ENCODER_BINDING.get_outputs()

        for layer in range(num_cross_layers):
            key_chunk = encoder_outputs[layer].numpy()
            value_chunk = encoder_outputs[num_cross_layers + layer].numpy()
            cross_k[layer] = (
                key_chunk
                if cross_k[layer] is None
                else np.concatenate(
                    (cross_k[layer], key_chunk), axis=CROSS_KEY_SEQ_AXIS
                )
            )
            cross_v[layer] = (
                value_chunk
                if cross_v[layer] is None
                else np.concatenate(
                    (cross_v[layer], value_chunk), axis=CROSS_VALUE_SEQ_AXIS
                )
            )

        # Encoder self-caches remain on-device and feed the next chunk directly.
        att_k_ort = encoder_outputs[num_cross:num_cross + num_layer_en]
        att_v_ort = encoder_outputs[
            num_cross + num_layer_en:num_cross + 2 * num_layer_en
        ]
        cnn_ort = encoder_outputs[
            num_cross + 2 * num_layer_en:num_cross + 3 * num_layer_en
        ]

        ctc_ids_all.extend(encoder_outputs[-1].numpy()[0].tolist())
        partial_ids = ctc_collapse(ctc_ids_all)
        text = TOKENIZER.decode_ids(partial_ids)
        n_text = len(partial_ids)
        if not is_final_chunk:
            print(f"  [partial {slice_end / SAMPLE_RATE:5.2f}s] {text}")
            continue

        cross_kv_by_name = {
            **{
                f"en_key_layer_{layer}": _ort_value(
                    array_for(
                        PREFILL_INPUT_META[f"en_key_layer_{layer}"],
                        cross_k[layer],
                        axes={CROSS_KEY_SEQ_AXIS: cross_k[layer].shape[CROSS_KEY_SEQ_AXIS]},
                    )
                )
                for layer in range(num_cross_layers)
            },
            **{
                f"en_value_layer_{layer}": _ort_value(
                    array_for(
                        PREFILL_INPUT_META[f"en_value_layer_{layer}"],
                        cross_v[layer],
                        axes={CROSS_VALUE_SEQ_AXIS: cross_v[layer].shape[CROSS_VALUE_SEQ_AXIS]},
                    )
                )
                for layer in range(num_cross_layers)
            },
        }

        attention_tokens, decode_steps, decode_time = (
            _run_merged_attention_prefix(
                cross_kv_by_name,
                generate_limit=min(4, MAX_SEQ_LEN - prompt_ids_np.shape[1]),
            )
        )
        total_decode_steps += decode_steps
        total_decode_time += decode_time
        lang_str, region_str, _, _ = build_text(attention_tokens)
        print(
            f"  [FINAL {slice_end / SAMPLE_RATE:5.2f}s | "
            f"{lang_str}-{region_str}] {text}"
        )

    elapsed = time.time() - start_time
    rtf = elapsed / (audio_len / SAMPLE_RATE)
    decode_rate = total_decode_steps / total_decode_time
    print(f"\nDetected: {lang_str}-{region_str}")
    print(
        f"\nASR Result:\n{text}\n\n"
        f"RTF: {rtf:.3f}   ({elapsed:.3f}s for {audio_len / SAMPLE_RATE:.2f}s "
        f"audio, {n_text} text tokens; merged decode {decode_rate:.2f} token/s; "
        "1 graph launch/token)"
    )
    print("-" * 106)
