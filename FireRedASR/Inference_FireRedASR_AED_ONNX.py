"""Run FireRedASR-AED with merged prefill/decode graphs and shared weights."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
import sentencepiece as spm
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
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
    scalar_for,
    resolve_supported_language,
)
from Shared_Merged import DEFAULT_MODEL_FILE_NAMES, attach_shared_initializers


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the merged/shared FireRedASR AED ONNX pipeline."
    )
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=SCRIPT_DIR / "FireRedASR_Optimized",
        help="Folder containing merged graphs and FireRedASR_SharedInitializers.onnx(.data).",
    )
    parser.add_argument(
        "--vocab-path",
        "--dict-path",
        dest="vocab_path",
        type=Path,
        default=None,
        help="Optional token dictionary; defaults to dict.txt in the model folder.",
    )
    parser.add_argument(
        "--tokenizer-path",
        "--bpe-model-path",
        dest="tokenizer_path",
        type=Path,
        default=None,
        help="Optional SentencePiece model; defaults to train_bpe1000.model in the model folder.",
    )
    return parser.parse_args()


ARGS = _parse_args()
ONNX_FOLDER = ARGS.onnx_folder.expanduser().resolve()
METADATA_PATH = ONNX_FOLDER / "ASR_Metadata.onnx"
VOCAB_PATH = (
    ARGS.vocab_path.expanduser().resolve()
    if ARGS.vocab_path is not None
    else ONNX_FOLDER / "dict.txt"
)
TOKENIZER_PATH = (
    ARGS.tokenizer_path.expanduser().resolve()
    if ARGS.tokenizer_path is not None
    else ONNX_FOLDER / "train_bpe1000.model"
)


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally path-only. Edit this section for all
# decoding, audio, demo, and runtime behavior.
USE_SAMPLING = False  # Sampling takes precedence over deterministic decoding.
TEMPERATURE = 0.8
TOP_K = 3
TOP_P = 0.95
REPEAT_PENALTY = 0.8  # 1.0 selects greedy; another value selects penalty-greedy.
SAMPLING_REPETITION_PENALTY = 1.0
PENALTY_RANGE = 10
USE_NORMALISE_AUDIO = False
SLIDING_WINDOW = 0
DECODE_MAX_LEN = 0
LANGUAGE = "zh"

test_audio = model_audio_paths("fireredasr")


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
    *,
    audio_pcm_scale: int,
    target_rms: float = 4096.0,
) -> np.ndarray:
    """Convert raw PCM once; Kaldi fbank keeps the int16 numeric range."""
    if not USE_NORMALISE_AUDIO and input_audio_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio_int16, dtype=input_audio_dtype)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= target_rms / (rms + 1e-7)
            np.clip(audio, -32768.0, 32767.0, out=audio)
    if input_audio_dtype != np.dtype(np.int16):
        audio *= np.float32(1.0 / audio_pcm_scale)
    return audio.astype(input_audio_dtype)


# ============================================================================
# Standalone tokenizer implementation
# ============================================================================
class TokenDict:
    def __init__(self, dict_path, unk=""):
        self.id2word, self.word2id = self.read_dict(dict_path)
        self.unk = unk
        self.unkid = self.word2id[unk] if unk else -1

    def get(self, key, default):
        if isinstance(default, str):
            default = self.word2id[default]
        return self.word2id.get(key, default)

    def __getitem__(self, key):
        if isinstance(key, str):
            if self.unk:
                return self.word2id.get(key, self.word2id[self.unk])
            return self.word2id[key]
        if isinstance(key, (int, np.integer)):
            return self.id2word[int(key)]
        return self.word2id[key]

    def __len__(self):
        return len(self.id2word)

    @staticmethod
    def read_dict(dict_path):
        id2word, word2id = [], {}
        with open(dict_path, encoding="utf8") as file:
            for line_number, line in enumerate(file):
                pieces = line.strip().split()
                if len(pieces) >= 2:
                    word, index = pieces[0], int(pieces[1])
                elif len(pieces) == 1:
                    word, index = pieces[0], line_number
                else:
                    logging.info(
                        "Empty dictionary line %s:%s becomes a literal space",
                        dict_path,
                        line_number,
                    )
                    word, index = " ", line_number
                if word == "<space>":
                    word = " "
                word2id[word] = index
                id2word.append(word)
        return id2word, word2id


class ChineseCharEnglishSpmTokenizer:
    SPM_SPACE = "▁"

    def __init__(self, dict_path, spm_model, unk="<unk>", space="<space>"):
        self.dict = TokenDict(dict_path, unk=unk)
        self.space = space
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(spm_model)

    def tokenize(self, text, replace_punc=True):
        text = text.upper()
        if replace_punc:
            text = re.sub(r"[，。？！,\.!?]", " ", text)
        pattern = re.compile(r"([\u3400-\u4dbf\u4e00-\u9fff])")
        parts = [part for part in pattern.split(text.strip()) if part.strip()]
        tokens = []
        for part in parts:
            if pattern.fullmatch(part):
                tokens.append(part)
            else:
                tokens.extend(self.sp.EncodeAsPieces(part.strip()))
        return tokens, [self.dict.get(token, self.dict.unk) for token in tokens]

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        tokens = (
            [self.dict[int(index)] for index in inputs]
            if len(inputs) > 0 and isinstance(inputs[0], (int, np.integer))
            else inputs
        )
        text = join_symbol.join(tokens)
        return text.replace(self.SPM_SPACE, " ").strip() if replace_spm_space else text


# ============================================================================
# ORT helpers
# ============================================================================
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


def _make_session(path: Path, shared_path: Path | None = None):
    options = _build_session_options()
    refs = None
    if shared_path is not None:
        refs = attach_shared_initializers(options, shared_path)
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
    if refs is not None:
        # The memmaps and OrtValues must outlive SessionOptions.add_initializer().
        session._native_llm_shared_initializers = refs
    return session


def _run(session, binding):
    session.run_with_iobinding(binding, run_options=RUN_OPTIONS)


def _inputs(session):
    return [value.name for value in session.get_inputs()]


def _outputs(session):
    return [value.name for value in session.get_outputs()]


def _ort_value(array, target_device=None):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(array), target_device or DEVICE_TYPE, DEVICE_ID
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
LANGUAGE_CODE, LANGUAGE_ENTRY = resolve_supported_language(
    SUPPORTED_LANGUAGES, LANGUAGE
)

tokenizer = ChineseCharEnglishSpmTokenizer(str(VOCAB_PATH), str(TOKENIZER_PATH))
SOS_TOKEN = SPECIAL_TOKEN_IDS["sos"]
STOP_TOKENS = {SPECIAL_TOKEN_IDS["stop"]}

# Sampling owns its sign-aware repetition penalty and takes precedence. The
# deterministic path keeps the legacy direct logit multiplier isolated.
if USE_SAMPLING:
    STRATEGY = "sampling"
    USE_PENALTY = False
else:
    USE_PENALTY = REPEAT_PENALTY != 1.0
    STRATEGY = "penalty_greedy" if USE_PENALTY else "greedy"

GRAPH_PAIRS = {
    "greedy": ("prefill_greedy", "decode_greedy"),
    "penalty_greedy": ("prefill_penalty_greedy", "decode_penalty_greedy"),
    "sampling": ("prefill_sampling", "decode_sampling"),
}
PREFILL_ROLE, DECODE_ROLE = GRAPH_PAIRS[STRATEGY]
PREFILL_PATH = ONNX_FOLDER / MODEL_FILES[PREFILL_ROLE]
DECODE_PATH = ONNX_FOLDER / MODEL_FILES[DECODE_ROLE]
SHARED_PATH = ONNX_FOLDER / MODEL_FILES["shared_initializers"]


def _leading_names(names, prefix):
    result = []
    for name in names:
        if not name.startswith(prefix):
            break
        result.append(name)
    return result


def _plan_merged_io(session, strategy, is_decode):
    inputs = _inputs(session)
    outputs = _outputs(session)
    state_inputs = _leading_names(inputs, "in_de_")
    state_outputs = _leading_names(outputs, "out_de_")

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

    kv_seq_candidates = [
        name
        for name in outputs
        if name.startswith(("prefill_kv_seq_len", "decode_kv_seq_len_next"))
    ]
    kv_seq_output = kv_seq_candidates[0]
    cross_inputs = [
        name for name in inputs if name.startswith(("en_key_", "en_value_"))
    ]
    cross_outputs = [
        name
        for name in outputs
        if name.startswith(("encoder_en_key_", "encoder_en_value_"))
    ]
    cross_source = cross_inputs if is_decode else [
        name.removeprefix("encoder_") for name in cross_outputs
    ]
    key_cross = [name for name in cross_source if name.startswith("en_key_")]
    num_layers = len(key_cross)
    if strategy == "sampling":
        save_inputs = ["sampling_previous_ids"]
    elif strategy == "penalty_greedy":
        save_inputs = ["greedy_save_id_in"]
        if is_decode:
            save_inputs.append("penalty_save_id_in")
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
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "cross_inputs": cross_inputs,
        "cross_outputs": cross_outputs,
        "token_input": "embed_input_ids",
        "kv_seq_input": "decode_kv_seq_len" if is_decode else None,
        "kv_seq_output": kv_seq_output,
        "max_output": max_output,
        "next_token_output": next_token_output,
        "save_inputs": save_inputs,
        "save_output": save_output,
        "sampling_inputs": sampling_inputs,
        "num_layers": num_layers,
    }


def _empty_self_kv(meta):
    sequence_axis = _self_kv_sequence_axis(meta.name)
    return filled_for(meta, axes={0: 1, sequence_axis: 0})


def _bind_array(binding, meta, value, keepalive, *, axes, target_device=None):
    ort_value = _ort_value(
        array_for(meta, value, axes=axes), target_device
    )
    keepalive.append(ort_value)
    binding.bind_ortvalue_input(meta.name, ort_value)
    return ort_value


def _bind_scalar(binding, meta, value, keepalive, target_device=None):
    ort_value = _ort_value(scalar_for(meta, value), target_device)
    keepalive.append(ort_value)
    binding.bind_ortvalue_input(meta.name, ort_value)
    return ort_value


def _bind_feedback(binding, meta, value):
    binding.bind_ortvalue_input(meta.name, value)


def _bind_sampling_controls(binding, plan, input_meta, keepalive):
    values = {
        "sampling_temperature": [TEMPERATURE],
        "sampling_top_k": [TOP_K],
        "sampling_top_p": [TOP_P],
        "sampling_repetition_penalty": [SAMPLING_REPETITION_PENALTY],
    }
    for name in plan["sampling_inputs"]:
        _bind_scalar(binding, input_meta[name], values[name][0], keepalive)


print("\nLoading merged FireRedASR sessions and one shared initializer mmap ...")
PREFILL_SESSION = _make_session(PREFILL_PATH, SHARED_PATH)
DECODE_SESSION = _make_session(DECODE_PATH, SHARED_PATH)
print(f"Usable Providers: {DECODE_SESSION.get_providers()}")
print(
    f"Decoder strategy: {STRATEGY}; sessions=2 (Encoder+prefill + decode); "
    "decode launches/token=1; shared initializer blob=1."
)
PREFILL_PLAN = _plan_merged_io(PREFILL_SESSION, STRATEGY, False)
DECODE_PLAN = _plan_merged_io(DECODE_SESSION, STRATEGY, True)
KV_NUM_TENSORS = len(DECODE_PLAN["state_inputs"])
PREFILL_INPUT_META = metadata_by_name(PREFILL_SESSION.get_inputs())
DECODE_INPUT_META = metadata_by_name(DECODE_SESSION.get_inputs())


def _self_kv_sequence_axis(name):
    meta = DECODE_INPUT_META.get(name)
    if meta is None:
        meta = PREFILL_INPUT_META.get(name)
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

AUDIO_INPUT = PREFILL_INPUT_META["audio"]
AUDIO_NP_DTYPE = numpy_dtype(AUDIO_INPUT)

def _prefill(audio_buffer, audio_window):
    binding = PREFILL_SESSION.io_binding()
    keepalive = []
    audio_buffer.update_inplace(
        array_for(
            AUDIO_INPUT,
            audio_window,
            axes={0: 1, 1: 1, 2: audio_window.shape[2]},
        )
    )
    _bind_feedback(binding, AUDIO_INPUT, audio_buffer)
    kv_device = "cpu" if DEVICE_TYPE == "dml" else DEVICE_TYPE
    for name in PREFILL_PLAN["state_inputs"]:
        meta = PREFILL_INPUT_META[name]
        _bind_array(
            binding,
            meta,
            _empty_self_kv(meta),
            keepalive,
            axes={0: 1, _self_kv_sequence_axis(name): 0},
            target_device=kv_device,
        )
    token_meta = PREFILL_INPUT_META[PREFILL_PLAN["token_input"]]
    prompt = array_for(token_meta, [[SOS_TOKEN]], axes={0: 1, 1: 1})
    _bind_array(
        binding,
        token_meta,
        prompt,
        keepalive,
        axes={0: 1, 1: 1},
    )
    _bind_scalar(binding, PREFILL_INPUT_META["prefill_ids_len"], 1, keepalive)
    _bind_scalar(binding, PREFILL_INPUT_META["prefill_history_len"], 0, keepalive)
    for name in PREFILL_PLAN["save_inputs"]:
        meta = PREFILL_INPUT_META[name]
        _bind_array(
            binding,
            meta,
            filled_for(meta, axes={0: 1, 1: 0}),
            keepalive,
            axes={0: 1, 1: 0},
        )
    _bind_sampling_controls(binding, PREFILL_PLAN, PREFILL_INPUT_META, keepalive)
    _bind_device_outputs(binding, PREFILL_PLAN["outputs"])
    _run(PREFILL_SESSION, binding)
    return binding.get_outputs()


def _decode_tokens(prefill_outputs, generate_limit):
    state = prefill_outputs[:KV_NUM_TENSORS]
    cross_kv = {
        decode_name: prefill_outputs[PREFILL_OUTPUT_INDEX[prefill_name]]
        for prefill_name, decode_name in zip(
            PREFILL_PLAN["cross_outputs"],
            DECODE_PLAN["cross_inputs"],
        )
    }
    next_token = prefill_outputs[
        PREFILL_OUTPUT_INDEX[PREFILL_PLAN["next_token_output"]]
    ]
    kv_seq_len = prefill_outputs[
        PREFILL_OUTPUT_INDEX[PREFILL_PLAN["kv_seq_output"]]
    ]
    selected = int(
        prefill_outputs[
            PREFILL_OUTPUT_INDEX[PREFILL_PLAN["max_output"]]
        ].numpy().reshape(-1)[0]
    )
    save_id = (
        prefill_outputs[PREFILL_OUTPUT_INDEX[PREFILL_PLAN["save_output"]]]
        if PREFILL_PLAN["save_output"] is not None
        else None
    )

    generated_tokens = []
    generated_count = 0
    if selected not in STOP_TOKENS and generate_limit > 0:
        generated_count = 1
        if save_id is None:
            generated_tokens.append(selected)

    # Batch-one decoding alternates two bindings so every growing cache, history,
    # and kv_seq_len output is distinct from the same invocation's input buffers.
    bindings = [DECODE_SESSION.io_binding(), DECODE_SESSION.io_binding()]
    static_keepalive = [[], []]
    for binding, keepalive in zip(bindings, static_keepalive):
        for name in DECODE_PLAN["cross_inputs"]:
            _bind_feedback(
                binding,
                DECODE_INPUT_META[name],
                cross_kv[name],
            )
        if STRATEGY == "penalty_greedy":
            _bind_scalar(
                binding,
                DECODE_INPUT_META["penalty_penalty_range"],
                PENALTY_RANGE,
                keepalive,
            )
        _bind_sampling_controls(binding, DECODE_PLAN, DECODE_INPUT_META, keepalive)

    penalty_off = penalty_on = None
    if "penalty_penalty_value" in DECODE_PLAN["inputs"]:
        penalty_meta = DECODE_INPUT_META["penalty_penalty_value"]
        penalty_off = _ort_value(scalar_for(penalty_meta, 1.0))
        penalty_on = _ort_value(scalar_for(penalty_meta, REPEAT_PENALTY))
        for keepalive in static_keepalive:
            keepalive.extend((penalty_off, penalty_on))

    final_save_id = save_id
    decode_steps = 0
    start = time.time()
    while generated_count < generate_limit and selected not in STOP_TOKENS:
        binding = bindings[decode_steps & 1]
        _bind_feedback(
            binding,
            DECODE_INPUT_META[DECODE_PLAN["token_input"]],
            next_token,
        )
        _bind_feedback(
            binding,
            DECODE_INPUT_META[DECODE_PLAN["kv_seq_input"]],
            kv_seq_len,
        )
        for name, value in zip(DECODE_PLAN["state_inputs"], state):
            _bind_feedback(
                binding,
                DECODE_INPUT_META[name],
                value,
            )
        for name in DECODE_PLAN["save_inputs"]:
            _bind_feedback(binding, DECODE_INPUT_META[name], save_id)
        if penalty_on is not None:
            _bind_feedback(
                binding,
                DECODE_INPUT_META["penalty_penalty_value"],
                penalty_on if generated_count >= PENALTY_RANGE else penalty_off,
            )

        binding.clear_binding_outputs()
        _bind_device_outputs(binding, DECODE_PLAN["outputs"])
        _run(DECODE_SESSION, binding)
        outputs = binding.get_outputs()
        state = outputs[:KV_NUM_TENSORS]
        next_token = outputs[
            DECODE_OUTPUT_INDEX[DECODE_PLAN["next_token_output"]]
        ]
        kv_seq_len = outputs[
            DECODE_OUTPUT_INDEX[DECODE_PLAN["kv_seq_output"]]
        ]
        selected = int(
            outputs[DECODE_OUTPUT_INDEX[DECODE_PLAN["max_output"]]]
            .numpy()
            .reshape(-1)[0]
        )
        if DECODE_PLAN["save_output"] is not None:
            save_id = outputs[
                DECODE_OUTPUT_INDEX[DECODE_PLAN["save_output"]]
            ]
            final_save_id = save_id
        if selected not in STOP_TOKENS:
            generated_count += 1
            if save_id is None:
                generated_tokens.append(selected)
        decode_steps += 1

    elapsed = time.time() - start
    if final_save_id is not None:
        generated_tokens = []
        for token in final_save_id.numpy()[0]:
            token = int(token)
            if token in STOP_TOKENS or len(generated_tokens) >= generate_limit:
                break
            generated_tokens.append(token)
    return generated_tokens, decode_steps, elapsed


# ============================================================================
# End-to-end demo
# ============================================================================
for test in test_audio:
    print("-" * 106)
    print(f"\nTest Input Audio: {test}")
    segment = (
        AudioSegment.from_file(test)
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .set_sample_width(2)
    )
    raw_audio = np.asarray(segment.get_array_of_samples(), dtype=np.int16)
    audio_len = raw_audio.size
    audio_prefix = resolve_shape(
        AUDIO_INPUT, axes={0: 1, 1: 1, 2: audio_len}
    )[:2]
    audio = prepare_audio_input(
        raw_audio.reshape(*audio_prefix, audio_len),
        AUDIO_NP_DTYPE,
        audio_pcm_scale=AUDIO_PCM_SCALE,
    )

    fixed_shape = PREFILL_INPUT_META["audio"].shape[-1]
    window_length = (
        audio_len
        if is_dynamic_dim(fixed_shape)
        else fixed_shape
    )
    audio_shape = resolve_shape(
        AUDIO_INPUT, axes={0: 1, 1: 1, 2: window_length}
    )
    stride = window_length if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    if audio_len <= window_length:
        windows = 1
    else:
        windows = int(np.ceil((audio_len - window_length) / stride)) + 1
    aligned_length = (windows - 1) * stride + window_length
    if audio.shape[-1] < aligned_length:
        audio = np.concatenate(
            [
                audio,
                np.zeros(
                    (*audio_shape[:2], aligned_length - audio.shape[-1]),
                    dtype=numpy_dtype(AUDIO_INPUT),
                ),
            ],
            axis=-1,
        )

    audio_buffer = _ort_value(
        filled_for(AUDIO_INPUT, axes={0: 1, 1: 1, 2: window_length})
    )
    all_tokens = []
    total_decode_steps = 0
    total_decode_time = 0.0
    start_time = time.time()
    for window_index in range(windows):
        start_sample = window_index * stride
        audio_window = array_for(
            AUDIO_INPUT,
            audio[:, :, start_sample : start_sample + window_length],
            axes={0: audio_shape[0], 1: audio_shape[1], 2: window_length},
        )
        prefill_outputs = _prefill(
            audio_buffer,
            audio_window,
        )
        first_cross = PREFILL_PLAN["cross_outputs"][0]
        encoder_time_len = prefill_outputs[
            PREFILL_OUTPUT_INDEX[first_cross]
        ].shape()[-1]
        generate_limit = min(
            MAX_SEQ_LEN - 1,
            DECODE_MAX_LEN if DECODE_MAX_LEN > 0 else encoder_time_len,
        )
        window_tokens, decode_steps, decode_time = _decode_tokens(
            prefill_outputs, generate_limit
        )
        all_tokens.extend(window_tokens)
        total_decode_steps += decode_steps
        total_decode_time += decode_time

    elapsed = time.time() - start_time
    rtf = elapsed / (audio_len / SAMPLE_RATE)
    text = tokenizer.detokenize(all_tokens)
    decode_rate = total_decode_steps / total_decode_time
    print(
        f"\nASR Result:\n{text}\n\n"
        f"RTF: {rtf:.3f}   ({elapsed:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, "
        f"{len(all_tokens)} tokens; merged decode {decode_rate:.2f} token/s; "
        "1 graph launch/token)"
    )
    print("-" * 106)
