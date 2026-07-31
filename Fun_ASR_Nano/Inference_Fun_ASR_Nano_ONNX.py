"""Run FunASR-Nano with shared-weight merged prefill/decode ONNX graphs."""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer

import Shared_Merged


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
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
    resolve_supported_language,
    scalar_for,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Fun-ASR-Nano merged ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=_SCRIPT_DIR / "Fun_ASR_Nano_Optimized",
        help="Folder containing merged graphs and FunASR_Nano_SharedInitializers.onnx(.data).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Optional autoregressive tokenizer directory; defaults to Qwen3-0.6B in the model folder.",
    )
    parser.add_argument(
        "--ctc-vocab-path",
        "--vocab-path",
        dest="ctc_vocab_path",
        type=Path,
        default=None,
        help="Optional CTC vocabulary file; defaults to multilingual.tiktoken in the model folder.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
METADATA_MODEL_NAME = "ASR_Metadata.onnx"

# Audio behavior
# IMPORTANT: CLI options are intentionally limited to model/tokenizer/vocabulary
# paths. Edit this section for all decoding, audio, demo, and runtime behavior.
USE_NORMALISE_AUDIO = False
USE_CTC_DECODER = True
SLIDING_WINDOW = 0

# Decode strategy. Sampling takes precedence over direct penalty and plain greedy.
USE_SAMPLING = False
TEMPERATURE = 0.8
TOP_K = 10
TOP_P = 0.95
SAMPLING_REPETITION_PENALTY = 1.0
PENALTY_RANGE = 10
REPEAT_PENALTY = 0.8                # 1.0 selects greedy; another value selects penalty-greedy.
GENERATION_RESERVE_TOKENS = 10      # Keep this many decoder positions unused at the context limit.
USE_MULTILINGUAL_DEMO = False       # Add the bundled Korean example and prompt for the MLT checkpoint.

# ONNX Runtime
ORT_Accelerate_Providers = []
ORT_LOG = False
ORT_FP16 = False
MAX_THREADS = 0
DEVICE_ID = 0


def _read_metadata(path: Path) -> dict[str, str]:
    model = onnx.load(str(path), load_external_data=False)
    return {prop.key: prop.value for prop in model.metadata_props}


model_meta = _read_metadata(onnx_folder / METADATA_MODEL_NAME)
MODEL_FILE_NAMES = Shared_Merged.model_file_names()
AUDIO_PCM_SCALE = int(model_meta["audio_pcm_scale"])
SAMPLE_RATE = int(model_meta["sample_rate"])
MAX_SEQ_LEN = int(model_meta["max_seq_len"])
SPECIAL_TOKEN_IDS = load_special_token_ids(model_meta)
CTC_TOKEN_IDS = SPECIAL_TOKEN_IDS["ctc"]
STOP_TOKEN_SET = set(
    SPECIAL_TOKEN_IDS["stop"]
    if isinstance(SPECIAL_TOKEN_IDS["stop"], list)
    else [SPECIAL_TOKEN_IDS["stop"]]
)
SUPPORTED_LANGUAGES = load_supported_languages(model_meta)

if USE_SAMPLING:
    STRATEGY = "sampling"
    USE_PENALTY = False
else:
    USE_PENALTY = REPEAT_PENALTY != 1.0
    STRATEGY = "penalty_greedy" if USE_PENALTY else "greedy"
resolved_tokenizer_path = (
    _ARGS.tokenizer_path.expanduser().resolve()
    if _ARGS.tokenizer_path is not None
    else onnx_folder / "Qwen3-0.6B"
)
resolved_ctc_tokenizer_path = (
    _ARGS.ctc_vocab_path.expanduser().resolve()
    if _ARGS.ctc_vocab_path is not None
    else onnx_folder / "multilingual.tiktoken"
)
test_audio = model_audio_paths(
    "fun_asr_nano_mlt" if USE_MULTILINGUAL_DEMO else "fun_asr_nano"
)
demo_languages = ["zh", "en", "yue", "ja"]
if USE_MULTILINGUAL_DEMO:
    demo_languages.append("ko")
resolved_demo_languages = [
    resolve_supported_language(SUPPORTED_LANGUAGES, language)
    for language in demo_languages[:len(test_audio)]
]


# ---------------------------------------------------------------------------
# Audio and CTC helpers
# ---------------------------------------------------------------------------
def prepare_audio_input(
    audio_int16: np.ndarray,
    target_dtype: np.dtype,
    *,
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
            np.clip(audio, -32768.0, 32767.0, out=audio)
    if target_dtype != np.dtype(np.int16):
        audio *= np.float32(1.0 / audio_pcm_scale)
    return np.ascontiguousarray(audio, dtype=target_dtype)


class CTCTokenizer:
    """Standalone decoder matching FunASR's multilingual SenseVoice tokenizer."""

    _PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _LANGS = (
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar",
        "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu",
        "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa",
        "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
        "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
        "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw",
        "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
    )
    _AUDIO_EVENTS = ("ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter", "/Laughter", "Applause", "/Applause")
    _EMOTIONS = ("HAPPY", "SAD", "ANGRY", "NEUTRAL")

    def __init__(
        self,
        vocab_path: Path,
        ctc_token_ids: dict[str, int],
        num_languages: int = 8749,
    ):
        import tiktoken

        with open(vocab_path) as vocab_file:
            ranks = {
                base64.b64decode(token): int(rank)
                for token, rank in (line.split() for line in vocab_file if line)
            }
        n_vocab = len(ranks)
        special_tokens: dict[str, int] = {}
        specials = [
            "<|endoftext|>", "<|startoftranscript|>",
            *[f"<|{lang}|>" for lang in self._LANGS[:num_languages]],
            *[f"<|{event}|>" for event in self._AUDIO_EVENTS],
            *[f"<|{emotion}|>" for emotion in self._EMOTIONS],
            "<|translate|>", "<|transcribe|>", "<|startoflm|>", "<|startofprev|>",
            "<|nospeech|>", "<|notimestamps|>",
            *[f"<|SPECIAL_TOKEN_{index}|>" for index in range(1, 51)],
            *[f"<|{index * 0.02:.2f}|>" for index in range(1501)],
        ]
        for token in specials:
            special_tokens[token] = n_vocab
            n_vocab += 1
        self.encoding = tiktoken.Encoding(
            name="multilingual",
            explicit_n_vocab=n_vocab,
            pat_str=self._PAT,
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )
        self.blank_id = int(ctc_token_ids["blank"])
        self.no_speech = int(ctc_token_ids["no_speech"])
        self.timestamp_begin = int(ctc_token_ids["timestamp_begin"])

    def decode(self, token_ids) -> str:
        filtered = [
            token
            for token in token_ids
            if token < self.timestamp_begin
            and token not in (self.blank_id, self.no_speech)
        ]
        return self.encoding.decode(filtered)


# ---------------------------------------------------------------------------
# ORT configuration and shared-initializer session construction
# ---------------------------------------------------------------------------
def _new_session_options(*, shared: bool = False) -> onnxruntime.SessionOptions:
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 0 if ORT_LOG else 4
    options.log_verbosity_level = 4
    options.inter_op_num_threads = MAX_THREADS
    options.intra_op_num_threads = MAX_THREADS
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # Shared-weight sessions must retain the mmap-backed initializer OrtValues. Enabling graph
    # rewrites can fold/copy those initializers into a private optimized graph per session,
    # defeating physical sharing. Encoder/CTC remain fully optimized standalone sessions.
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if shared else onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
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


run_options = onnxruntime.RunOptions()
run_options.log_severity_level = 0 if ORT_LOG else 4
run_options.log_verbosity_level = 4
run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        "device_type": "CPU", "precision": "ACCURACY",
        "num_of_threads": MAX_THREADS if MAX_THREADS else 8, "num_streams": 1,
        "enable_opencl_throttling": False, "enable_qdq_optimizer": False,
        "disable_dynamic_shapes": False,
    }]
    device_type, ort_device_kind = "cpu", C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        "device_id": DEVICE_ID, "gpu_mem_limit": 24 * 1024 ** 3,
        "arena_extend_strategy": "kNextPowerOfTwo", "cudnn_conv_algo_search": "EXHAUSTIVE",
        "sdpa_kernel": "2", "use_tf32": "1", "fuse_conv_bias": "0",
        "cudnn_conv_use_max_workspace": "1", "cudnn_conv1d_pad_to_nc1d": "0",
        "tunable_op_enable": "0", "tunable_op_tuning_enable": "0",
        "tunable_op_max_tuning_duration_ms": 10, "do_copy_in_default_stream": "0",
        "enable_cuda_graph": "0", "prefer_nhwc": "0",
        "enable_skip_layer_norm_strict_mode": "0", "use_ep_level_unified_stream": "0",
    }]
    device_type, ort_device_kind = "cuda", C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        "device_id": DEVICE_ID, "performance_preference": "high_performance",
        "device_filter": "gpu", "disable_metacommands": "false",
        "enable_graph_capture": "false", "enable_graph_serialization": "false",
    }]
    device_type, ort_device_kind = "dml", C.OrtDevice.dml()
else:
    provider_options = None
    device_type, ort_device_kind = "cpu", C.OrtDevice.cpu()

ORT_DEVICE = C.OrtDevice(ort_device_kind, C.OrtDevice.default_memory(), DEVICE_ID)
KV_DEVICE = "cpu" if device_type == "dml" else device_type
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None
)
shared_model_path = onnx_folder / MODEL_FILE_NAMES["shared_initializers"]


def create_session(path: Path, *, shared: bool = False) -> onnxruntime.InferenceSession:
    options = _new_session_options(shared=shared)
    refs = Shared_Merged.attach_shared_initializers(options, shared_model_path) if shared else None
    session = onnxruntime.InferenceSession(
        str(path),
        sess_options=options,
        providers=ORT_Accelerate_Providers,
        provider_options=provider_options,
        disabled_optimizers=disabled_optimizers,
    )
    if refs is not None:
        session._funasr_shared_initializer_refs = refs
    return session


def _run(session, binding) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _bind_outputs(binding, names) -> None:
    for name in names:
        binding._iobinding.bind_output(name, ORT_DEVICE)


def _ort_value(array, device: str | None = None):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(array), device or device_type, DEVICE_ID)


def _state_sequence_axis(meta) -> int:
    dynamic_axes = [axis for axis, dim in enumerate(meta.shape) if is_dynamic_dim(dim)]
    if 0 in dynamic_axes:
        dynamic_axes.remove(0)
    return dynamic_axes[-1]


def _zero_state(meta):
    seq_axis = _state_sequence_axis(meta)
    return filled_for(meta, axes={0: 1, seq_axis: 0})


# ---------------------------------------------------------------------------
# Session loading and merged I/O planning
# ---------------------------------------------------------------------------
graph_pair = {
    "greedy": (MODEL_FILE_NAMES["prefill_greedy"], MODEL_FILE_NAMES["decode_greedy"]),
    "penalty_greedy": (MODEL_FILE_NAMES["prefill_penalty_greedy"], MODEL_FILE_NAMES["decode_penalty_greedy"]),
    "sampling": (MODEL_FILE_NAMES["prefill_sampling"], MODEL_FILE_NAMES["decode_sampling"]),
}
prefill_name, decode_name = graph_pair[STRATEGY]

ort_session_Embed = create_session(onnx_folder / MODEL_FILE_NAMES["embed"], shared=True)
prefill_session = create_session(onnx_folder / prefill_name, shared=True)
decode_session = create_session(onnx_folder / decode_name, shared=True)
decode_input_names = [meta.name for meta in decode_session.get_inputs()]
KV_NUM_TENSORS = next(
    (index for index, name in enumerate(decode_input_names) if not name.startswith("in_")),
    len(decode_input_names),
)
NUM_LAYERS = KV_NUM_TENSORS // 2
if USE_CTC_DECODER:
    ort_session_CTC = create_session(onnx_folder / MODEL_FILE_NAMES["ctc_decoder"])
    ctc_tokenizer = CTCTokenizer(resolved_ctc_tokenizer_path, CTC_TOKEN_IDS)

print(
    f"\nModel metadata: strategy={STRATEGY}, layers={NUM_LAYERS}, "
    f"kv_tensors={KV_NUM_TENSORS}."
)
print(f"Usable Providers: {decode_session.get_providers()}")
print(
    "Generation sessions: 2 merged sessions (Encoder+prefill + decode); "
    "the split runtime used Main + rotary + Embed + selection/penalty sessions per token."
)


def _plan_merged_io(session, strategy: str, decode: bool) -> dict:
    inputs = [meta.name for meta in session.get_inputs()]
    outputs = [meta.name for meta in session.get_outputs()]
    state_inputs = inputs[:KV_NUM_TENSORS]
    state_outputs = outputs[:KV_NUM_TENSORS]

    aux_output_names = {"encoder_enc_normed"} & set(outputs)
    tail = [
        name for name in outputs[KV_NUM_TENSORS:] if name not in aux_output_names
    ]
    if strategy == "greedy":
        max_idx, kv_seq_out = tail
        save_out = None
    else:
        max_idx, save_out, kv_seq_out = tail

    if strategy == "greedy":
        save_inputs = []
    elif strategy == "sampling":
        save_inputs = ["sampling_previous_ids"]
    elif not decode:
        save_inputs = ["penalty_greedy_save_id_in"]
    else:
        save_inputs = ["penalty_save_id_in", "penalty_greedy_save_id_in"]

    plan = {
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "max_idx": max_idx,
        "kv_seq_out": kv_seq_out,
        "save_out": save_out,
        "save_inputs": save_inputs,
        "ctc_out": "encoder_enc_normed" if not decode and "encoder_enc_normed" in outputs else None,
        "kv_seq_in": next((name for name in inputs if name.startswith("decode_") and "kv_seq_len" in name), None),
        "token_in": next((name for name in inputs if name == "decode_embed_input_ids"), None),
    }
    return plan


prefill_plan = _plan_merged_io(prefill_session, STRATEGY, decode=False)
decode_plan = _plan_merged_io(decode_session, STRATEGY, decode=True)
embed_input_meta = ort_session_Embed.get_inputs()[0]
embed_output_meta = ort_session_Embed.get_outputs()[0]
prefill_input_meta = metadata_by_name(prefill_session.get_inputs())
decode_input_meta = metadata_by_name(decode_session.get_inputs())
if USE_CTC_DECODER:
    ctc_input_meta = ort_session_CTC.get_inputs()[0]


def _sampling_scalars(input_meta: dict[str, object]) -> list[tuple[str, onnxruntime.OrtValue]]:
    controls = (
        ("sampling_temperature", TEMPERATURE),
        ("sampling_top_k", TOP_K),
        ("sampling_top_p", TOP_P),
        ("sampling_repetition_penalty", SAMPLING_REPETITION_PENALTY),
    )
    return [
        (name, _ort_value(scalar_for(input_meta[name], value)))
        for name, value in controls
    ]

# Encoder+prefill audio contract is graph-driven; no standalone Encoder session exists.
encoder_audio_meta = prefill_input_meta["audio"]
shape_value_in_Encoder = encoder_audio_meta.shape[-1]
audio_dtype = numpy_dtype(encoder_audio_meta)

# Tokenizer and task prompt embeddings (standalone Embed remains necessary for Encoder query_embed).
tokenizer = AutoTokenizer.from_pretrained(str(resolved_tokenizer_path))
embed_input_name = embed_input_meta.name
embed_output_name = embed_output_meta.name
prompt_embeddings = []
for language_code, language_entry in resolved_demo_languages:
    del language_code
    prompt_ids = language_entry["prompt_token_ids"]
    raw_ids = np.asarray([prompt_ids], dtype=numpy_dtype(embed_input_meta))
    token_ids = array_for(
        embed_input_meta,
        raw_ids,
        axes={0: 1, 1: raw_ids.shape[1]},
    )
    binding = ort_session_Embed.io_binding()
    binding.bind_ortvalue_input(embed_input_name, _ort_value(token_ids))
    _bind_outputs(binding, [embed_output_name])
    _run(ort_session_Embed, binding)
    # Keep each prompt independent of the Embed session's output arena.
    prompt_output = array_for(
        embed_output_meta,
        binding.get_outputs()[0].numpy(),
        axes={0: 1, 1: token_ids.shape[1]},
    )
    prompt_embeddings.append(_ort_value(prompt_output))


def _bind_typed(binding, name: str, value, input_meta: dict[str, object], keepalive: list, device: str | None = None, axes: dict[int, int] | None = None):
    if isinstance(value, onnxruntime.OrtValue):
        ort_value = value
    else:
        ort_value = _ort_value(array_for(input_meta[name], value, axes=axes), device)
    keepalive.append(ort_value)
    binding.bind_ortvalue_input(name, ort_value)
    return ort_value


def _run_generation(audio_value, query_embed) -> tuple[
    list[int], int, float, float, str
]:
    """Run one Encoder+prefill and a two-binding decode ping-pong per audio window."""
    prefill_binding = prefill_session.io_binding()
    keepalive: list = []
    for name in prefill_plan["state_inputs"]:
        sequence_axis = _state_sequence_axis(prefill_input_meta[name])
        _bind_typed(
            prefill_binding,
            name,
            _zero_state(prefill_input_meta[name]),
            prefill_input_meta,
            keepalive,
            KV_DEVICE,
            axes={0: 1, sequence_axis: 0},
        )
    _bind_typed(prefill_binding, "audio", audio_value, prefill_input_meta, keepalive)
    _bind_typed(prefill_binding, "query_embed", query_embed, prefill_input_meta, keepalive)
    _bind_typed(
        prefill_binding,
        "prefill_history_len",
        scalar_for(prefill_input_meta["prefill_history_len"], 0),
        prefill_input_meta,
        keepalive,
        axes={0: 1},
    )
    for name in prefill_plan["save_inputs"]:
        _bind_typed(
            prefill_binding,
            name,
            filled_for(prefill_input_meta[name], axes={0: 1, 1: 0}),
            prefill_input_meta,
            keepalive,
            axes={0: 1, 1: 0},
        )
    if STRATEGY == "sampling":
        for name, value in _sampling_scalars(prefill_input_meta):
            keepalive.append(value)
            prefill_binding.bind_ortvalue_input(name, value)
    _bind_outputs(prefill_binding, prefill_plan["outputs"])

    prefill_start = time.time()
    _run(prefill_session, prefill_binding)
    prefill_elapsed = time.time() - prefill_start
    prefill_outputs = prefill_binding.get_outputs()
    prefill_by_name = dict(zip(prefill_plan["outputs"], prefill_outputs))

    selected_token = int(prefill_by_name[prefill_plan["max_idx"]].numpy().flat[0])
    state = prefill_outputs[:KV_NUM_TENSORS]
    kv_seq_len = prefill_by_name[prefill_plan["kv_seq_out"]]
    ids_len = int(kv_seq_len.numpy().flat[0])
    generation_limit = max(0, MAX_SEQ_LEN - GENERATION_RESERVE_TOKENS - ids_len)
    next_token = prefill_by_name[prefill_plan["max_idx"]]
    saved_ids = prefill_by_name[prefill_plan["save_out"]] if prefill_plan["save_out"] else None
    ctc_features = (
        prefill_by_name[prefill_plan["ctc_out"]]
        if prefill_plan["ctc_out"] is not None
        else None
    )
    ctc_text = ""
    if USE_CTC_DECODER:
        ctc_binding = ort_session_CTC.io_binding()
        ctc_binding.bind_ortvalue_input(
            ctc_input_meta.name,
            ctc_features,
        )
        ctc_output_names = [meta.name for meta in ort_session_CTC.get_outputs()]
        _bind_outputs(ctc_binding, ctc_output_names)
        _run(ort_session_CTC, ctc_binding)
        ctc_ids = ctc_binding.get_outputs()[0].numpy()
        if ctc_ids.size:
            ctc_text = ctc_tokenizer.decode(ctc_ids.tolist())

    generated: list[int] = []
    generated_count = 0
    last_accepted_saved_ids = None
    if selected_token not in STOP_TOKEN_SET and generated_count < generation_limit:
        if saved_ids is None:
            generated.append(selected_token)
        elif saved_ids is not None:
            last_accepted_saved_ids = saved_ids
        generated_count = 1

    decode_bindings = [decode_session.io_binding(), decode_session.io_binding()]
    decode_keepalive: list[list] = [[], []]
    penalty_values = []
    sampling_scalars = _sampling_scalars(decode_input_meta) if STRATEGY == "sampling" else []
    for index, binding in enumerate(decode_bindings):
        if "penalty_penalty_value" in decode_plan["inputs"]:
            penalty_values.append(_bind_typed(
                binding,
                "penalty_penalty_value",
                scalar_for(decode_input_meta["penalty_penalty_value"], 1.0),
                decode_input_meta,
                decode_keepalive[index],
                axes={0: 1},
            ))
            _bind_typed(
                binding,
                "penalty_penalty_range",
                scalar_for(decode_input_meta["penalty_penalty_range"], PENALTY_RANGE),
                decode_input_meta,
                decode_keepalive[index],
                axes={0: 1},
            )
        for name, value in sampling_scalars:
            decode_keepalive[index].append(value)
            binding.bind_ortvalue_input(name, value)
        _bind_outputs(binding, decode_plan["outputs"])

    penalty_activated = False
    decode_steps = 0
    decode_start = time.time()
    while generated_count < generation_limit and selected_token not in STOP_TOKEN_SET:
        binding_index = decode_steps & 1
        binding = decode_bindings[binding_index]
        # Dynamic tensors are rebound to the previous step's device-auto outputs. Inputs and
        # outputs never alias within one invocation; alternating bindings provides the ping-pong.
        binding.bind_ortvalue_input(decode_plan["kv_seq_in"], kv_seq_len)
        binding.bind_ortvalue_input(decode_plan["token_in"], next_token)
        for name, value in zip(decode_plan["state_inputs"], state):
            binding.bind_ortvalue_input(name, value)
        for name in decode_plan["save_inputs"]:
            binding.bind_ortvalue_input(name, saved_ids)
        # IOBinding retains the OrtValues allocated on its previous use. KV and saved-id
        # outputs grow every token, so clear and re-device-bind all outputs before reuse;
        # otherwise ORT attempts to write the larger state into stale smaller buffers.
        binding.clear_binding_outputs()
        _bind_outputs(binding, decode_plan["outputs"])

        # The split runtime starts penalty only after a full PENALTY_RANGE history exists.
        # Until then the merged penalty graph multiplies by exactly 1.0, preserving bit parity.
        if penalty_values and not penalty_activated and generated_count >= PENALTY_RANGE:
            active = scalar_for(
                decode_input_meta["penalty_penalty_value"], REPEAT_PENALTY
            )
            for value in penalty_values:
                value.update_inplace(active)
            penalty_activated = True

        _run(decode_session, binding)
        outputs = binding.get_outputs()
        by_name = dict(zip(decode_plan["outputs"], outputs))
        candidate = int(by_name[decode_plan["max_idx"]].numpy().flat[0])
        decode_steps += 1
        if candidate in STOP_TOKEN_SET:
            selected_token = candidate
            break

        state = outputs[:KV_NUM_TENSORS]
        kv_seq_len = by_name[decode_plan["kv_seq_out"]]
        next_token = by_name[decode_plan["max_idx"]]
        if decode_plan["save_out"]:
            saved_ids = by_name[decode_plan["save_out"]]
        selected_token = candidate
        if saved_ids is None:
            generated.append(candidate)
        elif saved_ids is not None:
            last_accepted_saved_ids = saved_ids
        generated_count += 1

    decode_elapsed = time.time() - decode_start
    if last_accepted_saved_ids is not None:
        generated = [
            int(token)
            for token in last_accepted_saved_ids.numpy()[0, :generated_count]
            if int(token) not in STOP_TOKEN_SET
        ]
    return generated, generated_count, prefill_elapsed, decode_elapsed, ctc_text


# ---------------------------------------------------------------------------
# Audio windows: one Encoder+prefill, optional CTC, then decode generation.
# ---------------------------------------------------------------------------
for prompt_embed, test in zip(prompt_embeddings, test_audio):
    print("-" * 106)
    print(f"\nTest Input Audio: {test}")
    audio_int16 = np.array(
        AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(),
        dtype=np.int16,
    )
    audio_full_len = len(audio_int16)
    audio = prepare_audio_input(
        audio_int16.reshape(1, 1, -1),
        audio_dtype,
        audio_pcm_scale=AUDIO_PCM_SCALE,
    )
    input_audio_length = (
        audio_full_len
        if is_dynamic_dim(shape_value_in_Encoder)
        else int(shape_value_in_Encoder)
    )
    stride = input_audio_length if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    if audio_full_len > input_audio_length:
        num_windows = int(np.ceil((audio_full_len - input_audio_length) / stride)) + 1
        pad = (num_windows - 1) * stride + input_audio_length - audio_full_len
        padded_audio = filled_for(
            encoder_audio_meta,
            axes={0: 1, 1: 1, 2: audio_full_len + pad},
        )
        padded_audio[..., :audio_full_len] = audio
        audio = padded_audio
    elif audio_full_len < input_audio_length:
        padded_audio = filled_for(
            encoder_audio_meta,
            axes={0: 1, 1: 1, 2: input_audio_length},
        )
        padded_audio[..., :audio_full_len] = audio
        audio = padded_audio

    asr_tokens: list[int] = []
    ctc_result = ""
    total_prefill_time = 0.0
    total_decode_time = 0.0
    total_decode_tokens = 0
    rtf_start = time.time()
    audio_buffer = _ort_value(filled_for(
        encoder_audio_meta, axes={0: 1, 1: 1, 2: input_audio_length}
    ))

    for slice_start in range(0, audio.shape[-1] - input_audio_length + 1, stride):
        audio_window = array_for(
            encoder_audio_meta,
            audio[..., slice_start:slice_start + input_audio_length],
            axes={0: 1, 1: 1, 2: input_audio_length},
        )
        audio_buffer.update_inplace(audio_window)
        tokens, token_count, prefill_time, decode_time, ctc_text = _run_generation(
            audio_buffer,
            prompt_embed,
        )
        ctc_result += ctc_text

        asr_tokens.extend(tokens)
        total_decode_tokens += token_count
        total_prefill_time += prefill_time
        total_decode_time += decode_time

    asr_result = tokenizer.decode(asr_tokens, skip_special_tokens=True)
    elapsed = time.time() - rtf_start
    decode_speed = total_decode_tokens / total_decode_time
    print(f"\nLLM: {asr_result}", end="", flush=True)
    if USE_CTC_DECODER:
        print(f"\n\nCTC: {ctc_result}", end="", flush=True)
    print(
        f"\n\nMerged decode: {decode_speed:.3f} token/s "
        f"(prefill={total_prefill_time:.3f}s, decode={total_decode_time:.3f}s)"
        f"\nRTF: {elapsed / (audio_full_len / SAMPLE_RATE):.3f}"
    )
    print("-" * 106)
