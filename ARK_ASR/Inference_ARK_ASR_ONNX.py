"""Run ARK-ASR with merged prefill/decode ONNX graphs and shared weights."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
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
    metadata_by_name,
    load_special_token_ids,
    numpy_dtype,
    scalar_for,
)
from Shared_Merged import (
    DEFAULT_MODEL_FILE_NAMES,
    attach_shared_initializers,
    references_shared_bundle,
)


# ============================================================================
# Paths and demo inputs
# ============================================================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run merged ARK-ASR ONNX inference.")
    parser.add_argument(
        "--onnx-folder",
        "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=_SCRIPT_DIR / "ARK_ASR_Optimized",
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
_METADATA_MODEL_NAME = DEFAULT_MODEL_FILE_NAMES["metadata"]

# Test audio for inference validation.
test_audio = model_audio_paths("fun_asr_nano_mlt")
TASK_PROMPTS = ["Please transcribe this audio."]


# ============================================================================
# User configuration
# ============================================================================
# IMPORTANT: CLI options are intentionally limited to model/tokenizer paths.
# Edit this section for all decoding, audio, demo, and runtime behavior.
USE_NORMALISE_AUDIO = False

USE_SAMPLING = False                # Sampling takes precedence over deterministic decoding.
TEMPERATURE = 0.8
TOP_K = 10
TOP_P = 0.95
SAMPLING_REPETITION_PENALTY = 1.0
PENALTY_RANGE = 10
REPEAT_PENALTY = 1.0                # ARK source decoding is plain greedy by default.

ORT_Accelerate_Providers = []       # ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "DmlExecutionProvider"]
ORT_LOG = False
ORT_FP16 = False
MAX_THREADS = 0
DEVICE_ID = 0

# ============================================================================
# Metadata and utility helpers
# ============================================================================
def build_prompt_tail_ids(
    tokenizer: AutoTokenizer,
    special_token_ids: dict,
    task_prompt: str,
) -> List[int]:
    task_ids = tokenizer.encode(task_prompt, add_special_tokens=False)
    return [
        int(special_token_ids["end_audio"]),
        *[int(token_id) for token_id in task_ids],
        int(special_token_ids["assistant"]),
    ]


def decode_ark_output(
    tokenizer: AutoTokenizer,
    token_ids: List[int],
    remove_token_ids: set[int],
) -> str:
    visible_ids = [token_id for token_id in token_ids if token_id not in remove_token_ids]
    return tokenizer.decode(visible_ids, skip_special_tokens=False).strip()


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


def _empty_kv(value_meta) -> np.ndarray:
    return filled_for(value_meta, axes={0: 1, 2: 0})


def _leading_state_names(values, prefix: str) -> list[str]:
    names = [value.name for value in values]
    count = 0
    while count < len(names) and names[count].startswith(prefix):
        count += 1
    return names[:count]


def _derive_merged_kv_layout(
    prefill_session: onnxruntime.InferenceSession,
) -> tuple[int, int]:
    prefill_inputs = _leading_state_names(prefill_session.get_inputs(), "past_")
    return len(prefill_inputs), len(prefill_inputs) // 2


def _plan_merged_io(
    session: onnxruntime.InferenceSession,
    strategy: str,
    kv_num_tensors: int,
    is_decode: bool,
) -> dict:
    inputs = [value.name for value in session.get_inputs()]
    outputs = [value.name for value in session.get_outputs()]
    state_inputs = inputs[:kv_num_tensors]
    state_outputs = outputs[:kv_num_tensors]

    tail = outputs[kv_num_tensors:]
    if strategy == "greedy":
        max_out, kv_seq_out = tail
        save_out = None
    else:
        max_out, save_out, kv_seq_out = tail

    if strategy == "greedy":
        save_inputs: list[str] = []
    elif strategy == "sampling":
        save_inputs = ["sampling_previous_ids"]
    elif strategy == "penalty_greedy" and not is_decode:
        save_inputs = ["penalty_greedy_save_id_in"]
    else:
        save_inputs = ["penalty_save_id_in", "penalty_greedy_save_id_in"]

    return {
        "inputs": inputs,
        "outputs": outputs,
        "state_inputs": state_inputs,
        "state_outputs": state_outputs,
        "save_inputs": save_inputs,
        "save_out": save_out,
        "max_out": max_out,
        "kv_seq_out": kv_seq_out,
    }


def _resolve_strategy() -> tuple[str, bool]:
    if USE_SAMPLING:
        strategy = "sampling"
        use_direct_penalty = False
    else:
        use_direct_penalty = REPEAT_PENALTY != 1.0
        strategy = "penalty_greedy" if use_direct_penalty else "greedy"
    return strategy, use_direct_penalty


def _sampling_scalar_values(
    input_meta: dict[str, object],
) -> list[tuple[str, onnxruntime.OrtValue]]:
    controls = (
        ("sampling_temperature", TEMPERATURE),
        ("sampling_top_k", TOP_K),
        ("sampling_top_p", TOP_P),
        ("sampling_repetition_penalty", SAMPLING_REPETITION_PENALTY),
    )
    return [
        (name, _ort_from_numpy(scalar_for(input_meta[name], value)))
        for name, value in controls
    ]


def _persistent_embed(
    session: onnxruntime.InferenceSession,
    token_ids,
    input_meta,
    output_meta,
    consumer_meta,
) -> onnxruntime.OrtValue:
    token_count = len(token_ids)
    ids = array_for(
        input_meta,
        [token_ids],
        axes={0: 1, 1: token_count},
    )
    output = session.run([output_meta.name], {input_meta.name: ids})[0]
    output = array_for(
        output_meta,
        output,
        axes={0: 1, 1: token_count},
    )
    output = array_for(
        consumer_meta,
        output,
        axes={0: 1, 1: token_count},
    )
    return _ort_from_numpy(output)


# ============================================================================
# Merged runtime
# ============================================================================
def main() -> None:
    print("Starting merged ONNX Runtime inference ...\n")
    metadata = _load_metadata(onnx_folder)
    model_files = dict(DEFAULT_MODEL_FILE_NAMES)

    audio_pcm_scale = int(metadata["audio_pcm_scale"])
    max_audio_samples = int(metadata["max_audio_samples"])
    sample_rate = int(metadata["sample_rate"])
    max_seq_len = int(metadata["max_seq_len"])
    special_token_ids = load_special_token_ids(metadata)
    stop_token_set = {int(token_id) for token_id in special_token_ids["stop"]}
    remove_token_set = {
        int(token_id) for token_id in special_token_ids["remove_from_output"]
    }

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH), trust_remote_code=True
    )
    strategy, use_direct_penalty = _resolve_strategy()
    is_sampling = strategy == "sampling"
    graph_pair = {
        "greedy": ("prefill_greedy", "decode_greedy"),
        "penalty_greedy": ("prefill_penalty_greedy", "decode_penalty_greedy"),
        "sampling": ("prefill_sampling", "decode_sampling"),
    }
    prefill_role, decode_role = graph_pair[strategy]

    print("Loading sessions ...")
    shared_path = onnx_folder / model_files["shared_initializers"]
    # When the tied embedding table was shared into the bundle, the Embed graph
    # reads it from the shared blob and must attach it (mmap once, shared with the
    # transformer sessions); a self-contained Embed graph attaches nothing.
    embed_shared_path = (
        shared_path
        if references_shared_bundle(
            onnx_folder / model_files["embed"],
            model_files["shared_initializers_data"],
        )
        else None
    )
    embed_session = _make_session(onnx_folder / model_files["embed"], embed_shared_path)
    prefill_session = _make_session(onnx_folder / model_files[prefill_role], shared_path)
    decode_session = _make_session(onnx_folder / model_files[decode_role], shared_path)
    kv_num_tensors, num_layers = _derive_merged_kv_layout(prefill_session)
    print(f"  Strategy         : {strategy}")
    if is_sampling:
        print(
            "  Sampling         : "
            f"temperature={TEMPERATURE}, top_k={TOP_K}, top_p={TOP_P}, "
            f"repetition_penalty={SAMPLING_REPETITION_PENALTY}"
        )
    print(f"  Usable Providers : {decode_session.get_providers()}")
    print(f"  KV layout        : {num_layers} layers, {kv_num_tensors} leading tensors")

    prefill_plan = _plan_merged_io(prefill_session, strategy, kv_num_tensors, False)
    decode_plan = _plan_merged_io(decode_session, strategy, kv_num_tensors, True)
    embed_input_meta = embed_session.get_inputs()[0]
    embed_output_meta = embed_session.get_outputs()[0]
    prefill_input_meta = metadata_by_name(prefill_session.get_inputs())
    decode_input_meta = metadata_by_name(decode_session.get_inputs())
    audio_meta = prefill_input_meta["audio"]
    audio_dtype = numpy_dtype(audio_meta)
    audio_sample_dim = audio_meta.shape[2]

    test_audio_list = [test_audio] if isinstance(test_audio, str) else list(test_audio)
    configured_task_prompts = TASK_PROMPTS
    if len(configured_task_prompts) == 1:
        task_prompts = configured_task_prompts * len(test_audio_list)
    elif len(configured_task_prompts) == len(test_audio_list):
        task_prompts = configured_task_prompts
    else:
        task_prompts = configured_task_prompts
    prompt_tail_embeddings: list[onnxruntime.OrtValue] = []
    for task_prompt in task_prompts:
        prompt_tail_embeddings.append(
            _persistent_embed(
                embed_session,
                build_prompt_tail_ids(tokenizer, special_token_ids, task_prompt),
                embed_input_meta,
                embed_output_meta,
                prefill_input_meta["prompt_tail_embed"],
            )
        )

    history_len_zero = _ort_from_numpy(
        scalar_for(prefill_input_meta["prefill_history_len"], 0)
    )

    hidden_states_buffer = _ort_from_numpy(
        filled_for(embed_output_meta, axes={0: 1, 1: 1})
    )
    decode_embed_binding = embed_session.io_binding()
    decode_embed_binding.bind_ortvalue_output(embed_output_meta.name, hidden_states_buffer)

    # Both decode bindings share immutable scalar inputs.  Dynamic state always
    # crosses from one binding to the other, so no graph output aliases its input.
    decode_bindings = [decode_session.io_binding(), decode_session.io_binding()]
    penalty_value = None
    if use_direct_penalty:
        penalty_value = _ort_from_numpy(
            scalar_for(decode_input_meta["penalty_penalty_value"], REPEAT_PENALTY)
        )
        penalty_range = _ort_from_numpy(
            scalar_for(decode_input_meta["penalty_penalty_range"], PENALTY_RANGE)
        )
    decode_sampling_scalars = (
        _sampling_scalar_values(decode_input_meta) if is_sampling else []
    )
    for binding in decode_bindings:
        binding.bind_ortvalue_input("hidden_states", hidden_states_buffer)
        if use_direct_penalty:
            binding.bind_ortvalue_input("penalty_penalty_value", penalty_value)
            binding.bind_ortvalue_input("penalty_penalty_range", penalty_range)
        for name, value in decode_sampling_scalars:
            binding.bind_ortvalue_input(name, value)

    for prompt_tail_embed, task_prompt, test_path in zip(
        prompt_tail_embeddings,
        task_prompts,
        test_audio_list,
    ):
        audio_segment = AudioSegment.from_file(test_path)
        audio_pcm = np.asarray(
            audio_segment.set_channels(1)
            .set_frame_rate(sample_rate)
            .get_array_of_samples(),
            dtype=np.int16,
        )

        audio_pcm = audio_pcm[:max_audio_samples]
        original_audio_len = len(audio_pcm)
        if not is_dynamic_dim(audio_sample_dim):
            audio_pcm = audio_pcm[:int(audio_sample_dim)]
        audio = prepare_audio_input(
            audio_pcm.reshape(1, 1, -1),
            audio_dtype,
            audio_pcm_scale,
        )
        audio = array_for(
            audio_meta,
            audio,
            axes={0: 1, 1: 1, 2: audio.shape[2]},
        )
        audio_value = _ort_from_numpy(audio)

        print(
            f"\nTest audio : {test_path}   "
            f"({original_audio_len / sample_rate:.2f} s)"
        )
        if task_prompt:
            print(f"  Task prompt     : {task_prompt}")
        print("-" * 70)

        # One launch owns audio encoding, ARK prompt assembly, rotary/mask,
        # transformer prefill, and first-token selection for the chosen strategy.
        start_time = time.time()
        prefill_binding = prefill_session.io_binding()
        prefill_state_values = []
        for name in prefill_plan["state_inputs"]:
            value = _ort_from_numpy(_empty_kv(prefill_input_meta[name]))
            prefill_state_values.append(value)
            prefill_binding.bind_ortvalue_input(name, value)
        prefill_binding.bind_ortvalue_input("audio", audio_value)
        prefill_binding.bind_ortvalue_input("prompt_tail_embed", prompt_tail_embed)
        prefill_binding.bind_ortvalue_input("prefill_history_len", history_len_zero)
        empty_history_values = []
        if prefill_plan["save_inputs"]:
            for name in prefill_plan["save_inputs"]:
                empty_history = _ort_from_numpy(filled_for(
                    prefill_input_meta[name], axes={0: 1, 1: 0}
                ))
                empty_history_values.append(empty_history)
                prefill_binding.bind_ortvalue_input(
                    name,
                    empty_history,
                )
        prefill_sampling_scalars = (
            _sampling_scalar_values(prefill_input_meta) if is_sampling else []
        )
        for name, value in prefill_sampling_scalars:
            prefill_binding.bind_ortvalue_input(name, value)
        _bind_device_outputs(prefill_binding, prefill_plan["outputs"])

        prefill_start = time.time()
        _run(prefill_session, prefill_binding)
        prefill_elapsed = time.time() - prefill_start
        prefill_outputs = prefill_binding.get_outputs()
        prefill_positions = {
            name: index for index, name in enumerate(prefill_plan["outputs"])
        }

        state_values = prefill_outputs[:kv_num_tensors]
        kv_seq_len = prefill_outputs[prefill_positions[prefill_plan["kv_seq_out"]]]
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
            prefill_outputs[prefill_positions[prefill_plan["max_out"]]].numpy().flat[0]
        )
        next_token = prefill_outputs[prefill_positions[prefill_plan["max_out"]]]
        if strategy in ("penalty_greedy", "sampling"):
            save_id = prefill_outputs[prefill_positions[prefill_plan["save_out"]]]
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
            name: index for index, name in enumerate(decode_plan["outputs"])
        }
        decode_steps = 0
        decode_start = time.time()

        while (
            generated_count < generation_limit
            and selected_token not in stop_token_set
        ):
            # Standalone Embed serves ARK's prompt tail and each decode token; only
            # the transformer stage is one merged run per token.
            decode_embed_binding.bind_ortvalue_input(embed_input_meta.name, next_token)
            _run(embed_session, decode_embed_binding)

            binding = decode_bindings[decode_steps & 1]
            for name, value in zip(decode_plan["state_inputs"], state_values):
                binding.bind_ortvalue_input(name, value)
            binding.bind_ortvalue_input("decode_kv_seq_len", kv_seq_len)
            for name in decode_plan["save_inputs"]:
                binding.bind_ortvalue_input(name, save_id)

            # Device-bound outputs are fresh for this binding invocation.  They
            # become the peer binding's inputs on the next step (ping-pong).
            binding.clear_binding_outputs()
            _bind_device_outputs(binding, decode_plan["outputs"])
            _run(decode_session, binding)
            outputs = binding.get_outputs()

            state_values = outputs[:kv_num_tensors]
            kv_seq_len = outputs[decode_positions[decode_plan["kv_seq_out"]]]
            selected_token = int(
                outputs[decode_positions[decode_plan["max_out"]]].numpy().flat[0]
            )
            next_token = outputs[decode_positions[decode_plan["max_out"]]]
            if strategy in ("penalty_greedy", "sampling"):
                save_id = outputs[decode_positions[decode_plan["save_out"]]]
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
                for token in final_save_id.numpy()[0]:
                    token = int(token)
                    if token in stop_token_set:
                        break
                    generated_tokens.append(token)

        asr_result = decode_ark_output(
            tokenizer, generated_tokens, remove_token_set
        )

        total_elapsed = time.time() - start_time
        rtf = total_elapsed / (original_audio_len / sample_rate)
        decode_rate = decode_steps / decode_elapsed
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
