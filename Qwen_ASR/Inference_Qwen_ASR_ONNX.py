import time
from typing import List, Optional, Tuple
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer


# ============================================================================
# Paths
# ============================================================================
download_path                  = r'/home/DakeQQ/Downloads/Qwen3-ASR-0.6B'                                    # Set the path where the Qwen3-ASR-[0.6B, 1.7B] model downloaded.
onnx_model_Encoder             = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Qwen3_ASR_Encoder.onnx'         # The exported onnx model path.
onnx_model_Embed               = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Qwen3_ASR_Decoder_Embed.onnx'
onnx_model_Main                = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Qwen3_ASR_Decoder_Main.onnx'
onnx_model_Rotary_Mask_Prefill = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Rotary_Mask_Text_Prefill.onnx'
onnx_model_Rotary_Mask_Decode  = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Rotary_Mask_Text_Decode.onnx'
onnx_model_Greedy              = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Greedy_Search.onnx'
onnx_model_First_Beam          = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/First_Beam_Search.onnx'
onnx_model_Second_Beam         = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Second_Beam_Search.onnx'
onnx_model_Penalty             = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Apply_Penalty.onnx'
onnx_model_Argmax              = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Argmax.onnx'
onnx_model_Concat_Embed        = r'/home/DakeQQ/Downloads/Qwen_ASR_Optimized/Concat_Embed.onnx'


# Test audio for inference validation.
test_audio       = ["./example/zh.mp3", "./example/en.mp3", "./example/yue.mp3", "./example/ja.mp3", "./example/ko.mp3"]
LANGUAGE_PROMPTS = ["Chinese", "English", "Cantonese", "", ""]      # Use English words for the language. Set "" for auto-detection mode.
TASK_PROMPTS     = ["", "tribal chieftain", "", "", ""]             # Put the Hot Words or task hint here (optional).


# ============================================================================
# Runtime Configuration
# ============================================================================
SAMPLE_RATE            = 16000              # The model parameter, do not edit the value.
STOP_TOKEN             = [151643, 151645]   # The stop_id in Qwen is "151643" & "151645".
MAX_SEQ_LEN            = 1024               # The max context length, including prompt + audio + decode tokens.
MAX_INPUT_AUDIO_LENGTH = SAMPLE_RATE * 30   # The maximum input audio length (30s × 16000 samples/s).

USE_BEAM_SEARCH        = False      # Use beam search or greedy search.
TOP_K                  = 3          # The top k candidate in decoding.
BEAM_SIZE              = 3          # Number of beams in searching.
PENALTY_RANGE          = 10         # Penalizes the most recent output. "10" means the last 10 tokens.
REPEAT_PENALTY         = 1.0        # Range from 0.0 to 1.0; "1.0" means no penalty.

ORT_Accelerate_Providers = []       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                = False      # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16               = False      # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS            = 0          # Parallel CPU threads. Set 0 for auto.
DEVICE_ID              = 0          # Default to zero.


# ============================================================================
# Special-Token IDs
# ============================================================================
_ASR_TEXT_TAG        = "<asr_text>"
_LANG_PREFIX         = "language "


# ============================================================================
# Utility Helpers
# ============================================================================
def parse_asr_output(raw: str, user_language: Optional[str] = None) -> Tuple[str, str]:
	if not raw:
		return "", ""
	s = str(raw).strip()
	if not s:
		return "", ""
	if user_language:
		return user_language, s
	if _ASR_TEXT_TAG not in s:
		return "", s
	meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
	lang = ""
	idx = meta_part.lower().find(_LANG_PREFIX)
	if idx >= 0:
		lang = meta_part[idx + len(_LANG_PREFIX):].strip()
		if lang:
			lang = lang[:1].upper() + lang[1:].lower()
	return lang, text_part.strip()


def build_query_prompt_ids(tokenizer: AutoTokenizer, system_prompt: str) -> List[int]:
	query_ids: List[int] = []
	if system_prompt:
		query_ids.extend(tokenizer.encode(system_prompt, add_special_tokens=False))
	return query_ids


def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
	audio = audio.astype(np.float32)
	rms = np.sqrt(np.mean(audio * audio))
	if rms > 0:
		audio *= target_rms * 32768.0 / (rms + 1e-7)
	return np.clip(audio, -32768.0, 32767.0).astype(np.int16)


def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
	ro = onnxruntime.RunOptions()
	ro.log_severity_level = 0 if not silent else 4
	ro.log_verbosity_level = 4
	ro.add_run_config_entry("disable_synchronize_execution_providers", "0")
	return ro


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
	opts = onnxruntime.SessionOptions()
	opts.log_severity_level = 0 if ORT_LOG else 4
	opts.log_verbosity_level = 4
	opts.inter_op_num_threads = MAX_THREADS
	opts.intra_op_num_threads = MAX_THREADS
	opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
	opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

	cfgs = {
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
			if ORT_FP16 else ""
		),
	}
	for key, value in cfgs.items():
		opts.add_session_config_entry(key, value)
	return opts


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
	provider_options = [{
		'device_type': 'CPU',
		'precision': 'ACCURACY',
		'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,
		'num_streams': 1,
		'enable_opencl_throttling': False,
		'enable_qdq_optimizer': False,
		'disable_dynamic_shapes': False,
	}]
	device_type = "cpu"
	_ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
	provider_options = [{
		'device_id': DEVICE_ID,
		'gpu_mem_limit': 24 * (1024 ** 3),
		'arena_extend_strategy': 'kNextPowerOfTwo',
		'cudnn_conv_algo_search': 'EXHAUSTIVE',
		'sdpa_kernel': '2',
		'use_tf32': '1',
		'fuse_conv_bias': '0',
		'cudnn_conv_use_max_workspace': '1',
		'cudnn_conv1d_pad_to_nc1d': '0',
		'tunable_op_enable': '0',
		'tunable_op_tuning_enable': '0',
		'tunable_op_max_tuning_duration_ms': 10,
		'do_copy_in_default_stream': '0',
		'enable_cuda_graph': '0',
		'prefer_nhwc': '0',
		'enable_skip_layer_norm_strict_mode': '0',
		'use_ep_level_unified_stream': '0',
	}]
	device_type = "cuda"
	_ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
	provider_options = [{
		'device_id': DEVICE_ID,
		'performance_preference': 'high_performance',
		'device_filter': 'gpu',
		'disable_metacommands': 'false',
		'enable_graph_capture': 'false',
		'enable_graph_serialization': 'false',
	}]
	device_type = "dml"
	_ort_device_type = C.OrtDevice.dml()

else:
	provider_options = None
	device_type = "cpu"
	_ort_device_type = C.OrtDevice.cpu()

_ort_device_obj = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
session_opts_ort = _build_session_opts_ort()
run_options = _build_run_options(silent=not ORT_LOG)
disabled_opts = (
	["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
	if ORT_FP16 else None
)
_packed = {
	'sess_options': session_opts_ort,
	'providers': ORT_Accelerate_Providers or ["CPUExecutionProvider"],
	'provider_options': provider_options,
	'disabled_optimizers': disabled_opts,
}


def _make_session(path: str) -> onnxruntime.InferenceSession:
	return onnxruntime.InferenceSession(path, **_packed)


def _ort_from_numpy(arr: np.ndarray) -> onnxruntime.OrtValue:
	return onnxruntime.OrtValue.ortvalue_from_numpy(arr, device_type, DEVICE_ID)


def _ort_zeros(shape: tuple[int, ...], dtype: np.dtype) -> onnxruntime.OrtValue:
	return onnxruntime.OrtValue.ortvalue_from_numpy(
		np.zeros(shape, dtype=dtype),
		device_type,
		DEVICE_ID,
	)


def _ort_from_data(data, dtype: np.dtype) -> onnxruntime.OrtValue:
	return onnxruntime.OrtValue.ortvalue_from_numpy(
		np.array(data, dtype=dtype),
		device_type,
		DEVICE_ID,
	)


def _bind_inputs(binding, names, values) -> None:
	for name, value in zip(names, values):
		binding.bind_ortvalue_input(name, value)


def _bind_device_outputs(binding, names) -> None:
	for name in names:
		binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding) -> None:
	session.run_with_iobinding(binding, run_options=run_options)


def _in_names(session) -> List[str]:
	return [x.name for x in session.get_inputs()]


def _out_names(session) -> List[str]:
	return [x.name for x in session.get_outputs()]


def main() -> None:
	print("Starting ONNX Runtime inference ...\n")
	print("Loading tokenizer ...")
	tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

	print("Loading sessions ...")
	ort_session_Encoder = _make_session(onnx_model_Encoder)
	ort_session_Embed = _make_session(onnx_model_Embed)
	ort_session_Rotary_Prefill = _make_session(onnx_model_Rotary_Mask_Prefill)
	ort_session_Rotary_Decode = _make_session(onnx_model_Rotary_Mask_Decode)
	ort_session_Main = _make_session(onnx_model_Main)
	ort_session_Concat = _make_session(onnx_model_Concat_Embed)
	print(f"  Usable Providers : {ort_session_Main.get_providers()}")

	binding_Encoder = ort_session_Encoder.io_binding()
	binding_Embed = ort_session_Embed.io_binding()
	binding_Rotary_Prefill = ort_session_Rotary_Prefill.io_binding()
	binding_Rotary_Decode = ort_session_Rotary_Decode.io_binding()
	binding_Main = ort_session_Main.io_binding()
	binding_Concat = ort_session_Concat.io_binding()

	in_name_Encoder = _in_names(ort_session_Encoder)
	out_name_Encoder = _out_names(ort_session_Encoder)
	in_name_Embed = _in_names(ort_session_Embed)
	out_name_Embed = _out_names(ort_session_Embed)
	in_name_RP = _in_names(ort_session_Rotary_Prefill)
	out_name_RP = _out_names(ort_session_Rotary_Prefill)
	in_name_RD = _in_names(ort_session_Rotary_Decode)
	out_name_RD = _out_names(ort_session_Rotary_Decode)
	in_name_Main = _in_names(ort_session_Main)
	out_name_Main = _out_names(ort_session_Main)
	in_name_Concat = _in_names(ort_session_Concat)
	out_name_Concat = _out_names(ort_session_Concat)

	num_keys_values = len(out_name_Main) - 1
	main_hidden_idx = num_keys_values
	main_rotary_cos_idx = main_hidden_idx + 1
	main_rotary_sin_idx = main_hidden_idx + 2
	main_attention_mask_idx = main_hidden_idx + 3
	beam_kv_logits_count = num_keys_values + 1
	beam_save_id_idx = beam_kv_logits_count
	first_beam_size_idx = beam_save_id_idx + 1
	second_beam_previous_prob_idx = beam_save_id_idx + 1
	second_beam_size_idx = beam_save_id_idx + 2
	second_beam_top_k_idx = beam_save_id_idx + 3
	beam_save_id_out_idx = num_keys_values
	beam_score_out_idx = beam_save_id_out_idx + 1
	beam_ids_out_idx = beam_save_id_out_idx + 2
	beam_max_idx = beam_save_id_out_idx + 3

	main_inputs_meta = ort_session_Main._inputs_meta
	hidden_size = ort_session_Embed._outputs_meta[0].shape[2]
	kv_dtype_np = np.float16 if "float16" in main_inputs_meta[0].type else np.float32
	hidden_dtype_np = np.float16 if "float16" in main_inputs_meta[main_hidden_idx].type else np.float32
	logits_meta = ort_session_Main._outputs_meta[beam_save_id_out_idx]
	logits_dtype_np = np.float16 if "float16" in logits_meta.type else np.float32
	vocab_size = logits_meta.shape[1]

	num_layers_rt = num_keys_values // 2
	in_name_Main_kv = in_name_Main[:num_keys_values]
	out_name_Main_kv = out_name_Main[:num_keys_values]
	out_name_Main_logits = out_name_Main[beam_save_id_out_idx]

	past_keys_Main = _ort_zeros((1, main_inputs_meta[0].shape[1], 1, main_inputs_meta[0].shape[3], 0), kv_dtype_np)
	past_values_Main = _ort_zeros((1, main_inputs_meta[num_layers_rt].shape[1], 1, 0, main_inputs_meta[num_layers_rt].shape[4]), kv_dtype_np)

	use_beam_search = USE_BEAM_SEARCH
	top_k = TOP_K
	beam_size = BEAM_SIZE
	if use_beam_search and top_k < beam_size:
		top_k = beam_size
	if top_k < 2 or beam_size < 2:
		use_beam_search = False
		print("  [WARNING] Beam search requires BEAM_SIZE>=2 and TOP_K>=2; falling back to greedy.")
	if not use_beam_search:
		beam_size = 1

	use_penalty = REPEAT_PENALTY != 1.0
	stop_token_set = set(STOP_TOKEN)

	init_history_len_ort = _ort_from_data([0], np.int64)
	top_k_ort = _ort_from_data([top_k], np.int64)
	beam_size_ort = _ort_from_data([beam_size], np.int64)

	rotary_meta = ort_session_Rotary_Decode._outputs_meta
	rotary_dtype_np = np.float16 if "float16" in rotary_meta[0].type else np.float32
	decode_batch = beam_size if use_beam_search else 1

	rotary_cos_buf = _ort_zeros(tuple(int(dim) for dim in rotary_meta[0].shape), rotary_dtype_np)
	rotary_sin_buf = _ort_zeros(tuple(int(dim) for dim in rotary_meta[1].shape), rotary_dtype_np)
	hidden_states_buf = _ort_zeros((decode_batch, 1, hidden_size), hidden_dtype_np)
	save_id_buf = _ort_zeros((beam_size if use_beam_search else 1, 0), np.int32)
	attention_mask_buf = _ort_zeros((1, 1, 1, 1, 1), hidden_dtype_np)
	prefill_logits_buf = _ort_zeros((1, vocab_size), logits_dtype_np)
	decode_logits_buf = _ort_zeros((decode_batch, vocab_size), logits_dtype_np)
	max_idx_buf = _ort_zeros((1, 1), np.int32)

	if use_beam_search:
		print("\n  [INFO] Beam search active - transcription is shown only after full decode.\n")
		beam_ids_buf = _ort_zeros((beam_size, 1), np.int32)
		beam_score_buf = _ort_zeros((beam_size, 1), logits_dtype_np)

		ort_session_First_Beam = _make_session(onnx_model_First_Beam)
		ort_session_Second_Beam = _make_session(onnx_model_Second_Beam)
		binding_First_Beam = ort_session_First_Beam.io_binding()
		binding_Second_Beam = ort_session_Second_Beam.io_binding()
		in_name_First_Beam = _in_names(ort_session_First_Beam)
		out_name_First_Beam = _out_names(ort_session_First_Beam)
		in_name_Second_Beam = _in_names(ort_session_Second_Beam)
		out_name_Second_Beam = _out_names(ort_session_Second_Beam)

		in_name_First_Beam_parts = in_name_First_Beam[:beam_kv_logits_count]
		out_name_First_Beam_parts = out_name_First_Beam[:beam_kv_logits_count]
		in_name_Second_Beam_parts = in_name_Second_Beam[:beam_kv_logits_count]
		out_name_Second_Beam_parts = out_name_Second_Beam[:beam_kv_logits_count]

		binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[beam_save_id_idx], save_id_buf)
		binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[first_beam_size_idx], beam_size_ort)
		binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_size_idx], beam_size_ort)
		binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_top_k_idx], top_k_ort)
	else:
		ort_session_Greedy = _make_session(onnx_model_Greedy)
		binding_Greedy = ort_session_Greedy.io_binding()
		in_name_Greedy = _in_names(ort_session_Greedy)
		out_name_Greedy = _out_names(ort_session_Greedy)
		binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)
		binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)

		ort_session_Argmax = _make_session(onnx_model_Argmax)
		binding_Argmax = ort_session_Argmax.io_binding()
		in_name_Argmax = _in_names(ort_session_Argmax)[0]
		out_name_Argmax = _out_names(ort_session_Argmax)[0]
		binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)
		save_id_numpy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

	if use_penalty:
		ort_session_Penalty = _make_session(onnx_model_Penalty)
		binding_Penalty = ort_session_Penalty.io_binding()
		in_name_Penalty = _in_names(ort_session_Penalty)
		out_name_Penalty = _out_names(ort_session_Penalty)[0]
		num_penalty_inputs = len(in_name_Penalty)
		if num_penalty_inputs > 2:
			penalty_dtype_np = (
				np.float16 if "float16" in ort_session_Penalty._inputs_meta[2].type else np.float32
			)
			penalty_value_ort = _ort_from_data([REPEAT_PENALTY], penalty_dtype_np)
			binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value_ort)
		if num_penalty_inputs > 3:
			penalty_range_ort = _ort_from_data([PENALTY_RANGE], np.int64)
			binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range_ort)

	binding_Rotary_Prefill.bind_ortvalue_input(in_name_RP[1], init_history_len_ort)
	binding_Rotary_Decode.bind_ortvalue_output(out_name_RD[0], rotary_cos_buf)
	binding_Rotary_Decode.bind_ortvalue_output(out_name_RD[1], rotary_sin_buf)

	test_audio_list = [test_audio] if isinstance(test_audio, str) else list(test_audio)
	if len(TASK_PROMPTS) == 1:
		task_prompt_list = TASK_PROMPTS * len(test_audio_list)
	elif len(TASK_PROMPTS) != len(test_audio_list):
		raise ValueError("TASK_PROMPTS must contain either one prompt or one prompt per test audio.")
	else:
		task_prompt_list = TASK_PROMPTS

	if len(LANGUAGE_PROMPTS) == 1:
		language_prompt_list = LANGUAGE_PROMPTS * len(test_audio_list)
	elif len(LANGUAGE_PROMPTS) != len(test_audio_list):
		raise ValueError("LANGUAGE_PROMPTS must contain either one prompt or one prompt per test audio.")
	else:
		language_prompt_list = LANGUAGE_PROMPTS

	init_all_outputs_Embed: List[onnxruntime.OrtValue] = []
	for prompt in task_prompt_list:
		if prompt:
			prompt_ids = np.array([build_query_prompt_ids(tokenizer, prompt)], dtype=np.int32)
			prompt_ids_ort = _ort_from_numpy(prompt_ids)
			binding_Embed.bind_ortvalue_input(in_name_Embed[0], prompt_ids_ort)
			_bind_device_outputs(binding_Embed, [out_name_Embed[0]])
			_run(ort_session_Embed, binding_Embed)
			prompt_embed = _ort_from_data(binding_Embed.get_outputs()[0].numpy(), hidden_dtype_np)  # Create a new ort tensor.
		else:
			prompt_embed = _ort_zeros((1, 0, hidden_size), hidden_dtype_np)
		init_all_outputs_Embed.append(prompt_embed)

	init_all_outputs_Lang_Embed: List[Optional[onnxruntime.OrtValue]] = []
	for lang_name in language_prompt_list:
		if lang_name:
			lang_ids = np.array([tokenizer.encode(lang_name, add_special_tokens=False)], dtype=np.int32)
			lang_ids_ort = _ort_from_numpy(lang_ids)
			binding_Embed.bind_ortvalue_input(in_name_Embed[0], lang_ids_ort)
			_bind_device_outputs(binding_Embed, [out_name_Embed[0]])
			_run(ort_session_Embed, binding_Embed)
			lang_embed = _ort_from_data(binding_Embed.get_outputs()[0].numpy(), hidden_dtype_np)  # Create a new ort tensor.
			init_all_outputs_Lang_Embed.append(lang_embed)
		else:
			init_all_outputs_Lang_Embed.append(None)

	binding_Embed.bind_ortvalue_output(out_name_Embed[0], hidden_states_buf)
	if use_beam_search:
		binding_Embed.bind_ortvalue_input(in_name_Embed[0], beam_ids_buf)
	else:
		binding_Embed.bind_ortvalue_input(in_name_Embed[0], max_idx_buf)

	for prompt_embed, lang_embed, system_prompt, lang_prompt, test_path in zip(
		init_all_outputs_Embed,
		init_all_outputs_Lang_Embed,
		task_prompt_list,
		language_prompt_list,
		test_audio_list,
	):
		try:
			audio_seg = AudioSegment.from_file(test_path)
			audio_pcm = np.array(audio_seg.set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
		except Exception:
			print(f"\n  [WARN] Cannot load '{test_path}'")
			continue

		print(f"\nTest audio : {test_path}   ({len(audio_pcm) / SAMPLE_RATE:.2f} s)")
		if system_prompt:
			print(f"  System prompt : {system_prompt}")
		if lang_prompt:
			print(f"  Language prompt: {lang_prompt}")
		print("-" * 70)

		audio_pcm = normalise_audio(audio_pcm)
		audio_len = len(audio_pcm)
		audio_pcm = audio_pcm[:MAX_INPUT_AUDIO_LENGTH]
		audio_np = audio_pcm.reshape(1, 1, -1)
		audio_ort = _ort_from_numpy(audio_np)

		t0 = time.time()

		binding_Encoder.bind_ortvalue_input(in_name_Encoder[0], audio_ort)
		binding_Encoder.bind_ortvalue_input(in_name_Encoder[1], prompt_embed)
		_bind_device_outputs(binding_Encoder, out_name_Encoder)
		_run(ort_session_Encoder, binding_Encoder)
		hidden_states, ids_len = binding_Encoder.get_outputs()

		if lang_embed is not None:
			binding_Concat.bind_ortvalue_input(in_name_Concat[0], hidden_states)
			binding_Concat.bind_ortvalue_input(in_name_Concat[1], lang_embed)
			_bind_device_outputs(binding_Concat, out_name_Concat)
			_run(ort_session_Concat, binding_Concat)
			hidden_states, ids_len = binding_Concat.get_outputs()
			ids_len_val = ids_len.numpy()
		else:
			ids_len_val = ids_len.numpy()

		t_enc = time.time()
		print(f"  Encode done ({t_enc - t0:.3f}s)")

		binding_Rotary_Prefill.bind_ortvalue_input(in_name_RP[0], ids_len)
		_bind_device_outputs(binding_Rotary_Prefill, out_name_RP)
		_run(ort_session_Rotary_Prefill, binding_Rotary_Prefill)
		rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Prefill.get_outputs()

		binding_Rotary_Decode.bind_ortvalue_input(in_name_RD[0], kv_seq_len)
		binding_Rotary_Decode.bind_ortvalue_output(out_name_RD[2], kv_seq_len)

		binding_Main.bind_ortvalue_input(in_name_Main[main_hidden_idx], hidden_states)
		binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_cos_idx], rotary_cos)
		binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_sin_idx], rotary_sin)
		binding_Main.bind_ortvalue_input(in_name_Main[main_attention_mask_idx], attention_mask)
		for index in range(num_layers_rt):
			binding_Main.bind_ortvalue_input(in_name_Main[index], past_keys_Main)
		for index in range(num_layers_rt):
			binding_Main.bind_ortvalue_input(in_name_Main[num_layers_rt + index], past_values_Main)
		_bind_device_outputs(binding_Main, out_name_Main_kv)
		binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

		if use_penalty:
			binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
			binding_Penalty.bind_ortvalue_output(out_name_Penalty, prefill_logits_buf)

		if not use_beam_search and use_penalty:
			binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
		elif not use_beam_search:
			binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)

		if not use_beam_search and use_penalty:
			binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

		is_prefill_step = True
		num_decode = 0
		save_id = save_id_buf
		generate_limit = max(MAX_SEQ_LEN - 10 - ids_len_val, 0)

		while num_decode < generate_limit:
			_run(ort_session_Main, binding_Main)
			outputs_Main = binding_Main.get_outputs()

			if use_penalty and num_decode >= PENALTY_RANGE:
				binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
				_run(ort_session_Penalty, binding_Penalty)

			if use_beam_search:
				if is_prefill_step:
					_bind_inputs(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
					_bind_device_outputs(binding_First_Beam, out_name_First_Beam_parts)
					binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[beam_score_out_idx], beam_score_buf)
					binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[beam_ids_out_idx], beam_ids_buf)
					binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[beam_max_idx], max_idx_buf)
					_run(ort_session_First_Beam, binding_First_Beam)
					outputs_Beam = binding_First_Beam.get_outputs()
				else:
					_bind_inputs(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
					_bind_device_outputs(binding_Second_Beam, out_name_Second_Beam_parts)
					if num_decode < 2:
						binding_Second_Beam.bind_ortvalue_input(
							in_name_Second_Beam[second_beam_previous_prob_idx],
							beam_score_buf,
						)
						binding_Second_Beam.bind_ortvalue_output(
							out_name_Second_Beam[beam_score_out_idx],
							beam_score_buf,
						)
						binding_Second_Beam.bind_ortvalue_output(
							out_name_Second_Beam[beam_ids_out_idx],
							beam_ids_buf,
						)
						binding_Second_Beam.bind_ortvalue_output(
							out_name_Second_Beam[beam_max_idx],
							max_idx_buf,
						)
					_run(ort_session_Second_Beam, binding_Second_Beam)
					outputs_Beam = binding_Second_Beam.get_outputs()

				max_token = max_idx_buf.numpy().flat[0]
				if max_token in stop_token_set:
					break

				save_id = outputs_Beam[num_keys_values]
				_bind_inputs(binding_Main, in_name_Main_kv, outputs_Beam[:num_keys_values])
				binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[beam_save_id_idx], save_id)
			else:
				if use_penalty:
					binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_obj)
					_run(ort_session_Greedy, binding_Greedy)
					save_id = binding_Greedy.get_outputs()[1]
				else:
					_run(ort_session_Argmax, binding_Argmax)

				max_token = max_idx_buf.numpy().flat[0]
				if max_token in stop_token_set:
					break

				if use_penalty:
					binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
				else:
					save_id_numpy[num_decode] = max_token

				_bind_inputs(binding_Main, in_name_Main_kv, outputs_Main[:num_keys_values])

			_bind_device_outputs(binding_Main, out_name_Main_kv)

			if is_prefill_step:
				binding_Main.bind_ortvalue_input(in_name_Main[main_hidden_idx], hidden_states_buf)
				binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_cos_idx], rotary_cos_buf)
				binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_sin_idx], rotary_sin_buf)
				binding_Main.bind_ortvalue_input(in_name_Main[main_attention_mask_idx], attention_mask_buf)
				binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)

				if use_penalty:
					binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
					binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

				if not use_beam_search and use_penalty:
					binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
				elif not use_beam_search:
					binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

				is_prefill_step = False

			_run(ort_session_Embed, binding_Embed)
			_run(ort_session_Rotary_Decode, binding_Rotary_Decode)
			num_decode += 1

		if use_beam_search or use_penalty:
			decoded_ids = save_id.numpy()[0]
		else:
			decoded_ids = save_id_numpy[:num_decode]

		raw_result = tokenizer.decode(decoded_ids, skip_special_tokens=True).strip()
		detected_language, asr_result = parse_asr_output(raw_result)
		t_total = time.time() - t0
		rtf = t_total / max(audio_len / SAMPLE_RATE, 1e-6)

		if detected_language:
			print(f"\nDetected language : {detected_language}")
		print(f"\nTranscription:\n  {asr_result}")
		print(f"\nRTF : {rtf:.3f}   total {t_total:.2f}s")
		print("-" * 70)


if __name__ == "__main__":
	main()
