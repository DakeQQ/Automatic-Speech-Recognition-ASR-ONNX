import time
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer

tokenizer_path = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B'                                  # Set the tokenizer path.
onnx_model_A = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Encoder.onnx'                 # The exported onnx model path.
onnx_model_B = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Embed.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Main.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/Greedy_Search.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/First_Beam_Search.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/Second_Beam_Search.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/Reset_Penality.onnx'
onnx_model_H = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/Argmax.onnx'


# The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/yue.mp3", "./example/ja.mp3"]       # The test audio list.
task_prompt = ["将语音转写成中文：", "将语音转写成英文：", "将语音转写成粤语：", "将语音转写成日文："]           # The prompt of transcription task.


# official_demo_prompt = """请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。
# 
# 
# **上下文信息：**
# 
# 
# 热词列表：[开放时间]
# 语音转写成中文：
# 
# """

# Audio & STFT Configuration
SAMPLE_RATE = 16000                  # The model parameter, do not edit the value.
USE_NORMALIZER = True                # If true, use the audio normalizer to make the loudness consistent.

STOP_TOKEN = [151643, 151645]        # The stop_id in Qwen is "151643" & "151645"
MAX_SEQ_LEN = 1024                   # The max context length.

# Input & Processing Limits
MAX_INPUT_AUDIO_LENGTH = 320000      # The maximum input audio length.
SLIDING_WINDOW = 0                   # Set the sliding window step for test audio reading; use 0 to disable.

# Decoding Strategy
USE_BEAM_SEARCH = False              # It recommended to use greedy search for Fun-ASR-Nano.
TOP_K = 3                            # The top k candidate in decoding.
BEAM_SIZE = 3                        # Number of beams in searching.
MAX_BEAM_SIZE = 10                   # Max beams for exported model.
REPEAT_PENALITY = 1.0                # Range from 0.0 to 1.0; "1.0" means no penality.
PENALTY_RANGE = 10                   # Penalizes the most recent output. "10" means the last 10 tokens.

# Runtime & Export Settings
MAX_THREADS = 0                      # Parllel CPU threads. Set 0 for auto.
DEVICE_ID = 0                        # Default to zero.
ORT_Accelerate_Providers = []        # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                     # else keep empty.

# ONNX Runtime settings
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,  # The default value is 8. Edit freely.
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '0',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
    provider_options = None


def normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
session_opts.log_severity_level = 4                 # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                # Fatal level, it an adjustable value.
run_options.log_severity_level = 4                  # Fatal level, it an adjustable value.
run_options.log_verbosity_level = 4                 # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')


def bind_ort_values(binding, names, values, num=0):
    if num != 0:
        for i in range(num):
            binding.bind_ortvalue_input(names[i], values[i])
    else:
        for name, val in zip(names, values):
            binding.bind_ortvalue_input(name, val)


def bind_outputs_generic(binding, output_names, device_type):
    for name in output_names:
        binding._iobinding.bind_output(name, device_type)


def create_ortvalue(data, dtype, device_type, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device_type, device_id)


_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_A = ort_session_A.io_binding()
shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = [x.name for x in ort_session_A.get_inputs()]
out_name_A = [x.name for x in ort_session_A.get_outputs()]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_B = ort_session_B.io_binding()
in_name_B = ort_session_B.get_inputs()[0].name
out_name_B = [ort_session_B.get_outputs()[0].name]

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_C = ort_session_C.io_binding()
print(f"\nUsable Providers: {ort_session_C.get_providers()}")

in_meta_C = ort_session_C.get_inputs()
out_meta_C = ort_session_C.get_outputs()
in_name_C = [x.name for x in in_meta_C]
out_name_C = [x.name for x in out_meta_C]
amount_of_outputs_C = len(out_name_C)
in_name_C_parts = in_name_C[:-2]

num_layers = (amount_of_outputs_C - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
vocab_size = ort_session_C._outputs_meta[num_keys_values].shape[1]

generate_limit = MAX_SEQ_LEN - 20
topK = create_ortvalue([TOP_K], np.int64, device_type, DEVICE_ID)
beam_size = create_ortvalue([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")
    BEAM_SIZE = 1

do_repeat_penalty = (REPEAT_PENALITY != 1.0)

if USE_BEAM_SEARCH:
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_E = ort_session_E.io_binding()
    in_name_E = [x.name for x in ort_session_E.get_inputs()]
    out_name_E = [x.name for x in ort_session_E.get_outputs()]
    in_name_E_parts = in_name_E[:num_keys_values_plus_1]
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_F = ort_session_F.io_binding()
    in_name_F = [x.name for x in ort_session_F.get_inputs()]
    out_name_F = [x.name for x in ort_session_F.get_outputs()]
    in_name_F_parts = in_name_F[:num_keys_values_plus_1]
    ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_G = ort_session_G.io_binding()
    in_name_G = [x.name for x in ort_session_G.get_inputs()]
    out_name_G = [x.name for x in ort_session_G.get_outputs()]
    penality_dtype = np.float16 if 'float16' in ort_session_E._inputs_meta[num_keys_values_plus_4].type else np.float32
    penality_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
    init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
    init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
    init_penality_reset_count = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros([BEAM_SIZE, 1], dtype=np.int32), device_type, DEVICE_ID)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], init_save_id_beam)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], init_repeat_penality)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_3], penality_value)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_4], beam_size)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_4], penality_value)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_5], beam_size)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_6], topK)
    binding_G.bind_ortvalue_input(in_name_G[2], init_penality_reset_count)
else:
    BEAM_SIZE = 1
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if do_repeat_penalty:
        ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_D = ort_session_D.io_binding()
        in_name_D = [x.name for x in ort_session_D.get_inputs()]
        out_name_D = [x.name for x in ort_session_D.get_outputs()]
        penality_dtype = np.float16 if 'float16' in ort_session_D._inputs_meta[2].type else np.float32
        penality_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
        penalty_shape = (BEAM_SIZE, vocab_size)
        init_penalty = np.ones(penalty_shape, dtype=penality_dtype)
        current_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(init_penalty, device_type, DEVICE_ID)
        next_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(init_penalty, device_type, DEVICE_ID)
        binding_D.bind_ortvalue_input(in_name_D[2], penality_value)
        binding_D.bind_output(name=out_name_D[0], device_type=device_type, device_id=DEVICE_ID)
        binding_D.bind_output(name=out_name_D[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
        init_penality_reset_count = 0
    else:
        ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_H = ort_session_H.io_binding()
        in_name_H = ort_session_H.get_inputs()[0].name
        out_name_H = [ort_session_H.get_outputs()[0].name]

init_ids_len_1 = create_ortvalue([1], np.int64, device_type, DEVICE_ID)
init_history_len = create_ortvalue([0], np.int64, device_type, DEVICE_ID)
init_attention_mask_0 = create_ortvalue([0], np.int8, device_type, DEVICE_ID)
init_attention_mask_1 = create_ortvalue([1], np.int8, device_type, DEVICE_ID)

if 'dml' in device_type:
    kv_device = 'cpu'
    kv_device_id = 0
else:
    kv_device = device_type
    kv_device_id = DEVICE_ID

init_past_keys_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[0].shape[1], 1, ort_session_C._inputs_meta[0].shape[3], 0), dtype=np.float16), kv_device, kv_device_id)
init_past_values_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[num_layers].shape[1], 1, 0, ort_session_C._inputs_meta[num_layers].shape[4]), dtype=np.float16), kv_device, kv_device_id)

init_all_outputs_B = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    binding_B.bind_ortvalue_input(in_name_B, input_ids)
    bind_outputs_generic(binding_B, out_name_B, _ort_device_type)
    ort_session_B.run_with_iobinding(binding_B, run_options=run_options)
    init_all_outputs_B.append(onnxruntime.OrtValue.ortvalue_from_numpy(binding_B.get_outputs()[0].numpy(), device_type, DEVICE_ID))

for prompt_embed, test in zip(init_all_outputs_B, test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_NORMALIZER: 
        audio = normalizer(audio, 8192.0)
    audio_full_len = len(audio)
    INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_full_len) if isinstance(shape_value_in_A, str) else shape_value_in_A
    stride_step = INPUT_AUDIO_LENGTH if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    audio = audio.reshape(1, 1, -1)
    if audio_full_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_full_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        pad_amount = ((num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH) - audio_full_len
        zeros = np.zeros([1, 1, pad_amount], dtype=audio.dtype)
        audio = np.concatenate((audio, zeros), axis=-1)
    elif audio_full_len < INPUT_AUDIO_LENGTH:
        zeros = np.zeros([1, 1, INPUT_AUDIO_LENGTH - audio_full_len], dtype=audio.dtype)
        audio = np.concatenate((audio, zeros), axis=-1)
    aligned_len = audio.shape[-1]
    asr_result = ""
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    rtf_time = time.time()
    while slice_end <= aligned_len:
        audio_slice = audio[..., slice_start: slice_end]
        ort_audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio_slice, device_type, DEVICE_ID)
        binding_A.bind_ortvalue_input(in_name_A[0], ort_audio)
        binding_A.bind_ortvalue_input(in_name_A[1], prompt_embed)
        bind_outputs_generic(binding_A, out_name_A, _ort_device_type)
        ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
        all_outputs_A = binding_A.get_outputs()
        i = 0
        j = num_layers
        while i < j:
            binding_C.bind_ortvalue_input(in_name_C[i], init_past_keys_C)
            i += 1
        j = i + num_layers
        while i < j:
            binding_C.bind_ortvalue_input(in_name_C[i], init_past_values_C)
            i += 1
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values], all_outputs_A[0])
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_1], init_history_len)
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_2], all_outputs_A[1])
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_3], init_attention_mask_1)
        if USE_BEAM_SEARCH:
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], init_save_id_beam)
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], init_repeat_penality)
            if do_repeat_penalty:
                binding_G.bind_ortvalue_input(in_name_G[2], init_penality_reset_count)
        else:
            if do_repeat_penalty:
                current_penalty.update_inplace(init_penalty)
                next_penalty.update_inplace(init_penalty)
                binding_D.bind_output(name=out_name_D[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
                init_penality_reset_count = 0

        num_decode = 0
        limit = generate_limit - all_outputs_A[1].numpy()
        start_time = time.time()
        while num_decode < limit:
            bind_outputs_generic(binding_C, out_name_C, _ort_device_type)
            ort_session_C.run_with_iobinding(binding_C, run_options=run_options)
            all_outputs_C = binding_C.get_outputs()
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    bind_ort_values(binding_E, in_name_E_parts, all_outputs_C)
                    bind_outputs_generic(binding_E, out_name_E, _ort_device_type)
                    ort_session_E.run_with_iobinding(binding_E, run_options=run_options)
                    all_outputs_E = binding_E.get_outputs()
                    max_logits_idx = all_outputs_E[num_keys_values_plus_4].numpy()
                    if max_logits_idx in STOP_TOKEN:
                        print("\nBad first token generated, stopping decoding.\n")
                        break
                else:
                    bind_ort_values(binding_F, in_name_F_parts, all_outputs_C)
                    bind_outputs_generic(binding_F, out_name_F, _ort_device_type)
                    ort_session_F.run_with_iobinding(binding_F, run_options=run_options)
                    all_outputs_F = binding_F.get_outputs()
                    max_logits_idx = all_outputs_F[num_keys_values_plus_4].numpy()
                    if max_logits_idx in STOP_TOKEN:
                        break
                if do_repeat_penalty and (num_decode >= PENALTY_RANGE):
                    binding_G.bind_ortvalue_input(in_name_G[0], all_outputs_F[num_keys_values_plus_1])
                    binding_G.bind_ortvalue_input(in_name_G[1], all_outputs_F[num_keys_values_plus_2])
                    bind_outputs_generic(binding_G, out_name_G, _ort_device_type)
                    ort_session_G.run_with_iobinding(binding_G, run_options=run_options)
                    all_outputs_G = binding_G.get_outputs()
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_2], all_outputs_G[0])
                    binding_G.bind_ortvalue_input(in_name_G[2], all_outputs_G[1])
                if num_decode < 1:
                    bind_ort_values(binding_C, in_name_C_parts, all_outputs_E)
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_E[num_keys_values])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_1], all_outputs_E[num_keys_values_plus_1])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_2], all_outputs_E[num_keys_values_plus_2])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_3], all_outputs_E[num_keys_values_plus_3])
                else:
                    bind_ort_values(binding_C, in_name_C_parts, all_outputs_F)
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_F[num_keys_values])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_1], all_outputs_F[num_keys_values_plus_1])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_2], all_outputs_F[num_keys_values_plus_2])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_3], all_outputs_F[num_keys_values_plus_3])
            else:
                if do_repeat_penalty:
                    binding_D.bind_ortvalue_input(in_name_D[0], all_outputs_C[num_keys_values])
                    binding_D.bind_ortvalue_input(in_name_D[1], current_penalty)
                    ort_session_D.run_with_iobinding(binding_D, run_options=run_options)
                    all_outputs_D = binding_D.get_outputs()
                    max_logits_idx = all_outputs_D[0].numpy().flat[0]
                    if max_logits_idx in STOP_TOKEN:
                        break
                    if num_decode >= PENALTY_RANGE:
                        reset_ids = save_id_greedy[init_penality_reset_count]
                        if reset_ids != max_logits_idx:
                            tmp = next_penalty.numpy()
                            tmp[:, reset_ids] = 1.0
                            next_penalty.update_inplace(tmp)
                        init_penality_reset_count += 1
                    current_penalty, next_penalty = next_penalty, current_penalty
                    binding_D.bind_output(name=out_name_D[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_D[0])
                else:
                    binding_H.bind_ortvalue_input(in_name_H, all_outputs_C[num_keys_values])
                    bind_outputs_generic(binding_H, out_name_H, _ort_device_type)
                    ort_session_H.run_with_iobinding(binding_H)
                    all_outputs_H = binding_H.get_outputs()
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_H[0])
                    max_logits_idx = all_outputs_H[0].numpy().flat[0]
                    if max_logits_idx in STOP_TOKEN:
                        break
                bind_ort_values(binding_C, in_name_C_parts, all_outputs_C)
                save_id_greedy[num_decode] = max_logits_idx
            bind_outputs_generic(binding_B, out_name_B, _ort_device_type)
            ort_session_B.run_with_iobinding(binding_B)
            binding_C.bind_ortvalue_input(in_name_C[num_keys_values], binding_B.get_outputs()[0])
            binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_1], all_outputs_C[num_keys_values_plus_1])
            if num_decode < 1:
                binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_2], init_ids_len_1)
                binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_3], init_attention_mask_0)
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        if num_decode > 0:
            if USE_BEAM_SEARCH:
                asr_result += tokenizer.decode(all_outputs_F[num_keys_values_plus_1].numpy()[0, :num_decode], skip_special_tokens=True)
            else:
                asr_result += tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")
    print(asr_result, end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_full_len / SAMPLE_RATE)):.3f}")
    print("----------------------------------------------------------------------------------------------------------")

