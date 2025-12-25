import time
import numpy as np
import onnxruntime
from pydub import AudioSegment
from transformers import AutoTokenizer

tokenizer_path = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B'                                       # Set the tokenizer path.
onnx_model_A = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Encoder.onnx'                      # The exported onnx model path.
onnx_model_B = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Embed.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Main.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Greedy_Search.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_First_Beam_Search.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Second_Beam_Search.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized/FunASR_Nano_Decoder_Reset_Penality.onnx'

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
REPEAT_PENALITY = 0.8                # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                  # Penalizes the most recent output. "10" means the last 10 tokens.

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
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    provider_options = None


def normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


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
run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(len(out_name_A))]


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B = in_name_B[0].name
out_name_B = [out_name_B[0].name]


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
print(f"\nUsable Providers: {ort_session_C.get_providers()}")
model_dtype = ort_session_C._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
amount_of_outputs_C = len(out_name_C)
in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
out_name_C = [out_name_C[i].name for i in range(amount_of_outputs_C)]


generate_limit = MAX_SEQ_LEN - 20                   # 20 = length of basic ids
num_layers = (amount_of_outputs_C - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7
vocab_size = ort_session_C._outputs_meta[num_keys_values].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([REPEAT_PENALITY], dtype=model_dtype), device_type, DEVICE_ID)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")
    TOP_K = BEAM_SIZE


if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")


if USE_BEAM_SEARCH:
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    amount_of_outputs_F = len(out_name_F)
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(amount_of_outputs_F)]
    
    ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_G = ort_session_G.get_inputs()
    out_name_G = ort_session_G.get_outputs()
    in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
    out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]

    input_feed_E = {
        in_name_E[num_keys_values_plus_3]: penality_value,
        in_name_E[num_keys_values_plus_4]: beam_size
    }

    input_feed_F = {
        in_name_F[num_keys_values_plus_5]: penality_value,
        in_name_F[num_keys_values_plus_6]: beam_size,
        in_name_F[num_keys_values_plus_7]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    input_feed_D = {in_name_D[2]: penality_value}


if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False


init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
if device_type != 'dml':
    init_past_keys_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[0].shape[1], 1, ort_session_C._inputs_meta[0].shape[3], 0), dtype=model_dtype), device_type, DEVICE_ID)
    init_past_values_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[num_layers].shape[1], 1, 0, ort_session_C._inputs_meta[num_layers].shape[4]), dtype=model_dtype), device_type, DEVICE_ID)
else:
    init_past_keys_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[0].shape[1], 1, ort_session_C._inputs_meta[0].shape[3], 0), dtype=model_dtype), 'cpu', 0)
    init_past_values_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[num_layers].shape[1], 1, 0, ort_session_C._inputs_meta[num_layers].shape[4]), dtype=model_dtype), 'cpu', 0)
init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
init_batch_size_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)


if USE_BEAM_SEARCH:
    input_feed_E[in_name_E[num_keys_values_plus_1]] = init_save_id_beam
    input_feed_E[in_name_E[num_keys_values_plus_2]] = init_repeat_penality
else:
    input_feed_D[in_name_D[1]] = init_repeat_penality
    input_feed_D[in_name_D[3]] = init_batch_size_greedy


if do_repeat_penality:
    if USE_BEAM_SEARCH:
        input_feed_G = {in_name_G[2]: penality_reset_count_beam_init}
    else:
        penality_reset_count_greedy = 0


# Start to run FunASR-Nano
input_feed_A = {}
input_feed_C = {}

init_all_outputs_B = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    input_feed_B = {in_name_B: input_ids}
    init_all_outputs_B.append(ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0])

# Load the input audio
for prompt_embed, test in zip(init_all_outputs_B, test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_NORMALIZER:
        audio = normalizer(audio, 8192.0)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in_A, str):
        INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_len)  # You can adjust it.
    else:
        INPUT_AUDIO_LENGTH = shape_value_in_A
    if SLIDING_WINDOW <= 0:
        stride_step = INPUT_AUDIO_LENGTH
    else:
        stride_step = SLIDING_WINDOW
    if audio_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]

    asr_result = ""
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    rtf_time = time.time()
    while slice_end <= aligned_len:
        input_feed_A[in_name_A[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start: slice_end], device_type, DEVICE_ID)
        input_feed_A[in_name_A[1]] = prompt_embed
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
        input_feed_C[in_name_C[num_keys_values]] = all_outputs_A[0]
        input_feed_C[in_name_C[num_keys_values_plus_1]] = init_history_len
        input_feed_C[in_name_C[num_keys_values_plus_2]] = all_outputs_A[1]
        input_feed_C[in_name_C[num_keys_values_plus_3]] = init_attention_mask_1
        for i in range(num_layers):
            input_feed_C[in_name_C[i]] = init_past_keys_C
        for i in range(num_layers, num_keys_values):
            input_feed_C[in_name_C[i]] = init_past_values_C

        if USE_BEAM_SEARCH:
            input_feed_E[in_name_E[num_keys_values_plus_1]] = init_save_id_beam
            input_feed_E[in_name_E[num_keys_values_plus_2]] = init_repeat_penality
            if do_repeat_penality:
                input_feed_G[in_name_G[2]] = penality_reset_count_beam_init
        else:
            input_feed_D[in_name_D[1]] = init_repeat_penality
            penality_reset_count_greedy = 0

        num_decode = 0
        limit = generate_limit - all_outputs_A[1].numpy()
        start_time = time.time()
        while num_decode < limit:
            all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    input_feed_E.update(zip(in_name_E[:num_keys_values_plus_1], all_outputs_C))
                    all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    max_logits_idx = all_outputs_E[num_keys_values_plus_5].numpy()
                    input_feed_F[in_name_F[num_keys_values_plus_4]] = all_outputs_E[num_keys_values_plus_4]
                    if do_repeat_penality:
                        input_feed_G[in_name_G[3]] = all_outputs_E[num_keys_values_plus_4]
                else:
                    input_feed_F.update(zip(in_name_F[:num_keys_values_plus_1], all_outputs_C))
                    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                    max_logits_idx = all_outputs_F[num_keys_values_plus_4].numpy()
                if max_logits_idx in STOP_TOKEN:
                    save_id = all_outputs_F[num_keys_values_plus_1].numpy()[0, :num_decode]  # 0 is the Top_1
                    asr_result += tokenizer.decode(save_id, skip_special_tokens=True)
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    input_feed_G[in_name_G[0]] = all_outputs_F[num_keys_values_plus_1]
                    input_feed_G[in_name_G[1]] = all_outputs_F[num_keys_values_plus_2]
                    all_outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)
                    input_feed_G[in_name_G[2]] = all_outputs_G[2]
                    input_feed_F[in_name_F[num_keys_values_plus_1]] = all_outputs_G[0]
                    input_feed_F[in_name_F[num_keys_values_plus_2]] = all_outputs_G[1]
                if num_decode < 1:
                    input_feed_C.update(zip(in_name_C[:num_keys_values], all_outputs_E))
                    input_feed_B[in_name_B] = all_outputs_E[num_keys_values]
                    input_feed_F[in_name_F[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_F[in_name_F[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
                    input_feed_F[in_name_F[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
                else:
                    input_feed_C.update(zip(in_name_C[:num_keys_values], all_outputs_F))
                    input_feed_B[in_name_B] = all_outputs_F[num_keys_values]
                    input_feed_F[in_name_F[num_keys_values_plus_1]] = all_outputs_F[num_keys_values_plus_1]
                    input_feed_F[in_name_F[num_keys_values_plus_2]] = all_outputs_F[num_keys_values_plus_2]
                    input_feed_F[in_name_F[num_keys_values_plus_3]] = all_outputs_F[num_keys_values_plus_3]
                input_feed_C[in_name_C[num_keys_values]] = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0]
            else:
                input_feed_D[in_name_D[0]] = all_outputs_C[num_keys_values]
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                max_logits_idx = all_outputs_D[0].numpy()[0, 0]
                if max_logits_idx in STOP_TOKEN:
                    asr_result += tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    reset_ids = save_id_greedy[penality_reset_count_greedy]
                    if reset_ids != max_logits_idx:
                        repeat_penality = all_outputs_D[1].numpy()
                        repeat_penality[:, reset_ids] = 1.0
                        input_feed_D[in_name_D[1]].update_inplace(repeat_penality)
                    penality_reset_count_greedy += 1
                else:
                    input_feed_D[in_name_D[1]] = all_outputs_D[1]
                input_feed_D[in_name_D[0]] = all_outputs_D[0]
                input_feed_C.update(zip(in_name_C[:num_keys_values], all_outputs_C))
                input_feed_B[in_name_B] = all_outputs_D[0]
                input_feed_C[in_name_C[num_keys_values]] = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0]
                save_id_greedy[num_decode] = max_logits_idx
            input_feed_C[in_name_C[num_keys_values_plus_1]] = all_outputs_C[num_keys_values_plus_1]
            if num_decode < 1:
                input_feed_C[in_name_C[num_keys_values_plus_2]] = init_ids_len_1
                input_feed_C[in_name_C[num_keys_values_plus_3]] = init_attention_mask_0
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")
    print(asr_result, end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_len / SAMPLE_RATE)):.3f}")
    print("----------------------------------------------------------------------------------------------------------")
