import time
import numpy as np
import onnxruntime
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


def normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


def get_sess_info(model_path, opts, providers, p_opts, r_opts):
    sess = onnxruntime.InferenceSession(model_path, sess_options=opts, providers=providers, provider_options=p_opts, run_options=r_opts)
    inputs = [x.name for x in sess.get_inputs()]
    outputs = [x.name for x in sess.get_outputs()]
    return sess, inputs, outputs


def run_greedy_decoding(encoded_audio, encoded_len, limit):
    input_feed_C = {
        in_name_C[num_keys_values]: encoded_audio,
        in_name_C[num_keys_values_plus_1]: init_history_len,
        in_name_C[num_keys_values_plus_2]: encoded_len,
        in_name_C[num_keys_values_plus_3]: init_att_mask_1
    }
    for i in range(num_layers): input_feed_C[in_name_C[i]] = init_past_keys
    for i in range(num_layers, num_keys_values): input_feed_C[in_name_C[i]] = init_past_vals
    
    decoded_ids = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    local_result = ""
    num_decode = 0
    input_feed_B = {} 
    input_feed_D = {}
    input_feed_H = {}
    
    if do_repeat_penalty:
        input_feed_D = {in_name_D[1]: init_rp, in_name_D[2]: penality_val_ort, in_name_D[3]: init_batch_greedy}
        penalty_reset_count = 0 
    
    start_time = time.time()
    while num_decode < limit:
        outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C, run_options=run_options)
        logits_ort = outputs_C[num_keys_values]
        if do_repeat_penalty:
            input_feed_D[in_name_D[0]] = logits_ort
            outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D, run_options=run_options)
            max_logits_idx = outputs_D[0].numpy().flat[0]
            next_embed_input = outputs_D[0]
            if num_decode >= PENALITY_RANGE:
                reset_id = decoded_ids[penalty_reset_count]
                if reset_id != max_logits_idx:
                    rp_arr = outputs_D[1].numpy()
                    rp_arr[:, reset_id] = 1.0
                    input_feed_D[in_name_D[1]].update_inplace(rp_arr)
                penalty_reset_count += 1
            else:
                input_feed_D[in_name_D[1]] = outputs_D[1]
        else:
            input_feed_H[in_name_H] = logits_ort
            outputs_H = ort_session_H.run_with_ort_values(out_name_H, input_feed_H, run_options=run_options)[0]
            max_logits_idx = outputs_H.numpy().flat[0]
            next_embed_input = outputs_H
        if max_logits_idx in STOP_TOKEN:
            local_result += tokenizer.decode(decoded_ids[:num_decode], skip_special_tokens=True)
            break
        decoded_ids[num_decode] = max_logits_idx
        input_feed_C.update(zip(in_name_C[:num_keys_values], outputs_C))
        input_feed_B[in_name_B] = next_embed_input
        next_embed = ort_session_B.run_with_ort_values(out_name_B, input_feed_B, run_options=run_options)[0]
        input_feed_C[in_name_C[num_keys_values]] = next_embed
        input_feed_C[in_name_C[num_keys_values_plus_1]] = outputs_C[num_keys_values_plus_1]
        if num_decode < 1:
            input_feed_C[in_name_C[num_keys_values_plus_2]] = init_ids_len_1
            input_feed_C[in_name_C[num_keys_values_plus_3]] = init_att_mask_0
        num_decode += 1
        
    print(f"\nDecode (Greedy): {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")
    return local_result


def run_beam_decoding(encoded_audio, encoded_len, limit):
    input_feed_C = {
        in_name_C[num_keys_values]: encoded_audio,
        in_name_C[num_keys_values_plus_1]: init_history_len,
        in_name_C[num_keys_values_plus_2]: encoded_len,
        in_name_C[num_keys_values_plus_3]: init_att_mask_1
    }
    for i in range(num_layers): input_feed_C[in_name_C[i]] = init_past_keys
    for i in range(num_layers, num_keys_values): input_feed_C[in_name_C[i]] = init_past_vals

    local_result = ""
    num_decode = 0
    input_feed_E = {in_name_E[-2]: penality_val_ort, in_name_E[-1]: beam_size_ort}
    input_feed_F = {in_name_F[-3]: penality_val_ort, in_name_F[-2]: beam_size_ort, in_name_F[-1]: topK_ort}
    input_feed_G = {}
    input_feed_B = {}
    
    input_feed_E[in_name_E[num_keys_values_plus_1]] = init_save_id
    input_feed_E[in_name_E[num_keys_values_plus_2]] = init_rp
    
    if do_repeat_penalty:
        input_feed_G[in_name_G[2]] = init_reset_cnt

    start_time = time.time()
    while num_decode < limit:
        outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C, run_options=run_options)
        if num_decode < 1:
            input_feed_E.update(zip(in_name_E[:num_keys_values_plus_1], outputs_C))
            outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E, run_options=run_options)
            input_feed_F[in_name_F[-4]] = outputs_E[-2]
            if do_repeat_penalty:
                input_feed_G[in_name_G[3]] = outputs_E[-2]
            input_feed_C.update(zip(in_name_C[:num_keys_values], outputs_E))
            input_feed_B[in_name_B] = outputs_E[num_keys_values]
            input_feed_F[in_name_F[num_keys_values_plus_1]] = outputs_E[num_keys_values_plus_1]
            input_feed_F[in_name_F[num_keys_values_plus_2]] = outputs_E[num_keys_values_plus_2] 
            input_feed_F[in_name_F[num_keys_values_plus_3]] = outputs_E[num_keys_values_plus_3]
        else:
            input_feed_F.update(zip(in_name_F[:num_keys_values_plus_1], outputs_C))
            outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F, run_options=run_options)
            max_logits_idx = outputs_F[-1].numpy()
            if max_logits_idx in STOP_TOKEN:
                save_id = outputs_F[num_keys_values_plus_1].numpy()[0, :num_decode]
                local_result += tokenizer.decode(save_id, skip_special_tokens=True)
                break
            if do_repeat_penalty and (num_decode >= PENALITY_RANGE):
                input_feed_G[in_name_G[0]] = outputs_F[num_keys_values_plus_1]
                input_feed_G[in_name_G[1]] = outputs_F[num_keys_values_plus_2]
                outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G, run_options=run_options)
                input_feed_G[in_name_G[2]] = outputs_G[2]
                input_feed_F[in_name_F[num_keys_values_plus_1]] = outputs_G[0]
                input_feed_F[in_name_F[num_keys_values_plus_2]] = outputs_G[1]
            else:
                input_feed_F[in_name_F[num_keys_values_plus_1]] = outputs_F[num_keys_values_plus_1]
                input_feed_F[in_name_F[num_keys_values_plus_2]] = outputs_F[num_keys_values_plus_2]
            input_feed_F[in_name_F[num_keys_values_plus_3]] = outputs_F[num_keys_values_plus_3]
            input_feed_C.update(zip(in_name_C[:num_keys_values], outputs_F))
            input_feed_B[in_name_B] = outputs_F[num_keys_values]
        next_embed = ort_session_B.run_with_ort_values(out_name_B, input_feed_B, run_options=run_options)[0]
        input_feed_C[in_name_C[num_keys_values]] = next_embed
        input_feed_C[in_name_C[num_keys_values_plus_1]] = outputs_C[num_keys_values_plus_1]
        
        if num_decode == 0:
            input_feed_C[in_name_C[num_keys_values_plus_2]] = init_ids_len_1
            input_feed_C[in_name_C[num_keys_values_plus_3]] = init_att_mask_0
        num_decode += 1
    print(f"\nDecode (Beam): {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")
    return local_result


ort_session_A, in_name_A, out_name_A = get_sess_info(onnx_model_A, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]

ort_session_B, in_name_B, out_name_B = get_sess_info(onnx_model_B, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
in_name_B, out_name_B = in_name_B[0], [out_name_B[0]]

ort_session_C, in_name_C, out_name_C = get_sess_info(onnx_model_C, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
print(f"\nUsable Providers: {ort_session_C.get_providers()}")

model_dtype_str = ort_session_C._inputs_meta[-2].type
model_dtype = np.float16 if 'float16' in model_dtype_str else np.float32
amount_of_outputs_C = len(out_name_C)
num_layers = (amount_of_outputs_C - 2) // 2
num_keys_values = num_layers * 2
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
vocab_size = ort_session_C._outputs_meta[num_keys_values].shape[1]
do_repeat_penalty = (REPEAT_PENALITY != 1.0)

if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search: TOP_K adjusted to match BEAM_SIZE.")
    TOP_K = BEAM_SIZE
if (TOP_K < 2) or (BEAM_SIZE < 2):
    if USE_BEAM_SEARCH:
        print("\nBeam Search settings too low. Falling back to Greedy Search.")
    USE_BEAM_SEARCH = False

ort_session_D, ort_session_E, ort_session_F, ort_session_G, ort_session_H = None, None, None, None, None
in_name_D, out_name_D = [], []
in_name_E, out_name_E = [], []
in_name_F, out_name_F = [], []
in_name_G, out_name_G = [], []
in_name_H, out_name_H = "", []

if USE_BEAM_SEARCH:
    ort_session_E, in_name_E, out_name_E = get_sess_info(onnx_model_E, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
    ort_session_F, in_name_F, out_name_F = get_sess_info(onnx_model_F, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
    ort_session_G, in_name_G, out_name_G = get_sess_info(onnx_model_G, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
else:
    BEAM_SIZE = 1
    if do_repeat_penalty:
        ort_session_D, in_name_D, out_name_D = get_sess_info(onnx_model_D, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
    else:
        ort_session_H, in_name_H_list, out_name_H = get_sess_info(onnx_model_H, session_opts, ORT_Accelerate_Providers, provider_options, run_options)
        in_name_H = in_name_H_list[0]

generate_limit = MAX_SEQ_LEN - 20
topK_ort = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size_ort = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_val_ort = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([REPEAT_PENALITY], dtype=model_dtype), device_type, DEVICE_ID)

init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_batch_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_att_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_att_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
init_rp = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
init_save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
init_reset_cnt = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)

dim_k = ort_session_C._inputs_meta[0].shape[3]
dim_v = ort_session_C._inputs_meta[num_layers].shape[4]
dim_k1 = ort_session_C._inputs_meta[0].shape[1]
dim_v1 = ort_session_C._inputs_meta[num_layers].shape[1]
kv_device = 'cpu' if device_type == 'dml' else device_type
init_past_keys = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, dim_k1, 1, dim_k, 0), dtype=np.float16), kv_device, 0 if kv_device == 'cpu' else DEVICE_ID)
init_past_vals = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, dim_v1, 1, 0, dim_v), dtype=np.float16), kv_device, 0 if kv_device == 'cpu' else DEVICE_ID)
input_feed_A = {}

init_all_outputs_B = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids_ort = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    init_all_outputs_B.append(ort_session_B.run_with_ort_values(out_name_B, {in_name_B: input_ids_ort}, run_options=run_options)[0])

for prompt_embed, test_file in zip(init_all_outputs_B, test_audio):
    print("-" * 105)
    print(f"\nTest Input Audio: {test_file}")
    audio = np.array(AudioSegment.from_file(test_file).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_NORMALIZER: audio = normalizer(audio, 8192.0)
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
    final_asr_result = ""
    slice_start = 0
    rtf_time = time.time()
    while slice_start + INPUT_AUDIO_LENGTH <= aligned_len:
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        input_feed_A[in_name_A[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start: slice_end], device_type, DEVICE_ID)
        input_feed_A[in_name_A[1]] = prompt_embed
        outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A, run_options=run_options)
        encoded_audio_ort = outputs_A[0]
        encoded_len = outputs_A[1].numpy()
        current_limit = generate_limit - encoded_len
        if USE_BEAM_SEARCH:
            res = run_beam_decoding(encoded_audio_ort, outputs_A[1], current_limit)
        else:
            res = run_greedy_decoding(encoded_audio_ort, outputs_A[1], current_limit)
        final_asr_result += res
        slice_start += stride_step
        
    print(final_asr_result, end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_full_len / SAMPLE_RATE)):.3f}")
    print("-" * 105)

