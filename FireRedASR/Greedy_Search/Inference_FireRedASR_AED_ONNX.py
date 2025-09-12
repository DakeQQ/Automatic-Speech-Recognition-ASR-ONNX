import sys
import time
import numpy as np
import onnxruntime
from pydub import AudioSegment


project_path = "/home/DakeQQ/Downloads/FireRedASR-main"                                           # The FireRedASR Github project path.
download_path = "/home/DakeQQ/Downloads/FireRedASR-AED-L"                                         # The FireRedASR-AED model download path.
onnx_model_A = "/home/DakeQQ/Downloads/FireRedASR_Optimized/FireRedASR_AED_L-Encoder.onnx"        # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/FireRedASR_Optimized/FireRedASR_AED_L-Decoder.onnx"        # The exported onnx model path.
test_audio = ["../example/zh.mp3", "../example/zh_1.wav", "../example/zh_2.wav"]                  # The test audio list.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
MAX_THREADS = 4                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SEQ_LEN = 80                        # It should keep the same with exported model.
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
STOP_TOKEN = [4]                        # 4 is the end token for FireRedASR series model.

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
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
            'fuse_conv_bias': '0',
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
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


if project_path not in sys.path:
    sys.path.append(project_path)

from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                  # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
output_names_A = []
for i in range(len(out_name_A)):
    output_names_A.append(out_name_A[i].name)

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
model_dtype = ort_session_B._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32

in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]

generate_limit = MAX_SEQ_LEN - 1  # 1 = length of initial input_ids
num_layers = (amount_of_outputs_B - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
tokenizer = ChineseCharEnglishSpmTokenizer(download_path + "/dict.txt", download_path + "/train_bpe1000.model")

# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(320000, audio_len)  # You can adjust it.
    else:
        INPUT_AUDIO_LENGTH = shape_value_in
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

    # Start to run Whisper
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    input_ids = np.array([[3]], dtype=np.int32)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
    ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
    attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
    attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[0].shape[0], ort_session_B._inputs_meta[0].shape[1], 0), dtype=model_dtype), device_type, DEVICE_ID)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[num_layers].shape[0], 0, ort_session_B._inputs_meta[num_layers].shape[2]), dtype=model_dtype), device_type, DEVICE_ID)
    layer_indices = np.arange(num_keys_values_plus_2, num_keys_values_plus_2 + num_keys_values, dtype=np.int32)
    input_feed_B = {
        in_name_B[num_keys_values]: input_ids,
        in_name_B[num_keys_values_plus_1]: history_len,
        in_name_B[-2]: ids_len,
        in_name_B[-1]: attention_mask_1,
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B

    num_decode = 0
    save_token = []
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {
            in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], device_type, DEVICE_ID)})
        for i in range(num_keys_values):
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-2])[0, 0]
            if max_logit_ids in STOP_TOKEN:
                break
            for i in range(amount_of_outputs_B):
                input_feed_B[in_name_B[i]] = all_outputs_B[i]
            if num_decode < 2:
                input_feed_B[in_name_B[-1]] = attention_mask_0
                input_feed_B[in_name_B[-2]] = ids_len_1
            save_token.append(max_logit_ids)
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    text = ("".join([tokenizer.dict[int(id)] for id in save_token])).replace(tokenizer.SPM_SPACE, ' ').strip()
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")

if project_path in sys.path:
    sys.path.remove(project_path)
