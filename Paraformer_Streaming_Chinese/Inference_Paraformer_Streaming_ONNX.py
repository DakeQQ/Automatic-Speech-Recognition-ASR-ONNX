import json
import time
import numpy as np
import onnxruntime
from pydub import AudioSegment


tokens_path = "/home/DakeQQ/Downloads/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/tokens.json"    # The Paraformer tokens download path.
onnx_model_A = "/home/DakeQQ/Downloads/Paraformer_Optimized/Paraformer_Streaming_Encoder.onnx"                          # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Paraformer_Optimized/Paraformer_Streaming_Decoder.onnx"                          # The exported onnx model path.
test_audio = "./zh.wav"                                                                                                 # The test audio path.

# Only support CPU and f32, q8f32, currently.
ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
FSMN_DE_PAD = 10                        # The model parameter, do not edit the value.
MAX_THREADS = 8                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
            'num_streams': 1,
            'enable_opencl_throttling': True,
            'enable_qdq_optimizer': False                 # Enable it carefully
        }
    ]
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 8 * 1024 * 1024 * 1024,      # 8 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'cudnn_conv_use_max_workspace': '1',
            'do_copy_in_default_stream': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'use_tf32': '0'
        }
    ]
else:
    # Please config by yourself for others providers.
    provider_options = None


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


with open(tokens_path, 'r', encoding='UTF-8') as json_file:
    tokenizer = np.array(json.load(json_file), dtype=np.str_)


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
shape_value_in = ort_session_A._inputs_meta[-1].shape[-1]
dtype_A = ort_session_A._inputs_meta[0].type
if "float16" in dtype_A:
    dtype_A = np.float16
else:
    dtype_A = np.float32
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
input_names_A = []
output_names_A = []
amount_of_outputs_A = len(out_name_A)
for i in range(len(in_name_A)):
    input_names_A.append(in_name_A[i].name)
for i in range(amount_of_outputs_A):
    output_names_A.append(out_name_A[i].name)


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
dtype_B = ort_session_B._inputs_meta[0].type
if "float16" in dtype_B:
    dtype_B = np.float16
else:
    dtype_B = np.float32
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
input_names_B = []
output_names_B = []
amount_of_outputs_B = len(out_name_B)
for i in range(len(in_name_B)):
    input_names_B.append(in_name_B[i].name)
for i in range(amount_of_outputs_B):
    output_names_B.append(out_name_B[i].name)


# Load the input audio
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(8800, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
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


ORT_Accelerate_Providers = ort_session_A.get_providers()[0]
print(f"\nUsable Providers: {ORT_Accelerate_Providers}\n")
if "CUDAExecutionProvider" in ORT_Accelerate_Providers or "TensorrtExecutionProvider" in ORT_Accelerate_Providers:
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    device_type = 'dml'
else:
    device_type = 'cpu'


# Initialize
amount_of_outputs_B -= 1   # The last 1 is max_logit_ids; pass it to the tokenizer.
num_layer_de = amount_of_outputs_B // 3
num_layer_en = (amount_of_outputs_A - 7) // 2
amount_of_outputs_A -= 3   # The last 3 are for model_B.

in_en_value_start = num_layer_en
in_previous_mel_features_start = in_en_value_start + num_layer_en
in_cif_hidden_start = in_previous_mel_features_start + 1
in_de_key_start = num_layer_de
in_de_value_start = in_de_key_start + num_layer_de
in_de_value_end = in_de_value_start + num_layer_de

in_en_key_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[0].shape[0], ort_session_A._inputs_meta[0].shape[1], 0), dtype=dtype_A), device_type, DEVICE_ID)
in_en_value_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[in_en_value_start].shape[0], 0, ort_session_A._inputs_meta[in_en_value_start].shape[2]), dtype=dtype_A), device_type, DEVICE_ID)
in_previous_mel_features = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[in_previous_mel_features_start].shape[0], ort_session_A._inputs_meta[in_previous_mel_features_start].shape[1], ort_session_A._inputs_meta[in_previous_mel_features_start].shape[2]), dtype=dtype_A), device_type, DEVICE_ID)
in_cif_hidden = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[in_cif_hidden_start].shape[0], ort_session_A._inputs_meta[in_cif_hidden_start].shape[1], ort_session_A._inputs_meta[in_cif_hidden_start].shape[2]), dtype=dtype_A), device_type, DEVICE_ID)
in_cif_alpha = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=dtype_A), device_type, DEVICE_ID)
start_idx = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=np.int64), device_type, DEVICE_ID)
in_de_fsmn_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[0].shape[0], ort_session_B._inputs_meta[0].shape[1], FSMN_DE_PAD), dtype=dtype_B), device_type, DEVICE_ID)
in_de_key_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[in_de_key_start].shape[0], ort_session_B._inputs_meta[in_de_key_start].shape[1], 0), dtype=dtype_B), device_type, DEVICE_ID)
in_de_value_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[in_de_value_start].shape[0], 0, ort_session_B._inputs_meta[in_de_value_start].shape[2]), dtype=dtype_B), device_type, DEVICE_ID)


def Initialize():
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    input_feed_B = {}
    input_feed_A = {
        in_name_A[-2].name: start_idx,
        in_name_A[-3].name: in_cif_alpha,
        in_name_A[-4].name: in_cif_hidden,
        in_name_A[-5].name: in_previous_mel_features
    }
    for i in range(num_layer_en):
        input_feed_A[in_name_A[i].name] = in_en_key_A
    for i in range(in_en_value_start, in_previous_mel_features_start):
        input_feed_A[in_name_A[i].name] = in_en_value_A
    for i in range(num_layer_de):
        input_feed_B[in_name_B[i].name] = in_de_fsmn_B
    for i in range(in_de_key_start, in_de_value_start):
        input_feed_B[in_name_B[i].name] = in_de_key_B
    for i in range(in_de_value_start, in_de_value_end):
        input_feed_B[in_name_B[i].name] = in_de_value_B
    return input_feed_A, input_feed_B, slice_start, slice_end


# Start to run Paraformer-Streaming
input_feed_A, input_feed_B, slice_start, slice_end = Initialize()
while True:
    input_feed_A[in_name_A[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start:slice_end], device_type, DEVICE_ID)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    start_time = time.time()
    all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, input_feed_A)
    if slice_end <= aligned_len:
        for i in range(amount_of_outputs_A):
            input_feed_A[in_name_A[i].name] = all_outputs_A[i]
    if onnxruntime.OrtValue.numpy(all_outputs_A[-1]) != 0:
        input_feed_B[in_name_B[-1].name] = all_outputs_A[-1]
        input_feed_B[in_name_B[-2].name] = all_outputs_A[-2]
        input_feed_B[in_name_B[-3].name] = all_outputs_A[-3]
        all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
        end_time = time.time()
        max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-1])[0]
        text = tokenizer[max_logit_ids].tolist()
        text = ''.join(text).replace("</s>", "")
        print(f"ASR: {text} / Time Cost: {end_time - start_time:.3f} Seconds")
        if slice_end > aligned_len:
            input_feed_A, input_feed_B, slice_start, slice_end = Initialize()  # Ready for next input audio.
            break
        for i in range(amount_of_outputs_B):
            input_feed_B[in_name_B[i].name] = all_outputs_B[i]
    elif slice_end > aligned_len:
        input_feed_A, input_feed_B, slice_start, slice_end = Initialize()      # Ready for next input audio.
        break

