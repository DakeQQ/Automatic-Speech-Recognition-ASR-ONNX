import time
import numpy as np
import onnxruntime
from pydub import AudioSegment
from transformers import AutoTokenizer


tokenizer_path = "/home/DakeQQ/Downloads/whisper-large-v3-turbo"                       # The whisper model tokenizer folder path.
onnx_model_A = "/home/DakeQQ/Downloads/Whisper_Optimized/Whisper_Encoder.onnx"         # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Whisper_Optimized/Whisper_Decoder.onnx"         # The exported onnx model path.
onnx_model_C = '/home/DakeQQ/Downloads/Whisper_Optimized/Greedy_Search.onnx'           # The exported onnx model path.
onnx_model_D = '/home/DakeQQ/Downloads/Whisper_Optimized/First_Beam_Search.onnx'       # The exported onnx model path.
onnx_model_E = '/home/DakeQQ/Downloads/Whisper_Optimized/Second_Beam_Search.onnx'      # The exported onnx model path.
onnx_model_F = '/home/DakeQQ/Downloads/Whisper_Optimized/Reset_Penality.onnx'          # The exported onnx model path.

test_audio = ["../example/zh.mp3", "../example/en.mp3", "../example/ja.mp3", "../example/ko.mp3"]     # The test audio list.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
USE_BEAM_SEARCH = True                  # Use beam search or greedy search.
TOP_K = 3                               # The top k candidate in decoding.
BEAM_SIZE = 3                           # Number of beams in searching.
REPEAT_PENALITY = 0.9                   # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                     # Penalizes the most recent output. "30" means the last 30 tokens.
MAX_THREADS = 4                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.
MAX_SEQ_LEN = 80                        # It should keep the same with exported model.
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.
TARGET_LANGUAGE = "en"                  # Choose a language listed in the get_language_id function's language_map.
TASK = 'transcribe'                     # Choose one of : ['transcribe', 'translate']
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.


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


tokenizer_path_lower = tokenizer_path.lower()
if ("v3" in tokenizer_path_lower) or ("crisperwhisper" in tokenizer_path_lower) or ("anime" in tokenizer_path_lower) or ("belle" in tokenizer_path_lower) or ("turbo" in tokenizer_path_lower) or ("distil" in tokenizer_path_lower):
    is_v3 = True
    if 'v0.3' in tokenizer_path_lower:
        custom_vocab = True
    else:
        custom_vocab = False
else:
    is_v3 = False
    custom_vocab = False


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * scaling_factor).astype(np.int16)
  

_LANGUAGE_DATA = {
    'af': {'id': 50327, 'custom_id': 18941, 'full_name': 'afrikaans'},
    'am': {'id': 50334, 'custom_id': 18948, 'full_name': 'amharic'},
    'ar': {'id': 50272, 'custom_id': 18886, 'full_name': 'arabic'},
    'as': {'id': 50350, 'custom_id': 18964, 'full_name': 'assamese'},
    'az': {'id': 50304, 'custom_id': 18918, 'full_name': 'azerbaijani'},
    'ba': {'id': 50355, 'custom_id': 18969, 'full_name': 'bashkir'},
    'be': {'id': 50330, 'custom_id': 18944, 'full_name': 'belarusian'},
    'bg': {'id': 50292, 'custom_id': 18906, 'full_name': 'bulgarian'},
    'bn': {'id': 50302, 'custom_id': 18916, 'full_name': 'bengali'},
    'bo': {'id': 50347, 'custom_id': 18961, 'full_name': 'tibetan'},
    'br': {'id': 50309, 'custom_id': 18923, 'full_name': 'breton'},
    'bs': {'id': 50315, 'custom_id': 18929, 'full_name': 'bosnian'},
    'ca': {'id': 50270, 'custom_id': 18884, 'full_name': 'catalan'},
    'cs': {'id': 50283, 'custom_id': 18897, 'full_name': 'czech'},
    'cy': {'id': 50297, 'custom_id': 18911, 'full_name': 'welsh'},
    'da': {'id': 50285, 'custom_id': 18899, 'full_name': 'danish'},
    'de': {'id': 50261, 'custom_id': 18875, 'full_name': 'german'},
    'el': {'id': 50281, 'custom_id': 18895, 'full_name': 'greek'},
    'en': {'id': 50259, 'custom_id': 18873, 'full_name': 'english'},
    'es': {'id': 50262, 'custom_id': 18876, 'full_name': 'spanish'},
    'et': {'id': 50307, 'custom_id': 18921, 'full_name': 'estonian'},
    'eu': {'id': 50310, 'custom_id': 18924, 'full_name': 'basque'},
    'fa': {'id': 50300, 'custom_id': 18914, 'full_name': 'persian'},
    'fi': {'id': 50277, 'custom_id': 18891, 'full_name': 'finnish'},
    'fo': {'id': 50338, 'custom_id': 18952, 'full_name': 'faroese'},
    'fr': {'id': 50265, 'custom_id': 18879, 'full_name': 'french'},
    'gl': {'id': 50319, 'custom_id': 18933, 'full_name': 'galician'},
    'gu': {'id': 50333, 'custom_id': 18947, 'full_name': 'gujarati'},
    'ha': {'id': 50354, 'custom_id': 18968, 'full_name': 'hausa'},
    'haw': {'id': 50352, 'custom_id': 18966, 'full_name': 'hawaiian'},
    'he': {'id': 50279, 'custom_id': 18893, 'full_name': 'hebrew'},
    'hi': {'id': 50276, 'custom_id': 18890, 'full_name': 'hindi'},
    'hr': {'id': 50291, 'custom_id': 18905, 'full_name': 'croatian'},
    'ht': {'id': 50339, 'custom_id': 18953, 'full_name': 'haitian creole'},
    'hu': {'id': 50286, 'custom_id': 18900, 'full_name': 'hungarian'},
    'hy': {'id': 50312, 'custom_id': 18926, 'full_name': 'armenian'},
    'id': {'id': 50275, 'custom_id': 18889, 'full_name': 'indonesian'},
    'is': {'id': 50311, 'custom_id': 18925, 'full_name': 'icelandic'},
    'it': {'id': 50274, 'custom_id': 18888, 'full_name': 'italian'},
    'ja': {'id': 50266, 'custom_id': 18880, 'full_name': 'japanese'},
    'jw': {'id': 50356, 'custom_id': 18970, 'full_name': 'javanese'},
    'ka': {'id': 50329, 'custom_id': 18943, 'full_name': 'georgian'},
    'kk': {'id': 50316, 'custom_id': 18930, 'full_name': 'kazakh'},
    'km': {'id': 50323, 'custom_id': 18937, 'full_name': 'khmer'},
    'kn': {'id': 50306, 'custom_id': 18920, 'full_name': 'kannada'},
    'ko': {'id': 50264, 'custom_id': 18878, 'full_name': 'korean'},
    'la': {'id': 50294, 'custom_id': 18908, 'full_name': 'latin'},
    'lb': {'id': 50345, 'custom_id': 18959, 'full_name': 'luxembourgish'},
    'ln': {'id': 50353, 'custom_id': 18967, 'full_name': 'lingala'},
    'lo': {'id': 50336, 'custom_id': 18950, 'full_name': 'lao'},
    'lt': {'id': 50293, 'custom_id': 18907, 'full_name': 'lithuanian'},
    'lv': {'id': 50301, 'custom_id': 18915, 'full_name': 'latvian'},
    'mg': {'id': 50349, 'custom_id': 18963, 'full_name': 'malagasy'},
    'mi': {'id': 50295, 'custom_id': 18909, 'full_name': 'maori'},
    'mk': {'id': 50308, 'custom_id': 18922, 'full_name': 'macedonian'},
    'ml': {'id': 50296, 'custom_id': 18910, 'full_name': 'malayalam'},
    'mn': {'id': 50314, 'custom_id': 18928, 'full_name': 'mongolian'},
    'mr': {'id': 50320, 'custom_id': 18934, 'full_name': 'marathi'},
    'ms': {'id': 50282, 'custom_id': 18896, 'full_name': 'malay'},
    'mt': {'id': 50343, 'custom_id': 18957, 'full_name': 'maltese'},
    'my': {'id': 50346, 'custom_id': 18960, 'full_name': 'burmese'},
    'ne': {'id': 50313, 'custom_id': 18927, 'full_name': 'nepali'},
    'nl': {'id': 50271, 'custom_id': 18885, 'full_name': 'dutch'},
    'nn': {'id': 50342, 'custom_id': 18956, 'full_name': 'nynorsk'},
    'no': {'id': 50288, 'custom_id': 18902, 'full_name': 'norwegian'},
    'oc': {'id': 50328, 'custom_id': 18942, 'full_name': 'occitan'},
    'pa': {'id': 50321, 'custom_id': 18935, 'full_name': 'punjabi'},
    'pl': {'id': 50269, 'custom_id': 18883, 'full_name': 'polish'},
    'ps': {'id': 50340, 'custom_id': 18954, 'full_name': 'pashto'},
    'pt': {'id': 50267, 'custom_id': 18881, 'full_name': 'portuguese'},
    'ro': {'id': 50284, 'custom_id': 18898, 'full_name': 'romanian'},
    'ru': {'id': 50263, 'custom_id': 18877, 'full_name': 'russian'},
    'sa': {'id': 50344, 'custom_id': 18958, 'full_name': 'sanskrit'},
    'sd': {'id': 50332, 'custom_id': 18946, 'full_name': 'sindhi'},
    'si': {'id': 50322, 'custom_id': 18936, 'full_name': 'sinhala'},
    'sk': {'id': 50298, 'custom_id': 18912, 'full_name': 'slovak'},
    'sl': {'id': 50305, 'custom_id': 18919, 'full_name': 'slovenian'},
    'sn': {'id': 50324, 'custom_id': 18938, 'full_name': 'shona'},
    'so': {'id': 50326, 'custom_id': 18940, 'full_name': 'somali'},
    'sq': {'id': 50317, 'custom_id': 18931, 'full_name': 'albanian'},
    'sr': {'id': 50303, 'custom_id': 18917, 'full_name': 'serbian'},
    'su': {'id': 50357, 'custom_id': 18971, 'full_name': 'sundanese'},
    'sv': {'id': 50273, 'custom_id': 18887, 'full_name': 'swedish'},
    'sw': {'id': 50318, 'custom_id': 18932, 'full_name': 'swahili'},
    'ta': {'id': 50287, 'custom_id': 18901, 'full_name': 'tamil'},
    'te': {'id': 50299, 'custom_id': 18913, 'full_name': 'telugu'},
    'tg': {'id': 50331, 'custom_id': 18945, 'full_name': 'tajik'},
    'th': {'id': 50289, 'custom_id': 18903, 'full_name': 'thai'},
    'tk': {'id': 50341, 'custom_id': 18955, 'full_name': 'turkmen'},
    'tl': {'id': 50348, 'custom_id': 18962, 'full_name': 'tagalog'},
    'tr': {'id': 50268, 'custom_id': 18882, 'full_name': 'turkish'},
    'tt': {'id': 50351, 'custom_id': 18965, 'full_name': 'tatar'},
    'uk': {'id': 50280, 'custom_id': 18894, 'full_name': 'ukrainian'},
    'ur': {'id': 50290, 'custom_id': 18904, 'full_name': 'urdu'},
    'uz': {'id': 50337, 'custom_id': 18951, 'full_name': 'uzbek'},
    'vi': {'id': 50278, 'custom_id': 18892, 'full_name': 'vietnamese'},
    'yi': {'id': 50335, 'custom_id': 18949, 'full_name': 'yiddish'},
    'yo': {'id': 50325, 'custom_id': 18939, 'full_name': 'yoruba'},
    'yue': {'id': 50358, 'custom_id': 18972, 'full_name': 'cantonese'},
    'zh': {'id': 50260, 'custom_id': 18874, 'full_name': 'chinese'},
}


_FULL_NAME_TO_CODE = {data['full_name']: code for code, data in _LANGUAGE_DATA.items()}


_ALIAS_TO_CODE = {
    'united states': 'en', 'us': 'en',
    'united kingdom': 'en', 'uk': 'en', 'gb': 'en',
    'france': 'fr',
    'germany': 'de',
    'spain': 'es',
    'china': 'zh',
    'japan': 'ja',
    'korea': 'ko',
}


def get_language_id(language_input, custom_vocab=False):
    normalized_input = language_input.lower().strip()
    lang_code = None
    if normalized_input in _LANGUAGE_DATA:
        lang_code = normalized_input
    elif normalized_input in _FULL_NAME_TO_CODE:
        lang_code = _FULL_NAME_TO_CODE[normalized_input]
    elif normalized_input in _ALIAS_TO_CODE:
        lang_code = _ALIAS_TO_CODE[normalized_input]
    if lang_code:
        language_info = _LANGUAGE_DATA[lang_code]
        id_key = 'custom_id' if custom_vocab else 'id'
        return language_info.get(id_key)
    return None


def get_task_id(task_input, is_v3, custom_vocab=False):
    task_input = task_input.lower()
    if custom_vocab:
        stop_token = 18871
        start_token = 18872
        task_map = {
            'translate': 18973,
            'transcribe': 18974
        }
        return start_token, [stop_token], task_map[task_input]
    stop_token = 50257
    start_token = 50258
    if is_v3:
        task_map = {
            'translate': 50359,
            'transcribe':  50360
        }
        return start_token, [stop_token], task_map[task_input]
    else:
        task_map = {
            'translate': 50358,
            'transcribe': 50359
        }
        return start_token, [stop_token], task_map[task_input]


def remove_repeated_parts(ids, repeat_words_threshold, ids_len):
    if ids_len <= repeat_words_threshold:
        return ids
    side_L = repeat_words_threshold // 2
    side_R = side_L + 1
    boundary = ids_len - side_L
    for i in range(side_L, boundary):
        for j in range(i + repeat_words_threshold, boundary):
            check = []
            for k in range(-side_L, side_R):
                if ids[j + k] == ids[i + k]:
                    check.append(True)
                else:
                    check.append(False)
                    break
            if False not in check:
                return ids[: j - side_L]
    return ids


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
out_name_A = [out_name_A[i].name for i in range(len(out_name_A))]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
providers = ort_session_B.get_providers()
print(f"\nUsable Providers: {providers}")
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

generate_limit = MAX_SEQ_LEN - 5  # 5 = length of inital input_ids
num_layers = (amount_of_outputs_B - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values2_plus_2 = num_keys_values_plus_2 + num_keys_values
vocab_size = ort_session_B._outputs_meta[num_keys_values].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=model_dtype), device_type, DEVICE_ID)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]
    
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]

    input_feed_D = {
        in_name_D[-2]: penality_value,
        in_name_D[-1]: beam_size
    }

    input_feed_E = {
        in_name_E[-3]: penality_value,
        in_name_E[-2]: beam_size,
        in_name_E[-1]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
    out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
    input_feed_C = {in_name_C[2]: penality_value}

if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False

# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    if ("example/en" in test) or ("example/zh" in test) or ("example/ja" in test) or ("example/ko" in test):
        language = test.split("/")[-1].split(".")[0]
    else:
        language = TARGET_LANGUAGE
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(240000, audio_len)  # You can adjust it.
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
    start, STOP_TOKEN, task = get_task_id(TASK, is_v3, custom_vocab)
    input_ids = np.array([[start, get_language_id(language, custom_vocab), task]], dtype=np.int32)
    batch_size = input_ids.shape[0]
    repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
    ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
    attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
    attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)

  if device_type != 'dml':
        past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, ort_session_B._outputs_meta[0].shape[1],  0), dtype=model_dtype), device_type, DEVICE_ID)
        past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, 0, ort_session_B._outputs_meta[num_layers].shape[2]), dtype=model_dtype), device_type, DEVICE_ID)
    else:
        past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, ort_session_B._outputs_meta[0].shape[1], 0), dtype=model_dtype), 'cpu', DEVICE_ID)
        past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, 0, ort_session_B._outputs_meta[num_layers].shape[2]), dtype=model_dtype), 'cpu', DEVICE_ID)

    input_feed_B = {
        in_name_B[num_keys_values]: input_ids,
        in_name_B[num_keys_values_plus_1]: history_len,
        in_name_B[-2]: ids_len,
        in_name_B[-1]: attention_mask_1
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B

    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
        input_feed_D[in_name_D[num_keys_values_plus_1]] = save_id_beam
        input_feed_D[in_name_D[num_keys_values_plus_2]] = repeat_penality
    else:
        batch_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([batch_size], dtype=np.int64), device_type, DEVICE_ID)
        input_feed_C[in_name_C[1]] = repeat_penality
        input_feed_C[in_name_C[3]] = batch_size

    if do_repeat_penality:
        if USE_BEAM_SEARCH:
            input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
        else:
            penality_reset_count_greedy = 0

    num_decode = 0
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], device_type, DEVICE_ID)})
        input_feed_B.update(zip(in_name_B[num_keys_values_plus_2: num_keys_values2_plus_2], all_outputs_A))
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    input_feed_D.update(zip(in_name_D[:num_keys_values_plus_1], all_outputs_B))
                    all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_D[-1])
                    input_feed_E[in_name_E[-4]] = all_outputs_D[-2]
                    if do_repeat_penality:
                        input_feed_F[in_name_F[3]] = all_outputs_D[-2]
                else:
                    input_feed_E.update(zip(in_name_E[:num_keys_values_plus_1], all_outputs_B))
                    all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_E[-1])
                if max_logits_idx in STOP_TOKEN:
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_F[in_name_F[1]] = all_outputs_E[num_keys_values_plus_2]
                    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                    input_feed_F[in_name_F[2]] = all_outputs_F[2]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_F[0]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_F[1]
                if num_decode < 1:
                    input_feed_B.update(zip(in_name_B[:num_keys_values_plus_1], all_outputs_D))
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_D[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_D[num_keys_values_plus_3]
                else:
                    input_feed_B.update(zip(in_name_B[:num_keys_values_plus_1], all_outputs_E))
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
            else:
                input_feed_C[in_name_C[0]] = all_outputs_B[num_keys_values]
                all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
                max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_C[0])[0, 0]
                if max_logits_idx in STOP_TOKEN:
                    break
                input_feed_B[in_name_B[num_keys_values]] = all_outputs_C[0]
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    reset_ids = save_id_greedy[penality_reset_count_greedy]
                    if reset_ids != max_logits_idx:
                        repeat_penality = onnxruntime.OrtValue.numpy(all_outputs_C[1])
                        repeat_penality[..., reset_ids] = 1.0
                        all_outputs_C[1] = onnxruntime.OrtValue.ortvalue_from_numpy(repeat_penality, device_type, DEVICE_ID)
                    penality_reset_count_greedy += 1
                input_feed_C[in_name_C[1]] = all_outputs_C[1]
                save_id_greedy[num_decode] = max_logits_idx
                input_feed_B.update(zip(in_name_B[:num_keys_values], all_outputs_B))
            input_feed_B[in_name_B[num_keys_values_plus_1]] = all_outputs_B[num_keys_values_plus_1]
            if num_decode < 1:
                input_feed_B[in_name_B[-1]] = attention_mask_0
                input_feed_B[in_name_B[-2]] = ids_len_1
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    if num_decode > 0:
        if USE_BEAM_SEARCH:
            save_id_beam = onnxruntime.OrtValue.numpy(all_outputs_E[num_keys_values_plus_1])[0]
            for i, idx in enumerate(save_id_beam):
                if idx in STOP_TOKEN:
                    save_id_beam = save_id_beam[:i]
                    break
            save_token_array = remove_repeated_parts(save_id_beam, 3, save_id_beam.shape[-1])          # To handle "over-talking".
        else:
            save_token_array = remove_repeated_parts(save_id_greedy[:num_decode], 3, num_decode)    # To handle "over-talking".
        text, _ = tokenizer._decode_asr(
            [{
                "tokens": save_token_array.reshape(1, -1)
            }],
            return_timestamps=None,  # Do not support return timestamps
            return_language=None,
            time_precision=0
        )
        print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
        print("----------------------------------------------------------------------------------------------------------")
