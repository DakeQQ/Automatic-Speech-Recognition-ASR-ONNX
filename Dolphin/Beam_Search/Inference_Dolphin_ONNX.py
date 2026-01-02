import time
import onnxruntime
import numpy as np
from pydub import AudioSegment

model_path = "/home/DakeQQ/Downloads/dolphin-small"                                          # The dolphin project download path. Currently, only support dolphin-small and dolphin-base.
onnx_model_A = "/home/DakeQQ/Downloads/Dolphin_Optimized/Dolphin_Encoder.onnx"               # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Dolphin_Optimized/Dolphin_Decoder.onnx"               # The exported onnx model path.
onnx_model_C = '/home/DakeQQ/Downloads/Dolphin_Optimized/Greedy_Search.onnx'                 # The exported onnx model path.
onnx_model_D = '/home/DakeQQ/Downloads/Dolphin_Optimized/First_Beam_Search.onnx'             # The exported onnx model path.
onnx_model_E = '/home/DakeQQ/Downloads/Dolphin_Optimized/Second_Beam_Search.onnx'            # The exported onnx model path.
onnx_model_F = '/home/DakeQQ/Downloads/Dolphin_Optimized/Reset_Penality.onnx'                # The exported onnx model path.
onnx_model_G = '/home/DakeQQ/Downloads/Dolphin_Optimized/Argmax.onnx'                        # The exported onnx model path.

save_vocab = "/home/DakeQQ/Downloads/Dolphin_ONNX/vocab_Dolphin.txt"                                        # The exported Dolphin vocab path.
TARGET_LANGUAGE = "Auto-Auto"                                                                               # See 'LANGUAGE_REGION' for detail.
test_audio = ["../example/zh.mp3", "../example/zh-Shanghai.wav", "../example/ja.mp3", "../example/ko.mp3"]  # The test audio list.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
USE_BEAM_SEARCH = True                  # Use beam search or greedy search.
TOP_K = 3                               # The top k candidate in decoding.
BEAM_SIZE = 3                           # Number of beams in searching.
REPEAT_PENALITY = 0.9                   # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                     # Penalizes the most recent output. "30" means the last 30 tokens.
MAX_THREADS = 4                         # Max CPU parallel threads.
DEVICE_ID = 0                           # The GPU id, default to 0.
MAX_SEQ_LEN = 72                        # It should keep the same with exported model.
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
STOP_TOKEN = [40000]                    # 40000 is the end token for Dolphin series model.


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS,
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
            'fuse_conv_bias': '0',
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '0',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10000,
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
  

LANGUAGE_REGION = {
    # ───────────────────────────── Auto Detection ─────────────────────────────
    "Auto"                         : "auto-auto",
    "Auto-Auto"                    : "auto-auto",
    "Chinese-Auto"                 : "zh-auto",
    "Mandarin-Auto"                : "zh-auto",
    "Yue-Auto"                     : "ct-NULL",
    "Tamil-Auto"                   : "ta-auto",
    "Urdu-Auto"                    : "ur-auto",
    "Arabic-Auto"                  : "ar-auto",

    "自动"                          : "auto",
    "自动-自动"                      : "auto-auto",
    "中文-自动"                      : "zh-auto",
    "普通话-自动"                     : "zh-auto",
    "粤语-自动"                      : "ct-NULL",
    "泰米尔语-自动"                   : "ta-auto",
    "乌尔都语-自动"                   : "ur-auto",
    "阿拉伯语-自动"                   : "ar-auto",

    # ───────────────────────────── Chinese variants ─────────────────────────────
    "Chinese"                       : "zh-CN",
    "Mandarin"                      : "zh-CN",
    "Chinese-Mandarin"              : "zh-CN",
    "Chinese-Taiwan"                : "zh-TW",
    "Chinese-Wuyu"                  : "zh-WU",
    "Chinese-Sichuan"               : "zh-SICHUAN",
    "Chinese-Shanxi"                : "zh-SHANXI",
    "Chinese-Anhui"                 : "zh-ANHUI",
    "Chinese-Tianjin"               : "zh-TIANJIN",
    "Chinese-Ningxia"               : "zh-NINGXIA",
    "Chinese-Shaanxi"               : "zh-SHAANXI",
    "Chinese-Hebei"                 : "zh-HEBEI",
    "Chinese-Shandong"              : "zh-SHANDONG",
    "Chinese-Guangdong"             : "zh-GUANGDONG",
    "Chinese-Shanghai"              : "zh-SHANGHAI",
    "Chinese-Hubei"                 : "zh-HUBEI",
    "Chinese-Liaoning"              : "zh-LIAONING",
    "Chinese-Gansu"                 : "zh-GANSU",
    "Chinese-Fujian"                : "zh-FUJIAN",
    "Chinese-Hunan"                 : "zh-HUNAN",
    "Chinese-Henan"                 : "zh-HENAN",
    "Chinese-Yunnan"                : "zh-YUNNAN",
    "Chinese-Minnan"                : "zh-MINNAN",
    "Chinese-Wenzhou"               : "zh-WENZHOU",

    "中文"                           : "zh-CN",
    "普通话"                         : "zh-CN",
    "中文-普通话"                    : "zh-CN",
    "中文-台湾"                      : "zh-TW",
    "中文-吴语"                      : "zh-WU",
    "中文-四川话"                    : "zh-SICHUAN",
    "中文-山西话"                    : "zh-SHANXI",
    "中文-安徽话"                    : "zh-ANHUI",
    "中文-天津话"                    : "zh-TIANJIN",
    "中文-宁夏话"                    : "zh-NINGXIA",
    "中文-陕西话"                    : "zh-SHAANXI",
    "中文-河北话"                    : "zh-HEBEI",
    "中文-山东话"                    : "zh-SHANDONG",
    "中文-广东话"                    : "zh-GUANGDONG",
    "中文-上海话"                    : "zh-SHANGHAI",
    "中文-湖北话"                    : "zh-HUBEI",
    "中文-辽宁话"                    : "zh-LIAONING",
    "中文-甘肃话"                    : "zh-GANSU",
    "中文-福建话"                    : "zh-FUJIAN",
    "中文-湖南话"                    : "zh-HUNAN",
    "中文-河南话"                    : "zh-HENAN",
    "中文-云南话"                    : "zh-YUNNAN",
    "中文-闽南语"                    : "zh-MINNAN",
    "中文-温州话"                    : "zh-WENZHOU",

    # ───────────────────────────── Yue-Cantonese variants ───────────────────────────
    "Yue-Unknown"                  : "ct-NULL",
    "Yue-Hongkong"                 : "ct-HK",
    "Yue-Guangdong"                : "ct-GZ",

    "粤语-未知"                     : "ct-NULL",
    "粤语-香港"                     : "ct-HK",
    "粤语-广东"                     : "ct-GZ",

    # ───────────────────────────── East-Asian languages ──────────────────────────────
    "Japanese"                      : "ja-JP",
    "Korean"                        : "ko-KR",

    "日文"                           : "ja-JP",
    "日语"                           : "ja-JP",
    "韩语"                           : "ko-KR",

    # ───────────────────────────── South-East Asian languages ─────────────────────────
    "Thai"                          : "th-TH",
    "Indonesian"                    : "id-ID",
    "Vietnamese"                    : "vi-VN",
    "Malay"                         : "ms-MY",
    "Burmese"                       : "my-MM",
    "Tagalog"                       : "tl-PH",
    "Khmer"                         : "km-KH",
    "Javanese"                      : "jv-ID",
    "Lao"                           : "lo-LA",
    "Filipino"                      : "fil-PH",
    "Sundanese"                     : "su-ID",

    "泰语"                            : "th-TH",
    "印度尼西亚语"                     : "id-ID",
    "越南语"                          : "vi-VN",
    "马来语"                          : "ms-MY",
    "缅甸语"                          : "my-MM",
    "塔加洛语"                        : "tl-PH",
    "高棉语"                          : "km-KH",
    "爪哇语"                          : "jv-ID",
    "老挝语"                          : "lo-LA",
    "菲律宾语"                        : "fil-PH",
    "巽他语"                          : "su-ID",

    # ───────────────────────────── South-Asian languages ──────────────────────────────
    "Hindi"                         : "hi-IN",
    "Bengali"                       : "bn-BD",
    "Tamil-Singaporean"             : "ta-SG",
    "Tamil-Sri Lankan"              : "ta-LK",
    "Tamil-India"                   : "ta-IN",
    "Tamil-Malaysia"                : "ta-MY",
    "Telugu"                        : "te-IN",
    "Gujarati"                      : "gu-IN",
    "Oriya"                         : "or-IN",
    "Odia"                          : "or-IN",
    "Nepali"                        : "ne-NP",
    "Sinhala"                       : "si-LK",
    "Panjabi"                       : "pa-IN",
    "Kashmiri"                      : "ks-IN",
    "Marathi"                       : "mr-IN",

    "印地语"                         : "hi-IN",
    "孟加拉语"                       : "bn-BD",
    "泰米尔语-新加坡"                 : "ta-SG",
    "泰米尔语-斯里兰卡"                : "ta-LK",
    "泰米尔语-印度"                   : "ta-IN",
    "泰米尔语-马来西亚"                : "ta-MY",
    "泰卢固语"                        : "te-IN",
    "古吉拉特语"                      : "gu-IN",
    "奥里亚语"                        : "or-IN",
    "尼泊尔语"                        : "ne-NP",
    "僧伽罗语"                        : "si-LK",
    "旁遮普语"                        : "pa-IN",
    "克什米尔语"                      : "ks-IN",
    "马拉地语"                        : "mr-IN",

    # ───────────────────────────── Middle-Eastern languages ───────────────────────────
    "Urdu"                          : "ur-PK",
    "Urdu-Islamic Republic of Pakistan": "ur-PK",
    "Urdu-India"                    : "ur-IN",
    "Persian"                       : "fa-IR",
    "Pushto"                        : "ps-AF",

    "乌尔都语"                        : "ur-PK",
    "乌尔都语-印度"                    : "ur-IN",
    "波斯语"                          : "fa-IR",
    "普什图语"                        : "ps-AF",

    # ───────────────────────────── Arabic variants ──────────────────────────────
    "Arabic"                        : "ar-GLA",
    "Arabic-Morocco"                : "ar-MA",
    "Arabic-Saudi Arabia"           : "ar-SA",
    "Arabic-Egypt"                  : "ar-EG",
    "Arabic-Kuwait"                 : "ar-KW",
    "Arabic-Libya"                  : "ar-LY",
    "Arabic-Jordan"                 : "ar-JO",
    "Arabic-U.A.E."                 : "ar-AE",
    "Arabic-Levant"                 : "ar-LVT",

    "阿拉伯语"                        : "ar-GLA",
    "阿拉伯语-摩洛哥"                  : "ar-MA",
    "阿拉伯语-沙特"                    : "ar-SA",
    "阿拉伯语-埃及"                    : "ar-EG",
    "阿拉伯语-科威特"                  : "ar-KW",
    "阿拉伯语-利比亚"                  : "ar-LY",
    "阿拉伯语-约旦"                    : "ar-JO",
    "阿拉伯语-阿联酋"                  : "ar-AE",
    "阿拉伯语-黎凡特"                  : "ar-LVT",

    # ───────────────────────────── Central-Asian languages ────────────────────────────
    "Uighur"                        : "ug-CN",
    "Uzbek"                         : "uz-UZ",
    "Kazakh"                        : "kk-KZ",
    "Mongolian"                     : "mn-MN",
    "Kabyle"                        : "kab-NULL",
    "Bashkir"                       : "ba-NULL",
    "Tajik"                         : "tg-TJ",
    "Kirghiz"                       : "ky-KG",
    "Azerbaijani"                   : "az-AZ",

    "维吾尔语"                        : "ug-CN",
    "乌兹别克语"                      : "uz-UZ",
    "哈萨克语"                        : "kk-KZ",
    "蒙古语"                          : "mn-MN",
    "卡拜尔语"                        : "kab-NULL",
    "巴什基尔语"                      : "ba-NULL",
    "塔吉克语"                        : "tg-TJ",
    "吉尔吉斯语"                      : "ky-KG",
    "阿塞拜疆语"                      : "az-AZ",

    # ───────────────────────────── Eastern-European languages ─────────────────────────
    "Russian"                       : "ru-RU",
    "俄语"                           : "ru-RU",
}


class Tokenizer:
    def __init__(self, filename):
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.num_vocab = 0
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def num_vocab(self):
        return self.num_vocab


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                 # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
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
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A = [out_name_A[i].name for i in range(len(out_name_A))]

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

ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
in_name_G = ort_session_G.get_inputs()[0].name
out_name_G = [ort_session_G.get_outputs()[0].name]

generate_limit = MAX_SEQ_LEN - 6          # 6 = length of initial input_ids
num_layers = (amount_of_outputs_B - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=np.float32), device_type, DEVICE_ID)
tokenizer = Tokenizer(save_vocab)
vocab_size = tokenizer.num_vocab

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

language_region = LANGUAGE_REGION.get(TARGET_LANGUAGE, "NONE")
if language_region == "NONE":
    TARGET_LANGUAGE = "Auto-Auto"
    print(f"\nThe language or region:{TARGET_LANGUAGE} not found. \nFallback to auto detection.")
language_region = language_region.split("-")
lang_id = f"<{language_region[0]}>"
region_id = f"<{language_region[1]}>"

if lang_id != "<auto>":
    detect_language = False
    lang_id = tokenizer.encode(lang_id)
else:
    detect_language = True

if not detect_language:
    if region_id != "<auto>":
        detect_region = False
        region_id = tokenizer.encode(region_id)
    else:
        detect_region = True
else:
    detect_region = True

history_len_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)

ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
ids_len_2 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([2], dtype=np.int64), device_type, DEVICE_ID)
ids_len_5 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([5], dtype=np.int64), device_type, DEVICE_ID)

ids_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
ids_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
ids_7 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([7], dtype=np.int64), device_type, DEVICE_ID)
ids_145 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), device_type, DEVICE_ID)
ids_324 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([324], dtype=np.int64), device_type, DEVICE_ID)
ids_39999 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999]], dtype=np.int32), device_type, DEVICE_ID)  # int32
ids_vocab_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([vocab_size], dtype=np.int64), device_type, DEVICE_ID)

init_BATCH = 1
if device_type != 'dml':
    init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((init_BATCH, ort_session_B._outputs_meta[0].shape[1], ort_session_B._outputs_meta[0].shape[2], 0), dtype=np.float32), device_type, DEVICE_ID)
    init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((init_BATCH, ort_session_B._outputs_meta[num_layers].shape[1], 0, ort_session_B._outputs_meta[num_layers].shape[3]), dtype=np.float32), device_type, DEVICE_ID)
else:
    init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((init_BATCH, ort_session_B._outputs_meta[0].shape[1], ort_session_B._outputs_meta[0].shape[2], 0), dtype=np.float32), 'cpu', 0)
    init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((init_BATCH, ort_session_B._outputs_meta[num_layers].shape[1], 0, ort_session_B._outputs_meta[num_layers].shape[3]), dtype=np.float32), 'cpu', 0)
layer_indices = np.arange(num_keys_values_plus_2, num_keys_values_plus_2 + num_keys_values, dtype=np.int32)

# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalizer(audio)
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

    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH

    # Start to run dolphin
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], device_type, DEVICE_ID)})
        input_feed_B = {
            in_name_B[-1]: attention_mask_1,
            in_name_B[num_keys_values_plus_1]: history_len_0,
        }
        for i in range(num_layers):
            input_feed_B[in_name_B[i]] = init_past_keys_B
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]
        for i in range(num_layers, num_keys_values):
            input_feed_B[in_name_B[i]] = init_past_values_B
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]

        if detect_language:
            print("\nAutomatically detect which language it is.")
            input_feed_B[in_name_B[-4]] = ids_7
            input_feed_B[in_name_B[-3]] = ids_145
            input_feed_B[in_name_B[-2]] = ids_len_2
            input_feed_B[in_name_B[num_keys_values]] = ids_39999
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            lang_id = onnxruntime.OrtValue.numpy(ort_session_G.run_with_ort_values(out_name_G, {in_name_G: all_outputs_B[-2]})[0])[0] + 7
            for i in range(num_layers):
                input_feed_B[in_name_B[i]] = init_past_keys_B
            for i in range(num_layers, num_keys_values):
                input_feed_B[in_name_B[i]] = init_past_values_B
            
        if detect_region:
            print("\nAutomatically detect which region it is.")
            input_feed_B[in_name_B[-4]] = ids_145
            input_feed_B[in_name_B[-3]] = ids_324
            input_feed_B[in_name_B[-2]] = ids_len_2
            input_feed_B[in_name_B[num_keys_values]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, lang_id]], dtype=np.int32), device_type, DEVICE_ID)
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            region_id = onnxruntime.OrtValue.numpy(ort_session_G.run_with_ort_values(out_name_G, {in_name_G: all_outputs_B[-2]})[0])[0] + 145
            for i in range(num_layers):
                input_feed_B[in_name_B[i]] = init_past_keys_B
            for i in range(num_layers, num_keys_values):
                input_feed_B[in_name_B[i]] = init_past_values_B

        if detect_language or detect_region:
            lang_str = tokenizer.decode(lang_id)
            region_str = tokenizer.decode(region_id)
            message = f"\nThis audio belongs to {lang_str}-{region_str}."
            message = message.replace("<", "").replace(">", "")
            print(message)
        else:
            print(f"\nThis audio belongs to {TARGET_LANGUAGE}.")
            
        input_feed_B[in_name_B[-4]] = ids_0
        input_feed_B[in_name_B[-3]] = ids_vocab_size
        input_feed_B[in_name_B[-2]] = ids_len_5
        input_feed_B[in_name_B[num_keys_values]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, lang_id, region_id, 6, 324]], dtype=np.int32), device_type, DEVICE_ID)  # start_id = 39999; itn = 5; asr = 6; no_timestamp = 324
        repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=np.float32), device_type, DEVICE_ID)

        if USE_BEAM_SEARCH:
            save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
            input_feed_D[in_name_D[num_keys_values_plus_1]] = save_id_beam
            input_feed_D[in_name_D[num_keys_values_plus_2]] = repeat_penality
        else:
            batch_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([init_BATCH], dtype=np.int64), device_type, DEVICE_ID)
            input_feed_C[in_name_C[1]] = repeat_penality
            input_feed_C[in_name_C[3]] = batch_size
    
        if do_repeat_penality:
            if USE_BEAM_SEARCH:
                input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
            else:
                penality_reset_count_greedy = 0
        
        num_decode = 0
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    for i in range(num_keys_values_plus_1):
                        input_feed_D[in_name_D[i]] = all_outputs_B[i]
                    all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_D[-1])
                    input_feed_E[in_name_E[-4]] = all_outputs_D[-2]
                    if do_repeat_penality:
                        input_feed_F[in_name_F[3]] = all_outputs_D[-2]
                else:
                    for i in range(num_keys_values_plus_1):
                        input_feed_E[in_name_E[i]] = all_outputs_B[i]
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
                    for i in range(num_keys_values_plus_1):
                        input_feed_B[in_name_B[i]] = all_outputs_D[i]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_D[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_D[num_keys_values_plus_3]
                else:
                    for i in range(num_keys_values_plus_1):
                        input_feed_B[in_name_B[i]] = all_outputs_E[i]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
            else:
                input_feed_C[in_name_C[0]] = all_outputs_B[-2]
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
                for i in range(num_keys_values):
                    input_feed_B[in_name_B[i]] = all_outputs_B[i]
            input_feed_B[in_name_B[num_keys_values_plus_1]] = all_outputs_B[num_keys_values_plus_1]
            if num_decode < 1:
                input_feed_B[in_name_B[-1]] = attention_mask_0
                input_feed_B[in_name_B[-2]] = ids_len_1
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.numpy(all_outputs_E[num_keys_values_plus_1])[0]
        for i, idx in enumerate(save_id_beam):
            if idx in STOP_TOKEN:
                save_token_array = save_id_beam[:i]
                break
    else:
        save_token_array = save_id_greedy[:num_decode]
    text = ""
    for i in save_token_array:
        text += tokenizer.decode(i)
    text = text.replace("▁", " ")
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")

