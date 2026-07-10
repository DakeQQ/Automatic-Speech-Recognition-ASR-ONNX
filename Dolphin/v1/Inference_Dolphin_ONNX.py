import argparse
import os
import sys
import time
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import numpy as np
from pydub import AudioSegment
try:
    import sentencepiece as spm
except ImportError:
    spm = None
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from Example_Audio import model_audio_paths


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Dolphin ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", default=os.path.join(_SCRIPT_DIR, "Dolphin_Optimized"), help="Folder containing ONNX graphs, for example Dolphin_Optimized or Dolphin_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()


onnx_folder            = os.path.abspath(_ARGS.onnx_folder)                          # Selected ONNX graph folder.
onnx_model_Metadata    = os.path.join(onnx_folder, "ASR_Matadata.onnx")                      # Tiny metadata carrier graph.
onnx_model_Encoder     = os.path.join(onnx_folder, "Dolphin_Encoder.onnx")                       # The exported onnx encoder model path.
onnx_model_Decoder     = os.path.join(onnx_folder, "Dolphin_Decoder.onnx")                       # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = os.path.join(onnx_folder, "Dolphin_Decoder_Embed.onnx")                 # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = os.path.join(onnx_folder, "Dolphin_Position_Mask_Prefill.onnx")         # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = os.path.join(onnx_folder, "Dolphin_Position_Mask_Decode.onnx")          # Decode position-embedding graph for the single new token.
onnx_model_Greedy      = os.path.join(onnx_folder, "Dolphin_Greedy_Search.onnx")                 # Greedy argmax + save_id history (used with Apply_Penality).
onnx_model_Argmax      = os.path.join(onnx_folder, "Dolphin_Argmax.onnx")                        # Bare argmax (greedy decoding without a repetition penalty).
onnx_model_First_Beam  = os.path.join(onnx_folder, "Dolphin_First_Beam_Search.onnx")             # First beam-search step.
onnx_model_Second_Beam = os.path.join(onnx_folder, "Dolphin_Second_Beam_Search.onnx")            # Subsequent beam-search steps.
onnx_model_Penality    = os.path.join(onnx_folder, "Dolphin_Apply_Penality.onnx")                # Sliding-window repetition penalty on the logits.
save_vocab             = os.path.join(onnx_folder, "vocab_Dolphin.txt")                          # The exported Dolphin vocab path.
TARGET_LANGUAGE        = "Auto-Auto"                                                             # See 'LANGUAGE_REGION' for detail.

test_audio = model_audio_paths("dolphin")                                                         # The test audio list.


USE_BEAM_SEARCH    = False              # Use beam search or greedy search.
REPEAT_PENALITY    = 1.0                # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE     = 20                 # Penalizes the most recent output. "30" means the last 30 tokens.
TOP_K              = 3                  # The top k candidate in decoding.
BEAM_SIZE          = 3                  # Number of beams in searching.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# ("int16" -> raw PCM ÷32768 in-graph; "float16"/"float" -> pre-normalised [-1, 1]); no manual setting needed.
USE_NORMALISE_AUDIO = False             # Apply RMS loudness normalisation before feeding the model. The reference Dolphin pipeline keeps the decoded waveform amplitude unchanged.
SLIDING_WINDOW     = 0                  # Set the sliding window step for test audio reading; use 0 to disable.


# ============================================================================
# ONNX Runtime runtime configuration (IOBinding + shared buffers, mirrors Qwen ASR)
# ============================================================================
ORT_Accelerate_Providers = []           # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                = False          # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16               = False          # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS            = 0              # Parallel CPU threads. Set 0 for auto.
DEVICE_ID              = 0              # Default to zero.


_INV_INT16_SCALE = np.float32(1.0 / 32768.0)   # pre-computed [-1, 1] normalisation scale, reused every call


def prepare_audio_input(audio_int16: np.ndarray, input_audio_dtype: str, target_rms: float = 8192.0) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's
    # audio input dtype exactly once (no float32<->int16 round-trip for the float paths).
    # `input_audio_dtype` is derived from the encoder's audio input tensor in the ONNX model.
    #   "INT16": raw PCM (the graph divides by 32768 internally).
    #   "F32"/"F16": normalised to [-1, 1] here (÷32768), because the float graph skips the
    #   in-model division; "F16" stores those values (the graph up-casts back to f32).
    if not USE_NORMALISE_AUDIO and input_audio_dtype == "INT16":
        return np.ascontiguousarray(audio_int16, dtype=np.int16)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= (target_rms / (rms + 1e-7))
            np.clip(audio, -32768.0, 32767.0, out=audio)
    if input_audio_dtype == "INT16":
        return audio.astype(np.int16)
    audio *= _INV_INT16_SCALE   # fold the pre-computed ÷32768 scale into the same float buffer
    if input_audio_dtype == "F16":
        return audio.astype(np.float16)
    return audio


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

    "自动"                          : "auto-auto",
    "自动-自动"                      : "auto-auto",
    "中文-自动"                      : "zh-auto",
    "普通话-自动"                    : "zh-auto",
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
    def __init__(self, filename, bpe_model=None):
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.num_vocab = 0
        self.sp = None
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)
        if spm is not None and bpe_model is not None and os.path.exists(bpe_model):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def decode_ids(self, ids):
        tokens = [self.decode(int(idx)) for idx in ids]
        tokens = [token for token in tokens if token is not None]
        if self.sp is not None:
            return self.sp.DecodePieces(tokens)
        return ''.join(tokens).replace("▁", " ")

    def num_vocab(self):
        return self.num_vocab


def _build_run_options(silent):
    ro = onnxruntime.RunOptions()
    ro.log_severity_level = 0 if not silent else 4
    ro.log_verbosity_level = 4
    ro.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return ro


def _build_session_opts_ort():
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


def _make_session(path):
    return onnxruntime.InferenceSession(path, **_packed)


def _ort_from_numpy(arr):
    return onnxruntime.OrtValue.ortvalue_from_numpy(arr, device_type, DEVICE_ID)


def _ort_zeros(shape, dtype):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device_type, DEVICE_ID)


def _ort_from_data(data, dtype):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device_type, DEVICE_ID)


def _bind_inputs(binding, names, values):
    for name, value in zip(names, values):
        binding.bind_ortvalue_input(name, value)


def _bind_device_outputs(binding, names):
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def _in_names(session):
    return [x.name for x in session.get_inputs()]


def _out_names(session):
    return [x.name for x in session.get_outputs()]


def _np_dtype(meta):
    return np.float16 if "float16" in meta.type else np.float32


print('\nLoading the model...')

# ---- Core pipeline sessions ----
ort_session_Metadata = _make_session(onnx_model_Metadata)
ort_session_Encoder = _make_session(onnx_model_Encoder)
print(f"\nUsable Providers: {ort_session_Encoder.get_providers()}")
shape_value_in = ort_session_Encoder._inputs_meta[0].shape[-1]
in_name_Encoder = _in_names(ort_session_Encoder)
out_name_Encoder = _out_names(ort_session_Encoder)
in_name_Encoder0 = in_name_Encoder[0]

# The audio input dtype is taken straight from the encoder's audio input tensor in the ONNX model,
# so it always matches how the model was exported: "int16" -> raw PCM (the graph divides by 32768),
# "float16"/"float" -> audio pre-normalised to [-1, 1] in prepare_audio_input.
_audio_input_type = ort_session_Encoder._inputs_meta[0].type
input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")
_audio_np_dtype = {"INT16": np.int16, "F32": np.float32, "F16": np.float16}[input_audio_dtype]   # sliding-window buffer dtype
binding_Encoder = ort_session_Encoder.io_binding()

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Dolphin.py to stamp the model metadata."
        )
    return int(value)


def _meta_int_list(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Dolphin.py to stamp the model metadata."
        )
    return [int(x) for x in value.split(",") if x != ""]


MAX_SEQ_LEN = _meta_int("max_seq_len")
SAMPLE_RATE = _meta_int("sample_rate")
METADATA_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
STOP_TOKEN = _meta_int_list("stop_token_ids")
print(f"\nModel metadata: {len(_model_meta)} keys "
    f"(max_seq_len={MAX_SEQ_LEN}, sample_rate={SAMPLE_RATE}, "
    f"input_audio_length={METADATA_INPUT_AUDIO_LENGTH}, "
      f"stop_token_ids={STOP_TOKEN}).")

ort_session_Decoder = _make_session(onnx_model_Decoder)
in_name_Decoder = _in_names(ort_session_Decoder)
out_name_Decoder = _out_names(ort_session_Decoder)
amount_of_outputs_Decoder = len(out_name_Decoder)
binding_Decoder = ort_session_Decoder.io_binding()

num_layers = (amount_of_outputs_Decoder - 1) // 2          # outputs = decoder K/V caches (2 * L) + logits.
num_keys_values = num_layers + num_layers
# Beam search I/O indices (Qwen ASR layout). Inputs: caches(2L), logits, save_id[, prev_prob], beam_size[, topK].
# Outputs: caches(2L), save_id, top_beam_prob, top_beam_indices (next token), max_logits_idx.
beam_logits_in_idx   = num_keys_values                # logits input to First/Second beam.
beam_save_id_in_idx  = num_keys_values + 1            # save_id input to First/Second beam.
first_beam_size_idx  = num_keys_values + 2            # beam_size input (First beam).
second_prev_prob_idx = num_keys_values + 2            # previous_prob input (Second beam).
second_beam_size_idx = num_keys_values + 3            # beam_size input (Second beam).
second_topk_idx      = num_keys_values + 4            # topK input (Second beam).
beam_save_id_out_idx = num_keys_values                # save_id output.
beam_prob_out_idx    = num_keys_values + 1            # top_beam_prob output.
beam_ids_out_idx     = num_keys_values + 2            # top_beam_indices output (next token to embed).
beam_max_out_idx     = num_keys_values + 3            # max_logits_idx output (stop check).
idx_en_key = num_keys_values                          # decoder inputs: en cross-attn keys start (2 * L).
idx_en_value = idx_en_key + num_layers                # en cross-attn values start (3 * L).
idx_hidden = idx_en_value + num_layers                # token-embedding (hidden_states) input (4 * L).
idx_position = idx_hidden + 1                         # position-embedding input (4 * L + 1); mask is in_name_Decoder[-1].
out_name_Decoder_kv = out_name_Decoder[:num_keys_values]
out_name_Decoder_logits = out_name_Decoder[num_keys_values]
in_name_Decoder_self_kv = in_name_Decoder[:num_keys_values]         # decoder self-attn K/V cache feedback (beam / greedy).
in_name_Decoder_en_kv = in_name_Decoder[idx_en_key: idx_hidden]     # encoder cross-attn K/V (rebound once per window).
in_name_Decoder_hidden = in_name_Decoder[idx_hidden]               # token-embedding (hidden_states) input.
in_name_Decoder_position = in_name_Decoder[idx_position]           # position-embedding input.
in_name_Decoder_mask = in_name_Decoder[-1]                         # causal attention-mask input.
vocab_size = ort_session_Decoder._outputs_meta[num_keys_values].shape[-1]
hidden_size = ort_session_Decoder._inputs_meta[idx_hidden].shape[-1]

kv_cache_dtype = _np_dtype(ort_session_Decoder._inputs_meta[0])
logits_dtype = _np_dtype(ort_session_Decoder._outputs_meta[num_keys_values])
hidden_dtype = _np_dtype(ort_session_Decoder._inputs_meta[idx_hidden])
position_dtype = _np_dtype(ort_session_Decoder._inputs_meta[idx_position])
mask_dtype = _np_dtype(ort_session_Decoder._inputs_meta[-1])

ort_session_Embed = _make_session(onnx_model_Embed)
in_name_Embed = _in_names(ort_session_Embed)
out_name_Embed = _out_names(ort_session_Embed)
in_name_Embed0 = in_name_Embed[0]
out_name_Embed0 = out_name_Embed[0]
binding_Embed = ort_session_Embed.io_binding()

ort_session_Prefill = _make_session(onnx_model_Prefill)
in_name_Prefill = _in_names(ort_session_Prefill)
out_name_Prefill = _out_names(ort_session_Prefill)
binding_Prefill = ort_session_Prefill.io_binding()

ort_session_Decode = _make_session(onnx_model_Decode)
in_name_Decode = _in_names(ort_session_Decode)
out_name_Decode = _out_names(ort_session_Decode)
in_name_Decode0 = in_name_Decode[0]
out_name_Decode_position = out_name_Decode[0]
out_name_Decode_kv_seq_len = out_name_Decode[1]
binding_Decode = ort_session_Decode.io_binding()

tokenizer = Tokenizer(save_vocab, os.path.join(onnx_folder, "bpe.model"))

# ---- Decoding-strategy resolution ----
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE
if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")
if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1
decode_batch = BEAM_SIZE
do_repeat_penality = REPEAT_PENALITY != 1.0

beam_size_ort = _ort_from_data([BEAM_SIZE], np.int64)
topK_ort = _ort_from_data([TOP_K], np.int64)

if USE_BEAM_SEARCH:
    ort_session_First_Beam = _make_session(onnx_model_First_Beam)
    in_name_First_Beam = _in_names(ort_session_First_Beam)
    out_name_First_Beam = _out_names(ort_session_First_Beam)
    binding_First_Beam = ort_session_First_Beam.io_binding()

    ort_session_Second_Beam = _make_session(onnx_model_Second_Beam)
    in_name_Second_Beam = _in_names(ort_session_Second_Beam)
    out_name_Second_Beam = _out_names(ort_session_Second_Beam)
    binding_Second_Beam = ort_session_Second_Beam.io_binding()

    # Pre-slice the beam I/O names once so the hot decode loop never re-slices these lists.
    in_name_First_Beam_kv = in_name_First_Beam[:num_keys_values]
    out_name_First_Beam_kv = out_name_First_Beam[:num_keys_values]
    in_name_First_Beam_logits = in_name_First_Beam[beam_logits_in_idx]
    in_name_First_Beam_save_id = in_name_First_Beam[beam_save_id_in_idx]
    out_name_First_Beam_save_id = out_name_First_Beam[beam_save_id_out_idx]
    out_name_First_Beam_prob = out_name_First_Beam[beam_prob_out_idx]
    out_name_First_Beam_ids = out_name_First_Beam[beam_ids_out_idx]
    out_name_First_Beam_max = out_name_First_Beam[beam_max_out_idx]
    in_name_Second_Beam_kv = in_name_Second_Beam[:num_keys_values]
    out_name_Second_Beam_kv = out_name_Second_Beam[:num_keys_values]
    in_name_Second_Beam_logits = in_name_Second_Beam[beam_logits_in_idx]
    in_name_Second_Beam_save_id = in_name_Second_Beam[beam_save_id_in_idx]
    in_name_Second_Beam_prev_prob = in_name_Second_Beam[second_prev_prob_idx]
    out_name_Second_Beam_save_id = out_name_Second_Beam[beam_save_id_out_idx]
    out_name_Second_Beam_prob = out_name_Second_Beam[beam_prob_out_idx]
    out_name_Second_Beam_ids = out_name_Second_Beam[beam_ids_out_idx]
    out_name_Second_Beam_max = out_name_Second_Beam[beam_max_out_idx]

    prob_dtype = _np_dtype(ort_session_First_Beam._outputs_meta[beam_prob_out_idx])

    # Shared beam buffers (fixed shape, reused every step).
    beam_ids_buf = _ort_zeros((BEAM_SIZE, 1), np.int32)              # top_beam_indices (next token to embed).
    beam_score_buf = _ort_zeros((BEAM_SIZE, 1), prob_dtype)         # top_beam_prob / previous_prob (self-aliased).

    # Static beam inputs (bound once).
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[first_beam_size_idx], beam_size_ort)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_size_idx], beam_size_ort)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_topk_idx], topK_ort)
else:
    ort_session_Greedy = _make_session(onnx_model_Greedy)
    in_name_Greedy = _in_names(ort_session_Greedy)
    out_name_Greedy = _out_names(ort_session_Greedy)
    binding_Greedy = ort_session_Greedy.io_binding()
    in_name_Greedy_logits = in_name_Greedy[0]
    in_name_Greedy_save_id = in_name_Greedy[1]
    out_name_Greedy_max = out_name_Greedy[0]
    out_name_Greedy_save_id = out_name_Greedy[1]

    ort_session_Argmax = _make_session(onnx_model_Argmax)
    in_name_Argmax = _in_names(ort_session_Argmax)
    out_name_Argmax = _out_names(ort_session_Argmax)
    binding_Argmax = ort_session_Argmax.io_binding()
    in_name_Argmax_logits = in_name_Argmax[0]
    out_name_Argmax_max = out_name_Argmax[0]

# Repetition penalty is a standalone pass applied to the logits before greedy / beam selection (Qwen ASR style).
if do_repeat_penality:
    ort_session_Penality = _make_session(onnx_model_Penality)
    in_name_Penality = _in_names(ort_session_Penality)
    out_name_Penality = _out_names(ort_session_Penality)[0]
    binding_Penality = ort_session_Penality.io_binding()
    in_name_Penality_logits = in_name_Penality[0]
    in_name_Penality_save_id = in_name_Penality[1]
    num_penality_inputs = len(in_name_Penality)
    # penality_range is baked into the graph via int(), so ORT may prune it as a dead input; guard the binds.
    if num_penality_inputs > 2:
        penality_value_dtype = _np_dtype(ort_session_Penality._inputs_meta[2])
        penality_value_ort = _ort_from_data([REPEAT_PENALITY], penality_value_dtype)
        binding_Penality.bind_ortvalue_input(in_name_Penality[2], penality_value_ort)
    if num_penality_inputs > 3:
        penality_range_ort = _ort_from_data([PENALITY_RANGE], np.int64)
        binding_Penality.bind_ortvalue_input(in_name_Penality[3], penality_range_ort)

# ---- Language / region target resolution (Dolphin-specific two-stage detection) ----
language_region = LANGUAGE_REGION.get(TARGET_LANGUAGE, "NONE")
if language_region == "NONE":
    TARGET_LANGUAGE = "Auto-Auto"
    print(f"\nThe language or region:{TARGET_LANGUAGE} not found. \nFallback to auto detection.")
    language_region = LANGUAGE_REGION[TARGET_LANGUAGE]
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

# ---- Fixed shared buffers (sized from model meta; the audio window is a fresh OrtValue per window) ----
history_len_ort = _ort_from_data([0], np.int64)                    # history_len = 0 (each prefill / detection pass starts fresh).
hidden_states_buf = _ort_zeros((decode_batch, 1, hidden_size), hidden_dtype)
position_buf = _ort_zeros((1, 1, hidden_size), position_dtype)
decode_mask_buf = _ort_zeros((1, 1, 1), mask_dtype)                # decode-phase mask is all-zeros (the new token sees every cached position).
prefill_logits_buf = _ort_zeros((1, vocab_size), logits_dtype)
decode_logits_buf = _ort_zeros((decode_batch, vocab_size), logits_dtype)
max_idx_buf = _ort_zeros((1, 1), np.int32)                        # next-token (greedy) / stop-check (beam) buffer.

detect_language_ids_buf = _ort_from_data([[39999]], np.int32)
detect_region_ids_np = np.empty((1, 2), dtype=np.int32)
detect_region_ids_np[0, 0] = 39999
detect_region_ids_buf = _ort_zeros((1, 2), np.int32)
input_ids_np = np.empty((1, 5), dtype=np.int32)
input_ids_np[0, [0, 3, 4]] = [39999, 6, 324]
input_ids_buf = _ort_zeros((1, 5), np.int32)
ids_len_1_ort = _ort_from_data([1], np.int64)
ids_len_2_ort = _ort_from_data([2], np.int64)
input_ids_len_ort = _ort_from_data([input_ids_np.shape[-1]], np.int64)

# Empty decoder self-attention caches (batch 1) reused for every detection / prefill pass.
past_keys_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[0].shape[1], ort_session_Decoder._outputs_meta[0].shape[2], 0), kv_cache_dtype)
past_values_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[num_layers].shape[1], 0, ort_session_Decoder._outputs_meta[num_layers].shape[3]), kv_cache_dtype)


def run_prefill_logits(prompt_ort, ids_len_ort):
    # Run Embed -> Prefill -> Decoder on a short prompt (empty self-caches + the current window's encoder cross-KV)
    # and return the FULL decoder logits for the last prompt token. Used for the language / region detection passes.
    _bind_inputs(binding_Decoder, in_name_Decoder[:num_layers], [past_keys_Decoder] * num_layers)
    _bind_inputs(binding_Decoder, in_name_Decoder[num_layers:num_keys_values], [past_values_Decoder] * num_layers)
    binding_Embed.bind_ortvalue_input(in_name_Embed0, prompt_ort)
    _bind_device_outputs(binding_Embed, out_name_Embed)
    _run(ort_session_Embed, binding_Embed)
    prompt_embed = binding_Embed.get_outputs()[0]
    _bind_inputs(binding_Prefill, in_name_Prefill, [ids_len_ort, history_len_ort])
    _bind_device_outputs(binding_Prefill, out_name_Prefill)
    _run(ort_session_Prefill, binding_Prefill)
    prefill_out = binding_Prefill.get_outputs()
    binding_Decoder.bind_ortvalue_input(in_name_Decoder_hidden, prompt_embed)
    binding_Decoder.bind_ortvalue_input(in_name_Decoder_position, prefill_out[0])
    binding_Decoder.bind_ortvalue_input(in_name_Decoder_mask, prefill_out[1])
    _bind_device_outputs(binding_Decoder, out_name_Decoder_kv)
    binding_Decoder.bind_ortvalue_output(out_name_Decoder_logits, prefill_logits_buf)
    _run(ort_session_Decoder, binding_Decoder)
    return prefill_logits_buf.numpy()

# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    audio_len = len(audio)
    audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(METADATA_INPUT_AUDIO_LENGTH, audio_len)
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
        audio = np.concatenate((audio, np.zeros((1, 1, pad_amount), dtype=audio.dtype)), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio = np.concatenate((audio, np.zeros((1, 1, INPUT_AUDIO_LENGTH - audio_len), dtype=audio.dtype)), axis=-1)
    aligned_len = audio.shape[-1]

    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    num_decode = 0
    audio_buffer = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), _audio_np_dtype)
    # Start to run dolphin
    start_time = time.time()
    while slice_end <= aligned_len:
        # ---- Encoder (per-clip shared buffer keeps Dolphin's exact dynamic input length while avoiding per-window OrtValue churn) ----
        audio_buffer.update_inplace(np.ascontiguousarray(audio[:, :, slice_start: slice_end]))
        binding_Encoder.bind_ortvalue_input(in_name_Encoder0, audio_buffer)
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
        _run(ort_session_Encoder, binding_Encoder)
        en_kv = binding_Encoder.get_outputs()                              # f16 cross-attn K/V, valid for the whole window.
        _bind_inputs(binding_Decoder, in_name_Decoder_en_kv, en_kv)

        # ---- Stage 1: language detection (SOS-only prompt, ids_len = 1) ----
        if detect_language:
            print("\nAutomatically detect which language it is.")
            detect_logits = run_prefill_logits(detect_language_ids_buf, ids_len_1_ort)
            lang_id = int(np.argmax(detect_logits[0, 7:145])) + 7
        # ---- Stage 2: region detection ([sos, lang] prompt, ids_len = 2) ----
        if detect_region:
            print("\nAutomatically detect which region it is.")
            detect_region_ids_np[0, 1] = lang_id
            detect_region_ids_buf.update_inplace(detect_region_ids_np)
            detect_logits = run_prefill_logits(detect_region_ids_buf, ids_len_2_ort)
            region_id = int(np.argmax(detect_logits[0, 145:324])) + 145

        if detect_language or detect_region:
            lang_str = tokenizer.decode(lang_id)
            region_str = tokenizer.decode(region_id)
            message = f"\nThis audio belongs to {lang_str}-{region_str}."
            message = message.replace("<", "").replace(">", "")
            print(message)
        else:
            print(f"\nThis audio belongs to {TARGET_LANGUAGE}.")

        # ---- Final prompt prefill: [sos = 39999, lang, region, asr = 6, notimestamp = 324] ----
        input_ids_np[0, 1] = lang_id
        input_ids_np[0, 2] = region_id
        input_ids_buf.update_inplace(input_ids_np)
        generate_limit = MAX_SEQ_LEN - input_ids_np.shape[-1]
        _bind_inputs(binding_Decoder, in_name_Decoder[:num_layers], [past_keys_Decoder] * num_layers)
        _bind_inputs(binding_Decoder, in_name_Decoder[num_layers:num_keys_values], [past_values_Decoder] * num_layers)

        binding_Embed.bind_ortvalue_input(in_name_Embed0, input_ids_buf)
        _bind_device_outputs(binding_Embed, out_name_Embed)
        _run(ort_session_Embed, binding_Embed)
        prompt_embed = binding_Embed.get_outputs()[0]

        _bind_inputs(binding_Prefill, in_name_Prefill, [input_ids_len_ort, history_len_ort])
        _bind_device_outputs(binding_Prefill, out_name_Prefill)
        _run(ort_session_Prefill, binding_Prefill)
        prefill_out = binding_Prefill.get_outputs()
        kv_seq_len_ort = prefill_out[2]

        binding_Decoder.bind_ortvalue_input(in_name_Decoder_hidden, prompt_embed)
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_position, prefill_out[0])
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_mask, prefill_out[1])
        _bind_device_outputs(binding_Decoder, out_name_Decoder_kv)
        binding_Decoder.bind_ortvalue_output(out_name_Decoder_logits, prefill_logits_buf)

        # Decode-phase position graph: kv_seq_len is incremented in place (input aliased to output).
        binding_Decode.bind_ortvalue_input(in_name_Decode0, kv_seq_len_ort)
        binding_Decode.bind_ortvalue_output(out_name_Decode_position, position_buf)
        binding_Decode.bind_ortvalue_output(out_name_Decode_kv_seq_len, kv_seq_len_ort)

        # Decode-phase embedding graph: next token -> shared hidden buffer.
        if USE_BEAM_SEARCH:
            binding_Embed.bind_ortvalue_input(in_name_Embed0, beam_ids_buf)
        else:
            binding_Embed.bind_ortvalue_input(in_name_Embed0, max_idx_buf)
        binding_Embed.bind_ortvalue_output(out_name_Embed0, hidden_states_buf)

        if USE_BEAM_SEARCH:
            save_id_buf = _ort_zeros((BEAM_SIZE, 0), np.int32)             # initial empty per-beam history.
            save_id = save_id_buf                                          # current on-device save_id (penalty / feedback).
            latest_save_id = save_id_buf                                   # last beam save_id for final detokenisation.
            binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[beam_save_id_in_idx], save_id_buf)
        else:
            save_id = _ort_zeros((1, 0), np.int32)                         # on-device greedy history (used when penalty is on).
            save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)         # host-side history (used by the argmax path).

        num_decode = 0
        is_prefill_step = True
        while num_decode < generate_limit:
            _run(ort_session_Decoder, binding_Decoder)
            outputs_Decoder = binding_Decoder.get_outputs()
            cur_logits_buf = prefill_logits_buf if is_prefill_step else decode_logits_buf

            # Repetition penalty: a standalone in-place pass over the most recent tokens (Qwen ASR style).
            if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                binding_Penality.bind_ortvalue_input(in_name_Penality_logits, cur_logits_buf)
                binding_Penality.bind_ortvalue_input(in_name_Penality_save_id, save_id)
                binding_Penality.bind_ortvalue_output(out_name_Penality, cur_logits_buf)
                _run(ort_session_Penality, binding_Penality)

            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    _bind_inputs(binding_First_Beam, in_name_First_Beam_kv, outputs_Decoder[:num_keys_values])
                    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam_logits, cur_logits_buf)
                    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam_save_id, save_id)
                    # Bind outputs in graph order; get_outputs() returns values in bind order, so the order must match the graph.
                    _bind_device_outputs(binding_First_Beam, out_name_First_Beam_kv)                                         # caches 0..2L-1
                    binding_First_Beam._iobinding.bind_output(out_name_First_Beam_save_id, _ort_device_obj)                  # save_id
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam_prob, beam_score_buf)                       # top_beam_prob
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam_ids, beam_ids_buf)                          # top_beam_indices (next token)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam_max, max_idx_buf)                           # max_logits_idx (stop check)
                    _run(ort_session_First_Beam, binding_First_Beam)
                    outputs_Beam = binding_First_Beam.get_outputs()
                else:
                    _bind_inputs(binding_Second_Beam, in_name_Second_Beam_kv, outputs_Decoder[:num_keys_values])
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_logits, cur_logits_buf)
                    if num_decode < 2:
                        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_prev_prob, beam_score_buf)
                    # Bind outputs in graph order every step; get_outputs() returns values in bind order.
                    _bind_device_outputs(binding_Second_Beam, out_name_Second_Beam_kv)                                       # caches 0..2L-1
                    binding_Second_Beam._iobinding.bind_output(out_name_Second_Beam_save_id, _ort_device_obj)                # save_id
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam_prob, beam_score_buf)                     # top_beam_prob
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam_ids, beam_ids_buf)                        # top_beam_indices (next token)
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam_max, max_idx_buf)                         # max_logits_idx (stop check)
                    _run(ort_session_Second_Beam, binding_Second_Beam)
                    outputs_Beam = binding_Second_Beam.get_outputs()

                save_id = outputs_Beam[beam_save_id_out_idx]
                latest_save_id = save_id
                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN:
                    break

                _bind_inputs(binding_Decoder, in_name_Decoder_self_kv, outputs_Beam[:num_keys_values])
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_save_id, save_id)
            else:
                if do_repeat_penality:
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy_logits, cur_logits_buf)
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy_save_id, save_id)
                    binding_Greedy.bind_ortvalue_output(out_name_Greedy_max, max_idx_buf)
                    binding_Greedy._iobinding.bind_output(out_name_Greedy_save_id, _ort_device_obj)
                    _run(ort_session_Greedy, binding_Greedy)
                    save_id = binding_Greedy.get_outputs()[1]
                else:
                    binding_Argmax.bind_ortvalue_input(in_name_Argmax_logits, cur_logits_buf)
                    binding_Argmax.bind_ortvalue_output(out_name_Argmax_max, max_idx_buf)
                    _run(ort_session_Argmax, binding_Argmax)

                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN:
                    break
                if not do_repeat_penality:
                    save_id_greedy[num_decode] = max_logits_idx
                _bind_inputs(binding_Decoder, in_name_Decoder_self_kv, outputs_Decoder[:num_keys_values])

            _bind_device_outputs(binding_Decoder, out_name_Decoder_kv)
            if is_prefill_step:
                binding_Decoder.bind_ortvalue_input(in_name_Decoder_hidden, hidden_states_buf)
                binding_Decoder.bind_ortvalue_input(in_name_Decoder_position, position_buf)
                binding_Decoder.bind_ortvalue_input(in_name_Decoder_mask, decode_mask_buf)
                is_prefill_step = False
            binding_Decoder.bind_ortvalue_output(out_name_Decoder_logits, decode_logits_buf)

            _run(ort_session_Embed, binding_Embed)
            _run(ort_session_Decode, binding_Decode)
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    rtf = count_time / max(audio_len / SAMPLE_RATE, 1e-6)

    if USE_BEAM_SEARCH:
        save_token_array = latest_save_id.numpy()[0]
        for i, idx in enumerate(save_token_array):
            if idx in STOP_TOKEN:
                save_token_array = save_token_array[:i]
                break
    elif do_repeat_penality:
        # Greedy with penalty keeps its token history on-device (GREEDY_SEARCH appends each step).
        save_token_array = save_id.numpy()[0]
        for i, idx in enumerate(save_token_array):
            if idx in STOP_TOKEN:
                save_token_array = save_token_array[:i]
                break
    else:
        save_token_array = save_id_greedy[:num_decode]
    text = tokenizer.decode_ids(save_token_array)
    print(f"\nASR Result:\n{text}\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, {num_decode} tokens)")
    print("----------------------------------------------------------------------------------------------------------")
