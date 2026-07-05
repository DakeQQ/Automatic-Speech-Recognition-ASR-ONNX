import argparse
import os
import sys
import time
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer, GenerationConfig


# ============================================================================
# Paths
# ============================================================================
def _parse_args():
    parser = argparse.ArgumentParser(description="Run Whisper ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Whisper_Optimized"), help="Folder containing ONNX graphs, for example Whisper_Optimized or Whisper_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()

model_path                = "/home/DakeQQ/Downloads/whisper-large-v3-turbo"                               # Whisper model folder (tokenizer + generation/feature config).
onnx_folder               = os.path.abspath(_ARGS.onnx_folder)   # Selected ONNX graph folder.
onnx_model_Metadata       = os.path.join(onnx_folder, "Whisper_Metadata.onnx")
onnx_model_Encoder        = os.path.join(onnx_folder, "Whisper_Encoder.onnx")               # The exported onnx model paths.
onnx_model_Decoder        = os.path.join(onnx_folder, "Whisper_Decoder.onnx")
onnx_model_Embed          = os.path.join(onnx_folder, "Whisper_Decoder_Embed.onnx")
onnx_model_Prefill        = os.path.join(onnx_folder, "Whisper_Position_Mask_Prefill.onnx")
onnx_model_Decode         = os.path.join(onnx_folder, "Whisper_Position_Mask_Decode.onnx")
onnx_model_Greedy         = os.path.join(onnx_folder, "Whisper_Greedy_Search.onnx")
onnx_model_Argmax         = os.path.join(onnx_folder, "Whisper_Argmax.onnx")
onnx_model_First_Beam     = os.path.join(onnx_folder, "Whisper_First_Beam_Search.onnx")
onnx_model_Second_Beam    = os.path.join(onnx_folder, "Whisper_Second_Beam_Search.onnx")
onnx_model_Penality       = os.path.join(onnx_folder, "Whisper_Apply_Penality.onnx")
onnx_model_No_Speech      = os.path.join(onnx_folder, "Whisper_No_Speech_Detection.onnx")


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                                              # Anchor helper imports to this script's folder, not the current working directory.
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from Example_Audio import model_audio_cases

test_audio_cases = model_audio_cases("whisper")                                                       # The test audio cases.
test_audio = [path for path, _language in test_audio_cases]                                           # The test audio list.


# ============================================================================
# Decoding configuration
# ============================================================================
USE_BEAM_SEARCH       = False           # Use beam search (True) or greedy search (False).
BEAM_SIZE             = 3               # Number of beams in searching.
TOP_K                 = 3               # Top-k candidates considered during decoding.

REPEAT_PENALITY       = 0.8             # Range 0.0 - 1.0; "1.0" means no penalty.
PENALITY_RANGE        = 20              # Penalizes the most recent output; "10" means the last 10 tokens.
REMOVE_REPEATED_PARTS = False           # Non-Whisper cleanup for runaway repetition;

TARGET_LANGUAGE       = "en"            # A language listed in the get_language_id function's language_map.
TASK                  = "transcribe"    # One of: ['transcribe', 'translate'].
DETECT_LANGUAGE       = True            # Whisper-style auto language detection; overrides TARGET_LANGUAGE when True.

NO_SPEECH_DETECTION   = True            # Skip silent / non-speech windows using the <|nospeech|> probability.
NO_SPEECH_THRESHOLD   = 0.6             # Whisper default; higher = stricter silence rejection.

SLIDING_WINDOW        = 0               # Sliding-window step for reading test audio; 0 disables.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# ("int16" -> raw PCM ÷32768 in-graph; "float16"/"float" -> pre-normalised [-1, 1]); no manual setting needed.
USE_NORMALISE_AUDIO   = False           # Apply RMS loudness normalisation before feeding the model. Whisper keeps the raw waveform amplitude by default.


# ============================================================================
# ONNX Runtime runtime configuration (mirrors the Qwen ASR inference pipeline)
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


# ============================================================================
# Load tokenizer / generation config / feature extractor and derive constants
# ============================================================================
tokenizer = AutoTokenizer.from_pretrained(model_path)
generation_config = GenerationConfig.from_pretrained(model_path)

model_path_lower = model_path.lower()
if ("v3" in model_path_lower) or ("crisperwhisper" in model_path_lower) or ("anime" in model_path_lower) or ("belle" in model_path_lower) or ("turbo" in model_path_lower) or ("distil" in model_path_lower):
    is_v3 = True
    custom_vocab = 'v0.3' in model_path_lower
else:
    is_v3 = False
    custom_vocab = False

no_timestamps_id = int(getattr(generation_config, "no_timestamps_token_id", tokenizer.convert_tokens_to_ids("<|notimestamps|>")))    # 50364 (v3): selects non-timestamp transcription.
begin_suppress_token_ids = tuple(int(token_id) for token_id in (getattr(generation_config, "begin_suppress_tokens", None) or ()))


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


def get_language_id(language_input, custom_vocab=False, generation_config=None):
    normalized_input = language_input.lower().strip()
    lang_code = None
    if normalized_input in _LANGUAGE_DATA:
        lang_code = normalized_input
    elif normalized_input in _FULL_NAME_TO_CODE:
        lang_code = _FULL_NAME_TO_CODE[normalized_input]
    elif normalized_input in _ALIAS_TO_CODE:
        lang_code = _ALIAS_TO_CODE[normalized_input]
    if lang_code:
        if generation_config is not None and hasattr(generation_config, "lang_to_id"):
            lang_token = f"<|{lang_code}|>"
            if lang_token in generation_config.lang_to_id:
                return int(generation_config.lang_to_id[lang_token])
        language_info = _LANGUAGE_DATA[lang_code]
        id_key = 'custom_id' if custom_vocab else 'id'
        return language_info.get(id_key)
    return None


def get_task_id(task_input, is_v3, custom_vocab=False, generation_config=None):
    task_input = task_input.lower()
    if generation_config is not None and hasattr(generation_config, "task_to_id"):
        start_token = getattr(generation_config, "decoder_start_token_id", None)
        stop_token = getattr(generation_config, "eos_token_id", None)
        if start_token is not None and stop_token is not None and task_input in generation_config.task_to_id:
            return int(start_token), [int(stop_token)], int(generation_config.task_to_id[task_input])
    if custom_vocab:
        stop_token = 18871
        start_token = 18872
        task_map = {'translate': 18973, 'transcribe': 18974}
        return start_token, [stop_token], task_map[task_input]
    stop_token = 50257
    start_token = 50258
    if is_v3:
        task_map = {'translate': 50359, 'transcribe': 50360}
        return start_token, [stop_token], task_map[task_input]
    task_map = {'translate': 50358, 'transcribe': 50359}
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


# ============================================================================
# ONNX Runtime infrastructure (IOBinding + shared buffers, mirrors Qwen ASR)
# ============================================================================
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


print("\nLoading the model...")

# ---- Core pipeline sessions ----
ort_session_Metadata = _make_session(onnx_model_Metadata)
ort_session_Encoder = _make_session(onnx_model_Encoder)
print(f"\nUsable Providers: {ort_session_Encoder.get_providers()}")
shape_value_in = ort_session_Encoder._inputs_meta[0].shape[-1]
in_name_Encoder = _in_names(ort_session_Encoder)
out_name_Encoder = _out_names(ort_session_Encoder)
in_name_Encoder0 = in_name_Encoder[0]
binding_Encoder = ort_session_Encoder.io_binding()

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Whisper.py to stamp the model metadata."
        )
    return int(value)


def _meta_str(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Whisper.py to stamp the model metadata."
        )
    return value


def _meta_bool(key):
    return _meta_str(key) == "1"


def _meta_int_list(key):
    value = _meta_str(key)
    return [int(part) for part in value.split(",") if part]


MAX_SEQ_LEN = _meta_int("max_seq_len")
SAMPLE_RATE = _meta_int("sample_rate")
MAX_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
model_variant = _meta_str("model_variant")
if model_variant not in ("v2", "v3"):
    raise ValueError(f"Unsupported Whisper model_variant metadata value: {model_variant!r}")
is_v3 = model_variant == "v3"
custom_vocab = _meta_bool("custom_vocab")
no_timestamps_id = _meta_int("no_timestamps_token_id")
begin_suppress_token_ids = tuple(_meta_int_list("begin_suppress_token_ids"))
print(f"\nModel metadata: {len(_model_meta)} keys "
      f"(max_seq_len={MAX_SEQ_LEN}, sample_rate={SAMPLE_RATE}, "
      f"input_audio_length={MAX_INPUT_AUDIO_LENGTH}, model_variant={model_variant}, "
      f"no_timestamps_token_id={no_timestamps_id}).")

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

if NO_SPEECH_DETECTION:
    ort_session_No_Speech = _make_session(onnx_model_No_Speech)
    in_name_No_Speech = _in_names(ort_session_No_Speech)
    out_name_No_Speech = _out_names(ort_session_No_Speech)
    in_name_No_Speech0 = in_name_No_Speech[0]
    binding_No_Speech = ort_session_No_Speech.io_binding()

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

# ---- Fixed shared buffers ----
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = MAX_INPUT_AUDIO_LENGTH
else:
    INPUT_AUDIO_LENGTH = int(shape_value_in)
if SLIDING_WINDOW <= 0:
    stride_step = INPUT_AUDIO_LENGTH
else:
    stride_step = SLIDING_WINDOW

begin_suppress_valid = [int(t) for t in begin_suppress_token_ids if 0 <= int(t) < vocab_size]

# The audio input dtype is taken straight from the encoder's audio input tensor in the ONNX model,
# so it always matches how the model was exported: "int16" -> raw PCM (the graph divides by 32768),
# "float16"/"float" -> audio pre-normalised to [-1, 1] in prepare_audio_input.
_audio_input_type = ort_session_Encoder._inputs_meta[0].type
input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")
_audio_np_dtype = {"INT16": np.int16, "F32": np.float32, "F16": np.float16}[input_audio_dtype]
audio_buffer = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), _audio_np_dtype)
hidden_states_buf = _ort_zeros((decode_batch, 1, hidden_size), hidden_dtype)
position_buf = _ort_zeros((1, 1, hidden_size), position_dtype)
decode_mask_buf = _ort_zeros((1, 1, 1), mask_dtype)                 # decode-phase mask is all-zeros (new token sees every cached position).
prefill_logits_buf = _ort_zeros((1, vocab_size), logits_dtype)
decode_logits_buf = _ort_zeros((decode_batch, vocab_size), logits_dtype)
max_idx_buf = _ort_zeros((1, 1), np.int32)                         # next-token (greedy) / stop-check (beam) buffer.

# Language-detection lookup tables (token id <-> language code).
if hasattr(generation_config, "lang_to_id"):
    language_token_to_code = {int(token_id): lang_token[2:-2] for lang_token, token_id in generation_config.lang_to_id.items()}
    language_token_ids = np.array(list(language_token_to_code.keys()), dtype=np.int64)
elif custom_vocab:
    language_token_ids = np.array([data['custom_id'] for data in _LANGUAGE_DATA.values()], dtype=np.int64)
    language_token_to_code = {int(token_id): code for code, token_id in zip(_LANGUAGE_DATA.keys(), language_token_ids)}
else:
    language_token_ids = np.array([data['id'] for data in _LANGUAGE_DATA.values()], dtype=np.int64)
    language_token_to_code = {int(token_id): code for code, token_id in zip(_LANGUAGE_DATA.keys(), language_token_ids)}


# ============================================================================
# Inference
# ============================================================================
# Task / SOT constants are invariant across every test clip (TASK is fixed); compute them once.
start, STOP_TOKEN, task = get_task_id(TASK, is_v3, custom_vocab, generation_config)
sot_ids = _ort_from_numpy(np.array([[start]], dtype=np.int32))  # SOT-only prompt for language / no-speech detection.
ids_len_1_ort = _ort_from_data([1], np.int64)                  # ids_len = 1 for the SOT detection prefill.
history_len_ort = _ort_from_data([0], np.int64)                # history_len = 0 (self-attn caches start empty per clip).
for language_idx, (test, demo_language) in enumerate(test_audio_cases):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    language = demo_language or TARGET_LANGUAGE

    audio_segment = AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    audio = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)  # Raw PCM; prepare_audio_input owns the optional RMS + dtype conversion.
    audio_len = len(audio)
    audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)
    if audio_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        silence_pad = np.zeros((1, 1, pad_amount), dtype=audio.dtype)  # Whisper pads with silence (zeros), not noise, to avoid hallucinated tokens.
        audio = np.concatenate((audio, silence_pad), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        silence_pad = np.zeros((1, 1, INPUT_AUDIO_LENGTH - audio_len), dtype=audio.dtype)
        audio = np.concatenate((audio, silence_pad), axis=-1)
    aligned_len = audio.shape[-1]

    # ---- Per-audio state ----
    first_window = True
    prefill_ready = False
    language_id = get_language_id(language, custom_vocab, generation_config)
    if language_id is None:
        raise ValueError(f"Unsupported language: {language}")
    input_ids = np.array([[start, language_id, task, no_timestamps_id]], dtype=np.int32)
    generate_limit = MAX_SEQ_LEN - input_ids.shape[-1]

    # Decoder self-attention caches start empty (batch = 1 at prefill).
    past_keys_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[0].shape[1], ort_session_Decoder._outputs_meta[0].shape[2], 0), kv_cache_dtype)
    past_values_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[num_layers].shape[1], 0, ort_session_Decoder._outputs_meta[num_layers].shape[3]), kv_cache_dtype)
    _bind_inputs(binding_Decoder, in_name_Decoder[:num_layers], [past_keys_Decoder] * num_layers)
    _bind_inputs(binding_Decoder, in_name_Decoder[num_layers:num_keys_values], [past_values_Decoder] * num_layers)

    if USE_BEAM_SEARCH:
        save_id_buf = _ort_zeros((BEAM_SIZE, 0), np.int32)          # initial empty per-beam history.
        save_id = save_id_buf                                       # current on-device save_id (penalty / feedback).
        latest_save_id = save_id_buf                                # last beam save_id for final detokenisation.
        binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[beam_save_id_in_idx], save_id_buf)
    else:
        save_id = _ort_zeros((1, 0), np.int32)                      # on-device greedy history (used when penalty is on).
        save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)      # host-side history (used by the argmax path).

    num_decode = 0
    is_prefill_step = True
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    start_time = time.time()
    while slice_end <= aligned_len:
        audio_buffer.update_inplace(np.ascontiguousarray(audio[:, :, slice_start: slice_end]))
        binding_Encoder.bind_ortvalue_input(in_name_Encoder0, audio_buffer)
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
        _run(ort_session_Encoder, binding_Encoder)
        en_kv = binding_Encoder.get_outputs()                              # cross-attn K/V, valid for the whole window.
        _bind_inputs(binding_Decoder, in_name_Decoder_en_kv, en_kv)

        if first_window and (DETECT_LANGUAGE or NO_SPEECH_DETECTION):
            first_window = False
            # Whisper reads the <|sot|> distribution: embed [sot] and run the decoder on a single token.
            binding_Embed.bind_ortvalue_input(in_name_Embed0, sot_ids)
            _bind_device_outputs(binding_Embed, out_name_Embed)
            _run(ort_session_Embed, binding_Embed)
            sot_embed = binding_Embed.get_outputs()[0]

            _bind_inputs(binding_Prefill, in_name_Prefill, [ids_len_1_ort, history_len_ort])
            _bind_device_outputs(binding_Prefill, out_name_Prefill)
            _run(ort_session_Prefill, binding_Prefill)
            det_prefill = binding_Prefill.get_outputs()

            binding_Decoder.bind_ortvalue_input(in_name_Decoder_hidden, sot_embed)
            binding_Decoder.bind_ortvalue_input(in_name_Decoder_position, det_prefill[0])
            binding_Decoder.bind_ortvalue_input(in_name_Decoder_mask, det_prefill[1])
            _bind_device_outputs(binding_Decoder, out_name_Decoder_kv)
            binding_Decoder.bind_ortvalue_output(out_name_Decoder_logits, prefill_logits_buf)
            _run(ort_session_Decoder, binding_Decoder)

            if DETECT_LANGUAGE:
                detect_logits = prefill_logits_buf.numpy().reshape(-1)
                detected_token = int(language_token_ids[np.argmax(detect_logits[language_token_ids])])
                language = language_token_to_code.get(detected_token, language)
                input_ids = np.array([[start, detected_token, task, no_timestamps_id]], dtype=np.int32)
                print(f"Detected Language: {language}")
            if NO_SPEECH_DETECTION:
                binding_No_Speech.bind_ortvalue_input(in_name_No_Speech0, prefill_logits_buf)
                _bind_device_outputs(binding_No_Speech, out_name_No_Speech)
                _run(ort_session_No_Speech, binding_No_Speech)
                no_speech_prob = float(binding_No_Speech.get_outputs()[0].numpy().reshape(-1)[0])
                print(f"No-Speech Probability: {no_speech_prob:.3f}")
                if no_speech_prob >= NO_SPEECH_THRESHOLD:
                    print("Window classified as silence / non-speech; skipping transcription.")
                    break

        if not prefill_ready:
            # Prefill: embed the full prompt and produce its position embedding + causal mask once.
            prefill_ready = True
            input_ids_ort = _ort_from_numpy(input_ids)
            ids_len_ort = _ort_from_data([input_ids.shape[-1]], np.int64)
            binding_Embed.bind_ortvalue_input(in_name_Embed0, input_ids_ort)
            _bind_device_outputs(binding_Embed, out_name_Embed)
            _run(ort_session_Embed, binding_Embed)
            prompt_embed = binding_Embed.get_outputs()[0]

            _bind_inputs(binding_Prefill, in_name_Prefill, [ids_len_ort, history_len_ort])
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

        while num_decode < generate_limit:
            _run(ort_session_Decoder, binding_Decoder)
            outputs_Decoder = binding_Decoder.get_outputs()
            cur_logits_buf = prefill_logits_buf if is_prefill_step else decode_logits_buf

            if num_decode < 1 and begin_suppress_valid:
                logits_np = cur_logits_buf.numpy()
                logits_np[..., begin_suppress_valid] = -np.inf
                cur_logits_buf.update_inplace(np.ascontiguousarray(logits_np))

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
                max_logits_idx = int(max_idx_buf.numpy().reshape(-1)[0])
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

                max_logits_idx = int(max_idx_buf.numpy().reshape(-1)[0])
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

    if num_decode > 0:
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
        if REMOVE_REPEATED_PARTS:
            save_token_array = remove_repeated_parts(save_token_array, 3, save_token_array.shape[-1])
        text, _ = tokenizer._decode_asr(
            [{
                "tokens": save_token_array.reshape(1, -1)
            }],
            return_timestamps=None,  # Do not support return timestamps
            return_language=None,
            time_precision=0
        )
        print(f"\nASR Result:\n{text}\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, {num_decode} tokens)")
        print("----------------------------------------------------------------------------------------------------------")
    else:
        print(f"\nASR Result:\n[no speech detected]\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio)")
        print("----------------------------------------------------------------------------------------------------------")
