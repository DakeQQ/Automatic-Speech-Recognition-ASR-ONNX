import time
import argparse
import sys
from pathlib import Path
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths


tokenizer_path                  = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B'                          # Set the tokenizer path.
ctc_tokenizer_path              = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/multilingual.tiktoken'               # SenseVoice multilingual tiktoken vocab for the CTC branch.

def _parse_args():
    parser = argparse.ArgumentParser(description="Run Fun-ASR-Nano ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "Fun_ASR_Nano_Optimized", help="Folder containing ONNX graphs, for example Fun_ASR_Nano_Optimized or Fun_ASR_Nano_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()

# Load the selected ONNX models from a local folder next to this script by default.
onnx_folder                     = _ARGS.onnx_folder.expanduser().resolve()                                      # Selected ONNX graph folder.

onnx_model_Metadata             = str(onnx_folder / "FunASR_Nano_Metadata.onnx")                                # Tiny metadata carrier graph.
onnx_model_Encoder              = str(onnx_folder / "FunASR_Nano_Encoder.onnx")                                 # The exported onnx model path.
onnx_model_CTC_Decoder          = str(onnx_folder / "FunASR_Nano_CTC_Decoder.onnx")                             # Optional fast CTC transcription head; loaded only when USE_CTC_DECODER=True.
onnx_model_Embed                = str(onnx_folder / "FunASR_Nano_Decoder_Embed.onnx")
onnx_model_Main                 = str(onnx_folder / "FunASR_Nano_Decoder_Main.onnx")
onnx_model_Rotary_Mask_Prefill  = str(onnx_folder / "FunASR_Nano_Rotary_Mask_Text_Prefill.onnx")
onnx_model_Rotary_Mask_Decode   = str(onnx_folder / "FunASR_Nano_Rotary_Mask_Text_Decode.onnx")
onnx_model_Greedy               = str(onnx_folder / "FunASR_Nano_Greedy_Search.onnx")
onnx_model_First_Beam           = str(onnx_folder / "FunASR_Nano_First_Beam_Search.onnx")
onnx_model_Second_Beam          = str(onnx_folder / "FunASR_Nano_Second_Beam_Search.onnx")
onnx_model_Penalty              = str(onnx_folder / "FunASR_Nano_Apply_Penalty.onnx")
onnx_model_Argmax               = str(onnx_folder / "FunASR_Nano_Argmax.onnx")


test_audio = model_audio_paths("fun_asr_nano_mlt" if "MLT" in tokenizer_path else "fun_asr_nano")      # The test audio list.
task_prompt = ["将语音转写成中文：", "将语音转写成英文：", "将语音转写成粤语：", "将语音转写成日文："]              # The prompt of transcription task.
if "MLT" in tokenizer_path:
    task_prompt += ["将语音转写成韩文："]


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
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model.
# Kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values; no division by 32768.
USE_NORMALISE_AUDIO = False             # If true, use the audio normalizer to make the loudness consistent.

USE_CTC_DECODER     = True              # If True, also run the fast CTC transcription head. Must match the exported/optimized model set.

# Input & Processing Limits
SLIDING_WINDOW         = 0              # Set the sliding window step for test audio reading; use 0 to disable.

# Decoding Strategy
USE_BEAM_SEARCH       = False           # It recommended to use greedy search for Fun-ASR-Nano.
TOP_K                 = 3               # The top k candidate in decoding.
BEAM_SIZE             = 3               # Number of beams in searching.
PENALTY_RANGE         = 10              # Penalizes the most recent output. "10" means the last 10 tokens.
REPEAT_PENALTY        = 1.0             # Range from 0.0 to 1.0; "1.0" means no penality.

# Runtime & Export Settings
ORT_Accelerate_Providers = []           # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                  = False        # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                 = False        # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS              = 0            # Parllel CPU threads. Set 0 for auto.
DEVICE_ID                = 0            # Default to zero.


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def prepare_audio_input(audio_int16: np.ndarray, input_audio_dtype: str, target_rms: float = 8192.0) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's audio input
    # dtype exactly once. `input_audio_dtype` is derived from the ONNX model's audio input tensor.
    # The Kaldi fbank front-end consumes the int16 numeric range directly, so the float variants
    # carry int16-range values (there is NO ÷32768 here).
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
    if input_audio_dtype == "F16":
        return audio.astype(np.float16)   # NOTE: int16-range in f16 is lossy (~±16 ULP near 32768)
    return audio                          # F32: int16-range values as float32 (kaldi keeps this range)


def bind_ort_in(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out(binding, names, device):
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# CTC TOKENIZER (standalone tiktoken; mirrors funasr SenseVoiceTokenizer / multilingual)
# ══════════════════════════════════════════════════════════════════════════════
class CTCTokenizer:
    """Minimal standalone tiktoken tokenizer for the CTC branch.

    Byte-for-byte reproduction of funasr's `SenseVoiceTokenizer` (the `multilingual`
    encoding), so no funasr dependency is needed at inference time. Only `decode` is
    implemented — it drops timestamp tokens exactly like funasr's `Tokenizer.decode`.
    """

    _PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    _LANGS = (
        'en', 'zh', 'de', 'es', 'ru', 'ko', 'fr', 'ja', 'pt', 'tr', 'pl', 'ca', 'nl', 'ar',
        'sv', 'it', 'id', 'hi', 'fi', 'vi', 'he', 'uk', 'el', 'ms', 'cs', 'ro', 'da', 'hu',
        'ta', 'no', 'th', 'ur', 'hr', 'bg', 'lt', 'la', 'mi', 'ml', 'cy', 'sk', 'te', 'fa',
        'lv', 'bn', 'sr', 'az', 'sl', 'kn', 'et', 'mk', 'br', 'eu', 'is', 'hy', 'ne', 'mn',
        'bs', 'kk', 'sq', 'sw', 'gl', 'mr', 'pa', 'si', 'km', 'sn', 'yo', 'so', 'af', 'oc',
        'ka', 'be', 'tg', 'sd', 'gu', 'am', 'yi', 'lo', 'uz', 'fo', 'ht', 'ps', 'tk', 'nn',
        'mt', 'sa', 'lb', 'my', 'bo', 'tl', 'mg', 'as', 'tt', 'haw', 'ln', 'ha', 'ba', 'jw',
        'su', 'yue', 'minnan', 'wuyu', 'dialect', 'zh/en', 'en/zh',
    )
    _AUDIO_EVENTS = ('ASR', 'AED', 'SER', 'Speech', '/Speech', 'BGM', '/BGM', 'Laughter', '/Laughter', 'Applause', '/Applause')
    _EMOTIONS = ('HAPPY', 'SAD', 'ANGRY', 'NEUTRAL')

    def __init__(self, vocab_path, num_languages=8749):
        import base64
        import tiktoken
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in open(vocab_path) if line)
        }
        n_vocab = len(ranks)
        special_tokens = {}
        specials = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            *[f"<|{lang}|>" for lang in self._LANGS[:num_languages]],
            *[f"<|{event}|>" for event in self._AUDIO_EVENTS],
            *[f"<|{emotion}|>" for emotion in self._EMOTIONS],
            "<|translate|>", "<|transcribe|>", "<|startoflm|>",
            "<|startofprev|>", "<|nospeech|>", "<|notimestamps|>",
            *[f"<|SPECIAL_TOKEN_{i}|>" for i in range(1, 51)],
            *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
        ]
        for token in specials:
            special_tokens[token] = n_vocab
            n_vocab += 1
        self.encoding = tiktoken.Encoding(
            name="multilingual",
            explicit_n_vocab=n_vocab,
            pat_str=self._PAT,
            mergeable_ranks=ranks,
            special_tokens=special_tokens,
        )
        self.timestamp_begin = special_tokens["<|0.00|>"]

    def decode(self, token_ids):
        token_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self.encoding.decode(token_ids)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# --- Encoder ---
ort_session_Metadata = create_session(onnx_model_Metadata, **packed_settings)
ort_session_Encoder = create_session(onnx_model_Encoder, **packed_settings)
binding_Encoder     = ort_session_Encoder.io_binding()
shape_value_in_Encoder = ort_session_Encoder._inputs_meta[0].shape[-1]
in_name_Encoder     = get_in_names(ort_session_Encoder)
out_name_Encoder    = get_out_names(ort_session_Encoder)

# The audio input dtype is taken straight from the encoder's audio input tensor in the ONNX model,
# so it always matches how the model was exported (kaldi fbank keeps the int16 numeric range;
# "float16"/"float" carry int16-range values with no ÷32768). The whole clip is converted once.
_audio_input_type = ort_session_Encoder._inputs_meta[0].type
input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Fun_ASR_Nano.py to stamp the model metadata."
        )
    return int(value)


def _meta_int_list(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Fun_ASR_Nano.py to stamp the model metadata."
        )
    return [int(x) for x in value.split(",") if x != ""]


def _meta_bool(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Encoder}. "
            f"Re-export with Export_Fun_ASR_Nano.py to stamp the model metadata."
        )
    return value == "1"


SAMPLE_RATE = _meta_int("sample_rate")
MAX_SEQ_LEN = _meta_int("max_seq_len")
MAX_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
STOP_TOKEN = _meta_int_list("stop_token_ids")
EXPORTED_USE_CTC_DECODER = _meta_bool("use_ctc_decoder")
if USE_CTC_DECODER and not EXPORTED_USE_CTC_DECODER:
    raise ValueError(
        "USE_CTC_DECODER=True but the loaded Fun_ASR_Nano metadata says use_ctc_decoder=0. "
        "Disable USE_CTC_DECODER or re-export with USE_CTC_DECODER=True."
    )
print(f"\nModel metadata: {len(_model_meta)} keys "
      f"(max_seq_len={MAX_SEQ_LEN}, sample_rate={SAMPLE_RATE}, "
      f"max_input_audio_length={MAX_INPUT_AUDIO_LENGTH}, stop_token_ids={STOP_TOKEN}, "
      f"use_ctc_decoder={EXPORTED_USE_CTC_DECODER}).")

# --- CTC Decoder (optional; loaded only when USE_CTC_DECODER) ---
if USE_CTC_DECODER:
    ort_session_CTC_Decoder = create_session(onnx_model_CTC_Decoder, **packed_settings)
    binding_CTC_Decoder     = ort_session_CTC_Decoder.io_binding()
    in_name_CTC_Decoder     = get_in_names(ort_session_CTC_Decoder)[0]
    out_name_CTC_Decoder    = get_out_names(ort_session_CTC_Decoder)
    ctc_tokenizer           = CTCTokenizer(ctc_tokenizer_path)

# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Rotary + Mask (Prefill) ---
ort_session_Rotary_Mask_Prefill = create_session(onnx_model_Rotary_Mask_Prefill, **packed_settings)
binding_Rotary_Mask_Prefill     = ort_session_Rotary_Mask_Prefill.io_binding()
in_name_Rotary_Mask_Prefill     = get_in_names(ort_session_Rotary_Mask_Prefill)
out_name_Rotary_Mask_Prefill    = get_out_names(ort_session_Rotary_Mask_Prefill)

# --- Rotary + Mask (Decode) ---
ort_session_Rotary_Mask_Decode = create_session(onnx_model_Rotary_Mask_Decode, **packed_settings)
binding_Rotary_Mask_Decode     = ort_session_Rotary_Mask_Decode.io_binding()
in_name_Rotary_Mask_Decode     = get_in_names(ort_session_Rotary_Mask_Decode)[0]
out_name_Rotary_Mask_Decode    = get_out_names(ort_session_Rotary_Mask_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)
num_keys_values        = amount_of_outputs_Main - 1

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4

in_name_Main_parts   = in_name_Main[:num_keys_values]
out_name_Main_kv     = out_name_Main[:num_keys_values]
out_name_Main_logits = out_name_Main[num_keys_values]

# Dtype introspection
kv_dtype_str             = ort_session_Main._inputs_meta[0].type
hidden_states_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values].type else np.float32
mask_dtype_Main          = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values_plus_3].type else np.float32
vocab_size               = ort_session_Main._outputs_meta[num_keys_values].shape[1]

_logits_out_meta  = ort_session_Main._outputs_meta[num_keys_values]
_logits_out_dtype = np.float16 if 'float16' in _logits_out_meta.type else np.float32


# ══════════════════════════════════════════════════════════════════════════════
# KV CACHE SETUP (float16 for FunASR-Nano)
# ══════════════════════════════════════════════════════════════════════════════
_meta = ort_session_Main._inputs_meta

kv_dtype_Main = np.float16 if 'float16' in kv_dtype_str else np.float32
num_layers    = num_keys_values // 2

past_keys_Main   = create_ort_with_shape((1, _meta[0].shape[1],          1, _meta[0].shape[3],          0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, _meta[num_layers].shape[1], 1, 0, _meta[num_layers].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)

STOP_TOKEN_SET = set(STOP_TOKEN)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & SHARED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

_rotary_meta = ort_session_Rotary_Mask_Decode._outputs_meta

# --- Scalar OrtValues ---
init_history_len = create_ort_with_data([0],         np.int64, device_type, DEVICE_ID)
topK             = create_ort_with_data([TOP_K],     np.int64, device_type, DEVICE_ID)
beam_size        = create_ort_with_data([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

# --- Decode-phase static buffers (bind once, reused every step) ---
attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1),                                 mask_dtype_Main,          device_type, DEVICE_ID)
rotary_cos_buf     = create_ort_with_shape((1, 1, 1, 1, _rotary_meta[0].shape[4]),          hidden_states_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf     = create_ort_with_shape((1, 1, 1, 1, _rotary_meta[1].shape[4]),          hidden_states_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf  = create_ort_with_shape((BEAM_SIZE, 1, _meta[num_keys_values].shape[2]), hidden_states_dtype_Main, device_type, DEVICE_ID)
save_id_buf        = create_ort_with_shape((BEAM_SIZE, 0),                                  np.int32,                 device_type, DEVICE_ID)

# --- Logits & token-index buffers ---
prefill_logits_buf = create_ort_with_shape((1, vocab_size),         _logits_out_dtype, device_type, DEVICE_ID)
decode_logits_buf  = create_ort_with_shape((BEAM_SIZE, vocab_size), _logits_out_dtype, device_type, DEVICE_ID)
max_idx_buf        = create_ort_with_shape((1, 1),                  np.int32,          device_type, DEVICE_ID)

generate_limit_base = MAX_SEQ_LEN - 10


# ══════════════════════════════════════════════════════════════════════════════
# DECODE HEAD SESSIONS (Beam Search OR Greedy/Argmax)
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")

    # --- First Beam ---
    ort_session_First_Beam    = create_session(onnx_model_First_Beam, **packed_settings)
    binding_First_Beam        = ort_session_First_Beam.io_binding()
    in_name_First_Beam        = get_in_names(ort_session_First_Beam)
    out_name_First_Beam       = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts  = in_name_First_Beam[:num_keys_values_plus_1]
    out_name_First_Beam_parts = out_name_First_Beam[:num_keys_values_plus_1]

    # --- Second Beam ---
    ort_session_Second_Beam    = create_session(onnx_model_Second_Beam, **packed_settings)
    binding_Second_Beam        = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam        = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam       = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts  = in_name_Second_Beam[:num_keys_values_plus_1]
    out_name_Second_Beam_parts = out_name_Second_Beam[:num_keys_values_plus_1]

    # --- Beam-specific buffers ---
    beam_ids_buf   = create_ort_with_shape((BEAM_SIZE, 1), np.int32,                 device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_states_dtype_Main, device_type, DEVICE_ID)

    # --- Static beam bindings ---
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_1], save_id_buf)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_2], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_3], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_4], topK)

else:
    # --- Greedy ---
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = get_in_names(ort_session_Greedy)
    out_name_Greedy    = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    # --- Argmax ---
    ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
    out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# PENALTY SESSION (optional — replaces Reset_Penalty_Beam/Greedy)
# ══════════════════════════════════════════════════════════════════════════════
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

    binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-EMBED TASK PROMPTS
# ══════════════════════════════════════════════════════════════════════════════
init_all_outputs_Embed = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
    bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
    run(ort_session_Embed, binding_Embed)
    init_all_outputs_Embed.append(onnxruntime.OrtValue.ortvalue_from_numpy(binding_Embed.get_outputs()[0].numpy(), device_type, DEVICE_ID))


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════
for prompt_embed, test in zip(init_all_outputs_Embed, test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    audio_full_len = len(audio)
    audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)                  # full clip -> target dtype, RMS once
    INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_full_len) if isinstance(shape_value_in_Encoder, str) else shape_value_in_Encoder
    stride_step = INPUT_AUDIO_LENGTH if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    if audio_full_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_full_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        pad_amount = ((num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH) - audio_full_len
        audio = np.concatenate((audio, np.zeros([1, 1, pad_amount], dtype=audio.dtype)), axis=-1)
    elif audio_full_len < INPUT_AUDIO_LENGTH:
        audio = np.concatenate((audio, np.zeros([1, 1, INPUT_AUDIO_LENGTH - audio_full_len], dtype=audio.dtype)), axis=-1)
    aligned_len = audio.shape[-1]
    asr_result = ""
    ctc_result = ""
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    rtf_time = time.time()

    while slice_end <= aligned_len:
        ort_audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start:slice_end], device_type, DEVICE_ID)

        # ══════════════════════════════════════════════════════════════
        # ENCODER
        # ══════════════════════════════════════════════════════════════
        binding_Encoder.bind_ortvalue_input(in_name_Encoder[0], ort_audio)
        binding_Encoder.bind_ortvalue_input(in_name_Encoder[1], prompt_embed)
        bind_ort_out(binding_Encoder, out_name_Encoder, _ort_device_type)
        run(ort_session_Encoder, binding_Encoder)
        all_outputs_Encoder = binding_Encoder.get_outputs()
        hidden_states = all_outputs_Encoder[0]   # concat_embed
        ids_len       = all_outputs_Encoder[1]   # ids_len
        if USE_CTC_DECODER:
            # Feed enc_normed (encoder output #2) to the standalone CTC decoder graph.
            binding_CTC_Decoder.bind_ortvalue_input(in_name_CTC_Decoder, all_outputs_Encoder[2])
            bind_ort_out(binding_CTC_Decoder, out_name_CTC_Decoder, _ort_device_type)
            run(ort_session_CTC_Decoder, binding_CTC_Decoder)
            ctc_token_ids = binding_CTC_Decoder.get_outputs()[0].numpy()   # 1-D int32: greedy-collapsed CTC tokens
            if ctc_token_ids.size > 0:
                ctc_result += ctc_tokenizer.decode(ctc_token_ids.tolist()).replace("<|nospeech|>", "")

        # ══════════════════════════════════════════════════════════════
        # PREFILL SETUP
        # ══════════════════════════════════════════════════════════════

        # --- Rotary + Mask (Prefill) ---
        binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[0], ids_len)
        binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[1], init_history_len)
        bind_ort_out(binding_Rotary_Mask_Prefill, out_name_Rotary_Mask_Prefill, _ort_device_type)
        run(ort_session_Rotary_Mask_Prefill, binding_Rotary_Mask_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Mask_Prefill.get_outputs()

        # --- Pre-bind Decode Rotary (static output buffers) ---
        binding_Rotary_Mask_Decode.bind_ortvalue_input(in_name_Rotary_Mask_Decode, kv_seq_len)
        binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[0], rotary_cos_buf)
        binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[1], rotary_sin_buf)
        binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[2], kv_seq_len)

        # --- Bind Main: non-KV inputs (prefill) ---
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        hidden_states)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask)

        # --- Bind Main: empty KV cache ---
        i = 0
        for _ in range(num_layers):
            binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
            i += 1
        for _ in range(num_layers):
            binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
            i += 1

        # --- Bind Main: outputs (prefill) ---
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
        binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

        # --- Bind Penalty to prefill logits ---
        if USE_PENALTY:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
            binding_Penalty.bind_ortvalue_output(out_name_Penalty,  prefill_logits_buf)

        # --- Bind decode head to prefill logits ---
        if USE_BEAM_SEARCH:
            binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values], prefill_logits_buf)
        elif USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0],   prefill_logits_buf)
            binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax,   prefill_logits_buf)
            binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)

        # --- Pre-bind Embed for decode phase ---
        binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

        # --- Reset greedy save_id for new window ---
        if not USE_BEAM_SEARCH and USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

        is_prefill_step = True
        num_decode = 0
        generate_limit = generate_limit_base - ids_len.numpy().flat[0]
        start_time = time.time()

        # ══════════════════════════════════════════════════════════════
        # DECODE LOOP
        # ══════════════════════════════════════════════════════════════
        while num_decode < generate_limit:

            # ── 1. Run Main Model ────────────────────────────────────
            run(ort_session_Main, binding_Main)
            outputs_Main = binding_Main.get_outputs()

            # ── 2. Apply Repetition Penalty ──────────────────────────
            if USE_PENALTY and num_decode >= PENALTY_RANGE:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
                run(ort_session_Penalty, binding_Penalty)

            # ── 3. Token Selection ───────────────────────────────────
            if USE_BEAM_SEARCH:
                # ── 3a. Beam Search ──────────────────────────────────
                if is_prefill_step:
                    bind_ort_in(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
                    bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_1], beam_score_buf)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_2], beam_ids_buf)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_3], max_idx_buf)
                    run(ort_session_First_Beam, binding_First_Beam)
                    outputs_Beam = binding_First_Beam.get_outputs()
                else:
                    bind_ort_in(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
                    bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
                    if num_decode < 2:
                        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_2], beam_score_buf)
                        binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_1], beam_score_buf)
                        binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_2], beam_ids_buf)
                        binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_3], max_idx_buf)
                    run(ort_session_Second_Beam, binding_Second_Beam)
                    outputs_Beam = binding_Second_Beam.get_outputs()

                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN_SET:
                    break

                save_id = outputs_Beam[num_keys_values]
                bind_ort_in(binding_Main, in_name_Main_parts, outputs_Beam)
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_1], save_id)

            else:
                # ── 3b. Greedy / Argmax ──────────────────────────────
                if USE_PENALTY:
                    binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
                    run(ort_session_Greedy, binding_Greedy)
                    save_id = binding_Greedy.get_outputs()[1]
                else:
                    run(ort_session_Argmax, binding_Argmax)

                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN_SET:
                    break

                if USE_PENALTY:
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
                else:
                    save_id_numpy[num_decode] = max_logits_idx

                bind_ort_in(binding_Main, in_name_Main_parts, outputs_Main)

            # ── 4. Re-bind Main KV outputs (fresh alloc each step) ───
            bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

            # ── 5. Transition: prefill → decode (once per window) ────
            if is_prefill_step:
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        hidden_states_buf)
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos_buf)
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin_buf)
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask_buf)
                binding_Main.bind_ortvalue_output(out_name_Main_logits,                decode_logits_buf)

                binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

                if USE_PENALTY:
                    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                    binding_Penalty.bind_ortvalue_output(out_name_Penalty,  decode_logits_buf)

                if USE_BEAM_SEARCH:
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values], decode_logits_buf)
                    binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
                elif USE_PENALTY:
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
                else:
                    binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

                is_prefill_step = False

            # ── 6. Prepare next step: Embed + Rotary ─────────────────
            run(ort_session_Embed, binding_Embed)
            run(ort_session_Rotary_Mask_Decode, binding_Rotary_Mask_Decode)
            num_decode += 1

        # ── End of window ────────────────────────────────────────────
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        if num_decode > 0:
            if USE_BEAM_SEARCH or USE_PENALTY:
                asr_result += tokenizer.decode(save_id.numpy()[0, :num_decode], skip_special_tokens=True)
            else:
                asr_result += tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")

    print("LLM: " + asr_result, end="", flush=True)
    if USE_CTC_DECODER:
        print(f"\n\nCTC: {ctc_result}", end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_full_len / SAMPLE_RATE)):.3f}")
    print("----------------------------------------------------------------------------------------------------------")
