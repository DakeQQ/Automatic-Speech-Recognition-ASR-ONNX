from pathlib import Path
import argparse
import sys
import time
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from sentencepiece import SentencePieceProcessor


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths
from ORT_IO import (
    array_for,
    filled_for,
    is_dynamic_dim,
    load_supported_languages,
    numpy_dtype,
    resolve_supported_language,
)

def _parse_args():
    parser = argparse.ArgumentParser(description="Run SenseVoice ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "SenseVoice_Optimized", help="Folder containing ONNX graphs, for example SenseVoice_Optimized or SenseVoice_ONNX.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Optional SentencePiece model; defaults to the bundled model in the ONNX folder.")
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
onnx_model_Metadata = str(onnx_folder / "ASR_Metadata.onnx")
onnx_model_A = str(onnx_folder / "SenseVoiceSmall.onnx")
TOKENIZER_PATH = (
    _ARGS.tokenizer_path.expanduser().resolve()
    if _ARGS.tokenizer_path is not None
    else onnx_folder / "chn_jpn_yue_eng_ko_spectok.bpe.model"
)


# ============================== User configuration ==============================
# IMPORTANT: CLI options are intentionally limited to model/tokenizer paths.
# Edit this section for demo, language, audio, and ONNX Runtime behavior.
test_audio = model_audio_paths("sensevoice")[0]  # The test audio path.
ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
# The audio input dtype is auto-detected from the model's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO = False             # Apply RMS loudness normalisation before feeding the model. SenseVoice normalises the input by default.
TARGET_LANGUAGE = "en"                  # Code/alias from artifact metadata: auto, zh, en, yue, ja, ko, or nospeech.
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.


# The SentencePiece tokenizer model is bundled inside the ONNX folder by the export / optimize step, so inference is stand-alone.
tokenizer = SentencePieceProcessor()
tokenizer.Load(str(TOKENIZER_PATH))


def prepare_audio_input(
    audio_int16: np.ndarray,
    input_audio_dtype: str,
    *,
    audio_pcm_scale: int,
    target_rms: float = 4096.0,
) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's audio input
    # dtype exactly once. `input_audio_dtype` is derived from the ONNX model's audio input tensor.
    # The metadata divisor defines the float graph's immutable PCM convention.
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
    audio *= np.float32(1.0 / audio_pcm_scale)
    if input_audio_dtype == "F16":
        return audio.astype(np.float16)   # NOTE: int16-range in f16 is lossy (~±16 ULP near 32768)
    return audio                          # F32: int16-range values as float32 (kaldi keeps this range)
  

# ============================== Runtime settings ==============================
ORT_LOG     = False             # Enable ONNX Runtime logging for debugging. Set False for best performance.
ORT_FP16    = False             # Set True if the loaded ONNX model was converted to FP16 (CPUs need ARM64-v8.2a or newer).
MAX_THREADS = 0                 # Parallel CPU threads for inter/intra-op. Set 0 for auto.
DEVICE_ID   = 0                 # Accelerator device index; default zero.


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 0 if ORT_LOG else 4
session_opts.log_verbosity_level = 4
session_opts.inter_op_num_threads = MAX_THREADS       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
for _cfg_key, _cfg_value in {
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
        "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else ""
    ),
}.items():
    session_opts.add_session_config_entry(_cfg_key, _cfg_value)


# Per-provider options plus the matching OrtDevice handle used for zero-copy IOBinding.
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
disabled_optimizers = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None
)

run_options = onnxruntime.RunOptions()
run_options.log_severity_level = 0 if ORT_LOG else 4
run_options.log_verbosity_level = 4
run_options.add_run_config_entry("disable_synchronize_execution_providers", "0")


ort_session_Metadata = onnxruntime.InferenceSession(
    onnx_model_Metadata,
    sess_options=session_opts,
    providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    provider_options=provider_options,
    disabled_optimizers=disabled_optimizers,
)
ort_session_A = onnxruntime.InferenceSession(
    onnx_model_A,
    sess_options=session_opts,
    providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    provider_options=provider_options,
    disabled_optimizers=disabled_optimizers,
)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
audio_input_meta = in_name_A[0]
language_input_meta = in_name_A[1]
shape_value_in = audio_input_meta.shape[-1]
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name
io_binding_A = ort_session_A.io_binding()

# The audio input dtype is taken straight from the model's audio input tensor in the ONNX model,
# so it always matches how the model was exported (kaldi fbank keeps the int16 numeric range;
# "float16"/"float" carry int16-range values with no ÷32768).
audio_np_dtype = numpy_dtype(audio_input_meta)
input_audio_dtype = (
    "INT16" if audio_np_dtype == np.dtype(np.int16) else
    "F16" if audio_np_dtype == np.dtype(np.float16) else "F32"
)

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

SAMPLE_RATE = int(_model_meta["sample_rate"])
AUDIO_PCM_SCALE = int(_model_meta["audio_pcm_scale"])

SUPPORTED_LANGUAGES = load_supported_languages(_model_meta)
LANGUAGE, LANGUAGE_ENTRY = resolve_supported_language(
    SUPPORTED_LANGUAGES,
    TARGET_LANGUAGE,
)
LANGUAGE_SELECTOR_INDEX = LANGUAGE_ENTRY.get("selector_index")
print(f"\nModel metadata: sample_rate={SAMPLE_RATE}, "
    f"language={LANGUAGE}, selector_index={LANGUAGE_SELECTOR_INDEX}.")


# Load the input audio
print(f"\nTest Input Audio: {test_audio}")
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio_len = len(audio)
audio = prepare_audio_input(
    audio.reshape(1, 1, -1),
    input_audio_dtype,
    audio_pcm_scale=AUDIO_PCM_SCALE,
)
if is_dynamic_dim(shape_value_in):
    input_audio_length = audio_len
else:
    input_audio_length = int(shape_value_in)
if SLIDING_WINDOW <= 0:
    stride_step = input_audio_length
else:
    stride_step = SLIDING_WINDOW
if audio_len > input_audio_length:
    num_windows = int(np.ceil((audio_len - input_audio_length) / stride_step)) + 1
    total_length_needed = (num_windows - 1) * stride_step + input_audio_length
    pad_amount = total_length_needed - audio_len
    zeros_pad = np.zeros((1, 1, pad_amount), dtype=audio.dtype)
    audio = np.concatenate((audio, zeros_pad), axis=-1)
elif audio_len < input_audio_length:
    zeros_pad = np.zeros((1, 1, input_audio_length - audio_len), dtype=audio.dtype)
    audio = np.concatenate((audio, zeros_pad), axis=-1)
aligned_len = audio.shape[-1]


# Start to run SenseVoice
slice_start = 0
slice_end = input_audio_length
language_idx = filled_for(
    language_input_meta,
    LANGUAGE_SELECTOR_INDEX,
    axes={0: 1},
)
print("\nRunning the SenseVoice by ONNX Runtime.")
text = ""

# CPUExecutionProvider can consume each NumPy window in place. Accelerators keep one device input
# buffer and update it per window; the language id remains device-resident for the whole clip.
if device_type == "cpu":
    audio_buffer = None
    io_binding_A.bind_cpu_input(in_name_A1, language_idx)
else:
    audio_buffer = onnxruntime.OrtValue.ortvalue_from_numpy(
        filled_for(audio_input_meta, axes={0: 1, 1: 1, 2: input_audio_length}),
        device_type,
        DEVICE_ID,
    )
    language_buffer = onnxruntime.OrtValue.ortvalue_from_numpy(language_idx, device_type, DEVICE_ID)
    io_binding_A.bind_ortvalue_input(in_name_A0, audio_buffer)
    io_binding_A.bind_ortvalue_input(in_name_A1, language_buffer)

start_time = time.time()
while slice_end <= aligned_len:
    audio_window = array_for(
        audio_input_meta,
        audio[:, :, slice_start: slice_end],
        axes={0: 1, 1: 1, 2: input_audio_length},
    )
    if audio_buffer is None:
        io_binding_A.bind_cpu_input(in_name_A0, audio_window)
    else:
        audio_buffer.update_inplace(audio_window)
    # Device-auto outputs retain their previous allocation. Rebind before every run because the
    # compact CTC token count is data-dependent and can change between sliding windows.
    io_binding_A._iobinding.bind_output(out_name_A0, _ort_device_obj)
    ort_session_A.run_with_iobinding(io_binding_A, run_options=run_options)
    token_ids = io_binding_A.get_outputs()[0].numpy()
    text += tokenizer.decode([token_ids.tolist()])[0]
    slice_start += stride_step
    slice_end = slice_start + input_audio_length
end_time = time.time()
real_time_factor = (end_time - start_time) / (audio_len / SAMPLE_RATE)
print(f"\nASR Result:\n{text}\n\nRTF: {real_time_factor:.4f}\n")
print("----------------------------------------------------------------------------------------------------------")
