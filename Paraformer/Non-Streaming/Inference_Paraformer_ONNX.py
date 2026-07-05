import json
from pathlib import Path
import argparse
import sys
import time
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths


tokens_path = "/home/DakeQQ/Downloads/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json"   # The Paraformer download path. Large Verion: [speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/tokens.json] / Small Verion: [speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1/tokens.json]
def _parse_args():
    parser = argparse.ArgumentParser(description="Run Paraformer ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "Paraformer_Optimized", help="Folder containing ONNX graphs, for example Paraformer_Optimized or Paraformer_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
onnx_model_Metadata = str(onnx_folder / "Paraformer_Metadata.onnx")
onnx_model_A = str(onnx_folder / "Paraformer.onnx")
test_audio = model_audio_paths("paraformer")[0]                                                                                                           # The test audio path.
    

# ============================== Runtime settings ==============================
ORT_Accelerate_Providers = []           # Accelerate devices, e.g. ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']; else keep empty.
ORT_LOG              = False             # Enable ONNX Runtime logging for debugging. Set False for best performance.
ORT_FP16             = False             # Set True if the loaded ONNX model was converted to FP16 (CPUs need ARM64-v8.2a or newer).
MAX_THREADS          = 0                 # Parallel CPU threads for inter/intra-op. Set 0 for auto.
DEVICE_ID            = 0                 # Accelerator device index; default zero.
SLIDING_WINDOW       = 0                 # Sliding window step for test audio reading; use 0 to disable.
# The audio input dtype is auto-detected from the model's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO  = False             # Apply RMS loudness normalisation before feeding the model (RMS-normalize the demo audio).
DECODE_MODE          = "zh"              # Token decoding mode ('zh' or 'en').


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


def decode_tokens(tokens, mode):
    if mode == "en":
        return " ".join(tokens).replace("</s>", "").replace("@@ ", "").strip()
    return "".join(tokens).replace("</s>", "").strip()


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


with open(tokens_path, 'r', encoding='UTF-8') as json_file:
    tokenizer = np.array(json.load(json_file), dtype=np.str_)


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
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
io_binding_A = ort_session_A.io_binding()

# The audio input dtype is taken straight from the model's audio input tensor in the ONNX model,
# so it always matches how the model was exported (kaldi fbank keeps the int16 numeric range;
# "float16"/"float" carry int16-range values with no ÷32768).
_audio_input_type = ort_session_A._inputs_meta[0].type
input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Paraformer.py to stamp the model metadata."
        )
    return int(value)


SAMPLE_RATE = _meta_int("sample_rate")
INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
print(f"\nModel metadata: {len(_model_meta)} keys "
      f"(input_audio_length={INPUT_AUDIO_LENGTH}, sample_rate={SAMPLE_RATE}).")


# Load the input audio
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
audio_len = len(audio)
audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(INPUT_AUDIO_LENGTH, audio_len)  # The LFR index buffer is exported for this maximum length.
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
    zeros_pad = np.zeros((1, 1, pad_amount), dtype=audio.dtype)
    audio = np.concatenate((audio, zeros_pad), axis=-1)
elif audio_len < INPUT_AUDIO_LENGTH:
    zeros_pad = np.zeros((1, 1, INPUT_AUDIO_LENGTH - audio_len), dtype=audio.dtype)
    audio = np.concatenate((audio, zeros_pad), axis=-1)
aligned_len = audio.shape[-1]

# Start to run Paraformer
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
text = np.array([], dtype=np.str_)

# Reuse one pre-allocated input buffer for every sliding window (update_inplace keeps the bound
# input zero-copy) and let ORT allocate the variable-length token output on the target device.
audio_buffer = onnxruntime.OrtValue.ortvalue_from_numpy(
    np.zeros((1, 1, INPUT_AUDIO_LENGTH), dtype=audio.dtype), device_type, DEVICE_ID
)
io_binding_A.bind_ortvalue_input(in_name_A0, audio_buffer)
io_binding_A._iobinding.bind_output(out_name_A0, _ort_device_obj)
io_binding_A._iobinding.bind_output(out_name_A1, _ort_device_obj)

start_time = time.time()
while slice_end <= aligned_len:
    audio_buffer.update_inplace(np.ascontiguousarray(audio[:, :, slice_start: slice_end]))
    ort_session_A.run_with_iobinding(io_binding_A, run_options=run_options)
    token_ids = io_binding_A.get_outputs()[0].numpy()
    text = np.concatenate((text, tokenizer[token_ids[0]]))
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
count_time = time.time() - start_time
rtf = count_time / max(audio_len / SAMPLE_RATE, 1e-6)
text = decode_tokens(text.tolist(), DECODE_MODE)
print(f"\nASR Result:\n{text}\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio)\n")
print("----------------------------------------------------------------------------------------------------------")

