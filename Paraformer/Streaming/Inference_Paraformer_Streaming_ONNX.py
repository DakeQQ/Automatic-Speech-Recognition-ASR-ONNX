import json
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths


tokens_path = "/home/DakeQQ/Downloads/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/tokens.json"  # The Paraformer tokens download path.


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Paraformer streaming ONNX inference.")
    parser.add_argument(
        "--onnx-folder", "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=_SCRIPT_DIR / "Paraformer_Optimized",
        help="Folder containing ONNX graphs, for example Paraformer_Optimized or Paraformer_ONNX.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder        = _ARGS.onnx_folder.expanduser().resolve()
onnx_model_Metadata = str(onnx_folder / "Paraformer_Streaming_Metadata.onnx")
onnx_model_Encoder = str(onnx_folder / "Paraformer_Streaming_Encoder.onnx")      # The exported onnx model path.
onnx_model_Decoder = str(onnx_folder / "Paraformer_Streaming_Decoder.onnx")      # The exported onnx model path.
test_audio         = model_audio_paths("paraformer")[0]                                  # The test audio path.

# Only support CPU and f32, q8f32, currently.
ORT_Accelerate_Providers = []       # If you have accelerate devices for: ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                    # else keep empty.
MAX_THREADS = 0                     # Max CPU parallel threads.
DEVICE_ID = 0                       # The GPU id, default to 0.

ORT_LOG  = False                    # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO = False         # Apply RMS loudness normalisation before feeding the model. Default keeps the raw int16 waveform amplitude.


def prepare_audio_input(audio_int16: np.ndarray, input_audio_dtype: str, target_rms: float = 8192.0) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's audio input
    # dtype exactly once. `input_audio_dtype` is derived from the ONNX model's audio input tensor.
    # The Kaldi fbank front-end consumes the int16 numeric range directly, so the float variants
    # carry int16-range values (there is NO ÷32768 here). For streaming the whole clip is converted
    # once (RMS computed over the full clip) before windows are sliced.
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


# ============================================================================
# ONNX Runtime settings (session options + run options + provider configs)
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
    opts.enable_cpu_mem_arena = True
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
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else ""
        ),
    }
    for key, value in cfgs.items():
        opts.add_session_config_entry(key, value)
    return opts


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',
            'precision': 'ACCURACY',
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,
            'disable_dynamic_shapes': False
        }
    ]
    device_type = "cpu"
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
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
            'use_ep_level_unified_stream': '0'
        }
    ]
    device_type = "cuda"
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',
            'device_filter': 'gpu',
            'disable_metacommands': 'false',
            'enable_graph_capture': 'false',
            'enable_graph_serialization': 'false'
        }
    ]
    device_type = "dml"
    _ort_device_type = C.OrtDevice.dml()
else:
    # Please config by yourself for others providers.
    provider_options = None
    device_type = "cpu"
    _ort_device_type = C.OrtDevice.cpu()


_ort_device_obj = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
session_opts = _build_session_opts_ort()
run_options = _build_run_options(silent=not ORT_LOG)
disabled_opts = ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None


# ============================================================================
# ONNX Runtime IOBinding helpers (shared, zero-copy buffers)
# ============================================================================
_packed = {
    'sess_options': session_opts,
    'providers': ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    'provider_options': provider_options,
    'disabled_optimizers': disabled_opts,
}


def _make_session(path):
    return onnxruntime.InferenceSession(path, **_packed)


def _ort_zeros(shape, dtype):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device_type, DEVICE_ID)


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


with open(tokens_path, 'r', encoding='UTF-8') as json_file:
    tokenizer = np.array(json.load(json_file), dtype=np.str_)


ort_session_Metadata = _make_session(onnx_model_Metadata)
ort_session_Encoder = _make_session(onnx_model_Encoder)
ort_session_Decoder = _make_session(onnx_model_Decoder)
shape_value_in = ort_session_Encoder._inputs_meta[-1].shape[-1]
in_name_Encoder = _in_names(ort_session_Encoder)
out_name_Encoder = _out_names(ort_session_Encoder)
in_name_Decoder = _in_names(ort_session_Decoder)
out_name_Decoder = _out_names(ort_session_Decoder)
binding_Encoder = ort_session_Encoder.io_binding()
binding_Decoder = ort_session_Decoder.io_binding()

# The audio input dtype is taken straight from the encoder's audio input tensor in the ONNX model
# (here the audio is the LAST encoder input), so it always matches how the model was exported (kaldi
# fbank keeps the int16 numeric range; "float16"/"float" carry int16-range values with no ÷32768).
_audio_input_type = ort_session_Encoder._inputs_meta[-1].type
input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Paraformer_Streaming.py to stamp the model metadata."
        )
    return int(value)


SAMPLE_RATE = _meta_int("sample_rate")
METADATA_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
FSMN_DE_PAD = _meta_int("fsmn_de_pad")
print(f"\nModel metadata: {len(_model_meta)} keys "
    f"(sample_rate={SAMPLE_RATE}, input_audio_length={METADATA_INPUT_AUDIO_LENGTH}, "
    f"fsmn_de_pad={FSMN_DE_PAD}).")


# Load the input audio
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)  # Raw int16 PCM == FunASR waveform * (1 << 15); prepare_audio_input owns the optional RMS + dtype conversion.
audio_len = len(audio)
audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)         # full clip -> target dtype, RMS once
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(METADATA_INPUT_AUDIO_LENGTH, audio_len)
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


print(f"\nUsable Providers: {ort_session_Encoder.get_providers()[0]}\n")


# Initialize
amount_of_outputs_Encoder = len(out_name_Encoder)
amount_of_outputs_Decoder = len(out_name_Decoder)
amount_of_outputs_Decoder -= 2   # The last 2 are max_logit_ids and num_id; pass them to the tokenizer.
num_layer_de = amount_of_outputs_Decoder // 3
num_layer_en = (amount_of_outputs_Encoder - 7) // 2
amount_of_outputs_Encoder -= 3   # The last 3 are for the Decoder.

in_en_value_start = num_layer_en
in_previous_mel_features_start = in_en_value_start + num_layer_en
in_cif_hidden_start = in_previous_mel_features_start + 1
in_de_key_start = num_layer_de
in_de_value_start = in_de_key_start + num_layer_de
in_de_value_end = in_de_value_start + num_layer_de

# Shared ORT buffers: allocated once, reused across every streaming window.
enc_in_meta = ort_session_Encoder._inputs_meta
dec_in_meta = ort_session_Decoder._inputs_meta
in_en_key_Encoder = _ort_zeros((enc_in_meta[0].shape[0], enc_in_meta[0].shape[1], 0), _np_dtype(enc_in_meta[0]))
in_en_value_Encoder = _ort_zeros((enc_in_meta[in_en_value_start].shape[0], 0, enc_in_meta[in_en_value_start].shape[2]), _np_dtype(enc_in_meta[in_en_value_start]))
in_previous_mel_features = _ort_zeros((enc_in_meta[in_previous_mel_features_start].shape[0], enc_in_meta[in_previous_mel_features_start].shape[1], enc_in_meta[in_previous_mel_features_start].shape[2]), _np_dtype(enc_in_meta[in_previous_mel_features_start]))
in_cif_hidden = _ort_zeros((enc_in_meta[in_cif_hidden_start].shape[0], enc_in_meta[in_cif_hidden_start].shape[1], enc_in_meta[in_cif_hidden_start].shape[2]), _np_dtype(enc_in_meta[in_cif_hidden_start]))
in_cif_alpha = _ort_zeros((1,), _np_dtype(enc_in_meta[-3]))
start_idx = _ort_zeros((1,), np.int64)
in_de_fsmn_Decoder = _ort_zeros((dec_in_meta[0].shape[0], dec_in_meta[0].shape[1], FSMN_DE_PAD), _np_dtype(dec_in_meta[0]))
in_de_key_Decoder = _ort_zeros((dec_in_meta[in_de_key_start].shape[0], dec_in_meta[in_de_key_start].shape[1], 0), _np_dtype(dec_in_meta[in_de_key_start]))
in_de_value_Decoder = _ort_zeros((dec_in_meta[in_de_value_start].shape[0], 0, dec_in_meta[in_de_value_start].shape[2]), _np_dtype(dec_in_meta[in_de_value_start]))
audio_buffer = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), audio.dtype)

# Bind the persistent encoder / decoder inputs once; cache outputs ping-pong on device.
for i in range(num_layer_en):
    binding_Encoder.bind_ortvalue_input(in_name_Encoder[i], in_en_key_Encoder)
for i in range(in_en_value_start, in_previous_mel_features_start):
    binding_Encoder.bind_ortvalue_input(in_name_Encoder[i], in_en_value_Encoder)
binding_Encoder.bind_ortvalue_input(in_name_Encoder[in_previous_mel_features_start], in_previous_mel_features)
binding_Encoder.bind_ortvalue_input(in_name_Encoder[in_cif_hidden_start], in_cif_hidden)
binding_Encoder.bind_ortvalue_input(in_name_Encoder[-3], in_cif_alpha)
binding_Encoder.bind_ortvalue_input(in_name_Encoder[-2], start_idx)
binding_Encoder.bind_ortvalue_input(in_name_Encoder[-1], audio_buffer)
_bind_device_outputs(binding_Encoder, out_name_Encoder)

for i in range(num_layer_de):
    binding_Decoder.bind_ortvalue_input(in_name_Decoder[i], in_de_fsmn_Decoder)
for i in range(in_de_key_start, in_de_value_start):
    binding_Decoder.bind_ortvalue_input(in_name_Decoder[i], in_de_key_Decoder)
for i in range(in_de_value_start, in_de_value_end):
    binding_Decoder.bind_ortvalue_input(in_name_Decoder[i], in_de_value_Decoder)


# Start to run Paraformer-Streaming
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
while True:
    audio_buffer.update_inplace(np.ascontiguousarray(audio[:, :, slice_start:slice_end]))
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    start_time = time.time()
    _run(ort_session_Encoder, binding_Encoder)
    all_outputs_Encoder = binding_Encoder.get_outputs()
    # Read the CIF fire count and hand the 3 tail outputs to the Decoder BEFORE the
    # encoder output slots get re-bound below (re-binding invalidates these OrtValues).
    cif_fired = all_outputs_Encoder[-1].numpy() != 0
    if cif_fired:
        binding_Decoder.bind_ortvalue_input(in_name_Decoder[-1], all_outputs_Encoder[-1])
        binding_Decoder.bind_ortvalue_input(in_name_Decoder[-2], all_outputs_Encoder[-2])
        binding_Decoder.bind_ortvalue_input(in_name_Decoder[-3], all_outputs_Encoder[-3])
        _bind_device_outputs(binding_Decoder, out_name_Decoder)
        _run(ort_session_Decoder, binding_Decoder)
        all_outputs_Decoder = binding_Decoder.get_outputs()
        end_time = time.time()
        max_logit_ids = all_outputs_Decoder[-2].numpy()[0]
        text = tokenizer[max_logit_ids].tolist()
        text = ''.join(text).replace("</s>", "")
        real_time_factor = (end_time - start_time) / (INPUT_AUDIO_LENGTH / SAMPLE_RATE)
        print(f"ASR: {text} / RTF: {real_time_factor:.4f}")
    if slice_end <= aligned_len:
        _bind_inputs(binding_Encoder, in_name_Encoder[:amount_of_outputs_Encoder], all_outputs_Encoder[:amount_of_outputs_Encoder])
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
    if cif_fired:
        if slice_end > aligned_len:
            break
        _bind_inputs(binding_Decoder, in_name_Decoder[:amount_of_outputs_Decoder], all_outputs_Decoder[:amount_of_outputs_Decoder])
    elif slice_end > aligned_len:
        break

