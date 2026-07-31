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
from ORT_IO import (
    array_for,
    filled_for,
    load_special_token_ids,
    load_supported_languages,
    metadata_by_name,
    numpy_dtype,
    resolve_supported_language,
    scalar_for,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Paraformer streaming ONNX inference.")
    parser.add_argument(
        "--onnx-folder", "--model-folder",
        dest="onnx_folder",
        type=Path,
        default=_SCRIPT_DIR / "Paraformer_Optimized",
        help="Folder containing ONNX graphs, for example Paraformer_Optimized or Paraformer_ONNX.",
    )
    parser.add_argument(
        "--vocab-path", "--tokenizer-path",
        dest="vocab_path",
        type=Path,
        default=None,
        help="Optional vocabulary file; defaults to Vocab_Paraformer.txt in the model folder.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder        = _ARGS.onnx_folder.expanduser().resolve()
onnx_model_Metadata = str(onnx_folder / "ASR_Metadata.onnx")
onnx_model_Encoder = str(onnx_folder / "Paraformer_Streaming_Encoder.onnx")      # The exported onnx model path.
onnx_model_Decoder = str(onnx_folder / "Paraformer_Streaming_Decoder.onnx")      # The exported onnx model path.
VOCAB_PATH = (
    _ARGS.vocab_path.expanduser().resolve()
    if _ARGS.vocab_path is not None
    else onnx_folder / "Vocab_Paraformer.txt"
)

# IMPORTANT: CLI options are intentionally limited to model/vocabulary paths.
# Edit this section for demo, audio, and ONNX Runtime behavior.
test_audio = model_audio_paths("paraformer")[0]  # The test audio path.
# The optimized default uses an F32 recurrent encoder and an F16 decoder with
# F32 state-interface Casts; both CPU and CUDA paths are validated.
ORT_Accelerate_Providers = ["CUDAExecutionProvider"]       # If you have accelerate devices for: ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                    # else keep empty.
MAX_THREADS = 0                     # Max CPU parallel threads.
DEVICE_ID = 0                       # The GPU id, default to 0.

ORT_LOG  = False                    # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO = False         # Apply RMS loudness normalisation before feeding the model. Default keeps the raw int16 waveform amplitude.


def prepare_audio_input(
    audio_int16: np.ndarray,
    input_audio_dtype: np.dtype,
    *,
    audio_pcm_scale: int,
    target_rms: float = 4096.0,
) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's audio input
    # dtype exactly once. `input_audio_dtype` is derived from the ONNX model's audio input NodeArg.
    # The metadata divisor defines the float graph's immutable PCM convention. For streaming the
    # whole clip is converted once (RMS computed over the full clip) before windows are sliced.
    input_audio_dtype = np.dtype(input_audio_dtype)
    if not USE_NORMALISE_AUDIO and input_audio_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio_int16, dtype=input_audio_dtype)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= (target_rms / (rms + 1e-7))
            np.clip(audio, -32768.0, 32767.0, out=audio)
    if input_audio_dtype != np.dtype(np.int16):
        audio *= np.float32(1.0 / audio_pcm_scale)
    return audio.astype(input_audio_dtype)


def decode_tokens(tokens, mode):
    if mode == "en":
        return " ".join(tokens).replace("@@ ", "").strip()
    return "".join(tokens).strip()


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


def _ort_from_numpy(value):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(value), device_type, DEVICE_ID
    )


def _bind_device_outputs(binding, names):
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def _cache_array(value_meta, history_axis, history_length):
    return filled_for(value_meta, axes={history_axis: history_length})


def _bind_transfers(binding, transfers, produced):
    for consumer_meta, producer_meta in transfers:
        binding.bind_ortvalue_input(
            consumer_meta.name,
            produced[producer_meta.name],
        )


# The vocab list is bundled inside the ONNX folder by the export / optimize step, so inference is stand-alone.
with open(VOCAB_PATH, 'r', encoding='UTF-8') as vocab_file:
    tokenizer = np.array([_line.rstrip('\n') for _line in vocab_file], dtype=np.str_)


ort_session_Metadata = _make_session(onnx_model_Metadata)
ort_session_Encoder = _make_session(onnx_model_Encoder)
ort_session_Decoder = _make_session(onnx_model_Decoder)
enc_input_meta = ort_session_Encoder.get_inputs()
enc_output_meta = ort_session_Encoder.get_outputs()
dec_input_meta = ort_session_Decoder.get_inputs()
dec_output_meta = ort_session_Decoder.get_outputs()
enc_inputs = metadata_by_name(enc_input_meta)
enc_outputs = metadata_by_name(enc_output_meta)
dec_inputs = metadata_by_name(dec_input_meta)
dec_outputs = metadata_by_name(dec_output_meta)
in_name_Encoder = [value.name for value in enc_input_meta]
out_name_Encoder = [value.name for value in enc_output_meta]
in_name_Decoder = [value.name for value in dec_input_meta]
out_name_Decoder = [value.name for value in dec_output_meta]
binding_Encoder = ort_session_Encoder.io_binding()
binding_Decoder = ort_session_Decoder.io_binding()

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}
SAMPLE_RATE = int(_model_meta["sample_rate"])
AUDIO_PCM_SCALE = int(_model_meta["audio_pcm_scale"])

SPECIAL_TOKEN_IDS = load_special_token_ids(_model_meta)
stop_value = SPECIAL_TOKEN_IDS["stop"]
STOP_TOKEN_IDS = stop_value if isinstance(stop_value, list) else [stop_value]

SUPPORTED_LANGUAGES = load_supported_languages(_model_meta)
_artifact_language = next(iter(SUPPORTED_LANGUAGES))
LANGUAGE, LANGUAGE_ENTRY = resolve_supported_language(
    SUPPORTED_LANGUAGES,
    _artifact_language,
)
DECODE_MODE = LANGUAGE_ENTRY.get("decode_mode")

NUM_LAYER_EN = len([name for name in in_name_Encoder if name.startswith("in_en_key_")])
NUM_LAYER_DE = len([name for name in in_name_Decoder if name.startswith("in_de_fsmn_")])
print(f"\nModel metadata: sample_rate={SAMPLE_RATE}, "
    f"language={LANGUAGE}, decode={DECODE_MODE}.")

encoder_key_inputs = [f"in_en_key_{i}" for i in range(NUM_LAYER_EN)]
encoder_value_inputs = [f"in_en_value_{i}" for i in range(NUM_LAYER_EN)]
encoder_key_outputs = [f"out_en_key_{i}" for i in range(NUM_LAYER_EN)]
encoder_value_outputs = [f"out_en_value_{i}" for i in range(NUM_LAYER_EN)]
decoder_fsmn_inputs = [f"in_de_fsmn_{i}" for i in range(NUM_LAYER_DE)]
decoder_key_inputs = [f"in_de_key_{i}" for i in range(NUM_LAYER_DE)]
decoder_value_inputs = [f"in_de_value_{i}" for i in range(NUM_LAYER_DE)]
decoder_fsmn_outputs = [f"out_de_fsmn_{i}" for i in range(NUM_LAYER_DE)]
decoder_key_outputs = [f"out_de_key_{i}" for i in range(NUM_LAYER_DE)]
decoder_value_outputs = [f"out_de_value_{i}" for i in range(NUM_LAYER_DE)]

encoder_feedback = []
for input_names, output_names in (
    (encoder_key_inputs, encoder_key_outputs),
    (encoder_value_inputs, encoder_value_outputs),
    (["in_previous_mel_features"], ["out_previous_mel_features"]),
    (["in_cif_hidden"], ["out_cif_hidden"]),
    (["in_cif_alphas"], ["out_cif_alphas"]),
    (["start_idx"], ["end_idx"]),
):
    for index in range(len(input_names)):
        input_meta = enc_inputs[input_names[index]]
        output_meta = enc_outputs[output_names[index]]
        encoder_feedback.append((input_meta, output_meta))

decoder_feedback = []
for input_names, output_names in (
    (decoder_fsmn_inputs, decoder_fsmn_outputs),
    (decoder_key_inputs, decoder_key_outputs),
    (decoder_value_inputs, decoder_value_outputs),
):
    for index in range(len(input_names)):
        input_meta = dec_inputs[input_names[index]]
        output_meta = dec_outputs[output_names[index]]
        decoder_feedback.append((input_meta, output_meta))

encoder_decoder_bridge = []
for name in ("encoder_out", "list_frame", "list_frame_len"):
    encoder_decoder_bridge.append((dec_inputs[name], enc_outputs[name]))

audio_meta = enc_inputs["audio"]
input_audio_dtype = numpy_dtype(audio_meta)
INPUT_AUDIO_LENGTH = int(audio_meta.shape[2])
FSMN_DE_PAD = int(dec_inputs[decoder_fsmn_inputs[0]].shape[2])


# Load the input audio
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)  # Raw int16 PCM == FunASR waveform * (1 << 15); prepare_audio_input owns the optional RMS + dtype conversion.
audio_len = len(audio)
audio = prepare_audio_input(
    audio.reshape(1, 1, -1),
    input_audio_dtype,
    audio_pcm_scale=AUDIO_PCM_SCALE,
)  # full clip -> target dtype, RMS once
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


# Shared ORT buffers: allocated once, reused across every streaming window.
encoder_input_buffers = {}
for name in encoder_key_inputs:
    encoder_input_buffers[name] = _ort_from_numpy(_cache_array(enc_inputs[name], 2, 0))
for name in encoder_value_inputs:
    encoder_input_buffers[name] = _ort_from_numpy(_cache_array(enc_inputs[name], 1, 0))
for name in ("in_previous_mel_features", "in_cif_hidden", "in_cif_alphas"):
    encoder_input_buffers[name] = _ort_from_numpy(filled_for(enc_inputs[name]))
encoder_input_buffers["start_idx"] = _ort_from_numpy(scalar_for(enc_inputs["start_idx"], 0))
encoder_input_buffers["audio"] = _ort_from_numpy(
    filled_for(audio_meta, axes={2: INPUT_AUDIO_LENGTH})
)

decoder_input_buffers = {}
for name in decoder_fsmn_inputs:
    decoder_input_buffers[name] = _ort_from_numpy(
        _cache_array(dec_inputs[name], 2, FSMN_DE_PAD)
    )
for name in decoder_key_inputs:
    decoder_input_buffers[name] = _ort_from_numpy(_cache_array(dec_inputs[name], 2, 0))
for name in decoder_value_inputs:
    decoder_input_buffers[name] = _ort_from_numpy(_cache_array(dec_inputs[name], 1, 0))

# Bind the persistent encoder / decoder inputs once; cache outputs ping-pong on device.
for name in in_name_Encoder:
    binding_Encoder.bind_ortvalue_input(name, encoder_input_buffers[name])
_bind_device_outputs(binding_Encoder, out_name_Encoder)

for name in decoder_fsmn_inputs + decoder_key_inputs + decoder_value_inputs:
    binding_Decoder.bind_ortvalue_input(name, decoder_input_buffers[name])


# Start to run Paraformer-Streaming
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
transcript_parts = []
while True:
    audio_window = array_for(
        audio_meta,
        audio[:, :, slice_start:slice_end],
        axes={2: INPUT_AUDIO_LENGTH},
    )
    encoder_input_buffers["audio"].update_inplace(audio_window)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    start_time = time.time()
    _run(ort_session_Encoder, binding_Encoder)
    outputs_Encoder = dict(zip(out_name_Encoder, binding_Encoder.get_outputs()))
    # Read the CIF fire count and hand the 3 tail outputs to the Decoder BEFORE the
    # encoder output slots get re-bound below (re-binding invalidates these OrtValues).
    cif_count = outputs_Encoder["list_frame_len"].numpy()
    cif_fired = bool(cif_count.reshape(-1)[0] != 0)
    if cif_fired:
        _bind_transfers(
            binding_Decoder, encoder_decoder_bridge, outputs_Encoder
        )
        _bind_device_outputs(binding_Decoder, out_name_Decoder)
        _run(ort_session_Decoder, binding_Decoder)
        outputs_Decoder = dict(zip(out_name_Decoder, binding_Decoder.get_outputs()))
        end_time = time.time()
        max_logit_ids = outputs_Decoder["max_logit_ids"].numpy().reshape(-1)
        max_logit_ids = max_logit_ids[
            ~np.isin(max_logit_ids, STOP_TOKEN_IDS)
        ]
        text = decode_tokens(tokenizer[max_logit_ids].tolist(), DECODE_MODE)
        transcript_parts.append(text)
        real_time_factor = (end_time - start_time) / (INPUT_AUDIO_LENGTH / SAMPLE_RATE)
        print(f"ASR: {text} / RTF: {real_time_factor:.4f}")
    if slice_end <= aligned_len:
        _bind_transfers(
            binding_Encoder, encoder_feedback, outputs_Encoder
        )
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
    if cif_fired:
        if slice_end > aligned_len:
            break
        _bind_transfers(
            binding_Decoder, decoder_feedback, outputs_Decoder
        )
    elif slice_end > aligned_len:
        break

text = "".join(transcript_parts)
print(f"\nFinal ASR Result: {text}")

