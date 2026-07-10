"""Run exported Parakeet TDT ASR ONNX graphs (OFFLINE) with ONNX Runtime IOBinding.

Auto-detects the graph set in the target folder (Parakeet_ASR_ONNX / Parakeet_ASR_Optimized) and runs a
single full-sequence encoder pass followed by the Token-and-Duration Transducer (TDT) greedy decode. The
decoder/joint carries its LSTM state across steps through shared ORT buffers (input==output aliasing) and
advances the encoder frame pointer by the per-step predicted duration, reproducing the HuggingFace
`ParakeetForTDT` greedy loop.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from tokenizers import Tokenizer

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Config
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Parakeet TDT ASR ONNX inference (offline).")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=None,
                        help="Folder with the ONNX graphs (Parakeet_ASR_ONNX or Parakeet_ASR_Optimized). "
                             "Defaults to auto-search of the known output folders.")
    return parser.parse_args()


_ARGS = _parse_args()

_GRAPH_NAMES = ("ASR_Matadata.onnx", "Parakeet_ASR_Encoder.onnx", "Parakeet_ASR_Decoder.onnx")
_DEFAULT_CANDIDATES = (
    _SCRIPT_DIR / "Parakeet_ASR_Optimized",
    _SCRIPT_DIR / "Parakeet_ASR_ONNX",
)


def _resolve_graphs():
    candidates = []
    if _ARGS.onnx_folder is not None:
        candidates.append(_ARGS.onnx_folder.expanduser().resolve())
    candidates.extend(c.resolve() for c in _DEFAULT_CANDIDATES)
    for folder in candidates:
        if (folder / _GRAPH_NAMES[0]).exists():
            return folder, _GRAPH_NAMES
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No Parakeet ASR ONNX graphs found. Looked in:\n  {tried}")


ONNX_FOLDER, (METADATA_NAME, ENCODER_NAME, DECODER_NAME) = _resolve_graphs()

TEST_AUDIO    = _REPO_ROOT / "Test_Examples" / "en" / "test_sample.wav"
USE_NORMALISE_AUDIO = False

ORT_Accelerate_Providers = []      # e.g. ['CUDAExecutionProvider', 'DmlExecutionProvider']
ORT_LOG       = False
MAX_THREADS   = 0                  # 0 = auto
DEVICE_ID     = 0

_INV_INT16    = np.float32(1.0 / 32768.0)


# ONNX Runtime helpers
def _build_session_opts() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level = 0 if ORT_LOG else 4
    opts.inter_op_num_threads = MAX_THREADS
    opts.intra_op_num_threads = MAX_THREADS
    opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    for key, value in {
        "session.set_denormal_as_zero": "1",
        "session.intra_op.allow_spinning": "1",
        "session.inter_op.allow_spinning": "1",
        "session.use_device_allocator_for_initializers": "1",
        "optimization.enable_cast_chain_elimination": "1",
    }.items():
        opts.add_session_config_entry(key, value)
    return opts


if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    _ort_device_type = C.OrtDevice.cuda()
    device_type = "cuda"
    provider_options = [{"device_id": DEVICE_ID}]
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    _ort_device_type = C.OrtDevice.dml()
    device_type = "dml"
    provider_options = [{"device_id": DEVICE_ID}]
else:
    _ort_device_type = C.OrtDevice.cpu()
    device_type = "cpu"
    provider_options = None

_ort_device_obj = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
_session_opts = _build_session_opts()
_packed = {
    "sess_options": _session_opts,
    "providers": ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    "provider_options": provider_options,
}


def _make_session(path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(path), **_packed)


def _ort_from_numpy(arr: np.ndarray) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), device_type, DEVICE_ID)


def _ort_shape(value: onnxruntime.OrtValue) -> tuple[int, ...]:
    shape_fn = getattr(value, "shape", None)
    if callable(shape_fn):
        return tuple(int(dim) for dim in shape_fn())
    return tuple(int(dim) for dim in value.numpy().shape)


def _in_names(session):
    return [x.name for x in session.get_inputs()]


def _out_names(session):
    return [x.name for x in session.get_outputs()]


def _np_dtype(type_str: str):
    if "float16" in type_str:
        return np.float16
    if "int64" in type_str:
        return np.int64
    if "int32" in type_str:
        return np.int32
    if "int16" in type_str:
        return np.int16
    return np.float32


# Audio helpers
def load_audio_int16(path, sample_rate: int) -> np.ndarray:
    seg = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
    return np.array(seg.get_array_of_samples(), dtype=np.int16)


def prepare_audio_input(audio_int16: np.ndarray, input_audio_dtype: str) -> np.ndarray:
    if not USE_NORMALISE_AUDIO and input_audio_dtype == "INT16":
        return np.ascontiguousarray(audio_int16, dtype=np.int16)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= (8192.0 / (rms + 1e-7))
            np.clip(audio, -32768.0, 32767.0, out=audio)
    if input_audio_dtype == "INT16":
        return audio.astype(np.int16)
    audio *= _INV_INT16
    return audio.astype(np.float16 if input_audio_dtype == "F16" else np.float32)


# Main
def main() -> None:
    print(f"Loading ONNX sessions from {ONNX_FOLDER} ...")
    sess_meta = _make_session(ONNX_FOLDER / METADATA_NAME)
    sess_enc = _make_session(ONNX_FOLDER / ENCODER_NAME)
    sess_dj = _make_session(ONNX_FOLDER / DECODER_NAME)
    print(f"  Providers: {sess_enc.get_providers()}")

    meta = sess_meta.get_modelmeta().custom_metadata_map or {}
    if meta.get("parakeet_asr_metadata_version") != "1":
        raise ValueError("Metadata version 1 missing. Re-export with Export_Parakeet_ASR.py.")

    def mi(key):
        return int(meta[key])

    lstm_layers = mi("decoder_layers")
    pred_hidden = mi("decoder_pred_hidden")
    blank_id = mi("blank_id")
    max_symbols = mi("max_symbols")
    sample_rate = mi("sample_rate")
    input_audio_dtype = meta.get("input_audio_dtype", "F32")
    tokenizer_name = "tokenizer.json"

    tokenizer_path = ONNX_FOLDER / tokenizer_name
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Re-export to copy it next to the graphs.")
    tok = Tokenizer.from_file(str(tokenizer_path))

    # Decoder/joint I/O and shared state buffers (state bound in==out for in-place carry).
    dj_in = _in_names(sess_dj)
    dj_out = _out_names(sess_dj)
    dj_enc_in, dj_frame_in, dj_token_in, dj_state_h_in, dj_state_c_in = dj_in
    dj_next_token_out, dj_is_blank_out, dj_duration_out, dj_state_h_next_out, dj_state_c_next_out = dj_out

    # LSTM state dtype follows the exported decoder graph (F32 or F16), never hard-coded.
    dj_in_specs = {spec.name: spec for spec in sess_dj.get_inputs()}
    state_dtype = _np_dtype(dj_in_specs[dj_state_h_in].type)

    frame_idx_np = np.zeros((1,), dtype=np.int32)
    frame_idx_ort = _ort_from_numpy(frame_idx_np)
    token_buf = _ort_from_numpy(np.array([[blank_id]], dtype=np.int32))
    state_h_buf = _ort_from_numpy(np.zeros((lstm_layers, 1, pred_hidden), dtype=state_dtype))
    state_c_buf = _ort_from_numpy(np.zeros((lstm_layers, 1, pred_hidden), dtype=state_dtype))
    is_blank_buf = _ort_from_numpy(np.zeros((1, 1), dtype=np.int32))
    duration_buf = _ort_from_numpy(np.zeros((1, 1), dtype=np.int32))

    binding_dj = sess_dj.io_binding()
    binding_dj.bind_ortvalue_input(dj_frame_in, frame_idx_ort)
    binding_dj.bind_ortvalue_input(dj_token_in, token_buf)
    binding_dj.bind_ortvalue_input(dj_state_h_in, state_h_buf)
    binding_dj.bind_ortvalue_input(dj_state_c_in, state_c_buf)
    binding_dj.bind_ortvalue_output(dj_next_token_out, token_buf)
    binding_dj.bind_ortvalue_output(dj_state_h_next_out, state_h_buf)
    binding_dj.bind_ortvalue_output(dj_state_c_next_out, state_c_buf)
    binding_dj.bind_ortvalue_output(dj_is_blank_out, is_blank_buf)
    binding_dj.bind_ortvalue_output(dj_duration_out, duration_buf)

    # Encoder: single full-sequence pass, enc_proj kept on device and fed straight to the decoder.
    audio_i16 = load_audio_int16(TEST_AUDIO, sample_rate)
    audio = prepare_audio_input(audio_i16, input_audio_dtype).reshape(1, 1, -1)
    audio_seconds = audio.shape[2] / sample_rate

    enc_in = _in_names(sess_enc)
    enc_out = _out_names(sess_enc)
    audio_buf = _ort_from_numpy(audio)
    binding_enc = sess_enc.io_binding()
    binding_enc.bind_ortvalue_input(enc_in[0], audio_buf)
    binding_enc._iobinding.bind_output(enc_out[0], _ort_device_obj)

    t0 = time.time()
    sess_enc.run_with_iobinding(binding_enc)
    enc_proj_ort = binding_enc.get_outputs()[0]
    n_frames = _ort_shape(enc_proj_ort)[1]
    binding_dj.bind_ortvalue_input(dj_enc_in, enc_proj_ort)

    # TDT greedy decode: advance the encoder frame pointer by the predicted duration each step.
    tokens = []
    frame_idx = 0
    steps = 0
    max_steps = max_symbols * n_frames
    while frame_idx < n_frames and steps < max_steps:
        frame_idx_np[0] = min(frame_idx, n_frames - 1)
        frame_idx_ort.update_inplace(frame_idx_np)
        sess_dj.run_with_iobinding(binding_dj)
        if int(is_blank_buf.numpy().flat[0]) == 0:
            tokens.append(token_buf.numpy().flat[0])
        frame_idx += duration_buf.numpy().flat[0]
        steps += 1
    elapsed = time.time() - t0

    text = tok.decode(tokens, skip_special_tokens=True)
    rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"Audio      : {Path(TEST_AUDIO).name}  ({audio_seconds:.2f}s, {n_frames} encoder frames)")
    print(f"Mode       : offline (single pass, TDT greedy)")
    print(f"Transcript : {text}")
    print(f"Tokens     : {len(tokens)}  ({tokens[:20]}{' ...' if len(tokens) > 20 else ''})")
    print(f"Steps      : {steps}")
    print(f"Elapsed    : {elapsed:.3f}s   RTF: {rtf:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
