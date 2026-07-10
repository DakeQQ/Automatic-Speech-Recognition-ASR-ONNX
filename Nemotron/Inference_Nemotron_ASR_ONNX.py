"""Run exported Nemotron ASR ONNX graphs (OFFLINE or STREAMING) with ONNX Runtime IOBinding.

Auto-detects the graph set in the target folder: offline (Nemotron_ASR_*) runs one full-sequence
encoder pass (single pass, or tumbling windows for a fixed-length encoder); streaming
(Nemotron_ASR_Streaming_*) drives a sliding fixed-length window through NeMo's cache-aware Conformer,
threading the attention/conv/mel caches chunk-to-chunk. Both share the carried-state RNN-T greedy
decoder. The mode is selected by the metadata ``streaming`` flag.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
import sentencepiece as spm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Config
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Nemotron ASR ONNX inference (offline or streaming).")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=None,
                        help="Folder with the ONNX graphs (offline Nemotron_ASR_* or streaming "
                             "Nemotron_ASR_Streaming_*, either _ONNX or _Optimized). "
                             "Defaults to auto-search of the known output folders.")
    return parser.parse_args()


_ARGS = _parse_args()

# The fused inference auto-detects offline vs streaming graphs in the target folder.
_OFFLINE_NAMES   = ("ASR_Matadata.onnx", "Nemotron_ASR_Encoder.onnx", "Nemotron_ASR_Decoder.onnx")
_STREAMING_NAMES = ("ASR_Matadata.onnx", "Nemotron_ASR_Streaming_Encoder.onnx",
                    "Nemotron_ASR_Streaming_Decoder.onnx")
_DEFAULT_CANDIDATES = (
    _SCRIPT_DIR / "Nemotron_ASR_Optimized",
    _SCRIPT_DIR / "Nemotron_ASR_ONNX",
    _SCRIPT_DIR / "Nemotron_ASR_Streaming_Optimized",
    _SCRIPT_DIR / "Nemotron_ASR_Streaming_ONNX",
)


def _probe_names(folder: Path):
    """Return the graph-name triple present in ``folder`` (streaming preferred), or None."""
    if (folder / _STREAMING_NAMES[0]).exists():
        return _STREAMING_NAMES
    if (folder / _OFFLINE_NAMES[0]).exists():
        return _OFFLINE_NAMES
    return None


def _resolve_graphs():
    """Pick the ONNX folder + graph names: explicit --onnx-folder first, then default candidates."""
    candidates = []
    if _ARGS.onnx_folder is not None:
        candidates.append(_ARGS.onnx_folder.expanduser().resolve())
    candidates.extend(c.resolve() for c in _DEFAULT_CANDIDATES)
    for folder in candidates:
        names = _probe_names(folder)
        if names is not None:
            return folder, names
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No Nemotron ASR ONNX graphs found. Looked in:\n  {tried}")


ONNX_FOLDER, (METADATA_NAME, ENCODER_NAME, DECODER_NAME) = _resolve_graphs()

TEST_AUDIO    = _REPO_ROOT / "Test_Examples" / "en" / "test_sample.wav"
# TARGET_LANG must match a metadata prompt key; "auto" uses auto-detect.
TARGET_LANG   = "en-US"
USE_NORMALISE_AUDIO = False
STRIP_LANG_TAGS = True
PRINT_PARTIALS  = True             # Streaming only: print the growing transcript after every chunk.

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


def _bind_inputs(binding, names, values) -> None:
    for name, value in zip(names, values):
        binding.bind_ortvalue_input(name, value)


def _bind_device_outputs(binding, names) -> None:
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


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


def _fixed_audio_length(session):
    """Return the fixed encoder sample count, or None for dynamic audio length."""
    inputs = session.get_inputs()
    if not inputs:
        return None
    spec = inputs[0]
    dim = spec.shape[2] if len(spec.shape) > 2 else None
    return dim if isinstance(dim, int) else None


# Audio/text helpers
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


def strip_lang_tags(text: str) -> str:
    import re
    return re.sub(r"<[a-z]{2}(-[A-Za-z]{2,4})?>", "", text).strip()


# Main
def main() -> None:
    label = "streaming " if METADATA_NAME == _STREAMING_NAMES[0] else ""
    print(f"Loading {label}ONNX sessions from {ONNX_FOLDER} ...")
    sess_meta = _make_session(ONNX_FOLDER / METADATA_NAME)
    sess_enc = _make_session(ONNX_FOLDER / ENCODER_NAME)
    sess_dj = _make_session(ONNX_FOLDER / DECODER_NAME)
    print(f"  Providers: {sess_enc.get_providers()}")

    meta = sess_meta.get_modelmeta().custom_metadata_map or {}
    if meta.get("nemotron_asr_metadata_version") != "1":
        raise ValueError("Metadata version 1 missing. Re-export with Export_Nemotron_ASR.py.")
    streaming = meta.get("streaming") == "1"

    def mi(key):
        return int(meta[key])

    import json
    lstm_layers = mi("decoder_layers")
    pred_hidden = mi("decoder_pred_hidden")
    blank_id = mi("blank_id")
    max_symbols = mi("max_symbols")
    sample_rate = mi("sample_rate")
    input_audio_dtype = meta.get("input_audio_dtype", "F32")
    prompt_dict = json.loads(meta["prompt_dictionary"])
    tokenizer_name = meta.get("tokenizer_model", "tokenizer.model")
    prompt_id = prompt_dict.get(TARGET_LANG, prompt_dict.get("auto", 101))

    sp = spm.SentencePieceProcessor()
    sp.load(str(ONNX_FOLDER / tokenizer_name))

    dj_names_in = _in_names(sess_dj)
    dj_names_out = _out_names(sess_dj)
    dj_enc_proj_in, dj_frame_idx_in, dj_token_in, dj_state_h_in, dj_state_c_in = dj_names_in
    dj_next_token_out, dj_is_blank_out, dj_state_h_next_out, dj_state_c_next_out = dj_names_out

    # Shared RNN-T greedy decoder: state buffers bound in==out for in-place updates.
    frame_idx_np = np.zeros((1,), dtype=np.int32)
    frame_idx_ort = _ort_from_numpy(np.zeros((1,), dtype=np.int32))
    token_buf = _ort_from_numpy(np.array([[blank_id]], dtype=np.int32))
    state_h_buf = _ort_from_numpy(np.zeros((lstm_layers, 1, pred_hidden), dtype=np.float32))
    state_c_buf = _ort_from_numpy(np.zeros((lstm_layers, 1, pred_hidden), dtype=np.float32))
    is_blank_buf = _ort_from_numpy(np.zeros((1, 1), dtype=np.int32))

    binding_dj = sess_dj.io_binding()
    binding_dj.bind_ortvalue_input(dj_frame_idx_in, frame_idx_ort)
    binding_dj.bind_ortvalue_input(dj_token_in, token_buf)
    binding_dj.bind_ortvalue_input(dj_state_h_in, state_h_buf)
    binding_dj.bind_ortvalue_input(dj_state_c_in, state_c_buf)
    binding_dj.bind_ortvalue_output(dj_next_token_out, token_buf)
    binding_dj.bind_ortvalue_output(dj_state_h_next_out, state_h_buf)
    binding_dj.bind_ortvalue_output(dj_state_c_next_out, state_c_buf)
    binding_dj.bind_ortvalue_output(dj_is_blank_out, is_blank_buf)

    def decode_segment(enc_proj_ort, out_tokens, n_frames=None):
        binding_dj.bind_ortvalue_input(dj_enc_proj_in, enc_proj_ort)
        if n_frames is None:
            n_frames = _ort_shape(enc_proj_ort)[1]
        for t in range(n_frames):
            frame_idx_np[0] = t
            frame_idx_ort.update_inplace(frame_idx_np)
            emitted = 0
            while emitted < max_symbols:
                sess_dj.run_with_iobinding(binding_dj)
                if int(is_blank_buf.numpy().flat[0]) != 0:
                    break
                out_tokens.append(int(token_buf.numpy().flat[0]))
                emitted += 1

    blank_token_np = np.array([[blank_id]], dtype=np.int32)
    zero_h_np = np.zeros((lstm_layers, 1, pred_hidden), dtype=np.float32)
    zero_c_np = np.zeros((lstm_layers, 1, pred_hidden), dtype=np.float32)

    def reset_decoder_state():
        token_buf.update_inplace(blank_token_np)
        state_h_buf.update_inplace(zero_h_np)
        state_c_buf.update_inplace(zero_c_np)

    audio_i16 = load_audio_int16(TEST_AUDIO, sample_rate)
    audio = prepare_audio_input(audio_i16, input_audio_dtype).reshape(1, 1, -1)

    if streaming:
        # Cache-aware sliding-window encoder; per-layer caches thread chunk-to-chunk, state CARRIES.
        valid_out_len = mi("valid_out_len")
        stride_samples = mi("stream_stride_samples")
        left_overlap = mi("stream_left_overlap")
        window_samples = mi("stream_window_samples")
        print(f"  target_lang={TARGET_LANG!r} -> prompt_id={prompt_id}   "
              f"window={window_samples} stride={stride_samples} valid_out_len={valid_out_len}")

        enc_names_in = _in_names(sess_enc)
        enc_names_out = _out_names(sess_enc)
        (enc_audio_in, enc_mel_cache_in, enc_chan_cache_in,
         enc_time_cache_in, enc_cache_len_in, enc_prompt_in) = enc_names_in[:6]
        (enc_proj_out, enc_mel_cache_next_out, enc_chan_cache_next_out,
         enc_time_cache_next_out, enc_cache_len_next_out) = enc_names_out[:5]
        enc_in_meta = {spec.name: spec for spec in sess_enc.get_inputs()}
        # The exported audio window shape is fixed; fall back to metadata if the session hides it.
        _audio_dim = enc_in_meta[enc_audio_in].shape[-1]
        if isinstance(_audio_dim, int):
            window_samples = _audio_dim
        audio_np_dtype = _np_dtype(enc_in_meta[enc_audio_in].type)

        def _zeros_like_input(name):
            spec = enc_in_meta[name]
            shape = tuple(int(d) for d in spec.shape)
            return _ort_from_numpy(np.zeros(shape, dtype=_np_dtype(spec.type)))

        audio_buf = _ort_from_numpy(np.zeros((1, 1, window_samples), dtype=audio_np_dtype))
        mel_cache_buf = _zeros_like_input(enc_mel_cache_in)
        chan_cache_buf = _zeros_like_input(enc_chan_cache_in)
        time_cache_buf = _zeros_like_input(enc_time_cache_in)
        cache_len_buf = _zeros_like_input(enc_cache_len_in)
        prompt_buf = _ort_from_numpy(np.array([prompt_id], dtype=_np_dtype(enc_in_meta[enc_prompt_in].type)))

        enc_in_map = {
            enc_audio_in: audio_buf, enc_mel_cache_in: mel_cache_buf,
            enc_chan_cache_in: chan_cache_buf, enc_time_cache_in: time_cache_buf,
            enc_cache_len_in: cache_len_buf, enc_prompt_in: prompt_buf,
        }
        # Feedback inputs (fed from the previous chunk's cache outputs) vs. persistent audio/prompt inputs.
        feedback_in = [enc_mel_cache_in, enc_chan_cache_in, enc_time_cache_in, enc_cache_len_in]
        feedback_out = [enc_mel_cache_next_out, enc_chan_cache_next_out, enc_time_cache_next_out, enc_cache_len_next_out]
        feedback_out_idx = [enc_names_out.index(n) for n in feedback_out]
        enc_proj_idx = enc_names_out.index(enc_proj_out)

        binding_enc = sess_enc.io_binding()
        _bind_inputs(binding_enc, enc_names_in, [enc_in_map[name] for name in enc_names_in])
        _bind_device_outputs(binding_enc, enc_names_out)

        def stream_window(signal, base):
            total = signal.shape[2]
            start = base - left_overlap
            end = start + window_samples
            seg = signal[:, :, max(0, start):min(total, end)]
            left_pad = max(0, -start)
            right_pad = max(0, end - total)
            if left_pad or right_pad:
                seg = np.pad(seg, ((0, 0), (0, 0), (left_pad, right_pad)))
            return np.ascontiguousarray(seg)

        total_samples = audio.shape[2]
        audio_seconds = total_samples / sample_rate
        num_chunks = (total_samples + stride_samples - 1) // stride_samples
        print(f"\nAudio      : {Path(TEST_AUDIO).name}  ({audio_seconds:.2f}s, {num_chunks} chunks)\n")

        tokens = []
        t0 = time.time()
        base = 0
        for k in range(num_chunks):
            window = stream_window(audio, base)
            audio_buf.update_inplace(window)
            sess_enc.run_with_iobinding(binding_enc)
            all_outputs = binding_enc.get_outputs()
            enc_proj_ort = all_outputs[enc_proj_idx]
            decode_segment(enc_proj_ort, tokens, valid_out_len)
            if PRINT_PARTIALS:
                partial = sp.decode(tokens)
                partial = strip_lang_tags(partial) if STRIP_LANG_TAGS else partial
                print(f"  [chunk {k + 1:2d}/{num_chunks}] {partial}")
            if k + 1 < num_chunks:
                # Thread cache outputs back as next-chunk cache inputs, then re-arm device outputs.
                _bind_inputs(binding_enc, feedback_in, [all_outputs[i] for i in feedback_out_idx])
                _bind_device_outputs(binding_enc, enc_names_out)
            base += stride_samples
        elapsed = time.time() - t0
        mode = f"streaming (cache-aware, {num_chunks} sliding windows)"
    else:
        # Offline full-sequence encoder: single pass, or tumbling windows for a fixed-length graph.
        print(f"  target_lang={TARGET_LANG!r} -> prompt_id={prompt_id}")
        fixed_len = _fixed_audio_length(sess_enc)
        if fixed_len is not None and audio.shape[2] < fixed_len:
            n = audio.shape[2]
            audio = np.ascontiguousarray(
                np.concatenate([audio, np.zeros((1, 1, fixed_len - n), dtype=audio.dtype)], axis=2))
            print(f"  Fixed-length encoder: padded audio {n} -> {fixed_len} samples with silence.")
        audio_seconds = audio.shape[2] / sample_rate
        t0 = time.time()

        binding_enc = sess_enc.io_binding()
        enc_names_in = _in_names(sess_enc)
        enc_names_out = _out_names(sess_enc)
        enc_audio_in, enc_prompt_in = enc_names_in[:2]
        enc_proj_out = enc_names_out[0]
        enc_proj_idx = enc_names_out.index(enc_proj_out)
        # Encoder inputs are bound once; audio is refreshed in the same OrtValue per segment.
        enc_audio_shape = ((1, 1, fixed_len) if (fixed_len is not None and audio.shape[2] > fixed_len)
                           else tuple(int(d) for d in audio.shape))
        audio_buf = _ort_from_numpy(np.zeros(enc_audio_shape, dtype=audio.dtype))
        prompt_buf = _ort_from_numpy(np.array([prompt_id], dtype=np.int32))
        enc_in_map = {enc_audio_in: audio_buf, enc_prompt_in: prompt_buf}
        _bind_inputs(binding_enc, enc_names_in, [enc_in_map[name] for name in enc_names_in])
        _bind_device_outputs(binding_enc, enc_names_out)

        def run_encoder(audio_np):
            audio_buf.update_inplace(np.ascontiguousarray(audio_np))
            sess_enc.run_with_iobinding(binding_enc)
            return binding_enc.get_outputs()[enc_proj_idx]

        tokens = []
        if fixed_len is not None and audio.shape[2] > fixed_len:
            total = audio.shape[2]
            n_windows = (total + fixed_len - 1) // fixed_len
            print(f"  Fixed-length encoder: audio {total} > {fixed_len} samples; "
                  f"decoding {n_windows} tumbling window(s).")
            pad_buf = np.zeros((1, 1, fixed_len), dtype=audio.dtype)
            start = 0
            for k in range(n_windows):
                end = start + fixed_len
                seg = audio[:, :, start:end]
                valid = seg.shape[2]
                if valid < fixed_len:
                    pad_buf[:, :, :valid] = seg
                    seg = pad_buf
                reset_decoder_state()
                enc_proj_ort = run_encoder(seg)
                decode_segment(enc_proj_ort, tokens)
                start = end
            mode = "offline (tumbling windows)"
        else:
            enc_proj_ort = run_encoder(audio)
            decode_segment(enc_proj_ort, tokens)
            mode = "offline (single pass)"
        elapsed = time.time() - t0

    text = sp.decode(tokens)
    display = strip_lang_tags(text) if STRIP_LANG_TAGS else text
    rtf = elapsed / audio_seconds if audio_seconds > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"Audio      : {Path(TEST_AUDIO).name}  ({audio_seconds:.2f}s)")
    print(f"Mode       : {mode}")
    print(f"Transcript : {display}")
    print(f"Raw        : {text}")
    print(f"Tokens     : {len(tokens)}  ({tokens[:20]}{' ...' if len(tokens) > 20 else ''})")
    print(f"Elapsed    : {elapsed:.3f}s   RTF: {rtf:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
