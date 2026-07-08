import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths


# ============================================================================================
#                                       User configuration
# ============================================================================================
# The tokens table (tokens.txt) is bundled inside the ONNX folder by the export / optimize step,
# so inference is stand-alone (no external zipformer data path needed).

TEST_AUDIO    = model_audio_paths("x_asr")
PRINT_STREAMING_PARTIALS = True     # Demo mode: print one partial line after every encoded chunk.

# The audio input dtype, audio frontend geometry, chunk geometry (decode_chunk_len), joiner_dim,
# context_size, blank_id and chunk_ms are all
# read automatically from the ONNX model metadata / tensor shapes at load time (XasrStreamingRunner.__init__),
# so none of them need to be set here or kept in sync with Export_X_ASR.py.

# ONNX Runtime settings (house template)
USE_NORMALISE_AUDIO = False         # Apply RMS loudness normalisation before feeding the model. The reference X-ASR pipeline keeps the decoded waveform amplitude unchanged.
ORT_LOG       = False               # Verbose ORT logging (False = fastest).
ORT_FP16      = False               # True only if the graph was converted to fp16.
MAX_THREADS   = 0                   # inter/intra-op CPU threads; 0 = auto.
DEVICE_ID     = 0
ORT_Accelerate_Providers = []       # e.g. ['CUDAExecutionProvider'] / ['OpenVINOExecutionProvider'] / ['DmlExecutionProvider']

def _parse_args():
    parser = argparse.ArgumentParser(description="Run X-ASR ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "X_ASR_Optimized", help="Folder containing ONNX graphs, for example X_ASR_Optimized or X_ASR_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
onnx_model_Metadata = str(onnx_folder / "X_ASR_Metadata.onnx")
onnx_encoder = str(onnx_folder / "X_ASR_Encoder.onnx")
onnx_decoder = str(onnx_folder / "X_ASR_Decoder.onnx")
onnx_joiner = str(onnx_folder / "X_ASR_Joiner.onnx")

_INV_INT16_SCALE = np.float32(1.0 / 32768.0)   # pre-computed [-1, 1] normalisation scale, reused every call


# ============================================================================================
#                    ONNX Runtime session / provider / device setup (house template)
# ============================================================================================
def _build_session_opts():
    so = onnxruntime.SessionOptions()
    so.log_severity_level = 0 if ORT_LOG else 4
    so.log_verbosity_level = 4
    so.inter_op_num_threads = MAX_THREADS
    so.intra_op_num_threads = MAX_THREADS
    so.enable_cpu_mem_arena = True
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    for k, v in {
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
        so.add_session_config_entry(k, v)
    return so


def _resolve_provider():
    if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
        opts = [{
            "device_type":              "CPU",
            "precision":                "ACCURACY",
            "num_of_threads":           MAX_THREADS if MAX_THREADS else 8,
            "num_streams":              1,
            "enable_opencl_throttling": False,
            "enable_qdq_optimizer":     False,
            "disable_dynamic_shapes":   False,
        }]
        return "cpu", C.OrtDevice.cpu(), opts
    if "CUDAExecutionProvider" in ORT_Accelerate_Providers:
        opts = [{
            "device_id":                        DEVICE_ID,
            "gpu_mem_limit":                    24 * (1024 ** 3),
            "arena_extend_strategy":            "kNextPowerOfTwo",
            "cudnn_conv_algo_search":           "EXHAUSTIVE",
            "sdpa_kernel":                      "2",
            "use_tf32":                         "1",
            "fuse_conv_bias":                   "0",
            "cudnn_conv_use_max_workspace":     "1",
            "cudnn_conv1d_pad_to_nc1d":         "0",
            "tunable_op_enable":                "0",
            "tunable_op_tuning_enable":         "0",
            "tunable_op_max_tuning_duration_ms": 10,
            "do_copy_in_default_stream":        "0",
            "enable_cuda_graph":                "0",
            "prefer_nhwc":                      "0",
            "enable_skip_layer_norm_strict_mode": "0",
            "use_ep_level_unified_stream":      "0",
        }]
        return "cuda", C.OrtDevice.cuda(), opts
    if "DmlExecutionProvider" in ORT_Accelerate_Providers:
        opts = [{
            "device_id":                  DEVICE_ID,
            "performance_preference":     "high_performance",
            "device_filter":              "gpu",
            "disable_metacommands":       "false",
            "enable_graph_capture":       "false",
            "enable_graph_serialization": "false",
        }]
        return "dml", C.OrtDevice.dml(), opts
    return "cpu", C.OrtDevice.cpu(), None


_SESS_OPTS = _build_session_opts()
_DEVICE_STR, _ORT_DEVICE_TYPE, _PROVIDER_OPTS = _resolve_provider()
_ORT_DEVICE = C.OrtDevice(_ORT_DEVICE_TYPE, C.OrtDevice.default_memory(), DEVICE_ID)
_DISABLED_OPT = ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None
_RUN_OPTS = onnxruntime.RunOptions()
_RUN_OPTS.log_severity_level = 0 if ORT_LOG else 4
_RUN_OPTS.add_run_config_entry("disable_synchronize_execution_providers", "0")


def _make_session(path):
    return onnxruntime.InferenceSession(
        path, sess_options=_SESS_OPTS,
        providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
        provider_options=_PROVIDER_OPTS, disabled_optimizers=_DISABLED_OPT,
    )


def _ort_zeros(shape, np_dtype):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=np_dtype), _DEVICE_STR, DEVICE_ID)


def _ort_from(arr):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), _DEVICE_STR, DEVICE_ID)


def _np_dtype_from_meta(meta):
    if "int64" in meta.type:
        return np.int64
    if "int32" in meta.type:
        return np.int32
    if "int16" in meta.type:
        return np.int16
    if "float16" in meta.type:
        return np.float16
    return np.float32


def _run(session, binding):
    session.run_with_iobinding(binding, run_options=_RUN_OPTS)


# ============================================================================================
#            kaldi snip_edges=False reflection padding (so the in-graph Conv1d fbank,
#            which frames like snip_edges=True, reproduces the training global fbank)
# ============================================================================================
def snip_edges_false_pad(waveform_1d: np.ndarray, window_length: int, hop_length: int):
    """(num_samples,) waveform -> (padded_waveform, num_frames). Reflection-pads the waveform
    exactly like Kaldi / torchaudio snip_edges=False, so that a stride-hop_length Conv1d over the
    padded signal yields the same frames as kaldi.fbank(..., snip_edges=False)."""
    wav = np.ascontiguousarray(waveform_1d)
    n = wav.shape[0]
    num_frames = (n + hop_length // 2) // hop_length
    pad = window_length // 2 - hop_length // 2
    reversed_wav = wav[::-1]
    if pad > 0:
        padded = np.concatenate([reversed_wav[-pad:], wav, reversed_wav])
    else:
        padded = np.concatenate([wav[-pad:], reversed_wav])
    return np.ascontiguousarray(padded), num_frames


def load_tokens(path):
    table = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                table[int(parts[1])] = parts[0]
            elif len(parts) == 1:
                table[len(table)] = parts[0]
    return table


# ============================================================================================
#          Deeply-optimized IOBinding streaming inference (encoder state ping-pong +
#          greedy transducer with decoder feedback, all zero-copy on pre-allocated buffers)
# ============================================================================================
class XasrStreamingRunner:
    def __init__(self):
        self.meta = _make_session(onnx_model_Metadata)
        self.enc = _make_session(onnx_encoder)
        self.dec = _make_session(onnx_decoder)
        self.joi = _make_session(onnx_joiner)
        self.enc_in = [i.name for i in self.enc.get_inputs()]
        self.enc_out = [o.name for o in self.enc.get_outputs()]
        self.n_states = len(self.enc_in) - 1                      # 116
        # ---- self-configure from ONNX metadata + tensor shapes (nothing to keep in sync with export) ----
        model_meta = self.meta.get_modelmeta().custom_metadata_map or {}

        def _required_meta_int(metadata, key, model_path):
            value = metadata.get(key)
            if value is None:
                raise KeyError(
                    f"Required metadata key '{key}' is missing from {model_path}. "
                    f"Re-export with Export_X_ASR.py to stamp the model metadata."
                )
            return int(value)

        audio_in = self.enc.get_inputs()[0]
        self.sample_rate = _required_meta_int(model_meta, "sample_rate", onnx_model_Metadata)
        self.chunk_ms = _required_meta_int(model_meta, "chunk_ms", onnx_model_Metadata)
        self.window_length = _required_meta_int(model_meta, "window_length", onnx_model_Metadata)
        self.hop_length = _required_meta_int(model_meta, "hop_length", onnx_model_Metadata)
        self.inv_sample_rate = 1.0 / self.sample_rate
        self.audio_chunk = audio_in.shape[2]                      # waveform samples per chunk (4880 @ 160ms)
        meta_audio_chunk = _required_meta_int(model_meta, "audio_chunk_samples", onnx_model_Metadata)
        if self.audio_chunk != meta_audio_chunk:
            raise ValueError(
                f"Encoder audio input shape says {self.audio_chunk} samples, but metadata audio_chunk_samples="
                f"{meta_audio_chunk}. Re-export all X-ASR graphs together."
            )
        self.audio_np_dtype = _np_dtype_from_meta(audio_in)       # ONNX audio dtype: int16 / float32 / float16
        self.input_audio_is_int16 = self.audio_np_dtype == np.int16
        self.T = (self.audio_chunk - self.window_length) // self.hop_length + 1
        enc_out_meta = self.enc.get_outputs()[0]
        enc_out_shape = [d if isinstance(d, int) else 1 for d in enc_out_meta.shape]
        self.frame_advance = _required_meta_int(model_meta, "decode_chunk_len", onnx_model_Metadata)
        self.context_size = self.dec.get_inputs()[0].shape[1]     # stateless predictor context (2)
        self.joiner_dim = self.joi.get_inputs()[0].shape[1]       # joiner space dim (512)
        self.blank_id = _required_meta_int(model_meta, "blank_id", onnx_model_Metadata)
        print(f"\nModel metadata: {len(model_meta)} keys "
              f"(sample_rate={self.sample_rate}, chunk_ms={self.chunk_ms}, "
              f"window/hop={self.window_length}/{self.hop_length}, "
              f"decode_chunk_len={self.frame_advance}, blank_id={self.blank_id}).")
        # ---- pre-allocate every buffer once ----
        self._x = _ort_zeros((1, 1, self.audio_chunk), self.audio_np_dtype)
        self._state_zeros = [self._init_state_array(m) for m in self.enc.get_inputs()[1:]]
        self._state_bank_a = [self._ort_zero_like_state(a) for a in self._state_zeros]
        self._state_bank_b = [self._ort_zero_like_state(a) for a in self._state_zeros]
        self._enc_out = _ort_zeros(enc_out_shape, _np_dtype_from_meta(enc_out_meta))
        # decoder / joiner shared buffers
        self._y_np = np.zeros((1, self.context_size), dtype=np.int32)
        self._y = _ort_zeros((1, self.context_size), np.int32)
        self._joi_e = _ort_zeros((1, self.joiner_dim), np.float32)
        self._joi_d = _ort_zeros((1, self.joiner_dim), np.float32)
        self.dec_in = self.dec.get_inputs()[0].name
        self.dec_out = self.dec.get_outputs()[0].name
        self.joi_in = [i.name for i in self.joi.get_inputs()]
        self.joi_out = self.joi.get_outputs()[0].name
        self._tok_id = _ort_zeros((1,), np.int32)                # joiner greedy token id (bound once, read per frame)
        self.dec_bind = self.dec.io_binding()
        self.joi_bind = self.joi.io_binding()
        # ---- bind every static shared buffer ONCE (updated in place / chained device->device) ----
        # Encoder ping-pong is folded into TWO fully pre-bound bindings (A->B and B->A) that share the
        # audio + encoder_out buffers, so a chunk just picks the binding by parity -- no per-chunk cache
        # re-binding at all.
        self.enc_bind_ab = self._build_enc_binding(self._state_bank_a, self._state_bank_b)
        self.enc_bind_ba = self._build_enc_binding(self._state_bank_b, self._state_bank_a)
        self._parity = 0
        # Decoder: context ids in (updated in place); decoder_out is written straight into the joiner's
        # decoder_out buffer (_joi_d), so there is no device->host->device hop between decoder and joiner.
        self.dec_bind.bind_ortvalue_input(self.dec_in, self._y)
        self.dec_bind.bind_ortvalue_output(self.dec_out, self._joi_d)
        # Joiner: per-frame encoder / decoder buffers in (updated in place / written by the decoder),
        # greedy token id out -> _tok_id.
        self.joi_bind.bind_ortvalue_input(self.joi_in[0], self._joi_e)
        self.joi_bind.bind_ortvalue_input(self.joi_in[1], self._joi_d)
        self.joi_bind.bind_ortvalue_output(self.joi_out, self._tok_id)

    def _build_enc_binding(self, in_bank, out_bank):
        # One fully pre-bound encoder binding: shared audio in + encoder_out out, plus the sliding caches
        # read from in_bank and written to out_bank. Two of these (A->B, B->A) replace every per-chunk rebind.
        b = self.enc.io_binding()
        b.bind_ortvalue_input(self.enc_in[0], self._x)
        b.bind_ortvalue_output(self.enc_out[0], self._enc_out)
        for nm, st in zip(self.enc_in[1:], in_bank):
            b.bind_ortvalue_input(nm, st)
        for nm, st in zip(self.enc_out[1:], out_bank):
            b.bind_ortvalue_output(nm, st)
        return b

    @staticmethod
    def _init_state_array(meta):
        shape = [d if isinstance(d, int) else 1 for d in meta.shape]
        return np.zeros(shape, dtype=_np_dtype_from_meta(meta))

    @staticmethod
    def _ort_zero_like_state(arr):
        return _ort_zeros(arr.shape, arr.dtype)

    def prepare_audio_input(self, audio_int16: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
        # Fold the optional RMS loudness normalisation and the model-dtype conversion into a single pass
        # over the raw int16 PCM that pydub returns, casting to the model's audio dtype exactly once.
        #   int16 input: raw PCM (the encoder graph divides by 32768 internally).
        #   float32/float16 input: normalised to [-1, 1] here (÷32768), because the float graph skips the
        #   in-model division; float16 stores those values (the graph up-casts back to f32).
        if not USE_NORMALISE_AUDIO and self.input_audio_is_int16:
            return np.ascontiguousarray(audio_int16, dtype=np.int16)
        audio = audio_int16.astype(np.float32)
        if USE_NORMALISE_AUDIO:
            rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
            if rms > 0:
                audio *= (target_rms / (rms + 1e-7))
                np.clip(audio, -32768.0, 32767.0, out=audio)
        if self.input_audio_is_int16:
            return audio.astype(np.int16)
        audio *= _INV_INT16_SCALE   # fold the pre-computed ÷32768 scale into the same float buffer
        return audio.astype(self.audio_np_dtype)   # float32 (no-op) or float16

    def reset(self):
        for bank in (self._state_bank_a, self._state_bank_b):
            for state, zero in zip(bank, self._state_zeros):
                state.update_inplace(zero)
        self._parity = 0

    def _run_decoder(self, hyp):
        # Refresh the predictor context; decoder_out is written in place into _joi_d (bound once in
        # __init__), so the joiner reads it with no device->host->device round-trip.
        self._y_np[0] = hyp[-self.context_size:]
        self._y.update_inplace(self._y_np)
        _run(self.dec, self.dec_bind)

    def encode_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """audio_chunk: (audio_chunk_samples,) waveform in the model's audio dtype. In-graph
        fbank + sliding-cache ping-pong, returns encoder_out (T',512)."""
        # Fold the old host-staging buffer + copy into a single host->device write: the reflection-padded
        # window is already contiguous in the model dtype, so reshape is a view and no extra host copy is made.
        self._x.update_inplace(np.ascontiguousarray(audio_chunk, dtype=self.audio_np_dtype).reshape(1, 1, -1))
        # Ping-pong without re-binding: even chunks read bank A / write bank B, odd chunks the reverse.
        _run(self.enc, self.enc_bind_ab if self._parity == 0 else self.enc_bind_ba)
        self._parity ^= 1
        return self._enc_out.numpy()[0]                              # (T', 512)

    def greedy(self, encoder_out: np.ndarray, hyp):
        """Advance greedy transducer decoding over the frames of one encoder chunk. The joiner I/O
        buffers are bound once in __init__; decoder_out already lives in _joi_d (written by _run_decoder)."""
        for t in range(encoder_out.shape[0]):
            self._joi_e.update_inplace(encoder_out[t:t + 1])
            _run(self.joi, self.joi_bind)
            y = int(self._tok_id.numpy()[0])
            if y != self.blank_id:
                hyp.append(y)
                self._run_decoder(hyp)                         # refreshes _joi_d in place for the next frame
        return hyp

    def _format_hyp(self, hyp, token_table) -> str:
        text = "".join(token_table.get(i, "") for i in hyp[self.context_size:])
        return text.replace("\u2581", " ").strip()

    def transcribe_stream(self, waveform_1d: np.ndarray, token_table):
        # waveform_1d: raw int16 PCM or [-1, 1] float, matching the model's audio dtype. Reflection-pad
        # exactly like Kaldi snip_edges=False, then slide raw-audio windows so the encoder's in-graph Conv1d fbank
        # reproduces the training global fbank frame-for-frame. Yields a visible partial after each chunk.
        padded, num_frames = snip_edges_false_pad(
            np.concatenate([waveform_1d, np.zeros(int(0.3 * self.sample_rate), dtype=waveform_1d.dtype)]),
            self.window_length,
            self.hop_length,
        )
        self.reset()
        hyp = [self.blank_id] * self.context_size
        self._run_decoder(hyp)                                 # initial decoder_out -> _joi_d
        frame_pos = 0                                           # current frame index into the global fbank
        chunk_index = 0
        while num_frames - frame_pos >= self.T:
            start = frame_pos * self.hop_length
            encoder_out = self.encode_chunk(padded[start:start + self.audio_chunk])
            frame_pos += self.frame_advance                    # decode_chunk_len fbank frames per chunk
            hyp = self.greedy(encoder_out, hyp)
            chunk_index += 1
            yield chunk_index, frame_pos * self.hop_length * self.inv_sample_rate, self._format_hyp(hyp, token_table)

    def transcribe(self, waveform_1d: np.ndarray, token_table) -> str:
        text = ""
        for _, _, text in self.transcribe_stream(waveform_1d, token_table):
            pass
        return text


# ============================================================================================
#                                            main
# ============================================================================================
if __name__ == "__main__":
    print("\n===== X-ASR ONNX inference =====")
    print("Loading exported models with IOBinding runtime ...")
    runner = XasrStreamingRunner()
    # The tokens table is bundled inside the ONNX folder by the export / optimize step, so inference is stand-alone.
    token_table = load_tokens(str(onnx_folder / "tokens.txt"))
    print(f"Providers: {runner.enc.get_providers()}  |  audio_chunk={runner.audio_chunk}  T={runner.T}  states={runner.n_states}")
    print(f"Auto-detected from ONNX: audio_dtype={np.dtype(runner.audio_np_dtype).name}  frame_advance={runner.frame_advance}  "
          f"context_size={runner.context_size}  joiner_dim={runner.joiner_dim}  blank_id={runner.blank_id}")

    for test in TEST_AUDIO:
        print("----------------------------------------------------------------------------------------------------------")
        print(f"\nTest Input Audio: {test}")
        seg = AudioSegment.from_file(test).set_channels(1).set_frame_rate(runner.sample_rate)
        audio_pcm = np.array(seg.get_array_of_samples(), dtype=np.int16)
        wav = runner.prepare_audio_input(audio_pcm)
        start_time = time.time()
        text = ""
        if PRINT_STREAMING_PARTIALS:
            print("\nStreaming partials:")
        for chunk_index, audio_seconds, partial in runner.transcribe_stream(wav, token_table):
            text = partial
            if PRINT_STREAMING_PARTIALS:
                print(f"[chunk {chunk_index:04d} | audio {audio_seconds:6.2f}s] {partial or '<blank>'}", flush=True)
        real_time_factor = (time.time() - start_time) / (len(wav) / runner.sample_rate)
        print(f"\nFinal ASR Result:\n{text}\n\nRTF: {real_time_factor:.4f}\n")
        print("----------------------------------------------------------------------------------------------------------")
