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
from ORT_IO import (
    array_for,
    filled_for,
    is_dynamic_dim,
    load_special_token_ids,
    metadata_by_name,
    numpy_dtype,
    resolve_shape,
)


# ============================================================================================
#                                       User configuration
# ============================================================================================
# IMPORTANT: CLI options are intentionally limited to model/vocabulary paths.
# Edit this section for demo, audio, streaming display, and ONNX Runtime behavior.
# The default tokens.txt is bundled inside the ONNX folder, so no path override is
# normally needed and inference remains stand-alone.

TEST_AUDIO    = model_audio_paths("x_asr")
PRINT_STREAMING_PARTIALS = True         # Demo mode: print one partial line after every encoded chunk.

# ONNX Runtime settings (house template)
USE_NORMALISE_AUDIO = False             # Apply RMS loudness normalisation before feeding the model. The reference X-ASR pipeline keeps the decoded waveform amplitude unchanged.
ORT_LOG       = False                   # Verbose ORT logging (False = fastest).
ORT_FP16      = False                   # True only if the graph was converted to fp16.
MAX_THREADS   = 2                       # Measured optimum on the target i7-1165G7; 0 lets ORT choose automatically.
DEVICE_ID     = 0
ORT_Accelerate_Providers = ["CUDAExecutionProvider"]           # e.g. ['CUDAExecutionProvider'] / ['OpenVINOExecutionProvider'] / ['DmlExecutionProvider']
CPU_DISABLE_MATMUL_ADD_FUSION = True    # ORT 1.27 wraps rank-3 MatMul+Add in costly Reshape/Gemm/Reshape chains.
CPU_DISABLE_NCHWC = True                # NCHWc reorders regress mean/tail latency on the target i7-1165G7.
CPU_EXTRA_DISABLED_OPTIMIZERS = [       # Individually benchmarked on the same CPU / ORT build.
    "ConvAddActivationFusion",
    "MatmulTransposeFusion",
]

def _parse_args():
    parser = argparse.ArgumentParser(description="Run X-ASR ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "X_ASR_Optimized", help="Folder containing ONNX graphs, for example X_ASR_Optimized or X_ASR_ONNX.")
    parser.add_argument("--vocab-path", "--tokenizer-path", dest="vocab_path", type=Path, default=None, help="Optional tokens.txt path; defaults to tokens.txt in the model folder.")
    return parser.parse_args()


_ARGS = _parse_args()
onnx_folder = _ARGS.onnx_folder.expanduser().resolve()
VOCAB_PATH = (
    _ARGS.vocab_path.expanduser().resolve()
    if _ARGS.vocab_path is not None
    else onnx_folder / "tokens.txt"
)
onnx_model_Metadata = str(onnx_folder / "ASR_Metadata.onnx")
onnx_encoder = str(onnx_folder / "X_ASR_Encoder.onnx")
onnx_decoder = str(onnx_folder / "X_ASR_Decoder.onnx")
onnx_joiner = str(onnx_folder / "X_ASR_Joiner.onnx")

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
_CPU_EP_ONLY = not ORT_Accelerate_Providers or set(ORT_Accelerate_Providers) == {"CPUExecutionProvider"}
_DISABLED_OPT = []
if ORT_FP16:
    _DISABLED_OPT.extend(["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"])
_DISABLED_OPT = _DISABLED_OPT or None
_ENCODER_DISABLED_OPT = list(_DISABLED_OPT or [])
if _CPU_EP_ONLY and CPU_DISABLE_MATMUL_ADD_FUSION:
    _ENCODER_DISABLED_OPT.append("MatMulAddFusion")
if _CPU_EP_ONLY and CPU_DISABLE_NCHWC:
    _ENCODER_DISABLED_OPT.append("NchwcTransformer")
if _CPU_EP_ONLY:
    _ENCODER_DISABLED_OPT.extend(CPU_EXTRA_DISABLED_OPTIMIZERS)
_ENCODER_DISABLED_OPT = _ENCODER_DISABLED_OPT or None
_RUN_OPTS = onnxruntime.RunOptions()
_RUN_OPTS.log_severity_level = 0 if ORT_LOG else 4
_RUN_OPTS.add_run_config_entry("disable_synchronize_execution_providers", "0")


def _make_session(path, disabled_optimizers=_DISABLED_OPT):
    return onnxruntime.InferenceSession(
        path, sess_options=_SESS_OPTS,
        providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
        provider_options=_PROVIDER_OPTS, disabled_optimizers=disabled_optimizers,
    )


def _ort_from(arr):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr), _DEVICE_STR, DEVICE_ID)


def _resolve_batch_one(value_meta):
    dynamic_axes = [
        axis for axis, dim in enumerate(value_meta.shape) if is_dynamic_dim(dim)
    ]
    return resolve_shape(value_meta, axes={0: 1} if dynamic_axes else None)


def _filled_batch_one(value_meta, fill_value=0):
    shape = _resolve_batch_one(value_meta)
    return filled_for(
        value_meta, fill_value,
        axes={0: shape[0]} if shape and is_dynamic_dim(value_meta.shape[0]) else None,
    )


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
        self.enc = _make_session(onnx_encoder, _ENCODER_DISABLED_OPT)
        self.dec = _make_session(onnx_decoder)
        self.joi = _make_session(onnx_joiner)
        self.enc_input_meta = self.enc.get_inputs()
        self.enc_output_meta = self.enc.get_outputs()
        self.dec_input_meta = self.dec.get_inputs()
        self.dec_output_meta = self.dec.get_outputs()
        self.joi_input_meta = self.joi.get_inputs()
        self.joi_output_meta = self.joi.get_outputs()
        self.enc_inputs = metadata_by_name(self.enc_input_meta)
        self.enc_outputs = metadata_by_name(self.enc_output_meta)
        self.dec_inputs = metadata_by_name(self.dec_input_meta)
        self.dec_outputs = metadata_by_name(self.dec_output_meta)
        self.joi_inputs = metadata_by_name(self.joi_input_meta)
        self.joi_outputs = metadata_by_name(self.joi_output_meta)
        self.enc_in = [value.name for value in self.enc_input_meta]
        self.enc_out = [value.name for value in self.enc_output_meta]
        self.dec_in_names = [value.name for value in self.dec_input_meta]
        self.dec_out_names = [value.name for value in self.dec_output_meta]
        self.joi_in = [value.name for value in self.joi_input_meta]
        self.joi_out_names = [value.name for value in self.joi_output_meta]
        # ---- self-configure from ONNX metadata + tensor shapes (nothing to keep in sync with export) ----
        model_meta = self.meta.get_modelmeta().custom_metadata_map or {}
        special_token_ids = load_special_token_ids(model_meta)

        indexed_keys = {
            int(name.removeprefix("cached_key_"))
            for name in self.enc_in
            if name.startswith("cached_key_") and name.removeprefix("cached_key_").isdigit()
        }
        num_layers = len(indexed_keys)
        expected_state_names = []
        state_kinds = (
            "cached_key", "cached_nonlin_attn", "cached_val1",
            "cached_val2", "cached_conv1", "cached_conv2",
        )
        for layer in range(num_layers):
            for kind in state_kinds:
                expected_state_names.append(f"{kind}_{layer}")
        expected_state_names.extend(("embed_states", "processed_lens"))
        expected_state_outputs = [f"new_{name}" for name in expected_state_names]
        self.n_states = len(expected_state_names)
        self._state_input_meta = [self.enc_inputs[name] for name in expected_state_names]
        self._state_output_meta = [self.enc_outputs[name] for name in expected_state_outputs]

        audio_in = self.enc_inputs["audio"]
        self.sample_rate = int(model_meta["sample_rate"])
        self.audio_pcm_scale = int(model_meta["audio_pcm_scale"])
        self.window_length = int(model_meta["window_length"])
        self.hop_length = int(model_meta["hop_length"])
        self.stream_stride_samples = int(model_meta["stream_stride_samples"])
        self.tail_padding_samples = int(model_meta["tail_padding_samples"])
        self.inv_sample_rate = 1.0 / self.sample_rate
        audio_shape = _resolve_batch_one(audio_in)
        self.audio_chunk = audio_shape[2]                         # waveform samples per chunk (4880 @ 160ms)
        self.audio_np_dtype = numpy_dtype(audio_in)               # ONNX audio dtype: int16 / float32 / float16
        self.input_audio_is_int16 = self.audio_np_dtype == np.dtype(np.int16)
        self.T = (self.audio_chunk - self.window_length) // self.hop_length + 1
        enc_out_meta = self.enc_outputs["encoder_out"]
        enc_out_shape = _resolve_batch_one(enc_out_meta)
        # The public encoder output is already output-downsampled; its static time
        # dimension is the exact number of source fbank frames advanced per chunk.
        self.frame_advance = enc_out_shape[1]
        self.stride_frames = self.stream_stride_samples // self.hop_length
        self.chunk_ms = int(round(self.stream_stride_samples * 1000 / self.sample_rate))
        dec_in_meta = self.dec_inputs["y"]
        dec_out_meta = self.dec_outputs["decoder_out"]
        joi_enc_meta = self.joi_inputs["encoder_out"]
        joi_out_meta = self.joi_outputs["max_token_id"]
        dec_in_shape = _resolve_batch_one(dec_in_meta)
        joi_enc_shape = _resolve_batch_one(joi_enc_meta)
        self.context_size = dec_in_shape[1]
        self.joiner_dim = joi_enc_shape[1]
        self.blank_id = special_token_ids["blank"]
        self.sos_eos_id = special_token_ids["sos_eos"]
        self.unknown_id = special_token_ids["unknown"]
        self.decoder_start_id = special_token_ids["decoder_start"]
        print(
            f"\nModel metadata: {len(model_meta)} keys "
            f"(sample_rate={self.sample_rate}, chunk_ms={self.chunk_ms}, "
            f"window/hop={self.window_length}/{self.hop_length}, "
            f"decode_chunk_len={self.frame_advance}, blank_id={self.blank_id})."
        )
        # ---- pre-allocate every buffer once ----
        self._x = _ort_from(filled_for(audio_in, axes={0: 1} if is_dynamic_dim(audio_in.shape[0]) else None))
        self._state_zeros = [_filled_batch_one(meta) for meta in self._state_input_meta]
        self._state_inputs = [_ort_from(value) for value in self._state_zeros]
        self._state_outputs = [
            _ort_from(_filled_batch_one(meta)) for meta in self._state_output_meta
        ]
        self._enc_out = _ort_from(_filled_batch_one(enc_out_meta))
        # decoder / joiner shared buffers
        self._y_np = _filled_batch_one(dec_in_meta)
        self._y = _ort_from(self._y_np)
        self._joi_e = _ort_from(_filled_batch_one(joi_enc_meta))
        self._decoder_out = _ort_from(_filled_batch_one(dec_out_meta))
        self.dec_in = dec_in_meta.name
        self.dec_out = dec_out_meta.name
        self.joi_out = joi_out_meta.name
        self._tok_id = _ort_from(_filled_batch_one(joi_out_meta))
        self.dec_bind = self.dec.io_binding()
        self.joi_bind = self.joi.io_binding()
        # ---- bind every static shared buffer ONCE (updated in place / chained device->device) ----
        self.enc_bind = self._build_enc_binding()
        # Decoder: context ids in (updated in place); decoder_out is written straight into the joiner's
        # decoder output buffer, so there is no device->host->device hop between decoder and joiner.
        self.dec_bind.bind_ortvalue_input(self.dec_in, self._y)
        self.dec_bind.bind_ortvalue_output(self.dec_out, self._decoder_out)
        # Joiner: per-frame encoder / decoder buffers in (updated in place / written by the decoder),
        # greedy token id out -> _tok_id.
        self.joi_bind.bind_ortvalue_input(self.joi_in[0], self._joi_e)
        self.joi_bind.bind_ortvalue_input(self.joi_in[1], self._decoder_out)
        self.joi_bind.bind_ortvalue_output(self.joi_out, self._tok_id)

    def _build_enc_binding(self):
        # Encoder inputs and outputs stay bound to distinct NodeArg-owned buffers.
        b = self.enc.io_binding()
        b.bind_ortvalue_input(self.enc_in[0], self._x)
        b.bind_ortvalue_output(self.enc_out[0], self._enc_out)
        for index in range(self.n_states):
            b.bind_ortvalue_input(
                self._state_input_meta[index].name,
                self._state_inputs[index],
            )
            b.bind_ortvalue_output(
                self._state_output_meta[index].name,
                self._state_outputs[index],
            )
        return b

    def prepare_audio_input(self, audio_int16: np.ndarray, target_rms: float = 4096.0) -> np.ndarray:
        # Fold the optional RMS loudness normalisation and the model-dtype conversion into a single pass
        # over the raw int16 PCM that pydub returns, casting to the model's audio dtype exactly once.
        #   int16 input: raw PCM (the encoder graph divides by 32768 internally).
        #   float32/float16 input: normalised to [-1, 1] here (÷32768), because the float graph skips the
        #   in-model division; float16 stores those values (the graph up-casts back to f32).
        if not USE_NORMALISE_AUDIO and self.input_audio_is_int16:
            return np.ascontiguousarray(audio_int16, dtype=self.audio_np_dtype)
        audio = audio_int16.astype(np.float32)
        if USE_NORMALISE_AUDIO:
            rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
            if rms > 0:
                audio *= (target_rms / (rms + 1e-7))
                np.clip(audio, -32768.0, 32767.0, out=audio)
        if self.input_audio_is_int16:
            return audio.astype(self.audio_np_dtype)
        audio *= np.float32(1.0 / self.audio_pcm_scale)
        return audio.astype(self.audio_np_dtype)   # float32 (no-op) or float16

    def reset(self):
        for state, zero in zip(self._state_inputs, self._state_zeros):
            state.update_inplace(zero)

    def _run_decoder(self, hyp):
        # Refresh the predictor context; decoder_out is written into the producer-owned buffer (bound once in
        # __init__), so the joiner reads it with no device->host->device round-trip.
        self._y_np[0] = hyp[-self.context_size:]
        self._y.update_inplace(self._y_np)
        _run(self.dec, self.dec_bind)

    def encode_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """audio_chunk: (audio_chunk_samples,) waveform in the model's audio dtype. In-graph
        fbank + sliding-cache ping-pong, returns encoder_out (T',512)."""
        # Fold the old host-staging buffer + copy into a single host->device write: the reflection-padded
        # window is already contiguous in the model dtype, so reshape is a view and no extra host copy is made.
        audio_value = array_for(
            self.enc_inputs[self.enc_in[0]],
            np.ascontiguousarray(audio_chunk, dtype=self.audio_np_dtype).reshape(1, 1, -1),
            axes={0: 1} if is_dynamic_dim(self.enc_inputs[self.enc_in[0]].shape[0]) else None,
        )
        self._x.update_inplace(audio_value)
        _run(self.enc, self.enc_bind)
        for state_input, state_output in zip(
            self._state_inputs,
            self._state_outputs,
        ):
            state_input.update_inplace(state_output)
        return self._enc_out.numpy()[0]                              # (T', 512)

    def greedy(self, encoder_out: np.ndarray, hyp):
        """Advance greedy transducer decoding over the frames of one encoder chunk. The joiner I/O
        buffers are bound once in __init__; decoder_out already lives in its producer-owned buffer."""
        for t in range(encoder_out.shape[0]):
            frame = array_for(
                self.joi_inputs[self.joi_in[0]], encoder_out[t:t + 1],
                axes={0: 1} if is_dynamic_dim(self.joi_inputs[self.joi_in[0]].shape[0]) else None,
            )
            self._joi_e.update_inplace(frame)
            _run(self.joi, self.joi_bind)
            token_ids = self._tok_id.numpy()
            y = int(token_ids.flat[0])
            if y != self.blank_id:
                hyp.append(y)
                self._run_decoder(hyp)                         # refreshes decoder output for the next frame
        return hyp

    def _format_hyp(self, hyp, token_table) -> str:
        text = "".join(token_table.get(i, "") for i in hyp[self.context_size:])
        return text.replace("\u2581", " ").strip()

    def transcribe_stream(self, waveform_1d: np.ndarray, token_table):
        # waveform_1d: raw int16 PCM or [-1, 1] float, matching the model's audio dtype. Reflection-pad
        # exactly like Kaldi snip_edges=False, then slide raw-audio windows so the encoder's in-graph Conv1d fbank
        # reproduces the training global fbank frame-for-frame. Yields a visible partial after each chunk.
        padded, num_frames = snip_edges_false_pad(
            np.concatenate([
                waveform_1d,
            np.zeros(self.tail_padding_samples, dtype=self.audio_np_dtype),
            ]),
            self.window_length,
            self.hop_length,
        )
        self.reset()
        hyp = [self.decoder_start_id] * (self.context_size - 1) + [self.blank_id]
        self._run_decoder(hyp)                                 # initialize the decoder output buffer
        frame_pos = 0                                           # current frame index into the global fbank
        chunk_index = 0
        while num_frames - frame_pos >= self.T:
            start = frame_pos * self.hop_length
            encoder_out = self.encode_chunk(padded[start:start + self.audio_chunk])
            frame_pos += self.stride_frames
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
    token_table = load_tokens(str(VOCAB_PATH))
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
