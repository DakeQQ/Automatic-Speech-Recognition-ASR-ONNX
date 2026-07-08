import argparse
import os
import sys
import time
import onnxruntime
from onnxruntime.capi import _pybind_state as C
import numpy as np
from pydub import AudioSegment
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from Example_Audio import model_audio_paths


def _parse_args():
    parser = argparse.ArgumentParser(description="Run Dolphin-CN-Dialect streaming ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", default=os.path.join(_SCRIPT_DIR, "Dolphin_CN_Dialect_Streaming_Optimized"), help="Folder containing ONNX graphs, for example Dolphin_CN_Dialect_Streaming_Optimized or Dolphin_CN_Dialect_Streaming_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()


onnx_folder            = os.path.abspath(_ARGS.onnx_folder)                          # Selected ONNX graph folder.
save_vocab             = os.path.join(onnx_folder, "vocab_Dolphin_CN_Dialect.txt")    # The exported Dolphin-CN-Dialect vocab path.
onnx_model_Metadata    = os.path.join(onnx_folder, "Dolphin_Metadata.onnx")                             # Tiny metadata carrier graph.
onnx_model_Encoder     = os.path.join(onnx_folder, "Dolphin_Encoder.onnx")                              # The exported onnx encoder model path.
onnx_model_Decoder     = os.path.join(onnx_folder, "Dolphin_Decoder.onnx")                              # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = os.path.join(onnx_folder, "Dolphin_Decoder_Embed.onnx")                        # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = os.path.join(onnx_folder, "Dolphin_Position_Mask_Prefill.onnx")                # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = os.path.join(onnx_folder, "Dolphin_Position_Mask_Decode.onnx")                 # Decode position-embedding graph for the single new token.
onnx_model_Argmax      = os.path.join(onnx_folder, "Dolphin_Argmax.onnx")                               # Bare argmax for the short lang/region attention pass (the transcript itself comes from CTC).


test_audio = model_audio_paths("dolphin_cn_dialect")                                                     # The test audio list.


# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO = False             # Apply RMS loudness normalisation before feeding the model. The reference Dolphin pipeline keeps the decoded waveform amplitude unchanged.
# Supported language configuration (README + units.txt):
# | Setting    | Supported values |
# | ---------- | ---------------- |
# | LANG_SYM   | zh, en |
# | REGION_SYM | CN, TW, WU, SICHUAN, SHANXI, ANHUI, TIANJIN, NINGXIA, SHAANXI, HEBEI |
# |            | SHANDONG, GUANGDONG, SHANGHAI, HUBEI, LIAONING, GANSU, FUJIAN, HUNAN |
# |            | HENAN, YUNNAN, MINNAN, WENZHOU, BEIJING, JILIN, NEIMENGGU, GUANGXI |
# |            | GUIZHOU, HEILONGJIANG, JIANGSU |
LANG_SYM           = ""                 # Force the language, e.g. "zh"/"en". Leave "" to auto-detect. Requires REGION_SYM too; otherwise auto-detect is used.
REGION_SYM         = ""                 # Force the region, e.g. "CN"/"TW"/"SHANGHAI". Leave "" to auto-detect. Both LANG_SYM and REGION_SYM must be set to skip detection.


# ============================================================================
# ONNX Runtime runtime configuration (IOBinding + shared buffers, mirrors Qwen ASR)
# ============================================================================
ORT_Accelerate_Providers = []           # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                = False          # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16               = False          # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS            = 0              # Parallel CPU threads. Set 0 for auto.
DEVICE_ID              = 0              # Default to zero.


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


class Tokenizer:
    # Char tokenizer for Dolphin-CN-Dialect (no bpe.model). Chinese characters map 1:1; English BPE-style pieces
    # carry the SentencePiece word-boundary marker "▁", which is rendered back as a space at detokenisation.
    def __init__(self, filename):
        self.str_to_idx = {}
        self.idx_to_str = {}
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def decode_ids(self, ids):
        tokens = [self.decode(int(idx)) for idx in ids]
        tokens = [token for token in tokens if token is not None]
        return ''.join(tokens).replace("▁", " ").strip()


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


print('\nLoading the model...')

# ---- Core pipeline sessions ----
ort_session_Metadata = _make_session(onnx_model_Metadata)
ort_session_Encoder = _make_session(onnx_model_Encoder)
print(f"\nUsable Providers: {ort_session_Encoder.get_providers()}")
shape_value_in = ort_session_Encoder._inputs_meta[0].shape[-1]
in_name_Encoder = _in_names(ort_session_Encoder)
out_name_Encoder = _out_names(ort_session_Encoder)
in_name_Encoder0 = in_name_Encoder[0]
binding_Encoder = ort_session_Encoder.io_binding()

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
            f"Re-export with Export_Dolphin_CN_Dialect_Streaming.py to stamp the model metadata."
        )
    return int(value)


def _meta_int_list(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Dolphin_CN_Dialect_Streaming.py to stamp the model metadata."
        )
    return [int(x) for x in value.split(",") if x != ""]


SAMPLE_RATE = _meta_int("sample_rate")
METADATA_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
STREAM_CHUNK_FRAMES = _meta_int("stream_chunk_frames")
SUBSAMPLING_FACTOR = _meta_int("subsampling_factor")
SUBSAMPLING_CONTEXT = _meta_int("subsampling_context")
HOP_LENGTH = _meta_int("hop_length")
WINDOW_LENGTH = _meta_int("window_length")
SOS_TOKEN = _meta_int("sos_token_id")
EOS_TOKEN = _meta_int("eos_token_id")
ASR_TOKEN = _meta_int("asr_token_id")
NOTIMESTAMP = _meta_int("notimestamp_id")
STOP_TOKEN = _meta_int_list("stop_token_ids")
print(f"\nModel metadata: {len(_model_meta)} keys "
    f"(sample_rate={SAMPLE_RATE}, input_audio_length={METADATA_INPUT_AUDIO_LENGTH}, "
    f"stream_chunk_frames={STREAM_CHUNK_FRAMES}, window/hop={WINDOW_LENGTH}/{HOP_LENGTH}, "
    f"sos/eos/asr={SOS_TOKEN}/{EOS_TOKEN}/{ASR_TOKEN}, "
      f"notimestamp={NOTIMESTAMP}, stop_token_ids={STOP_TOKEN}).")

# Streaming encoder I/O (Export_Dolphin_CN_Dialect_Streaming.py): inputs = audio + per-layer att_k(L) + att_v(L) + csgu cnn(L);
# outputs = cross-attn K/V (2 * NUM_LAYER_DE) + new att_k(L) + att_v(L) + cnn(L) + ctc_ids. The att/cnn caches carry history
# chunk-to-chunk so each encoder call is O(1) per chunk; the cross-attn K/V are concatenated across chunks on the host.
num_layer_en = (len(in_name_Encoder) - 1) // 3                     # audio + att_k(L) + att_v(L) + cnn(L)
en_cache_dtype = _np_dtype(ort_session_Encoder._inputs_meta[1])    # in_att_k_0 dtype (f16)
en_att_k_in = in_name_Encoder[1:1 + num_layer_en]
en_att_v_in = in_name_Encoder[1 + num_layer_en:1 + 2 * num_layer_en]
en_cnn_in = in_name_Encoder[1 + 2 * num_layer_en:1 + 3 * num_layer_en]
num_cross = len(out_name_Encoder) - 3 * num_layer_en - 1           # cross K/V outputs (2 * NUM_LAYER_DE); last output is ctc_ids
num_cross_layers = num_cross // 2                                  # NUM_LAYER_DE
en_att_shape = (ort_session_Encoder._outputs_meta[num_cross].shape[0], ort_session_Encoder._inputs_meta[1].shape[2])  # (heads, d_k)
csgu_shape = (ort_session_Encoder._inputs_meta[1 + 2 * num_layer_en].shape[1], ort_session_Encoder._inputs_meta[1 + 2 * num_layer_en].shape[2])
stream_window_samples = (((STREAM_CHUNK_FRAMES - 1) * SUBSAMPLING_FACTOR + SUBSAMPLING_CONTEXT) - 1) * HOP_LENGTH + WINDOW_LENGTH
stream_stride_samples = STREAM_CHUNK_FRAMES * SUBSAMPLING_FACTOR * HOP_LENGTH

ort_session_Decoder = _make_session(onnx_model_Decoder)
in_name_Decoder = _in_names(ort_session_Decoder)
out_name_Decoder = _out_names(ort_session_Decoder)
amount_of_outputs_Decoder = len(out_name_Decoder)
binding_Decoder = ort_session_Decoder.io_binding()

num_layers = (amount_of_outputs_Decoder - 1) // 2          # outputs = decoder K/V caches (2 * L) + logits.
num_keys_values = num_layers + num_layers
idx_en_key = num_keys_values                          # decoder inputs: en cross-attn keys start (2 * L).
idx_en_value = idx_en_key + num_layers                # en cross-attn values start (3 * L).
idx_hidden = idx_en_value + num_layers                # token-embedding (hidden_states) input (4 * L).
idx_position = idx_hidden + 1                         # position-embedding input (4 * L + 1); mask is in_name_Decoder[-1].
out_name_Decoder_kv = out_name_Decoder[:num_keys_values]
out_name_Decoder_logits = out_name_Decoder[num_keys_values]
in_name_Decoder_self_kv = in_name_Decoder[:num_keys_values]         # decoder self-attn K/V cache feedback (greedy argmax).
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

tokenizer = Tokenizer(save_vocab)

# ---- Greedy argmax for the short attention pass (lang/region only; the transcript itself comes from CTC) ----
ort_session_Argmax = _make_session(onnx_model_Argmax)
in_name_Argmax = _in_names(ort_session_Argmax)
out_name_Argmax = _out_names(ort_session_Argmax)
binding_Argmax = ort_session_Argmax.io_binding()
in_name_Argmax_logits = in_name_Argmax[0]
out_name_Argmax_max = out_name_Argmax[0]

# ---- Fixed shared buffers (sized from model meta; the audio window reuses a per-clip buffer, except the short tail) ----
# Dolphin-CN-Dialect-Streaming is an auto-detect model: the decode prefix is just [sos] and the model
# auto-generates [lang, region, asr, notimestamp, text..., eos]. No prompt hotwords / forced language passes.
history_len_ort = _ort_from_data([0], np.int64)                    # history_len = 0 (each prefill starts fresh).
hidden_states_buf = _ort_zeros((1, 1, hidden_size), hidden_dtype)
position_buf = _ort_zeros((1, 1, hidden_size), position_dtype)
decode_mask_buf = _ort_zeros((1, 1, 1), mask_dtype)                # decode-phase mask is all-zeros (the new token sees every cached position).
prefill_logits_buf = _ort_zeros((1, vocab_size), logits_dtype)
decode_logits_buf = _ort_zeros((1, vocab_size), logits_dtype)
max_idx_buf = _ort_zeros((1, 1), np.int32)                        # next-token / stop-check buffer.

# Language mode: with both LANG_SYM+REGION_SYM the prefix pins [sos, <lang>, <region>, <asr>, <notimestamp>] so the
# model only generates text; otherwise the prefix is [sos] and the model auto-detects lang/region/asr/notimestamp.
lang_id_fixed = tokenizer.encode(f"<{LANG_SYM}>") if LANG_SYM else None
region_id_fixed = tokenizer.encode(f"<{REGION_SYM}>") if REGION_SYM else None
specify_language = lang_id_fixed is not None and region_id_fixed is not None
if (LANG_SYM or REGION_SYM) and not specify_language:
    print(f"\nLanguage mode: ignoring LANG_SYM={LANG_SYM!r}/REGION_SYM={REGION_SYM!r} (both must be valid tokens). Falling back to auto-detect.")
ts_tail = [NOTIMESTAMP]
lang_prefix = [lang_id_fixed, region_id_fixed, ASR_TOKEN, *ts_tail] if specify_language else []
prompt_ids_np = np.array([[SOS_TOKEN, *lang_prefix]], dtype=np.int32)
if specify_language:
    print(f"\nLanguage: forced {LANG_SYM}-{REGION_SYM}  ->  prefix ids {prompt_ids_np.tolist()[0]}")
prompt_len = prompt_ids_np.shape[-1]
prompt_ids_buf = _ort_zeros((1, prompt_len), np.int32)
prompt_ids_buf.update_inplace(prompt_ids_np)
prompt_ids_len_ort = _ort_from_data([prompt_len], np.int64)

# The decode prefix is constant across all clips/windows, so its embedding, position embedding, mask and
# kv_seq_len are computed ONCE here and reused every prefill (only kv_seq_len is reset, since Decode mutates it).
prompt_embed_buf = _ort_zeros((1, prompt_len, hidden_size), hidden_dtype)
binding_Embed.bind_ortvalue_input(in_name_Embed0, prompt_ids_buf)
binding_Embed.bind_ortvalue_output(out_name_Embed0, prompt_embed_buf)
_run(ort_session_Embed, binding_Embed)
prompt_pos_buf = _ort_zeros((1, prompt_len, hidden_size), position_dtype)
prompt_mask_buf = _ort_zeros((1, prompt_len, prompt_len), mask_dtype)
prompt_kv_len_buf = _ort_zeros((1,), np.int64)
_bind_inputs(binding_Prefill, in_name_Prefill, [prompt_ids_len_ort, history_len_ort])
binding_Prefill.bind_ortvalue_output(out_name_Prefill[0], prompt_pos_buf)
binding_Prefill.bind_ortvalue_output(out_name_Prefill[1], prompt_mask_buf)
binding_Prefill.bind_ortvalue_output(out_name_Prefill[2], prompt_kv_len_buf)
_run(ort_session_Prefill, binding_Prefill)
prompt_kv_seq_len = np.ascontiguousarray(prompt_kv_len_buf.numpy())   # reset value for kv_seq_len each window.
kv_seq_len_ort = _ort_zeros((1,), np.int64)

# Empty decoder self-attention caches (batch 1) reused for every prefill pass.
past_keys_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[0].shape[1], ort_session_Decoder._outputs_meta[0].shape[2], 0), kv_cache_dtype)
past_values_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[num_layers].shape[1], 0, ort_session_Decoder._outputs_meta[num_layers].shape[3]), kv_cache_dtype)

def build_text(save_token_array):
    # Auto mode: tokens = [lang, region, asr, notimestamp, text...]; forced mode: text only (lang/region in prefix).
    toks = []
    for idx in save_token_array:
        if idx in STOP_TOKEN:
            break
        toks.append(int(idx))
    if specify_language:
        lang_s, region_s, text_ids = LANG_SYM, REGION_SYM, toks
    else:
        lang_s = tokenizer.decode(toks[0]).strip("<>") if len(toks) > 0 else "?"
        region_s = tokenizer.decode(toks[1]).strip("<>") if len(toks) > 1 else "?"
        text_ids = toks[4:]
    return lang_s, region_s, tokenizer.decode_ids(text_ids), len(text_ids)


def ctc_collapse(ids, blank_id=0):
    # dolphin remove_duplicates_and_blank: drop blanks + merge consecutive repeats. Special tokens (<=NOTIMESTAMP) skipped.
    out, prev = [], None
    for i in ids:
        i = int(i)
        if i != prev and i != blank_id and i > NOTIMESTAMP:
            out.append(i)
        prev = i
    return out


# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    audio_len = len(audio)
    audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)                  # full clip -> target dtype, RMS once
    INPUT_AUDIO_LENGTH = min(shape_value_in, audio_len) if isinstance(shape_value_in, int) else min(METADATA_INPUT_AUDIO_LENGTH, audio_len)
    audio = audio[:, :, :INPUT_AUDIO_LENGTH]                              # cap at the 30 s encoder limit; longer clips need a real cache.
    text = ""
    lang_str = region_str = "?"
    n_text = 0
    # Encoder self-caches (att K/V grow all-history; the csgu cnn state is a fixed lorder window) live ON-DEVICE
    # and are chained output->input each chunk (Nemotron / Paraformer-Streaming IOBinding pattern), so they never
    # round-trip through host. Only the per-chunk cross-attn K/V and CTC ids are copied to host (accumulated / collapsed).
    att_k_ort = [_ort_zeros((en_att_shape[0], 0, en_att_shape[1]), en_cache_dtype) for _ in range(num_layer_en)]
    att_v_ort = [_ort_zeros((en_att_shape[0], 0, en_att_shape[1]), en_cache_dtype) for _ in range(num_layer_en)]
    cnn_ort = [_ort_zeros((1, csgu_shape[0], csgu_shape[1]), en_cache_dtype) for _ in range(num_layer_en)]
    cross_k = [None] * num_cross_layers                                  # accumulated cross-attn key  (h, d, T)
    cross_v = [None] * num_cross_layers                                  # accumulated cross-attn value (h, T, d)
    ctc_ids_all = []                                                      # all CTC frame ids so far (for stable partial text)
    chunk_ends = list(range(stream_window_samples, INPUT_AUDIO_LENGTH, stream_stride_samples)) + [INPUT_AUDIO_LENGTH]
    win_start = 0
    # Reuse one pre-allocated buffer for every full-width streaming window (update_inplace keeps the bound
    # encoder input zero-copy); only the short tail window (< stream_window_samples) needs a one-off OrtValue.
    en_audio_buffer = _ort_zeros((1, 1, stream_window_samples), audio.dtype)
    # Start to run Dolphin-CN-Dialect-Streaming
    start_time = time.time()
    for chunk_no, slice_end in enumerate(chunk_ends):
        is_final_chunk = chunk_no == len(chunk_ends) - 1
        # ---- Encoder: ONE chunk; per-layer K/V + csgu caches carry history so this is O(N), not a full re-run ----
        en_chunk = np.ascontiguousarray(audio[:, :, win_start: slice_end])
        win_start += stream_stride_samples
        if en_chunk.shape[-1] == stream_window_samples:
            en_audio_buffer.update_inplace(en_chunk)                        # full-width window: refresh the shared buffer in place.
            en_audio = en_audio_buffer
        else:
            en_audio = _ort_from_numpy(en_chunk)                           # short tail window (variable length): one-off OrtValue.
        _bind_inputs(binding_Encoder, [in_name_Encoder0], [en_audio])
        _bind_inputs(binding_Encoder, en_att_k_in, att_k_ort)
        _bind_inputs(binding_Encoder, en_att_v_in, att_v_ort)
        _bind_inputs(binding_Encoder, en_cnn_in, cnn_ort)
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
        _run(ort_session_Encoder, binding_Encoder)
        en_out = binding_Encoder.get_outputs()
        for i in range(num_cross_layers):
            ck = en_out[i].numpy()
            cv = en_out[num_cross_layers + i].numpy()
            cross_k[i] = ck if cross_k[i] is None else np.concatenate((cross_k[i], ck), axis=2)
            cross_v[i] = cv if cross_v[i] is None else np.concatenate((cross_v[i], cv), axis=1)
        # Chain the self-caches on-device: keep the fresh output OrtValues (the io_binding holds them alive) and feed
        # them straight into the next chunk's inputs, replacing the old device->host (.numpy) -> device (_ort_from_numpy) hop.
        att_k_ort = en_out[num_cross: num_cross + num_layer_en]
        att_v_ort = en_out[num_cross + num_layer_en: num_cross + 2 * num_layer_en]
        cnn_ort = en_out[num_cross + 2 * num_layer_en: num_cross + 3 * num_layer_en]

        # ---- Stable streaming partial: CTC greedy collapse over all frames so far (cheap, monotonic, no re-decode) ----
        ctc_ids_all.extend(en_out[-1].numpy()[0].tolist())
        partial_ids = ctc_collapse(ctc_ids_all)
        text = tokenizer.decode_ids(partial_ids)
        n_text = len(partial_ids)
        if not is_final_chunk:
            print(f"  [partial {chunk_ends[chunk_no] / SAMPLE_RATE:5.2f}s] {text}")
            continue

        # ---- Final text = stable CTC; attention runs only to confirm lang/region (capped, no full re-decode) ----
        en_kv = [_ort_from_numpy(np.ascontiguousarray(c)) for c in cross_k] + [_ort_from_numpy(np.ascontiguousarray(c)) for c in cross_v]
        _bind_inputs(binding_Decoder, in_name_Decoder_en_kv, en_kv)

        # ---- Prefill: [sos] (embed/position/mask precomputed once, reused) ----
        # Attention runs only to confirm lang/region (auto: lang,region,asr,notimestamp); text comes from CTC, so cap short.
        generate_limit = 4
        _bind_inputs(binding_Decoder, in_name_Decoder[:num_layers], [past_keys_Decoder] * num_layers)
        _bind_inputs(binding_Decoder, in_name_Decoder[num_layers:num_keys_values], [past_values_Decoder] * num_layers)

        kv_seq_len_ort.update_inplace(prompt_kv_seq_len)               # reset (Decode increments kv_seq_len in place each step).
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_hidden, prompt_embed_buf)
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_position, prompt_pos_buf)
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_mask, prompt_mask_buf)
        _bind_device_outputs(binding_Decoder, out_name_Decoder_kv)
        binding_Decoder.bind_ortvalue_output(out_name_Decoder_logits, prefill_logits_buf)

        # Decode-phase position graph: kv_seq_len is incremented in place (input aliased to output).
        binding_Decode.bind_ortvalue_input(in_name_Decode0, kv_seq_len_ort)
        binding_Decode.bind_ortvalue_output(out_name_Decode_position, position_buf)
        binding_Decode.bind_ortvalue_output(out_name_Decode_kv_seq_len, kv_seq_len_ort)

        # Decode-phase embedding graph: next token -> shared hidden buffer.
        binding_Embed.bind_ortvalue_input(in_name_Embed0, max_idx_buf)
        binding_Embed.bind_ortvalue_output(out_name_Embed0, hidden_states_buf)

        save_id_greedy = np.zeros(generate_limit, dtype=np.int32)          # host-side token history for the short lang/region pass.

        num_decode = 0
        is_prefill_step = True
        while num_decode < generate_limit:
            _run(ort_session_Decoder, binding_Decoder)
            outputs_Decoder = binding_Decoder.get_outputs()
            cur_logits_buf = prefill_logits_buf if is_prefill_step else decode_logits_buf

            binding_Argmax.bind_ortvalue_input(in_name_Argmax_logits, cur_logits_buf)
            binding_Argmax.bind_ortvalue_output(out_name_Argmax_max, max_idx_buf)
            _run(ort_session_Argmax, binding_Argmax)

            max_logits_idx = max_idx_buf.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN:
                break
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

        save_token_array = save_id_greedy[:num_decode]
        lang_str, region_str, _, _ = build_text(save_token_array)          # attention prefix -> lang/region (reliable first 2 tokens)
        print(f"  [FINAL {chunk_ends[chunk_no] / SAMPLE_RATE:5.2f}s | {lang_str}-{region_str}] {text}")

    count_time = time.time() - start_time
    rtf = count_time / max(audio_len / SAMPLE_RATE, 1e-6)
    print(f"\nDetected: {lang_str}-{region_str}")
    print(f"\nASR Result:\n{text}\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, {n_text} text tokens)")
    print("----------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------")
