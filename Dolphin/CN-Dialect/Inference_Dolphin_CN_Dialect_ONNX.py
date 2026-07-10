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
    parser = argparse.ArgumentParser(description="Run Dolphin-CN-Dialect ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", default=os.path.join(_SCRIPT_DIR, "Dolphin_CN_Dialect_Optimized"), help="Folder containing ONNX graphs, for example Dolphin_CN_Dialect_Optimized or Dolphin_CN_Dialect_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()


onnx_folder            = os.path.abspath(_ARGS.onnx_folder)                          # Selected ONNX graph folder.
save_vocab             = os.path.join(onnx_folder, "vocab_Dolphin_CN_Dialect.txt")    # The exported Dolphin-CN-Dialect vocab path.
onnx_model_Metadata    = os.path.join(onnx_folder, "ASR_Matadata.onnx")                             # Tiny metadata carrier graph.
onnx_model_Encoder     = os.path.join(onnx_folder, "Dolphin_Encoder.onnx")                              # The exported onnx encoder model path.
onnx_model_Decoder     = os.path.join(onnx_folder, "Dolphin_Decoder.onnx")                              # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = os.path.join(onnx_folder, "Dolphin_Decoder_Embed.onnx")                        # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = os.path.join(onnx_folder, "Dolphin_Position_Mask_Prefill.onnx")                # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = os.path.join(onnx_folder, "Dolphin_Position_Mask_Decode.onnx")                 # Decode position-embedding graph for the single new token.
onnx_model_Greedy      = os.path.join(onnx_folder, "Dolphin_Greedy_Search.onnx")                        # Greedy argmax + save_id history (used with Apply_Penality).
onnx_model_Argmax      = os.path.join(onnx_folder, "Dolphin_Argmax.onnx")                               # Bare argmax (greedy decoding without a repetition penalty).
onnx_model_First_Beam  = os.path.join(onnx_folder, "Dolphin_First_Beam_Search.onnx")                    # First beam-search step.
onnx_model_Second_Beam = os.path.join(onnx_folder, "Dolphin_Second_Beam_Search.onnx")                   # Subsequent beam-search steps.
onnx_model_Penality    = os.path.join(onnx_folder, "Dolphin_Apply_Penality.onnx")                       # Sliding-window repetition penalty on the logits.


test_audio = model_audio_paths("dolphin_cn_dialect")                                                     # The test audio list.


USE_BEAM_SEARCH    = False              # Use beam search or greedy search.
REPEAT_PENALITY    = 1.0                # Range from 0.0 to 1.0; "1.0" means no penality (the Dolphin reference decoder applies none).
PENALITY_RANGE     = 20                 # Penalizes the most recent output. "20" means the last 20 tokens.
TOP_K              = 3                  # The top k candidate in decoding.
BEAM_SIZE          = 3                  # Number of beams in searching.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO = False             # Apply RMS loudness normalisation before feeding the model. The reference Dolphin pipeline keeps the decoded waveform amplitude unchanged.
SLIDING_WINDOW     = 0                  # Set the sliding window step for test audio reading; use 0 to disable.
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
HOTWORDS           = ["开饭时间"]         # Prompt-based hotwords (small.cn.prompt). Each word is char-tokenised and packed between PROMPT_START/PROMPT_END; use [] to disable biasing.


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
# "float16"/"float" carry int16-range values with no ÷32768).
_audio_input_type = ort_session_Encoder._inputs_meta[0].type
input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")
_audio_np_dtype = {"INT16": np.int16, "F32": np.float32, "F16": np.float16}[input_audio_dtype]   # sliding-window buffer dtype

# ---- Model metadata: the SOLE source of the runtime geometry / token constants ----
# Written by Export_Dolphin_CN_Dialect.py into ASR_Matadata.onnx; read here so the special-token IDs,
# max_seq_len and sample_rate never have to be kept in sync by hand. A missing key is a hard error
# (re-export with Export_Dolphin_CN_Dialect.py to stamp the metadata) -- there is no compiled-in fallback.
_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_Dolphin_CN_Dialect.py to stamp the model metadata."
        )
    return int(value)

MAX_SEQ_LEN  = _meta_int("max_seq_len")
SAMPLE_RATE  = _meta_int("sample_rate")
METADATA_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
SOS_TOKEN    = _meta_int("sos_token_id")
EOS_TOKEN    = _meta_int("eos_token_id")
ASR_TOKEN    = _meta_int("asr_token_id")
PROMPT_START = _meta_int("prompt_start_id")
PROMPT_END   = _meta_int("prompt_end_id")
NOTIMESTAMP  = _meta_int("notimestamp_id")
STOP_TOKEN   = [EOS_TOKEN]              # end token derived from the model's eos_token_id metadata.
print(f"\nModel metadata: {len(_model_meta)} keys "
      f"(max_seq_len={MAX_SEQ_LEN}, sample_rate={SAMPLE_RATE}, "
    f"input_audio_length={METADATA_INPUT_AUDIO_LENGTH}, "
      f"sos/eos/asr={SOS_TOKEN}/{EOS_TOKEN}/{ASR_TOKEN}, "
      f"prompt_start/end={PROMPT_START}/{PROMPT_END}, notimestamp={NOTIMESTAMP}).")

ort_session_Decoder = _make_session(onnx_model_Decoder)
in_name_Decoder = _in_names(ort_session_Decoder)
out_name_Decoder = _out_names(ort_session_Decoder)
amount_of_outputs_Decoder = len(out_name_Decoder)
binding_Decoder = ort_session_Decoder.io_binding()

num_layers = (amount_of_outputs_Decoder - 1) // 2          # outputs = decoder K/V caches (2 * L) + logits.
num_keys_values = num_layers + num_layers
# Beam search I/O indices (Qwen ASR layout). Inputs: caches(2L), logits, save_id[, prev_prob], beam_size[, topK].
# Outputs: caches(2L), save_id, top_beam_prob, top_beam_indices (next token), max_logits_idx.
beam_logits_in_idx   = num_keys_values                # logits input to First/Second beam.
beam_save_id_in_idx  = num_keys_values + 1            # save_id input to First/Second beam.
first_beam_size_idx  = num_keys_values + 2            # beam_size input (First beam).
second_prev_prob_idx = num_keys_values + 2            # previous_prob input (Second beam).
second_beam_size_idx = num_keys_values + 3            # beam_size input (Second beam).
second_topk_idx      = num_keys_values + 4            # topK input (Second beam).
beam_save_id_out_idx = num_keys_values                # save_id output.
beam_prob_out_idx    = num_keys_values + 1            # top_beam_prob output.
beam_ids_out_idx     = num_keys_values + 2            # top_beam_indices output (next token to embed).
beam_max_out_idx     = num_keys_values + 3            # max_logits_idx output (stop check).
idx_en_key = num_keys_values                          # decoder inputs: en cross-attn keys start (2 * L).
idx_en_value = idx_en_key + num_layers                # en cross-attn values start (3 * L).
idx_hidden = idx_en_value + num_layers                # token-embedding (hidden_states) input (4 * L).
idx_position = idx_hidden + 1                         # position-embedding input (4 * L + 1); mask is in_name_Decoder[-1].
out_name_Decoder_kv = out_name_Decoder[:num_keys_values]
out_name_Decoder_logits = out_name_Decoder[num_keys_values]
in_name_Decoder_self_kv = in_name_Decoder[:num_keys_values]         # decoder self-attn K/V cache feedback (beam / greedy).
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

# ---- Decoding-strategy resolution ----
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE
if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")
if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1
decode_batch = BEAM_SIZE
do_repeat_penality = REPEAT_PENALITY != 1.0

beam_size_ort = _ort_from_data([BEAM_SIZE], np.int64)
topK_ort = _ort_from_data([TOP_K], np.int64)

if USE_BEAM_SEARCH:
    ort_session_First_Beam = _make_session(onnx_model_First_Beam)
    in_name_First_Beam = _in_names(ort_session_First_Beam)
    out_name_First_Beam = _out_names(ort_session_First_Beam)
    binding_First_Beam = ort_session_First_Beam.io_binding()

    ort_session_Second_Beam = _make_session(onnx_model_Second_Beam)
    in_name_Second_Beam = _in_names(ort_session_Second_Beam)
    out_name_Second_Beam = _out_names(ort_session_Second_Beam)
    binding_Second_Beam = ort_session_Second_Beam.io_binding()

    # Pre-slice the beam I/O names once so the hot decode loop never re-slices these lists.
    in_name_First_Beam_kv = in_name_First_Beam[:num_keys_values]
    out_name_First_Beam_kv = out_name_First_Beam[:num_keys_values]
    in_name_First_Beam_logits = in_name_First_Beam[beam_logits_in_idx]
    in_name_First_Beam_save_id = in_name_First_Beam[beam_save_id_in_idx]
    out_name_First_Beam_save_id = out_name_First_Beam[beam_save_id_out_idx]
    out_name_First_Beam_prob = out_name_First_Beam[beam_prob_out_idx]
    out_name_First_Beam_ids = out_name_First_Beam[beam_ids_out_idx]
    out_name_First_Beam_max = out_name_First_Beam[beam_max_out_idx]
    in_name_Second_Beam_kv = in_name_Second_Beam[:num_keys_values]
    out_name_Second_Beam_kv = out_name_Second_Beam[:num_keys_values]
    in_name_Second_Beam_logits = in_name_Second_Beam[beam_logits_in_idx]
    in_name_Second_Beam_save_id = in_name_Second_Beam[beam_save_id_in_idx]
    in_name_Second_Beam_prev_prob = in_name_Second_Beam[second_prev_prob_idx]
    out_name_Second_Beam_save_id = out_name_Second_Beam[beam_save_id_out_idx]
    out_name_Second_Beam_prob = out_name_Second_Beam[beam_prob_out_idx]
    out_name_Second_Beam_ids = out_name_Second_Beam[beam_ids_out_idx]
    out_name_Second_Beam_max = out_name_Second_Beam[beam_max_out_idx]

    prob_dtype = _np_dtype(ort_session_First_Beam._outputs_meta[beam_prob_out_idx])

    # Shared beam buffers (fixed shape, reused every step).
    beam_ids_buf = _ort_zeros((BEAM_SIZE, 1), np.int32)              # top_beam_indices (next token to embed).
    beam_score_buf = _ort_zeros((BEAM_SIZE, 1), prob_dtype)         # top_beam_prob / previous_prob (self-aliased).

    # Static beam inputs (bound once).
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[first_beam_size_idx], beam_size_ort)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_size_idx], beam_size_ort)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_topk_idx], topK_ort)
else:
    ort_session_Greedy = _make_session(onnx_model_Greedy)
    in_name_Greedy = _in_names(ort_session_Greedy)
    out_name_Greedy = _out_names(ort_session_Greedy)
    binding_Greedy = ort_session_Greedy.io_binding()
    in_name_Greedy_logits = in_name_Greedy[0]
    in_name_Greedy_save_id = in_name_Greedy[1]
    out_name_Greedy_max = out_name_Greedy[0]
    out_name_Greedy_save_id = out_name_Greedy[1]

    ort_session_Argmax = _make_session(onnx_model_Argmax)
    in_name_Argmax = _in_names(ort_session_Argmax)
    out_name_Argmax = _out_names(ort_session_Argmax)
    binding_Argmax = ort_session_Argmax.io_binding()
    in_name_Argmax_logits = in_name_Argmax[0]
    out_name_Argmax_max = out_name_Argmax[0]

# Repetition penalty is a standalone pass applied to the logits before greedy / beam selection (Qwen ASR style).
if do_repeat_penality:
    ort_session_Penality = _make_session(onnx_model_Penality)
    in_name_Penality = _in_names(ort_session_Penality)
    out_name_Penality = _out_names(ort_session_Penality)[0]
    binding_Penality = ort_session_Penality.io_binding()
    in_name_Penality_logits = in_name_Penality[0]
    in_name_Penality_save_id = in_name_Penality[1]
    num_penality_inputs = len(in_name_Penality)
    # penality_range is baked into the graph via int(), so ORT may prune it as a dead input; guard the binds.
    if num_penality_inputs > 2:
        penality_value_dtype = _np_dtype(ort_session_Penality._inputs_meta[2])
        penality_value_ort = _ort_from_data([REPEAT_PENALITY], penality_value_dtype)
        binding_Penality.bind_ortvalue_input(in_name_Penality[2], penality_value_ort)
    if num_penality_inputs > 3:
        penality_range_ort = _ort_from_data([PENALITY_RANGE], np.int64)
        binding_Penality.bind_ortvalue_input(in_name_Penality[3], penality_range_ort)

# ---- Fixed shared buffers (sized from model meta; the audio window is a fresh OrtValue per window) ----
# Dolphin-CN-Dialect is a prompt model: the decode prefix is [sos, PROMPT_START, hotwords..., PROMPT_END] and the
# model runs with no timestamps. No explicit language/region detection passes.
history_len_ort = _ort_from_data([0], np.int64)                    # history_len = 0 (each prefill starts fresh).
hidden_states_buf = _ort_zeros((decode_batch, 1, hidden_size), hidden_dtype)
position_buf = _ort_zeros((1, 1, hidden_size), position_dtype)
decode_mask_buf = _ort_zeros((1, 1, 1), mask_dtype)                # decode-phase mask is all-zeros (the new token sees every cached position).
prefill_logits_buf = _ort_zeros((1, vocab_size), logits_dtype)
decode_logits_buf = _ort_zeros((decode_batch, vocab_size), logits_dtype)
max_idx_buf = _ort_zeros((1, 1), np.int32)                        # next-token (greedy) / stop-check (beam) buffer.

# Prompt-based hotword biasing (dolphin.transcribe._prepare_prompt_hotwords): each hotword is split char-level and
# the ids are concatenated (no separator) between PROMPT_START and PROMPT_END, e.g. "达摩院" -> [PROMPT_START,达,摩,院,PROMPT_END].
hotword_ids = [cid for word in HOTWORDS for ch in word for cid in (tokenizer.encode(ch),) if cid is not None]
# Language mode: with both LANG_SYM+REGION_SYM the prefix pins [<lang>, <region>, <asr>, <notimestamp>] so the model
# only generates text; otherwise the model auto-detects lang/region before text generation.
lang_id_fixed = tokenizer.encode(f"<{LANG_SYM}>") if LANG_SYM else None
region_id_fixed = tokenizer.encode(f"<{REGION_SYM}>") if REGION_SYM else None
specify_language = lang_id_fixed is not None and region_id_fixed is not None
if (LANG_SYM or REGION_SYM) and not specify_language:
    print(f"\nLanguage mode: ignoring LANG_SYM={LANG_SYM!r}/REGION_SYM={REGION_SYM!r} (both must be valid tokens). Falling back to auto-detect.")
lang_prefix = [lang_id_fixed, region_id_fixed, ASR_TOKEN, NOTIMESTAMP] if specify_language else []
prompt_ids_np = np.array([[SOS_TOKEN, PROMPT_START, *hotword_ids, PROMPT_END, *lang_prefix]], dtype=np.int32)
if hotword_ids:
    print(f"\nPrompt hotwords: {HOTWORDS}  ->  prompt ids {prompt_ids_np.tolist()[0]}")
if specify_language:
    print(f"\nLanguage: forced {LANG_SYM}-{REGION_SYM}  ->  prefix ids {prompt_ids_np.tolist()[0]}")
prompt_len = prompt_ids_np.shape[-1]
prompt_ids_buf = _ort_zeros((1, prompt_len), np.int32)
prompt_ids_buf.update_inplace(prompt_ids_np)
prompt_ids_len_ort = _ort_from_data([prompt_len], np.int64)

# The hotword prompt is constant across all clips/windows, so its embedding, position embedding, mask and
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

# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    audio_len = len(audio)
    audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(METADATA_INPUT_AUDIO_LENGTH, audio_len)
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
        audio = np.concatenate((audio, np.zeros((1, 1, pad_amount), dtype=audio.dtype)), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio = np.concatenate((audio, np.zeros((1, 1, INPUT_AUDIO_LENGTH - audio_len), dtype=audio.dtype)), axis=-1)
    aligned_len = audio.shape[-1]

    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    num_decode = 0
    generated_tokens = []
    audio_buffer = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), _audio_np_dtype)
    # Start to run Dolphin-CN-Dialect
    start_time = time.time()
    while slice_end <= aligned_len:
        # ---- Encoder (per-clip shared buffer keeps Dolphin's exact dynamic input length while avoiding per-window OrtValue churn) ----
        audio_buffer.update_inplace(np.ascontiguousarray(audio[:, :, slice_start: slice_end]))
        binding_Encoder.bind_ortvalue_input(in_name_Encoder0, audio_buffer)
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
        _run(ort_session_Encoder, binding_Encoder)
        en_kv = binding_Encoder.get_outputs()                              # f16 cross-attn K/V, valid for the whole window.
        _bind_inputs(binding_Decoder, in_name_Decoder_en_kv, en_kv)

        # ---- Prompt prefill: [sos, PROMPT_START, hotwords..., PROMPT_END] (embed/position/mask precomputed once, reused) ----
        generate_limit = MAX_SEQ_LEN - prompt_len
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
        if USE_BEAM_SEARCH:
            binding_Embed.bind_ortvalue_input(in_name_Embed0, beam_ids_buf)
        else:
            binding_Embed.bind_ortvalue_input(in_name_Embed0, max_idx_buf)
        binding_Embed.bind_ortvalue_output(out_name_Embed0, hidden_states_buf)

        if USE_BEAM_SEARCH:
            save_id_buf = _ort_zeros((BEAM_SIZE, 0), np.int32)             # initial empty per-beam history.
            save_id = save_id_buf                                          # current on-device save_id (penalty / feedback).
            latest_save_id = save_id_buf                                   # last beam save_id for final detokenisation.
            binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[beam_save_id_in_idx], save_id_buf)
        else:
            save_id = _ort_zeros((1, 0), np.int32)                         # on-device greedy history (used when penalty is on).
            save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)         # host-side history (used by the argmax path).

        num_decode = 0
        is_prefill_step = True
        while num_decode < generate_limit:
            _run(ort_session_Decoder, binding_Decoder)
            outputs_Decoder = binding_Decoder.get_outputs()
            cur_logits_buf = prefill_logits_buf if is_prefill_step else decode_logits_buf

            # Repetition penalty: a standalone in-place pass over the most recent tokens (Qwen ASR style).
            if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                binding_Penality.bind_ortvalue_input(in_name_Penality_logits, cur_logits_buf)
                binding_Penality.bind_ortvalue_input(in_name_Penality_save_id, save_id)
                binding_Penality.bind_ortvalue_output(out_name_Penality, cur_logits_buf)
                _run(ort_session_Penality, binding_Penality)

            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    _bind_inputs(binding_First_Beam, in_name_First_Beam_kv, outputs_Decoder[:num_keys_values])
                    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam_logits, cur_logits_buf)
                    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam_save_id, save_id)
                    # Bind outputs in graph order; get_outputs() returns values in bind order, so the order must match the graph.
                    _bind_device_outputs(binding_First_Beam, out_name_First_Beam_kv)                                         # caches 0..2L-1
                    binding_First_Beam._iobinding.bind_output(out_name_First_Beam_save_id, _ort_device_obj)                  # save_id
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam_prob, beam_score_buf)                       # top_beam_prob
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam_ids, beam_ids_buf)                          # top_beam_indices (next token)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam_max, max_idx_buf)                           # max_logits_idx (stop check)
                    _run(ort_session_First_Beam, binding_First_Beam)
                    outputs_Beam = binding_First_Beam.get_outputs()
                else:
                    _bind_inputs(binding_Second_Beam, in_name_Second_Beam_kv, outputs_Decoder[:num_keys_values])
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_logits, cur_logits_buf)
                    if num_decode < 2:
                        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_prev_prob, beam_score_buf)
                    # Bind outputs in graph order every step; get_outputs() returns values in bind order.
                    _bind_device_outputs(binding_Second_Beam, out_name_Second_Beam_kv)                                       # caches 0..2L-1
                    binding_Second_Beam._iobinding.bind_output(out_name_Second_Beam_save_id, _ort_device_obj)                # save_id
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam_prob, beam_score_buf)                     # top_beam_prob
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam_ids, beam_ids_buf)                        # top_beam_indices (next token)
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam_max, max_idx_buf)                         # max_logits_idx (stop check)
                    _run(ort_session_Second_Beam, binding_Second_Beam)
                    outputs_Beam = binding_Second_Beam.get_outputs()

                save_id = outputs_Beam[beam_save_id_out_idx]
                latest_save_id = save_id
                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN:
                    break

                _bind_inputs(binding_Decoder, in_name_Decoder_self_kv, outputs_Beam[:num_keys_values])
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam_save_id, save_id)
            else:
                if do_repeat_penality:
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy_logits, cur_logits_buf)
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy_save_id, save_id)
                    binding_Greedy.bind_ortvalue_output(out_name_Greedy_max, max_idx_buf)
                    binding_Greedy._iobinding.bind_output(out_name_Greedy_save_id, _ort_device_obj)
                    _run(ort_session_Greedy, binding_Greedy)
                    save_id = binding_Greedy.get_outputs()[1]
                else:
                    binding_Argmax.bind_ortvalue_input(in_name_Argmax_logits, cur_logits_buf)
                    binding_Argmax.bind_ortvalue_output(out_name_Argmax_max, max_idx_buf)
                    _run(ort_session_Argmax, binding_Argmax)

                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN:
                    break
                if not do_repeat_penality:
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
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    rtf = count_time / max(audio_len / SAMPLE_RATE, 1e-6)

    if USE_BEAM_SEARCH:
        save_token_array = latest_save_id.numpy()[0]
    elif do_repeat_penality:
        # Greedy with penalty keeps its token history on-device (GREEDY_SEARCH appends each step).
        save_token_array = save_id.numpy()[0]
    else:
        save_token_array = save_id_greedy[:num_decode]

    generated_tokens = []
    for idx in save_token_array:
        if idx in STOP_TOKEN:
            break
        generated_tokens.append(int(idx))

    # Auto mode emits 4 control tokens before text; forced mode puts those controls in the prefix, so generated ids are text only.
    if specify_language:
        lang_str, region_str = LANG_SYM, REGION_SYM
        text_start = 0
    else:
        lang_str = tokenizer.decode(generated_tokens[0]).strip("<>") if len(generated_tokens) > 0 else "?"
        region_str = tokenizer.decode(generated_tokens[1]).strip("<>") if len(generated_tokens) > 1 else "?"
        text_start = 4
    text_ids = generated_tokens[text_start:]
    text = tokenizer.decode_ids(text_ids)
    print(f"\nDetected: {lang_str}-{region_str}")
    print(f"\nASR Result:\n{text}\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, {len(text_ids)} text tokens)")
    print("----------------------------------------------------------------------------------------------------------")
