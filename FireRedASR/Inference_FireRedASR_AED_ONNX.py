import argparse
import re
import time
import logging
import sys
import numpy as np
import onnxruntime
import sentencepiece as spm
from onnxruntime.capi import _pybind_state as C
from pathlib import Path
from pydub import AudioSegment


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_paths


def _parse_args():
    parser = argparse.ArgumentParser(description="Run FireRedASR AED ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "FireRedASR_Optimized", help="Folder containing ONNX graphs, for example FireRedASR_Optimized or FireRedASR_ONNX. The tokenizer assets (dict.txt + train_bpe1000.model) are read from this same folder.")
    return parser.parse_args()


_ARGS = _parse_args()


# -- Optimized ONNX graph paths (Optimize_ONNX.py output): core pipeline (Embed keeps token ids out of the float decoder; Prefill / Decode build position embedding + causal mask) --
onnx_folder            = _ARGS.onnx_folder.expanduser().resolve()                # Selected ONNX graph folder (also holds the bundled tokenizer assets).

onnx_model_Metadata    = f"{onnx_folder}/FireRedASR_Metadata.onnx"               # Tiny metadata carrier graph.
onnx_model_Encoder     = f"{onnx_folder}/FireRedASR_Encoder.onnx"                # The exported ONNX encoder model path.
onnx_model_Decoder     = f"{onnx_folder}/FireRedASR_Decoder.onnx"                # The exported ONNX decoder (main, pure-float) model path.
onnx_model_Embed       = f"{onnx_folder}/FireRedASR_Decoder_Embed.onnx"          # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = f"{onnx_folder}/FireRedASR_Position_Mask_Prefill.onnx"  # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = f"{onnx_folder}/FireRedASR_Position_Mask_Decode.onnx"   # Decode position-embedding graph for the single new token.
onnx_model_Greedy      = f"{onnx_folder}/FireRedASR_Greedy_Search.onnx"          # Greedy argmax + save_id history (used with Apply_Penality).
onnx_model_Argmax      = f"{onnx_folder}/FireRedASR_Argmax.onnx"                 # Bare argmax (greedy decoding without a repetition penalty).
onnx_model_First_Beam  = f"{onnx_folder}/FireRedASR_First_Beam_Search.onnx"      # First beam-search step.
onnx_model_Second_Beam = f"{onnx_folder}/FireRedASR_Second_Beam_Search.onnx"     # Subsequent beam-search steps.
onnx_model_Penality    = f"{onnx_folder}/FireRedASR_Apply_Penality.onnx"         # Sliding-window repetition penalty on the logits.


test_audio = model_audio_paths("fireredasr")  # The test audio list.


USE_BEAM_SEARCH     = False   # Use beam search or greedy search.
DECODE_MAX_LEN      = 0       # Match original AED: 0 means use encoder time length.
REPEAT_PENALITY     = 1.0     # Original FireRedASR-AED uses no repetition penalty; keep 1.0 for parity.
PENALITY_RANGE      = 10      # Penalizes the most recent output. "30" means the last 30 tokens.
TOP_K               = 3       # The top k candidate in decoding.
BEAM_SIZE           = 3       # Number of beams in searching.
SLIDING_WINDOW      = 0       # Set the sliding window step for test audio reading; use 0 to disable.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# (kaldi fbank keeps the int16 numeric range, so "float16"/"float" carry int16-range values, no ÷32768); no manual setting needed.
USE_NORMALISE_AUDIO = False   # RMS-normalize the input audio before inference; False = feed raw int16 PCM (matches the original kaldi_native_fbank pipeline).

# ---- ONNX Runtime runtime configuration (IOBinding + shared buffers) ----
ORT_Accelerate_Providers = []     # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                  = False  # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16                 = False  # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS              = 0      # Parallel CPU threads. Set 0 for auto.
DEVICE_ID                = 0      # Default to zero.


print('\nLoading the FireRedASR models...')


# ── Tokenizer (inlined from the FireRedASR project so this script is standalone) ──
# Copied from fireredasr/data/token_dict.py (TokenDict) and
# fireredasr/tokenizer/aed_tokenizer.py (ChineseCharEnglishSpmTokenizer).
class TokenDict:
    def __init__(self, dict_path, unk=""):
        assert dict_path != ""
        self.id2word, self.word2id = self.read_dict(dict_path)
        self.unk = unk
        assert unk == "" or unk in self.word2id
        self.unkid = self.word2id[unk] if unk else -1

    def get(self, key, default):
        if type(default) == str:
            default = self.word2id[default]
        return self.word2id.get(key, default)

    def __getitem__(self, key):
        if type(key) == str:
            if self.unk:
                return self.word2id.get(key, self.word2id[self.unk])
            else:
                return self.word2id[key]
        elif type(key) == int:
            return self.id2word[key]
        else:
            raise TypeError("Key should be str or int")

    def __len__(self):
        return len(self.id2word)

    def __contains__(self, query):
        if type(query) == str:
            return query in self.word2id
        elif type(query) == int:
            return query in self.id2word
        else:
            raise TypeError("query should be str or int")

    def read_dict(self, dict_path):
        id2word, word2id = [], {}
        with open(dict_path, encoding='utf8') as f:
            for i, line in enumerate(f):
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    word, index = tokens[0], int(tokens[1])
                elif len(tokens) == 1:
                    word, index = tokens[0], i
                else:  # empty line or space
                    logging.info(f"Find empty line or space '{line.strip()}' in {dict_path}:L{i}, set to ' '")
                    word, index = " ", i
                assert len(id2word) == index
                assert len(word2id) == index
                if word == "<space>":
                    logging.info(f"NOTE: Find <space> in {dict_path}:L{i} and convert it to ' '")
                    word = " "
                word2id[word] = index
                id2word.append(word)
        assert len(id2word) == len(word2id)
        return id2word, word2id


class ChineseCharEnglishSpmTokenizer:
    """
    - One Chinese char is a token.
    - Split English word into SPM and one piece is a token.
    - Ignore ' ' between Chinese char
    - Replace ' ' between English word with "▁" by spm_model
    - Need to put SPM piece into dict file
    - If not set spm_model, will use English char and <space>
    """
    SPM_SPACE = "▁"

    def __init__(self, dict_path, spm_model, unk="<unk>", space="<space>"):
        self.dict = TokenDict(dict_path, unk=unk)
        self.space = space
        if spm_model:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)
        else:
            self.sp = None
            print("[WRAN] Not set spm_model, will use English char")
            print("[WARN] Please check how to deal with ' '(space)")
            if self.space not in self.dict:
                print("Please add <space> to your dict, or it will be <unk>")

    def tokenize(self, text, replace_punc=True):
        text = text.upper()
        tokens = []
        if replace_punc:
            text = re.sub(r"[，。？！,\.?!]", " ", text)
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        for part in parts:
            if pattern.fullmatch(part) is not None:
                tokens.append(part)
            else:
                if self.sp:
                    for piece in self.sp.EncodeAsPieces(part.strip()):
                        tokens.append(piece)
                else:
                    for char in part.strip():
                        tokens.append(char if char != " " else self.space)
        tokens_id = []
        for token in tokens:
            tokens_id.append(self.dict.get(token, self.dict.unk))
        return tokens, tokens_id

    def detokenize(self, inputs, join_symbol="", replace_spm_space=True):
        """inputs is ids or tokens, do not need self.sp"""
        if len(inputs) > 0 and type(inputs[0]) == int:
            tokens = [self.dict[id] for id in inputs]
        else:
            tokens = inputs
        s = f"{join_symbol}".join(tokens)
        if replace_spm_space:
            s = s.replace(self.SPM_SPACE, ' ').strip()
        return s


def _build_run_options(silent):
    ro                     = onnxruntime.RunOptions()
    ro.log_severity_level  = 0 if not silent else 4
    ro.log_verbosity_level = 4
    ro.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return ro


def _build_session_opts_ort():
    opts                          = onnxruntime.SessionOptions()
    opts.log_severity_level       = 0 if ORT_LOG else 4
    opts.log_verbosity_level      = 4
    opts.inter_op_num_threads     = MAX_THREADS
    opts.intra_op_num_threads     = MAX_THREADS
    opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    cfgs = {
        "session.set_denormal_as_zero"                 : "1",
        "session.intra_op.allow_spinning"               : "1",
        "session.inter_op.allow_spinning"               : "1",
        "session.enable_quant_qdq_cleanup"              : "1",
        "session.qdq_matmulnbits_accuracy_level"        : "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers" : "1",
        "session.graph_optimizations_loop_level"        : "2",
        "optimization.enable_gelu_approximation"        : "1",
        "optimization.minimal_build_optimizations"      : "",
        "optimization.enable_cast_chain_elimination"    : "1",
        "optimization.disable_specified_optimizers"     : (
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer"
            if ORT_FP16 else ""
        ),
    }
    for key, value in cfgs.items():
        opts.add_session_config_entry(key, value)
    return opts


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type'              : 'CPU',
        'precision'                : 'ACCURACY',
        'num_of_threads'           : MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams'              : 1,
        'enable_opencl_throttling' : False,
        'enable_qdq_optimizer'     : False,
        'disable_dynamic_shapes'   : False,
    }]
    device_type      = "cpu"
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id'                              : DEVICE_ID,
        'gpu_mem_limit'                          : 24 * (1024 ** 3),
        'arena_extend_strategy'                  : 'kNextPowerOfTwo',
        'cudnn_conv_algo_search'                 : 'EXHAUSTIVE',
        'sdpa_kernel'                            : '2',
        'use_tf32'                               : '1',
        'fuse_conv_bias'                         : '0',
        'cudnn_conv_use_max_workspace'           : '1',
        'cudnn_conv1d_pad_to_nc1d'               : '0',
        'tunable_op_enable'                      : '0',
        'tunable_op_tuning_enable'               : '0',
        'tunable_op_max_tuning_duration_ms'      : 10,
        'do_copy_in_default_stream'              : '0',
        'enable_cuda_graph'                      : '0',
        'prefer_nhwc'                            : '0',
        'enable_skip_layer_norm_strict_mode'     : '0',
        'use_ep_level_unified_stream'            : '0',
    }]
    device_type      = "cuda"
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id'                  : DEVICE_ID,
        'performance_preference'     : 'high_performance',
        'device_filter'              : 'gpu',
        'disable_metacommands'       : 'false',
        'enable_graph_capture'       : 'false',
        'enable_graph_serialization' : 'false',
    }]
    device_type      = "dml"
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = "cpu"
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_obj = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
session_opts_ort = _build_session_opts_ort()
run_options      = _build_run_options(silent=not ORT_LOG)
disabled_opts    = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16 else None
)
_packed = {
    'sess_options'        : session_opts_ort,
    'providers'           : ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    'provider_options'    : provider_options,
    'disabled_optimizers' : disabled_opts,
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

_model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}

def _meta_int(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_FireRedASR_AED.py to stamp the model metadata."
        )
    return int(value)


def _meta_int_list(key):
    value = _model_meta.get(key)
    if value is None:
        raise KeyError(
            f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
            f"Re-export with Export_FireRedASR_AED.py to stamp the model metadata."
        )
    return [int(x) for x in value.split(",") if x != ""]


METADATA_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
MAX_SEQ_LEN = _meta_int("max_seq_len")
SAMPLE_RATE = _meta_int("sample_rate")
SOS_TOKEN = _meta_int("sos_token_id")
STOP_TOKEN = _meta_int_list("stop_token_ids")
print(f"\nModel metadata: {len(_model_meta)} keys "
    f"(input_audio_length={METADATA_INPUT_AUDIO_LENGTH}, max_seq_len={MAX_SEQ_LEN}, "
      f"sample_rate={SAMPLE_RATE}, sos={SOS_TOKEN}, stop_token_ids={STOP_TOKEN}).")

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

# Tokenizer assets are bundled inside the ONNX folder by the export / optimize step, so inference is stand-alone.
tokenizer = ChineseCharEnglishSpmTokenizer(str(onnx_folder / "dict.txt"), str(onnx_folder / "train_bpe1000.model"))

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

# ---- Fixed shared buffers (sized from model meta; the audio window is a fresh OrtValue per clip) ----
history_len_ort = _ort_from_data([0], np.int64)                    # history_len = 0 (each prefill starts fresh).
hidden_states_buf = _ort_zeros((decode_batch, 1, hidden_size), hidden_dtype)
position_buf = _ort_zeros((1, 1, hidden_size), position_dtype)
decode_mask_buf = _ort_zeros((1, 1, 1), mask_dtype)                # decode-phase mask is all-zeros (the new token sees every cached position).
prefill_logits_buf = _ort_zeros((1, vocab_size), logits_dtype)
decode_logits_buf = _ort_zeros((decode_batch, vocab_size), logits_dtype)
max_idx_buf = _ort_zeros((1, 1), np.int32)                        # next-token (greedy) / stop-check (beam) buffer.

input_ids_np = np.array([[SOS_TOKEN]], dtype=np.int32)            # FireRedASR-AED prompt is the single SOS token.
input_ids_buf = _ort_from_numpy(input_ids_np)
input_ids_len_ort = _ort_from_data([input_ids_np.shape[-1]], np.int64)
max_generate_limit = MAX_SEQ_LEN - input_ids_np.shape[-1]

# FireRedASR-AED always starts from the same one-token SOS prompt. Precompute its embedding and
# prefill position/mask once, then keep the Embed graph rebound to the decode-token shared buffer.
prompt_embed_buf = _ort_zeros((1, input_ids_np.shape[-1], hidden_size), hidden_dtype)
prefill_position_buf = _ort_zeros((1, input_ids_np.shape[-1], hidden_size), position_dtype)
prefill_mask_buf = _ort_zeros((1, input_ids_np.shape[-1], input_ids_np.shape[-1]), mask_dtype)
prefill_kv_seq_len_seed = _ort_zeros((1,), np.int64)
kv_seq_len_seed_np = np.array([input_ids_np.shape[-1]], dtype=np.int64)
kv_seq_len_ort = _ort_zeros((1,), np.int64)

binding_Embed.bind_ortvalue_input(in_name_Embed0, input_ids_buf)
binding_Embed.bind_ortvalue_output(out_name_Embed0, prompt_embed_buf)
_run(ort_session_Embed, binding_Embed)

_bind_inputs(binding_Prefill, in_name_Prefill, [input_ids_len_ort, history_len_ort])
binding_Prefill.bind_ortvalue_output(out_name_Prefill[0], prefill_position_buf)
binding_Prefill.bind_ortvalue_output(out_name_Prefill[1], prefill_mask_buf)
binding_Prefill.bind_ortvalue_output(out_name_Prefill[2], prefill_kv_seq_len_seed)
_run(ort_session_Prefill, binding_Prefill)

if USE_BEAM_SEARCH:
    binding_Embed.bind_ortvalue_input(in_name_Embed0, beam_ids_buf)
else:
    binding_Embed.bind_ortvalue_input(in_name_Embed0, max_idx_buf)
binding_Embed.bind_ortvalue_output(out_name_Embed0, hidden_states_buf)

binding_Decode.bind_ortvalue_input(in_name_Decode0, kv_seq_len_ort)
binding_Decode.bind_ortvalue_output(out_name_Decode_position, position_buf)
binding_Decode.bind_ortvalue_output(out_name_Decode_kv_seq_len, kv_seq_len_ort)

# Empty decoder self-attention caches (batch 1) reused for every prefill pass.
past_keys_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[0].shape[1], ort_session_Decoder._outputs_meta[0].shape[2], 0), kv_cache_dtype)
past_values_Decoder = _ort_zeros((1, ort_session_Decoder._outputs_meta[num_layers].shape[1], 0, ort_session_Decoder._outputs_meta[num_layers].shape[3]), kv_cache_dtype)

# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)  # Raw int16 PCM; prepare_audio_input owns the optional RMS + dtype conversion.
    audio_len = len(audio)
    audio = prepare_audio_input(audio.reshape(1, 1, -1), input_audio_dtype)
    if SLIDING_WINDOW <= 0:
        INPUT_AUDIO_LENGTH = min(METADATA_INPUT_AUDIO_LENGTH, audio_len) if isinstance(shape_value_in, str) else shape_value_in
        stride_step = INPUT_AUDIO_LENGTH
    else:
        INPUT_AUDIO_LENGTH = min(METADATA_INPUT_AUDIO_LENGTH, audio_len) if isinstance(shape_value_in, str) else shape_value_in
        stride_step = SLIDING_WINDOW
    if SLIDING_WINDOW > 0 and audio_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif SLIDING_WINDOW > 0 and audio_len < INPUT_AUDIO_LENGTH:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]

    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    audio_buffer = _ort_zeros((1, 1, INPUT_AUDIO_LENGTH), _audio_np_dtype)
    save_token_array = np.zeros(0, dtype=np.int32)
    # Start to run FireRedASR
    start_time = time.time()
    while slice_end <= aligned_len:
        # ---- Encoder (per-clip shared buffer keeps the dynamic input length while avoiding per-window OrtValue churn) ----
        audio_buffer.update_inplace(np.ascontiguousarray(audio[:, :, slice_start: slice_end]))
        binding_Encoder.bind_ortvalue_input(in_name_Encoder0, audio_buffer)
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
        _run(ort_session_Encoder, binding_Encoder)
        encoder_outputs = binding_Encoder.get_outputs()                    # f16 cross-attn K/V, already sliced to valid encoder length.
        _bind_inputs(binding_Decoder, in_name_Decoder_en_kv, encoder_outputs)
        encoder_time_len = encoder_outputs[0].shape()[-1]
        generate_limit = min(max_generate_limit, DECODE_MAX_LEN if DECODE_MAX_LEN > 0 else encoder_time_len)

        # ---- Prompt prefill: embed [sos], build its position embedding + causal mask, run the decoder once ----
        _bind_inputs(binding_Decoder, in_name_Decoder[:num_layers], [past_keys_Decoder] * num_layers)
        _bind_inputs(binding_Decoder, in_name_Decoder[num_layers:num_keys_values], [past_values_Decoder] * num_layers)

        kv_seq_len_ort.update_inplace(kv_seq_len_seed_np)
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_hidden, prompt_embed_buf)
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_position, prefill_position_buf)
        binding_Decoder.bind_ortvalue_input(in_name_Decoder_mask, prefill_mask_buf)
        _bind_device_outputs(binding_Decoder, out_name_Decoder_kv)
        binding_Decoder.bind_ortvalue_output(out_name_Decoder_logits, prefill_logits_buf)

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

        if USE_BEAM_SEARCH:
            save_token_array = latest_save_id.numpy()[0]
            for i, idx in enumerate(save_token_array):
                if idx in STOP_TOKEN:
                    save_token_array = save_token_array[:i]
                    break
        elif do_repeat_penality:
            # Greedy with penalty keeps its token history on-device (GREEDY_SEARCH appends each step).
            save_token_array = save_id.numpy()[0]
            for i, idx in enumerate(save_token_array):
                if idx in STOP_TOKEN:
                    save_token_array = save_token_array[:i]
                    break
        else:
            save_token_array = save_id_greedy[:num_decode]
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    rtf = count_time / max(audio_len / SAMPLE_RATE, 1e-6)
    text = tokenizer.detokenize([int(id) for id in save_token_array])
    print(f"\nASR Result:\n{text}\n\nRTF: {rtf:.3f}   ({count_time:.3f}s for {audio_len / SAMPLE_RATE:.2f}s audio, {len(save_token_array)} tokens)")
    print("----------------------------------------------------------------------------------------------------------")
