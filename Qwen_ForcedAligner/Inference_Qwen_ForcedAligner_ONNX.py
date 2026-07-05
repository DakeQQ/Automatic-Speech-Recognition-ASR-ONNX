import argparse
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_cases

# transformers==4.57.6
#
# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-ForcedAligner-0.6B  ·  ONNX Runtime inference demo
# ──────────────────────────────────────────────────────────────────────────────
# Standalone inference pipeline for the ONNX graphs produced by
# Export_Qwen_ForcedAligner.py. The forced aligner is NON auto-regressive (NAR):
# given an audio clip and its transcript it classifies, in ONE forward pass, a
# timestamp bucket at every "<timestamp>" position — NO KV cache, NO decode loop,
# NO beam/greedy search.
#
# Pipeline (mirrors the export self-test):
#   1. tokenize transcript -> word_list, build
#      "<|audio_start|><|audio_pad|><|audio_end|>" + "w1<ts><ts>w2<ts><ts>..."
#   2. Embed graph          : text/timestamp ids   -> token embeddings
#   3. Encoder graph        : audio(int16)+text_emb -> [audio_start, audio, audio_end, text]
#   4. Rotary+Mask graph    : ids_len               -> cos, sin, causal mask
#   5. Decoder-Main graph   : embeds+cos+sin+mask   -> argmax bucket per position
#   6. gather buckets at "<timestamp>" positions, x 80 ms, monotonic-fix -> seconds
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen ForcedAligner ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "Qwen_ForcedAligner_Optimized", help="Folder containing ONNX graphs, for example Qwen_ForcedAligner_Optimized or Qwen_ForcedAligner_ONNX.")
    return parser.parse_args()


_ARGS = _parse_args()

download_path          = r'/home/DakeQQ/Downloads/Qwen3-ForcedAligner-0.6B'  # Local Qwen3-ForcedAligner-0.6B directory (used for the tokenizer).
onnx_folder            = _ARGS.onnx_folder.expanduser().resolve()          # Selected ONNX graph folder.
onnx_model_Metadata    = str(onnx_folder / "ForcedAligner_Metadata.onnx")
onnx_model_Embed       = str(onnx_folder / "ForcedAligner_Embed.onnx")
onnx_model_Encoder     = str(onnx_folder / "ForcedAligner_Encoder.onnx")
onnx_model_Rotary_Mask = str(onnx_folder / "ForcedAligner_Rotary_Mask.onnx")
onnx_model_Main        = str(onnx_folder / "ForcedAligner_Decoder_Main.onnx")

# Test (audio, transcript, language) triples for the inference demo.
# NOTE: the transcript MUST match what is spoken in the clip — forced alignment
# aligns a *known* transcript to the audio. Edit these to your own data.
_TEST_AUDIO = dict(model_audio_cases("qwen_forced_aligner"))
TEST_CASES = [
    (_TEST_AUDIO["zh"],  "開放時間：早上九點至下午五點。",  "Chinese"),
    (_TEST_AUDIO["en"],  "The tribal chieftain called for the boy, and presented him with fifty pieces of gold.", "English"),
    (_TEST_AUDIO["yue"], "呢幾個字都表達唔到我想講嘅意思。", "Cantonese"),
    (_TEST_AUDIO["ja"],  "うちの中学は弁当制で、持っていない場合は50円の学校販売のパンを買う。", "Japanese"),
    (_TEST_AUDIO["ko"],  "조금만 생각을 하면서 살면 훨씬 편할 거야.", "Korean"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Runtime Configuration
# ══════════════════════════════════════════════════════════════════════════════
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# ("int16" -> raw PCM ÷32768 in-graph; "float16"/"float" -> pre-normalised [-1, 1]); no manual setting needed.
USE_NORMALISE_AUDIO    = False             # Apply RMS loudness normalisation before feeding the model. Set False to pass raw audio through (only the dtype conversion is applied).

ORT_Accelerate_Providers = []     # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                  = False  # Enable ONNX Runtime logging for debugging.
ORT_FP16                 = False  # FP16 ONNX Runtime settings. For CPUs, requires ARM64-v8.2a or newer.
MAX_THREADS              = 0      # Parallel CPU threads. 0 = auto.
DEVICE_ID                = 0      # Default to zero.


_INV_INT16_SCALE = np.float32(1.0 / 32768.0)  # pre-computed [-1, 1] normalisation scale, reused every call


def prepare_audio_input(audio_int16: np.ndarray, input_audio_dtype: str, target_rms: float = 8192.0) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's
    # audio input dtype exactly once (no float32<->int16 round-trip for the float paths).
    # `input_audio_dtype` is derived from the encoder's audio input tensor in the ONNX model.
    #   "INT16": raw PCM (the graph divides by 32768 internally).
    #   "F32"/"F16": normalised to [-1, 1] here (÷32768), because the float graph skips the
    #   in-model division; "F16" stores those values (the graph up-casts back to f32).
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
    audio *= _INV_INT16_SCALE   # fold the pre-computed ÷32768 scale into the same float buffer
    if input_audio_dtype == "F16":
        return audio.astype(np.float16)
    return audio


# ══════════════════════════════════════════════════════════════════════════════
# Special-Token IDs & timestamp buckets  (verified from config.json / tokenizer_config.json)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# ── Forced-alignment text processor (port of Qwen3ForceAlignProcessor) ─────────
# ══════════════════════════════════════════════════════════════════════════════
class AlignerTextProcessor:
    """Word/character tokenizer + timestamp post-processing, byte-faithful to upstream."""

    def __init__(self) -> None:
        self.ko_tokenizer = None

    # ── unit tokenization ──────────────────────────────────────────────────────
    def is_kept_char(self, ch: str) -> bool:
        if ch == "'":
            return True
        cat = unicodedata.category(ch)
        return cat.startswith("L") or cat.startswith("N")

    def clean_token(self, token: str) -> str:
        return "".join(ch for ch in token if self.is_kept_char(ch))

    def is_cjk_char(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
        )

    def split_segment_with_chinese(self, seg: str) -> List[str]:
        tokens: List[str] = []
        buf: List[str] = []

        def flush_buf() -> None:
            nonlocal buf
            if buf:
                tokens.append("".join(buf))
                buf = []

        for ch in seg:
            if self.is_cjk_char(ch):
                flush_buf()
                tokens.append(ch)
            else:
                buf.append(ch)
        flush_buf()
        return tokens

    def tokenize_space_lang(self, text: str) -> List[str]:
        tokens: List[str] = []
        for seg in text.split():
            cleaned = self.clean_token(seg)
            if cleaned:
                tokens.extend(self.split_segment_with_chinese(cleaned))
        return tokens

    def tokenize_japanese(self, text: str) -> List[str]:
        try:
            import nagisa
            words = nagisa.tagging(text).words
        except Exception:
            print("  [WARN] 'nagisa' not available — falling back to whitespace/CJK tokenization for Japanese.")
            return self.tokenize_space_lang(text)
        tokens: List[str] = []
        for w in words:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def tokenize_korean(self, text: str) -> List[str]:
        try:
            if self.ko_tokenizer is None:
                from soynlp.tokenizer import LTokenizer
                self.ko_tokenizer = LTokenizer()
            raw_tokens = self.ko_tokenizer.tokenize(text)
        except Exception:
            print("  [WARN] 'soynlp' not available — falling back to whitespace/CJK tokenization for Korean.")
            return self.tokenize_space_lang(text)
        tokens: List[str] = []
        for w in raw_tokens:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def encode_timestamp(self, text: str, language: str) -> Tuple[List[str], str]:
        language = language.lower()
        if language == "japanese":
            word_list = self.tokenize_japanese(text)
        elif language == "korean":
            word_list = self.tokenize_korean(text)
        else:
            word_list = self.tokenize_space_lang(text)
        input_text = "<timestamp><timestamp>".join(word_list) + "<timestamp><timestamp>"
        input_text = "<|audio_start|><|audio_pad|><|audio_end|>" + input_text
        return word_list, input_text

    # ── timestamp post-processing ──────────────────────────────────────────────
    def fix_timestamp(self, data) -> List[int]:
        """Longest non-decreasing subsequence -> monotone repair of anomalies."""
        data = list(int(x) for x in data)
        n = len(data)
        if n == 0:
            return []

        dp = [1] * n
        parent = [-1] * n
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j

        max_length = max(dp)
        max_idx = dp.index(max_length)
        lis_indices: List[int] = []
        idx = max_idx
        while idx != -1:
            lis_indices.append(idx)
            idx = parent[idx]
        lis_indices.reverse()

        is_normal = [False] * n
        for idx in lis_indices:
            is_normal[idx] = True

        result = data.copy()
        i = 0
        while i < n:
            if not is_normal[i]:
                j = i
                while j < n and not is_normal[j]:
                    j += 1
                anomaly_count = j - i

                left_val = None
                for k in range(i - 1, -1, -1):
                    if is_normal[k]:
                        left_val = result[k]
                        break
                right_val = None
                for k in range(j, n):
                    if is_normal[k]:
                        right_val = result[k]
                        break

                if anomaly_count <= 2:
                    for k in range(i, j):
                        if left_val is None:
                            result[k] = right_val
                        elif right_val is None:
                            result[k] = left_val
                        else:
                            result[k] = left_val if (k - (i - 1)) <= (j - k) else right_val
                else:
                    if left_val is not None and right_val is not None:
                        step = (right_val - left_val) / (anomaly_count + 1)
                        for k in range(i, j):
                            result[k] = left_val + step * (k - i + 1)
                    elif left_val is not None:
                        for k in range(i, j):
                            result[k] = left_val
                    elif right_val is not None:
                        for k in range(i, j):
                            result[k] = right_val
                i = j
            else:
                i += 1
        return [int(res) for res in result]

    def parse_timestamp(self, word_list: List[str], timestamp) -> List[Dict]:
        timestamp_output: List[Dict] = []
        timestamp_fixed = self.fix_timestamp(timestamp)
        for i, word in enumerate(word_list):
            timestamp_output.append({
                "text": word,
                "start_time": timestamp_fixed[i * 2],
                "end_time": timestamp_fixed[i * 2 + 1],
            })
        return timestamp_output


# ══════════════════════════════════════════════════════════════════════════════
# ── ORT Session & Runtime Helpers ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
    ro = onnxruntime.RunOptions()
    ro.log_severity_level  = 4 if silent else 0
    ro.log_verbosity_level = 4
    ro.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return ro


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level       = 0 if ORT_LOG else 4
    opts.log_verbosity_level      = 4
    opts.inter_op_num_threads     = MAX_THREADS
    opts.intra_op_num_threads     = MAX_THREADS
    opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    _cfgs = {
        "session.set_denormal_as_zero":                  "1",
        "session.intra_op.allow_spinning":               "1",
        "session.inter_op.allow_spinning":               "1",
        "session.enable_quant_qdq_cleanup":              "1",
        "session.qdq_matmulnbits_accuracy_level":        "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level":        "2",
        "optimization.enable_gelu_approximation":        "1",
        "optimization.minimal_build_optimizations":      "",
        "optimization.enable_cast_chain_elimination":    "1",
        "optimization.disable_specified_optimizers":
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else "",
    }
    for k, v in _cfgs.items():
        opts.add_session_config_entry(k, v)
    return opts


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False,
    }]
    device_type      = "cpu"
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                    DEVICE_ID,
        'gpu_mem_limit':                24 * (1024 ** 3),
        'arena_extend_strategy':        'kNextPowerOfTwo',
        'cudnn_conv_algo_search':       'EXHAUSTIVE',
        'sdpa_kernel':                  '2',
        'use_tf32':                     '1',
        'do_copy_in_default_stream':    '0',
        'enable_cuda_graph':            '0',
    }]
    device_type      = "cuda"
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':              DEVICE_ID,
        'performance_preference': 'high_performance',
        'device_filter':          'gpu',
    }]
    device_type      = "dml"
    _ort_device_type = C.OrtDevice.dml()
else:
    provider_options = None
    device_type      = "cpu"
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_obj  = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
session_opts_ort = _build_session_opts_ort()
run_options      = _build_run_options(silent=not ORT_LOG)
disabled_opts    = ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"] if ORT_FP16 else None

_packed = dict(
    sess_options=session_opts_ort,
    providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    provider_options=provider_options,
    disabled_optimizers=disabled_opts,
)


def _make_session(path: str) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(path, **_packed)


def _ort_from_numpy(arr: np.ndarray) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(arr, device_type, DEVICE_ID)


def _bind_device_outputs(binding, names) -> None:
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _in_names(session):
    return [x.name for x in session.get_inputs()]


def _out_names(session):
    return [x.name for x in session.get_outputs()]


# ══════════════════════════════════════════════════════════════════════════════
# ── Inference Demo (single NAR forward per sample) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def run_inference() -> None:
    tokenizer         = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)
    aligner_processor = AlignerTextProcessor()

    print("Loading sessions …")
    ort_session_Metadata = _make_session(onnx_model_Metadata)
    ort_session_Embed   = _make_session(onnx_model_Embed)
    ort_session_Encoder = _make_session(onnx_model_Encoder)
    ort_session_Rotary  = _make_session(onnx_model_Rotary_Mask)
    ort_session_Main    = _make_session(onnx_model_Main)
    print(f"  Usable Providers : {ort_session_Main.get_providers()}")

    _model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}
    if _model_meta.get("qwen_forcedaligner_metadata_version") != "1":
        raise ValueError(
            f"Required Qwen_ForcedAligner metadata version 1 is missing from {onnx_model_Metadata}. "
            "Re-export with Export_Qwen_ForcedAligner.py to stamp the model metadata."
        )

    def _meta_int(key: str) -> int:
        value = _model_meta.get(key)
        if value is None:
            raise KeyError(
                f"Required metadata key '{key}' is missing from {onnx_model_Metadata}. "
                "Re-export with Export_Qwen_ForcedAligner.py to stamp the model metadata."
            )
        return int(value)

    SAMPLE_RATE = _meta_int("sample_rate")
    MAX_INPUT_AUDIO_LENGTH = _meta_int("input_audio_length")
    MAX_SEQ_LEN = _meta_int("max_seq_len")
    AUDIO_START_TOKEN_ID = _meta_int("audio_start_token_id")
    AUDIO_END_TOKEN_ID = _meta_int("audio_end_token_id")
    AUDIO_PAD_TOKEN_ID = _meta_int("audio_pad_token_id")
    TIMESTAMP_TOKEN_ID = _meta_int("timestamp_token_id")
    TIMESTAMP_SEGMENT_TIME = _meta_int("timestamp_segment_ms")
    print(
        f"  Model metadata: {len(_model_meta)} keys "
        f"(sample_rate={SAMPLE_RATE}, input_audio_length={MAX_INPUT_AUDIO_LENGTH}, "
        f"max_seq_len={MAX_SEQ_LEN}, timestamp_segment_ms={TIMESTAMP_SEGMENT_TIME})."
    )

    binding_Embed   = ort_session_Embed.io_binding()
    binding_Encoder = ort_session_Encoder.io_binding()
    binding_Rotary  = ort_session_Rotary.io_binding()
    binding_Main    = ort_session_Main.io_binding()

    in_name_Embed,   out_name_Embed   = _in_names(ort_session_Embed),   _out_names(ort_session_Embed)
    in_name_Encoder, out_name_Encoder = _in_names(ort_session_Encoder), _out_names(ort_session_Encoder)
    in_name_Rotary,  out_name_Rotary  = _in_names(ort_session_Rotary),  _out_names(ort_session_Rotary)
    in_name_Main,    out_name_Main    = _in_names(ort_session_Main),    _out_names(ort_session_Main)

    # Encoder text_embed input dtype (float16 or float32 depending on quantization).
    enc_text_meta = ort_session_Encoder._inputs_meta[1]
    text_embed_np = np.float16 if "float16" in enc_text_meta.type else np.float32

    # Audio input dtype comes straight from the encoder's audio input tensor in the ONNX model
    # ("int16" -> raw PCM ÷32768 in-graph; "float16"/"float" -> pre-normalised [-1, 1]).
    _audio_input_type = ort_session_Encoder._inputs_meta[0].type
    input_audio_dtype = "INT16" if "int16" in _audio_input_type else ("F16" if "float16" in _audio_input_type else "F32")

    for test_path, transcript, language in TEST_CASES:
        if not os.path.isfile(test_path):
            print(f"\n  [WARN] Cannot find '{test_path}' — skipping.")
            continue
        try:
            audio_seg = AudioSegment.from_file(test_path)
            audio_pcm = np.array(audio_seg.set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
        except Exception:
            print(f"\n  [WARN] Cannot load '{test_path}' — skipping.")
            continue

        print(f"\nTest audio : {test_path}   ({len(audio_pcm) / SAMPLE_RATE:.2f} s)")
        print(f"  Language   : {language}")
        print(f"  Transcript : {transcript}")
        print("─" * 70)

        # 1. Build the alignment prompt and tokenize it.
        word_list, input_text = aligner_processor.encode_timestamp(transcript, language)
        if not word_list:
            print("  [WARN] Transcript produced no alignable units — skipping.")
            continue
        full_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
        expected_audio_prefix = [AUDIO_START_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_END_TOKEN_ID]
        if full_ids[:3] != expected_audio_prefix:
            raise ValueError(
                "Tokenizer audio special-token ids do not match ONNX metadata: "
                f"got {full_ids[:3]}, expected {expected_audio_prefix}."
            )
        pad_pos = full_ids.index(AUDIO_PAD_TOKEN_ID)
        text_ts_ids = full_ids[pad_pos + 2:]   # tokens after <|audio_end|> (words + timestamps)
        text_len = len(text_ts_ids)

        audio_pcm = audio_pcm[:MAX_INPUT_AUDIO_LENGTH]
        audio_np  = prepare_audio_input(audio_pcm.reshape(1, 1, -1), input_audio_dtype)
        audio_ort = _ort_from_numpy(audio_np)
        text_ids_ort = _ort_from_numpy(np.array([text_ts_ids], dtype=np.int32))

        t0 = time.time()

        # 2. Embed text/timestamp ids.
        binding_Embed.bind_ortvalue_input(in_name_Embed[0], text_ids_ort)
        _bind_device_outputs(binding_Embed, out_name_Embed)
        _run(ort_session_Embed, binding_Embed)
        text_embed_ort = _ort_from_numpy(binding_Embed.get_outputs()[0].numpy().astype(text_embed_np))

        # 3. Encode audio and splice in the token embeddings.
        binding_Encoder.bind_ortvalue_input(in_name_Encoder[0], audio_ort)
        binding_Encoder.bind_ortvalue_input(in_name_Encoder[1], text_embed_ort)
        _bind_device_outputs(binding_Encoder, out_name_Encoder)
        _run(ort_session_Encoder, binding_Encoder)
        concat_embed_ort, ids_len_ort = binding_Encoder.get_outputs()
        ids_len_value = int(ids_len_ort.numpy().reshape(-1)[0])
        if ids_len_value > MAX_SEQ_LEN:
            raise ValueError(
                f"Encoded sequence length {ids_len_value} exceeds metadata max_seq_len={MAX_SEQ_LEN}. "
                "Use a shorter clip/transcript or re-export with a larger MAX_SEQ_LEN."
            )

        # 4. Rotary tables + causal mask for the full sequence.
        binding_Rotary.bind_ortvalue_input(in_name_Rotary[0], ids_len_ort)
        _bind_device_outputs(binding_Rotary, out_name_Rotary)
        _run(ort_session_Rotary, binding_Rotary)
        rotary_cos_ort, rotary_sin_ort, attention_mask_ort = binding_Rotary.get_outputs()

        # 5. One NAR forward -> a timestamp bucket at every position.
        binding_Main.bind_ortvalue_input(in_name_Main[0], concat_embed_ort)
        binding_Main.bind_ortvalue_input(in_name_Main[1], rotary_cos_ort)
        binding_Main.bind_ortvalue_input(in_name_Main[2], rotary_sin_ort)
        binding_Main.bind_ortvalue_input(in_name_Main[3], attention_mask_ort)
        _bind_device_outputs(binding_Main, out_name_Main)
        _run(ort_session_Main, binding_Main)
        output_ids = binding_Main.get_outputs()[0].numpy()[0]   # (L,)

        # 6. Gather buckets at "<timestamp>" positions and convert to seconds.
        total_len    = output_ids.shape[0]
        text_start   = total_len - text_len   # text block sits after [audio_start, audio, audio_end]
        ts_positions = [text_start + j for j, tok in enumerate(text_ts_ids) if tok == TIMESTAMP_TOKEN_ID]
        timestamp_ms = output_ids[ts_positions].astype(np.int64) * TIMESTAMP_SEGMENT_TIME
        aligned = aligner_processor.parse_timestamp(word_list, timestamp_ms)

        t_total = time.time() - t0
        rtf     = t_total / max(len(audio_pcm) / SAMPLE_RATE, 1e-6)

        print("  Timestamps :")
        for item in aligned:
            start_s = round(item["start_time"] / 1000.0, 3)
            end_s   = round(item["end_time"] / 1000.0, 3)
            print(f"    {start_s:7.3f}s → {end_s:7.3f}s   {item['text']}")
        print(f"\n  RTF : {rtf:.3f}   total {t_total:.2f}s")
        print("─" * 70)


if __name__ == "__main__":
    print("Starting ONNX Runtime inference …\n")
    run_inference()
