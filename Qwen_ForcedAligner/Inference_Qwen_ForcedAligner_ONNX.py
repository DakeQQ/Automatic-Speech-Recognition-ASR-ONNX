import argparse
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from transformers import AutoTokenizer

from Shared_Merged import (
    DEFAULT_MODEL_FILE_NAMES,
    attach_shared_initializers,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from Example_Audio import model_audio_cases
from ORT_IO import (
    array_for,
    is_dynamic_dim,
    metadata_by_name,
    load_special_token_ids,
    load_supported_languages,
    numpy_dtype,
    resolve_supported_language,
)

# transformers==4.57.6
#
# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-ForcedAligner-0.6B  ·  ONNX Runtime inference demo
# ──────────────────────────────────────────────────────────────────────────────
# Standalone inference pipeline for the merged ONNX bundle produced by
# Export_Qwen_ForcedAligner.py. The forced aligner is NON auto-regressive (NAR):
# given an audio clip and its transcript it classifies, in ONE forward pass, a
# timestamp bucket at every "<timestamp>" position — NO KV cache, NO decode loop,
# No autoregressive token-selection loop.
#
# Host pipeline:
#   1. split the transcript into word units, tokenize only lexical text, and
#      insert metadata-owned audio/timestamp IDs into the prompt explicitly
#   2. Merged graph (one run): Embed -> Encoder -> Rotary+Mask -> Decoder Main
#   3. gather buckets at "<timestamp>" positions, x 80 ms, monotonic-fix -> seconds
#
# Large initializers are mmap'd once from ForcedAligner_SharedInitializers.onnx.data
# and injected before session creation. There is no prefill/decode ping-pong to run:
# that mechanism is meaningful only for autoregressive models with a growing KV cache.
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen ForcedAligner ONNX inference.")
    parser.add_argument("--onnx-folder", "--model-folder", dest="onnx_folder", type=Path, default=_SCRIPT_DIR / "Qwen_ForcedAligner_Optimized", help="Folder containing ONNX graphs, for example Qwen_ForcedAligner_Optimized or Qwen_ForcedAligner_ONNX.")
    parser.add_argument("--tokenizer-path", type=Path, default=None, help="Optional tokenizer directory; defaults to tokenizer inside the model folder.")
    return parser.parse_args()


_ARGS = _parse_args()

onnx_folder            = _ARGS.onnx_folder.expanduser().resolve()          # Selected ONNX graph folder.
onnx_model_Metadata    = str(onnx_folder / DEFAULT_MODEL_FILE_NAMES["metadata"])
TOKENIZER_PATH = (
    _ARGS.tokenizer_path.expanduser().resolve()
    if _ARGS.tokenizer_path is not None
    else onnx_folder / "tokenizer"
)

# Test (audio, transcript, language) triples for the inference demo.
# NOTE: the transcript MUST match what is spoken in the clip — forced alignment
# aligns a *known* transcript to the audio. Edit these to your own data.
_TEST_AUDIO = {language: path for path, language in model_audio_cases("qwen_forced_aligner")}
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
# IMPORTANT: CLI options are intentionally limited to model/tokenizer paths.
# Edit this section and TEST_CASES above for demo, audio, and runtime behavior.
# The audio input dtype is auto-detected from the encoder's audio input tensor in the ONNX model
# ("int16" -> raw PCM scaled in-graph; "float16"/"float" -> metadata-scaled
# normalised audio); no manual setting is needed.
USE_NORMALISE_AUDIO    = False             # Apply RMS loudness normalisation before feeding the model. Set False to pass raw audio through (only the dtype conversion is applied).

ORT_Accelerate_Providers = ["CUDAExecutionProvider"]     # ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                  = False  # Enable ONNX Runtime logging for debugging.
ORT_FP16                 = False  # FP16 ONNX Runtime settings. For CPUs, requires ARM64-v8.2a or newer.
MAX_THREADS              = 0      # Parallel CPU threads. 0 = auto.
DEVICE_ID                = 0      # Default to zero.


def prepare_audio_input(
    audio_int16: np.ndarray,
    target_dtype: np.dtype,
    audio_pcm_scale: int,
    target_rms: float = 4096.0,
) -> np.ndarray:
    # Fold the optional RMS loudness normalisation and the model-dtype conversion into a
    # single pass over the raw int16 PCM that pydub returns, casting to the model's
    # audio input dtype exactly once (no float32<->int16 round-trip for the float paths).
    # `target_dtype` is derived exactly from the merged graph's public audio input.
    if not USE_NORMALISE_AUDIO and target_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio_int16, dtype=target_dtype)
    audio = audio_int16.astype(np.float32)
    if USE_NORMALISE_AUDIO:
        rms = np.sqrt(np.mean(audio * audio, dtype=np.float32), dtype=np.float32)
        if rms > 0:
            audio *= (target_rms / (rms + 1e-7))
            np.clip(
                audio,
                -float(audio_pcm_scale),
                float(audio_pcm_scale) - 1.0,
                out=audio,
            )
    if target_dtype == np.dtype(np.int16):
        return np.ascontiguousarray(audio, dtype=target_dtype)
    audio *= np.float32(1.0 / audio_pcm_scale)
    return np.ascontiguousarray(audio, dtype=target_dtype)


def _build_alignment_prompt_ids(
    tokenizer,
    word_units: List[str],
    special_token_ids: dict[str, int],
    timestamp_tokens_per_word: int,
) -> List[int]:
    """Encode lexical units and insert every immutable control ID from metadata."""
    audio_prefix = [
        int(special_token_ids["audio_start"]),
        int(special_token_ids["audio_pad"]),
        int(special_token_ids["audio_end"]),
    ]
    timestamp_id = int(special_token_ids["timestamp"])
    text_ts_ids: List[int] = []
    for word in word_units:
        lexical_ids = [
            int(token_id)
            for token_id in tokenizer.encode(word, add_special_tokens=False)
        ]
        text_ts_ids.extend(lexical_ids)
        text_ts_ids.extend([timestamp_id] * timestamp_tokens_per_word)
    return audio_prefix + text_ts_ids


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
        import nagisa
        words = nagisa.tagging(text).words
        tokens: List[str] = []
        for w in words:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def tokenize_korean(self, text: str) -> List[str]:
        if self.ko_tokenizer is None:
            from soynlp.tokenizer import LTokenizer
            self.ko_tokenizer = LTokenizer()
        raw_tokens = self.ko_tokenizer.tokenize(text)
        tokens: List[str] = []
        for w in raw_tokens:
            cleaned = self.clean_token(w)
            if cleaned:
                tokens.append(cleaned)
        return tokens

    def word_units(self, text: str, language: str) -> List[str]:
        language = language.lower()
        if language == "japanese":
            return self.tokenize_japanese(text)
        elif language == "korean":
            return self.tokenize_korean(text)
        return self.tokenize_space_lang(text)

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

    def parse_timestamp(
        self,
        word_list: List[str],
        timestamp,
        timestamp_tokens_per_word: int,
    ) -> List[Dict]:
        timestamp_output: List[Dict] = []
        timestamp_fixed = self.fix_timestamp(timestamp)
        for i, word in enumerate(word_list):
            group_start = i * timestamp_tokens_per_word
            timestamp_group = timestamp_fixed[
                group_start:group_start + timestamp_tokens_per_word
            ]
            timestamp_output.append({
                "text": word,
                "start_time": timestamp_group[0],
                "end_time": timestamp_group[-1],
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
run_options      = _build_run_options(silent=not ORT_LOG)

def _make_session(path: str, shared_path: str | None = None) -> onnxruntime.InferenceSession:
    session_options = _build_session_opts_ort()
    shared_refs = None
    if shared_path is not None:
        shared_refs = attach_shared_initializers(session_options, Path(shared_path))
    session = onnxruntime.InferenceSession(
        path,
        sess_options=session_options,
        providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
        provider_options=provider_options,
        disabled_optimizers=(
            ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
            if ORT_FP16 else None
        ),
    )
    if shared_refs is not None:
        # Memmaps and OrtValues must outlive the session. Dropping these references
        # would leave add_initializer() pointing at released memory.
        session._qwen_forced_aligner_shared_initializers = shared_refs
    return session


def _ort_from_numpy(arr: np.ndarray) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.ascontiguousarray(arr), device_type, DEVICE_ID
    )


def _bind_device_outputs(binding, names) -> None:
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _out_names(session):
    return [x.name for x in session.get_outputs()]


# ══════════════════════════════════════════════════════════════════════════════
# ── Inference Demo (single NAR forward per sample) ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def run_inference() -> None:
    print("Loading metadata …")
    ort_session_Metadata = _make_session(onnx_model_Metadata)
    _model_meta = ort_session_Metadata.get_modelmeta().custom_metadata_map or {}
    model_file_names = DEFAULT_MODEL_FILE_NAMES
    merged_path = onnx_folder / model_file_names["merged"]
    shared_path = onnx_folder / model_file_names["shared_initializers"]

    shared_initializer_path = str(shared_path) if shared_path.exists() else None
    print("Loading merged compute session …")
    ort_session_Merged = _make_session(str(merged_path), shared_initializer_path)
    print(f"  Usable Providers : {ort_session_Merged.get_providers()}")
    print("  Compute sessions : 1 merged (legacy split pipeline: 4)")

    # The tokenizer is bundled inside the selected model folder, so inference is stand-alone.
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), trust_remote_code=True)
    SPECIAL_TOKEN_IDS = load_special_token_ids(_model_meta)
    SUPPORTED_LANGUAGES = load_supported_languages(_model_meta)
    aligner_processor = AlignerTextProcessor()

    SAMPLE_RATE = int(_model_meta["sample_rate"])
    TIMESTAMP_TOKEN_ID = SPECIAL_TOKEN_IDS["timestamp"]
    TIMESTAMP_SEGMENT_TIME = int(_model_meta["timestamp_segment_ms"])
    TIMESTAMP_TOKENS_PER_WORD = int(_model_meta["timestamp_tokens_per_word"])
    AUDIO_PCM_SCALE = int(_model_meta["audio_pcm_scale"])
    print(
        f"  Model metadata: sample_rate={SAMPLE_RATE}, "
        f"timestamp_segment_ms={TIMESTAMP_SEGMENT_TIME}, "
        f"timestamp_tokens_per_word={TIMESTAMP_TOKENS_PER_WORD}."
    )

    binding_Merged = ort_session_Merged.io_binding()
    out_name_Merged = _out_names(ort_session_Merged)

    input_meta = metadata_by_name(ort_session_Merged.get_inputs())
    audio_meta = input_meta["audio"]
    input_ids_meta = input_meta["input_ids"]
    audio_dtype = numpy_dtype(audio_meta)
    audio_sample_dim = audio_meta.shape[2]
    if not is_dynamic_dim(audio_sample_dim):
        runtime_audio_limit = int(audio_sample_dim)
    else:
        runtime_audio_limit = None

    for test_path, transcript, language in TEST_CASES:
        language_code, language_entry = resolve_supported_language(
            SUPPORTED_LANGUAGES, language
        )
        audio_seg = AudioSegment.from_file(test_path)
        audio_pcm = np.array(audio_seg.set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)

        print(f"\nTest audio : {test_path}   ({len(audio_pcm) / SAMPLE_RATE:.2f} s)")
        print(f"  Language   : {language_entry['name']} ({language_code})")
        print(f"  Transcript : {transcript}")
        print("─" * 70)

        # 1. Tokenize lexical units, then insert metadata-owned control IDs.
        word_list = aligner_processor.word_units(transcript, language_entry["name"])
        if not word_list:
            print("  [WARN] Transcript produced no alignable units — skipping.")
            continue
        full_ids = _build_alignment_prompt_ids(
            tokenizer,
            word_list,
            SPECIAL_TOKEN_IDS,
            TIMESTAMP_TOKENS_PER_WORD,
        )
        text_ts_ids = full_ids[3:]
        text_len = len(text_ts_ids)

        if runtime_audio_limit is not None:
            audio_pcm = audio_pcm[:runtime_audio_limit]
        audio_np = prepare_audio_input(
            audio_pcm.reshape(1, 1, -1),
            audio_dtype,
            AUDIO_PCM_SCALE,
        )
        audio_np = array_for(
            audio_meta,
            audio_np,
            axes={0: 1, 1: 1, 2: audio_np.shape[2]},
        )
        text_ids_np = array_for(
            input_ids_meta,
            [text_ts_ids],
            axes={0: 1, 1: text_len},
        )
        audio_ort = _ort_from_numpy(audio_np)
        text_ids_ort = _ort_from_numpy(text_ids_np)

        t0 = time.time()

        # 2. One NAR graph launch: Embed -> Encoder -> Rotary+Mask -> Decoder Main.
        binding_Merged.bind_ortvalue_input("audio", audio_ort)
        binding_Merged.bind_ortvalue_input("input_ids", text_ids_ort)
        _bind_device_outputs(binding_Merged, out_name_Merged)
        _run(ort_session_Merged, binding_Merged)
        output_ids_array = binding_Merged.get_outputs()[0].numpy()
        output_ids = output_ids_array[0]   # (L,)

        # 3. Gather buckets at "<timestamp>" positions and convert to seconds.
        total_len    = output_ids.shape[0]
        text_start   = total_len - text_len   # text block sits after [audio_start, audio, audio_end]
        ts_positions = [text_start + j for j, tok in enumerate(text_ts_ids) if tok == TIMESTAMP_TOKEN_ID]
        timestamp_ms = output_ids[ts_positions].astype(np.int64) * TIMESTAMP_SEGMENT_TIME
        aligned = aligner_processor.parse_timestamp(
            word_list,
            timestamp_ms,
            TIMESTAMP_TOKENS_PER_WORD,
        )

        t_total = time.time() - t0
        rtf     = t_total / (len(audio_pcm) / SAMPLE_RATE)

        print("  Timestamps :")
        for item in aligned:
            start_s = round(item["start_time"] / 1000.0, 3)
            end_s   = round(item["end_time"] / 1000.0, 3)
            print(f"    {start_s:7.3f}s → {end_s:7.3f}s   {item['text']}")
        print(f"\n  RTF : {rtf:.3f}   total {t_total:.2f}s   graph launches: 1 (split: 4)")
        print("─" * 70)


if __name__ == "__main__":
    print("Starting ONNX Runtime inference …\n")
    run_inference()
