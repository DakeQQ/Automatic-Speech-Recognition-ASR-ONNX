"""
Standalone Qwen3-ASR transformers inference script.

This vendors the minimal qwen_asr source closure needed to run the same path as
`demo.py` without importing the `qwen_asr` package:

    Qwen3ASRModel.from_pretrained(...)
      -> AutoModel.from_pretrained(...)
      -> AutoProcessor.from_pretrained(...)
      -> transcribe(...)
      -> normalize_audios(...)
      -> split_audio_into_chunks(...)
      -> processor(text=..., audio=..., ...)
      -> model.generate(...)
      -> parse_asr_output(...)

The script keeps the upstream model/config/processor logic local and adds an
optional trace mode that exposes the chunk-level prompt and tensor shapes.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import base64
import io
import re
import urllib.request
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.activations import ACT2FN
from transformers.audio_utils import AudioInput
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import TextInput
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.generic import TransformersKwargs, check_model_inputs

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Vendored configuration_qwen3_asr.py
# ---------------------------------------------------------------------------

class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = scale_embedding
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRTextConfig(PretrainedConfig):
    model_type = "qwen3_asr_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=128000,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"
    attribute_map = {}
    sub_configs = {
        "audio_config": Qwen3ASRAudioEncoderConfig,
        "text_config": Qwen3ASRTextConfig,
    }

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id=151646,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config
        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        return self.text_config


class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {"thinker_config": Qwen3ASRThinkerConfig}

    def __init__(self, thinker_config=None, support_languages=None, **kwargs):
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
        self.thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.support_languages = support_languages

    def get_text_config(self, decoder=False) -> PretrainedConfig:
        return self.thinker_config.get_text_config()


# ---------------------------------------------------------------------------
# Vendored processing_qwen3_asr.py
# ---------------------------------------------------------------------------

class Qwen3ASRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "padding_side": "left"},
        "audio_kwargs": {"sampling_rate": 16000, "padding": True, "return_attention_mask": True},
    }


def _get_feat_extract_output_lengths(input_lengths):
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3ASRProcessor(ProcessorMixin):
    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "WhisperFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, feature_extractor=None, tokenizer=None, chat_template=None):
        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)
        self.audio_token = self.tokenizer.audio_token
        self.audio_bos_token = self.tokenizer.audio_bos_token
        self.audio_eos_token = self.tokenizer.audio_eos_token

    def __call__(self, text: TextInput = None, audio: AudioInput = None, **kwargs) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")
        output_kwargs = self._merge_kwargs(
            Qwen3ASRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = True
            output_kwargs["audio_kwargs"]["truncation"] = False
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop("attention_mask")
            audio_inputs["input_features"] = audio_inputs.pop("input_features")
            audio_lengths = iter(_get_feat_extract_output_lengths(audio_inputs["feature_attention_mask"].sum(-1)))
        else:
            audio_inputs = {}
            audio_lengths = iter([])
        if not isinstance(text, list):
            text = [text]
        text = self.replace_multimodal_special_tokens(text, audio_lengths)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        return BatchFeature(data={**text_inputs, **audio_inputs}, tensor_type=kwargs.get("return_tensors"))

    def replace_multimodal_special_tokens(self, text, audio_lengths):
        processed_text = []
        for sample in text:
            special_tokens = [re.escape(tok) for tok in [self.audio_token]]
            pattern = "|".join(special_tokens)
            positions = sorted([(match.start(), match.group()) for match in re.finditer(pattern, sample)])
            for _, special_token in positions:
                if special_token == self.audio_token:
                    sample = sample.replace(self.audio_token, "<|audio_placeholder|>" * next(audio_lengths), 1)
            sample = sample.replace("<|audio_placeholder|>", self.audio_token)
            processed_text.append(sample)
        return processed_text

    def apply_chat_template(self, conversations, chat_template=None, **kwargs):
        return super().apply_chat_template(conversations, chat_template, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names + ["feature_attention_mask"]))


# ---------------------------------------------------------------------------
# Vendored inference/utils.py pieces used by demo.py path
# ---------------------------------------------------------------------------

AudioLike = Union[str, Tuple[np.ndarray, int]]
MaybeList = Union[Any, List[Any]]

SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200
MIN_ASR_INPUT_SECONDS = 0.5
SUPPORTED_LANGUAGES: List[str] = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish", "Portuguese",
    "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese", "Japanese", "Turkish",
    "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech", "Filipino",
    "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def normalize_language_name(language: str) -> str:
    if language is None:
        raise ValueError("language is None")
    s = str(language).strip()
    if not s:
        raise ValueError("language is empty")
    return s[:1].upper() + s[1:].lower()


def validate_language(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language}. Supported: {SUPPORTED_LANGUAGES}")


def ensure_list(x: MaybeList) -> List[Any]:
    return x if isinstance(x, list) else [x]


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False


def is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False


def decode_base64_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


def load_audio_any(x: str) -> Tuple[np.ndarray, int]:
    if is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def normalize_audio_input(a: AudioLike) -> np.ndarray:
    if isinstance(a, str):
        audio, sr = load_audio_any(a)
    elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        audio, sr = a[0], int(a[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(a)}")
    audio = to_mono(np.asarray(audio))
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    return float_range_normalize(audio)


def normalize_audios(audios: Union[AudioLike, List[AudioLike]]) -> List[np.ndarray]:
    return [normalize_audio_input(a) for a in ensure_list(audios)]


def chunk_list(xs: List[Any], chunk_size: int) -> Iterable[List[Any]]:
    if chunk_size <= 0:
        yield xs
        return
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


@dataclass(frozen=True)
class AudioChunk:
    orig_index: int
    chunk_index: int
    wav: np.ndarray
    sr: int
    offset_sec: float


def split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> List[Tuple[np.ndarray, float]]:
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1).astype(np.float32)
    total_len = int(wav.shape[0])
    total_sec = total_len / float(sr)
    if total_sec <= max_chunk_sec:
        return [(wav, 0.0)]
    max_len = int(max_chunk_sec * sr)
    expand = int(search_expand_sec * sr)
    win = max(4, int((min_window_ms / 1000.0) * sr))
    chunks: List[Tuple[np.ndarray, float]] = []
    start = 0
    offset_sec = 0.0
    while (total_len - start) > max_len:
        cut = start + max_len
        left = max(start, cut - expand)
        right = min(total_len, cut + expand)
        if right - left <= win:
            boundary = cut
        else:
            seg = wav[left:right]
            seg_abs = np.abs(seg)
            window_sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
            min_pos = int(np.argmin(window_sums))
            local = seg_abs[min_pos : min_pos + win]
            inner = int(np.argmin(local))
            boundary = left + min_pos + inner
        boundary = int(max(boundary, start + 1))
        boundary = int(min(boundary, total_len))
        chunk = wav[start:boundary]
        chunks.append((chunk, offset_sec))
        offset_sec += (boundary - start) / float(sr)
        start = boundary
    tail = wav[start:total_len]
    chunks.append((tail, offset_sec))
    min_len = int(MIN_ASR_INPUT_SECONDS * sr)
    padded: List[Tuple[np.ndarray, float]] = []
    for c, off in chunks:
        if c.shape[0] < min_len:
            pad = min_len - int(c.shape[0])
            c = np.pad(c, (0, pad), mode="constant", constant_values=0.0).astype(np.float32)
        padded.append((c, off))
    return padded


def detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    def fix_char_repeats(s: str, thresh: int) -> str:
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i : i + count])
                i += count
        return "".join(res)

    def fix_pattern_repeats(s: str, thresh: int, max_len: int = 20) -> str:
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        found = False
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i : i + k]
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx : start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index : end_index + k] == pattern:
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            result.append(s[i])
            i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text = fix_char_repeats(text, threshold)
    return fix_pattern_repeats(text, threshold)


def parse_asr_output(raw: str, user_language: Optional[str] = None) -> Tuple[str, str]:
    if raw is None:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""
    s = detect_and_fix_repetitions(s)
    if user_language:
        return user_language, s
    if _ASR_TEXT_TAG not in s:
        return "", s.strip()
    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    meta_lower = meta_part.lower()
    if "language none" in meta_lower:
        t = text_part.strip()
        return ("", "") if not t else ("", t)
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break
    return lang, text_part.strip()


def merge_languages(langs: List[str]) -> str:
    out: List[str] = []
    prev = None
    for x in langs:
        x = (x or "").strip()
        if not x or x == prev:
            continue
        out.append(x)
        prev = x
    return ",".join(out)


# ---------------------------------------------------------------------------
# Vendored modeling_qwen3_asr.py
# ---------------------------------------------------------------------------

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3ASRTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3ASRTextAttention(nn.Module):
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3ASRTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3ASRThinkerTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3ASRTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRTextMLP(config)
        self.input_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3ASRPreTrainedModel(PreTrainedModel):
    config: Qwen3ASRConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {"attentions": Qwen3ASRTextAttention}


@dataclass
class Qwen3ASRThinkerCausalLMOutputWithPast(MoeCausalLMOutputWithPast):
    """
    Args:
        rope_deltas (`torch.LongTensor` of shape `(batch_size,)`, optional):
            The rope index difference between sequence length and multimodal rope.
    """

    rope_deltas: Optional[torch.LongTensor] = None


class Qwen3ASRPreTrainedModelForConditionalGeneration(Qwen3ASRPreTrainedModel):
    def get_rope_index(self, attention_mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = attention_mask.float().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)
        return position_ids, mrope_position_deltas


class Qwen3ASRAudioAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.num_key_value_groups = 1
        self.config = config
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        seq_length, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.out_proj(attn_output)


class Qwen3ASRAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states, cu_seqlens=cu_seqlens, attention_mask=attention_mask, **kwargs)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        return (hidden_states,)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer("positional_embedding", torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1), persistent=False)


@auto_docstring(custom_intro="""
Transformer encoder consisting of self attention layers.
""")
class Qwen3ASRAudioEncoder(Qwen3ASRPreTrainedModel):
    config: Qwen3ASRAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__(config)
        self.positional_embedding = SinusoidsPositionEmbedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList([Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize
        self.post_init()

    def forward(self, input_features, feature_lens=None, aftercnn_lens=None):
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()
        chunk_lengths = torch.tensor([self.n_window * 2] * chunk_num.sum(), dtype=torch.long, device=feature_lens.device)
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2
        chunk_list_ = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list_, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))
        positional_embedding = self.positional_embedding.positional_embedding[: padded_embed.shape[1], :].unsqueeze(0).to(padded_embed.dtype)
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens)[0]
        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)


class Qwen3ASRThinkerTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3ASRConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling.get("rope_type", "default") if getattr(config, "rope_scaling", None) else "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3ASRThinkerTextAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = None

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(self, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor], attention_mask: Optional[torch.Tensor], past_key_values: Optional[Cache] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs: Unpack[FlashAttentionKwargs]):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


@auto_docstring(custom_intro="Text part of Qwen3ASRThinker")
class Qwen3ASRThinkerTextModel(Qwen3ASRPreTrainedModel):
    config: Qwen3ASRConfig
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3ASRThinkerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3ASRThinkerTextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    @auto_docstring
    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.LongTensor] = None, past_key_values: Optional[Cache] = None, inputs_embeds: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None, cache_position: Optional[torch.LongTensor] = None, **kwargs: Unpack[FlashAttentionKwargs]) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


@auto_docstring(custom_intro="The Qwen3ASRThinker model with audio encoder and language model.")
class Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRPreTrainedModelForConditionalGeneration, GenerationMixin):
    config: Qwen3ASRThinkerConfig
    base_model_prefix = "thinker"
    _tied_weights_keys = ["model.embed_tokens.weight", "lm_head.weight"]
    _no_split_modules = ["Qwen3ASRAudioEncoderLayer", "Qwen3ASRThinkerTextDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_audio_features(self, input_features: torch.FloatTensor, feature_attention_mask: Optional[torch.LongTensor] = None, audio_feature_lengths: Optional[torch.LongTensor] = None):
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None
        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            audio_output = self.audio_tower(input_feature[:, :feature_len], feature_lens=feature_len.unsqueeze(0))
            audio_features.append(audio_output.last_hidden_state)
        return torch.cat(audio_features, dim=0)

    def get_placeholder_mask(self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor):
        if input_ids is None:
            special_audio_mask = (inputs_embeds == self.get_input_embeddings()(torch.tensor(self.config.audio_token_id, dtype=torch.long, device=inputs_embeds.device))).all(-1)
        else:
            special_audio_mask = input_ids == self.config.audio_token_id
        return special_audio_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)

    @can_return_tuple
    @auto_docstring
    def forward(self, input_ids=None, input_features=None, attention_mask=None, feature_attention_mask=None, audio_feature_lengths=None, position_ids=None, past_key_values=None, inputs_embeds=None, rope_deltas=None, labels=None, use_cache=None, cache_position=None, **kwargs) -> Union[tuple, Qwen3ASRThinkerCausalLMOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if input_features is not None:
            audio_features = self.get_audio_features(input_features, feature_attention_mask=feature_attention_mask, audio_feature_lengths=audio_feature_lengths)
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(attention_mask)
                self.rope_deltas = rope_deltas - delta0
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device).view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta).unsqueeze(0).expand(3, -1, -1)
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size)
        return Qwen3ASRThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, position_ids=None, use_cache=True, input_features=None, feature_attention_mask=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            **kwargs,
        )
        model_inputs["position_ids"] = None
        if cache_position[0] != 0:
            model_inputs["input_features"] = None
        return model_inputs


class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.config = config
        self.thinker = Qwen3ASRThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()

    def get_support_languages(self):
        return self.config.support_languages

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor] = None, max_new_tokens: int = 4096, eos_token_id: int | list[int] = [151645, 151643], **kwargs):
        shared_kwargs = {}
        thinker_kwargs = {"max_new_tokens": max_new_tokens, "eos_token_id": eos_token_id}
        for key, value in kwargs.items():
            if key == "feature_attention_mask":
                thinker_kwargs[key] = value
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
        return self.thinker.generate(input_ids=input_ids, return_dict_in_generate=True, **thinker_kwargs)


# ---------------------------------------------------------------------------
# Standalone wrapper for the exact demo.py path
# ---------------------------------------------------------------------------

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


@dataclass
class ASRChunkTrace:
    orig_index: int
    chunk_index: int
    offset_sec: float
    duration_sec: float
    prompt: str
    input_ids_shape: Tuple[int, ...]
    input_features_shape: Tuple[int, ...]
    feature_attention_mask_shape: Tuple[int, ...]
    raw_decoded: str
    language: str
    text: str


@dataclass
class ASRTranscription:
    language: str
    text: str
    chunk_traces: Optional[List[ASRChunkTrace]] = None


class Qwen3ASRModel:
    def __init__(self, model: Any, processor: Any, max_inference_batch_size: int = 32, max_new_tokens: int = 512):
        self.backend = "transformers"
        self.model = model
        self.processor = processor
        self.max_inference_batch_size = int(max_inference_batch_size)
        self.max_new_tokens = max_new_tokens
        self.device = getattr(model, "device", None)
        if self.device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        self.dtype = getattr(model, "dtype", torch.float32)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, max_inference_batch_size: int = 32, max_new_tokens: int = 512, **kwargs) -> "Qwen3ASRModel":
        model = AutoModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, fix_mistral_regex=True)
        return cls(model=model, processor=processor, max_inference_batch_size=max_inference_batch_size, max_new_tokens=max_new_tokens)

    def get_supported_languages(self) -> List[str]:
        return list(SUPPORTED_LANGUAGES)

    def _build_messages(self, context: str, audio_payload: Any) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_payload}]},
        ]

    def _build_text_prompt(self, context: str, force_language: Optional[str]) -> str:
        msgs = self._build_messages(context=context, audio_payload="")
        base = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        if force_language:
            base = base + f"language {force_language}<asr_text>"
        return base

    @torch.no_grad()
    def _infer_asr_transformers(
        self,
        contexts: List[str],
        wavs: List[np.ndarray],
        languages: List[Optional[str]],
        chunk_meta: List[AudioChunk],
        trace: bool,
    ) -> Tuple[List[str], List[ASRChunkTrace]]:
        outs: List[str] = []
        traces: List[ASRChunkTrace] = []
        texts = [self._build_text_prompt(context=c, force_language=fl) for c, fl in zip(contexts, languages)]
        batch_size = self.max_inference_batch_size
        if batch_size is None or batch_size < 0:
            batch_size = len(texts)
        for i in range(0, len(texts), batch_size):
            sub_text = texts[i : i + batch_size]
            sub_wavs = wavs[i : i + batch_size]
            sub_meta = chunk_meta[i : i + batch_size]
            sub_langs = languages[i : i + batch_size]
            inputs = self.processor(text=sub_text, audio=sub_wavs, return_tensors="pt", padding=True)
            input_ids_shape = tuple(inputs["input_ids"].shape)
            input_features_shape = tuple(inputs["input_features"].shape)
            feature_attention_mask_shape = tuple(inputs["feature_attention_mask"].shape)
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            text_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            decoded = self.processor.batch_decode(
                text_ids.sequences[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            outs.extend(list(decoded))
            if trace:
                for prompt, meta, forced_lang, raw_decoded in zip(sub_text, sub_meta, sub_langs, decoded):
                    language, text = parse_asr_output(raw_decoded, user_language=forced_lang)
                    traces.append(
                        ASRChunkTrace(
                            orig_index=meta.orig_index,
                            chunk_index=meta.chunk_index,
                            offset_sec=meta.offset_sec,
                            duration_sec=meta.wav.shape[0] / float(meta.sr),
                            prompt=prompt,
                            input_ids_shape=input_ids_shape,
                            input_features_shape=input_features_shape,
                            feature_attention_mask_shape=feature_attention_mask_shape,
                            raw_decoded=raw_decoded,
                            language=language,
                            text=text,
                        )
                    )
        return outs, traces

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[AudioLike, List[AudioLike]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        trace: bool = False,
    ) -> List[ASRTranscription]:
        wavs = normalize_audios(audio)
        n = len(wavs)
        ctxs = context if isinstance(context, list) else [context]
        if len(ctxs) == 1 and n > 1:
            ctxs = ctxs * n
        if len(ctxs) != n:
            raise ValueError(f"Batch size mismatch: audio={n}, context={len(ctxs)}")
        if language is None:
            langs_in = [None] * n
        else:
            langs_in = language if isinstance(language, list) else [language]
            if len(langs_in) == 1 and n > 1:
                langs_in = langs_in * n
            if len(langs_in) != n:
                raise ValueError(f"Batch size mismatch: audio={n}, language={len(langs_in)}")
        langs_norm: List[Optional[str]] = []
        for item in langs_in:
            if item is None or str(item).strip() == "":
                langs_norm.append(None)
            else:
                ln = normalize_language_name(str(item))
                validate_language(ln)
                langs_norm.append(ln)
        chunks: List[AudioChunk] = []
        for i, wav in enumerate(wavs):
            parts = split_audio_into_chunks(wav=wav, sr=SAMPLE_RATE, max_chunk_sec=MAX_ASR_INPUT_SECONDS)
            for j, (chunk_wav, offset_sec) in enumerate(parts):
                chunks.append(AudioChunk(orig_index=i, chunk_index=j, wav=chunk_wav, sr=SAMPLE_RATE, offset_sec=offset_sec))
        chunk_ctx = [ctxs[c.orig_index] for c in chunks]
        chunk_lang = [langs_norm[c.orig_index] for c in chunks]
        chunk_wavs = [c.wav for c in chunks]
        raw_outputs, traces = self._infer_asr_transformers(chunk_ctx, chunk_wavs, chunk_lang, chunks, trace)
        per_chunk_lang: List[str] = []
        per_chunk_text: List[str] = []
        for out, forced_lang in zip(raw_outputs, chunk_lang):
            lang, txt = parse_asr_output(out, user_language=forced_lang)
            per_chunk_lang.append(lang)
            per_chunk_text.append(txt)
        out_langs: List[List[str]] = [[] for _ in range(n)]
        out_texts: List[List[str]] = [[] for _ in range(n)]
        out_traces: List[List[ASRChunkTrace]] = [[] for _ in range(n)]
        for c, lang_pred, txt in zip(chunks, per_chunk_lang, per_chunk_text):
            out_langs[c.orig_index].append(lang_pred)
            out_texts[c.orig_index].append(txt)
        if trace:
            for trace_item in traces:
                out_traces[trace_item.orig_index].append(trace_item)
        results: List[ASRTranscription] = []
        for i in range(n):
            results.append(
                ASRTranscription(
                    language=merge_languages(out_langs[i]),
                    text="".join([t for t in out_texts[i] if t is not None]),
                    chunk_traces=out_traces[i] if trace else None,
                )
            )
        return results


def print_trace(result: ASRTranscription) -> None:
    if not result.chunk_traces:
        return
    print("\nChunk Trace")
    print("-" * 60)
    for item in result.chunk_traces:
        print(f"chunk={item.chunk_index} offset={item.offset_sec:.3f}s dur={item.duration_sec:.3f}s")
        print(f"input_ids_shape={item.input_ids_shape}")
        print(f"input_features_shape={item.input_features_shape}")
        print(f"feature_attention_mask_shape={item.feature_attention_mask_shape}")
        print(f"prompt={item.prompt}")
        print(f"raw_decoded={item.raw_decoded}")
        print(f"parsed_language={item.language}")
        print(f"parsed_text={item.text}")
        print("-" * 60)


if __name__ == "__main__":
    AUDIO_URL = "../example/en.mp3"
    print("Loading Qwen3-ASR-0.6B model...")
    model = Qwen3ASRModel.from_pretrained(
        "/home/DakeQQ/Downloads/Qwen3-ASR-0.6B",
        dtype=torch.float32,
        device_map="cpu",
        max_inference_batch_size=32,
        max_new_tokens=256,
    )
    print(f"Transcribing: {AUDIO_URL}")
    results = model.transcribe(audio=AUDIO_URL, language=None, trace=True)
    print(f"\n{'=' * 50}")
    print(f"Language: {results[0].language}")
    print(f"Text: {results[0].text}")
    print(f"{'=' * 50}")
    print_trace(results[0])
