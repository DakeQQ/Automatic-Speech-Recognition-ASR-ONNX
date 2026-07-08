import gc
import subprocess
import sys
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel

from STFT_Process import STFT_Process

# ──────────────────────────────────────────────────────────────────────────────
# Qwen3-ForcedAligner-0.6B  ·  deeply-optimized ONNX export + inference demo
# ──────────────────────────────────────────────────────────────────────────────
# The forced aligner is NON auto-regressive (NAR): given an audio clip and its
# transcript it classifies, in ONE forward pass, a timestamp bucket at every
# "<timestamp>" position. There is therefore NO KV cache, NO decode loop and NO
# beam/greedy search — a dramatic simplification over the ASR decoder.
#
# Pipeline (mirrors qwen_asr/inference/qwen3_forced_aligner.py):
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
download_path               = r'/home/DakeQQ/Downloads/Qwen3-ForcedAligner-0.6B'             # Local Qwen3-ForcedAligner-0.6B directory (must contain model.safetensors).
onnx_folder                 = Path(__file__).resolve().parent / "Qwen_ForcedAligner_ONNX"  # Local folder next to this script holding all exported ONNX graphs; created automatically if missing.
onnx_folder.mkdir(parents=True, exist_ok=True)
onnx_model_Metadata         = str(onnx_folder / "ForcedAligner_Metadata.onnx")
onnx_model_Embed            = str(onnx_folder / "ForcedAligner_Embed.onnx")
onnx_model_Encoder          = str(onnx_folder / "ForcedAligner_Encoder.onnx")
onnx_model_Rotary_Mask      = str(onnx_folder / "ForcedAligner_Rotary_Mask.onnx")
onnx_model_Main             = str(onnx_folder / "ForcedAligner_Decoder_Main.onnx")


# ══════════════════════════════════════════════════════════════════════════════
# Audio & STFT Configuration  (identical to Qwen3-ASR-0.6B front-end)
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE                 = 16000      # Model parameter, do not edit.
WINDOW_TYPE                 = 'hann'     # STFT window function.
N_MELS                      = 128        # Mel bands. Do not edit.
NFFT_STFT                   = 400        # FFT size.
WINDOW_LENGTH               = 400        # Window length.
HOP_LENGTH                  = 160        # STFT hop.
INPUT_AUDIO_DTYPE           = "INT16"    # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. "INT16" feeds raw PCM (÷32768 in-graph); "F32"/"F16" feed audio pre-normalised to [-1, 1].

# Model / runtime limits
MAX_INPUT_AUDIO_LENGTH      = SAMPLE_RATE * 30  # Max audio the encoder buffers are sized for (30 s). Raise for longer clips.
MAX_SEQ_LEN                 = 1024              # Max total sequence length (audio tokens + 2 + text tokens). Scale with audio length.
PREVENT_F16_OVERFLOW        = False             # Prevent float16 overflow. Set True for Q4F16 / Q8F16 / F16 quantization.

# Weight-Quantization-Friendly Reorder (EXACT, zero runtime cost; helps only when weight-quant group_size < head_dim)
REORDER_DOWNPROJ_FOR_QUANT  = True       # Reorder MLP intermediate channels so down_proj block-quant groups are magnitude-homogeneous (absorbed into gate_up + down_proj).
REORDER_OPROJ_FOR_QUANT     = True       # Reorder each head's head_dim so o_proj sub-head groups are homogeneous (compensated on the qkv v-rows).
REORDER_KEY                 = "absmean"  # Channel key: "absmean" | "L4" | "rms" | "std".

DYNAMIC_AXES                = True       # Default dynamic axis is the input length.

# Runtime & export settings
OPSET                       = 20         # ONNX opset version.
# ══════════════════════════════════════════════════════════════════════════════
# Special-Token IDs & timestamp buckets  (verified from config.json / tokenizer_config.json)
# ══════════════════════════════════════════════════════════════════════════════
AUDIO_START_TOKEN_ID        = 151669     # <|audio_start|>
AUDIO_END_TOKEN_ID          = 151670     # <|audio_end|>
AUDIO_PAD_TOKEN_ID          = 151676     # <|audio_pad|> (audio placeholder)
TIMESTAMP_TOKEN_ID          = 151705     # <timestamp>
TIMESTAMP_SEGMENT_TIME      = 80         # ms per classification bucket.


def build_model_metadata(*sections):
    metadata = {}
    for section in sections:
        for key, value in section.items():
            if value is None:
                continue
            if isinstance(value, bool):
                metadata[str(key)] = "1" if value else "0"
            elif isinstance(value, (list, tuple)):
                metadata[str(key)] = ",".join(str(item) for item in value)
            else:
                metadata[str(key)] = str(value)
    return metadata


def write_onnx_metadata(onnx_path, metadata):
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


def _token_id(tokenizer, token, fallback):
    try:
        value = tokenizer.convert_tokens_to_ids(token)
        if value is not None and value != getattr(tokenizer, "unk_token_id", None):
            return int(value)
    except Exception:
        pass
    try:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) == 1:
            return int(ids[0])
    except Exception:
        pass
    return int(fallback)


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


# ══════════════════════════════════════════════════════════════════════════════
# ── Vendored Qwen3-ForcedAligner configuration + modeling (transformers==4.57.6) ─
#
# The Qwen3-ForcedAligner configuration and modeling classes are inlined here (they
# previously lived in ./modeling_modified/inference_forced_aligner_standalone.py)
# so this export script is fully standalone. The forced aligner shares the exact
# Qwen3-ASR audio encoder and Qwen3 text decoder; only the top-level head differs
# (a ``classify_num``-bucket timestamp classifier instead of the vocab-size LM
# head). Only the closure required to load the checkpoint through
# AutoModel.from_pretrained is kept; the processor, audio/text I/O helpers and the
# high-level inference wrapper were dropped because every forward() consumed by the
# export is re-implemented below.
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuration ─────────────────────────────────────────────────────────────
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
        # Build the sub-configs BEFORE `super().__init__()`: transformers>=5 runs
        # `@strict` validators (e.g. `validate_token_ids` -> `get_text_config()`)
        # inside `super().__init__()`, so `self.text_config` must already exist.
        # On the transformers 4.x path this ordering is harmless.
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
        self.user_token_id = user_token_id
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range
        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        # `getattr` guard stays safe even if a validator calls this before
        # `__init__` has finished assigning `text_config`.
        return getattr(self, "text_config", self)


class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {"thinker_config": Qwen3ASRThinkerConfig}

    def __init__(self, thinker_config=None, support_languages=None, **kwargs):
        # Build the sub-config BEFORE `super().__init__()`: transformers>=5 runs
        # `@strict` validators inside `super().__init__()` that call
        # `get_text_config()` -> `self.thinker_config`, which would otherwise not
        # exist yet (AttributeError on newer transformers).
        if thinker_config is None:
            thinker_config = {}
        if isinstance(thinker_config, dict):
            thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.thinker_config = thinker_config
        self.support_languages = support_languages
        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        thinker_config = getattr(self, "thinker_config", None)
        if thinker_config is None:
            return self
        return thinker_config.get_text_config(decoder=decoder, encoder=encoder)


# ── Modeling ──────────────────────────────────────────────────────────────────
# Only the ``__init__`` closure that builds the module tree is kept (so
# AutoModel.from_pretrained can load the checkpoint and expose the weights/buffers
# the ONNX graphs read). Every forward()/generate() + their helpers were removed —
# the export re-implements all computation in the FORCED_ALIGNER_* wrappers below.
class Qwen3ASRTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


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


class Qwen3ASRTextMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


class Qwen3ASRThinkerTextDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3ASRTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRTextMLP(config)
        self.input_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


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


class Qwen3ASRPreTrainedModelForConditionalGeneration(Qwen3ASRPreTrainedModel):
    pass


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


class Qwen3ASRAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig):
        super().__init__()
        self.self_attn = Qwen3ASRAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)
        self.final_layer_norm = nn.LayerNorm(config.d_model)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer("positional_embedding", torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1), persistent=False)


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


class Qwen3ASRThinkerTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3ASRConfig, device=None):
        super().__init__()
        self.rope_type = config.rope_scaling.get("rope_type", "default") if getattr(config, "rope_scaling", None) else "default"
        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS.get(self.rope_type)
        if self.rope_init_fn is not None:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        else:
            # transformers>=5 removed the "default" entry from ROPE_INIT_FUNCTIONS
            # (plain RoPE is no longer registry-based). Compute the standard,
            # non-scaled inverse frequencies directly so this module stays agnostic
            # to the installed transformers version.
            base = getattr(config, "rope_theta", None)
            if base is None and getattr(config, "rope_scaling", None):
                base = config.rope_scaling.get("rope_theta")
            base = 10000.0 if base is None else base
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
            self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)


class Qwen3ASRThinkerTextModel(Qwen3ASRPreTrainedModel):
    config: Qwen3ASRConfig
    _no_split_modules = ["Qwen3ASRThinkerTextDecoderLayer"]
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Qwen3ASRThinkerTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3ASRThinkerTextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()


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
        # transformers>=5 pops generation token-ids (e.g. pad_token_id) out of the
        # config, so it may no longer be an attribute; fall back to -1 defensively.
        pad_token_id = getattr(self.config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)


class Qwen3ASRForConditionalGeneration(Qwen3ASRPreTrainedModel, GenerationMixin):
    config_class = Qwen3ASRConfig

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__(config)
        self.config = config
        self.thinker = Qwen3ASRThinkerForConditionalGeneration._from_config(config.thinker_config)
        self.post_init()


# ── Forced-aligner head overrides ─────────────────────────────────────────────
# The forced aligner reuses every Qwen3-ASR module above; only the top-level head
# changes from a vocab-size LM head to a ``classify_num``-bucket timestamp
# classifier. These two subclasses swap that head in (weight tying is disabled
# because the classify head no longer shares shape with the token embeddings).
class Qwen3ForcedAlignerThinker(Qwen3ASRThinkerForConditionalGeneration):
    """Thinker whose head is a timestamp classifier (``classify_num`` buckets)."""

    # The classify head is independent of the input embeddings (different shape),
    # so weight tying must never be attempted.
    _tied_weights_keys = None

    def __init__(self, config):
        # Bypass Qwen3ASRThinkerForConditionalGeneration.__init__ which would build a
        # vocab_size LM head; replicate everything except the head construction.
        Qwen3ASRPreTrainedModelForConditionalGeneration.__init__(self, config)
        self.audio_tower = Qwen3ASRAudioEncoder._from_config(config.audio_config)
        self.vocab_size = config.text_config.vocab_size
        self.model = Qwen3ASRThinkerTextModel._from_config(config.text_config)
        classify_num = int(getattr(config, "classify_num", 5000))
        self.lm_head = nn.Linear(config.text_config.hidden_size, classify_num, bias=False)
        pad_id = getattr(config, "pad_token_id", None)
        self.pad_token_id = pad_id if pad_id is not None else -1
        self.rope_deltas = None
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        # Classify head is not tied to the token embedding table.
        return

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, value):
        return


class Qwen3ForcedAlignerForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    """Top-level forced aligner model that owns the classification thinker."""

    config_class = Qwen3ASRConfig

    def __init__(self, config):
        # Bypass Qwen3ASRForConditionalGeneration.__init__ so the thinker is our
        # classification variant rather than the LM-head variant.
        Qwen3ASRPreTrainedModel.__init__(self, config)
        self.config = config
        self.thinker = Qwen3ForcedAlignerThinker._from_config(config.thinker_config)
        self.post_init()

    def tie_weights(self, *args, **kwargs):
        return


# Register so ``AutoModel.from_pretrained`` resolves the forced aligner variant
# (config `model_type` == "qwen3_asr" -> Qwen3ForcedAlignerForConditionalGeneration).
AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ForcedAlignerForConditionalGeneration)


# ══════════════════════════════════════════════════════════════════════════════
# ── Weight-fusion helpers (shared with the ASR export) ────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _get_feat_extract_output_lengths(input_lengths: Tensor) -> Tensor:
    input_lengths_leave = input_lengths % 100
    feat_lengths = torch.clamp(input_lengths_leave - 1, min=0) // 2 + 1
    feat_lengths = feat_lengths * (input_lengths_leave > 0).to(feat_lengths.dtype)
    feat_lengths_2 = torch.clamp(feat_lengths - 1, min=0) // 2 + 1
    feat_lengths_2 = feat_lengths_2 * (feat_lengths > 0).to(feat_lengths_2.dtype)
    feat_lengths_3 = torch.clamp(feat_lengths_2 - 1, min=0) // 2 + 1
    feat_lengths_3 = feat_lengths_3 * (feat_lengths_2 > 0).to(feat_lengths_3.dtype)
    return feat_lengths_3 + (input_lengths // 100) * 13


def replace_gelu_with_tanh(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh(child)


def absorb_layer_norm_affine(norm: torch.nn.LayerNorm, linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype))
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias   = None


def refresh_non_persistent_buffers(model: torch.nn.Module, text_cfg) -> None:
    """Re-materialize the ``__init__``-computed, non-persistent buffers this export
    reads directly. transformers>=5 overwrites every non-persistent buffer with
    ``torch.empty_like`` during ``from_pretrained`` and only re-initializes a known
    set of module types afterwards (a RotaryEmbedding is only refilled when it owns
    an ``original_inv_freq`` attribute, and the Whisper-style sinusoidal position
    embedding is not handled at all) — leaving them as uninitialized memory. That
    silently corrupts the RoPE table and audio positions, producing garbage output.
    On transformers 4.x these buffers already hold the correct values, so this is a
    harmless no-op there."""
    # ── Default/plain RoPE inverse frequencies ────────────────────────────
    llm = getattr(model.thinker, "model", None)
    rotary = getattr(llm, "rotary_emb", None)
    if rotary is not None and getattr(rotary, "inv_freq", None) is not None:
        base = getattr(text_cfg, "rope_theta", None)
        if base is None and getattr(text_cfg, "rope_scaling", None):
            base = text_cfg.rope_scaling.get("rope_theta")
        base = 10000.0 if base is None else float(base)
        head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads
        dim = int(head_dim * getattr(text_cfg, "partial_rotary_factor", 1.0))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        rotary.inv_freq = inv_freq.to(rotary.inv_freq.dtype)
        if hasattr(rotary, "original_inv_freq"):
            rotary.original_inv_freq = rotary.inv_freq
    # ── Whisper-style sinusoidal audio positional embedding ─────────────────
    sinu = getattr(model.thinker.audio_tower, "positional_embedding", None)
    if sinu is not None and getattr(sinu, "positional_embedding", None) is not None:
        length, channels = int(sinu.positional_embedding.shape[0]), int(sinu.positional_embedding.shape[1])
        log_timescale_increment = np.log(10000) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
        sinu.positional_embedding = pe.to(sinu.positional_embedding.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# ── Audio Encoder (fused mel front-end + windowed encoder + prompt concat) ────
# ══════════════════════════════════════════════════════════════════════════════
class FORCED_ALIGNER_ENCODER(torch.nn.Module):
    """Audio path identical to the Qwen3-ASR encoder; emits the NAR input embedding
    sequence [<|audio_start|>, audio, <|audio_end|>, text/timestamp embeddings]."""

    def __init__(self, audio_tower: torch.nn.Module, embed_tokens: torch.nn.Embedding) -> None:
        super().__init__()
        self.audio_tower = audio_tower.float()
        replace_gelu_with_tanh(self.audio_tower)
        self.audio_tower.act = torch.nn.GELU(approximate="tanh")
        for layer in self.audio_tower.layers:
            layer.activation_fn = torch.nn.GELU(approximate="tanh")
        self._fuse_encoder_weights()

        self.stft = STFT_Process(
            model_type="stft_B",
            n_fft=NFFT_STFT,
            win_length=WINDOW_LENGTH,
            hop_len=HOP_LENGTH,
            max_frames=0,
            window_type=WINDOW_TYPE,
            center_pad=True,
            pad_mode="reflect",
        ).eval()

        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=(NFFT_STFT // 2) + 1,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1).unsqueeze(0).contiguous()
        self.register_buffer("mel_filters", mel_filters.to(dtype=torch.float32), persistent=False)

        chunk_size            = int(self.audio_tower.n_window * 2)          # 100
        self.chunk_size       = chunk_size
        self.chunk_size_minus = chunk_size - 1
        chunk_aftercnn        = int(_get_feat_extract_output_lengths(torch.tensor([chunk_size])).item())  # 13
        self.chunk_aftercnn   = chunk_aftercnn
        chunks_per_window     = int(self.audio_tower.n_window_infer // chunk_size)  # 8
        self.chunks_per_window = chunks_per_window
        self.chunks_per_window_minus = chunks_per_window - 1
        tokens_per_window     = chunks_per_window * chunk_aftercnn           # 104
        self.tokens_per_window = tokens_per_window
        self.model_dim        = int(self.audio_tower.config.d_model)
        self.output_dim       = int(self.audio_tower.config.output_dim)
        self.num_heads        = int(self.audio_tower.layers[0].self_attn.num_heads)
        self.head_dim         = self.model_dim // self.num_heads

        max_mel_frames  = MAX_INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
        self.max_chunks  = (max_mel_frames + self.chunk_size_minus) // chunk_size
        self.max_windows = (self.max_chunks + chunks_per_window - 1) // chunks_per_window
        max_total_chunks_padded = self.max_windows * chunks_per_window
        key_mask_lookup = torch.zeros(1, tokens_per_window + 1, 1, 1, tokens_per_window, dtype=torch.int8)
        for n in range(tokens_per_window + 1):
            key_mask_lookup[0, n, 0, 0, n:] = -128
        valid_mask_float_lookup = torch.zeros(tokens_per_window + 1, tokens_per_window, 1, dtype=torch.int8)
        for n in range(tokens_per_window + 1):
            valid_mask_float_lookup[n, :n, 0] = 1

        self.pos = self.audio_tower.positional_embedding.positional_embedding[:self.chunk_aftercnn].unsqueeze(0)
        self.register_buffer("window_chunks_full", torch.arange(self.max_windows + 1, dtype=torch.int32) * chunks_per_window, persistent=False)
        self.register_buffer("chunk_starts_full", torch.arange(self.max_chunks + 1, dtype=torch.int32) * chunk_size, persistent=False)
        self.register_buffer("aftercnn_lens_lookup", _get_feat_extract_output_lengths(torch.arange(chunk_size + 1)).int(), persistent=False)
        self.register_buffer("chunk_pad_zeros", torch.zeros((max_total_chunks_padded, chunk_aftercnn, self.model_dim), dtype=torch.int8), persistent=False)
        self.register_buffer("aftercnn_pad_zeros", torch.zeros(chunks_per_window, dtype=torch.int8), persistent=False)
        self.register_buffer("key_mask_lookup", key_mask_lookup, persistent=False)
        self.register_buffer("valid_mask_float_lookup", valid_mask_float_lookup, persistent=False)
        self.register_buffer("mel_pad_zeros", torch.zeros((1, N_MELS, chunk_size), dtype=torch.int8), persistent=False)
        self.register_buffer("inv_int16", torch.tensor([1.0 / 32768.0], dtype=torch.float32), persistent=False)
        # int16 audio is raw PCM (normalised in forward via ÷32768); f32/f16 audio is
        # assumed pre-normalised to [-1, 1], so the in-graph division is skipped.
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")

        with torch.no_grad():
            audio_start = torch.tensor([[AUDIO_START_TOKEN_ID]], dtype=torch.int32)
            audio_end   = torch.tensor([[AUDIO_END_TOKEN_ID]], dtype=torch.int32)
            self.register_buffer("audio_start_embed", embed_tokens(audio_start).float(), persistent=False)
            self.register_buffer("audio_end_embed", embed_tokens(audio_end).float(), persistent=False)

    def _fuse_encoder_weights(self) -> None:
        with torch.no_grad():
            for layer in self.audio_tower.layers:
                attn = layer.self_attn
                qkv  = torch.nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features, bias=True)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                qkv.bias.copy_(torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0))
                absorb_layer_norm_affine(layer.self_attn_layer_norm, qkv)

                scale_sqrt = attn.scaling ** 0.5
                q_out = attn.q_proj.out_features
                k_out = attn.k_proj.out_features
                qkv.weight.data[:q_out].mul_(scale_sqrt)
                qkv.weight.data[q_out:q_out + k_out].mul_(scale_sqrt)
                qkv.bias.data[:q_out].mul_(scale_sqrt)
                qkv.bias.data[q_out:q_out + k_out].mul_(scale_sqrt)

                absorb_layer_norm_affine(layer.final_layer_norm, layer.fc1)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj

            absorb_layer_norm_affine(self.audio_tower.ln_post, self.audio_tower.proj1)

    def forward(self, audio: Tensor, text_embed: Tensor) -> Tuple[Tensor, Tensor]:
        # WhisperFeatureExtractor-equivalent log-mel front-end (no pre-emphasis / DC removal).
        if self.input_audio_is_int16:
            audio = audio.float() * self.inv_int16
        else:
            audio = audio.float()
        real, imag = self.stft(audio)
        real, imag = real[..., :-1], imag[..., :-1]
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_filters, power)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        input_features = mel * 0.25 + 1.0
        feature_len = input_features.shape[-1].unsqueeze(0)
        num_chunks = (feature_len + self.chunk_size_minus) // self.chunk_size
        pad_frames = self.chunk_starts_full[num_chunks] - feature_len
        padded_features = torch.cat([input_features, self.mel_pad_zeros[..., :pad_frames].float()], dim=-1)
        chunks = padded_features.reshape(1, N_MELS, num_chunks, self.chunk_size).permute(2, 0, 1, 3)
        chunk_starts = self.chunk_starts_full[:num_chunks]
        raw_chunk_lens = torch.clamp(feature_len - chunk_starts, min=0, max=self.chunk_size)
        aftercnn_lens = self.aftercnn_lens_lookup[raw_chunk_lens]
        x = F.gelu(self.audio_tower.conv2d1(chunks), approximate="tanh")
        x = F.gelu(self.audio_tower.conv2d2(x), approximate="tanh")
        x = F.gelu(self.audio_tower.conv2d3(x), approximate="tanh")
        x = self.audio_tower.conv_out(x.permute(0, 3, 1, 2).contiguous().view(num_chunks, self.chunk_aftercnn, -1))
        hidden_states = x + self.pos
        num_windows = (num_chunks + self.chunks_per_window_minus) // self.chunks_per_window
        total_chunks_padded = self.window_chunks_full[num_windows]
        pad_chunks = total_chunks_padded - num_chunks
        hidden_states = torch.cat([hidden_states, self.chunk_pad_zeros[:pad_chunks].float()], dim=0)
        aftercnn_lens = torch.cat([aftercnn_lens, self.aftercnn_pad_zeros[:pad_chunks].int()])
        hidden_states = hidden_states.reshape(num_windows, self.tokens_per_window, self.model_dim)
        valid_counts = aftercnn_lens.reshape(num_windows, self.chunks_per_window).sum(dim=1, dtype=torch.int32)
        valid_mask_float = self.valid_mask_float_lookup[valid_counts].float()
        key_mask = self.key_mask_lookup[:, valid_counts].float()
        for layer in self.audio_tower.layers:
            residual = hidden_states
            normed = layer.self_attn_layer_norm(hidden_states)
            qkv = layer.self_attn.qkv(normed).view(num_windows, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.split(1, dim=0)
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(attn + key_mask, dim=-1)
            attn = torch.matmul(attn, v).transpose(2, 3).reshape(num_windows, -1, layer.self_attn.out_proj.in_features)
            hidden_states = residual + layer.self_attn.out_proj(attn)
            residual = hidden_states
            normed = layer.final_layer_norm(hidden_states)
            hidden_states = residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))
        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj2(self.audio_tower.act(self.audio_tower.proj1(hidden_states)))
        hidden_states = hidden_states * valid_mask_float
        hidden_states = hidden_states.reshape(1, -1, self.output_dim)
        encoded_len = aftercnn_lens.sum(dtype=torch.int32).long()
        audio_hidden = hidden_states[:, :encoded_len]
        concat_embed = torch.cat([self.audio_start_embed, audio_hidden, self.audio_end_embed, text_embed], dim=1)
        ids_len = concat_embed.shape[1].unsqueeze(0)
        return concat_embed, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoder Embedding ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class FORCED_ALIGNER_EMBED(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embed_tokens = model.thinker.model.embed_tokens.float()

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# ── Rotary + Causal Mask (single NAR forward, history_len == 0) ───────────────
# ══════════════════════════════════════════════════════════════════════════════
class FORCED_ALIGNER_ROTARY_MASK(torch.nn.Module):
    def __init__(self, llm: torch.nn.Module, max_seq_len: int) -> None:
        super().__init__()
        self.register_buffer(
            "attention_mask",
            (1.0 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128,
            persistent=False,
        )
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = llm.rotary_emb.inv_freq.float()
        theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos, sin = torch.cos(theta), torch.sin(theta)
        # flip-RoPE: pre-duplicate cos and pre-negate/duplicate sin (rotate_half via view+flip).
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, ids_len: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        rotary_cos = self.cos_rotary_pos_emb[:, :ids_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, :ids_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :ids_len].float()
        return rotary_cos, rotary_sin, attention_mask


# ══════════════════════════════════════════════════════════════════════════════
# ── Simplified RMS LayerNorm (ORT fused SimplifiedLayerNormalization) ──────────
# ══════════════════════════════════════════════════════════════════════════════
class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Emit ONNX Runtime's fused ``SimplifiedLayerNormalization`` (RMS norm) in place of the
    Mul -> ReduceSum -> Add -> Sqrt -> Div chain.

    ORT registers this op in the DEFAULT ONNX domain (``kOnnxDomain``) with SinceVersion 1, so it is
    available at any opset. It computes ``Y = X * rsqrt(mean(X^2, axis) + epsilon) * scale`` and, with
    ``stash_type = 1``, performs the mean/rsqrt reduction in float32 regardless of the I/O dtype (so no
    manual float16-overflow guard is needed). ``forward`` runs only during export tracing; the exported
    graph uses ``symbolic`` (the whole Function collapses into one node).
    """

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance   = x.float().pow(2).mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x, scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    """Return ``SimplifiedLayerNormalization(x, scale, axis, epsilon)`` -- a single fused RMS-norm node."""
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoder Main (deeply fused transformer, NAR, timestamp classifier) ────────
# ══════════════════════════════════════════════════════════════════════════════
class FORCED_ALIGNER_DECODER_MAIN(torch.nn.Module):
    def __init__(
        self,
        model:        torch.nn.Module,
        num_heads:    int,
        num_kv_heads: int,
        head_dim:     int,
        num_layers:   int,
        hidden_size:  int,
    ) -> None:
        super().__init__()
        self.llm      = model.thinker.model.float()
        self.lm_head  = model.thinker.lm_head.float()   # classification head (classify_num buckets)
        self.head_dim       = head_dim
        self.head_dim_half  = head_dim // 2
        self.num_heads      = num_heads
        self.num_kv_heads   = num_kv_heads
        self.num_kv_groups  = num_heads // num_kv_heads
        self.qk_heads       = num_heads + num_kv_heads
        self.num_layers     = num_layers

        # RMS norm is emitted as ORT's fused SimplifiedLayerNormalization (default ONNX domain): it computes
        # y = x * rsqrt(mean(x^2) + eps) * scale and reduces in float32 (stash_type=1). Feeding scale = 1/sqrt(N)
        # (N = normalized size) and the model's per-element eps reproduces this file's sum-based
        # r = x * rsqrt(sum(x^2) + N*eps) EXACTLY, and the float32 reduction makes the PREVENT_F16_OVERFLOW
        # activation pre-scale unnecessary. The g*sqrt(N) norm weight stays absorbed into the following linear.
        hidden_rms_norm = self.llm.layers[0].input_layernorm
        qk_rms_norm     = self.llm.layers[0].self_attn.q_norm
        self.hidden_rms_norm_eps = float(getattr(hidden_rms_norm, "variance_epsilon", getattr(hidden_rms_norm, "eps", 1e-6)))
        self.qk_rms_norm_eps     = float(getattr(qk_rms_norm, "variance_epsilon", getattr(qk_rms_norm, "eps", self.hidden_rms_norm_eps)))
        self.register_buffer("hidden_norm_scale", torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32))
        self.register_buffer("qk_norm_scale",     torch.full((head_dim,),    head_dim ** -0.5,    dtype=torch.float32))

        self._fuse_weights(hidden_size)

        # Quantization-friendly weight reorder (exact, zero runtime cost; absorbed into the fused weights).
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _fuse_weights(self, hidden_size: int) -> None:
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.llm.layers:
                attn = layer.self_attn
                has_bias = any(proj.bias is not None for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
                qkv = torch.nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                if has_bias:
                    def _bias_or_zero(proj: torch.nn.Linear) -> Tensor:
                        return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
                    qkv.bias.copy_(torch.cat([_bias_or_zero(attn.q_proj), _bias_or_zero(attn.k_proj), _bias_or_zero(attn.v_proj)], dim=0))

                input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(input_norm_weight)
                attn.qkv = qkv

                combined_scale = scale_factor * norm_factor_qk
                attn.q_norm.weight.mul_(combined_scale)
                attn.k_norm.weight.mul_(combined_scale)
                q_norm_rep = attn.q_norm.weight.repeat(self.num_heads)
                k_norm_rep = attn.k_norm.weight.repeat(self.num_kv_heads)
                attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_rep, k_norm_rep], dim=0).view(1, 1, 1, self.qk_heads, self.head_dim))

                gate = layer.mlp.gate_proj
                up   = layer.mlp.up_proj
                post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
                layer.mlp.gate_up_proj = gate_up

                del attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm
                del layer.input_layernorm, layer.post_attention_layernorm
                del layer.mlp.gate_proj, layer.mlp.up_proj

            # Absorb the (sum-based) final RMS norm into the classification head. The
            # classify head is independent of the token embeddings so this is safe.
            final_norm_weight = self.llm.norm.weight.unsqueeze(0) * norm_factor
            self.lm_head.weight = torch.nn.Parameter(self.lm_head.weight.data.clone() * final_norm_weight)
            del self.llm.norm

    # ── Quantization-friendly weight reorder (exact, zero runtime cost) ─────────
    def _reorder_downproj_for_quant(self, key: str) -> None:
        """Reorder each layer's MLP intermediate channels so contiguous block-quant groups over down_proj's
        INPUT axis hold magnitude-homogeneous channels (smaller per-group scale -> lower weight-quant error).
        The SAME permutation is applied to gate_up's OUTPUT rows (gate + up halves) and down_proj's INPUT
        columns, so act(gate)*up @ down_proj is unchanged -- fully absorbed, no runtime de-permutation.
        """
        with torch.no_grad():
            for layer in self.llm.layers:
                W = layer.mlp.down_proj.weight              # (hidden, intermediate)
                a = W.abs()
                if key == "rms":
                    stat = (W * W).mean(0).sqrt()
                elif key == "L4":
                    stat = a.pow(4).mean(0).pow(0.25)
                elif key == "std":
                    stat = W.std(0)
                else:                                       # "absmean" (default / fallback)
                    stat = a.mean(0)
                perm  = torch.argsort(stat)
                inter = layer.mlp.down_proj.in_features
                gu    = layer.mlp.gate_up_proj.weight       # (2*intermediate, hidden): [gate; up]
                layer.mlp.gate_up_proj.weight.copy_(torch.cat([gu[:inter][perm], gu[inter:][perm]], dim=0))
                layer.mlp.down_proj.weight.copy_(W[:, perm])

    def _reorder_oproj_for_quant(self, key: str) -> None:
        """Reorder each head's head_dim so contiguous SUB-head o_proj quant groups (group_size < head_dim)
        are magnitude-homogeneous. ONE head_dim permutation per kv head (shared by its GQA query heads) is
        applied to o_proj's INPUT columns and to the v-rows of the fused qkv; q/k/RoPE/qk_norm untouched.
        Fully absorbed into the weights. Pure win for the f16 KV cache (exact V-cache round-trip).
        """
        H, KVH, Dh, qk_heads = self.num_heads, self.num_kv_heads, self.head_dim, self.qk_heads
        G = H // KVH
        with torch.no_grad():
            for layer in self.llm.layers:
                Wo  = layer.self_attn.o_proj.weight                 # (hidden, H*head_dim)
                Woc = Wo.view(Wo.shape[0], H, Dh)                   # (hidden, H, head_dim)
                perms = []
                for kvh in range(KVH):                              # one order per kv head
                    cols = Woc[:, kvh * G:(kvh + 1) * G, :]         # its G query heads combined
                    a = cols.abs()
                    if key == "rms":
                        stat = (cols * cols).mean(dim=(0, 1)).sqrt()
                    elif key == "std":
                        stat = cols.reshape(-1, Dh).std(0)
                    elif key == "L4":
                        stat = a.pow(4).mean(dim=(0, 1)).pow(0.25)
                    else:                                           # "absmean" (default / fallback)
                        stat = a.mean(dim=(0, 1))
                    perms.append(torch.argsort(stat))
                # o_proj input columns: permute each head's head_dim by its kv-head order
                Woc2 = Woc.clone()
                for h in range(H):
                    Woc2[:, h, :] = Woc2[:, h, perms[h // G]]
                Wo.copy_(Woc2.reshape(Wo.shape[0], H * Dh))
                # compensate on the v-rows of the fused qkv (head_dim); q/k rows untouched
                Wq  = layer.self_attn.qkv.weight                    # (total_heads*head_dim, hidden)
                Wqr = Wq.view(-1, Dh, Wq.shape[1]).clone()          # (total_heads, head_dim, hidden)
                for kvh in range(KVH):
                    Wqr[qk_heads + kvh] = Wqr[qk_heads + kvh][perms[kvh]]
                Wq.copy_(Wqr.reshape(Wq.shape[0], Wq.shape[1]))

    def _rms_norm(self, x: Tensor, scale: Tensor, eps: float) -> Tensor:
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x: Tensor, batch_size: int) -> Tensor:
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, hidden_states: Tensor, rotary_cos: Tensor, rotary_sin: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size = hidden_states.shape[0]
        for layer in self.llm.layers:
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_kv_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_kv_heads], dim=-2)
            qk = self._rms_norm(qk, self.qk_norm_scale, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_cos + self._rotate_half(qk, batch_size) * rotary_sin
            q, k = torch.split(qk_rot, [self.num_heads, self.num_kv_heads], dim=-2)
            q = q.reshape(batch_size, -1, self.num_kv_heads, self.num_kv_groups, self.head_dim).permute(0, 2, 3, 1, 4)
            k = k.permute(0, 3, 2, 4, 1)   # (B, KVH, 1, D, L) — pre-transposed for Q @ K
            v = v.transpose(1, 3)          # (B, KVH, 1, L, D)
            attn = torch.matmul(q, k) + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)
        logits = self.lm_head(hidden_states)                       # (1, L, classify_num)
        output_ids = torch.argmax(logits, dim=-1).to(torch.int32)  # (1, L) timestamp bucket per position
        return output_ids


# ══════════════════════════════════════════════════════════════════════════════
# ── Export ────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def export_all() -> None:
    if not os.path.isfile(os.path.join(download_path, "model.safetensors")):
        raise FileNotFoundError(
            f"'{download_path}/model.safetensors' not found.\n"
            f"Download the weights first, e.g.:\n"
            f"  huggingface-cli download Qwen/Qwen3-ForcedAligner-0.6B --local-dir {download_path}"
        )
    os.makedirs(os.path.dirname(onnx_model_Main), exist_ok=True)
    print("\nExport start …\n")
    config = AutoConfig.from_pretrained(download_path, trust_remote_code=True)
    model  = AutoModel.from_pretrained(download_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

    audio_cfg    = config.thinker_config.audio_config
    text_cfg     = config.thinker_config.text_config
    # transformers>=5 leaves __init__-computed non-persistent buffers (RoPE inv_freq,
    # sinusoidal audio positions) as uninitialized memory after from_pretrained;
    # recompute them before any graph reads their values. No-op on transformers 4.x.
    refresh_non_persistent_buffers(model, text_cfg)
    num_layers   = text_cfg.num_hidden_layers
    num_heads    = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim     = text_cfg.head_dim
    hidden_size  = text_cfg.hidden_size
    classify_num = int(getattr(config.thinker_config, "classify_num", 5000))

    print(f"  Encoder    : layers={audio_cfg.encoder_layers}, d_model={audio_cfg.d_model}, output_dim={audio_cfg.output_dim}")
    print(f"  Decoder    : layers={num_layers}, heads={num_heads}/{num_kv_heads} GQA, head_dim={head_dim}")
    print(f"  Classifier : {classify_num} timestamp buckets x {TIMESTAMP_SEGMENT_TIME} ms (max {classify_num * TIMESTAMP_SEGMENT_TIME / 1000:.0f}s)")

    # ── Decoder Embed ─────────────────────────────────────────────────────────
    embed_mod = FORCED_ALIGNER_EMBED(model).eval()
    dummy_ids = torch.ones((1, 16), dtype=torch.int32)
    torch.onnx.export(
        embed_mod, (dummy_ids,), onnx_model_Embed,
        input_names=["input_ids"], output_names=["text_embed"],
        dynamic_axes={"input_ids": {0: "batch", 1: "ids_len"}, "text_embed": {0: "batch", 1: "ids_len"}} if DYNAMIC_AXES else None,
        opset_version=OPSET, dynamo=False,
    )
    del embed_mod, dummy_ids
    gc.collect()
    print("  [1/4] Embed exported")

    # ── Audio Encoder ─────────────────────────────────────────────────────────
    encoder     = FORCED_ALIGNER_ENCODER(model.thinker.audio_tower, model.thinker.model.embed_tokens.float()).eval()
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    dummy_audio = torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)
    dummy_text  = torch.ones((1, 8, hidden_size), dtype=torch.float32)
    torch.onnx.export(
        encoder, (dummy_audio, dummy_text), onnx_model_Encoder,
        input_names=["audio", "text_embed"], output_names=["concat_embed", "ids_len"],
        dynamic_axes={"audio": {2: "audio_len"}, "text_embed": {1: "num_token"}, "concat_embed": {1: "total_len"}} if DYNAMIC_AXES else None,
        opset_version=OPSET, dynamo=False,
    )
    del encoder, dummy_audio, dummy_text
    gc.collect()
    print("  [2/4] Encoder exported")

    # ── Rotary + Causal Mask ──────────────────────────────────────────────────
    rotary_mod = FORCED_ALIGNER_ROTARY_MASK(model.thinker.model, MAX_SEQ_LEN).eval()
    dummy_ids_len = torch.tensor([16], dtype=torch.int64)
    torch.onnx.export(
        rotary_mod, (dummy_ids_len,), onnx_model_Rotary_Mask,
        input_names=["ids_len"], output_names=["rotary_cos", "rotary_sin", "attention_mask"],
        dynamic_axes={"rotary_cos": {1: "ids_len"}, "rotary_sin": {1: "ids_len"}, "attention_mask": {3: "ids_len", 4: "kv_seq_len"}} if DYNAMIC_AXES else None,
        opset_version=OPSET, dynamo=False,
    )
    del rotary_mod
    gc.collect()
    print("  [3/4] Rotary+Mask exported")

    # ── Decoder Main ──────────────────────────────────────────────────────────
    decoder_main = FORCED_ALIGNER_DECODER_MAIN(model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size).eval()
    del model
    gc.collect()

    dummy_len      = 16
    hidden_states  = torch.ones((1, dummy_len, hidden_size),          dtype=torch.float32)
    rotary_cos     = torch.ones((1, dummy_len, 1, 1, head_dim),       dtype=torch.float32)
    rotary_sin     = torch.zeros((1, dummy_len, 1, 1, head_dim),      dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1, dummy_len, dummy_len),     dtype=torch.float32)
    torch.onnx.export(
        decoder_main, (hidden_states, rotary_cos, rotary_sin, attention_mask), onnx_model_Main,
        input_names=["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"],
        output_names=["output_ids"],
        dynamic_axes={
            "hidden_states":  {1: "ids_len"},
            "rotary_cos":     {1: "ids_len"},
            "rotary_sin":     {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
            "output_ids":     {1: "ids_len"},
        } if DYNAMIC_AXES else None,
        opset_version=OPSET, dynamo=False,
    )
    del decoder_main, hidden_states, rotary_cos, rotary_sin, attention_mask
    gc.collect()
    print("  [4/4] Decoder-Main exported")

    metadata_marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        onnx_model_Metadata,
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False,
    )
    del metadata_marker

    onnx_metadata = build_model_metadata(
        {
            "qwen_forcedaligner_metadata_version": 1,
            "producer": "Export_Qwen_ForcedAligner.py",
            "sample_rate": SAMPLE_RATE,
            "input_audio_length": MAX_INPUT_AUDIO_LENGTH,
            "input_audio_dtype": INPUT_AUDIO_DTYPE,
            "max_seq_len": MAX_SEQ_LEN,
            "activations_fp16": False,
            "opset": OPSET,
        },
        {
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "num_timestamp_buckets": classify_num,
            "timestamp_segment_ms": TIMESTAMP_SEGMENT_TIME,
            "audio_encoder_layers": audio_cfg.encoder_layers,
            "audio_encoder_d_model": audio_cfg.d_model,
            "audio_encoder_output_dim": audio_cfg.output_dim,
            "num_mels": N_MELS,
            "nfft_stft": NFFT_STFT,
            "window_length": WINDOW_LENGTH,
            "hop_length": HOP_LENGTH,
            "window_type": WINDOW_TYPE,
        },
        {
            "audio_start_token_id": _token_id(tokenizer, "<|audio_start|>", AUDIO_START_TOKEN_ID),
            "audio_end_token_id": _token_id(tokenizer, "<|audio_end|>", AUDIO_END_TOKEN_ID),
            "audio_pad_token_id": _token_id(tokenizer, "<|audio_pad|>", AUDIO_PAD_TOKEN_ID),
            "timestamp_token_id": _token_id(tokenizer, "<timestamp>", TIMESTAMP_TOKEN_ID),
        },
    )
    metadata_targets = [onnx_model_Metadata, onnx_model_Embed, onnx_model_Encoder, onnx_model_Rotary_Mask, onnx_model_Main]
    written = []
    for target in metadata_targets:
        if not Path(target).exists():
            continue
        write_onnx_metadata(target, onnx_metadata)
        written.append(target)
    print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(written)} ONNX graph(s):")
    for key in sorted(onnx_metadata):
        print(f"    {key} = {onnx_metadata[key]}")

    # ── Save the tokenizer into the ONNX folder so the exported folder runs inference ──
    # stand-alone (no external Qwen3-ForcedAligner model path needed at inference time).
    try:
        _tokenizer_dir = onnx_folder / "tokenizer"
        tokenizer.save_pretrained(str(_tokenizer_dir))
        print(f"[Tokenizer] Saved tokenizer -> {_tokenizer_dir}")
    except Exception as _exc:  # noqa: BLE001 - a failed save must not abort the auto demo
        print(f"[Tokenizer] Skipped tokenizer bundle ({_exc})")
    print("\nExport complete.\n")


def run_exported_inference() -> None:
    print("─" * 70)
    print("Running ONNX Runtime demo via Inference_Qwen_ForcedAligner_ONNX.py ...\n")
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "Inference_Qwen_ForcedAligner_ONNX.py"), "--onnx-folder", str(onnx_folder)],
        check=True,
    )


if __name__ == "__main__":
    export_all()
    run_exported_inference()
