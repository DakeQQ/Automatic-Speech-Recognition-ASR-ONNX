import gc
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import torchaudio
from STFT_Process import STFT_Process                                             # The custom STFT/ISTFT can be exported in ONNX format.
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig


model_path = str(Path.home() / "Downloads" / "whisper-large-v3-turbo")  # Source Whisper model (HF) download path.

SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_DIR = SCRIPT_DIR / "Whisper_ONNX"
_raw_onnx_temp = tempfile.TemporaryDirectory(prefix="whisper_export_")
_raw_onnx_dir = Path(_raw_onnx_temp.name)

import Shared_Merged


MODEL_FILE_NAMES = dict(Shared_Merged.DEFAULT_MODEL_FILE_NAMES)

# Split graphs are temporary merge constituents. Encoder / NoSpeech / Metadata are copied to ONNX_DIR.
onnx_model_Metadata       = str(_raw_onnx_dir / MODEL_FILE_NAMES["metadata"])
onnx_model_Encoder        = str(_raw_onnx_dir / MODEL_FILE_NAMES["encoder"])
onnx_model_Decoder        = str(_raw_onnx_dir / MODEL_FILE_NAMES["main"])
onnx_model_Embed          = str(_raw_onnx_dir / MODEL_FILE_NAMES["embed"])
onnx_model_Prefill        = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_prefill"])
onnx_model_Decode         = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_decode"])

# -- Exported ONNX graph paths: token selection, repetition penalty, and no-speech detection --
onnx_model_Begin_Suppress = str(_raw_onnx_dir / MODEL_FILE_NAMES["begin_suppress"])
onnx_model_Greedy         = str(_raw_onnx_dir / MODEL_FILE_NAMES["greedy"])
onnx_model_Argmax         = str(_raw_onnx_dir / MODEL_FILE_NAMES["argmax"])
onnx_model_Sampling       = str(_raw_onnx_dir / MODEL_FILE_NAMES["sampling"])
onnx_model_Penalty        = str(_raw_onnx_dir / MODEL_FILE_NAMES["penalty"])
onnx_model_No_Speech      = str(_raw_onnx_dir / MODEL_FILE_NAMES["no_speech"])

# -- Export configuration --
INPUT_AUDIO_DTYPE          = "F32"      # ONNX audio input dtype: "INT16", "F32", or "F16".
USE_FP16_KV                = True       # Keep cache, cross-KV, position, and mask storage in FP16 for normal deployment exports.
COMPUTE_IN_F32             = False      # FP16-cache-only option: upcast attention compute while retaining FP16 storage.
KV_DTYPE                   = torch.float16 if USE_FP16_KV else torch.float32
REORDER_DOWNPROJ_FOR_QUANT = True       # Apply the exact fc1/fc2 channel reorder.
REORDER_OPROJ_FOR_QUANT    = True       # Apply the exact self-attention V/o_proj channel reorder.
REORDER_KEY                = "absmean"
OPSET                      = 20


tokenizer = AutoTokenizer.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
generation_config = GenerationConfig.from_pretrained(model_path)
INPUT_AUDIO_LENGTH = int(getattr(feature_extractor, "n_samples", 480000))
NFFT_STFT = int(getattr(feature_extractor, "n_fft", 400))
HOP_LENGTH = int(getattr(feature_extractor, "hop_length", 160))
SAMPLE_RATE = int(getattr(feature_extractor, "sampling_rate", 16000))
WINDOW_LENGTH = NFFT_STFT
WINDOW_TYPE = "hann"
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
no_timestamps_id = int(getattr(generation_config, "no_timestamps_token_id", tokenizer.convert_tokens_to_ids("<|notimestamps|>")))    # 50364 (v3) / 50363 (v2): selects non-timestamp transcription.
no_speech_id = int(tokenizer.convert_tokens_to_ids("<|nospeech|>") or (no_timestamps_id - 1))                                      # 50363 (v3) / 50362 (v2): Whisper's silence-detection token.
begin_suppress_token_ids = tuple(int(token_id) for token_id in (getattr(generation_config, "begin_suppress_tokens", None) or ()))
_LANGUAGE_DATA = {
    'af': {'id': 50327, 'custom_id': 18941, 'full_name': 'afrikaans'},
    'am': {'id': 50334, 'custom_id': 18948, 'full_name': 'amharic'},
    'ar': {'id': 50272, 'custom_id': 18886, 'full_name': 'arabic'},
    'as': {'id': 50350, 'custom_id': 18964, 'full_name': 'assamese'},
    'az': {'id': 50304, 'custom_id': 18918, 'full_name': 'azerbaijani'},
    'ba': {'id': 50355, 'custom_id': 18969, 'full_name': 'bashkir'},
    'be': {'id': 50330, 'custom_id': 18944, 'full_name': 'belarusian'},
    'bg': {'id': 50292, 'custom_id': 18906, 'full_name': 'bulgarian'},
    'bn': {'id': 50302, 'custom_id': 18916, 'full_name': 'bengali'},
    'bo': {'id': 50347, 'custom_id': 18961, 'full_name': 'tibetan'},
    'br': {'id': 50309, 'custom_id': 18923, 'full_name': 'breton'},
    'bs': {'id': 50315, 'custom_id': 18929, 'full_name': 'bosnian'},
    'ca': {'id': 50270, 'custom_id': 18884, 'full_name': 'catalan'},
    'cs': {'id': 50283, 'custom_id': 18897, 'full_name': 'czech'},
    'cy': {'id': 50297, 'custom_id': 18911, 'full_name': 'welsh'},
    'da': {'id': 50285, 'custom_id': 18899, 'full_name': 'danish'},
    'de': {'id': 50261, 'custom_id': 18875, 'full_name': 'german'},
    'el': {'id': 50281, 'custom_id': 18895, 'full_name': 'greek'},
    'en': {'id': 50259, 'custom_id': 18873, 'full_name': 'english'},
    'es': {'id': 50262, 'custom_id': 18876, 'full_name': 'spanish'},
    'et': {'id': 50307, 'custom_id': 18921, 'full_name': 'estonian'},
    'eu': {'id': 50310, 'custom_id': 18924, 'full_name': 'basque'},
    'fa': {'id': 50300, 'custom_id': 18914, 'full_name': 'persian'},
    'fi': {'id': 50277, 'custom_id': 18891, 'full_name': 'finnish'},
    'fo': {'id': 50338, 'custom_id': 18952, 'full_name': 'faroese'},
    'fr': {'id': 50265, 'custom_id': 18879, 'full_name': 'french'},
    'gl': {'id': 50319, 'custom_id': 18933, 'full_name': 'galician'},
    'gu': {'id': 50333, 'custom_id': 18947, 'full_name': 'gujarati'},
    'ha': {'id': 50354, 'custom_id': 18968, 'full_name': 'hausa'},
    'haw': {'id': 50352, 'custom_id': 18966, 'full_name': 'hawaiian'},
    'he': {'id': 50279, 'custom_id': 18893, 'full_name': 'hebrew'},
    'hi': {'id': 50276, 'custom_id': 18890, 'full_name': 'hindi'},
    'hr': {'id': 50291, 'custom_id': 18905, 'full_name': 'croatian'},
    'ht': {'id': 50339, 'custom_id': 18953, 'full_name': 'haitian creole'},
    'hu': {'id': 50286, 'custom_id': 18900, 'full_name': 'hungarian'},
    'hy': {'id': 50312, 'custom_id': 18926, 'full_name': 'armenian'},
    'id': {'id': 50275, 'custom_id': 18889, 'full_name': 'indonesian'},
    'is': {'id': 50311, 'custom_id': 18925, 'full_name': 'icelandic'},
    'it': {'id': 50274, 'custom_id': 18888, 'full_name': 'italian'},
    'ja': {'id': 50266, 'custom_id': 18880, 'full_name': 'japanese'},
    'jw': {'id': 50356, 'custom_id': 18970, 'full_name': 'javanese'},
    'ka': {'id': 50329, 'custom_id': 18943, 'full_name': 'georgian'},
    'kk': {'id': 50316, 'custom_id': 18930, 'full_name': 'kazakh'},
    'km': {'id': 50323, 'custom_id': 18937, 'full_name': 'khmer'},
    'kn': {'id': 50306, 'custom_id': 18920, 'full_name': 'kannada'},
    'ko': {'id': 50264, 'custom_id': 18878, 'full_name': 'korean'},
    'la': {'id': 50294, 'custom_id': 18908, 'full_name': 'latin'},
    'lb': {'id': 50345, 'custom_id': 18959, 'full_name': 'luxembourgish'},
    'ln': {'id': 50353, 'custom_id': 18967, 'full_name': 'lingala'},
    'lo': {'id': 50336, 'custom_id': 18950, 'full_name': 'lao'},
    'lt': {'id': 50293, 'custom_id': 18907, 'full_name': 'lithuanian'},
    'lv': {'id': 50301, 'custom_id': 18915, 'full_name': 'latvian'},
    'mg': {'id': 50349, 'custom_id': 18963, 'full_name': 'malagasy'},
    'mi': {'id': 50295, 'custom_id': 18909, 'full_name': 'maori'},
    'mk': {'id': 50308, 'custom_id': 18922, 'full_name': 'macedonian'},
    'ml': {'id': 50296, 'custom_id': 18910, 'full_name': 'malayalam'},
    'mn': {'id': 50314, 'custom_id': 18928, 'full_name': 'mongolian'},
    'mr': {'id': 50320, 'custom_id': 18934, 'full_name': 'marathi'},
    'ms': {'id': 50282, 'custom_id': 18896, 'full_name': 'malay'},
    'mt': {'id': 50343, 'custom_id': 18957, 'full_name': 'maltese'},
    'my': {'id': 50346, 'custom_id': 18960, 'full_name': 'burmese'},
    'ne': {'id': 50313, 'custom_id': 18927, 'full_name': 'nepali'},
    'nl': {'id': 50271, 'custom_id': 18885, 'full_name': 'dutch'},
    'nn': {'id': 50342, 'custom_id': 18956, 'full_name': 'nynorsk'},
    'no': {'id': 50288, 'custom_id': 18902, 'full_name': 'norwegian'},
    'oc': {'id': 50328, 'custom_id': 18942, 'full_name': 'occitan'},
    'pa': {'id': 50321, 'custom_id': 18935, 'full_name': 'punjabi'},
    'pl': {'id': 50269, 'custom_id': 18883, 'full_name': 'polish'},
    'ps': {'id': 50340, 'custom_id': 18954, 'full_name': 'pashto'},
    'pt': {'id': 50267, 'custom_id': 18881, 'full_name': 'portuguese'},
    'ro': {'id': 50284, 'custom_id': 18898, 'full_name': 'romanian'},
    'ru': {'id': 50263, 'custom_id': 18877, 'full_name': 'russian'},
    'sa': {'id': 50344, 'custom_id': 18958, 'full_name': 'sanskrit'},
    'sd': {'id': 50332, 'custom_id': 18946, 'full_name': 'sindhi'},
    'si': {'id': 50322, 'custom_id': 18936, 'full_name': 'sinhala'},
    'sk': {'id': 50298, 'custom_id': 18912, 'full_name': 'slovak'},
    'sl': {'id': 50305, 'custom_id': 18919, 'full_name': 'slovenian'},
    'sn': {'id': 50324, 'custom_id': 18938, 'full_name': 'shona'},
    'so': {'id': 50326, 'custom_id': 18940, 'full_name': 'somali'},
    'sq': {'id': 50317, 'custom_id': 18931, 'full_name': 'albanian'},
    'sr': {'id': 50303, 'custom_id': 18917, 'full_name': 'serbian'},
    'su': {'id': 50357, 'custom_id': 18971, 'full_name': 'sundanese'},
    'sv': {'id': 50273, 'custom_id': 18887, 'full_name': 'swedish'},
    'sw': {'id': 50318, 'custom_id': 18932, 'full_name': 'swahili'},
    'ta': {'id': 50287, 'custom_id': 18901, 'full_name': 'tamil'},
    'te': {'id': 50299, 'custom_id': 18913, 'full_name': 'telugu'},
    'tg': {'id': 50331, 'custom_id': 18945, 'full_name': 'tajik'},
    'th': {'id': 50289, 'custom_id': 18903, 'full_name': 'thai'},
    'tk': {'id': 50341, 'custom_id': 18955, 'full_name': 'turkmen'},
    'tl': {'id': 50348, 'custom_id': 18962, 'full_name': 'tagalog'},
    'tr': {'id': 50268, 'custom_id': 18882, 'full_name': 'turkish'},
    'tt': {'id': 50351, 'custom_id': 18965, 'full_name': 'tatar'},
    'uk': {'id': 50280, 'custom_id': 18894, 'full_name': 'ukrainian'},
    'ur': {'id': 50290, 'custom_id': 18904, 'full_name': 'urdu'},
    'uz': {'id': 50337, 'custom_id': 18951, 'full_name': 'uzbek'},
    'vi': {'id': 50278, 'custom_id': 18892, 'full_name': 'vietnamese'},
    'yi': {'id': 50335, 'custom_id': 18949, 'full_name': 'yiddish'},
    'yo': {'id': 50325, 'custom_id': 18939, 'full_name': 'yoruba'},
    'yue': {'id': 50358, 'custom_id': 18972, 'full_name': 'cantonese'},
    'zh': {'id': 50260, 'custom_id': 18874, 'full_name': 'chinese'},
}


_ALIAS_TO_CODE = {
    'united states': 'en', 'us': 'en',
    'united kingdom': 'en', 'uk': 'en', 'gb': 'en',
    'france': 'fr',
    'germany': 'de',
    'spain': 'es',
    'china': 'zh',
    'japan': 'ja',
    'korea': 'ko',
}


def build_model_metadata(*sections):
    def _norm(value):
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return str(value)

    merged = {}
    for section in sections:
        for key, value in section.items():
            if value is not None:
                merged[key] = _norm(value)
    return merged


def write_metadata_carrier(onnx_path, metadata):
    Shared_Merged.replace_onnx_metadata(
        onnx_path,
        {str(key): str(value) for key, value in metadata.items()},
    )


def _bias_or_zero(linear):
    return linear.bias if linear.bias is not None else torch.zeros(linear.out_features, dtype=linear.weight.dtype)


def absorb_layer_norm_affine(norm, linear):
    # Fold a LayerNorm's affine (gamma, beta) into the following Linear so the norm becomes affine-less:
    #   Linear(gamma * xhat + beta) = (W * gamma) @ xhat + (W @ beta + b)
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype))
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))   # b += W @ beta  (uses pre-scaled W)
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))                     # W *= gamma  (per input channel)
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias = None


class BEGIN_SUPPRESS(torch.nn.Module):
    """Apply Whisper's begin-only suppression before the first decode head."""

    def __init__(self, token_ids, vocab_size):
        super(BEGIN_SUPPRESS, self).__init__()
        bias = torch.zeros((1, vocab_size), dtype=torch.float32)
        valid_ids = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < vocab_size]
        if valid_ids:
            bias[:, valid_ids] = float("-inf")
        self.register_buffer("bias", bias)

    def forward(self, logits):
        return logits + self.bias


class GREEDY_SEARCH(torch.nn.Module):
    # Pure argmax that also appends the chosen token to the running save_id history (Qwen ASR style).
    # Used when a repetition penalty is active so APPLY_PENALTY can read the on-device history.
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        return max_logits_idx, torch.cat([save_id, max_logits_idx], dim=-1)


class ARGMAX(torch.nn.Module):
    # Bare argmax (Qwen ASR style); used for greedy decoding when no repetition penalty is applied.
    def __init__(self):
        super(ARGMAX, self).__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_min", torch.tensor(self.GUMBEL_EPS, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_max",
            torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        inv_penalty = torch.reciprocal(repetition_penalty)
        prev_logits = torch.gather(logits, 1, previous_ids)
        prev_scores = torch.where(
            prev_logits < 0.0,
            prev_logits * repetition_penalty,
            prev_logits * inv_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids, prev_scores)
        scores = scores * torch.reciprocal(temperature)

        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep_topp = (sorted_cumsum - sorted_probs) <= top_p
        sorted_scores = torch.where(keep_topp, sorted_scores, self.neg_inf)

        noise = torch.clamp(
            torch.rand_like(sorted_scores), self.gumbel_min, self.gumbel_max
        )
        gumbel = -torch.log(-torch.log(noise))
        winner = torch.argmax(sorted_scores + gumbel, dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        save_id = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_id


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super(METADATA_CARRIER, self).__init__()

    def forward(self, marker):
        return marker


class APPLY_PENALTY(torch.nn.Module):
    # Sliding-window repetition penalty (Qwen ASR style): multiply the logits of the most recent
    # `penalty_range` tokens by `penalty_value`. The merged runtime binds penalty_value=1.0
    # before the history window fills, then switches to the configured multiplier; this preserves
    # the split runtime's conditional invocation without another graph launch.
    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        # Keep penalty_range tensor-valued so it remains a runtime ONNX input. The legacy
        # exporter lowers the negative bound directly to Neg + Slice without `.item()`.
        target_indices = save_id[:, -penalty_range:].long()
        penalised = logits.gather(1, target_indices) * penalty_value
        return logits.scatter(1, target_indices, penalised)


class NO_SPEECH_DETECTION(torch.nn.Module):
    def __init__(self, no_speech_token, suppress_tokens, vocab_size):
        super(NO_SPEECH_DETECTION, self).__init__()
        self.no_speech_token = no_speech_token
        # The decoder emits logits with the permanent suppress-token bias already applied (-128 on every
        # suppressed id, which includes <|nospeech|>). Re-add that bias here so P(<|nospeech|>) is read from
        # the pre-suppression distribution, matching Whisper's silence cue. unsuppress_bias is 0 everywhere
        # except +128 on suppressed ids, so (logits + unsuppress_bias) reconstructs the raw proj_out logits.
        unsuppress_bias = torch.zeros((1, vocab_size), dtype=torch.float32)
        if suppress_tokens is not None:
            unsuppress_bias[:, suppress_tokens] = float(128.0)
        self.register_buffer('unsuppress_bias', unsuppress_bias)

    def forward(self, logits):
        return torch.softmax(logits + self.unsuppress_bias, dim=-1)[:, self.no_speech_token]


class WHISPER_ENCODER(torch.nn.Module):
    def __init__(self, whisper, stft_model, nfft_stft, n_mels, sample_rate, num_layers_de):
        super(WHISPER_ENCODER, self).__init__()
        self.encoder = whisper.encoder
        self.decoder = whisper.decoder
        self.stft_model = stft_model
        self.register_buffer(
            'fbank',
            torchaudio.functional.melscale_fbanks(
                nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, "slaney", 'slaney'
            ).transpose(0, 1).unsqueeze(0),
        )
        self.num_layers_de = num_layers_de
        # int16 audio is raw PCM (normalised in forward via ÷32768); f32/f16 audio is
        # assumed pre-normalised to [-1, 1]. The immutable scale is folded into the
        # Conv1d DFT kernel so no full-waveform Mul remains in the captured graph.
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")
        self.num_heads = self.encoder.layers[0].self_attn.num_heads
        self.head_dim = self.encoder.layers[0].self_attn.head_dim
        self.hidden_size = self.encoder.layers[0].self_attn.out_proj.in_features
        self.cross_num_heads = self.decoder.layers[0].encoder_attn.num_heads
        self.cross_head_dim = self.decoder.layers[0].encoder_attn.head_dim
        self.cross_hidden_size = self.cross_num_heads * self.cross_head_dim
        self._fuse_weights()

    def _fuse_weights(self):
        # Fuse self-attention q/k/v into a single Linear, fold the d**-0.25 attention scale into q & k,
        # absorb self_attn_layer_norm into qkv and final_layer_norm into fc1.
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            for encoder_layer in self.encoder.layers:
                attn = encoder_layer.self_attn
                out_features = attn.q_proj.out_features
                qkv = torch.nn.Linear(attn.q_proj.in_features, out_features * 3, bias=True)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                qkv.bias.copy_(torch.cat([_bias_or_zero(attn.q_proj), _bias_or_zero(attn.k_proj), _bias_or_zero(attn.v_proj)], dim=0))
                qkv.weight.data[:out_features * 2].mul_(scale)   # fold attention scale into q and k
                qkv.bias.data[:out_features].mul_(scale)         # q bias only (k has no bias)
                absorb_layer_norm_affine(encoder_layer.self_attn_layer_norm, qkv)
                absorb_layer_norm_affine(encoder_layer.final_layer_norm, encoder_layer.fc1)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj
            # All decoder layers project the same final encoder state. Concatenate every layer's
            # K projection followed by every V projection into one Linear. This removes L-1 large
            # MatMul/Add/Cast pipelines while retaining the exact per-layer output cache contract.
            cross_scale = float(self.cross_head_dim ** -0.25)
            key_weights = []
            key_biases = []
            value_weights = []
            value_biases = []
            for decoder_layer in self.decoder.layers:
                cross_attn = decoder_layer.encoder_attn
                key_weights.append(cross_attn.k_proj.weight.detach() * cross_scale)
                key_biases.append(_bias_or_zero(cross_attn.k_proj).detach() * cross_scale)
                value_weights.append(cross_attn.v_proj.weight.detach())
                value_biases.append(_bias_or_zero(cross_attn.v_proj).detach())

            first_cross_attn = self.decoder.layers[0].encoder_attn
            self.cross_kv = torch.nn.Linear(
                first_cross_attn.k_proj.in_features,
                self.num_layers_de * self.cross_hidden_size * 2,
                bias=True,
                device=first_cross_attn.k_proj.weight.device,
                dtype=first_cross_attn.k_proj.weight.dtype,
            )
            self.cross_kv.weight.copy_(torch.cat(key_weights + value_weights, dim=0))
            self.cross_kv.bias.copy_(torch.cat(key_biases + value_biases, dim=0))
            for decoder_layer in self.decoder.layers:
                cross_attn = decoder_layer.encoder_attn
                del cross_attn.k_proj, cross_attn.v_proj

    def forward(self, audio):
        audio = audio.float()
        power = self.stft_model(audio)                                            # Packed STFT computes power directly and omits Whisper's discarded final frame.
        mel_features = torch.matmul(self.fbank, power).clamp(min=1e-10).log10()
        mel_features = torch.maximum(mel_features, mel_features.max() - 8.0)
        mel_features = (mel_features + 4.0) * 0.25
        hidden_states = torch.nn.functional.gelu(self.encoder.conv2(torch.nn.functional.gelu(self.encoder.conv1(mel_features)))).transpose(1, 2)
        hidden_states = hidden_states + self.encoder.embed_positions.weight[:hidden_states.shape[1]]
        for encoder_layer in self.encoder.layers:
            hidden_states_norm = encoder_layer.self_attn_layer_norm(hidden_states)
            qkv = encoder_layer.self_attn.qkv(hidden_states_norm).view(-1, 3 * self.num_heads, self.head_dim).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)                            # each (num_heads, T, head_dim)
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1), v).transpose(0, 1).reshape(1, -1, self.hidden_size)
            hidden_states_attn = encoder_layer.self_attn.out_proj(attn)
            hidden_states_attn += hidden_states
            hidden_states = hidden_states_attn + encoder_layer.fc2(encoder_layer.activation_fn(encoder_layer.fc1(encoder_layer.final_layer_norm(hidden_states_attn))))
        hidden_states = self.encoder.layer_norm(hidden_states)
        cross_kv = self.cross_kv(hidden_states).to(KV_DTYPE)
        keys, values = cross_kv.split(self.num_layers_de * self.cross_hidden_size, dim=-1)
        # Audio batch is statically one in the encoder contract. Flattening that singleton with
        # signal length avoids dynamic batch-shape construction before the two aggregate layouts.
        keys = keys.reshape(-1, self.num_layers_de, self.cross_num_heads, self.cross_head_dim)
        values = values.reshape(-1, self.num_layers_de, self.cross_num_heads, self.cross_head_dim)
        keys = keys.permute(1, 2, 3, 0).unbind(dim=0)                      # L * (heads, head_dim, signal_len)
        values = values.permute(1, 2, 0, 3).unbind(dim=0)                  # L * (heads, signal_len, head_dim)
        return *keys, *values


class WHISPER_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Qwen's Decoder_Embed) so the int
    # token ids never enter the float-only decode graph.
    def __init__(self, decoder):
        super(WHISPER_DECODER_EMBED, self).__init__()
        self.embed_tokens = decoder.embed_tokens

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class WHISPER_PREFILL(torch.nn.Module):
    # Prefill-phase position-embedding + causal-mask generator (mirrors Qwen's Rotary_Mask_Prefill).
    # Consumes the int lengths and emits float position embedding + float attention mask so the decoder
    # main graph stays integer-free.
    def __init__(self, decoder, max_seq_len, attention_dtype):
        super(WHISPER_PREFILL, self).__init__()
        self.register_buffer('position_weight', decoder.embed_positions.weight.unsqueeze(0).to(KV_DTYPE))
        self.register_buffer(
            'attention_mask',
            torch.triu(
                torch.full((1, max_seq_len, max_seq_len), -128.0, dtype=attention_dtype),
                diagonal=1,
            ),
        )

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len]
        return position_embed, attention_mask, kv_seq_len


class WHISPER_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Qwen's Rotary_Mask_Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, decoder):
        super(WHISPER_DECODE, self).__init__()
        self.register_buffer('position_weight', decoder.embed_positions.weight.unsqueeze(0).to(KV_DTYPE))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class WHISPER_DECODER(torch.nn.Module):
    def __init__(self, whisper, suppress_tokens, num_layers_de):
        super(WHISPER_DECODER, self).__init__()
        self.whisper = whisper
        self.decoder = whisper.model.decoder
        self.suppress_tokens = suppress_tokens
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = not USE_FP16_KV or COMPUTE_IN_F32
        self.idx_en_key = self.num_layers_de + self.num_layers_de            # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + self.num_layers_de             # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + self.num_layers_de             # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                              # position-embedding input (4 * L + 1)
        self.num_heads = self.decoder.layers[0].self_attn.num_heads
        self.head_dim = self.decoder.layers[0].self_attn.head_dim
        self.hidden_size = self.decoder.layers[0].self_attn.out_proj.in_features
        self.cross_num_heads = self.decoder.layers[0].encoder_attn.num_heads
        self.cross_head_dim = self.decoder.layers[0].encoder_attn.head_dim
        suppress_tokens_penalty = torch.zeros((1, self.whisper.proj_out.out_features), dtype=torch.float32)
        if self.suppress_tokens is not None:
            suppress_tokens_penalty[:, self.suppress_tokens] = float(-128.0)
        self.register_buffer('suppress_tokens_penalty', suppress_tokens_penalty)
        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _fuse_weights(self):
        # Fuse self-attention q/k/v into one Linear (fold d**-0.25 into q & k, absorb self_attn_layer_norm),
        # fold the cross-attention scale into encoder_attn.q_proj and absorb encoder_attn_layer_norm into it,
        # and absorb final_layer_norm into fc1.
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.decoder.layers:
                attn = decoder_layer.self_attn
                out_features = attn.q_proj.out_features
                qkv = torch.nn.Linear(attn.q_proj.in_features, out_features * 3, bias=True)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                qkv.bias.copy_(torch.cat([_bias_or_zero(attn.q_proj), _bias_or_zero(attn.k_proj), _bias_or_zero(attn.v_proj)], dim=0))
                qkv.weight.data[:out_features * 2].mul_(scale)
                qkv.bias.data[:out_features].mul_(scale)
                absorb_layer_norm_affine(decoder_layer.self_attn_layer_norm, qkv)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj
                cross_attn = decoder_layer.encoder_attn
                cross_attn.q_proj.weight.data.mul_(cross_scale)
                if cross_attn.q_proj.bias is not None:
                    cross_attn.q_proj.bias.data.mul_(cross_scale)
                absorb_layer_norm_affine(decoder_layer.encoder_attn_layer_norm, cross_attn.q_proj)
                absorb_layer_norm_affine(decoder_layer.final_layer_norm, decoder_layer.fc1)

    @staticmethod
    def _channel_stat(weight, key, dims):
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(dim=dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            if isinstance(dims, tuple):
                keep_dim = weight.shape[-1]
                return weight.reshape(-1, keep_dim).std(0)
            return weight.std(dim=dims)
        if key != "absmean":
            raise ValueError(f"Unsupported REORDER_KEY: {key!r}")
        return absolute.mean(dim=dims)

    def _reorder_downproj_for_quant(self, key):
        """Permute fc1 outputs and fc2 inputs by the same intermediate-channel order."""
        with torch.no_grad():
            for decoder_layer in self.decoder.layers:
                fc1 = decoder_layer.fc1
                fc2 = decoder_layer.fc2
                permutation = torch.argsort(self._channel_stat(fc2.weight, key, 0))
                fc1.weight.copy_(fc1.weight[permutation])
                if fc1.bias is not None:
                    fc1.bias.copy_(fc1.bias[permutation])
                fc2.weight.copy_(fc2.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute each self-attention V head and the matching o_proj input columns."""
        num_heads = self.num_heads
        head_dim = self.head_dim
        hidden_size = self.hidden_size
        with torch.no_grad():
            for decoder_layer in self.decoder.layers:
                attention = decoder_layer.self_attn
                output_weight = attention.out_proj.weight
                output_by_head = output_weight.view(output_weight.shape[0], num_heads, head_dim)
                permutations = [
                    torch.argsort(self._channel_stat(output_by_head[:, head, :], key, 0))
                    for head in range(num_heads)
                ]

                reordered_output = output_by_head.clone()
                for head, permutation in enumerate(permutations):
                    reordered_output[:, head, :] = output_by_head[:, head, permutation]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv = attention.qkv
                value_weight = qkv.weight[2 * hidden_size:].view(num_heads, head_dim, qkv.in_features)
                reordered_value_weight = value_weight.clone()
                for head, permutation in enumerate(permutations):
                    reordered_value_weight[head] = value_weight[head, permutation]
                qkv.weight[2 * hidden_size:].copy_(reordered_value_weight.reshape(hidden_size, qkv.in_features))

                if qkv.bias is not None:
                    value_bias = qkv.bias[2 * hidden_size:].view(num_heads, head_dim)
                    reordered_value_bias = value_bias.clone()
                    for head, permutation in enumerate(permutations):
                        reordered_value_bias[head] = value_bias[head, permutation]
                    qkv.bias[2 * hidden_size:].copy_(reordered_value_bias.reshape(hidden_size))

    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        # f16-storage / f32-compute (COMPUTE_IN_F32): the causal mask is kept f16 at the graph boundary (I/O
        # dtype unchanged) and upcast to f32 ONCE here, shared by every layer. Minimum-cast path uses it as-is (f16).
        attn_mask = attention_mask.float() if self.compute_in_f32 else attention_mask
        save_de_keys = []
        save_de_values = []
        for idx, decoder_layer in enumerate(self.decoder.layers):
            hidden_states_norm = decoder_layer.self_attn_layer_norm(hidden_states)
            # Self-attention. OFF (minimum-cast): cast the fused QKV DOWN to f16 before the split so
            # Q@K/mask/softmax/attn@V run in f16 on the f16 K/V cache; the context is cast back to f32 for out_proj.
            # ON (COMPUTE_IN_F32): keep the f16 K/V *storage* (K/V still cast to f16 before the cache concat, so
            # the cache I/O dtype is unchanged) but upcast K/V to f32 at the matmul use points and keep
            # Q/mask/softmax in f32 -- f16 storage, f32 compute. Q is never downcast.
            qkv = decoder_layer.self_attn.qkv(hidden_states_norm)
            if not self.compute_in_f32:
                qkv = qkv.half()
            qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = qkv.split(self.num_heads, dim=1)                            # each (batch, num_heads, ids_len, head_dim)
            if self.compute_in_f32:
                k = k.to(KV_DTYPE)
                v = v.to(KV_DTYPE)
            k = torch.cat((all_inputs[idx], k.transpose(-1, -2)), dim=-1)  # f16 key cache (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v), dim=-2)  # f16 value cache
            save_de_keys.append(k)
            save_de_values.append(v)
            if self.compute_in_f32:
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k.float()) + attn_mask, dim=-1), v.float()).transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
            else:
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k) + attn_mask, dim=-1), v).transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float()
            hidden_states_attn = decoder_layer.self_attn.out_proj(attn)
            hidden_states_attn += hidden_states
            hidden_states_attn_norm = decoder_layer.encoder_attn_layer_norm(hidden_states_attn)
            # Cross-attention against the f16 encoder cross-KV cache. OFF: downcast Q to f16 and run in f16 on the
            # f16 cross cache, context back to f32. ON: keep Q in f32 and upcast the f16 cross K/V to f32 at the
            # matmul use points (the cross cache is produced f16 by the encoder; its I/O dtype is unchanged).
            q = decoder_layer.encoder_attn.q_proj(hidden_states_attn_norm).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            if self.compute_in_f32:
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = decoder_layer.encoder_attn.out_proj(attn.transpose(1, 2).reshape(batch_size, -1, self.hidden_size))
            else:
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = decoder_layer.encoder_attn.out_proj(attn.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_states_attn
            hidden_states = hidden_state_cross + decoder_layer.fc2(decoder_layer.activation_fn(decoder_layer.fc1(decoder_layer.final_layer_norm(hidden_state_cross))))
        hidden_states = self.decoder.layer_norm(hidden_states[:, -1])
        logits = self.whisper.proj_out(hidden_states)
        if self.suppress_tokens is not None:
            logits = logits + self.suppress_tokens_penalty
        return *save_de_keys, *save_de_values, logits


print('\nExport start...\n')
_raw_onnx_dir.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    HIDDEN_SIZE = model.config.d_model
    NUM_HEAD_DE = model.model.config.decoder_attention_heads
    HEAD_DIM_DE = model.model.decoder.layers._modules['0'].self_attn.head_dim
    NUM_LAYER_DE = model.config.decoder_layers
    N_MELS = model.config.num_mel_bins
    VOCAB_SIZE = model.config.vocab_size
    supported_languages = {}
    aliases_by_code = {}
    for alias, code in _ALIAS_TO_CODE.items():
        aliases_by_code.setdefault(code, []).append(alias)
    canonical_language_codes = {
        token[2:-2] for token in generation_config.lang_to_id
    }
    for token, token_id in generation_config.lang_to_id.items():
        code = token[2:-2]
        language_data = _LANGUAGE_DATA.get(code)
        supported_languages[code] = {
            "name": language_data["full_name"].title(),
            "aliases": sorted(
                set([language_data["full_name"], *aliases_by_code.get(code, [])])
                - {code}
                - canonical_language_codes
            ),
            "token_id": int(token_id),
            "prompt_token_ids": [],
        }
    task_ids = {
        str(task): int(token_id)
        for task, token_id in (getattr(generation_config, "task_to_id", None) or {}).items()
    }
    special_token_ids = {
        "bos": int(tokenizer.bos_token_id),
        "decoder_start": int(generation_config.decoder_start_token_id),
        "eos": int(generation_config.eos_token_id),
        "pad": int(tokenizer.pad_token_id),
        "unknown": int(tokenizer.unk_token_id),
        "stop": [int(generation_config.eos_token_id)],
        "no_speech": no_speech_id,
        "no_timestamps": no_timestamps_id,
        "tasks": task_ids,
    }
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH
    ENCODER_SIGNAL_LENGTH = (STFT_SIGNAL_LENGTH + 1) // 2
    MAX_SEQ_LEN = int(model.config.max_target_positions)

    # Attention scaling (d**-0.25 on q & k) is now folded into the fused qkv/kv weights inside the
    # WHISPER_ENCODER / WHISPER_DECODER modules, so no separate pre-scaling loop is needed here.

    custom_stft = STFT_Process(
        model_type='stft_B_power',
        n_fft=NFFT_STFT,
        win_length=WINDOW_LENGTH,
        hop_len=HOP_LENGTH,
        max_frames=0,
        window_type=WINDOW_TYPE,
        pad_mode='reflect',
        center_pad=True,
        input_scale=(1.0 / 32768.0) if INPUT_AUDIO_DTYPE == "INT16" else 1.0,
        drop_last_frame=True,
    ).eval()
    whisper_encoder = WHISPER_ENCODER(model.model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, NUM_LAYER_DE)

    output_names = []
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)
    dynamic_axes = {'audio': {2: 'audio_len'}}
    for i in range(NUM_LAYER_DE):
        name = f'en_key_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'signal_len'}

    torch.onnx.export(
        whisper_encoder,
        (audio,),
        onnx_model_Encoder,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del whisper_encoder
    del audio
    del custom_stft
    del name
    del output_names
    del dynamic_axes
    gc.collect()

    suppress_tokens_config = getattr(generation_config, "suppress_tokens", None)
    if suppress_tokens_config is None:
        suppress_tokens = None
    else:
        suppress_tokens = torch.tensor(suppress_tokens_config, dtype=torch.int64)

    # ── Decoder token-embedding graph (mirrors Qwen Decoder_Embed; keeps int ids out of the decoder) ─
    whisper_embed = WHISPER_DECODER_EMBED(model.model.decoder)
    embed_input_ids = torch.ones((1, 4), dtype=torch.int32)
    torch.onnx.export(
        whisper_embed,
        (embed_input_ids,),
        onnx_model_Embed,
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'ids_len'},
            'hidden_states': {0: 'batch', 1: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del whisper_embed
    del embed_input_ids

    # ── Prefill position-embedding + causal-mask graph (mirrors Qwen Rotary_Mask_Prefill) ────────────
    attention_mask_dtype = KV_DTYPE
    whisper_prefill = WHISPER_PREFILL(model.model.decoder, MAX_SEQ_LEN, attention_mask_dtype)
    prefill_ids_len = torch.tensor([4], dtype=torch.int64)
    prefill_history_len = torch.tensor([0], dtype=torch.int64)
    torch.onnx.export(
        whisper_prefill,
        (prefill_ids_len, prefill_history_len),
        onnx_model_Prefill,
        input_names=['ids_len', 'history_len'],
        output_names=['position_embed', 'attention_mask', 'kv_seq_len'],
        dynamic_axes={
            'position_embed': {1: 'ids_len'},
            'attention_mask': {1: 'ids_len', 2: 'kv_seq_len'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del whisper_prefill
    del prefill_ids_len
    del prefill_history_len

    # ── Decode position-embedding graph for the single new token (mirrors Qwen Rotary_Mask_Decode) ──
    whisper_decode = WHISPER_DECODE(model.model.decoder)
    decode_kv_seq_len = torch.tensor([4], dtype=torch.int64)
    torch.onnx.export(
        whisper_decode,
        (decode_kv_seq_len,),
        onnx_model_Decode,
        input_names=['kv_seq_len'],
        output_names=['position_embed', 'kv_seq_len_next'],
        dynamic_axes={},
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del whisper_decode
    del decode_kv_seq_len
    gc.collect()

    # ── Decoder main graph (pure float: token + position embeddings and the mask arrive as inputs) ──
    whisper_decoder = WHISPER_DECODER(model, suppress_tokens, NUM_LAYER_DE)
    save_encoder_key = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, ENCODER_SIGNAL_LENGTH), dtype=KV_DTYPE)
    save_encoder_value = torch.zeros((NUM_HEAD_DE, ENCODER_SIGNAL_LENGTH, HEAD_DIM_DE), dtype=KV_DTYPE)
    batch_size = 1  # Production decoding is batch one; the sequence axes remain dynamic.
    past_key_de = torch.zeros((batch_size, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=KV_DTYPE)
    past_value_de = torch.zeros((batch_size, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=KV_DTYPE)
    hidden_states_de = torch.ones((batch_size, 1, HIDDEN_SIZE), dtype=torch.float32)
    position_embed_de = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1), dtype=attention_mask_dtype)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {}

    for i in range(NUM_LAYER_DE):
        name = f'in_de_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_key_de)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_de_key_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_value_de)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len'}
        name = f'out_de_value_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len_plus_ids_len'}

    for i in range(NUM_LAYER_DE):
        name = f'en_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(save_encoder_key)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(save_encoder_value)
        dynamic_axes[name] = {1: 'signal_len'}

    input_names.append('hidden_states')
    all_inputs.append(hidden_states_de)
    dynamic_axes['hidden_states'] = {0: 'batch', 1: 'ids_len'}
    input_names.append('position_embed')
    all_inputs.append(position_embed_de)
    dynamic_axes['position_embed'] = {1: 'ids_len'}
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    dynamic_axes['attention_mask'] = {1: 'ids_len', 2: 'kv_seq_len'}

    output_names.append('logits')
    dynamic_axes['logits'] = {0: 'batch'}

    torch.onnx.export(
        whisper_decoder,
        tuple(all_inputs),
        onnx_model_Decoder,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del model
    del whisper_decoder
    del save_encoder_key
    del save_encoder_value
    del hidden_states_de
    del position_embed_de
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes

    begin_suppress = BEGIN_SUPPRESS(begin_suppress_token_ids, VOCAB_SIZE)
    begin_logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    torch.onnx.export(
        begin_suppress,
        (begin_logits,),
        onnx_model_Begin_Suppress,
        input_names=['logits_in'],
        output_names=['logits_out'],
        dynamic_axes={
            'logits_in': {0: 'batch'},
            'logits_out': {0: 'batch'},
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False,
    )
    del begin_suppress
    del begin_logits

    no_speech_detection = NO_SPEECH_DETECTION(no_speech_id, suppress_tokens, VOCAB_SIZE)
    no_speech_logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    torch.onnx.export(
        no_speech_detection,
        (no_speech_logits,),
        onnx_model_No_Speech,
        input_names=['logits'],
        output_names=['no_speech_prob'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'no_speech_prob': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del no_speech_detection
    del no_speech_logits

    logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    save_id = torch.zeros((1, 10), dtype=torch.int32)  # Representative dynamic history.
    penalty_value = torch.ones((1,), dtype=torch.float32)
    penalty_range = torch.ones((1,), dtype=torch.int64)

    # ── Greedy Search (argmax + save_id history; used together with APPLY_PENALTY) ──────────────────
    torch.onnx.export(
        GREEDY_SEARCH(),
        (logits[[0]], save_id[[0]]),
        onnx_model_Greedy,
        input_names=['logits', 'save_id_in'],
        output_names=['max_logits_idx', 'save_id_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'max_logits_idx': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len_out'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Argmax (greedy decoding without a repetition penalty) ────────────────────────────────────────
    torch.onnx.export(
        ARGMAX(),
        (logits,),
        onnx_model_Argmax,
        input_names=['logits'],
        output_names=['max_logits_idx'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Apply Penalty (sliding-window repetition penalty on the logits) ─────────────────────────────
    torch.onnx.export(
        APPLY_PENALTY(),
        (logits, save_id, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
        output_names=['logits_out'],
        dynamic_axes={
            'logits_in': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'logits_out': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Top-K / Top-P sampling with standard repetition penalty ─────────────────────────────────────
    sampling_temperature = torch.tensor([0.8], dtype=torch.float32)
    sampling_top_k = torch.tensor([50], dtype=torch.int32)
    sampling_top_p = torch.tensor([0.95], dtype=torch.float32)
    sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            sampling_temperature,
            sampling_top_k,
            sampling_top_p,
            sampling_repetition_penalty,
            save_id,
        ),
        onnx_model_Sampling,
        input_names=[
            'logits', 'temperature', 'top_k', 'top_p',
            'repetition_penalty', 'previous_ids'
        ],
        output_names=['sampled_id', 'save_id_out'],
        dynamic_axes={
            'previous_ids': {1: 'history_len'},
            'save_id_out': {1: 'history_len'},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    del past_key_de
    del past_value_de
    del logits
    del save_id
    del penalty_value
    del penalty_range
    del sampling_temperature
    del sampling_top_k
    del sampling_top_p
    del sampling_repetition_penalty
    gc.collect()

    metadata_marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        onnx_model_Metadata,
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )
    del metadata_marker

    onnx_metadata = build_model_metadata(
        {
            "audio_pcm_scale": 32768,
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
        },
    )

    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    gc.collect()

for _folder in (_raw_onnx_dir, ONNX_DIR):
    Shared_Merged.delete_obsolete_strategy_artifacts(
        _folder,
        model_file_names=MODEL_FILE_NAMES,
    )
if ONNX_DIR.exists():
    shutil.rmtree(ONNX_DIR)
ONNX_DIR.mkdir(parents=True)
print("\n[SharedMerged] Building Whisper strategy graphs + shared initializer bundle ...")
_bundle = Shared_Merged.build_shared_merged_bundle(
    _raw_onnx_dir,
    out_folder=ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
)
_standalones = Shared_Merged.copy_runtime_standalones(
    _raw_onnx_dir,
    ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
)
_embed_dedup = _bundle["embed_dedup"]
_shared_stats = _bundle["shared_stats"]
if _embed_dedup.get("applied"):
    print(
        "    Tied Embed/LM-head table deduplicated: "
        f"{_embed_dedup['bytes_eliminated']} bytes; inserted {_embed_dedup['inserted_node']}"
    )
else:
    print(f"    Tied Embed/LM-head dedup skipped: {_embed_dedup.get('reason', 'not applicable')}")
print(
    "    Shared initializer physical dedup: "
    f"{_shared_stats['initializer_count']} logical -> {_shared_stats['unique_data_count']} unique "
    f"({_shared_stats['logical_data_bytes']} -> {_shared_stats['physical_data_bytes']} bytes)"
)
for _name, _path in _bundle["graphs"].items():
    print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
write_metadata_carrier(
    ONNX_DIR / MODEL_FILE_NAMES["metadata"], onnx_metadata
)
print(
    f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
    f"({Path(_bundle['shared_data']).stat().st_size} bytes)"
)
print(f"    Copied {len(_standalones)} runtime standalone graph(s): NoSpeech, Metadata")

_raw_onnx_temp.cleanup()

# ── Save the tokenizer + generation config into the final merged folder so it runs ──
# inference stand-alone (no external Whisper model path needed at inference time).
_tokenizer_dir = str(ONNX_DIR / "tokenizer")
tokenizer.save_pretrained(_tokenizer_dir)
generation_config.save_pretrained(_tokenizer_dir)
print(f"[Tokenizer] Saved tokenizer + generation config -> {_tokenizer_dir}")

print('\nExport done!\n')
if subprocess.call(
    [
        sys.executable,
        str(SCRIPT_DIR / "Inference_Whisper_ONNX.py"),
        "--onnx-folder",
        str(ONNX_DIR),
    ],
    cwd=str(SCRIPT_DIR),
) != 0:
    raise RuntimeError("Whisper inference failed after export.")
