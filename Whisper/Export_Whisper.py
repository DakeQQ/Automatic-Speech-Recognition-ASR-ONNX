import gc
import subprocess
import sys
import os
import time
import torch
import torchaudio
import numpy as np
from STFT_Process import STFT_Process                                             # The custom STFT/ISTFT can be exported in ONNX format.
from transformers import AutoFeatureExtractor, AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig


model_path = "/home/DakeQQ/Downloads/whisper-large-v3-turbo"    # Source Whisper model (HF) download path.

# -- Exported ONNX graph paths: core pipeline (Embed keeps token ids out of the float decoder; Prefill / Decode build position embedding + causal mask) --
onnx_folder               = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Whisper_ONNX")   # Local folder next to this script holding all exported ONNX graphs; created automatically if missing.
os.makedirs(onnx_folder, exist_ok=True)
onnx_model_Metadata       = os.path.join(onnx_folder, "Whisper_Metadata.onnx")
onnx_model_Encoder        = os.path.join(onnx_folder, "Whisper_Encoder.onnx")
onnx_model_Decoder        = os.path.join(onnx_folder, "Whisper_Decoder.onnx")
onnx_model_Embed          = os.path.join(onnx_folder, "Whisper_Decoder_Embed.onnx")
onnx_model_Prefill        = os.path.join(onnx_folder, "Whisper_Position_Mask_Prefill.onnx")
onnx_model_Decode         = os.path.join(onnx_folder, "Whisper_Position_Mask_Decode.onnx")

# -- Exported ONNX graph paths: beam / greedy search, repetition-penalty reset, no-speech detection --
onnx_model_Greedy         = os.path.join(onnx_folder, "Whisper_Greedy_Search.onnx")
onnx_model_Argmax         = os.path.join(onnx_folder, "Whisper_Argmax.onnx")
onnx_model_First_Beam     = os.path.join(onnx_folder, "Whisper_First_Beam_Search.onnx")
onnx_model_Second_Beam    = os.path.join(onnx_folder, "Whisper_Second_Beam_Search.onnx")
onnx_model_Penality       = os.path.join(onnx_folder, "Whisper_Apply_Penality.onnx")
onnx_model_No_Speech      = os.path.join(onnx_folder, "Whisper_No_Speech_Detection.onnx")

# -- Decoding strategy --
USE_BEAM_SEARCH       = False           # Use beam search (True) or greedy search (False).
BEAM_SIZE             = 3               # Number of beams in searching.
TOP_K                 = 3               # Top-k candidates considered during decoding.
MAX_BEAM_SIZE         = 10              # Max beams supported by the exported model.
MAX_SEQ_LEN           = 448             # Maximum decoded sequence length; must be <= 448.

# -- Repetition penalty --
REPEAT_PENALITY       = 0.8             # Range 0.0 - 1.0; "1.0" means no penalty.
PENALITY_RANGE        = 20              # Penalizes the most recent output; "30" means the last 30 tokens.
REMOVE_REPEATED_PARTS = False           # Non-Whisper cleanup for runaway repetition;

# -- Language / task --
TARGET_LANGUAGE       = "en"            # A language listed in the get_language_id function's language_map.
TASK                  = "transcribe"    # One of: ['transcribe', 'translate'].
DETECT_LANGUAGE       = True            # Whisper-style auto language detection; overrides TARGET_LANGUAGE when True.

# -- No-speech detection --
NO_SPEECH_DETECTION   = True            # Skip silent / non-speech windows using the <|nospeech|> probability.
NO_SPEECH_THRESHOLD   = 0.6             # Whisper default; higher = stricter silence rejection.

# -- Audio / STFT front-end --
INPUT_AUDIO_LENGTH    = 480000          # Whisper's default 30-second chunk; overwritten from preprocessor_config when available.
SAMPLE_RATE           = 16000           # Model sample rate, do not edit.
WINDOW_TYPE           = "hann"          # Window function used in the STFT.
NFFT_STFT             = 400             # Number of FFT components for the STFT; edit carefully.
WINDOW_LENGTH         = 400             # Length of windowing; edit carefully.
HOP_LENGTH            = 160             # Samples between successive STFT frames; edit carefully.
PRE_EMPHASIZE         = 0.97            # Audio pre-emphasis coefficient.
INPUT_AUDIO_DTYPE     = "INT16"         # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. "INT16" feeds raw PCM (÷32768 in-graph); "F32"/"F16" feed audio pre-normalised to [-1, 1].
# N_MELS              = 80              # Set from the Whisper model config (num_mel_bins); edit carefully.

# -- Export --
OPSET                 = 18              # ONNX opset version.


if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


model_path_lower = model_path.lower()
if ("v3" in model_path_lower) or ("crisperwhisper" in model_path_lower) or ("anime" in model_path_lower) or ("belle" in model_path_lower) or ("turbo" in model_path_lower) or ("distil" in model_path_lower):
    is_v3 = True
    if 'v0.3' in model_path_lower:
        custom_vocab = True
    else:
        custom_vocab = False
    print("\nExport the Whisper-V3")
else:
    is_v3 = False
    custom_vocab = False
    print("\nExport the Whisper-V2")


tokenizer = AutoTokenizer.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
generation_config = GenerationConfig.from_pretrained(model_path)
INPUT_AUDIO_LENGTH = int(getattr(feature_extractor, "n_samples", INPUT_AUDIO_LENGTH))
NFFT_STFT = int(getattr(feature_extractor, "n_fft", NFFT_STFT))
HOP_LENGTH = int(getattr(feature_extractor, "hop_length", HOP_LENGTH))
SAMPLE_RATE = int(getattr(feature_extractor, "sampling_rate", SAMPLE_RATE))
WINDOW_LENGTH = NFFT_STFT
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
MAX_INPUT_AUDIO_LENGTH = INPUT_AUDIO_LENGTH
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


_FULL_NAME_TO_CODE = {data['full_name']: code for code, data in _LANGUAGE_DATA.items()}


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


def get_language_id(language_input, custom_vocab=False, generation_config=None):
    normalized_input = language_input.lower().strip()
    lang_code = None
    if normalized_input in _LANGUAGE_DATA:
        lang_code = normalized_input
    elif normalized_input in _FULL_NAME_TO_CODE:
        lang_code = _FULL_NAME_TO_CODE[normalized_input]
    elif normalized_input in _ALIAS_TO_CODE:
        lang_code = _ALIAS_TO_CODE[normalized_input]
    if lang_code:
        if generation_config is not None and hasattr(generation_config, "lang_to_id"):
            lang_token = f"<|{lang_code}|>"
            if lang_token in generation_config.lang_to_id:
                return int(generation_config.lang_to_id[lang_token])
        language_info = _LANGUAGE_DATA[lang_code]
        id_key = 'custom_id' if custom_vocab else 'id'
        return language_info.get(id_key)
    return None


def get_task_id(task_input, is_v3, custom_vocab=False, generation_config=None):
    task_input = task_input.lower()
    if generation_config is not None and hasattr(generation_config, "task_to_id"):
        start_token = getattr(generation_config, "decoder_start_token_id", None)
        stop_token = getattr(generation_config, "eos_token_id", None)
        if start_token is not None and stop_token is not None and task_input in generation_config.task_to_id:
            return int(start_token), [int(stop_token)], int(generation_config.task_to_id[task_input])
    if custom_vocab:
        stop_token = 18871
        start_token = 18872
        task_map = {
            'translate': 18973,
            'transcribe': 18974
        }
        return start_token, [stop_token], task_map[task_input]
    stop_token = 50257
    start_token = 50258
    if is_v3:
        task_map = {
            'translate': 50359,
            'transcribe':  50360
        }
        return start_token, [stop_token], task_map[task_input]
    else:
        task_map = {
            'translate': 50358,
            'transcribe': 50359
        }
        return start_token, [stop_token], task_map[task_input]


def remove_repeated_parts(ids, repeat_words_threshold, ids_len):
    if ids_len <= repeat_words_threshold:
        return ids
    side_L = repeat_words_threshold // 2
    side_R = side_L + 1
    boundary = ids_len - side_L
    for i in range(side_L, boundary):
        for j in range(i + repeat_words_threshold, boundary):
            check = []
            for k in range(-side_L, side_R):
                if ids[j + k] == ids[i + k]:
                    check.append(True)
                else:
                    check.append(False)
                    break
            if False not in check:
                return ids[: j - side_L]
    return ids


def build_model_metadata(*sections):
    def _norm(value):
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    merged = {}
    for section in sections:
        for key, value in section.items():
            if value is not None:
                merged[key] = _norm(value)
    return merged


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


class GREEDY_SEARCH(torch.nn.Module):
    # Pure argmax that also appends the chosen token to the running save_id history (Qwen ASR style).
    # Used when a repetition penalty is active so APPLY_PENALITY can read the on-device history.
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


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super(METADATA_CARRIER, self).__init__()

    def forward(self, marker):
        return marker


class APPLY_PENALITY(torch.nn.Module):
    # Sliding-window repetition penalty (Qwen ASR style): multiply the logits of the most recent
    # `penality_range` tokens by `penality_value`. penality_range is baked into the graph at export
    # via int(), so the runtime input only needs to match the export value.
    def __init__(self):
        super(APPLY_PENALITY, self).__init__()

    def forward(self, logits, save_id, penality_value, penality_range):
        penality_range_val = int(penality_range.item())
        target_indices = save_id[:, -penality_range_val:].long()
        penalised = logits.gather(1, target_indices) * penality_value
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


class FIRST_BEAM_SEARCH(torch.nn.Module):
    # First beam step (Qwen ASR style): expand the single prefilled sequence into `beam_size` beams.
    # Scores are log-probabilities via logits - logsumexp(logits); no repetition penalty here (it is a
    # separate APPLY_PENALITY pass on the logits beforehand).
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp
        for i in range(self.num_keys_values):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]
        return *self.save_keys_values, save_id, top_beam_prob.transpose(0, 1), top_beam_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    # Subsequent beam steps (Qwen ASR style): accumulate log-prob scores, pick the global top-`beam_size`
    # continuations from the top-`topK` per beam, and reorder the K/V caches / save_id via index_select.
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=True)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = flat_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)
        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)
        return *self.save_keys_values, save_id, top_beam_prob.unsqueeze(-1), top_beam_indices, max_logits_idx


class WHISPER_ENCODER(torch.nn.Module):
    def __init__(self, whisper, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(WHISPER_ENCODER, self).__init__()
        self.encoder = whisper.encoder
        self.decoder = whisper.decoder
        self.stft_model = stft_model
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, "slaney", 'slaney')).transpose(0, 1).unsqueeze(0)
        self.save_encoder_key = [None] * num_layers_de
        self.save_encoder_value = [None] * num_layers_de
        self.inv_int16 = float(1.0 / 32768.0)
        # int16 audio is raw PCM (normalised in forward via ÷32768); f32/f16 audio is
        # assumed pre-normalised to [-1, 1], so the in-graph division is skipped.
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")
        self.pre_emphasis = float(pre_emphasis)
        self.num_heads = self.encoder.layers[0].self_attn.num_heads
        self.head_dim = self.encoder.layers[0].self_attn.head_dim
        self.hidden_size = self.encoder.layers[0].self_attn.out_proj.in_features
        self.cross_num_heads = self.decoder.layers[0].encoder_attn.num_heads
        self.cross_head_dim = self.decoder.layers[0].encoder_attn.head_dim
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
            # Cross-attention keys/values are produced here from the encoder output; fuse k/v into one Linear
            # and fold the cross-attention d**-0.25 scale into the key half.
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.decoder.layers:
                cross_attn = decoder_layer.encoder_attn
                kv = torch.nn.Linear(cross_attn.k_proj.in_features, cross_attn.k_proj.out_features * 2, bias=True)
                kv.weight.copy_(torch.cat([cross_attn.k_proj.weight, cross_attn.v_proj.weight], dim=0))
                kv.bias.copy_(torch.cat([_bias_or_zero(cross_attn.k_proj), _bias_or_zero(cross_attn.v_proj)], dim=0))
                kv.weight.data[:cross_attn.k_proj.out_features].mul_(cross_scale)   # fold cross scale into k
                cross_attn.kv = kv
                del cross_attn.k_proj, cross_attn.v_proj

    def forward(self, audio):
        if self.input_audio_is_int16:
            audio = audio.float() * self.inv_int16                               # Raw PCM -> [-1, 1]; matches Whisper (no DC offset / pre-emphasis / gain normalization).
        else:
            audio = audio.float()                                                # f32/f16 input is already normalised to [-1, 1].
        real_part, imag_part = self.stft_model(audio)                 # Whisper STFT uses center padding with reflect mode.
        power = (real_part * real_part + imag_part * imag_part)[:, :, :-1]        # Power spectrum; Whisper drops the final STFT frame: stft[..., :-1].
        mel_features = torch.matmul(self.fbank, power).clamp(min=1e-10).log10()
        mel_features = torch.maximum(mel_features, mel_features.max() - 8.0)
        mel_features = (mel_features + 4.0) * 0.25
        hidden_states = torch.nn.functional.gelu(self.encoder.conv2(torch.nn.functional.gelu(self.encoder.conv1(mel_features)))).transpose(1, 2)
        hidden_states = hidden_states + self.encoder.embed_positions.weight[:hidden_states.shape[1]].float()
        for encoder_layer in self.encoder.layers:
            hidden_states_norm = encoder_layer.self_attn_layer_norm(hidden_states)
            qkv = encoder_layer.self_attn.qkv(hidden_states_norm).view(-1, 3 * self.num_heads, self.head_dim).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)                            # each (num_heads, T, head_dim)
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1), v).transpose(0, 1).reshape(1, -1, self.hidden_size)
            hidden_states_attn = encoder_layer.self_attn.out_proj(attn)
            hidden_states_attn += hidden_states
            hidden_states = hidden_states_attn + encoder_layer.fc2(encoder_layer.activation_fn(encoder_layer.fc1(encoder_layer.final_layer_norm(hidden_states_attn))))
        hidden_states = self.encoder.layer_norm(hidden_states)
        for i, decoder_layer in enumerate(self.decoder.layers):
            cross_kv = decoder_layer.encoder_attn.kv(hidden_states).view(-1, 2 * self.cross_num_heads, self.cross_head_dim).transpose(0, 1)
            k, v = cross_kv.split(self.cross_num_heads, dim=0)                   # each (num_heads, T, head_dim)
            self.save_encoder_key[i] = k.half().transpose(1, 2)                  # f16 cross-attention key   (num_heads, head_dim, T)
            self.save_encoder_value[i] = v.half()                                # f16 cross-attention value (num_heads, T, head_dim)
        return *self.save_encoder_key, *self.save_encoder_value


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
    def __init__(self, decoder, max_seq_len):
        super(WHISPER_PREFILL, self).__init__()
        self.register_buffer('position_weight', decoder.embed_positions.weight.unsqueeze(0).half())
        self.register_buffer('attention_mask', (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128)

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len].half()
        return position_embed, attention_mask, kv_seq_len


class WHISPER_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Qwen's Rotary_Mask_Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, decoder):
        super(WHISPER_DECODE, self).__init__()
        self.register_buffer('position_weight', decoder.embed_positions.weight.unsqueeze(0).half())

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
        self.idx_en_key = self.num_layers_de + self.num_layers_de            # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + self.num_layers_de             # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + self.num_layers_de             # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                              # position-embedding input (4 * L + 1)
        self.save_de_keys = [None] * self.num_layers_de
        self.save_de_values = [None] * self.num_layers_de
        self.num_heads = self.decoder.layers[0].self_attn.num_heads
        self.head_dim = self.decoder.layers[0].self_attn.head_dim
        self.hidden_size = self.decoder.layers[0].self_attn.out_proj.in_features
        self.cross_num_heads = self.decoder.layers[0].encoder_attn.num_heads
        self.cross_head_dim = self.decoder.layers[0].encoder_attn.head_dim
        self.suppress_tokens_penality = torch.zeros((1, self.whisper.proj_out.out_features), dtype=torch.float32)
        if self.suppress_tokens is not None:
            self.suppress_tokens_penality[:, self.suppress_tokens] = float(-128.0)
        self._fuse_weights()

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

    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for idx, decoder_layer in enumerate(self.decoder.layers):
            hidden_states_norm = decoder_layer.self_attn_layer_norm(hidden_states)
            qkv = decoder_layer.self_attn.qkv(hidden_states_norm).view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = qkv.split(self.num_heads, dim=1)                            # each (batch, num_heads, ids_len, head_dim)
            k = torch.cat((all_inputs[idx], k.half().transpose(-1, -2)), dim=-1)  # f16 key cache (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v.half()), dim=-2)  # f16 value cache
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q.half(), k) + attention_mask, dim=-1), v).transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float()
            hidden_states_attn = decoder_layer.self_attn.out_proj(attn)
            hidden_states_attn += hidden_states
            hidden_states_attn_norm = decoder_layer.encoder_attn_layer_norm(hidden_states_attn)
            q = decoder_layer.encoder_attn.q_proj(hidden_states_attn_norm).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
            hidden_state_cross = decoder_layer.encoder_attn.out_proj(attn.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_states_attn
            hidden_states = hidden_state_cross + decoder_layer.fc2(decoder_layer.activation_fn(decoder_layer.fc1(decoder_layer.final_layer_norm(hidden_state_cross))))
        hidden_states = self.decoder.layer_norm(hidden_states[:, -1])
        logits = self.whisper.proj_out(hidden_states)
        if self.suppress_tokens is not None:
            logits = logits + self.suppress_tokens_penality
        return *self.save_de_keys, *self.save_de_values, logits


print('\nExport start...\n')
with torch.inference_mode():
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=False).eval()
    except:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True).eval()
    HIDDEN_SIZE = model.config.d_model
    NUM_HEAD_EN = model.model.config.encoder_attention_heads
    NUM_HEAD_DE = model.model.config.decoder_attention_heads
    HEAD_DIM_EN = model.model.encoder.layers._modules['0'].self_attn.head_dim
    HEAD_DIM_DE = model.model.decoder.layers._modules['0'].self_attn.head_dim
    NUM_LAYER_DE = model.config.decoder_layers
    N_MELS = model.config.num_mel_bins
    VOCAB_SIZE = model.config.vocab_size
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
    if MAX_SEQ_LEN > model.config.max_target_positions:
        MAX_SEQ_LEN = model.config.max_target_positions

    # Attention scaling (d**-0.25 on q & k) is now folded into the fused qkv/kv weights inside the
    # WHISPER_ENCODER / WHISPER_DECODER modules, so no separate pre-scaling loop is needed here.

    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE, pad_mode='reflect', center_pad=True).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    whisper_encoder = WHISPER_ENCODER(model.model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)

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
    whisper_prefill = WHISPER_PREFILL(model.model.decoder, MAX_SEQ_LEN)
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
    save_encoder_key = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float16)
    save_encoder_value = torch.zeros((NUM_HEAD_DE, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_DE), dtype=torch.float16)
    batch_size = 3  # Dummy batch value for the export trace.
    past_key_de = torch.zeros((batch_size, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float16)
    past_value_de = torch.zeros((batch_size, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float16)
    hidden_states_de = torch.ones((batch_size, 1, HIDDEN_SIZE), dtype=torch.float32)
    position_embed_de = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1), dtype=torch.float16)   # f16 mask matches the minimum-cast f16 attention scores

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

    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    logits = torch.ones((BEAM_SIZE, VOCAB_SIZE), dtype=torch.float32)
    save_id = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)            # 10 = dummy history length
    previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
    penality_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
    penality_range = torch.tensor([PENALITY_RANGE], dtype=torch.int64)

    # ── Greedy Search (argmax + save_id history; used together with APPLY_PENALITY) ──────────────────
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

    # ── Apply Penality (sliding-window repetition penalty on the logits) ─────────────────────────────
    torch.onnx.export(
        APPLY_PENALITY(),
        (logits, save_id, penality_value, penality_range),
        onnx_model_Penality,
        input_names=['logits_in', 'save_id_in', 'penality_value', 'penality_range'],
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

    # ── First Beam Search ────────────────────────────────────────────────────────────────────────────
    first_beam_search = FIRST_BEAM_SEARCH(NUM_LAYER_DE)
    past_keys_greedy = past_key_de[[0]]
    past_values_greedy = past_value_de[[0]]

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_DE):
        name = f'in_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_keys_greedy)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_key_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        name = f'in_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_values_greedy)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len'}
        name = f'out_value_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len_plus_ids_len'}
    input_names.append('logits')
    all_inputs.append(logits[[0]])
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('save_id_out')
    output_names.append('top_beam_prob')
    output_names.append('top_beam_indices')
    output_names.append('max_logits_idx')
    dynamic_axes['logits'] = {0: 'batch'}
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}

    torch.onnx.export(
        first_beam_search,
        tuple(all_inputs),
        onnx_model_First_Beam,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Second Beam Search (same output layout as First Beam) ────────────────────────────────────────
    all_inputs = []
    input_names = []
    for i in range(NUM_LAYER_DE):
        name = f'in_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_key_de)
    for i in range(NUM_LAYER_DE):
        name = f'in_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_value_de)
    input_names.append('logits')
    all_inputs.append(logits)
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}

    second_beam_search = SECOND_BEAM_SEARCH(NUM_LAYER_DE)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_Second_Beam,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    del first_beam_search
    del second_beam_search
    del past_key_de
    del past_value_de
    del past_keys_greedy
    del past_values_greedy
    del logits
    del previous_prob
    del save_id
    del penality_value
    del penality_range
    del topK
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
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
            "whisper_metadata_version": 1,
            "producer": "Export_Whisper.py",
            "model_variant": "v3" if is_v3 else "v2",
            "custom_vocab": custom_vocab,
        },
        {
            "num_decoder_layers": NUM_LAYER_DE,
            "num_encoder_heads": NUM_HEAD_EN,
            "num_decoder_heads": NUM_HEAD_DE,
            "encoder_head_dim": HEAD_DIM_EN,
            "decoder_head_dim": HEAD_DIM_DE,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "num_mels": N_MELS,
            "nfft_stft": NFFT_STFT,
            "window_length": WINDOW_LENGTH,
            "hop_length": HOP_LENGTH,
            "window_type": WINDOW_TYPE,
            "pre_emphasis": PRE_EMPHASIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "input_audio_length": INPUT_AUDIO_LENGTH,
            "input_audio_dtype": INPUT_AUDIO_DTYPE,
            "no_timestamps_token_id": no_timestamps_id,
            "no_speech_token_id": no_speech_id,
            "begin_suppress_token_ids": ",".join(str(t) for t in begin_suppress_token_ids),
            "suppress_token_ids": ",".join(str(t) for t in (suppress_tokens.tolist() if suppress_tokens is not None else [])),
        },
    )

    _metadata_targets = [
        onnx_model_Metadata, onnx_model_Encoder, onnx_model_Decoder, onnx_model_Embed,
        onnx_model_Prefill, onnx_model_Decode, onnx_model_Greedy,
        onnx_model_Argmax, onnx_model_First_Beam, onnx_model_Second_Beam,
        onnx_model_Penality, onnx_model_No_Speech,
    ]
    _written, _skipped = [], []
    for _target in _metadata_targets:
        if not os.path.exists(_target):
            continue
        try:
            write_onnx_metadata(_target, onnx_metadata)
            _written.append(os.path.basename(_target))
        except Exception as _exc:  # noqa: BLE001 - one bad graph must not abort export
            _skipped.append(f"{os.path.basename(_target)} ({_exc})")

    print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(_written)} ONNX graph(s):")
    for _key in sorted(onnx_metadata):
        print(f"    {_key} = {onnx_metadata[_key]}")
    if _skipped:
        print("[Metadata] Skipped (kept usable, metadata not written):")
        for _entry in _skipped:
            print(f"    {_entry}")
    gc.collect()

print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_Whisper_ONNX.py ...')
subprocess.run(
    [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Inference_Whisper_ONNX.py"), "--onnx-folder", onnx_folder],
    check=True,
)
