import gc
import time
import onnxruntime
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig


model_path = "/home/DakeQQ/Downloads/whisper-large-v3-turbo"                                          # The Whisper project download path.
onnx_model_A = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Encoder.onnx"                             # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Decoder.onnx"                             # The exported onnx model path.
if "3.5" in model_path:
    test_audio = ["../example/en.mp3"]                                                                    # The test audio list.
else:
    test_audio = ["../example/zh.mp3", "../example/en.mp3", "../example/ja.mp3", "../example/ko.mp3"]     # The test audio list.


DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Whisper series models only support dynamic_axes due to their transformer structure.
INPUT_AUDIO_LENGTH = 240000                                 # The maximum input audio length.
WINDOW_TYPE = 'hann'                                        # Type of window function used in the STFT
# N_MELS = 80                                               # Setting by whisper model config. Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 400                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
TARGET_LANGUAGE = "en"                                      # Choose a language listed in the get_language_id function's language_map.
TASK = 'transcribe'                                         # Choose one of : ['transcribe', 'translate']
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SEQ_LEN = 80                                            # It should less than 448.


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


def normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)
  

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


def get_language_id(language_input, custom_vocab=False):
    normalized_input = language_input.lower().strip()
    lang_code = None
    if normalized_input in _LANGUAGE_DATA:
        lang_code = normalized_input
    elif normalized_input in _FULL_NAME_TO_CODE:
        lang_code = _FULL_NAME_TO_CODE[normalized_input]
    elif normalized_input in _ALIAS_TO_CODE:
        lang_code = _ALIAS_TO_CODE[normalized_input]
    if lang_code:
        language_info = _LANGUAGE_DATA[lang_code]
        id_key = 'custom_id' if custom_vocab else 'id'
        return language_info.get(id_key)
    return None


def get_task_id(task_input, is_v3, custom_vocab=False):
    task_input = task_input.lower()
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
        return np.array([ids], dtype=np.int32)
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
                return np.array([ids[: j - side_L]], dtype=np.int32)
    return np.array([ids], dtype=np.int32)


class WHISPER_ENCODER(torch.nn.Module):
    def __init__(self, whisper, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, num_layers_de, num_head_en, num_head_de):
        super(WHISPER_ENCODER, self).__init__()
        self.encoder = whisper.encoder
        self.decoder = whisper.decoder
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 0, sample_rate // 2, n_mels, sample_rate, "slaney", 'slaney')).transpose(0, 1).unsqueeze(0)
        self.save_encoder_key = [None] * num_layers_de * num_head_de
        self.save_encoder_value = [None] * num_layers_de * num_head_de
        self.inv_int16 = float(1.0 / 32768.0)
        self.num_head_en = num_head_en
        self.num_head_de = num_head_de

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-7).log10()
        mel_features = torch.maximum(mel_features, mel_features.max() - 8.0)
        mel_features = (mel_features + 4.0) * 0.25
        hidden_states = torch.nn.functional.gelu(self.encoder.conv2(torch.nn.functional.gelu(self.encoder.conv1(mel_features)))).transpose(1, 2)
        hidden_states = hidden_states + self.encoder.embed_positions.weight[:hidden_states.shape[1]].float()
        for encoder_layer in self.encoder.layers:
            hidden_states_norm = encoder_layer.self_attn_layer_norm(hidden_states)
            q = torch.matmul(hidden_states_norm, encoder_layer.self_attn.q_proj.weight[0]) + encoder_layer.self_attn.q_proj.bias[0]
            k = torch.matmul(hidden_states_norm, encoder_layer.self_attn.k_proj.weight[0]).transpose(1, 2)
            v = torch.matmul(hidden_states_norm, encoder_layer.self_attn.v_proj.weight[0]) + encoder_layer.self_attn.v_proj.bias[0]
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k), dim=-1), v)
            hidden_states_attn = torch.matmul(attn, encoder_layer.self_attn.out_proj.weight[0])
            for i in range(1, self.num_head_en):
                q = torch.matmul(hidden_states_norm, encoder_layer.self_attn.q_proj.weight[i]) + encoder_layer.self_attn.q_proj.bias[i]
                k = torch.matmul(hidden_states_norm, encoder_layer.self_attn.k_proj.weight[i]).transpose(1, 2)
                v = torch.matmul(hidden_states_norm, encoder_layer.self_attn.v_proj.weight[i]) + encoder_layer.self_attn.v_proj.bias[i]
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k), dim=-1), v)
                hidden_states_attn += torch.matmul(attn, encoder_layer.self_attn.out_proj.weight[i])
            hidden_states_attn = hidden_states_attn + encoder_layer.self_attn.out_proj.bias + hidden_states
            hidden_states = hidden_states_attn + encoder_layer.fc2(encoder_layer.activation_fn(encoder_layer.fc1(encoder_layer.final_layer_norm(hidden_states_attn))))
        hidden_states = self.encoder.layer_norm(hidden_states)
        count = 0
        for decoder_layer in self.decoder.layers:
            for i in range(self.num_head_de):
                self.save_encoder_key[count] = torch.matmul(hidden_states, decoder_layer.encoder_attn.k_proj.weight[i]).transpose(1, 2)
                self.save_encoder_value[count] = torch.matmul(hidden_states, decoder_layer.encoder_attn.v_proj.weight[i]) + decoder_layer.encoder_attn.v_proj.bias[i]
                count += 1
        return *self.save_encoder_key, *self.save_encoder_value


class WHISPER_DECODER(torch.nn.Module):
    def __init__(self, whisper, max_seq_len, suppress_tokens, num_layers_de, num_head_de):
        super(WHISPER_DECODER, self).__init__()
        self.whisper = whisper
        self.decoder = whisper.model.decoder
        self.suppress_tokens = suppress_tokens
        self.num_layers_head_de = num_layers_de * num_head_de
        self.num_layers_head_de_2 = self.num_layers_head_de + self.num_layers_head_de
        self.num_layers_head_de_2_plus_1 = self.num_layers_head_de_2 + 1
        self.num_layers_head_de_2_plus_2 = self.num_layers_head_de_2 + 2
        self.num_layers_head_de_3_plus = self.num_layers_head_de_2_plus_2 + self.num_layers_head_de
        self.save_de_keys = [None] * self.num_layers_head_de
        self.save_de_values = [None] * self.num_layers_head_de
        self.num_head_de = num_head_de
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.suppress_tokens_penality = torch.ones((1, self.whisper.proj_out.out_features), dtype=torch.float32)
        self.decoder.embed_positions.weight.data = self.decoder.embed_positions.weight.data.unsqueeze(0)
        if self.suppress_tokens is not None:
            self.suppress_tokens_penality[:, self.suppress_tokens] = float(-128.0)

    def forward(self, *all_inputs):
        input_ids = all_inputs[self.num_layers_head_de_2]
        history_len = all_inputs[self.num_layers_head_de_2_plus_1]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        hidden_states = self.decoder.embed_tokens(input_ids) + self.decoder.embed_positions.weight[:, history_len: kv_seq_len]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        count = 0
        for idx, decoder_layer in enumerate(self.decoder.layers):
            idx *= self.num_head_de
            hidden_states_norm = decoder_layer.self_attn_layer_norm(hidden_states)
            q = torch.matmul(hidden_states_norm, decoder_layer.self_attn.q_proj.weight[0]) + decoder_layer.self_attn.q_proj.bias[0]
            k = torch.matmul(hidden_states_norm, decoder_layer.self_attn.k_proj.weight[0]).transpose(1, 2)
            v = torch.matmul(hidden_states_norm, decoder_layer.self_attn.v_proj.weight[0]) + decoder_layer.self_attn.v_proj.bias[0]
            k = torch.cat((all_inputs[idx], k), dim=2)
            v = torch.cat((all_inputs[idx + self.num_layers_head_de], v), dim=1)
            self.save_de_keys[count] = k
            self.save_de_values[count] = v
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)
            hidden_states_attn = torch.matmul(attn, decoder_layer.self_attn.out_proj.weight[0])
            count += 1
            for i in range(1, self.num_head_de):
                q = torch.matmul(hidden_states_norm, decoder_layer.self_attn.q_proj.weight[i]) + decoder_layer.self_attn.q_proj.bias[i]
                k = torch.matmul(hidden_states_norm, decoder_layer.self_attn.k_proj.weight[i]).transpose(1, 2)
                v = torch.matmul(hidden_states_norm, decoder_layer.self_attn.v_proj.weight[i]) + decoder_layer.self_attn.v_proj.bias[i]
                k = torch.cat((all_inputs[idx + i], k), dim=2)
                v = torch.cat((all_inputs[idx + i + self.num_layers_head_de], v), dim=1)
                self.save_de_keys[count] = k
                self.save_de_values[count] = v
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)
                hidden_states_attn += torch.matmul(attn, decoder_layer.self_attn.out_proj.weight[i])
                count += 1
            hidden_states_attn = hidden_states_attn + decoder_layer.self_attn.out_proj.bias + hidden_states
            hidden_states_attn_norm = decoder_layer.encoder_attn_layer_norm(hidden_states_attn)
            q = torch.matmul(hidden_states_attn_norm, decoder_layer.encoder_attn.q_proj.weight[0]) + decoder_layer.encoder_attn.q_proj.bias[0]
            attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, all_inputs[idx + self.num_layers_head_de_2_plus_2]), dim=-1), all_inputs[idx + self.num_layers_head_de_3_plus])
            hidden_state_cross = torch.matmul(attn, decoder_layer.encoder_attn.out_proj.weight[0])
            for i in range(1, self.num_head_de):
                q = torch.matmul(hidden_states_attn_norm, decoder_layer.encoder_attn.q_proj.weight[i]) + decoder_layer.encoder_attn.q_proj.bias[i]
                attn = torch.matmul(torch.nn.functional.softmax(torch.matmul(q, all_inputs[idx + i + self.num_layers_head_de_2_plus_2]), dim=-1), all_inputs[idx + i + self.num_layers_head_de_3_plus])
                hidden_state_cross += torch.matmul(attn, decoder_layer.encoder_attn.out_proj.weight[i])
            hidden_state_cross = hidden_state_cross + decoder_layer.encoder_attn.out_proj.bias + hidden_states_attn
            hidden_states = hidden_state_cross + decoder_layer.fc2(decoder_layer.activation_fn(decoder_layer.fc1(decoder_layer.final_layer_norm(hidden_state_cross))))
        hidden_states = self.decoder.layer_norm(hidden_states[:, -1])
        lm_logits = self.whisper.proj_out(hidden_states)
        if self.suppress_tokens is not None:
            lm_logits = lm_logits + self.suppress_tokens_penality
        max_logit_ids = torch.argmax(lm_logits, dim=-1, keepdim=True).int()
        return *self.save_de_keys, *self.save_de_values, max_logit_ids, kv_seq_len


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
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
    if MAX_SEQ_LEN > model.config.max_target_positions:
        MAX_SEQ_LEN = model.config.max_target_positions
        
    scaling = float(HEAD_DIM_EN ** -0.25)
    for i in model.model.encoder.layers._modules:
        model.model.encoder.layers._modules[i].self_attn.q_proj.weight.data *= scaling
        model.model.encoder.layers._modules[i].self_attn.q_proj.bias.data *= scaling
        model.model.encoder.layers._modules[i].self_attn.k_proj.bias.data *= scaling

        model.model.encoder.layers._modules[i].self_attn.q_proj.weight.data = model.model.encoder.layers._modules[i].self_attn.q_proj.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.encoder.layers._modules[i].self_attn.q_proj.bias.data = model.model.encoder.layers._modules[i].self_attn.q_proj.bias.data.view(NUM_HEAD_EN, 1, HEAD_DIM_EN).contiguous()
        model.model.encoder.layers._modules[i].self_attn.k_proj.weight.data = model.model.encoder.layers._modules[i].self_attn.k_proj.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.encoder.layers._modules[i].self_attn.v_proj.weight.data = model.model.encoder.layers._modules[i].self_attn.v_proj.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.encoder.layers._modules[i].self_attn.v_proj.bias.data = model.model.encoder.layers._modules[i].self_attn.v_proj.bias.data.view(NUM_HEAD_EN, 1, HEAD_DIM_EN).contiguous()
        model.model.encoder.layers._modules[i].self_attn.out_proj.weight.data = model.model.encoder.layers._modules[i].self_attn.out_proj.weight.data.view(HIDDEN_SIZE, NUM_HEAD_EN, HEAD_DIM_EN).permute(1, 2, 0).contiguous()
        model.model.encoder.layers._modules[i].self_attn.out_proj.bias.data = model.model.encoder.layers._modules[i].self_attn.out_proj.bias.data.view(1, 1, -1).contiguous()
        
    scaling = float(HEAD_DIM_DE ** -0.25)
    for i in model.model.decoder.layers._modules:
        model.model.decoder.layers._modules[i].self_attn.q_proj.weight.data *= scaling
        model.model.decoder.layers._modules[i].self_attn.q_proj.bias.data *= scaling
        model.model.decoder.layers._modules[i].self_attn.k_proj.weight.data *= scaling

        model.model.decoder.layers._modules[i].self_attn.q_proj.weight.data = model.model.decoder.layers._modules[i].self_attn.q_proj.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.decoder.layers._modules[i].self_attn.q_proj.bias.data = model.model.decoder.layers._modules[i].self_attn.q_proj.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.model.decoder.layers._modules[i].self_attn.k_proj.weight.data = model.model.decoder.layers._modules[i].self_attn.k_proj.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.decoder.layers._modules[i].self_attn.v_proj.weight.data = model.model.decoder.layers._modules[i].self_attn.v_proj.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.decoder.layers._modules[i].self_attn.v_proj.bias.data = model.model.decoder.layers._modules[i].self_attn.v_proj.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.model.decoder.layers._modules[i].self_attn.out_proj.weight.data = model.model.decoder.layers._modules[i].self_attn.out_proj.weight.data.view(HIDDEN_SIZE, NUM_HEAD_DE, HEAD_DIM_DE).permute(1, 2, 0).contiguous()
        model.model.decoder.layers._modules[i].self_attn.out_proj.bias.data = model.model.decoder.layers._modules[i].self_attn.out_proj.bias.data.view(1, 1, -1).contiguous()
        
    scaling = float(model.model.decoder.layers._modules['0'].encoder_attn.head_dim ** -0.25)
    for i in model.model.decoder.layers._modules:
        model.model.decoder.layers._modules[i].encoder_attn.q_proj.weight.data *= scaling
        model.model.decoder.layers._modules[i].encoder_attn.q_proj.bias.data *= scaling
        model.model.decoder.layers._modules[i].encoder_attn.k_proj.weight.data *= scaling
        
        model.model.decoder.layers._modules[i].encoder_attn.q_proj.weight.data = model.model.decoder.layers._modules[i].encoder_attn.q_proj.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.decoder.layers._modules[i].encoder_attn.q_proj.bias.data = model.model.decoder.layers._modules[i].encoder_attn.q_proj.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.model.decoder.layers._modules[i].encoder_attn.k_proj.weight.data = model.model.decoder.layers._modules[i].encoder_attn.k_proj.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.decoder.layers._modules[i].encoder_attn.v_proj.weight.data = model.model.decoder.layers._modules[i].encoder_attn.v_proj.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.model.decoder.layers._modules[i].encoder_attn.v_proj.bias.data = model.model.decoder.layers._modules[i].encoder_attn.v_proj.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.model.decoder.layers._modules[i].encoder_attn.out_proj.weight.data = model.model.decoder.layers._modules[i].encoder_attn.out_proj.weight.data.view(HIDDEN_SIZE, NUM_HEAD_EN, HEAD_DIM_DE).permute(1, 2, 0).contiguous()
        model.model.decoder.layers._modules[i].encoder_attn.out_proj.bias.data = model.model.decoder.layers._modules[i].encoder_attn.out_proj.bias.data.view(1, 1, -1).contiguous()
        
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    whisper_encoder = WHISPER_ENCODER(model.model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE, NUM_HEAD_EN, NUM_HEAD_DE)

    output_names = []
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    dynamic_axes = {'audio': {2: 'audio_len'}}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_EN):
            name = f'en_key_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_EN):
            name = f'en_value_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {1: 'signal_len'}
      
    torch.onnx.export(
        whisper_encoder,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del whisper_encoder
    del audio
    del custom_stft
    del name
    del output_names
    del dynamic_axes
    gc.collect()

    if is_v3:
        try:
            generation_config = GenerationConfig.from_pretrained(model_path)
            suppress_tokens = torch.tensor(generation_config.suppress_tokens, dtype=torch.int64)
        except:
            suppress_tokens = None
    else:
        suppress_tokens = None
        
    whisper_decoder = WHISPER_DECODER(model, MAX_SEQ_LEN, suppress_tokens, NUM_LAYER_DE, NUM_HEAD_DE)
    input_ids = torch.ones([1, 3], dtype=torch.int32)
    ids_len = torch.tensor([input_ids.shape[-1]], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    save_encoder_key = torch.zeros((1, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float32)
    save_encoder_value = torch.zeros((1, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float32)
    past_key_de = torch.zeros((1, HEAD_DIM_DE, 0), dtype=torch.float32)
    past_value_de = torch.zeros((1, 0, HEAD_DIM_DE), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}

    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_de_key_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_key_de)
            dynamic_axes[name] = {2: 'history_len'}
            name = f'out_de_key_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_de_value_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_value_de)
            dynamic_axes[name] = {1: 'history_len'}
            name = f'out_de_value_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {1: 'history_len_plus_ids_len'}

    input_names.append('input_ids')
    all_inputs.append(input_ids)
    input_names.append('history_len')
    all_inputs.append(history_len)

    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_EN):
            name = f'en_key_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(save_encoder_key)
            dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_EN):
            name = f'en_value_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(save_encoder_value)
            dynamic_axes[name] = {1: 'signal_len'}

    input_names.append('ids_len')
    all_inputs.append(ids_len)
    all_inputs.append(attention_mask)
    input_names.append('attention_mask')
    output_names.append('max_logit_id')
    output_names.append('kv_seq_len')

    torch.onnx.export(
        whisper_decoder,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model
    del whisper_decoder
    del input_ids
    del ids_len
    del history_len
    del save_encoder_key
    del save_encoder_value
    del past_key_de
    del past_value_de
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes
print('\nExport done!\n\nStart to run Whisper by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4        # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
output_names_A = []
for i in range(len(out_name_A)):
    output_names_A.append(out_name_A[i].name)

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]

generate_limit = MAX_SEQ_LEN - 5  # 5 = length of initial input_ids
num_layers = (amount_of_outputs_B - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    if ("example/en" in test) or ("example/zh" in test) or ("example/ja" in test) or ("example/ko" in test):
        language = test.split("/")[-1].split(".")[0]
    else:
        language = TARGET_LANGUAGE
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalizer(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(240000, audio_len)  # You can adjust it.
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
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]

    # Start to run Whisper
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    start, STOP_TOKEN, task = get_task_id(TASK, is_v3, custom_vocab)
    input_ids = np.array([[start, get_language_id(language, custom_vocab), task]], dtype=np.int32)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), 'cpu', 0)
    ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, 'cpu', 0)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
    attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
    attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._outputs_meta[0].shape[1], 0), dtype=np.float32), 'cpu', 0)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, ort_session_B._outputs_meta[num_layers].shape[2]), dtype=np.float32), 'cpu', 0)
    layer_indices = np.arange(num_keys_values_plus_2, num_keys_values_plus_2 + num_keys_values, dtype=np.int32)
    input_feed_B = {
        in_name_B[num_keys_values]: input_ids,
        in_name_B[num_keys_values_plus_1]: history_len,
        in_name_B[-2]: ids_len,
        in_name_B[-1]: attention_mask_1,
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B

    num_decode = 0
    save_token = []
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start:slice_end], 'cpu', 0)})
        for i in range(num_keys_values):
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-2])[0, 0]
            if max_logit_ids in STOP_TOKEN:
                break
            for i in range(amount_of_outputs_B):
                input_feed_B[in_name_B[i]] = all_outputs_B[i]
            if num_decode < 2:
                input_feed_B[in_name_B[-1]] = attention_mask_0
                input_feed_B[in_name_B[-2]] = ids_len_1
            save_token.append(max_logit_ids)
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    if num_decode > 0:
        save_token_array = remove_repeated_parts(save_token, 3, num_decode)  # To handle "over-talking".
        text, _ = tokenizer._decode_asr(
            [{
                "tokens": save_token_array
            }],
            return_timestamps=None,  # Do not support return timestamps
            return_language=None,
            time_precision=0
        )
        print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {(num_decode + 1) / count_time:.3f} tokens/s")
        print("----------------------------------------------------------------------------------------------------------")
