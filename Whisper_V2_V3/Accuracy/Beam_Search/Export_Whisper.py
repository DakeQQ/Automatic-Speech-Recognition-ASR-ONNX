import gc
import time
import onnxruntime
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from STFT_Process import STFT_Process                                             # The custom STFT/ISTFT can be exported in ONNX format.
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig


model_path = "/home/DakeQQ/Downloads/whisper-large-v3-turbo"                      # The Whisper project download path.
onnx_model_A = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Encoder.onnx"         # Assign a path where the exported onnx model stored.
onnx_model_B = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Decoder.onnx"         # Assign a path where the exported onnx model stored.
onnx_model_C = '/home/DakeQQ/Downloads/Whisper_ONNX/Greedy_Search.onnx'           # Assign a path where the exported onnx model stored.
onnx_model_D = '/home/DakeQQ/Downloads/Whisper_ONNX/First_Beam_Search.onnx'       # Assign a path where the exported onnx model stored.
onnx_model_E = '/home/DakeQQ/Downloads/Whisper_ONNX/Second_Beam_Search.onnx'      # Assign a path where the exported onnx model stored.
onnx_model_F = '/home/DakeQQ/Downloads/Whisper_ONNX/Reset_Penality.onnx'          # Assign a path where the exported onnx model stored.


if "3.5" in model_path:
    test_audio = ["../example/en.mp3"]                                                                    # The test audio list.
else:
    test_audio = ["../example/zh.mp3", "../example/en.mp3", "../example/ja.mp3", "../example/ko.mp3"]     # The test audio list.


USE_BEAM_SEARCH = True                                      # Use beam search or greedy search.
INPUT_AUDIO_LENGTH = 240000                                 # The maximum input audio length.
MAX_SEQ_LEN = 72                                            # It should less than 448.
REPEAT_PENALITY = 0.9                                       # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                                         # Penalizes the most recent output. "30" means the last 30 tokens.
MAX_BEAM_SIZE = 8                                           # Max beams for exported model.
TOP_K = 3                                                   # The top k candidate in decoding.
BEAM_SIZE = 3                                               # Number of beams in searching.
TARGET_LANGUAGE = "en"                                      # Choose a language listed in the get_language_id function's language_map.
TASK = 'transcribe'                                         # Choose one of : ['transcribe', 'translate']
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WINDOW_TYPE = 'hann'                                        # Type of window function used in the STFT
# N_MELS = 80                                               # Setting by whisper model config. Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
STOP_TOKEN = [50257]                                        # 50257 is the end token for common Whisper series model.


if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


model_path_lower = model_path.lower()
if ("v3" in model_path_lower) or ("crisperwhisper" in model_path_lower) or ("anime" in model_path_lower) or ("belle" in model_path_lower) or ("turbo" in model_path_lower) or ("distil" in model_path_lower):
    is_v3 = True
    print("\nExport the Whisper-V3")
else:
    is_v3 = False
    print("\nExport the Whisper-V2")


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)
  

def get_language_id(language_input):
    # Define the dictionary mapping language tags to their IDs
    language_map = {
        'af': 50327, 'am': 50334, 'ar': 50272, 'as': 50350, 'az': 50304,
        'ba': 50355, 'be': 50330, 'bg': 50292, 'bn': 50302, 'bo': 50347,
        'br': 50309, 'bs': 50315, 'ca': 50270, 'cs': 50283, 'cy': 50297,
        'da': 50285, 'de': 50261, 'el': 50281, 'en': 50259, 'es': 50262,
        'et': 50307, 'eu': 50310, 'fa': 50300, 'fi': 50277, 'fo': 50338,
        'fr': 50265, 'gl': 50319, 'gu': 50333, 'haw': 50352, 'ha': 50354,
        'he': 50279, 'hi': 50276, 'hr': 50291, 'ht': 50339, 'hu': 50286,
        'hy': 50312, 'id': 50275, 'is': 50311, 'it': 50274, 'ja': 50266,
        'jw': 50356, 'ka': 50329, 'kk': 50316, 'km': 50323, 'kn': 50306,
        'ko': 50264, 'la': 50294, 'lb': 50345, 'ln': 50353, 'lo': 50336,
        'lt': 50293, 'lv': 50301, 'mg': 50349, 'mi': 50295, 'mk': 50308,
        'ml': 50296, 'mn': 50314, 'mr': 50320, 'ms': 50282, 'mt': 50343,
        'my': 50346, 'ne': 50313, 'nl': 50271, 'nn': 50342, 'no': 50288,
        'oc': 50328, 'pa': 50321, 'pl': 50269, 'ps': 50340, 'pt': 50267,
        'ro': 50284, 'ru': 50263, 'sa': 50344, 'sd': 50332, 'si': 50322,
        'sk': 50298, 'sl': 50305, 'sn': 50324, 'so': 50326, 'sq': 50317,
        'sr': 50303, 'su': 50357, 'sv': 50273, 'sw': 50318, 'ta': 50287,
        'te': 50299, 'tg': 50331, 'th': 50289, 'tk': 50341, 'tl': 50348,
        'tr': 50268, 'tt': 50351, 'uk': 50280, 'ur': 50290, 'uz': 50337,
        'vi': 50278, 'yi': 50335, 'yo': 50325, 'zh': 50260
    }

    # Normalize the input to lowercase
    language_input = language_input.lower()

    # Handle special cases for full language names
    full_language_names = {
        'afrikaans':       'af',  'amharic':        'am',  'arabic':         'ar',  'assamese':       'as',
        'azerbaijani':     'az',  'bashkir':        'ba',  'belarusian':     'be',  'bulgarian':      'bg',
        'bengali':         'bn',  'tibetan':        'bo',  'breton':         'br',  'bosnian':        'bs',
        'catalan':         'ca',  'czech':          'cs',  'welsh':          'cy',  'danish':         'da',
        'german':          'de',  'greek':          'el',  'english':        'en',  'spanish':        'es',
        'estonian':        'et',  'basque':         'eu',  'persian':        'fa',  'finnish':        'fi',
        'faroese':         'fo',  'french':         'fr',  'galician':       'gl',  'gujarati':       'gu',
        'hawaiian':        'haw', 'hausa':          'ha',  'hebrew':         'he',  'hindi':          'hi',
        'croatian':        'hr',  'haitian creole': 'ht',  'hungarian':      'hu',  'armenian':       'hy',
        'indonesian':      'id',  'icelandic':      'is',  'italian':        'it',  'japanese':       'ja',
        'javanese':       'jw',   'georgian':       'ka',  'kazakh':         'kk',  'khmer':          'km',
        'kannada':        'kn',   'korean':         'ko',  'latin':          'la',  'luxembourgish':  'lb',
        'lingala':        'ln',   'lao':            'lo',  'lithuanian':     'lt',  'latvian':        'lv',
        'malagasy':       'mg',   'maori':          'mi',  'macedonian':     'mk',  'malayalam':      'ml',
        'mongolian':      'mn',   'marathi':        'mr',  'malay':          'ms',  'maltese':        'mt',
        'burmese':        'my',   'nepali':         'ne',  'dutch':          'nl',  'nynorsk':        'nn',
        'norwegian':      'no',   'occitan':        'oc',  'punjabi':        'pa',  'polish':         'pl',
        'pashto':         'ps',   'portuguese':     'pt',  'romanian':       'ro',  'russian':        'ru',
        'sanskrit':       'sa',   'sindhi':         'sd',  'sinhala':        'si',  'slovak':         'sk',
        'slovenian':      'sl',   'shona':          'sn',  'somali':         'so',  'albanian':       'sq',
        'serbian':        'sr',   'sundanese':      'su',  'swedish':        'sv',  'swahili':        'sw',
        'tamil':          'ta',   'telugu':         'te',  'tajik':          'tg',  'thai':           'th',
        'turkmen':        'tk',   'tagalog':        'tl',  'turkish':        'tr',  'tatar':          'tt',
        'ukrainian':      'uk',   'urdu':           'ur',  'uzbek':          'uz',  'vietnamese':     'vi',
        'yiddish':        'yi',   'yoruba':         'yo',  'chinese':        'zh'
    }

    # Check if the input is a full language name and convert to code
    if language_input in full_language_names:
        language_input = full_language_names[language_input]

    # Return the corresponding ID or None if not found
    return language_map.get(language_input)


def get_task_id(task_input, is_v3):
    task_input = task_input.lower()
    if is_v3:
        task_map = {
            'translate': 50359,
            'transcribe':  50360
        }
        return task_map[task_input], 50363, 50364
    else:
        task_map = {
            'translate': 50358,
            'transcribe': 50359
        }
        return task_map[task_input], 50362, 50363


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


class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, logits, repeat_penality, penality_value, batch_size):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        batch_indices = self.batch_indices[:batch_size].long()
        repeat_penality[batch_indices, max_logits_idx.squeeze(-1)] *= penality_value
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers, num_head_de):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = (num_layers + num_layers) * num_head_de
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers, num_head_de):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = (num_layers + num_layers) * num_head_de
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-8]
        save_id = all_inputs[-7]
        repeat_penality = all_inputs[-6]
        previous_prob = all_inputs[-5]
        batch_indices = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=False)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[[0]]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count, batch_indices):
        repeat_penality[batch_indices, save_id[batch_indices, penality_reset_count[batch_indices]]] = 1.0
        penality_reset_count += 1
        return save_id, repeat_penality, penality_reset_count


class WHISPER_ENCODER(torch.nn.Module):
    def __init__(self, whisper, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, num_layers_de, num_head_en, num_head_de):
        super(WHISPER_ENCODER, self).__init__()
        self.encoder = whisper.encoder
        self.decoder = whisper.decoder
        self.stft_model = stft_model
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, "slaney", 'slaney')).transpose(0, 1).unsqueeze(0)
        self.save_encoder_key = [None] * num_layers_de * num_head_de
        self.save_encoder_value = [None] * num_layers_de * num_head_de
        self.inv_int16 = float(1.0 / 32768.0)
        self.pre_emphasis = float(pre_emphasis)
        self.num_head_en = num_head_en
        self.num_head_de = num_head_de

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-5).log10()
        mel_features = torch.maximum(mel_features, mel_features.max() - 8.0)
        mel_features = mel_features * 0.25 + 1.0
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
        logits = self.whisper.proj_out(hidden_states)
        if self.suppress_tokens is not None:
            logits = logits + self.suppress_tokens_penality
        return *self.save_de_keys, *self.save_de_values, logits, kv_seq_len


print('\nExport start...\n')
with torch.inference_mode():
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=False).eval()
    except:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True).eval()
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
        
    scaling = float(HEAD_DIM_EN ** -0.25)
    for i in model.model.encoder.layers._modules:
        model.model.encoder.layers._modules[i].self_attn.q_proj.weight.data *= scaling
        model.model.encoder.layers._modules[i].self_attn.q_proj.bias.data *= scaling
        model.model.encoder.layers._modules[i].self_attn.k_proj.weight.data *= scaling

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
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del whisper_encoder
    del audio
    del custom_stft
    del name
    del output_names
    del dynamic_axes
    gc.collect()

    if is_v3:
        generation_config = GenerationConfig.from_pretrained(model_path)
        suppress_tokens = torch.tensor(generation_config.suppress_tokens, dtype=torch.int64)
    else:
        suppress_tokens = None
    whisper_decoder = WHISPER_DECODER(model, MAX_SEQ_LEN, suppress_tokens, NUM_LAYER_DE, NUM_HEAD_DE)
    input_ids = torch.ones((3, 1), dtype=torch.int32)  # '3' is a dummy value
    ids_len = torch.tensor([input_ids.shape[-1]], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    save_encoder_key = torch.zeros((1, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float32)
    save_encoder_value = torch.zeros((1, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float32)
    batch_size = input_ids.shape[0]
    past_key_de = torch.zeros((batch_size, HEAD_DIM_DE, 0), dtype=torch.float32)
    past_value_de = torch.zeros((batch_size, 0, HEAD_DIM_DE), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {'input_ids': {0: 'batch', 1: 'ids_len'}}

    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_de_key_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_key_de)
            dynamic_axes[name] = {0: 'batch', 2: 'history_len'}
            name = f'out_de_key_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 2: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_de_value_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_value_de)
            dynamic_axes[name] = {0: 'batch', 1: 'history_len'}
            name = f'out_de_value_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 1: 'history_len_plus_ids_len'}

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
    output_names.append('logits')
    dynamic_axes['logits'] = {0: 'batch'}
    output_names.append('kv_seq_len')

    torch.onnx.export(
        whisper_decoder,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del whisper_decoder
    del input_ids
    del ids_len
    del history_len
    del save_encoder_key
    del save_encoder_value
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes

    greedy = GREEDY_SEARCH()
    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, VOCAB_SIZE), dtype=torch.float32)
    penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
    logits = torch.ones((beam_size, VOCAB_SIZE), dtype=torch.float32)
    penality_value = torch.tensor(REPEAT_PENALITY, dtype=torch.float32)
    batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

    torch.onnx.export(
        greedy,
        (logits, repeat_penality, penality_value, beam_size),  # Reuse the beam_size tensor as batch_size during export process.
        onnx_model_C,
        input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
        output_names=['max_logits_idx', 'repeat_penality_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del greedy

    first_beam_search = FIRST_BEAM_SEARCH(NUM_LAYER_DE, NUM_HEAD_DE)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
    previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
    past_keys_greedy = past_key_de[[0]]
    past_values_greedy = past_value_de[[0]]

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_key_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_keys_greedy)
            dynamic_axes[name] = {0: 'batch', 2: 'history_len'}
            name = f'out_key_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 2: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_value_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_values_greedy)
            dynamic_axes[name] = {0: 'batch', 1: 'history_len'}
            name = f'out_value_layer_{i}_head_{j}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 1: 'history_len_plus_ids_len'}
    input_names.append('logits')
    all_inputs.append(logits[[0]])
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('top_beam_indices')
    output_names.append('save_id_out')
    output_names.append('repeat_penality_out')
    output_names.append('batch_indices')
    output_names.append('top_beam_prob')
    output_names.append('max_logits_idx')
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['repeat_penality_in'] = {0: 'batch'}
    dynamic_axes['repeat_penality_out'] = {0: 'batch'}
    dynamic_axes['logits'] = {0: 'batch'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}
    dynamic_axes['batch_indices'] = {0: 'batch'}

    torch.onnx.export(
        first_beam_search,
        tuple(all_inputs),
        onnx_model_D,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )

    all_inputs = []
    input_names = []
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_key_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_key_de)
    for i in range(NUM_LAYER_DE):
        for j in range(NUM_HEAD_DE):
            name = f'in_value_layer_{i}_head_{j}'
            input_names.append(name)
            all_inputs.append(past_value_de)
    input_names.append('logits')
    all_inputs.append(logits)
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('batch_indices')
    all_inputs.append(batch_indices)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}
    output_names.remove("batch_indices")

    second_beam_search = SECOND_BEAM_SEARCH(NUM_LAYER_DE, NUM_HEAD_DE)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )

    reset_penality = RESET_PENALITY()
    torch.onnx.export(
        reset_penality,
        (save_id, repeat_penality, penality_reset_count, batch_indices),
        onnx_model_F,
        input_names=['save_id_in', 'repeat_penality_in', 'penality_reset_count_in', 'batch_indices'],
        output_names=['save_id_out', 'repeat_penality_out', 'penality_reset_count_out'],
        dynamic_axes={
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'penality_reset_count_in': {0: 'batch'},
            'penality_reset_count_out': {0: 'batch'},
            'batch_indices': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )

    del first_beam_search
    del reset_penality
    del second_beam_search
    del batch_indices
    del past_keys_greedy
    del past_values_greedy
    del logits
    del previous_prob
    del save_id
    del repeat_penality
    del penality_reset_count
    del topK
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    gc.collect()

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
out_name_A = [out_name_A[i].name for i in range(len(out_name_A))]

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
vocab_size = ort_session_B._outputs_meta[num_keys_values].shape[-1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), 'cpu', 0)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), 'cpu', 0)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=np.float32), 'cpu', 0)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]
    
    input_feed_D = {
        in_name_D[-2]: penality_value,
        in_name_D[-1]: beam_size
    }
    input_feed_E = {
        in_name_E[-3]: penality_value,
        in_name_E[-2]: beam_size,
        in_name_E[-1]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
    out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
    input_feed_C = {in_name_C[2]: penality_value}

if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
    if USE_BEAM_SEARCH:
        penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), 'cpu', 0)
    else:
        save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
else:
    do_repeat_penality = False


# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    if ("example/en" in test) or ("example/zh" in test) or ("example/ja" in test) or ("example/ko" in test):
        language = test.split("/")[-1].split(".")[0]
    else:
        language = TARGET_LANGUAGE
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
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
    input_ids = np.array([[50258, get_language_id(language), get_task_id(TASK, is_v3)[0]]], dtype=np.int32)
    batch_size = input_ids.shape[0]
    repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=np.float32), 'cpu', 0)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), 'cpu', 0)
    ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, 'cpu', 0)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
    attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
    attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, ort_session_B._outputs_meta[0].shape[1], 0), dtype=np.float32), 'cpu', 0)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, 0, ort_session_B._outputs_meta[num_layers].shape[2]), dtype=np.float32), 'cpu', 0)
    layer_indices = np.arange(num_keys_values_plus_2, num_keys_values_plus_2 + num_keys_values, dtype=np.int32)

    input_feed_B = {
        in_name_B[num_keys_values]: input_ids,
        in_name_B[num_keys_values_plus_1]: history_len,
        in_name_B[-2]: ids_len,
        in_name_B[-1]: attention_mask_1
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i]] = past_keys_B
    for i in range(num_layers, num_keys_values):
        input_feed_B[in_name_B[i]] = past_values_B

    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), 'cpu', 0)
        input_feed_D[in_name_D[num_keys_values_plus_1]] = save_id_beam
        input_feed_D[in_name_D[num_keys_values_plus_2]] = repeat_penality
    else:
        batch_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([batch_size], dtype=np.int64), 'cpu', 0)
        input_feed_C[in_name_C[1]] = repeat_penality
        input_feed_C[in_name_C[3]] = batch_size

    if do_repeat_penality:
        if USE_BEAM_SEARCH:
            input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
        else:
            penality_reset_count_greedy = 0

    num_decode = 0
    save_token = []
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], 'cpu', 0)})
        for i in range(num_keys_values):
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    for i in range(num_keys_values_plus_1):
                        input_feed_D[in_name_D[i]] = all_outputs_B[i]
                    all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_D[-1])
                    input_feed_E[in_name_E[-4]] = all_outputs_D[-2]
                    if do_repeat_penality:
                        input_feed_F[in_name_F[3]] = all_outputs_D[-2]
                else:
                    for i in range(num_keys_values_plus_1):
                        input_feed_E[in_name_E[i]] = all_outputs_B[i]
                    all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_E[-1])
                if max_logits_idx in STOP_TOKEN:
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_F[in_name_F[1]] = all_outputs_E[num_keys_values_plus_2]
                    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                    input_feed_F[in_name_F[2]] = all_outputs_F[2]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_F[0]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_F[1]
                if num_decode < 1:
                    for i in range(num_keys_values_plus_1):
                        input_feed_B[in_name_B[i]] = all_outputs_D[i]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_D[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_D[num_keys_values_plus_3]
                else:
                    for i in range(num_keys_values_plus_1):
                        input_feed_B[in_name_B[i]] = all_outputs_E[i]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
            else:
                input_feed_C[in_name_C[0]] = all_outputs_B[num_keys_values]
                all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
                max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_C[0])[0, 0]
                if max_logits_idx in STOP_TOKEN:
                    break
                input_feed_B[in_name_B[num_keys_values]] = all_outputs_C[0]
                if do_repeat_penality and (num_decode >= PENALITY_RANGE) and (save_id_greedy[penality_reset_count_greedy] != max_logits_idx):
                    repeat_penality = onnxruntime.OrtValue.numpy(all_outputs_C[1])
                    repeat_penality[..., penality_reset_count_greedy] = 1.0
                    penality_reset_count_greedy += 1
                input_feed_C[in_name_C[1]] = all_outputs_C[1]
                save_id_greedy[num_decode] = max_logits_idx
                for i in range(num_keys_values):
                    input_feed_B[in_name_B[i]] = all_outputs_B[i]
            input_feed_B[in_name_B[num_keys_values_plus_1]] = all_outputs_B[num_keys_values_plus_1]
            if num_decode < 1:
                input_feed_B[in_name_B[-1]] = attention_mask_0
                input_feed_B[in_name_B[-2]] = ids_len_1
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    if num_decode == generate_limit:
        num_decode -= 1
    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.numpy(all_outputs_E[num_keys_values_plus_1])[0]
        for i in range(num_decode, -1, -1):
            if save_id_beam[i] not in STOP_TOKEN:
                save_id_beam = save_id_beam[:i + 1]
                break
        save_token_array = remove_repeated_parts(save_id_beam, 3, save_id_beam.shape[-1])       # To handle "over-talking".
    else:
        save_token_array = remove_repeated_parts(save_id_greedy[:num_decode], 3, num_decode)    # To handle "over-talking".
    text, _ = tokenizer._decode_asr(
        [{
            "tokens": save_token_array.reshape(1, -1)
        }],
        return_timestamps=None,  # Do not support return timestamps
        return_language=None,
        time_precision=0
    )
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")
