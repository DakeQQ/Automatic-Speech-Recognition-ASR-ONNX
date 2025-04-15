import gc
import math
import time
import site
import shutil
import onnxruntime
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/whisper-large-v3"                                            # The Whisper project download path.
onnx_model_A = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Encoder.onnx"                         # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Decoder.onnx"                         # The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/ja.mp3", "./example/ko.mp3"]     # The test audio list.


DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Whisper series models only support dynamic_axes due to their transformer structure.
INPUT_AUDIO_LENGTH = 160000                                 # The maximum input audio length.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
# N_MELS = 80                                               # Setting by whisper model config. Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 400                                             # Number of FFT components for the STFT process, edit it carefully.
NFFT_FBANK = 512                                            # Number of FFT components for the FBank process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
TARGET_LANGUAGE = "en"                                      # Choose a language listed in the get_language_id function's language_map.
TASK = 'transcribe'                                         # Choose one of : ['transcribe', 'translate']
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SEQ_LEN = 64                                            # It should less than 448.
STOP_TOKEN = [50257]                                        # 50257 is the end token for common Whisper series model.

if NFFT_FBANK < NFFT_STFT:
    NFFT_FBANK = NFFT_STFT
if NFFT_FBANK > INPUT_AUDIO_LENGTH:
    NFFT_FBANK = INPUT_AUDIO_LENGTH
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


shutil.copyfile('./modeling_modified/modeling_whisper.py', site.getsitepackages()[-1] + "/transformers/models/whisper/modeling_whisper.py")
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig


if "v3" in model_path or "V3" in model_path:
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


def remove_repeated_parts(ids, repeat_words_threshold):
    ids_len = len(ids)
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
    def __init__(self, whisper, stft_model, nfft_stft, nfft_fbank, stft_signal_len, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(WHISPER_ENCODER, self).__init__()
        self.encoder = whisper.encoder
        self.decoder = whisper.decoder
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_fbank // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, "slaney", 'slaney')).transpose(0, 1).unsqueeze(0)
        self.padding = torch.zeros((1, (nfft_fbank - nfft_stft) // 2, stft_signal_len), dtype=torch.int8)
        self.save_encoder_key = [None] * num_layers_de
        self.save_encoder_value = [None] * num_layers_de
        self.inv_int16 = float(1.0 / 32768.0)


    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        power = real_part * real_part + imag_part * imag_part
        if self.padding.shape[1] != 0:
            power = torch.cat((power, self.padding[:, :, :power.shape[-1]].float()), dim=1)
        mel_features = torch.matmul(self.fbank, power).clamp(min=1e-5).log10()
        mel_features = torch.maximum(mel_features, mel_features.max() - 8.0)
        mel_features = mel_features * 0.25 + 1.0
        encoder_hidden_states = self.encoder(mel_features)
        for idx, decoder_layer in enumerate(self.decoder.layers):
            self.save_encoder_key[idx] = decoder_layer.encoder_attn._shape(decoder_layer.encoder_attn.k_proj(encoder_hidden_states), -1, 1).transpose(1, 2)
            self.save_encoder_value[idx] = decoder_layer.encoder_attn._shape(decoder_layer.encoder_attn.v_proj(encoder_hidden_states), -1, 1)
        return *self.save_encoder_key, *self.save_encoder_value


class WHISPER_DECODER(torch.nn.Module):
    def __init__(self, whisper, max_seq_len, suppress_tokens, num_layers_de):
        super(WHISPER_DECODER, self).__init__()
        self.whisper = whisper
        self.decoder = whisper.model.decoder
        self.suppress_tokens = suppress_tokens
        self.num_layers_de_2 = num_layers_de + num_layers_de
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def forward(self, *all_inputs):
        attention_mask = all_inputs[-1]
        input_ids = all_inputs[self.num_layers_de_2]
        ids_len = input_ids.shape[1].unsqueeze(0)
        history_len = all_inputs[0].shape[-1].unsqueeze(0)
        kv_seq_len = ids_len + history_len
        task_embeds = self.decoder.embed_tokens(input_ids) + self.decoder.embed_positions.weight[history_len:kv_seq_len]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * attention_mask).float()
        outputs = self.decoder(*(all_inputs + tuple([task_embeds, attention_mask])))
        lm_logits = self.whisper.proj_out(outputs[0][:, -1])
        if self.suppress_tokens is not None:
            lm_logits[:, self.suppress_tokens] = -65504.0
        indices = torch.argmax(lm_logits, dim=-1, keepdim=True).int()
        return outputs[1:], indices


print('\nExport start...\n')
with torch.inference_mode():
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=False).eval()
    except:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True, use_safetensors=True).eval()
    HIDDEN_SIZE = model.config.d_model
    NUM_HEAD_EN = model.model.config.encoder_attention_heads
    NUM_HEAD_DE = model.model.config.decoder_attention_heads
    HEAD_DIM_EN = HIDDEN_SIZE // NUM_HEAD_EN
    HEAD_DIM_DE = HIDDEN_SIZE // NUM_HEAD_DE
    NUM_LAYER_EN = model.config.encoder_layers
    NUM_LAYER_DE = model.config.decoder_layers
    N_MELS = model.config.num_mel_bins
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
    if MAX_SEQ_LEN > model.config.max_target_positions:
        MAX_SEQ_LEN = model.config.max_target_positions
        
    scaling = float(math.pow(model.model.encoder.layers._modules['0'].self_attn.head_dim, -0.5))
    for i in model.model.encoder.layers._modules:
        model.model.encoder.layers._modules[i].self_attn.q_proj.weight.data = model.model.encoder.layers._modules[i].self_attn.q_proj.weight.data * scaling
        model.model.encoder.layers._modules[i].self_attn.q_proj.bias.data = model.model.encoder.layers._modules[i].self_attn.q_proj.bias.data * scaling
    scaling = float(math.pow(model.model.decoder.layers._modules['0'].self_attn.head_dim, -0.5))
    for i in model.model.decoder.layers._modules:
        model.model.decoder.layers._modules[i].self_attn.q_proj.weight.data = model.model.decoder.layers._modules[i].self_attn.q_proj.weight.data * scaling
        model.model.decoder.layers._modules[i].self_attn.q_proj.bias.data = model.model.decoder.layers._modules[i].self_attn.q_proj.bias.data * scaling
    scaling = float(math.pow(model.model.decoder.layers._modules['0'].encoder_attn.head_dim, -0.5))
    for i in model.model.decoder.layers._modules:
        model.model.decoder.layers._modules[i].encoder_attn.q_proj.weight.data = model.model.decoder.layers._modules[i].encoder_attn.q_proj.weight.data * scaling
        model.model.decoder.layers._modules[i].encoder_attn.q_proj.bias.data = model.model.decoder.layers._modules[i].encoder_attn.q_proj.bias.data * scaling
    
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    whisper_encoder = WHISPER_ENCODER(model.model, custom_stft, NFFT_STFT, NFFT_FBANK, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)

    output_names = []
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    dynamic_axes = {'audio': {2: 'audio_len'}}
    for i in range(NUM_LAYER_EN):
        name = f'en_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_EN):
        name = f'en_value_{i}'
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
    whisper_decoder = WHISPER_DECODER(model, MAX_SEQ_LEN, suppress_tokens, NUM_LAYER_DE)
    input_ids = torch.tensor([[50258, get_language_id(TARGET_LANGUAGE), get_task_id(TASK, True)[0]]], dtype=torch.int32)
    save_encoder_key = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float32)
    save_encoder_value = torch.zeros((NUM_HEAD_EN, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float32)
    past_key_de = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float32)
    past_value_de = torch.zeros((NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)

    input_names = []
    keys_values = []
    output_names = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}

    for i in range(NUM_LAYER_DE):
        name = f'in_de_key_{i}'
        input_names.append(name)
        keys_values.append(past_key_de)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_de_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_value_{i}'
        input_names.append(name)
        keys_values.append(past_value_de)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_de_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'history_len_plus_ids_len'}

    input_names.append('input_ids')
    keys_values.append(input_ids)

    for i in range(NUM_LAYER_DE):
        name = f'en_key_{i}'
        input_names.append(name)
        keys_values.append(save_encoder_key)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_{i}'
        input_names.append(name)
        keys_values.append(save_encoder_value)
        dynamic_axes[name] = {1: 'signal_len'}

    input_names.append('attention_mask')
    output_names.append('max_logit_id')

    torch.onnx.export(
        whisper_decoder,
        tuple(keys_values + [attention_mask]),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del whisper_decoder
    del input_ids
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
session_opts.log_severity_level = 3         # error level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")


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
input_names_B = []
output_names_B = []
amount_of_outputs = len(out_name_B)
for i in range(len(in_name_B)):
    input_names_B.append(in_name_B[i].name)
for i in range(amount_of_outputs):
    output_names_B.append(out_name_B[i].name)

generate_limit = MAX_SEQ_LEN - 5  # 5 = length of inital input_ids
num_layers = (amount_of_outputs - 1) // 2
num_layers_2 = num_layers + num_layers
num_layers_4 = num_layers_2 + num_layers_2
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
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(160000, audio_len)  # You can adjust it.
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
        final_slice = audio[:, :, -pad_amount:]
        white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        white_noise = (np.sqrt(np.mean(audio * audio)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]

    # Start to run Whisper
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[50258, get_language_id(language), get_task_id(TASK, is_v3)[0]]], dtype=np.int32), 'cpu', 0)
    attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[0].shape[0], ort_session_B._inputs_meta[0].shape[1], 0), dtype=np.float32), 'cpu', 0)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[num_layers].shape[0], 0, ort_session_B._inputs_meta[num_layers].shape[2]), dtype=np.float32), 'cpu', 0)
    layer_indices = np.arange(num_layers_2, num_layers_4, dtype=np.int32) + 1
    input_feed_B = {
        in_name_B[-1].name: attention_mask,
        in_name_B[num_layers_2].name: input_ids
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i].name] = past_keys_B
    for i in range(num_layers, num_layers_2):
        input_feed_B[in_name_B[i].name] = past_values_B
    num_decode = 0
    save_token = []
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], 'cpu', 0)})
        for i in range(num_layers_2):
            input_feed_B[in_name_B[layer_indices[i]].name] = all_outputs_A[i]
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
            max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-1])[0][0]
            if max_logit_ids in STOP_TOKEN:
                break
            for i in range(amount_of_outputs):
                input_feed_B[in_name_B[i].name] = all_outputs_B[i]
            if num_decode < 1:
                input_feed_B[in_name_B[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
            num_decode += 1
            save_token.append(max_logit_ids)
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    save_token_array = remove_repeated_parts(save_token, 3)  # To handle "over-talking".
    text, _ = tokenizer._decode_asr(
        [{
            "tokens": save_token_array
        }],
        return_timestamps=None,                                                # Do not support return timestamps
        return_language=None,
        time_precision=0
    )
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")

