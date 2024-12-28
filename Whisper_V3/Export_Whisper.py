import gc
import time
import shutil
import numpy as np
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig

from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/whisper-large-v3"                                            # The Whisper project download path.
onnx_model_A = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Encoder.onnx"                         # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Whisper_ONNX/Whisper_Decoder.onnx"                         # The exported onnx model path.
modified_path = './modeling_modified/modeling_whisper.py'
transformers_whisper_path = "/home/DakeQQ/anaconda3/envs/python_312/lib/python3.12/site-packages/transformers/models/whisper/modeling_whisper.py"  # The original modeling_whisper.py path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/ja.mp3", "./example/ko.mp3"]   # The test audio list.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
provider_options = []
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Whisper series models only support dynamic_axes due to their transformer structure.
INPUT_AUDIO_LENGTH = 16000                                  # Just a dummy value here.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
# N_MELS = 80                                               # Setting by whisper model config. Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT = 512                                                  # Number of FFT components for the STFT process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
TARGET_LANGUAGE = "en"                                      # Choose a language listed in the get_language_id function's language_map.
TASK = 'transcribe'                                         # Choose one of : ['transcribe', 'translate']
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SEQ_LEN = 64                                            # It should less than 448.
STOP_TOKEN = 50257                                          # 50257 is the end token for common Whisper series model.

if NFFT > INPUT_AUDIO_LENGTH:
    INPUT_AUDIO_LENGTH = NFFT
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


shutil.copyfile(modified_path, transformers_whisper_path)


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


def get_task_id(task_input, use_v3):
    task_input = task_input.lower()
    if use_v3:
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
    def __init__(self, whisper, stft_model, nfft, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(WHISPER_ENCODER, self).__init__()
        self.encoder = whisper.encoder
        self.decoder = whisper.decoder
        self.stft_model = stft_model
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 20, 8000, n_mels, sample_rate, "slaney",'slaney')).transpose(0, 1).unsqueeze(0)
        self.save_encoder_key = [None] * num_layers_de
        self.save_encoder_value = [None] * num_layers_de
        self.inv_int16 = 1.0 / 32768.0

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).clamp(min=1e-5).log10()
        mel_features = torch.maximum(mel_features, mel_features.max() - 8.0)
        mel_features = (mel_features + 4.0) * 0.25
        encoder_hidden_states = self.encoder(mel_features)
        for idx, decoder_layer in enumerate(self.decoder.layers):
            self.save_encoder_key[idx] = decoder_layer.encoder_attn._shape(decoder_layer.encoder_attn.k_proj(encoder_hidden_states), -1, 1).half()
            self.save_encoder_value[idx] = decoder_layer.encoder_attn._shape(decoder_layer.encoder_attn.v_proj(encoder_hidden_states), -1, 1).half()
        save_encoder_key = torch.stack(self.save_encoder_key, dim=0)
        save_encoder_value = torch.stack(self.save_encoder_value, dim=0)
        return save_encoder_key.transpose(-1, -2), save_encoder_value


class WHISPER_DECODER(torch.nn.Module):
    def __init__(self, whisper, max_seq_len, suppress_tokens):
        super(WHISPER_DECODER, self).__init__()
        self.whisper = whisper
        self.decoder = whisper.model.decoder
        self.suppress_tokens = suppress_tokens
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8)))

    def forward(self, input_ids, save_encoder_key, save_encoder_value, past_key_de, past_value_de, ids_len, history_len, attention_mask):
        kv_seq_len = ids_len + history_len
        task_embeds = (self.decoder.embed_tokens(input_ids) + self.decoder.embed_positions.weight[history_len: kv_seq_len])
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len] * attention_mask
        hidden_states, past_key_de, past_value_de = self.decoder(hidden_states=task_embeds, attention_mask=attention_mask, past_key_de=past_key_de, past_value_de=past_value_de, past_key_en=save_encoder_key, past_value_en=save_encoder_value)
        lm_logits = self.whisper.proj_out(hidden_states[-1])
        lm_logits[self.suppress_tokens] = -65504.0
        indices = torch.argmax(lm_logits, dim=-1)
        return indices.int(), past_key_de, past_value_de


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
    NUM_LAYER_DE = model.config.decoder_layers
    N_MELS = model.config.num_mel_bins
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
    if MAX_SEQ_LEN > model.config.max_target_positions:
        MAX_SEQ_LEN = model.config.max_target_positions
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    whisper_encoder = WHISPER_ENCODER(model.model, custom_stft, NFFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        whisper_encoder,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=['save_encoder_key', 'save_encoder_value'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'save_encoder_key': {3: 'signal_len'},
            'save_encoder_value': {2: 'signal_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del whisper_encoder
    del audio
    del custom_stft
    gc.collect()

    generation_config = GenerationConfig.from_pretrained(model_path)
    suppress_tokens = torch.tensor(generation_config.suppress_tokens, dtype=torch.int64)
    whisper_decoder = WHISPER_DECODER(model, MAX_SEQ_LEN, suppress_tokens)
    input_ids = torch.tensor([50258, get_language_id(TARGET_LANGUAGE), get_task_id(TASK, True)[0]], dtype=torch.int32)
    save_encoder_key = torch.zeros((NUM_LAYER_DE, NUM_HEAD_EN, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float16)
    save_encoder_value = torch.zeros((NUM_LAYER_DE, NUM_HEAD_EN, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float16)
    past_key_de = torch.zeros((NUM_LAYER_DE, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float16)
    past_value_de = torch.zeros((NUM_LAYER_DE, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float16)
    ids_len = torch.tensor([input_ids.shape[0]], dtype=torch.int64)
    history_len = torch.tensor([past_key_de.shape[-2]], dtype=torch.int64)
    attention_mask = torch.tensor([-65504.0], dtype=torch.float32)
    torch.onnx.export(
        whisper_decoder,
        (input_ids, save_encoder_key, save_encoder_value, past_key_de, past_value_de, ids_len, history_len, attention_mask),
        onnx_model_B,
        input_names=['input_ids', 'save_encoder_key', 'save_encoder_value', 'past_key_de_in', 'past_value_de_in', 'ids_len', 'history_len', 'attention_mask'],
        output_names=['token_ids', 'past_key_de_out', 'past_value_de_out'],
        do_constant_folding=True,
        dynamic_axes={
            'input_ids': {0: 'ids_len'},
            'save_encoder_key': {3: 'signal_len'},
            'save_encoder_value': {2: 'signal_len'},
            'past_key_de_in': {2: 'history_len'},
            'past_value_de_in': {2: 'history_len'},
            'past_key_de_out': {2: 'history_len'},
            'past_value_de_out': {2: 'history_len'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del model
    del whisper_decoder
    del input_ids
    del save_encoder_key
    del save_encoder_value
    del past_key_de
    del past_value_de
    del ids_len
    del history_len
    del attention_mask
print('\nExport done!\n\nStart to run Whisper by ONNX Runtime.\n\nNow, loading the model...')


# # ONNX Runtime settings
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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options)
model_type = ort_session_B._inputs_meta[-1].type
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
in_name_B2 = in_name_B[2].name
in_name_B3 = in_name_B[3].name
in_name_B4 = in_name_B[4].name
in_name_B5 = in_name_B[5].name
in_name_B6 = in_name_B[6].name
in_name_B7 = in_name_B[7].name
out_name_B0 = out_name_B[0].name
out_name_B1 = out_name_B[1].name
out_name_B2 = out_name_B[2].name

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    if ("example/en" in test) or ("example/zh" in test) or ("example/ja" in test) or ("example/ko" in test):
        language = test.split("/")[-1].split(".")[0]
    else:
        language = TARGET_LANGUAGE
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
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
    input_ids = np.array([50258, get_language_id(language), get_task_id(TASK, True)[0]], dtype=np.int32)
    ids_len = np.array([input_ids.shape[0]], dtype=np.int64)
    history_len = np.array([0], dtype=np.int64)
    past_key_de = np.zeros((ort_session_B._inputs_meta[3].shape[0], ort_session_B._inputs_meta[3].shape[1], history_len[0], ort_session_B._inputs_meta[3].shape[-1]), dtype=np.float16)
    past_value_de = np.zeros((ort_session_B._inputs_meta[4].shape[0], ort_session_B._inputs_meta[4].shape[1], history_len[0], ort_session_B._inputs_meta[4].shape[-1]), dtype=np.float16)
    if 'float16' in model_type:
        attention_mask = np.array([-65504.0], dtype=np.float16)
    else:
        attention_mask = np.array([-65504.0], dtype=np.float32)
    num_decode = 0
    save_token = []
    start_time = time.time()
    while slice_end <= aligned_len:
        save_encoder_key, save_encoder_value = ort_session_A.run([out_name_A0, out_name_A1], {in_name_A0: audio[:, :, slice_start: slice_end]})
        while history_len < MAX_SEQ_LEN:
            input_ids, past_key_de, past_value_de = ort_session_B.run([out_name_B0, out_name_B1, out_name_B2], {
                in_name_B0: input_ids,
                in_name_B1: save_encoder_key,
                in_name_B2: save_encoder_value,
                in_name_B3: past_key_de,
                in_name_B4: past_value_de,
                in_name_B5: ids_len,
                in_name_B6: history_len,
                in_name_B7: attention_mask
            })
            if STOP_TOKEN in input_ids:
                break
            if num_decode < 1:
                history_len += ids_len
                ids_len[0] = 1
                attention_mask[0] = 0
            else:
                history_len += 1
            num_decode += 1
            save_token.append(input_ids)
            input_ids = np.array([input_ids], dtype=np.int32)
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    save_token_array = remove_repeated_parts(save_token, 3)  # To handle "over-talking".
    text, _ = tokenizer._decode_asr(
        [{
            "tokens": save_token_array
        }],
        return_timestamps=None,  # Do not support return timestamps
        return_language=None,
        time_precision=None
    )
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")
