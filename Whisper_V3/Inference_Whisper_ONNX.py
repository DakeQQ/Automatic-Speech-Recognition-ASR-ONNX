import time
import numpy as np
import onnxruntime
from pydub import AudioSegment
from transformers import AutoTokenizer


download_path = "/home/DakeQQ/Downloads/whisper-large-v3"                                         # The whisper model download path.
onnx_model_A = "/home/DakeQQ/Downloads/Whisper_Optimized/Whisper_Encoder.ort"                     # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Whisper_Optimized/Whisper_Decoder.ort"                     # The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/ja.mp3", "./example/ko.mp3"]     # The test audio list.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
provider_options = []
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
TARGET_LANGUAGE = "en"                  # Choose a language listed in the get_language_id function's language_map.
TASK = 'transcribe'                     # Choose one of : ['transcribe', 'translate']
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SEQ_LEN = 64                        # It should keep the same with exported model.
STOP_TOKEN = 50257                      # 50257 is the end token for common Whisper series model.


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


tokenizer = AutoTokenizer.from_pretrained(download_path)


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
        time_precision=0
    )
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")
