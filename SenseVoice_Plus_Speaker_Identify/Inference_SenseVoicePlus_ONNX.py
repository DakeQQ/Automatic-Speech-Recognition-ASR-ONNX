import time
import numpy as np
import onnxruntime
from pydub import AudioSegment
from sentencepiece import SentencePieceProcessor


tokenizer_path = "/home/DakeQQ/Downloads/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model"     # The SenseVoice download path.
onnx_model_A = "/home/DakeQQ/Downloads/SenseVoice_Optimized/SenseVoiceSmallPlus.ort"               # The exported onnx model path.
test_audio = "./test_sample.wav"                                                                   # The test audio path.


ORT_Accelerate_Providers = []           # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                        # else keep empty.
SAMPLE_RATE = 16000                     # The model parameter, do not edit the value.
TARGET_LANGUAGE = 2                     # Choose one of indices ['auto' = 0, 'zh' = 1, 'en' = 2, 'yue' = 3, 'ja' = 4, 'ko' = 5, 'nospeech' = 6]
SLIDING_WINDOW = 0                      # Set the sliding window step for test audio reading; use 0 to disable.
SIMILARITY_THRESHOLD = 0.5              # Threshold to determine the speaker's identity. You can adjust it.


tokenizer = SentencePieceProcessor()
tokenizer.Load(tokenizer_path)


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


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
model_type = ort_session_A._inputs_meta[2].type
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
in_name_A2 = in_name_A[2].name
in_name_A3 = in_name_A[3].name
if isinstance(shape_value_in, int):
    in_name_A4 = in_name_A[4].name
    dynamic_axes = False
else:
    in_name_A4 = None
    dynamic_axes = True
out_name_A0 = out_name_A[0].name
out_name_A1 = out_name_A[1].name
out_name_A2 = out_name_A[2].name
out_name_A3 = out_name_A[3].name
out_name_A4 = out_name_A[4].name

num_speakers = np.array([1], dtype=np.int64)  # At least 1.
if dynamic_axes:
    saved_embed = np.zeros((ort_session_A._inputs_meta[2].shape[0], 2), dtype=np.float32)  # At least 2.
    saved_dot = np.ones((1, 2), dtype=np.float32)
    empty_embed = np.zeros((ort_session_A._inputs_meta[2].shape[0], 1), dtype=np.float32)
    empty_dot = np.ones((1, 1), dtype=np.float32)
else:
    saved_embed = np.zeros((ort_session_A._inputs_meta[2].shape[0], ort_session_A._inputs_meta[2].shape[1]), dtype=np.float32)
    saved_dot = np.ones((1, ort_session_A._inputs_meta[2].shape[1]), dtype=np.float32)
    empty_embed = None
    empty_dot = None
if "float16" in model_type:
    saved_embed = saved_embed.astype(np.float16)
    saved_dot = saved_dot.astype(np.float16)
    if dynamic_axes:
        empty_embed = empty_embed.astype(np.float16)
        empty_dot = empty_dot.astype(np.float16)


# Load the input audio
print(f"\nTest Input Audio: {test_audio}")
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
if dynamic_axes:
    INPUT_AUDIO_LENGTH = min(320000, audio_len)  # You can adjust it.
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

# Start to run SenseVoicePlus
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
language_idx = np.array([TARGET_LANGUAGE], dtype=np.int32)
while slice_end <= aligned_len:
    input_feed = {
                in_name_A0: audio[:, :, slice_start: slice_end],
                in_name_A1: language_idx,
                in_name_A2: saved_embed,
                in_name_A3: saved_dot
            }
    if not dynamic_axes:
        input_feed[in_name_A4] = num_speakers
    start_time = time.time()
    token_ids, target_speaker_id, speaker_score, speaker_embed, speaker_embed_dot = ort_session_A.run([out_name_A0, out_name_A1, out_name_A2, out_name_A3, out_name_A4], input_feed)
    end_time = time.time()
    if speaker_score >= SIMILARITY_THRESHOLD:
        saved_embed[:, target_speaker_id] = (saved_embed[:, target_speaker_id] + speaker_embed) * 0.5
        saved_dot[:, target_speaker_id] = (saved_dot[:, target_speaker_id] + speaker_embed_dot) * 0.5
        speaker_id = target_speaker_id
        print(f"\nLocate the identified speaker with ID = {speaker_id}, Similarity = {speaker_score[0]:.3f}")
    else:
        saved_embed[:, num_speakers] = speaker_embed
        saved_dot[:, num_speakers] = speaker_embed_dot
        speaker_id = num_speakers[0]
        print(f"\nIt's an unknown speaker. Assign it a new ID = {speaker_id}")
        num_speakers += 1
        if dynamic_axes:
            saved_embed = np.concatenate((saved_embed, empty_embed), axis=1)
            saved_dot = np.concatenate((saved_dot, empty_dot), axis=1)
    text = tokenizer.decode(token_ids.tolist())
    print(f"\nSpeaker_ID_{speaker_id}: {text}\n\nTime Cost: {end_time - start_time:.3f} Seconds\n")
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    print("----------------------------------------------------------------------------------------------------------")
