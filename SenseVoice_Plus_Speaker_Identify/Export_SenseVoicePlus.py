import gc
import time
import shutil
import site

import numpy as np
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment
from funasr import AutoModel
from modelscope.models.base import Model

from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path_asr = "/home/DakeQQ/Downloads/SenseVoiceSmall"                                                                                                 # The SenseVoice download path.
model_path_speaker = "/home/DakeQQ/Downloads/speech_eres2netv2_sv_zh-cn_16k-common"                                                                       # The SenseVoice download path.
onnx_model_A = "/home/iamj/Downloads/SenseVoice_ONNX/SenseVoiceSmallPlus.onnx"                                                                          # The exported onnx model path.
test_audio = [model_path_asr + "/example/zh.mp3", model_path_asr + "/example/en.mp3", model_path_asr + "/example/yue.mp3", model_path_asr + "/example/ja.mp3", model_path_asr + "/example/ko.mp3", model_path_asr + "/example/ko.mp3"]   # The test audio list. Duplicate the last one for Speaker Identify.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 128000 if not DYNAMIC_AXES else 320000 # Set for static axis export: the length of the audio input signal (in samples). If using DYNAMIC_AXES, default to 320000, you can adjust it.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT = 400                                                  # Number of FFT components for the STFT process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
LFR_M = 7                                                   # The model parameter, do not edit the value.
LFR_N = 6                                                   # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
TARGET_LANGUAGE = 0                                         # Choose one of indices ['auto' = 0, 'zh' = 1, 'en' = 2, 'yue' = 3, 'ja' = 4, 'ko' = 5, 'nospeech' = 6]
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SPEAKERS = 50                                           # Maximum number of saved speaker features.
HIDDEN_SIZE = 192                                           # Model hidden size. Do not edit it.
SIMILARITY_THRESHOLD = 0.5                                  # Threshold to determine the speaker's identity. You can adjust it.
USE_EMOTION = False                                          # Output the emotion tag or not.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if NFFT > INPUT_AUDIO_LENGTH:
    NFFT = INPUT_AUDIO_LENGTH
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


shutil.copyfile('./modeling_modified/ERes2NetV2.py', site.getsitepackages()[-1] + "/modelscope/models/audio/sv/ERes2NetV2.py")


class SENSE_VOICE_PLUS(torch.nn.Module):
    def __init__(self, sense_voice, stft_model, nfft, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, ref_len, cmvn_means, cmvn_vars, eres2netv2, use_emo):
        super(SENSE_VOICE_PLUS, self).__init__()
        self.eres2netv2 = eres2netv2
        self.embed_sys = sense_voice.embed
        self.encoder = sense_voice.encoder
        self.ctc_lo = sense_voice.ctc.ctc_lo
        self.stft_model = stft_model
        self.cmvn_means = cmvn_means
        self.cmvn_vars = cmvn_vars
        self.T_lfr = lfr_len
        self.blank_id = sense_voice.blank_id
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 20, 8000, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=ref_len + self.lfr_m_factor - 1)
        self.system_embed = self.embed_sys(torch.tensor([1, 2, 14], dtype=torch.int32)).unsqueeze(0) if use_emo else self.embed_sys(torch.tensor([5, 14], dtype=torch.int32)).unsqueeze(0)
        self.language_embed = self.embed_sys(torch.tensor([0, 3, 4, 7, 11, 12, 13], dtype=torch.int32)).unsqueeze(0).half()  # Original dict: {'auto': 0, 'zh': 3, 'en': 4, 'yue': 7, 'ja': 11, 'ko': 12, 'nospeech': 13}

    def forward(self, audio, language_idx, saved_embed, saved_dot, num_speakers):
        audio = audio.float()
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-5).log()
        speaker_embed = self.eres2netv2.forward(mel_features - mel_features.mean(dim=1, keepdim=True))
        speaker_embed_T = speaker_embed.transpose(0, 1)
        speaker_embed_dot = torch.matmul(speaker_embed, speaker_embed_T)
        if not DYNAMIC_AXES:
            saved_embed = saved_embed[:, :num_speakers]
            saved_dot = saved_dot[:, :num_speakers]
        speaker_score = torch.matmul(speaker_embed, saved_embed) / torch.sqrt(speaker_embed_dot * saved_dot)
        speaker_score, target_speaker_id = torch.max(speaker_score, dim=-1)
        mel_features = mel_features.transpose(1, 2)
        left_padding = mel_features[:, [0], :].repeat(1, self.lfr_m_factor, 1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        mel_features = padded_inputs[:, self.indices_mel.clamp(max=padded_inputs.shape[1] - 1)].reshape(1, self.T_lfr, -1)
        mel_features = torch.cat((self.language_embed[:, language_idx].float(), self.system_embed, mel_features), dim=1)
        encoder_out = self.encoder((mel_features + self.cmvn_means) * self.cmvn_vars)
        token_ids = self.ctc_lo(encoder_out).argmax(dim=-1).int()
        shifted_tensor = torch.roll(token_ids, shifts=-1, dims=-1)
        mask = ((token_ids != shifted_tensor) & (token_ids != self.blank_id)).to(torch.int32)
        mask[..., 0] = 1
        return token_ids.index_select(-1, torch.nonzero(mask, as_tuple=True)[-1]), target_speaker_id.int(), speaker_score, speaker_embed_T, speaker_embed_dot


print('\nExport start ...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model_asr = AutoModel(
        model=model_path_asr,
        trust_remote_code=True,
        disable_update=True,
        remote_code="./modeling_modified/model.py",
        device="cpu",
        LFR_LENGTH=LFR_LENGTH,
        FEATURE_SIZE=560,  # The model parameter, do not edit the value.
        USE_EMOTION=USE_EMOTION
    )
    encoder_output_size_factor = (model_asr.model.encoder.output_size()) ** 0.5
    model_asr.model.embed.weight.data *= encoder_output_size_factor
    CMVN_MEANS = model_asr.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model_asr.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    tokenizer = model_asr.kwargs['tokenizer']
    model_speaker = Model.from_pretrained(
        model_name_or_path=model_path_speaker,
        disable_update=True,
        device="cpu",
    ).embedding_model.eval()
    sense_voice_plus = SENSE_VOICE_PLUS(model_asr.model.eval(), custom_stft, NFFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, STFT_SIGNAL_LENGTH, CMVN_MEANS, CMVN_VARS, model_speaker, USE_EMOTION)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    language_idx = torch.tensor([0], dtype=torch.int32)
    saved_embed = torch.ones((HIDDEN_SIZE, MAX_SPEAKERS), dtype=torch.float32)
    saved_dot = torch.ones((1, MAX_SPEAKERS), dtype=torch.float32)
    num_speakers = torch.tensor([1], dtype=torch.int64)
    torch.onnx.export(
        sense_voice_plus,
        (audio, language_idx, saved_embed, saved_dot, num_speakers),
        onnx_model_A,
        input_names=['audio', 'language_idx', 'saved_embed', 'saved_dot', 'num_speakers'],
        output_names=['token_ids', 'target_speaker_id', 'speaker_score', 'speaker_embed', 'speaker_embed_dot'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'saved_embed': {1: 'max_speakers'},
            'saved_dot': {1: 'max_speakers'},
            'token_ids': {1: 'num_token'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del model_asr
    del model_speaker
    del sense_voice_plus
    del audio
    del language_idx
    del CMVN_VARS
    del CMVN_MEANS
    gc.collect()
print('\nExport done!\n\nStart to run SenseVoicePlus by ONNX Runtime.\n\nNow, loading the model...')


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
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if dynamic_axes:
        INPUT_AUDIO_LENGTH = min(163840, audio_len)  # You can adjust it.
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
    language_idx = np.array([language_idx + 1], dtype=np.int32)
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
        text = tokenizer.decode(token_ids.tolist())[0]
        print(f"\nSpeaker_ID_{speaker_id}: {text}\n\nTime Cost: {end_time - start_time:.3f} Seconds\n")
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        print("----------------------------------------------------------------------------------------------------------")
      
