import gc
import time

import numpy as np
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment
from funasr import AutoModel

from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path = "/home/DakeQQ/Downloads/SenseVoiceSmall"                                     # The SenseVoice download path.
onnx_model_A = "/home/DakeQQ/Downloads/SenseVoice_ONNX/SenseVoiceSmall.onnx"              # The exported onnx model path.
test_audio = [model_path + "/example/zh.mp3", model_path + "/example/en.mp3", model_path + "/example/yue.mp3", model_path + "/example/ja.mp3", model_path + "/example/ko.mp3"]   # The test audio list.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 160000                                 # The maximum input audio length.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 400                                             # Number of FFT components for the STFT process, edit it carefully.
NFFT_FBANK = 512                                            # Number of FFT components for the FBank process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
LFR_M = 7                                                   # The model parameter, do not edit the value.
LFR_N = 6                                                   # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
TARGET_LANGUAGE = 0                                         # Choose one of indices ['auto' = 0, 'zh' = 1, 'en' = 2, 'yue' = 3, 'ja' = 4, 'ko' = 5, 'nospeech' = 6]
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
USE_EMOTION = True                                          # Output the emotion tag or not.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if NFFT_FBANK < NFFT_STFT:
    NFFT_FBANK = NFFT_STFT
if NFFT_FBANK > INPUT_AUDIO_LENGTH:
    NFFT_FBANK = INPUT_AUDIO_LENGTH
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class SENSE_VOICE(torch.nn.Module):
    def __init__(self, sense_voice, stft_model, nfft_stft, nfft_fbank, stft_signal_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, cmvn_means, cmvn_vars, use_emo):
        super(SENSE_VOICE, self).__init__()
        self.embed_sys = sense_voice.embed
        self.encoder = sense_voice.encoder
        self.ctc_lo = sense_voice.ctc.ctc_lo
        self.stft_model = stft_model
        self.cmvn_means = cmvn_means
        self.cmvn_vars = cmvn_vars
        self.T_lfr = lfr_len
        self.blank_id = sense_voice.blank_id
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_fbank // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.padding = torch.zeros((1, (nfft_fbank - nfft_stft) // 2, stft_signal_len), dtype=torch.int8)
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=stft_signal_len + self.lfr_m_factor - 1)
        self.system_embed = self.embed_sys(torch.tensor([1, 2, 14], dtype=torch.int32)).unsqueeze(0) if use_emo else self.embed_sys(torch.tensor([5, 14], dtype=torch.int32)).unsqueeze(0)
        self.language_embed = self.embed_sys(torch.tensor([0, 3, 4, 7, 11, 12, 13], dtype=torch.int32)).unsqueeze(0).half()  # Original dict: {'auto': 0, 'zh': 3, 'en': 4, 'yue': 7, 'ja': 11, 'ko': 12, 'nospeech': 13}
        self.int_inv16 = float(1.0 / 32768.0)
  
    def forward(self, audio, language_idx):
        audio = audio.float() * self.int_inv16
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        power = real_part * real_part + imag_part * imag_part
        if self.padding.shape[1] != 0:
            power = torch.cat((power, self.padding[:, :, :power.shape[-1]].float()), dim=1)
        mel_features = torch.matmul(self.fbank, power).transpose(1, 2).clamp(min=1e-5).log()
        left_padding = mel_features[:, [0], :]
        left_padding = torch.cat([left_padding for _ in range(self.lfr_m_factor)], dim=1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        mel_features = padded_inputs[:, self.indices_mel.clamp(max=padded_inputs.shape[1] - 1)].reshape(1, self.T_lfr, -1)
        mel_features = torch.cat((self.language_embed[:, language_idx].float(), self.system_embed, mel_features), dim=1)
        encoder_out = self.encoder((mel_features + self.cmvn_means) * self.cmvn_vars)
        token_ids = self.ctc_lo(encoder_out).argmax(dim=-1).int()
        shifted_tensor = torch.roll(token_ids, shifts=-1, dims=-1)
        mask = ((token_ids != shifted_tensor) & (token_ids != self.blank_id)).to(torch.int32)
        mask[..., 0] = 1
        return token_ids.index_select(-1, torch.nonzero(mask, as_tuple=True)[-1])


print('\nExport start ...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        disable_update=True,
        remote_code="./modeling_modified/model.py",
        device="cpu",
        LFR_LENGTH=LFR_LENGTH,
        FEATURE_SIZE=560,  # The model parameter, do not edit the value.
        USE_EMOTION=USE_EMOTION
    )
    encoder_output_size_factor = (model.model.encoder.output_size()) ** 0.5
    model.model.embed.weight.data *= encoder_output_size_factor
    CMVN_MEANS = model.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    tokenizer = model.kwargs['tokenizer']
    sense_voice = SENSE_VOICE(model.model.eval(), custom_stft, NFFT_STFT, NFFT_FBANK, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, CMVN_MEANS, CMVN_VARS, USE_EMOTION)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    language_idx = torch.tensor([0], dtype=torch.int32)
    torch.onnx.export(
        sense_voice,
        (audio, language_idx),
        onnx_model_A,
        input_names=['audio', 'language_idx'],
        output_names=['token_ids'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'token_ids': {1: 'num_token'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del model
    del sense_voice
    del audio
    del language_idx
    del CMVN_VARS
    del CMVN_MEANS
    gc.collect()
print('\nExport done!\n\nStart to run SenseVoice by ONNX Runtime.\n\nNow, loading the model...')


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
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
in_name_A1 = in_name_A[1].name
out_name_A0 = out_name_A[0].name


# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
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

    # Start to run SenseVoice
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    language_idx = np.array([language_idx + 1], dtype=np.int32)
    while slice_end <= aligned_len:
        start_time = time.time()
        token_ids = ort_session_A.run(
            [out_name_A0],
            {
                in_name_A0: audio[:, :, slice_start: slice_end],
                in_name_A1: language_idx
            })[0]
        end_time = time.time()
        text = tokenizer.decode(token_ids.tolist())[0]
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        print(f"\nASR Result:\n{text}\n\nTime Cost: {end_time - start_time:.3f} Seconds\n")
        print("----------------------------------------------------------------------------------------------------------")
