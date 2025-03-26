import gc
import time
import site
import shutil
import json

import numpy as np
import onnxruntime
import torch
import torchaudio
from pydub import AudioSegment

from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1"      # The Paraformer-Chinese download path.
onnx_model_A = "/home/DakeQQ/Downloads/Paraformer_ONNX/Paraformer.onnx"                                     # The exported onnx model path.
test_audio = "./zh.mp3"                                                                                     # The test audio list.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 48000 if not DYNAMIC_AXES else 320000  # Set for static axis export: the length of the audio input signal (in samples). If using DYNAMIC_AXES, default to 320000, you can adjust it but less than 320000 (20 seconds).
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT = 400                                                  # Number of FFT components for the STFT process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
LFR_M = 7                                                   # The model parameter, do not edit the value.
LFR_N = 6                                                   # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if NFFT > INPUT_AUDIO_LENGTH:
    NFFT = INPUT_AUDIO_LENGTH
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


python_package_path = site.getsitepackages()[-1]
shutil.copyfile('./modeling_modified/attention.py', python_package_path + '/funasr/models/sanm/attention.py')
shutil.copyfile('./modeling_modified/encoder.py', python_package_path + '/funasr/models/sanm/encoder.py')
shutil.copyfile('./modeling_modified/decoder.py', python_package_path + '/funasr/models/sanm/decoder.py')
shutil.copyfile('./modeling_modified/positionwise_feed_forward.py', python_package_path + '/funasr/models/sanm/positionwise_feed_forward.py')
shutil.copyfile('./modeling_modified/cif_predictor.py', python_package_path + '/funasr/models/paraformer/cif_predictor.py')
shutil.copyfile('./modeling_modified/model.py', python_package_path + '/funasr/models/paraformer/model.py')
shutil.copyfile('./modeling_modified/embedding.py', python_package_path + '/funasr/models/transformer/embedding.py')
from funasr import AutoModel


class PARAFORMER(torch.nn.Module):
    def __init__(self, paraformer, stft_model, nfft, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, ref_len, cmvn_means, cmvn_vars):
        super(PARAFORMER, self).__init__()
        self.encoder = paraformer.encoder
        self.calc_predictor = paraformer.calc_predictor
        self.cal_decoder_with_predictor = paraformer.cal_decoder_with_predictor
        self.stft_model = stft_model
        self.cmvn_means = cmvn_means
        self.cmvn_vars = cmvn_vars
        self.T_lfr = lfr_len
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft // 2 + 1, 20, 8000, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=ref_len + self.lfr_m_factor - 1)

    def forward(self, audio):
        audio = audio.float()
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, torch.sqrt(real_part * real_part + imag_part * imag_part)).transpose(1, 2).clamp(min=1e-5).log()
        left_padding = mel_features[:, [0], :].repeat(1, self.lfr_m_factor, 1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        mel_features = padded_inputs[:, self.indices_mel.clamp(max=padded_inputs.shape[1] - 1)].reshape(1, self.T_lfr, -1)
        encoder_out = self.encoder((mel_features + self.cmvn_means) * self.cmvn_vars)
        pre_acoustic_embeds = self.calc_predictor(encoder_out)
        decoder_outs = self.cal_decoder_with_predictor(encoder_out, pre_acoustic_embeds)
        return decoder_outs.argmax(dim=-1).int()


print('\nExport start ...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = AutoModel(
        model=model_path,
        disable_update=True,
        device="cpu"
    )
    encoder_output_size_factor = (model.model.encoder.output_size()) ** 0.5
    CMVN_MEANS = model.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    tokenizer = model.kwargs['tokenizer']
    paraformer = PARAFORMER(model.model.eval(), custom_stft, NFFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, STFT_SIGNAL_LENGTH, CMVN_MEANS, CMVN_VARS)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        paraformer,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=['token_ids'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'token_ids': {1: 'num_token'}
        } if DYNAMIC_AXES else None,
        opset_version=17
    )
    del model
    del audio
    del CMVN_VARS
    del CMVN_MEANS
    gc.collect()
print('\nExport done!\n\nStart to run Paraformer by ONNX Runtime.\n\nNow, loading the model...')


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


with open(model_path + "/tokens.json", 'r', encoding='UTF-8') as json_file:
    tokenizer = np.array(json.load(json_file), dtype=np.str_)


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


# Load the input audio
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples())
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

# Start to run Paraformer
slice_start = 0
slice_end = INPUT_AUDIO_LENGTH
while slice_end <= aligned_len:
    start_time = time.time()
    token_ids = ort_session_A.run(
        [out_name_A0],
        {
            in_name_A0: audio[:, :, slice_start: slice_end],
        })[0]
    end_time = time.time()
    text = tokenizer[token_ids[0]].tolist()
    if '</s>' in text:
        text.remove('</s>')
    text = ''.join(text)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    print(f"\nASR Result:\n{text}\n\nTime Cost: {end_time - start_time:.3f} Seconds\n")
    print("----------------------------------------------------------------------------------------------------------")
