import gc
import time
import json
import torch
import torchaudio
import numpy as np
import onnxruntime
from funasr import AutoModel
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


model_path = "/home/DakeQQ/Downloads/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"      # The Paraformer-Chinese-Online-Streaming download path.
onnx_model_A = "/home/DakeQQ/Downloads/Paraformer_ONNX/Paraformer_Streaming_Encoder.onnx"                    # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Paraformer_ONNX/Paraformer_Streaming_Decoder.onnx"                    # The exported onnx model path.
test_audio = "./zh.wav"                                                                                      # The test audio.


ORT_Accelerate_Providers = []                               # If you have accelerate devices for : ['CUDAExecutionProvider', 'TensorrtExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'MIGraphXExecutionProvider', 'AzureExecutionProvider']
                                                            # else keep empty.
MAX_CONTINUE_STREAMING = 503                                # 503 = Max 30 seconds streaming audio input. # 1005 = Max 60 seconds streaming audio input.
INPUT_AUDIO_LENGTH = 8800                                   # The fixed input audio segment length, edit it carefully.
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
NFFT_FBANK = 512                                            # Number of FFT components for the FBank process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
LFR_M = 7                                                   # The model parameter, do not edit the value.
LFR_N = 6                                                   # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
LOOK_BACK_A = 5                                             # The model parameter, edit it carefully. 5 for 8800 input audio length. 10 for 8800*2 ...
LOOK_BACK_B = 10                                            # The model parameter, edit it carefully. 10 for 8800 input audio length. 20 for 8800*2 ...
LOOK_BACK_ENCODER = 4                                       # The model parameter, edit it carefully.
LOOK_BACK_DECODER = 1                                       # The model parameter, edit it carefully.
DYNAMIC_AXES = True                                         # The dynamic_axes setting. Do not turn off for the Paraformer Streaming model.


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class PARAFORMER_ENCODER(torch.nn.Module):
    def __init__(self, paraformer, stft_model, nfft_stft, nfft_fbank, stft_signal_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, cmvn_means, cmvn_vars, cif_hidden_size, fsmn_hidden_size, feature_size, look_back_A, look_back_B, look_back_en, max_continue_streaming):
        super(PARAFORMER_ENCODER, self).__init__()
        self.inv_int16 = float(1.0 / 32768.0)
        self.threshold = float(1.0)
        self.look_back_A = look_back_A
        self.look_back_B = look_back_B
        self.look_back_en = -(look_back_en * look_back_B) - self.look_back_A
        self.encoder = paraformer.encoder
        self.predictor = paraformer.predictor
        self.stft_model = stft_model
        self.cmvn_means = cmvn_means
        self.cmvn_vars = cmvn_vars
        self.T_lfr = lfr_len
        self.cif_hidden_size = cif_hidden_size
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_fbank // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.nfft_fbank = nfft_fbank
        if self.nfft_stft > self.nfft_fbank:
            self.padding = torch.zeros((1, n_mels, (nfft_stft - nfft_fbank) // 2), dtype=torch.float32)
            self.fbank = torch.cat((self.fbank, self.padding), dim=-1)
        else:
            self.padding = torch.zeros((1, (nfft_fbank - nfft_stft) // 2, stft_signal_len), dtype=torch.int8)
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=stft_signal_len + self.lfr_m_factor - 1)
        self.total_encoders = list(self.encoder.encoders0) + list(self.encoder.encoders)
        self.cache_layer_num_en = len(self.total_encoders)
        self.save_keys_en = [None] * self.cache_layer_num_en
        self.save_values_en = [None] * self.cache_layer_num_en
        self.pad_zeros_predictor = torch.zeros((1, cif_hidden_size, 1), dtype=torch.float32)
        self.pad_zeros_fsmn = torch.zeros((1, fsmn_hidden_size, self.encoder.encoders0._modules["0"].self_attn.pad_fn.padding[0]), dtype=torch.float32)
        positions = torch.arange(1, max_continue_streaming, dtype=torch.int32).unsqueeze(0)
        self.position_encoding = self.encoder.embed.encode(positions, feature_size).half()

    def forward(self, *all_inputs):
        previous_mel_features = all_inputs[-5]
        cif_hidden = all_inputs[-4]
        cif_alphas = all_inputs[-3]
        start_idx = all_inputs[-2]
        audio = all_inputs[-1]
        audio = audio.float() * self.inv_int16
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        power = real_part * real_part + imag_part * imag_part
        if self.nfft_fbank > self.nfft_stft:
            power = torch.cat((power, self.padding[:, :, :power.shape[-1]].float()), dim=1)
        mel_features = torch.matmul(self.fbank, power).transpose(1, 2).clamp(min=1e-5).log()
        left_padding = mel_features[:, [0], :]
        left_padding = torch.cat([left_padding for _ in range(self.lfr_m_factor)], dim=1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        mel_features = padded_inputs[:, self.indices_mel.clamp(max=padded_inputs.shape[1] - 1)].reshape(1, self.T_lfr, -1)
        mel_features = (mel_features - self.cmvn_means) * self.cmvn_vars
        end_idx = start_idx + mel_features.shape[1]
        mel_features = mel_features + self.position_encoding[:, start_idx:end_idx]
        x = torch.cat([previous_mel_features, mel_features], dim=1)
        previous_mel_features = x[:, -self.look_back_A:]
        for layer_idx, encoder_layer in enumerate(self.total_encoders):
            if layer_idx > 0:
                residual = x
            q_k_v = encoder_layer.self_attn.linear_q_k_v(encoder_layer.norm1(x))
            q, k, v = torch.split(q_k_v, self.cif_hidden_size, dim=-1)
            q = q.reshape(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            k = k.reshape(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0)
            v_h = v.reshape(-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).transpose(0, 1)
            k = torch.cat([all_inputs[layer_idx], k], dim=2)
            v_h = torch.cat([all_inputs[layer_idx + self.cache_layer_num_en], v_h], dim=1)
            self.save_keys_en[layer_idx] = k[:, :, self.look_back_en:-self.look_back_A]
            self.save_values_en[layer_idx] = v_h[:, self.look_back_en:-self.look_back_A]
            v_fsmn = torch.cat([self.pad_zeros_fsmn, v.transpose(1, 2), self.pad_zeros_fsmn], dim=-1)
            v_fsmn = encoder_layer.self_attn.fsmn_block(v_fsmn).transpose(1, 2)
            v_fsmn += v
            q = q * (encoder_layer.self_attn.d_k ** (-0.5))
            x = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v_h).transpose(0, 1).contiguous().view(1, -1, self.cif_hidden_size)
            x = encoder_layer.self_attn.linear_out(x)
            x += v_fsmn
            if layer_idx > 0:
                x += residual
            x += encoder_layer.feed_forward.w_2(encoder_layer.feed_forward.activation(encoder_layer.feed_forward.w_1(encoder_layer.norm2(x))))
        encoder_out = self.encoder.after_norm(x)
        output = torch.cat([self.pad_zeros_predictor, encoder_out.transpose(1, 2), self.pad_zeros_predictor], dim=-1)
        output = torch.relu(self.predictor.cif_conv1d(output)).transpose(1, 2)
        alphas = torch.sigmoid(self.predictor.cif_output(output))
        alphas = torch.nn.functional.relu(alphas).squeeze()
        list_frame = []
        save_condition = []
        condition_A = cif_alphas < self.threshold
        condition_B = ~condition_A
        condition_A = condition_A.float()
        condition_B = condition_B.float()
        save_condition.append(condition_B)
        if self.threshold != 1.0:
            frames = cif_alphas * cif_hidden * condition_A + self.threshold * cif_hidden * condition_B
        else:
            frames = cif_alphas * cif_hidden * condition_A + cif_hidden * condition_B
        list_frame.append(frames)
        if self.threshold != 1.0:
            cif_alphas -= self.threshold * condition_B
        else:
            cif_alphas -= condition_B
        frames = frames * condition_A + cif_alphas * cif_hidden * condition_B
        for i in range(self.look_back_B):
            alpha = alphas[i]
            threshold = self.threshold - cif_alphas
            condition_A = alpha < threshold
            condition_B = ~condition_A
            condition_A = condition_A.float()
            condition_B = condition_B.float()
            save_condition.append(condition_B)
            hidden = encoder_out[:, [i]]
            frames = (frames + alpha * hidden) * condition_A + (frames + threshold * hidden) * condition_B
            list_frame.append(frames)
            cif_alphas = cif_alphas + alpha
            if self.threshold != 1.0:
                cif_alphas -= self.threshold * condition_B
            else:
                cif_alphas -= condition_B
            frames = frames * condition_A + cif_alphas * hidden * condition_B
        list_frame = torch.cat(list_frame, dim=1)
        cif_hidden = list_frame[:, [-1]] / cif_alphas
        list_frame = list_frame.index_select(1, torch.nonzero(torch.cat(save_condition, dim=0), as_tuple=True)[-1])
        list_frame_len = list_frame.shape[1]
        return *self.save_keys_en, *self.save_values_en, previous_mel_features, cif_hidden, cif_alphas, end_idx, encoder_out, list_frame, list_frame_len


class PARAFORMER_DECODER(torch.nn.Module):
    def __init__(self, paraformer, look_back_B, look_back_de, cif_hidden_size):
        super(PARAFORMER_DECODER, self).__init__()
        self.look_back_B = look_back_B
        self.look_back_de = look_back_de * look_back_B
        self.decoder = paraformer.decoder
        self.cif_hidden_size = cif_hidden_size
        self.fsmn_kernal_size_minus = -torch.tensor([self.decoder.decoders._modules["0"].self_attn.kernel_size - 1], dtype=torch.int64)
        self.cache_layer_num_de = len(self.decoder.decoders)
        self.cache_layer_num_de_2 = self.cache_layer_num_de + self.cache_layer_num_de
        self.save_fsmn_de = [None] * self.cache_layer_num_de
        self.save_keys_de = [None] * self.cache_layer_num_de
        self.save_values_de = [None] * self.cache_layer_num_de

    def forward(self, *all_inputs):
        encoder_out = all_inputs[-3]
        list_frame = all_inputs[-2]
        list_frame_len = all_inputs[-1]
        look_back = self.fsmn_kernal_size_minus - list_frame_len
        for layer_idx, decoder_layer in enumerate(self.decoder.decoders):
            residual = list_frame
            list_frame = decoder_layer.norm1(list_frame)
            list_frame = decoder_layer.feed_forward.w_2(decoder_layer.feed_forward.norm(decoder_layer.feed_forward.activation(decoder_layer.feed_forward.w_1(list_frame))))
            list_frame = decoder_layer.norm2(list_frame)
            x = torch.cat((all_inputs[layer_idx], list_frame.transpose(1, 2)), dim=-1)[:, :, look_back:]
            self.save_fsmn_de[layer_idx] = x
            x = decoder_layer.self_attn.fsmn_block(x).transpose(1, 2)
            x += list_frame + residual
            residual = x
            q = decoder_layer.src_attn.linear_q(decoder_layer.norm3(x))
            q = q.reshape(-1, decoder_layer.src_attn.h, decoder_layer.src_attn.d_k).transpose(0, 1)
            k_v = decoder_layer.src_attn.linear_k_v(encoder_out)
            k, v = torch.split(k_v, self.cif_hidden_size, dim=-1)
            k = k.reshape(-1, decoder_layer.src_attn.h, decoder_layer.src_attn.d_k).permute(1, 2, 0)
            v = v.reshape(-1, decoder_layer.src_attn.h, decoder_layer.src_attn.d_k).transpose(0, 1)
            k = torch.cat([all_inputs[layer_idx + self.cache_layer_num_de], k], dim=2)
            v = torch.cat([all_inputs[layer_idx + self.cache_layer_num_de_2], v], dim=1)
            self.save_keys_de[layer_idx] = k[:, :, -self.look_back_de:]
            self.save_values_de[layer_idx] = v[:, -self.look_back_de:]
            q = q * (decoder_layer.src_attn.d_k ** (-0.5))
            x = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v).transpose(0, 1).contiguous().view(1, -1, self.cif_hidden_size)
            x = decoder_layer.src_attn.linear_out(x)
            list_frame = residual + x
        x = self.decoder.decoders3[0].norm1(list_frame)
        x = self.decoder.decoders3[0].feed_forward.w_2(self.decoder.decoders3[0].feed_forward.norm(self.decoder.decoders3[0].feed_forward.activation(self.decoder.decoders3[0].feed_forward.w_1(x))))
        x = self.decoder.output_layer(self.decoder.after_norm(x))
        max_logit_ids = torch.argmax(x, dim=-1, keepdim=False).int()
        return *self.save_fsmn_de, *self.save_keys_de, *self.save_values_de, max_logit_ids


print('\nExport Encoder Part...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, n_mels=N_MELS, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = AutoModel(
        model=model_path,
        disable_update=True,
        device="cpu"
    )
    encoder_output_size_factor = (model.model.encoder.output_size()) ** 0.5
    CMVN_MEANS = model.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    tokenizer = model.kwargs['tokenizer']
    model = model.model.eval()
    NUM_LAYER_EN = len(model.encoder.encoders0) + len(model.encoder.encoders)
    NUM_LAYER_DE = len(model.decoder.decoders)
    FEATURE_SIZE = model.encoder.encoders0._modules["0"].in_size
    CIF_HIDDEN_SIZE = model.encoder.encoders0._modules["0"].size
    FSMN_HIDDEN_SIZE = model.decoder.decoders._modules["0"].size
    NUM_HEAD_EN = model.encoder.encoders0._modules["0"].self_attn.h
    HEAD_DIM_EN = model.encoder.encoders0._modules["0"].self_attn.d_k
    NUM_HEAD_DE = model.decoder.decoders._modules["0"].src_attn.h
    HEAD_DIM_DE = model.decoder.decoders._modules["0"].src_attn.d_k
    FSMN_DE_PAD = model.decoder.decoders._modules["0"].self_attn.pad_fn.padding[0]

    key_en = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, 0), dtype=torch.float32)
    value_en = torch.zeros((NUM_HEAD_EN, 0, HEAD_DIM_EN), dtype=torch.float32)
    previous_mel_features = torch.zeros((1, LOOK_BACK_A, FEATURE_SIZE), dtype=torch.float32)
    cif_hidden = torch.zeros((1, 1, CIF_HIDDEN_SIZE), dtype=torch.float32)
    cif_alphas = torch.zeros(1, dtype=torch.float32)
    start_idx = torch.zeros(1, dtype=torch.int64)
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_EN):
        name = f'in_en_key_{i}'
        input_names.append(name)
        all_inputs.append(key_en)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_en_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus'}
    for i in range(NUM_LAYER_EN):
        name = f'in_en_value_{i}'
        input_names.append(name)
        all_inputs.append(value_en)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_en_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'history_len_plus'}
    input_names.append("in_previous_mel_features")
    all_inputs.append(previous_mel_features)
    output_names.append("out_previous_mel_features")
    input_names.append("in_cif_hidden")
    all_inputs.append(cif_hidden)
    output_names.append("out_cif_hidden")
    input_names.append("in_cif_alphas")
    all_inputs.append(cif_alphas)
    output_names.append("out_cif_alphas")
    input_names.append("start_idx")
    all_inputs.append(start_idx)
    output_names.append("end_idx")
    input_names.append("audio")
    output_names.append("encoder_out")
    output_names.append("list_frame")
    dynamic_axes["list_frame"] = {1: 'list_frame_len'}
    output_names.append("list_frame_len")
    all_inputs.append(audio)

    paraformer_encoder = PARAFORMER_ENCODER(model, custom_stft, NFFT_STFT, NFFT_FBANK, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, CMVN_MEANS, CMVN_VARS, CIF_HIDDEN_SIZE, FSMN_HIDDEN_SIZE, FEATURE_SIZE, LOOK_BACK_A, LOOK_BACK_B, LOOK_BACK_ENCODER, MAX_CONTINUE_STREAMING)
    torch.onnx.export(
        paraformer_encoder,
        tuple(all_inputs),
        onnx_model_A,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17
    )
    del paraformer_encoder
    del audio
    del key_en
    del value_en
    del previous_mel_features
    del cif_hidden
    del cif_alphas
    del start_idx
    del CMVN_VARS
    del CMVN_MEANS
    del all_inputs
    del input_names
    del output_names
    del dynamic_axes
    gc.collect()
    print('\nDone Encoder Part!\n\nExport Decoder Part...')

    key_de = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float32)
    value_de = torch.zeros((NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float32)
    fsmn_de = torch.zeros((1, FSMN_HIDDEN_SIZE, FSMN_DE_PAD), dtype=torch.float32)
    encoder_out = torch.zeros((1, LOOK_BACK_A + LFR_LENGTH, CIF_HIDDEN_SIZE), dtype=torch.float32)
    list_frame = torch.zeros((1, 1, CIF_HIDDEN_SIZE), dtype=torch.float32)
    list_frame_len = torch.tensor(1, dtype=torch.int64)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_fsmn_{i}'
        input_names.append(name)
        all_inputs.append(fsmn_de)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_de_fsmn_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus'}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_key_{i}'
        input_names.append(name)
        all_inputs.append(key_de)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_de_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus'}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_value_{i}'
        input_names.append(name)
        all_inputs.append(value_de)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_de_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'history_len_plus'}

    input_names.append("encoder_out")
    all_inputs.append(encoder_out)
    input_names.append("list_frame")
    dynamic_axes["list_frame"] = {1: 'list_frame_len'}
    all_inputs.append(list_frame)
    input_names.append("list_frame_len")
    all_inputs.append(list_frame_len)
    output_names.append("max_logit_ids")
    dynamic_axes["max_logit_ids"] = {-1: 'token_len'}

    paraformer_decoder = PARAFORMER_DECODER(model, LOOK_BACK_B, LOOK_BACK_DECODER, CIF_HIDDEN_SIZE)
    torch.onnx.export(
        paraformer_decoder,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=17
    )
    del paraformer_decoder
    del key_de
    del value_de
    del all_inputs
    del input_names
    del output_names
    del dynamic_axes
print('\nExport done!\n\nStart to run Paraformer-Streaming by ONNX Runtime.\n\nNow, loading the model...')


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
shape_value_in = ort_session_A._inputs_meta[-1].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
input_names_A = []
output_names_A = []
amount_of_outputs_A = len(out_name_A)
for i in range(len(in_name_A)):
    input_names_A.append(in_name_A[i].name)
for i in range(amount_of_outputs_A):
    output_names_A.append(out_name_A[i].name)


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
input_names_B = []
output_names_B = []
amount_of_outputs_B = len(out_name_B)
for i in range(len(in_name_B)):
    input_names_B.append(in_name_B[i].name)
for i in range(amount_of_outputs_B):
    output_names_B.append(out_name_B[i].name)


# Load the input audio
audio = np.array(AudioSegment.from_file(test_audio).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
audio = normalize_to_int16(audio)
audio_len = len(audio)
audio = audio.reshape(1, 1, -1)
if isinstance(shape_value_in, str):
    INPUT_AUDIO_LENGTH = min(8800, audio_len)  # You can adjust it.
else:
    INPUT_AUDIO_LENGTH = shape_value_in
stride_step = INPUT_AUDIO_LENGTH
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


# Initialize
num_layer_en = (amount_of_outputs_A - 7) // 2
num_layer_de = (amount_of_outputs_B - 1) // 3
amount_of_outputs_B -= 1
amount_of_outputs_A -= 3

in_en_value_start = num_layer_en
in_previous_mel_features_start = in_en_value_start + num_layer_en
in_cif_hidden_start = in_previous_mel_features_start + 1
in_de_key_start = num_layer_de
in_de_value_start = in_de_key_start + num_layer_de
in_de_value_end = in_de_value_start + num_layer_de

in_en_key_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[0].shape[0], ort_session_A._inputs_meta[0].shape[1], 0), dtype=np.float32))
in_en_value_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[in_en_value_start].shape[0], 0, ort_session_A._inputs_meta[in_en_value_start].shape[2]), dtype=np.float32))
in_previous_mel_features = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[in_previous_mel_features_start].shape[0], ort_session_A._inputs_meta[in_previous_mel_features_start].shape[1], ort_session_A._inputs_meta[in_previous_mel_features_start].shape[2]), dtype=np.float32))
in_cif_hidden = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_A._inputs_meta[in_cif_hidden_start].shape[0], ort_session_A._inputs_meta[in_cif_hidden_start].shape[1], ort_session_A._inputs_meta[in_cif_hidden_start].shape[2]), dtype=np.float32))
in_cif_alpha = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=np.float32))
start_idx = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(1, dtype=np.int64))
in_de_fsmn_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[0].shape[0], ort_session_B._inputs_meta[0].shape[1], FSMN_DE_PAD), dtype=np.float32))
in_de_key_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[in_de_key_start].shape[0], ort_session_B._inputs_meta[in_de_key_start].shape[1], 0), dtype=np.float32))
in_de_value_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[in_de_value_start].shape[0], 0, ort_session_B._inputs_meta[in_de_value_start].shape[2]), dtype=np.float32))


def Initialize():
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    input_feed_B = {}
    input_feed_A = {
        in_name_A[-2].name: start_idx,
        in_name_A[-3].name: in_cif_alpha,
        in_name_A[-4].name: in_cif_hidden,
        in_name_A[-5].name: in_previous_mel_features
    }
    for i in range(num_layer_en):
        input_feed_A[in_name_A[i].name] = in_en_key_A
    for i in range(in_en_value_start, in_previous_mel_features_start):
        input_feed_A[in_name_A[i].name] = in_en_value_A
    for i in range(num_layer_de):
        input_feed_B[in_name_B[i].name] = in_de_fsmn_B
    for i in range(in_de_key_start, in_de_value_start):
        input_feed_B[in_name_B[i].name] = in_de_key_B
    for i in range(in_de_value_start, in_de_value_end):
        input_feed_B[in_name_B[i].name] = in_de_value_B
    return input_feed_A, input_feed_B, slice_start, slice_end


# Start to run Paraformer-Streaming
input_feed_A, input_feed_B, slice_start, slice_end = Initialize()
while True:
    input_feed_A[in_name_A[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], 'cpu', 0)
    slice_start += stride_step
    slice_end = slice_start + INPUT_AUDIO_LENGTH
    start_time = time.time()
    all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, input_feed_A)
    if slice_end <= aligned_len:
        for i in range(amount_of_outputs_A):
            input_feed_A[in_name_A[i].name] = all_outputs_A[i]
    if onnxruntime.OrtValue.numpy(all_outputs_A[-1]) != 0:
        input_feed_B[in_name_B[-1].name] = all_outputs_A[-1]
        input_feed_B[in_name_B[-2].name] = all_outputs_A[-2]
        input_feed_B[in_name_B[-3].name] = all_outputs_A[-3]
        all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
        end_time = time.time()
        max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-1])[0]
        text = tokenizer[max_logit_ids].tolist()
        text = ''.join(text).replace("</s>", "")
        print(f"ASR: {text} / Time Cost: {end_time - start_time:.3f} Seconds")
        if slice_end > aligned_len:
            input_feed_A, input_feed_B, slice_start, slice_end = Initialize()  # Ready for next input audio.
            break
        for i in range(amount_of_outputs_B):
            input_feed_B[in_name_B[i].name] = all_outputs_B[i]
    elif slice_end > aligned_len:
        input_feed_A, input_feed_B, slice_start, slice_end = Initialize()      # Ready for next input audio.
        break
