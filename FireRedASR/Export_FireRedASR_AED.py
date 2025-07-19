import gc
import sys
import shutil
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.


project_path = "/home/DakeQQ/Downloads/FireRedASR-main"                                    # The FireRedASR Github project path.
model_path = "/home/DakeQQ/Downloads/FireRedASR-AED-L"                                     # The FireRedASR-AED model download path.
onnx_model_A = "/home/DakeQQ/Downloads/FireRedASR_ONNX/FireRedASR_AED_L-Encoder.onnx"      # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/FireRedASR_ONNX/FireRedASR_AED_L-Decoder.onnx"      # The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/zh_1.wav", "./example/zh_2.wav"]              # The test audio list.


DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 240000                                 # Set for maximum input audio length.
WINDOW_TYPE = 'hann'                                        # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
MAX_SEQ_LEN = 80                                            # Set an appropriate value.
STOP_TOKEN = [4]                                            # 4 is the end token for FireRedASR-AED series model.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.


if project_path not in sys.path:
    sys.path.append(project_path)

shutil.copyfile('./modeling_modified/conformer_encoder.py', project_path + "/fireredasr/models/module/conformer_encoder.py")
shutil.copyfile('./modeling_modified/fireredasr.py', project_path + "/fireredasr/models/fireredasr.py")
from fireredasr.models.fireredasr import FireRedAsr
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)


class FIRE_RED_ENCODER(torch.nn.Module):
    def __init__(self, fire_red, feat_extractor, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis):
        super(FIRE_RED_ENCODER, self).__init__()
        self.model = fire_red
        self.stft_model = stft_model
        self.cmvn_means = torch.from_numpy(feat_extractor.cmvn.means).float().view(1, 1, -1)
        self.cmvn_vars = torch.from_numpy(feat_extractor.cmvn.inverse_std_variences).float().view(1, 1, -1)
        self.model.encoder.positional_encoding.pe.data = self.model.encoder.positional_encoding.pe.data.half()
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.save_en_keys = [None] * self.model.decoder.n_layers
        self.save_en_values = [None] * self.model.decoder.n_layers
        self.inv_int16 = float(1.0 / 32768.0)
  
    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-5).log()
        mel_features = (mel_features - self.cmvn_means) * self.cmvn_vars
        enc_outputs = self.model.encoder(mel_features)
        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            self.save_en_keys[idx] = torch.matmul(enc_outputs, decoder_layer.cross_attn.w_ks.weight).transpose(1, 2)
            self.save_en_values[idx] = torch.matmul(enc_outputs, decoder_layer.cross_attn.w_vs.weight) + decoder_layer.cross_attn.w_vs.bias
        return *self.save_en_keys, *self.save_en_values


class FIRE_RED_DECODER(torch.nn.Module):
    def __init__(self, fire_red, max_seq_len):
        super(FIRE_RED_DECODER, self).__init__()
        self.model = fire_red
        self.num_layers_de_2 = self.model.decoder.n_layers + self.model.decoder.n_layers
        self.num_layers_de_2_plus_1 = self.num_layers_de_2 + 1
        self.num_layers_de_2_plus_2 = self.num_layers_de_2 + 2
        self.num_layers_de_2_plus_3 = self.num_layers_de_2 + 3
        self.num_layers_de_3_plus = self.num_layers_de_2_plus_3 + self.model.decoder.n_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.model.decoder.tgt_word_emb.weight *= self.model.decoder.scale
        self.model.decoder.positional_encoding.pe.data = self.model.decoder.positional_encoding.pe.data[:, :max_seq_len].half()
        self.save_de_keys = [None] * self.model.decoder.n_layers
        self.save_de_values = [None] * self.model.decoder.n_layers

    def forward(self, *all_inputs):
        history_len = all_inputs[self.num_layers_de_2]
        input_ids = all_inputs[self.num_layers_de_2_plus_1]
        ids_len = all_inputs[self.num_layers_de_2_plus_2]
        kv_seq_len = history_len + ids_len
        hidden_state = self.model.decoder.tgt_word_emb(input_ids) + self.model.decoder.positional_encoding.pe[:, history_len: kv_seq_len].float()
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            hidden_state_norm = decoder_layer.self_attn_norm(hidden_state)
            q = torch.matmul(hidden_state_norm, decoder_layer.self_attn.w_qs.weight) + decoder_layer.self_attn.w_qs.bias
            k = torch.matmul(hidden_state_norm, decoder_layer.self_attn.w_ks.weight).transpose(1, 2)
            v = torch.matmul(hidden_state_norm, decoder_layer.self_attn.w_vs.weight) + decoder_layer.self_attn.w_vs.bias
            k = torch.cat((all_inputs[idx], k), dim=2)
            v = torch.cat((all_inputs[idx + self.model.decoder.n_layers], v), dim=1)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)
            hidden_state_attn = torch.matmul(hidden_state_attn, decoder_layer.self_attn.fc.weight).sum(dim=0, keepdim=True) + decoder_layer.self_attn.fc.bias
            hidden_state_attn += hidden_state
            q = torch.matmul(decoder_layer.cross_attn_norm(hidden_state_attn), decoder_layer.cross_attn.w_qs.weight) + decoder_layer.cross_attn.w_qs.bias
            hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.num_layers_de_2_plus_3]), dim=-1), all_inputs[idx + self.num_layers_de_3_plus])
            hidden_state_cross = torch.matmul(hidden_state_cross, decoder_layer.cross_attn.fc.weight).sum(dim=0, keepdim=True) + decoder_layer.cross_attn.fc.bias
            hidden_state_cross += hidden_state_attn
            hidden_state = hidden_state_cross + decoder_layer.mlp(decoder_layer.mlp_norm(hidden_state_cross))
        max_logit_idx = torch.argmax(self.model.decoder.tgt_word_prj(self.model.decoder.layer_norm_out(hidden_state)[:, -1]), dim=-1, keepdim=True).int()
        return *self.save_de_keys, *self.save_de_values, kv_seq_len, max_logit_idx


print('\nStart to export the Encoder part.\n')
with torch.inference_mode():
    if 'aed' in model_path or 'AED' in model_path or 'Aed' in model_path:
        model = FireRedAsr.from_pretrained("aed", model_path)
        feat_extractor = model.feat_extractor
        model = model.model.float()
        HIDDEN_SIZE = model.encoder.odim
        NUM_HEAD_EN = model.encoder.layer_stack._modules['0'].mhsa.n_head
        NUM_HEAD_DE = model.decoder.layer_stack._modules['0'].self_attn.n_head
        NUM_LAYER_DE = model.decoder.n_layers
        HEAD_DIM_EN = model.encoder.layer_stack._modules['0'].mhsa.d_k
        HEAD_DIM_DE = model.decoder.layer_stack._modules['0'].self_attn.d_k
        scaling = float(HEAD_DIM_EN ** -0.25)
        for i in model.encoder.layer_stack._modules:
            model.encoder.layer_stack._modules[i].mhsa.w_qs.weight.data *= scaling
            model.encoder.layer_stack._modules[i].mhsa.w_ks.weight.data *= scaling
            model.encoder.layer_stack._modules[i].mhsa.linear_pos.weight.data *= scaling
            model.encoder.layer_stack._modules[i].mhsa.pos_bias_u.data = model.encoder.layer_stack._modules[i].mhsa.pos_bias_u.data.unsqueeze(1) * scaling
            model.encoder.layer_stack._modules[i].mhsa.pos_bias_v.data = model.encoder.layer_stack._modules[i].mhsa.pos_bias_v.data.unsqueeze(1) * scaling

            model.encoder.layer_stack._modules[i].mhsa.w_qs.weight.data = model.encoder.layer_stack._modules[i].mhsa.w_qs.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.encoder.layer_stack._modules[i].mhsa.w_ks.weight.data = model.encoder.layer_stack._modules[i].mhsa.w_ks.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.encoder.layer_stack._modules[i].mhsa.w_vs.weight.data = model.encoder.layer_stack._modules[i].mhsa.w_vs.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.encoder.layer_stack._modules[i].mhsa.linear_pos.weight.data = model.encoder.layer_stack._modules[i].mhsa.linear_pos.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.encoder.layer_stack._modules[i].mhsa.fc.weight.data = model.encoder.layer_stack._modules[i].mhsa.fc.weight.data.view(HIDDEN_SIZE, NUM_HEAD_EN, HEAD_DIM_EN).permute(1, 2, 0).contiguous()

        scaling = float(HEAD_DIM_DE ** -0.25)
        for i in model.decoder.layer_stack._modules:
            model.decoder.layer_stack._modules[i].self_attn.w_qs.weight.data *= scaling
            model.decoder.layer_stack._modules[i].self_attn.w_qs.bias.data *= scaling
            model.decoder.layer_stack._modules[i].self_attn.w_ks.weight.data *= scaling

            model.decoder.layer_stack._modules[i].self_attn.w_qs.weight.data = model.decoder.layer_stack._modules[i].self_attn.w_qs.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.decoder.layer_stack._modules[i].self_attn.w_qs.bias.data = model.decoder.layer_stack._modules[i].self_attn.w_qs.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
            model.decoder.layer_stack._modules[i].self_attn.w_ks.weight.data = model.decoder.layer_stack._modules[i].self_attn.w_ks.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.decoder.layer_stack._modules[i].self_attn.w_vs.weight.data = model.decoder.layer_stack._modules[i].self_attn.w_vs.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.decoder.layer_stack._modules[i].self_attn.w_vs.bias.data = model.decoder.layer_stack._modules[i].self_attn.w_vs.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
            model.decoder.layer_stack._modules[i].self_attn.fc.weight.data = model.decoder.layer_stack._modules[i].self_attn.fc.weight.data.view(HIDDEN_SIZE, NUM_HEAD_DE, HEAD_DIM_DE).permute(1, 2, 0).contiguous()
            model.decoder.layer_stack._modules[i].self_attn.fc.bias.data = model.decoder.layer_stack._modules[i].self_attn.fc.bias.data.view(1, 1, -1).contiguous()

        scaling = float(model.decoder.layer_stack._modules['0'].cross_attn.d_k ** -0.25)
        for i in model.decoder.layer_stack._modules:
            model.decoder.layer_stack._modules[i].cross_attn.w_qs.weight.data *= scaling
            model.decoder.layer_stack._modules[i].cross_attn.w_qs.bias.data *= scaling
            model.decoder.layer_stack._modules[i].cross_attn.w_ks.weight.data *= scaling

            model.decoder.layer_stack._modules[i].cross_attn.w_qs.weight.data = model.decoder.layer_stack._modules[i].cross_attn.w_qs.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.decoder.layer_stack._modules[i].cross_attn.w_qs.bias.data = model.decoder.layer_stack._modules[i].cross_attn.w_qs.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
            model.decoder.layer_stack._modules[i].cross_attn.w_ks.weight.data = model.decoder.layer_stack._modules[i].cross_attn.w_ks.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.decoder.layer_stack._modules[i].cross_attn.w_vs.weight.data = model.decoder.layer_stack._modules[i].cross_attn.w_vs.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
            model.decoder.layer_stack._modules[i].cross_attn.w_vs.bias.data = model.decoder.layer_stack._modules[i].cross_attn.w_vs.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
            model.decoder.layer_stack._modules[i].cross_attn.fc.weight.data = model.decoder.layer_stack._modules[i].cross_attn.fc.weight.data.view(HIDDEN_SIZE, NUM_HEAD_DE, HEAD_DIM_DE).permute(1, 2, 0).contiguous()
            model.decoder.layer_stack._modules[i].cross_attn.fc.bias.data = model.decoder.layer_stack._modules[i].cross_attn.fc.bias.data.view(1, 1, -1).contiguous()

        custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
        fire_red_encoder = FIRE_RED_ENCODER(model, feat_extractor, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE)

        output_names = []
        audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
        dynamic_axes = {'audio': {2: 'audio_len'}}
        for i in range(NUM_LAYER_DE):
            name = f'en_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {2: 'signal_len'}
        for i in range(NUM_LAYER_DE):
            name = f'en_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {1: 'signal_len'}

        torch.onnx.export(
            fire_red_encoder,
            (audio,),
            onnx_model_A,
            input_names=['audio'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=17,
            external_data=True,
        )
        del fire_red_encoder
        del audio
        del custom_stft
        del name
        del output_names
        del dynamic_axes
        gc.collect()
        print("\nExport Done!\n\nStart to export the Decoder part.")

        fire_red_decoder = FIRE_RED_DECODER(model, MAX_SEQ_LEN)
        input_ids = torch.tensor([[3]], dtype=torch.int32)
        ids_len = torch.tensor([input_ids.shape[-1]], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        save_encoder_key = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float32)
        save_encoder_value = torch.zeros((NUM_HEAD_EN, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float32)
        past_key_de = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float32)
        past_value_de = torch.zeros((NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float32)
        attention_mask = torch.tensor([1], dtype=torch.int8)

        input_names = []
        all_inputs = []
        output_names = []
        dynamic_axes = {}

        for i in range(NUM_LAYER_DE):
            name = f'in_de_key_{i}'
            input_names.append(name)
            all_inputs.append(past_key_de)
            dynamic_axes[name] = {2: 'history_len'}
            name = f'out_de_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
        for i in range(NUM_LAYER_DE):
            name = f'in_de_value_{i}'
            input_names.append(name)
            all_inputs.append(past_value_de)
            dynamic_axes[name] = {1: 'history_len'}
            name = f'out_de_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {1: 'history_len_plus_ids_len'}

        input_names.append('history_len')
        all_inputs.append(history_len)
        input_names.append('input_ids')
        all_inputs.append(input_ids)
        input_names.append('ids_len')
        all_inputs.append(ids_len)

        for i in range(NUM_LAYER_DE):
            name = f'en_key_{i}'
            input_names.append(name)
            all_inputs.append(save_encoder_key)
            dynamic_axes[name] = {2: 'signal_len'}
        for i in range(NUM_LAYER_DE):
            name = f'en_value_{i}'
            input_names.append(name)
            all_inputs.append(save_encoder_value)
            dynamic_axes[name] = {1: 'signal_len'}

        input_names.append('attention_mask')
        all_inputs.append(attention_mask)
        output_names.append('max_logit_id')
        output_names.append('kv_seq_len')

        torch.onnx.export(
            fire_red_decoder,
            tuple(all_inputs),
            onnx_model_B,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
            do_constant_folding=True,
            opset_version=17
        )
        del model
        del fire_red_decoder
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
    else:
        print("Currently, only support the FireRedASR-AED")

if project_path in sys.path:
    sys.path.remove(project_path)
print('\nExport done!\n\nStart to run FireRedASR by ONNX Runtime.\n\nNow, loading the model...')


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
input_names_B = []
output_names_B = []
amount_of_outputs = len(out_name_B)
for i in range(len(in_name_B)):
    input_names_B.append(in_name_B[i].name)
for i in range(amount_of_outputs):
    output_names_B.append(out_name_B[i].name)

generate_limit = MAX_SEQ_LEN - 1  # 1 = length of input_ids
num_layers = (amount_of_outputs - 2) // 2
num_layers_2 = num_layers + num_layers
num_layers_4 = num_layers_2 + num_layers_2
num_layers_2_plus_1 = num_layers_2 + 1
num_layers_2_plus_2 = num_layers_2 + 2

tokenizer = ChineseCharEnglishSpmTokenizer(model_path + "/dict.txt", model_path + "/train_bpe1000.model")

# Load the input audio
for language_idx, test in enumerate(test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
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

    # Start to run FireRedASR
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    input_ids = np.array([[3]], dtype=np.int32)
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), 'cpu', 0)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, 'cpu', 0)
    history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
    attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[0].shape[0], ort_session_B._inputs_meta[0].shape[1], 0), dtype=np.float32), 'cpu', 0)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[num_layers].shape[0], 0, ort_session_B._inputs_meta[num_layers].shape[2]), dtype=np.float32), 'cpu', 0)
    layer_indices = np.arange(num_layers_2, num_layers_4, dtype=np.int32) + 3
    input_feed_B = {
        in_name_B[-1].name: attention_mask,
        in_name_B[num_layers_2].name: history_len,
        in_name_B[num_layers_2_plus_1].name: input_ids,
        in_name_B[num_layers_2_plus_2].name: ids_len
    }
    for i in range(num_layers):
        input_feed_B[in_name_B[i].name] = past_keys_B
    for i in range(num_layers, num_layers_2):
        input_feed_B[in_name_B[i].name] = past_values_B
    num_decode = 0
    save_token = []
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start:slice_end], 'cpu', 0)})
        for i in range(num_layers_2):
            input_feed_B[in_name_B[layer_indices[i]].name] = all_outputs_A[i]
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
            max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[-1])
            num_decode += 1
            if max_logit_ids in STOP_TOKEN:
                break
            for i in range(amount_of_outputs):
                input_feed_B[in_name_B[i].name] = all_outputs_B[i]
            if num_decode < 2:
                input_feed_B[in_name_B[-1].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
            save_token.append(max_logit_ids)
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    text = ("".join([tokenizer.dict[int(id[0][0])] for id in save_token])).replace(tokenizer.SPM_SPACE, ' ').strip()
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")
