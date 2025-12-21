import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
from pydub import AudioSegment
from funasr import AutoModel
from transformers import AutoTokenizer
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512'                                                 # Set the path where the [Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512] downloaded.  URL: https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 / https://modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512
tokenizer_path = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B'                                  # Set the tokenizer path.
onnx_model_A = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Encoder.onnx'                      # The exported onnx model path.
onnx_model_B = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Embed.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Main.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Greedy_Search.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_First_Beam_Search.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Second_Beam_Search.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Reset_Penality.onnx'

# The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/yue.mp3", "./example/ja.mp3"]           # The test audio list.
task_prompt = ["将语音转写成中文：", "将语音转写成英文：", "将语音转写成粤语：", "将语音转写成日文："]               # The prompt of transcription task.


if "MLT" in model_path:
    test_audio += ["./example/ko.mp3"]
    task_prompt += ["将语音转写成韩文："]


# Audio & STFT Configuration
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
WINDOW_TYPE = 'hamming'                                     # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 400                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
USE_NORMALIZER = True                                       # If true, use the audio normalizer to make the loudness consistent.

# Model Parameters
LFR_M = 7                                                   # The model parameter, do not edit the value.
LFR_N = 6                                                   # The model parameter, do not edit the value.
STOP_TOKEN = [151643, 151645]                               # The stop_id in Qwen is "151643" & "151645"
MAX_SEQ_LEN = 1024                                          # The max context length.

# Input & Processing Limits
MAX_INPUT_AUDIO_LENGTH = 320000                             # The maximum input audio length.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.

# Decoding Strategy
USE_BEAM_SEARCH = False                                     # Use beam search or greedy search. It recommended to use greedy search for Fun-ASR-Nano.
TOP_K = 3                                                   # The top k candidate in decoding.
BEAM_SIZE = 3                                               # Number of beams in searching.
MAX_BEAM_SIZE = 10                                          # Max beams for exported model.
REPEAT_PENALITY = 0.8                                       # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                                         # Penalizes the most recent output. "10" means the last 10 tokens.

# Runtime & Export Settings
MAX_THREADS = 0                                             # Parllel CPU threads. Set 0 for auto.
DEVICE_ID = 0                                               # Default to zero.
OPSET = 17                                                  # ONNX Runtime opset version.


MAX_STFT_SIGNAL_LENGTH = MAX_INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (MAX_STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > MAX_INPUT_AUDIO_LENGTH:
    HOP_LENGTH = MAX_INPUT_AUDIO_LENGTH


def normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


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
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
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
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
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


class FUNASR_NANO_ENCODER(torch.nn.Module):
    def __init__(self, funasr_nano, stft_model, nfft_stft, max_stft_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, _tokenizer):
        super(FUNASR_NANO_ENCODER, self).__init__()
        self.funasr_nano = funasr_nano.float()
        self.stft_model = stft_model
        self.T_lfr = lfr_len
        self.lfr_n = lfr_n
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int64).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int64)
        self.indices_mel = indices.clamp(max=max_stft_len + self.lfr_m_factor - 1)
        self.output_size_factor = self.funasr_nano.audio_encoder.output_size() ** 0.5
        self.position_encoding = self.funasr_nano.audio_encoder.embed(torch.zeros([1, max_stft_len, 560], dtype=torch.float32))
        num_head = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.d_k
        self.pad_zeros = torch.zeros((1, num_head * head_dim, 5), dtype=torch.float32)
        factor = float(head_dim ** (-0.25))
        self.total_encoders = list(self.funasr_nano.audio_encoder.encoders0) + list(self.funasr_nano.audio_encoder.encoders) + list(self.funasr_nano.audio_encoder.tp_encoders)
        in_siz = self.funasr_nano.audio_encoder.encoders._modules["0"].in_size
        in_size_2 = in_siz + in_siz
        for encoder_layer in self.total_encoders:
            encoder_layer.self_attn.linear_q_k_v.weight.data[:in_size_2] *= factor
            encoder_layer.self_attn.linear_q_k_v.bias.data[:in_size_2] *= factor
            encoder_layer.self_attn.linear_q_w = encoder_layer.self_attn.linear_q_k_v.weight.data[:in_siz].view(num_head, head_dim, -1).transpose(1, 2).contiguous()
            encoder_layer.self_attn.linear_q_b = encoder_layer.self_attn.linear_q_k_v.bias.data[:in_siz].view(num_head, 1, head_dim).contiguous()
            encoder_layer.self_attn.linear_k_w = encoder_layer.self_attn.linear_q_k_v.weight.data[in_siz:in_size_2].view(num_head, head_dim, -1).transpose(1, 2).contiguous()
            encoder_layer.self_attn.linear_k_b = encoder_layer.self_attn.linear_q_k_v.bias.data[in_siz:in_size_2].view(num_head, 1, head_dim).contiguous()
            encoder_layer.self_attn.linear_v_w = encoder_layer.self_attn.linear_q_k_v.weight.data[in_size_2:].transpose(0, 1).unsqueeze(0).contiguous()
            encoder_layer.self_attn.linear_v_b = encoder_layer.self_attn.linear_q_k_v.bias.data[in_size_2:].view(1, 1, -1).contiguous()
            encoder_layer.self_attn.linear_out_w = encoder_layer.self_attn.linear_out.weight.data.view(-1, num_head, head_dim).permute(1, 2, 0).contiguous()
            encoder_layer.self_attn.linear_out_b = encoder_layer.self_attn.linear_out.bias.data.view(1, 1, -1).contiguous()
        
        num_head = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        for block in self.funasr_nano.audio_adaptor.blocks:
            block.self_attn.linear_q.weight.data *= factor
            block.self_attn.linear_q.bias.data *= factor
            block.self_attn.linear_k.weight.data *= factor
            block.self_attn.linear_k.bias.data *= factor
            block.self_attn.linear_q.weight.data = block.self_attn.linear_q.weight.data.view(num_head, head_dim, -1).transpose(1, 2).contiguous()
            block.self_attn.linear_q.bias.data = block.self_attn.linear_q.bias.data.view(num_head, head_dim, -1).view(num_head, 1, head_dim).contiguous()
            block.self_attn.linear_k.weight.data = block.self_attn.linear_k.weight.data.view(num_head, head_dim, -1).transpose(1, 2).contiguous()
            block.self_attn.linear_k.bias.data = block.self_attn.linear_k.bias.data.view(num_head, 1, head_dim).contiguous()
            block.self_attn.linear_v.weight.data = block.self_attn.linear_v.weight.data.view(num_head, head_dim, -1).transpose(1, 2).contiguous()
            block.self_attn.linear_v.bias.data = block.self_attn.linear_v.bias.data.view(num_head, 1, head_dim).contiguous()
            block.self_attn.linear_out.weight.data = block.self_attn.linear_out.weight.data.view(-1, num_head, head_dim).permute(1, 2, 0).contiguous()
            block.self_attn.linear_out.bias.data = block.self_attn.linear_out.bias.data.view(1, 1, -1).contiguous()

        head_ids = _tokenizer.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n", return_tensors="pt")
        tail_ids = _tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        self.head_embed = self.funasr_nano.llm.model.embed_tokens(head_ids)
        self.tail_embed = self.funasr_nano.llm.model.embed_tokens(tail_ids)
        self.fake_token = torch.zeros(max_stft_len + 1, dtype=torch.int64)
        for i in range(self.fake_token.shape[0]):
            self.fake_token[i] = (((i - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1

    def forward(self, audio, query_embed):
        audio = audio.float()
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = (torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2) + 1e-7).log()
        features_len = mel_features.shape[1].unsqueeze(0)
        left_padding = mel_features[:, [0], :]
        left_padding = torch.cat([left_padding for _ in range(self.lfr_m_factor)], dim=1)
        padded_inputs = torch.cat((left_padding, mel_features), dim=1)
        _len = features_len // self.lfr_n - 1
        mel_features = padded_inputs[:, self.indices_mel[:_len]].reshape(1, _len, -1)
        x = mel_features * self.output_size_factor + self.position_encoding[:, :_len].float()
        for encoder_layer in self.funasr_nano.audio_encoder.encoders0 + self.funasr_nano.audio_encoder.encoders:
            x1 = encoder_layer.norm1(x)
            q_h = torch.matmul(x1, encoder_layer.self_attn.linear_q_w) + encoder_layer.self_attn.linear_q_b
            k_h = (torch.matmul(x1, encoder_layer.self_attn.linear_k_w) + encoder_layer.self_attn.linear_k_b).transpose(1, 2)
            v = torch.matmul(x1, encoder_layer.self_attn.linear_v_w) + encoder_layer.self_attn.linear_v_b
            v_h = torch.reshape(v, (-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k)).transpose(0, 1)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.matmul(torch.softmax(torch.matmul(q_h, k_h), dim=-1), v_h)
            attn = torch.matmul(attn, encoder_layer.self_attn.linear_out_w).sum(dim=0, keepdim=True) + encoder_layer.self_attn.linear_out_b + fsmn_memory
            if encoder_layer.in_size == encoder_layer.size:
                x += attn
            else:
                x = attn
            x = x + encoder_layer.feed_forward(encoder_layer.norm2(x))
        x = self.funasr_nano.audio_encoder.after_norm(x)
        for encoder_layer in self.funasr_nano.audio_encoder.tp_encoders:
            x1 = encoder_layer.norm1(x)
            q_h = torch.matmul(x1, encoder_layer.self_attn.linear_q_w) + encoder_layer.self_attn.linear_q_b
            k_h = (torch.matmul(x1, encoder_layer.self_attn.linear_k_w) + encoder_layer.self_attn.linear_k_b).transpose(1, 2)
            v = torch.matmul(x1, encoder_layer.self_attn.linear_v_w) + encoder_layer.self_attn.linear_v_b
            v_h = torch.reshape(v, (-1, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k)).transpose(0, 1)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.matmul(torch.softmax(torch.matmul(q_h, k_h), dim=-1), v_h)
            attn = torch.matmul(attn, encoder_layer.self_attn.linear_out_w).sum(dim=0, keepdim=True) + encoder_layer.self_attn.linear_out_b + fsmn_memory
            x += attn
            x = x + encoder_layer.feed_forward(encoder_layer.norm2(x))
        x = self.funasr_nano.audio_encoder.tp_norm(x)
        x = self.funasr_nano.audio_adaptor.linear1(x)
        x = self.funasr_nano.audio_adaptor.relu(x)
        x = self.funasr_nano.audio_adaptor.linear2(x)
        for block in self.funasr_nano.audio_adaptor.blocks:
            x1 = block.norm1(x)
            q = torch.matmul(x1, block.self_attn.linear_q.weight) + block.self_attn.linear_q.bias
            k = (torch.matmul(x1, block.self_attn.linear_k.weight) + block.self_attn.linear_k.bias).transpose(1, 2)
            v = torch.matmul(x1, block.self_attn.linear_v.weight) + block.self_attn.linear_v.bias
            attn = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v)
            attn = torch.matmul(attn, block.self_attn.linear_out.weight.data).sum(dim=0, keepdim=True) + block.self_attn.linear_out.bias.data
            x += attn
            x = x + block.feed_forward(block.norm2(x))
        x = x[:, :self.fake_token[features_len]]
        concat_embed = torch.cat([self.head_embed, query_embed, x, self.tail_embed], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


class FUNASR_NANO_DECODER_EMBED(torch.nn.Module):
    def __init__(self, funasr_nano):
        super(FUNASR_NANO_DECODER_EMBED, self).__init__()
        self.funasr_nano_decoder_embed = funasr_nano.llm.model.embed_tokens.float()
        
    def forward(self, input_ids):
        return self.funasr_nano_decoder_embed(input_ids)


class FUNASR_NANO_DECODER_MAIN(torch.nn.Module):
    def __init__(self, funasr_nano, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(FUNASR_NANO_DECODER_MAIN, self).__init__()
        self.funasr_nano_decoder_main = funasr_nano.llm.float()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)
        
        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.funasr_nano_decoder_main.model.layers._modules[f'{i}'].self_attn.q_norm.weight.data *= scale_factor
            self.funasr_nano_decoder_main.model.layers._modules[f'{i}'].self_attn.k_norm.weight.data *= scale_factor

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        idx_theta = (position_ids * self.funasr_nano_decoder_main.model.rotary_emb.inv_freq).unsqueeze(0).unsqueeze(0)
        cos_rotary_pos_emb = torch.cos(idx_theta)
        sin_rotary_pos_emb = torch.sin(idx_theta)
        self.cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).half()
        self.sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).half()

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def rotate_half(self, x, head_dim_half, dim):
        x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
        return torch.cat((-x2, x1), dim=dim)

    def repeat_k(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, head_dim, -1)

    def repeat_v(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, -1, head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.funasr_nano_decoder_main.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(batch_size, -1, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
            q = (layer.self_attn.q_norm.weight * (q / torch.sqrt(q.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).transpose(1, 2)
            k = (layer.self_attn.k_norm.weight * (k / torch.sqrt(k.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))).permute(0, 3, 2, 4, 1)
            q = q * rotary_pos_emb_cos_q + self.rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + self.rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.funasr_nano_decoder_main.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        logits = self.funasr_nano_decoder_main.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits, kv_seq_len


print('\nExport start ...\n')
with torch.inference_mode():
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        remote_code="./modeling_modified/model.py",
        device="cpu",
        disable_update=True
    )

    num_heads = model.model.llm.config.num_attention_heads
    num_key_value_heads = model.model.llm.config.num_key_value_heads
    head_dim = model.model.llm.config.head_dim
    num_layers = model.model.llm.config.num_hidden_layers
    vocab_size = model.model.llm.model.vocab_size
    hidden_size = model.model.llm.model.embed_tokens.embedding_dim
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    funasr_nano_encoder = FUNASR_NANO_ENCODER(model.model, custom_stft, NFFT_STFT, MAX_STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, tokenizer)
    audio = torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=torch.int16)
    query_embed = torch.ones((1, 10, hidden_size), dtype=torch.float32)  # "10" is just a dummy value.
    torch.onnx.export(
        funasr_nano_encoder,
        (audio, query_embed),
        onnx_model_A,
        input_names=['audio', 'query_embed'],
        output_names=['concat_embed', 'ids_len'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'query_embed': {1: 'num_token'},
            'concat_embed': {1: 'num_token'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del funasr_nano_encoder
    del audio
    del custom_stft
    gc.collect()

    batch_size = 3
    ids_len = torch.tensor([10], dtype=torch.long)      
    history_len = torch.tensor([0], dtype=torch.long)
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)
    past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
    past_values = torch.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)
    kv_seq_len = history_len + ids_len

    model_B = FUNASR_NANO_DECODER_EMBED(model.model)
    torch.onnx.export(
        model_B,
        (input_ids,),
        onnx_model_B,
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'ids_len'},
            'hidden_states': {0: 'batch', 1: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del model_B
    del input_ids
    
    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'hidden_states': {0: 'batch', 1: 'ids_len'}}
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 4: 'ks_seq_len'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'ks_seq_len'}
    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('logits')
    output_names.append('kv_seq_len')
    dynamic_axes['logits'] = {0: 'batch'}

    model_C = FUNASR_NANO_DECODER_MAIN(model.model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)
    torch.onnx.export(
        model_C,
        tuple(all_inputs),
        onnx_model_C,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del model
    del model_C
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del num_heads
    del num_key_value_heads
    del head_dim
    del hidden_size
    del ids_len
    del history_len
    del batch_size
    del hidden_states
    del attention_mask
    del kv_seq_len
    gc.collect()
    
    greedy = GREEDY_SEARCH()
    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
    logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
    batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

    torch.onnx.export(
        greedy,
        (logits, repeat_penality, penality_value, beam_size),
        # Reuse the beam_size tensor as batch_size during export process.
        onnx_model_D,
        input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
        output_names=['max_logits_idx', 'repeat_penality_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del greedy

    first_beam_search = FIRST_BEAM_SEARCH(num_layers)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
    previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
    past_keys_greedy = past_keys[[0]]
    past_values_greedy = past_values[[0]]

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys_greedy)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 4: 'kv_seq_len'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values_greedy)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'kv_seq_len'}
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
    output_names.append('top_beam_prob')
    output_names.append('batch_indices')
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
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del first_beam_search

    all_inputs = []
    input_names = []
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
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

    second_beam_search = SECOND_BEAM_SEARCH(num_layers)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_F,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del second_beam_search
    del num_layers
    del past_keys
    del past_values
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del logits
    del beam_size
    del penality_value

    reset_penality = RESET_PENALITY()
    torch.onnx.export(
        reset_penality,
        (save_id, repeat_penality, penality_reset_count, batch_indices),
        onnx_model_G,
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
        opset_version=OPSET,
        dynamo=False
    )
    del reset_penality
    del save_id
    del repeat_penality
    del penality_reset_count
    del batch_indices

print('\nExport done!\n\nStart to run FunASR-Nano by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
session_opts.log_severity_level = 4                 # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                # Fatal level, it an adjustable value.
run_options.log_severity_level = 4                  # Fatal level, it an adjustable value.
run_options.log_verbosity_level = 4                 # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')

ORT_Accelerate_Providers = ['CPUExecutionProvider']
provider_options = None
device_type = 'cpu'
DEVICE_ID = 0

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(len(out_name_A))]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B = in_name_B[0].name
out_name_B = [out_name_B[0].name]

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
print(f"\nUsable Providers: {ort_session_C.get_providers()}")
model_dtype = ort_session_C._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
amount_of_outputs_C = len(out_name_C)
in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
out_name_C = [out_name_C[i].name for i in range(amount_of_outputs_C)]


generate_limit = MAX_SEQ_LEN - 20                   # 20 = length of basic ids
num_layers = (amount_of_outputs_C - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7
vocab_size = ort_session_C._outputs_meta[num_keys_values].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([REPEAT_PENALITY], dtype=model_dtype), device_type, DEVICE_ID)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")
    TOP_K = BEAM_SIZE


if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")


if USE_BEAM_SEARCH:
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    amount_of_outputs_F = len(out_name_F)
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(amount_of_outputs_F)]
    
    ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_G = ort_session_G.get_inputs()
    out_name_G = ort_session_G.get_outputs()
    in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
    out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]

    input_feed_E = {
        in_name_E[num_keys_values_plus_3]: penality_value,
        in_name_E[num_keys_values_plus_4]: beam_size
    }

    input_feed_F = {
        in_name_F[num_keys_values_plus_5]: penality_value,
        in_name_F[num_keys_values_plus_6]: beam_size,
        in_name_F[num_keys_values_plus_7]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    input_feed_D = {in_name_D[2]: penality_value}


if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False


init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
if device_type != 'dml':
    init_past_keys_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[0].shape[1], 1, ort_session_C._inputs_meta[0].shape[3], 0), dtype=model_dtype), device_type, DEVICE_ID)
    init_past_values_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[num_layers].shape[1], 1, 0, ort_session_C._inputs_meta[num_layers].shape[4]), dtype=model_dtype), device_type, DEVICE_ID)
else:
    init_past_keys_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[0].shape[1], 1, ort_session_C._inputs_meta[0].shape[3], 0), dtype=model_dtype), 'cpu', 0)
    init_past_values_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[num_layers].shape[1], 1, 0, ort_session_C._inputs_meta[num_layers].shape[4]), dtype=model_dtype), 'cpu', 0)
init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
init_batch_size_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)


if USE_BEAM_SEARCH:
    input_feed_E[in_name_E[num_keys_values_plus_1]] = init_save_id_beam
    input_feed_E[in_name_E[num_keys_values_plus_2]] = init_repeat_penality
else:
    input_feed_D[in_name_D[1]] = init_repeat_penality
    input_feed_D[in_name_D[3]] = init_batch_size_greedy


if do_repeat_penality:
    if USE_BEAM_SEARCH:
        input_feed_G = {in_name_G[2]: penality_reset_count_beam_init}
    else:
        penality_reset_count_greedy = 0


# Start to run FunASR-Nano
input_feed_A = {}
input_feed_C = {}

init_all_outputs_B = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    input_feed_B = {in_name_B: input_ids}
    init_all_outputs_B.append(ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0])

# Load the input audio
for prompt_embed, test in zip(init_all_outputs_B, test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_NORMALIZER:
        audio = normalizer(audio, 8192.0)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in_A, str):
        INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_len)  # You can adjust it.
    else:
        INPUT_AUDIO_LENGTH = shape_value_in_A
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

    asr_result = ""
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    rtf_time = time.time()
    while slice_end <= aligned_len:
        input_feed_A[in_name_A[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start: slice_end], device_type, DEVICE_ID)
        input_feed_A[in_name_A[1]] = prompt_embed
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
        input_feed_C[in_name_C[num_keys_values]] = all_outputs_A[0]
        input_feed_C[in_name_C[num_keys_values_plus_1]] = init_history_len
        input_feed_C[in_name_C[num_keys_values_plus_2]] = all_outputs_A[1]
        input_feed_C[in_name_C[num_keys_values_plus_3]] = init_attention_mask_1
        for i in range(num_layers):
            input_feed_C[in_name_C[i]] = init_past_keys_C
        for i in range(num_layers, num_keys_values):
            input_feed_C[in_name_C[i]] = init_past_values_C

        if USE_BEAM_SEARCH:
            input_feed_E[in_name_E[num_keys_values_plus_1]] = init_save_id_beam
            input_feed_E[in_name_E[num_keys_values_plus_2]] = init_repeat_penality
            input_feed_G[in_name_G[2]] = penality_reset_count_beam_init
        else:
            input_feed_D[in_name_D[1]] = init_repeat_penality
            penality_reset_count_greedy = 0

        num_decode = 0
        limit = generate_limit - all_outputs_A[1].numpy()
        start_time = time.time()
        while num_decode < limit:
            all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    input_feed_E.update(zip(in_name_E[:num_keys_values_plus_1], all_outputs_C))
                    all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    max_logits_idx = all_outputs_E[num_keys_values_plus_5].numpy()
                    input_feed_F[in_name_F[num_keys_values_plus_4]] = all_outputs_E[num_keys_values_plus_4]
                    if do_repeat_penality:
                        input_feed_G[in_name_G[3]] = all_outputs_E[num_keys_values_plus_4]
                else:
                    input_feed_F.update(zip(in_name_F[:num_keys_values_plus_1], all_outputs_C))
                    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                    max_logits_idx = all_outputs_F[num_keys_values_plus_4].numpy()
                if max_logits_idx in STOP_TOKEN:
                    save_id = all_outputs_F[num_keys_values_plus_1].numpy()[0, :num_decode]  # 0 is the Top_1
                    asr_result += tokenizer.decode(save_id, skip_special_tokens=True)
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    input_feed_G[in_name_G[0]] = all_outputs_F[num_keys_values_plus_1]
                    input_feed_G[in_name_G[1]] = all_outputs_F[num_keys_values_plus_2]
                    all_outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)
                    input_feed_G[in_name_G[2]] = all_outputs_G[2]
                    input_feed_F[in_name_F[num_keys_values_plus_1]] = all_outputs_G[0]
                    input_feed_F[in_name_F[num_keys_values_plus_2]] = all_outputs_G[1]
                if num_decode < 1:
                    input_feed_C.update(zip(in_name_C[:num_keys_values], all_outputs_E))
                    input_feed_B[in_name_B] = all_outputs_E[num_keys_values]
                    input_feed_F[in_name_F[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_F[in_name_F[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
                    input_feed_F[in_name_F[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
                else:
                    input_feed_C.update(zip(in_name_C[:num_keys_values], all_outputs_F))
                    input_feed_B[in_name_B] = all_outputs_F[num_keys_values]
                    input_feed_F[in_name_F[num_keys_values_plus_1]] = all_outputs_F[num_keys_values_plus_1]
                    input_feed_F[in_name_F[num_keys_values_plus_2]] = all_outputs_F[num_keys_values_plus_2]
                    input_feed_F[in_name_F[num_keys_values_plus_3]] = all_outputs_F[num_keys_values_plus_3]
                input_feed_C[in_name_C[num_keys_values]] = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0]
            else:
                input_feed_D[in_name_D[0]] = all_outputs_C[num_keys_values]
                all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                max_logits_idx = all_outputs_D[0].numpy()[0, 0]
                if max_logits_idx in STOP_TOKEN:
                    asr_result += tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    reset_ids = save_id_greedy[penality_reset_count_greedy]
                    if reset_ids != max_logits_idx:
                        repeat_penality = all_outputs_D[1].numpy()
                        repeat_penality[:, reset_ids] = 1.0
                        input_feed_D[in_name_D[1]].update_inplace(repeat_penality)
                    penality_reset_count_greedy += 1
                else:
                    input_feed_D[in_name_D[1]] = all_outputs_D[1]
                input_feed_D[in_name_D[0]] = all_outputs_D[0]
                input_feed_C.update(zip(in_name_C[:num_keys_values], all_outputs_C))
                input_feed_B[in_name_B] = all_outputs_D[0]
                input_feed_C[in_name_C[num_keys_values]] = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)[0]
                save_id_greedy[num_decode] = max_logits_idx
            input_feed_C[in_name_C[num_keys_values_plus_1]] = all_outputs_C[num_keys_values_plus_1]
            if num_decode < 1:
                input_feed_C[in_name_C[num_keys_values_plus_2]] = init_ids_len_1
                input_feed_C[in_name_C[num_keys_values_plus_3]] = init_attention_mask_0
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")
    print(asr_result, end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_len / SAMPLE_RATE)):.3f}")
    print("----------------------------------------------------------------------------------------------------------")
