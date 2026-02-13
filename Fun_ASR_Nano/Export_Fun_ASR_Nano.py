import gc
import time
import torch
import torchaudio
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from funasr import AutoModel
from transformers import AutoTokenizer
from STFT_Process import STFT_Process                                                                    # The custom STFT/ISTFT can be exported in ONNX format.

model_path = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512'                                                 # Set the path where the [Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512] downloaded.  URL: https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 / https://modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512
tokenizer_path = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B'                                  # Set the tokenizer path.
onnx_model_A = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Encoder.onnx'                      # The exported onnx model path.
onnx_model_B = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Embed.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Main.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Greedy_Search.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/First_Beam_Search.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Second_Beam_Search.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Reset_Penality.onnx'
onnx_model_H = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Argmax.onnx'

# The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/yue.mp3", "./example/ja.mp3"]          # The test audio list.
task_prompt = ["将语音转写成中文：", "将语音转写成英文：", "将语音转写成粤语：", "将语音转写成日文："]              # The prompt of transcription task.


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
REPEAT_PENALITY = 1.0                                       # Range from 0.0 to 1.0; "1.0" means no penality.
PENALTY_RANGE = 10                                          # Penalizes the most recent output. "10" means the last 10 tokens.

# Runtime & Export Settings
PREVENT_F16_OVERFLOW = False                                # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization. Not recommended.
MAX_THREADS = 0                                             # Parllel CPU threads. Set 0 for auto.
DEVICE_ID = 0                                               # Default to zero.
OPSET = 17                                                  # ONNX Runtime opset version.
ORT_Accelerate_Providers = []


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


class ARGMAX(torch.nn.Module):
    def __init__(self):
        super(ARGMAX, self).__init__()
        pass

    def forward(self, logits):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True)
        return max_logits_idx.int()


class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()

    def forward(self, logits, repeat_penality, penality_value):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        repeat_penality.scatter_(1, max_logits_idx, repeat_penality.gather(1, max_logits_idx) * penality_value)
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        repeat_penality.scatter_(1, top_beam_indices, repeat_penality.gather(1, top_beam_indices) * penality_value)
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-7]
        save_id = all_inputs[-6]
        repeat_penality = all_inputs[-5]
        previous_prob = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=True)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        repeat_penality.scatter_(1, top_beam_indices, repeat_penality.gather(1, top_beam_indices) * penality_value)
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[0]
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count):
        token_indices = save_id.gather(1, penality_reset_count).long()
        repeat_penality.scatter_(1, token_indices, 1.0)
        penality_reset_count += 1
        return repeat_penality, penality_reset_count


class FUNASR_NANO_ENCODER(torch.nn.Module):
    def __init__(self, funasr_nano, stft_model, nfft_stft, max_stft_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, _tokenizer):
        super(FUNASR_NANO_ENCODER, self).__init__()
        self.funasr_nano = funasr_nano.float()
        self._replace_gelu_with_tanh_approximation(self.funasr_nano)
        self.stft_model = stft_model
        self.T_lfr = lfr_len
        self.lfr_n = lfr_n
        self.pre_emphasis = torch.tensor(pre_emphasis, dtype=torch.float32).view(1, 1, -1)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None,'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=max_stft_len + self.lfr_m_factor - 1).to(torch.int16)
        self.output_size_factor = self.funasr_nano.audio_encoder.output_size() ** 0.5
        self.variance_epsilon = torch.tensor([1e-7], dtype=torch.float32)
        self.position_encoding = self.funasr_nano.audio_encoder.embed(torch.zeros([1, max_stft_len, 560], dtype=torch.float32))
        num_head = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.d_k
        self.pad_zeros = torch.zeros((1, num_head * head_dim, 5), dtype=torch.float32)
        scale_factor = head_dim ** (-0.25)
        self.total_encoders = list(self.funasr_nano.audio_encoder.encoders0) + list(self.funasr_nano.audio_encoder.encoders) + list(self.funasr_nano.audio_encoder.tp_encoders)
        in_size = self.funasr_nano.audio_encoder.encoders._modules["0"].in_size
        for encoder_layer in self.total_encoders:
            encoder_layer.self_attn.linear_q_k_v.weight.data[:-in_size] *= scale_factor
            encoder_layer.self_attn.linear_q_k_v.bias.data[:-in_size] *= scale_factor

            # Fuse encoder_layer.norm1 into encoder_layer.self_attn.linear_q_k_v
            norm = encoder_layer.norm1
            linear = encoder_layer.self_attn.linear_q_k_v

            # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters in LayerNorm so it only performs (x - mu) / sigma
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

            # Fuse encoder_layer.norm2 into encoder_layer.feed_forward.w_1
            norm = encoder_layer.norm2
            linear = encoder_layer.feed_forward.w_1

            # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters in LayerNorm
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

        head_dim = self.funasr_nano.audio_adaptor.blocks._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        for block in self.funasr_nano.audio_adaptor.blocks:
            block.self_attn.linear_q.weight.data *= factor
            block.self_attn.linear_q.bias.data *= factor
            block.self_attn.linear_k.weight.data *= factor
            block.self_attn.linear_k.bias.data *= factor

            # Fusing q, k, v
            in_features = block.self_attn.linear_q.in_features
            out_features = block.self_attn.linear_q.out_features + block.self_attn.linear_k.out_features + block.self_attn.linear_v.out_features
            block.self_attn.linear_q_k_v = torch.nn.Linear(in_features, out_features, bias=True)
            block.self_attn.linear_q_k_v.weight.data = torch.cat([block.self_attn.linear_q.weight.data, block.self_attn.linear_k.weight.data, block.self_attn.linear_v.weight.data], dim=0)
            block.self_attn.linear_q_k_v.bias.data = torch.cat([block.self_attn.linear_q.bias.data, block.self_attn.linear_k.bias.data, block.self_attn.linear_v.bias.data], dim=0)
            block.self_attn.size = out_features // 3
            del block.self_attn.linear_q
            del block.self_attn.linear_k
            del block.self_attn.linear_v

            # Fuse block.norm1 into block.self_attn.linear_q_k_v
            norm = block.norm1
            linear = block.self_attn.linear_q_k_v

            # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters in LayerNorm so it only performs (x - mu) / sigma
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

            # Fuse block.norm2 into block.feed_forward.w_1
            norm = block.norm2
            linear = block.feed_forward.w_1

            # 1. Update Bias
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

        # Fuse audio_encoder.tp_norm into audio_adaptor.linear1
        norm = self.funasr_nano.audio_encoder.tp_norm
        linear = self.funasr_nano.audio_adaptor.linear1

        # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

        # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

        # 3. Disable affine parameters in LayerNorm so it only performs (x - mu) / sigma
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias = None

        head_ids = _tokenizer.encode("<|im_start|>user\n", return_tensors="pt")
        tail_ids = _tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        self.head_embed = self.funasr_nano.llm.model.embed_tokens(head_ids)
        self.tail_embed = self.funasr_nano.llm.model.embed_tokens(tail_ids)
        self.fake_token = torch.zeros(max_stft_len + 1, dtype=torch.int16)
        for i in range(self.fake_token.shape[0]):
            self.fake_token[i] = (((i - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)
    
    def forward(self, audio, query_embed):
        audio = audio.float()
        audio = audio - torch.mean(audio)  # Remove DC Offset
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[..., :1], audio[..., 1:] - self.pre_emphasis * audio[..., :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = (torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2) + self.variance_epsilon).log() * self.output_size_factor
        features_len = mel_features.shape[1].unsqueeze(0)
        left_padding = mel_features[:, [0]]
        padded_inputs = torch.cat([left_padding] * self.lfr_m_factor + [mel_features], dim=1)
        _len = features_len // self.lfr_n - 1
        mel_features = padded_inputs[:, self.indices_mel[:_len].int()].reshape(1, _len, -1)
        x = mel_features + self.position_encoding[:, :_len].float()
        for encoder_layer in self.funasr_nano.audio_encoder.encoders0 + self.funasr_nano.audio_encoder.encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0, 3)
            q_h, k_h, v_h = qkv.split([1, 1, 1], dim=0)
            v = v_h.transpose(1, 2).reshape(1, -1, encoder_layer.size)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.softmax(torch.matmul(q_h, k_h.transpose(-1, -2)), dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v_h).transpose(1, 2).reshape(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            if encoder_layer.in_size == encoder_layer.size:
                x += attn
            else:
                x = attn
            x = x + encoder_layer.feed_forward.w_2(encoder_layer.feed_forward.activation(encoder_layer.feed_forward.w_1(encoder_layer.norm2(x))))
        x = self.funasr_nano.audio_encoder.after_norm(x)
        for encoder_layer in self.funasr_nano.audio_encoder.tp_encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0, 3)
            q_h, k_h, v_h = qkv.split([1, 1, 1], dim=0)
            v = v_h.transpose(1, 2).reshape(1, -1, encoder_layer.size)
            fsmn_memory = encoder_layer.self_attn.fsmn_block(torch.cat([self.pad_zeros, v.transpose(1, 2), self.pad_zeros], dim=-1)).transpose(1, 2) + v
            attn = torch.softmax(torch.matmul(q_h, k_h.transpose(-1, -2)), dim=-1)
            attn = torch.matmul(attn, v_h).transpose(1, 2).reshape(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            x += attn
            x = x + encoder_layer.feed_forward.w_2(encoder_layer.feed_forward.activation(encoder_layer.feed_forward.w_1(encoder_layer.norm2(x))))
        x = self.funasr_nano.audio_encoder.tp_norm(x)
        x = self.funasr_nano.audio_adaptor.linear1(x)
        x = self.funasr_nano.audio_adaptor.relu(x)
        x = self.funasr_nano.audio_adaptor.linear2(x)
        for block in self.funasr_nano.audio_adaptor.blocks:
            x1 = block.norm1(x)
            qkv = block.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, block.self_attn.h, block.self_attn.d_k).permute(1, 2, 0, 3)
            q, k, v = qkv.split([1, 1, 1], dim=0)
            attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
            attn = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, block.self_attn.linear_out.in_features)
            attn = block.self_attn.linear_out(attn)
            x += attn
            x = x + block.feed_forward.w_2(block.feed_forward.activation(block.feed_forward.w_1(block.norm2(x))))
        x = x[:, :self.fake_token[features_len].to(torch.int64)]
        concat_embed = torch.cat([self.head_embed, query_embed, x, self.tail_embed], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


class FUNASR_NANO_DECODER_EMBED(torch.nn.Module):
    def __init__(self, funasr_nano):
        super(FUNASR_NANO_DECODER_EMBED, self).__init__()
        self.funasr_nano_decoder_embed = funasr_nano.llm.model.embed_tokens.float()
        
    def forward(self, input_ids):
        return self.funasr_nano_decoder_embed(input_ids)


class FUNASR_NANO_DECODER_MAIN(torch.nn.Module):
    def __init__(self, funasr_nano, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super(FUNASR_NANO_DECODER_MAIN, self).__init__()
        self.funasr_nano = funasr_nano.llm.float()
        self._replace_gelu_with_tanh_approximation(self.funasr_nano)
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        
        # Scaling Factors
        scale_factor = head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = head_dim ** 0.5
        
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        idx_theta = (position_ids * self.funasr_nano.model.rotary_emb.inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        self.cos_rotary_pos_emb = torch.cat([cos, cos], dim=-1).half()
        self.sin_rotary_pos_emb = torch.cat([-sin, sin], dim=-1).half()
        
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        # --- Fuse / Rearrange weights ---
        with torch.no_grad():
            for layer in self.funasr_nano.model.layers:
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj

                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)

                in_features = int(q_proj.in_features)
                out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
                has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)

                # 1) qkv Fusion
                qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

                if has_bias:
                    dtype = qkv.weight.dtype
                    qb = q_proj.bias if q_proj.bias is not None else torch.zeros(q_proj.out_features, dtype=dtype)
                    kb = k_proj.bias if k_proj.bias is not None else torch.zeros(k_proj.out_features, dtype=dtype)
                    vb = v_proj.bias if v_proj.bias is not None else torch.zeros(v_proj.out_features, dtype=dtype)
                    qkv.bias.copy_(torch.cat([qb, kb, vb], dim=0))
                
                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj

                # 2) QK Norm Fusion (Combine Q and K norm weights)
                layer.self_attn.q_norm.weight.mul_(scale_factor * norm_factor_qk)
                layer.self_attn.k_norm.weight.mul_(scale_factor * norm_factor_qk)
                
                q_norm_w = layer.self_attn.q_norm.weight.repeat(self.num_heads)
                k_norm_w = layer.self_attn.k_norm.weight.repeat(self.num_key_value_heads)
                
                # Shape: [1, 1, num_heads + num_kv_heads, head_dim] for broadcasting
                layer.self_attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_w, k_norm_w], dim=0).view(1, 1, 1, -1, self.head_dim))
                del layer.self_attn.q_norm
                del layer.self_attn.k_norm

                # 3) Fuse input rms norm into qkv
                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                layer.self_attn.qkv = qkv
                del layer.input_layernorm

                # 4) Fuse post attn norm into MLP
                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj

                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)

                gate_weight = gate.weight * w
                up_weight = up.weight * w
                gate_up.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))

                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj
                del layer.mlp.up_proj
                del layer.post_attention_layernorm

            # 5) Fuse final norm
            w = self.funasr_nano.model.norm.weight.unsqueeze(0) * norm_factor
            self.funasr_nano.lm_head.weight.mul_(w)
            del self.funasr_nano.model.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def rotate_half(self, x, batch_size):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        mask = all_inputs[-1]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * mask).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.funasr_nano.model.layers):
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            if PREVENT_F16_OVERFLOW:
                qk = qk * self.overflow_scale
            qk = qk * torch.rsqrt(qk.square().sum(dim=-1, keepdim=True)) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self.rotate_half(qk, batch_size) * rotary_pos_emb_sin
            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            k = k.half().permute(0, 3, 2, 4, 1)
            v = v.half().transpose(1, 3)
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            attn = torch.matmul(q, k.float())
            attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v.float())
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        hidden_states = hidden_states[:, -1]
        if PREVENT_F16_OVERFLOW:
            hidden_states = hidden_states * self.overflow_scale
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
        logits = self.funasr_nano.lm_head(hidden_states)
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
    past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=torch.float16)
    past_values = torch.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=torch.float16)
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

    model_C = FUNASR_NANO_DECODER_MAIN(model.model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size)
    torch.onnx.export(
        model_C,
        tuple(all_inputs),
        onnx_model_C,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
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

    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_reset_count = torch.zeros([beam_size, 1], dtype=torch.int32)
    logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)

    torch.onnx.export(
        GREEDY_SEARCH(),
        (logits, repeat_penality, penality_value),
        # Reuse the beam_size tensor as batch_size during export process.
        onnx_model_D,
        input_names=['logits', 'repeat_penality_in', 'penality_value'],
        output_names=['max_logits_idx', 'repeat_penality_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )

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
    output_names.append('max_logits_idx')
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['repeat_penality_in'] = {0: 'batch'}
    dynamic_axes['repeat_penality_out'] = {0: 'batch'}
    dynamic_axes['logits'] = {0: 'batch'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}

    torch.onnx.export(
        FIRST_BEAM_SEARCH(num_layers + num_layers),
        tuple(all_inputs),
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )

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
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}

    torch.onnx.export(
        SECOND_BEAM_SEARCH(num_layers + num_layers),
        tuple(all_inputs),
        onnx_model_F,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del num_layers
    del past_keys
    del past_values
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del beam_size
    del penality_value

    torch.onnx.export(
        RESET_PENALITY(),
        (save_id, repeat_penality, penality_reset_count),
        onnx_model_G,
        input_names=['save_id_in', 'repeat_penality_in', 'penality_reset_count_in'],
        output_names=['repeat_penality_out', 'penality_reset_count_out'],
        dynamic_axes={
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'penality_reset_count_in': {0: 'batch'},
            'penality_reset_count_out': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del save_id
    del repeat_penality
    del penality_reset_count

    torch.onnx.export(
        ARGMAX(),
        (logits,),
        onnx_model_H,
        input_names=['logits'], output_names=['max_logits_idx'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del logits
print('\nExport done!\n\nStart to run FunASR-Nano by ONNX Runtime.\n\nNow, loading the model...')


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

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
run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,  # The default value is 8. Edit freely.
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '0',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
    provider_options = None
    

def bind_ort_values(binding, names, values, num=0):
    if num != 0:
        for i in range(num):
            binding.bind_ortvalue_input(names[i], values[i])
    else:
        for name, val in zip(names, values):
            binding.bind_ortvalue_input(name, val)


def bind_outputs_generic(binding, output_names, device_type):
    for name in output_names:
        binding._iobinding.bind_output(name, device_type)


def create_ortvalue(data, dtype, device_type, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device_type, device_id)


_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_A = ort_session_A.io_binding()
shape_value_in_A = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = [x.name for x in ort_session_A.get_inputs()]
out_name_A = [x.name for x in ort_session_A.get_outputs()]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_B = ort_session_B.io_binding()
in_name_B = ort_session_B.get_inputs()[0].name
out_name_B = [ort_session_B.get_outputs()[0].name]

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_C = ort_session_C.io_binding()
print(f"\nUsable Providers: {ort_session_C.get_providers()}")

in_meta_C = ort_session_C.get_inputs()
out_meta_C = ort_session_C.get_outputs()
in_name_C = [x.name for x in in_meta_C]
out_name_C = [x.name for x in out_meta_C]
amount_of_outputs_C = len(out_name_C)
in_name_C_parts = in_name_C[:-2]

num_layers = (amount_of_outputs_C - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
vocab_size = ort_session_C._outputs_meta[num_keys_values].shape[1]

generate_limit = MAX_SEQ_LEN - 20
topK = create_ortvalue([TOP_K], np.int64, device_type, DEVICE_ID)
beam_size = create_ortvalue([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")
    BEAM_SIZE = 1

do_repeat_penalty = (REPEAT_PENALITY != 1.0)

if USE_BEAM_SEARCH:
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_E = ort_session_E.io_binding()
    in_name_E = [x.name for x in ort_session_E.get_inputs()]
    out_name_E = [x.name for x in ort_session_E.get_outputs()]
    in_name_E_parts = in_name_E[:num_keys_values_plus_1]
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_F = ort_session_F.io_binding()
    in_name_F = [x.name for x in ort_session_F.get_inputs()]
    out_name_F = [x.name for x in ort_session_F.get_outputs()]
    in_name_F_parts = in_name_F[:num_keys_values_plus_1]
    ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_G = ort_session_G.io_binding()
    in_name_G = [x.name for x in ort_session_G.get_inputs()]
    out_name_G = [x.name for x in ort_session_G.get_outputs()]
    penality_dtype = np.float16 if 'float16' in ort_session_E._inputs_meta[num_keys_values_plus_4].type else np.float32
    penality_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
    init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
    init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
    init_penality_reset_count = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros([BEAM_SIZE, 1], dtype=np.int32), device_type, DEVICE_ID)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], init_save_id_beam)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], init_repeat_penality)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_3], penality_value)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_4], beam_size)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_4], penality_value)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_5], beam_size)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_6], topK)
    binding_G.bind_ortvalue_input(in_name_G[2], init_penality_reset_count)
else:
    BEAM_SIZE = 1
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if do_repeat_penalty:
        ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_D = ort_session_D.io_binding()
        in_name_D = [x.name for x in ort_session_D.get_inputs()]
        out_name_D = [x.name for x in ort_session_D.get_outputs()]
        penality_dtype = np.float16 if 'float16' in ort_session_D._inputs_meta[2].type else np.float32
        penality_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
        penalty_shape = (BEAM_SIZE, vocab_size)
        init_penalty = np.ones(penalty_shape, dtype=penality_dtype)
        current_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(init_penalty, device_type, DEVICE_ID)
        next_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(init_penalty, device_type, DEVICE_ID)
        binding_D.bind_ortvalue_input(in_name_D[2], penality_value)
        binding_D.bind_output(name=out_name_D[0], device_type=device_type, device_id=DEVICE_ID)
        binding_D.bind_output(name=out_name_D[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
        init_penality_reset_count = 0
    else:
        ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_H = ort_session_H.io_binding()
        in_name_H = ort_session_H.get_inputs()[0].name
        out_name_H = [ort_session_H.get_outputs()[0].name]

init_ids_len_1 = create_ortvalue([1], np.int64, device_type, DEVICE_ID)
init_history_len = create_ortvalue([0], np.int64, device_type, DEVICE_ID)
init_attention_mask_0 = create_ortvalue([0], np.int8, device_type, DEVICE_ID)
init_attention_mask_1 = create_ortvalue([1], np.int8, device_type, DEVICE_ID)

if 'dml' in device_type:
    kv_device = 'cpu'
    kv_device_id = 0
else:
    kv_device = device_type
    kv_device_id = DEVICE_ID

init_past_keys_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[0].shape[1], 1, ort_session_C._inputs_meta[0].shape[3], 0), dtype=np.float16), kv_device, kv_device_id)
init_past_values_C = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_C._inputs_meta[num_layers].shape[1], 1, 0, ort_session_C._inputs_meta[num_layers].shape[4]), dtype=np.float16), kv_device, kv_device_id)

init_all_outputs_B = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    binding_B.bind_ortvalue_input(in_name_B, input_ids)
    bind_outputs_generic(binding_B, out_name_B, _ort_device_type)
    ort_session_B.run_with_iobinding(binding_B, run_options=run_options)
    init_all_outputs_B.append(onnxruntime.OrtValue.ortvalue_from_numpy(binding_B.get_outputs()[0].numpy(), device_type, DEVICE_ID))

for prompt_embed, test in zip(init_all_outputs_B, test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_NORMALIZER: 
        audio = normalizer(audio, 8192.0)
    audio_full_len = len(audio)
    INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_full_len) if isinstance(shape_value_in_A, str) else shape_value_in_A
    stride_step = INPUT_AUDIO_LENGTH if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    audio = audio.reshape(1, 1, -1)
    if audio_full_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_full_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        pad_amount = ((num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH) - audio_full_len
        zeros = np.zeros([1, 1, pad_amount], dtype=audio.dtype)
        audio = np.concatenate((audio, zeros), axis=-1)
    elif audio_full_len < INPUT_AUDIO_LENGTH:
        zeros = np.zeros([1, 1, INPUT_AUDIO_LENGTH - audio_full_len], dtype=audio.dtype)
        audio = np.concatenate((audio, zeros), axis=-1)
    aligned_len = audio.shape[-1]
    asr_result = ""
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    rtf_time = time.time()
    while slice_end <= aligned_len:
        audio_slice = audio[..., slice_start: slice_end]
        ort_audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio_slice, device_type, DEVICE_ID)
        binding_A.bind_ortvalue_input(in_name_A[0], ort_audio)
        binding_A.bind_ortvalue_input(in_name_A[1], prompt_embed)
        bind_outputs_generic(binding_A, out_name_A, _ort_device_type)
        ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
        all_outputs_A = binding_A.get_outputs()
        i = 0
        j = num_layers
        while i < j:
            binding_C.bind_ortvalue_input(in_name_C[i], init_past_keys_C)
            i += 1
        j = i + num_layers
        while i < j:
            binding_C.bind_ortvalue_input(in_name_C[i], init_past_values_C)
            i += 1
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values], all_outputs_A[0])
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_1], init_history_len)
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_2], all_outputs_A[1])
        binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_3], init_attention_mask_1)
        if USE_BEAM_SEARCH:
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], init_save_id_beam)
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], init_repeat_penality)
            if do_repeat_penalty:
                binding_G.bind_ortvalue_input(in_name_G[2], init_penality_reset_count)
        else:
            if do_repeat_penalty:
                current_penalty.update_inplace(init_penalty)
                next_penalty.update_inplace(init_penalty)
                binding_D.bind_output(name=out_name_D[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
                init_penality_reset_count = 0

        num_decode = 0
        limit = generate_limit - all_outputs_A[1].numpy()
        start_time = time.time()
        while num_decode < limit:
            bind_outputs_generic(binding_C, out_name_C, _ort_device_type)
            ort_session_C.run_with_iobinding(binding_C, run_options=run_options)
            all_outputs_C = binding_C.get_outputs()
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    bind_ort_values(binding_E, in_name_E_parts, all_outputs_C)
                    bind_outputs_generic(binding_E, out_name_E, _ort_device_type)
                    ort_session_E.run_with_iobinding(binding_E, run_options=run_options)
                    all_outputs_E = binding_E.get_outputs()
                    max_logits_idx = all_outputs_E[num_keys_values_plus_4].numpy()
                    if max_logits_idx in STOP_TOKEN:
                        print("\nBad first token generated, stopping decoding.\n")
                        break
                else:
                    bind_ort_values(binding_F, in_name_F_parts, all_outputs_C)
                    bind_outputs_generic(binding_F, out_name_F, _ort_device_type)
                    ort_session_F.run_with_iobinding(binding_F, run_options=run_options)
                    all_outputs_F = binding_F.get_outputs()
                    max_logits_idx = all_outputs_F[num_keys_values_plus_4].numpy()
                    if max_logits_idx in STOP_TOKEN:
                        break
                if do_repeat_penalty and (num_decode >= PENALTY_RANGE):
                    binding_G.bind_ortvalue_input(in_name_G[0], all_outputs_F[num_keys_values_plus_1])
                    binding_G.bind_ortvalue_input(in_name_G[1], all_outputs_F[num_keys_values_plus_2])
                    bind_outputs_generic(binding_G, out_name_G, _ort_device_type)
                    ort_session_G.run_with_iobinding(binding_G, run_options=run_options)
                    all_outputs_G = binding_G.get_outputs()
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_2], all_outputs_G[0])
                    binding_G.bind_ortvalue_input(in_name_G[2], all_outputs_G[1])
                if num_decode < 1:
                    bind_ort_values(binding_C, in_name_C_parts, all_outputs_E)
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_E[num_keys_values])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_1], all_outputs_E[num_keys_values_plus_1])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_2], all_outputs_E[num_keys_values_plus_2])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_3], all_outputs_E[num_keys_values_plus_3])
                else:
                    bind_ort_values(binding_C, in_name_C_parts, all_outputs_F)
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_F[num_keys_values])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_1], all_outputs_F[num_keys_values_plus_1])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_2], all_outputs_F[num_keys_values_plus_2])
                    binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_3], all_outputs_F[num_keys_values_plus_3])
            else:
                if do_repeat_penalty:
                    binding_D.bind_ortvalue_input(in_name_D[0], all_outputs_C[num_keys_values])
                    binding_D.bind_ortvalue_input(in_name_D[1], current_penalty)
                    ort_session_D.run_with_iobinding(binding_D, run_options=run_options)
                    all_outputs_D = binding_D.get_outputs()
                    max_logits_idx = all_outputs_D[0].numpy().flat[0]
                    if max_logits_idx in STOP_TOKEN:
                        break
                    if num_decode >= PENALTY_RANGE:
                        reset_ids = save_id_greedy[init_penality_reset_count]
                        if reset_ids != max_logits_idx:
                            tmp = next_penalty.numpy()
                            tmp[:, reset_ids] = 1.0
                            next_penalty.update_inplace(tmp)
                        init_penality_reset_count += 1
                    current_penalty, next_penalty = next_penalty, current_penalty
                    binding_D.bind_output(name=out_name_D[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_D[0])
                else:
                    binding_H.bind_ortvalue_input(in_name_H, all_outputs_C[num_keys_values])
                    bind_outputs_generic(binding_H, out_name_H, _ort_device_type)
                    ort_session_H.run_with_iobinding(binding_H)
                    all_outputs_H = binding_H.get_outputs()
                    binding_B.bind_ortvalue_input(in_name_B, all_outputs_H[0])
                    max_logits_idx = all_outputs_H[0].numpy().flat[0]
                    if max_logits_idx in STOP_TOKEN:
                        break
                bind_ort_values(binding_C, in_name_C_parts, all_outputs_C)
                save_id_greedy[num_decode] = max_logits_idx
            bind_outputs_generic(binding_B, out_name_B, _ort_device_type)
            ort_session_B.run_with_iobinding(binding_B)
            binding_C.bind_ortvalue_input(in_name_C[num_keys_values], binding_B.get_outputs()[0])
            binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_1], all_outputs_C[num_keys_values_plus_1])
            if num_decode < 1:
                binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_2], init_ids_len_1)
                binding_C.bind_ortvalue_input(in_name_C[num_keys_values_plus_3], init_attention_mask_0)
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        if num_decode > 0:
            if USE_BEAM_SEARCH:
                asr_result += tokenizer.decode(all_outputs_F[num_keys_values_plus_1].numpy()[0, :num_decode], skip_special_tokens=True)
            else:
                asr_result += tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")
    print(asr_result, end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_full_len / SAMPLE_RATE)):.3f}")
    print("----------------------------------------------------------------------------------------------------------")

