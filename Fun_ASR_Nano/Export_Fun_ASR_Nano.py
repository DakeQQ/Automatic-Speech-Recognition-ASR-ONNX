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
from STFT_Process import STFT_Process

model_path                      = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512'                                # Set the path where the [Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512] downloaded.  URL: https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 / https://modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512
tokenizer_path                  = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B'                     # Set the tokenizer path.
onnx_model_Encoder              = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Encoder.onnx'       # The exported onnx model path.
onnx_model_Embed                = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Embed.onnx'
onnx_model_Main                 = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/FunASR_Nano_Decoder_Main.onnx'
onnx_model_Rotary_Mask_Prefill  = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Rotary_Mask_Text_Prefill.onnx'
onnx_model_Rotary_Mask_Decode   = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Rotary_Mask_Text_Decode.onnx'
onnx_model_Greedy               = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam           = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam          = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty              = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax               = r'/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX/Argmax.onnx'


# The exported onnx model path.
test_audio = ["./example/zh.mp3", "./example/en.mp3", "./example/yue.mp3", "./example/ja.mp3"]          # The test audio list.
task_prompt = ["将语音转写成中文：", "将语音转写成英文：", "将语音转写成粤语：", "将语音转写成日文："]              # The prompt of transcription task.


if "MLT" in model_path:
    test_audio += ["./example/ko.mp3"]
    task_prompt += ["将语音转写成韩文："]


# Audio & STFT Configuration
SAMPLE_RATE = 16000                 # The model parameter, do not edit the value.
WINDOW_TYPE = 'hamming'             # Type of window function used in the STFT
N_MELS = 80                         # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 400                     # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                 # Length of windowing, edit it carefully.
HOP_LENGTH = 160                    # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                # For audio preprocessing.
USE_NORMALIZER = True               # If true, use the audio normalizer to make the loudness consistent.

# Model Parameters
LFR_M = 7                           # The model parameter, do not edit the value.
LFR_N = 6                           # The model parameter, do not edit the value.
STOP_TOKEN = [151643, 151645]       # The stop_id in Qwen is "151643" & "151645"
MAX_SEQ_LEN = 1024                  # The max context length.
PREVENT_F16_OVERFLOW = False        # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization. Not recommended.

# Input & Processing Limits
MAX_INPUT_AUDIO_LENGTH = 320000     # The maximum input audio length.
SLIDING_WINDOW = 0                  # Set the sliding window step for test audio reading; use 0 to disable.
DYNAMIC_AXES = True                 # The default dynamic_axes is the input audio length. Note that some providers only support static axes.

# Decoding Strategy
USE_BEAM_SEARCH = False             # Use beam search or greedy search. It recommended to use greedy search for Fun-ASR-Nano.
TOP_K = 3                           # The top k candidate in decoding.
BEAM_SIZE = 3                       # Number of beams in searching.
PENALTY_RANGE = 10                  # Penalizes the most recent output. "10" means the last 10 tokens.
MAX_BEAM_SIZE = 10                  # Max beams for exported model.
REPEAT_PENALTY = 1.0                # Range from 0.0 to 1.0; "1.0" means no penalty.

# Runtime & Export Settings
ORT_Accelerate_Providers = []       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG = False                     # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS = 0                     # Parllel CPU threads. Set 0 for auto.
DEVICE_ID = 0                       # Default to zero.
OPSET = 17                          # ONNX Runtime opset version.


MAX_STFT_SIGNAL_LENGTH = MAX_INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
LFR_LENGTH = (MAX_STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > MAX_INPUT_AUDIO_LENGTH:
    HOP_LENGTH = MAX_INPUT_AUDIO_LENGTH


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: select the token with the highest logit."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=False)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)

        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx
        )


# ══════════════════════════════════════════════════════════════════════════════
# Penalty & Utility Modules
# ══════════════════════════════════════════════════════════════════════════════
class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask
# ══════════════════════════════════════════════════════════════════════════════
class ROTARY_MASK_PREFILL(torch.nn.Module):
    """Precompute rotary embeddings and causal mask for the prefill phase."""

    def __init__(self, llm, max_seq_len):
        super().__init__()

        # Causal attention mask: upper triangle → -128
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128

        # Precompute rotary embeddings
        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    @staticmethod
    def _build_rotary_table(llm, max_seq_len):
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(idx_theta), torch.sin(idx_theta)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_MASK_DECODE(torch.nn.Module):
    """Provide rotary embeddings for a single decode step."""

    def __init__(self, llm, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len_next].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len_next].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Encoder Module (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
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
            v_fsmn = v_h.transpose(-1, -2).reshape(1, encoder_layer.size, -1)
            fsmn_in = torch.cat([self.pad_zeros, v_fsmn, self.pad_zeros], dim=-1)
            fsmn_out = encoder_layer.self_attn.fsmn_block(fsmn_in)
            fsmn_memory = (fsmn_out + v_fsmn).transpose(1, 2).reshape(1, -1, encoder_layer.size)
            attn = torch.matmul(q_h, k_h.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v_h).transpose(1, 2).reshape(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn_out = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            if encoder_layer.in_size == encoder_layer.size:
                x = x + attn_out
            else:
                x = attn_out
            x = x + encoder_layer.feed_forward.w_2(encoder_layer.feed_forward.activation(encoder_layer.feed_forward.w_1(encoder_layer.norm2(x))))
        x = self.funasr_nano.audio_encoder.after_norm(x)
        for encoder_layer in self.funasr_nano.audio_encoder.tp_encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0, 3)
            q_h, k_h, v_h = qkv.split([1, 1, 1], dim=0)
            v_fsmn = v_h.transpose(-1, -2).reshape(1, encoder_layer.size, -1)
            fsmn_in = torch.cat([self.pad_zeros, v_fsmn, self.pad_zeros], dim=-1)
            fsmn_out = encoder_layer.self_attn.fsmn_block(fsmn_in)
            fsmn_memory = (fsmn_out + v_fsmn).transpose(1, 2).reshape(1, -1, encoder_layer.size)
            attn = torch.matmul(q_h, k_h.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v_h).transpose(1, 2).reshape(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn_out = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            x = x + attn_out
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
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, block.self_attn.linear_out.in_features)
            attn_out = block.self_attn.linear_out(attn)
            x = x + attn_out
            x = x + block.feed_forward.w_2(block.feed_forward.activation(block.feed_forward.w_1(block.norm2(x))))
        x = x[:, :self.fake_token[features_len].to(torch.int64)]
        concat_embed = torch.cat([self.head_embed, query_embed, x, self.tail_embed], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Module
# ══════════════════════════════════════════════════════════════════════════════
class FUNASR_NANO_DECODER_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, funasr_nano):
        super(FUNASR_NANO_DECODER_EMBED, self).__init__()
        self.funasr_nano_decoder_embed = funasr_nano.llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.funasr_nano_decoder_embed(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Main Transformer Decoder Module
# ══════════════════════════════════════════════════════════════════════════════
class FUNASR_NANO_DECODER_MAIN(torch.nn.Module):
    """
    Main transformer decoder module that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (RoPE) received as external inputs
      - F16 KV cache management
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
    """

    def __init__(self, funasr_nano, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super(FUNASR_NANO_DECODER_MAIN, self).__init__()
        self.funasr_nano = funasr_nano.llm.float()
        self._replace_gelu_with_tanh_approximation(self.funasr_nano)

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = num_heads + num_key_value_heads

        # ── Layer count ──────────────────────────────────────────────────
        self.num_layers = num_layers

        # ── Overflow guard ───────────────────────────────────────────────
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self._fuse_weights(hidden_size)

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """
        Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP.
        """
        scale_factor = self.head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.funasr_nano.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # Absorb final RMSNorm into lm_head
            final_norm_weight = self.funasr_nano.model.norm.weight.unsqueeze(0) * norm_factor
            self.funasr_nano.lm_head.weight.mul_(final_norm_weight)
            del self.funasr_nano.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        # Store split dimensions for later use
        attn.q_out_features = int(q_proj.out_features)
        attn.k_out_features = int(k_proj.out_features)
        attn.v_out_features = int(v_proj.out_features)
        attn.qkv_in_features = in_features

        del attn.q_proj, attn.k_proj, attn.v_proj

        # ── Fuse QK norms (absorb scale factors) ────────────────────
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)

        q_norm_repeated = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        # ── Absorb input LayerNorm into QKV weights ─────────────────
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up = layer.mlp.gate_proj, layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Recursively replace exact GELU with tanh-approximated GELU for ONNX compatibility."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                FUNASR_NANO_DECODER_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def _rotate_half(self, x, batch_size):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime.
        """
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0]

        for i, layer in enumerate(self.funasr_nano.model.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            # Fused QKV projection & reshape
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)

            # QK normalization & rotary embedding
            qk = self._rms_norm(qk) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # Split into query and key, reshape query for GQA
            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)

            # F16 KV cache
            k = k.half().permute(0, 3, 2, 4, 1)
            v = v.half().transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v

            attn = torch.matmul(q, k.float()) + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v.float())

            # Output projection & residual
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(hidden_states[:, -1])
        logits = self.funasr_nano.lm_head(hidden_states)

        return *self.save_key, *self.save_value, logits


print('\nExport start ...\n')
with torch.inference_mode():

    # ══════════════════════════════════════════════════════════════════
    # Load Model & Extract Config
    # ══════════════════════════════════════════════════════════════════
    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, win_length=WINDOW_LENGTH, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()
    model = AutoModel(
        model=model_path,
        trust_remote_code=True,
        remote_code="./modeling_modified/model.py",
        device="cpu",
        disable_update=True
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    llm_config   = model.model.llm.config
    llm_model    = model.model.llm.model
    num_layers   = llm_config.num_hidden_layers
    num_heads    = llm_config.num_attention_heads
    num_kv_heads = llm_config.num_key_value_heads
    head_dim     = llm_config.head_dim
    vocab_size   = llm_model.vocab_size
    hidden_size  = llm_model.embed_tokens.embedding_dim

    # ══════════════════════════════════════════════════════════════════
    # Build Dummy Tensors for Tracing
    # ══════════════════════════════════════════════════════════════════
    batch_size  = BEAM_SIZE
    ids_len     = torch.tensor([10], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    kv_seq_len  = ids_len + history_len
    beam_size   = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    logits      = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)

    # KV cache spec: list of (name, concat_dim) — F16 only for FunASR-Nano
    kv_specs = [('key', 4), ('value', 3)]
    kv_dtype = torch.float16

    kv_tensors = {
        'key':   torch.zeros((batch_size, num_kv_heads, 1, head_dim, history_len), dtype=kv_dtype),
        'value': torch.zeros((batch_size, num_kv_heads, 1, history_len, head_dim), dtype=kv_dtype)
    }

    # ══════════════════════════════════════════════════════════════════
    # Helper: Build KV I/O names, tensors, and dynamic axes
    # ══════════════════════════════════════════════════════════════════
    def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
        inputs, in_names, out_names, axes = [], [], [], {}
        for name, dim in kv_specs:
            tensor = tensors_dict[name]
            for i in range(num_layers):
                in_n  = f'in_{name}_{i}'
                out_n = f'out_{name}_{i}'
                inputs.append(tensor)
                in_names.append(in_n)
                out_names.append(out_n)
                axes[in_n]  = {0: batch_axis, dim: seq_axis}
                axes[out_n] = {0: batch_axis, dim: out_seq_axis}
        return inputs, in_names, out_names, axes

    # ══════════════════════════════════════════════════════════════════
    # Export: Encoder
    # ══════════════════════════════════════════════════════════════════
    funasr_nano_encoder = FUNASR_NANO_ENCODER(
        model.model, custom_stft, NFFT_STFT, MAX_STFT_SIGNAL_LENGTH,
        N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, tokenizer
    )
    torch.onnx.export(
        funasr_nano_encoder,
        (torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=torch.int16),
         torch.ones((1, 10, hidden_size), dtype=torch.float32)),
        onnx_model_Encoder,
        input_names=['audio', 'query_embed'],
        output_names=['concat_embed', 'ids_len'],
        dynamic_axes={
            'audio':        {2: 'audio_len'},
            'query_embed':  {1: 'num_token'},
            'concat_embed': {1: 'num_token'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del funasr_nano_encoder, custom_stft
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Export: Decoder Embed
    # ══════════════════════════════════════════════════════════════════
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    torch.onnx.export(
        FUNASR_NANO_DECODER_EMBED(model.model),
        (input_ids,),
        onnx_model_Embed,
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids':     {0: 'batch', 1: 'ids_len'},
            'hidden_states': {0: 'batch', 1: 'ids_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del input_ids

    # ══════════════════════════════════════════════════════════════════
    # Export: Rotary + Mask (Prefill)
    # ══════════════════════════════════════════════════════════════════
    torch.onnx.export(
        ROTARY_MASK_PREFILL(model.model, MAX_SEQ_LEN),
        (ids_len, history_len),
        onnx_model_Rotary_Mask_Prefill,
        input_names=['ids_len', 'history_len'],
        output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
        dynamic_axes={
            'rotary_cos':     {1: 'ids_len'},
            'rotary_sin':     {1: 'ids_len'},
            'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: Rotary + Mask (Decode)
    # ══════════════════════════════════════════════════════════════════
    torch.onnx.export(
        ROTARY_MASK_DECODE(model.model, MAX_SEQ_LEN),
        (kv_seq_len,),
        onnx_model_Rotary_Mask_Decode,
        input_names=['kv_seq_len'],
        output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: Decoder Main (Transformer Layers)
    # ══════════════════════════════════════════════════════════════════
    kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)

    hidden_states  = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
    rotary_cos     = torch.zeros((1, ids_len, 1, 1, head_dim), dtype=torch.float32)
    rotary_sin     = rotary_cos
    attention_mask = torch.zeros((1, 1, 1, ids_len, kv_seq_len), dtype=torch.float32)

    all_inputs   = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
    input_names  = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
    output_names = kv_out_names + ['logits']
    dynamic_axes = {
        **kv_axes,
        'hidden_states':  {0: 'batch', 1: 'ids_len'},
        'logits':         {0: 'batch'},
        'rotary_cos':     {1: 'ids_len'},
        'rotary_sin':     {1: 'ids_len'},
        'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
    }

    model_Main = FUNASR_NANO_DECODER_MAIN(model.model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size)
    del model

    torch.onnx.export(
        model_Main,
        tuple(all_inputs),
        onnx_model_Main,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del model_Main, hidden_states, attention_mask, all_inputs
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Export: Greedy Search
    # ══════════════════════════════════════════════════════════════════
    save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)  # 10 is a dummy value.

    torch.onnx.export(
        GREEDY_SEARCH(),
        (logits, save_id_in),
        onnx_model_Greedy,
        input_names=['logits', 'save_id_in'],
        output_names=['max_logits_idx', 'save_id_out'],
        dynamic_axes={
            'logits':         {0: 'batch'},
            'save_id_in':     {0: 'batch', 1: 'history_len'},
            'save_id_out':    {0: 'batch', 1: 'history_len'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: First Beam Search
    # ══════════════════════════════════════════════════════════════════
    num_layers_beam = num_layers * len(kv_specs)
    # First beam uses single-batch KV (batch dim = 1)
    kv_tensors_Greedy = {k: v[[0]] for k, v in kv_tensors.items()}
    kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_Greedy)
    # Remove output axes — first beam outputs have variable batch, not tracked here
    kv_input_only_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names}

    torch.onnx.export(
        FIRST_BEAM_SEARCH(num_layers_beam),
        tuple(kv_ins + [logits[[0]], save_id_in, beam_size]),
        onnx_model_First_Beam,
        input_names=kv_in_names + ['logits', 'save_id_in', 'beam_size'],
        output_names=(
            ['out_' + n[3:] for n in kv_in_names] + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx']
        ),
        dynamic_axes={
            **kv_input_only_axes,
            'logits':           {0: 'batch'},
            'save_id_in':       {0: 'batch', 1: 'history_len'},
            'top_beam_prob':    {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx':   {0: 'batch'},
            'save_id_out':      {0: 'batch', 1: 'history_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: Second Beam Search
    # ══════════════════════════════════════════════════════════════════
    kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)
    previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
    topK = torch.tensor([TOP_K], dtype=torch.int64)

    torch.onnx.export(
        SECOND_BEAM_SEARCH(num_layers_beam),
        tuple(kv_ins + [logits, save_id_in, previous_prob, beam_size, topK]),
        onnx_model_Second_Beam,
        input_names=kv_in_names + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
        output_names=kv_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
        dynamic_axes={
            **kv_axes,
            'logits':           {0: 'batch'},
            'save_id_in':       {0: 'batch', 1: 'history_len'},
            'previous_prob':    {0: 'batch'},
            'save_id_out':      {0: 'batch', 1: 'history_len'},
            'top_beam_prob':    {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx':   {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del kv_tensors_Greedy, previous_prob, topK

    # ══════════════════════════════════════════════════════════════════
    # Export: Apply Penalty
    # ══════════════════════════════════════════════════════════════════
    penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
    penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)

    torch.onnx.export(
        APPLY_PENALTY(),
        (logits, save_id_in, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
        output_names=['logits_out'],
        dynamic_axes={
            'logits_in':  {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'logits_out': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del save_id_in, penalty_value, penalty_range

    # ══════════════════════════════════════════════════════════════════
    # Export: Argmax
    # ══════════════════════════════════════════════════════════════════
    torch.onnx.export(
        ARGMAX(),
        (logits,),
        onnx_model_Argmax,
        input_names=['logits'],
        output_names=['max_logits_idx'],
        dynamic_axes={
            'logits':         {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del logits
    gc.collect()

print('\nExport done!\n\nStart to run FunASR-Nano by ONNX Runtime.\n\nNow, loading the model...')


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def normalizer(_audio, target_value=8192.0):
    _audio = _audio.astype(np.float32)
    rms = np.sqrt(np.mean((_audio * _audio), dtype=np.float32), dtype=np.float32)
    _audio *= (target_value / (rms + 1e-7))
    np.clip(_audio, -32768.0, 32767.0, out=_audio)
    return _audio.astype(np.int16)


def bind_ort_in(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out(binding, names, device):
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# --- Encoder ---
ort_session_Encoder = create_session(onnx_model_Encoder, **packed_settings)
binding_Encoder     = ort_session_Encoder.io_binding()
shape_value_in_Encoder = ort_session_Encoder._inputs_meta[0].shape[-1]
in_name_Encoder     = get_in_names(ort_session_Encoder)
out_name_Encoder    = get_out_names(ort_session_Encoder)

# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Rotary + Mask (Prefill) ---
ort_session_Rotary_Mask_Prefill = create_session(onnx_model_Rotary_Mask_Prefill, **packed_settings)
binding_Rotary_Mask_Prefill     = ort_session_Rotary_Mask_Prefill.io_binding()
in_name_Rotary_Mask_Prefill     = get_in_names(ort_session_Rotary_Mask_Prefill)
out_name_Rotary_Mask_Prefill    = get_out_names(ort_session_Rotary_Mask_Prefill)

# --- Rotary + Mask (Decode) ---
ort_session_Rotary_Mask_Decode = create_session(onnx_model_Rotary_Mask_Decode, **packed_settings)
binding_Rotary_Mask_Decode     = ort_session_Rotary_Mask_Decode.io_binding()
in_name_Rotary_Mask_Decode     = get_in_names(ort_session_Rotary_Mask_Decode)[0]
out_name_Rotary_Mask_Decode    = get_out_names(ort_session_Rotary_Mask_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)
num_keys_values        = amount_of_outputs_Main - 1

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4

in_name_Main_parts   = in_name_Main[:num_keys_values]
out_name_Main_kv     = out_name_Main[:num_keys_values]
out_name_Main_logits = out_name_Main[num_keys_values]

# Dtype introspection
kv_dtype_str             = ort_session_Main._inputs_meta[0].type
hidden_states_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values].type else np.float32
vocab_size               = ort_session_Main._outputs_meta[num_keys_values].shape[1]

_logits_out_meta  = ort_session_Main._outputs_meta[num_keys_values]
_logits_out_dtype = np.float16 if 'float16' in _logits_out_meta.type else np.float32


# ══════════════════════════════════════════════════════════════════════════════
# KV CACHE SETUP (float16 for FunASR-Nano)
# ══════════════════════════════════════════════════════════════════════════════
_meta = ort_session_Main._inputs_meta

kv_dtype_Main = np.float16 if 'float16' in kv_dtype_str else np.float32
num_layers    = num_keys_values // 2

past_keys_Main   = create_ort_with_shape((1, _meta[0].shape[1],          1, _meta[0].shape[3],          0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, _meta[num_layers].shape[1], 1, 0, _meta[num_layers].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)

STOP_TOKEN_SET = set(STOP_TOKEN)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & SHARED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

_rotary_meta = ort_session_Rotary_Mask_Decode._outputs_meta

# --- Scalar OrtValues ---
init_history_len = create_ort_with_data([0], np.int64, device_type, DEVICE_ID)
topK             = create_ort_with_data([TOP_K],     np.int64, device_type, DEVICE_ID)
beam_size        = create_ort_with_data([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

# --- Decode-phase static buffers (bind once, reused every step) ---
attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1),                                  hidden_states_dtype_Main, device_type, DEVICE_ID)
rotary_cos_buf     = create_ort_with_shape((1, 1, 1, 1, _rotary_meta[0].shape[4]),           hidden_states_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf     = create_ort_with_shape((1, 1, 1, 1, _rotary_meta[1].shape[4]),           hidden_states_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf  = create_ort_with_shape((BEAM_SIZE, 1, _meta[num_keys_values].shape[2]),  hidden_states_dtype_Main, device_type, DEVICE_ID)
save_id_buf        = create_ort_with_shape((BEAM_SIZE, 0),                                   np.int32,                 device_type, DEVICE_ID)

# --- Logits & token-index buffers ---
prefill_logits_buf = create_ort_with_shape((1, vocab_size),         _logits_out_dtype, device_type, DEVICE_ID)
decode_logits_buf  = create_ort_with_shape((BEAM_SIZE, vocab_size), _logits_out_dtype, device_type, DEVICE_ID)
max_idx_buf        = create_ort_with_shape((1, 1),                  np.int32,          device_type, DEVICE_ID)

generate_limit_base = MAX_SEQ_LEN - 10


# ══════════════════════════════════════════════════════════════════════════════
# DECODE HEAD SESSIONS (Beam Search OR Greedy/Argmax)
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")

    # --- First Beam ---
    ort_session_First_Beam    = create_session(onnx_model_First_Beam, **packed_settings)
    binding_First_Beam        = ort_session_First_Beam.io_binding()
    in_name_First_Beam        = get_in_names(ort_session_First_Beam)
    out_name_First_Beam       = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts  = in_name_First_Beam[:num_keys_values_plus_1]
    out_name_First_Beam_parts = out_name_First_Beam[:num_keys_values_plus_1]

    # --- Second Beam ---
    ort_session_Second_Beam    = create_session(onnx_model_Second_Beam, **packed_settings)
    binding_Second_Beam        = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam        = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam       = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts  = in_name_Second_Beam[:num_keys_values_plus_1]
    out_name_Second_Beam_parts = out_name_Second_Beam[:num_keys_values_plus_1]

    # --- Beam-specific buffers ---
    beam_ids_buf   = create_ort_with_shape((BEAM_SIZE, 1), np.int32,                 device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_states_dtype_Main, device_type, DEVICE_ID)

    # --- Static beam bindings ---
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_1], save_id_buf)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_2], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_3], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_4], topK)

else:
    # --- Greedy ---
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = get_in_names(ort_session_Greedy)
    out_name_Greedy    = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    # --- Argmax ---
    ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
    out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# PENALTY SESSION (optional — replaces Reset_Penalty_Beam/Greedy)
# ══════════════════════════════════════════════════════════════════════════════
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

    binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-EMBED TASK PROMPTS
# ══════════════════════════════════════════════════════════════════════════════
init_all_outputs_Embed = []
for i in task_prompt:
    tokens = tokenizer(i, return_tensors='np')['input_ids'].astype(np.int32)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
    binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
    bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
    run(ort_session_Embed, binding_Embed)
    init_all_outputs_Embed.append(onnxruntime.OrtValue.ortvalue_from_numpy(binding_Embed.get_outputs()[0].numpy(), device_type, DEVICE_ID))


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════
for prompt_embed, test in zip(init_all_outputs_Embed, test_audio):
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    if USE_NORMALIZER:
        audio = normalizer(audio, 8192.0)
    audio_full_len = len(audio)
    INPUT_AUDIO_LENGTH = min(MAX_INPUT_AUDIO_LENGTH, audio_full_len) if isinstance(shape_value_in_Encoder, str) else shape_value_in_Encoder
    stride_step = INPUT_AUDIO_LENGTH if SLIDING_WINDOW <= 0 else SLIDING_WINDOW
    audio = audio.reshape(1, 1, -1)
    if audio_full_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_full_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        pad_amount = ((num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH) - audio_full_len
        audio = np.concatenate((audio, np.zeros([1, 1, pad_amount], dtype=audio.dtype)), axis=-1)
    elif audio_full_len < INPUT_AUDIO_LENGTH:
        audio = np.concatenate((audio, np.zeros([1, 1, INPUT_AUDIO_LENGTH - audio_full_len], dtype=audio.dtype)), axis=-1)
    aligned_len = audio.shape[-1]
    asr_result = ""
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    rtf_time = time.time()

    while slice_end <= aligned_len:
        ort_audio = onnxruntime.OrtValue.ortvalue_from_numpy(audio[..., slice_start:slice_end], device_type, DEVICE_ID)

        # ══════════════════════════════════════════════════════════════
        # ENCODER
        # ══════════════════════════════════════════════════════════════
        binding_Encoder.bind_ortvalue_input(in_name_Encoder[0], ort_audio)
        binding_Encoder.bind_ortvalue_input(in_name_Encoder[1], prompt_embed)
        bind_ort_out(binding_Encoder, out_name_Encoder, _ort_device_type)
        run(ort_session_Encoder, binding_Encoder)
        all_outputs_Encoder = binding_Encoder.get_outputs()
        hidden_states = all_outputs_Encoder[0]   # concat_embed
        ids_len       = all_outputs_Encoder[1]   # ids_len

        # ══════════════════════════════════════════════════════════════
        # PREFILL SETUP
        # ══════════════════════════════════════════════════════════════

        # --- Rotary + Mask (Prefill) ---
        binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[0], ids_len)
        binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[1], init_history_len)
        bind_ort_out(binding_Rotary_Mask_Prefill, out_name_Rotary_Mask_Prefill, _ort_device_type)
        run(ort_session_Rotary_Mask_Prefill, binding_Rotary_Mask_Prefill)
        rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Mask_Prefill.get_outputs()

        # --- Pre-bind Decode Rotary (static output buffers) ---
        binding_Rotary_Mask_Decode.bind_ortvalue_input(in_name_Rotary_Mask_Decode, kv_seq_len)
        binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[0], rotary_cos_buf)
        binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[1], rotary_sin_buf)
        binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[2], kv_seq_len)

        # --- Bind Main: non-KV inputs (prefill) ---
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        hidden_states)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask)

        # --- Bind Main: empty KV cache ---
        i = 0
        for _ in range(num_layers):
            binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
            i += 1
        for _ in range(num_layers):
            binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
            i += 1

        # --- Bind Main: outputs (prefill) ---
        bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
        binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

        # --- Bind Penalty to prefill logits ---
        if USE_PENALTY:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
            binding_Penalty.bind_ortvalue_output(out_name_Penalty,  prefill_logits_buf)

        # --- Bind decode head to prefill logits ---
        if USE_BEAM_SEARCH:
            binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values], prefill_logits_buf)
        elif USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0],   prefill_logits_buf)
            binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax,   prefill_logits_buf)
            binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)

        # --- Pre-bind Embed for decode phase ---
        binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

        # --- Reset greedy save_id for new window ---
        if not USE_BEAM_SEARCH and USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

        is_prefill_step = True
        num_decode = 0
        generate_limit = generate_limit_base - ids_len.numpy().flat[0]
        start_time = time.time()

        # ══════════════════════════════════════════════════════════════
        # DECODE LOOP
        # ══════════════════════════════════════════════════════════════
        while num_decode < generate_limit:

            # ── 1. Run Main Model ────────────────────────────────────
            run(ort_session_Main, binding_Main)
            outputs_Main = binding_Main.get_outputs()

            # ── 2. Apply Repetition Penalty ──────────────────────────
            if USE_PENALTY and num_decode >= PENALTY_RANGE:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
                run(ort_session_Penalty, binding_Penalty)

            # ── 3. Token Selection ───────────────────────────────────
            if USE_BEAM_SEARCH:
                # ── 3a. Beam Search ──────────────────────────────────
                if is_prefill_step:
                    bind_ort_in(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
                    bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_1], beam_score_buf)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_2], beam_ids_buf)
                    binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_3], max_idx_buf)
                    run(ort_session_First_Beam, binding_First_Beam)
                    outputs_Beam = binding_First_Beam.get_outputs()
                else:
                    bind_ort_in(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
                    bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
                    if num_decode < 2:
                        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_2], beam_score_buf)
                        binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_1], beam_score_buf)
                        binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_2], beam_ids_buf)
                        binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_3], max_idx_buf)
                    run(ort_session_Second_Beam, binding_Second_Beam)
                    outputs_Beam = binding_Second_Beam.get_outputs()

                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN_SET:
                    break

                save_id = outputs_Beam[num_keys_values]
                bind_ort_in(binding_Main, in_name_Main_parts, outputs_Beam)
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_1], save_id)

            else:
                # ── 3b. Greedy / Argmax ──────────────────────────────
                if USE_PENALTY:
                    binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
                    run(ort_session_Greedy, binding_Greedy)
                    save_id = binding_Greedy.get_outputs()[1]
                else:
                    run(ort_session_Argmax, binding_Argmax)

                max_logits_idx = max_idx_buf.numpy().flat[0]
                if max_logits_idx in STOP_TOKEN_SET:
                    break

                if USE_PENALTY:
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
                else:
                    save_id_numpy[num_decode] = max_logits_idx

                bind_ort_in(binding_Main, in_name_Main_parts, outputs_Main)

            # ── 4. Re-bind Main KV outputs (fresh alloc each step) ───
            bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

            # ── 5. Transition: prefill → decode (once per window) ────
            if is_prefill_step:
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        hidden_states_buf)
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos_buf)
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin_buf)
                binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask_buf)
                binding_Main.bind_ortvalue_output(out_name_Main_logits,                decode_logits_buf)

                binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

                if USE_PENALTY:
                    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                    binding_Penalty.bind_ortvalue_output(out_name_Penalty,  decode_logits_buf)

                if USE_BEAM_SEARCH:
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values], decode_logits_buf)
                    binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
                elif USE_PENALTY:
                    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
                else:
                    binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

                is_prefill_step = False

            # ── 6. Prepare next step: Embed + Rotary ─────────────────
            run(ort_session_Embed, binding_Embed)
            run(ort_session_Rotary_Mask_Decode, binding_Rotary_Mask_Decode)
            num_decode += 1

        # ── End of window ────────────────────────────────────────────
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
        if num_decode > 0:
            if USE_BEAM_SEARCH or USE_PENALTY:
                asr_result += tokenizer.decode(save_id.numpy()[0, :num_decode], skip_special_tokens=True)
            else:
                asr_result += tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)
        print(f"\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s\n")

    print(asr_result, end="", flush=True)
    print(f"\n\nRTF: {((time.time() - rtf_time) / (audio_full_len / SAMPLE_RATE)):.3f}")
    print("----------------------------------------------------------------------------------------------------------")
    
