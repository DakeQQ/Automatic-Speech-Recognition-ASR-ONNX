import gc
import time
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
import torchaudio
from onnxruntime.capi import _pybind_state as C
from pydub import AudioSegment
from torch import Tensor
from transformers import AutoConfig, AutoModel, AutoTokenizer
# transformers==4.57.6

from STFT_Process import STFT_Process
from modeling_modified.inference_asr_standalone import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
download_path                  = r'/home/DakeQQ/Downloads/Qwen3-ASR-0.6B'                                   # Set the path where the Qwen3-ASR-[0.6B, 1.7B] model downloaded.
onnx_model_Encoder             = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Qwen3_ASR_Encoder.onnx'             # The exported onnx model path.
onnx_model_Embed               = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Qwen3_ASR_Decoder_Embed.onnx'
onnx_model_Main                = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Qwen3_ASR_Decoder_Main.onnx'
onnx_model_Rotary_Mask_Prefill = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Rotary_Mask_Text_Prefill.onnx'
onnx_model_Rotary_Mask_Decode  = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Rotary_Mask_Text_Decode.onnx'
onnx_model_Greedy              = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam          = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam         = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty             = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax              = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Argmax.onnx'
onnx_model_Concat_Embed        = r'/home/DakeQQ/Downloads/Qwen_ASR_ONNX/Concat_Embed.onnx'

# Test audio for post-export validation.
test_audio       = ["./example/zh.mp3", "./example/en.mp3", "./example/yue.mp3", "./example/ja.mp3", "./example/ko.mp3"]
LANGUAGE_PROMPTS = ["Chinese", "English", "Cantonese", "", ""]      # Use English words for the language. Set "" for auto-detection mode.
TASK_PROMPTS     = ["", "tribal chieftain", "", "", ""]             # Put the Hot Words or task hint here (optional).


# ═══════════ㄢ═══════════════════════════════════════════════════════════════════
# Audio & STFT Configuration
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE            = 16000      # The model parameter, do not edit the value.
WINDOW_TYPE            = 'hann'     # Type of window function used in the STFT.
N_MELS                 = 128        # Number of Mel bands to generate in the Mel-spectrogram. Do not edit.
NFFT_STFT              = 400        # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH          = 400        # Length of windowing, edit it carefully.
HOP_LENGTH             = 160        # Number of samples between successive frames in the STFT, edit it carefully.

# Model Parameters
STOP_TOKEN             = [151643, 151645]  # The stop_id in Qwen is "151643" & "151645".
MAX_SEQ_LEN            = 1024       # The max context length, including prompt + audio + decode tokens.
USE_FP16_KV            = True       # Use fp16 KV cache for memory efficiency.
PREVENT_F16_OVERFLOW   = False      # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.

# Input & Processing Limits
MAX_INPUT_AUDIO_LENGTH = SAMPLE_RATE * 30  # The maximum input audio length (30s × 16000 samples/s).
DYNAMIC_AXES           = True       # The default dynamic_axes is the input audio length. Note that some providers only support static axes.

# Decoding Strategy
USE_BEAM_SEARCH        = False      # Use beam search or greedy search.
TOP_K                  = 3          # The top k candidate in decoding.
BEAM_SIZE              = 3          # Number of beams in searching.
PENALTY_RANGE          = 10         # Penalizes the most recent output. "10" means the last 10 tokens.
MAX_BEAM_SIZE          = 10         # Max beams for exported model (static batch dimension in ONNX).
REPEAT_PENALTY         = 1.0        # Range from 0.0 to 1.0; "1.0" means no penalty.

# Runtime & Export Settings
ORT_Accelerate_Providers = []       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
ORT_LOG                = False      # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16               = False      # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
MAX_THREADS            = 0          # Parallel CPU threads. Set 0 for auto.
DEVICE_ID              = 0          # Default to zero.
OPSET                  = 23         # ONNX Runtime opset version.


# ══════════════════════════════════════════════════════════════════════════════
# Audio Special-Token IDs
# ══════════════════════════════════════════════════════════════════════════════
AUDIO_START_TOKEN_ID   = 151669     # Audio start token ID.
AUDIO_END_TOKEN_ID     = 151670     # Audio end token ID.
IM_START_TOKEN_ID      = 151644     # <|im_start|> token ID.
IM_END_TOKEN_ID        = 151645     # <|im_end|> token ID.
SYSTEM_TOKEN_ID        = 8948       # "system" token ID.
USER_TOKEN_ID          = 872        # "user" token ID.
ASSISTANT_TOKEN_ID     = 77091      # "assistant" token ID.
NEWLINE_TOKEN_ID       = 198        # Newline "\n" token ID.
_ASR_TEXT_TAG          = "<asr_text>"
_LANG_PREFIX           = "language "


# ══════════════════════════════════════════════════════════════════════════════
# ── Utility Helpers ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def parse_asr_output(raw: str, user_language: Optional[str] = None) -> Tuple[str, str]:
    if not raw:
        return "", ""
    s = str(raw).strip()
    if not s:
        return "", ""
    if user_language:
        return user_language, s
    if _ASR_TEXT_TAG not in s:
        return "", s
    meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    lang = ""
    idx = meta_part.lower().find(_LANG_PREFIX)
    if idx >= 0:
        lang = meta_part[idx + len(_LANG_PREFIX):].strip()
        if lang:
            lang = lang[:1].upper() + lang[1:].lower()
    return lang, text_part.strip()


def _get_feat_extract_output_lengths(input_lengths: Tensor) -> Tensor:
    input_lengths_leave = input_lengths % 100
    feat_lengths = torch.clamp(input_lengths_leave - 1, min=0) // 2 + 1
    feat_lengths = feat_lengths * (input_lengths_leave > 0).to(feat_lengths.dtype)
    feat_lengths_2 = torch.clamp(feat_lengths - 1, min=0) // 2 + 1
    feat_lengths_2 = feat_lengths_2 * (feat_lengths > 0).to(feat_lengths_2.dtype)
    feat_lengths_3 = torch.clamp(feat_lengths_2 - 1, min=0) // 2 + 1
    feat_lengths_3 = feat_lengths_3 * (feat_lengths_2 > 0).to(feat_lengths_3.dtype)
    return feat_lengths_3 + (input_lengths // 100) * 13


def replace_gelu_with_tanh(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh(child)


def absorb_layer_norm_affine(norm: torch.nn.LayerNorm, linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(
                torch.zeros(linear.out_features, dtype=linear.weight.dtype)
            )
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias   = None


def build_query_prompt_ids(tokenizer: AutoTokenizer, system_prompt: str) -> List[int]:
    query_ids: List[int] = []
    if system_prompt:
        query_ids.extend(tokenizer.encode(system_prompt, add_special_tokens=False))
    return query_ids


def get_kv_io(
    tensors_dict:  Dict[str, Tensor],
    kv_specs:      Sequence[Tuple[str, int]],
    num_layers:    int,
    batch_axis:    str = "batch",
    seq_axis:      str = "history_len",
    out_seq_axis:  str = "kv_seq_len",
) -> Tuple[List[Tensor], List[str], List[str], Dict[str, Dict[int, str]]]:
    """Build KV I/O tensor lists, name lists, and dynamic-axes dict for onnx.export."""
    inputs:       List[Tensor]              = []
    input_names:  List[str]                 = []
    output_names: List[str]                 = []
    dynamic_axes: Dict[str, Dict[int, str]] = {}
    for name, dim in kv_specs:
        tensor = tensors_dict[name]
        for i in range(num_layers):
            in_name  = f"past_{name}_{i}"
            out_name = f"present_{name}_{i}"
            inputs.append(tensor)
            input_names.append(in_name)
            output_names.append(out_name)
            dynamic_axes[in_name]  = {0: batch_axis, dim: seq_axis}
            dynamic_axes[out_name] = {0: batch_axis, dim: out_seq_axis}
    return inputs, input_names, output_names, dynamic_axes


# ══════════════════════════════════════════════════════════════════════════════
# ── Audio Encoder (fused mel + encoder + prompt concat) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_ENCODER(torch.nn.Module):
    def __init__(
        self,
        audio_tower: torch.nn.Module,
        embed_tokens: torch.nn.Embedding,
        head_ids: Sequence[int],
        tail_ids: Sequence[int],
        query_suffix_ids: Sequence[int],
    ) -> None:
        super().__init__()
        self.audio_tower = audio_tower.float()
        replace_gelu_with_tanh(self.audio_tower)
        self.audio_tower.act = torch.nn.GELU(approximate="tanh")
        for layer in self.audio_tower.layers:
            layer.activation_fn = torch.nn.GELU(approximate="tanh")
        self._fuse_encoder_weights()

        self.stft = STFT_Process(
            model_type="stft_B",
            n_fft=NFFT_STFT,
            win_length=WINDOW_LENGTH,
            hop_len=HOP_LENGTH,
            max_frames=0,
            window_type=WINDOW_TYPE,
            center_pad=True,
            pad_mode="reflect",
        ).eval()

        mel_filters = torchaudio.functional.melscale_fbanks(
            n_freqs=(NFFT_STFT // 2) + 1,
            f_min=0.0,
            f_max=SAMPLE_RATE / 2,
            n_mels=N_MELS,
            sample_rate=SAMPLE_RATE,
            norm="slaney",
            mel_scale="slaney",
        ).transpose(0, 1).unsqueeze(0).contiguous()
        self.register_buffer("mel_filters", mel_filters.to(dtype=torch.float32), persistent=False)

        chunk_size           = int(self.audio_tower.n_window * 2)          # 100
        self.chunk_size      = chunk_size
        self.chunk_size_minus = chunk_size - 1
        chunk_aftercnn       = int(_get_feat_extract_output_lengths(torch.tensor([chunk_size])).item()) # 13
        self.chunk_aftercnn  = chunk_aftercnn
        chunks_per_window    = int(self.audio_tower.n_window_infer // chunk_size)  # 8
        self.chunks_per_window = chunks_per_window
        self.chunks_per_window_minus = chunks_per_window - 1
        tokens_per_window    = chunks_per_window * chunk_aftercnn           # 104
        self.tokens_per_window = tokens_per_window
        self.model_dim       = int(self.audio_tower.config.d_model)
        self.output_dim      = int(self.audio_tower.config.output_dim)
        self.num_heads       = int(self.audio_tower.layers[0].self_attn.num_heads)
        self.head_dim        = self.model_dim // self.num_heads

        max_mel_frames = MAX_INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
        self.max_chunks = (max_mel_frames + self.chunk_size_minus) // chunk_size
        self.max_windows = (self.max_chunks + chunks_per_window - 1) // chunks_per_window
        max_total_chunks_padded = self.max_windows * chunks_per_window
        key_mask_lookup = torch.zeros(1, tokens_per_window + 1, 1, 1, tokens_per_window, dtype=torch.int8)
        for n in range(tokens_per_window + 1):
            key_mask_lookup[0, n, 0, 0, n:] = -128
        valid_mask_float_lookup = torch.zeros(tokens_per_window + 1, tokens_per_window, 1, dtype=torch.int8)
        for n in range(tokens_per_window + 1):
            valid_mask_float_lookup[n, :n, 0] = 1

        self.pos = self.audio_tower.positional_embedding.positional_embedding[:self.chunk_aftercnn].unsqueeze(0)
        self.register_buffer("window_chunks_full", torch.arange(self.max_windows + 1, dtype=torch.int32) * chunks_per_window, persistent=False)
        self.register_buffer("chunk_starts_full", torch.arange(self.max_chunks + 1, dtype=torch.int32) * chunk_size, persistent=False)
        self.register_buffer("aftercnn_lens_lookup",  _get_feat_extract_output_lengths(torch.arange(chunk_size + 1)).int(), persistent=False)
        self.register_buffer("chunk_pad_zeros", torch.zeros((max_total_chunks_padded, chunk_aftercnn, self.model_dim), dtype=torch.int8), persistent=False)
        self.register_buffer("aftercnn_pad_zeros", torch.zeros(chunks_per_window, dtype=torch.int8), persistent=False)
        self.register_buffer("key_mask_lookup", key_mask_lookup, persistent=False)
        self.register_buffer("valid_mask_float_lookup", valid_mask_float_lookup, persistent=False)
        self.register_buffer("mel_pad_zeros", torch.zeros((1, N_MELS, chunk_size), dtype=torch.int8), persistent=False)
        self.register_buffer("inv_int16", torch.tensor([1.0 / 32768.0], dtype=torch.float32), persistent=False)

        with torch.no_grad():
            head_ids_tensor = torch.tensor([list(head_ids)], dtype=torch.int32)
            tail_ids_tensor = torch.tensor([list(tail_ids)], dtype=torch.int32)
            query_suffix_tensor = torch.tensor([list(query_suffix_ids)], dtype=torch.int32)
            self.register_buffer("head_embed", embed_tokens(head_ids_tensor).float(), persistent=False)
            self.register_buffer("tail_embed", embed_tokens(tail_ids_tensor).float(), persistent=False)
            self.register_buffer("query_suffix_embed", embed_tokens(query_suffix_tensor).float(), persistent=False)
        self.head_len = self.head_embed.shape[1]
        self.tail_len = self.tail_embed.shape[1]

    def _fuse_encoder_weights(self) -> None:
        with torch.no_grad():
            for layer in self.audio_tower.layers:
                attn = layer.self_attn
                qkv  = torch.nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features, bias=True)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                qkv.bias.copy_(torch.cat([attn.q_proj.bias, attn.k_proj.bias, attn.v_proj.bias], dim=0))
                absorb_layer_norm_affine(layer.self_attn_layer_norm, qkv)

                scale_sqrt = attn.scaling ** 0.5
                q_out = attn.q_proj.out_features
                k_out = attn.k_proj.out_features
                qkv.weight.data[:q_out].mul_(scale_sqrt)
                qkv.weight.data[q_out:q_out + k_out].mul_(scale_sqrt)
                qkv.bias.data[:q_out].mul_(scale_sqrt)
                qkv.bias.data[q_out:q_out + k_out].mul_(scale_sqrt)

                absorb_layer_norm_affine(layer.final_layer_norm, layer.fc1)
                attn.qkv = qkv
                del attn.q_proj, attn.k_proj, attn.v_proj

            absorb_layer_norm_affine(self.audio_tower.ln_post, self.audio_tower.proj1)

    def forward(self, audio: Tensor, query_embed: Tensor) -> Tuple[Tensor, Tensor]:
        # Matches WhisperFeatureExtractor exactly: no pre-emphasis, no mean/DC removal,
        # reflect-padded STFT, drop the trailing frame (stft[..., :-1]), then slaney log-mel.
        audio = audio.float() * self.inv_int16
        real, imag = self.stft(audio)
        real, imag = real[..., :-1], imag[..., :-1]
        power = real * real + imag * imag
        mel = torch.matmul(self.mel_filters, power)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.amax(dim=(-2, -1), keepdim=True) - 8.0)
        input_features = mel * 0.25 + 1.0
        feature_len = input_features.shape[-1].unsqueeze(0)
        num_chunks = (feature_len + self.chunk_size_minus) // self.chunk_size
        pad_frames = self.chunk_starts_full[num_chunks] - feature_len
        padded_features = torch.cat([input_features, self.mel_pad_zeros[..., :pad_frames].float()], dim=-1)
        chunks = padded_features.reshape(1, N_MELS, num_chunks, self.chunk_size).permute(2, 0, 1, 3)
        chunk_starts = self.chunk_starts_full[:num_chunks]
        raw_chunk_lens = torch.clamp(feature_len - chunk_starts, min=0, max=self.chunk_size)
        aftercnn_lens = self.aftercnn_lens_lookup[raw_chunk_lens]
        x = F.gelu(self.audio_tower.conv2d1(chunks), approximate="tanh")
        x = F.gelu(self.audio_tower.conv2d2(x), approximate="tanh")
        x = F.gelu(self.audio_tower.conv2d3(x), approximate="tanh")
        x = self.audio_tower.conv_out(x.permute(0, 3, 1, 2).contiguous().view(num_chunks, self.chunk_aftercnn, -1))
        hidden_states = x + self.pos
        num_windows = (num_chunks + self.chunks_per_window_minus) // self.chunks_per_window
        total_chunks_padded = self.window_chunks_full[num_windows]
        pad_chunks = total_chunks_padded - num_chunks
        hidden_states = torch.cat([hidden_states, self.chunk_pad_zeros[:pad_chunks].float()], dim=0)
        aftercnn_lens = torch.cat([aftercnn_lens, self.aftercnn_pad_zeros[:pad_chunks].int()])
        hidden_states = hidden_states.reshape(num_windows, self.tokens_per_window, self.model_dim)
        valid_counts = aftercnn_lens.reshape(num_windows, self.chunks_per_window).sum(dim=1, dtype=torch.int32)
        valid_mask_float = self.valid_mask_float_lookup[valid_counts].float()
        key_mask = self.key_mask_lookup[:, valid_counts]
        for layer in self.audio_tower.layers:
            residual = hidden_states
            normed = layer.self_attn_layer_norm(hidden_states)
            qkv = layer.self_attn.qkv(normed).view(num_windows, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.split(1, dim=0)
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(attn + key_mask, dim=-1)
            attn = torch.matmul(attn, v).transpose(2, 3).reshape(num_windows, -1, layer.self_attn.out_proj.in_features)
            hidden_states = residual + layer.self_attn.out_proj(attn)
            residual = hidden_states
            normed = layer.final_layer_norm(hidden_states)
            hidden_states = residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))
        hidden_states = self.audio_tower.ln_post(hidden_states)
        hidden_states = self.audio_tower.proj2(self.audio_tower.act(self.audio_tower.proj1(hidden_states)))
        hidden_states = hidden_states * valid_mask_float
        hidden_states = hidden_states.reshape(1, -1, self.output_dim)
        encoded_len = aftercnn_lens.sum(dtype=torch.int32).long()
        audio_hidden = hidden_states[:, :encoded_len]
        concat_embed = torch.cat([self.head_embed, query_embed, self.query_suffix_embed, audio_hidden, self.tail_embed], dim=1)
        ids_len = concat_embed.shape[1].unsqueeze(0)
        return concat_embed, ids_len


# ══════════════════════════════════════════════════════════════════════════════
# ── Rotary + Mask — Prefill Phase ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_ROTARY_MASK_PREFILL(torch.nn.Module):
    def __init__(self, llm: torch.nn.Module, max_seq_len: int) -> None:
        super().__init__()
        self.register_buffer("attention_mask", (1.0 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128, persistent=False)
        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    @staticmethod
    def _build_rotary_table(
        llm: torch.nn.Module, max_seq_len: int
    ) -> Tuple[Tensor, Tensor]:
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq     = llm.rotary_emb.inv_freq.float()
        theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(theta), torch.sin(theta)

    def forward(
        self, ids_len: Tensor, history_len: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


# ══════════════════════════════════════════════════════════════════════════════
# ── Rotary — Decode Phase (single new token, no mask output) ──────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, llm: torch.nn.Module, max_seq_len: int) -> None:
        super().__init__()
        cos, sin = QWEN3_ASR_ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoder Embedding ──────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_DECODER_EMBED(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.embed_tokens = model.thinker.model.embed_tokens.float()

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoder Main (deeply fused transformer) ───────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
class QWEN3_ASR_DECODER_MAIN(torch.nn.Module):
    def __init__(
        self,
        model:        torch.nn.Module,
        num_heads:    int,
        num_kv_heads: int,
        head_dim:     int,
        num_layers:   int,
        hidden_size:  int
    ) -> None:
        super().__init__()
        self.llm      = model.thinker.model.float()
        self.lm_head  = model.thinker.lm_head.float()
        self.head_dim       = head_dim
        self.head_dim_half  = head_dim // 2
        self.num_heads      = num_heads
        self.num_kv_heads   = num_kv_heads
        self.num_kv_groups  = num_heads // num_kv_heads
        self.qk_heads       = num_heads + num_kv_heads   # total heads before split
        self.num_layers     = num_layers
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)

        # Pre-computed RMS norm eps (scaled by dimension since size is absorbed into weights)
        rms_norm_eps = torch.tensor([1e-6], dtype=torch.float32)
        rms_eps_hidden = rms_norm_eps * hidden_size
        rms_eps_head = rms_norm_eps * head_dim
        if PREVENT_F16_OVERFLOW:
            rms_eps_hidden *= self.overflow_scale.square()
            rms_eps_head *= self.overflow_scale.square()
        self.register_buffer("rms_eps_hidden", torch.tensor([rms_eps_hidden], dtype=torch.float32))
        self.register_buffer("rms_eps_head", torch.tensor([rms_eps_head], dtype=torch.float32))

        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers

        self._fuse_weights(hidden_size)

    # ── Weight fusion (runs once at init) ──────────────────────────────────────
    def _fuse_weights(self, hidden_size: int) -> None:
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.llm.layers:
                attn = layer.self_attn
                has_bias = any(proj.bias is not None for proj in (attn.q_proj, attn.k_proj, attn.v_proj))
                qkv = torch.nn.Linear(attn.q_proj.in_features, attn.q_proj.out_features + attn.k_proj.out_features + attn.v_proj.out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0))
                if has_bias:
                    def _bias_or_zero(proj: torch.nn.Linear) -> Tensor:
                        return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
                    qkv.bias.copy_(torch.cat([_bias_or_zero(attn.q_proj), _bias_or_zero(attn.k_proj), _bias_or_zero(attn.v_proj)], dim=0))

                input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(input_norm_weight)
                attn.qkv = qkv

                combined_scale = scale_factor * norm_factor_qk
                attn.q_norm.weight.mul_(combined_scale)
                attn.k_norm.weight.mul_(combined_scale)
                q_norm_rep = attn.q_norm.weight.repeat(self.num_heads)
                k_norm_rep = attn.k_norm.weight.repeat(self.num_kv_heads)
                attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_rep, k_norm_rep], dim=0).view(1, 1, 1, self.qk_heads, self.head_dim))

                gate = layer.mlp.gate_proj
                up   = layer.mlp.up_proj
                post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
                gate_up.weight.copy_(torch.cat([gate.weight * post_norm_weight, up.weight * post_norm_weight], dim=0))
                layer.mlp.gate_up_proj = gate_up

                del attn.q_proj, attn.k_proj, attn.v_proj, attn.q_norm, attn.k_norm
                del layer.input_layernorm, layer.post_attention_layernorm
                del layer.mlp.gate_proj, layer.mlp.up_proj

            final_norm_weight = self.llm.norm.weight.unsqueeze(0) * norm_factor
            self.lm_head.weight.mul_(final_norm_weight)
            del self.llm.norm

    def _rms_norm(self, x: Tensor, eps: Tensor) -> Tensor:
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)

    def _rotate_half(self, x: Tensor, batch_size: int) -> Tensor:
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs: Tensor) -> Tuple[Tensor, ...]:
        hidden_states  = all_inputs[-4]
        rotary_cos     = all_inputs[-3]
        rotary_sin     = all_inputs[-2]
        attention_mask = all_inputs[-1]
        batch_size     = hidden_states.shape[0]
        for i, layer in enumerate(self.llm.layers):
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.rms_eps_hidden)
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_kv_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_kv_heads], dim=-2)
            qk = self._rms_norm(qk, self.rms_eps_head) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_cos + self._rotate_half(qk, batch_size) * rotary_sin
            q, k = torch.split(qk_rot, [self.num_heads, self.num_kv_heads], dim=-2)
            q = q.reshape(batch_size, -1, self.num_kv_heads, self.num_kv_groups, self.head_dim).permute(0, 2, 3, 1, 4)
            k = k.half().permute(0, 3, 2, 4, 1)
            v = v.half().transpose(1, 3)
            k = torch.cat((all_inputs[i],                   k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i]   = k
            self.save_value[i] = v
            attn = torch.matmul(q, k.float()) + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v.float())
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.rms_eps_hidden)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features] * 2, dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
        hidden_states = self._rms_norm(hidden_states[:, -1], self.rms_eps_hidden)
        logits = self.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoding Strategy Modules (mirrors Export_Fun_ASR_Nano.py) ────────────────
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    def forward(self, logits: Tensor, save_id: Tensor) -> Tuple[Tensor, Tensor]:
        max_idx = torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)
        return max_idx, torch.cat([save_id, max_idx], dim=-1)


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers: int) -> None:
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values: List[Optional[Tensor]] = [None] * self.total_layers

    def forward(self, *all_inputs: Tensor) -> Tuple[Tensor, ...]:
        logits    = all_inputs[-3]
        save_id   = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp  = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))

        top_beam_indices = top_beam_indices.transpose(0, 1).to(torch.int32)
        save_id          = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx   = top_beam_indices[[0]]

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx,
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers: int) -> None:
        super().__init__()
        self.total_layers     = total_layers
        self.save_keys_values: List[Optional[Tensor]] = [None] * self.total_layers

    def forward(self, *all_inputs: Tensor) -> Tuple[Tensor, ...]:
        logits        = all_inputs[-5]
        save_id       = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size     = all_inputs[-2]
        top_k         = all_inputs[-1]

        row_logsumexp          = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob    = top_k_logits - row_logsumexp
        current_prob  = (top_k_prob + previous_prob).view(-1)

        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index       = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).to(torch.int32)
        max_logits_idx   = top_beam_indices[[0]]
        save_id          = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx,
        )


class APPLY_PENALTY(torch.nn.Module):
    def forward(
        self,
        logits:        Tensor,
        save_id:       Tensor,
        penalty_value: Tensor,
        penalty_range: Tensor,
    ) -> Tensor:
        penalty_range_val = int(penalty_range.item())
        target_indices    = save_id[:, -penalty_range_val:].long()
        penalised         = logits.gather(1, target_indices) * penalty_value
        return logits.scatter(1, target_indices, penalised)


class ARGMAX(torch.nn.Module):
    def forward(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True).to(torch.int32)


class CONCAT_EMBED(torch.nn.Module):
    def __init__(self, embed_tokens: torch.nn.Embedding, tokenizer):
        super().__init__()
        with torch.no_grad():
            suffix_ids = torch.tensor([tokenizer.encode("<asr_text>", add_special_tokens=False)], dtype=torch.int32)
            self.register_buffer("suffix_embed", embed_tokens(suffix_ids).float(), persistent=False)

    def forward(self, codec_embed_0: Tensor, codec_embed_1: Tensor) -> Tensor:
        # The encoder tail already ends with the "language " prefix, so here we only
        # append the forced language name followed by the "<asr_text>" tag.
        concat_embed = torch.cat([codec_embed_0, codec_embed_1, self.suffix_embed], dim=1)
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# ── Export Loop ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
print("\nExport start …\n")

with torch.inference_mode():

    # ── Load model ────────────────────────────────────────────────────────────
    config = AutoConfig.from_pretrained(download_path, trust_remote_code=True)
    model  = AutoModel.from_pretrained(download_path, dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

    audio_cfg  = config.thinker_config.audio_config
    text_cfg   = config.thinker_config.text_config
    num_layers   = text_cfg.num_hidden_layers
    num_heads    = text_cfg.num_attention_heads
    num_kv_heads = text_cfg.num_key_value_heads
    head_dim     = text_cfg.head_dim
    hidden_size  = text_cfg.hidden_size
    vocab_size   = text_cfg.vocab_size
    kv_dtype     = torch.float16 if USE_FP16_KV else torch.float32
    kv_specs     = [("key", 4), ("value", 3)]

    print(f"  Encoder : layers={audio_cfg.encoder_layers}, d_model={audio_cfg.d_model}, output_dim={audio_cfg.output_dim}")
    print(f"  Decoder : layers={num_layers}, heads={num_heads}/{num_kv_heads} GQA, head_dim={head_dim}")
    print(f"  KV dtype: {'float16' if USE_FP16_KV else 'float32'}")

    head_ids = [IM_START_TOKEN_ID, SYSTEM_TOKEN_ID, NEWLINE_TOKEN_ID]
    # The assistant turn is trained to always begin with the "language " prefix.
    # Baking it into the encoder tail primes the decoder so that even in auto-detect
    # mode it reliably emits the detected language; a bare "assistant\n" context makes
    # the quantized decoder degenerate into garbage on the very first token.
    tail_ids = [
        AUDIO_END_TOKEN_ID,
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        ASSISTANT_TOKEN_ID,
        NEWLINE_TOKEN_ID,
    ] + tokenizer.encode(_LANG_PREFIX, add_special_tokens=False)
    dummy_query_ids = torch.tensor([build_query_prompt_ids(tokenizer, TASK_PROMPTS[0])], dtype=torch.int32)
    dummy_query_embed = model.thinker.model.embed_tokens(dummy_query_ids).float()

    ids_len     = torch.tensor([16], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    kv_seq_len  = ids_len + history_len
    beam_size   = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    top_k_t     = torch.tensor([TOP_K], dtype=torch.int64)
    logits      = torch.ones((MAX_BEAM_SIZE, vocab_size), dtype=torch.float32)
    save_id     = torch.zeros((MAX_BEAM_SIZE, 0), dtype=torch.int32)

    kv_tensors  = {
        "key":   torch.zeros((MAX_BEAM_SIZE, num_kv_heads, 1, head_dim, 0), dtype=kv_dtype),
        "value": torch.zeros((MAX_BEAM_SIZE, num_kv_heads, 1, 0, head_dim), dtype=kv_dtype),
    }

    query_suffix_ids = [
        IM_END_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        IM_START_TOKEN_ID,
        USER_TOKEN_ID,
        NEWLINE_TOKEN_ID,
        AUDIO_START_TOKEN_ID,
    ]

    # ── Fused Audio Encoder (pre-process + encoder in one graph) ─────────────
    encoder     = QWEN3_ASR_ENCODER(
        model.thinker.audio_tower,
        model.thinker.model.embed_tokens.float(),
        head_ids,
        tail_ids,
        query_suffix_ids,
    ).eval()
    dummy_audio = torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=torch.int16)
    torch.onnx.export(
        encoder,
        (dummy_audio, dummy_query_embed),
        onnx_model_Encoder,
        input_names=["audio", "query_embed"],
        output_names=["concat_embed", "ids_len"],
        dynamic_axes={
            "audio": {2: "audio_len"},
            "query_embed": {1: "num_token"},
            "concat_embed": {1: "total_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del encoder, dummy_audio, dummy_query_ids, dummy_query_embed
    gc.collect()

    # ── Decoder Embed ─────────────────────────────────────────────────────────
    embed_mod  = QWEN3_ASR_DECODER_EMBED(model).eval()
    dummy_ids  = torch.ones((1, 16), dtype=torch.int32)
    torch.onnx.export(
        embed_mod,
        (dummy_ids,),
        onnx_model_Embed,
        input_names=["input_ids"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_ids":     {0: "batch", 1: "ids_len"},
            "hidden_states": {0: "batch", 1: "ids_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del embed_mod
    gc.collect()

    # ── Rotary + Mask — Prefill ────────────────────────────────────────────────
    rotary_prefill = QWEN3_ASR_ROTARY_MASK_PREFILL(model.thinker.model, MAX_SEQ_LEN).eval()
    torch.onnx.export(
        rotary_prefill,
        (ids_len, history_len),
        onnx_model_Rotary_Mask_Prefill,
        input_names=["ids_len", "history_len"],
        output_names=["rotary_cos", "rotary_sin", "attention_mask", "kv_seq_len"],
        dynamic_axes={
            "rotary_cos":     {1: "ids_len"},
            "rotary_sin":     {1: "ids_len"},
            "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del rotary_prefill
    gc.collect()

    # ── Rotary — Decode (no attention_mask output) ─────────────────────────────
    rotary_decode = QWEN3_ASR_ROTARY_MASK_DECODE(model.thinker.model, MAX_SEQ_LEN).eval()
    torch.onnx.export(
        rotary_decode,
        (kv_seq_len,),
        onnx_model_Rotary_Mask_Decode,
        input_names=["kv_seq_len"],
        output_names=["rotary_cos", "rotary_sin", "kv_seq_len_next"],
        dynamic_axes={},
        opset_version=OPSET,
        dynamo=False,
    )
    del rotary_decode
    gc.collect()

    # ── Decoder Main ──────────────────────────────────────────────────────────
    kv_inputs, kv_input_names, kv_output_names, kv_axes = get_kv_io(kv_tensors, kv_specs, num_layers)

    hidden_states  = torch.ones((MAX_BEAM_SIZE, ids_len.item(), hidden_size), dtype=torch.float32)
    rotary_cos     = torch.ones((1, ids_len.item(), 1, 1, head_dim),          dtype=torch.float32)
    rotary_sin     = torch.zeros((1, ids_len.item(), 1, 1, head_dim),         dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1, ids_len.item(), ids_len.item()),   dtype=torch.float32)

    all_inputs   = kv_inputs + [hidden_states, rotary_cos, rotary_sin, attention_mask]
    input_names  = kv_input_names + ["hidden_states", "rotary_cos", "rotary_sin", "attention_mask"]
    output_names = kv_output_names + ["logits"]
    dynamic_axes = {
        **kv_axes,
        "hidden_states":  {0: "batch", 1: "ids_len"},
        "rotary_cos":     {1: "ids_len"},
        "rotary_sin":     {1: "ids_len"},
        "attention_mask": {3: "ids_len", 4: "kv_seq_len"},
        "logits":         {0: "batch"},
    }

    # ── Concat Embed ──────────────────────────────────────────────────────────
    concat_embed = CONCAT_EMBED(model.thinker.model.embed_tokens, tokenizer).eval()
    dummy_embed_0 = torch.ones((1, 5, hidden_size), dtype=torch.float32)
    dummy_embed_1 = torch.ones((1, 3, hidden_size), dtype=torch.float32)
    torch.onnx.export(
        concat_embed,
        (dummy_embed_0, dummy_embed_1),
        onnx_model_Concat_Embed,
        input_names=["codec_embed_0", "codec_embed_1"],
        output_names=["concat_embed", "concat_len"],
        dynamic_axes={
            "codec_embed_0": {1: "seq_len_0"},
            "codec_embed_1": {1: "seq_len_1"},
            "concat_embed":  {1: "total_seq_len"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del concat_embed, dummy_embed_0, dummy_embed_1
    gc.collect()

    # ── Decoder Main ──────────────────────────────────────────────────────────
    decoder_main = QWEN3_ASR_DECODER_MAIN(model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size).eval()
    del model
    gc.collect()

    torch.onnx.export(
        decoder_main,
        tuple(all_inputs),
        onnx_model_Main,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,
    )
    del decoder_main
    gc.collect()

    # ── Greedy Search ─────────────────────────────────────────────────────────
    torch.onnx.export(
        GREEDY_SEARCH().eval(),
        (logits[[0]], save_id[[0]]),
        onnx_model_Greedy,
        input_names=["logits", "save_id_in"],
        output_names=["max_logits_idx", "save_id_out"],
        dynamic_axes={
            "logits":         {0: "batch"},
            "save_id_in":     {0: "batch", 1: "history_len"},
            "max_logits_idx": {0: "batch"},
            "save_id_out":    {0: "batch", 1: "history_len_out"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── Apply Penalty ─────────────────────────────────────────────────────────
    save_id_10 = torch.zeros((MAX_BEAM_SIZE, 10), dtype=torch.int32)   # 10 = dummy hist len
    penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
    penalty_range = torch.tensor([PENALTY_RANGE],  dtype=torch.int64)
    torch.onnx.export(
        APPLY_PENALTY().eval(),
        (logits, save_id_10, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=["logits_in", "save_id_in", "penalty_value", "penalty_range"],
        output_names=["logits_out"],
        dynamic_axes={
            "logits_in":  {0: "batch"},
            "save_id_in": {0: "batch", 1: "history_len"},
            "logits_out": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── Argmax ────────────────────────────────────────────────────────────────
    torch.onnx.export(
        ARGMAX().eval(),
        (logits,),
        onnx_model_Argmax,
        input_names=["logits"],
        output_names=["max_logits_idx"],
        dynamic_axes={
            "logits":         {0: "batch"},
            "max_logits_idx": {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── First Beam Search ─────────────────────────────────────────────────────
    num_kv_entries     = num_layers * len(kv_specs)
    single_kv_tensors  = {name: tensor[[0]] for name, tensor in kv_tensors.items()}
    first_kv_inputs, first_kv_in_names, _, first_kv_axes = get_kv_io(single_kv_tensors, kv_specs, num_layers)
    first_kv_out_names = [f"out_{n[5:]}" for n in first_kv_in_names]  # "past_" → "out_"

    first_beam_save_id = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)
    torch.onnx.export(
        FIRST_BEAM_SEARCH(num_kv_entries).eval(),
        tuple(first_kv_inputs + [logits[[0]], first_beam_save_id, beam_size]),
        onnx_model_First_Beam,
        input_names=first_kv_in_names + ["logits", "save_id_in", "beam_size"],
        output_names=first_kv_out_names + ["save_id_out", "top_beam_prob", "top_beam_indices", "max_logits_idx"],
        dynamic_axes={
            **{n: ax for n, ax in first_kv_axes.items() if n in first_kv_in_names},
            "logits":           {0: "batch"},
            "save_id_in":       {0: "batch", 1: "history_len"},
            "save_id_out":      {0: "batch", 1: "history_len_out"},
            "top_beam_prob":    {0: "batch"},
            "top_beam_indices": {0: "batch"},
            "max_logits_idx":   {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    # ── Second Beam Search ────────────────────────────────────────────────────
    previous_prob = torch.zeros((MAX_BEAM_SIZE, 1), dtype=torch.float32)
    torch.onnx.export(
        SECOND_BEAM_SEARCH(num_kv_entries).eval(),
        tuple(kv_inputs + [logits, save_id, previous_prob, beam_size, top_k_t]),
        onnx_model_Second_Beam,
        input_names=kv_input_names + ["logits", "save_id_in", "previous_prob", "beam_size", "top_k"],
        output_names=kv_output_names + ["save_id_out", "top_beam_prob", "top_beam_indices", "max_logits_idx"],
        dynamic_axes={
            **kv_axes,
            "logits":           {0: "batch"},
            "save_id_in":       {0: "batch", 1: "history_len"},
            "previous_prob":    {0: "batch"},
            "save_id_out":      {0: "batch", 1: "history_len_out"},
            "top_beam_prob":    {0: "batch"},
            "top_beam_indices": {0: "batch"},
            "max_logits_idx":   {0: "batch"},
        },
        opset_version=OPSET,
        dynamo=False,
    )
    del kv_inputs, kv_tensors, first_kv_inputs, single_kv_tensors
    del logits, save_id, previous_prob
    gc.collect()

print("\nExport complete.\n")
print("─" * 70)
print("Starting ONNX Runtime inference …\n")
print("Loading sessions …")


# ══════════════════════════════════════════════════════════════════════════════
# ── ORT Session & Runtime Helpers ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def _build_run_options(silent: bool) -> onnxruntime.RunOptions:
    ro = onnxruntime.RunOptions()
    ro.log_severity_level  = 0 if not silent else 4
    ro.log_verbosity_level = 4
    ro.add_run_config_entry("disable_synchronize_execution_providers", "0")
    return ro


def _build_session_opts_ort() -> onnxruntime.SessionOptions:
    opts = onnxruntime.SessionOptions()
    opts.log_severity_level           = 0 if ORT_LOG else 4
    opts.log_verbosity_level          = 4
    opts.inter_op_num_threads         = MAX_THREADS
    opts.intra_op_num_threads         = MAX_THREADS
    opts.execution_mode               = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level     = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    _cfgs = {
        "session.set_denormal_as_zero":                  "1",
        "session.intra_op.allow_spinning":               "1",
        "session.inter_op.allow_spinning":               "1",
        "session.enable_quant_qdq_cleanup":              "1",
        "session.qdq_matmulnbits_accuracy_level":        "2" if ORT_FP16 else "4",
        "session.use_device_allocator_for_initializers": "1",
        "session.graph_optimizations_loop_level":        "2",
        "optimization.enable_gelu_approximation":        "1",
        "optimization.minimal_build_optimizations":      "",
        "optimization.enable_cast_chain_elimination":    "1",
        "optimization.disable_specified_optimizers":
            "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" if ORT_FP16 else "",
    }
    for k, v in _cfgs.items():
        opts.add_session_config_entry(k, v)
    return opts


# ── Execution provider setup ──────────────────────────────────────────────────
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
    device_type      = "cpu"
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
    device_type      = "cuda"
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
    device_type      = "dml"
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = "cpu"
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_obj  = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device        = "cpu" if "dml" in device_type else device_type

session_opts_ort = _build_session_opts_ort()
run_options      = _build_run_options(silent=not ORT_LOG)
disabled_opts    = (
    ["CastFloat16Transformer", "FuseFp16InitializerToFp32NodeTransformer"]
    if ORT_FP16 else None
)

_packed = dict(
    sess_options=session_opts_ort,
    providers=ORT_Accelerate_Providers or ["CPUExecutionProvider"],
    provider_options=provider_options,
    disabled_optimizers=disabled_opts,
)


def _make_session(path: str) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(path, **_packed)


def _ort_from_numpy(arr: np.ndarray) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(arr, device_type, DEVICE_ID)


def _ort_zeros(shape: tuple[int, ...], dtype: np.dtype) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device_type, DEVICE_ID)


def _ort_from_data(data, dtype: np.dtype) -> onnxruntime.OrtValue:
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device_type, DEVICE_ID)


def _bind_inputs(binding, names, values) -> None:
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def _bind_device_outputs(binding, names) -> None:
    for name in names:
        binding._iobinding.bind_output(name, _ort_device_obj)


def _run(session, binding) -> None:
    session.run_with_iobinding(binding, run_options=run_options)


def _in_names(session):
    return [x.name for x in session.get_inputs()]


def _out_names(session):
    return [x.name for x in session.get_outputs()]


# ══════════════════════════════════════════════════════════════════════════════
# ── Load all ORT Sessions ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
ort_session_Encoder        = _make_session(onnx_model_Encoder)
ort_session_Embed          = _make_session(onnx_model_Embed)
ort_session_Rotary_Prefill = _make_session(onnx_model_Rotary_Mask_Prefill)
ort_session_Rotary_Decode  = _make_session(onnx_model_Rotary_Mask_Decode)
ort_session_Main           = _make_session(onnx_model_Main)
ort_session_Concat         = _make_session(onnx_model_Concat_Embed)
print(f"  Usable Providers : {ort_session_Main.get_providers()}")

binding_Encoder        = ort_session_Encoder.io_binding()
binding_Embed          = ort_session_Embed.io_binding()
binding_Rotary_Prefill = ort_session_Rotary_Prefill.io_binding()
binding_Rotary_Decode  = ort_session_Rotary_Decode.io_binding()
binding_Main           = ort_session_Main.io_binding()
binding_Concat         = ort_session_Concat.io_binding()

in_name_Encoder,  out_name_Encoder  = _in_names(ort_session_Encoder),        _out_names(ort_session_Encoder)
in_name_Embed,    out_name_Embed    = _in_names(ort_session_Embed),          _out_names(ort_session_Embed)
in_name_RP,       out_name_RP       = _in_names(ort_session_Rotary_Prefill), _out_names(ort_session_Rotary_Prefill)
in_name_RD,       out_name_RD       = _in_names(ort_session_Rotary_Decode),  _out_names(ort_session_Rotary_Decode)
in_name_Main,     out_name_Main     = _in_names(ort_session_Main),           _out_names(ort_session_Main)
in_name_Concat,   out_name_Concat   = _in_names(ort_session_Concat),         _out_names(ort_session_Concat)

num_keys_values              = len(out_name_Main) - 1
main_hidden_idx              = num_keys_values
main_rotary_cos_idx          = main_hidden_idx + 1
main_rotary_sin_idx          = main_hidden_idx + 2
main_attention_mask_idx      = main_hidden_idx + 3
beam_kv_logits_count         = num_keys_values + 1
beam_save_id_idx             = beam_kv_logits_count
first_beam_size_idx          = beam_save_id_idx + 1
second_beam_previous_prob_idx = beam_save_id_idx + 1
second_beam_size_idx         = beam_save_id_idx + 2
second_beam_top_k_idx        = beam_save_id_idx + 3
beam_save_id_out_idx         = num_keys_values
beam_score_out_idx           = beam_save_id_out_idx + 1
beam_ids_out_idx             = beam_save_id_out_idx + 2
beam_max_idx                 = beam_save_id_out_idx + 3

_meta           = ort_session_Main._inputs_meta
hidden_size     = ort_session_Embed._outputs_meta[0].shape[2]
kv_dtype_str    = _meta[0].type
kv_dtype_np     = np.float16 if "float16" in kv_dtype_str else np.float32
hidden_dtype_np = np.float16 if "float16" in _meta[main_hidden_idx].type else np.float32
logits_meta     = ort_session_Main._outputs_meta[beam_save_id_out_idx]
logits_dtype_np = np.float16 if "float16" in logits_meta.type else np.float32
vocab_size      = logits_meta.shape[1]

num_layers_rt        = num_keys_values // 2
in_name_Main_kv      = in_name_Main[:num_keys_values]
out_name_Main_kv     = out_name_Main[:num_keys_values]
out_name_Main_logits = out_name_Main[beam_save_id_out_idx]

past_keys_Main   = _ort_zeros((1, _meta[0].shape[1], 1, _meta[0].shape[3], 0), kv_dtype_np)
past_values_Main = _ort_zeros((1, _meta[num_layers_rt].shape[1], 1, 0, _meta[num_layers_rt].shape[4]), kv_dtype_np)


# ══════════════════════════════════════════════════════════════════════════════
# ── Decoding Strategy Validation & Session Loading ────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE
if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("  [WARNING] Beam search requires BEAM_SIZE≥2 and TOP_K≥2; falling back to greedy.")
if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY    = (REPEAT_PENALTY != 1.0)
STOP_TOKEN_SET = set(STOP_TOKEN)

init_history_len_ort = _ort_from_data([0],         np.int64)
top_k_ort            = _ort_from_data([TOP_K],     np.int64)
beam_size_ort        = _ort_from_data([BEAM_SIZE], np.int64)

rotary_meta     = ort_session_Rotary_Decode._outputs_meta
rotary_dtype_np = np.float16 if "float16" in rotary_meta[0].type else np.float32
decode_batch    = BEAM_SIZE if USE_BEAM_SEARCH else 1

rotary_cos_buf     = _ort_zeros(tuple(int(dim) for dim in rotary_meta[0].shape), rotary_dtype_np)
rotary_sin_buf     = _ort_zeros(tuple(int(dim) for dim in rotary_meta[1].shape), rotary_dtype_np)
hidden_states_buf  = _ort_zeros((decode_batch, 1, hidden_size),              hidden_dtype_np)
save_id_buf        = _ort_zeros((BEAM_SIZE if USE_BEAM_SEARCH else 1, 0),    np.int32)
attention_mask_buf = _ort_zeros((1, 1, 1, 1, 1),                             hidden_dtype_np)
prefill_logits_buf = _ort_zeros((1, vocab_size),                             logits_dtype_np)
decode_logits_buf  = _ort_zeros((decode_batch, vocab_size),                  logits_dtype_np)
max_idx_buf        = _ort_zeros((1, 1),                                      np.int32)

if USE_BEAM_SEARCH:
    print("\n  [INFO] Beam search active — transcription is shown only after full decode.\n")
    beam_ids_buf   = _ort_zeros((BEAM_SIZE, 1), np.int32)
    beam_score_buf = _ort_zeros((BEAM_SIZE, 1), logits_dtype_np)

    ort_session_First_Beam  = _make_session(onnx_model_First_Beam)
    ort_session_Second_Beam = _make_session(onnx_model_Second_Beam)
    binding_First_Beam      = ort_session_First_Beam.io_binding()
    binding_Second_Beam     = ort_session_Second_Beam.io_binding()
    in_name_First_Beam      = _in_names(ort_session_First_Beam)
    out_name_First_Beam     = _out_names(ort_session_First_Beam)
    in_name_Second_Beam     = _in_names(ort_session_Second_Beam)
    out_name_Second_Beam    = _out_names(ort_session_Second_Beam)

    in_name_First_Beam_parts   = in_name_First_Beam[:beam_kv_logits_count]
    out_name_First_Beam_parts  = out_name_First_Beam[:beam_kv_logits_count]
    in_name_Second_Beam_parts  = in_name_Second_Beam[:beam_kv_logits_count]
    out_name_Second_Beam_parts = out_name_Second_Beam[:beam_kv_logits_count]

    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[beam_save_id_idx],        save_id_buf)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[first_beam_size_idx],     beam_size_ort)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_size_idx],  beam_size_ort)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_top_k_idx], top_k_ort)

else:
    ort_session_Greedy = _make_session(onnx_model_Greedy)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = _in_names(ort_session_Greedy)
    out_name_Greedy    = _out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1],   save_id_buf)
    binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)

    ort_session_Argmax = _make_session(onnx_model_Argmax)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = _in_names(ort_session_Argmax)[0]
    out_name_Argmax    = _out_names(ort_session_Argmax)[0]
    binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

if USE_PENALTY:
    ort_session_Penalty = _make_session(onnx_model_Penalty)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = _in_names(ort_session_Penalty)
    out_name_Penalty    = _out_names(ort_session_Penalty)[0]
    num_penalty_inputs  = len(in_name_Penalty)
    if num_penalty_inputs > 2:
        penalty_dtype_np = np.float16 if "float16" in ort_session_Penalty._inputs_meta[2].type else np.float32
        penalty_value_ort = _ort_from_data([REPEAT_PENALTY], penalty_dtype_np)
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value_ort)
    if num_penalty_inputs > 3:
        penalty_range_ort = _ort_from_data([PENALTY_RANGE], np.int64)
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range_ort)


# ══════════════════════════════════════════════════════════════════════════════
# ── Audio Normaliser ─────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def normalise_audio(audio: np.ndarray, target_rms: float = 8192.0) -> np.ndarray:
    _audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(_audio * _audio, dtype=np.float32), dtype=np.float32)
    if rms > 0:
        _audio *= (target_rms / (rms + 1e-7))
        np.clip(_audio, -32768.0, 32767.0, out=_audio)
        return _audio.astype(np.int16)
    else:
        return audio
        

binding_Rotary_Prefill.bind_ortvalue_input(in_name_RP[1], init_history_len_ort)
binding_Rotary_Decode.bind_ortvalue_output(out_name_RD[0], rotary_cos_buf)
binding_Rotary_Decode.bind_ortvalue_output(out_name_RD[1], rotary_sin_buf)


# ══════════════════════════════════════════════════════════════════════════════
# ── Pre-embed Prompt Queries ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
test_audio_list = [test_audio] if isinstance(test_audio, str) else list(test_audio)
if len(TASK_PROMPTS) == 1:
    task_prompt_list = TASK_PROMPTS * len(test_audio_list)
elif len(TASK_PROMPTS) != len(test_audio_list):
    raise ValueError("TASK_PROMPTS must contain either one prompt or one prompt per test audio.")
else:
    task_prompt_list = TASK_PROMPTS

if len(LANGUAGE_PROMPTS) == 1:
    language_prompt_list = LANGUAGE_PROMPTS * len(test_audio_list)
elif len(LANGUAGE_PROMPTS) != len(test_audio_list):
    raise ValueError("LANGUAGE_PROMPTS must contain either one prompt or one prompt per test audio.")
else:
    language_prompt_list = LANGUAGE_PROMPTS

init_all_outputs_Embed: List[onnxruntime.OrtValue] = []
for prompt in task_prompt_list:
    if prompt:
        prompt_ids = np.array([build_query_prompt_ids(tokenizer, prompt)], dtype=np.int32)
        prompt_ids_ort = _ort_from_numpy(prompt_ids)
        binding_Embed.bind_ortvalue_input(in_name_Embed[0], prompt_ids_ort)
        _bind_device_outputs(binding_Embed, [out_name_Embed[0]])
        _run(ort_session_Embed, binding_Embed)
        prompt_embed = _ort_from_data(binding_Embed.get_outputs()[0].numpy(), hidden_dtype_np)  # Create a new ort tensor.
    else:
        prompt_embed = _ort_zeros((1, 0, hidden_size), hidden_dtype_np)
    init_all_outputs_Embed.append(prompt_embed)

# Pre-embed language name prompts (just the name, e.g. "Chinese")
# The encoder tail already ends with the "language " prefix; CONCAT_EMBED only adds
# the forced language name and the "<asr_text>" tag.
init_all_outputs_Lang_Embed: List[Optional[onnxruntime.OrtValue]] = []
for lang_name in language_prompt_list:
    if lang_name:
        lang_ids = np.array([tokenizer.encode(lang_name, add_special_tokens=False)], dtype=np.int32)
        lang_ids_ort = _ort_from_numpy(lang_ids)
        binding_Embed.bind_ortvalue_input(in_name_Embed[0], lang_ids_ort)
        _bind_device_outputs(binding_Embed, [out_name_Embed[0]])
        _run(ort_session_Embed, binding_Embed)
        lang_embed = _ort_from_data(binding_Embed.get_outputs()[0].numpy(), hidden_dtype_np)  # Create a new ort tensor.
        init_all_outputs_Lang_Embed.append(lang_embed)
    else:
        init_all_outputs_Lang_Embed.append(None)

binding_Embed.bind_ortvalue_output(out_name_Embed[0], hidden_states_buf)
if USE_BEAM_SEARCH:
    binding_Embed.bind_ortvalue_input(in_name_Embed[0], beam_ids_buf)
else:
    binding_Embed.bind_ortvalue_input(in_name_Embed[0], max_idx_buf)


# ══════════════════════════════════════════════════════════════════════════════
# ── Main Inference Loop ───────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
for prompt_embed, lang_embed, system_prompt, lang_prompt, test_path in zip(
    init_all_outputs_Embed, init_all_outputs_Lang_Embed, task_prompt_list, language_prompt_list, test_audio_list
):
    try:
        audio_seg = AudioSegment.from_file(test_path)
        audio_pcm = np.array(audio_seg.set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.int16)
    except Exception:
        print(f"\n  [WARN] Cannot load '{test_path}'")
        continue

    print(f"\nTest audio : {test_path}   ({len(audio_pcm) / SAMPLE_RATE:.2f} s)")
    if system_prompt:
        print(f"  System prompt : {system_prompt}")
    if lang_prompt:
        print(f"  Language prompt: {lang_prompt}")
    print("─" * 70)

    audio_pcm = normalise_audio(audio_pcm)
    audio_len = len(audio_pcm)
    audio_pcm = audio_pcm[:MAX_INPUT_AUDIO_LENGTH]
    audio_np  = audio_pcm.reshape(1, 1, -1)
    audio_ort = _ort_from_numpy(audio_np)

    t0 = time.time()

    # Pass only the system prompt embed to the encoder
    binding_Encoder.bind_ortvalue_input(in_name_Encoder[0], audio_ort)
    binding_Encoder.bind_ortvalue_input(in_name_Encoder[1], prompt_embed)
    _bind_device_outputs(binding_Encoder, out_name_Encoder)
    _run(ort_session_Encoder, binding_Encoder)
    hidden_states, ids_len = binding_Encoder.get_outputs()

    # Append language directive after the encoder output (after "assistant\n")
    if lang_embed:
        binding_Concat.bind_ortvalue_input(in_name_Concat[0], hidden_states)
        binding_Concat.bind_ortvalue_input(in_name_Concat[1], lang_embed)
        _bind_device_outputs(binding_Concat, out_name_Concat)
        _run(ort_session_Concat, binding_Concat)
        hidden_states, ids_len = binding_Concat.get_outputs()
        ids_len_val = ids_len.numpy()
    else:
        ids_len_val = ids_len.numpy()

    t_enc = time.time()
    print(f"  Encode done ({t_enc - t0:.3f}s)")

    binding_Rotary_Prefill.bind_ortvalue_input(in_name_RP[0], ids_len)
    _bind_device_outputs(binding_Rotary_Prefill, out_name_RP)
    _run(ort_session_Rotary_Prefill, binding_Rotary_Prefill)
    rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Prefill.get_outputs()

    binding_Rotary_Decode.bind_ortvalue_input(in_name_RD[0], kv_seq_len)
    binding_Rotary_Decode.bind_ortvalue_output(out_name_RD[2], kv_seq_len)

    binding_Main.bind_ortvalue_input(in_name_Main[main_hidden_idx], hidden_states)
    binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_cos_idx], rotary_cos)
    binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_sin_idx], rotary_sin)
    binding_Main.bind_ortvalue_input(in_name_Main[main_attention_mask_idx], attention_mask)
    for i in range(num_layers_rt):
        binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
    for i in range(num_layers_rt):
        binding_Main.bind_ortvalue_input(in_name_Main[num_layers_rt + i], past_values_Main)
    _bind_device_outputs(binding_Main, out_name_Main_kv)
    binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

    if USE_PENALTY:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
        binding_Penalty.bind_ortvalue_output(out_name_Penalty, prefill_logits_buf)

    if not USE_BEAM_SEARCH and USE_PENALTY:
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], prefill_logits_buf)
    elif not USE_BEAM_SEARCH:
        binding_Argmax.bind_ortvalue_input(in_name_Argmax, prefill_logits_buf)

    # Reset save_id for new audio to prevent cross-utterance penalty leakage
    if not USE_BEAM_SEARCH and USE_PENALTY:
        binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    is_prefill_step = True
    num_decode      = 0
    save_id         = save_id_buf
    generate_limit  = max(MAX_SEQ_LEN - 10 - ids_len_val, 0)

    while num_decode < generate_limit:
        _run(ort_session_Main, binding_Main)
        outputs_Main = binding_Main.get_outputs()

        if USE_PENALTY and num_decode >= PENALTY_RANGE:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
            _run(ort_session_Penalty, binding_Penalty)

        if USE_BEAM_SEARCH:
            if is_prefill_step:
                _bind_inputs(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
                _bind_device_outputs(binding_First_Beam, out_name_First_Beam_parts)
                binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[beam_score_out_idx], beam_score_buf)
                binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[beam_ids_out_idx], beam_ids_buf)
                binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[beam_max_idx], max_idx_buf)
                _run(ort_session_First_Beam, binding_First_Beam)
                outputs_Beam = binding_First_Beam.get_outputs()
            else:
                _bind_inputs(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
                _bind_device_outputs(binding_Second_Beam, out_name_Second_Beam_parts)
                if num_decode < 2:
                    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[second_beam_previous_prob_idx], beam_score_buf)
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[beam_score_out_idx], beam_score_buf)
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[beam_ids_out_idx], beam_ids_buf)
                    binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[beam_max_idx], max_idx_buf)
                _run(ort_session_Second_Beam, binding_Second_Beam)
                outputs_Beam = binding_Second_Beam.get_outputs()

            max_token = max_idx_buf.numpy().flat[0]
            if max_token in STOP_TOKEN_SET:
                break

            save_id = outputs_Beam[num_keys_values]
            _bind_inputs(binding_Main, in_name_Main_kv, outputs_Beam[:num_keys_values])
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[beam_save_id_idx], save_id)

        else:
            if USE_PENALTY:
                binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_obj)
                _run(ort_session_Greedy, binding_Greedy)
                save_id = binding_Greedy.get_outputs()[1]
            else:
                _run(ort_session_Argmax, binding_Argmax)

            max_token = max_idx_buf.numpy().flat[0]
            if max_token in STOP_TOKEN_SET:
                break

            if USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
            else:
                save_id_numpy[num_decode] = max_token

            _bind_inputs(binding_Main, in_name_Main_kv, outputs_Main[:num_keys_values])

        _bind_device_outputs(binding_Main, out_name_Main_kv)

        if is_prefill_step:
            binding_Main.bind_ortvalue_input(in_name_Main[main_hidden_idx], hidden_states_buf)
            binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_cos_idx], rotary_cos_buf)
            binding_Main.bind_ortvalue_input(in_name_Main[main_rotary_sin_idx], rotary_sin_buf)
            binding_Main.bind_ortvalue_input(in_name_Main[main_attention_mask_idx], attention_mask_buf)
            binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)

            if USE_PENALTY:
                binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
                binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

            if not USE_BEAM_SEARCH and USE_PENALTY:
                binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
            elif not USE_BEAM_SEARCH:
                binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

            is_prefill_step = False

        _run(ort_session_Embed, binding_Embed)
        _run(ort_session_Rotary_Decode, binding_Rotary_Decode)
        num_decode += 1

    if USE_BEAM_SEARCH or USE_PENALTY:
        decoded_ids = save_id.numpy()[0]
    else:
        decoded_ids = save_id_numpy[:num_decode]

    raw_result = tokenizer.decode(decoded_ids, skip_special_tokens=True).strip()
    if not lang_embed and raw_result:
        raw_result = _LANG_PREFIX + raw_result  # encoder primed "language "; restore it for parsing
    detected_language, asr_result = parse_asr_output(raw_result)
    t_total = time.time() - t0
    rtf     = t_total / max(audio_len / SAMPLE_RATE, 1e-6)

    if detected_language:
        print(f"\nDetected language : {detected_language}")
    print(f"\nTranscription:\n  {asr_result}")
    print(f"\nRTF : {rtf:.3f}   total {t_total:.2f}s")
    print("─" * 70)
    
