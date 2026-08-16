import gc
import json
import subprocess
import sys
import os
import copy
import shutil
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
import dolphin
import torchaudio.compliance.kaldi as kaldi   # Used at export time to bake Kaldi's exact triangular mel filterbank as a constant.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = Path(_SCRIPT_DIR).parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ============================================================================================
#                                       User configuration
# ============================================================================================
model_path = str(Path.home() / "Downloads" / "dolphin-cn-dialect-small-prompt")  # Dolphin-CN-Dialect project path.

_MODEL_VARIANT             = "small.cn.prompt"
_MODEL_MAX_AUDIO_SAMPLES   = 480000     # 30 seconds at 16 kHz; exported as runtime metadata.
_MODEL_MAX_DECODER_SEQ_LEN = 448        # Must not exceed the 5000-position decoder table.
INPUT_AUDIO_DTYPE          = "F32"      # "INT16", "F32", or "F16". F32/F16 still carry int16-range values (no /32768).
USE_FP16_KV                = True       # Keep cache, cross-KV, position, and mask storage in FP16 for normal deployment exports.
COMPUTE_IN_F32             = False      # FP16-cache-only option: upcast attention compute while retaining FP16 storage.
REORDER_DOWNPROJ_FOR_QUANT = True       # Exact FFN intermediate-channel reorder, absorbed into w_1/w_2.
REORDER_OPROJ_FOR_QUANT    = True       # Exact self-attention V/linear_out per-head reorder.
REORDER_KEY                = "absmean"  # "absmean" | "L4" | "rms" | "std".
OPSET                      = 20         # ONNX opset version.
KV_DTYPE                   = torch.float16 if USE_FP16_KV else torch.float32


ONNX_DIR = Path(_SCRIPT_DIR) / "Dolphin_CN_Dialect_ONNX"
_raw_onnx_temp = tempfile.TemporaryDirectory(prefix="dolphin-cn-export-")
_raw_onnx_dir = Path(_raw_onnx_temp.name)

MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Dolphin_Encoder.onnx",
    "main": "Dolphin_Decoder.onnx",
    "embed": "Dolphin_Decoder_Embed.onnx",
    "position_prefill": "Dolphin_Position_Mask_Prefill.onnx",
    "position_decode": "Dolphin_Position_Mask_Decode.onnx",
    "greedy": "Dolphin_Greedy_Search.onnx",
    "argmax": "Dolphin_Argmax.onnx",
    "sampling": "Dolphin_TopKTopPSampling.onnx",
    "penalty": "Dolphin_Apply_Penalty.onnx",
    "prefill_greedy": "Dolphin_PrefillGreedy.onnx",
    "prefill_penalty_greedy": "Dolphin_PrefillPenaltyGreedy.onnx",
    "prefill_sampling": "Dolphin_PrefillSampling.onnx",
    "decode_greedy": "Dolphin_DecodeGreedy.onnx",
    "decode_penalty_greedy": "Dolphin_DecodePenaltyGreedy.onnx",
    "decode_sampling": "Dolphin_DecodeSampling.onnx",
    "shared_initializers": "Dolphin_SharedInitializers.onnx",
}
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"

onnx_model_Metadata = str(_raw_onnx_dir / MODEL_FILE_NAMES["metadata"])
onnx_model_Encoder = str(_raw_onnx_dir / MODEL_FILE_NAMES["encoder"])
onnx_model_Decoder = str(_raw_onnx_dir / MODEL_FILE_NAMES["main"])
onnx_model_Embed = str(_raw_onnx_dir / MODEL_FILE_NAMES["embed"])
onnx_model_Prefill = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_prefill"])
onnx_model_Decode = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_decode"])
onnx_model_Greedy = str(_raw_onnx_dir / MODEL_FILE_NAMES["greedy"])
onnx_model_Argmax = str(_raw_onnx_dir / MODEL_FILE_NAMES["argmax"])
onnx_model_Sampling = str(_raw_onnx_dir / MODEL_FILE_NAMES["sampling"])
onnx_model_Penalty = str(_raw_onnx_dir / MODEL_FILE_NAMES["penalty"])
save_vocab = str(_raw_onnx_dir / "vocab_Dolphin_CN_Dialect.txt")


# Dolphin-CN-Dialect has no frontend_conf; dolphin.processor.extract_feats uses
# torchaudio.compliance.kaldi.fbank with these model-trained Kaldi defaults.
WINDOW_TYPE   = "povey"
N_MELS        = 80
NFFT_STFT     = 512
WINDOW_LENGTH = 400
HOP_LENGTH    = 160
PRE_EMPHASIZE = 0.97
LOW_FREQ      = 20.0
SAMPLE_RATE   = 16000

_METADATA_PENALTY_RANGE = 20



if HOP_LENGTH > _MODEL_MAX_AUDIO_SAMPLES:
    HOP_LENGTH = _MODEL_MAX_AUDIO_SAMPLES
class Tokenizer:
    # Char tokenizer for Dolphin-CN-Dialect (no bpe.model). Chinese characters map 1:1; English BPE-style pieces
    # carry the SentencePiece word-boundary marker "▁", which is rendered back as a space at detokenisation.
    def __init__(self, filename):
        self.str_to_idx = {}
        self.idx_to_str = {}
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def decode_ids(self, ids):
        tokens = [self.decode(int(idx)) for idx in ids]
        tokens = [token for token in tokens if token is not None]
        return ''.join(tokens).replace("▁", " ").strip()


def _bias_or_zero(linear):
    return linear.bias if linear.bias is not None else torch.zeros(linear.out_features, dtype=linear.weight.dtype)


def fold_norm_into_linear(norm, linear):
    # Absorb a LayerNorm affine (gamma/beta) forward into the next Linear: W'=W*gamma, b'=b+W@beta.
    # The LayerNorm is left affine-free so its forward call still performs the (x-mean)/std normalisation.
    linear.bias.data.add_(linear.weight.data @ norm.bias.data)
    linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.weight.data.fill_(1.0)
    norm.bias.data.zero_()


def kaldi_window(window_type, win_length):
    window_type = window_type.lower()
    if window_type == "hamming":
        return torch.hamming_window(win_length, periodic=False, alpha=0.54, beta=0.46)
    if window_type in ("hanning", "hann"):
        return torch.hann_window(win_length, periodic=False)
    if window_type == "povey":
        return torch.hann_window(win_length, periodic=False).pow(0.85)
    if window_type == "rectangular":
        return torch.ones(win_length, dtype=torch.float32)
    if window_type == "blackman":
        blackman_coeff = 0.42
        n = torch.arange(win_length, dtype=torch.float32)
        angle = 2.0 * torch.pi / (win_length - 1)
        return blackman_coeff - 0.5 * torch.cos(angle * n) + (0.5 - blackman_coeff) * torch.cos(2.0 * angle * n)
    raise ValueError(f"Unsupported Kaldi window type: {window_type}")


def create_kaldi_stft_kernel(n_fft, win_length, window_type, pre_emphasis):
    # Bake Kaldi's per-frame pipeline (DC removal -> pre-emphasis -> window -> DFT) into a single Conv1d kernel,
    # so the whole fbank front-end is one Conv1d in the exported graph.
    window = kaldi_window(window_type, win_length)
    freq = torch.arange(n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
    time_index = torch.arange(win_length, dtype=torch.float32).unsqueeze(0)
    omega = (2.0 * torch.pi / n_fft) * freq * time_index
    real_basis = torch.cos(omega) * window.unsqueeze(0)
    imag_basis = -torch.sin(omega) * window.unsqueeze(0)

    dc_remove = torch.eye(win_length, dtype=torch.float32) - torch.full((win_length, win_length), 1.0 / win_length, dtype=torch.float32)
    previous_sample = torch.zeros((win_length, win_length), dtype=torch.float32)
    previous_sample[0, 0] = 1.0
    previous_sample[1:, :-1] = torch.eye(win_length - 1, dtype=torch.float32)
    pre_emphasis_matrix = torch.eye(win_length, dtype=torch.float32) - float(pre_emphasis) * previous_sample
    frame_transform = torch.matmul(pre_emphasis_matrix, dc_remove)

    real_kernel = torch.matmul(real_basis, frame_transform)
    imag_kernel = torch.matmul(imag_basis, frame_transform)
    return torch.cat([real_kernel, imag_kernel], dim=0).unsqueeze(1)


class KaldiFbank(torch.nn.Module):
    # Numerically matches torchaudio.compliance.kaldi.fbank(dither=0, frame_length=25, frame_shift=10,
    # num_mel_bins=80, window_type='povey', use_power=True, use_log_fbank=True, snip_edges=True). The input audio
    # is raw int16 PCM (== the reference's waveform * (1 << 15)), so no /32768 rescaling is applied here.
    def __init__(self, n_fft, win_length, hop_len, n_mels, sample_rate, window_type, pre_emphasis, low_freq):
        super().__init__()
        self.hop_len = hop_len
        self.n_freqs = n_fft // 2 + 1
        self.register_buffer("stft_kernel", create_kaldi_stft_kernel(n_fft, win_length, window_type, pre_emphasis))
        mel_bins, _ = kaldi.get_mel_banks(n_mels, n_fft, sample_rate, low_freq, 0.0, 100.0, -500.0, 1.0)
        mel_bins = torch.nn.functional.pad(mel_bins, (0, 1), mode="constant", value=0.0)
        self.register_buffer("mel_bins", mel_bins.unsqueeze(0).to(torch.float32))
        self.register_buffer("epsilon", torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32))

    def forward(self, audio):
        stft = torch.nn.functional.conv1d(audio.float(), self.stft_kernel, stride=self.hop_len)
        real_power, imag_power = torch.split(stft * stft, self.n_freqs, dim=1)
        power = real_power + imag_power
        mel = torch.matmul(self.mel_bins, power)
        log_mel = torch.maximum(mel, self.epsilon).log()
        return log_mel.transpose(1, 2)


class GREEDY_SEARCH(torch.nn.Module):
    # Pure argmax that also appends the chosen token to the running save_id history (Qwen ASR style).
    # Used when a repetition penalty is active so APPLY_PENALTY can read the on-device history.
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        return max_logits_idx, torch.cat([save_id, max_logits_idx], dim=-1)


class ARGMAX(torch.nn.Module):
    # Bare argmax (Qwen ASR style); used for greedy decoding when no repetition penalty is applied.
    def __init__(self):
        super(ARGMAX, self).__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super(METADATA_CARRIER, self).__init__()

    def forward(self, marker):
        return marker


class APPLY_PENALTY(torch.nn.Module):
    # Sliding-window repetition penalty (Qwen ASR style): multiply the logits of the most recent
    # `penalty_range` tokens by `penalty_value`. Keep penalty_range as a live int64[1] graph input;
    # indexing element zero avoids `.item()` and the fragile scalar-Reshape Slice lowering.
    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalised = logits.gather(1, target_indices) * penalty_value
        return logits.scatter(1, target_indices, penalised)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_min", torch.tensor(self.GUMBEL_EPS, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_max",
            torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
        inv_penalty = torch.reciprocal(repetition_penalty)
        prev_logits = torch.gather(logits, 1, previous_ids)
        prev_scores = torch.where(
            prev_logits < 0.0,
            prev_logits * repetition_penalty,
            prev_logits * inv_penalty,
        )
        scores = torch.scatter(logits, 1, previous_ids, prev_scores)
        scores = scores * torch.reciprocal(temperature)

        sorted_scores, sorted_indices = torch.topk(
            scores, k=top_k, dim=-1, largest=True, sorted=True
        )
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep_topp = (sorted_cumsum - sorted_probs) <= top_p
        sorted_scores = torch.where(keep_topp, sorted_scores, self.neg_inf)

        noise = torch.clamp(
            torch.rand_like(sorted_scores), self.gumbel_min, self.gumbel_max
        )
        gumbel = -torch.log(-torch.log(noise))
        winner = torch.argmax(sorted_scores + gumbel, dim=-1, keepdim=True)
        sampled_id = torch.gather(sorted_indices, 1, winner).int()
        save_id = torch.cat([previous_ids, sampled_id], dim=-1)
        return sampled_id, save_id


class DOLPHIN_ENCODER(torch.nn.Module):
    def __init__(self, dolphin, fbank_model, num_layers_de):
        super(DOLPHIN_ENCODER, self).__init__()
        encoder = copy.deepcopy(dolphin.encoder)
        self.fbank_model = fbank_model

        # GlobalCMVN (JSON global_cmvn): forward is (x - mean) * istd, where istd is already the inverse std.
        self.register_buffer("cmvn_mean", encoder.global_cmvn.mean.detach().float().clone())
        self.register_buffer("cmvn_istd", encoder.global_cmvn.istd.detach().float().clone())

        # Encoder components
        self.subsampling_conv = encoder.embed.conv
        self.embed = encoder.embed.out[0]                            # Conv2dSubsampling4 projection (Linear 14592 -> 768)
        position_encode = encoder.embed.pos_enc                      # RelPositionalEncoding (rel_pos, NOT rel_pos_v1)
        # Conv2dSubsampling4 applies x = x * xscale inside pos_enc; fold that scale into the projection here.
        self.embed.weight.data *= position_encode.xscale
        self.embed.bias.data *= position_encode.xscale
        self.encoders = encoder.encoders
        self.num_heads = self.encoders[0].attn.h
        self.head_dim = self.encoders[0].attn.d_k
        self.hidden_size = self.embed.out_features
        self.cgmlp_split_size = self.encoders[0].cgmlp.channel_proj2.in_features
        self.cross_num_heads = dolphin.decoder.decoders[0].src_attn.h
        self.cross_head_dim = dolphin.decoder.decoders[0].src_attn.d_k
        self.after_norm_eps = encoder.after_norm.eps
        self.register_buffer("affine_free_norm_weight", torch.ones(self.hidden_size, dtype=torch.float32))
        self.register_buffer("affine_free_norm_bias", torch.zeros(self.hidden_size, dtype=torch.float32))
        self._fuse_weights(encoder.after_norm, dolphin.decoder.decoders)
        # Pre-apply linear_pos + view + permute once per layer over the full pe; forward slices all layers then gathers.
        # Preserve the previous f16-table numerical contract while storing only the final per-layer table.
        pe_full = position_encode.pe.detach().to(KV_DTYPE).float()
        self.register_buffer(
            "position_projection",
            torch.stack(
                [
                    encoder_layer.attn.linear_pos(pe_full)
                    .view(-1, self.num_heads, self.head_dim)
                    .permute(1, 2, 0)
                    for encoder_layer in self.encoders
                ],
                dim=0,
            ).to(KV_DTYPE),
        )
        for encoder_layer in self.encoders:
            del encoder_layer.attn.linear_pos

    def _fuse_weights(self, after_norm, decoder_layers):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            for encoder_layer in self.encoders:
                attn = encoder_layer.attn
                out_features = attn.linear_q.out_features
                qkv = torch.nn.Linear(attn.linear_q.in_features, out_features * 3, bias=True)
                qkv.weight.copy_(torch.cat([attn.linear_q.weight, attn.linear_k.weight, attn.linear_v.weight], dim=0))
                qkv.bias.copy_(torch.cat([_bias_or_zero(attn.linear_q), _bias_or_zero(attn.linear_k), _bias_or_zero(attn.linear_v)], dim=0))
                qkv.weight.data[:out_features * 2].mul_(scale)
                qkv.bias.data[:out_features * 2].mul_(scale)
                attn.linear_pos.weight.data.mul_(scale)
                attn.pos_bias_u.data = attn.pos_bias_u.data.unsqueeze(1) * scale
                attn.pos_bias_v.data = attn.pos_bias_v.data.unsqueeze(1) * scale
                attn.qkv = qkv
                del attn.linear_q, attn.linear_k, attn.linear_v

                # Fold norm_mha -> qkv (pre-attention LayerNorm absorbed into the fused QKV linear).
                fold_norm_into_linear(encoder_layer.norm_mha, qkv)
                # Fold norm_mlp -> channel_proj1; norm_ff_macaron -> macaron w_1; norm_ff -> ff w_1.
                fold_norm_into_linear(encoder_layer.norm_mlp, encoder_layer.cgmlp.channel_proj1[0])
                fold_norm_into_linear(encoder_layer.norm_ff_macaron, encoder_layer.feed_forward_macaron.w_1)
                fold_norm_into_linear(encoder_layer.norm_ff, encoder_layer.feed_forward.w_1)
                # Absorb the 0.5 macaron ff_scale into both w_2 outputs, then make ff_scale a no-op.
                encoder_layer.feed_forward_macaron.w_2.weight.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward_macaron.w_2.bias.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward.w_2.weight.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward.w_2.bias.data.mul_(encoder_layer.ff_scale)
                encoder_layer.ff_scale = 1.0

            cross_scale = float(self.cross_head_dim ** -0.25)
            after_gamma = after_norm.weight.data.clone()
            after_beta = after_norm.bias.data.clone()
            decoder_layers = list(decoder_layers)
            self.cross_kv_projections = torch.nn.ModuleList()
            self.cross_kv_width = 2 * self.cross_num_heads * self.cross_head_dim
            for decoder_layer in decoder_layers:
                cross_attn = decoder_layer.src_attn
                fused = torch.nn.Linear(
                    cross_attn.linear_k.in_features,
                    self.cross_kv_width,
                    bias=True,
                )
                key_weight = cross_attn.linear_k.weight.detach().clone().mul_(cross_scale)
                key_bias = _bias_or_zero(cross_attn.linear_k).detach().clone().mul_(cross_scale)
                fused.weight.copy_(
                    torch.cat((key_weight, cross_attn.linear_v.weight.detach().clone()), dim=0)
                )
                fused.bias.copy_(
                    torch.cat(
                        (key_bias, _bias_or_zero(cross_attn.linear_v).detach().clone()),
                        dim=0,
                    )
                )
                # Fold per layer to preserve the original reduction and output geometry exactly.
                fused.bias.add_(fused.weight @ after_beta)
                fused.weight.mul_(after_gamma)
                self.cross_kv_projections.append(fused)

    def _affine_free_norm(self, value, eps):
        return F.layer_norm(
            value,
            (self.hidden_size,),
            self.affine_free_norm_weight,
            self.affine_free_norm_bias,
            eps,
        )

    def forward(self, audio):
        # Kaldi fbank front-end (int16 audio -> log-mel) then GlobalCMVN, matching dolphin.processor.extract_feats
        # for models without a frontend_conf block.
        mel_features = self.fbank_model(audio)
        mel_features = (mel_features - self.cmvn_mean) * self.cmvn_istd
        embed = self.subsampling_conv(mel_features.unsqueeze(1))
        embed_len = embed.shape[-2]
        x = self.embed(embed.transpose(1, 2).reshape(1, -1, self.embed.in_features))
        pos_p = self.position_projection[:, :, :, :embed_len].float()
        for idx, encoder_layer in enumerate(self.encoders):
            x = x + encoder_layer.feed_forward_macaron(
                self._affine_free_norm(x, encoder_layer.norm_ff_macaron.eps)
            )  # ff_scale(0.5) already folded into macaron w_2
            # norm_mha and norm_mlp become the same affine-free LayerNorm after their affines are folded.
            x_normalized = self._affine_free_norm(x, encoder_layer.norm_mha.eps)
            x1 = x_normalized
            qkv = encoder_layer.attn.qkv(x1).view(-1, 3 * self.num_heads, self.head_dim).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)
            p = pos_p[idx]
            q_with_bias_u = q + encoder_layer.attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(1, 2))
            matrix_bd = torch.matmul(q_with_bias_v, p)                       # rel_pos + use_sdpa: NO rel_shift (pos_emb length == time)
            x1 = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v)
            x1 = encoder_layer.attn.linear_out(x1.transpose(0, 1).reshape(1, -1, self.hidden_size))
            x2 = encoder_layer.cgmlp.channel_proj1(x_normalized)
            x_r, x_g = torch.split(x2, self.cgmlp_split_size, dim=-1)
            x_g = encoder_layer.cgmlp.csgu.conv(encoder_layer.cgmlp.csgu.norm(x_g).transpose(1, 2)).transpose(1, 2)
            x2 = encoder_layer.cgmlp.channel_proj2(x_r * x_g)
            x_concat = torch.cat([x1, x2], dim=-1)
            x_concat = x_concat + encoder_layer.depthwise_conv_fusion(
                x_concat.transpose(1, 2)
            ).transpose(1, 2)
            x = x + encoder_layer.merge_proj(x_concat)
            x = x + encoder_layer.feed_forward(
                self._affine_free_norm(x, encoder_layer.norm_ff.eps)
            )  # ff_scale(0.5) already folded into ff w_2
            x = encoder_layer.norm_final(x)
        enc_outputs = self._affine_free_norm(x, self.after_norm_eps)
        save_en_keys = []
        save_en_values = []
        for cross_kv_projection in self.cross_kv_projections:
            cross_kv = cross_kv_projection(enc_outputs).to(KV_DTYPE).view(
                -1, 2 * self.cross_num_heads, self.cross_head_dim
            ).transpose(0, 1)
            key, value = cross_kv.split(self.cross_num_heads, dim=0)
            save_en_keys.append(key.transpose(1, 2))   # f16 key   (num_heads, head_dim, T)
            save_en_values.append(value)               # f16 value (num_heads, T, head_dim)
        return *save_en_keys, *save_en_values


class DOLPHIN_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Whisper/Qwen Decoder_Embed) so the int
    # token ids never enter the float-only decode graph. The positional xscale is folded into the embedding
    # weight here (the absolute position embedding itself is added inside the decoder main graph).
    def __init__(self, dolphin):
        super(DOLPHIN_DECODER_EMBED, self).__init__()
        self.embed = copy.deepcopy(dolphin.decoder.embed[0])
        self.embed.weight.data *= dolphin.decoder.embed[1].xscale

    def forward(self, input_ids):
        return self.embed(input_ids)


class DOLPHIN_PREFILL(torch.nn.Module):
    # Prefill-phase position-embedding + causal-mask generator (mirrors Whisper/Qwen Prefill).
    # Consumes the int lengths and emits float position embedding + float attention mask so the decoder
    # main graph stays integer-free.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_PREFILL, self).__init__()
        self.emit_fp32_mask = USE_FP16_KV and COMPUTE_IN_F32
        self.register_buffer(
            "position_weight",
            dolphin.decoder.embed[1].pe[:, :max_seq_len].detach().to(KV_DTYPE).clone(),
        )
        self.register_buffer(
            "attention_mask",
            (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128,
        )

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len].to(KV_DTYPE)
        if self.emit_fp32_mask:
            attention_mask = attention_mask.float()
        return position_embed, attention_mask, kv_seq_len


class DOLPHIN_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Whisper/Qwen Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_DECODE, self).__init__()
        self.register_buffer(
            "position_weight",
            dolphin.decoder.embed[1].pe[:, :max_seq_len].detach().to(KV_DTYPE).clone(),
        )

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class DOLPHIN_DECODER(torch.nn.Module):
    def __init__(self, dolphin, num_layers_de):
        super(DOLPHIN_DECODER, self).__init__()
        decoder = copy.deepcopy(dolphin.decoder)
        self.decoders = decoder.decoders
        self.output_layer = decoder.output_layer
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = not USE_FP16_KV or COMPUTE_IN_F32
        self.idx_en_key = num_layers_de + num_layers_de         # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + num_layers_de     # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + num_layers_de     # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                 # position-embedding input (4 * L + 1)
        self.num_heads = self.decoders[0].self_attn.h
        self.head_dim = self.decoders[0].self_attn.d_k
        self.hidden_size = self.output_layer.in_features
        self.cross_num_heads = self.decoders[0].src_attn.h
        self.cross_head_dim = self.decoders[0].src_attn.d_k
        self.after_norm_eps = decoder.after_norm.eps
        self.register_buffer("affine_free_norm_weight", torch.ones(self.hidden_size, dtype=torch.float32))
        self.register_buffer("affine_free_norm_bias", torch.zeros(self.hidden_size, dtype=torch.float32))
        self._fuse_weights(decoder.after_norm)
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _fuse_weights(self, after_norm):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.decoders:
                attn = decoder_layer.self_attn
                out_features = attn.linear_q.out_features
                qkv = torch.nn.Linear(attn.linear_q.in_features, out_features * 3, bias=True)
                qkv.weight.copy_(torch.cat([attn.linear_q.weight, attn.linear_k.weight, attn.linear_v.weight], dim=0))
                qkv.bias.copy_(torch.cat([_bias_or_zero(attn.linear_q), _bias_or_zero(attn.linear_k), _bias_or_zero(attn.linear_v)], dim=0))
                qkv.weight.data[:out_features * 2].mul_(scale)
                qkv.bias.data[:out_features * 2].mul_(scale)
                attn.qkv = qkv
                del attn.linear_q, attn.linear_k, attn.linear_v

                cross_attn = decoder_layer.src_attn
                cross_attn.linear_q.weight.data.mul_(cross_scale)
                cross_attn.linear_q.bias.data.mul_(cross_scale)

                # Fold the decoder layer norms forward: norm1->self qkv, norm2->cross linear_q, norm3->w_1.
                fold_norm_into_linear(decoder_layer.norm1, qkv)
                fold_norm_into_linear(decoder_layer.norm2, cross_attn.linear_q)
                fold_norm_into_linear(decoder_layer.norm3, decoder_layer.feed_forward.w_1)
            # Absorb the decoder's final after_norm into the output projection.
            fold_norm_into_linear(after_norm, self.output_layer)

    def _affine_free_norm(self, value, eps):
        return F.layer_norm(
            value,
            (self.hidden_size,),
            self.affine_free_norm_weight,
            self.affine_free_norm_bias,
            eps,
        )

    @staticmethod
    def _channel_stat(weight, key, dims):
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(dim=dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            if isinstance(dims, tuple):
                return weight.reshape(-1, weight.shape[-1]).std(0)
            return weight.std(dim=dims)
        if key != "absmean":
            raise ValueError(f"Unsupported REORDER_KEY: {key!r}")
        return absolute.mean(dim=dims)

    def _reorder_downproj_for_quant(self, key):
        """Permute decoder w_1 outputs and matching w_2 input columns."""
        with torch.no_grad():
            for decoder_layer in self.decoders:
                w_1 = decoder_layer.feed_forward.w_1
                w_2 = decoder_layer.feed_forward.w_2
                permutation = torch.argsort(self._channel_stat(w_2.weight, key, 0))
                w_1.weight.copy_(w_1.weight[permutation])
                if w_1.bias is not None:
                    w_1.bias.copy_(w_1.bias[permutation])
                w_2.weight.copy_(w_2.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute each self-attention V head and matching linear_out columns."""
        with torch.no_grad():
            for decoder_layer in self.decoders:
                attention = decoder_layer.self_attn
                output_weight = attention.linear_out.weight
                qkv = attention.qkv
                output_by_head = output_weight.view(
                    output_weight.shape[0], self.num_heads, self.head_dim
                )
                permutations = [
                    torch.argsort(self._channel_stat(output_by_head[:, head], key, 0))
                    for head in range(self.num_heads)
                ]

                reordered_output = output_by_head.clone()
                for head, permutation in enumerate(permutations):
                    reordered_output[:, head] = output_by_head[:, head, permutation]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                value_weight = qkv.weight[2 * self.hidden_size:].view(
                    self.num_heads, self.head_dim, qkv.in_features
                )
                reordered_value_weight = value_weight.clone()
                for head, permutation in enumerate(permutations):
                    reordered_value_weight[head] = value_weight[head, permutation]
                qkv.weight[2 * self.hidden_size:].copy_(
                    reordered_value_weight.reshape(self.hidden_size, qkv.in_features)
                )

                if qkv.bias is not None:
                    value_bias = qkv.bias[2 * self.hidden_size:].view(
                        self.num_heads, self.head_dim
                    )
                    reordered_value_bias = value_bias.clone()
                    for head, permutation in enumerate(permutations):
                        reordered_value_bias[head] = value_bias[head, permutation]
                    qkv.bias[2 * self.hidden_size:].copy_(
                        reordered_value_bias.reshape(self.hidden_size)
                    )

    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        # Prefill emits the causal mask in attention-compute dtype, so every layer
        # shares it directly without an additional precision-boundary Cast.
        attn_mask = attention_mask
        save_de_keys = []
        save_de_values = []
        for idx, decoder_layer in enumerate(self.decoders):
            hidden_states_norm = self._affine_free_norm(hidden_states, decoder_layer.norm1.eps)
            # Self-attention. OFF (minimum-cast): cast the fused QKV DOWN to f16 before the split so
            # Q@K/mask/softmax/attn@V run in f16 on the f16 K/V cache; the context is cast back to f32 for linear_out.
            # ON (COMPUTE_IN_F32): keep the f16 K/V *storage* (K/V still cast to f16 before the cache concat, so
            # the cache I/O dtype is unchanged) but upcast K/V to f32 at the matmul use points and keep
            # Q/mask/softmax in f32 -- f16 storage, f32 compute. Q is never downcast.
            qkv = decoder_layer.self_attn.qkv(hidden_states_norm)
            if not self.compute_in_f32:
                qkv = qkv.half()
            qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = qkv.split(self.num_heads, dim=1)
            if self.compute_in_f32:
                k = k.to(KV_DTYPE)
                v = v.to(KV_DTYPE)
            k = torch.cat((all_inputs[idx], k.transpose(-1, -2)), dim=-1)           # f16 key cache   (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v), dim=-2)       # f16 value cache (batch, num_heads, kv_seq_len, head_dim)
            save_de_keys.append(k)
            save_de_values.append(v)
            if self.compute_in_f32:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k.float()) + attn_mask, dim=-1), v.float()).transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
            else:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attn_mask, dim=-1), v).transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float()
            hidden_state_attn = decoder_layer.self_attn.linear_out(hidden_state_attn)
            hidden_state_attn += hidden_states
            # Cross-attention against the f16 encoder cross-KV cache. OFF: downcast Q to f16 and run in f16 on the
            # f16 cross cache, context back to f32. ON: keep Q in f32 and upcast the f16 cross K/V to f32 at the
            # matmul use points (the cross cache is produced f16 by the encoder; its I/O dtype is unchanged).
            q = decoder_layer.src_attn.linear_q(
                self._affine_free_norm(hidden_state_attn, decoder_layer.norm2.eps)
            ).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            if self.compute_in_f32:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size))
            else:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(
                self._affine_free_norm(hidden_state_cross, decoder_layer.norm3.eps)
            )
        hidden_states = self._affine_free_norm(hidden_states[:, -1], self.after_norm_eps)
        logits = self.output_layer(hidden_states)
        return *save_de_keys, *save_de_values, logits


# ══════════════════════════════════════════════════════════════════════════════════
# ONNX METADATA  (store immutable runtime facts in ASR_Metadata.onnx)
# ──────────────────────────────────────────────────────────────────────────────────
# The inference runtime used to hard-code the special-token IDs, max_seq_len and sample_rate as
# constants that HAD to be kept in sync with this exporter. Stamping the same facts into the metadata
# carrier's `metadata_props` lets the runtime read them directly (in ONNX Runtime:
# InferenceSession.get_modelmeta().custom_metadata_map), removing the fragile "keep the inference
# constants in sync with the exporter" duplication. All values are stored as strings.
# ══════════════════════════════════════════════════════════════════════════════════


def build_model_metadata(*sections):
    """Merge metadata sections (dicts) into one normalized {str: str} map for ONNX metadata_props.

    bool -> '1'/'0', everything else -> str(); None values are dropped so optional facts stay absent
    instead of being written as the literal string 'None'.
    """
    def _norm(value):
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (dict, list)):
            return json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        return str(value)

    merged = {}
    for section in sections:
        for key, value in section.items():
            if value is not None:
                merged[key] = _norm(value)
    return merged


def build_cn_supported_languages(token_to_id, special_token_ids):
    """Build the prompt catalog from real CN language and region tokens."""
    language_start = special_token_ids["asr"] + 1
    region_start = token_to_id["<CN>"]
    reserved_region_ids = [
        token_id
        for token, token_id in token_to_id.items()
        if token.startswith("<DIALECT_RES_")
    ]
    region_end = min(reserved_region_ids)

    ordered_tokens = sorted(
        ((token_id, token) for token, token_id in token_to_id.items()),
        key=lambda item: item[0],
    )
    language_tokens = [
        (token[1:-1], token_id)
        for token_id, token in ordered_tokens
        if language_start <= token_id < region_start
        and token.startswith("<")
        and token.endswith(">")
    ]
    region_tokens = [
        (token[1:-1], token_id)
        for token_id, token in ordered_tokens
        if region_start <= token_id < region_end
        and token.startswith("<")
        and token.endswith(">")
    ]
    catalog = {
        "auto-auto": {
            "name": "auto-auto",
            "aliases": ["auto"],
            "prompt_token_ids": [],
        }
    }
    for language, language_id in language_tokens:
        catalog[f"{language}-auto"] = {
            "name": f"{language}-auto",
            "aliases": (
                [language, "Chinese", "中文"]
                if language == "zh"
                else [language, "English", "英文"]
            ),
            "prompt_token_ids": [language_id],
        }
        valid_regions = region_tokens if language == "zh" else region_tokens[:1]
        for region, region_id in valid_regions:
            code = f"{language}-{region}"
            catalog[code] = {
                "name": code,
                "aliases": [],
                "prompt_token_ids": [
                    language_id,
                    region_id,
                    special_token_ids["asr"],
                    special_token_ids["notimestamp"],
                ],
            }
    return catalog


def write_onnx_metadata(onnx_path, metadata, *, replace=True):
    """Write sorted `metadata_props`, optionally replacing the exact property map.

    load_external_data=False keeps any big `*.data` weights on disk untouched (only the graph proto +
    metadata are rewritten). Replacement is required for the metadata carrier so stale keys cannot
    survive; composed graph-local diagnostics use the default merge behavior.
    """
    import onnx  # lazy: only needed when actually exporting
    model = onnx.load(onnx_path, load_external_data=False)
    values = {} if replace else {prop.key: prop.value for prop in model.metadata_props}
    values.update({str(key): str(value) for key, value in metadata.items()})
    del model.metadata_props[:]
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save(model, onnx_path)


def write_metadata_carrier(onnx_path, metadata):
    write_onnx_metadata(
        onnx_path,
        {str(key): str(value) for key, value in metadata.items()},
        replace=True,
    )


print('\nExport start...\n')
_raw_onnx_dir.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    model = dolphin.load_model(_MODEL_VARIANT, model_path, "cpu")
    model.eval()
    # Build the vocab in token-id order from units.txt ("<token> <id>" per line),
    # mirroring dolphin.tokenizer._read_symbol_table parsing.
    os.makedirs(os.path.dirname(save_vocab), exist_ok=True)
    id_to_token = {}
    with open(os.path.join(model_path, "units.txt"), 'r', encoding='utf-8') as units_file:
        for vocab_line in units_file:
            arr = vocab_line.strip().split()
            if len(arr) >= 2:
                id_to_token[int(arr[1])] = arr[0]
    with open(save_vocab, 'w', encoding='utf-8') as file:
        for idx in range(len(id_to_token)):
            file.write(id_to_token[idx] + '\n')
    HIDDEN_SIZE = model.decoder.output_layer.in_features
    NUM_HEAD_EN = model.encoder.encoders._modules['0'].attn.h
    NUM_HEAD_DE = model.decoder.decoders._modules['0'].self_attn.h
    HEAD_DIM_DE = model.decoder.decoders._modules['0'].self_attn.d_k
    NUM_LAYER_DE = len(model.decoder.decoders)
    VOCAB_SIZE = model.vocab_size
    STFT_SIGNAL_LENGTH = (_MODEL_MAX_AUDIO_SAMPLES - WINDOW_LENGTH) // HOP_LENGTH + 1   # Kaldi snip_edges=True framing.

    custom_fbank = KaldiFbank(NFFT_STFT, WINDOW_LENGTH, HOP_LENGTH, N_MELS, SAMPLE_RATE, WINDOW_TYPE, PRE_EMPHASIZE, LOW_FREQ).eval()
    dolphin_encoder = DOLPHIN_ENCODER(model, custom_fbank, NUM_LAYER_DE)
    output_names = []
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, _MODEL_MAX_AUDIO_SAMPLES), dtype=_audio_export_dtype)
    dynamic_axes = {'audio': {2: 'audio_len'}}
    for i in range(NUM_LAYER_DE):
        name = f'en_key_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'signal_len'}

    torch.onnx.export(
        dolphin_encoder,
        (audio,),
        onnx_model_Encoder,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del dolphin_encoder
    del audio
    del custom_fbank
    del name
    del output_names
    del dynamic_axes
    gc.collect()

    # ── Decoder token-embedding graph (keeps int ids out of the decoder; xscale folded into the embedding) ──
    dolphin_embed = DOLPHIN_DECODER_EMBED(model)
    embed_input_ids = torch.ones((1, 3), dtype=torch.int32)
    torch.onnx.export(
        dolphin_embed,
        (embed_input_ids,),
        onnx_model_Embed,
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'ids_len'},
            'hidden_states': {0: 'batch', 1: 'ids_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del dolphin_embed
    del embed_input_ids

    # ── Prefill position-embedding + causal-mask graph ──
    dolphin_prefill = DOLPHIN_PREFILL(model, _MODEL_MAX_DECODER_SEQ_LEN)
    prefill_ids_len = torch.tensor([3], dtype=torch.int64)
    prefill_history_len = torch.tensor([0], dtype=torch.int64)
    torch.onnx.export(
        dolphin_prefill,
        (prefill_ids_len, prefill_history_len),
        onnx_model_Prefill,
        input_names=['ids_len', 'history_len'],
        output_names=['position_embed', 'attention_mask', 'kv_seq_len'],
        dynamic_axes={
            'position_embed': {1: 'ids_len'},
            'attention_mask': {1: 'ids_len', 2: 'kv_seq_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del dolphin_prefill
    del prefill_ids_len
    del prefill_history_len

    # ── Decode position-embedding graph for the single new token ──
    dolphin_decode = DOLPHIN_DECODE(model, _MODEL_MAX_DECODER_SEQ_LEN)
    decode_kv_seq_len = torch.tensor([3], dtype=torch.int64)
    torch.onnx.export(
        dolphin_decode,
        (decode_kv_seq_len,),
        onnx_model_Decode,
        input_names=['kv_seq_len'],
        output_names=['position_embed', 'kv_seq_len_next'],
        dynamic_axes={},
        opset_version=OPSET,
        dynamo=False
    )
    del dolphin_decode
    del decode_kv_seq_len
    gc.collect()

    # ── Decoder main graph (pure float: token + position embeddings and the mask arrive as inputs) ──
    dolphin_decoder = DOLPHIN_DECODER(model, NUM_LAYER_DE)
    save_encoder_key = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, STFT_SIGNAL_LENGTH // 2 + 1), dtype=KV_DTYPE)
    save_encoder_value = torch.zeros((NUM_HEAD_DE, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_DE), dtype=KV_DTYPE)
    batch_size = 3  # Dummy batch value for the export trace.
    past_key_de = torch.zeros((batch_size, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=KV_DTYPE)
    past_value_de = torch.zeros((batch_size, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=KV_DTYPE)
    hidden_states_de = torch.ones((batch_size, 1, HIDDEN_SIZE), dtype=torch.float32)
    position_embed_de = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    attention_mask_dtype = torch.float32 if (USE_FP16_KV and COMPUTE_IN_F32) else KV_DTYPE
    attention_mask = torch.zeros((1, 1, 1), dtype=attention_mask_dtype)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {}

    for i in range(NUM_LAYER_DE):
        name = f'in_de_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_key_de)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_de_key_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_value_de)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len'}
        name = f'out_de_value_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len_plus_ids_len'}

    for i in range(NUM_LAYER_DE):
        name = f'en_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(save_encoder_key)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(save_encoder_value)
        dynamic_axes[name] = {1: 'signal_len'}

    input_names.append('hidden_states')
    all_inputs.append(hidden_states_de)
    dynamic_axes['hidden_states'] = {0: 'batch', 1: 'ids_len'}
    input_names.append('position_embed')
    all_inputs.append(position_embed_de)
    dynamic_axes['position_embed'] = {1: 'ids_len'}
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    dynamic_axes['attention_mask'] = {1: 'ids_len', 2: 'kv_seq_len'}

    output_names.append('logits')
    dynamic_axes['logits'] = {0: 'batch', 1: 'vocab_range'}

    torch.onnx.export(
        dolphin_decoder,
        tuple(all_inputs),
        onnx_model_Decoder,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del model
    del dolphin_decoder
    del save_encoder_key
    del save_encoder_value
    del hidden_states_de
    del position_embed_de
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes

    # Representative values used only to trace dynamic selection-head inputs.
    export_history_length = 10
    logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    save_id = torch.zeros((1, export_history_length), dtype=torch.int32)
    penalty_value = torch.tensor([1.0], dtype=torch.float32)
    penalty_range = torch.tensor([_METADATA_PENALTY_RANGE], dtype=torch.int64)

    # ── Greedy Search (argmax + save_id history; used together with APPLY_PENALTY) ──
    torch.onnx.export(
        GREEDY_SEARCH(),
        (logits[[0]], save_id[[0]]),
        onnx_model_Greedy,
        input_names=['logits', 'save_id_in'],
        output_names=['max_logits_idx', 'save_id_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'max_logits_idx': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len_out'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Argmax (greedy decoding without a repetition penalty) ──
    torch.onnx.export(
        ARGMAX(),
        (logits,),
        onnx_model_Argmax,
        input_names=['logits'],
        output_names=['max_logits_idx'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Apply Penalty (sliding-window repetition penalty on the logits) ──
    torch.onnx.export(
        APPLY_PENALTY(),
        (logits, save_id, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
        output_names=['logits_out'],
        dynamic_axes={
            'logits_in': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'logits_out': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Top-K / Top-P sampling with standard repetition penalty ──
    sampling_temperature = torch.tensor([0.8], dtype=torch.float32)
    sampling_top_k = torch.tensor([50], dtype=torch.int32)
    sampling_top_p = torch.tensor([0.95], dtype=torch.float32)
    sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            sampling_temperature,
            sampling_top_k,
            sampling_top_p,
            sampling_repetition_penalty,
            save_id,
        ),
        onnx_model_Sampling,
        input_names=[
            'logits', 'temperature', 'top_k', 'top_p',
            'repetition_penalty', 'previous_ids'
        ],
        output_names=['sampled_id', 'save_id_out'],
        dynamic_axes={
            'previous_ids': {1: 'history_len'},
            'save_id_out': {1: 'history_len_out'},
        },
        opset_version=OPSET,
        dynamo=False,
    )

    del past_key_de
    del past_value_de
    del logits
    del save_id
    del penalty_value
    del penalty_range
    del sampling_temperature
    del sampling_top_k
    del sampling_top_p
    del sampling_repetition_penalty
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════════════
    # Stamp model metadata into ASR_Metadata.onnx
    # ──────────────────────────────────────────────────────────────────────────────────
    # Special-token IDs come directly from units.txt so metadata cannot drift from the model.
    # Runtime capacities collapse duplicated inference constants into metadata lookups.
    # ══════════════════════════════════════════════════════════════════════════════════
    token_to_id = {token: idx for idx, token in id_to_token.items()}

    required_special_tokens = {
        "blank": "<blank>",
        "sos": "<sos>",
        "stop": "<eos>",
        "asr": "<asr>",
        "notimestamp": "<notimestamp>",
        "prompt_start": "<PROMPT_START>",
        "prompt_end": "<PROMPT_END>",
    }
    special_token_ids = {
        key: int(token_to_id[piece])
        for key, piece in required_special_tokens.items()
    }
    supported_languages = build_cn_supported_languages(
        token_to_id,
        special_token_ids,
    )

    metadata_marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        onnx_model_Metadata,
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )
    del metadata_marker

    onnx_metadata = build_model_metadata(
        {
            "max_seq_len": _MODEL_MAX_DECODER_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "prompt_control_token_count": 4,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
        },
    )

    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    gc.collect()

import Shared_Merged

if ONNX_DIR.exists():
    shutil.rmtree(ONNX_DIR)
ONNX_DIR.mkdir(parents=True)
print("\n[SharedMerged] Building Dolphin strategy graphs + shared initializer bundle ...")
_bundle = Shared_Merged.build_shared_merged_bundle(
    _raw_onnx_dir,
    out_folder=ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
    retain_prefill_logits=True,
    merge_encoder_into_prefill=True,
)
Shared_Merged.copy_runtime_standalones(
    _raw_onnx_dir,
    ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
    include_encoder=False,
)
shutil.copy2(save_vocab, ONNX_DIR / Path(save_vocab).name)
for _name, _path in _bundle["graphs"].items():
    print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
write_metadata_carrier(ONNX_DIR / MODEL_FILE_NAMES["metadata"], onnx_metadata)
print(
    f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
    f"({Path(_bundle['shared_data']).stat().st_size} bytes)"
)
print("    Runtime standalone graphs: Metadata only; Encoder/Main are optimizer donors.")

_raw_onnx_temp.cleanup()
print(f'\n[Cleanup] Removed raw ONNX staging folder: {_raw_onnx_dir}')
print('\nExport done!\n')
print(f'Final ONNX models retained in: {ONNX_DIR}')
subprocess.run(
    [
        sys.executable,
        str(Path(_SCRIPT_DIR) / "Inference_Dolphin_CN_Dialect_ONNX.py"),
        "--onnx-folder",
        str(ONNX_DIR),
    ],
    cwd=_SCRIPT_DIR,
    check=True,
)
