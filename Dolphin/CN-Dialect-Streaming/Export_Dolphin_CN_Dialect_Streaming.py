import gc
import json
import os
import copy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
import dolphin
import torchaudio.compliance.kaldi as kaldi   # Used at export time to bake Kaldi's exact triangular mel filterbank as a constant.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =================================================================================================
# User configuration: these are the only values intended to be edited for a deployment export.
# =================================================================================================
model_path          = str(Path.home() / "Downloads" / "dolphin-cn-dialect-small-streaming")  # Downloaded small.cn.streaming checkpoint folder.
INPUT_AUDIO_LENGTH  = 480000   # Maximum runtime audio: 30 seconds at the checkpoint's 16-kHz sample rate.
STREAM_CHUNK_FRAMES = 16       # 16 encoder frames = 640 ms with this checkpoint's 4x/10-ms frontend.
MAX_SEQ_LEN         = 448      # Runtime decoder limit; validated against the checkpoint position table.
INPUT_AUDIO_DTYPE   = "F32"    # "INT16", "F32", or "F16"; float inputs still carry int16-range PCM values.
USE_FP16_KV         = True     # Keep recurrent/cache storage in FP16 for normal deployment exports.
COMPUTE_IN_F32      = False    # FP16-cache-only option: upcast attention compute while retaining FP16 storage.
KV_DTYPE            = torch.float16 if USE_FP16_KV else torch.float32


# Export implementation policy (not user/model configuration).
MODEL_VARIANT                  = "small.cn.streaming"
REORDER_DOWNPROJ_FOR_QUANT     = True
REORDER_OPROJ_FOR_QUANT        = True
REORDER_KEY                    = "absmean"  # "absmean" | "L4" | "rms" | "std".
OPSET                          = 20


# Fixed artifact layout. The split folder is temporary staging; only the merged runtime bundle is retained.
onnx_folder         = os.path.join(_SCRIPT_DIR, "Dolphin_CN_Dialect_Streaming_ONNX")
_split_export_temp  = tempfile.TemporaryDirectory(prefix="dolphin-streaming-export-")
split_export_folder = _split_export_temp.name

MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Dolphin_Encoder.onnx",
    "main": "Dolphin_Decoder.onnx",
    "embed": "Dolphin_Decoder_Embed.onnx",
    "position_prefill": "Dolphin_Position_Mask_Prefill.onnx",
    "position_decode": "Dolphin_Position_Mask_Decode.onnx",
    "argmax": "Dolphin_Argmax.onnx",
    "prefill_greedy": "Dolphin_PrefillGreedy.onnx",
    "decode_greedy": "Dolphin_DecodeGreedy.onnx",
    "shared_initializers": "Dolphin_SharedInitializers.onnx",
    "vocab": "vocab_Dolphin_CN_Dialect.txt",
}
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"

onnx_model_Metadata    = os.path.join(split_export_folder, MODEL_FILE_NAMES["metadata"])                                  # Tiny metadata carrier graph.
onnx_model_Encoder     = os.path.join(split_export_folder, MODEL_FILE_NAMES["encoder"])                                   # Streaming encoder; remains standalone.
onnx_model_Decoder     = os.path.join(split_export_folder, MODEL_FILE_NAMES["main"])                                      # Temporary decoder Main donor.
onnx_model_Embed       = os.path.join(split_export_folder, MODEL_FILE_NAMES["embed"])                                     # Temporary token-embedding shell.
onnx_model_Prefill     = os.path.join(split_export_folder, MODEL_FILE_NAMES["position_prefill"])                          # Temporary prefill position/mask shell.
onnx_model_Decode      = os.path.join(split_export_folder, MODEL_FILE_NAMES["position_decode"])                           # Temporary decode position shell.
onnx_model_Argmax      = os.path.join(split_export_folder, MODEL_FILE_NAMES["argmax"])                                    # Temporary greedy head.
save_vocab             = os.path.join(split_export_folder, MODEL_FILE_NAMES["vocab"])                                     # Copied into the final deployment folder.


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
    # Absorb a LayerNorm affine into the next Linear. Export wrappers use one shared affine-free
    # LayerNorm afterward, so the original normalization modules are discarded.
    linear.bias.data.add_(linear.weight.data @ norm.bias.data)
    linear.weight.data.mul_(norm.weight.data.unsqueeze(0))


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


class ARGMAX(torch.nn.Module):
    # Bare argmax: greedy next-token pick for the short lang/region attention pass (the transcript itself comes from CTC).
    def __init__(self):
        super(ARGMAX, self).__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super(METADATA_CARRIER, self).__init__()

    def forward(self, marker):
        return marker


class DOLPHIN_ENCODER(torch.nn.Module):
    def __init__(self, dolphin, fbank_model, num_layers_de, max_encoder_seq_len):
        super(DOLPHIN_ENCODER, self).__init__()
        self.fbank_model = fbank_model
        source_encoder = dolphin.encoder
        source_decoder = dolphin.decoder

        # Copy only modules reached by forward(). The old wrapper duplicated the complete 421M-parameter
        # ASR model even though encoder export never executes the decoder embedding/output stack.
        self.subsampling_conv = copy.deepcopy(source_encoder.embed.conv)
        self.embed = copy.deepcopy(source_encoder.embed.out[0])
        self.encoder_layers = copy.deepcopy(source_encoder.encoders)
        self.ctc_lo = copy.deepcopy(dolphin.ctc.ctc_lo)

        # GlobalCMVN (JSON global_cmvn): forward is (x - mean) * istd, where istd is already the inverse std.
        self.register_buffer('cmvn_mean', source_encoder.global_cmvn.mean.detach().float().clone())
        self.register_buffer('cmvn_istd', source_encoder.global_cmvn.istd.detach().float().clone())

        # Encoder components
        self.num_layers_en = len(self.encoder_layers)
        self.num_layers_de = int(num_layers_de)
        self.compute_in_f32 = not USE_FP16_KV or COMPUTE_IN_F32
        first_encoder_layer = self.encoder_layers._modules['0']
        self.csgu_channels = first_encoder_layer.cgmlp.csgu.conv.in_channels
        position_encode = source_encoder.embed.pos_enc               # RelPositionalEncoding (rel_pos, NOT rel_pos_v1)
        # Conv2dSubsampling4 applies x = x * xscale inside pos_enc; fold that scale into the projection here.
        self.embed.weight.data *= position_encode.xscale
        self.embed.bias.data *= position_encode.xscale
        self.num_heads = first_encoder_layer.attn.h
        self.head_dim = first_encoder_layer.attn.d_k
        self.hidden_size = self.embed.out_features
        self.cross_num_heads = source_decoder.decoders._modules['0'].src_attn.h
        self.cross_head_dim = source_decoder.decoders._modules['0'].src_attn.d_k
        # Streaming model is CAUSAL: merge fusion conv (depthwise) and CSGU conv each left-pad lorder = kernel-1 zeros.
        self.merge_lorder = first_encoder_layer.lorder
        self.csgu_lorder = first_encoder_layer.cgmlp.csgu.lorder
        self.norm_shape = (self.hidden_size,)
        self.norm_eps = float(first_encoder_layer.norm_mha.eps)
        norm_dtype = self.embed.weight.dtype
        self.register_buffer('norm_weight', torch.ones(self.norm_shape, dtype=norm_dtype))
        self.register_buffer('norm_bias', torch.zeros(self.norm_shape, dtype=norm_dtype))
        self.register_buffer(
            'merge_zero_pad',
            torch.zeros(
                (1, first_encoder_layer.depthwise_conv_fusion.in_channels, self.merge_lorder),
                dtype=norm_dtype,
            ),
        )
        self._fuse_weights(source_encoder.after_norm, source_decoder.decoders)
        # Pre-apply linear_pos + view + permute once per layer over the full pe; forward slices all layers then gathers.
        pe_full = position_encode.pe[:, :max_encoder_seq_len].to(KV_DTYPE).float()
        self.register_buffer(
            'pos_p',
            torch.stack([
                encoder_layer.attn.linear_pos(pe_full).view(
                    -1, self.num_heads, self.head_dim
                ).permute(1, 2, 0)
                for encoder_layer in self.encoder_layers
            ], dim=0).to(KV_DTYPE),
        )

    def _norm(self, value):
        return F.layer_norm(
            value,
            self.norm_shape,
            self.norm_weight,
            self.norm_bias,
            self.norm_eps,
        )

    def _fuse_weights(self, after_norm, decoder_layers):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            for encoder_layer in self.encoder_layers:
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

                del encoder_layer.norm_mha
                del encoder_layer.norm_mlp
                del encoder_layer.norm_ff_macaron
                del encoder_layer.norm_ff

            cross_scale = float(self.cross_head_dim ** -0.25)
            fold_norm_into_linear(after_norm, self.ctc_lo)
            after_gamma = after_norm.weight.data.clone()
            after_beta = after_norm.bias.data.clone()
            self.cross_kv = torch.nn.ModuleList()
            for decoder_layer in decoder_layers:
                cross_attn = decoder_layer.src_attn
                out_features = cross_attn.linear_k.out_features
                kv = torch.nn.Linear(cross_attn.linear_k.in_features, out_features * 2, bias=True)
                kv.weight.copy_(torch.cat([cross_attn.linear_k.weight, cross_attn.linear_v.weight], dim=0))
                kv.bias.copy_(torch.cat([_bias_or_zero(cross_attn.linear_k), _bias_or_zero(cross_attn.linear_v)], dim=0))
                kv.weight.data[:out_features].mul_(cross_scale)
                kv.bias.data[:out_features].mul_(cross_scale)
                # Fold encoder.after_norm into every cross-attn kv (enc_outputs after_norm becomes identity).
                kv.bias.data.add_(kv.weight.data @ after_beta)
                kv.weight.data.mul_(after_gamma)
                self.cross_kv.append(kv)

    def forward(self, audio, audio_len, *caches):
        # ONE streaming chunk (mirrors dolphin EBranchformerEncoder.forward_chunk, left=-1 all-history cache):
        # audio is the overlapped window for this chunk; the per-layer att K/V + csgu conv caches carry history.
        audio = audio[:, :, :audio_len[0]]
        att_k_cache = caches[:self.num_layers_en]                               # f16 (h, hist, d_k)
        att_v_cache = caches[self.num_layers_en:2 * self.num_layers_en]         # f16 (h, hist, d_k)
        cnn_cache = caches[2 * self.num_layers_en:3 * self.num_layers_en]       # f16 (1, csgu_ch, lorder)
        mel_features = self.fbank_model(audio)
        mel_features = (mel_features - self.cmvn_mean) * self.cmvn_istd
        embed = self.subsampling_conv(mel_features.unsqueeze(1))
        embed_len = embed.shape[-2].unsqueeze(0)
        x = self.embed(embed.permute(0, 2, 1, 3).flatten(2))
        hist_len = att_k_cache[0].shape[1].unsqueeze(0)
        kv_len = hist_len + embed_len
        pos_p = self.pos_p[:, :, :, :kv_len]                                    # pe[:, :key_size] (offset==hist, all left)
        if self.compute_in_f32:
            pos_p = pos_p.float()
        new_att_k, new_att_v, new_cnn = [], [], []
        for idx, encoder_layer in enumerate(self.encoder_layers):
            x = x + encoder_layer.feed_forward_macaron(self._norm(x))  # ff_scale(0.5) already folded into macaron w_2
            x_norm = self._norm(x)                                     # shared by attention and cgMLP
            qkv = encoder_layer.attn.qkv(x_norm).view(
                -1, 3 * self.num_heads, self.head_dim
            ).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)
            k = torch.cat([att_k_cache[idx], k.to(KV_DTYPE)], dim=1)
            v = torch.cat([att_v_cache[idx], v.to(KV_DTYPE)], dim=1)
            new_att_k.append(k)
            new_att_v.append(v)
            p = pos_p[idx]
            # Streaming encoder self-attention over the f16 K/V history cache. COMPUTE_IN_F32 (ON): keep the f16
            # cache *storage* (k/v are cast to f16 before the concat above, so the cache I/O dtype is unchanged)
            # but upcast K/V to f32 at the matmul use points (Q/p/pos_bias stay f32). OFF (minimum-cast): downcast
            # the small non-stored operands (q+pos_bias, p) to f16 and run the attention in f16 on the f16 cache;
            # the context is cast back to f32 for linear_out.
            if self.compute_in_f32:
                matrix_ac = torch.matmul(q + encoder_layer.attn.pos_bias_u, k.transpose(1, 2).float())
                matrix_bd = torch.matmul(q + encoder_layer.attn.pos_bias_v, p)       # rel_pos + use_sdpa: NO rel_shift
                x1 = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v.float())
            else:
                matrix_ac = torch.matmul((q + encoder_layer.attn.pos_bias_u).half(), k.transpose(1, 2))
                matrix_bd = torch.matmul((q + encoder_layer.attn.pos_bias_v).half(), p)  # rel_pos + use_sdpa: NO rel_shift
                x1 = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v).float()
            x1 = encoder_layer.attn.linear_out(x1.transpose(0, 1).reshape(1, -1, self.hidden_size))
            x2 = encoder_layer.cgmlp.channel_proj1(x_norm)
            x_r, x_g = x2.split(self.csgu_channels, dim=-1)
            x_g = encoder_layer.cgmlp.csgu.norm(x_g).transpose(1, 2)
            x_g = torch.cat([cnn_cache[idx].float(), x_g], dim=2)                # causal CSGU conv: prepend cached lorder frames
            new_cnn.append(x_g[:, :, -self.csgu_lorder:].to(KV_DTYPE))
            x_g = encoder_layer.cgmlp.csgu.conv(x_g).transpose(1, 2)
            x2 = encoder_layer.cgmlp.channel_proj2(x_r * x_g)
            x_concat = torch.cat([x1, x2], dim=-1)
            x_fusion = torch.cat((self.merge_zero_pad, x_concat.transpose(1, 2)), dim=2)
            x_concat = x_concat + encoder_layer.depthwise_conv_fusion(x_fusion).transpose(1, 2)
            x = x + encoder_layer.merge_proj(x_concat)
            x = x + encoder_layer.feed_forward(self._norm(x))  # ff_scale(0.5) already folded into ff w_2
            x = encoder_layer.norm_final(x)
        enc_outputs = self._norm(x)
        ctc_ids = self.ctc_lo(enc_outputs).argmax(dim=-1).int()                 # CTC top-1 ids for this chunk (stable streaming text)
        save_k, save_v = [], []
        for projection in self.cross_kv:
            cross_kv = projection(enc_outputs).to(KV_DTYPE).view(
                -1, 2 * self.cross_num_heads, self.cross_head_dim
            ).transpose(0, 1)
            k, v = cross_kv.split(self.cross_num_heads, dim=0)
            save_k.append(k.transpose(1, 2))                                    # each key:   (h, d, chunk)
            save_v.append(v)                                                     # each value: (h, chunk, d)
        return *save_k, *save_v, *new_att_k, *new_att_v, *new_cnn, ctc_ids


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
        self.register_buffer(
            'position_weight',
            dolphin.decoder.embed[1].pe[:, :max_seq_len].detach().clone().to(KV_DTYPE),
        )
        self.register_buffer('attention_mask', (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128)

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len].to(KV_DTYPE)
        return position_embed, attention_mask, kv_seq_len


class DOLPHIN_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Whisper/Qwen Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_DECODE, self).__init__()
        self.register_buffer(
            'position_weight',
            dolphin.decoder.embed[1].pe[:, :max_seq_len].detach().clone().to(KV_DTYPE),
        )

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class DOLPHIN_DECODER(torch.nn.Module):
    def __init__(self, dolphin, num_layers_de, apply_reorders=True):
        super(DOLPHIN_DECODER, self).__init__()
        source_decoder = dolphin.decoder
        self.decoder_layers = copy.deepcopy(source_decoder.decoders)
        self.output_layer = copy.deepcopy(source_decoder.output_layer)
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = not USE_FP16_KV or COMPUTE_IN_F32
        self.idx_en_key = num_layers_de + num_layers_de         # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + num_layers_de     # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + num_layers_de     # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                 # position-embedding input (4 * L + 1)
        first_decoder_layer = self.decoder_layers._modules['0']
        self.num_heads = first_decoder_layer.self_attn.h
        self.head_dim = first_decoder_layer.self_attn.d_k
        self.hidden_size = self.output_layer.in_features
        self.cross_num_heads = first_decoder_layer.src_attn.h
        self.cross_head_dim = first_decoder_layer.src_attn.d_k
        self.norm_shape = (self.hidden_size,)
        self.norm_eps = float(first_decoder_layer.norm1.eps)
        norm_dtype = self.output_layer.weight.dtype
        self.register_buffer('norm_weight', torch.ones(self.norm_shape, dtype=norm_dtype))
        self.register_buffer('norm_bias', torch.zeros(self.norm_shape, dtype=norm_dtype))
        self._fuse_weights(source_decoder.after_norm)
        if apply_reorders:
            self.apply_quant_reorders()

    def apply_quant_reorders(self):
        """Apply each configured compensation once, after fusion and before export."""
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _norm(self, value):
        return F.layer_norm(
            value,
            self.norm_shape,
            self.norm_weight,
            self.norm_bias,
            self.norm_eps,
        )

    def _fuse_weights(self, after_norm):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.decoder_layers:
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
                del decoder_layer.norm1
                del decoder_layer.norm2
                del decoder_layer.norm3
                del cross_attn.linear_k
                del cross_attn.linear_v
            # Absorb the decoder's final after_norm into the output projection.
            fold_norm_into_linear(after_norm, self.output_layer)

    @staticmethod
    def _channel_stat(weight, key, dims):
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(dim=dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            if isinstance(dims, tuple):
                keep_dim = weight.shape[-1]
                return weight.reshape(-1, keep_dim).std(0)
            return weight.std(dim=dims)
        if key != "absmean":
            raise ValueError(f"Unsupported REORDER_KEY: {key!r}")
        return absolute.mean(dim=dims)

    def _reorder_downproj_for_quant(self, key):
        """Permute each decoder w_1 output and matching w_2 input exactly."""
        with torch.no_grad():
            for decoder_layer in self.decoder_layers:
                w_1 = decoder_layer.feed_forward.w_1
                w_2 = decoder_layer.feed_forward.w_2
                permutation = torch.argsort(self._channel_stat(w_2.weight, key, 0))
                w_1.weight.copy_(w_1.weight[permutation])
                if w_1.bias is not None:
                    w_1.bias.copy_(w_1.bias[permutation])
                w_2.weight.copy_(w_2.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute self-attention V rows and matching linear_out input columns per head."""
        num_heads = self.num_heads
        head_dim = self.head_dim
        hidden_size = self.hidden_size
        with torch.no_grad():
            for decoder_layer in self.decoder_layers:
                attention = decoder_layer.self_attn
                output_weight = attention.linear_out.weight
                output_by_head = output_weight.view(output_weight.shape[0], num_heads, head_dim)
                qkv = attention.qkv
                value_weight = qkv.weight[2 * hidden_size:].view(num_heads, head_dim, qkv.in_features)
                permutations = [
                    torch.argsort(self._channel_stat(output_by_head[:, head], key, 0))
                    for head in range(num_heads)
                ]
                reordered_output = output_by_head.clone()
                for head, permutation in enumerate(permutations):
                    reordered_output[:, head] = output_by_head[:, head, permutation]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                reordered_value_weight = value_weight.clone()
                for head, permutation in enumerate(permutations):
                    reordered_value_weight[head] = value_weight[head, permutation]
                qkv.weight[2 * hidden_size:].copy_(reordered_value_weight.reshape(hidden_size, qkv.in_features))

                if qkv.bias is not None:
                    value_bias = qkv.bias[2 * hidden_size:].view(num_heads, head_dim)
                    reordered_value_bias = value_bias.clone()
                    for head, permutation in enumerate(permutations):
                        reordered_value_bias[head] = value_bias[head, permutation]
                    qkv.bias[2 * hidden_size:].copy_(reordered_value_bias.reshape(hidden_size))


    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        # f16-storage / f32-compute (COMPUTE_IN_F32): the causal mask is kept f16 at the graph boundary (I/O
        # dtype unchanged) and upcast to f32 ONCE here, shared by every layer. Minimum-cast path uses it as-is (f16).
        attn_mask = attention_mask.float() if self.compute_in_f32 else attention_mask
        save_de_keys, save_de_values = [], []
        for idx, decoder_layer in enumerate(self.decoder_layers):
            hidden_states_norm = self._norm(hidden_states)
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
            q = decoder_layer.src_attn.linear_q(self._norm(hidden_state_attn)).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            if self.compute_in_f32:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size))
            else:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(self._norm(hidden_state_cross))
        hidden_states = self._norm(hidden_states[:, -1])
        logits = self.output_layer(hidden_states)
        return *save_de_keys, *save_de_values, logits


def build_model_metadata(*sections):
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
    """Build the prompt catalog from real streaming vocabulary entries."""
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
    import onnx
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


os.makedirs(split_export_folder, exist_ok=True)
print(f'\nExport start... Raw graphs: {split_export_folder}\n')
with torch.inference_mode():
    model = dolphin.load_model(MODEL_VARIANT, model_path, "cpu")
    model.eval()

    # Checkpoint/model constants used to construct the graph and stamp runtime metadata.
    # These are derived here rather than exposed as user settings.
    model_configs = model.model_configs
    dataset_conf = model_configs["dataset_conf"]
    fbank_conf = dataset_conf["fbank_conf"]
    SAMPLE_RATE = int(dataset_conf["resample_conf"]["resample_rate"])
    N_MELS = int(fbank_conf["num_mel_bins"])
    WINDOW_LENGTH = round(float(fbank_conf["frame_length"]) * SAMPLE_RATE / 1000.0)
    HOP_LENGTH = round(float(fbank_conf["frame_shift"]) * SAMPLE_RATE / 1000.0)
    WINDOW_TYPE = str(fbank_conf.get("window_type", "povey"))
    PRE_EMPHASIZE = float(fbank_conf.get("preemphasis_coefficient", 0.97))
    LOW_FREQ = float(fbank_conf.get("low_freq", 20.0))
    NFFT_STFT = (
        1 << (WINDOW_LENGTH - 1).bit_length()
        if fbank_conf.get("round_to_power_of_two", True)
        else WINDOW_LENGTH
    )
    SUBSAMPLING_FACTOR = int(model.encoder.embed.subsampling_rate)
    SUBSAMPLING_CONTEXT = int(model.encoder.embed.right_context) + 1
    decoder_position_capacity = int(model.decoder.embed[1].pe.shape[1])

    # Export-only dummy dimensions. They trace representative ranks/layouts; all declared
    # runtime sequence axes remain dynamic in the exported ONNX interfaces.
    stream_window_mel_frames = (
        (STREAM_CHUNK_FRAMES - 1) * SUBSAMPLING_FACTOR
        + SUBSAMPLING_CONTEXT
    )
    stream_window_samples = (
        (stream_window_mel_frames - 1) * HOP_LENGTH + WINDOW_LENGTH
    )
    stream_stride_samples = (
        STREAM_CHUNK_FRAMES * SUBSAMPLING_FACTOR * HOP_LENGTH
    )
    max_stream_chunks = 1 + max(
        0,
        INPUT_AUDIO_LENGTH
        - stream_window_samples
        + stream_stride_samples
        - 1,
    ) // stream_stride_samples
    # Every full window emits STREAM_CHUNK_FRAMES; a final short window emits no more.
    max_encoder_seq_len = max_stream_chunks * STREAM_CHUNK_FRAMES
    dummy_cross_seq_len = (
        (INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1
    ) // 2 + 1

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
    HEAD_DIM_EN = model.encoder.encoders._modules['0'].attn.d_k
    HEAD_DIM_DE = model.decoder.decoders._modules['0'].self_attn.d_k
    NUM_LAYER_DE = len(model.decoder.decoders)
    VOCAB_SIZE = model.vocab_size

    custom_fbank = KaldiFbank(NFFT_STFT, WINDOW_LENGTH, HOP_LENGTH, N_MELS, SAMPLE_RATE, WINDOW_TYPE, PRE_EMPHASIZE, LOW_FREQ).eval()
    dolphin_encoder = DOLPHIN_ENCODER(model, custom_fbank, NUM_LAYER_DE, max_encoder_seq_len)
    NUM_LAYER_EN = dolphin_encoder.num_layers_en
    CSGU_LORDER = dolphin_encoder.csgu_lorder
    CSGU_CHANNELS = dolphin_encoder.csgu_channels
    # Streaming chunk window: emit STREAM_CHUNK_FRAMES encoder frames; overlap by right_context(6) mel frames (no subsample cache).
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, stream_window_samples), dtype=_audio_export_dtype)
    audio_len = torch.tensor([stream_window_samples], dtype=torch.int64)
    en_att_k = [torch.zeros((NUM_HEAD_EN, 0, HEAD_DIM_EN), dtype=KV_DTYPE) for _ in range(NUM_LAYER_EN)]
    en_att_v = [torch.zeros((NUM_HEAD_EN, 0, HEAD_DIM_EN), dtype=KV_DTYPE) for _ in range(NUM_LAYER_EN)]
    en_cnn = [torch.zeros((1, CSGU_CHANNELS, CSGU_LORDER), dtype=KV_DTYPE) for _ in range(NUM_LAYER_EN)]
    input_names = ['audio', 'audio_len']
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_DE):
        output_names.append(f'en_key_layer_{i}'); dynamic_axes[f'en_key_layer_{i}'] = {2: 'chunk_len'}
    for i in range(NUM_LAYER_DE):
        output_names.append(f'en_value_layer_{i}'); dynamic_axes[f'en_value_layer_{i}'] = {1: 'chunk_len'}
    for i in range(NUM_LAYER_EN):
        input_names.append(f'in_att_k_{i}'); dynamic_axes[f'in_att_k_{i}'] = {1: 'history_len'}
    for i in range(NUM_LAYER_EN):
        input_names.append(f'in_att_v_{i}'); dynamic_axes[f'in_att_v_{i}'] = {1: 'history_len'}
    for i in range(NUM_LAYER_EN):
        input_names.append(f'in_cnn_{i}')
    for i in range(NUM_LAYER_EN):
        output_names.append(f'out_att_k_{i}'); dynamic_axes[f'out_att_k_{i}'] = {1: 'history_plus_chunk'}
    for i in range(NUM_LAYER_EN):
        output_names.append(f'out_att_v_{i}'); dynamic_axes[f'out_att_v_{i}'] = {1: 'history_plus_chunk'}
    for i in range(NUM_LAYER_EN):
        output_names.append(f'out_cnn_{i}')
    output_names.append('ctc_ids'); dynamic_axes['ctc_ids'] = {1: 'chunk_len'}

    torch.onnx.export(
        dolphin_encoder,
        (audio, audio_len, *en_att_k, *en_att_v, *en_cnn),
        onnx_model_Encoder,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del dolphin_encoder
    del audio
    del audio_len
    del custom_fbank
    del en_att_k
    del en_att_v
    del en_cnn
    del input_names
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
    dolphin_prefill = DOLPHIN_PREFILL(model, MAX_SEQ_LEN)
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
    dolphin_decode = DOLPHIN_DECODE(model, MAX_SEQ_LEN)
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
    dolphin_decoder = DOLPHIN_DECODER(model, NUM_LAYER_DE, apply_reorders=False)
    reorder_reference_model = os.path.join(split_export_folder, "Dolphin_Decoder_ReorderReference.onnx")
    save_encoder_key = torch.zeros(
        (NUM_HEAD_DE, HEAD_DIM_DE, dummy_cross_seq_len),
        dtype=KV_DTYPE,
    )
    save_encoder_value = torch.zeros(
        (NUM_HEAD_DE, dummy_cross_seq_len, HEAD_DIM_DE),
        dtype=KV_DTYPE,
    )
    batch_size = 3  # Dummy batch value for the export trace.
    past_key_de = torch.zeros((batch_size, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=KV_DTYPE)
    past_value_de = torch.zeros((batch_size, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=KV_DTYPE)
    hidden_states_de = torch.ones((batch_size, 1, HIDDEN_SIZE), dtype=torch.float32)
    position_embed_de = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1), dtype=KV_DTYPE)

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

    # Temporary golden: same fused Main immediately before channel reordering.
    # It is used only for the checkpoint-backed reorder identity gate and is
    # removed with the split staging directory after validation.
    torch.onnx.export(
        dolphin_decoder,
        tuple(all_inputs),
        reorder_reference_model,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    dolphin_decoder.apply_quant_reorders()
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

    # ── Argmax (greedy decoding for the short lang/region attention pass; the transcript comes from CTC) ──
    logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
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
    del past_key_de
    del past_value_de
    del logits
    gc.collect()

    token_to_id = {token: idx for idx, token in id_to_token.items()}

    special_token_ids = {
        "blank": int(token_to_id["<blank>"]),
        "sos": int(token_to_id["<sos>"]),
        "stop": int(token_to_id["<eos>"]),
        "asr": int(token_to_id["<asr>"]),
        "notimestamp": int(token_to_id["<notimestamp>"]),
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
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "stream_stride_samples": stream_stride_samples,
            "prompt_control_token_count": 4,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
        },
    )

    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    gc.collect()

import Shared_Merged

if os.path.exists(onnx_folder):
    shutil.rmtree(onnx_folder)
os.makedirs(onnx_folder, exist_ok=True)
print("\n[SharedMerged] Building Dolphin greedy prefill/decode graphs ...")
_bundle = Shared_Merged.build_shared_merged_bundle(
    Path(split_export_folder),
    out_folder=Path(onnx_folder),
    model_file_names=MODEL_FILE_NAMES,
)
_standalones = Shared_Merged.copy_runtime_standalones(
    Path(split_export_folder), Path(onnx_folder), MODEL_FILE_NAMES
)
shutil.copy2(save_vocab, os.path.join(onnx_folder, MODEL_FILE_NAMES["vocab"]))
write_metadata_carrier(
    os.path.join(onnx_folder, MODEL_FILE_NAMES["metadata"]),
    onnx_metadata,
)
for _name, _path in _bundle["graphs"].items():
    print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
print(
    f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
    f"({Path(_bundle['shared_data']).stat().st_size} bytes)"
)
_shared_storage = _bundle["shared_storage"]
print(
    "    Shared ranges: "
    f"{_shared_storage['unique_data_ranges']} unique / "
    f"{_shared_storage['initializer_count']} names; "
    f"deduplicated {_shared_storage['content_deduplicated_tensors']} tensor(s), "
    f"{_shared_storage['content_deduplicated_bytes']} bytes"
)
print(f"    Copied {len(_standalones)} standalone graph(s): Encoder, Metadata")

_split_export_temp.cleanup()
print(f"\n[Cleanup] Removed raw ONNX staging folder: {split_export_folder}")

print('\nExport done!\n')
print(f'Final ONNX models retained in: {onnx_folder}')
subprocess.run(
    [
        sys.executable,
        str(Path(_SCRIPT_DIR) / "Inference_Dolphin_CN_Dialect_Streaming_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ],
    cwd=_SCRIPT_DIR,
    check=True,
)
