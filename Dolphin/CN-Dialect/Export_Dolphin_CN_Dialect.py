import gc
import subprocess
import sys
import os
import copy
import time
import torch
import dolphin
import torchaudio
import torchaudio.compliance.kaldi as kaldi   # Used at export time to bake Kaldi's exact triangular mel filterbank as a constant.
import numpy as np
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


model_path             = "/home/DakeQQ/Downloads/dolphin-cn-dialect-small-prompt"                                 # The Dolphin-CN-Dialect project download path (small.cn.prompt).
onnx_folder            = os.path.join(_SCRIPT_DIR, "Dolphin_CN_Dialect_ONNX")                                   # Local folder next to this script holding all exported ONNX graphs; created automatically if missing.
os.makedirs(onnx_folder, exist_ok=True)
onnx_model_Metadata    = os.path.join(onnx_folder, "Dolphin_Metadata.onnx")                                     # Tiny metadata carrier graph.
onnx_model_Encoder     = os.path.join(onnx_folder, "Dolphin_Encoder.onnx")                                      # The exported onnx encoder model path.
onnx_model_Decoder     = os.path.join(onnx_folder, "Dolphin_Decoder.onnx")                                      # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = os.path.join(onnx_folder, "Dolphin_Decoder_Embed.onnx")                                # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = os.path.join(onnx_folder, "Dolphin_Position_Mask_Prefill.onnx")                        # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = os.path.join(onnx_folder, "Dolphin_Position_Mask_Decode.onnx")                         # Decode position-embedding graph for the single new token.
onnx_model_Greedy      = os.path.join(onnx_folder, "Dolphin_Greedy_Search.onnx")                                # Greedy argmax + save_id history (used with Apply_Penality).
onnx_model_Argmax      = os.path.join(onnx_folder, "Dolphin_Argmax.onnx")                                       # Bare argmax (greedy decoding without a repetition penalty).
onnx_model_First_Beam  = os.path.join(onnx_folder, "Dolphin_First_Beam_Search.onnx")                            # First beam-search step.
onnx_model_Second_Beam = os.path.join(onnx_folder, "Dolphin_Second_Beam_Search.onnx")                           # Subsequent beam-search steps.
onnx_model_Penality    = os.path.join(onnx_folder, "Dolphin_Apply_Penality.onnx")                               # Sliding-window repetition penalty on the logits.
save_vocab             = os.path.join(onnx_folder, "vocab_Dolphin_CN_Dialect.txt")                              # The exported Dolphin-CN-Dialect vocab path.



USE_BEAM_SEARCH    = False      # Use beam search or greedy search.
INPUT_AUDIO_LENGTH = 480000     # The maximum input audio length. Must less than 480000 (30 seconds).
MAX_SEQ_LEN        = 448        # It should less than 5000 (the decoder positional-encoding table length).
REPEAT_PENALITY    = 1.0        # Range from 0.0 to 1.0; "1.0" means no penality (the Dolphin reference decoder applies none).
PENALITY_RANGE     = 20         # Penalizes the most recent output. "20" means the last 20 tokens.
MAX_BEAM_SIZE      = 10         # Max beams for exported model.
TOP_K              = 3          # The top k candidate in decoding.
BEAM_SIZE          = 3          # Number of beams in searching.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ---- Kaldi fbank front-end parameters (Dolphin-CN-Dialect has NO frontend_conf: dolphin.processor.extract_feats calls torchaudio.compliance.kaldi.fbank) ----
WINDOW_TYPE     = 'povey'       # Kaldi's default window for fbank (Hann ** 0.85).
N_MELS          = 80            # Number of Mel bands (fbank_conf.num_mel_bins), edit it carefully.
NFFT_STFT       = 512           # Kaldi zero-pads the 400-sample (25 ms) frame to the next power of two before the FFT.
WINDOW_LENGTH   = 400           # frame_length 25 ms * 16 kHz, edit it carefully.
HOP_LENGTH      = 160           # frame_shift 10 ms * 16 kHz, edit it carefully.
PRE_EMPHASIZE   = 0.97          # Kaldi's default pre-emphasis coefficient for fbank.
LOW_FREQ        = 20.0          # Kaldi's default low_freq for the mel filterbank.
SAMPLE_RATE     = 16000         # The model parameter, do not edit the value.
INPUT_AUDIO_DTYPE   = "INT16"   # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
# ---- Dolphin-CN-Dialect special tokens (units.txt) ----
SOS_TOKEN       = 2             # <sos>
EOS_TOKEN       = 3             # <eos>
ASR_TOKEN       = 4             # <asr>
PROMPT_START    = 106           # <PROMPT_START>
PROMPT_END      = 107           # <PROMPT_END>
NOTIMESTAMP     = 109           # <notimestamp>
# Supported language configuration (README + units.txt):
# | Setting    | Supported values |
# | ---------- | ---------------- |
# | LANG_SYM   | zh, en |
# | REGION_SYM | CN, TW, WU, SICHUAN, SHANXI, ANHUI, TIANJIN, NINGXIA, SHAANXI, HEBEI |
# |            | SHANDONG, GUANGDONG, SHANGHAI, HUBEI, LIAONING, GANSU, FUJIAN, HUNAN |
# |            | HENAN, YUNNAN, MINNAN, WENZHOU, BEIJING, JILIN, NEIMENGGU, GUANGXI |
# |            | GUIZHOU, HEILONGJIANG, JIANGSU |
LANG_SYM        = "zh"          # Force the language, e.g. "zh"/"en". Leave "" to auto-detect. Requires REGION_SYM too; otherwise auto-detect is used.
REGION_SYM      = "SHANGHAI"    # Force the region, e.g. "CN"/"TW"/"SHANGHAI". Leave "" to auto-detect. Both LANG_SYM and REGION_SYM must be set to skip detection.
HOTWORDS        = ["开饭时间"]   # Prompt-based hotwords (small.cn.prompt). Each word is char-tokenised and packed between PROMPT_START/PROMPT_END; use [] to disable biasing.
STOP_TOKEN      = [EOS_TOKEN]   # 3 is the end token for Dolphin-CN-Dialect.
OPSET           = 18            # ONNX opset version for the export.



if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
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
    # Used when a repetition penalty is active so APPLY_PENALITY can read the on-device history.
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


class APPLY_PENALITY(torch.nn.Module):
    # Sliding-window repetition penalty (Qwen ASR style): multiply the logits of the most recent
    # `penality_range` tokens by `penality_value`. penality_range is baked into the graph at export
    # via int(), so the runtime input only needs to match the export value.
    def __init__(self):
        super(APPLY_PENALITY, self).__init__()

    def forward(self, logits, save_id, penality_value, penality_range):
        penality_range_val = int(penality_range.item())
        target_indices = save_id[:, -penality_range_val:].long()
        penalised = logits.gather(1, target_indices) * penality_value
        return logits.scatter(1, target_indices, penalised)


class FIRST_BEAM_SEARCH(torch.nn.Module):
    # First beam step (Qwen ASR style): expand the single prefilled sequence into `beam_size` beams.
    # Scores are log-probabilities via logits - logsumexp(logits); no repetition penalty here (it is a
    # separate APPLY_PENALITY pass on the logits beforehand).
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp
        for i in range(self.num_keys_values):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]
        return *self.save_keys_values, save_id, top_beam_prob.transpose(0, 1), top_beam_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    # Subsequent beam steps (Qwen ASR style): accumulate log-prob scores, pick the global top-`beam_size`
    # continuations from the top-`topK` per beam, and reorder the K/V caches / save_id via index_select.
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=True)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = flat_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)
        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)
        return *self.save_keys_values, save_id, top_beam_prob.unsqueeze(-1), top_beam_indices, max_logits_idx


class DOLPHIN_ENCODER(torch.nn.Module):
    def __init__(self, dolphin, fbank_model, num_layers_de):
        super(DOLPHIN_ENCODER, self).__init__()
        self.dolphin = copy.deepcopy(dolphin)
        self.fbank_model = fbank_model

        # GlobalCMVN (JSON global_cmvn): forward is (x - mean) * istd, where istd is already the inverse std.
        self.cmvn_mean = self.dolphin.encoder.global_cmvn.mean.float()
        self.cmvn_istd = self.dolphin.encoder.global_cmvn.istd.float()

        # Encoder components
        self.save_en_keys = [None] * num_layers_de
        self.save_en_values = [None] * num_layers_de
        self.embed = self.dolphin.encoder.embed.out[0]               # Conv2dSubsampling4 projection (Linear 14592 -> 768)
        self.position_encode = self.dolphin.encoder.embed.pos_enc    # RelPositionalEncoding (rel_pos, NOT rel_pos_v1)
        # Conv2dSubsampling4 applies x = x * xscale inside pos_enc; fold that scale into the projection here.
        self.embed.weight.data *= self.position_encode.xscale
        self.embed.bias.data *= self.position_encode.xscale
        # rel_pos positional table: pos_emb is the leading length-T slice pe[:, :T] (no 2T-1 shift trick).
        self.register_buffer('position_encode_pe', self.position_encode.pe.half())
        self.num_heads = self.dolphin.encoder.encoders._modules['0'].attn.h
        self.head_dim = self.dolphin.encoder.encoders._modules['0'].attn.d_k
        self.hidden_size = self.embed.out_features
        self.cross_num_heads = self.dolphin.decoder.decoders._modules['0'].src_attn.h
        self.cross_head_dim = self.dolphin.decoder.decoders._modules['0'].src_attn.d_k
        self._fuse_weights()
        # Pre-apply linear_pos + view + permute once per layer over the full pe; forward slices all layers then gathers.
        pe_full = self.position_encode_pe.float()
        self.pos_p = torch.stack([encoder_layer.attn.linear_pos(pe_full).view(-1, self.num_heads, self.head_dim).permute(1, 2, 0)
                                  for encoder_layer in self.dolphin.encoder.encoders], dim=0).half()

    def _fuse_weights(self):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            for encoder_layer in self.dolphin.encoder.encoders:
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
            after_norm = self.dolphin.encoder.after_norm
            after_gamma = after_norm.weight.data.clone()
            after_beta = after_norm.bias.data.clone()
            for decoder_layer in self.dolphin.decoder.decoders:
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
                cross_attn.kv = kv
                del cross_attn.linear_k, cross_attn.linear_v
            after_norm.weight.data.fill_(1.0)
            after_norm.bias.data.zero_()

    def forward(self, audio):
        # Kaldi fbank front-end (int16 audio -> log-mel) then GlobalCMVN, matching dolphin.processor.extract_feats
        # for models without a frontend_conf block.
        mel_features = self.fbank_model(audio)
        mel_features = (mel_features - self.cmvn_mean) * self.cmvn_istd
        embed = self.dolphin.encoder.embed.conv(mel_features.unsqueeze(1))
        embed_len = embed.shape[-2].unsqueeze(0)
        x = self.embed(embed.transpose(1, 2).contiguous().view(1, embed_len, -1))
        pos_p = self.pos_p[:, :, :, :embed_len].float()
        for idx, encoder_layer in enumerate(self.dolphin.encoder.encoders):
            x = x + encoder_layer.feed_forward_macaron(encoder_layer.norm_ff_macaron(x))  # ff_scale(0.5) already folded into macaron w_2
            x1 = encoder_layer.norm_mha(x)
            qkv = encoder_layer.attn.qkv(x1).view(-1, 3 * self.num_heads, self.head_dim).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)
            p = pos_p[idx]
            q_with_bias_u = q + encoder_layer.attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(1, 2))
            matrix_bd = torch.matmul(q_with_bias_v, p)                       # rel_pos + use_sdpa: NO rel_shift (pos_emb length == time)
            x1 = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v)
            x1 = encoder_layer.attn.linear_out(x1.transpose(0, 1).reshape(1, -1, self.hidden_size))
            x2 = encoder_layer.cgmlp.channel_proj1(encoder_layer.norm_mlp(x))
            x_r, x_g = x2.chunk(2, dim=-1)
            x_g = encoder_layer.cgmlp.csgu.conv(encoder_layer.cgmlp.csgu.norm(x_g).transpose(1, 2)).transpose(1, 2)
            x2 = encoder_layer.cgmlp.channel_proj2(x_r * x_g)
            x_concat = torch.cat([x1, x2], dim=-1)
            x_concat = x_concat + encoder_layer.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
            x = x + encoder_layer.merge_proj(x_concat)
            x = x + encoder_layer.feed_forward(encoder_layer.norm_ff(x))  # ff_scale(0.5) already folded into ff w_2
            x = encoder_layer.norm_final(x)
        enc_outputs = self.dolphin.encoder.after_norm(x)
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            cross_kv = decoder_layer.src_attn.kv(enc_outputs).view(-1, 2 * self.cross_num_heads, self.cross_head_dim).transpose(0, 1)
            k, v = cross_kv.split(self.cross_num_heads, dim=0)
            self.save_en_keys[idx] = k.half().transpose(1, 2)       # f16 cross-attention key   (num_heads, head_dim, T)
            self.save_en_values[idx] = v.half()                     # f16 cross-attention value (num_heads, T, head_dim)
        return *self.save_en_keys, *self.save_en_values


class DOLPHIN_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Whisper/Qwen Decoder_Embed) so the int
    # token ids never enter the float-only decode graph. The positional xscale is folded into the embedding
    # weight here (the absolute position embedding itself is added inside the decoder main graph).
    def __init__(self, dolphin):
        super(DOLPHIN_DECODER_EMBED, self).__init__()
        self.dolphin = copy.deepcopy(dolphin)
        self.embed = self.dolphin.decoder.embed[0]
        self.position_encode = self.dolphin.decoder.embed[1]
        self.embed.weight.data *= self.position_encode.xscale

    def forward(self, input_ids):
        return self.embed(input_ids)


class DOLPHIN_PREFILL(torch.nn.Module):
    # Prefill-phase position-embedding + causal-mask generator (mirrors Whisper/Qwen Prefill).
    # Consumes the int lengths and emits float position embedding + float attention mask so the decoder
    # main graph stays integer-free.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_PREFILL, self).__init__()
        position_encode = copy.deepcopy(dolphin.decoder.embed[1])
        self.register_buffer('position_weight', position_encode.pe[:, :max_seq_len].half())
        self.register_buffer('attention_mask', (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128)

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len].half()   # f16 mask matches the minimum-cast f16 attention scores
        return position_embed, attention_mask, kv_seq_len


class DOLPHIN_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Whisper/Qwen Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_DECODE, self).__init__()
        position_encode = copy.deepcopy(dolphin.decoder.embed[1])
        self.register_buffer('position_weight', position_encode.pe[:, :max_seq_len].half())

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class DOLPHIN_DECODER(torch.nn.Module):
    def __init__(self, dolphin, num_layers_de):
        super(DOLPHIN_DECODER, self).__init__()
        self.dolphin = copy.deepcopy(dolphin)
        self.num_layers_de = num_layers_de
        self.idx_en_key = num_layers_de + num_layers_de         # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + num_layers_de     # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + num_layers_de     # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                 # position-embedding input (4 * L + 1)
        self.save_de_keys = [None] * num_layers_de
        self.save_de_values = [None] * num_layers_de
        self.num_heads = self.dolphin.decoder.decoders._modules['0'].self_attn.h
        self.head_dim = self.dolphin.decoder.decoders._modules['0'].self_attn.d_k
        self.hidden_size = self.dolphin.decoder.output_layer.in_features
        self.cross_num_heads = self.dolphin.decoder.decoders._modules['0'].src_attn.h
        self.cross_head_dim = self.dolphin.decoder.decoders._modules['0'].src_attn.d_k
        self._fuse_weights()

    def _fuse_weights(self):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.dolphin.decoder.decoders:
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
            fold_norm_into_linear(self.dolphin.decoder.after_norm, self.dolphin.decoder.output_layer)

    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            hidden_states_norm = decoder_layer.norm1(hidden_states)
            qkv = decoder_layer.self_attn.qkv(hidden_states_norm).view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = qkv.split(self.num_heads, dim=1)
            k = torch.cat((all_inputs[idx], k.half().transpose(-1, -2)), dim=-1)           # f16 key cache   (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v.half()), dim=-2)       # f16 value cache (batch, num_heads, kv_seq_len, head_dim)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q.half(), k) + attention_mask, dim=-1), v).transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float()
            hidden_state_attn = decoder_layer.self_attn.linear_out(hidden_state_attn)
            hidden_state_attn += hidden_states
            q = decoder_layer.src_attn.linear_q(decoder_layer.norm2(hidden_state_attn)).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
            hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(decoder_layer.norm3(hidden_state_cross))
        hidden_states = self.dolphin.decoder.after_norm(hidden_states[:, -1])
        logits = self.dolphin.decoder.output_layer(hidden_states)
        return *self.save_de_keys, *self.save_de_values, logits


# ══════════════════════════════════════════════════════════════════════════════════
# ONNX METADATA  (embed model geometry + special tokens into every graph)
# ──────────────────────────────────────────────────────────────────────────────────
# The inference runtime used to hard-code the special-token IDs, max_seq_len and sample_rate as
# constants that HAD to be kept in sync with this exporter. Stamping the same facts into every graph's
# `metadata_props` lets the runtime read them directly (in ONNX Runtime:
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
        return str(value)

    merged = {}
    for section in sections:
        for key, value in section.items():
            if value is not None:
                merged[key] = _norm(value)
    return merged


def write_onnx_metadata(onnx_path, metadata):
    """Add/overwrite `metadata_props` on an ONNX file in place, preserving external-weight sidecars.

    load_external_data=False keeps any big `*.data` weights on disk untouched (only the graph proto +
    metadata are rewritten), so this is safe and cheap for every exported Dolphin graph.
    """
    import onnx  # lazy: only needed when actually exporting
    model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


print('\nExport start...\n')
with torch.inference_mode():
    model = dolphin.load_model("small.cn.prompt", model_path, "cpu")
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
    HEAD_DIM_EN = model.encoder.encoders._modules['0'].attn.d_k
    HEAD_DIM_DE = model.decoder.decoders._modules['0'].self_attn.d_k
    NUM_LAYER_DE = len(model.decoder.decoders)
    VOCAB_SIZE = model.vocab_size
    CROSS_NUM_HEAD_DE = model.decoder.decoders._modules['0'].src_attn.h    # decoder cross-attention heads (attend encoder KV).
    CROSS_HEAD_DIM_DE = model.decoder.decoders._modules['0'].src_attn.d_k  # decoder cross-attention head dim.
    STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1   # Kaldi snip_edges=True framing.

    custom_fbank = KaldiFbank(NFFT_STFT, WINDOW_LENGTH, HOP_LENGTH, N_MELS, SAMPLE_RATE, WINDOW_TYPE, PRE_EMPHASIZE, LOW_FREQ).eval()
    dolphin_encoder = DOLPHIN_ENCODER(model, custom_fbank, NUM_LAYER_DE)
    output_names = []
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)
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
    dolphin_decoder = DOLPHIN_DECODER(model, NUM_LAYER_DE)
    save_encoder_key = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float16)
    save_encoder_value = torch.zeros((NUM_HEAD_DE, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_DE), dtype=torch.float16)
    batch_size = 3  # Dummy batch value for the export trace.
    past_key_de = torch.zeros((batch_size, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float16)
    past_value_de = torch.zeros((batch_size, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float16)
    hidden_states_de = torch.ones((batch_size, 1, HIDDEN_SIZE), dtype=torch.float32)
    position_embed_de = torch.ones((1, 1, HIDDEN_SIZE), dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 1), dtype=torch.float16)   # f16 mask matches the minimum-cast f16 attention scores

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

    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    logits = torch.ones((BEAM_SIZE, VOCAB_SIZE), dtype=torch.float32)
    save_id = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)            # 10 = dummy history length
    previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
    penality_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
    penality_range = torch.tensor([PENALITY_RANGE], dtype=torch.int64)

    # ── Greedy Search (argmax + save_id history; used together with APPLY_PENALITY) ──
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
        opset_version=17,
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
        opset_version=17,
        dynamo=False
    )

    # ── Apply Penality (sliding-window repetition penalty on the logits) ──
    torch.onnx.export(
        APPLY_PENALITY(),
        (logits, save_id, penality_value, penality_range),
        onnx_model_Penality,
        input_names=['logits_in', 'save_id_in', 'penality_value', 'penality_range'],
        output_names=['logits_out'],
        dynamic_axes={
            'logits_in': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'logits_out': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )

    # ── First Beam Search ──
    first_beam_search = FIRST_BEAM_SEARCH(NUM_LAYER_DE)
    past_keys_greedy = past_key_de[[0]]
    past_values_greedy = past_value_de[[0]]

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_DE):
        name = f'in_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_keys_greedy)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_key_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        name = f'in_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_values_greedy)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len'}
        name = f'out_value_layer_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 2: 'history_len_plus_ids_len'}
    input_names.append('logits')
    all_inputs.append(logits[[0]])
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('save_id_out')
    output_names.append('top_beam_prob')
    output_names.append('top_beam_indices')
    output_names.append('max_logits_idx')
    dynamic_axes['logits'] = {0: 'batch'}
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}

    torch.onnx.export(
        first_beam_search,
        tuple(all_inputs),
        onnx_model_First_Beam,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )

    # ── Second Beam Search (same output layout as First Beam) ──
    all_inputs = []
    input_names = []
    for i in range(NUM_LAYER_DE):
        name = f'in_key_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_key_de)
    for i in range(NUM_LAYER_DE):
        name = f'in_value_layer_{i}'
        input_names.append(name)
        all_inputs.append(past_value_de)
    input_names.append('logits')
    all_inputs.append(logits)
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}

    second_beam_search = SECOND_BEAM_SEARCH(NUM_LAYER_DE)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_Second_Beam,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )

    del first_beam_search
    del second_beam_search
    del past_key_de
    del past_value_de
    del past_keys_greedy
    del past_values_greedy
    del logits
    del previous_prob
    del save_id
    del penality_value
    del penality_range
    del topK
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    gc.collect()

    # ══════════════════════════════════════════════════════════════════════════════════
    # Stamp model metadata into every exported graph
    # ──────────────────────────────────────────────────────────────────────────────────
    # The SAME facts are written into every graph so the runtime can read them from whichever model it
    # queries. Special-token IDs are taken from the tokenizer (units.txt) so they can never drift from
    # the model; the verified module constants are the fallback. Geometry / max_seq_len / sample_rate
    # collapse the inference script's hand-kept constants into simple metadata lookups.
    # ══════════════════════════════════════════════════════════════════════════════════
    token_to_id = {token: idx for idx, token in id_to_token.items()}

    def _special_token_id(piece, fallback):
        tid = token_to_id.get(piece)
        return int(tid) if tid is not None else int(fallback)

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
            "dolphin_metadata_version": 1,
            "producer":                 "Export_Dolphin_CN_Dialect.py",
            "model_variant":            "small.cn.prompt",
        },
        {   # ── model geometry ──
            "num_decoder_layers": NUM_LAYER_DE,
            "num_encoder_heads":  NUM_HEAD_EN,
            "num_decoder_heads":  NUM_HEAD_DE,
            "encoder_head_dim":   HEAD_DIM_EN,
            "decoder_head_dim":   HEAD_DIM_DE,
            "cross_num_heads":    CROSS_NUM_HEAD_DE,
            "cross_head_dim":     CROSS_HEAD_DIM_DE,
            "hidden_size":        HIDDEN_SIZE,
            "vocab_size":         VOCAB_SIZE,
            "max_seq_len":        MAX_SEQ_LEN,
            "sample_rate":        SAMPLE_RATE,
            "input_audio_length": INPUT_AUDIO_LENGTH,
            "num_mels":           N_MELS,
            "nfft_stft":          NFFT_STFT,
            "window_length":      WINDOW_LENGTH,
            "hop_length":         HOP_LENGTH,
            "window_type":        WINDOW_TYPE,
            "pre_emphasis":       PRE_EMPHASIZE,
        },
        {   # ── special tokens (units.txt-derived, module constants as fallback) ──
            "sos_token_id":    _special_token_id("<sos>", SOS_TOKEN),
            "eos_token_id":    _special_token_id("<eos>", EOS_TOKEN),
            "asr_token_id":    _special_token_id("<asr>", ASR_TOKEN),
            "prompt_start_id": _special_token_id("<PROMPT_START>", PROMPT_START),
            "prompt_end_id":   _special_token_id("<PROMPT_END>", PROMPT_END),
            "notimestamp_id":  _special_token_id("<notimestamp>", NOTIMESTAMP),
        },
    )

    _metadata_targets = [
        onnx_model_Metadata, onnx_model_Encoder, onnx_model_Decoder, onnx_model_Embed,
        onnx_model_Prefill, onnx_model_Decode, onnx_model_Greedy,
        onnx_model_Argmax, onnx_model_First_Beam, onnx_model_Second_Beam,
        onnx_model_Penality,
    ]
    _written, _skipped = [], []
    for _target in _metadata_targets:
        if not os.path.exists(_target):
            continue
        try:
            write_onnx_metadata(_target, onnx_metadata)
            _written.append(os.path.basename(_target))
        except Exception as _exc:  # noqa: BLE001 - one bad graph must not abort the whole export
            _skipped.append(f"{os.path.basename(_target)} ({_exc})")

    print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(_written)} ONNX graph(s):")
    for _key in sorted(onnx_metadata):
        print(f"    {_key} = {onnx_metadata[_key]}")
    if _skipped:
        print("[Metadata] Skipped (kept usable, metadata not written):")
        for _entry in _skipped:
            print(f"    {_entry}")
    gc.collect()

print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_Dolphin_CN_Dialect_ONNX.py ...')
subprocess.run(
    [sys.executable, os.path.join(_SCRIPT_DIR, "Inference_Dolphin_CN_Dialect_ONNX.py"), "--onnx-folder", onnx_folder],
    check=True,
)
