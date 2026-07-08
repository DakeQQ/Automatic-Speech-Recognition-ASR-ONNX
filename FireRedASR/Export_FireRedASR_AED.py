import gc
import subprocess
import shutil
import os
import sys
import importlib
import torch
import torchaudio
from pathlib import Path


# ── Set your base download directory ONCE; every path below is derived from it (edit once, applies everywhere). ──
DOWNLOAD_ROOT = "/home/DakeQQ/Downloads"                                                                    # Root folder that holds the model dir + Github project + ONNX output folders.
SCRIPT_DIR = Path(__file__).resolve().parent                                                             # This script's directory; exported ONNX product folders are created here.

# ── Set the model download directory below. The FireRedASR version (v1 / v2) is auto-detected from its name: ──
#     v1: FireRedASR-AED-L   (uses the `fireredasr`  package from FireRedASR-main)
#     v2: FireRedASR2-AED    (uses the `fireredasr2` package from FireRedASR2S-main)
model_path = f"{DOWNLOAD_ROOT}/FireRedASR2-AED"                                                           # The FireRedASR-AED (v1) / FireRedASR2-AED (v2) model download path.
onnx_dir     = str(SCRIPT_DIR / "FireRedASR_ONNX")

# -- Auto-detect the version from the model download path name (v2 dir names contain "ASR2", e.g. FireRedASR2-AED) --
IS_V2 = "ASR2" in os.path.basename(model_path.rstrip("/"))
if IS_V2:
    project_path = f"{DOWNLOAD_ROOT}/FireRedASR2S-main/fireredasr2s"                                      # FireRedASR2S Github project path (provides the `fireredasr2` package).
    package_name = "fireredasr2"
else:
    project_path = f"{DOWNLOAD_ROOT}/FireRedASR-main"                                     # FireRedASR v1 Github project path (provides the `fireredasr` package).
    package_name = "fireredasr"
os.makedirs(onnx_dir, exist_ok=True)

# -- Exported ONNX graph paths: core pipeline (Embed keeps token ids out of the float decoder; Prefill / Decode build position embedding + causal mask) --
onnx_model_Metadata    = f"{onnx_dir}/FireRedASR_Metadata.onnx"                  # Tiny metadata carrier graph.
onnx_model_Encoder     = f"{onnx_dir}/FireRedASR_Encoder.onnx"                    # The exported onnx encoder model path.
onnx_model_Decoder     = f"{onnx_dir}/FireRedASR_Decoder.onnx"                    # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = f"{onnx_dir}/FireRedASR_Decoder_Embed.onnx"              # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = f"{onnx_dir}/FireRedASR_Position_Mask_Prefill.onnx"      # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = f"{onnx_dir}/FireRedASR_Position_Mask_Decode.onnx"       # Decode position-embedding graph for the single new token.
onnx_model_Greedy      = f"{onnx_dir}/FireRedASR_Greedy_Search.onnx"              # Greedy argmax + save_id history (used with Apply_Penality).
onnx_model_Argmax      = f"{onnx_dir}/FireRedASR_Argmax.onnx"                     # Bare argmax (greedy decoding without a repetition penalty).
onnx_model_First_Beam  = f"{onnx_dir}/FireRedASR_First_Beam_Search.onnx"          # First beam-search step.
onnx_model_Second_Beam = f"{onnx_dir}/FireRedASR_Second_Beam_Search.onnx"         # Subsequent beam-search steps.
onnx_model_Penality    = f"{onnx_dir}/FireRedASR_Apply_Penality.onnx"             # Sliding-window repetition penalty on the logits.


script_dir = Path(__file__).resolve().parent


USE_BEAM_SEARCH = False                                     # Use beam search or greedy search.
INPUT_AUDIO_LENGTH = 480000                                 # The maximum input audio length.
MAX_SEQ_LEN = 448                                           # It should less than 448.
DECODE_MAX_LEN = 0                                          # Match original AED: 0 means use encoder time length.
REPEAT_PENALITY = 1.0                                       # Original FireRedASR-AED uses no repetition penalty; keep 1.0 for parity.
PENALITY_RANGE = 10                                         # Penalizes the most recent output. "30" means the last 30 tokens.
MAX_BEAM_SIZE = 8                                           # Max beams for exported model.
TOP_K = 3                                                   # The top k candidate in decoding.
BEAM_SIZE = 3                                               # Number of beams in searching.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WINDOW_TYPE = 'povey'                                       # Kaldi fbank window type used by the original FireRedASR pipeline.
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
INPUT_AUDIO_DTYPE = "INT16"                                 # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
STOP_TOKEN = [4]                                            # 4 is the end token for FireRedASR-AED series model.
SOS_TOKEN = 3                                               # 3 is the start token for FireRedASR-AED series model.
COMPUTE_IN_F32 = False                                      # F16 KV-cache compute precision. False = minimum-cast f16 attention (self + cross Q@K/mask/softmax/attn@V run in f16 on the f16 caches; the context is cast back to f32). True = keep the f16 KV *storage* (cache I/O dtype unchanged) but upcast K/V (and the mask, internally) to f32 at the attention use points, keeping Q/softmax in f32 (f16 storage, f32 compute).
OPSET = 17                                                  # ONNX opset version for the export.


if project_path not in sys.path:
    sys.path.append(project_path)


# The source below is the ONNX-export-friendly replacement for the Conformer encoder module
# inside the selected FireRedASR project (v1 or v2 — their encoders are numerically identical).
# It is inlined here (instead of being copied from a ./modeling_modified folder) so this export
# script is fully standalone, and is written into the project package before it is imported.
_CONFORMER_ENCODER_SOURCE = r'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerEncoder(nn.Module):
    def __init__(self, idim, n_layers, n_head, d_model,
                 residual_dropout=0.1, dropout_rate=0.1, kernel_size=33,
                 pe_maxlen=5000):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.pad_zeros = torch.zeros((1, 6, 80), dtype=torch.float32)  # 80 = n_mels
        self.pos_num_layers = n_layers
        self.pos_heads = n_head
        self.pos_head_dim = d_model // n_head

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = RelPosEmbConformerBlock(d_model, n_head,
                        residual_dropout,
                        dropout_rate, kernel_size)
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths):
        padded_input = torch.cat((padded_input, self.pad_zeros), dim=1)
        # Conv2dSubsampling rebuilds the padding mask from output_lengths, so the original
        # padding_position_is_0 (a dynamic in-place scatter loop) was dead and is dropped.
        enc_output, input_lengths, valid_lengths, src_mask = self.input_preprocessor(padded_input, input_lengths)
        pos_emb = self.positional_encoding(input_lengths)
        pos_p = torch.matmul(pos_emb, self.pos_weight).reshape(-1, self.pos_num_layers, self.pos_heads, self.pos_head_dim).permute(1, 2, 3, 0)
        for idx, enc_layer in enumerate(self.layer_stack):
            enc_output = enc_layer(enc_output, pos_p[idx], input_lengths, slf_attn_mask=src_mask, pad_mask=src_mask)
        return enc_output, input_lengths, valid_lengths, src_mask


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head,
                 residual_dropout=0.1,
                 dropout_rate=0.1, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model,
                                             residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size,
                                         dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, x_len, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * (x + self.ffn1(x))
        out = self.mhsa(out, out, out, pos_emb, x_len, mask=slf_attn_mask)
        out = self.conv(out, pad_mask)
        out = 0.5 * (out + self.ffn2(out))
        out = self.layer_norm(out)
        return out


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim, d_model, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)
        self.out_size = self.out.in_features

    def forward(self, x, input_lengths):
        x = x.unsqueeze(1)
        x = self.conv(x)
        output_lengths = (input_lengths - 3) // 2 + 1
        output_lengths = (output_lengths - 3) // 2 + 1
        max_len = x.shape[2]
        indices = torch.arange(max_len, device=x.device).expand(1, -1)
        mask = indices < output_lengths.unsqueeze(1)
        x = self.out(x.transpose(1, 2).reshape(1, -1, self.out_size))
        return x, x.shape[1].unsqueeze(0), output_lengths, mask.unsqueeze(1)


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer('pe', pe)
        self.Tmax_half = pe.size(1) // 2
        self.Tmax_half_plus = self.Tmax_half + 1

    def forward(self, x_len):
        return self.pe[:, self.Tmax_half_plus - x_len: self.Tmax_half + x_len].float()


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model*4)
        nonlinear = Swish()
        dropout_pre = nn.Dropout(dropout_rate)
        linear_project = nn.Linear(d_model*4, d_model)
        dropout_post = nn.Dropout(dropout_rate)
        self.net = nn.Sequential(pre_layer_norm,
                                 linear_expand,
                                 nonlinear,
                                 dropout_pre,
                                 linear_project,
                                 dropout_post)

    def forward(self, x):
        return self.net(x) + x


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size=33, dropout_rate=0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model*4, kernel_size=1, bias=False)
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(d_model*2, d_model*2,
                                        kernel_size, stride=1,
                                        padding=self.padding,
                                        groups=d_model*2, bias=False)
        self.batch_norm = nn.LayerNorm(d_model*2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model*2, d_model, kernel_size=1, bias=False)

    def forward(self, x, mask):
        residual = x
        out = self.pre_layer_norm(x)
        out = out.transpose(1, 2)
        out = out.masked_fill(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = F.glu(out, dim=1)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)
        out = self.pointwise_conv2(out)
        out = out.masked_fill(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k

        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)

        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)

        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)

    def forward_qkv(self, q, k, v):
        # Self-attention feeds q == k == v, and layer_norm_q/k/v normalize that shared input identically
        # (same eps / shape), differing only in affine. The affine is folded into the fused qkv Linear (gamma
        # into the weight, beta into the qkv bias), so one affine-less normalization drives all three matmuls.
        normed = self.layer_norm_q(q)
        qkv = self.qkv(normed).reshape(-1, 3 * self.n_head, self.d_k).transpose(0, 1)
        q, k, v = qkv.split(self.n_head, dim=0)
        k = k.transpose(1, 2)
        return q, k, v

    def forward_output(self, output, residual):
        output = torch.matmul(output, self.fc.weight).sum(dim=0, keepdim=True)
        return output + residual


class ScaledDotProductAttention(nn.Module):
    def forward_attention(self, attn, v, mask):
        key_mask = mask.squeeze(1).eq(0).unsqueeze(1)
        attn = attn.masked_fill(key_mask, -float('inf'))
        attn = torch.softmax(attn, dim=-1).masked_fill(key_mask, 0.0)
        return torch.matmul(attn, v)


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__(n_head, d_model,
                         residual_dropout)
        d_k = d_model // n_head
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.zero_pad = torch.zeros((n_head, 2048, 1), dtype=torch.int8)  # 2048 is about 30 seconds audio input.

    def _rel_shift(self, x, x_len):
        x_padded = torch.cat([self.zero_pad[:, :x_len].float(), x], dim=-1)
        x_padded = x_padded.view(self.n_head, -1, x_len)
        x = x_padded[:, 1:].view_as(x)
        return x[:, :, :x_len]

    def forward(self, q, k, v, pos_p, x_len, mask=None):
        residual = q
        q, k, v = self.forward_qkv(q, k, v)
        p = pos_p
        q_with_bias_u = q + self.pos_bias_u
        q_with_bias_v = q + self.pos_bias_v
        matrix_ac = torch.matmul(q_with_bias_u, k)
        matrix_bd = torch.matmul(q_with_bias_v, p)
        matrix_bd = self._rel_shift(matrix_bd, x_len)
        attn_scores = matrix_ac + matrix_bd
        output = self.attention.forward_attention(attn_scores, v, mask=mask)
        return self.forward_output(output, residual)
'''.lstrip("\n")

# Overwrite only the Conformer encoder module of the selected FireRedASR project with the ONNX-export-friendly
# variant above. The v1 (FireRedASR-AED) and v2 (FireRedASR2-AED) encoders are numerically identical, so one
# export source serves both. The rest of the package (fireredasr_aed.py, the decoder, CMVN / fbank feature
# extractor) is imported unmodified; the tokenizer is inlined above so no tokenizer import is needed.
_encoder_module_path = f"{project_path}/{package_name}/models/module/conformer_encoder.py"
with open(_encoder_module_path, "w", encoding="utf-8") as _f:
    _f.write(_CONFORMER_ENCODER_SOURCE)

ASRFeatExtractor = importlib.import_module(f"{package_name}.data.asr_feat").ASRFeatExtractor
FireRedAsrAed = importlib.import_module(f"{package_name}.models.fireredasr_aed").FireRedAsrAed


def load_fireredasr_aed_model(model_path):
    # Mirrors {package}/asr.py::load_fireredasr_aed_model: rebuild FireRedAsrAed from the checkpoint args and
    # load the weights (strict=False so any unused CTC head / LLM-only tensors present in v2 are ignored).
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    return model


STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1   # The length after STFT processed
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
def _bias_or_zero(linear):
    return linear.bias if linear.bias is not None else torch.zeros(linear.out_features, dtype=linear.weight.dtype)


def absorb_layer_norm_affine(norm, linear):
    # Fold a LayerNorm's affine (gamma, beta) into the following nn.Linear so the norm becomes affine-less:
    #   Linear(gamma * xhat + beta) = (W * gamma) @ xhat + (W @ beta + b)
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype))
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))   # b += W @ beta  (uses pre-scaled W)
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))                     # W *= gamma  (per input channel)
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias = None


def _kaldi_fbank_stft_kernel(n_fft, win_length, pre_emphasis):
    frame = torch.arange(win_length, dtype=torch.float64)
    window = torch.pow(0.5 - 0.5 * torch.cos(2.0 * torch.pi * frame / (win_length - 1)), 0.85)
    freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float64).unsqueeze(1)
    omega = 2.0 * torch.pi * freqs * frame.unsqueeze(0) / n_fft
    kernels = []
    for trig in (torch.cos(omega), -torch.sin(omega)):
        coeff = trig * window.unsqueeze(0)
        framed = torch.zeros_like(coeff)
        framed[:, 0] += coeff[:, 0] * (1.0 - pre_emphasis)
        framed[:, 1:] += coeff[:, 1:]
        framed[:, :-1] += -pre_emphasis * coeff[:, 1:]
        framed -= framed.sum(dim=1, keepdim=True) / win_length
        kernels.append(framed)
    return torch.cat(kernels, dim=0).float().unsqueeze(1)


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


class FIRE_RED_ENCODER(torch.nn.Module):
    def __init__(self, fire_red, feat_extractor, nfft_stft, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(FIRE_RED_ENCODER, self).__init__()
        self.model = fire_red
        self.cmvn_means = torch.from_numpy(feat_extractor.cmvn.means).float().view(1, 1, -1)
        self.cmvn_vars = torch.from_numpy(feat_extractor.cmvn.inverse_std_variences).float().view(1, 1, -1)
        self.model.encoder.positional_encoding.pe.data = self.model.encoder.positional_encoding.pe.data.half()
        self.register_buffer('stft_kernel', _kaldi_fbank_stft_kernel(nfft_stft, WINDOW_LENGTH, float(pre_emphasis)))
        self.register_buffer('fbank', (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0))
        self.nfft_stft = nfft_stft
        self.hop_length = HOP_LENGTH
        self.log_floor = float(torch.finfo(torch.float32).eps)
        self.save_en_keys = [None] * num_layers_de
        self.save_en_values = [None] * num_layers_de
        self.num_heads = self.model.encoder.layer_stack._modules['0'].mhsa.n_head
        self.head_dim = self.model.encoder.layer_stack._modules['0'].mhsa.d_k
        self.hidden_size = self.model.encoder.odim
        self.cross_num_heads = self.model.decoder.layer_stack._modules['0'].cross_attn.n_head
        self.cross_head_dim = self.model.decoder.layer_stack._modules['0'].cross_attn.d_k
        self._fuse_weights()

    def _fuse_weights(self):
        with torch.no_grad():
            # Encoder relative-position self-attention: fold the d_k**-0.25 scale into q / k / linear_pos and the
            # two position biases, then reshape q / k / v / linear_pos to per-head (H, hidden, d_k) and fc to
            # (H, d_k, hidden) so the inlined ConformerEncoder forward produces per-head outputs in one matmul each.
            scale = float(self.head_dim ** -0.25)
            pos_weights = []
            for encoder_layer in self.model.encoder.layer_stack:
                mhsa = encoder_layer.mhsa
                mhsa.w_qs.weight.data.mul_(scale)
                mhsa.w_ks.weight.data.mul_(scale)
                mhsa.linear_pos.weight.data.mul_(scale)
                mhsa.pos_bias_u.data = mhsa.pos_bias_u.data.unsqueeze(1) * scale
                mhsa.pos_bias_v.data = mhsa.pos_bias_v.data.unsqueeze(1) * scale

                # Fuse encoder self-attention Q/K/V into one 2-D Linear. This keeps the folded LayerNorm beta as
                # a real linear bias and avoids three rank-3 MatMul nodes per encoder layer.
                ln_q, ln_k, ln_v = mhsa.layer_norm_q, mhsa.layer_norm_k, mhsa.layer_norm_v
                q_weight = mhsa.w_qs.weight.data.clone()
                k_weight = mhsa.w_ks.weight.data.clone()
                v_weight = mhsa.w_vs.weight.data.clone()
                q_bias = torch.matmul(q_weight, ln_q.bias.data)
                k_bias = torch.matmul(k_weight, ln_k.bias.data)
                v_bias = torch.matmul(v_weight, ln_v.bias.data)
                q_weight.mul_(ln_q.weight.data.unsqueeze(0))
                k_weight.mul_(ln_k.weight.data.unsqueeze(0))
                v_weight.mul_(ln_v.weight.data.unsqueeze(0))
                qkv = torch.nn.Linear(mhsa.w_qs.in_features, mhsa.w_qs.out_features * 3, bias=True)
                qkv.weight.copy_(torch.cat([q_weight, k_weight, v_weight], dim=0))
                qkv.bias.copy_(torch.cat([q_bias, k_bias, v_bias], dim=0))
                mhsa.qkv = qkv
                del mhsa.w_qs, mhsa.w_ks, mhsa.w_vs

                pos_weights.append(mhsa.linear_pos.weight.data.clone())
                del mhsa.linear_pos
                mhsa.fc.weight.data = mhsa.fc.weight.data.view(self.hidden_size, self.num_heads, self.head_dim).permute(1, 2, 0).contiguous()

                # Collapse the three identical self-attention LayerNorms into one affine-less normalization:
                # qkv now carries beta in its bias and gamma in its weight; the norm itself stays affine-less.
                for ln in (ln_q, ln_k, ln_v):
                    ln.weight = None
                    ln.bias = None
                    ln.elementwise_affine = False

                # Absorb each Conformer feed-forward pre-LayerNorm into its expand Linear (net[0] -> net[1]).
                absorb_layer_norm_affine(encoder_layer.ffn1.net[0], encoder_layer.ffn1.net[1])
                absorb_layer_norm_affine(encoder_layer.ffn2.net[0], encoder_layer.ffn2.net[1])
            self.model.encoder.register_buffer('pos_weight', torch.cat(pos_weights, dim=0).transpose(0, 1).contiguous())

            # Decoder cross-attention keys/values are produced from the encoder output here; fuse w_ks + w_vs into one
            # Linear and fold the cross-attention d_k**-0.25 scale into the key half (mirrors Whisper's encoder_attn.kv).
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.model.decoder.layer_stack:
                cross_attn = decoder_layer.cross_attn
                out_features = cross_attn.w_ks.out_features
                kv = torch.nn.Linear(cross_attn.w_ks.in_features, out_features * 2, bias=True)
                kv.weight.copy_(torch.cat([cross_attn.w_ks.weight, cross_attn.w_vs.weight], dim=0))
                kv.bias.copy_(torch.cat([_bias_or_zero(cross_attn.w_ks), _bias_or_zero(cross_attn.w_vs)], dim=0))
                kv.weight.data[:out_features].mul_(cross_scale)
                kv.bias.data[:out_features].mul_(cross_scale)
                cross_attn.kv = kv
                del cross_attn.w_ks, cross_attn.w_vs

    def forward(self, audio):
        audio = audio.float()
        spectrum = torch.nn.functional.conv1d(audio, self.stft_kernel, stride=self.hop_length)
        spectrum_square = spectrum * spectrum                       # square once over all 514 channels (real^2 / imag^2 interleaved into one Mul)
        real_part_sq, imag_part_sq = spectrum_square.split(self.nfft_stft // 2 + 1, dim=1)
        mel_features = torch.matmul(self.fbank, real_part_sq + imag_part_sq).transpose(1, 2).clamp(min=self.log_floor).log()
        mel_features = (mel_features - self.cmvn_means) * self.cmvn_vars
        features_len = torch._shape_as_tensor(mel_features)[1].unsqueeze(0)
        enc_outputs, _, valid_lengths, _ = self.model.encoder(mel_features, features_len)
        valid_len = valid_lengths.squeeze(0)
        enc_outputs = enc_outputs[:, :valid_len.to(torch.int64)]
        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            cross_kv = decoder_layer.cross_attn.kv(enc_outputs).half().view(-1, 2 * self.cross_num_heads, self.cross_head_dim).transpose(0, 1)
            k, v = cross_kv.split(self.cross_num_heads, dim=0)
            self.save_en_keys[idx] = k.transpose(1, 2)      # f16 cross-attention key   (num_heads, head_dim, T)
            self.save_en_values[idx] = v                    # f16 cross-attention value (num_heads, T, head_dim)
        return *self.save_en_keys, *self.save_en_values


class FIRE_RED_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Whisper/Qwen Decoder_Embed) so the int
    # token ids never enter the float-only decode graph. The d_model**0.5 scale is folded into the embedding
    # weight here (the absolute position embedding itself is added inside the decoder main graph). The MAIN
    # decoder decouples tgt_word_prj before this runs, so scaling the embedding leaves the logits unscaled.
    def __init__(self, fire_red):
        super(FIRE_RED_DECODER_EMBED, self).__init__()
        self.embed = fire_red.decoder.tgt_word_emb
        self.embed.weight.data *= fire_red.decoder.scale

    def forward(self, input_ids):
        return self.embed(input_ids)


class FIRE_RED_PREFILL(torch.nn.Module):
    # Prefill-phase position-embedding + causal-mask generator (mirrors Whisper/Qwen Prefill).
    # Consumes the int lengths and emits float position embedding + float attention mask so the decoder
    # main graph stays integer-free.
    def __init__(self, fire_red, max_seq_len):
        super(FIRE_RED_PREFILL, self).__init__()
        self.register_buffer('position_weight', fire_red.decoder.positional_encoding.pe[:, :max_seq_len].half())
        self.register_buffer('attention_mask', (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128)

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len].half()   # f16 mask matches the minimum-cast f16 attention scores
        return position_embed, attention_mask, kv_seq_len


class FIRE_RED_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Whisper/Qwen Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, fire_red, max_seq_len):
        super(FIRE_RED_DECODE, self).__init__()
        self.register_buffer('position_weight', fire_red.decoder.positional_encoding.pe[:, :max_seq_len].half())

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class FIRE_RED_DECODER(torch.nn.Module):
    def __init__(self, fire_red, num_layers_de):
        super(FIRE_RED_DECODER, self).__init__()
        self.model = fire_red
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = COMPUTE_IN_F32
        self.idx_en_key = num_layers_de + num_layers_de         # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + num_layers_de     # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + num_layers_de     # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                 # position-embedding input (4 * L + 1)
        self.save_de_keys = [None] * num_layers_de
        self.save_de_values = [None] * num_layers_de
        self.num_heads = self.model.decoder.layer_stack._modules['0'].self_attn.n_head
        self.head_dim = self.model.decoder.layer_stack._modules['0'].self_attn.d_k
        self.hidden_size = self.model.decoder.tgt_word_prj.in_features
        self.cross_num_heads = self.model.decoder.layer_stack._modules['0'].cross_attn.n_head
        self.cross_head_dim = self.model.decoder.layer_stack._modules['0'].cross_attn.d_k
        # tgt_word_prj.weight is tied to tgt_word_emb.weight; decouple (clone) so the logits projection stays
        # unscaled while the Embed graph folds the d_model**0.5 scale into the embedding.
        self.model.decoder.tgt_word_prj.weight = torch.nn.Parameter(self.model.decoder.tgt_word_prj.weight.detach().clone())
        self._fuse_weights()

    def _fuse_weights(self):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            cross_scale = float(self.cross_head_dim ** -0.25)
            for decoder_layer in self.model.decoder.layer_stack:
                attn = decoder_layer.self_attn
                out_features = attn.w_qs.out_features
                qkv = torch.nn.Linear(attn.w_qs.in_features, out_features * 3, bias=True)
                qkv.weight.copy_(torch.cat([attn.w_qs.weight, attn.w_ks.weight, attn.w_vs.weight], dim=0))
                qkv.bias.copy_(torch.cat([_bias_or_zero(attn.w_qs), _bias_or_zero(attn.w_ks), _bias_or_zero(attn.w_vs)], dim=0))
                qkv.weight.data[:out_features * 2].mul_(scale)
                qkv.bias.data[:out_features * 2].mul_(scale)
                attn.qkv = qkv
                del attn.w_qs, attn.w_ks, attn.w_vs
                absorb_layer_norm_affine(decoder_layer.self_attn_norm, qkv)

                cross_attn = decoder_layer.cross_attn
                cross_attn.w_qs.weight.data.mul_(cross_scale)
                cross_attn.w_qs.bias.data.mul_(cross_scale)
                absorb_layer_norm_affine(decoder_layer.cross_attn_norm, cross_attn.w_qs)
                absorb_layer_norm_affine(decoder_layer.mlp_norm, decoder_layer.mlp.w_1)
            # Fold the final decoder LayerNorm affine into the (decoupled) logits projection.
            absorb_layer_norm_affine(self.model.decoder.layer_norm_out, self.model.decoder.tgt_word_prj)

    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        # f16-storage / f32-compute (COMPUTE_IN_F32): the causal mask is kept f16 at the graph boundary (I/O
        # dtype unchanged) and upcast to f32 ONCE here, shared by every layer. Minimum-cast path uses it as-is (f16).
        attn_mask = attention_mask.float() if self.compute_in_f32 else attention_mask
        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            hidden_states_norm = decoder_layer.self_attn_norm(hidden_states)
            # Self-attention. OFF (minimum-cast): cast the fused QKV DOWN to f16 before the split so
            # Q@K/mask/softmax/attn@V run in f16 on the f16 K/V cache; the context is cast back to f32 for fc.
            # ON (COMPUTE_IN_F32): keep the f16 K/V *storage* (K/V still cast to f16 before the cache concat, so
            # the cache I/O dtype is unchanged) but upcast K/V to f32 at the matmul use points and keep
            # Q/mask/softmax in f32 -- f16 storage, f32 compute. Q is never downcast.
            qkv = decoder_layer.self_attn.qkv(hidden_states_norm)
            if not self.compute_in_f32:
                qkv = qkv.half()
            qkv = qkv.view(batch_size, -1, 3 * self.num_heads, self.head_dim).transpose(1, 2)
            q, k, v = qkv.split(self.num_heads, dim=1)
            if self.compute_in_f32:
                k = k.half()   # f16 K storage (no-op in the minimum-cast path: qkv is already f16)
                v = v.half()   # f16 V storage
            k = torch.cat((all_inputs[idx], k.transpose(-1, -2)), dim=-1)            # f16 key cache   (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v), dim=-2)        # f16 value cache (batch, num_heads, kv_seq_len, head_dim)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            if self.compute_in_f32:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k.float()) + attn_mask, dim=-1), v.float()).transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
            else:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attn_mask, dim=-1), v).transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float()
            hidden_state_attn = decoder_layer.self_attn.fc(hidden_state_attn)
            hidden_state_attn += hidden_states
            # Cross-attention against the f16 encoder cross-KV cache. OFF: downcast Q to f16 and run in f16 on
            # the f16 cross cache, context back to f32. ON: keep Q in f32 and upcast the f16 cross K/V to f32 at
            # the matmul use points (the cross cache is produced f16 by the encoder; its I/O dtype is unchanged).
            q = decoder_layer.cross_attn.w_qs(decoder_layer.cross_attn_norm(hidden_state_attn)).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            if self.compute_in_f32:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = decoder_layer.cross_attn.fc(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size))
            else:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = decoder_layer.cross_attn.fc(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.mlp(decoder_layer.mlp_norm(hidden_state_cross))
        hidden_states = self.model.decoder.layer_norm_out(hidden_states[:, -1])
        logits = self.model.decoder.tgt_word_prj(hidden_states)
        return *self.save_de_keys, *self.save_de_values, logits


def build_model_metadata(*sections):
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
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


def read_dict_token_ids(dict_path):
    token_to_id = {}
    with open(dict_path, encoding='utf8') as file:
        for index, line in enumerate(file):
            tokens = line.strip().split()
            if len(tokens) >= 2:
                token_to_id[tokens[0]] = int(tokens[1])
            elif len(tokens) == 1:
                token_to_id[tokens[0]] = index
    return token_to_id


print('\nStart to export the Encoder part.\n')
with torch.inference_mode():
    if 'aed' in model_path or 'AED' in model_path or 'Aed' in model_path:
        feat_extractor = ASRFeatExtractor(model_path + "/cmvn.ark")
        model = load_fireredasr_aed_model(model_path + "/model.pth.tar").float()
        model.eval()
        HIDDEN_SIZE = model.encoder.odim
        NUM_HEAD_EN = model.encoder.layer_stack._modules['0'].mhsa.n_head
        NUM_HEAD_DE = model.decoder.layer_stack._modules['0'].self_attn.n_head
        NUM_LAYER_DE = model.decoder.n_layers
        HEAD_DIM_EN = model.encoder.layer_stack._modules['0'].mhsa.d_k
        HEAD_DIM_DE = model.decoder.layer_stack._modules['0'].self_attn.d_k
        CROSS_NUM_HEAD_DE = model.decoder.layer_stack._modules['0'].cross_attn.n_head
        CROSS_HEAD_DIM_DE = model.decoder.layer_stack._modules['0'].cross_attn.d_k
        VOCAB_SIZE = model.decoder.tgt_word_prj.out_features

        # All attention weight fusion + scale folding now happens inside each module's _fuse_weights(); no external
        # pre-scaling loops are needed here. ORDER MATTERS: build the encoder first (it fuses the decoder cross-attn
        # k/v into `kv` and deletes w_ks/w_vs), then the decoder main (decouples tgt_word_prj before the embedding is
        # scaled, fuses self-attn qkv, folds the cross-attn q scale), then the embedding graph (scales the embedding).
        fire_red_encoder = FIRE_RED_ENCODER(model, feat_extractor, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)

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
            fire_red_encoder,
            (audio,),
            onnx_model_Encoder,
            input_names=['audio'],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False,
            external_data=True
        )
        del fire_red_encoder
        del audio
        del name
        del output_names
        del dynamic_axes
        gc.collect()
        print("\nExport Done!\n\nStart to export the Decoder part.")

        # ── Decoder main graph (pure float: token + position embeddings and the mask arrive as inputs) ──
        fire_red_decoder = FIRE_RED_DECODER(model, NUM_LAYER_DE)
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
        dynamic_axes['logits'] = {0: 'batch'}

        torch.onnx.export(
            fire_red_decoder,
            tuple(all_inputs),
            onnx_model_Decoder,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False
        )
        del fire_red_decoder
        del save_encoder_key
        del save_encoder_value
        del hidden_states_de
        del position_embed_de
        del attention_mask
        del input_names
        del output_names
        del dynamic_axes
        gc.collect()

        # ── Decoder token-embedding graph (keeps int ids out of the decoder; scale folded into the embedding) ──
        fire_red_embed = FIRE_RED_DECODER_EMBED(model)
        embed_input_ids = torch.ones((1, 1), dtype=torch.int32)
        torch.onnx.export(
            fire_red_embed,
            (embed_input_ids,),
            onnx_model_Embed,
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
        del fire_red_embed
        del embed_input_ids

        # ── Prefill position-embedding + causal-mask graph ──
        fire_red_prefill = FIRE_RED_PREFILL(model, MAX_SEQ_LEN)
        prefill_ids_len = torch.tensor([1], dtype=torch.int64)
        prefill_history_len = torch.tensor([0], dtype=torch.int64)
        torch.onnx.export(
            fire_red_prefill,
            (prefill_ids_len, prefill_history_len),
            onnx_model_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['position_embed', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'position_embed': {1: 'ids_len'},
                'attention_mask': {1: 'ids_len', 2: 'kv_seq_len'}
            },
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False
        )
        del fire_red_prefill
        del prefill_ids_len
        del prefill_history_len

        # ── Decode position-embedding graph for the single new token ──
        fire_red_decode = FIRE_RED_DECODE(model, MAX_SEQ_LEN)
        decode_kv_seq_len = torch.tensor([1], dtype=torch.int64)
        torch.onnx.export(
            fire_red_decode,
            (decode_kv_seq_len,),
            onnx_model_Decode,
            input_names=['kv_seq_len'],
            output_names=['position_embed', 'kv_seq_len_next'],
            dynamic_axes={},
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False
        )
        del model
        del fire_red_decode
        del decode_kv_seq_len
        gc.collect()
    else:
        print("Currently, only support the FireRedASR-AED")

    # ── Decode-strategy graphs (Qwen ASR style) ──
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
        opset_version=OPSET,
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
        do_constant_folding=True,
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
        do_constant_folding=True,
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

    token_to_id = read_dict_token_ids(model_path + "/dict.txt")
    sos_token_id = int(token_to_id.get("<sos>", SOS_TOKEN))
    eos_token_id = int(token_to_id.get("<eos>", STOP_TOKEN[0]))
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
            "fireredasr_metadata_version": 1,
            "producer": "Export_FireRedASR_AED.py",
            "model_variant": "FireRedASR2-AED" if IS_V2 else "FireRedASR-AED",
            "compute_in_f32": COMPUTE_IN_F32,
        },
        {
            "num_decoder_layers": NUM_LAYER_DE,
            "num_encoder_heads": NUM_HEAD_EN,
            "num_decoder_heads": NUM_HEAD_DE,
            "encoder_head_dim": HEAD_DIM_EN,
            "decoder_head_dim": HEAD_DIM_DE,
            "cross_num_heads": CROSS_NUM_HEAD_DE,
            "cross_head_dim": CROSS_HEAD_DIM_DE,
            "hidden_size": HIDDEN_SIZE,
            "vocab_size": VOCAB_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "input_audio_length": INPUT_AUDIO_LENGTH,
            "num_mels": N_MELS,
            "nfft_stft": NFFT_STFT,
            "window_length": WINDOW_LENGTH,
            "hop_length": HOP_LENGTH,
            "window_type": WINDOW_TYPE,
            "pre_emphasis": PRE_EMPHASIZE,
        },
        {
            "sos_token_id": sos_token_id,
            "eos_token_id": eos_token_id,
            "stop_token_ids": ",".join(str(t) for t in [eos_token_id]),
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
        except Exception as _exc:  # noqa: BLE001 - one bad graph must not abort export
            _skipped.append(f"{os.path.basename(_target)} ({_exc})")

    print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(_written)} ONNX graph(s):")
    for _key in sorted(onnx_metadata):
        print(f"    {_key} = {onnx_metadata[_key]}")
    if _skipped:
        print("[Metadata] Skipped (kept usable, metadata not written):")
        for _entry in _skipped:
            print(f"    {_entry}")
    gc.collect()


if project_path in sys.path:
    sys.path.remove(project_path)

# ── Copy the tokenizer assets (vocab + BPE model) into the ONNX folder so the exported folder runs ──
# inference stand-alone (no external FireRedASR source path needed at inference time).
for _asset in ("dict.txt", "train_bpe1000.model"):
    _src = os.path.join(model_path, _asset)
    _dst = os.path.join(onnx_dir, _asset)
    try:
        shutil.copy2(_src, _dst)
        print(f"[Tokenizer] Copied {_asset} -> {onnx_dir}")
    except Exception as _exc:  # noqa: BLE001 - a failed copy must not abort the auto demo
        print(f"[Tokenizer] Skipped {_asset} ({_exc})")

print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_FireRedASR_AED_ONNX.py ...')
subprocess.run(
    [
        sys.executable,
        str(SCRIPT_DIR / "Inference_FireRedASR_AED_ONNX.py"),
        "--onnx-folder", onnx_dir,
    ],
    check=True,
)
