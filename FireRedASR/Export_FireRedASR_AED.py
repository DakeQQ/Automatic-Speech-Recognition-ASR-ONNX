import gc
import json
import shutil
import os
import subprocess
import sys
import importlib
import tempfile
import types
import torch
import torchaudio
from pathlib import Path

import Shared_Merged


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

# ============================== USER CONFIG ==============================
DOWNLOAD_ROOT = str(Path.home() / "Downloads")
model_path = f"{DOWNLOAD_ROOT}/FireRedASR2-AED"
INPUT_AUDIO_LENGTH = 480000
MAX_SEQ_LEN = 448
REORDER_DOWNPROJ_FOR_QUANT = True
REORDER_OPROJ_FOR_QUANT = True
REORDER_KEY = "absmean"
INPUT_AUDIO_DTYPE = "F32"
USE_FP16_KV = True
COMPUTE_IN_F32 = False
KV_DTYPE = torch.float16 if USE_FP16_KV else torch.float32
OPSET = 20
# ============================ END USER CONFIG ============================


ONNX_DIR = SCRIPT_DIR / "FireRedASR_ONNX"

# Split graphs are ephemeral merge constituents. They are deleted immediately after merged parity
# validation, including when export fails, so only the final merged/shared deployment is preserved.
_raw_onnx_temp = tempfile.TemporaryDirectory(prefix="fireredasr_export_")
_raw_onnx_dir = Path(_raw_onnx_temp.name)
onnx_dir = str(ONNX_DIR)

# -- Auto-detect the version from the model download path name (v2 dir names contain "ASR2", e.g. FireRedASR2-AED) --
IS_V2 = "ASR2" in os.path.basename(model_path.rstrip("/"))
if IS_V2:
    project_path = f"{DOWNLOAD_ROOT}/FireRedASR2S-main/fireredasr2s"
    package_name = "fireredasr2"
else:
    project_path = f"{DOWNLOAD_ROOT}/FireRedASR-main"
    package_name = "fireredasr"
os.makedirs(_raw_onnx_dir, exist_ok=True)

MODEL_FILE_NAMES = dict(Shared_Merged.DEFAULT_MODEL_FILE_NAMES)

# -- Exported ONNX graph paths: core pipeline (Embed keeps token ids out of the float decoder; Prefill / Decode build position embedding + causal mask) --
onnx_model_Metadata    = str(_raw_onnx_dir / MODEL_FILE_NAMES["metadata"])          # Tiny metadata carrier graph.
onnx_model_Encoder     = str(_raw_onnx_dir / MODEL_FILE_NAMES["encoder"])           # The exported onnx encoder model path.
onnx_model_Decoder     = str(_raw_onnx_dir / MODEL_FILE_NAMES["main"])              # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = str(_raw_onnx_dir / MODEL_FILE_NAMES["embed"])             # Token-embedding graph, absorbed into merged graphs.
onnx_model_Prefill     = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_prefill"])  # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_decode"])   # Decode position-embedding graph for one token.
onnx_model_Greedy      = str(_raw_onnx_dir / MODEL_FILE_NAMES["penalty_greedy"])    # Greedy argmax + save_id history.
onnx_model_Argmax      = str(_raw_onnx_dir / MODEL_FILE_NAMES["greedy"])            # Bare argmax.
onnx_model_Sampling    = str(_raw_onnx_dir / MODEL_FILE_NAMES["sampling"])          # Top-K/Top-P sampling + token history.
onnx_model_Penalty     = str(_raw_onnx_dir / MODEL_FILE_NAMES["penalty"])           # Sliding-window repetition penalty.

for _artifact_folder in (
    _raw_onnx_dir,
    ONNX_DIR,
    SCRIPT_DIR / "FireRedASR_Optimized",
):
    _removed_artifacts = Shared_Merged.delete_obsolete_strategy_artifacts(
        _artifact_folder,
        MODEL_FILE_NAMES,
    )
    if _removed_artifacts:
        print(
            f"[Cleanup] Removed {len(_removed_artifacts)} obsolete strategy "
            f"artifact(s) from {_artifact_folder}."
        )


# Fixed FireRedASR model constants and metadata defaults; these are not user tunables.
_MODEL_NUM_MELS = 80
_MODEL_NFFT_STFT = 512
_MODEL_WINDOW_LENGTH = 400
_MODEL_HOP_LENGTH = 160
_MODEL_PRE_EMPHASIS = 0.97
_MODEL_SAMPLE_RATE = 16000
_METADATA_PENALTY_RANGE = 10


if project_path not in sys.path:
    sys.path.append(project_path)


# The source below is the ONNX-export-friendly replacement for the Conformer encoder module
# inside the selected FireRedASR project (v1 or v2 — their encoders are numerically identical).
# It is inlined here (instead of being copied from a ./modeling_modified folder) so this export
# script is fully standalone, then installed in memory before FireRedAsrAed is imported.
_CONFORMER_ENCODER_SOURCE = r'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class _OnnxMaskedFill(torch.autograd.Function):
    """Preserve masked_fill semantics while emitting one standard Where node."""
    @staticmethod
    def forward(ctx, value, mask, fill_value):
        return torch.where(mask, fill_value, value)

    @staticmethod
    def symbolic(g, value, mask, fill_value):
        return g.op("Where", mask, fill_value, value)


class _OnnxSplitThreeDim0(torch.autograd.Function):
    """Emit Split with one shared int64 split-size initializer."""
    @staticmethod
    def forward(ctx, value, split_sizes):
        sizes = tuple(int(size) for size in split_sizes.tolist())
        return tuple(torch.split(value, sizes, dim=0))

    @staticmethod
    def symbolic(g, value, split_sizes):
        return g.op("Split", value, split_sizes, axis_i=0, outputs=3)


class _OnnxReshape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, shape):
        target = [int(dim) for dim in shape.tolist()]
        target = [value.shape[index] if dim == 0 else dim for index, dim in enumerate(target)]
        return value.reshape(tuple(target))

    @staticmethod
    def symbolic(g, value, shape):
        return g.op("Reshape", value, shape, allowzero_i=0)


class _OnnxSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, starts, ends, axes, steps):
        index = [slice(None)] * value.dim()
        for start, end, axis, step in zip(
                starts.tolist(), ends.tolist(), axes.tolist(), steps.tolist()):
            index[int(axis)] = slice(int(start), int(end), int(step))
        return value[tuple(index)]

    @staticmethod
    def symbolic(g, value, starts, ends, axes, steps):
        return g.op("Slice", value, starts, ends, axes, steps)


class _OnnxSplitDim0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, split_sizes, output_count):
        sizes = tuple(int(size) for size in split_sizes.tolist())
        return tuple(torch.split(value, sizes, dim=0))

    @staticmethod
    def symbolic(g, value, split_sizes, output_count):
        return g.op(
            "Split", value, split_sizes,
            axis_i=0, outputs=output_count)


class _OnnxSqueeze(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, axes):
        result = value
        for axis in sorted((int(item) for item in axes.tolist()), reverse=True):
            result = result.squeeze(axis)
        return result

    @staticmethod
    def symbolic(g, value, axes):
        return g.op("Squeeze", value, axes)


class ConformerEncoder(nn.Module):
    def __init__(self, idim, n_layers, n_head, d_model,
                 residual_dropout=0.1, dropout_rate=0.1, kernel_size=33,
                 pe_maxlen=5000):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.register_buffer('pad_zeros', torch.zeros((1, 6, 80), dtype=torch.float32))
        self.register_buffer('mask_zero', torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer('mask_neg_inf', torch.tensor(-float('inf'), dtype=torch.float32))
        self.register_buffer('residual_half', torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer('qkv_split_sizes', torch.full((3,), n_head, dtype=torch.int64))
        self.register_buffer('qkv_shape', torch.tensor([-1, 3 * n_head, d_model // n_head], dtype=torch.int64))
        self.register_buffer('rel_shift_zero', torch.zeros((n_head, 2048, 1), dtype=torch.float32))
        self.register_buffer('rel_shift_shape_prefix', torch.tensor([0, -1], dtype=torch.int64))
        self.register_buffer('rel_shift_infer', torch.tensor([-1], dtype=torch.int64))
        self.register_buffer('slice_zero', torch.tensor([0], dtype=torch.int64))
        self.register_buffer('slice_one', torch.tensor([1], dtype=torch.int64))
        self.register_buffer('slice_end_max', torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64))
        self.register_buffer('slice_axis_two', torch.tensor([2], dtype=torch.int64))
        self.register_buffer('layer_split_sizes', torch.ones(n_layers, dtype=torch.int64))
        self.pos_num_layers = n_layers
        self.pos_heads = n_head
        self.pos_head_dim = d_model // n_head

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = RelPosEmbConformerBlock(
                d_model, n_head, kernel_size,
                self.qkv_split_sizes, self.qkv_shape,
                self.slice_zero, self.slice_one,
                self.slice_end_max, self.slice_one,
                self.slice_axis_two, self.slice_one)
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths):
        padded_input = torch.cat((padded_input, self.pad_zeros), dim=1)
        # Conv2dSubsampling rebuilds the padding mask from output_lengths, so the original
        # padding_position_is_0 (a dynamic in-place scatter loop) was dead and is dropped.
        enc_output, input_lengths, valid_lengths, src_mask = self.input_preprocessor(padded_input, input_lengths)
        pos_start = self.positional_encoding.Tmax_half_plus - input_lengths
        pos_end = self.positional_encoding.Tmax_half + input_lengths
        pos_emb = _OnnxSlice.apply(
            self.positional_encoding.pe, pos_start, pos_end,
            self.slice_one, self.slice_one).float()
        pos_p = torch.matmul(pos_emb, self.pos_weight).reshape(-1, self.pos_num_layers, self.pos_heads, self.pos_head_dim).permute(1, 2, 3, 0)
        pos_layers = _OnnxSplitDim0.apply(
            pos_p, self.layer_split_sizes, self.pos_num_layers)
        rel_shift_zero = _OnnxSlice.apply(
            self.rel_shift_zero, self.slice_zero, input_lengths,
            self.slice_one, self.slice_one)
        rel_shift_shape = torch.cat((self.rel_shift_shape_prefix, input_lengths))
        rel_shift_restore_shape = torch.cat((self.slice_zero, input_lengths, self.rel_shift_infer))
        for idx, enc_layer in enumerate(self.layer_stack):
            pos_layer = _OnnxSqueeze.apply(pos_layers[idx], self.slice_zero)
            enc_output = enc_layer(
                enc_output, pos_layer, input_lengths,
                slf_attn_mask=src_mask, pad_mask=src_mask,
            mask_zero=self.mask_zero, mask_neg_inf=self.mask_neg_inf,
            residual_half=self.residual_half,
            rel_shift_zero=rel_shift_zero,
            rel_shift_shape=rel_shift_shape,
            rel_shift_restore_shape=rel_shift_restore_shape)
        return enc_output, input_lengths, valid_lengths, src_mask


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head, kernel_size=33,
                 qkv_split_sizes=None, qkv_shape=None,
                 slice_start_zero=None, slice_start_one=None,
                 slice_end_max=None, slice_axis_one=None,
                 slice_axis_two=None, slice_step_one=None):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model)
        self.mhsa = RelPosMultiHeadAttention(
            n_head, d_model,
            qkv_split_sizes, qkv_shape,
            slice_start_zero, slice_start_one,
            slice_end_max, slice_axis_one,
            slice_axis_two, slice_step_one)
        self.conv = ConformerConvolution(d_model, kernel_size)
        self.ffn2 = ConformerFeedForward(d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, x_len, slf_attn_mask=None, pad_mask=None,
                mask_zero=None, mask_neg_inf=None, residual_half=None,
                rel_shift_zero=None, rel_shift_shape=None,
                rel_shift_restore_shape=None):
        out = residual_half * (x + self.ffn1(x))
        out = self.mhsa(
            out, out, out, pos_emb, x_len, mask=slf_attn_mask,
            mask_zero=mask_zero, mask_neg_inf=mask_neg_inf,
            rel_shift_zero=rel_shift_zero,
            rel_shift_shape=rel_shift_shape,
            rel_shift_restore_shape=rel_shift_restore_shape)
        out = self.conv(out, pad_mask, mask_zero)
        out = residual_half * (out + self.ffn2(out))
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
    def __init__(self, d_model):
        super().__init__()
        pre_layer_norm = nn.LayerNorm(d_model)
        linear_expand = nn.Linear(d_model, d_model*4)
        nonlinear = Swish()
        linear_project = nn.Linear(d_model*4, d_model)
        self.net = nn.Sequential(pre_layer_norm,
                                 linear_expand,
                                 nonlinear,
                                 nn.Identity(),
                                 linear_project,
                                 nn.Identity())

    def forward(self, x):
        return self.net(x) + x


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size=33):
        super().__init__()
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

    def forward(self, x, mask, mask_zero):
        residual = x
        out = self.pre_layer_norm(x)
        invalid_mask = mask.ne(1)
        invalid_mask_last = invalid_mask.transpose(1, 2)
        out = _OnnxMaskedFill.apply(out, invalid_mask_last, mask_zero)
        out = self.pointwise_linear1(out)
        out = F.glu(out, dim=-1)
        out = self.depthwise_conv(out.transpose(1, 2))
        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = self.pointwise_linear2(out)
        out = _OnnxMaskedFill.apply(out, invalid_mask_last, mask_zero)
        return out + residual


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model):
        super().__init__()
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
        qkv = _OnnxReshape.apply(self.qkv(normed), self.qkv_shape).transpose(0, 1)
        q, k, v = _OnnxSplitThreeDim0.apply(qkv, self.qkv_split_sizes)
        k = k.transpose(1, 2)
        return q, k, v

    def forward_output(self, output, residual):
        output = self.fc(
            output.transpose(0, 1).reshape(1, -1, self.fc.in_features)
        )
        return output + residual


class ScaledDotProductAttention(nn.Module):
    def forward_attention(self, attn, v, mask, mask_zero, mask_neg_inf):
        key_mask = mask.squeeze(1).eq(0).unsqueeze(1)
        attn = _OnnxMaskedFill.apply(attn, key_mask, mask_neg_inf)
        attn = torch.softmax(attn, dim=-1)
        attn = _OnnxMaskedFill.apply(attn, key_mask, mask_zero)
        return torch.matmul(attn, v)


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model,
                 qkv_split_sizes=None, qkv_shape=None,
                 slice_start_zero=None, slice_start_one=None,
                 slice_end_max=None, slice_axis_one=None,
                 slice_axis_two=None, slice_step_one=None):
        super().__init__(n_head, d_model)
        d_k = d_model // n_head
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)
        self.qkv_split_sizes = qkv_split_sizes
        self.qkv_shape = qkv_shape
        self.slice_start_zero = slice_start_zero
        self.slice_start_one = slice_start_one
        self.slice_end_max = slice_end_max
        self.slice_axis_one = slice_axis_one
        self.slice_axis_two = slice_axis_two
        self.slice_step_one = slice_step_one

    def _rel_shift(self, x, x_len, zero, shape, restore_shape):
        x_padded = torch.cat([zero, x], dim=-1)
        x_padded = _OnnxReshape.apply(x_padded, shape)
        x_padded = _OnnxSlice.apply(
            x_padded, self.slice_start_one, self.slice_end_max,
            self.slice_axis_one, self.slice_step_one)
        x = _OnnxReshape.apply(x_padded, restore_shape)
        return _OnnxSlice.apply(
            x, self.slice_start_zero, x_len,
            self.slice_axis_two, self.slice_step_one)

    def forward(self, q, k, v, pos_p, x_len, mask=None,
                mask_zero=None, mask_neg_inf=None, rel_shift_zero=None,
                rel_shift_shape=None, rel_shift_restore_shape=None):
        residual = q
        q, k, v = self.forward_qkv(q, k, v)
        p = pos_p
        q_with_bias_u = q + self.pos_bias_u
        q_with_bias_v = q + self.pos_bias_v
        matrix_ac = torch.matmul(q_with_bias_u, k)
        matrix_bd = torch.matmul(q_with_bias_v, p)
        matrix_bd = self._rel_shift(
            matrix_bd, x_len, rel_shift_zero,
            rel_shift_shape, rel_shift_restore_shape)
        attn_scores = matrix_ac + matrix_bd
        output = self.attention.forward_attention(
            attn_scores, v, mask, mask_zero, mask_neg_inf)
        return self.forward_output(output, residual)
'''.lstrip("\n")

# Install the ONNX-friendly Conformer module in memory before importing FireRedAsrAed. This preserves the
# downloaded FireRed source tree and avoids the old export-time site-package/project-file mutation.
_encoder_module_name = f"{package_name}.models.module.conformer_encoder"
_encoder_module = types.ModuleType(_encoder_module_name)
_encoder_module.__file__ = f"<{_encoder_module_name}:onnx-export>"
_encoder_module.__package__ = f"{package_name}.models.module"
exec(compile(_CONFORMER_ENCODER_SOURCE, _encoder_module.__file__, "exec"), _encoder_module.__dict__)
sys.modules[_encoder_module_name] = _encoder_module

ASRFeatExtractor = importlib.import_module(f"{package_name}.data.asr_feat").ASRFeatExtractor
FireRedAsrAed = importlib.import_module(f"{package_name}.models.fireredasr_aed").FireRedAsrAed


def load_fireredasr_aed_model(model_path):
    # Mirrors {package}/asr.py::load_fireredasr_aed_model: rebuild FireRedAsrAed from the checkpoint args and
    # load the weights (strict=False so any unused CTC head / LLM-only tensors present in v2 are ignored).
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model


_EXPORT_HOP_LENGTH = min(_MODEL_HOP_LENGTH, INPUT_AUDIO_LENGTH)
STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // _EXPORT_HOP_LENGTH + 1
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


class _OnnxReshape(torch.autograd.Function):
    """Use ONNX Reshape's copied dimension instead of a dynamic Shape/Concat target."""
    @staticmethod
    def forward(ctx, value, shape):
        target = [int(dim) for dim in shape.tolist()]
        target = [value.shape[index] if dim == 0 else dim for index, dim in enumerate(target)]
        return value.reshape(tuple(target))

    @staticmethod
    def symbolic(g, value, shape):
        return g.op("Reshape", value, shape, allowzero_i=0)


class _OnnxSlice(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, starts, ends, axes, steps):
        index = [slice(None)] * value.dim()
        for start, end, axis, step in zip(
                starts.tolist(), ends.tolist(), axes.tolist(), steps.tolist()):
            index[int(axis)] = slice(int(start), int(end), int(step))
        return value[tuple(index)]

    @staticmethod
    def symbolic(g, value, starts, ends, axes, steps):
        return g.op("Slice", value, starts, ends, axes, steps)


class _OnnxSplitThreeDim1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, split_sizes):
        sizes = tuple(int(size) for size in split_sizes.tolist())
        return tuple(torch.split(value, sizes, dim=1))

    @staticmethod
    def symbolic(g, value, split_sizes):
        return g.op("Split", value, split_sizes, axis_i=1, outputs=3)


class _OnnxSplitTwoDim0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, split_sizes):
        sizes = tuple(int(size) for size in split_sizes.tolist())
        return tuple(torch.split(value, sizes, dim=0))

    @staticmethod
    def symbolic(g, value, split_sizes):
        return g.op("Split", value, split_sizes, axis_i=0, outputs=2)


class _OnnxExactGelu(torch.autograd.Function):
    """Keep PyTorch exact GELU in eager mode and share its three ONNX scalars."""
    @staticmethod
    def forward(ctx, value, sqrt_two, one, half):
        return torch.nn.functional.gelu(value, approximate="none")

    @staticmethod
    def symbolic(g, value, sqrt_two, one, half):
        erf = g.op("Erf", g.op("Div", value, sqrt_two))
        scaled = g.op("Mul", value, g.op("Add", erf, one))
        return g.op("Mul", scaled, half)


class _OnnxGatherLastToken(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, index):
        return value[:, int(index.item())]

    @staticmethod
    def symbolic(g, value, index):
        return g.op("Gather", value, index, axis_i=1)


class _OnnxGatherElementsInt32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices):
        return torch.gather(value, 1, indices.long())

    @staticmethod
    def symbolic(g, value, indices):
        return g.op("GatherElements", value, indices, axis_i=1)


class _OnnxScatterElementsInt32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, indices, updates):
        return value.scatter(1, indices.long(), updates)

    @staticmethod
    def symbolic(g, value, indices, updates):
        return g.op("ScatterElements", value, indices, updates, axis_i=1)


def _share_affine_less_layer_norm(module, hidden_size):
    """Replace exporter-created per-call ones/zeros with one immutable pair."""
    shared_weight = torch.nn.Parameter(torch.ones(hidden_size), requires_grad=False)
    shared_bias = torch.nn.Parameter(torch.zeros(hidden_size), requires_grad=False)
    shared_count = 0
    for layer_norm in module.modules():
        if not isinstance(layer_norm, torch.nn.LayerNorm):
            continue
        if layer_norm.elementwise_affine or tuple(layer_norm.normalized_shape) != (hidden_size,):
            continue
        layer_norm.elementwise_affine = True
        layer_norm.weight = shared_weight
        layer_norm.bias = shared_bias
        shared_count += 1
    return shared_count


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


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    """Qwen-v3-compatible Top-K/Top-P sampling with on-device history."""

    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "neg_inf",
            torch.tensor(self.NEG_INF, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gumbel_min",
            torch.tensor(self.GUMBEL_EPS, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "gumbel_max",
            torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        logits,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        previous_ids,
    ):
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


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super(METADATA_CARRIER, self).__init__()

    def forward(self, marker):
        return marker


class APPLY_PENALTY(torch.nn.Module):
    # Sliding-window repetition penalty (Qwen ASR style): multiply the logits of the most recent
    # `penalty_range` tokens by `penalty_value`. The one-element int64 range remains a live ONNX input.
    def __init__(self):
        super(APPLY_PENALTY, self).__init__()
        self.register_buffer(
            'slice_end_max',
            torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64),
        )
        self.register_buffer('slice_one', torch.tensor([1], dtype=torch.int64))

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = _OnnxSlice.apply(
            save_id, -penalty_range, self.slice_end_max,
            self.slice_one, self.slice_one)
        penalised = _OnnxGatherElementsInt32.apply(
            logits, target_indices) * penalty_value
        return _OnnxScatterElementsInt32.apply(
            logits, target_indices, penalised)


class FIRE_RED_ENCODER(torch.nn.Module):
    def __init__(self, fire_red, feat_extractor, nfft_stft, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(FIRE_RED_ENCODER, self).__init__()
        self.model = fire_red
        self.register_buffer('cmvn_means', torch.from_numpy(feat_extractor.cmvn.means).float().view(1, 1, -1))
        self.register_buffer('cmvn_vars', torch.from_numpy(feat_extractor.cmvn.inverse_std_variences).float().view(1, 1, -1))
        self.model.encoder.positional_encoding.pe.data = self.model.encoder.positional_encoding.pe.data.to(KV_DTYPE)
        self.register_buffer('stft_kernel', _kaldi_fbank_stft_kernel(nfft_stft, _MODEL_WINDOW_LENGTH, float(pre_emphasis)))
        self.register_buffer('fbank', (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0))
        self.slice_zero = self.model.encoder.slice_zero
        self.slice_one = self.model.encoder.slice_one
        self.nfft_stft = nfft_stft
        self.hop_length = _EXPORT_HOP_LENGTH
        self.log_floor = float(torch.finfo(torch.float32).eps)
        self.save_en_keys = [None] * num_layers_de
        self.save_en_values = [None] * num_layers_de
        self.num_heads = self.model.encoder.layer_stack._modules['0'].mhsa.n_head
        self.head_dim = self.model.encoder.layer_stack._modules['0'].mhsa.d_k
        self.hidden_size = self.model.encoder.odim
        self.cross_num_heads = self.model.decoder.layer_stack._modules['0'].cross_attn.n_head
        self.cross_head_dim = self.model.decoder.layer_stack._modules['0'].cross_attn.d_k
        self.register_buffer(
            'cross_kv_split_sizes',
            torch.full((2,), self.cross_num_heads, dtype=torch.int64),
        )
        self.register_buffer(
            'cross_kv_shape',
            torch.tensor([-1, 2 * self.cross_num_heads, self.cross_head_dim], dtype=torch.int64),
        )
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

                # Collapse the three identical self-attention LayerNorms into one affine-less normalization:
                # qkv now carries beta in its bias and gamma in its weight; the norm itself stays affine-less.
                for ln in (ln_q, ln_k, ln_v):
                    ln.weight = None
                    ln.bias = None
                    ln.elementwise_affine = False
                del mhsa.layer_norm_k, mhsa.layer_norm_v

                convolution = encoder_layer.conv
                convolution.pointwise_linear1 = torch.nn.Linear(
                    convolution.pointwise_conv1.in_channels,
                    convolution.pointwise_conv1.out_channels,
                    bias=False,
                )
                convolution.pointwise_linear1.weight.copy_(
                    convolution.pointwise_conv1.weight.squeeze(-1)
                )
                convolution.pointwise_linear2 = torch.nn.Linear(
                    convolution.pointwise_conv2.in_channels,
                    convolution.pointwise_conv2.out_channels,
                    bias=False,
                )
                convolution.pointwise_linear2.weight.copy_(
                    convolution.pointwise_conv2.weight.squeeze(-1)
                )
                del convolution.pointwise_conv1, convolution.pointwise_conv2

                # Absorb each Conformer feed-forward pre-LayerNorm into its expand Linear (net[0] -> net[1]).
                absorb_layer_norm_affine(encoder_layer.ffn1.net[0], encoder_layer.ffn1.net[1])
                absorb_layer_norm_affine(encoder_layer.ffn2.net[0], encoder_layer.ffn2.net[1])
            self.model.encoder.register_buffer('pos_weight', torch.cat(pos_weights, dim=0).transpose(0, 1).contiguous())
            _share_affine_less_layer_norm(self.model.encoder, self.hidden_size)

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
        real_part_sq, imag_part_sq = spectrum_square.split(
            self.nfft_stft // 2 + 1, dim=1)
        mel_features = torch.matmul(self.fbank, real_part_sq + imag_part_sq).transpose(1, 2).clamp(min=self.log_floor).log()
        mel_features = (mel_features - self.cmvn_means) * self.cmvn_vars
        features_len = torch._shape_as_tensor(mel_features)[1].unsqueeze(0)
        enc_outputs, _, valid_lengths, _ = self.model.encoder(mel_features, features_len)
        enc_outputs = _OnnxSlice.apply(
            enc_outputs, self.slice_zero, valid_lengths,
            self.slice_one, self.slice_one)
        for idx, decoder_layer in enumerate(self.model.decoder.layer_stack):
            cross_kv = _OnnxReshape.apply(
                decoder_layer.cross_attn.kv(enc_outputs).to(KV_DTYPE),
                self.cross_kv_shape).transpose(0, 1)
            k, v = _OnnxSplitTwoDim0.apply(cross_kv, self.cross_kv_split_sizes)
            self.save_en_keys[idx] = k.transpose(1, 2)      # f16 cross-attention key   (num_heads, head_dim, T)
            self.save_en_values[idx] = v                    # f16 cross-attention value (num_heads, T, head_dim)
        return *self.save_en_keys, *self.save_en_values


class FIRE_RED_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Whisper/Qwen Decoder_Embed) so the int
    # token ids never enter the float-only decode graph. The d_model**0.5 scale is folded into the embedding
    # output here (the absolute position embedding itself is added inside the decoder main graph). Keeping
    # the table pristine preserves its exact tie to tgt_word_prj for one-source block quantization.
    def __init__(self, fire_red):
        super(FIRE_RED_DECODER_EMBED, self).__init__()
        self.embed = fire_red.decoder.tgt_word_emb
        self.scale = float(fire_red.decoder.scale)

    def forward(self, input_ids):
        return self.embed(input_ids) * self.scale


class FIRE_RED_PREFILL(torch.nn.Module):
    # Prefill-phase position-embedding + causal-mask generator (mirrors Whisper/Qwen Prefill).
    # Consumes the int lengths and emits float position embedding + float attention mask so the decoder
    # main graph stays integer-free.
    def __init__(self, fire_red, max_seq_len):
        super(FIRE_RED_PREFILL, self).__init__()
        self.emit_fp32_mask = USE_FP16_KV and COMPUTE_IN_F32
        self.register_buffer('position_weight', fire_red.decoder.positional_encoding.pe[:, :max_seq_len].to(KV_DTYPE))
        attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.register_buffer('attention_mask', attention_mask.to(KV_DTYPE))
        self.register_buffer('slice_one', torch.tensor([1], dtype=torch.int64))
        self.register_buffer('mask_starts', torch.tensor([0, 0], dtype=torch.int64))
        self.register_buffer('mask_axes', torch.tensor([1, 2], dtype=torch.int64))
        self.register_buffer('mask_steps', torch.tensor([1, 1], dtype=torch.int64))

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = _OnnxSlice.apply(
            self.position_weight, history_len, kv_seq_len,
            self.slice_one, self.slice_one).float()
        mask_ends = torch.cat((ids_len, kv_seq_len))
        attention_mask = _OnnxSlice.apply(
            self.attention_mask, self.mask_starts, mask_ends,
            self.mask_axes, self.mask_steps)
        if self.emit_fp32_mask:
            attention_mask = attention_mask.float()
        return position_embed, attention_mask, kv_seq_len


class FIRE_RED_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Whisper/Qwen Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, fire_red, max_seq_len):
        super(FIRE_RED_DECODE, self).__init__()
        self.register_buffer('position_weight', fire_red.decoder.positional_encoding.pe[:, :max_seq_len].to(KV_DTYPE))
        self.register_buffer('one', torch.ones(1, dtype=torch.int64))

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + self.one
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class FIRE_RED_DECODER(torch.nn.Module):
    def __init__(self, fire_red, num_layers_de):
        super(FIRE_RED_DECODER, self).__init__()
        self.model = fire_red
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = not USE_FP16_KV or COMPUTE_IN_F32
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
        self.register_buffer(
            'self_qkv_shape',
            torch.tensor([0, -1, 3 * self.num_heads, self.head_dim], dtype=torch.int64),
        )
        self.register_buffer(
            'cross_q_shape',
            torch.tensor([0, -1, self.cross_num_heads, self.cross_head_dim], dtype=torch.int64),
        )
        self.register_buffer(
            'context_shape',
            torch.tensor([0, -1, self.hidden_size], dtype=torch.int64),
        )
        self.register_buffer(
            'self_qkv_split_sizes',
            torch.full((3,), self.num_heads, dtype=torch.int64),
        )
        self.register_buffer('gelu_sqrt_two', torch.tensor(2.0, dtype=torch.float32).sqrt())
        self.register_buffer('gelu_one', torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer('gelu_half', torch.tensor(0.5, dtype=torch.float32))
        self.register_buffer('last_token_index', torch.tensor(-1, dtype=torch.int64))
        # Keep the tied vocabulary projection pristine. Embed applies d_model**0.5 after lookup, while the
        # final LayerNorm keeps its affine parameters live before this projection.
        self.model.decoder.tgt_word_prj.weight = torch.nn.Parameter(self.model.decoder.tgt_word_prj.weight.detach().clone())
        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

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
            _share_affine_less_layer_norm(self.model.decoder, self.hidden_size)

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
        """Permute w_1 outputs and w_2 inputs by one exact intermediate-channel order."""
        with torch.no_grad():
            for decoder_layer in self.model.decoder.layer_stack:
                w_1 = decoder_layer.mlp.w_1
                w_2 = decoder_layer.mlp.w_2
                permutation = torch.argsort(self._channel_stat(w_2.weight, key, 0))
                w_1.weight.copy_(w_1.weight[permutation])
                if w_1.bias is not None:
                    w_1.bias.copy_(w_1.bias[permutation])
                w_2.weight.copy_(w_2.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute each self-attention V head and matching fc input columns exactly."""
        num_heads = self.num_heads
        head_dim = self.head_dim
        hidden_size = self.hidden_size
        with torch.no_grad():
            for decoder_layer in self.model.decoder.layer_stack:
                attention = decoder_layer.self_attn
                output_weight = attention.fc.weight
                output_by_head = output_weight.view(
                    output_weight.shape[0], num_heads, head_dim
                )
                permutations = [
                    torch.argsort(
                        self._channel_stat(output_by_head[:, head, :], key, 0)
                    )
                    for head in range(num_heads)
                ]

                reordered_output = output_by_head.clone()
                for head, permutation in enumerate(permutations):
                    reordered_output[:, head, :] = output_by_head[
                        :, head, permutation
                    ]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                qkv = attention.qkv
                value_weight = qkv.weight[2 * hidden_size:].view(
                    num_heads, head_dim, qkv.in_features
                )
                reordered_value_weight = value_weight.clone()
                for head, permutation in enumerate(permutations):
                    reordered_value_weight[head] = value_weight[head, permutation]
                qkv.weight[2 * hidden_size:].copy_(
                    reordered_value_weight.reshape(hidden_size, qkv.in_features)
                )

                if qkv.bias is not None:
                    value_bias = qkv.bias[2 * hidden_size:].view(num_heads, head_dim)
                    reordered_value_bias = value_bias.clone()
                    for head, permutation in enumerate(permutations):
                        reordered_value_bias[head] = value_bias[head, permutation]
                    qkv.bias[2 * hidden_size:].copy_(
                        reordered_value_bias.reshape(hidden_size)
                    )


    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        # Prefill emits the causal mask in attention-compute dtype, so every layer
        # shares it directly without an additional precision-boundary Cast.
        attn_mask = attention_mask
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
            qkv = _OnnxReshape.apply(qkv, self.self_qkv_shape).transpose(1, 2)
            q, k, v = _OnnxSplitThreeDim1.apply(qkv, self.self_qkv_split_sizes)
            if self.compute_in_f32:
                k = k.to(KV_DTYPE)
                v = v.to(KV_DTYPE)
            k = torch.cat((all_inputs[idx], k.transpose(-1, -2)), dim=-1)            # f16 key cache   (batch, num_heads, head_dim, kv_seq_len)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v), dim=-2)        # f16 value cache (batch, num_heads, kv_seq_len, head_dim)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            if self.compute_in_f32:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k.float()) + attn_mask, dim=-1), v.float()).transpose(1, 2)
            else:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attn_mask, dim=-1), v).transpose(1, 2).float()
            hidden_state_attn = _OnnxReshape.apply(hidden_state_attn, self.context_shape)
            hidden_state_attn = decoder_layer.self_attn.fc(hidden_state_attn)
            hidden_state_attn += hidden_states
            # Cross-attention against the f16 encoder cross-KV cache. OFF: downcast Q to f16 and run in f16 on
            # the f16 cross cache, context back to f32. ON: keep Q in f32 and upcast the f16 cross K/V to f32 at
            # the matmul use points (the cross cache is produced f16 by the encoder; its I/O dtype is unchanged).
            q = decoder_layer.cross_attn.w_qs(decoder_layer.cross_attn_norm(hidden_state_attn))
            q = _OnnxReshape.apply(q, self.cross_q_shape).transpose(1, 2)
            if self.compute_in_f32:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = _OnnxReshape.apply(hidden_state_cross.transpose(1, 2), self.context_shape)
                hidden_state_cross = decoder_layer.cross_attn.fc(hidden_state_cross)
            else:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = _OnnxReshape.apply(hidden_state_cross.transpose(1, 2).float(), self.context_shape)
                hidden_state_cross = decoder_layer.cross_attn.fc(hidden_state_cross)
            hidden_state_cross += hidden_state_attn
            mlp_hidden = decoder_layer.mlp.w_1(decoder_layer.mlp_norm(hidden_state_cross))
            mlp_hidden = _OnnxExactGelu.apply(
                mlp_hidden, self.gelu_sqrt_two, self.gelu_one, self.gelu_half)
            hidden_states = hidden_state_cross + decoder_layer.mlp.w_2(mlp_hidden)
        hidden_states = self.model.decoder.layer_norm_out(
            _OnnxGatherLastToken.apply(hidden_states, self.last_token_index))
        logits = self.model.decoder.tgt_word_prj(hidden_states)
        return *self.save_de_keys, *self.save_de_values, logits


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


def write_metadata_carrier(onnx_path, metadata):
    Shared_Merged.replace_onnx_metadata(
        onnx_path,
        {str(key): str(value) for key, value in metadata.items()},
    )


def build_token_metadata(dict_path):
    """Derive the immutable decoder token contract from FireRed's dictionary."""
    token_to_id = {}
    with open(dict_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            pieces = line.strip().split()
            if len(pieces) >= 2:
                token, token_id = pieces[0], int(pieces[1])
            elif len(pieces) == 1:
                token, token_id = pieces[0], line_number
            else:
                token, token_id = " ", line_number
            token_to_id[token] = token_id
    role_tokens = {
        "blank": "<blank>",
        "unknown": "<unk>",
        "pad": "<pad>",
        "sos": "<sos>",
        "stop": "<eos>",
    }
    special_token_ids = {
        role: token_to_id[token] for role, token in role_tokens.items()
    }
    supported_languages = {
        "zh": {
            "name": "Chinese",
            "aliases": ["chinese", "mandarin", "cn", "中文", "普通话"],
            "prompt_token_ids": [],
        },
        "en": {
            "name": "English",
            "aliases": ["english", "eng"],
            "prompt_token_ids": [],
        },
    }
    return len(token_to_id), special_token_ids, supported_languages


print('\nStart to export the Encoder part.\n')
with torch.inference_mode():
    if 'aed' in model_path or 'AED' in model_path or 'Aed' in model_path:
        feat_extractor = ASRFeatExtractor(model_path + "/cmvn.ark")
        model = load_fireredasr_aed_model(model_path + "/model.pth.tar").float()
        model.eval()
        HIDDEN_SIZE = model.encoder.odim
        NUM_HEAD_DE = model.decoder.layer_stack._modules['0'].self_attn.n_head
        NUM_LAYER_DE = model.decoder.n_layers
        HEAD_DIM_DE = model.decoder.layer_stack._modules['0'].self_attn.d_k
        CROSS_NUM_HEAD_DE = model.decoder.layer_stack._modules['0'].cross_attn.n_head
        CROSS_HEAD_DIM_DE = model.decoder.layer_stack._modules['0'].cross_attn.d_k
        VOCAB_SIZE = model.decoder.tgt_word_prj.out_features
        (
            DICTIONARY_VOCAB_SIZE,
            SPECIAL_TOKEN_IDS,
            SUPPORTED_LANGUAGES,
        ) = build_token_metadata(os.path.join(model_path, "dict.txt"))

        # All attention weight fusion + scale folding now happens inside each module's _fuse_weights(); no external
        # pre-scaling loops are needed here. ORDER MATTERS: build the encoder first (it fuses the decoder cross-attn
        # k/v into `kv` and deletes w_ks/w_vs), then the decoder main (decouples tgt_word_prj before the embedding is
        # scaled, fuses self-attn qkv, folds the cross-attn q scale), then the embedding graph (scales the embedding).
        fire_red_encoder = FIRE_RED_ENCODER(
            model,
            feat_extractor,
            _MODEL_NFFT_STFT,
            _MODEL_NUM_MELS,
            _MODEL_SAMPLE_RATE,
            _MODEL_PRE_EMPHASIS,
            NUM_LAYER_DE,
        )

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
        save_encoder_key = torch.zeros((CROSS_NUM_HEAD_DE, CROSS_HEAD_DIM_DE, STFT_SIGNAL_LENGTH // 2 + 1), dtype=KV_DTYPE)
        save_encoder_value = torch.zeros((CROSS_NUM_HEAD_DE, STFT_SIGNAL_LENGTH // 2 + 1, CROSS_HEAD_DIM_DE), dtype=KV_DTYPE)
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

    # ── Decode-strategy split graphs ──
    # Trace-only values establish ranks; every control remains a live ONNX input.
    logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    save_id = torch.zeros((1, 10), dtype=torch.int32)
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

    # ── Top-K / Top-P Sampling ──
    sampling_temperature = torch.tensor([0.8], dtype=torch.float32)
    sampling_top_k = torch.tensor([50], dtype=torch.int32)
    sampling_top_p = torch.tensor([0.95], dtype=torch.float32)
    sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    sampling_previous_ids = torch.zeros((1, 10), dtype=torch.int32)
    torch.onnx.export(
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            sampling_temperature,
            sampling_top_k,
            sampling_top_p,
            sampling_repetition_penalty,
            sampling_previous_ids,
        ),
        onnx_model_Sampling,
        input_names=[
            'logits',
            'temperature',
            'top_k',
            'top_p',
            'repetition_penalty',
            'previous_ids',
        ],
        output_names=['sampled_id', 'save_id_out'],
        dynamic_axes={
            'previous_ids': {1: 'history_len'},
            'save_id_out': {1: 'history_len_out'},
        },
        do_constant_folding=True,
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
    del sampling_previous_ids
    gc.collect()

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
            "audio_pcm_scale": 1,
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": _MODEL_SAMPLE_RATE,
            "special_token_ids": SPECIAL_TOKEN_IDS,
            "supported_languages": SUPPORTED_LANGUAGES,
        },
    )

    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    gc.collect()


if project_path in sys.path:
    sys.path.remove(project_path)

# ── Convert the ephemeral split graphs into six merged strategy graphs backed by one shared blob. ──
if ONNX_DIR.exists():
    shutil.rmtree(ONNX_DIR)
ONNX_DIR.mkdir(parents=True)
print("\n[SharedMerged] Building FireRedASR strategy graphs + shared initializer bundle ...")
_bundle = Shared_Merged.build_shared_merged_bundle(
    _raw_onnx_dir,
    out_folder=ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
)
Shared_Merged.copy_runtime_standalones(
    _raw_onnx_dir,
    ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
)
for _name, _path in _bundle["graphs"].items():
    print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
write_metadata_carrier(
    ONNX_DIR / MODEL_FILE_NAMES["metadata"], onnx_metadata
)
print(
    f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
    f"({Path(_bundle['shared_data']).stat().st_size} bytes)"
)
# ── Copy tokenizer assets rather than moving them, so both source and final folders remain intact. ──
for _asset in ("dict.txt", "train_bpe1000.model"):
    _src = os.path.join(model_path, _asset)
    _dst = os.path.join(onnx_dir, _asset)
    shutil.copy2(_src, _dst)
    print(f"[Tokenizer] Copied {_asset} -> {onnx_dir}")

# No raw split graph survives the export process.
_raw_onnx_temp.cleanup()

print('\nExport done!\n')
subprocess.run(
    [
        sys.executable,
        str(SCRIPT_DIR / "Inference_FireRedASR_AED_ONNX.py"),
        "--onnx-folder",
        str(ONNX_DIR),
    ],
    cwd=str(SCRIPT_DIR),
    check=True,
)
