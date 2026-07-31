import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from funasr import AutoModel
from torch.onnx.operators import reshape_from_tensor_shape
from Rewrite_Paraformer_Streaming_ONNX import rewrite_folder


# =================================================================================================
# User configuration: these are the only values intended to be edited for a deployment export.
# =================================================================================================
model_path             = str(Path.home() / "Downloads" / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online")  # Downloaded Paraformer-Chinese-Online-Streaming checkpoint folder.
MAX_CONTINUE_STREAMING = 502                                # 502 = Max 30 seconds streaming audio input. # 1003 = Max 60 seconds streaming audio input.
INPUT_AUDIO_LENGTH     = 8000                               # The fixed input audio segment length, edit it carefully.
INPUT_AUDIO_DTYPE      = "F32"                              # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
USE_FP16_KV            = True                               # Keep recurrent K/V storage in FP16 for normal deployment exports.
PREVENT_F16_OVERFLOW   = False                              # Set True before export if the front-end will be converted to fp16.
COMPUTE_IN_F32         = False                              # FP16-cache compute precision. False keeps attention in F16; True upcasts at attention use points.
CACHE_DTYPE            = torch.float16 if USE_FP16_KV else torch.float32


# Export implementation policy; these values control graph construction and are not deployment settings.
DYNAMIC_AXES           = True                               # The dynamic_axes setting. Do not turn off for the Paraformer Streaming model.
OPSET                  = 20                                 # <= 20


# Fixed checkpoint constants and metadata defaults; these are not user tunables.
WINDOW_TYPE            = 'hamming'                          # Kaldi fbank window used by the trained online model.
N_MELS                 = 80                                 # Number of model Mel bands.
NFFT_STFT              = 512                                # Kaldi rounds the 25 ms frame length up to 512.
WINDOW_LENGTH          = 400                                # 25 ms at the fixed 16-kHz sample rate.
HOP_LENGTH             = 160                                # 10 ms at the fixed 16-kHz sample rate.
SAMPLE_RATE            = 16000                              # Fixed checkpoint sample rate.
LFR_M                  = 7                                  # Fixed low-frame-rate stacking width.
LFR_N                  = 6                                  # Fixed low-frame-rate stacking stride.
PRE_EMPHASIZE          = 0.97                               # Fixed Kaldi per-frame pre-emphasis coefficient.
LOOK_BACK_ENCODER      = 4                                  # Fixed encoder history multiplier.
LOOK_BACK_DECODER      = 1                                  # Fixed decoder history multiplier.


# Fixed artifact layout. Raw torch.onnx.export graphs are build intermediates in an auto-cleaned
# system temporary directory; the workspace retains only the rewritten runtime bundle.
script_folder          = Path(__file__).resolve().parent
onnx_folder            = script_folder / "Paraformer_ONNX"
_raw_onnx_temp         = tempfile.TemporaryDirectory(prefix="paraformer_streaming_export_")
raw_onnx_folder        = Path(_raw_onnx_temp.name)
raw_onnx_model_Metadata = str(raw_onnx_folder / "ASR_Metadata.onnx")
raw_onnx_model_Encoder = str(raw_onnx_folder / "Paraformer_Streaming_Encoder.onnx")
raw_onnx_model_Decoder = str(raw_onnx_folder / "Paraformer_Streaming_Decoder.onnx")
raw_vocab_path         = str(raw_onnx_folder / "Vocab_Paraformer.txt")
onnx_folder.mkdir(parents=True, exist_ok=True)


# Derived dummy-input and cache dimensions. These follow from the user-selected chunk length and
# the fixed checkpoint contract; they are implementation details rather than independent settings.
LFR_M_FACTOR = (LFR_M - 1) // 2                             # Number of left-context frames replicated before LFR stacking.
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1   # The Kaldi snip_edges=True fbank frame count for one chunk.
LFR_LENGTH = (LFR_M_FACTOR + STFT_SIGNAL_LENGTH) // LFR_N + 1                  # Must match the dynamic _len computed inside forward().


LOOK_BACK_B = LFR_LENGTH                                    # Current encoder chunk width after LFR stacking.
LOOK_BACK_C = LOOK_BACK_B // 2                              # Fixed half-chunk overlap retained between calls.
LOOK_BACK_A = 0                                             # No additional left feature context for this checkpoint.


_SPECIAL_TOKEN_CANDIDATES = {
    "blank": ("<blank>", "<blk>", "<eps>"),
    "eos": ("</s>", "<eos>"),
    "unknown": ("<unk>", "<unknown>", "[UNK]"),
    "pad": ("<pad>", "[PAD]"),
    "bos": ("<s>", "<bos>", "<sos>"),
}


def _find_token_id(token_list, role, *, required):
    candidates = _SPECIAL_TOKEN_CANDIDATES[role]
    matches = [
        (token_id, token)
        for token_id, token in enumerate(token_list)
        if token in candidates
    ]
    if not matches:
        if required:
            raise ValueError(
                f"Tokenizer is missing required {role} token; expected exactly "
                f"one of {candidates}."
            )
        return None
    if len(matches) != 1:
        raise ValueError(
            f"Tokenizer has ambiguous {role} tokens {matches}; expected exactly "
            f"one of {candidates}."
        )
    return matches[0][0]


def build_tokenizer_metadata(token_list, model):
    token_list = list(token_list)
    if not token_list or any(not isinstance(token, str) for token in token_list):
        raise TypeError("Tokenizer token_list must be a non-empty string list.")
    declared_vocab_size = int(model.vocab_size)
    projection_vocab_size = int(model.decoder.output_layer.out_features)
    vocab_size = len(token_list)
    if vocab_size != declared_vocab_size or vocab_size != projection_vocab_size:
        raise ValueError(
            "Tokenizer/model vocabulary mismatch: "
            f"token_list={vocab_size}, model.vocab_size={declared_vocab_size}, "
            f"decoder.output_layer={projection_vocab_size}."
        )

    blank_id = _find_token_id(token_list, "blank", required=True)
    eos_id = _find_token_id(token_list, "eos", required=True)
    special_token_ids = {
        "blank": blank_id,
        "eos": eos_id,
        "stop": [eos_id],
    }
    for role in ("unknown", "pad", "bos"):
        token_id = _find_token_id(token_list, role, required=False)
        if token_id is not None:
            special_token_ids[role] = token_id

    for role, attr_name in (("blank", "blank_id"), ("bos", "sos"), ("eos", "eos")):
        model_token_id = getattr(model, attr_name, None)
        if model_token_id is not None and role in special_token_ids:
            if int(model_token_id) != special_token_ids[role]:
                raise ValueError(
                    f"Tokenizer {role} ID {special_token_ids[role]} disagrees with "
                    f"model.{attr_name}={model_token_id}."
                )

    supported_languages = {
        "zh": {
            "name": "Chinese",
            "aliases": ["Chinese", "Mandarin", "zh-CN", "中文"],
            "prompt_token_ids": [],
            "decode_mode": "zh",
        }
    }
    return special_token_ids, supported_languages


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


def write_onnx_metadata(onnx_path, metadata, *, replace=False):
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    if replace:
        del model.metadata_props[:]
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, onnx_path)


def write_metadata_carrier(onnx_path, metadata):
    """Exact-replace the ASR metadata carrier."""
    carrier_path = Path(onnx_path)
    expected = {str(key): str(value) for key, value in metadata.items()}
    write_onnx_metadata(carrier_path, expected, replace=True)


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


def absorb_layer_norm_affine(norm, linear):
    """Fold a LayerNorm's affine (weight, bias) into the linear that consumes its output.

        new_bias   = linear.bias + linear.weight @ norm.bias
        new_weight = linear.weight * norm.weight        (scales the linear's input columns)

    The learned affine is replaced by a shared identity scale/shift, so at runtime the learned
    scale/shift live inside the following GEMM weight/bias while ONNX can keep one fused
    LayerNormalization node. The linear must be the sole consumer of the normalised tensor.
    """
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype))
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
        norm.weight.data.fill_(1.0)
        norm.bias.data.zero_()
    norm._onnx_affine_folded = True


def share_folded_layer_norm_affines(module):
    """Share identity LayerNorm parameters so the legacy exporter emits two initializers per
    normalized width instead of recreating full one/zero Constant tensors at every call site.
    """
    shared_affines = {}
    for norm in module.modules():
        if not isinstance(norm, torch.nn.LayerNorm) or not getattr(norm, "_onnx_affine_folded", False):
            continue
        key = (tuple(norm.normalized_shape), norm.weight.dtype, norm.weight.device)
        if key not in shared_affines:
            shared_affines[key] = (norm.weight, norm.bias)
        else:
            norm.weight, norm.bias = shared_affines[key]


def fold_symmetric_pad_into_conv(pad_module, conv):
    """Fold a zero-valued symmetric ConstantPad1d into the following Conv1d's own padding.

    nn.Conv1d already performs zero padding, so ConstantPad1d((p, p), 0.0) feeding a padding=0
    convolution is exactly a convolution with padding=p. Folding removes the standalone Pad node
    (and its two zero buffers) from the exported graph while keeping the result bit-identical. Only
    symmetric, zero-valued pads are expressible by Conv1d's scalar padding, so anything else is
    rejected rather than silently mis-folded.
    """
    left, right = pad_module.padding
    if float(pad_module.value) != 0.0 or left != right:
        raise ValueError(f"Cannot fold pad {pad_module.padding!r} (value={pad_module.value}) into Conv1d.")
    conv.padding = (int(left),)


def replace_gelu_with_tanh(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.GELU):
            setattr(module, name, torch.nn.GELU(approximate="tanh"))
        else:
            replace_gelu_with_tanh(child)


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
    """Re-implements the FunASR WavFrontendOnline Kaldi fbank (kaldi.fbank) front-end as a single
    conv1d so it can be exported to ONNX. The conv kernel folds the per-frame DC-offset removal,
    pre-emphasis and analysis window into the windowed DFT basis; snip_edges=True framing follows
    from the strided conv (no centre padding). This matches the original waveform * (1 << 15) ->
    kaldi.fbank(num_mel_bins, frame_length=25, frame_shift=10, window_type='hamming') path.
    """

    def __init__(self, n_fft, win_length, hop_len, n_mels, sample_rate, window_type, pre_emphasis):
        super().__init__()
        self.hop_len = hop_len
        self.n_freqs = n_fft // 2 + 1
        self.register_buffer("stft_kernel", create_kaldi_stft_kernel(n_fft, win_length, window_type, pre_emphasis))
        mel_bins, _ = kaldi.get_mel_banks(n_mels, n_fft, sample_rate, 20.0, 0.0, 100.0, -500.0, 1.0)
        mel_bins = torch.nn.functional.pad(mel_bins, (0, 1), mode="constant", value=0.0)
        self.register_buffer("mel_bins", mel_bins.unsqueeze(0).to(torch.float32))
        power_scale = 0.01 if PREVENT_F16_OVERFLOW else 1.0
        self.register_buffer("power_scale", torch.tensor([power_scale], dtype=torch.float32), persistent=False)
        self.register_buffer("epsilon", torch.tensor(torch.finfo(torch.float32).eps * (power_scale ** 2), dtype=torch.float32))
        self.register_buffer("log_power_scale", torch.tensor(np.log(power_scale ** 2), dtype=torch.float32), persistent=False)

    def forward(self, audio):
        stft = torch.nn.functional.conv1d(audio.float(), self.stft_kernel, stride=self.hop_len)
        if PREVENT_F16_OVERFLOW:
            stft = stft * self.power_scale                                                # one scale over the 2*n_freqs channels (== scaling real / imag separately)
        real_power, imag_power = torch.split(stft * stft, self.n_freqs, dim=1)            # one square over the 2*n_freqs channels, then split (== real^2 / imag^2)
        power = real_power + imag_power
        mel = torch.matmul(self.mel_bins, power)
        log_mel = torch.maximum(mel, self.epsilon).log()
        if PREVENT_F16_OVERFLOW:
            log_mel = log_mel - self.log_power_scale
        return log_mel.transpose(1, 2)


class PARAFORMER_ENCODER(torch.nn.Module):
    def __init__(self, paraformer, fbank_model, stft_signal_len, lfr_m, lfr_n, lfr_len, cmvn_means, cmvn_vars, cif_hidden_size, fsmn_hidden_size, feature_size, look_back_A, look_back_B, look_back_C, look_back_en, max_continue_streaming):
        super(PARAFORMER_ENCODER, self).__init__()
        self.look_back_A = look_back_A
        self.look_back_B = look_back_B
        self.look_back_C = look_back_C
        self.look_back_en = -(look_back_en * look_back_B) - self.look_back_C
        self.encoder = paraformer.encoder
        self.predictor = paraformer.predictor
        self.fbank_model = fbank_model
        self.register_buffer("cmvn_means", cmvn_means)
        self.register_buffer("cmvn_vars", cmvn_vars)
        self.T_lfr = lfr_len
        self.cif_hidden_size = cif_hidden_size
        self.lfr_feature_size = feature_size                                                # static LFR-stacked feature width (n_mels * lfr_m)
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        indices = (indices - ((lfr_m - 1) // 2)).clamp(min=0, max=stft_signal_len - 1).reshape(-1)
        self.register_buffer("indices_mel", indices.to(torch.int32))                       # direct int32 LFR gather indices; clamping reproduces left-frame replication
        self.register_buffer("lfr_shape", torch.tensor([1, self.T_lfr, self.lfr_feature_size], dtype=torch.int64))
        self.register_buffer("cif_one", torch.tensor(1.0, dtype=torch.float32))
        self.total_encoders = list(self.encoder.encoders0) + list(self.encoder.encoders)
        self.cache_layer_num_en = len(self.total_encoders)
        self.cache_dtype = CACHE_DTYPE
        self.compute_in_f32 = self.cache_dtype == torch.float32 or COMPUTE_IN_F32
        self.save_keys_en = [None] * self.cache_layer_num_en
        self.save_values_en = [None] * self.cache_layer_num_en
        positions = torch.arange(1, max_continue_streaming, dtype=torch.int32).unsqueeze(0)
        self.register_buffer("position_encoding", self.encoder.embed.encode(positions, feature_size).to(self.cache_dtype))

        first_attn = self.total_encoders[0].self_attn
        self.register_buffer("q_shape", torch.tensor([-1, first_attn.h, first_attn.d_k], dtype=torch.int64))
        self.register_buffer("kv_shape", torch.tensor([-1, 2 * first_attn.h, first_attn.d_k], dtype=torch.int64))
        self.register_buffer("qkv_shape", torch.tensor([-1, 3 * first_attn.h, first_attn.d_k], dtype=torch.int64))
        self.register_buffer("context_shape", torch.tensor([1, -1, self.cif_hidden_size], dtype=torch.int64))
        for encoder_layer in self.total_encoders:
            attn = encoder_layer.self_attn
            if attn.h != first_attn.h or attn.d_k != first_attn.d_k:
                raise ValueError("All streaming encoder layers must use one shared head layout.")

        # Fold the attention scale (1 / sqrt(d_k)) into the q/k projection weights so the inlined
        # attention can use a plain q @ k matmul, then absorb every LayerNorm affine into the linear
        # that consumes it, and fold each symmetric FSMN / CIF zero-pad into its following Conv1d.
        # All folds are exact in float32 and keep the fused LayerNormalization / Conv ops; the scale
        # and affine folds commute because one scales the linear's output rows and the other its
        # input columns.
        replace_gelu_with_tanh(self.encoder)
        replace_gelu_with_tanh(self.predictor)
        factor = float(self.encoder.encoders._modules["0"].self_attn.d_k ** -0.25)
        for encoder_layer in self.total_encoders:
            attn = encoder_layer.self_attn
            attn.linear_q_k_v.weight.data[:-self.cif_hidden_size] *= factor
            attn.linear_q_k_v.bias.data[:-self.cif_hidden_size] *= factor
            absorb_layer_norm_affine(encoder_layer.norm1, attn.linear_q_k_v)
            absorb_layer_norm_affine(encoder_layer.norm2, encoder_layer.feed_forward.w_1)
            fold_symmetric_pad_into_conv(attn.pad_fn, attn.fsmn_block)                       # FSMN symmetric zero-pad -> Conv1d padding
        fold_symmetric_pad_into_conv(self.predictor.pad, self.predictor.cif_conv1d)          # CIF symmetric zero-pad -> Conv1d padding
        share_folded_layer_norm_affines(self)

    def forward(self, *all_inputs):
        previous_mel_features = all_inputs[-5]
        cif_hidden = all_inputs[-4]
        cif_alphas = all_inputs[-3]
        start_idx = all_inputs[-2]
        audio = all_inputs[-1]
        mel_features = self.fbank_model(audio)
        mel_features = torch.index_select(mel_features, 1, self.indices_mel)
        mel_features = reshape_from_tensor_shape(mel_features, self.lfr_shape)                # fixed chunk -> one direct Gather and one initializer-backed Reshape
        mel_features = (mel_features + self.cmvn_means) * self.cmvn_vars
        end_idx = start_idx + self.T_lfr
        mel_features += self.position_encoding[:, start_idx:end_idx]
        x = torch.cat([previous_mel_features, mel_features], dim=1)
        previous_mel_features = x[:, -(self.look_back_A + self.look_back_C):]
        for layer_idx, encoder_layer in enumerate(self.total_encoders):
            attn = encoder_layer.self_attn
            if layer_idx > 0:
                residual = x
            qkv = attn.linear_q_k_v(encoder_layer.norm1(x))
            _, v = torch.split(qkv, [2 * self.cif_hidden_size, self.cif_hidden_size], dim=-1)  # retain the f32 V branch used by the FSMN
            if self.compute_in_f32:
                q = reshape_from_tensor_shape(qkv[:, :, :self.cif_hidden_size], self.q_shape).transpose(0, 1)
                kv = reshape_from_tensor_shape(qkv[:, :, self.cif_hidden_size:].to(self.cache_dtype), self.kv_shape).transpose(0, 1)
                k_time, v_h = torch.split(kv, attn.h, dim=0)
                k = k_time.transpose(1, 2)
            else:
                qkv_heads = reshape_from_tensor_shape(qkv, self.qkv_shape).transpose(0, 1)    # one shared layout conversion for Q/K/V
                q, k_time, v_h = torch.split(qkv_heads, attn.h, dim=0)
                # Keep these casts after Split/Transpose. Combining them into one pre-layout Cast
                # changes ORT's f16 MatMul reduction path by up to one cache ULP on CPU.
                q = q.to(self.cache_dtype)
                k = k_time.transpose(1, 2).to(self.cache_dtype)
                v_h = v_h.to(self.cache_dtype)
            k = torch.cat([all_inputs[layer_idx], k], dim=2)
            v_h = torch.cat([all_inputs[layer_idx + self.cache_layer_num_en], v_h], dim=1)
            self.save_keys_en[layer_idx] = k[:, :, self.look_back_en:-self.look_back_C]
            self.save_values_en[layer_idx] = v_h[:, self.look_back_en:-self.look_back_C]
            v_fsmn = attn.fsmn_block(v.transpose(1, 2)).transpose(1, 2) + v                  # FSMN symmetric pad folded into the Conv1d
            if self.compute_in_f32:
                # f16 storage, f32 compute: upcast the f16 K/V cache to f32 at the matmul use points (Q stays f32).
                context = torch.matmul(torch.softmax(torch.matmul(q, k.float()), dim=-1), v_h.float()).transpose(0, 1)
            else:
                context = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v_h).transpose(0, 1).float()  # minimum-cast f16 attention, then one context upcast
            context = reshape_from_tensor_shape(context, self.context_shape)
            x = attn.linear_out(context) + v_fsmn
            if layer_idx > 0:
                x += residual
            ff = encoder_layer.feed_forward
            x += ff.w_2(ff.activation(ff.w_1(encoder_layer.norm2(x))))
        encoder_out = self.encoder.after_norm(x)
        output = torch.relu(self.predictor.cif_conv1d(encoder_out.transpose(1, 2))).transpose(1, 2)   # CIF symmetric pad folded into the Conv1d
        alphas = torch.sigmoid(self.predictor.cif_output(output)).squeeze()                            # relu(sigmoid()) == sigmoid() (sigmoid >= 0)
        list_frame = []
        save_condition = []
        condition_A = (cif_alphas < self.cif_one).float()
        condition_B = self.cif_one - condition_A                                      # exact 0/1 complement; avoids LogicalNot + a second Cast
        save_condition.append(condition_B)
        frames = cif_alphas * cif_hidden * condition_A + cif_hidden * condition_B
        list_frame.append(frames)
        cif_alphas -= condition_B
        frames = frames * condition_A + cif_alphas * cif_hidden * condition_B
        alpha_steps = torch.split(alphas[self.look_back_A:self.look_back_A + self.look_back_B], 1, dim=0)
        hidden_steps = torch.split(encoder_out[:, self.look_back_A:self.look_back_A + self.look_back_B], 1, dim=1)
        for alpha, hidden in zip(alpha_steps, hidden_steps):
            threshold = self.cif_one - cif_alphas
            condition_A = (alpha < threshold).float()
            condition_B = self.cif_one - condition_A
            save_condition.append(condition_B)
            frames = (frames + alpha * hidden) * condition_A + (frames + threshold * hidden) * condition_B
            list_frame.append(frames)
            cif_alphas = cif_alphas + alpha
            cif_alphas -= condition_B
            frames = frames * condition_A + cif_alphas * hidden * condition_B
        list_frame = torch.cat(list_frame, dim=1)
        cif_hidden = list_frame[:, [-1]] / cif_alphas
        list_frame = list_frame.index_select(1, torch.nonzero(torch.cat(save_condition, dim=0), as_tuple=True)[-1])
        list_frame_len = list_frame.shape[1]
        return *self.save_keys_en, *self.save_values_en, previous_mel_features, cif_hidden, cif_alphas, end_idx, encoder_out, list_frame, list_frame_len


class PARAFORMER_DECODER(torch.nn.Module):
    def __init__(self, paraformer, look_back_B, look_back_C, look_back_de, cif_hidden_size, cache_layer_num_de):
        super(PARAFORMER_DECODER, self).__init__()
        self.look_back_de = look_back_de * look_back_B
        self.decoder = paraformer.decoder
        self.cif_hidden_size = cif_hidden_size
        self.fsmn_history = self.decoder.decoders._modules["0"].self_attn.kernel_size - 1
        self.cache_layer_num_de = cache_layer_num_de
        self.cache_dtype = CACHE_DTYPE
        self.compute_in_f32 = self.cache_dtype == torch.float32 or COMPUTE_IN_F32
        self.cache_layer_num_de_2 = cache_layer_num_de + cache_layer_num_de
        self.save_fsmn_de = [None] * cache_layer_num_de
        self.save_keys_de = [None] * cache_layer_num_de
        self.save_values_de = [None] * cache_layer_num_de

        first_cross = self.decoder.decoders._modules["0"].src_attn
        self.register_buffer("q_shape", torch.tensor([-1, first_cross.h, first_cross.d_k], dtype=torch.int64))
        self.register_buffer("kv_shape", torch.tensor([-1, 2 * first_cross.h, first_cross.d_k], dtype=torch.int64))
        self.register_buffer("context_shape", torch.tensor([1, -1, self.cif_hidden_size], dtype=torch.int64))

        # Fold the cross-attention scale into the q / k projections, then absorb every LayerNorm
        # affine into the linear that consumes it (norm2 is left intact because the decoder reuses
        # its normalised output for both the FSMN branch and the residual add, so it has two
        # consumers and cannot be folded).
        replace_gelu_with_tanh(self.decoder)
        factor = float(self.decoder.decoders._modules["0"].src_attn.d_k ** -0.25)
        for decoder_layer in self.decoder.decoders:
            if decoder_layer.src_attn.h != first_cross.h or decoder_layer.src_attn.d_k != first_cross.d_k:
                raise ValueError("All streaming decoder layers must use one shared head layout.")
            decoder_layer.src_attn.linear_q.weight.data *= factor
            decoder_layer.src_attn.linear_q.bias.data *= factor
            decoder_layer.src_attn.linear_k_v.weight.data[:cif_hidden_size] *= factor
            decoder_layer.src_attn.linear_k_v.bias.data[:cif_hidden_size] *= factor
            absorb_layer_norm_affine(decoder_layer.norm1, decoder_layer.feed_forward.w_1)
            absorb_layer_norm_affine(decoder_layer.feed_forward.norm, decoder_layer.feed_forward.w_2)
            absorb_layer_norm_affine(decoder_layer.norm3, decoder_layer.src_attn.linear_q)
        for decoder_layer in self.decoder.decoders3:
            absorb_layer_norm_affine(decoder_layer.norm1, decoder_layer.feed_forward.w_1)
            absorb_layer_norm_affine(decoder_layer.feed_forward.norm, decoder_layer.feed_forward.w_2)
        absorb_layer_norm_affine(self.decoder.after_norm, self.decoder.output_layer)
        share_folded_layer_norm_affines(self)

    def forward(self, *all_inputs):
        encoder_out = all_inputs[-3]
        list_frame = all_inputs[-2]
        list_frame_len = all_inputs[-1]
        for layer_idx, decoder_layer in enumerate(self.decoder.decoders):
            ff = decoder_layer.feed_forward
            cross = decoder_layer.src_attn
            residual = list_frame
            list_frame = decoder_layer.norm1(list_frame)
            list_frame = ff.w_2(ff.norm(ff.activation(ff.w_1(list_frame))))
            list_frame = decoder_layer.norm2(list_frame)
            if self.fsmn_history:
                fsmn_history = all_inputs[layer_idx][:, :, -self.fsmn_history:]
            else:
                fsmn_history = all_inputs[layer_idx][:, :, :0]
            x = torch.cat((fsmn_history, list_frame.transpose(1, 2)), dim=-1)            # trim before Cat: same last history+current window, smaller temporary
            self.save_fsmn_de[layer_idx] = (
                x[:, :, -self.fsmn_history:]
                if self.fsmn_history
                else x[:, :, :0]
            )
            x = decoder_layer.self_attn.fsmn_block(x).transpose(1, 2)
            x += list_frame + residual
            residual = x
            q = reshape_from_tensor_shape(cross.linear_q(decoder_layer.norm3(x)), self.q_shape).transpose(0, 1)
            kv = reshape_from_tensor_shape(cross.linear_k_v(encoder_out).to(self.cache_dtype), self.kv_shape).transpose(0, 1)  # one cast/layout conversion for K/V
            k_time, v = torch.split(kv, cross.h, dim=0)
            k = k_time.transpose(1, 2)
            k = torch.cat([all_inputs[layer_idx + self.cache_layer_num_de], k], dim=2)
            v = torch.cat([all_inputs[layer_idx + self.cache_layer_num_de_2], v], dim=1)
            self.save_keys_de[layer_idx] = k[:, :, -self.look_back_de:]
            self.save_values_de[layer_idx] = v[:, -self.look_back_de:]
            if self.compute_in_f32:
                context = torch.matmul(torch.softmax(torch.matmul(q, k.float()), dim=-1), v.float()).transpose(0, 1)   # cast the f16 cache back to f32 for the matmuls
            else:
                # minimum-cast: downcast Q to f16 and run the attention in f16 on the f16 cache, then cast the context back to f32.
                context = torch.matmul(torch.softmax(torch.matmul(q.to(self.cache_dtype), k), dim=-1), v).transpose(0, 1).float()
            context = reshape_from_tensor_shape(context, self.context_shape)
            list_frame = residual + cross.linear_out(context)
        decoder_layer = self.decoder.decoders3[0]
        ff = decoder_layer.feed_forward
        x = ff.w_2(ff.norm(ff.activation(ff.w_1(decoder_layer.norm1(list_frame)))))
        x = self.decoder.output_layer(self.decoder.after_norm(x))
        max_logit_ids = torch.argmax(x, dim=-1, keepdim=False).int()                      # (1, list_frame_len) int32 token id per fired CIF frame
        num_id = list_frame_len.to(torch.int32).unsqueeze(0)                              # fixed shape [1]: reuse the authoritative token count instead of Shape -> Gather
        return *self.save_fsmn_de, *self.save_keys_de, *self.save_values_de, max_logit_ids, num_id


print('\nExport Encoder Part...\n')
with torch.inference_mode():
    custom_fbank = KaldiFbank(NFFT_STFT, WINDOW_LENGTH, HOP_LENGTH, N_MELS, SAMPLE_RATE, WINDOW_TYPE, PRE_EMPHASIZE).eval()  # Kaldi-faithful fbank front-end (matches FunASR WavFrontendOnline).
    model = AutoModel(
        model=model_path,
        disable_update=True,
        device="cpu"
    )
    encoder_output_size_factor = (model.model.encoder.output_size()) ** 0.5
    CMVN_MEANS = model.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    tokenizer = model.kwargs['tokenizer']
    token_list = list(tokenizer.token_list)
    special_token_ids, supported_languages = build_tokenizer_metadata(
        token_list,
        model.model,
    )
    # Save to text file
    with open(raw_vocab_path, 'w', encoding='utf-8') as f:
        for token in token_list:
            f.write(f'{token}\n')
          
    model = model.model.eval()
    NUM_LAYER_EN = len(model.encoder.encoders0) + len(model.encoder.encoders)
    NUM_LAYER_DE = len(model.decoder.decoders)
    FEATURE_SIZE = model.encoder.encoders0._modules["0"].in_size
    CIF_HIDDEN_SIZE = model.encoder.encoders0._modules["0"].size
    FSMN_HIDDEN_SIZE = model.decoder.decoders._modules["0"].size
    NUM_HEAD_EN = model.encoder.encoders0._modules["0"].self_attn.h
    HEAD_DIM_EN = model.encoder.encoders0._modules["0"].self_attn.d_k
    NUM_HEAD_DE = model.decoder.decoders._modules["0"].src_attn.h
    HEAD_DIM_DE = model.decoder.decoders._modules["0"].src_attn.d_k
    FSMN_DE_PAD = model.decoder.decoders._modules["0"].self_attn.pad_fn.padding[0]

    key_en = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, 0), dtype=CACHE_DTYPE)
    value_en = torch.zeros((NUM_HEAD_EN, 0, HEAD_DIM_EN), dtype=CACHE_DTYPE)
    previous_mel_features = torch.zeros((1, LOOK_BACK_A + LOOK_BACK_C, FEATURE_SIZE), dtype=torch.float32)
    cif_hidden = torch.zeros((1, 1, CIF_HIDDEN_SIZE), dtype=torch.float32)
    cif_alphas = torch.zeros(1, dtype=torch.float32)
    start_idx = torch.zeros(1, dtype=torch.int64)
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_EN):
        name = f'in_en_key_{i}'
        input_names.append(name)
        all_inputs.append(key_en)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_en_key_{i}'
        output_names.append(name)
    for i in range(NUM_LAYER_EN):
        name = f'in_en_value_{i}'
        input_names.append(name)
        all_inputs.append(value_en)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_en_value_{i}'
        output_names.append(name)
    input_names.append("in_previous_mel_features")
    all_inputs.append(previous_mel_features)
    output_names.append("out_previous_mel_features")
    input_names.append("in_cif_hidden")
    all_inputs.append(cif_hidden)
    output_names.append("out_cif_hidden")
    input_names.append("in_cif_alphas")
    all_inputs.append(cif_alphas)
    output_names.append("out_cif_alphas")
    input_names.append("start_idx")
    all_inputs.append(start_idx)
    output_names.append("end_idx")
    output_names.append("encoder_out")
    output_names.append("list_frame")
    output_names.append("list_frame_len")
    input_names.append("audio")
    all_inputs.append(audio)
    dynamic_axes["list_frame"] = {1: 'list_frame_len'}

    paraformer_encoder = PARAFORMER_ENCODER(model, custom_fbank, STFT_SIGNAL_LENGTH, LFR_M, LFR_N, LFR_LENGTH, CMVN_MEANS, CMVN_VARS, CIF_HIDDEN_SIZE, FSMN_HIDDEN_SIZE, FEATURE_SIZE, LOOK_BACK_A, LOOK_BACK_B, LOOK_BACK_C, LOOK_BACK_ENCODER, MAX_CONTINUE_STREAMING)
    torch.onnx.export(
        paraformer_encoder,
        tuple(all_inputs),
        raw_onnx_model_Encoder,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del paraformer_encoder
    del audio
    del key_en
    del value_en
    del previous_mel_features
    del cif_hidden
    del cif_alphas
    del start_idx
    del CMVN_VARS
    del CMVN_MEANS
    del all_inputs
    del input_names
    del output_names
    del dynamic_axes
    gc.collect()
    print('\nDone Encoder Part!\n\nExport Decoder Part...')

    key_de = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=CACHE_DTYPE)
    value_de = torch.zeros((NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=CACHE_DTYPE)
    fsmn_de = torch.zeros((1, FSMN_HIDDEN_SIZE, FSMN_DE_PAD), dtype=torch.float32)
    encoder_out = torch.zeros((1, LOOK_BACK_A + LOOK_BACK_C + LFR_LENGTH, CIF_HIDDEN_SIZE), dtype=torch.float32)
    list_frame = torch.zeros((1, 1, CIF_HIDDEN_SIZE), dtype=torch.float32)
    list_frame_len = torch.tensor(1, dtype=torch.int64)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_fsmn_{i}'
        input_names.append(name)
        all_inputs.append(fsmn_de)
        name = f'out_de_fsmn_{i}'
        output_names.append(name)
    for i in range(NUM_LAYER_DE):
        name = f'in_de_key_{i}'
        input_names.append(name)
        all_inputs.append(key_de)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_de_key_{i}'
        output_names.append(name)
    for i in range(NUM_LAYER_DE):
        name = f'in_de_value_{i}'
        input_names.append(name)
        all_inputs.append(value_de)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_de_value_{i}'
        output_names.append(name)
    input_names.append("encoder_out")
    all_inputs.append(encoder_out)
    input_names.append("list_frame")
    dynamic_axes["list_frame"] = {1: 'list_frame_len'}
    all_inputs.append(list_frame)
    input_names.append("list_frame_len")
    all_inputs.append(list_frame_len)
    output_names.append("max_logit_ids")
    dynamic_axes["max_logit_ids"] = {-1: 'token_len'}
    output_names.append("num_id")

    paraformer_decoder = PARAFORMER_DECODER(model, LOOK_BACK_B, LOOK_BACK_C, LOOK_BACK_DECODER, CIF_HIDDEN_SIZE, NUM_LAYER_DE)
    torch.onnx.export(
        paraformer_decoder,
        tuple(all_inputs),
        raw_onnx_model_Decoder,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del paraformer_decoder
    del key_de
    del value_de
    del all_inputs
    del input_names
    del output_names
    del dynamic_axes
    metadata_marker = torch.zeros((1,), dtype=torch.int64)
    torch.onnx.export(
        METADATA_CARRIER(),
        (metadata_marker,),
        raw_onnx_model_Metadata,
        input_names=["metadata_marker"],
        output_names=["metadata_marker_out"],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )
    del metadata_marker

    onnx_metadata = build_model_metadata(
        {
            "sample_rate": SAMPLE_RATE,
            "audio_pcm_scale": 1,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
        },
    )
    write_metadata_carrier(raw_onnx_model_Metadata, onnx_metadata)

    print(
        f"\n[Metadata] Stamped and verified {len(onnx_metadata)} keys in "
        f"{Path(raw_onnx_model_Metadata).name}:"
    )
    for _key in sorted(onnx_metadata):
        print(f"    {_key} = {onnx_metadata[_key]}")

    print("\n[Targeted rewrite] Hoisting exporter-generated Constant nodes into shared initializers:")
    try:
        for _name, _report in rewrite_folder(raw_onnx_folder, onnx_folder):
            print(
                f"    {_name}: {_report['raw_nodes']} -> {_report['final_nodes']} nodes; "
                f"removed {_report['constant_nodes_removed']} Constant nodes, "
                f"added {_report['unique_initializers_added']} shared initializers"
            )
        write_metadata_carrier(
            Path(onnx_folder) / Path(raw_onnx_model_Metadata).name,
            onnx_metadata,
        )
    finally:
        _raw_onnx_temp.cleanup()
    print("[Targeted rewrite] Removed automatic raw-export staging directory.")
    gc.collect()
print('\nExport done!\n')
if subprocess.call(
    [
        sys.executable,
        str(script_folder / "Inference_Paraformer_Streaming_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ],
    cwd=str(script_folder),
) != 0:
    raise RuntimeError("Paraformer Streaming inference failed after export.")
