import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.compliance.kaldi as kaldi

from funasr import AutoModel


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================== Path settings ==============================
# Set this single path to the Paraformer download you want to export. The language
# (zh or en) is auto-detected from the folder, so there is no separate switch to set.
DOWNLOADS_DIR   = Path.home() / "Downloads"
MODEL_PATH      = DOWNLOADS_DIR / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  # The Paraformer download to export.
ONNX_OUTPUT_DIR = SCRIPT_DIR / "Paraformer_ONNX"                                                        # Where exported artifacts are written.
ONNX_MODEL_PATH = ONNX_OUTPUT_DIR / "Paraformer.onnx"                                                   # The exported onnx model path.
VOCAB_FILE_PATH = ONNX_OUTPUT_DIR / "Vocab_Paraformer.txt"                                              # Save the vocab list.
ONNX_METADATA_PATH = ONNX_OUTPUT_DIR / "ASR_Metadata.onnx"                                       # Tiny metadata carrier graph.


# ============================== Language profiles ==============================
# Only the decode cleanup differs between languages; the front-end and model
# forward logic are shared. The active profile is selected by detect_language.
LANGUAGE_PROFILES = {
    "zh": {
        "decode_mode": "zh",
    },
    "en": {
        "decode_mode": "en",
    },
}


def detect_language(model_dir):
    """Infer 'zh' or 'en' from a single Paraformer download folder.

    Primary signal is the folder name (the official downloads are tagged '...-en-...'
    / 'vocab4199' for English and '...-zh-...' / 'vocab8404' for Chinese). If the name
    is ambiguous we fall back to the token list: the English BPE vocab carries '@@'
    continuation markers, the Chinese character vocab does not.
    """
    name = Path(model_dir).name.lower()
    if "-en-" in name or "_en-" in name or "vocab4199" in name:
        return "en"
    if "zh" in name or "vocab8404" in name:
        return "zh"
    tokens_file = Path(model_dir) / "tokens.json"
    if tokens_file.is_file():
        with open(tokens_file, "r", encoding="utf-8") as handle:
            tokens = json.load(handle)
        return "en" if any("@@" in str(token) for token in tokens) else "zh"
    raise ValueError(f"Cannot determine language (zh/en) from model folder: {model_dir!r}")


model_path = str(MODEL_PATH)                                                # The selected Paraformer download path.
LANGUAGE   = detect_language(model_path)                                    # Auto-detected from the model folder.
PROFILE    = LANGUAGE_PROFILES[LANGUAGE]


# ============================== Runtime paths ==============================
onnx_model_A = str(ONNX_MODEL_PATH)                                         # The exported onnx model path.
onnx_model_Metadata = str(ONNX_METADATA_PATH)                               # Tiny metadata carrier graph.
vocab_path   = str(VOCAB_FILE_PATH)                                         # Save the vocab list.


# ============================== Export / runtime settings ==============================
DYNAMIC_AXES         = True                                                 # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH   = 480000                                               # The maximum input audio length. Must be <= 480000 (30 seconds).
DECODER_CROSS_KV_GROUP_SIZE = 4                                             # Fuse this many decoder memory K/V projections into each GEMM (4 balances launch reduction and peak memory).
WINDOW_TYPE          = "hamming"                                            # Type of window function used in the STFT.
N_MELS               = 80                                                   # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT            = 512                                                  # Kaldi fbank defaults to 512 for both zh and en profiles.
WINDOW_LENGTH        = 400                                                  # Length of windowing (25 ms analysis window).
HOP_LENGTH           = 160                                                  # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE          = 16000                                                # The model parameter, do not edit the value.
LFR_M                = 7                                                    # The model parameter, do not edit the value.
LFR_N                = 6                                                    # The model parameter, do not edit the value.
PRE_EMPHASIZE        = 0.97                                                 # For audio preprocessing.
FRONTEND_TYPE        = "kaldi"                                              # Front-end implementation ('kaldi').
DECODE_MODE          = PROFILE["decode_mode"]                               # Token decoding mode ('zh' or 'en').
INPUT_AUDIO_DTYPE    = "F32"                                                # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
OPSET                = 20                                                   # <= 20


# ============================== Derived values ==============================
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1   # The length after Kaldi snip_edges=True fbank framing.
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N


def sinusoidal_encode(positions, depth, dtype=torch.float32):
    # Re-implements FunASR SinusoidalPositionEncoder.encode used by SANMEncoder.
    positions = positions.type(dtype)
    log_timescale_increment = torch.log(torch.tensor([10000], dtype=dtype)) / (depth / 2 - 1)
    inv_timescales = torch.exp(torch.arange(depth / 2).type(dtype) * (-log_timescale_increment))
    inv_timescales = torch.reshape(inv_timescales, [positions.size(0), -1])
    scaled_time = torch.reshape(positions, [1, -1, 1]) * torch.reshape(inv_timescales, [1, 1, -1])
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=2).type(dtype)


_SPECIAL_TOKEN_CANDIDATES = {
    "blank": ("<blank>", "<blk>", "<eps>"),
    "eos": ("</s>", "<eos>"),
    "unknown": ("<unk>", "<unknown>", "[UNK]"),
    "pad": ("<pad>", "[PAD]"),
    "bos": ("<s>", "<bos>", "<sos>"),
}
_LANGUAGE_METADATA = {
    "zh": {
        "name": "Chinese",
        "aliases": ["Chinese", "Mandarin", "zh-CN", "中文"],
    },
    "en": {
        "name": "English",
        "aliases": ["English", "en-US"],
    },
}


def _find_token_id(token_list, role):
    candidates = _SPECIAL_TOKEN_CANDIDATES[role]
    return next(
        (token_id for token_id, token in enumerate(token_list) if token in candidates),
        None,
    )


def build_tokenizer_metadata(token_list, language, decode_mode):
    token_list = list(token_list)

    blank_id = _find_token_id(token_list, "blank")
    eos_id = _find_token_id(token_list, "eos")
    special_token_ids = {
        "blank": blank_id,
        "eos": eos_id,
        "stop": [eos_id],
    }
    for role in ("unknown", "pad", "bos"):
        token_id = _find_token_id(token_list, role)
        if token_id is not None:
            special_token_ids[role] = token_id

    language_metadata = _LANGUAGE_METADATA[language]
    supported_languages = {
        language: {
            **language_metadata,
            "prompt_token_ids": [],
            "decode_mode": decode_mode,
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
    write_onnx_metadata(
        onnx_path,
        {str(key): str(value) for key, value in metadata.items()},
        replace=True,
    )


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


def _output_scale_tensor(output_scale, device):
    scale = torch.as_tensor(output_scale, dtype=torch.float64, device=device)
    if scale.ndim == 0:
        return scale
    scale = scale.reshape(-1)
    return scale


def fold_linear_output_scale(linear, output_scale):
    """Fold an immutable per-output scale into a Linear in float64, then store float32."""
    with torch.no_grad():
        weight = linear.weight.detach().to(torch.float64)
        bias = (linear.bias.detach().to(torch.float64) if linear.bias is not None
                else torch.zeros(linear.out_features, dtype=torch.float64, device=weight.device))
        scale = _output_scale_tensor(output_scale, weight.device)
        if scale.ndim == 0:
            weight = weight * scale
            bias = bias * scale
        else:
            weight = weight * scale.unsqueeze(1)
            bias = bias * scale
        linear.weight.copy_(weight.to(linear.weight.dtype))
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(bias.to(linear.weight.dtype))
        else:
            linear.bias.copy_(bias.to(linear.bias.dtype))
    linear._onnx_output_scale_folded = True


def absorb_layer_norm_affine(norm, linear, output_scale=1.0):
    """Fold a LayerNorm's affine (weight, bias) into the linear that consumes its output.

        new_bias   = linear.bias + linear.weight @ norm.bias
        new_weight = linear.weight * norm.weight        (scales the linear's input columns)

    The learned affine is replaced by an identity scale/shift, which is later shared across
    compatible LayerNorms so ONNX emits reusable initializers instead of per-call constants.
    Immutable transforms are evaluated in float64 and rounded once to the model dtype. The
    linear must be the sole consumer of the normalised tensor.
    """
    with torch.no_grad():
        weight = linear.weight.detach().to(torch.float64)
        bias = (linear.bias.detach().to(torch.float64) if linear.bias is not None
                else torch.zeros(linear.out_features, dtype=torch.float64, device=weight.device))
        scale = _output_scale_tensor(output_scale, weight.device)
        if scale.ndim == 0:
            weight = weight * scale
            bias = bias * scale
        else:
            weight = weight * scale.unsqueeze(1)
            bias = bias * scale
        bias = bias + torch.matmul(weight, norm.bias.detach().to(torch.float64))
        weight = weight * norm.weight.detach().to(torch.float64).unsqueeze(0)
        linear.weight.copy_(weight.to(linear.weight.dtype))
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(bias.to(linear.weight.dtype))
        else:
            linear.bias.copy_(bias.to(linear.bias.dtype))
        norm.weight.fill_(1.0)
        norm.bias.zero_()
    norm._onnx_affine_folded = True


def share_folded_layer_norm_affines(module):
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
    from the exported graph while keeping the result bit-identical. Only symmetric, zero-valued
    pads are expressible by Conv1d's scalar padding, so anything else is rejected rather than
    silently mis-folded.
    """
    left, right = pad_module.padding
    conv.padding = (int(left),)


def fold_depthwise_residual_into_conv(conv):
    """Fold ``depthwise_conv(x) + x`` into the convolution's centre tap."""
    kernel_size = int(conv.kernel_size[0])
    with torch.no_grad():
        weight = conv.weight.detach().to(torch.float64)
        weight[:, 0, kernel_size // 2] += 1.0
        conv.weight.copy_(weight.to(conv.weight.dtype))
    conv._onnx_identity_folded = True


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
    def __init__(self, n_fft, win_length, hop_len, n_mels, sample_rate, window_type, pre_emphasis):
        super().__init__()
        self.hop_len = hop_len
        self.n_freqs = n_fft // 2 + 1
        stft_kernel = create_kaldi_stft_kernel(n_fft, win_length, window_type, pre_emphasis)
        mel_bins, _ = kaldi.get_mel_banks(n_mels, n_fft, sample_rate, 20.0, 0.0, 100.0, -500.0, 1.0)
        mel_bins = torch.nn.functional.pad(mel_bins, (0, 1), mode="constant", value=0.0)
        self.register_buffer("stft_kernel", stft_kernel)
        self.register_buffer("mel_bins", mel_bins.unsqueeze(0).to(torch.float32))
        self.register_buffer("epsilon", torch.tensor(torch.finfo(torch.float32).eps, dtype=torch.float32))

    def forward(self, audio):
        stft = torch.nn.functional.conv1d(audio.float(), self.stft_kernel, stride=self.hop_len)
        real_power, imag_power = torch.split(stft * stft, self.n_freqs, dim=1)            # one square over the 2*n_freqs channels, then split (== real^2 / imag^2)
        power = real_power + imag_power
        mel = torch.matmul(self.mel_bins, power)
        log_mel = torch.maximum(mel, self.epsilon).log()
        return log_mel.transpose(1, 2)


class PARAFORMER(torch.nn.Module):
    def __init__(self, paraformer, fbank_model, n_mels, lfr_m, lfr_n, lfr_len, cmvn_means, cmvn_vars,
                 cif_hidden_size, cross_kv_group_size=DECODER_CROSS_KV_GROUP_SIZE):
        super(PARAFORMER, self).__init__()
        self.encoder = paraformer.encoder
        self.predictor = paraformer.predictor
        self.decoder = paraformer.decoder
        self.fbank_model = fbank_model
        self.register_buffer("cmvn_vars", cmvn_vars.detach().clone())
        self.T_lfr = lfr_len
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cif_hidden_size = cif_hidden_size
        self.lfr_m_factor = (lfr_m - 1) // 2
        self.lfr_feature_size = n_mels * lfr_m                                          # static LFR-stacked feature width
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int64).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int64) - self.lfr_m_factor
        self.register_buffer("indices_mel", indices.clamp(min=0).reshape(-1))  # int64 matches dynamic shape arithmetic and avoids a runtime Cast

        # Fold the attention scale (1 / sqrt(d_k)) into the q/k projection weights so the inlined
        # attention can use a plain q @ k matmul without a separate scaling step, then absorb every
        # LayerNorm affine into the linear that consumes it. The immutable transforms are evaluated
        # together in float64 and rounded once, while runtime stays entirely in the model dtype.
        head_dim = self.encoder.encoders._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        total_encoders = list(self.encoder.encoders0) + list(self.encoder.encoders)
        for encoder_layer in total_encoders:
            attn = encoder_layer.self_attn
            qk_scale = torch.ones(attn.linear_q_k_v.out_features, dtype=torch.float64,
                                  device=attn.linear_q_k_v.weight.device)
            qk_scale[:-cif_hidden_size] = factor
            absorb_layer_norm_affine(encoder_layer.norm1, attn.linear_q_k_v, qk_scale)
            absorb_layer_norm_affine(encoder_layer.norm2, encoder_layer.feed_forward.w_1)

        head_dim = self.decoder.decoders._modules["0"].src_attn.d_k
        factor = float(head_dim ** (-0.25))
        for decoder_layer in self.decoder.decoders:
            cross = decoder_layer.src_attn
            absorb_layer_norm_affine(decoder_layer.norm1, decoder_layer.feed_forward.w_1)
            absorb_layer_norm_affine(decoder_layer.feed_forward.norm, decoder_layer.feed_forward.w_2)
            absorb_layer_norm_affine(decoder_layer.norm3, cross.linear_q, factor)
            kv_scale = torch.ones(cross.linear_k_v.out_features, dtype=torch.float64,
                                  device=cross.linear_k_v.weight.device)
            kv_scale[:cif_hidden_size] = factor
            fold_linear_output_scale(cross.linear_k_v, kv_scale)

        # decoders3 are FFN-only blocks (no self/cross attention); fold their two LayerNorms too,
        # and finally fold the decoder's trailing after_norm into the output projection.
        for decoder_layer in self.decoder.decoders3:
            absorb_layer_norm_affine(decoder_layer.norm1, decoder_layer.feed_forward.w_1)
            absorb_layer_norm_affine(decoder_layer.feed_forward.norm, decoder_layer.feed_forward.w_2)
        absorb_layer_norm_affine(self.decoder.after_norm, self.decoder.output_layer)

        # Fold every symmetric zero pad into Conv1d, then fold each FSMN's depthwise_conv(x) + x
        # identity into the centre tap. This removes 66 full activation-sized residual Adds.
        for encoder_layer in total_encoders:
            fold_symmetric_pad_into_conv(encoder_layer.self_attn.pad_fn, encoder_layer.self_attn.fsmn_block)
            fold_depthwise_residual_into_conv(encoder_layer.self_attn.fsmn_block)
            encoder_layer.self_attn.pad_fn = None
        for decoder_layer in self.decoder.decoders:
            fold_symmetric_pad_into_conv(decoder_layer.self_attn.pad_fn, decoder_layer.self_attn.fsmn_block)
            fold_depthwise_residual_into_conv(decoder_layer.self_attn.fsmn_block)
            decoder_layer.self_attn.pad_fn = None
        fold_symmetric_pad_into_conv(self.predictor.pad, self.predictor.cif_conv1d)
        self.predictor.pad = None
        share_folded_layer_norm_affines(self)

        # Flatten the FunASR Sequential containers into plain lists so forward() can iterate the
        # layers explicitly (one inlined block per layer in the exported graph).
        self.encoder_layers = total_encoders
        self.decoder_att_layers = list(self.decoder.decoders)
        self.decoder_ffn_layers = list(self.decoder.decoders3)
        # Every decoder layer projects the same encoder memory. Concatenate immutable K/V weights
        # in bounded groups, execute one larger GEMM per group, then split in head-major layout.
        # Four layers reduce 16 MatMul/Add/Reshape/Transpose/Split chains to four while limiting the
        # maximum extra activation to about 8 MiB at the 500-frame production maximum.
        self.cross_kv_group_size = int(cross_kv_group_size)
        self.cross_kv_group_counts = []
        for group_idx, group_start in enumerate(range(0, len(self.decoder_att_layers), self.cross_kv_group_size)):
            group_layers = self.decoder_att_layers[group_start:group_start + self.cross_kv_group_size]
            kv_linears = [layer.src_attn.linear_k_v for layer in group_layers]
            self.register_buffer(
                f"cross_kv_weight_{group_idx}",
                torch.cat([linear.weight.detach() for linear in kv_linears], dim=0).contiguous(),
            )
            self.register_buffer(
                f"cross_kv_bias_{group_idx}",
                torch.cat([linear.bias.detach() for linear in kv_linears], dim=0).contiguous(),
            )
            self.cross_kv_group_counts.append(len(group_layers))
            for layer in group_layers:
                layer.src_attn.linear_k_v = None

        # Constants that the original FunASR modules build internally; precomputed here so the
        # export no longer depends on the patched modeling files.
        positions = torch.arange(1, lfr_len + 1, dtype=torch.int32).unsqueeze(0)
        position_encoding = sinusoidal_encode(positions, n_mels * lfr_m)
        encoder_input_bias = (cmvn_means.to(torch.float64) * cmvn_vars.to(torch.float64)
                              + position_encoding.to(torch.float64)).to(torch.float32)
        self.register_buffer("encoder_input_bias", encoder_input_bias)
        self.encoder.embed = None
        self.decoder.embed = None
        self.register_buffer("predictor_tail_threshold", torch.reshape(torch.tensor([self.predictor.tail_threshold], dtype=torch.float32), (1, 1)))
        self.register_buffer("predictor_start_zero", torch.zeros((1, 1), dtype=torch.float32))
        self.register_buffer("predictor_zeros", torch.zeros((1, 1, cif_hidden_size), dtype=torch.float32))
        self.register_buffer("cif_frame_zero", torch.zeros((1, cif_hidden_size), dtype=torch.float32))
        self.register_buffer("cif_one", torch.ones((1,), dtype=torch.int32))

    def forward(self, audio):
        # ----- Front-end -> LFR stacking -----
        mel_features = self.fbank_model(audio)
        mel_len = torch._shape_as_tensor(mel_features)[1]
        _len = (mel_len + self.lfr_n - 1) // self.lfr_n
        lfr_indices = torch.minimum(self.indices_mel[:_len * self.lfr_m], mel_len - 1)
        mel_features = torch.index_select(mel_features, 1, lfr_indices).reshape(1, -1, self.lfr_feature_size)

        # ----- Encoder: SANMEncoder (CMVN + sinusoidal position encoding + SANM blocks) -----
        enc = mel_features * self.cmvn_vars + self.encoder_input_bias[:, :_len]
        for layer in self.encoder_layers:
            attn = layer.self_attn
            hidden = attn.h * attn.d_k
            qkv = attn.linear_q_k_v(layer.norm1(enc))                                 # fused q/k/v projection (one GEMM)
            v = qkv[:, :, 2 * hidden:]                                                # (1, time, hidden) reused by FSMN
            q, k, v_h = torch.split(qkv.view(-1, 3 * attn.h, attn.d_k).transpose(0, 1), attn.h, dim=0)  # one reshape splits all heads
            scores = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)        # k.transpose -> (head, d_k, time)
            context = torch.matmul(scores, v_h).transpose(0, 1).reshape(1, -1, hidden)
            fsmn = attn.fsmn_block(v.transpose(1, 2)).transpose(1, 2)                 # pad and identity residual folded into Conv1d
            att_out = attn.linear_out(context) + fsmn
            enc = enc + att_out if layer.in_size == layer.size else att_out
            ff = layer.feed_forward
            enc = enc + ff.w_2(ff.activation(ff.w_1(layer.norm2(enc))))
        encoder_out = self.encoder.after_norm(enc)

        # ----- CIF predictor: CifPredictorV2 (alpha weights + continuous integrate-and-fire) -----
        context = encoder_out.transpose(1, 2)
        conv_out = torch.relu(self.predictor.cif_conv1d(context)).transpose(1, 2)               # CIF pad folded into the Conv1d
        alphas = torch.sigmoid(self.predictor.cif_output(conv_out)).squeeze(-1)                 # relu(sigmoid()) == sigmoid() (sigmoid >= 0)
        alphas = torch.cat([alphas, self.predictor_tail_threshold], dim=-1)
        cif_hidden = torch.cat([encoder_out, self.predictor_zeros], dim=1)
        # FunASR deliberately accumulates alpha in float64 and rounds once to float32; a genuine
        # fp32 ONNX CumSum can miss an integer boundary and change the transcript.
        prefix_sum = torch.cumsum(alphas, dim=-1, dtype=torch.float64).float()
        prefix_sum_floor = torch.floor(prefix_sum)
        dislocation_floor = torch.cat([self.predictor_start_zero, prefix_sum_floor[:, :-1]], dim=1)
        fire_idxs = prefix_sum_floor > dislocation_floor
        fire_indices = torch.nonzero(fire_idxs[0], as_tuple=False).squeeze(1)
        prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1) * cif_hidden, dim=1)
        frames = torch.index_select(prefix_sum_hidden, 1, fire_indices).squeeze(0)
        remains = torch.index_select(prefix_sum - prefix_sum_floor, 1, fire_indices).squeeze(0)
        fired_hidden = torch.index_select(cif_hidden, 1, fire_indices).squeeze(0)
        completed_prefix = frames - remains.unsqueeze(1) * fired_hidden
        completed_prefix = torch.cat([self.cif_frame_zero, completed_prefix], dim=0)
        acoustic_embeds = (completed_prefix[1:] - completed_prefix[:-1]).unsqueeze(0)  # zero-fire safe (1, token, dim)
        num_id = prefix_sum_floor[:, -1].to(torch.int32)                              # fixed shape [1]: authoritative CIF fire count

        # ----- Decoder: ParaformerSANMDecoder (FFN -> SANM self-attn -> cross-attn per block) -----
        memory = encoder_out
        # Conv1d cannot consume a zero-length token axis. Append one immutable zero frame, gather
        # max(num_id, 1) rows with native int32 indices, and remove that dummy from token_ids below.
        # Dynamic Range avoids the fragile scalar-Slice lowering used by several ONNX optimizers.
        safe_num_id = torch.maximum(num_id, self.cif_one)
        safe_token_indices = torch.arange(safe_num_id[0], dtype=torch.int32, device=acoustic_embeds.device)
        dec = torch.index_select(torch.cat([acoustic_embeds, self.predictor_zeros], dim=1), 1, safe_token_indices)
        layer_idx = 0
        for group_idx, group_count in enumerate(self.cross_kv_group_counts):
            first_cross = self.decoder_att_layers[layer_idx].src_attn
            grouped_kv = F.linear(
                memory,
                getattr(self, f"cross_kv_weight_{group_idx}"),
                getattr(self, f"cross_kv_bias_{group_idx}"),
            )
            grouped_kv = grouped_kv.reshape(-1, 2 * group_count * first_cross.h, first_cross.d_k).transpose(0, 1)
            kv_heads = torch.split(grouped_kv, first_cross.h, dim=0)
            for local_idx in range(group_count):
                layer = self.decoder_att_layers[layer_idx]
                ff = layer.feed_forward
                x = ff.w_2(ff.norm(ff.activation(ff.w_1(layer.norm1(dec)))))
                sa = layer.self_attn
                sa_in = layer.norm2(x)
                fsmn = sa.fsmn_block(sa_in.transpose(1, 2)).transpose(1, 2)             # pad and identity residual folded into Conv1d
                x = dec + fsmn
                cross = layer.src_attn
                c_in = layer.norm3(x)
                q = cross.linear_q(c_in).view(-1, cross.h, cross.d_k).transpose(0, 1)
                k = kv_heads[2 * local_idx]
                v = kv_heads[2 * local_idx + 1]
                scores = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)
                c_out = torch.matmul(scores, v).transpose(0, 1).reshape(1, -1, self.cif_hidden_size)
                dec = x + cross.linear_out(c_out)
                layer_idx += 1
        for layer in self.decoder_ffn_layers:
            ff = layer.feed_forward
            dec = ff.w_2(ff.norm(ff.activation(ff.w_1(layer.norm1(dec)))))
        decoder_out = self.decoder.output_layer(self.decoder.after_norm(dec))
        token_ids = decoder_out.argmax(dim=-1).int()
        output_token_indices = torch.arange(num_id[0], dtype=torch.int32, device=token_ids.device)
        token_ids = torch.index_select(token_ids, 1, output_token_indices)             # (1, num_token) int32 token ids; empty when CIF fires zero times
        return token_ids, num_id


print('\nExport start ...\n')
with torch.inference_mode():
    Path(onnx_model_A).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(vocab_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    custom_fbank = KaldiFbank(NFFT_STFT, WINDOW_LENGTH, HOP_LENGTH, N_MELS, SAMPLE_RATE, WINDOW_TYPE, PRE_EMPHASIZE).eval()
    print(f"Language: {LANGUAGE}; frontend: {FRONTEND_TYPE}; decode: {DECODE_MODE}")
    model = AutoModel(
        model=model_path,
        disable_update=True,
        device="cpu"
    )
    encoder_output_size_factor = (model.model.encoder.output_size()) ** 0.5
    CMVN_MEANS = model.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    CIF_HIDDEN_SIZE = model.model.encoder.encoders0._modules["0"].size
    tokenizer = model.kwargs['tokenizer']
    token_list = list(tokenizer.token_list)
    special_token_ids, supported_languages = build_tokenizer_metadata(
        token_list,
        LANGUAGE,
        DECODE_MODE,
    )
    # Save to text file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in token_list:
            f.write(f'{token}\n')
  
    paraformer = PARAFORMER(model.model.eval(), custom_fbank, N_MELS, LFR_M, LFR_N, LFR_LENGTH, CMVN_MEANS, CMVN_VARS, CIF_HIDDEN_SIZE)
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)
    torch.onnx.export(
        paraformer,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=['token_ids', 'num_id'],
        do_constant_folding=True,
        dynamic_axes={
            'audio': {2: 'audio_len'},
            'token_ids': {1: 'num_token'}
        } if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
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
            "sample_rate": SAMPLE_RATE,
            "audio_pcm_scale": 1,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
        },
    )
    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    del model
    del audio
    del CMVN_VARS
    del CMVN_MEANS
    gc.collect()
print('\nExport done!\n')
if subprocess.call(
    [
        sys.executable,
        str(SCRIPT_DIR / "Inference_Paraformer_ONNX.py"),
        "--onnx-folder",
        str(ONNX_OUTPUT_DIR),
    ],
    cwd=str(SCRIPT_DIR),
) != 0:
    raise RuntimeError("Paraformer inference failed after export.")
