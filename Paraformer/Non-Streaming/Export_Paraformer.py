import gc
import subprocess
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi

from funasr import AutoModel


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================== Path settings ==============================
# Set this single path to the Paraformer download you want to export. The language
# (zh or en) is auto-detected from the folder, so there is no separate switch to set.
DOWNLOADS_DIR   = Path("/home/DakeQQ/Downloads")
MODEL_PATH      = DOWNLOADS_DIR / "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"  # The Paraformer download to export.
ONNX_OUTPUT_DIR = SCRIPT_DIR / "Paraformer_ONNX"                                                        # Where exported artifacts are written.
ONNX_MODEL_PATH = ONNX_OUTPUT_DIR / "Paraformer.onnx"                                                   # The exported onnx model path.
VOCAB_FILE_PATH = ONNX_OUTPUT_DIR / "Vocab_Paraformer.txt"                                              # Save the vocab list.
ONNX_METADATA_PATH = ONNX_OUTPUT_DIR / "Paraformer_Metadata.onnx"                                       # Tiny metadata carrier graph.


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
PREVENT_F16_OVERFLOW = False                                                # Set True before export if the front-end will be converted to fp16.
INPUT_AUDIO_LENGTH   = 480000                                               # The maximum input audio length. Must less than 480000 (30 seconds).
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
INPUT_AUDIO_DTYPE    = "INT16"                                              # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
OPSET                = 18                                                   # <= 20


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
def decode_tokens(tokens, mode):
    if mode == "en":
        return " ".join(tokens).replace("</s>", "").replace("@@ ", "").strip()
    return "".join(tokens).replace("</s>", "").strip()


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


class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


def absorb_layer_norm_affine(norm, linear):
    """Fold a LayerNorm's affine (weight, bias) into the linear that consumes its output.

        new_bias   = linear.bias + linear.weight @ norm.bias
        new_weight = linear.weight * norm.weight        (scales the linear's input columns)

    The LayerNorm is then switched to affine-free, so at runtime it only normalises while the
    scale/shift live inside the following GEMM weight/bias. This is exact in float32 and keeps
    the single fused LayerNormalization op. The linear must be the sole consumer of the
    normalised tensor.
    """
    with torch.no_grad():
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=linear.weight.dtype))
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.elementwise_affine = False
    norm.weight = None
    norm.bias = None


def fold_symmetric_pad_into_conv(pad_module, conv):
    """Fold a zero-valued symmetric ConstantPad1d into the following Conv1d's own padding.

    nn.Conv1d already performs zero padding, so ConstantPad1d((p, p), 0.0) feeding a padding=0
    convolution is exactly a convolution with padding=p. Folding removes the standalone Pad node
    from the exported graph while keeping the result bit-identical. Only symmetric, zero-valued
    pads are expressible by Conv1d's scalar padding, so anything else is rejected rather than
    silently mis-folded.
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


class PARAFORMER(torch.nn.Module):
    def __init__(self, paraformer, fbank_model, stft_signal_len, n_mels, lfr_m, lfr_n, lfr_len, cmvn_means, cmvn_vars, cif_hidden_size):
        super(PARAFORMER, self).__init__()
        self.encoder = paraformer.encoder
        self.predictor = paraformer.predictor
        self.decoder = paraformer.decoder
        replace_gelu_with_tanh(self.encoder)
        replace_gelu_with_tanh(self.predictor)
        replace_gelu_with_tanh(self.decoder)
        self.fbank_model = fbank_model
        self.register_buffer("cmvn_means", cmvn_means, persistent=False)
        self.register_buffer("cmvn_vars", cmvn_vars, persistent=False)
        self.T_lfr = lfr_len
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.lfr_m_factor = (lfr_m - 1) // 2
        self.lfr_feature_size = n_mels * lfr_m                                          # static LFR-stacked feature width
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.register_buffer("indices_mel", indices.clamp(max=stft_signal_len + self.lfr_m_factor - 1).reshape(-1), persistent=False)  # int32 LFR gather indices

        # Fold the attention scale (1 / sqrt(d_k)) into the q/k projection weights so the inlined
        # attention can use a plain q @ k matmul without a separate scaling step, then absorb every
        # LayerNorm affine into the linear that consumes it. Both folds are exact in float32 and keep
        # the fused LayerNormalization op. Scale-fold runs before LayerNorm-fold; the two commute
        # because one scales the linear's output rows and the other scales its input columns.
        head_dim = self.encoder.encoders._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        total_encoders = list(self.encoder.encoders0) + list(self.encoder.encoders)
        for encoder_layer in total_encoders:
            attn = encoder_layer.self_attn
            attn.linear_q_k_v.weight.data[:-cif_hidden_size] *= factor
            attn.linear_q_k_v.bias.data[:-cif_hidden_size] *= factor
            absorb_layer_norm_affine(encoder_layer.norm1, attn.linear_q_k_v)
            absorb_layer_norm_affine(encoder_layer.norm2, encoder_layer.feed_forward.w_1)

        head_dim = self.decoder.decoders._modules["0"].src_attn.d_k
        factor = float(head_dim ** (-0.25))
        for decoder_layer in self.decoder.decoders:
            cross = decoder_layer.src_attn
            cross.linear_q.weight.data *= factor
            cross.linear_q.bias.data *= factor
            cross.linear_k_v.weight.data[:cif_hidden_size] *= factor
            cross.linear_k_v.bias.data[:cif_hidden_size] *= factor
            absorb_layer_norm_affine(decoder_layer.norm1, decoder_layer.feed_forward.w_1)
            absorb_layer_norm_affine(decoder_layer.feed_forward.norm, decoder_layer.feed_forward.w_2)
            absorb_layer_norm_affine(decoder_layer.norm3, cross.linear_q)

        # decoders3 are FFN-only blocks (no self/cross attention); fold their two LayerNorms too,
        # and finally fold the decoder's trailing after_norm into the output projection.
        for decoder_layer in self.decoder.decoders3:
            absorb_layer_norm_affine(decoder_layer.norm1, decoder_layer.feed_forward.w_1)
            absorb_layer_norm_affine(decoder_layer.feed_forward.norm, decoder_layer.feed_forward.w_2)
        absorb_layer_norm_affine(self.decoder.after_norm, self.decoder.output_layer)

        # Fold every (symmetric, zero) FSMN / CIF pad into its following Conv1d so the exported graph
        # drops the standalone Pad nodes. Conv1d already zero-pads, so each fold is bit-identical.
        for encoder_layer in total_encoders:
            fold_symmetric_pad_into_conv(encoder_layer.self_attn.pad_fn, encoder_layer.self_attn.fsmn_block)
        for decoder_layer in self.decoder.decoders:
            fold_symmetric_pad_into_conv(decoder_layer.self_attn.pad_fn, decoder_layer.self_attn.fsmn_block)
        fold_symmetric_pad_into_conv(self.predictor.pad, self.predictor.cif_conv1d)

        # Flatten the FunASR Sequential containers into plain lists so forward() can iterate the
        # layers explicitly (one inlined block per layer in the exported graph).
        self.encoder_layers = total_encoders
        self.decoder_att_layers = list(self.decoder.decoders)
        self.decoder_ffn_layers = list(self.decoder.decoders3)
        assert getattr(self.decoder, "decoders2", None) is None, \
            "Inlined decoder assumes att_layer_num == num_blocks (decoders2 must be None)."

        # Constants that the original FunASR modules build internally; precomputed here so the
        # export no longer depends on the patched modeling files.
        positions = torch.arange(1, 502, dtype=torch.int32).unsqueeze(0)   # 502 -> up to 30 seconds of audio
        self.register_buffer("position_encoding", sinusoidal_encode(positions, n_mels * lfr_m).half(), persistent=False)
        self.register_buffer("predictor_tail_threshold", torch.reshape(torch.tensor([self.predictor.tail_threshold], dtype=torch.float32), (1, 1)), persistent=False)
        self.register_buffer("predictor_start_zero", torch.zeros((1, 1), dtype=torch.float32), persistent=False)
        self.register_buffer("predictor_zeros", torch.zeros((1, 1, cif_hidden_size), dtype=torch.float32), persistent=False)
        self.register_buffer("cif_frame_zero", torch.zeros((1, cif_hidden_size), dtype=torch.float32), persistent=False)

    def forward(self, audio):
        # ----- Front-end -> LFR stacking -----
        mel_features = self.fbank_model(audio)
        left_padding = mel_features[:, [0]]
        right_padding = mel_features[:, [-1]]
        padded_inputs = torch.cat([left_padding] * self.lfr_m_factor + [mel_features] + [right_padding] * self.lfr_m, dim=1)
        _len = (mel_features.shape[1] + self.lfr_n - 1) // self.lfr_n
        mel_features = torch.index_select(padded_inputs, 1, self.indices_mel[:_len * self.lfr_m]).reshape(1, -1, self.lfr_feature_size)

        # ----- Encoder: SANMEncoder (CMVN + sinusoidal position encoding + SANM blocks) -----
        enc = (mel_features + self.cmvn_means) * self.cmvn_vars
        enc = enc + self.position_encoding[:, :_len, :].float()
        for layer in self.encoder_layers:
            attn = layer.self_attn
            hidden = attn.h * attn.d_k
            qkv = attn.linear_q_k_v(layer.norm1(enc))                                 # fused q/k/v projection (one GEMM)
            v = qkv[:, :, 2 * hidden:]                                                # (1, time, hidden) reused by FSMN
            q, k, v_h = torch.split(qkv.view(-1, 3 * attn.h, attn.d_k).transpose(0, 1), attn.h, dim=0)  # one reshape splits all heads
            scores = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)        # k.transpose -> (head, d_k, time)
            context = torch.matmul(scores, v_h).transpose(0, 1).reshape(1, -1, hidden)
            fsmn = attn.fsmn_block(v.transpose(1, 2)).transpose(1, 2) + v             # FSMN pad folded into the Conv1d
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
        prefix_sum = torch.cumsum(alphas, dim=-1, dtype=torch.float32)
        prefix_sum_floor = torch.floor(prefix_sum)
        dislocation_floor = torch.floor(torch.cat([self.predictor_start_zero, prefix_sum[:, :-1]], dim=1))
        fire_idxs = (prefix_sum_floor - dislocation_floor) > 0
        fires = fire_idxs.float() + prefix_sum - prefix_sum_floor
        prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1) * cif_hidden, dim=1)
        frames = prefix_sum_hidden[fire_idxs]
        shift_frames = torch.cat([self.cif_frame_zero, frames[:-1]], dim=0)
        remains = fires - torch.floor(fires)
        remain_frames = remains[fire_idxs].reshape(-1, 1) * cif_hidden[fire_idxs]
        shift_remain_frames = torch.cat([self.cif_frame_zero, remain_frames[:-1]], dim=0)
        acoustic_embeds = (frames - shift_frames + shift_remain_frames - remain_frames).unsqueeze(0)  # (1, token, dim)

        # ----- Decoder: ParaformerSANMDecoder (FFN -> SANM self-attn -> cross-attn per block) -----
        memory = encoder_out
        dec = acoustic_embeds
        for layer in self.decoder_att_layers:
            ff = layer.feed_forward
            x = ff.w_2(ff.norm(ff.activation(ff.w_1(layer.norm1(dec)))))
            sa = layer.self_attn
            sa_in = layer.norm2(x)
            fsmn = sa.fsmn_block(sa_in.transpose(1, 2)).transpose(1, 2) + sa_in                 # FSMN pad folded into the Conv1d
            x = dec + fsmn
            cross = layer.src_attn
            c_in = layer.norm3(x)
            q = cross.linear_q(c_in).view(-1, cross.h, cross.d_k).transpose(0, 1)
            k, v = torch.split(cross.linear_k_v(memory).view(-1, 2 * cross.h, cross.d_k).transpose(0, 1), cross.h, dim=0)
            scores = torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)
            c_out = torch.matmul(scores, v).transpose(0, 1).reshape(1, -1, cross.linear_out.in_features)
            dec = x + cross.linear_out(c_out)
        for layer in self.decoder_ffn_layers:
            ff = layer.feed_forward
            dec = ff.w_2(ff.norm(ff.activation(ff.w_1(layer.norm1(dec)))))
        decoder_out = self.decoder.output_layer(self.decoder.after_norm(dec))
        token_ids = decoder_out.argmax(dim=-1).int()                                # (1, num_token) int32 token ids
        num_id = torch._shape_as_tensor(token_ids)[1].to(torch.int32).unsqueeze(0)  # fixed shape [1]: count of decoded tokens
        return token_ids, num_id


print('\nExport start ...\n')
with torch.inference_mode():
    Path(onnx_model_A).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(vocab_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    if FRONTEND_TYPE != "kaldi":
        raise ValueError(f"Unsupported PARAFORMER_FRONTEND={FRONTEND_TYPE!r}; use 'kaldi'.")
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
    FIRST_ATTN = model.model.encoder.encoders0._modules["0"].self_attn
    NUM_HEADS = FIRST_ATTN.h
    HEAD_DIM = FIRST_ATTN.d_k
    tokenizer = model.kwargs['tokenizer']
    # Save to text file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in tokenizer.token_list:
            f.write(f'{token}\n')
  
    paraformer = PARAFORMER(model.model.eval(), custom_fbank, STFT_SIGNAL_LENGTH, N_MELS, LFR_M, LFR_N, LFR_LENGTH, CMVN_MEANS, CMVN_VARS, CIF_HIDDEN_SIZE)
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
            "paraformer_metadata_version": 1,
            "producer": "Export_Paraformer.py",
            "language": LANGUAGE,
        },
        {
            "num_heads": NUM_HEADS,
            "head_dim": HEAD_DIM,
            "hidden_size": CIF_HIDDEN_SIZE,
            "vocab_size": len(tokenizer.token_list),
            "num_mels": N_MELS,
            "nfft_stft": NFFT_STFT,
            "window_length": WINDOW_LENGTH,
            "hop_length": HOP_LENGTH,
            "lfr_m": LFR_M,
            "lfr_n": LFR_N,
            "sample_rate": SAMPLE_RATE,
            "input_audio_length": INPUT_AUDIO_LENGTH,
        },
    )
    _written, _skipped = [], []
    for _target in [onnx_model_Metadata, onnx_model_A]:
        try:
            write_onnx_metadata(_target, onnx_metadata)
            _written.append(Path(_target).name)
        except Exception as _exc:  # noqa: BLE001 - one bad graph must not abort export
            _skipped.append(f"{Path(_target).name} ({_exc})")

    print(f"\n[Metadata] Stamped {len(onnx_metadata)} keys into {len(_written)} ONNX graph(s):")
    for _key in sorted(onnx_metadata):
        print(f"    {_key} = {onnx_metadata[_key]}")
    if _skipped:
        print("[Metadata] Skipped (kept usable, metadata not written):")
        for _entry in _skipped:
            print(f"    {_entry}")
    del model
    del audio
    del CMVN_VARS
    del CMVN_MEANS
    gc.collect()
print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_Paraformer_ONNX.py ...')
subprocess.run(
    [sys.executable, str(SCRIPT_DIR / "Inference_Paraformer_ONNX.py"), "--onnx-folder", str(ONNX_OUTPUT_DIR)],
    check=True,
)
