import gc
import subprocess
import sys
from pathlib import Path
import torch
import numpy as np
import torchaudio.compliance.kaldi as kaldi
from funasr import AutoModel


model_path         = "/home/DakeQQ/Downloads/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"    # The Paraformer-Chinese-Online-Streaming download path.
onnx_folder        = Path(__file__).resolve().parent / "Paraformer_ONNX"                                          # Local folder next to this script holding all exported ONNX graphs; created automatically if missing.
onnx_folder.mkdir(parents=True, exist_ok=True)
onnx_model_Metadata = str(onnx_folder / "ASR_Matadata.onnx")                                      # Tiny metadata carrier graph.
onnx_model_Encoder = str(onnx_folder / "Paraformer_Streaming_Encoder.onnx")                                         # The exported onnx model path.
onnx_model_Decoder = str(onnx_folder / "Paraformer_Streaming_Decoder.onnx")                                         # The exported onnx model path.
vocab_path         = str(onnx_folder / "Vocab_Paraformer.txt")                                                      # Save the vocab list.


MAX_CONTINUE_STREAMING = 502                                # 502 = Max 30 seconds streaming audio input. # 1003 = Max 60 seconds streaming audio input.
INPUT_AUDIO_LENGTH     = 8000                               # The fixed input audio segment length, edit it carefully.
WINDOW_TYPE            = 'hamming'                          # Type of window function used by Kaldi fbank. The online model is trained with 'hamming'.
N_MELS                 = 80                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT              = 512                                # Kaldi fbank rounds the 25 ms (400 sample) frame length up to the next power of two (512).
WINDOW_LENGTH          = 400                                # Length of windowing (25 ms analysis window), edit it carefully.
HOP_LENGTH             = 160                                # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE            = 16000                              # The model parameter, do not edit the value.
LFR_M                  = 7                                  # The model parameter, do not edit the value.
LFR_N                  = 6                                  # The model parameter, do not edit the value.
PRE_EMPHASIZE          = 0.97                               # Kaldi fbank per-frame pre-emphasis coefficient.
INPUT_AUDIO_DTYPE      = "INT16"                            # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
PREVENT_F16_OVERFLOW   = False                              # Set True before export if the front-end will be converted to fp16.
COMPUTE_IN_F32         = False                              # F16-cache compute precision. False = minimum-cast f16 attention (encoder self-attn + decoder cross-attn Q@K/softmax/attn@V run in f16 on the f16 caches; the context is cast back to f32). True = keep the f16 cache *storage* (cache I/O dtype unchanged) but upcast K/V to f32 at the attention use points, keeping Q/softmax in f32 (f16 storage, f32 compute).
LOOK_BACK_ENCODER      = 4                                  # The model parameter, edit it carefully.
LOOK_BACK_DECODER      = 1                                  # The model parameter, edit it carefully.
DYNAMIC_AXES           = True                               # The dynamic_axes setting. Do not turn off for the Paraformer Streaming model.
OPSET                  = 18                                 # <= 20


LFR_M_FACTOR = (LFR_M - 1) // 2                             # Number of left-context frames replicated before LFR stacking.
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1   # The Kaldi snip_edges=True fbank frame count for one chunk.
LFR_LENGTH = (LFR_M_FACTOR + STFT_SIGNAL_LENGTH) // LFR_N + 1                  # Must match the dynamic _len computed inside forward().


LOOK_BACK_B = LFR_LENGTH                                    # The model parameter, edit it carefully. 10 for 8800 input audio length. 20 for 8800*2 ...
LOOK_BACK_C = LOOK_BACK_B // 2                              # The model parameter, edit it carefully. 5 for 8800 input audio length. 10 for 8800*2 ...
LOOK_BACK_A = 0                                             # The model parameter, edit it carefully. 5 for 8800 input audio length. 10 for 8800*2 ...
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
        self.threshold = float(1.0)
        self.look_back_A = look_back_A
        self.look_back_B = look_back_B
        self.look_back_C = look_back_C
        self.look_back_en = -(look_back_en * look_back_B) - self.look_back_C
        self.encoder = paraformer.encoder
        self.predictor = paraformer.predictor
        self.fbank_model = fbank_model
        self.cmvn_means = cmvn_means
        self.cmvn_vars = cmvn_vars
        self.T_lfr = lfr_len
        self.lfr_n = lfr_n
        self.cif_hidden_size = cif_hidden_size
        self.lfr_m_factor = (lfr_m - 1) // 2
        self.lfr_feature_size = feature_size                                                # static LFR-stacked feature width (n_mels * lfr_m)
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=stft_signal_len + self.lfr_m_factor - 1).to(torch.int32)  # int32 LFR gather indices
        self.total_encoders = list(self.encoder.encoders0) + list(self.encoder.encoders)
        self.cache_layer_num_en = len(self.total_encoders)
        self.compute_in_f32 = COMPUTE_IN_F32
        self.save_keys_en = [None] * self.cache_layer_num_en
        self.save_values_en = [None] * self.cache_layer_num_en
        positions = torch.arange(1, max_continue_streaming, dtype=torch.int32).unsqueeze(0)
        self.position_encoding = self.encoder.embed.encode(positions, feature_size).half()

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

    def forward(self, *all_inputs):
        previous_mel_features = all_inputs[-5]
        cif_hidden = all_inputs[-4]
        cif_alphas = all_inputs[-3]
        start_idx = all_inputs[-2]
        audio = all_inputs[-1]
        mel_features = self.fbank_model(audio)
        left_padding = mel_features[:, [0]]
        padded_inputs = torch.cat([left_padding] * self.lfr_m_factor + [mel_features], dim=1)
        mel_features = padded_inputs[:, self.indices_mel].reshape(1, self.T_lfr, self.lfr_feature_size)   # fixed chunk -> static LFR stack (no Shape read)
        mel_features = (mel_features + self.cmvn_means) * self.cmvn_vars
        end_idx = start_idx + self.T_lfr
        mel_features += self.position_encoding[:, start_idx:end_idx]
        x = torch.cat([previous_mel_features, mel_features], dim=1)
        previous_mel_features = x[:, -(self.look_back_A + self.look_back_C):]
        for layer_idx, encoder_layer in enumerate(self.total_encoders):
            attn = encoder_layer.self_attn
            if layer_idx > 0:
                residual = x
            q, k, v = torch.split(attn.linear_q_k_v(encoder_layer.norm1(x)), self.cif_hidden_size, dim=-1)
            q = q.view(-1, attn.h, attn.d_k).transpose(0, 1)
            k = k.view(-1, attn.h, attn.d_k).permute(1, 2, 0).half()                         # keep the key cache in float16
            v_h = v.view(-1, attn.h, attn.d_k).transpose(0, 1).half()                        # keep the value cache in float16
            k = torch.cat([all_inputs[layer_idx], k], dim=2)
            v_h = torch.cat([all_inputs[layer_idx + self.cache_layer_num_en], v_h], dim=1)
            self.save_keys_en[layer_idx] = k[:, :, self.look_back_en:-self.look_back_C]
            self.save_values_en[layer_idx] = v_h[:, self.look_back_en:-self.look_back_C]
            v_fsmn = attn.fsmn_block(v.transpose(1, 2)).transpose(1, 2) + v                  # FSMN symmetric pad folded into the Conv1d
            if self.compute_in_f32:
                # f16 storage, f32 compute: upcast the f16 K/V cache to f32 at the matmul use points (Q stays f32).
                context = torch.matmul(torch.softmax(torch.matmul(q, k.float()), dim=-1), v_h.float()).transpose(0, 1).reshape(1, -1, self.cif_hidden_size)
            else:
                context = torch.matmul(torch.softmax(torch.matmul(q.half(), k), dim=-1), v_h).transpose(0, 1).reshape(1, -1, self.cif_hidden_size).float()   # minimum-cast: run attention in f16 on the f16 cache, then cast the context back to f32
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
        condition_A = cif_alphas < self.threshold
        condition_B = ~condition_A
        condition_A = condition_A.float()
        condition_B = condition_B.float()
        save_condition.append(condition_B)
        if self.threshold != 1.0:
            frames = cif_alphas * cif_hidden * condition_A + self.threshold * cif_hidden * condition_B
        else:
            frames = cif_alphas * cif_hidden * condition_A + cif_hidden * condition_B
        list_frame.append(frames)
        if self.threshold != 1.0:
            cif_alphas -= self.threshold * condition_B
        else:
            cif_alphas -= condition_B
        frames = frames * condition_A + cif_alphas * cif_hidden * condition_B
        for i in range(self.look_back_A, self.look_back_A + self.look_back_B):
            alpha = alphas[i]
            threshold = self.threshold - cif_alphas
            condition_A = alpha < threshold
            condition_B = ~condition_A
            condition_A = condition_A.float()
            condition_B = condition_B.float()
            save_condition.append(condition_B)
            hidden = encoder_out[:, [i]]
            frames = (frames + alpha * hidden) * condition_A + (frames + threshold * hidden) * condition_B
            list_frame.append(frames)
            cif_alphas = cif_alphas + alpha
            if self.threshold != 1.0:
                cif_alphas -= self.threshold * condition_B
            else:
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
        self.look_back_B = look_back_B
        self.look_back_C = look_back_C
        self.look_back_de = look_back_de * look_back_B
        self.decoder = paraformer.decoder
        self.cif_hidden_size = cif_hidden_size
        self.fsmn_kernal_size_minus = -torch.tensor([self.decoder.decoders._modules["0"].self_attn.kernel_size - 1], dtype=torch.int64)
        self.cache_layer_num_de = cache_layer_num_de
        self.compute_in_f32 = COMPUTE_IN_F32
        self.cache_layer_num_de_2 = cache_layer_num_de + cache_layer_num_de
        self.save_fsmn_de = [None] * cache_layer_num_de
        self.save_keys_de = [None] * cache_layer_num_de
        self.save_values_de = [None] * cache_layer_num_de

        # Fold the cross-attention scale into the q / k projections, then absorb every LayerNorm
        # affine into the linear that consumes it (norm2 is left intact because the decoder reuses
        # its normalised output for both the FSMN branch and the residual add, so it has two
        # consumers and cannot be folded).
        replace_gelu_with_tanh(self.decoder)
        factor = float(self.decoder.decoders._modules["0"].src_attn.d_k ** -0.25)
        for decoder_layer in self.decoder.decoders:
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

    def forward(self, *all_inputs):
        encoder_out = all_inputs[-3]
        list_frame = all_inputs[-2]
        list_frame_len = all_inputs[-1]
        look_back = self.fsmn_kernal_size_minus - list_frame_len
        for layer_idx, decoder_layer in enumerate(self.decoder.decoders):
            ff = decoder_layer.feed_forward
            cross = decoder_layer.src_attn
            residual = list_frame
            list_frame = decoder_layer.norm1(list_frame)
            list_frame = ff.w_2(ff.norm(ff.activation(ff.w_1(list_frame))))
            list_frame = decoder_layer.norm2(list_frame)
            x = torch.cat((all_inputs[layer_idx], list_frame.transpose(1, 2)), dim=-1)[:, :, look_back:]
            self.save_fsmn_de[layer_idx] = x
            x = decoder_layer.self_attn.fsmn_block(x).transpose(1, 2)
            x += list_frame + residual
            residual = x
            q = cross.linear_q(decoder_layer.norm3(x)).view(-1, cross.h, cross.d_k).transpose(0, 1)
            k, v = torch.split(cross.linear_k_v(encoder_out), self.cif_hidden_size, dim=-1)
            k = k.half().view(-1, cross.h, cross.d_k).permute(1, 2, 0)                       # keep the key cache in float16
            v = v.half().view(-1, cross.h, cross.d_k).transpose(0, 1)                        # keep the value cache in float16
            k = torch.cat([all_inputs[layer_idx + self.cache_layer_num_de], k], dim=2)
            v = torch.cat([all_inputs[layer_idx + self.cache_layer_num_de_2], v], dim=1)
            self.save_keys_de[layer_idx] = k[:, :, -self.look_back_de:]
            self.save_values_de[layer_idx] = v[:, -self.look_back_de:]
            if self.compute_in_f32:
                context = torch.matmul(torch.softmax(torch.matmul(q, k.float()), dim=-1), v.float()).transpose(0, 1).reshape(1, -1, self.cif_hidden_size)   # cast the f16 cache back to f32 for the matmuls
            else:
                # minimum-cast: downcast Q to f16 and run the attention in f16 on the f16 cache, then cast the context back to f32.
                context = torch.matmul(torch.softmax(torch.matmul(q.half(), k), dim=-1), v).transpose(0, 1).reshape(1, -1, self.cif_hidden_size).float()
            list_frame = residual + cross.linear_out(context)
        decoder_layer = self.decoder.decoders3[0]
        ff = decoder_layer.feed_forward
        x = ff.w_2(ff.norm(ff.activation(ff.w_1(decoder_layer.norm1(list_frame)))))
        x = self.decoder.output_layer(self.decoder.after_norm(x))
        max_logit_ids = torch.argmax(x, dim=-1, keepdim=False).int()                      # (1, list_frame_len) int32 token id per fired CIF frame
        num_id = torch._shape_as_tensor(max_logit_ids)[1].to(torch.int32).unsqueeze(0)    # fixed shape [1]: count of decoded tokens
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
    # Save to text file
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in tokenizer.token_list:
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

    key_en = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, 0), dtype=torch.float16)        # float16 key cache
    value_en = torch.zeros((NUM_HEAD_EN, 0, HEAD_DIM_EN), dtype=torch.float16)       # float16 value cache
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
        onnx_model_Encoder,
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

    key_de = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float16)         # float16 key cache
    value_de = torch.zeros((NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float16)        # float16 value cache
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
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_de_fsmn_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus'}
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
        onnx_model_Decoder,
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
            "paraformer_streaming_metadata_version": 1,
            "producer": "Export_Paraformer_Streaming.py",
            "compute_in_f32": COMPUTE_IN_F32,
        },
        {
            "num_encoder_layers": NUM_LAYER_EN,
            "num_decoder_layers": NUM_LAYER_DE,
            "num_encoder_heads": NUM_HEAD_EN,
            "num_decoder_heads": NUM_HEAD_DE,
            "encoder_head_dim": HEAD_DIM_EN,
            "decoder_head_dim": HEAD_DIM_DE,
            "hidden_size": CIF_HIDDEN_SIZE,
            "feature_size": FEATURE_SIZE,
            "fsmn_hidden_size": FSMN_HIDDEN_SIZE,
            "fsmn_de_pad": FSMN_DE_PAD,
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
    _metadata_targets = [onnx_model_Metadata]
    _written, _skipped = [], []
    for _target in _metadata_targets:
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
    gc.collect()
print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_Paraformer_Streaming_ONNX.py ...')
subprocess.run(
    [sys.executable, str(Path(__file__).resolve().parent / "Inference_Paraformer_Streaming_ONNX.py"), "--onnx-folder", str(onnx_folder)],
    check=True,
)
