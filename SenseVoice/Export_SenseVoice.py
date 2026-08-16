import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torchaudio.compliance.kaldi as kaldi  # Used at export time to bake Kaldi's exact triangular mel filterbank as a constant.
from funasr import AutoModel

model_path = str(Path.home() / "Downloads" / "SenseVoiceSmall")                         # The SenseVoice download path.
onnx_folder = Path(__file__).resolve().parent / "SenseVoice_ONNX"                       # Local folder next to this script holding the exported ONNX graph; created automatically if missing.
onnx_folder.mkdir(parents=True, exist_ok=True)
onnx_model_Metadata = str(onnx_folder / "ASR_Metadata.onnx")      # Tiny metadata carrier graph.
onnx_model_A = str(onnx_folder / "SenseVoiceSmall.onnx")              # The exported onnx model path.


DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. Note that some providers only support static axes.
INPUT_AUDIO_LENGTH = 480000                                 # The maximum input audio length.
INPUT_AUDIO_DTYPE = "F32"                                   # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).
OPSET = 20                                                  # ONNX opset version.
WINDOW_TYPE = 'hamming'                                     # Type of window function used in the STFT (Kaldi uses the symmetric Hamming window).
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # FFT size: Kaldi zero-pads each 400-sample (25 ms) frame to the next power of two before the FFT.
WINDOW_LENGTH = 400                                         # Length of windowing (frame length, 25 ms), edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
LFR_M = 7                                                   # The model parameter, do not edit the value.
LFR_N = 6                                                   # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
USE_EMOTION = True                                          # Output the emotion tag or not.
USE_FUSED_SKIP_LAYER_NORM = False                           # CPU-only alternative: improves 30 s inputs but regresses representative 10 s inputs on this target; False keeps the portable graph.


# The selector index is the row in the exported seven-row language embedding table;
# prompt_token_id is the authoritative key in SenseVoiceSmall.model.embed/lid_dict.
SENSEVOICE_LANGUAGE_PROFILES = (
    ("auto", "Automatic language detection", ["automatic", "detect"], 0),
    ("zh", "Chinese", ["Chinese", "Mandarin", "zh-CN", "中文"], 3),
    ("en", "English", ["English", "en-US"], 4),
    ("yue", "Cantonese", ["Cantonese", "zh-yue", "粤语", "粵語"], 7),
    ("ja", "Japanese", ["Japanese", "jp", "日本語"], 11),
    ("ko", "Korean", ["Korean", "kr", "한국어"], 12),
    ("nospeech", "No speech", ["no-speech", "silence"], 13),
)
LANGUAGE_PROMPT_TOKEN_IDS = tuple(
    prompt_token_id
    for _, _, _, prompt_token_id in SENSEVOICE_LANGUAGE_PROFILES
)


STFT_SIGNAL_LENGTH = (INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1   # Number of fbank frames (Kaldi snip_edges=True framing).
LFR_LENGTH = (STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


DYNAMIC_AXES_MAP = {
    'audio': {2: 'audio_len'},                              # input-driven waveform sample count; affects Conv/LFR Slice lengths.
    'token_ids': {0: 'num_token'}                           # data-dependent final CTC token count after repeat/blank collapse.
} if DYNAMIC_AXES else None
# Dynamic-axis contract when DYNAMIC_AXES is enabled:
# audio axis 2 ('audio_len'): input-shape-driven, expected >= WINDOW_LENGTH and <= INPUT_AUDIO_LENGTH used at export;
# it drives Conv output length, bounded LFR Slice/Gather, and the encoder time dimension.
# token_ids axis 0 ('num_token'): data-dependent final CTC token count after repeat/blank collapse.
# num_id output is fixed shape [1] and stores the same dynamic token count; direct compact output introduces
# NonZero/Gather in ONNX, which is intentional for this export interface.
class METADATA_CARRIER(torch.nn.Module):
    def forward(self, marker):
        return marker


class _SKIP_LAYER_NORMALIZATION(torch.autograd.Function):
    EPSILON = 1e-5

    @staticmethod
    def forward(ctx, x, skip, gamma, beta):
        residual = x + skip
        normalized = torch.nn.functional.layer_norm(residual, gamma.shape, gamma, beta, _SKIP_LAYER_NORMALIZATION.EPSILON)
        return normalized, residual

    @staticmethod
    def symbolic(graph, x, skip, gamma, beta):
        outputs = graph.op(
            "com.microsoft::SkipLayerNormalization",
            x,
            skip,
            gamma,
            beta,
            epsilon_f=_SKIP_LAYER_NORMALIZATION.EPSILON,
            outputs=4,
        )
        outputs[0].setType(x.type())
        outputs[3].setType(x.type())
        return outputs[0], outputs[3]


class _INTEGER_DIVIDE(torch.autograd.Function):
    """Export non-negative integer division directly as ONNX Div.

    The legacy exporter otherwise inserts int64 -> float -> int64 Casts for
    floor/truncating division. Both operands here are non-negative int64 shape
    scalars, so ONNX integer Div has exactly the required floor semantics.
    """

    @staticmethod
    def forward(ctx, numerator, denominator):
        return torch.div(numerator, denominator, rounding_mode='trunc')

    @staticmethod
    def symbolic(graph, numerator, denominator):
        output = graph.op("Div", numerator, denominator)
        output.setType(numerator.type())
        return output


class SENSE_VOICE(torch.nn.Module):
    """Standalone, ONNX-export-friendly SenseVoiceSmall.

    Every sub-module forward (encoder, SANM self-attention, FSMN memory, position-wise
    feed-forward, LayerNorm and the sinusoidal position encoding) is re-implemented inline
    below, so the graph can be traced end-to-end without the './modeling_modified' package.
    Only the trained weights are taken from the loaded funasr model.
    """

    def __init__(self, sense_voice, cif_hidden_size, nfft_stft, win_length, hop_length, stft_signal_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, cmvn_means, cmvn_vars, use_emo, use_fused_skip_layer_norm):
        super(SENSE_VOICE, self).__init__()
        # ---- Modules carrying the trained weights (their forwards are inlined in this class) ----
        self.encoder = sense_voice.encoder                      # SenseVoiceEncoderSmall: encoders0 + encoders + tp_encoders + norms
        self.ctc_lo = sense_voice.ctc.ctc_lo                    # CTC output projection (Linear -> vocab)
        self.blank_id = sense_voice.blank_id

        # ---- Kaldi-faithful log-Mel front-end (numerically matches torchaudio.compliance.kaldi.fbank) ----
        # One Conv1d kernel folds, per frame: DC-offset removal, pre-emphasis (replicate boundary),
        # the symmetric Hamming window and the one-sided windowed DFT (nfft_stft-point frequency basis
        # sampled over the win_length-sample frame). Conv stride = hop_length with no padding reproduces
        # Kaldi's snip_edges=True framing: frame m spans samples [m * hop_length, m * hop_length + win_length).
        self.hop_length = hop_length
        self.fbank_freq = nfft_stft // 2 + 1
        window = torch.hamming_window(win_length, periodic=False, alpha=0.54, beta=0.46, dtype=torch.float32)
        freqs = torch.arange(self.fbank_freq, dtype=torch.float32).unsqueeze(1)               # (fbank_freq, 1)
        samples = torch.arange(win_length, dtype=torch.float32).unsqueeze(0)                  # (1, win_length)
        omega = (2.0 * torch.pi / nfft_stft) * freqs * samples
        cos_basis = torch.cos(omega) * window
        sin_basis = -torch.sin(omega) * window

        def fold_frontend(basis):
            shifted = torch.cat([basis[:, 1:], torch.zeros_like(basis[:, :1])], dim=1)        # basis[:, n + 1], zero past the frame edge
            folded = basis - pre_emphasis * shifted                                           # pre-emphasis: pf[n] = s[n] - c * s[n - 1]
            folded[:, 0] = folded[:, 0] - pre_emphasis * basis[:, 0]                          # replicate boundary: pf[0] = (1 - c) * s[0]
            return folded - folded.mean(dim=1, keepdim=True)                                  # per-frame DC removal (subtract the frame mean)

        self.register_buffer('fbank_kernel', torch.cat([fold_frontend(cos_basis), fold_frontend(sin_basis)], dim=0).unsqueeze(1).contiguous(), persistent=True)  # (2 * fbank_freq, 1, win_length)
        self.log_eps = float(torch.finfo(torch.float32).eps)                                  # Kaldi's log floor (FLT_EPSILON)

        # Kaldi triangular mel filterbank (low_freq = 20 Hz, high_freq = Nyquist) over the
        # nfft_stft spectrum, padded with a zero Nyquist column and stored as (fbank_freq, n_mels).
        mel_banks, _ = kaldi.get_mel_banks(n_mels, nfft_stft, float(sample_rate), 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer('mel_filters', torch.nn.functional.pad(mel_banks, (0, 1), value=0.0).transpose(0, 1).contiguous(), persistent=True)

        self.lfr_n = lfr_n
        self.lfr_m_factor = (lfr_m - 1) // 2
        self.T_lfr = lfr_len
        self.feature_size = n_mels * lfr_m                     # LFR-stacked feature width (= 560); static Reshape target in forward()
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int64).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int64) - self.lfr_m_factor
        # Match ONNX's int64 shape arithmetic so the dynamic upper clamp needs no run-time Cast.
        self.register_buffer('indices_mel', indices.clamp(min=0).contiguous(), persistent=True)
        self.register_buffer('lfr_divisor', torch.tensor(lfr_n, dtype=torch.int64), persistent=True)

        # ---- Pre-computed prompt embeddings prepended to the speech features ----
        embed_sys = sense_voice.embed
        system_embed = embed_sys(torch.tensor([1, 2, 14], dtype=torch.int32)) if use_emo else embed_sys(torch.tensor([5, 14], dtype=torch.int32))
        language_embed = embed_sys(torch.tensor(LANGUAGE_PROMPT_TOKEN_IDS, dtype=torch.int32)).half().float()  # Preserve half-rounded values, stored in their final graph dtype.

        # ---- SANM self-attention geometry (identical for every encoder block) ----
        first_attn = self.encoder.encoders0[0].self_attn
        self.num_head = first_attn.h                            # number of attention heads (4)
        self.head_dim = first_attn.d_k                          # per-head dimension (128)
        self.hidden_size = cif_hidden_size                      # encoder hidden size (512 = num_head * head_dim)
        self.use_fused_skip_layer_norm = use_fused_skip_layer_norm
        kernel_size = first_attn.fsmn_block.kernel_size[0]      # FSMN depth-wise conv length (11)
        fsmn_left_pad = (kernel_size - 1) // 2                  # symmetric zero padding for the FSMN conv (5 each side)

        # ---- Per-encoder-block folds applied once in __init__, so forward() stays a clean tensor-only path ----
        self._prepare_sanm_for_export(cif_hidden_size, fsmn_left_pad)

        # ---- Pre-computed sinusoidal position encoding (depth = feature_size = 560) ----
        feature_size = self.feature_size                       # encoder input feature size (= 560 = n_mels * lfr_m)
        prompt_length = 1 + system_embed.shape[0]
        positions = torch.arange(1, lfr_len + prompt_length + 1, dtype=torch.float32)
        log_timescale_increment = torch.log(torch.tensor([10000.0], dtype=torch.float32)) / (feature_size / 2 - 1)
        inv_timescales = torch.exp(torch.arange(feature_size / 2, dtype=torch.float32) * (-log_timescale_increment)).reshape(1, -1)
        scaled_time = positions.reshape(-1, 1) * inv_timescales
        position_encoding = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1).half().float()

        # Prompt positions are invariant and can be baked into their embeddings exactly. Speech
        # retains the checkpoint's original (features + mean) * scale + position evaluation order;
        # folding this affine chain exceeded the strict long-sequence internal-logit tolerance.
        language_embed = language_embed + position_encoding[:1]
        system_embed = system_embed + position_encoding[1:prompt_length]
        self.register_buffer('language_embed', language_embed.detach().contiguous(), persistent=True)
        self.register_buffer('system_embed', system_embed.detach().contiguous(), persistent=True)
        self.register_buffer('cmvn_means', cmvn_means.reshape(1, feature_size).contiguous(), persistent=True)
        self.register_buffer('cmvn_vars', cmvn_vars.reshape(1, feature_size).contiguous(), persistent=True)
        self.register_buffer('speech_position', position_encoding[prompt_length:].contiguous(), persistent=True)

    def _prepare_sanm_for_export(self, cif_hidden_size, fsmn_left_pad):
        """Migrate an already-loaded FunASR checkpoint instance to its export-only parameter layout."""
        factor = float(self.head_dim ** (-0.25))               # 1/sqrt(head_dim), split evenly across q and k
        encoder_layers = list(self.encoder.encoders0) + list(self.encoder.encoders) + list(self.encoder.tp_encoders)
        with torch.no_grad():
            for encoder_layer in encoder_layers:
                attn = encoder_layer.self_attn
                attn.linear_q_k_v.weight[:-cif_hidden_size].mul_(factor)  # scale q & k rows only
                attn.linear_q_k_v.bias[:-cif_hidden_size].mul_(factor)
                attn.fsmn_block.weight[:, 0, fsmn_left_pad].add_(1.0)     # fold fsmn(v) + v into the center tap
                attn.fsmn_block.bias = torch.nn.Parameter(attn.linear_out.bias.detach().clone(), requires_grad=False)
                attn.linear_out.bias = None                              # Conv carries the shared parallel-branch bias
                attn.fsmn_block.padding = (fsmn_left_pad,)                # fold symmetric padding into the Conv op

    @staticmethod
    def layer_norm(x, norm):
        # funasr LayerNorm == standard affine LayerNorm evaluated in float32 (input is already float32 here).
        return torch.nn.functional.layer_norm(x, norm.normalized_shape, norm.weight, norm.bias, norm.eps)

    def sanm_block(self, x, layer):
        """One SANM encoder block: multi-head self-attention + FSMN memory + position-wise feed-forward."""
        # Keep the encoder sequence as (time, hidden), making every biased Linear export as one
        # standard Gemm instead of MatMul + Add. Reshape and transpose q|k|v jointly, then split,
        # so 70 blocks share one layout conversion rather than three independent conversions each.
        qkv = layer.self_attn.linear_q_k_v(self.layer_norm(x, layer.norm1))
        qkv = qkv.reshape(-1, 3, self.num_head, self.head_dim).permute(1, 2, 0, 3)
        q, k, v = torch.split(qkv, 1, dim=0)                              # each: (1, head, time, dim)
        k = k.transpose(-2, -1)                                          # (1, head, dim, time)
        # --- scaled dot-product attention (the 1/sqrt(head_dim) scale is already folded into q & k) ---
        context = torch.matmul(torch.softmax(torch.matmul(q, k), dim=-1), v)               # (1, head, time, dim)
        context = context.permute(0, 2, 1, 3).reshape(-1, self.hidden_size)                 # (time, hidden)
        # --- FSMN memory: depth-wise conv over the value (symmetric zero-pad folded into the Conv), plus the value itself ---
        fsmn_input = v.permute(0, 1, 3, 2).reshape(1, self.hidden_size, -1)
        fsmn_memory = layer.self_attn.fsmn_block(fsmn_input).transpose(1, 2).reshape(-1, self.hidden_size)
        # linear_out has no bias after _prepare_sanm_for_export(); supplying the FSMN matrix as
        # F.linear's bias lowers to Gemm.C and removes a separate full-sequence Add.
        attention_branch = torch.nn.functional.linear(context, layer.self_attn.linear_out.weight, fsmn_memory)
        # --- residual added only when in/out sizes match (encoders0[0] projects 560 -> 512, so it is skipped there) ---
        if layer.in_size == layer.size:
            if self.use_fused_skip_layer_norm:
                normalized_attention, attention = _SKIP_LAYER_NORMALIZATION.apply(
                    attention_branch, x, layer.norm2.weight, layer.norm2.bias
                )
            else:
                attention = attention_branch + x
                normalized_attention = self.layer_norm(attention, layer.norm2)
        else:
            attention = attention_branch
            normalized_attention = self.layer_norm(attention, layer.norm2)
        # --- position-wise feed-forward: Linear -> ReLU -> Linear, added as a residual ---
        return attention + layer.feed_forward.w_2(torch.relu(layer.feed_forward.w_1(normalized_attention)))

    def encode(self, xs):
        """Inlined SenseVoiceEncoderSmall.forward over the canonical (time, hidden) layout."""
        for layer in self.encoder.encoders0:                   # first block: projects feature_size (560) -> hidden (512), no residual
            xs = self.sanm_block(xs, layer)
        for layer in self.encoder.encoders:                    # remaining encoder blocks
            xs = self.sanm_block(xs, layer)
        xs = self.layer_norm(xs, self.encoder.after_norm)
        for layer in self.encoder.tp_encoders:                 # transformer-postnet blocks
            xs = self.sanm_block(xs, layer)
        return self.layer_norm(xs, self.encoder.tp_norm)

    def forward(self, audio, language_idx):
        # ===== 1. Kaldi-style log-Mel filterbank =====
        # The Hamming window, per-frame DC removal, per-frame pre-emphasis and the windowed DFT are all
        # folded into fbank_kernel, so one Conv1d turns the raw waveform into the one-sided complex spectrum.
        spectrum = torch.nn.functional.conv1d(audio.float(), self.fbank_kernel, stride=self.hop_length)
        real_power, imag_power = torch.split(spectrum * spectrum, self.fbank_freq, dim=1)     # one square over the 514 channels, then split (== real^2 / imag^2)
        power = (real_power + imag_power).transpose(1, 2)                                     # (1, frames, fbank_freq)
        mel_features = torch.matmul(power, self.mel_filters).clamp(min=self.log_eps).log()    # (1, frames, n_mels)
        # ===== 2. Low-Frame-Rate (LFR) stacking =====
        num_frames = torch._shape_as_tensor(mel_features)[1]
        _len = _INTEGER_DIVIDE.apply(num_frames + self.lfr_n - 1, self.lfr_divisor)
        lfr_indices = torch.minimum(self.indices_mel[:_len], num_frames - 1)
        mel_features = mel_features[:, lfr_indices].reshape(-1, self.feature_size)
        # ===== 3. CMVN + speech position, then prepend position-folded prompts =====
        mel_features = (mel_features + self.cmvn_means) * self.cmvn_vars
        mel_features = mel_features + self.speech_position[:_len]
        mel_features = torch.cat([self.language_embed[language_idx], self.system_embed, mel_features], dim=0)
        encoder_out = self.encode(mel_features)
        # ===== 4. CTC head + greedy collapse (drop repeats and blanks) =====
        token_ids = self.ctc_lo(encoder_out).argmax(dim=-1)                                 # keep ArgMax int64 through collapse
        shifted_tensor = torch.cat([token_ids[1:], token_ids[[0]]], dim=0)
        token_keep_mask = (token_ids != shifted_tensor) & (token_ids != self.blank_id)
        keep_indices = torch.nonzero(token_keep_mask, as_tuple=False).squeeze(1)
        token_ids = torch.index_select(token_ids, 0, keep_indices).int()                    # cast only the compact public output to int32
        num_id = torch._shape_as_tensor(token_ids).to(torch.int32)
        return token_ids, num_id


def build_supported_languages():
    return {
        code: {
            "name": name,
            "aliases": aliases,
            "selector_index": selector_index,
            "prompt_token_ids": [prompt_token_id],
        }
        for selector_index, (code, name, aliases, prompt_token_id) in enumerate(
            SENSEVOICE_LANGUAGE_PROFILES
        )
    }
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


print('\nExport start ...\n')
with torch.inference_mode():
    model = AutoModel(
        model=model_path,
        disable_update=True,
        device="cpu"
    )  # Loads the built-in funasr SenseVoiceSmall; every forward() is re-implemented (inlined) in SENSE_VOICE below, so no './modeling_modified' is needed.
    encoder_output_size_factor = (model.model.encoder.output_size()) ** 0.5
    model.model.embed.weight.data *= encoder_output_size_factor
    CMVN_MEANS = model.kwargs['frontend'].cmvn[0].repeat(1, 1, 1)
    CMVN_VARS = (model.kwargs['frontend'].cmvn[1] * encoder_output_size_factor).repeat(1, 1, 1)
    CIF_HIDDEN_SIZE = model.model.encoder.encoders0._modules["0"].size
    supported_languages = build_supported_languages()
    sense_voice = SENSE_VOICE(model.model.eval(), CIF_HIDDEN_SIZE, NFFT_STFT, WINDOW_LENGTH, HOP_LENGTH, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, CMVN_MEANS, CMVN_VARS, USE_EMOTION, USE_FUSED_SKIP_LAYER_NORM)
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype)
    language_idx = torch.tensor([0], dtype=torch.int32)
    torch.onnx.export(
        sense_voice,
        (audio, language_idx),
        onnx_model_A,
        input_names=['audio', 'language_idx'],
        output_names=['token_ids', 'num_id'],
        export_params=True,
        do_constant_folding=True,
        dynamic_axes=DYNAMIC_AXES_MAP,
        keep_initializers_as_inputs=False,
        training=torch.onnx.TrainingMode.EVAL,
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
            "supported_languages": supported_languages,
        },
    )
    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    del model
    del sense_voice
    del audio
    del language_idx
    del CMVN_VARS
    del CMVN_MEANS
    gc.collect()

# ── Copy the SentencePiece tokenizer model into the ONNX folder so the exported folder runs ──
# inference stand-alone (no external SenseVoice download path needed at inference time).
for _asset in ("chn_jpn_yue_eng_ko_spectok.bpe.model",):
    _src = Path(model_path) / _asset
    _dst = onnx_folder / _asset
    shutil.copy2(_src, _dst)
    print(f"[Tokenizer] Copied {_asset} -> {onnx_folder}")

print('\nExport done!\n')
subprocess.run(
    [
        sys.executable,
        str(onnx_folder.parent / "Inference_SenseVoice_ONNX.py"),
        "--onnx-folder",
        str(onnx_folder),
    ],
    cwd=str(onnx_folder.parent),
    check=True,
)
