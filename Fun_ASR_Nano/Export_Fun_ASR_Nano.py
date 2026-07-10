import os
os.environ["HF_HUB_OFFLINE"] = "1"          # Load all HuggingFace assets from local disk; never contact huggingface.co
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # Disable transformers' network HEAD/update checks

import gc
import shutil
import subprocess
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from funasr import AutoModel
from funasr.register import tables
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from STFT_Process import STFT_Process


model_path                     = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512'            # Set the path where the [Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512] downloaded. URL: https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512 / https://modelscope.cn/models/FunAudioLLM/Fun-ASR-MLT-Nano-2512
tokenizer_path                 = r'/home/DakeQQ/Downloads/Fun-ASR-Nano-2512/Qwen3-0.6B' # Set the tokenizer path.

# Store (and later load) the exported ONNX models in a local folder next to this script; created automatically if missing.
onnx_folder                    = Path(__file__).resolve().parent / "Fun_ASR_Nano_ONNX"  # Local folder holding all exported ONNX graphs.
onnx_folder.mkdir(parents=True, exist_ok=True)

onnx_model_Metadata            = str(onnx_folder / "ASR_Matadata.onnx")         # Tiny metadata carrier graph.
onnx_model_Encoder             = str(onnx_folder / "FunASR_Nano_Encoder.onnx")          # The exported onnx model path.
onnx_model_CTC_Decoder         = str(onnx_folder / "FunASR_Nano_CTC_Decoder.onnx")      # Optional fast CTC transcription head; exported & loaded only when USE_CTC_DECODER=True.
onnx_model_Embed               = str(onnx_folder / "FunASR_Nano_Decoder_Embed.onnx")
onnx_model_Main                = str(onnx_folder / "FunASR_Nano_Decoder_Main.onnx")
onnx_model_Rotary_Mask_Prefill = str(onnx_folder / "FunASR_Nano_Rotary_Mask_Text_Prefill.onnx")
onnx_model_Rotary_Mask_Decode  = str(onnx_folder / "FunASR_Nano_Rotary_Mask_Text_Decode.onnx")
onnx_model_Greedy              = str(onnx_folder / "FunASR_Nano_Greedy_Search.onnx")
onnx_model_First_Beam          = str(onnx_folder / "FunASR_Nano_First_Beam_Search.onnx")
onnx_model_Second_Beam         = str(onnx_folder / "FunASR_Nano_Second_Beam_Search.onnx")
onnx_model_Penalty             = str(onnx_folder / "FunASR_Nano_Apply_Penalty.onnx")
onnx_model_Argmax              = str(onnx_folder / "FunASR_Nano_Argmax.onnx")



# Audio & STFT Configuration
SAMPLE_RATE                   = 16000       # The model parameter, do not edit the value.
WINDOW_TYPE                   = 'hamming'   # Type of window function used in the STFT. Kaldi WavFrontend uses a symmetric Hamming window.
N_MELS                        = 80          # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT                     = 512         # FFT size. Kaldi rounds the 400-sample window up to the next power of two (round_to_power_of_two).
WINDOW_LENGTH                 = 400         # Kaldi frame length (25 ms = 400 samples). Zero-padded to NFFT_STFT before the DFT.
HOP_LENGTH                    = 160         # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE                 = 0.97        # Kaldi per-frame pre-emphasis coefficient (baked into the STFT kernel).
INPUT_AUDIO_DTYPE             = "INT16"     # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. Kaldi fbank works on the int16 numeric range, so "F32"/"F16" carry int16-range values (no ÷32768).

# Model Parameters
LFR_M                         = 7                 # The model parameter, do not edit the value.
LFR_N                         = 6                 # The model parameter, do not edit the value.
STOP_TOKEN                    = [151643, 151645]  # The stop_id in Qwen is "151643" & "151645"
MAX_SEQ_LEN                   = 1024              # The max context length.
USE_FP16_KV                   = True              # Use fp16 KV cache + minimum-cast f16 attention (q/k/v/mask/softmax in f16; only the context is cast back to f32 for o_proj).
COMPUTE_IN_F32                = False             # F16-KV compute precision. False = minimum-cast f16 attention (above). True = keep the f16 KV *storage* (cache I/O dtype unchanged) but upcast K/V (and the mask, internally) to f32 at the attention use points, Q/softmax in f32 (f16 storage, f32 compute). No effect when USE_FP16_KV=False.
USE_CTC_DECODER               = True              # If True, the Encoder graph also emits the fast CTC transcription (greedy-collapsed token ids). If False, the CTC decoder & head are excluded to shrink the model and cut computation.

# Weight-Quantization-Friendly Reorder (EXACT, zero runtime cost; helps only when weight-quant group_size < head_dim)
REORDER_DOWNPROJ_FOR_QUANT    = True       # Reorder MLP intermediate channels so down_proj block-quant groups are magnitude-homogeneous (absorbed into gate_up + down_proj).
REORDER_OPROJ_FOR_QUANT       = True       # Reorder each head's head_dim so o_proj sub-head groups are homogeneous (compensated on the qkv v-rows). Pure win for f16 KV.
REORDER_KEY                   = "absmean"  # Channel key: "absmean" (robust; best at group=32) | "L4" (best at group=128) | "rms" | "std".

# Input & Processing Limits
MAX_INPUT_AUDIO_LENGTH        = 480000  # The maximum input audio length.
DYNAMIC_AXES                  = True    # The default dynamic_axes is the input audio length. Note that some providers only support static axes.

# Decoding Strategy
USE_BEAM_SEARCH                = False  # Use beam search or greedy search. It recommended to use greedy search for Fun-ASR-Nano.
TOP_K                          = 3      # The top k candidate in decoding.
BEAM_SIZE                      = 3      # Number of beams in searching.
PENALTY_RANGE                  = 10     # Penalizes the most recent output. "10" means the last 10 tokens.
MAX_BEAM_SIZE                  = 10     # Max beams for exported model.
REPEAT_PENALTY                 = 1.0    # Range from 0.0 to 1.0; "1.0" means no penalty.

# Runtime & Export Settings
OPSET                        = 20  # ONNX Runtime opset version.


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


MAX_STFT_SIGNAL_LENGTH = (MAX_INPUT_AUDIO_LENGTH - WINDOW_LENGTH) // HOP_LENGTH + 1   # Kaldi snip_edges frame count (no centering)
LFR_LENGTH = (MAX_STFT_SIGNAL_LENGTH + LFR_N - 1) // LFR_N
if HOP_LENGTH > MAX_INPUT_AUDIO_LENGTH:
    HOP_LENGTH = MAX_INPUT_AUDIO_LENGTH


# ══════════════════════════════════════════════════════════════════════════════
# Inlined FunASR-Nano modeling code (standalone — no external `modeling_modified`).
# Only the module-tree construction (`__init__`) is required so that funasr's
# `AutoModel` can build the graph and load the `model.pt` weights; every forward
# pass is re-implemented (optimized) by the export wrappers further below.
# ══════════════════════════════════════════════════════════════════════════════
class CTC(nn.Module):
    """CTC module (inlined from Fun-ASR `ctc.py`)."""

    def __init__(self, odim, encoder_output_size, dropout_rate=0.0, reduce=True, blank_id=0, **kwargs):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.blank_id = blank_id
        self.ctc_loss = torch.nn.CTCLoss(reduction="none", blank=blank_id)
        self.reduce = reduce

    def log_softmax(self, hs_pad):
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


@tables.register("model_classes", "FunASRNano")
class FunASRNano(nn.Module):
    """Fun-ASR-Nano model builder (inlined from the official `model.py`).

    Only `__init__` (module construction) and the trivial `encode` helper are kept;
    the training / inference methods are intentionally dropped because the ONNX
    export re-implements every forward pass in the dedicated wrapper modules.
    """

    def __init__(
        self,
        audio_encoder: str = None,
        audio_encoder_conf: dict = None,
        audio_adaptor: str = None,
        audio_adaptor_conf: dict = None,
        llm: str = None,
        llm_conf: dict = None,
        input_size: int = 80,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        # audio encoder
        hub = audio_encoder_conf.get("hub", None)
        self.audio_encoder_activation_checkpoint = audio_encoder_conf.get("activation_checkpoint", False)
        if hub == "ms":
            model = AutoModel(model=audio_encoder, model_revision="master")
            audio_encoder_output_size = (
                model.model.encoder_output_size if hasattr(model.model, "encoder_output_size") else -1
            )
            audio_encoder = (
                model.model.model.encoder if hasattr(model.model, "model") else model.model.encoder
            )
        else:
            encoder_class = tables.encoder_classes.get(audio_encoder)
            audio_encoder = encoder_class(input_size=input_size, **audio_encoder_conf)
            audio_encoder_output_size = audio_encoder.output_size()
        freeze = audio_encoder_conf.get("freeze", True)
        if freeze:
            for _, param in audio_encoder.named_parameters():
                param.requires_grad = False
            audio_encoder.eval()
        self.audio_encoder = audio_encoder

        # llm
        self.llm = None
        init_param_path = llm_conf.get("init_param_path", None)
        llm_dim = None
        llm_load_kwargs = llm_conf.get("load_kwargs", {})
        config = AutoConfig.from_pretrained(init_param_path)
        model = AutoModelForCausalLM.from_config(config, **llm_load_kwargs)
        freeze = llm_conf.get("freeze", True)
        if freeze:
            for _, param in model.named_parameters():
                param.requires_grad = False
            model.eval()
        if llm_conf.get("activation_checkpoint", False):
            model.gradient_checkpointing_enable()
        self.llm = model.to(torch.float32)
        llm_dim = model.get_input_embeddings().weight.shape[-1]

        # adaptor
        adaptor_class = tables.adaptor_classes.get(audio_adaptor)
        if audio_encoder_output_size > 0:
            audio_adaptor_conf["encoder_dim"] = audio_encoder_output_size
        audio_adaptor_conf["llm_dim"] = (llm_dim if llm_dim is not None else audio_adaptor_conf["llm_dim"])
        audio_adaptor = adaptor_class(**audio_adaptor_conf)
        freeze = audio_adaptor_conf.get("freeze", False)
        if freeze:
            for _, param in audio_adaptor.named_parameters():
                param.requires_grad = False
            audio_adaptor.eval()
        self.audio_adaptor = audio_adaptor
        self.use_low_frame_rate = audio_adaptor_conf.get("use_low_frame_rate", False)

        # ctc decoder (present when the checkpoint ships CTC weights)
        self.ctc_decoder = None
        ctc_decoder_class = tables.adaptor_classes.get(kwargs.get("ctc_decoder", None))
        if ctc_decoder_class is not None:
            ctc_tokenizer = (
                kwargs.get("ctc_tokenizer", None)
                if "ctc_tokenizer" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer"]
            )
            ctc_tokenizer_conf = (
                kwargs.get("ctc_tokenizer_conf", None)
                if "ctc_tokenizer_conf" in kwargs
                else kwargs["dataset_conf"]["ctc_tokenizer_conf"]
            )
            if ctc_tokenizer is not None and ctc_tokenizer_conf is not None:
                ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer)
                ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)
                self.ctc_tokenizer = ctc_tokenizer
            assert ctc_tokenizer is not None, f"ctc_tokenizer must be set"

            ctc_vocab_size = kwargs.get("ctc_vocab_size", 60515)
            ctc_decoder_conf = kwargs.get("ctc_decoder_conf", {})
            if audio_encoder_output_size > 0:
                ctc_decoder_conf["encoder_dim"] = audio_encoder_output_size
            self.ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
            init_param_path = ctc_decoder_conf.get("init_param_path", None)
            if init_param_path is not None:
                src_state = torch.load(init_param_path, map_location="cpu")
                self.ctc_decoder.load_state_dict(src_state, strict=False)
            freeze = ctc_decoder_conf.get("freeze", False)
            if freeze:
                for _, param in self.ctc_decoder.named_parameters():
                    param.requires_grad = False
                self.ctc_decoder.eval()

            ctc_conf = kwargs.get("ctc_conf", {})
            self.blank_id = ctc_conf.get("blank_id", ctc_vocab_size - 1)
            self.ctc_weight = kwargs.get("ctc_weight", 0.3)
            self.ctc = CTC(
                odim=ctc_vocab_size,
                encoder_output_size=audio_encoder_output_size,
                blank_id=self.blank_id,
                **ctc_conf,
            )
            self.detach_ctc_decoder = kwargs.get("detach_ctc_decoder", True)
            self.error_calculator = None

        self.length_normalized_loss = length_normalized_loss

    def encode(self, speech, speech_lengths):
        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        return encoder_out, encoder_out_lens


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: select the token with the highest logit."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]

        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=True)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)

        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx
        )


# ══════════════════════════════════════════════════════════════════════════════
# Penalty & Utility Modules
# ══════════════════════════════════════════════════════════════════════════════
class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, marker):
        return marker


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask
# ══════════════════════════════════════════════════════════════════════════════
class ROTARY_MASK_PREFILL(torch.nn.Module):
    """Precompute rotary embeddings and causal mask for the prefill phase."""

    def __init__(self, llm, max_seq_len):
        super().__init__()

        # Mask dtype = the KV-cache dtype, so the exported mask I/O dtype is stable: float16 with the f16 KV
        # cache, float32 with a float32 KV cache. Deliberately NOT gated on COMPUTE_IN_F32 -- the f16-storage
        # f32-compute path upcasts this f16 mask to f32 INTERNALLY (mask I/O + inference runtime unchanged).
        # It is added to the attention scores in FUNASR_NANO_DECODER_MAIN.
        self.mask_dtype = torch.float16 if USE_FP16_KV else torch.float32

        # Causal attention mask: upper triangle → -128
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128

        # Precompute rotary embeddings
        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    @staticmethod
    def _build_rotary_table(llm, max_seq_len):
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(idx_theta), torch.sin(idx_theta)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].to(self.mask_dtype)
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_MASK_DECODE(torch.nn.Module):
    """Provide rotary embeddings for a single decode step."""

    def __init__(self, llm, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Encoder Module (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class FUNASR_NANO_ENCODER(torch.nn.Module):
    def __init__(self, funasr_nano, stft_model, nfft_stft, max_stft_len, n_mels, sample_rate, pre_emphasis, lfr_m, lfr_n, lfr_len, _tokenizer, use_ctc_decoder=False):
        super(FUNASR_NANO_ENCODER, self).__init__()
        self.funasr_nano = funasr_nano.float()
        self.use_ctc_decoder = use_ctc_decoder
        self._replace_gelu_with_tanh_approximation(self.funasr_nano)
        self.stft_model = stft_model
        self.T_lfr = lfr_len
        self.lfr_n = lfr_n
        # Mel filterbank matching Kaldi's get_mel_banks (exactly what WavFrontend uses
        # through kaldi.fbank); falls back to torchaudio.melscale_fbanks if unavailable.
        try:
            from torchaudio.compliance.kaldi import get_mel_banks
            mel_banks = get_mel_banks(n_mels, nfft_stft, sample_rate, 20.0, 0.0, 100.0, -500.0, 1.0)[0]  # (n_mels, nfft_stft // 2)
            mel_banks = torch.nn.functional.pad(mel_banks, (0, 1))                                        # zero Nyquist -> (n_mels, nfft_stft // 2 + 1)
            self.fbank = mel_banks.unsqueeze(0).to(torch.float32)
        except Exception:
            self.fbank = (torchaudio.functional.melscale_fbanks(nfft_stft // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, None, 'htk')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.lfr_m_factor = (lfr_m - 1) // 2
        indices = torch.arange(0, self.T_lfr * lfr_n, lfr_n, dtype=torch.int32).unsqueeze(1) + torch.arange(lfr_m, dtype=torch.int32)
        self.indices_mel = indices.clamp(max=max_stft_len + self.lfr_m_factor - 1).to(torch.int16)
        self.output_size_factor = self.funasr_nano.audio_encoder.output_size() ** 0.5
        self.variance_epsilon = torch.tensor([1.1920928955078125e-07], dtype=torch.float32)  # torch.finfo(float32).eps == Kaldi's fbank log floor
        self.position_encoding = self.funasr_nano.audio_encoder.embed(torch.zeros([1, max_stft_len, 560], dtype=torch.float32))
        num_head = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.h
        head_dim = self.funasr_nano.audio_encoder.encoders._modules["0"].self_attn.d_k
        self.pad_zeros = torch.zeros((1, num_head * head_dim, 5), dtype=torch.float32)
        scale_factor = head_dim ** (-0.25)
        self.total_encoders = list(self.funasr_nano.audio_encoder.encoders0) + list(self.funasr_nano.audio_encoder.encoders) + list(self.funasr_nano.audio_encoder.tp_encoders)
        in_size = self.funasr_nano.audio_encoder.encoders._modules["0"].in_size
        for encoder_layer in self.total_encoders:
            encoder_layer.self_attn.linear_q_k_v.weight.data[:-in_size] *= scale_factor
            encoder_layer.self_attn.linear_q_k_v.bias.data[:-in_size] *= scale_factor

            # Fuse encoder_layer.norm1 into encoder_layer.self_attn.linear_q_k_v
            norm = encoder_layer.norm1
            linear = encoder_layer.self_attn.linear_q_k_v

            # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters in LayerNorm so it only performs (x - mu) / sigma
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

            # Fuse encoder_layer.norm2 into encoder_layer.feed_forward.w_1
            norm = encoder_layer.norm2
            linear = encoder_layer.feed_forward.w_1

            # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters in LayerNorm
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

        self._fuse_adaptor_blocks(self.funasr_nano.audio_adaptor.blocks)

        # Fold audio_encoder.tp_norm's affine into audio_adaptor.linear1 so tp_norm only performs
        # (x - mu) / sigma at runtime (its output `enc_normed` is therefore affine-free). When the
        # CTC head is enabled, FUNASR_NANO_CTC_DECODER folds the SAME affine into its own
        # ctc_decoder.linear1; it is built BEFORE this encoder so the affine is still present there.
        norm = self.funasr_nano.audio_encoder.tp_norm
        linear = self.funasr_nano.audio_adaptor.linear1
        # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
        # 3. Disable affine parameters in LayerNorm so it only performs (x - mu) / sigma
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias = None

        head_ids = _tokenizer.encode("<|im_start|>user\n", return_tensors="pt")
        tail_ids = _tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")
        self.head_embed = self.funasr_nano.llm.model.embed_tokens(head_ids)
        self.tail_embed = self.funasr_nano.llm.model.embed_tokens(tail_ids)
        self.fake_token = torch.zeros(max_stft_len + 1, dtype=torch.int16)
        for i in range(self.fake_token.shape[0]):
            self.fake_token[i] = (((i - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    @staticmethod
    def _fuse_adaptor_blocks(blocks):
        """Fuse a Transformer-adaptor block stack (audio_adaptor / ctc_decoder) for fast inference:
        fold head_dim^-0.25 into q & k, merge q/k/v into a single linear, and absorb
        norm1 -> linear_q_k_v and norm2 -> feed_forward.w_1 (LayerNorms left affine-free)."""
        head_dim = blocks._modules["0"].self_attn.d_k
        factor = float(head_dim ** (-0.25))
        for block in blocks:
            block.self_attn.linear_q.weight.data *= factor
            block.self_attn.linear_q.bias.data *= factor
            block.self_attn.linear_k.weight.data *= factor
            block.self_attn.linear_k.bias.data *= factor

            # Fusing q, k, v
            in_features = block.self_attn.linear_q.in_features
            out_features = block.self_attn.linear_q.out_features + block.self_attn.linear_k.out_features + block.self_attn.linear_v.out_features
            block.self_attn.linear_q_k_v = torch.nn.Linear(in_features, out_features, bias=True)
            block.self_attn.linear_q_k_v.weight.data = torch.cat([block.self_attn.linear_q.weight.data, block.self_attn.linear_k.weight.data, block.self_attn.linear_v.weight.data], dim=0)
            block.self_attn.linear_q_k_v.bias.data = torch.cat([block.self_attn.linear_q.bias.data, block.self_attn.linear_k.bias.data, block.self_attn.linear_v.bias.data], dim=0)
            block.self_attn.size = out_features // 3
            del block.self_attn.linear_q
            del block.self_attn.linear_k
            del block.self_attn.linear_v

            # Fuse block.norm1 into block.self_attn.linear_q_k_v
            norm = block.norm1
            linear = block.self_attn.linear_q_k_v

            # 1. Update Bias: b_new = b_lin + W_lin @ b_norm
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight: W_new = W_lin * w_norm (broadcasting w_norm over rows of W_lin)
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters in LayerNorm so it only performs (x - mu) / sigma
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

            # Fuse block.norm2 into block.feed_forward.w_1
            norm = block.norm2
            linear = block.feed_forward.w_1

            # 1. Update Bias
            linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))

            # 2. Update Weight
            linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

            # 3. Disable affine parameters
            norm.elementwise_affine = False
            norm.weight = None
            norm.bias = None

    def forward(self, audio, query_embed):
        audio = audio.float()
        # Per-frame DC removal + pre-emphasis + symmetric Hamming window + zero-pad to
        # NFFT_STFT are all baked into the Kaldi-compatible STFT kernel (see STFT_Process).
        real_part, imag_part = self.stft_model(audio)
        mel_features = torch.maximum(torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2), self.variance_epsilon).log() * self.output_size_factor
        features_len = mel_features.shape[1].unsqueeze(0)
        left_padding = mel_features[:, [0]]
        padded_inputs = torch.cat([left_padding] * self.lfr_m_factor + [mel_features], dim=1)
        _len = features_len // self.lfr_n - 1
        # Number of speech tokens given to the LLM must equal FunASR's fake_token_len,
        # which is computed from the LFR frame count T_lfr = ceil(mel_frames / lfr_n)
        # (WavFrontend.apply_lfr), NOT the raw mel-frame count. Indexing fake_token by
        # features_len (mel frames) over-feeds ~lfr_n x too many speech tokens.
        lfr_len = (features_len + self.lfr_n - 1) // self.lfr_n
        mel_features = padded_inputs[:, self.indices_mel[:_len].int()].reshape(1, _len, -1)
        x = mel_features + self.position_encoding[:, :_len].float()
        for encoder_layer in self.funasr_nano.audio_encoder.encoders0 + self.funasr_nano.audio_encoder.encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0, 3)
            q_h, k_h, v_h = qkv.split([1, 1, 1], dim=0)
            v_fsmn = v_h.transpose(-1, -2).reshape(1, encoder_layer.size, -1)
            fsmn_in = torch.cat([self.pad_zeros, v_fsmn, self.pad_zeros], dim=-1)
            fsmn_out = encoder_layer.self_attn.fsmn_block(fsmn_in)
            fsmn_memory = (fsmn_out + v_fsmn).transpose(1, 2).reshape(1, -1, encoder_layer.size)
            attn = torch.matmul(q_h, k_h.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v_h).transpose(1, 2).reshape(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn_out = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            if encoder_layer.in_size == encoder_layer.size:
                x = x + attn_out
            else:
                x = attn_out
            x = x + encoder_layer.feed_forward.w_2(encoder_layer.feed_forward.activation(encoder_layer.feed_forward.w_1(encoder_layer.norm2(x))))
        x = self.funasr_nano.audio_encoder.after_norm(x)
        for encoder_layer in self.funasr_nano.audio_encoder.tp_encoders:
            x1 = encoder_layer.norm1(x)
            qkv = encoder_layer.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, encoder_layer.self_attn.h, encoder_layer.self_attn.d_k).permute(1, 2, 0, 3)
            q_h, k_h, v_h = qkv.split([1, 1, 1], dim=0)
            v_fsmn = v_h.transpose(-1, -2).reshape(1, encoder_layer.size, -1)
            fsmn_in = torch.cat([self.pad_zeros, v_fsmn, self.pad_zeros], dim=-1)
            fsmn_out = encoder_layer.self_attn.fsmn_block(fsmn_in)
            fsmn_memory = (fsmn_out + v_fsmn).transpose(1, 2).reshape(1, -1, encoder_layer.size)
            attn = torch.matmul(q_h, k_h.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v_h).transpose(1, 2).reshape(1, -1, encoder_layer.self_attn.linear_out.in_features)
            attn_out = encoder_layer.self_attn.linear_out(attn) + fsmn_memory
            x = x + attn_out
            x = x + encoder_layer.feed_forward.w_2(encoder_layer.feed_forward.activation(encoder_layer.feed_forward.w_1(encoder_layer.norm2(x))))
        enc_normed = self.funasr_nano.audio_encoder.tp_norm(x)
        x = self.funasr_nano.audio_adaptor.linear1(enc_normed)
        x = self.funasr_nano.audio_adaptor.relu(x)
        x = self.funasr_nano.audio_adaptor.linear2(x)
        for block in self.funasr_nano.audio_adaptor.blocks:
            x1 = block.norm1(x)
            qkv = block.self_attn.linear_q_k_v(x1)
            qkv = qkv.view(-1, 3, block.self_attn.h, block.self_attn.d_k).permute(1, 2, 0, 3)
            q, k, v = qkv.split([1, 1, 1], dim=0)
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, block.self_attn.linear_out.in_features)
            attn_out = block.self_attn.linear_out(attn)
            x = x + attn_out
            x = x + block.feed_forward.w_2(block.feed_forward.activation(block.feed_forward.w_1(block.norm2(x))))
        x = x[:, :self.fake_token[lfr_len].to(torch.int64)]
        concat_embed = torch.cat([self.head_embed, query_embed, x, self.tail_embed], dim=1)
        if self.use_ctc_decoder:
            # Emit enc_normed (the affine-free tp_norm output) so the separate
            # FUNASR_NANO_CTC_DECODER graph can consume it for the fast CTC transcription.
            return concat_embed, concat_embed.shape[1].unsqueeze(0), enc_normed
        return concat_embed, concat_embed.shape[1].unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# CTC Decoder Module (optional — exported & loaded only when USE_CTC_DECODER is True)
# ══════════════════════════════════════════════════════════════════════════════
class FUNASR_NANO_CTC_DECODER(torch.nn.Module):
    """Standalone fast CTC transcription head.

    Consumes `enc_normed` (the affine-free tp_norm output emitted by the encoder) and runs the
    ctc_decoder (Transformer adaptor, downsample_rate=1 -> no reshape) + CTC projection, then
    performs a greedy collapse in-graph: keep a frame only when its argmax token differs from the
    next frame AND is not the blank id. It fuses its OWN weights in __init__ (merge q/k/v, fold
    head_dim^-0.25 into q & k, absorb norm1/norm2, fold tp_norm's affine into linear1), so the graph
    runs the minimal number of ops. Returns the compacted token ids and their count.
    """

    def __init__(self, funasr_nano):
        super(FUNASR_NANO_CTC_DECODER, self).__init__()
        self.ctc_decoder = funasr_nano.ctc_decoder.float()
        self.ctc_lo = funasr_nano.ctc.ctc_lo.float()
        self.blank_id = int(funasr_nano.blank_id)

        # Fuse weights & fold scales in the ctc_decoder blocks (merge q/k/v into one linear, fold
        # head_dim^-0.25 into q & k, absorb norm1 -> linear_q_k_v and norm2 -> feed_forward.w_1).
        FUNASR_NANO_ENCODER._fuse_adaptor_blocks(self.ctc_decoder.blocks)

        # Fold audio_encoder.tp_norm's affine into ctc_decoder.linear1. `enc_normed` (this graph's
        # input) is the affine-free tp_norm output, so the affine must live in linear1. This module is
        # built BEFORE the encoder zeros tp_norm, so the affine is still available here.
        norm = funasr_nano.audio_encoder.tp_norm
        linear = self.ctc_decoder.linear1
        # b_new = b_lin + W_lin @ b_norm ; W_new = W_lin * w_norm
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm.bias.data))
        linear.weight.data.mul_(norm.weight.data.unsqueeze(0))

    def forward(self, enc_normed):
        c = self.ctc_decoder.linear1(enc_normed)
        c = self.ctc_decoder.relu(c)
        c = self.ctc_decoder.linear2(c)
        for block in self.ctc_decoder.blocks:
            c1 = block.norm1(c)
            qkv = block.self_attn.linear_q_k_v(c1)
            qkv = qkv.view(-1, 3, block.self_attn.h, block.self_attn.d_k).permute(1, 2, 0, 3)
            q, k, v = qkv.split([1, 1, 1], dim=0)
            attn = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v).transpose(1, 2).reshape(1, -1, block.self_attn.linear_out.in_features)
            c = c + block.self_attn.linear_out(attn)
            c = c + block.feed_forward.w_2(block.feed_forward.activation(block.feed_forward.w_1(block.norm2(c))))
        token_ids = self.ctc_lo(c).argmax(dim=-1).view(-1).int()                    # int32 token id per frame
        shifted_tensor = torch.cat([token_ids[1:], token_ids[[0]]], dim=0)
        token_keep_mask = (token_ids != shifted_tensor) & (token_ids != self.blank_id)
        keep_indices = torch.nonzero(token_keep_mask, as_tuple=True)[0]
        token_ids = torch.index_select(token_ids, 0, keep_indices)                  # 1-D compacted token ids
        num_id = torch._shape_as_tensor(token_ids)[0].to(torch.int32).unsqueeze(0)
        return token_ids, num_id


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Module
# ══════════════════════════════════════════════════════════════════════════════
class FUNASR_NANO_DECODER_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, funasr_nano):
        super(FUNASR_NANO_DECODER_EMBED, self).__init__()
        self.funasr_nano_decoder_embed = funasr_nano.llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.funasr_nano_decoder_embed(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Simplified RMS LayerNorm (ORT fused SimplifiedLayerNormalization)
# ══════════════════════════════════════════════════════════════════════════════
class SIMPLIFIED_LAYER_NORM(torch.autograd.Function):
    """Emit ONNX Runtime's fused ``SimplifiedLayerNormalization`` (RMS norm) in place of the
    Mul -> ReduceSum -> Add -> Sqrt -> Div chain.

    ORT registers this op in the DEFAULT ONNX domain (``kOnnxDomain``) with SinceVersion 1, so it is
    available at any opset. It computes ``Y = X * rsqrt(mean(X^2, axis) + epsilon) * scale`` and, with
    ``stash_type = 1``, performs the mean/rsqrt reduction in float32 regardless of the I/O dtype (so no
    manual float16-overflow guard is needed). ``forward`` runs only during export tracing; the exported
    graph uses ``symbolic`` (the whole Function collapses into one node).
    """

    @staticmethod
    def forward(ctx, x, scale, epsilon, axis):
        variance   = x.float().pow(2).mean(dim=axis, keepdim=True)
        normalized = x.float() * torch.rsqrt(variance + epsilon)
        return (normalized * scale).to(scale.dtype)

    @staticmethod
    def symbolic(g, x, scale, epsilon, axis):
        return g.op(
            "SimplifiedLayerNormalization",
            x, scale,
            axis_i=axis,
            epsilon_f=epsilon,
            stash_type_i=1,
        )


def simplified_layer_norm(x, scale, epsilon, axis=-1):
    """Return ``SimplifiedLayerNormalization(x, scale, axis, epsilon)`` -- a single fused RMS-norm node."""
    return SIMPLIFIED_LAYER_NORM.apply(x, scale, float(epsilon), axis)


# ══════════════════════════════════════════════════════════════════════════════
# Main Transformer Decoder Module
# ══════════════════════════════════════════════════════════════════════════════
class FUNASR_NANO_DECODER_MAIN(torch.nn.Module):
    """
    Main transformer decoder module that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (RoPE) received as external inputs
      - F16 KV cache management
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
    """

    def __init__(self, funasr_nano, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super(FUNASR_NANO_DECODER_MAIN, self).__init__()
        self.funasr_nano = funasr_nano.llm.float()
        self._replace_gelu_with_tanh_approximation(self.funasr_nano)

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = num_heads + num_key_value_heads

        # ── Layer count ──────────────────────────────────────────────────
        self.num_layers = num_layers

        # ── Minimum-cast f16 KV attention toggle ─────────────────────────
        self.use_fp16_kv = USE_FP16_KV
        self.compute_in_f32 = COMPUTE_IN_F32

        # RMS norm is emitted as ORT's fused SimplifiedLayerNormalization (default ONNX domain): it computes
        # y = x * rsqrt(mean(x^2) + eps) * scale and reduces in float32 (stash_type=1). Feeding scale = 1/sqrt(N)
        # (N = normalized size) and the model's per-element eps reproduces this file's sum-based
        # r = x * rsqrt(sum(x^2) + N*eps) EXACTLY, and the float32 reduction makes the PREVENT_F16_OVERFLOW
        # activation pre-scale unnecessary. The g*sqrt(N) norm weight stays absorbed into the following linear.
        rms_norm_eps = funasr_nano.llm.config.rms_norm_eps
        self.hidden_rms_norm_eps = float(rms_norm_eps)
        self.qk_rms_norm_eps     = float(rms_norm_eps)
        self.register_buffer("hidden_norm_scale", torch.full((hidden_size,), hidden_size ** -0.5, dtype=torch.float32))
        self.register_buffer("qk_norm_scale",     torch.full((head_dim,),    head_dim ** -0.5,    dtype=torch.float32))

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self._fuse_weights(hidden_size)

        # Quantization-friendly weight reorder (exact, zero runtime cost; absorbed into the fused weights).
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """
        Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP.
        """
        scale_factor = self.head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.funasr_nano.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # Absorb final RMSNorm into lm_head
            final_norm_weight = self.funasr_nano.model.norm.weight.unsqueeze(0) * norm_factor
            self.funasr_nano.lm_head.weight.mul_(final_norm_weight)
            del self.funasr_nano.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:
            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)
            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        # Store split dimensions for later use
        attn.q_out_features = int(q_proj.out_features)
        attn.k_out_features = int(k_proj.out_features)
        attn.v_out_features = int(v_proj.out_features)
        attn.qkv_in_features = in_features

        del attn.q_proj, attn.k_proj, attn.v_proj

        # ── Fuse QK norms (absorb scale factors) ────────────────────
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)

        q_norm_repeated = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        # ── Absorb input LayerNorm into QKV weights ─────────────────
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up = layer.mlp.gate_proj, layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Quantization-friendly weight reorder (exact, zero runtime cost)
    # ══════════════════════════════════════════════════════════════════════
    def _reorder_downproj_for_quant(self, key):
        """Reorder each layer's MLP intermediate channels so contiguous block-quant groups over down_proj's
        INPUT axis hold magnitude-homogeneous channels (smaller per-group scale -> lower weight-quant error).
        The SAME permutation is applied to gate_up's OUTPUT rows (gate + up halves) and down_proj's INPUT
        columns, so act(gate)*up @ down_proj is unchanged -- fully absorbed, no runtime de-permutation.
        """
        with torch.no_grad():
            for layer in self.funasr_nano.model.layers:
                W = layer.mlp.down_proj.weight              # (hidden, intermediate)
                a = W.abs()
                if key == "rms":
                    stat = (W * W).mean(0).sqrt()
                elif key == "L4":
                    stat = a.pow(4).mean(0).pow(0.25)
                elif key == "std":
                    stat = W.std(0)
                else:                                       # "absmean" (default / fallback)
                    stat = a.mean(0)
                perm  = torch.argsort(stat)
                inter = layer.mlp.down_proj.in_features
                gu    = layer.mlp.gate_up_proj.weight       # (2*intermediate, hidden): [gate; up]
                layer.mlp.gate_up_proj.weight.copy_(torch.cat([gu[:inter][perm], gu[inter:][perm]], dim=0))
                layer.mlp.down_proj.weight.copy_(W[:, perm])

    def _reorder_oproj_for_quant(self, key):
        """Reorder each head's head_dim so contiguous SUB-head o_proj quant groups (group_size < head_dim)
        are magnitude-homogeneous. ONE head_dim permutation per kv head (shared by its GQA query heads) is
        applied to o_proj's INPUT columns and to the v-rows of the fused qkv; q/k/RoPE/qk_norm untouched.
        Fully absorbed into the weights. Pure win for the f16 KV cache (exact V-cache round-trip).
        """
        H, KVH, Dh, qk_heads = self.num_heads, self.num_key_value_heads, self.head_dim, self.qk_heads
        G = H // KVH
        with torch.no_grad():
            for layer in self.funasr_nano.model.layers:
                Wo  = layer.self_attn.o_proj.weight                 # (hidden, H*head_dim)
                Woc = Wo.view(Wo.shape[0], H, Dh)                   # (hidden, H, head_dim)
                perms = []
                for kvh in range(KVH):                              # one order per kv head
                    cols = Woc[:, kvh * G:(kvh + 1) * G, :]         # its G query heads combined
                    a = cols.abs()
                    if key == "rms":
                        stat = (cols * cols).mean(dim=(0, 1)).sqrt()
                    elif key == "std":
                        stat = cols.reshape(-1, Dh).std(0)
                    elif key == "L4":
                        stat = a.pow(4).mean(dim=(0, 1)).pow(0.25)
                    else:                                           # "absmean" (default / fallback)
                        stat = a.mean(dim=(0, 1))
                    perms.append(torch.argsort(stat))
                # o_proj input columns: permute each head's head_dim by its kv-head order
                Woc2 = Woc.clone()
                for h in range(H):
                    Woc2[:, h, :] = Woc2[:, h, perms[h // G]]
                Wo.copy_(Woc2.reshape(Wo.shape[0], H * Dh))
                # compensate on the v-rows of the fused qkv (head_dim); q/k rows untouched
                Wq  = layer.self_attn.qkv.weight                    # (total_heads*head_dim, hidden)
                Wqr = Wq.view(-1, Dh, Wq.shape[1]).clone()          # (total_heads, head_dim, hidden)
                for kvh in range(KVH):
                    Wqr[qk_heads + kvh] = Wqr[qk_heads + kvh][perms[kvh]]
                Wq.copy_(Wqr.reshape(Wq.shape[0], Wq.shape[1]))

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Recursively replace exact GELU with tanh-approximated GELU for ONNX compatibility."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                FUNASR_NANO_DECODER_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x, scale, eps):
        """RMS norm via ORT's fused SimplifiedLayerNormalization (reduces in float32, stash_type=1)."""
        return simplified_layer_norm(x, scale, eps)

    def _rotate_half(self, x, batch_size):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime.
        """
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0]

        # f16-storage / f32-compute (COMPUTE_IN_F32): keep the causal mask f16 at the graph boundary (I/O
        # dtype unchanged) but upcast it to f32 ONCE here, shared by every layer (cast loop-invariant
        # constants once). In every other mode it is used as-is (f16 minimum-cast, or f32 for a float32 cache).
        attn_mask = attention_mask.float() if (self.use_fp16_kv and self.compute_in_f32) else attention_mask

        for i, layer in enumerate(self.funasr_nano.model.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)

            # Fused QKV projection & reshape
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)

            # QK normalization & rotary embedding
            qk = self._rms_norm(qk, self.qk_norm_scale, self.qk_rms_norm_eps) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # Split into query and key
            # Minimum-cast float16 KV attention (OFF): cast qk_rot + V DOWN to f16 before the split so
            # Q@K/mask/softmax/attn@V run in f16; only the context is cast back to f32 for o_proj.
            # COMPUTE_IN_F32 (ON): keep the f16 KV *storage* (K/V are still cast to f16 before the cache
            # concat, so the cache I/O dtype is unchanged) but upcast the f16 K/V to f32 at the matmul use
            # points and keep Q/mask/softmax in f32 -- f16 storage, f32 compute. Q is never downcast.
            if self.use_fp16_kv and not self.compute_in_f32:
                qk_rot = qk_rot.half()

            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            if self.use_fp16_kv:
                k = k.half()   # f16 KV storage (no-op in the minimum-cast path: qk_rot is already f16)
                v = v.half()
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim).permute(0, 2, 3, 1, 4)
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v

            if self.use_fp16_kv and self.compute_in_f32:
                attn = torch.matmul(q, k.float()) + attn_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v.float())
            else:
                attn = torch.matmul(q, k) + attn_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v)

            # Output projection & residual
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            if self.use_fp16_kv and not self.compute_in_f32:
                attn = attn.float()
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states, self.hidden_norm_scale, self.hidden_rms_norm_eps)

            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(hidden_states[:, -1], self.hidden_norm_scale, self.hidden_rms_norm_eps)
        logits = self.funasr_nano.lm_head(hidden_states)

        return *self.save_key, *self.save_value, logits


print('\nExport start ...\n')
with torch.inference_mode():

    # ══════════════════════════════════════════════════════════════════
    # Load Model & Extract Config
    # ══════════════════════════════════════════════════════════════════
    custom_stft = STFT_Process(
        model_type='stft_B',
        win_length=WINDOW_LENGTH,      # Kaldi 400-sample frame (conv width)
        hop_len=HOP_LENGTH,
        dft_size=NFFT_STFT,            # zero-pad 400 -> 512 before the DFT (Kaldi round_to_power_of_two)
        pre_emphasis=PRE_EMPHASIZE,    # per-frame pre-emphasis, baked into the kernel
        remove_dc=True,                # per-frame DC-offset removal, baked into the kernel
        window_type=WINDOW_TYPE,
        window_periodic=False          # symmetric Hamming window (Kaldi convention)
    ).eval()
    model = AutoModel(
        model=model_path,
        device="cpu",
        disable_update=True
    )  # FunASRNano is registered above via @tables.register, so no remote_code is needed.
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    ctc_tokenizer = model.model.ctc_tokenizer if USE_CTC_DECODER else None   # tiktoken tokenizer for the CTC branch; kept before `del model`.

    llm_config   = model.model.llm.config
    llm_model    = model.model.llm.model
    num_layers   = llm_config.num_hidden_layers
    num_heads    = llm_config.num_attention_heads
    num_kv_heads = llm_config.num_key_value_heads
    head_dim     = llm_config.head_dim
    vocab_size   = llm_model.vocab_size
    hidden_size  = llm_model.embed_tokens.embedding_dim

    # ══════════════════════════════════════════════════════════════════
    # Build Dummy Tensors for Tracing
    # ══════════════════════════════════════════════════════════════════
    batch_size  = BEAM_SIZE
    ids_len     = torch.tensor([10], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    kv_seq_len  = ids_len + history_len
    beam_size   = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    logits      = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)

    # KV cache spec: list of (name, concat_dim). F16 KV cache when USE_FP16_KV (minimum-cast attention).
    kv_specs = [('key', 4), ('value', 3)]
    kv_dtype = torch.float16 if USE_FP16_KV else torch.float32

    kv_tensors = {
        'key':   torch.zeros((batch_size, num_kv_heads, 1, head_dim, history_len), dtype=kv_dtype),
        'value': torch.zeros((batch_size, num_kv_heads, 1, history_len, head_dim), dtype=kv_dtype)
    }

    # ══════════════════════════════════════════════════════════════════
    # Helper: Build KV I/O names, tensors, and dynamic axes
    # ══════════════════════════════════════════════════════════════════
    def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
        inputs, in_names, out_names, axes = [], [], [], {}
        for name, dim in kv_specs:
            tensor = tensors_dict[name]
            for i in range(num_layers):
                in_n  = f'in_{name}_{i}'
                out_n = f'out_{name}_{i}'
                inputs.append(tensor)
                in_names.append(in_n)
                out_names.append(out_n)
                axes[in_n]  = {0: batch_axis, dim: seq_axis}
                axes[out_n] = {0: batch_axis, dim: out_seq_axis}
        return inputs, in_names, out_names, axes

    # ══════════════════════════════════════════════════════════════════
    # Build CTC Decoder FIRST (optional) so it folds tp_norm's affine into its own linear1
    # before the encoder makes tp_norm affine-free.
    # ══════════════════════════════════════════════════════════════════
    if USE_CTC_DECODER:
        funasr_nano_ctc_decoder = FUNASR_NANO_CTC_DECODER(model.model)
        ctc_enc_dim = model.model.ctc_decoder.linear1.in_features   # == enc_normed last dim

    # ══════════════════════════════════════════════════════════════════
    # Export: Encoder
    # ══════════════════════════════════════════════════════════════════
    funasr_nano_encoder = FUNASR_NANO_ENCODER(
        model.model, custom_stft, NFFT_STFT, MAX_STFT_SIGNAL_LENGTH,
        N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, LFR_M, LFR_N, LFR_LENGTH, tokenizer, USE_CTC_DECODER
    )
    encoder_output_names = ['concat_embed', 'ids_len']
    encoder_dynamic_axes = {
        'audio':        {2: 'audio_len'},
        'query_embed':  {1: 'num_token'},
        'concat_embed': {1: 'num_token'}
    }
    if USE_CTC_DECODER:
        encoder_output_names += ['enc_normed']
        encoder_dynamic_axes['enc_normed'] = {1: 'enc_len'}
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    torch.onnx.export(
        funasr_nano_encoder,
        (torch.ones((1, 1, MAX_INPUT_AUDIO_LENGTH), dtype=_audio_export_dtype),
         torch.ones((1, 10, hidden_size), dtype=torch.float32)),
        onnx_model_Encoder,
        input_names=['audio', 'query_embed'],
        output_names=encoder_output_names,
        dynamic_axes=encoder_dynamic_axes if DYNAMIC_AXES else None,
        opset_version=OPSET,
        dynamo=False
    )
    del funasr_nano_encoder, custom_stft
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Export: CTC Decoder (optional standalone fast-transcription head)
    # ══════════════════════════════════════════════════════════════════
    if USE_CTC_DECODER:
        torch.onnx.export(
            funasr_nano_ctc_decoder,   # built before the encoder; self-contained fusion done in its __init__
            (torch.ones((1, 20, ctc_enc_dim), dtype=torch.float32),),
            onnx_model_CTC_Decoder,
            input_names=['enc_normed'],
            output_names=['ctc_token_ids', 'ctc_num_id'],
            dynamic_axes={
                'enc_normed':    {1: 'enc_len'},
                'ctc_token_ids': {0: 'ctc_num_token'}
            } if DYNAMIC_AXES else None,
            opset_version=OPSET,
            dynamo=False
        )
        del funasr_nano_ctc_decoder
        gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Export: Decoder Embed
    # ══════════════════════════════════════════════════════════════════
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    torch.onnx.export(
        FUNASR_NANO_DECODER_EMBED(model.model),
        (input_ids,),
        onnx_model_Embed,
        input_names=['input_ids'],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids':     {0: 'batch', 1: 'ids_len'},
            'hidden_states': {0: 'batch', 1: 'ids_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del input_ids

    # ══════════════════════════════════════════════════════════════════
    # Export: Rotary + Mask (Prefill)
    # ══════════════════════════════════════════════════════════════════
    torch.onnx.export(
        ROTARY_MASK_PREFILL(model.model, MAX_SEQ_LEN),
        (ids_len, history_len),
        onnx_model_Rotary_Mask_Prefill,
        input_names=['ids_len', 'history_len'],
        output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
        dynamic_axes={
            'rotary_cos':     {1: 'ids_len'},
            'rotary_sin':     {1: 'ids_len'},
            'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: Rotary + Mask (Decode)
    # ══════════════════════════════════════════════════════════════════
    torch.onnx.export(
        ROTARY_MASK_DECODE(model.model, MAX_SEQ_LEN),
        (kv_seq_len,),
        onnx_model_Rotary_Mask_Decode,
        input_names=['kv_seq_len'],
        output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
        dynamic_axes=None,
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: Decoder Main (Transformer Layers)
    # ══════════════════════════════════════════════════════════════════
    kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)

    hidden_states  = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
    rotary_cos     = torch.zeros((1, ids_len, 1, 1, head_dim), dtype=torch.float32)
    rotary_sin     = rotary_cos
    # The mask is added to the (minimum-cast) f16 / f32 attention scores, so it must match the KV dtype.
    attention_mask = torch.zeros((1, 1, 1, ids_len, kv_seq_len), dtype=kv_dtype)

    all_inputs   = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
    input_names  = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
    output_names = kv_out_names + ['logits']
    dynamic_axes = {
        **kv_axes,
        'hidden_states':  {0: 'batch', 1: 'ids_len'},
        'logits':         {0: 'batch'},
        'rotary_cos':     {1: 'ids_len'},
        'rotary_sin':     {1: 'ids_len'},
        'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
    }

    model_Main = FUNASR_NANO_DECODER_MAIN(model.model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size)
    del model

    torch.onnx.export(
        model_Main,
        tuple(all_inputs),
        onnx_model_Main,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del model_Main, hidden_states, attention_mask, all_inputs
    gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Export: Greedy Search
    # ══════════════════════════════════════════════════════════════════
    save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)  # 10 is a dummy value.

    torch.onnx.export(
        GREEDY_SEARCH(),
        (logits, save_id_in),
        onnx_model_Greedy,
        input_names=['logits', 'save_id_in'],
        output_names=['max_logits_idx', 'save_id_out'],
        dynamic_axes={
            'logits':         {0: 'batch'},
            'save_id_in':     {0: 'batch', 1: 'history_len'},
            'save_id_out':    {0: 'batch', 1: 'history_len'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: First Beam Search
    # ══════════════════════════════════════════════════════════════════
    num_layers_beam = num_layers * len(kv_specs)
    # First beam uses single-batch KV (batch dim = 1)
    kv_tensors_Greedy = {k: v[[0]] for k, v in kv_tensors.items()}
    kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_Greedy)
    # Remove output axes — first beam outputs have variable batch, not tracked here
    kv_input_only_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names}

    torch.onnx.export(
        FIRST_BEAM_SEARCH(num_layers_beam),
        tuple(kv_ins + [logits[[0]], save_id_in, beam_size]),
        onnx_model_First_Beam,
        input_names=kv_in_names + ['logits', 'save_id_in', 'beam_size'],
        output_names=(
            ['out_' + n[3:] for n in kv_in_names] + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx']
        ),
        dynamic_axes={
            **kv_input_only_axes,
            'logits':           {0: 'batch'},
            'save_id_in':       {0: 'batch', 1: 'history_len'},
            'top_beam_prob':    {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx':   {0: 'batch'},
            'save_id_out':      {0: 'batch', 1: 'history_len'}
        },
        opset_version=OPSET,
        dynamo=False
    )

    # ══════════════════════════════════════════════════════════════════
    # Export: Second Beam Search
    # ══════════════════════════════════════════════════════════════════
    kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)
    previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
    topK = torch.tensor([TOP_K], dtype=torch.int64)

    torch.onnx.export(
        SECOND_BEAM_SEARCH(num_layers_beam),
        tuple(kv_ins + [logits, save_id_in, previous_prob, beam_size, topK]),
        onnx_model_Second_Beam,
        input_names=kv_in_names + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
        output_names=kv_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
        dynamic_axes={
            **kv_axes,
            'logits':           {0: 'batch'},
            'save_id_in':       {0: 'batch', 1: 'history_len'},
            'previous_prob':    {0: 'batch'},
            'save_id_out':      {0: 'batch', 1: 'history_len'},
            'top_beam_prob':    {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx':   {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del kv_tensors_Greedy, previous_prob, topK

    # ══════════════════════════════════════════════════════════════════
    # Export: Apply Penalty
    # ══════════════════════════════════════════════════════════════════
    penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
    penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)

    torch.onnx.export(
        APPLY_PENALTY(),
        (logits, save_id_in, penalty_value, penalty_range),
        onnx_model_Penalty,
        input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
        output_names=['logits_out'],
        dynamic_axes={
            'logits_in':  {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'logits_out': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del save_id_in, penalty_value, penalty_range

    # ══════════════════════════════════════════════════════════════════
    # Export: Argmax
    # ══════════════════════════════════════════════════════════════════
    torch.onnx.export(
        ARGMAX(),
        (logits,),
        onnx_model_Argmax,
        input_names=['logits'],
        output_names=['max_logits_idx'],
        dynamic_axes={
            'logits':         {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del logits
    gc.collect()

    onnx_metadata = build_model_metadata(
        {
            "fun_asr_nano_metadata_version": 1,
            "producer": "Export_Fun_ASR_Nano.py",
        },
        {
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "max_seq_len": MAX_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "input_audio_length": MAX_INPUT_AUDIO_LENGTH,
            "max_input_audio_length": MAX_INPUT_AUDIO_LENGTH,
            "use_fp16_kv": USE_FP16_KV,
            "compute_in_f32": COMPUTE_IN_F32,
            "use_ctc_decoder": USE_CTC_DECODER,
            "is_mlt": "MLT" in tokenizer_path,
            "num_mels": N_MELS,
            "nfft_stft": NFFT_STFT,
            "window_length": WINDOW_LENGTH,
            "hop_length": HOP_LENGTH,
            "lfr_m": LFR_M,
            "lfr_n": LFR_N,
        },
        {
            "stop_token_ids": ",".join(str(t) for t in STOP_TOKEN),
        },
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

    _metadata_targets = [onnx_model_Metadata]
    _written, _skipped = [], []
    for _target in _metadata_targets:
        if not Path(_target).exists():
            continue
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

# ── Bundle the tokenizer into the export folder so the ONNX model set is self-contained ──
# The inference script loads its tokenizer(s) from this folder when the local copy is present
# (falling back to the configured source paths), so the exported folder runs stand-alone.
_tokenizer_dst = onnx_folder / Path(tokenizer_path).name
try:
    shutil.copytree(tokenizer_path, _tokenizer_dst, dirs_exist_ok=True)
    print(f"[Tokenizer] Copied tokenizer -> {_tokenizer_dst}")
except Exception as _exc:  # noqa: BLE001 - a failed copy must not abort the demo run
    print(f"[Tokenizer] Skipped tokenizer copy ({_exc}); inference will use tokenizer_path.")

if USE_CTC_DECODER:
    _ctc_vocab_src = Path(model_path) / "multilingual.tiktoken"
    _ctc_vocab_dst = onnx_folder / _ctc_vocab_src.name
    try:
        shutil.copyfile(_ctc_vocab_src, _ctc_vocab_dst)
        print(f"[Tokenizer] Copied CTC vocab -> {_ctc_vocab_dst}")
    except Exception as _exc:  # noqa: BLE001 - a failed copy must not abort the demo run
        print(f"[Tokenizer] Skipped CTC vocab copy ({_exc}); inference will use ctc_tokenizer_path.")

print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_Fun_ASR_Nano_ONNX.py ...')
subprocess.run(
    [sys.executable, str(Path(__file__).resolve().parent / "Inference_Fun_ASR_Nano_ONNX.py"), "--onnx-folder", str(onnx_folder)],
    check=True,
)
