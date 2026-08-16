import gc
import json
import os
import shutil
import copy
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
import torch.nn.functional as F
import dolphin                         
import torchaudio
import sentencepiece as spm
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 


# ============================================================================================
#                                       User configuration
# ============================================================================================
model_path = str(Path.home() / "Downloads" / "dolphin-small")  # Dolphin project path.
_MODEL_MAX_AUDIO_SAMPLES = 480000
_MODEL_MAX_DECODER_SEQ_LEN = 448
INPUT_AUDIO_DTYPE   = "F32"             # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. "INT16" feeds raw PCM (÷32768 in-graph); "F32"/"F16" feed audio pre-normalised to [-1, 1].
USE_FP16_KV     = True                  # Keep cache, cross-KV, position, and mask storage in FP16 for normal deployment exports.
COMPUTE_IN_F32  = False                 # FP16-cache-only option: upcast attention compute while retaining FP16 storage.
KV_DTYPE        = torch.float16 if USE_FP16_KV else torch.float32
REORDER_DOWNPROJ_FOR_QUANT = True       # Exact FFN intermediate-channel reorder, absorbed into w_1/w_2.
REORDER_OPROJ_FOR_QUANT    = True       # Exact self-attention V/linear_out per-head reorder.
REORDER_KEY                = "absmean"  # "absmean" | "L4" | "rms" | "std".
CROSS_KV_GROUP_SIZE        = 2          # Fuse two decoder cross-KV projections per Gemm; bounds the max f32 temporary to ~8.8 MiB.
OPSET                      = 20         # ONNX opset version for the export.


ONNX_DIR = Path(_SCRIPT_DIR) / "Dolphin_ONNX"
_raw_onnx_temp = tempfile.TemporaryDirectory(prefix="dolphin-v1-export-")
_raw_onnx_dir = Path(_raw_onnx_temp.name)

MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Dolphin_Encoder.onnx",
    "main": "Dolphin_Decoder.onnx",
    "embed": "Dolphin_Decoder_Embed.onnx",
    "position_prefill": "Dolphin_Position_Mask_Prefill.onnx",
    "position_decode": "Dolphin_Position_Mask_Decode.onnx",
    "greedy": "Dolphin_Greedy_Search.onnx",
    "argmax": "Dolphin_Argmax.onnx",
    "sampling": "Dolphin_TopKTopPSampling.onnx",
    "penalty": "Dolphin_Apply_Penalty.onnx",
    "prefill_greedy": "Dolphin_PrefillGreedy.onnx",
    "prefill_penalty_greedy": "Dolphin_PrefillPenaltyGreedy.onnx",
    "prefill_sampling": "Dolphin_PrefillSampling.onnx",
    "probe_prefill_greedy": "Dolphin_ProbePrefillGreedy.onnx",
    "probe_prefill_penalty_greedy": "Dolphin_ProbePrefillPenaltyGreedy.onnx",
    "probe_prefill_sampling": "Dolphin_ProbePrefillSampling.onnx",
    "decode_greedy": "Dolphin_DecodeGreedy.onnx",
    "decode_penalty_greedy": "Dolphin_DecodePenaltyGreedy.onnx",
    "decode_sampling": "Dolphin_DecodeSampling.onnx",
    "shared_initializers": "Dolphin_SharedInitializers.onnx",
}
MODEL_FILE_NAMES["shared_initializers_data"] = MODEL_FILE_NAMES["shared_initializers"] + ".data"

onnx_model_Metadata = str(_raw_onnx_dir / MODEL_FILE_NAMES["metadata"])
onnx_model_Encoder = str(_raw_onnx_dir / MODEL_FILE_NAMES["encoder"])
onnx_model_Decoder = str(_raw_onnx_dir / MODEL_FILE_NAMES["main"])
onnx_model_Embed = str(_raw_onnx_dir / MODEL_FILE_NAMES["embed"])
onnx_model_Prefill = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_prefill"])
onnx_model_Decode = str(_raw_onnx_dir / MODEL_FILE_NAMES["position_decode"])
onnx_model_Greedy = str(_raw_onnx_dir / MODEL_FILE_NAMES["greedy"])
onnx_model_Argmax = str(_raw_onnx_dir / MODEL_FILE_NAMES["argmax"])
onnx_model_Sampling = str(_raw_onnx_dir / MODEL_FILE_NAMES["sampling"])
onnx_model_Penalty = str(_raw_onnx_dir / MODEL_FILE_NAMES["penalty"])
save_vocab = str(_raw_onnx_dir / "vocab_Dolphin.txt")


# Fixed Dolphin-v1 model contract. These values define exported tensor geometry and metadata;
# they are not runtime/user choices and must remain aligned with the trained frontend/model.
WINDOW_TYPE = "hann"
N_MELS = 80
NFFT_STFT = 512
WINDOW_LENGTH = 400
HOP_LENGTH = 160
PRE_EMPHASIZE = 0.0
SAMPLE_RATE = 16000


LANGUAGE_REGION = {
    # ───────────────────────────── Auto Detection ─────────────────────────────
    "Auto"                         : "auto-auto",
    "Auto-Auto"                    : "auto-auto",
    "Chinese-Auto"                 : "zh-auto",
    "Mandarin-Auto"                : "zh-auto",
    "Yue-Auto"                     : "ct-NULL",
    "Tamil-Auto"                   : "ta-auto",
    "Urdu-Auto"                    : "ur-auto",
    "Arabic-Auto"                  : "ar-auto",

    "自动"                          : "auto-auto",
    "自动-自动"                      : "auto-auto",
    "中文-自动"                      : "zh-auto",
    "普通话-自动"                    : "zh-auto",
    "粤语-自动"                      : "ct-NULL",
    "泰米尔语-自动"                   : "ta-auto",
    "乌尔都语-自动"                   : "ur-auto",
    "阿拉伯语-自动"                   : "ar-auto",

    # ───────────────────────────── Chinese variants ─────────────────────────────
    "Chinese"                       : "zh-CN",
    "Mandarin"                      : "zh-CN",
    "Chinese-Mandarin"              : "zh-CN",
    "Chinese-Taiwan"                : "zh-TW",
    "Chinese-Wuyu"                  : "zh-WU",
    "Chinese-Sichuan"               : "zh-SICHUAN",
    "Chinese-Shanxi"                : "zh-SHANXI",
    "Chinese-Anhui"                 : "zh-ANHUI",
    "Chinese-Tianjin"               : "zh-TIANJIN",
    "Chinese-Ningxia"               : "zh-NINGXIA",
    "Chinese-Shaanxi"               : "zh-SHAANXI",
    "Chinese-Hebei"                 : "zh-HEBEI",
    "Chinese-Shandong"              : "zh-SHANDONG",
    "Chinese-Guangdong"             : "zh-GUANGDONG",
    "Chinese-Shanghai"              : "zh-SHANGHAI",
    "Chinese-Hubei"                 : "zh-HUBEI",
    "Chinese-Liaoning"              : "zh-LIAONING",
    "Chinese-Gansu"                 : "zh-GANSU",
    "Chinese-Fujian"                : "zh-FUJIAN",
    "Chinese-Hunan"                 : "zh-HUNAN",
    "Chinese-Henan"                 : "zh-HENAN",
    "Chinese-Yunnan"                : "zh-YUNNAN",
    "Chinese-Minnan"                : "zh-MINNAN",
    "Chinese-Wenzhou"               : "zh-WENZHOU",

    "中文"                           : "zh-CN",
    "普通话"                         : "zh-CN",
    "中文-普通话"                    : "zh-CN",
    "中文-台湾"                      : "zh-TW",
    "中文-吴语"                      : "zh-WU",
    "中文-四川话"                    : "zh-SICHUAN",
    "中文-山西话"                    : "zh-SHANXI",
    "中文-安徽话"                    : "zh-ANHUI",
    "中文-天津话"                    : "zh-TIANJIN",
    "中文-宁夏话"                    : "zh-NINGXIA",
    "中文-陕西话"                    : "zh-SHAANXI",
    "中文-河北话"                    : "zh-HEBEI",
    "中文-山东话"                    : "zh-SHANDONG",
    "中文-广东话"                    : "zh-GUANGDONG",
    "中文-上海话"                    : "zh-SHANGHAI",
    "中文-湖北话"                    : "zh-HUBEI",
    "中文-辽宁话"                    : "zh-LIAONING",
    "中文-甘肃话"                    : "zh-GANSU",
    "中文-福建话"                    : "zh-FUJIAN",
    "中文-湖南话"                    : "zh-HUNAN",
    "中文-河南话"                    : "zh-HENAN",
    "中文-云南话"                    : "zh-YUNNAN",
    "中文-闽南语"                    : "zh-MINNAN",
    "中文-温州话"                    : "zh-WENZHOU",

    # ───────────────────────────── Yue-Cantonese variants ───────────────────────────
    "Yue-Unknown"                  : "ct-NULL",
    "Yue-Hongkong"                 : "ct-HK",
    "Yue-Guangdong"                : "ct-GZ",

    "粤语-未知"                     : "ct-NULL",
    "粤语-香港"                     : "ct-HK",
    "粤语-广东"                     : "ct-GZ",

    # ───────────────────────────── East-Asian languages ──────────────────────────────
    "Japanese"                      : "ja-JP",
    "Korean"                        : "ko-KR",

    "日文"                           : "ja-JP",
    "日语"                           : "ja-JP",
    "韩语"                           : "ko-KR",

    # ───────────────────────────── South-East Asian languages ─────────────────────────
    "Thai"                          : "th-TH",
    "Indonesian"                    : "id-ID",
    "Vietnamese"                    : "vi-VN",
    "Malay"                         : "ms-MY",
    "Burmese"                       : "my-MM",
    "Tagalog"                       : "tl-PH",
    "Khmer"                         : "km-KH",
    "Javanese"                      : "jv-ID",
    "Lao"                           : "lo-LA",
    "Filipino"                      : "fil-PH",
    "Sundanese"                     : "su-ID",

    "泰语"                            : "th-TH",
    "印度尼西亚语"                     : "id-ID",
    "越南语"                          : "vi-VN",
    "马来语"                          : "ms-MY",
    "缅甸语"                          : "my-MM",
    "塔加洛语"                        : "tl-PH",
    "高棉语"                          : "km-KH",
    "爪哇语"                          : "jv-ID",
    "老挝语"                          : "lo-LA",
    "菲律宾语"                        : "fil-PH",
    "巽他语"                          : "su-ID",

    # ───────────────────────────── South-Asian languages ──────────────────────────────
    "Hindi"                         : "hi-IN",
    "Bengali"                       : "bn-BD",
    "Tamil-Singaporean"             : "ta-SG",
    "Tamil-Sri Lankan"              : "ta-LK",
    "Tamil-India"                   : "ta-IN",
    "Tamil-Malaysia"                : "ta-MY",
    "Telugu"                        : "te-IN",
    "Gujarati"                      : "gu-IN",
    "Oriya"                         : "or-IN",
    "Odia"                          : "or-IN",
    "Nepali"                        : "ne-NP",
    "Sinhala"                       : "si-LK",
    "Panjabi"                       : "pa-IN",
    "Kashmiri"                      : "ks-IN",
    "Marathi"                       : "mr-IN",

    "印地语"                         : "hi-IN",
    "孟加拉语"                       : "bn-BD",
    "泰米尔语-新加坡"                 : "ta-SG",
    "泰米尔语-斯里兰卡"                : "ta-LK",
    "泰米尔语-印度"                   : "ta-IN",
    "泰米尔语-马来西亚"                : "ta-MY",
    "泰卢固语"                        : "te-IN",
    "古吉拉特语"                      : "gu-IN",
    "奥里亚语"                        : "or-IN",
    "尼泊尔语"                        : "ne-NP",
    "僧伽罗语"                        : "si-LK",
    "旁遮普语"                        : "pa-IN",
    "克什米尔语"                      : "ks-IN",
    "马拉地语"                        : "mr-IN",

    # ───────────────────────────── Middle-Eastern languages ───────────────────────────
    "Urdu"                          : "ur-PK",
    "Urdu-Islamic Republic of Pakistan": "ur-PK",
    "Urdu-India"                    : "ur-IN",
    "Persian"                       : "fa-IR",
    "Pushto"                        : "ps-AF",

    "乌尔都语"                        : "ur-PK",
    "乌尔都语-印度"                    : "ur-IN",
    "波斯语"                          : "fa-IR",
    "普什图语"                        : "ps-AF",

    # ───────────────────────────── Arabic variants ──────────────────────────────
    "Arabic"                        : "ar-GLA",
    "Arabic-Morocco"                : "ar-MA",
    "Arabic-Saudi Arabia"           : "ar-SA",
    "Arabic-Egypt"                  : "ar-EG",
    "Arabic-Kuwait"                 : "ar-KW",
    "Arabic-Libya"                  : "ar-LY",
    "Arabic-Jordan"                 : "ar-JO",
    "Arabic-U.A.E."                 : "ar-AE",
    "Arabic-Levant"                 : "ar-LVT",

    "阿拉伯语"                        : "ar-GLA",
    "阿拉伯语-摩洛哥"                  : "ar-MA",
    "阿拉伯语-沙特"                    : "ar-SA",
    "阿拉伯语-埃及"                    : "ar-EG",
    "阿拉伯语-科威特"                  : "ar-KW",
    "阿拉伯语-利比亚"                  : "ar-LY",
    "阿拉伯语-约旦"                    : "ar-JO",
    "阿拉伯语-阿联酋"                  : "ar-AE",
    "阿拉伯语-黎凡特"                  : "ar-LVT",

    # ───────────────────────────── Central-Asian languages ────────────────────────────
    "Uighur"                        : "ug-CN",
    "Uzbek"                         : "uz-UZ",
    "Kazakh"                        : "kk-KZ",
    "Mongolian"                     : "mn-MN",
    "Kabyle"                        : "kab-NULL",
    "Bashkir"                       : "ba-NULL",
    "Tajik"                         : "tg-TJ",
    "Kirghiz"                       : "ky-KG",
    "Azerbaijani"                   : "az-AZ",

    "维吾尔语"                        : "ug-CN",
    "乌兹别克语"                      : "uz-UZ",
    "哈萨克语"                        : "kk-KZ",
    "蒙古语"                          : "mn-MN",
    "卡拜尔语"                        : "kab-NULL",
    "巴什基尔语"                      : "ba-NULL",
    "塔吉克语"                        : "tg-TJ",
    "吉尔吉斯语"                      : "ky-KG",
    "阿塞拜疆语"                      : "az-AZ",

    # ───────────────────────────── Eastern-European languages ─────────────────────────
    "Russian"                       : "ru-RU",
    "俄语"                           : "ru-RU",
}


class Tokenizer:
    def __init__(self, filename, bpe_model=None):
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.num_vocab = 0
        self.sp = None
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)
        if spm is not None and bpe_model is not None and os.path.exists(bpe_model):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(bpe_model)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def decode_ids(self, ids):
        tokens = [self.decode(int(idx)) for idx in ids]
        tokens = [token for token in tokens if token is not None]
        if self.sp is not None:
            return self.sp.DecodePieces(tokens)
        return ''.join(tokens).replace("▁", " ")

    def num_vocab(self):
        return self.num_vocab
        

def rel_shift(x, x_len, zero_pad, n_head):
    x_padded = torch.cat([zero_pad[:, :x_len], x], dim=-1)
    x_padded = x_padded.view(n_head, -1, x_len)
    x = x_padded[:, 1:].view_as(x)
    return x[:, :, :x_len]


def _bias_or_zero(linear):
    return linear.bias if linear.bias is not None else torch.zeros(linear.out_features, dtype=linear.weight.dtype)


def fold_norm_into_linear(norm, linear):
    # Absorb a LayerNorm affine (gamma/beta) forward into the next Linear: W'=W*gamma, b'=b+W@beta.
    # The LayerNorm is left affine-free so its forward call still performs the (x-mean)/std normalisation.
    linear.bias.data.add_(linear.weight.data @ norm.bias.data)
    linear.weight.data.mul_(norm.weight.data.unsqueeze(0))
    norm.weight.data.fill_(1.0)
    norm.bias.data.zero_()


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


class METADATA_CARRIER(torch.nn.Module):
    def __init__(self):
        super(METADATA_CARRIER, self).__init__()

    def forward(self, marker):
        return marker


class APPLY_PENALTY(torch.nn.Module):
    # Sliding-window repetition penalty (Qwen ASR style): multiply the logits of the most recent
    # `penalty_range` tokens by `penalty_value`. Keep penalty_range as a live int64[1] graph input;
    # indexing element zero avoids `.item()` and the fragile scalar-Reshape Slice lowering.
    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range[0]:].long()
        penalised = logits.gather(1, target_indices) * penalty_value
        return logits.scatter(1, target_indices, penalised)


class TOPK_TOPP_SAMPLING(torch.nn.Module):
    NEG_INF = float("-inf")
    GUMBEL_EPS = 1.0e-7

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "neg_inf", torch.tensor(self.NEG_INF, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_min", torch.tensor(self.GUMBEL_EPS, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "gumbel_max",
            torch.tensor(1.0 - self.GUMBEL_EPS, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, logits, temperature, top_k, top_p, repetition_penalty, previous_ids):
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


class DOLPHIN_ENCODER(torch.nn.Module):
    def __init__(
        self,
        dolphin,
        stft_model,
        nfft_stft,
        n_mels,
        sample_rate,
        pre_emphasis,
        num_layers_de,
        max_audio_length,
        cross_kv_group_size,
    ):
        super(DOLPHIN_ENCODER, self).__init__()
        source_model = dolphin.s2t_model
        source_decoder_layers = list(source_model.decoder.decoders)
        # The encoder graph only needs the encoder plus decoder cross-K/V weights.  Avoid copying
        # the decoder embedding, self-attention, FFNs, and vocabulary head into this wrapper.
        self.encoder = copy.deepcopy(source_model.encoder)
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.nfft_stft = nfft_stft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.cross_kv_group_size = int(cross_kv_group_size)

        # Calculate frequency bins for STFT and filterbank
        self.stft_bins = nfft_stft // 2 + 1  # Number of frequency bins from STFT

        # Create mel filterbank - ensure it matches STFT output dimensions
        fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=self.stft_bins,
            f_min=0,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm='slaney',
            mel_scale='slaney'
        ).transpose(0, 1).unsqueeze(0)
        self.register_buffer('fbank', fbank)
                     
        # Normalization parameters (global_cmvn stores mean/std as float64; cast to float32)
        self.inv_int16 = float(1.0 / 32768.0)
        # int16 audio is raw PCM (normalised in forward via ÷32768); f32/f16 audio is
        # assumed pre-normalised to [-1, 1], so the in-graph division is skipped.
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")
        if self.input_audio_is_int16:
            # 2^-15 is exact in float32.  Folding it into the immutable DFT kernel removes a
            # full-audio Mul while preserving the Conv accumulation bit-for-bit on this model.
            self.stft_model.stft_kernel.mul_(self.inv_int16)
        self.register_buffer('cmvn_mean', self.encoder.global_cmvn.mean.float().clone())
        self.register_buffer('inv_std', (1.0 / self.encoder.global_cmvn.std).float().clone())
        del self.encoder.global_cmvn

        self.save_en_keys = [None] * num_layers_de
        self.save_en_values = [None] * num_layers_de
        self.embed = self.encoder.embed.out[0]
        position_encode = self.encoder.embed.pos_enc
        self.embed.weight.data *= position_encode.xscale
        self.embed.bias.data *= position_encode.xscale
        self.num_heads = self.encoder.encoders._modules['0'].attn.h
        self.head_dim = self.encoder.encoders._modules['0'].attn.d_k
        self.hidden_size = self.embed.out_features
        self.cross_num_heads = source_decoder_layers[0].src_attn.h
        self.cross_head_dim = source_decoder_layers[0].src_attn.d_k

        # Bound the relative-position table by the declared maximum audio length.  The original
        # 9,999-position projection was 184,301,568 bytes; 30 seconds reaches only 749 encoder
        # frames, so the complete required symmetric table has 1,497 positions.
        max_embed_len = int(max_audio_length)
        if getattr(self.stft_model, '_center_pad', False):
            max_embed_len = max_embed_len // self.stft_model.hop_len + 1
        else:
            max_embed_len = (
                max_embed_len - self.stft_model.n_fft
            ) // self.stft_model.hop_len + 1
        for module in self.encoder.embed.conv:
            if isinstance(module, torch.nn.Conv2d):
                kernel = module.kernel_size[0]
                stride = module.stride[0]
                padding = module.padding[0]
                dilation = module.dilation[0]
                max_embed_len = (
                    max_embed_len + 2 * padding - dilation * (kernel - 1) - 1
                ) // stride + 1
        full_position_half = position_encode.pe.size(1) // 2
        self.max_embed_len = max_embed_len
        self.position_encode_pe_half = max_embed_len - 1
        self.register_buffer(
            'zero_pad',
            torch.zeros((self.num_heads, max_embed_len, 1), dtype=torch.float32),
        )

        self._fuse_weights(source_decoder_layers)
        pe_full = position_encode.pe[
            :,
            full_position_half - max_embed_len + 1:
            full_position_half + max_embed_len,
        ].to(KV_DTYPE).float()
        pos_p = torch.stack([
            encoder_layer.attn.linear_pos(pe_full)
            .view(-1, self.num_heads, self.head_dim)
            .permute(1, 2, 0)
            for encoder_layer in self.encoder.encoders
        ], dim=0).to(KV_DTYPE)
        self.register_buffer('pos_p', pos_p)
        # These construction-only modules/tables are no longer reachable from forward().
        del self.encoder.embed.out
        del self.encoder.embed.pos_enc

    def _fuse_weights(self, source_decoder_layers):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            folded_norms = []
            for encoder_layer in self.encoder.encoders:
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

                # Fold every layer-norm affine forward into its consumer: norm_mha->qkv, the two FFN norms
                # into w_1, norm_mlp->channel_proj1. csgu.norm is left separate (its conv is zero-padded, so
                # folding the beta would be wrong at boundary frames -> garbage/repeated output).
                fold_norm_into_linear(encoder_layer.norm_mha, qkv)
                fold_norm_into_linear(encoder_layer.norm_ff_macaron, encoder_layer.feed_forward_macaron.w_1)
                fold_norm_into_linear(encoder_layer.norm_ff, encoder_layer.feed_forward.w_1)
                fold_norm_into_linear(encoder_layer.norm_mlp, encoder_layer.cgmlp.channel_proj1[0])
                folded_norms.extend((
                    encoder_layer.norm_ff_macaron,
                    encoder_layer.norm_mha,
                    encoder_layer.norm_mlp,
                    encoder_layer.norm_ff,
                ))
                # Absorb the encoder's residual FFN scale into both output projections.
                encoder_layer.feed_forward_macaron.w_2.weight.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward_macaron.w_2.bias.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward.w_2.weight.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward.w_2.bias.data.mul_(encoder_layer.ff_scale)
                encoder_layer.ff_scale = 1.0

                # depthwise_conv_fusion(x) + x is another depthwise convolution whose centre
                # coefficient is incremented by one.  This removes one full [T, 2H] Add.
                fusion = encoder_layer.depthwise_conv_fusion
                kernel = fusion.kernel_size[0]
                centre = kernel // 2
                fusion.weight.data[:, 0, centre].add_(1.0)

            cross_scale = float(self.cross_head_dim ** -0.25)
            # Snapshot after_norm gamma/beta ONCE: it folds into EVERY cross-attn kv, so we must not let the
            # helper zero it on the first layer (that would leave layers 1+ with gamma1/beta0 = garbage).
            after_gamma = self.encoder.after_norm.weight.data.clone()
            after_beta = self.encoder.after_norm.bias.data.clone()
            self.cross_kv_groups = torch.nn.ModuleList()
            self.cross_kv_group_counts = []
            for group_start in range(0, len(source_decoder_layers), self.cross_kv_group_size):
                group_layers = source_decoder_layers[
                    group_start:group_start + self.cross_kv_group_size
                ]
                group_weights = []
                group_biases = []
                for decoder_layer in group_layers:
                    cross_attn = decoder_layer.src_attn
                    out_features = cross_attn.linear_k.out_features
                    kv_weight = torch.cat([
                        cross_attn.linear_k.weight,
                        cross_attn.linear_v.weight,
                    ], dim=0).detach().clone()
                    kv_bias = torch.cat([
                        _bias_or_zero(cross_attn.linear_k),
                        _bias_or_zero(cross_attn.linear_v),
                    ], dim=0).detach().clone()
                    kv_weight[:out_features].mul_(cross_scale)
                    kv_bias[:out_features].mul_(cross_scale)
                    kv_bias.add_(kv_weight @ after_beta)
                    kv_weight.mul_(after_gamma.unsqueeze(0))
                    group_weights.append(kv_weight)
                    group_biases.append(kv_bias)
                grouped_kv = torch.nn.Linear(
                    group_weights[0].shape[1],
                    sum(weight.shape[0] for weight in group_weights),
                    bias=True,
                )
                grouped_kv.weight.copy_(torch.cat(group_weights, dim=0))
                grouped_kv.bias.copy_(torch.cat(group_biases, dim=0))
                self.cross_kv_groups.append(grouped_kv)
                self.cross_kv_group_counts.append(len(group_layers))
            # after_norm folded into all kv copies above; collapse it to identity once.
            self.encoder.after_norm.weight.data.fill_(1.0)
            self.encoder.after_norm.bias.data.zero_()
            folded_norms.append(self.encoder.after_norm)

            norm_eps = {float(norm.eps) for norm in folded_norms}
            self.folded_norm_eps = norm_eps.pop()
            self.folded_norm_shape = (self.hidden_size,)
            self.register_buffer(
                'folded_norm_weight',
                torch.ones(self.hidden_size, dtype=after_gamma.dtype),
            )
            self.register_buffer(
                'folded_norm_bias',
                torch.zeros(self.hidden_size, dtype=after_gamma.dtype),
            )

    def _folded_norm(self, x):
        return F.layer_norm(
            x,
            self.folded_norm_shape,
            self.folded_norm_weight,
            self.folded_norm_bias,
            self.folded_norm_eps,
        )

    def forward(self, audio):
        # Match the reference DefaultFrontend exactly: int16->[-1,1], torch.stft(center=True -> reflect pad),
        # power spectrum, Slaney mel (f_min=0), clamp(1e-10), natural log, then global-CMVN. No DC removal, no pre-emphasis.
        if self.input_audio_is_int16:
            audio = audio.float()  # 2^-15 is already folded into stft_kernel.
        else:
            audio = audio.float()
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio)
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-10).log()
        mel_features = (mel_features - self.cmvn_mean) * self.inv_std
        embed = self.encoder.embed.conv(mel_features.unsqueeze(0))
        embed_len = embed.shape[-2].unsqueeze(0)
        # Batch is contractually one for audio.  Keep the Transformer path as [time, hidden] so
        # every immutable Linear lowers to one Gemm instead of MatMul + Add.
        x = self.embed(
            embed.transpose(1, 2).contiguous().view(-1, self.embed.in_features)
        )
        pos_p = self.pos_p[:, :, :, self.position_encode_pe_half - embed_len + 1: self.position_encode_pe_half + embed_len].float()
        for idx, encoder_layer in enumerate(self.encoder.encoders):
            x = x + encoder_layer.feed_forward_macaron(self._folded_norm(x))  # ff_scale(0.5) already folded into macaron w_2
            # norm_mha and norm_mlp are now the same affine-free LayerNorm.  Preserve the
            # normalized [T,H] tensor until the cgMLP branch consumes it instead of reducing twice.
            branch_norm = self._folded_norm(x)
            qkv = encoder_layer.attn.qkv(branch_norm).view(-1, 3 * self.num_heads, self.head_dim).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)
            p = pos_p[idx]
            q_with_bias_u = q + encoder_layer.attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(1, 2))
            matrix_bd = torch.matmul(q_with_bias_v, p)
            matrix_bd = rel_shift(matrix_bd, embed_len, self.zero_pad, encoder_layer.attn.h)
            x1 = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v)
            x1 = encoder_layer.attn.linear_out(
                x1.transpose(0, 1).reshape(-1, self.hidden_size)
            )
            x2 = encoder_layer.cgmlp.channel_proj1[0](branch_norm)
            x2 = encoder_layer.cgmlp.channel_proj1[1](x2)
            split_size = encoder_layer.cgmlp.channel_proj1[0].out_features // 2
            x_r, x_g = torch.split(x2, [split_size, split_size], dim=-1)
            x_g = encoder_layer.cgmlp.csgu.conv(
                encoder_layer.cgmlp.csgu.norm(x_g).transpose(0, 1)
            ).transpose(0, 1)
            x2 = encoder_layer.cgmlp.channel_proj2(x_r * x_g)
            x_concat = torch.cat([x1, x2], dim=-1)
            x_concat = encoder_layer.depthwise_conv_fusion(
                x_concat.transpose(0, 1)
            ).transpose(0, 1).reshape(-1, encoder_layer.merge_proj.in_features)
            x = x + encoder_layer.merge_proj(x_concat)
            x = x + encoder_layer.feed_forward(self._folded_norm(x))  # ff_scale(0.5) already folded into ff w_2
            x = encoder_layer.norm_final(x)
        enc_outputs = self._folded_norm(x)
        output_index = 0
        for grouped_kv, layer_count in zip(
            self.cross_kv_groups,
            self.cross_kv_group_counts,
        ):
            cross_kv = grouped_kv(enc_outputs).to(KV_DTYPE).view(
                -1,
                2 * layer_count * self.cross_num_heads,
                self.cross_head_dim,
            ).transpose(0, 1)
            projected = torch.split(cross_kv, self.cross_num_heads, dim=0)
            for local_index in range(layer_count):
                k = projected[2 * local_index]
                v = projected[2 * local_index + 1]
                self.save_en_keys[output_index] = k.transpose(1, 2)  # f16 key   (heads, dim, T)
                self.save_en_values[output_index] = v                # f16 value (heads, T, dim)
                output_index += 1
        return *self.save_en_keys, *self.save_en_values


class DOLPHIN_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Whisper/Qwen Decoder_Embed) so the int
    # token ids never enter the float-only decode graph. The positional xscale is folded into the embedding
    # weight here (the absolute position embedding itself is added inside the decoder main graph).
    def __init__(self, dolphin):
        super(DOLPHIN_DECODER_EMBED, self).__init__()
        source_decoder = dolphin.s2t_model.decoder
        self.embed = copy.deepcopy(source_decoder.embed[0])
        self.embed.weight.data *= source_decoder.embed[1].xscale

    def forward(self, input_ids):
        return self.embed(input_ids)


class DOLPHIN_PREFILL(torch.nn.Module):
    # Prefill-phase position-embedding + causal-mask generator (mirrors Whisper/Qwen Prefill).
    # Consumes the int lengths and emits float position embedding + float attention mask so the decoder
    # main graph stays integer-free.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_PREFILL, self).__init__()
        self.emit_fp32_mask = USE_FP16_KV and COMPUTE_IN_F32
        position_encode = dolphin.s2t_model.decoder.embed[1]
        self.register_buffer(
            'position_weight',
            position_encode.pe[:, :max_seq_len].to(KV_DTYPE).clone(),
        )
        self.register_buffer(
            'attention_mask',
            ((1 - torch.tril(torch.ones(
                [1, max_seq_len, max_seq_len], dtype=torch.int8
            ))) * -128).to(KV_DTYPE),
        )

    def forward(self, ids_len, history_len):
        kv_seq_len = history_len + ids_len
        position_embed = self.position_weight[:, history_len: kv_seq_len].float()
        attention_mask = self.attention_mask[:, :ids_len, :kv_seq_len]
        if self.emit_fp32_mask:
            attention_mask = attention_mask.float()
        return position_embed, attention_mask, kv_seq_len


class DOLPHIN_DECODE(torch.nn.Module):
    # Decode-phase position-embedding generator for the single new token (mirrors Whisper/Qwen Decode).
    # The decode attention mask is all-zeros (the new token attends to every cached position), so it is fed
    # as a static buffer at runtime and no mask is produced here.
    def __init__(self, dolphin, max_seq_len):
        super(DOLPHIN_DECODE, self).__init__()
        position_encode = dolphin.s2t_model.decoder.embed[1]
        self.register_buffer(
            'position_weight',
            position_encode.pe[:, :max_seq_len].to(KV_DTYPE).clone(),
        )

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class DOLPHIN_DECODER(torch.nn.Module):
    def __init__(self, dolphin, num_layers_de):
        super(DOLPHIN_DECODER, self).__init__()
        source_decoder = dolphin.s2t_model.decoder
        # Embed is exported separately.  Copy only the decoder layers, final norm, and vocabulary
        # projection instead of duplicating the entire ASR model (including its encoder and embed table).
        self.layers = copy.deepcopy(source_decoder.decoders)
        self.after_norm = copy.deepcopy(source_decoder.after_norm)
        self.output_layer = copy.deepcopy(source_decoder.output_layer)
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = not USE_FP16_KV or COMPUTE_IN_F32
        self.idx_en_key = num_layers_de + num_layers_de         # en cross-attn keys start (2 * L)
        self.idx_en_value = self.idx_en_key + num_layers_de     # en cross-attn values start (3 * L)
        self.idx_hidden = self.idx_en_value + num_layers_de     # token-embedding input (4 * L)
        self.idx_position = self.idx_hidden + 1                 # position-embedding input (4 * L + 1)
        self.save_de_keys = [None] * num_layers_de
        self.save_de_values = [None] * num_layers_de
        self.num_heads = self.layers._modules['0'].self_attn.h
        self.head_dim = self.layers._modules['0'].self_attn.d_k
        self.hidden_size = self.output_layer.in_features
        self.cross_num_heads = self.layers._modules['0'].src_attn.h
        self.cross_head_dim = self.layers._modules['0'].src_attn.d_k
        self._fuse_weights()
        if REORDER_DOWNPROJ_FOR_QUANT:
            self._reorder_downproj_for_quant(REORDER_KEY)
        if REORDER_OPROJ_FOR_QUANT:
            self._reorder_oproj_for_quant(REORDER_KEY)

    def _fuse_weights(self):
        with torch.no_grad():
            scale = float(self.head_dim ** -0.25)
            cross_scale = float(self.cross_head_dim ** -0.25)
            folded_norms = []
            for decoder_layer in self.layers:
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
                folded_norms.extend((decoder_layer.norm1, decoder_layer.norm2, decoder_layer.norm3))
            # Absorb the decoder's final after_norm into the output projection.
            fold_norm_into_linear(self.after_norm, self.output_layer)
            folded_norms.append(self.after_norm)
            norm_eps = {float(norm.eps) for norm in folded_norms}
            self.folded_norm_eps = norm_eps.pop()
            self.folded_norm_shape = (self.hidden_size,)
            self.register_buffer(
                'folded_norm_weight',
                torch.ones(self.hidden_size, dtype=self.output_layer.weight.dtype),
            )
            self.register_buffer(
                'folded_norm_bias',
                torch.zeros(self.hidden_size, dtype=self.output_layer.weight.dtype),
            )

    def _folded_norm(self, x):
        return F.layer_norm(
            x,
            self.folded_norm_shape,
            self.folded_norm_weight,
            self.folded_norm_bias,
            self.folded_norm_eps,
        )

    @staticmethod
    def _channel_stat(weight, key, dims):
        absolute = weight.abs()
        if key == "rms":
            return (weight * weight).mean(dim=dims).sqrt()
        if key == "L4":
            return absolute.pow(4).mean(dim=dims).pow(0.25)
        if key == "std":
            if isinstance(dims, tuple):
                return weight.reshape(-1, weight.shape[-1]).std(0)
            return weight.std(dim=dims)
        if key != "absmean":
            raise ValueError(f"Unsupported REORDER_KEY: {key!r}")
        return absolute.mean(dim=dims)

    def _reorder_downproj_for_quant(self, key):
        """Permute each FFN w_1 output and its matching w_2 input column."""
        with torch.no_grad():
            for decoder_layer in self.layers:
                w_1 = decoder_layer.feed_forward.w_1
                w_2 = decoder_layer.feed_forward.w_2
                permutation = torch.argsort(self._channel_stat(w_2.weight, key, 0))
                w_1.weight.copy_(w_1.weight[permutation])
                if w_1.bias is not None:
                    w_1.bias.copy_(w_1.bias[permutation])
                w_2.weight.copy_(w_2.weight[:, permutation])

    def _reorder_oproj_for_quant(self, key):
        """Permute self-attention V rows and matching linear_out columns per head."""
        with torch.no_grad():
            for decoder_layer in self.layers:
                attention = decoder_layer.self_attn
                output_weight = attention.linear_out.weight
                qkv = attention.qkv
                output_by_head = output_weight.view(
                    output_weight.shape[0], self.num_heads, self.head_dim
                )
                permutations = [
                    torch.argsort(self._channel_stat(output_by_head[:, head], key, 0))
                    for head in range(self.num_heads)
                ]
                reordered_output = output_by_head.clone()
                for head, permutation in enumerate(permutations):
                    reordered_output[:, head] = output_by_head[:, head, permutation]
                output_weight.copy_(reordered_output.reshape_as(output_weight))

                value_weight = qkv.weight[2 * self.hidden_size:].view(
                    self.num_heads, self.head_dim, qkv.in_features
                )
                reordered_value_weight = value_weight.clone()
                for head, permutation in enumerate(permutations):
                    reordered_value_weight[head] = value_weight[head, permutation]
                qkv.weight[2 * self.hidden_size:].copy_(
                    reordered_value_weight.reshape(self.hidden_size, qkv.in_features)
                )
                if qkv.bias is not None:
                    value_bias = qkv.bias[2 * self.hidden_size:].view(
                        self.num_heads, self.head_dim
                    )
                    reordered_value_bias = value_bias.clone()
                    for head, permutation in enumerate(permutations):
                        reordered_value_bias[head] = value_bias[head, permutation]
                    qkv.bias[2 * self.hidden_size:].copy_(
                        reordered_value_bias.reshape(self.hidden_size)
                    )

    def forward(self, *all_inputs):
        # Pure float graph: token embedding + position embedding are produced by the separate Embed / Prefill /
        # Decode graphs and arrive here as float tensors, so the decode path has no integer I/O.
        hidden_states = all_inputs[self.idx_hidden] + all_inputs[self.idx_position]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        # Prefill emits the causal mask in attention-compute dtype, so every layer
        # shares it directly without an additional precision-boundary Cast.
        attn_mask = attention_mask
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states_norm = self._folded_norm(hidden_states)
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
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            if self.compute_in_f32:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k.float()) + attn_mask, dim=-1), v.float()).transpose(1, 2).reshape(batch_size, -1, self.hidden_size)
            else:
                hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attn_mask, dim=-1), v).transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float()
            hidden_state_attn = decoder_layer.self_attn.linear_out(hidden_state_attn)
            hidden_state_attn += hidden_states
            # Cross-attention against the f16 encoder cross-KV cache. OFF: downcast Q to f16 and run in f16 on the
            # f16 cross cache, context back to f32. ON: keep Q in f32 and upcast the f16 cross K/V to f32 at the
            # matmul use points (the cross cache is produced f16 by the encoder; its I/O dtype is unchanged).
            q = decoder_layer.src_attn.linear_q(self._folded_norm(hidden_state_attn)).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            if self.compute_in_f32:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size))
            else:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(self._folded_norm(hidden_state_cross))
        hidden_states = self._folded_norm(hidden_states[:, -1])
        logits = self.output_layer(hidden_states)
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


def build_v1_supported_languages(token_to_id, special_token_ids):
    """Group configured aliases and resolve every prompt against units.txt."""
    aliases_by_value = {}
    for alias, model_value in LANGUAGE_REGION.items():
        aliases_by_value.setdefault(model_value, []).append(alias)
    aliases_by_value.setdefault("auto-auto", [])
    # Uighur also has a trained unknown-region combination in the model token
    # table, in addition to the public alias map's preferred ug-CN spelling.
    if "<ug>" in token_to_id and "<NULL>" in token_to_id:
        aliases_by_value.setdefault("ug-NULL", [])

    catalog = {}
    required_languages = set()
    required_regions = set()
    for model_value in aliases_by_value:
        language, _, region = model_value.partition("-")
        if model_value != "auto-auto":
            required_languages.add(language)
        if region != "auto":
            required_regions.add(region)

    language_start = special_token_ids["asr"] + 1
    region_start = None
    for token_id in range(language_start, special_token_ids["notimestamp"]):
        token = next(
            (piece for piece, value in token_to_id.items() if value == token_id),
            None,
        )
        body = token[1:-1]
        if body.upper() == body and body.lower() != body:
            region_start = token_id
            break
    language_end = region_start
    region_end = special_token_ids["notimestamp"]

    for model_value, aliases in aliases_by_value.items():
        language, region = model_value.split("-", 1)
        if model_value == "auto-auto":
            prompt_token_ids = []
        else:
            language_id = token_to_id[f"<{language}>"]
            if region == "auto":
                prompt_token_ids = [language_id]
            else:
                region_id = token_to_id[f"<{region}>"]
                prompt_token_ids = [
                    language_id,
                    region_id,
                    special_token_ids["asr"],
                    special_token_ids["notimestamp"],
                ]
        catalog[model_value] = {
            "name": model_value,
            "aliases": sorted(set(aliases), key=str.casefold),
            "prompt_token_ids": prompt_token_ids,
        }
    return catalog, (language_start, language_end, region_start, region_end)


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


print('\nExport start...\n')
with torch.inference_mode():
    _raw_onnx_dir.mkdir(parents=True, exist_ok=True)
    if 'small' in model_path.lower():
        model_size = 'small'
    else:
        model_size = 'base'
    model = dolphin.load_model(model_size, model_path, "cpu").eval()
    # The current dolphin package exposes encoder/decoder/ctc directly on the
    # ASRModel instance; the former `s2t_model` wrapper attribute was removed.
    # Alias it back to the model itself so the export logic below stays intact.
    object.__setattr__(model, "s2t_model", model)
    # Build the vocab in token-id order from units.txt ("<token> <id>" per line),
    # mirroring dolphin.tokenizer._read_symbol_table parsing.
    id_to_token = {}
    with open(os.path.join(model_path, "units.txt"), 'r', encoding='utf-8') as units_file:
        for vocab_line in units_file:
            arr = vocab_line.strip().split()
            if len(arr) >= 2:
                id_to_token[int(arr[1])] = arr[0]
    with open(save_vocab, 'w', encoding='utf-8') as file:
        for idx in range(len(id_to_token)):
            file.write(id_to_token[idx] + '\n')
    # Copy the SentencePiece bpe.model next to the exported vocab so the ONNX folder is
    # self-contained and Inference_Dolphin_ONNX.py no longer needs the original model_path.
    src_bpe_model = os.path.join(model_path, "bpe.model")
    dst_bpe_model = str(_raw_onnx_dir / "bpe.model")
    if os.path.exists(src_bpe_model):
        shutil.copyfile(src_bpe_model, dst_bpe_model)
        print(f"Copied bpe.model -> {dst_bpe_model}")
    else:
        print(f"Note: {src_bpe_model} not found; skipping bpe.model copy.")
    HIDDEN_SIZE = model.s2t_model.decoder.output_layer.in_features
    NUM_HEAD_EN = model.s2t_model.encoder.encoders._modules['0'].attn.h
    NUM_HEAD_DE = model.s2t_model.decoder.decoders._modules['0'].self_attn.h
    HEAD_DIM_DE = model.s2t_model.decoder.decoders._modules['0'].self_attn.d_k
    NUM_LAYER_DE = len(model.s2t_model.decoder.decoders)
    VOCAB_SIZE = model.s2t_model.vocab_size
    STFT_SIGNAL_LENGTH = _MODEL_MAX_AUDIO_SAMPLES // HOP_LENGTH + 1

    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, pad_mode='reflect', center_pad=True).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    dolphin_encoder = DOLPHIN_ENCODER(
        model,
        custom_stft,
        NFFT_STFT,
        N_MELS,
        SAMPLE_RATE,
        PRE_EMPHASIZE,
        NUM_LAYER_DE,
        _MODEL_MAX_AUDIO_SAMPLES,
        CROSS_KV_GROUP_SIZE,
    )
    output_names = []
    _audio_export_dtype = {"INT16": torch.int16, "F32": torch.float32, "F16": torch.float16}[INPUT_AUDIO_DTYPE]
    audio = torch.ones((1, 1, _MODEL_MAX_AUDIO_SAMPLES), dtype=_audio_export_dtype)
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
    del custom_stft
    del name
    del output_names
    del dynamic_axes
    gc.collect()

    # ── Decoder token-embedding graph (keeps int ids out of the decoder; xscale folded into the embedding) ──
    dolphin_embed = DOLPHIN_DECODER_EMBED(model)
    embed_input_ids = torch.ones((1, 5), dtype=torch.int32)
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
    dolphin_prefill = DOLPHIN_PREFILL(model, _MODEL_MAX_DECODER_SEQ_LEN)
    prefill_ids_len = torch.tensor([5], dtype=torch.int64)
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
    dolphin_decode = DOLPHIN_DECODE(model, _MODEL_MAX_DECODER_SEQ_LEN)
    decode_kv_seq_len = torch.tensor([5], dtype=torch.int64)
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
    save_encoder_key = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, STFT_SIGNAL_LENGTH // 2 + 1), dtype=KV_DTYPE)
    save_encoder_value = torch.zeros((NUM_HEAD_DE, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_DE), dtype=KV_DTYPE)
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
    
    # Trace-only values for dynamic selection and direct-penalty inputs.
    logits = torch.ones((1, VOCAB_SIZE), dtype=torch.float32)
    save_id = torch.zeros((1, 10), dtype=torch.int32)  # Dummy history length.
    penalty_value = torch.tensor([1.0], dtype=torch.float32)
    penalty_range = torch.tensor([20], dtype=torch.int64)

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

    # ── Top-K / Top-P sampling with standard repetition penalty ──
    sampling_temperature = torch.tensor([0.8], dtype=torch.float32)
    sampling_top_k = torch.tensor([50], dtype=torch.int32)
    sampling_top_p = torch.tensor([0.95], dtype=torch.float32)
    sampling_repetition_penalty = torch.tensor([1.0], dtype=torch.float32)
    torch.onnx.export(
        TOPK_TOPP_SAMPLING(),
        (
            logits,
            sampling_temperature,
            sampling_top_k,
            sampling_top_p,
            sampling_repetition_penalty,
            save_id,
        ),
        onnx_model_Sampling,
        input_names=[
            'logits', 'temperature', 'top_k', 'top_p',
            'repetition_penalty', 'previous_ids'
        ],
        output_names=['sampled_id', 'save_id_out'],
        dynamic_axes={
            'previous_ids': {1: 'history_len'},
            'save_id_out': {1: 'history_len_out'},
        },
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
    gc.collect()

    token_to_id = {token: idx for idx, token in id_to_token.items()}

    required_special_tokens = {
        "blank": "<blank>",
        "sos": "<sos>",
        "stop": "<eos>",
        "asr": "<asr>",
        "notimestamp": "<notimestamp>",
    }
    special_token_ids = {
        role: int(token_to_id[piece])
        for role, piece in required_special_tokens.items()
    }
    supported_languages, token_ranges = build_v1_supported_languages(
        token_to_id,
        special_token_ids,
    )
    language_token_start, language_token_end, region_token_start, region_token_end = (
        token_ranges
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
            "max_seq_len": _MODEL_MAX_DECODER_SEQ_LEN,
            "sample_rate": SAMPLE_RATE,
            "audio_pcm_scale": 32768,
            "prompt_control_token_count": 4,
            "language_token_start": language_token_start,
            "language_token_end": language_token_end,
            "region_token_start": region_token_start,
            "region_token_end": region_token_end,
            "special_token_ids": special_token_ids,
            "supported_languages": supported_languages,
        },
    )

    write_metadata_carrier(onnx_model_Metadata, onnx_metadata)

    gc.collect()

import Shared_Merged

if ONNX_DIR.exists():
    shutil.rmtree(ONNX_DIR)
ONNX_DIR.mkdir(parents=True)
print("\n[SharedMerged] Building Dolphin v1 strategy graphs + shared initializer bundle ...")
_bundle = Shared_Merged.build_shared_merged_bundle(
    _raw_onnx_dir,
    out_folder=ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
    retain_prefill_logits=True,
    probe_aware=True,
)
Shared_Merged.copy_runtime_standalones(
    _raw_onnx_dir,
    ONNX_DIR,
    model_file_names=MODEL_FILE_NAMES,
    include_encoder=False,
)
shutil.copy2(save_vocab, ONNX_DIR / Path(save_vocab).name)
if os.path.exists(dst_bpe_model):
    shutil.copy2(dst_bpe_model, ONNX_DIR / "bpe.model")
for _name, _path in _bundle["graphs"].items():
    print(f"    {_name} ({Path(_path).stat().st_size} bytes)")
write_metadata_carrier(ONNX_DIR / MODEL_FILE_NAMES["metadata"], onnx_metadata)
print(
    f"    {MODEL_FILE_NAMES['shared_initializers_data']} "
    f"({Path(_bundle['shared_data']).stat().st_size} bytes)"
)
print("    Standalone graphs: Encoder + Metadata; no Qwen KV helpers.")

_raw_onnx_temp.cleanup()
print(f"[Raw] Deleted temporary split export at {_raw_onnx_dir}")
print('\nExport done!\n')
subprocess.run(
    [
        sys.executable,
        str(Path(_SCRIPT_DIR) / "Inference_Dolphin_ONNX.py"),
        "--onnx-folder",
        str(ONNX_DIR),
    ],
    cwd=_SCRIPT_DIR,
    check=True,
)
