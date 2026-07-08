import gc
import subprocess
import sys
import os
import shutil
import copy
import torch
import dolphin                         
import torchaudio
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.
try:
    import sentencepiece as spm
except ImportError:
    spm = None
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 


model_path             = "/home/DakeQQ/Downloads/dolphin-small"                                   # The dolphin project download path. Currently, only support dolphin-small and dolphin-base.
onnx_folder            = os.path.join(_SCRIPT_DIR, "Dolphin_ONNX")                              # Local folder next to this script holding all exported ONNX graphs; created automatically if missing.
os.makedirs(onnx_folder, exist_ok=True)
onnx_model_Metadata    = os.path.join(onnx_folder, "Dolphin_Metadata.onnx")                     # Tiny metadata carrier graph.
onnx_model_Encoder     = os.path.join(onnx_folder, "Dolphin_Encoder.onnx")                      # The exported onnx encoder model path.
onnx_model_Decoder     = os.path.join(onnx_folder, "Dolphin_Decoder.onnx")                      # The exported onnx decoder (main, pure-float) model path.
onnx_model_Embed       = os.path.join(onnx_folder, "Dolphin_Decoder_Embed.onnx")                # Token-embedding graph (keeps int ids out of the decoder).
onnx_model_Prefill     = os.path.join(onnx_folder, "Dolphin_Position_Mask_Prefill.onnx")        # Prefill position-embedding + causal-mask graph.
onnx_model_Decode      = os.path.join(onnx_folder, "Dolphin_Position_Mask_Decode.onnx")         # Decode position-embedding graph for the single new token.
onnx_model_Greedy      = os.path.join(onnx_folder, "Dolphin_Greedy_Search.onnx")                # Greedy argmax + save_id history (used with Apply_Penality).
onnx_model_Argmax      = os.path.join(onnx_folder, "Dolphin_Argmax.onnx")                       # Bare argmax (greedy decoding without a repetition penalty).
onnx_model_First_Beam  = os.path.join(onnx_folder, "Dolphin_First_Beam_Search.onnx")            # First beam-search step.
onnx_model_Second_Beam = os.path.join(onnx_folder, "Dolphin_Second_Beam_Search.onnx")           # Subsequent beam-search steps.
onnx_model_Penality    = os.path.join(onnx_folder, "Dolphin_Apply_Penality.onnx")               # Sliding-window repetition penalty on the logits.
save_vocab             = os.path.join(onnx_folder, "vocab_Dolphin.txt")                         # The exported Dolphin vocab path.
TARGET_LANGUAGE        = "Auto-Auto"                                                            # See 'LANGUAGE_REGION' for detail.



USE_BEAM_SEARCH    = False    # Use beam search or greedy search.
INPUT_AUDIO_LENGTH = 480000   # The maximum input audio length. Must less than 480000 (30 seconds).
MAX_SEQ_LEN        = 448      # It should less than 448.
REPEAT_PENALITY    = 1.0      # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE     = 20       # Penalizes the most recent output. "30" means the last 30 tokens.
MAX_BEAM_SIZE      = 10       # Max beams for exported model.
TOP_K              = 3        # The top k candidate in decoding.
BEAM_SIZE          = 3        # Number of beams in searching.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WINDOW_TYPE     = 'hann'      # Type of window function used in the STFT.
N_MELS          = 80          # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT       = 512         # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH   = 400         # Length of windowing, edit it carefully.
HOP_LENGTH      = 160         # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE   = 0.0         # Dolphin's DefaultFrontend applies NO pre-emphasis; keep 0.0 to match the reference feature extractor.
SAMPLE_RATE     = 16000       # The model parameter, do not edit the value.
INPUT_AUDIO_DTYPE   = "INT16" # ONNX audio input dtype: "INT16", "F32", or "F16". Must match export. "INT16" feeds raw PCM (÷32768 in-graph); "F32"/"F16" feed audio pre-normalised to [-1, 1].
STOP_TOKEN      = [40000]     # 40000 is the end token for Dolphin series model.
COMPUTE_IN_F32  = False       # F16 KV-cache compute precision. False = minimum-cast f16 attention (self + cross Q@K/mask/softmax/attn@V run in f16 on the f16 caches; the context is cast back to f32). True = keep the f16 KV *storage* (cache I/O dtype unchanged) but upcast K/V (and the mask, internally) to f32 at the attention use points, keeping Q/softmax in f32 (f16 storage, f32 compute).
OPSET           = 18          # ONNX opset version for the export.



if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH
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
    x_padded = torch.cat([zero_pad[:, :x_len].float(), x], dim=-1)
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
    def __init__(self, dolphin, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(DOLPHIN_ENCODER, self).__init__()
        self.dolphin = copy.deepcopy(dolphin.s2t_model)
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.nfft_stft = nfft_stft
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        # Calculate frequency bins for STFT and filterbank
        self.stft_bins = nfft_stft // 2 + 1  # Number of frequency bins from STFT

        # Create mel filterbank - ensure it matches STFT output dimensions
        self.fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=self.stft_bins,
            f_min=0,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm='slaney',
            mel_scale='slaney'
        ).transpose(0, 1).unsqueeze(0)
                     
        # Normalization parameters (global_cmvn stores mean/std as float64; cast to float32)
        self.inv_int16 = float(1.0 / 32768.0)
        # int16 audio is raw PCM (normalised in forward via ÷32768); f32/f16 audio is
        # assumed pre-normalised to [-1, 1], so the in-graph division is skipped.
        self.input_audio_is_int16 = (INPUT_AUDIO_DTYPE == "INT16")
        self.cmvn_mean = self.dolphin.encoder.global_cmvn.mean.float()
        self.inv_std = (1.0 / self.dolphin.encoder.global_cmvn.std).float()

        # Encoder components (keeping your original structure)
        self.zero_pad = torch.zeros((self.dolphin.encoder.encoders._modules['0'].attn.h, 2048, 1), dtype=torch.int8)
        self.save_en_keys = [None] * num_layers_de
        self.save_en_values = [None] * num_layers_de
        self.embed = self.dolphin.encoder.embed.out[0]
        self.position_encode = self.dolphin.encoder.embed.pos_enc
        self.embed.weight.data *= self.position_encode.xscale
        self.embed.bias.data *= self.position_encode.xscale
        self.position_encode_pe_half = self.position_encode.pe.size(1) // 2
        # Pre-compute the full symmetric rel-pos table once; forward only slices+casts the needed window.
        self.position_encode_pe = self.position_encode.pe.half()
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

                # Fold every layer-norm affine forward into its consumer: norm_mha->qkv, the two FFN norms
                # into w_1, norm_mlp->channel_proj1. csgu.norm is left separate (its conv is zero-padded, so
                # folding the beta would be wrong at boundary frames -> garbage/repeated output).
                fold_norm_into_linear(encoder_layer.norm_mha, qkv)
                fold_norm_into_linear(encoder_layer.norm_ff_macaron, encoder_layer.feed_forward_macaron.w_1)
                fold_norm_into_linear(encoder_layer.norm_ff, encoder_layer.feed_forward.w_1)
                fold_norm_into_linear(encoder_layer.norm_mlp, encoder_layer.cgmlp.channel_proj1[0])
                # Absorb the encoder's residual FFN scale into both output projections.
                encoder_layer.feed_forward_macaron.w_2.weight.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward_macaron.w_2.bias.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward.w_2.weight.data.mul_(encoder_layer.ff_scale)
                encoder_layer.feed_forward.w_2.bias.data.mul_(encoder_layer.ff_scale)
                encoder_layer.ff_scale = 1.0

            cross_scale = float(self.cross_head_dim ** -0.25)
            # Snapshot after_norm gamma/beta ONCE: it folds into EVERY cross-attn kv, so we must not let the
            # helper zero it on the first layer (that would leave layers 1+ with gamma1/beta0 = garbage).
            after_gamma = self.dolphin.encoder.after_norm.weight.data.clone()
            after_beta = self.dolphin.encoder.after_norm.bias.data.clone()
            for decoder_layer in self.dolphin.decoder.decoders:
                cross_attn = decoder_layer.src_attn
                out_features = cross_attn.linear_k.out_features
                kv = torch.nn.Linear(cross_attn.linear_k.in_features, out_features * 2, bias=True)
                kv.weight.copy_(torch.cat([cross_attn.linear_k.weight, cross_attn.linear_v.weight], dim=0))
                kv.bias.copy_(torch.cat([_bias_or_zero(cross_attn.linear_k), _bias_or_zero(cross_attn.linear_v)], dim=0))
                kv.weight.data[:out_features].mul_(cross_scale)
                kv.bias.data[:out_features].mul_(cross_scale)
                # Absorb the encoder's after_norm affine into every cross-attn kv (enc_outputs = after_norm(x)).
                kv.bias.data.add_(kv.weight.data @ after_beta)
                kv.weight.data.mul_(after_gamma.unsqueeze(0))
                cross_attn.kv = kv
                del cross_attn.linear_k, cross_attn.linear_v
            # after_norm folded into all kv copies above; collapse it to identity once.
            self.dolphin.encoder.after_norm.weight.data.fill_(1.0)
            self.dolphin.encoder.after_norm.bias.data.zero_()

    def forward(self, audio):
        # Match the reference DefaultFrontend exactly: int16->[-1,1], torch.stft(center=True -> reflect pad),
        # power spectrum, Slaney mel (f_min=0), clamp(1e-10), natural log, then global-CMVN. No DC removal, no pre-emphasis.
        if self.input_audio_is_int16:
            audio = audio.float() * self.inv_int16
        else:
            audio = audio.float()
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio)
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-10).log()
        mel_features = (mel_features - self.cmvn_mean) * self.inv_std
        embed = self.dolphin.encoder.embed.conv(mel_features.unsqueeze(0))
        embed_len = embed.shape[-2].unsqueeze(0)
        x = self.embed(embed.transpose(1, 2).contiguous().view(1, embed_len, -1))
        pos_p = self.pos_p[:, :, :, self.position_encode_pe_half - embed_len + 1: self.position_encode_pe_half + embed_len].float()
        for idx, encoder_layer in enumerate(self.dolphin.encoder.encoders):
            x = x + encoder_layer.feed_forward_macaron(encoder_layer.norm_ff_macaron(x))  # ff_scale(0.5) already folded into macaron w_2
            x1 = encoder_layer.norm_mha(x)
            qkv = encoder_layer.attn.qkv(x1).view(-1, 3 * self.num_heads, self.head_dim).transpose(0, 1)
            q, k, v = qkv.split(self.num_heads, dim=0)
            p = pos_p[idx]
            q_with_bias_u = q + encoder_layer.attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k.transpose(1, 2))
            matrix_bd = torch.matmul(q_with_bias_v, p)
            matrix_bd = rel_shift(matrix_bd, embed_len, self.zero_pad, encoder_layer.attn.h)
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
            cross_kv = decoder_layer.src_attn.kv(enc_outputs).half().view(-1, 2 * self.cross_num_heads, self.cross_head_dim).transpose(0, 1)
            k, v = cross_kv.split(self.cross_num_heads, dim=0)
            self.save_en_keys[idx] = k.transpose(1, 2)       # f16 cross-attention key   (num_heads, head_dim, T)
            self.save_en_values[idx] = v                     # f16 cross-attention value (num_heads, T, head_dim)
        return *self.save_en_keys, *self.save_en_values


class DOLPHIN_DECODER_EMBED(torch.nn.Module):
    # Token-embedding graph kept separate from the decoder (mirrors Whisper/Qwen Decoder_Embed) so the int
    # token ids never enter the float-only decode graph. The positional xscale is folded into the embedding
    # weight here (the absolute position embedding itself is added inside the decoder main graph).
    def __init__(self, dolphin):
        super(DOLPHIN_DECODER_EMBED, self).__init__()
        self.dolphin = copy.deepcopy(dolphin.s2t_model)
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
        position_encode = copy.deepcopy(dolphin.s2t_model.decoder.embed[1])
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
        position_encode = copy.deepcopy(dolphin.s2t_model.decoder.embed[1])
        self.register_buffer('position_weight', position_encode.pe[:, :max_seq_len].half())

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        position_embed = self.position_weight[:, kv_seq_len].float()
        return position_embed, kv_seq_len_next


class DOLPHIN_DECODER(torch.nn.Module):
    def __init__(self, dolphin, num_layers_de):
        super(DOLPHIN_DECODER, self).__init__()
        self.dolphin = copy.deepcopy(dolphin.s2t_model)
        self.num_layers_de = num_layers_de
        self.compute_in_f32 = COMPUTE_IN_F32
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
        # f16-storage / f32-compute (COMPUTE_IN_F32): the causal mask is kept f16 at the graph boundary (I/O
        # dtype unchanged) and upcast to f32 ONCE here, shared by every layer. Minimum-cast path uses it as-is (f16).
        attn_mask = attention_mask.float() if self.compute_in_f32 else attention_mask
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            hidden_states_norm = decoder_layer.norm1(hidden_states)
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
                k = k.half()   # f16 K storage (no-op in the minimum-cast path: qkv is already f16)
                v = v.half()   # f16 V storage
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
            q = decoder_layer.src_attn.linear_q(decoder_layer.norm2(hidden_state_attn)).view(batch_size, -1, self.cross_num_heads, self.cross_head_dim).transpose(1, 2)
            if self.compute_in_f32:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.idx_en_key].float()), dim=-1), all_inputs[idx + self.idx_en_value].float())
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size))
            else:
                hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q.half(), all_inputs[idx + self.idx_en_key]), dim=-1), all_inputs[idx + self.idx_en_value])
                hidden_state_cross = decoder_layer.src_attn.linear_out(hidden_state_cross.transpose(1, 2).reshape(batch_size, -1, self.hidden_size).float())
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(decoder_layer.norm3(hidden_state_cross))
        hidden_states = self.dolphin.decoder.after_norm(hidden_states[:, -1])
        logits = self.dolphin.decoder.output_layer(hidden_states)
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


print('\nExport start...\n')
with torch.inference_mode():
    if 'small' in model_path.lower():
        model_size = 'small'
    else:
        model_size = 'base'
    model = dolphin.load_model(model_size, model_path, "cpu")
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
    dst_bpe_model = os.path.join(onnx_folder, "bpe.model")
    if os.path.exists(src_bpe_model):
        shutil.copyfile(src_bpe_model, dst_bpe_model)
        print(f"Copied bpe.model -> {dst_bpe_model}")
    else:
        print(f"Note: {src_bpe_model} not found; skipping bpe.model copy.")
    HIDDEN_SIZE = model.s2t_model.decoder.output_layer.in_features
    NUM_HEAD_EN = model.s2t_model.encoder.encoders._modules['0'].attn.h
    NUM_HEAD_DE = model.s2t_model.decoder.decoders._modules['0'].self_attn.h
    HEAD_DIM_EN = model.s2t_model.encoder.encoders._modules['0'].attn.d_k
    HEAD_DIM_DE = model.s2t_model.decoder.decoders._modules['0'].self_attn.d_k
    NUM_LAYER_DE = len(model.s2t_model.decoder.decoders)
    VOCAB_SIZE = model.s2t_model.vocab_size
    CROSS_NUM_HEAD_DE = model.s2t_model.decoder.decoders._modules['0'].src_attn.h
    CROSS_HEAD_DIM_DE = model.s2t_model.decoder.decoders._modules['0'].src_attn.d_k
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1

    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE, pad_mode='reflect', center_pad=True).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    dolphin_encoder = DOLPHIN_ENCODER(model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)
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
    dolphin_prefill = DOLPHIN_PREFILL(model, MAX_SEQ_LEN)
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
    dolphin_decode = DOLPHIN_DECODE(model, MAX_SEQ_LEN)
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
            "producer": "Export_Dolphin.py",
            "model_variant": model_size,
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
            "stop_token_ids": ",".join(str(t) for t in [_special_token_id("<eos>", STOP_TOKEN[0])]),
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
    
print('\nExport done!\n')
print('Running ONNX Runtime demo via Inference_Dolphin_ONNX.py ...')
subprocess.run(
    [sys.executable, os.path.join(_SCRIPT_DIR, "Inference_Dolphin_ONNX.py"), "--onnx-folder", onnx_folder],
    check=True,
)
