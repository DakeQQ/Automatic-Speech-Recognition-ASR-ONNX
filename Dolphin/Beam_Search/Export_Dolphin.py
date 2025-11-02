import gc
import copy
import time
import torch
import dolphin                         # Currently, not support for Python >= 3.12
import torchaudio
import onnxruntime
import numpy as np
from pydub import AudioSegment
from STFT_Process import STFT_Process  # The custom STFT/ISTFT can be exported in ONNX format.

model_path = "/home/DakeQQ/Downloads/dolphin-small"                                     # The dolphin project download path. Currently, only support dolphin-small and dolphin-base.
onnx_model_A = "/home/DakeQQ/Downloads/Dolphin_ONNX/Dolphin_Encoder.onnx"               # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Dolphin_ONNX/Dolphin_Decoder.onnx"               # The exported onnx model path.
onnx_model_C = '/home/DakeQQ/Downloads/Dolphin_ONNX/Greedy_Search.onnx'                 # Assign a path where the exported onnx model stored.
onnx_model_D = '/home/DakeQQ/Downloads/Dolphin_ONNX/First_Beam_Search.onnx'             # Assign a path where the exported onnx model stored.
onnx_model_E = '/home/DakeQQ/Downloads/Dolphin_ONNX/Second_Beam_Search.onnx'            # Assign a path where the exported onnx model stored.
onnx_model_F = '/home/DakeQQ/Downloads/Dolphin_ONNX/Reset_Penality.onnx'                # Assign a path where the exported onnx model stored.
onnx_model_G = '/home/DakeQQ/Downloads/Dolphin_ONNX/Argmax.onnx'                        # Assign a path where the exported onnx model stored.

save_vocab = "/home/DakeQQ/Downloads/Dolphin_ONNX/vocab_Dolphin.txt"                    # The exported Dolphin vocab path.
TARGET_LANGUAGE = "Auto-Auto"                                                           # See 'LANGUAGE_REGION' for detail.
test_audio = ["../example/zh.mp3", "../example/zh-Shanghai.wav", "../example/ja.mp3", "../example/ko.mp3"]  # The test audio list.


USE_BEAM_SEARCH = True                                      # Use beam search or greedy search.
INPUT_AUDIO_LENGTH = 240000                                 # The maximum input audio length. Must less than 480000 (30 seconds).
MAX_SEQ_LEN = 72                                            # It should less than 448.
REPEAT_PENALITY = 0.9                                       # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                                         # Penalizes the most recent output. "30" means the last 30 tokens.
MAX_BEAM_SIZE = 8                                           # Max beams for exported model.
TOP_K = 3                                                   # The top k candidate in decoding.
BEAM_SIZE = 3                                               # Number of beams in searching.
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
WINDOW_TYPE = 'hann'                                        # Type of window function used in the STFT
N_MELS = 80                                                 # Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
WINDOW_LENGTH = 400                                         # Length of windowing, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
STOP_TOKEN = [40000]                                        # 40000 is the end token for Dolphin series model.


if HOP_LENGTH > INPUT_AUDIO_LENGTH:
    HOP_LENGTH = INPUT_AUDIO_LENGTH


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * float(scaling_factor)).astype(np.int16)
  

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

    "自动"                          : "auto",
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
    def __init__(self, filename):
        self.str_to_idx = {}
        self.idx_to_str = {}
        self.num_vocab = 0
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token
        self.num_vocab = len(self.idx_to_str)

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def num_vocab(self):
        return self.num_vocab
        

def rel_shift(x, x_len, zero_pad, n_head):
    x_padded = torch.cat([zero_pad[:, :x_len].float(), x], dim=-1)
    x_padded = x_padded.view(n_head, -1, x_len)
    x = x_padded[:, 1:].view_as(x)
    return x[:, :, :x_len]


class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, logits, repeat_penality, penality_value, batch_size):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        batch_indices = self.batch_indices[:batch_size].long()
        repeat_penality[batch_indices, max_logits_idx.squeeze(-1)] *= penality_value
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-8]
        save_id = all_inputs[-7]
        repeat_penality = all_inputs[-6]
        previous_prob = all_inputs[-5]
        batch_indices = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=False)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[[0]]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count, batch_indices):
        repeat_penality[batch_indices, save_id[batch_indices, penality_reset_count[batch_indices]]] = 1.0
        penality_reset_count += 1
        return save_id, repeat_penality, penality_reset_count
    
    
class ARGMAX(torch.nn.Module):
    def __init__(self):
        super(ARGMAX, self).__init__()
        pass
    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=False).int()
    

class DOLPHIN_ENCODER(torch.nn.Module):
    def __init__(self, dolphin, stft_model, nfft_stft, n_mels, sample_rate, pre_emphasis,
                 num_layers_de):
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
            f_min=20,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
            norm='slaney',
            mel_scale='slaney'
        ).transpose(0, 1).unsqueeze(0)
                     
        # Normalization parameters
        self.inv_int16 = float(1.0 / 32768.0)
        self.inv_std = 1.0 / self.dolphin.normalize.std

        # Encoder components (keeping your original structure)
        self.zero_pad = torch.zeros((self.dolphin.encoder.encoders._modules['0'].attn.h, 2048, 1), dtype=torch.int8)
        self.save_en_keys = [None] * num_layers_de
        self.save_en_values = [None] * num_layers_de
        self.embed = self.dolphin.encoder.embed.out[0]
        self.position_encode = self.dolphin.encoder.embed.out[1]
        self.position_encode.pe = self.position_encode.pe.half()
        self.embed.weight.data *= self.position_encode.xscale
        self.embed.bias.data *= self.position_encode.xscale
        self.position_encode_pe_half = self.position_encode.pe.size(1) // 2

    def forward(self, audio):
        audio = audio.float() * self.inv_int16
        audio = audio - torch.mean(audio, dim=-1, keepdim=True)
        if self.pre_emphasis > 0:
            audio = torch.cat([audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]], dim=-1)
        real_part, imag_part = self.stft_model(audio, 'constant')
        mel_features = torch.matmul(self.fbank, real_part * real_part + imag_part * imag_part).transpose(1, 2).clamp(min=1e-5).log()
        mel_features = (mel_features - self.dolphin.normalize.mean) * self.inv_std
        embed = self.dolphin.encoder.embed.conv(mel_features.unsqueeze(0))
        embed_len = embed.shape[-2].unsqueeze(0)
        x = self.embed(embed.transpose(1, 2).contiguous().view(1, embed_len, -1))
        pos_emb = self.position_encode.pe[:, self.position_encode_pe_half - embed_len + 1: self.position_encode_pe_half + embed_len].float()
        for encoder_layer in self.dolphin.encoder.encoders:
            x = x + encoder_layer.ff_scale * encoder_layer.feed_forward_macaron(encoder_layer.norm_ff_macaron(x))
            x1 = encoder_layer.norm_mha(x)
            q = torch.matmul(x1, encoder_layer.attn.linear_q.weight.data) + encoder_layer.attn.linear_q.bias.data
            k = (torch.matmul(x1, encoder_layer.attn.linear_k.weight.data) + encoder_layer.attn.linear_k.bias.data).transpose(1, 2)
            v = torch.matmul(x1, encoder_layer.attn.linear_v.weight.data) + encoder_layer.attn.linear_v.bias.data
            p = torch.matmul(pos_emb, encoder_layer.attn.linear_pos.weight.data).transpose(1, 2)
            q_with_bias_u = q + encoder_layer.attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k)
            matrix_bd = torch.matmul(q_with_bias_v, p)
            matrix_bd = rel_shift(matrix_bd, embed_len, self.zero_pad, encoder_layer.attn.h)
            x1 = torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v)
            x1 = torch.matmul(x1, encoder_layer.attn.linear_out.weight.data).sum(dim=0, keepdim=True) + encoder_layer.attn.linear_out.bias.data
            x2 = encoder_layer.cgmlp.channel_proj1(encoder_layer.norm_mlp(x))
            x_r, x_g = x2.chunk(2, dim=-1)
            x_g = encoder_layer.cgmlp.csgu.conv(encoder_layer.cgmlp.csgu.norm(x_g).transpose(1, 2)).transpose(1, 2)
            x2 = encoder_layer.cgmlp.channel_proj2(x_r * x_g)
            x_concat = torch.cat([x1, x2], dim=-1)
            x_concat = x_concat + encoder_layer.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
            x = x + encoder_layer.merge_proj(x_concat)
            x = x + encoder_layer.ff_scale * encoder_layer.feed_forward(encoder_layer.norm_ff(x))
            x = encoder_layer.norm_final(x)
        enc_outputs = self.dolphin.encoder.after_norm(x)
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            self.save_en_keys[idx] = (torch.matmul(enc_outputs, decoder_layer.src_attn.linear_k.weight.data) + decoder_layer.src_attn.linear_k.bias.data).transpose(1, 2)
            self.save_en_values[idx] = torch.matmul(enc_outputs, decoder_layer.src_attn.linear_v.weight.data) + decoder_layer.src_attn.linear_v.bias.data
        return *self.save_en_keys, *self.save_en_values


class DOLPHIN_DECODER(torch.nn.Module):
    def __init__(self, dolphin, max_seq_len, num_layers_de):
        super(DOLPHIN_DECODER, self).__init__()
        self.dolphin = copy.deepcopy(dolphin.s2t_model)
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.embed = self.dolphin.decoder.embed[0]
        self.position_encode = self.dolphin.decoder.embed[1]
        self.position_encode.pe = self.position_encode.pe[:, :max_seq_len].half()
        self.embed.weight.data *= self.position_encode.xscale
        self.num_layers_de = num_layers_de
        self.num_layers_de_2 = num_layers_de + num_layers_de
        self.num_layers_de_2_plus_1 = self.num_layers_de_2 + 1
        self.num_layers_de_2_plus_2 = self.num_layers_de_2 + 2
        self.num_layers_de_3_plus = self.num_layers_de_2_plus_2 + num_layers_de
        self.save_de_keys = [None] * num_layers_de
        self.save_de_values = [None] * num_layers_de

    def forward(self, *all_inputs):
        input_ids = all_inputs[self.num_layers_de_2]
        history_len = all_inputs[self.num_layers_de_2_plus_1]
        language_start = all_inputs[-4]
        language_end = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        hidden_states = self.embed(input_ids) + self.position_encode.pe[:, history_len: kv_seq_len].float()
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        hidden_states = hidden_states.unsqueeze(1)
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            hidden_states_norm = decoder_layer.norm1(hidden_states)
            q = torch.matmul(hidden_states_norm, decoder_layer.self_attn.linear_q.weight.data) + decoder_layer.self_attn.linear_q.bias.data
            k = (torch.matmul(hidden_states_norm, decoder_layer.self_attn.linear_k.weight.data) + decoder_layer.self_attn.linear_k.bias.data).transpose(-1, -2)
            v = torch.matmul(hidden_states_norm, decoder_layer.self_attn.linear_v.weight.data) + decoder_layer.self_attn.linear_v.bias.data
            k = torch.cat((all_inputs[idx], k), dim=-1)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v), dim=-2)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            hidden_state_attn = torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v)
            hidden_state_attn = torch.matmul(hidden_state_attn, decoder_layer.self_attn.linear_out.weight.data).sum(dim=1, keepdim=True) + decoder_layer.self_attn.linear_out.bias.data
            hidden_state_attn += hidden_states
            q = torch.matmul(decoder_layer.norm2(hidden_state_attn), decoder_layer.src_attn.linear_q.weight.data) + decoder_layer.src_attn.linear_q.bias.data
            hidden_state_cross = torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.num_layers_de_2_plus_2]), dim=-1), all_inputs[idx + self.num_layers_de_3_plus])
            hidden_state_cross = torch.matmul(hidden_state_cross, decoder_layer.src_attn.linear_out.weight.data).sum(dim=1, keepdim=True) + decoder_layer.src_attn.linear_out.bias.data
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(decoder_layer.norm3(hidden_state_cross))
        hidden_states = self.dolphin.decoder.after_norm(hidden_states.squeeze(1)[:, -1])
        logits = self.dolphin.decoder.output_layer(hidden_states)
        logits = logits[:, language_start: language_end]
        return *self.save_de_keys, *self.save_de_values, logits, kv_seq_len


print('\nExport start...\n')
with torch.inference_mode():
    if 'small' in model_path.lower():
        model_size = 'small'
    else:
        model_size = 'base'
    model = dolphin.load_model(model_size, model_path, "cpu")
    with open(save_vocab, 'w', encoding='utf-8') as file:
        for token in model.s2t_model.token_list:
            file.write(token + '\n')
    HIDDEN_SIZE = model.s2t_model.decoder.output_layer.in_features
    NUM_HEAD_EN = model.s2t_model.encoder.encoders._modules['0'].attn.h
    NUM_HEAD_DE = model.s2t_model.decoder.decoders._modules['0'].self_attn.h
    HEAD_DIM_EN = model.s2t_model.encoder.encoders._modules['0'].attn.d_k
    HEAD_DIM_DE = model.s2t_model.decoder.decoders._modules['0'].self_attn.d_k
    NUM_LAYER_DE = len(model.s2t_model.decoder.decoders)
    VOCAB_SIZE = model.s2t_model.vocab_size
    STFT_SIGNAL_LENGTH = INPUT_AUDIO_LENGTH // HOP_LENGTH + 1
        
    scaling = float(HEAD_DIM_EN ** -0.25)
    for i in model.s2t_model.encoder.encoders._modules:
        model.s2t_model.encoder.encoders._modules[i].attn.linear_q.weight.data *= scaling
        model.s2t_model.encoder.encoders._modules[i].attn.linear_q.bias.data *= scaling
        model.s2t_model.encoder.encoders._modules[i].attn.linear_k.weight.data *= scaling
        model.s2t_model.encoder.encoders._modules[i].attn.linear_k.bias.data *= scaling
        model.s2t_model.encoder.encoders._modules[i].attn.linear_pos.weight.data *= scaling
        model.s2t_model.encoder.encoders._modules[i].attn.pos_bias_u.data = model.s2t_model.encoder.encoders._modules[i].attn.pos_bias_u.data.unsqueeze(1) * scaling
        model.s2t_model.encoder.encoders._modules[i].attn.pos_bias_v.data = model.s2t_model.encoder.encoders._modules[i].attn.pos_bias_v.data.unsqueeze(1) * scaling
    
        model.s2t_model.encoder.encoders._modules[i].attn.linear_q.weight.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_q.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_q.bias.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_q.bias.data.view(NUM_HEAD_EN, 1, HEAD_DIM_EN).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_k.weight.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_k.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_k.bias.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_k.bias.data.view(NUM_HEAD_EN, 1, HEAD_DIM_EN).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_v.weight.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_v.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_v.bias.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_v.bias.data.view(NUM_HEAD_EN, 1, HEAD_DIM_EN).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_pos.weight.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_pos.weight.data.view(NUM_HEAD_EN, HEAD_DIM_EN, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_out.weight.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_out.weight.data.view(HIDDEN_SIZE, NUM_HEAD_EN, HEAD_DIM_EN).permute(1, 2, 0).contiguous()
        model.s2t_model.encoder.encoders._modules[i].attn.linear_out.bias.data = model.s2t_model.encoder.encoders._modules[i].attn.linear_out.bias.data.view(1, 1, -1).contiguous()

    scaling = float(HEAD_DIM_DE ** -0.25)
    for i in model.s2t_model.decoder.decoders._modules:
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.bias.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.bias.data *= scaling
        
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.weight.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.bias.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.weight.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.bias.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_v.weight.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_v.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_v.bias.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_v.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_out.weight.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_out.weight.data.view(HIDDEN_SIZE, NUM_HEAD_DE, HEAD_DIM_DE).permute(1, 2, 0).contiguous()
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_out.bias.data = model.s2t_model.decoder.decoders._modules[i].self_attn.linear_out.bias.data.view(1, 1, -1).contiguous()
        
    scaling = float(model.s2t_model.decoder.decoders._modules['0'].src_attn.d_k ** -0.25)
    for i in model.s2t_model.decoder.decoders._modules:
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.bias.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.bias.data *= scaling
        
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.weight.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.bias.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.weight.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.bias.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_v.weight.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_v.weight.data.view(NUM_HEAD_DE, HEAD_DIM_DE, HIDDEN_SIZE).transpose(1, 2).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_v.bias.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_v.bias.data.view(NUM_HEAD_DE, 1, HEAD_DIM_DE).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_out.weight.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_out.weight.data.view(HIDDEN_SIZE, NUM_HEAD_DE, HEAD_DIM_DE).permute(1, 2, 0).contiguous()
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_out.bias.data = model.s2t_model.decoder.decoders._modules[i].src_attn.linear_out.bias.data.view(1, 1, -1).contiguous()

    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, win_length=WINDOW_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    dolphin_encoder = DOLPHIN_ENCODER(model, custom_stft, NFFT_STFT, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)
    output_names = []
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
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
        onnx_model_A,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del dolphin_encoder
    del audio
    del custom_stft
    del name
    del output_names
    del dynamic_axes
    gc.collect()

    dolphin_decoder = DOLPHIN_DECODER(model, MAX_SEQ_LEN, NUM_LAYER_DE)
    input_ids = torch.ones((3, 1), dtype=torch.int32)  # '3' is a dummy value
    ids_len = torch.tensor([input_ids.shape[-1]], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    save_encoder_key = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float32)
    save_encoder_value = torch.zeros((NUM_HEAD_EN, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float32)
    batch_size = input_ids.shape[0]
    past_key_de = torch.zeros((batch_size, NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float32)
    past_value_de = torch.zeros((batch_size, NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)
    language_start = torch.tensor([0], dtype=torch.int64)
    language_end = torch.tensor([40002], dtype=torch.int64)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {'input_ids': {0: 'batch', 1: 'ids_len'}}

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

    input_names.append('input_ids')
    all_inputs.append(input_ids)
    input_names.append('history_len')
    all_inputs.append(history_len)

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

    all_inputs.append(language_start)
    input_names.append('language_start')
    all_inputs.append(language_end)
    input_names.append('language_end')
    all_inputs.append(ids_len)
    input_names.append('ids_len')
    all_inputs.append(attention_mask)
    input_names.append('attention_mask')
    output_names.append('logits')
    dynamic_axes['logits'] = {0: 'batch', 1: 'vocab_range'}
    output_names.append('kv_seq_len')

    torch.onnx.export(
        dolphin_decoder,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del dolphin_decoder
    del input_ids
    del ids_len
    del history_len
    del save_encoder_key
    del save_encoder_value
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes
    del language_start
    del language_end
    
    greedy = GREEDY_SEARCH()
    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, VOCAB_SIZE), dtype=torch.float32)
    penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
    logits = torch.ones((beam_size, VOCAB_SIZE), dtype=torch.float32)
    penality_value = torch.tensor(REPEAT_PENALITY, dtype=torch.float32)
    batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

    torch.onnx.export(
        greedy,
        (logits, repeat_penality, penality_value, beam_size),  # Reuse the beam_size tensor as batch_size during export process.
        onnx_model_C,
        input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
        output_names=['max_logits_idx', 'repeat_penality_out'],
        dynamic_axes={
            'logits': {0: 'batch', 1: 'vocab_range'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )

    first_beam_search = FIRST_BEAM_SEARCH(NUM_LAYER_DE)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
    previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
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
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('top_beam_indices')
    output_names.append('save_id_out')
    output_names.append('repeat_penality_out')
    output_names.append('top_beam_prob')
    output_names.append('batch_indices')
    output_names.append('max_logits_idx')
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['repeat_penality_in'] = {0: 'batch'}
    dynamic_axes['repeat_penality_out'] = {0: 'batch'}
    dynamic_axes['logits'] = {0: 'batch', 1: 'vocab_range'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}
    dynamic_axes['batch_indices'] = {0: 'batch'}

    torch.onnx.export(
        first_beam_search,
        tuple(all_inputs),
        onnx_model_D,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )

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
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('batch_indices')
    all_inputs.append(batch_indices)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}
    output_names.remove("batch_indices")

    second_beam_search = SECOND_BEAM_SEARCH(NUM_LAYER_DE)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )

    reset_penality = RESET_PENALITY()
    torch.onnx.export(
        reset_penality,
        (save_id, repeat_penality, penality_reset_count, batch_indices),
        onnx_model_F,
        input_names=['save_id_in', 'repeat_penality_in', 'penality_reset_count_in', 'batch_indices'],
        output_names=['save_id_out', 'repeat_penality_out', 'penality_reset_count_out'],
        dynamic_axes={
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'penality_reset_count_in': {0: 'batch'},
            'penality_reset_count_out': {0: 'batch'},
            'batch_indices': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )

    argmax = ARGMAX()
    torch.onnx.export(
        argmax,
        (logits,),
        onnx_model_G,
        input_names=['logits'],
        output_names=['max_logits_idx'],
        dynamic_axes={
            'logits': {0: 'batch', 1: 'vocab_range'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )

    del greedy
    del first_beam_search
    del reset_penality
    del second_beam_search
    del argmax
    del batch_indices
    del past_key_de
    del past_value_de
    del past_keys_greedy
    del past_values_greedy
    del logits
    del previous_prob
    del save_id
    del repeat_penality
    del penality_reset_count
    del topK
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    gc.collect()
    
print('\nExport done!\n\nStart to run Dolphin by ONNX Runtime.\n\nNow, loading the model...')


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4         # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4        # Fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0       # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0       # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True    # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A = [out_name_A[i].name for i in range(len(out_name_A))]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]

ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_G = ort_session_G.get_inputs()[0].name
out_name_G = [ort_session_G.get_outputs()[0].name]

generate_limit = MAX_SEQ_LEN - 6          # 6 = length of initial input_ids
num_layers = (amount_of_outputs_B - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), 'cpu', 0)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), 'cpu', 0)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=np.float32), 'cpu', 0)
tokenizer = Tokenizer(save_vocab)
vocab_size = tokenizer.num_vocab

# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]
    
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]
    
    input_feed_D = {
        in_name_D[-2]: penality_value,
        in_name_D[-1]: beam_size
    }
    input_feed_E = {
        in_name_E[-3]: penality_value,
        in_name_E[-2]: beam_size,
        in_name_E[-1]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
    in_name_C = ort_session_C.get_inputs()
    out_name_C = ort_session_C.get_outputs()
    in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
    out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
    input_feed_C = {in_name_C[2]: penality_value}

if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), 'cpu', 0)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False

language_region = LANGUAGE_REGION.get(TARGET_LANGUAGE, "NONE")
if language_region == "NONE":
    TARGET_LANGUAGE = "Auto-Auto"
    print(f"\nThe language or region:{TARGET_LANGUAGE} not found. \nFallback to auto detection.")
language_region = language_region.split("-")
lang_id = f"<{language_region[0]}>"
region_id = f"<{language_region[1]}>"

if lang_id != "<auto>":
    detect_language = False
    lang_id = tokenizer.encode(lang_id)
else:
    detect_language = True

if not detect_language:
    if region_id != "<auto>":
        detect_region = False
        region_id = tokenizer.encode(region_id)
    else:
        detect_region = True
else:
    detect_region = True

history_len_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)

ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
ids_len_2 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([2], dtype=np.int64), 'cpu', 0)
ids_len_5 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([5], dtype=np.int64), 'cpu', 0)

ids_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
ids_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
ids_7 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([7], dtype=np.int64), 'cpu', 0)
ids_145 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), 'cpu', 0)
ids_324 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([324], dtype=np.int64), 'cpu', 0)
ids_39999 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999]], dtype=np.int32), 'cpu', 0)  # int32
ids_vocab_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([vocab_size], dtype=np.int64), 'cpu', 0)

init_BATCH = 1
init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((init_BATCH, ort_session_B._outputs_meta[0].shape[1], ort_session_B._outputs_meta[0].shape[2], 0), dtype=np.float32), 'cpu', 0)
init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((init_BATCH, ort_session_B._outputs_meta[num_layers].shape[1], 0, ort_session_B._outputs_meta[num_layers].shape[3]), dtype=np.float32), 'cpu', 0)
layer_indices = np.arange(num_keys_values_plus_2, num_keys_values_plus_2 + num_keys_values, dtype=np.int32)

# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(240000, audio_len)  # You can adjust it.
    else:
        INPUT_AUDIO_LENGTH = shape_value_in
    if SLIDING_WINDOW <= 0:
        stride_step = INPUT_AUDIO_LENGTH
    else:
        stride_step = SLIDING_WINDOW
    if audio_len > INPUT_AUDIO_LENGTH:
        num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH) / stride_step)) + 1
        total_length_needed = (num_windows - 1) * stride_step + INPUT_AUDIO_LENGTH
        pad_amount = total_length_needed - audio_len
        final_slice = audio[:, :, -pad_amount:].astype(np.float32)
        white_noise = (np.sqrt(np.mean(final_slice * final_slice)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    elif audio_len < INPUT_AUDIO_LENGTH:
        audio_float = audio.astype(np.float32)
        white_noise = (np.sqrt(np.mean(audio_float * audio_float)) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH - audio_len))).astype(audio.dtype)
        audio = np.concatenate((audio, white_noise), axis=-1)
    aligned_len = audio.shape[-1]

    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    # Start to run dolphin
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], 'cpu', 0)})
        input_feed_B = {
            in_name_B[-1]: attention_mask_1,
            in_name_B[num_keys_values_plus_1]: history_len_0,
        }
        for i in range(num_layers):
            input_feed_B[in_name_B[i]] = init_past_keys_B
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]
        for i in range(num_layers, num_keys_values):
            input_feed_B[in_name_B[i]] = init_past_values_B
            input_feed_B[in_name_B[layer_indices[i]]] = all_outputs_A[i]

        if detect_language:
            print("\nAutomatically detect which language it is.")
            input_feed_B[in_name_B[-4]] = ids_7
            input_feed_B[in_name_B[-3]] = ids_145
            input_feed_B[in_name_B[-2]] = ids_len_2
            input_feed_B[in_name_B[num_keys_values]] = ids_39999
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            lang_id = onnxruntime.OrtValue.numpy(ort_session_G.run_with_ort_values(out_name_G, {in_name_G: all_outputs_B[-2]})[0])[0] + 7
            for i in range(num_layers):
                input_feed_B[in_name_B[i]] = init_past_keys_B
            for i in range(num_layers, num_keys_values):
                input_feed_B[in_name_B[i]] = init_past_values_B
            
        if detect_region:
            print("\nAutomatically detect which region it is.")
            input_feed_B[in_name_B[-4]] = ids_145
            input_feed_B[in_name_B[-3]] = ids_324
            input_feed_B[in_name_B[-2]] = ids_len_2
            input_feed_B[in_name_B[num_keys_values]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, lang_id]], dtype=np.int32), 'cpu', 0)
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            region_id = onnxruntime.OrtValue.numpy(ort_session_G.run_with_ort_values(out_name_G, {in_name_G: all_outputs_B[-2]})[0])[0] + 145
            for i in range(num_layers):
                input_feed_B[in_name_B[i]] = init_past_keys_B
            for i in range(num_layers, num_keys_values):
                input_feed_B[in_name_B[i]] = init_past_values_B

        if detect_language or detect_region:
            lang_str = tokenizer.decode(lang_id)
            region_str = tokenizer.decode(region_id)
            message = f"\nThis audio belongs to {lang_str}-{region_str}."
            message = message.replace("<", "").replace(">", "")
            print(message)
        else:
            print(f"\nThis audio belongs to {TARGET_LANGUAGE}.")
            
        input_feed_B[in_name_B[-4]] = ids_0
        input_feed_B[in_name_B[-3]] = ids_vocab_size
        input_feed_B[in_name_B[-2]] = ids_len_5
        input_feed_B[in_name_B[num_keys_values]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, lang_id, region_id, 6, 324]], dtype=np.int32), 'cpu', 0)  # start_id = 39999; itn = 5; asr = 6; no_timestamp = 324
        repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=np.float32), 'cpu', 0)

        if USE_BEAM_SEARCH:
            save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), 'cpu', 0)
            input_feed_D[in_name_D[num_keys_values_plus_1]] = save_id_beam
            input_feed_D[in_name_D[num_keys_values_plus_2]] = repeat_penality
        else:
            batch_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([init_BATCH], dtype=np.int64), 'cpu', 0)
            input_feed_C[in_name_C[1]] = repeat_penality
            input_feed_C[in_name_C[3]] = batch_size
    
        if do_repeat_penality:
            if USE_BEAM_SEARCH:
                input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
            else:
                penality_reset_count_greedy = 0
        
        num_decode = 0
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
            if USE_BEAM_SEARCH:
                if num_decode < 1:
                    for i in range(num_keys_values_plus_1):
                        input_feed_D[in_name_D[i]] = all_outputs_B[i]
                    all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_D[-1])
                    input_feed_E[in_name_E[-4]] = all_outputs_D[-2]
                    if do_repeat_penality:
                        input_feed_F[in_name_F[3]] = all_outputs_D[-2]
                else:
                    for i in range(num_keys_values_plus_1):
                        input_feed_E[in_name_E[i]] = all_outputs_B[i]
                    all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
                    max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_E[-1])
                if max_logits_idx in STOP_TOKEN:
                    break
                if do_repeat_penality and (num_decode >= PENALITY_RANGE):
                    input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_F[in_name_F[1]] = all_outputs_E[num_keys_values_plus_2]
                    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
                    input_feed_F[in_name_F[2]] = all_outputs_F[2]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_F[0]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_F[1]
                if num_decode < 1:
                    for i in range(num_keys_values_plus_1):
                        input_feed_B[in_name_B[i]] = all_outputs_D[i]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_D[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_D[num_keys_values_plus_3]
                else:
                    for i in range(num_keys_values_plus_1):
                        input_feed_B[in_name_B[i]] = all_outputs_E[i]
                    input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
                    input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
                    input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
            else:
                input_feed_C[in_name_C[0]] = all_outputs_B[-2]
                all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
                max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_C[0])[0, 0]
                if max_logits_idx in STOP_TOKEN:
                    break
                input_feed_B[in_name_B[num_keys_values]] = all_outputs_C[0]
                if do_repeat_penality and (num_decode >= PENALITY_RANGE) and (save_id_greedy[penality_reset_count_greedy] != max_logits_idx):
                    repeat_penality = onnxruntime.OrtValue.numpy(all_outputs_C[1])
                    repeat_penality[..., penality_reset_count_greedy] = 1.0
                    all_outputs_C[1] = onnxruntime.OrtValue.ortvalue_from_numpy(repeat_penality, 'cpu', 0)
                    penality_reset_count_greedy += 1
                input_feed_C[in_name_C[1]] = all_outputs_C[1]
                save_id_greedy[num_decode] = max_logits_idx
                for i in range(num_keys_values):
                    input_feed_B[in_name_B[i]] = all_outputs_B[i]
            input_feed_B[in_name_B[num_keys_values_plus_1]] = all_outputs_B[num_keys_values_plus_1]
            if num_decode < 1:
                input_feed_B[in_name_B[-1]] = attention_mask_0
                input_feed_B[in_name_B[-2]] = ids_len_1
            num_decode += 1
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    if USE_BEAM_SEARCH:
        save_id_beam = onnxruntime.OrtValue.numpy(all_outputs_E[num_keys_values_plus_1])[0]
        for i, idx in enumerate(save_id_beam):
            if idx in STOP_TOKEN:
                save_token_array = save_id_beam[:i]
                break
    else:
        save_token_array = save_id_greedy[:num_decode]
    text = ""
    for i in save_token_array:
        text += tokenizer.decode(i)
    text = text.replace("▁", " ")
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")
