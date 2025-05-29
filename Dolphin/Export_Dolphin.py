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


model_path = "/home/DakeQQ/Downloads/dolphin-small"                                             # The dolphin project download path. Currently, only support dolphin-small and dolphin-base.
onnx_model_A = "/home/DakeQQ/Downloads/Dolphin_ONNX/Dolphin_Encoder.onnx"                       # The exported onnx model path.
onnx_model_B = "/home/DakeQQ/Downloads/Dolphin_ONNX/Dolphin_Decoder.onnx"                       # The exported onnx model path.
save_vocab = "/home/DakeQQ/Downloads/Dolphin_ONNX/vocab_Dolphin.txt"                            # The exported Dolphin vocab path.
TARGET_LANGUAGE = "Auto-Auto"                                                                   # See 'LANGUAGE_REGION' for detail.
test_audio = ["./example/zh.mp3", "./example/zh-Shanghai.wav", "./example/ja.mp3", "./example/ko.mp3"]  # The test audio list.

DYNAMIC_AXES = True                                         # The default dynamic_axes is the input audio length. dolphin series models only support dynamic_axes due to their transformer structure.
INPUT_AUDIO_LENGTH = 160000                                 # The maximum input audio length. Must less than 480000 (30 seconds).
WINDOW_TYPE = 'kaiser'                                      # Type of window function used in the STFT
N_MELS = 80                                                 # Setting by dolphin model config. Number of Mel bands to generate in the Mel-spectrogram, edit it carefully.
NFFT_STFT = 512                                             # Number of FFT components for the STFT process, edit it carefully.
NFFT_FBANK = 512                                            # Number of FFT components for the FBank process, edit it carefully.
HOP_LENGTH = 160                                            # Number of samples between successive frames in the STFT, edit it carefully.
SAMPLE_RATE = 16000                                         # The model parameter, do not edit the value.
PRE_EMPHASIZE = 0.97                                        # For audio preprocessing.
SLIDING_WINDOW = 0                                          # Set the sliding window step for test audio reading; use 0 to disable.
MAX_SEQ_LEN = 72                                            # It should less than 448.
STOP_TOKEN = [40000]                                        # 40000 is the end token for Dolphin model.


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
    "Chinese-Auto"                 : "auto-auto",
    "Yue-Auto"                     : "ct-NULL",
    "Tamil-auto"                   : "ta-auto",
    "Urdu-auto"                    : "ur-auto",
    "Arabic-auto"                  : "ar-auto",

    "自动"                          : "auto",
    "自动-自动"                      : "auto-auto",
    "中文-自动"                      : "zh-auto",
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
    def __init__(self, filename=None):
        self.str_to_idx = {}
        self.idx_to_str = {}
        if filename:
            self.load_from_file(filename)

    def load_from_file(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            for idx, line in enumerate(file):
                token = line.rstrip('\n')
                self.str_to_idx[token] = idx
                self.idx_to_str[idx] = token

    def encode(self, token):
        return self.str_to_idx.get(token)

    def decode(self, idx):
        return self.idx_to_str.get(idx)

    def __len__(self):
        return len(self.str_to_idx)


def rel_shift(x, x_len, zero_pad, n_head):
    x_padded = torch.cat([zero_pad[:, :x_len].float(), x], dim=-1)
    x_padded = x_padded.view(n_head, -1, x_len)
    x = x_padded[:, 1:].view_as(x)
    return x[:, :, :x_len]


class DOLPHIN_ENCODER(torch.nn.Module):
    def __init__(self, dolphin, stft_model, nfft_stft, nfft_fbank, stft_signal_len, n_mels, sample_rate, pre_emphasis, num_layers_de):
        super(DOLPHIN_ENCODER, self).__init__()
        self.dolphin = copy.deepcopy(dolphin.s2t_model)
        self.stft_model = stft_model
        self.pre_emphasis = float(pre_emphasis)
        self.fbank = (torchaudio.functional.melscale_fbanks(nfft_fbank // 2 + 1, 20, sample_rate // 2, n_mels, sample_rate, 'slaney', 'slaney')).transpose(0, 1).unsqueeze(0)
        self.nfft_stft = nfft_stft
        self.nfft_fbank = nfft_fbank
        if self.nfft_stft > self.nfft_fbank:
            self.padding = torch.zeros((1, n_mels, (nfft_stft - nfft_fbank) // 2), dtype=torch.float32)
            self.fbank = torch.cat((self.fbank, self.padding), dim=-1)
        else:
            self.padding = torch.zeros((1, (nfft_fbank - nfft_stft) // 4, stft_signal_len), dtype=torch.int8)
        self.inv_int16 = float(1.0 / 32768.0)
        self.inv_std = 1.0 / self.dolphin.normalize.std
        self.zero_pad = torch.zeros((self.dolphin.encoder.encoders._modules['0'].attn.h, 2048, 1), dtype=torch.int8)  # 2048 is about 30 seconds audio input.
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
        audio -= torch.mean(audio)  # Remove DC Offset
        audio = torch.cat((audio[:, :, :1], audio[:, :, 1:] - self.pre_emphasis * audio[:, :, :-1]), dim=-1)  # Pre Emphasize
        real_part, imag_part = self.stft_model(audio, 'constant')
        power = real_part * real_part + imag_part * imag_part
        if self.nfft_fbank > self.nfft_stft:
            padding = self.padding[:, :, :power.shape[-1]].float()
            power = torch.cat((padding, power, padding), dim=1)
        mel_features = torch.matmul(self.fbank, power).transpose(1, 2).clamp(min=1e-5).log()
        mel_features = (mel_features - self.dolphin.normalize.mean) * self.inv_std
        embed = self.dolphin.encoder.embed.conv(mel_features.unsqueeze(0))
        embed_len = embed.shape[-2].unsqueeze(0)
        x = self.embed(embed.transpose(1, 2).contiguous().view(1, embed_len, -1))
        pos_emb = self.position_encode.pe[:, self.position_encode_pe_half - embed_len + 1: self.position_encode_pe_half + embed_len].float()
        for encoder_layer in self.dolphin.encoder.encoders:
            x += encoder_layer.ff_scale * encoder_layer.feed_forward_macaron(encoder_layer.norm_ff_macaron(x))
            x1 = encoder_layer.norm_mha(x)
            q = encoder_layer.attn.linear_q(x1).view(-1, encoder_layer.attn.h, encoder_layer.attn.d_k).transpose(0, 1)
            k = encoder_layer.attn.linear_k(x1).view(-1, encoder_layer.attn.h, encoder_layer.attn.d_k).permute(1, 2, 0)
            v = encoder_layer.attn.linear_v(x1).view(-1, encoder_layer.attn.h, encoder_layer.attn.d_k).transpose(0, 1)
            p = encoder_layer.attn.linear_pos(pos_emb).view(-1, encoder_layer.attn.h, encoder_layer.attn.d_k).permute(1, 2, 0)
            q_with_bias_u = q + encoder_layer.attn.pos_bias_u
            q_with_bias_v = q + encoder_layer.attn.pos_bias_v
            matrix_ac = torch.matmul(q_with_bias_u, k)
            matrix_bd = torch.matmul(q_with_bias_v, p)
            matrix_bd = rel_shift(matrix_bd, embed_len, self.zero_pad, encoder_layer.attn.h)
            x1 = encoder_layer.attn.linear_out(torch.matmul(torch.softmax(matrix_ac + matrix_bd, dim=-1), v).transpose(0, 1).contiguous().view(1, -1, encoder_layer.attn.linear_out.in_features))
            x2 = encoder_layer.norm_mlp(x)
            x2 = encoder_layer.cgmlp.channel_proj1(x2)
            x2 = encoder_layer.cgmlp.csgu(x2)
            x2 = encoder_layer.cgmlp.channel_proj2(x2)
            x_concat = torch.cat([x1, x2], dim=-1)
            x_concat += encoder_layer.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
            x += encoder_layer.merge_proj(x_concat)
            x += encoder_layer.ff_scale * encoder_layer.feed_forward(encoder_layer.norm_ff(x))
            x = encoder_layer.norm_final(x)
        enc_outputs = self.dolphin.encoder.after_norm(x)
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            self.save_en_keys[idx] = decoder_layer.src_attn.linear_k(enc_outputs).view(-1, decoder_layer.src_attn.h, decoder_layer.src_attn.d_k).permute(1, 2, 0)
            self.save_en_values[idx] = decoder_layer.src_attn.linear_v(enc_outputs).view(-1, decoder_layer.src_attn.h, decoder_layer.src_attn.d_k).transpose(0, 1)
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
        self.num_layers_de_2_plus = self.num_layers_de_2 + 3
        self.num_layers_de_3_plus = self.num_layers_de_2_plus + num_layers_de
        self.save_de_keys = [None] * num_layers_de
        self.save_de_values = [None] * num_layers_de

    def forward(self, *all_inputs):
        input_ids = all_inputs[self.num_layers_de_2]
        history_len = all_inputs[self.num_layers_de_2 + 1]
        ids_len = all_inputs[self.num_layers_de_2 + 2]
        kv_seq_len = history_len + ids_len
        hidden_states = self.embed(input_ids) + self.position_encode.pe[:, history_len: kv_seq_len].float()
        language_start = all_inputs[-3]
        language_end = all_inputs[-2]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for idx, decoder_layer in enumerate(self.dolphin.decoder.decoders):
            hidden_states_norm = decoder_layer.norm1(hidden_states)
            q = decoder_layer.self_attn.linear_q(hidden_states_norm).view(-1, decoder_layer.self_attn.h, decoder_layer.self_attn.d_k).transpose(0, 1)
            k = decoder_layer.self_attn.linear_k(hidden_states_norm).view(-1, decoder_layer.self_attn.h, decoder_layer.self_attn.d_k).permute(1, 2, 0)
            v = decoder_layer.self_attn.linear_v(hidden_states_norm).view(-1, decoder_layer.self_attn.h, decoder_layer.self_attn.d_k).transpose(0, 1)
            k = torch.cat((all_inputs[idx], k), dim=2)
            v = torch.cat((all_inputs[idx + self.num_layers_de], v), dim=1)
            self.save_de_keys[idx] = k
            self.save_de_values[idx] = v
            hidden_state_attn = decoder_layer.self_attn.linear_out(torch.matmul(torch.softmax(torch.matmul(q, k) + attention_mask, dim=-1), v).transpose(0, 1).contiguous().view(1, -1, decoder_layer.self_attn.linear_out.in_features))
            hidden_state_attn += hidden_states
            q = decoder_layer.src_attn.linear_q(decoder_layer.norm2(hidden_state_attn)).view(-1, decoder_layer.src_attn.h, decoder_layer.src_attn.d_k).transpose(0, 1)
            hidden_state_cross = decoder_layer.src_attn.linear_out(torch.matmul(torch.softmax(torch.matmul(q, all_inputs[idx + self.num_layers_de_2_plus]), dim=-1), all_inputs[idx + self.num_layers_de_3_plus]).transpose(0, 1).contiguous().view(1, -1, decoder_layer.src_attn.linear_out.in_features))
            hidden_state_cross += hidden_state_attn
            hidden_states = hidden_state_cross + decoder_layer.feed_forward(decoder_layer.norm3(hidden_state_cross))
        hidden_states = self.dolphin.decoder.output_layer(self.dolphin.decoder.after_norm(hidden_states[:, -1]))
        max_logit_ids = torch.argmax(hidden_states[:, language_start: language_end], dim=-1, keepdim=True).int()
        return *self.save_de_keys, *self.save_de_values, max_logit_ids, kv_seq_len


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
    scaling = float(HEAD_DIM_DE ** -0.25)
    for i in model.s2t_model.decoder.decoders._modules:
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_q.bias.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].self_attn.linear_k.bias.data *= scaling
    scaling = float(model.s2t_model.decoder.decoders._modules['0'].src_attn.d_k ** -0.25)
    for i in model.s2t_model.decoder.decoders._modules:
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_q.bias.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.weight.data *= scaling
        model.s2t_model.decoder.decoders._modules[i].src_attn.linear_k.bias.data *= scaling

    custom_stft = STFT_Process(model_type='stft_B', n_fft=NFFT_STFT, hop_len=HOP_LENGTH, max_frames=0, window_type=WINDOW_TYPE).eval()  # The max_frames is not the key parameter for STFT, but it is for ISTFT.
    dolphin_encoder = DOLPHIN_ENCODER(model, custom_stft, NFFT_STFT, NFFT_FBANK, STFT_SIGNAL_LENGTH, N_MELS, SAMPLE_RATE, PRE_EMPHASIZE, NUM_LAYER_DE)

    output_names = []
    audio = torch.ones((1, 1, INPUT_AUDIO_LENGTH), dtype=torch.int16)
    dynamic_axes = {'audio': {2: 'audio_len'}}
    for i in range(NUM_LAYER_DE):
        name = f'en_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'signal_len'}

    torch.onnx.export(
        dolphin_encoder,
        (audio,),
        onnx_model_A,
        input_names=['audio'],
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
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
    input_ids = torch.tensor([[39999, 143, 175, 5, 6, 324]], dtype=torch.int32)
    ids_len = torch.tensor([input_ids.shape[-1]], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    save_encoder_key = torch.zeros((NUM_HEAD_EN, HEAD_DIM_EN, STFT_SIGNAL_LENGTH // 2 + 1), dtype=torch.float32)
    save_encoder_value = torch.zeros((NUM_HEAD_EN, STFT_SIGNAL_LENGTH // 2 + 1, HEAD_DIM_EN), dtype=torch.float32)
    past_key_de = torch.zeros((NUM_HEAD_DE, HEAD_DIM_DE, 0), dtype=torch.float32)
    past_value_de = torch.zeros((NUM_HEAD_DE, 0, HEAD_DIM_DE), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)
    language_start = torch.tensor([0], dtype=torch.int64)
    language_end = torch.tensor([40002], dtype=torch.int64)

    input_names = []
    all_inputs = []
    output_names = []
    dynamic_axes = {'input_ids': {1: 'ids_len'}}

    for i in range(NUM_LAYER_DE):
        name = f'in_de_key_{i}'
        input_names.append(name)
        all_inputs.append(past_key_de)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_de_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
    for i in range(NUM_LAYER_DE):
        name = f'in_de_value_{i}'
        input_names.append(name)
        all_inputs.append(past_value_de)
        dynamic_axes[name] = {1: 'history_len'}
        name = f'out_de_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'history_len_plus_ids_len'}

    input_names.append('input_ids')
    all_inputs.append(input_ids)
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)

    for i in range(NUM_LAYER_DE):
        name = f'en_key_{i}'
        input_names.append(name)
        all_inputs.append(save_encoder_key)
        dynamic_axes[name] = {2: 'signal_len'}
    for i in range(NUM_LAYER_DE):
        name = f'en_value_{i}'
        input_names.append(name)
        all_inputs.append(save_encoder_value)
        dynamic_axes[name] = {1: 'signal_len'}

    all_inputs.append(language_start)
    input_names.append('language_start')
    all_inputs.append(language_end)
    input_names.append('language_end')
    all_inputs.append(attention_mask)
    input_names.append('attention_mask')
    output_names.append('max_logit_id')
    output_names.append('kv_seq_len')

    torch.onnx.export(
        dolphin_decoder,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_AXES else None,
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
    del past_key_de
    del past_value_de
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes
    del language_start
    del language_end
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

tokenizer = Tokenizer(save_vocab)

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
print(f"\nUsable Providers: {ort_session_A.get_providers()}")
shape_value_in = ort_session_A._inputs_meta[0].shape[-1]
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
output_names_A = []
for i in range(len(out_name_A)):
    output_names_A.append(out_name_A[i].name)

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
input_names_B = []
output_names_B = []
amount_of_outputs = len(out_name_B)
amount_of_inputs = len(in_name_B)
for i in range(amount_of_inputs):
    input_names_B.append(in_name_B[i].name)
for i in range(amount_of_outputs):
    output_names_B.append(out_name_B[i].name)

generate_limit = MAX_SEQ_LEN - 6          # 6 = length of initial input_ids
num_layers = (amount_of_outputs - 2) // 2
num_layers_2 = num_layers + num_layers
num_layers_4 = num_layers_2 + num_layers_2
num_layers_2_plus_1 = num_layers_2 + 1
num_layers_2_plus_2 = num_layers_2 + 2
language_start_indices = amount_of_inputs - 3
language_end_indices = amount_of_inputs - 2
attention_mask_indices = amount_of_inputs - 1
max_logit_ids_indices = amount_of_outputs - 2

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

init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
init_attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
init_past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[0].shape[0], ort_session_B._inputs_meta[0].shape[1], 0), dtype=np.float32), 'cpu', 0)
init_past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_B._inputs_meta[num_layers].shape[0], 0, ort_session_B._inputs_meta[num_layers].shape[2]), dtype=np.float32), 'cpu', 0)
layer_indices = np.arange(num_layers_2, num_layers_4, dtype=np.int32) + 3

# Load the input audio
for test in test_audio:
    print("----------------------------------------------------------------------------------------------------------")
    print(f"\nTest Input Audio: {test}")
    audio = np.array(AudioSegment.from_file(test).set_channels(1).set_frame_rate(SAMPLE_RATE).get_array_of_samples(), dtype=np.float32)
    audio = normalize_to_int16(audio)
    audio_len = len(audio)
    audio = audio.reshape(1, 1, -1)
    if isinstance(shape_value_in, str):
        INPUT_AUDIO_LENGTH = min(320000, audio_len)  # You can adjust it.
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

    save_token = []
    num_decode = 0
    slice_start = 0
    slice_end = INPUT_AUDIO_LENGTH
    # Start to run dolphin
    start_time = time.time()
    while slice_end <= aligned_len:
        all_outputs_A = ort_session_A.run_with_ort_values(output_names_A, {in_name_A0: onnxruntime.OrtValue.ortvalue_from_numpy(audio[:, :, slice_start: slice_end], 'cpu', 0)})
        input_feed_B = {
            input_names_B[attention_mask_indices]: init_attention_mask,
            input_names_B[num_layers_2_plus_1]: init_history_len,
        }
        for i in range(num_layers):
            input_feed_B[input_names_B[i]] = init_past_keys_B
        for i in range(num_layers, num_layers_2):
            input_feed_B[input_names_B[i]] = init_past_values_B
        for i in range(num_layers_2):
            input_feed_B[input_names_B[layer_indices[i]]] = all_outputs_A[i]

        if detect_language:
            print("\nAutomatically detect which language it is.")
            input_feed_B[input_names_B[language_start_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([7], dtype=np.int64), 'cpu', 0)
            input_feed_B[input_names_B[language_end_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), 'cpu', 0)
            input_feed_B[input_names_B[num_layers_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999]], dtype=np.int32), 'cpu', 0)
            input_feed_B[input_names_B[num_layers_2_plus_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
            all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
            lang_id = onnxruntime.OrtValue.numpy(all_outputs_B[max_logit_ids_indices])[0][0] + 7
            for i in range(num_layers):
                input_feed_B[input_names_B[i]] = init_past_keys_B
            for i in range(num_layers, num_layers_2):
                input_feed_B[input_names_B[i]] = init_past_values_B
            
        if detect_region:
            print("\nAutomatically detect which region it is.")
            input_feed_B[input_names_B[language_start_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), 'cpu', 0)
            input_feed_B[input_names_B[language_end_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([324], dtype=np.int64), 'cpu', 0)
            input_feed_B[input_names_B[num_layers_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, lang_id]], dtype=np.int32), 'cpu', 0)
            input_feed_B[input_names_B[num_layers_2_plus_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([2], dtype=np.int64), 'cpu', 0)
            all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
            region_id = onnxruntime.OrtValue.numpy(all_outputs_B[max_logit_ids_indices])[0][0] + 145
            for i in range(num_layers):
                input_feed_B[input_names_B[i]] = init_past_keys_B
            for i in range(num_layers, num_layers_2):
                input_feed_B[input_names_B[i]] = init_past_values_B

        if detect_region or detect_region:
            lang_str = tokenizer.decode(lang_id)
            region_str = tokenizer.decode(region_id)
            message = f"\nThis audio belongs to {lang_str}-{region_str}."
            message = message.replace("<", "").replace(">", "")
            print(message)
        else:
            print(f"\nThis audio belongs to {TARGET_LANGUAGE}.")
            
        input_ids = np.array([[39999, lang_id, region_id, 6, 324]], dtype=np.int32)  # start_id = 39999; itn = 5; asr = 6; no_timestamp = 324
        ids_len = np.array([input_ids.shape[1]], dtype=np.int64)
        input_feed_B[input_names_B[num_layers_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, 'cpu', 0)
        input_feed_B[input_names_B[num_layers_2_plus_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len, 'cpu', 0)
        input_feed_B[input_names_B[language_start_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
        input_feed_B[input_names_B[language_end_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([40002], dtype=np.int64), 'cpu', 0)

        num_decode = 0
        while num_decode < generate_limit:
            all_outputs_B = ort_session_B.run_with_ort_values(output_names_B, input_feed_B)
            max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_B[max_logit_ids_indices])[0][0]
            num_decode += 1
            if max_logit_ids in STOP_TOKEN:
                break
            for i in range(amount_of_outputs):
                input_feed_B[input_names_B[i]] = all_outputs_B[i]
            if num_decode < 2:
                input_feed_B[input_names_B[attention_mask_indices]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
                input_feed_B[input_names_B[num_layers_2_plus_2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
            save_token.append(max_logit_ids)
        slice_start += stride_step
        slice_end = slice_start + INPUT_AUDIO_LENGTH
    count_time = time.time() - start_time
    text = ""
    for i in save_token:
        text += tokenizer.decode(i)
    text = text.replace("▁", " ")
    print(f"\nASR Result:\n{text}\n\nTime Cost: {count_time:.3f} Seconds\n\nDecode Speed: {num_decode / count_time:.3f} tokens/s")
    print("----------------------------------------------------------------------------------------------------------")
