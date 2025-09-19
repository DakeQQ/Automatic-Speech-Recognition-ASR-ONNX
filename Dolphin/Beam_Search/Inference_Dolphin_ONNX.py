# Standard library imports
import os
import gc
import re
import json
import time
import site
import shutil
import platform
from pathlib import Path
from datetime import timedelta

# Third-party imports
import psutil
import cpuinfo
import numpy as np
import gradio as gr
import soundfile as sf
import onnxruntime
from pydub import AudioSegment
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from transformers import AutoTokenizer
from sentencepiece import SentencePieceProcessor
from VAD.TEN.include.ten_vad import TenVad
from ASR.FireRedASR.AED.L.Tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer

PYTHON_PACKAGE = site.getsitepackages()[-1]
shutil.copyfile(r'./VAD/Silero/utils_vad.py',    PYTHON_PACKAGE + r'/silero_vad/utils_vad.py')
shutil.copyfile(r'./VAD/Silero/model.py',        PYTHON_PACKAGE + r'/silero_vad/model.py')
shutil.copyfile(r'./VAD/Silero/silero_vad.onnx', PYTHON_PACKAGE + r'/silero_vad/data/silero_vad.onnx')
from silero_vad import load_silero_vad, get_speech_timestamps
from faster_whisper.vad import get_speech_timestamps as get_speech_timestamps_FW, VadOptions


physical_cores = psutil.cpu_count(logical=False)
print(f'\n找到 {physical_cores} 个物理 CPU 核心。Found {physical_cores} physical CPU cores.\n')


DEVICE_ID = 0
PENALITY_RANGE = 10         # For ASR decode.
MAX_SEQ_LEN_LLM = 85        # Do not edit it.
MAX_SEQ_LEN_ASR = 85        # Do not edit it.
MAX_ASR_SEGMENT = 320000    # Do not edit it.
SAMPLE_RATE_48K = 48000     # Do not edit it.
SAMPLE_RATE_16K = 16000     # Do not edit it.


TASK_LIST = ['转录 Transcribe', '转录 + 翻译 Transcribe + Translate']

HARDWARE_LIST = ['CPU', 'GPU_NPU']

WHISPER_ASR_LIST = [
    'Arabic-Whisper-v3',
    'Basque-Whisper-v3',
    'Cantonese-Yue-Whisper-v3',
    'Chinese-Belle-Punct-Whisper-v3',
    'Chinese-Hakka-Taiwan-Whisper-v3',
    'Chinese-Minnan-Taiwan-Whisper-v2',
    'Chinese-Taiwan-Whisper-v3',
    'CrisperWhisper-Whisper-v3',
    'Danish-Whisper-v3',
    'English-Indian-Whisper-v3',
    'English-Whisper-v3.5',
    'French-Whisper-v3',
    'German-Swiss-Whisper-v3',
    'German-Whisper-v3',
    'Greek-Whisper-v3',
    'Italian-v0.2-Whisper-v3',
    'Japanese-Anime-v0.3-Whisper-v3',
    'Japanese-Whisper-v3',
    'Korean-Whisper-v3',
    'Malaysian-Whisper-v3',
    'Official-Turbo-Whisper-v3',
    'Official-Whisper-v3',
    'Persian-Whisper-v3',
    'Polish-Whisper-v3'
    'Portuguese-Whisper-v3',
    'Russian-Whisper-v3',
    'Serbian-Whisper-v3',
    'Spanish-Whisper-v3',
    'Thai-Whisper-v3',
    'Turkish-Whisper-v3',
    'Urdu-Whisper-v3',
    'Vietnamese-Whisper-v3'
]

ASR_LIST = [
    'SenseVoice-Small',
    'Paraformer-Large',
    'FireRedASR-AED-L',
    'Dolphin-Small'
]
for i in WHISPER_ASR_LIST:
    ASR_LIST.append(i)

DENOISER_LIST = [
    'NONE',
    'GTCRN - denoise_level_1',
    'DFSMN - denoise_level_1',
    'MossFormer2_SE_48K - denoise_level_2',
    'ZipEnhancer - denoise_level_3',
    'MossFormerGAN_SE_16K - denoise_level_4',
    'MelBandRoformer - denoise_level_5'
]

VAD_LIST = [
    'NONE',
    'Faster_Whisper-Silero',
    'Official-Silero-v6',
    'Pyannote-3.0',
    'HumAware',
    'NVIDIA-NeMo-MarbleNet-v2.0',
    'TEN'
]

LLM_LIST = [
    'Qwen-3-4B-Instruct-2507-Abliterated',
    'Qwen-3-8B-Abliterated-v2',
    'Hunyuan-MT-7B-Abliterated',
    'Seed-X-PRO-7B',
    'Whisper'
]

WHISPER_LANGUAGE_LIST = [
        "Chinese           - 中文",
        "English           - 英语",
        "Japanese          - 日语",
        "Korean            - 韩语",
        "Cantonese         - 粤语",
        "Afrikaans         - 南非荷兰语",
        "Amharic           - 阿姆哈拉语",
        "Arabic            - 阿拉伯语",
        "Assamese          - 阿萨姆语",
        "Azerbaijani       - 阿塞拜疆语",
        "Bashkir           - 巴什基尔语",
        "Basque            - 巴斯克语",
        "Belarusian        - 白俄罗斯语",
        "Bengali           - 孟加拉语",
        "Bosnian           - 波斯尼亚语",
        "Breton            - 布列塔尼语",
        "Bulgarian         - 保加利亚语",
        "Burmese           - 缅甸语",
        "Catalan           - 加泰罗尼亚语",
        "Croatian          - 克罗地亚语",
        "Czech             - 捷克语",
        "Danish            - 丹麦语",
        "Dutch             - 荷兰语",
        "Estonian          - 爱沙尼亚语",
        "Faroese           - 法罗语",
        "Finnish           - 芬兰语",
        "French            - 法语",
        "Galician          - 加利西亚语",
        "Georgian          - 格鲁吉亚语",
        "German            - 德语",
        "Greek             - 希腊语",
        "Gujarati          - 古吉拉特语",
        "Haitian creole    - 海地克里奥尔语",
        "Hausa             - 豪萨语",
        "Hawaiian          - 夏威夷语",
        "Hebrew            - 希伯来语",
        "Hindi             - 印地语",
        "Hungarian         - 匈牙利语",
        "Icelandic         - 冰岛语",
        "Indonesian        - 印度尼西亚语",
        "Italian           - 意大利语",
        "Javanese          - 爪哇语",
        "Kannada           - 卡纳达语",
        "Kazakh            - 哈萨克语",
        "Khmer             - 高棉语",
        "Latin             - 拉丁语",
        "Latvian           - 拉脱维亚语",
        "Lingala           - 林加拉语",
        "Lithuanian        - 立陶宛语",
        "Luxembourgish     - 卢森堡语",
        "Macedonian        - 马其顿语",
        "Malagasy          - 马达加斯加语",
        "Malaysian         - 马来语",
        "Malayalam         - 马拉雅拉姆语",
        "Maltese           - 马耳他语",
        "Maori             - 毛利语",
        "Marathi           - 马拉地语",
        "Mongolian         - 蒙古语",
        "Nepali            - 尼泊尔语",
        "Norwegian         - 挪威语",
        "Nynorsk           - 新挪威语",
        "Occitan           - 奥克语",
        "Pashto            - 普什图语",
        "Persian           - 波斯语",
        "Polish            - 波兰语",
        "Portuguese        - 葡萄牙语",
        "Punjabi           - 旁遮普语",
        "Romanian          - 罗马尼亚语",
        "Russian           - 俄语",
        "Sanskrit          - 梵语",
        "Serbian           - 塞尔维亚语",
        "Shona             - 绍纳语",
        "Sindhi            - 信德语",
        "Sinhala           - 僧伽罗语",
        "Slovak            - 斯洛伐克语",
        "Slovenian         - 斯洛文尼亚语",
        "Somali            - 索马里语",
        "Spanish           - 西班牙语",
        "Sundanese         - 巽他语",
        "Swahili           - 斯瓦希里语",
        "Swedish           - 瑞典语",
        "Tagalog           - 他加禄语",
        "Tajik             - 塔吉克语",
        "Tamil             - 泰米尔语",
        "Tatar             - 塔塔尔语",
        "Telugu            - 泰卢固语",
        "Thai              - 泰语",
        "Tibetan           - 藏语",
        "Turkish           - 土耳其语",
        "Turkmen           - 土库曼语",
        "Ukrainian         - 乌克兰语",
        "Urdu              - 乌尔都语",
        "Uzbek             - 乌兹别克语",
        "Vietnamese        - 越南语",
        "Welsh             - 威尔士语",
        "Yiddish           - 意第绪语",
        "Yoruba            - 约鲁巴语"
]

WHISPER_LANGUAGE_MAP = {
    'Afrikaans':      {'id': 50327, 'custom_id': 18941},
    'Albanian':       {'id': 50317, 'custom_id': 18931},
    'Amharic':        {'id': 50334, 'custom_id': 18948},
    'Arabic':         {'id': 50272, 'custom_id': 18886},
    'Armenian':       {'id': 50312, 'custom_id': 18926},
    'Assamese':       {'id': 50350, 'custom_id': 18964},
    'Azerbaijani':    {'id': 50304, 'custom_id': 18918},
    'Bashkir':        {'id': 50355, 'custom_id': 18969},
    'Basque':         {'id': 50310, 'custom_id': 18924},
    'Belarusian':     {'id': 50330, 'custom_id': 18944},
    'Bengali':        {'id': 50302, 'custom_id': 18916},
    'Bosnian':        {'id': 50315, 'custom_id': 18929},
    'Breton':         {'id': 50309, 'custom_id': 18923},
    'Bulgarian':      {'id': 50292, 'custom_id': 18906},
    'Burmese':        {'id': 50346, 'custom_id': 18960},
    'Cantonese':      {'id': 50358, 'custom_id': 18972},
    'Catalan':        {'id': 50270, 'custom_id': 18884},
    'Chinese':        {'id': 50260, 'custom_id': 18874},
    'Croatian':       {'id': 50291, 'custom_id': 18905},
    'Czech':          {'id': 50283, 'custom_id': 18897},
    'Danish':         {'id': 50285, 'custom_id': 18899},
    'Dutch':          {'id': 50271, 'custom_id': 18885},
    'English':        {'id': 50259, 'custom_id': 18873},
    'Estonian':       {'id': 50307, 'custom_id': 18921},
    'Faroese':        {'id': 50338, 'custom_id': 18952},
    'Finnish':        {'id': 50277, 'custom_id': 18891},
    'French':         {'id': 50265, 'custom_id': 18879},
    'Galician':       {'id': 50319, 'custom_id': 18933},
    'Georgian':       {'id': 50329, 'custom_id': 18943},
    'German':         {'id': 50261, 'custom_id': 18875},
    'Greek':          {'id': 50281, 'custom_id': 18895},
    'Gujarati':       {'id': 50333, 'custom_id': 18947},
    'Haitian creole': {'id': 50339, 'custom_id': 18953},
    'Hausa':          {'id': 50354, 'custom_id': 18968},
    'Hawaiian':       {'id': 50352, 'custom_id': 18966},
    'Hebrew':         {'id': 50279, 'custom_id': 18893},
    'Hindi':          {'id': 50276, 'custom_id': 18890},
    'Hungarian':      {'id': 50286, 'custom_id': 18900},
    'Icelandic':      {'id': 50311, 'custom_id': 18925},
    'Indonesian':     {'id': 50275, 'custom_id': 18889},
    'Italian':        {'id': 50274, 'custom_id': 18888},
    'Japanese':       {'id': 50266, 'custom_id': 18880},
    'Javanese':       {'id': 50356, 'custom_id': 18970},
    'Kannada':        {'id': 50306, 'custom_id': 18920},
    'Kazakh':         {'id': 50316, 'custom_id': 18930},
    'Khmer':          {'id': 50323, 'custom_id': 18937},
    'Korean':         {'id': 50264, 'custom_id': 18878},
    'Lao':            {'id': 50336, 'custom_id': 18950},
    'Latin':          {'id': 50294, 'custom_id': 18908},
    'Latvian':        {'id': 50301, 'custom_id': 18915},
    'Lingala':        {'id': 50353, 'custom_id': 18967},
    'Lithuanian':     {'id': 50293, 'custom_id': 18907},
    'Luxembourgish':  {'id': 50345, 'custom_id': 18959},
    'Macedonian':     {'id': 50308, 'custom_id': 18922},
    'Malagasy':       {'id': 50349, 'custom_id': 18963},
    'Malaysian':      {'id': 50282, 'custom_id': 18896},
    'Malayalam':      {'id': 50296, 'custom_id': 18910},
    'Maltese':        {'id': 50343, 'custom_id': 18957},
    'Maori':          {'id': 50295, 'custom_id': 18909},
    'Marathi':        {'id': 50320, 'custom_id': 18934},
    'Mongolian':      {'id': 50314, 'custom_id': 18928},
    'Nepali':         {'id': 50313, 'custom_id': 18927},
    'Norwegian':      {'id': 50288, 'custom_id': 18902},
    'Nynorsk':        {'id': 50342, 'custom_id': 18956},
    'Occitan':        {'id': 50328, 'custom_id': 18942},
    'Pashto':         {'id': 50340, 'custom_id': 18954},
    'Persian':        {'id': 50300, 'custom_id': 18914},
    'Polish':         {'id': 50269, 'custom_id': 18883},
    'Portuguese':     {'id': 50267, 'custom_id': 18881},
    'Punjabi':        {'id': 50321, 'custom_id': 18935},
    'Romanian':       {'id': 50284, 'custom_id': 18898},
    'Russian':        {'id': 50263, 'custom_id': 18877},
    'Sanskrit':       {'id': 50344, 'custom_id': 18958},
    'Serbian':        {'id': 50303, 'custom_id': 18917},
    'Shona':          {'id': 50324, 'custom_id': 18938},
    'Sindhi':         {'id': 50332, 'custom_id': 18946},
    'Sinhala':        {'id': 50322, 'custom_id': 18936},
    'Slovak':         {'id': 50298, 'custom_id': 18912},
    'Slovenian':      {'id': 50305, 'custom_id': 18919},
    'Somali':         {'id': 50326, 'custom_id': 18940},
    'Spanish':        {'id': 50262, 'custom_id': 18876},
    'Sundanese':      {'id': 50357, 'custom_id': 18971},
    'Swahili':        {'id': 50318, 'custom_id': 18932},
    'Swedish':        {'id': 50273, 'custom_id': 18887},
    'Tagalog':        {'id': 50348, 'custom_id': 18962},
    'Tajik':          {'id': 50331, 'custom_id': 18945},
    'Tamil':          {'id': 50287, 'custom_id': 18901},
    'Tatar':          {'id': 50351, 'custom_id': 18965},
    'Telugu':         {'id': 50299, 'custom_id': 18913},
    'Thai':           {'id': 50289, 'custom_id': 18903},
    'Tibetan':        {'id': 50347, 'custom_id': 18961},
    'Turkish':        {'id': 50268, 'custom_id': 18882},
    'Turkmen':        {'id': 50341, 'custom_id': 18955},
    'Ukrainian':      {'id': 50280, 'custom_id': 18894},
    'Urdu':           {'id': 50290, 'custom_id': 18904},
    'Uzbek':          {'id': 50337, 'custom_id': 18951},
    'Vietnamese':     {'id': 50278, 'custom_id': 18892},
    'Welsh':          {'id': 50297, 'custom_id': 18911},
    'Yiddish':        {'id': 50335, 'custom_id': 18949},
    'Yoruba':         {'id': 50325, 'custom_id': 18939}
}

SENSEVOICE_LANGUAGE_LIST = [
    "Auto      - 自动",
    "Chinese   - 中文",
    "English   - 英语",
    "Cantonese - 粤语",
    "Japanese  - 日語",
    "Korean    - 韩语"
]

SENSEVOICE_LANGUAGE_MAP = {region.split('-')[0].strip(): idx for idx, region in enumerate(SENSEVOICE_LANGUAGE_LIST)}

DOLPHIN_LANGUAGE_LIST = [
            # Auto-detection options
            "Auto-Auto",             "Mandarin-Auto",          "Yue-Auto",              "Tamil-Auto",
            "Urdu-Auto",             "Arabic-Auto",
            "自动-自动",               "中文-自动",               "粤语-自动",               "泰米尔语-自动",
            "乌尔都语-自动",            "阿拉伯语-自动",

            # Chinese varieties (English)
            "Chinese-Mandarin",       "Chinese-Taiwan",        "Chinese-Wuyu",          "Chinese-Sichuan",
            "Chinese-Shanxi",         "Chinese-Anhui",         "Chinese-Tianjin",       "Chinese-Ningxia",
            "Chinese-Shaanxi",        "Chinese-Hebei",         "Chinese-Shandong",      "Chinese-Guangdong",
            "Chinese-Shanghai",       "Chinese-Hubei",         "Chinese-Liaoning",      "Chinese-Gansu",
            "Chinese-Fujian",         "Chinese-Hunan",         "Chinese-Henan",         "Chinese-Yunnan",
            "Chinese-Minnan",         "Chinese-Wenzhou",

            # Chinese varieties (Chinese)
            "中文-普通话",             "中文-台湾",               "中文-吴语",              "中文-四川话",
            "中文-山西话",             "中文-安徽话",             "中文-天津话",             "中文-宁夏话",
            "中文-陕西话",             "中文-河北话",             "中文-山东话",             "中文-广东话",
            "中文-上海话",             "中文-湖北话",             "中文-辽宁话",             "中文-甘肃话",
            "中文-福建话",             "中文-湖南话",             "中文-河南话",             "中文-云南话",
            "中文-闽南语",             "中文-温州话",

            # Cantonese and East Asian languages
            "Yue-Unknown",           "Yue-Hongkong",           "Yue-Guangdong",         "Japanese",
            "Korean",                "Thai",                   "Vietnamese",
            "粤语-未知",               "粤语-香港",               "粤语-广东",              "日语",
            "韩语",                   "泰语",                    "越南语",

            # Southeast Asian languages
            "Indonesian",            "Malaysian",              "Burmese",               "Tagalog",
            "Khmer",                 "Javanese",               "Lao",                   "Filipino",            "Sundanese",
            "印度尼西亚语",            "马来语",                  "缅甸语",                 "塔加洛语",
            "高棉语",                 "爪哇语",                  "老挝语",                 "菲律宾语",              "巽他语",

            # South Asian languages
            "Hindi",                 "Bengali",                "Tamil-Singaporean",     "Tamil-Sri Lankan",
            "Tamil-India",           "Tamil-Malaysia",         "Telugu",                "Gujarati",
            "Oriya",                 "Odia",                   "Nepali",                "Sinhala",
            "Panjabi",               "Kashmiri",               "Marathi",
            "印地语",                 "孟加拉语",                "泰米尔语-新加坡",          "泰米尔语-斯里兰卡",
            "泰米尔语-印度",           "泰米尔语-马来西亚",         "泰卢固语",                "古吉拉特语",
            "奥里亚语",               "尼泊尔语",                 "僧伽罗语",                "旁遮普语",
            "克什米尔语",              "马拉地语",

            # West Asian languages
            "Urdu",                  "Urdu-India",             "Persian",               "Pushto",
            "乌尔都语",               "乌尔都语-印度",             "波斯语",                 "普什图语",

            # Arabic varieties
            "Arabic",                "Arabic-Morocco",         "Arabic-Saudi Arabia",   "Arabic-Egypt",
            "Arabic-Kuwait",         "Arabic-Libya",           "Arabic-Jordan",         "Arabic-U.A.E.",
            "Arabic-Levant",
            "阿拉伯语",               "阿拉伯语-摩洛哥",           "阿拉伯语-沙特",            "阿拉伯语-埃及",
            "阿拉伯语-科威特",         "阿拉伯语-利比亚",           "阿拉伯语-约旦",            "阿拉伯语-阿联酋",
            "阿拉伯语-黎凡特",

            # Central Asian and Turkic languages
            "Uighur",               "Uzbek",                  "Kazakh",                 "Mongolian",
            "Kabyle",               "Bashkir",                "Tajik",                  "Kirghiz",             "Azerbaijani",
            "维吾尔语",               "乌兹别克语",              "哈萨克语",                 "蒙古语",
            "卡拜尔语",               "巴什基尔语",              "塔吉克语",                 "吉尔吉斯语",            "阿塞拜疆语",

            # Other languages
            "Russian",              "俄语"
        ]

DOLPHIN_LANGUAGE_MAP = {
    # ───────────────────────────── Auto Detection ─────────────────────────────
    "Auto-Auto"                    : "auto-auto",
    "Mandarin-Auto"                : "zh-auto",
    "Yue-Auto"                     : "ct-NULL",
    "Tamil-Auto"                   : "ta-auto",
    "Urdu-Auto"                    : "ur-auto",
    "Arabic-Auto"                  : "ar-auto",

    "自动-自动"                      : "auto-auto",
    "中文-自动"                      : "zh-auto",
    "粤语-自动"                      : "ct-NULL",
    "泰米尔语-自动"                   : "ta-auto",
    "乌尔都语-自动"                   : "ur-auto",
    "阿拉伯语-自动"                   : "ar-auto",

    # ───────────────────────────── Chinese variants ─────────────────────────────
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

    "日语"                           : "ja-JP",
    "韩语"                           : "ko-KR",

    # ───────────────────────────── South-East Asian languages ─────────────────────────
    "Thai"                          : "th-TH",
    "Indonesian"                    : "id-ID",
    "Vietnamese"                    : "vi-VN",
    "Malaysian"                     : "ms-MY",
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
    "俄语"                           : "ru-RU"
}

INV_DOLPHIN_LANGUAGE_MAP = {
    # ───────────────────────────── Chinese variants ─────────────────────────────
    "zh-CN"                         : "Chinese-Mandarin",
    "zh-TW"                         : "Chinese-Taiwan",
    "zh-WU"                         : "Chinese-Wuyu",
    "zh-SICHUAN"                    : "Chinese-Sichuan",
    "zh-SHANXI"                     : "Chinese-Shanxi",
    "zh-ANHUI"                      : "Chinese-Anhui",
    "zh-TIANJIN"                    : "Chinese-Tianjin",
    "zh-NINGXIA"                    : "Chinese-Ningxia",
    "zh-SHAANXI"                    : "Chinese-Shaanxi",
    "zh-HEBEI"                      : "Chinese-Hebei",
    "zh-SHANDONG"                   : "Chinese-Shandong",
    "zh-GUANGDONG"                  : "Chinese-Guangdong",
    "zh-SHANGHAI"                   : "Chinese-Shanghai",
    "zh-HUBEI"                      : "Chinese-Hubei",
    "zh-LIAONING"                   : "Chinese-Liaoning",
    "zh-GANSU"                      : "Chinese-Gansu",
    "zh-FUJIAN"                     : "Chinese-Fujian",
    "zh-HUNAN"                      : "Chinese-Hunan",
    "zh-HENAN"                      : "Chinese-Henan",
    "zh-YUNNAN"                     : "Chinese-Yunnan",
    "zh-MINNAN"                     : "Chinese-Minnan",
    "zh-WENZHOU"                    : "Chinese-Wenzhou",

    # ───────────────────────────── Yue-Cantonese variants ───────────────────────────
    "ct-NULL"                       : "Yue-Unknown",
    "ct-HK"                         : "Yue-Hongkong",
    "ct-GZ"                         : "Yue-Guangdong",

    # ───────────────────────────── East-Asian languages ──────────────────────────────
    "ja-JP"                         : "Japanese",
    "ko-KR"                         : "Korean",

    # ───────────────────────────── South-East Asian languages ─────────────────────────
    "th-TH"                         : "Thai",
    "id-ID"                         : "Indonesian",
    "vi-VN"                         : "Vietnamese",
    "ms-MY"                         : "Malaysian",
    "my-MM"                         : "Burmese",
    "tl-PH"                         : "Tagalog",
    "km-KH"                         : "Khmer",
    "jv-ID"                         : "Javanese",
    "lo-LA"                         : "Lao",
    "fil-PH"                        : "Filipino",
    "su-ID"                         : "Sundanese",

    # ───────────────────────────── South-Asian languages ──────────────────────────────
    "hi-IN"                         : "Hindi",
    "bn-BD"                         : "Bengali",
    "ta-SG"                         : "Tamil-Singaporean",
    "ta-LK"                         : "Tamil-Sri Lankan",
    "ta-IN"                         : "Tamil-India",
    "ta-MY"                         : "Tamil-Malaysia",
    "te-IN"                         : "Telugu",
    "gu-IN"                         : "Gujarati",
    "or-IN"                         : "Odia",  # Note: "Oriya" would be overwritten by "Odia"
    "ne-NP"                         : "Nepali",
    "si-LK"                         : "Sinhala",
    "pa-IN"                         : "Panjabi",
    "ks-IN"                         : "Kashmiri",
    "mr-IN"                         : "Marathi",

    # ───────────────────────────── Middle-Eastern languages ───────────────────────────
    "ur-PK"                         : "Urdu-Islamic Republic of Pakistan",  # Note: "Urdu" would be overwritten
    "ur-IN"                         : "Urdu-India",
    "fa-IR"                         : "Persian",
    "ps-AF"                         : "Pushto",

    # ───────────────────────────── Arabic variants ──────────────────────────────
    "ar-GLA"                        : "Arabic",
    "ar-MA"                         : "Arabic-Morocco",
    "ar-SA"                         : "Arabic-Saudi Arabia",
    "ar-EG"                         : "Arabic-Egypt",
    "ar-KW"                         : "Arabic-Kuwait",
    "ar-LY"                         : "Arabic-Libya",
    "ar-JO"                         : "Arabic-Jordan",
    "ar-AE"                         : "Arabic-U.A.E.",
    "ar-LVT"                        : "Arabic-Levant",

    # ───────────────────────────── Central-Asian languages ────────────────────────────
    "ug-CN"                         : "Uighur",
    "uz-UZ"                         : "Uzbek",
    "kk-KZ"                         : "Kazakh",
    "mn-MN"                         : "Mongolian",
    "kab-NULL"                      : "Kabyle",
    "ba-NULL"                       : "Bashkir",
    "tg-TJ"                         : "Tajik",
    "ky-KG"                         : "Kirghiz",
    "az-AZ"                         : "Azerbaijani",

    # ───────────────────────────── Eastern-European languages ─────────────────────────
    "ru-RU"                         : "Russian",
}

SEED_X_LANGUAGE_LIST = [
    "Arabic             - 阿拉伯语",
    "Chinese            - 中文",
    "Czech              - 捷克语",
    "Danish             - 丹麦语",
    "Dutch              - 荷兰语",
    "English            - 英语",
    "Finnish            - 芬兰语",
    "French             - 法语",
    "German             - 德语",
    "Hungarian          - 匈牙利语",
    "Indonesian         - 印度尼西亚语",
    "Italian            - 意大利语",
    "Japanese           - 日语",
    "Korean             - 韩语",
    "Malaysian          - 马来语",
    "Norwegian          - 挪威语",
    "Norwegian Bokmål   - 挪威博克马尔语",
    "Polish             - 波兰语",
    "Portuguese         - 葡萄牙语",
    "Romanian           - 罗马尼亚语",
    "Russian            - 俄语",
    "Spanish            - 西班牙语",
    "Swedish            - 瑞典语",
    "Thai               - 泰语",
    "Turkish            - 土耳其语",
    "Ukrainian          - 乌克兰语",
    "Vietnamese         - 越南语"
]

SEED_X_LANGUAGE_MAP = {
    "Arabic"               : "ar",
    "Chinese"              : "zh",
    "Czech"                : "cs",
    "Danish"               : "da",
    "Dutch"                : "nl",
    "English"              : "en",
    "Finnish"              : "fi",
    "French"               : "fr",
    "German"               : "de",
    "Hungarian"            : "hu",
    "Indonesian"           : "id",
    "Italian"              : "it",
    "Japanese"             : "ja",
    "Korean"               : "ko",
    "Malaysian"            : "ms",
    "Norwegian"            : "no",
    "Norwegian Bokmål"     : "nb",
    "Polish"               : "pl",
    "Portuguese"           : "pt",
    "Romanian"             : "ro",
    "Russian"              : "ru",
    "Spanish"              : "es",
    "Swedish"              : "sv",
    "Thai"                 : "th",
    "Turkish"              : "tr",
    "Ukrainian"            : "uk",
    "Vietnamese"           : "vi",
}

HUNYUAN_LANGUAGE_LIST = [
    "Arabic                 - 阿拉伯语",
    "Bengali                - 孟加拉语",
    "Burmese                - 缅甸语",
    "Cantonese              - 粤语",
    "Chinese                - 中文",
    "Chinese (Traditional)  - 繁体中文",
    "Czech                  - 捷克语",
    "Dutch                  - 荷兰语",
    "English                - 英语",
    "Filipino               - 菲律宾语",
    "French                 - 法语",
    "German                 - 德语",
    "Gujarati               - 古吉拉特语",
    "Hebrew                 - 希伯来语",
    "Hindi                  - 印地语",
    "Indonesian             - 印尼语",
    "Italian                - 意大利语",
    "Japanese               - 日语",
    "Kazakh                 - 哈萨克语",
    "Khmer                  - 高棉语",
    "Korean                 - 韩语",
    "Malaysian              - 马来语",
    "Marathi                - 马拉地语",
    "Mongolian              - 蒙古语",
    "Persian                - 波斯语",
    "Polish                 - 波兰语",
    "Portuguese             - 葡萄牙语",
    "Russian                - 俄语",
    "Spanish                - 西班牙语",
    "Tamil                  - 泰米尔语",
    "Telugu                 - 泰卢固语",
    "Thai                   - 泰语",
    "Tibetan                - 藏语",
    "Turkish                - 土耳其语",
    "Ukrainian              - 乌克兰语",
    "Urdu                   - 乌尔都语",
    "Uyghur                 - 维吾尔语",
    "Vietnamese             - 越南语",
]

HUNYUAN_LANGUAGE_MAP = {
        "ar":      {"english_name": "Arabic",                  "chinese_name": "阿拉伯语"},
        "bn":      {"english_name": "Bengali",                 "chinese_name": "孟加拉语"},
        "my":      {"english_name": "Burmese",                 "chinese_name": "缅甸语"},
        "yue":     {"english_name": "Cantonese",               "chinese_name": "粤语"},
        "zh":      {"english_name": "Chinese",                 "chinese_name": "中文"},
        "zh-Hant": {"english_name": "Chinese (Traditional)",   "chinese_name": "繁体中文"},
        "cs":      {"english_name": "Czech",                   "chinese_name": "捷克语"},
        "nl":      {"english_name": "Dutch",                   "chinese_name": "荷兰语"},
        "en":      {"english_name": "English",                 "chinese_name": "英语"},
        "tl":      {"english_name": "Filipino",                "chinese_name": "菲律宾语"},
        "fr":      {"english_name": "French",                  "chinese_name": "法语"},
        "de":      {"english_name": "German",                  "chinese_name": "德语"},
        "gu":      {"english_name": "Gujarati",                "chinese_name": "古吉拉特语"},
        "he":      {"english_name": "Hebrew",                  "chinese_name": "希伯来语"},
        "hi":      {"english_name": "Hindi",                   "chinese_name": "印地语"},
        "id":      {"english_name": "Indonesian",              "chinese_name": "印尼语"},
        "it":      {"english_name": "Italian",                 "chinese_name": "意大利语"},
        "ja":      {"english_name": "Japanese",                "chinese_name": "日语"},
        "kk":      {"english_name": "Kazakh",                  "chinese_name": "哈萨克语"},
        "km":      {"english_name": "Khmer",                   "chinese_name": "高棉语"},
        "ko":      {"english_name": "Korean",                  "chinese_name": "韩语"},
        "ms":      {"english_name": "Malaysian",               "chinese_name": "马来语"},
        "mr":      {"english_name": "Marathi",                 "chinese_name": "马拉地语"},
        "mn":      {"english_name": "Mongolian",               "chinese_name": "蒙古语"},
        "fa":      {"english_name": "Persian",                 "chinese_name": "波斯语"},
        "pl":      {"english_name": "Polish",                  "chinese_name": "波兰语"},
        "pt":      {"english_name": "Portuguese",              "chinese_name": "葡萄牙语"},
        "ru":      {"english_name": "Russian",                 "chinese_name": "俄语"},
        "es":      {"english_name": "Spanish",                 "chinese_name": "西班牙语"},
        "ta":      {"english_name": "Tamil",                   "chinese_name": "泰米尔语"},
        "te":      {"english_name": "Telugu",                  "chinese_name": "泰卢固语"},
        "th":      {"english_name": "Thai",                    "chinese_name": "泰语"},
        "bo":      {"english_name": "Tibetan",                 "chinese_name": "藏语"},
        "tr":      {"english_name": "Turkish",                 "chinese_name": "土耳其语"},
        "uk":      {"english_name": "Ukrainian",               "chinese_name": "乌克兰语"},
        "ur":      {"english_name": "Urdu",                    "chinese_name": "乌尔都语"},
        "ug":      {"english_name": "Uyghur",                  "chinese_name": "维吾尔语"},
        "vi":      {"english_name": "Vietnamese",              "chinese_name": "越南语"},
    }

HUNYUAN_EXTRA_ALIASES = {
        "Tagalog": "tl",
        "Farsi": "fa",
        "Myanmar": "my",
        "Uighur": "ug",
        "Traditional Chinese": "zh-Hant",
        "Cambodian": "km"
}
LANGUAGE_ALIAS_MAP = {}
for abbr, data in HUNYUAN_LANGUAGE_MAP.items():
    LANGUAGE_ALIAS_MAP[abbr] = abbr
    LANGUAGE_ALIAS_MAP[data["english_name"]] = abbr
    LANGUAGE_ALIAS_MAP[data["chinese_name"]] = abbr
LANGUAGE_ALIAS_MAP.update(HUNYUAN_EXTRA_ALIASES)

MEDIA_EXTENSIONS = (
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.alac', '.aiff', '.m4a',

    # Video formats
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.3gp',

    # Movie formats (often overlaps with video)
    '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.3gp',

    # Music formats (often overlaps with audio)
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.alac', '.aiff', '.m4a'
)


def update_task(dropdown_task):
    if "转录 + 翻译 Transcribe + Translate" == dropdown_task:
        update_A = gr.update(visible=True)
        update_B = gr.update(visible=True)
    else:
        update_A = gr.update(visible=False)
        update_B = gr.update(visible=False)
    return update_A, update_B


def update_translate_language(dropdown_model_llm):
    if "Whisper" in dropdown_model_llm:
        update_A = gr.update(choices=[WHISPER_LANGUAGE_LIST[1]], value=WHISPER_LANGUAGE_LIST[1])
        update_B = gr.update(value=WHISPER_ASR_LIST[0], choices=WHISPER_ASR_LIST)
        update_C = gr.update(visible=False)
    elif "Seed" in dropdown_model_llm:
        update_A = gr.update(value=SEED_X_LANGUAGE_LIST[1], choices=SEED_X_LANGUAGE_LIST)
        update_B = gr.update(value=ASR_LIST[0], choices=ASR_LIST)
        update_C = gr.update(visible=False)
    elif "Hunyuan" in dropdown_model_llm:
        update_A = gr.update(value=HUNYUAN_LANGUAGE_LIST[4], choices=HUNYUAN_LANGUAGE_LIST)
        update_B = gr.update(value=ASR_LIST[0], choices=ASR_LIST)
        update_C = gr.update(visible=False)
    else:
        update_A = gr.update(value=WHISPER_LANGUAGE_LIST[0], choices=WHISPER_LANGUAGE_LIST)
        update_B = gr.update(value=ASR_LIST[0], choices=ASR_LIST)
        update_C = gr.update(visible=True)
    return update_A, update_B, update_C


def update_transcribe_language(dropdown_model_asr):
    lower_dropdown_model_asr = dropdown_model_asr.lower()
    if "whisper" in lower_dropdown_model_asr:
        lang = dropdown_model_asr.split('-')[0].strip()
        update_A = gr.update(visible=True, value=WHISPER_LANGUAGE_LIST[2], choices=WHISPER_LANGUAGE_LIST)
        for language in WHISPER_LANGUAGE_LIST:
            if lang in language:
                update_A = gr.update(visible=True, value=language, choices=[language])
                break
    elif "sensevoice-small" in lower_dropdown_model_asr:
        update_A = gr.update(visible=True, value=SENSEVOICE_LANGUAGE_LIST[4], choices=SENSEVOICE_LANGUAGE_LIST)
    elif "fireredasr-aed-l" in lower_dropdown_model_asr:
        update_A = gr.update(visible=True, value=WHISPER_LANGUAGE_LIST[0], choices=[WHISPER_LANGUAGE_LIST[0]])
    elif "paraformer-large" in lower_dropdown_model_asr:
        update_A = gr.update(visible=True, value=WHISPER_LANGUAGE_LIST[0], choices=[WHISPER_LANGUAGE_LIST[0], WHISPER_LANGUAGE_LIST[1]])
    elif "dolphin-small" in lower_dropdown_model_asr:
        update_A = gr.update(value=DOLPHIN_LANGUAGE_LIST[0], choices=DOLPHIN_LANGUAGE_LIST)
    else:
        update_A = gr.update(visible=False)
    if ('sensevoice-small' in lower_dropdown_model_asr) or ('paraformer-large' in lower_dropdown_model_asr):
        update_B = gr.update(visible=False)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
    else:
        update_B = gr.update(visible=True)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
    return update_A, update_B, update_C, update_D


def update_denoiser(dropdown_model_denoiser):
    if 'NONE' == dropdown_model_denoiser:
        return gr.update(visible=False), gr.update(visible=False)
    else:
        return gr.update(visible=True), gr.update(visible=True)


def update_vad(dropdown_model_vad):
    if "Pyannote" in dropdown_model_vad:
        update_A = gr.update(visible=False, value=0.4)
        update_B = gr.update(visible=False, value=0.4)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=True, value=0.3)
        update_F = gr.update(visible=True, value=0.05)
        update_G = gr.update(visible=True, value=400)
    elif "Silero" in dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
        update_E = gr.update(visible=True, value=0.3)
        update_F = gr.update(visible=True, value=0.05)
        update_G = gr.update(visible=True, value=400)
    elif "HumAware" in dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
        update_E = gr.update(visible=True, value=0.3)
        update_F = gr.update(visible=True, value=0.05)
        update_G = gr.update(visible=True, value=400)
    elif "MarbleNet" in dropdown_model_vad:
        update_A = gr.update(visible=True, value=0.4)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
        update_E = gr.update(visible=True, value=0.3)
        update_F = gr.update(visible=True, value=0.05)
        update_G = gr.update(visible=True, value=400)
    elif dropdown_model_vad == "TEN":
        update_A = gr.update(visible=True, value=0.5)
        update_B = gr.update(visible=True, value=0.4)
        update_C = gr.update(visible=True)
        update_D = gr.update(visible=True)
        update_E = gr.update(visible=True, value=0.3)
        update_F = gr.update(visible=True, value=0.05)
        update_G = gr.update(visible=True, value=400)
    else:
        update_A = gr.update(visible=False)
        update_B = gr.update(visible=False)
        update_C = gr.update(visible=False)
        update_D = gr.update(visible=False)
        update_E = gr.update(visible=False)
        update_F = gr.update(visible=False)
        update_G = gr.update(visible=False)
    return update_A, update_B, update_C, update_D, update_E, update_F, update_G


def get_task_id_whisper(task_input, is_v3, custom_vocab=False):
    if custom_vocab:
        stop_token = 18871
        start_token = 18872
        task_map = {
            'Translate': 18973,
            'Transcribe': 18974
        }
        return start_token, [stop_token], task_map[task_input]
    stop_token = 50257
    start_token = 50258
    if is_v3:
        task_map = {
            'Translate': 50359,
            'Transcribe':  50360
        }
        return start_token, [stop_token], task_map[task_input]
    else:
        task_map = {
            'Translate': 50358,
            'Transcribe': 50359
        }
        return start_token, [stop_token], task_map[task_input]
    
    
def get_language_whisper(language_input, custom_vocab=False):
    if custom_vocab:
        return WHISPER_LANGUAGE_MAP[language_input]['custom_id']
    return WHISPER_LANGUAGE_MAP[language_input]['id']


def get_language_sensevoice(language_input):
    return SENSEVOICE_LANGUAGE_MAP.get(language_input)

    
def get_language_hunyuan(language_input):
    canonical_abbr = LANGUAGE_ALIAS_MAP.get(language_input)
    if canonical_abbr:
        lang_data = HUNYUAN_LANGUAGE_MAP.get(canonical_abbr)
        return lang_data["english_name"], lang_data["chinese_name"]
    else:
        return None, None


def get_language_seedx(language_input):
    if language_input in SEED_X_LANGUAGE_MAP:
        abbr = f"<{SEED_X_LANGUAGE_MAP[language_input]}>"
        return abbr
    else:
        return None


def remove_repeated_parts(ids, repeat_words_threshold, ids_len):
    if ids_len <= repeat_words_threshold:
        return ids
    side_L = repeat_words_threshold // 2
    side_R = side_L + 1
    boundary = ids_len - side_L
    for i in range(side_L, boundary):
        for j in range(i + repeat_words_threshold, boundary):
            check = []
            for k in range(-side_L, side_R):
                if ids[j + k] == ids[i + k]:
                    check.append(True)
                else:
                    check.append(False)
                    break
            if False not in check:
                return ids[: j - side_L]
    return ids


def process_timestamps(timestamps, fusion_threshold=1.0, min_duration=0.5):
    # Filter out short durations
    if min_duration > 0.0:
        filtered_timestamps = [(start, end) for start, end in timestamps if (end - start) > min_duration]
    else:
        filtered_timestamps = timestamps
    del timestamps

    # Fuse and filter timestamps
    if fusion_threshold > 0.0:
        fused_timestamps = []
        for start, end in filtered_timestamps:
            if fused_timestamps and (start - fused_timestamps[-1][1] <= fusion_threshold):
                fused_timestamps[-1] = (
                    fused_timestamps[-1][0],
                    max(end, fused_timestamps[-1][1])
                )
            else:
                fused_timestamps.append((start, end))
        return fused_timestamps
    else:
        return filtered_timestamps


def vad_to_timestamps(vad_output, frame_duration, pad=0.0):
    timestamps = []
    start = None
    max_len = len(vad_output) * frame_duration
    frame_duration_plus = frame_duration + pad
    # Extract raw timestamps
    for i, silence in enumerate(vad_output):
        if silence:
            if start is not None:  # End of the current speaking segment
                end = i * frame_duration + frame_duration_plus
                if end > max_len:
                    end = max_len
                timestamps.append((start, end))
                start = None
        else:
            if start is None:  # Start of a new speaking segment
                start = i * frame_duration - pad
                if start < 0.0:
                    start = 0.0
    # Handle the case where speech continues until the end
    if start is not None:
        timestamps.append((start, max_len))
    return timestamps


def format_time(seconds):
    """Convert Seconds to VTT time format 'hh:mm:ss.mmm'."""
    td = timedelta(seconds=seconds)
    td_sec = td.total_seconds()
    total_seconds = int(td_sec)
    milliseconds = int((td_sec - total_seconds) * 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def normalize_to_int16(audio):
    max_val = np.max(np.abs(audio))
    scaling_factor = 32767.0 / max_val if max_val > 0 else 1.0
    return (audio * scaling_factor).astype(np.int16)


class Dolphin_Tokenizer:
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


def MAIN_PROCESS(
        task,
        hardware,
        parallel_threads,
        file_path_input,
        translate_language,
        transcribe_language,
        slide_top_k_asr,
        slide_beam_size_asr,
        slide_repeat_penality_value_asr,
        model_asr,
        model_vad,
        model_denoiser,
        model_llm,
        llm_prompt,
        switcher_run_test,
        switcher_denoiser_cache,
        slider_vad_pad,
        slider_denoise_factor,
        slider_vad_SPEAKING_SCORE,
        slider_vad_SILENCE_SCORE,
        slider_vad_FUSION_THRESHOLD,
        slider_vad_MIN_SPEECH_DURATION,
        slider_vad_MAX_SPEECH_DURATION,
        slider_vad_MIN_SILENCE_DURATION
):
    def create_ort_session(_device_type, _has_npu, _onnx_model_x, _ORT_Accelerate_Providers, _session_opts,
                           _provider_options, tag):
        ort_session_x = None
        device_type_x = 'cpu'
        use_sync_operations = True

        def try_create_session(device_type, device_name):
            """Helper function to create ONNX session with given device type"""
            try:
                _provider_options[0]['device_type'] = device_type
                session = onnxruntime.InferenceSession(
                    _onnx_model_x,
                    sess_options=_session_opts,
                    providers=_ORT_Accelerate_Providers,
                    provider_options=_provider_options
                )
                if len(session.get_providers()) > 1:
                    print(f"\n{tag}: OpenVINO-{device_name}使用成功。OpenVINO-{device_name} is used successfully.")
                    return session, True
                else:
                    print(f"\n{tag}: OpenVINO-{device_name} 使用失败。OpenVINO-{device_name} usage failed.")
                    return None, False
            except:
                print(f"\n{tag}: OpenVINO-{device_name} 使用失败。OpenVINO-{device_name} usage failed.")
                return None, False

        if _device_type != 'cpu':
            try:
                if 'OpenVINOExecutionProvider' in _ORT_Accelerate_Providers[0]:
                    # Try NPU first if available, then fallback to GPU
                    if _has_npu:
                        ort_session_x, success = try_create_session('NPU', 'NPU')
                        if success:
                            device_type_x = 'npu'

                    # If NPU failed or not available, try GPU
                    if ort_session_x is None:
                        ort_session_x, success = try_create_session('GPU', 'GPU')
                        if success:
                            device_type_x = _device_type
                else:
                    # Non-OpenVINO provider
                    ort_session_x = onnxruntime.InferenceSession(_onnx_model_x, sess_options=_session_opts, providers=_ORT_Accelerate_Providers, provider_options=_provider_options)
                    if len(ort_session_x.get_providers()) > 1:
                        device_type_x = _device_type
                        print(f"\n{tag}: GPU_NPU 使用成功。GPU_NPU is used successfully.")
                    else:
                        print(f"\n{tag}: GPU_NPU 使用失败。GPU_NPU usage failed.")
                        ort_session_x = None
            except:
                print(f"\n{tag}: GPU_NPU 使用失败。GPU_NPU usage failed.")
                ort_session_x = None

        # Fallback to CPU if all other options failed
        if ort_session_x is None:
            _ORT_Accelerate_Providers = ['CPUExecutionProvider']
            _provider_options = None
            use_sync_operations = False
            _onnx_model_x = _onnx_model_x.replace('FP16', 'FP32')
            ort_session_x = onnxruntime.InferenceSession(_onnx_model_x, sess_options=_session_opts, providers=_ORT_Accelerate_Providers, provider_options=_provider_options)

        # Temporary fallback to 'cpu', due to the onnxruntime doesn't update yet.
        if device_type_x in ['gpu', 'npu', 'dml']:
            device_type_x = 'cpu'

        return ort_session_x, use_sync_operations, device_type_x, _ORT_Accelerate_Providers, _provider_options

    def inference_A(_inv_audio_len, _slice_start, _slice_end, _audio):
        wave_form = onnxruntime.OrtValue.ortvalue_from_numpy(_audio[..., _slice_start: _slice_end], device_type_A, DEVICE_ID)
        return _slice_start * _inv_audio_len, ort_session_A._sess.run_with_ort_values({in_name_A0: wave_form._ortvalue}, out_name_A0, run_options)

    def inference_C_sensevoice(_start, _end, _inv_audio_len, _audio, _sample_rate, _language_idx):
        _start_indices = _start * _sample_rate
        _audio = _audio[..., int(_start_indices): int(_end * _sample_rate)]
        audio_segment_len = _audio.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)
        _stride_step = INPUT_AUDIO_LENGTH
        _slice_start = 0
        _slice_end = INPUT_AUDIO_LENGTH
        saved_text = ""
        while _slice_start < audio_segment_len:
            wave_form = onnxruntime.OrtValue.ortvalue_from_numpy(_audio[..., _slice_start: _slice_end], device_type_C, DEVICE_ID)
            token_ids = ort_session_C._sess.run_with_ort_values({in_name_C0: wave_form._ortvalue, in_name_C1: _language_idx}, out_name_C, run_options)
            saved_text += tokenizer.decode(token_ids[0].numpy().tolist())[0]
            _slice_start += _stride_step
            _slice_end = _slice_start + INPUT_AUDIO_LENGTH
        return _start_indices * _inv_audio_len, saved_text + ";", (_start, _end)

    def inference_C_paraformer(_start, _end, _inv_audio_len, _audio, _sample_rate, _is_english):
        _start_indices = _start * _sample_rate
        _audio = _audio[..., int(_start_indices): int(_end * _sample_rate)]
        audio_segment_len = _audio.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)  # You can adjust it.
        _stride_step = INPUT_AUDIO_LENGTH
        _slice_start = 0
        _slice_end = INPUT_AUDIO_LENGTH
        saved_text = ""
        while _slice_start < audio_segment_len:
            wave_form = onnxruntime.OrtValue.ortvalue_from_numpy(_audio[..., _slice_start: _slice_end], device_type_C, DEVICE_ID)
            token_ids = ort_session_C._sess.run_with_ort_values({in_name_C0: wave_form._ortvalue}, out_name_C, run_options)
            saved_text += tokenizer[token_ids[0].numpy()[0]]
            _slice_start += _stride_step
            _slice_end = _slice_start + INPUT_AUDIO_LENGTH
        if _is_english:
            saved_text = " ".join(text).replace("@@ ", "")
        else:
            saved_text = "".join(text)
        saved_text = saved_text.replace("</s>", "")
        return _start_indices * _inv_audio_len, saved_text + ";", (_start, _end)

    def inference_CD_beam_search(_start, _end, _inv_audio_len, _audio, _sample_rate, _init_input_ids, _init_history_len, _init_ids_len, _init_ids_len_1, _init_attention_mask_D_0, _init_attention_mask_D_1, _init_past_keys_D, _init_past_values_D, _init_save_id_beam, _init_repeat_penality, _init_batch_size, _init_penality_reset_count_beam, _init_save_id_greedy, _is_whisper):
        _start_indices = _start * _sample_rate
        _audio = _audio[..., int(_start_indices): int(_end * _sample_rate)]
        audio_segment_len = _audio.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)  # You can adjust it.
        _stride_step = INPUT_AUDIO_LENGTH
        _slice_start = 0
        _slice_end = INPUT_AUDIO_LENGTH
        saved_text = ''
        while _slice_start < audio_segment_len:
            input_feed_D = {
                in_name_D[-1]: _init_attention_mask_D_1,
                in_name_D[-2]: _init_ids_len,
                in_name_D[num_keys_values]: _init_input_ids,
                in_name_D[num_keys_values_plus_1]: _init_history_len
            }
            for i in range(num_layers):
                input_feed_D[in_name_D[i]] = _init_past_keys_D
            for i in range(num_layers, num_keys_values):
                input_feed_D[in_name_D[i]] = _init_past_values_D
            if USE_BEAM_SEARCH:
                input_feed_H[in_name_H[num_keys_values_plus_1]] = _init_save_id_beam
                input_feed_H[in_name_H[num_keys_values_plus_2]] = _init_repeat_penality
            else:
                input_feed_G[in_name_G[1]] = _init_repeat_penality
                input_feed_G[in_name_G[3]] = _init_batch_size
            if DO_REPEAT_PENALITY:
                if USE_BEAM_SEARCH:
                    input_feed_J = {in_name_J[2]: _init_penality_reset_count_beam}
                else:
                    penality_reset_count_greedy = _init_penality_reset_count_beam
            num_decode = 0
            wave_form = onnxruntime.OrtValue.ortvalue_from_numpy(_audio[..., _slice_start: _slice_end], device_type_C, DEVICE_ID)
            all_outputs_C = ort_session_C._sess.run_with_ort_values({in_name_C0: wave_form._ortvalue}, out_name_C, run_options)
            input_feed_D.update(zip(in_name_D[num_keys_values_plus_2: num_keys_values2_plus_2], all_outputs_C))
            while num_decode < generate_limit:
                all_outputs_D = ort_session_D._sess.run_with_ort_values(input_feed_D, out_name_D, run_options)
                if USE_BEAM_SEARCH:
                    if num_decode < 1:
                        input_feed_H.update(zip(in_name_H[:num_keys_values_plus_1], all_outputs_D))
                        all_outputs_H = ort_session_H._sess.run_with_ort_values(input_feed_H, out_name_H, run_options)
                        max_logits_idx = all_outputs_H[amount_of_outputs_H_minus_1].numpy()
                        input_feed_I[in_name_I[-4]] = all_outputs_H[amount_of_outputs_H_minus_2]
                        if DO_REPEAT_PENALITY:
                            input_feed_J[in_name_J[3]] = all_outputs_H[amount_of_outputs_H_minus_2]
                    else:
                        input_feed_I.update(zip(in_name_I[:num_keys_values_plus_1], all_outputs_D))
                        all_outputs_I = ort_session_I._sess.run_with_ort_values(input_feed_I, out_name_I, run_options)
                        max_logits_idx = all_outputs_I[amount_of_outputs_I_minus_1].numpy()
                    if max_logits_idx in ASR_STOP_TOKEN:
                        break
                    if DO_REPEAT_PENALITY and (num_decode >= PENALITY_RANGE):
                        input_feed_J[in_name_J[0]] = all_outputs_I[num_keys_values_plus_1]
                        input_feed_J[in_name_J[1]] = all_outputs_I[num_keys_values_plus_2]
                        all_outputs_J = ort_session_J._sess.run_with_ort_values(input_feed_J, out_name_J, run_options)
                        input_feed_J[in_name_J[2]] = all_outputs_J[2]
                        input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_J[0]
                        input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_J[1]
                    if num_decode < 1:
                        input_feed_D.update(zip(in_name_D[:num_keys_values_plus_1], all_outputs_H))
                        input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_H[num_keys_values_plus_1]
                        input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_H[num_keys_values_plus_2]
                        input_feed_I[in_name_I[num_keys_values_plus_3]] = all_outputs_H[num_keys_values_plus_3]
                    else:
                        input_feed_D.update(zip(in_name_D[:num_keys_values_plus_1], all_outputs_I))
                        input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_I[num_keys_values_plus_1]
                        input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_I[num_keys_values_plus_2]
                        input_feed_I[in_name_I[num_keys_values_plus_3]] = all_outputs_I[num_keys_values_plus_3]
                else:
                    input_feed_G[in_name_G[0]] = all_outputs_D[num_keys_values]
                    all_outputs_G = ort_session_G._sess.run_with_ort_values(input_feed_G, out_name_G, run_options)
                    max_logits_idx = all_outputs_G[0].numpy()[0, 0]
                    if max_logits_idx in ASR_STOP_TOKEN:
                        break
                    input_feed_D[in_name_D[num_keys_values]] = all_outputs_G[0]
                    if DO_REPEAT_PENALITY and (num_decode >= PENALITY_RANGE) and (_init_save_id_greedy[penality_reset_count_greedy] != max_logits_idx):
                        repeat_penality = all_outputs_G[1].numpy()
                        repeat_penality[..., penality_reset_count_greedy] = 1.0
                        repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(repeat_penality, device_type_C, DEVICE_ID)
                        input_feed_G[in_name_G[1]] = repeat_penality._ortvalue
                        penality_reset_count_greedy += 1
                    else:
                        input_feed_G[in_name_G[1]] = all_outputs_G[1]
                    _init_save_id_greedy[num_decode] = max_logits_idx
                    input_feed_D.update(zip(in_name_D[:num_keys_values], all_outputs_D))
                input_feed_D[in_name_D[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
                if num_decode < 1:
                    input_feed_D[in_name_D[-1]] = _init_attention_mask_D_0
                    if _is_whisper:
                        input_feed_D[in_name_D[-2]] = _init_ids_len_1
                num_decode += 1
            _slice_start += _stride_step
            _slice_end = _slice_start + INPUT_AUDIO_LENGTH
            if num_decode > 0:
                if USE_BEAM_SEARCH:
                    save_id_beam = all_outputs_I[num_keys_values_plus_1].numpy()[0]
                    for i, idx in enumerate(save_id_beam):
                        if idx in ASR_STOP_TOKEN:
                            save_id_beam = save_id_beam[:i]
                            break
                    save_token_array = remove_repeated_parts(save_id_beam, 3, save_id_beam.shape[-1])      # To handle "over-talking".
                else:
                    save_token_array = remove_repeated_parts(_init_save_id_greedy[:num_decode], 3,  num_decode)  # To handle "over-talking".
                if _is_whisper:
                    text, _ = tokenizer._decode_asr(
                        [{
                            'tokens': save_token_array.reshape(1, -1)
                        }],
                        return_timestamps=None,  # Do not support return timestamps
                        return_language=None,
                        time_precision=0
                    )
                else:
                    text = (''.join([tokenizer.dict[int(id)] for id in save_token_array])).replace(tokenizer.SPM_SPACE, ' ').strip()
                saved_text += text
        return _start_indices * _inv_audio_len, saved_text + ';', (_start, _end)

    def inference_CD_dolphin(_start, _end, _inv_audio_len, _audio, _sample_rate, _init_input_ids, _init_history_len, _init_ids_len, _init_ids_len_1, _init_ids_len_2, _init_ids_0, _init_ids_len_5, _init_ids_7, _init_ids_145, _init_ids_324, _init_ids_39999, _init_ids_vocab_size, _init_attention_mask_D_0, _init_attention_mask_D_1, _init_past_keys_D, _init_past_values_D, _init_save_id_beam, _init_repeat_penality, _init_batch_size, _init_penality_reset_count_beam, _init_save_id_greedy, _lang_id, _region_id):
        start_indices = _start * _sample_rate
        _audio = _audio[..., int(start_indices): int(_end * _sample_rate)]
        audio_segment_len = _audio.shape[-1]
        INPUT_AUDIO_LENGTH = min(MAX_ASR_SEGMENT, audio_segment_len)  # You can adjust it.
        _stride_step = INPUT_AUDIO_LENGTH
        _slice_start = 0
        _slice_end = INPUT_AUDIO_LENGTH
        saved_text = ''
        while _slice_start < audio_segment_len:
            wave_form = onnxruntime.OrtValue.ortvalue_from_numpy(_audio[..., _slice_start: _slice_end], device_type_C, DEVICE_ID)
            all_outputs_C = ort_session_C._sess.run_with_ort_values({in_name_C0: wave_form._ortvalue}, out_name_C, run_options)
            input_feed_D = {
                in_name_D[-1]: _init_attention_mask_D_1,
                in_name_D[num_keys_values_plus_1]: _init_history_len,
            }
            for i in range(num_layers):
                input_feed_D[in_name_D[i]] = _init_past_keys_D
                input_feed_D[in_name_D[layer_indices[i]]] = all_outputs_C[i]
            for i in range(num_layers, num_keys_values):
                input_feed_D[in_name_D[i]] = _init_past_values_D
                input_feed_D[in_name_D[layer_indices[i]]] = all_outputs_C[i]
            if detect_language:
                input_feed_D[in_name_D[-4]] = _init_ids_7
                input_feed_D[in_name_D[-3]] = _init_ids_145
                input_feed_D[in_name_D[-2]] = _init_ids_len_2
                input_feed_D[in_name_D[num_keys_values]] = _init_ids_39999
                all_outputs_D = ort_session_D._sess.run_with_ort_values(input_feed_D, out_name_D, run_options)
                _lang_id = ort_session_K._sess.run_with_ort_values({in_name_K: all_outputs_D[num_keys_values]}, out_name_K, run_options)[0].numpy()[0] + 7
                for i in range(num_layers):
                    input_feed_D[in_name_D[i]] = _init_past_keys_D
                for i in range(num_layers, num_keys_values):
                    input_feed_D[in_name_D[i]] = _init_past_values_D
            if detect_region:
                input_feed_D[in_name_D[-4]] = _init_ids_145
                input_feed_D[in_name_D[-3]] = _init_ids_324
                input_feed_D[in_name_D[-2]] = _init_ids_len_2
                in_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, _lang_id]], dtype=np.int32), device_type_C, DEVICE_ID)
                input_feed_D[in_name_D[num_keys_values]] = in_ids._ortvalue
                all_outputs_D = ort_session_D._sess.run_with_ort_values(input_feed_D, out_name_D, run_options)
                _region_id = ort_session_K._sess.run_with_ort_values({in_name_K: all_outputs_D[num_keys_values]}, out_name_K, run_options)[0].numpy()[0] + 145
                for i in range(num_layers):
                    input_feed_D[in_name_D[i]] = _init_past_keys_D
                for i in range(num_layers, num_keys_values):
                    input_feed_D[in_name_D[i]] = _init_past_values_D
            if detect_language or detect_region:
                lang_str = tokenizer.decode(_lang_id)
                region_str = tokenizer.decode(_region_id)
                transcribe_language = INV_DOLPHIN_LANGUAGE_MAP.get(f'{lang_str}-{region_str}', 'Unknown')  # Update the global variable: transcribe_language
            input_feed_D[in_name_D[-4]] = _init_ids_0
            input_feed_D[in_name_D[-3]] = _init_ids_vocab_size
            input_feed_D[in_name_D[-2]] = _init_ids_len_5
            in_ids = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999, _lang_id, _region_id, 6, 324]], dtype=np.int32), device_type_C, DEVICE_ID)
            input_feed_D[in_name_D[num_keys_values]] = in_ids._ortvalue  # start_id = 39999; itn = 5; asr = 6; no_timestamp = 324
            if USE_BEAM_SEARCH:
                input_feed_H[in_name_H[num_keys_values_plus_1]] = _init_save_id_beam
                input_feed_H[in_name_H[num_keys_values_plus_2]] = _init_repeat_penality
            else:
                input_feed_G[in_name_G[1]] = _init_repeat_penality
                input_feed_G[in_name_G[3]] = _init_batch_size
            if DO_REPEAT_PENALITY:
                if USE_BEAM_SEARCH:
                    input_feed_J = {in_name_J[2]: _init_penality_reset_count_beam}
                else:
                    penality_reset_count_greedy = _init_penality_reset_count_beam
            num_decode = 0
            while num_decode < generate_limit:
                all_outputs_D = ort_session_D._sess.run_with_ort_values(input_feed_D, out_name_D, run_options)
                if USE_BEAM_SEARCH:
                    if num_decode < 1:
                        input_feed_H.update(zip(in_name_H[:num_keys_values_plus_1], all_outputs_D))
                        all_outputs_H = ort_session_H._sess.run_with_ort_values(input_feed_H, out_name_H, run_options)
                        max_logits_idx = all_outputs_H[amount_of_outputs_H_minus_1].numpy()
                        input_feed_I[in_name_I[-4]] = all_outputs_H[amount_of_outputs_H_minus_2]
                        if DO_REPEAT_PENALITY:
                            input_feed_J[in_name_J[3]] = all_outputs_H[amount_of_outputs_H_minus_2]
                    else:
                        input_feed_I.update(zip(in_name_I[:num_keys_values_plus_1], all_outputs_D))
                        all_outputs_I = ort_session_I._sess.run_with_ort_values(input_feed_I, out_name_I, run_options)
                        max_logits_idx = all_outputs_I[amount_of_outputs_I_minus_1].numpy()
                    if max_logits_idx in ASR_STOP_TOKEN:
                        break
                    if DO_REPEAT_PENALITY and (num_decode >= PENALITY_RANGE):
                        input_feed_J[in_name_J[0]] = all_outputs_I[num_keys_values_plus_1]
                        input_feed_J[in_name_J[1]] = all_outputs_I[num_keys_values_plus_2]
                        all_outputs_J = ort_session_J._sess.run_with_ort_values(input_feed_J, out_name_J, run_options)
                        input_feed_J[in_name_J[2]] = all_outputs_J[2]
                        input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_J[0]
                        input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_J[1]
                    if num_decode < 1:
                        input_feed_D.update(zip(in_name_D[:num_keys_values_plus_1], all_outputs_H))
                        input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_H[num_keys_values_plus_1]
                        input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_H[num_keys_values_plus_2]
                        input_feed_I[in_name_I[num_keys_values_plus_3]] = all_outputs_H[num_keys_values_plus_3]
                    else:
                        input_feed_D.update(zip(in_name_D[:num_keys_values_plus_1], all_outputs_I))
                        input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_I[num_keys_values_plus_1]
                        input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_I[num_keys_values_plus_2]
                        input_feed_I[in_name_I[num_keys_values_plus_3]] = all_outputs_I[num_keys_values_plus_3]
                else:
                    input_feed_G[in_name_G[0]] = all_outputs_D[num_keys_values]
                    all_outputs_G = ort_session_G._sess.run_with_ort_values(input_feed_G, out_name_G, run_options)
                    max_logits_idx = all_outputs_G[0].numpy()[0, 0]
                    if max_logits_idx in ASR_STOP_TOKEN:
                        break
                    input_feed_D[in_name_D[num_keys_values]] = all_outputs_G[0]
                    if DO_REPEAT_PENALITY and (num_decode >= PENALITY_RANGE) and (_init_save_id_greedy[penality_reset_count_greedy] != max_logits_idx):
                        repeat_penality = all_outputs_G[1].numpy()
                        repeat_penality[..., penality_reset_count_greedy] = 1.0
                        repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(repeat_penality, device_type_C, DEVICE_ID)
                        input_feed_G[in_name_G[1]] = repeat_penality._ortvalue
                        penality_reset_count_greedy += 1
                    else:
                        input_feed_G[in_name_G[1]] = all_outputs_G[1]
                    _init_save_id_greedy[num_decode] = max_logits_idx
                    input_feed_D.update(zip(in_name_D[:num_keys_values], all_outputs_D))
                input_feed_D[in_name_D[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
                if num_decode < 1:
                    input_feed_D[in_name_D[-1]] = _init_attention_mask_D_0
                    input_feed_D[in_name_D[-2]] = _init_ids_len_1
                num_decode += 1
            _slice_start += _stride_step
            _slice_end = _slice_start + INPUT_AUDIO_LENGTH
            if num_decode > 0:
                if USE_BEAM_SEARCH:
                    save_id_beam = all_outputs_I[num_keys_values_plus_1].numpy()[0]
                    for i, idx in enumerate(save_id_beam):
                        if idx in ASR_STOP_TOKEN:
                            save_id_beam = save_id_beam[:i]
                            break
                    save_token_array = save_id_beam
                else:
                    save_token_array = _init_save_id_greedy[:num_decode]
                for i in save_token_array:
                    saved_text += tokenizer.decode(i)
                saved_text = saved_text.replace('▁', ' ')
        return start_indices * _inv_audio_len, saved_text + ';', (_start, _end)

    def run_inference_x(func, args_list, progress_prefix="Progress"):
        results = []
        for args in args_list:
            res = func(*args)
            results.append(res)
            print(f"{progress_prefix}: {res[0]:.3f}%")
        return results

    def get_ort_device(bind_device_type, bind_device_id):
        return onnxruntime.capi._pybind_state.OrtDevice(onnxruntime.capi.onnxruntime_inference_collection.get_ort_device_type(bind_device_type, bind_device_id), onnxruntime.capi._pybind_state.OrtDevice.default_memory(), bind_device_id)

    def bind_inputs_to_device(io_binding, input_names, ortvalue, num_inputs):
        for i in range(num_inputs):
            io_binding.bind_ortvalue_input(input_names[i], ortvalue[i])

    def bind_outputs_to_device(io_binding, output_names, bind_device_type, num_outputs):
        for i in range(num_outputs):
            io_binding.bind_output(output_names[i], bind_device_type)

    total_process_time = time.time()
    print("----------------------------------------------------------------------------------------------------------")
    FIRST_RUN = True
    HAS_CACHE = False

    transcribe_language_dolphin = transcribe_language
    transcribe_language = transcribe_language.split('-')[0].strip()
    translate_language = translate_language.split('-')[0].strip()

    task_queue = []
    if os.path.isfile(file_path_input):
        if file_path_input.endswith(MEDIA_EXTENSIONS):
            task_queue.append(file_path_input)
        else:
            error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}' does not exist or is not in a legal media format."
            print(error)
    elif os.path.isdir(file_path_input):
        for file in os.listdir(file_path_input):
            if file.endswith(MEDIA_EXTENSIONS):
                task_queue.append(f"{file_path_input}/{file}")
            else:
                error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}/{file}' does not exist or is not in a legal media format."
                print(error)
    else:
        error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}' does not exist or is not in a legal media format."
        print(error)
        return error

    total_task = len(task_queue)
    if total_task < 1:
        error = f"\n指定的路径或文件不存在，或是不合规的媒体格式。\nThe specified path or file '{file_path_input}' does not exist or is not in a legal media format."
        print(error)
        return error
    else:
        print(f'\n找到了 {total_task} 个媒体文件。Totally {total_task} media found.')

    usable_providers = onnxruntime.get_available_providers()

    try:
        cpu_model = cpuinfo.get_cpu_info().get('brand_raw', 'unknown').lower()
        has_npu = 'intel' in cpu_model and 'ultra' in cpu_model
    except:
        has_npu = False

    if len(usable_providers) > 1:
        if hardware != "CPU":
            model_dtype = "FP16"
            has_npu = (('OpenVINOExecutionProvider' in usable_providers) and has_npu) or ('VitisAIExecutionProvider' in usable_providers) or ('QNNExecutionProvider' in usable_providers)
            cuda_options = {
                'device_id': DEVICE_ID,
                'gpu_mem_limit': 64 * 1024 * 1024 * 1024,    # 64 GB
                'arena_extend_strategy': 'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"]
                'cudnn_conv_algo_search': 'EXHAUSTIVE',      # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
                'sdpa_kernel': '2',                          # ["0", "1", "2"]
                'use_tf32': '1',
                'fuse_conv_bias': '0',
                'cudnn_conv_use_max_workspace': '1',
                'cudnn_conv1d_pad_to_nc1d': '1',
                'tunable_op_enable': '0',
                'tunable_op_tuning_enable': '0',
                'tunable_op_max_tuning_duration_ms': 1000,
                'do_copy_in_default_stream': '1',
                'enable_cuda_graph': '0',                    # Set to '0' to avoid potential errors when enabled.
                'prefer_nhwc': '0',
                'enable_skip_layer_norm_strict_mode': '0',
                'use_ep_level_unified_stream': '0',
            }
            rocm_options = {
                'device_id': DEVICE_ID,
                'do_copy_in_default_stream': True,
                'gpu_mem_limit': 64 * 1024 * 1024 * 1024,    # 64 GB,
                'arena_extend_strategy': 'kNextPowerOfTwo',  # ["kNextPowerOfTwo", "kSameAsRequested"],
            }
            if 'NvTensorRTRTXExecutionProvider' in usable_providers:
                device_type = 'cuda'
                ORT_Accelerate_Providers = ['NvTensorRTRTXExecutionProvider']
                if 'CUDAExecutionProvider' in usable_providers:
                    ORT_Accelerate_Providers = ['NvTensorRTRTXExecutionProvider', 'CUDAExecutionProvider']
            elif 'TensorrtExecutionProvider' in usable_providers:
                device_type = 'cuda'
                ORT_Accelerate_Providers = ['TensorrtExecutionProvider']
                if 'CUDAExecutionProvider' in usable_providers:
                    ORT_Accelerate_Providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
            elif 'CUDAExecutionProvider' in usable_providers:
                device_type = 'cuda'
                ORT_Accelerate_Providers = ['CUDAExecutionProvider']
            elif 'MIGraphXExecutionProvider' in usable_providers:
                device_type = 'gpu'
                ORT_Accelerate_Providers = ['MIGraphXExecutionProvider']
                if 'ROCMExecutionProvider' in usable_providers:
                    ORT_Accelerate_Providers = ['MIGraphXExecutionProvider', 'ROCMExecutionProvider']
            elif 'ROCMExecutionProvider' in usable_providers:
                device_type = 'gpu'
                ORT_Accelerate_Providers = ['ROCMExecutionProvider']
            elif 'VitisAIExecutionProvider' in usable_providers:
                device_type = 'npu'
                ORT_Accelerate_Providers = ['VitisAIExecutionProvider']
            elif 'WebGpuExecutionProvider' in usable_providers:
                device_type = 'webgpu'
                ORT_Accelerate_Providers = ['WebGpuExecutionProvider']
            elif 'DmlExecutionProvider' in usable_providers:
                device_type = 'dml'
                ORT_Accelerate_Providers = ['DmlExecutionProvider']
            elif 'OpenVINOExecutionProvider' in usable_providers:
                device_type = 'gpu'
                ORT_Accelerate_Providers = ['OpenVINOExecutionProvider']
            elif 'CoreMLExecutionProvider' in usable_providers:
                device_type = 'gpu'
                ORT_Accelerate_Providers = ['CoreMLExecutionProvider']
            elif 'QNNExecutionProvider' in usable_providers:
                device_type = 'npu'
                ORT_Accelerate_Providers = ['QNNExecutionProvider']
            else:
                device_type = 'cpu'
                ORT_Accelerate_Providers = ['CPUExecutionProvider']
        else:
            model_dtype = "FP32"
            device_type = 'cpu'
            ORT_Accelerate_Providers = ['CPUExecutionProvider']
    else:
        has_npu = False
        model_dtype = 'FP32'
        device_type = 'cpu'
        ORT_Accelerate_Providers = ['CPUExecutionProvider']

    if 'OpenVINOExecutionProvider' in ORT_Accelerate_Providers:
        provider_options = [
            {
                'device_type': device_type.upper(),
                'precision': 'ACCURACY',
                'model_priority': 'HIGH',
                'num_of_threads': parallel_threads,
                'num_streams': 1,
                'enable_opencl_throttling': False,
                'enable_qdq_optimizer': False,
                'disable_dynamic_shapes': False
            }
        ]
    elif 'NvTensorRTRTXExecutionProvider' in ORT_Accelerate_Providers[0]:
        if len(ORT_Accelerate_Providers) > 1:
            ORT_Accelerate_Providers = [
                ('NvTensorRTRTXExecutionProvider', {
                    'device_id': DEVICE_ID,
                    'nv_max_workspace_size': 0,      # 0 for Auto
                    'nv_dump_subgraphs': False,
                    'nv_cuda_graph_enable': False,
                    'nv_detailed_build_log': False,
                    'nv_profile_min_shapes': 'input_tensor_1:1x1x4800',
                    'nv_profile_max_shapes': 'input_tensor_1:1x1x960000',
                    'nv_profile_opt_shapes': 'input_tensor_1:1x1x480000'
                }),
                ('CUDAExecutionProvider', cuda_options)
            ]
            provider_options = None
        else:
            provider_options = [
                {
                    'device_id': DEVICE_ID,
                    'nv_max_workspace_size': 0,      # 0 for Auto
                    'nv_dump_subgraphs': False,
                    'nv_cuda_graph_enable': False,
                    'nv_detailed_build_log': False,
                    'nv_profile_min_shapes': 'input_tensor_1:1x1x4800',
                    'nv_profile_max_shapes': 'input_tensor_1:1x1x960000',
                    'nv_profile_opt_shapes': 'input_tensor_1:1x1x480000'
                }
            ]
    elif 'TensorrtExecutionProvider' in ORT_Accelerate_Providers[0]:
        trt_options = {
                    'device_id': DEVICE_ID,
                    'trt_detailed_build_log': False,
                    'trt_timing_cache_enable': False,
                    'trt_force_timing_cache': True,
                    'trt_timing_cache_path': "./Cache",
                    'trt_layer_norm_fp32_fallback': False,
                    'trt_context_memory_sharing_enable': True,
                    'trt_dump_subgraphs': True,
                    'trt_force_sequential_engine_build': False,  # For multi-GPU
                    'trt_dla_enable': True,
                    'trt_build_heuristics_enable': True,
                    'trt_sparsity_enable': True,
                    'trt_engine_hw_compatible': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': "./Cache",
                    'trt_cuda_graph_enable': False,
                    'trt_fp16_enable': True,
                    'trt_int8_enable': False,
                    'trt_max_partition_iterations': 1000,
                    'trt_min_subgraph_size': 1,
                    'trt_max_workspace_size': 64 * 1073741824,  # 64GB
                    'trt_builder_optimization_level': 5,
                    'trt_auxiliary_streams': 0,
                    'trt_profile_min_shapes': 'input_tensor_1:1x1x4800',
                    'trt_profile_max_shapes': 'input_tensor_1:1x1x960000',
                    'trt_profile_opt_shapes': 'input_tensor_1:1x1x480000'
                }
        if len(ORT_Accelerate_Providers) > 1:
            ORT_Accelerate_Providers = [
                ('TensorrtExecutionProvider', trt_options),
                ('CUDAExecutionProvider', cuda_options)
            ]
            provider_options = None
        else:
            provider_options = [trt_options]
    elif 'CUDAExecutionProvider' in ORT_Accelerate_Providers[0]:
        provider_options = [cuda_options]
    elif 'MIGraphXExecutionProvider' in ORT_Accelerate_Providers[0]:
        migx_options = {
                'device_id': DEVICE_ID,
                'migraphx_fp16_enable': 1,
                'migraphx_int8_enable': 0,
                'migraphx_save_compiled_model': 1,
                'migraphx_save_compiled_path': './Cache',
                'migraphx_load_compiled_model': 0,
                'migraphx_load_compiled_path': './Cache',
                'migraphx_exhaustive_tune': 1,
                'migraphx_mem_limit': 64 * 1024 * 1024 * 1024  # 64 GB
            }
        if len(ORT_Accelerate_Providers) > 1:
            ORT_Accelerate_Providers = [
                ('MIGraphXExecutionProvider', migx_options),
                ('ROCMExecutionProvider', rocm_options)
            ]
            provider_options = None
        else:
            provider_options = [migx_options]
    elif 'ROCMExecutionProvider' in ORT_Accelerate_Providers[0]:
        provider_options = [rocm_options]
    elif 'VitisAIExecutionProvider' in usable_providers:
        provider_options = [
            {
                'log_level': 'error'
            }
        ]
    elif 'DmlExecutionProvider' in ORT_Accelerate_Providers[0]:
        provider_options = [
            {
                'device_id': DEVICE_ID,
                'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
                'device_filter': 'gpu'                         # [any, npu, gpu]
            }
        ]
    elif 'CoreMLExecutionProvider' in ORT_Accelerate_Providers[0]:
        provider_options = [
            {
                'ModelFormat': 'MLProgram',                   # [MLProgram, NeuralNetwork]
                'MLComputeUnits': 'ALL',                      # [CPUOnly, CPUAndNeuralEngine, CPUAndGPU, ALL]
                'RequireStaticInputShapes': '0',              # [0, 1]
                'EnableOnSubgraphs': '1',                     # [0, 1]
                'SpecializationStrategy': 'Default',          # [Default, FastPrediction]
                'AllowLowPrecisionAccumulationOnGPU': '1',    # [0, 1]; 1: Use low precision data(float16) to accumulate data.
                'ProfileComputePlan': '0'                     # [0, 1]
            }
        ]
    elif 'QNNExecutionProvider' in ORT_Accelerate_Providers[0]:
        provider_options = [
            {
                'backend_type': 'htp',
                'backend_path': './QNN_Library/QnnHtp.dll',
                'profiling_level': 'off',
                'profiling_file_path': './Cache/qnn_profile_path.csv',
                'rpc_control_latency': '10',
                'vtcm_mb': '0',                              # 0 for auto
                'htp_performance_mode': 'burst',
                'qnn_context_priority': 'high',
                'htp_graph_finalization_optimization_mode': '3',
                'soc_model': '0',                            # 0 for auto
                'device_id': '0',
                'offload_graph_io_quantization': '1',
                'enable_htp_shared_memory_allocator': '0',
                'enable_htp_fp16_precision': '1',
                'ep.context_enable': '1',
                'ep.context_embed_mode': '1',
                'ep.context_file_path': './Cache/qnn_ctx.onnx'
            }
        ]
    else:
        provider_options = None
        print("\n仅 CPU 可用。Only CPU is available.")

    slider_denoise_factor_minus = 1.0 - slider_denoise_factor
    slider_denoise_factor = float(slider_denoise_factor)

    if model_denoiser != 'NONE':
        USE_DENOISED = True
        model_denoiser = model_denoiser.split("-")[0].strip()
        if model_denoiser != 'MelBandRoformer':
            denoiser_format = 'FP32'
        else:
            denoiser_format = model_dtype
    else:
        USE_DENOISED = False

    if USE_DENOISED:
        onnx_model_A = f'./Denoiser/{model_denoiser}/{denoiser_format}/{model_denoiser}.onnx'
        if os.path.isfile(onnx_model_A):
            print(f'\n找到了降噪器。Found the Denoiser-{model_denoiser}.')
        else:
            error = f"\n降噪器不存在。The Denoiser-{model_denoiser} doesn't exist."
            print(error)
            return error
    else:
        onnx_model_A = None
        print('\n此任务不使用降噪器。\nThis task is running without the denoiser.')

    if model_vad == VAD_LIST[1]:
        if os.path.isdir(PYTHON_PACKAGE + r'/faster_whisper'):
            vad_type = 1
            onnx_model_B = None
            print(f'\n找到了 VAD-Faster_Whisper-Silero。Found the VAD-Faster_Whisper-Silero.')
        else:
            error = "\nVAD-Faster_Whisper-Silero 不存在。请运行 pip install fastest-whisper --upgrade。\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'"
            print(error)
            return error
    elif model_vad == VAD_LIST[2]:
        if os.path.isdir(PYTHON_PACKAGE + r'/silero_vad'):
            vad_type = 2
            onnx_model_B = None
            print(f'\n找到了 VAD-Official_Silero。Found the VAD-Official_Silero.')
        else:
            error = "\nVAD-Official_Silero不存在。请运行 pip install silero-vad --upgrade。\nThe VAD-Official_Silero doesn't exist. Please run 'pip install silero-vad --upgrade'"
            print(error)
            return error
    elif model_vad == VAD_LIST[3]:
        if os.path.isfile(r'./VAD/Pyannote_Segmentation/pytorch_model.bin'):
            vad_type = 3
            onnx_model_B = None
            print(f'\n找到了 VAD-Pyannote_Segmentation。Found the VAD-Pyannote_Segmentation.')
        else:
            error = "\nVAD-Pyannote_Segmentation 不存在。请运行'pip install pyannote.audio --upgrade' 并从 https://huggingface.co/pyannote/segmentation-3.0 下载 pytorch_model.bin。\nThe VAD-Pyannote_Segmentation doesn't exist. Please run 'pip install pyannote.audio' --upgrade and Download the pytorch_model.bin from https://huggingface.co/pyannote/segmentation-3.0"
            print(error)
            print("\nVAD-Pyannote_Segmentation 不存在。回退到 Faster_Whisper-Silero。\nThe VAD-Pyannote_Segmentation doesn't exist. Fallback to Faster_Whisper-Silero.")
            if os.path.isdir(PYTHON_PACKAGE + r'/faster_whisper'):
                vad_type = 1
                onnx_model_B = None
                print(f'\n找到了 VAD-Faster_Whisper-Silero。Found the VAD-Faster_Whisper-Silero.')
            else:
                error = "\nVAD-Faster_Whisper-Silero 不存在。请运行 pip install fastest-whisper --upgrade。\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'"
                print(error)
                return error
    elif model_vad == VAD_LIST[4]:
        if os.path.isfile(r'./VAD/HumAware/HumAwareVAD.jit'):
            vad_type = 4
            onnx_model_B = None
            print(f'\n找到了 VAD-HumAware。Found the VAD-HumAware.')
        else:
            error = '\nVAD-HumAware不存在。'
            print(error)
            return error
    elif model_vad == VAD_LIST[5]:
        onnx_model_B = f'./VAD/NVIDIA_Frame_VAD_Multilingual_MarbleNet/FP32/NVIDIA_MarbleNet.onnx'
        if os.path.isfile(onnx_model_B):
            vad_type = 5
            print(f'\n找到了 VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet。Found the VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet.')
        else:
            error = "\nVAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet 不存在。\nThe VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet doesn't exist. "
            print(error)
            return error
    elif model_vad == VAD_LIST[6]:
        sys_os = platform.system()
        if sys_os == 'Darwin':
            # Current not support MAC + Python. Fallback to Default.
            print("\nTAN-VAD 目前不支持 MAC + Python。回退到 Faster_Whisper-Silero。\nThe TAN-VAD vurrently doesn't support MAC + Python. Fallback to Faster_Whisper-Silero.")
            if os.path.isdir(PYTHON_PACKAGE + "/faster_whisper"):
                vad_type = 1
                onnx_model_B = None
                print('\n找到了 VAD-Faster_Whisper-Silero。Found the VAD-Faster_Whisper-Silero.')
            else:
                error = "\nVAD-Faster_Whisper-Silero 不存在。请运行 pip install fastest-whisper --upgrade。\nThe VAD-Faster_Whisper-Silero doesn't exist. Please run 'pip install faster-whisper --upgrade'"
                print(error)
                return error
        else:
            if sys_os != "Windows":
                lib_path = r'./VAD/TEN/lib/Linux/x64/libten_vad.so'
                print('\n您正在使用 Linux 操作系统的 TEN-VAD。首次启动前，请运行以下命令。\nYou are using the TEN-VAD with Linux OS. Please run the following commands before the first launch.\n\nsudo apt update\nsudo apt install libc++1')
            else:
                lib_path = r'./VAD/TEN/lib/Windows/x64/libten_vad.dll'
            if os.path.isfile(lib_path):
                vad_type = 6
                onnx_model_B = None
                print('\n找到了 VAD-TEN。Found the VAD-TEN.')
            else:
                error = "\nVAD-TEN 不存在。\nThe VAD-TEN doesn't exist. "
                print(error)
                return error
    else:
        vad_type = -1
        onnx_model_B = None

    if 'Whisper' in model_asr:
        onnx_model_C = f'./ASR/Whisper/{model_dtype}/{model_asr}/Whisper_Encoder.onnx'
        onnx_model_D = f'./ASR/Whisper/{model_dtype}/{model_asr}/Whisper_Decoder.onnx'
        onnx_model_G = f'./ASR/Whisper/{model_dtype}/{model_asr}/Greedy_Search.onnx'
        onnx_model_H = f'./ASR/Whisper/{model_dtype}/{model_asr}/First_Beam_Search.onnx'
        onnx_model_I = f'./ASR/Whisper/{model_dtype}/{model_asr}/Second_Beam_Search.onnx'
        onnx_model_J = f'./ASR/Whisper/{model_dtype}/{model_asr}/Reset_Penality.onnx'
        if (
                os.path.isfile(onnx_model_C) and
                os.path.isfile(onnx_model_D) and
                os.path.isfile(onnx_model_G) and
                os.path.isfile(onnx_model_H) and
                os.path.isfile(onnx_model_I) and
                os.path.isfile(onnx_model_J)
        ):
            print(f'\n找到了 ASR。Found the {model_asr}.')
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error

        asr_type = 0
        tokenizer = AutoTokenizer.from_pretrained(f'./ASR/Whisper/Tokenizer/{model_asr}')
        if 'v0.3' in model_asr and 'Anime' in model_asr:  # For https://huggingface.co/efwkjn/whisper-ja-anime-v0.3
            custom_vocab = True
        else:
            custom_vocab = False
        target_language_id = get_language_whisper(transcribe_language, custom_vocab)
        if 'v3' in model_asr:
            USE_V3 = True
        else:
            USE_V3 = False
        if ('Whisper' in model_llm) and ('Translate' in task):
            whisper_start_token, ASR_STOP_TOKEN, target_task_id = get_task_id_whisper('Translate', USE_V3, custom_vocab)
        else:
            whisper_start_token, ASR_STOP_TOKEN, target_task_id = get_task_id_whisper('Transcribe', USE_V3, custom_vocab)
    elif 'SenseVoice' in model_asr:
        onnx_model_C = r'./ASR/SenseVoice/Small/FP32/SenseVoice.onnx'
        if os.path.isfile(onnx_model_C):
            print(f'\n找到了 ASR。Found the {model_asr}.')
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 1
        tokenizer = SentencePieceProcessor()
        tokenizer.Load(r'./ASR/SenseVoice/Small/Tokenizer/chn_jpn_yue_eng_ko_spectok.bpe.model')
        target_language_id = get_language_sensevoice(transcribe_language)
        target_task_id = None
        onnx_model_D = None
    elif 'Paraformer' in model_asr:
        if 'english' in transcribe_language.lower():
            is_english = True
            tokens_path = r'./ASR/Paraformer/English/Large/Tokenizer/tokens.json'
            onnx_model_C = r'./ASR/Paraformer/English/Large/FP32/Paraformer.onnx'
        else:
            is_english = False
            tokens_path = r'./ASR/Paraformer/Chinese/Large/Tokenizer/tokens.json'
            onnx_model_C = r'./ASR/Paraformer/Chinese/Large/FP32/Paraformer.onnx'
        if os.path.isfile(onnx_model_C):
            print(f'\n找到了 ASR。Found the {model_asr}.')
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 2
        with open(tokens_path, 'r', encoding='UTF-8') as json_file:
            tokenizer = np.array(json.load(json_file), dtype=np.str_)
        target_language_id = None
        target_task_id = None
        onnx_model_D = None
    elif 'FireRedASR' in model_asr:
        onnx_model_C = f'./ASR/FireRedASR/AED/L/{model_dtype}/FireRedASR_Encoder.onnx'
        onnx_model_D = f'./ASR/FireRedASR/AED/L/{model_dtype}/FireRedASR_Decoder.onnx'
        onnx_model_G = f'./ASR/FireRedASR/AED/L/{model_dtype}/Greedy_Search.onnx'
        onnx_model_H = f'./ASR/FireRedASR/AED/L/{model_dtype}/First_Beam_Search.onnx'
        onnx_model_I = f'./ASR/FireRedASR/AED/L/{model_dtype}/Second_Beam_Search.onnx'
        onnx_model_J = f'./ASR/FireRedASR/AED/L/{model_dtype}/Reset_Penality.onnx'
        if (
                os.path.isfile(onnx_model_C) and
                os.path.isfile(onnx_model_D) and
                os.path.isfile(onnx_model_G) and
                os.path.isfile(onnx_model_H) and
                os.path.isfile(onnx_model_I) and
                os.path.isfile(onnx_model_J)
        ):
            print(f'\n找到了 ASR。Found the {model_asr}.')
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 3
        tokenizer = ChineseCharEnglishSpmTokenizer(r'./ASR/FireRedASR/AED/L/Tokenizer/dict.txt', r'./ASR/FireRedASR/AED/L/Tokenizer/train_bpe1000.model')
    elif 'Dolphin' in model_asr:
        path = f'./ASR/Dolphin/Small/{model_dtype}/'
        onnx_model_C = path + 'Dolphin_Encoder.onnx'
        onnx_model_D = path + 'Dolphin_Decoder.onnx'
        onnx_model_G = path + 'Greedy_Search.onnx'
        onnx_model_H = path + 'First_Beam_Search.onnx'
        onnx_model_I = path + 'Second_Beam_Search.onnx'
        onnx_model_J = path + 'Reset_Penality.onnx'
        onnx_model_K = path + 'Argmax.onnx'
        if (
                os.path.isfile(onnx_model_C) and
                os.path.isfile(onnx_model_D) and
                os.path.isfile(onnx_model_G) and
                os.path.isfile(onnx_model_H) and
                os.path.isfile(onnx_model_I) and
                os.path.isfile(onnx_model_J) and
                os.path.isfile(onnx_model_K)
        ):
            print(f'\n找到了 ASR。Found the {model_asr}.')
        else:
            error = f"\n未找到模型。The {model_asr} doesn't exist."
            print(error)
            return error
        asr_type = 4
        tokenizer = Dolphin_Tokenizer(r'./ASR/Dolphin/Small/Tokenizer/vocab_Dolphin.txt')
        vocab_size = tokenizer.num_vocab
    else:
        error = f"\n未找到模型。The {model_asr} doesn't exist."
        print(error)
        return error

    # ONNX Runtime settings
    session_opts = onnxruntime.SessionOptions()
    run_options = onnxruntime.RunOptions()
    session_opts.log_severity_level = 4                      # Fatal level, it an adjustable value.
    session_opts.log_verbosity_level = 4                     # Fatal level, it an adjustable value.
    run_options.log_severity_level = 4                       # Fatal level, it an adjustable value.
    run_options.log_verbosity_level = 4                      # Fatal level, it an adjustable value.
    session_opts.inter_op_num_threads = parallel_threads     # Run different nodes with num_threads. Set 0 for auto.
    session_opts.intra_op_num_threads = parallel_threads     # Under the node, execute the operators with num_threads. Set 0 for auto.
    session_opts.enable_cpu_mem_arena = True                 # True for execute speed; False for less memory usage.
    session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
    session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
    session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
    session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
    session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
    session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
    session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
    session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
    session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
    session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
    
    run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')
    if 'QNNExecutionProvider' in ORT_Accelerate_Providers[0]:
        run_options.add_run_config_entry('qnn.htp_perf_mode', 'burst')
        run_options.add_run_config_entry('qnn.htp_perf_mode_post_run', 'burst')
        run_options.add_run_config_entry('qnn.rpc_control_latency', '10')

    print('----------------------------------------------------------------------------------------------------------')
    print('\n正在加载所需的模型和目标文件。Now loading the required models and target files.')
    # Load VAD model
    if vad_type == 1:
        slider_vad_MIN_SPEECH_DURATION_ms = int(slider_vad_MIN_SPEECH_DURATION * 1000)
    elif vad_type == 2:
        import torch
        torch.set_num_threads(parallel_threads)
        silero_vad = load_silero_vad(onnx=True)
        slider_vad_MIN_SPEECH_DURATION_ms = int(slider_vad_MIN_SPEECH_DURATION * 1000)
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 3:
        import torch
        torch.set_num_threads(parallel_threads)
        pyannote_vad = Model.from_pretrained("./VAD/Pyannote_Segmentation/pytorch_model.bin")
        pyannote_vad_pipeline = VoiceActivityDetection(segmentation=pyannote_vad)
        HYPER_PARAMETERS = {
            "min_duration_on": slider_vad_MIN_SPEECH_DURATION,
            "min_duration_off": slider_vad_FUSION_THRESHOLD
        }
        pyannote_vad_pipeline.instantiate(HYPER_PARAMETERS)
        slider_vad_pad = slider_vad_pad * 0.001
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 4:
        import torch
        torch.set_num_threads(parallel_threads)
        humaware_vad = torch.jit.load("./VAD/HumAware/HumAwareVAD.jit", map_location='cpu')
        humaware_vad = humaware_vad.float().eval()
        INPUT_AUDIO_LENGTH_B = 512
        stride_step_B = INPUT_AUDIO_LENGTH_B
        slider_vad_pad = slider_vad_pad * 0.001
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
    elif vad_type == 5:
        # Using CPU is fast enough.
        ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)
        device_type_B = 'cpu'
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")
        in_name_B = ort_session_B.get_inputs()
        out_name_B = ort_session_B.get_outputs()
        in_name_B0 = in_name_B[0].name
        out_name_B = [out_name_B[i].name for i in range(len(out_name_B))]
        slider_vad_SILENCE_SCORE = 1.0 - slider_vad_SILENCE_SCORE
        slider_vad_pad = slider_vad_pad * 0.001
    elif vad_type == 6:
        ten_vad = TenVad(256, 0.5, lib_path)  # TEN_VAD_FRAME_LENGTH = 256, standard threshold = 0.5
        INPUT_AUDIO_LENGTH_B = 256
        stride_step_B = INPUT_AUDIO_LENGTH_B
        slider_vad_pad = slider_vad_pad * 0.001
        print("\nVAD 可用的硬件 VAD Usable Providers: ['CPUExecutionProvider']")

    # Load ASR model
    ort_session_C, _, device_type_C, ORT_Accelerate_Providers_C, provider_options_C = create_ort_session(device_type, has_npu, onnx_model_C, ORT_Accelerate_Providers, session_opts, provider_options, 'ASR')
    provider_c = ort_session_C.get_providers()
    print(f'\nASR 可用的硬件 ASR-Usable Providers: {provider_c}')
    if (asr_type == 0) or (asr_type == 3) or (asr_type == 4):  # Whisper & FireRedASR & Dolphin
        if len(provider_c) < 2:
            onnx_model_D = onnx_model_D.replace('FP16', 'FP32')
            onnx_model_G = onnx_model_G.replace('FP16', 'FP32')
            onnx_model_H = onnx_model_H.replace('FP16', 'FP32')
            onnx_model_I = onnx_model_I.replace('FP16', 'FP32')
            onnx_model_J = onnx_model_J.replace('FP16', 'FP32')

        if slide_top_k_asr < slide_beam_size_asr:
            slide_top_k_asr = slide_beam_size_asr
        if (slide_top_k_asr < 2) or (slide_beam_size_asr < 2):
            USE_BEAM_SEARCH = False
            print('\n使用贪心搜索。Using Greedy Search.')
        else:
            USE_BEAM_SEARCH = True
            print('\n使用线束搜索。Using Beam Search.')
        if asr_type == 0:
            input_ids = np.array([[whisper_start_token, target_language_id, target_task_id]], dtype=np.int32)
            generate_limit = MAX_SEQ_LEN_ASR - 5  # 5 = length of initial input_ids
        elif asr_type == 3:
            ASR_STOP_TOKEN = [4]
            input_ids = np.array([[3]], dtype=np.int32)
            generate_limit = MAX_SEQ_LEN_ASR - 1  # 1 = length of initial input_ids
        else:
            ASR_STOP_TOKEN = [40000]
            input_ids = np.array([[39999]], dtype=np.int32)
            language_region = DOLPHIN_LANGUAGE_MAP.get(transcribe_language_dolphin, 'NONE').split('-')
            lang_id = f'<{language_region[0]}>'
            region_id = f'<{language_region[1]}>'
            if lang_id != '<auto>':
                detect_language = False
                lang_id = tokenizer.encode(lang_id)
            else:
                detect_language = True
            if not detect_language:
                if region_id != '<auto>':
                    detect_region = False
                    region_id = tokenizer.encode(region_id)
                else:
                    detect_region = True
            else:
                detect_region = True
            if len(provider_c) < 2:
                onnx_model_K = onnx_model_K.replace('FP16', 'FP32')
            generate_limit = MAX_SEQ_LEN_ASR - 6  # 6 = length of initial input_ids
            ort_session_K = onnxruntime.InferenceSession(onnx_model_K, sess_options=session_opts, providers=ORT_Accelerate_Providers_C, provider_options=provider_options_C)
            in_name_K = ort_session_K.get_inputs()[0].name
            out_name_K = [ort_session_K.get_outputs()[0].name]
            ids_len_2 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([2], dtype=np.int64), device_type_C, DEVICE_ID)
            ids_len_5 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([5], dtype=np.int64), device_type_C, DEVICE_ID)
            ids_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type_C, DEVICE_ID)
            ids_7 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([7], dtype=np.int64), device_type_C, DEVICE_ID)
            ids_145 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([145], dtype=np.int64), device_type_C, DEVICE_ID)
            ids_324 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([324], dtype=np.int64), device_type_C, DEVICE_ID)
            ids_39999 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([[39999]], dtype=np.int32), device_type_C, DEVICE_ID)  # int32
            ids_vocab_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([vocab_size], dtype=np.int64), device_type_C, DEVICE_ID)

            init_ids_len_2 = ids_len_2._ortvalue
            init_ids_len_5 = ids_len_5._ortvalue
            init_ids_0 = ids_0._ortvalue
            init_ids_7 = ids_7._ortvalue
            init_ids_145 = ids_145._ortvalue
            init_ids_324 = ids_324._ortvalue
            init_ids_39999 = ids_39999._ortvalue
            init_ids_vocab_size = ids_vocab_size._ortvalue

        ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers_C, provider_options=provider_options_C)
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        in_name_C0 = in_name_C[0].name
        out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]
        model_D_dtype = ort_session_D._inputs_meta[0].type
        if 'float16' in model_D_dtype:
            model_D_dtype = np.float16
        else:
            model_D_dtype = np.float32
        in_name_D = ort_session_D.get_inputs()
        out_name_D = ort_session_D.get_outputs()
        amount_of_outputs_D = len(out_name_D)
        in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
        out_name_D = [out_name_D[i].name for i in range(amount_of_outputs_D)]
        num_layers = (amount_of_outputs_D - 2) // 2
        num_keys_values = num_layers + num_layers
        num_keys_values_plus_1 = num_keys_values + 1
        num_keys_values_plus_2 = num_keys_values + 2
        num_keys_values_plus_3 = num_keys_values + 3
        num_keys_values2_plus_2 = num_keys_values_plus_2 + num_keys_values
        layer_indices = np.arange(num_keys_values_plus_2, num_keys_values_plus_2 + num_keys_values, dtype=np.int32)
        if asr_type != 4:
            vocab_size = ort_session_D._outputs_meta[num_keys_values].shape[1]
        attention_mask_D_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type_C, DEVICE_ID)
        attention_mask_D_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type_C, DEVICE_ID)
        history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type_C, DEVICE_ID)
        ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), device_type_C, DEVICE_ID)
        ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type_C, DEVICE_ID)
        input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type_C, DEVICE_ID)
        repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((slide_beam_size_asr, vocab_size), dtype=model_D_dtype), device_type_C, DEVICE_ID)
        penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(slide_repeat_penality_value_asr, dtype=model_D_dtype), device_type_C, DEVICE_ID)
        if slide_repeat_penality_value_asr != 1.0:
            DO_REPEAT_PENALITY = True
        else:
            DO_REPEAT_PENALITY = False
        if USE_BEAM_SEARCH:
            init_save_id_greedy = None
            penality_reset_count_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(slide_beam_size_asr, dtype=np.int32), device_type_C, DEVICE_ID)
            init_penality_reset_count_beam = penality_reset_count_beam._ortvalue
        else:
            init_penality_reset_count_beam = 0
            init_save_id_greedy = np.zeros(MAX_SEQ_LEN_ASR, dtype=np.int32)
        if asr_type != 0:
            if device_type_C != 'dml':
                past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_D._outputs_meta[0].shape[1], ort_session_D._outputs_meta[0].shape[2], 0), dtype=model_D_dtype), device_type_C, DEVICE_ID)
                past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_D._outputs_meta[num_layers].shape[1], 0, ort_session_D._outputs_meta[num_layers].shape[3]), dtype=model_D_dtype), device_type_C, DEVICE_ID)
            else:
                past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_D._outputs_meta[0].shape[1], ort_session_D._outputs_meta[0].shape[2], 0), dtype=model_D_dtype), 'cpu', DEVICE_ID)
                past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_D._outputs_meta[num_layers].shape[1], 0, ort_session_D._outputs_meta[num_layers].shape[3]), dtype=model_D_dtype), 'cpu', DEVICE_ID)
        else:
            if device_type_C != 'dml':
                past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_D._outputs_meta[0].shape[1],  0), dtype=model_D_dtype), device_type_C, DEVICE_ID)
                past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, ort_session_D._outputs_meta[num_layers].shape[2]), dtype=model_D_dtype), device_type_C, DEVICE_ID)
            else:
                past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_D._outputs_meta[0].shape[1], 0), dtype=model_D_dtype), 'cpu', DEVICE_ID)
                past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, ort_session_D._outputs_meta[num_layers].shape[2]), dtype=model_D_dtype), 'cpu', DEVICE_ID)
        batch_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type_C, DEVICE_ID)
        init_attention_mask_D_0 = attention_mask_D_0._ortvalue
        init_attention_mask_D_1 = attention_mask_D_1._ortvalue
        init_history_len = history_len._ortvalue
        init_ids_len = ids_len._ortvalue
        init_ids_len_1 = ids_len_1._ortvalue
        init_input_ids = input_ids._ortvalue
        init_repeat_penality = repeat_penality._ortvalue
        init_penality_value = penality_value._ortvalue
        init_past_keys_D = past_keys_D._ortvalue
        init_past_values_D = past_values_D._ortvalue
        init_batch_size = batch_size._ortvalue
        if USE_BEAM_SEARCH:
            topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([slide_top_k_asr], dtype=np.int64), device_type_C, DEVICE_ID)
            save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((slide_beam_size_asr, 0), dtype=np.int32), device_type_C, DEVICE_ID)
            beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([slide_beam_size_asr], dtype=np.int64), device_type_C, DEVICE_ID)
            init_topK = topK._ortvalue
            init_save_id_beam = save_id_beam._ortvalue
            init_beam_size = beam_size._ortvalue
            ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers_C, provider_options=provider_options_C)
            in_name_H = ort_session_H.get_inputs()
            out_name_H = ort_session_H.get_outputs()
            amount_of_outputs_H = len(out_name_H)
            amount_of_outputs_H_minus_1 = amount_of_outputs_H - 1
            amount_of_outputs_H_minus_2 = amount_of_outputs_H - 2
            in_name_H = [in_name_H[i].name for i in range(len(in_name_H))]
            out_name_H = [out_name_H[i].name for i in range(amount_of_outputs_H)]
            ort_session_I = onnxruntime.InferenceSession(onnx_model_I, sess_options=session_opts, providers=ORT_Accelerate_Providers_C, provider_options=provider_options_C)
            in_name_I = ort_session_I.get_inputs()
            out_name_I = ort_session_I.get_outputs()
            amount_of_outputs_I = len(out_name_I)
            amount_of_outputs_I_minus_1 = amount_of_outputs_I - 1
            in_name_I = [in_name_I[i].name for i in range(len(in_name_I))]
            out_name_I = [out_name_I[i].name for i in range(amount_of_outputs_I)]
            ort_session_J = onnxruntime.InferenceSession(onnx_model_J, sess_options=session_opts, providers=ORT_Accelerate_Providers_C, provider_options=provider_options_C)
            in_name_J = ort_session_J.get_inputs()
            out_name_J = ort_session_J.get_outputs()
            in_name_J = [in_name_J[i].name for i in range(len(in_name_J))]
            out_name_J = [out_name_J[i].name for i in range(len(out_name_J))]
            input_feed_H = {
                in_name_H[-2]: init_penality_value,
                in_name_H[-1]: init_beam_size
            }
            input_feed_I = {
                in_name_I[-3]: init_penality_value,
                in_name_I[-2]: init_beam_size,
                in_name_I[-1]: init_topK
            }
        else:
            init_topK = None
            init_save_id_beam = None
            init_beam_size = None
            ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers_C, provider_options=provider_options_C)
            in_name_G = ort_session_G.get_inputs()
            out_name_G = ort_session_G.get_outputs()
            in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
            out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]
            input_feed_G = {in_name_G[2]: init_penality_value}
    else:
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        in_name_C0 = in_name_C[0].name
        if asr_type == 1:  # SenseVoice
            in_name_C1 = in_name_C[1].name
            lang_idx = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([target_language_id], dtype=np.int32), device_type_C, DEVICE_ID)
            language_idx = lang_idx._ortvalue
        else:
            in_name_C1 = None
        out_name_C = [out_name_C[0].name]

    # Process Loop
    for input_audio in task_queue:
        print(f'\n加载音频文件 Loading the Input Media: {input_audio}')
        file_name = Path(input_audio).stem
        if USE_DENOISED:
            if switcher_denoiser_cache and Path(f'./Cache/{file_name}_{model_denoiser}.wav').exists():
                print('\n降噪音频文件已存在，改用缓存文件。The denoised audio file already exists. Using the cache instead.')
                USE_DENOISED = False
                HAS_CACHE = True
                audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE_16K).get_array_of_samples(), dtype=np.float32)
                audio = normalize_to_int16(audio).astype(np.float32)
                de_audio = np.array(AudioSegment.from_file(f'./Cache/{file_name}_{model_denoiser}.wav').set_channels(1).set_frame_rate(SAMPLE_RATE_16K).get_array_of_samples(), dtype=np.float32)
                min_len = min(audio.shape[-1], de_audio.shape[-1])
                audio = audio[:min_len] * slider_denoise_factor_minus + de_audio[:min_len] * slider_denoise_factor
                del de_audio
            else:
                audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE_48K).get_array_of_samples(), dtype=np.float32)
                if FIRST_RUN:
                    def setup_denoiser_session(onnx_model_A):
                        def try_create_session_with_config(providers, options, device_name, onnx_model_A, config_setup=None, config_cleanup=None):
                            try:
                                if config_setup:
                                    config_setup()
                                session = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=providers, provider_options=options)
                                if len(session.get_providers()) > 1:
                                    print(f'\nDenoiser: {device_name} 使用成功。{device_name} used successfully.')
                                    return session, True
                                else:
                                    print(f'\nDenoiser: {device_name} 使用失败。{device_name} usage failed.')
                                    return None, False
                            except:
                                print(f'\nDenoiser: {device_name} 使用失败。{device_name} usage failed.')
                                return None, False
                            finally:
                                if config_cleanup:
                                    config_cleanup()

                        def is_special_model():
                            return any(model in model_denoiser for model in['ZipEnhancer', 'MossFormerGAN_SE_16K', 'MossFormer2_SE_48K'])

                        # Load Denoiser model
                        ort_session_A = None
                        device_type_A = 'cpu'

                        if device_type == 'cpu':
                            # CPU-specific logic
                            if is_special_model():
                                if 'OpenVINOExecutionProvider' in usable_providers:
                                    # OpenVINO CPU configuration
                                    openvino_providers = ['OpenVINOExecutionProvider']
                                    openvino_options = [{
                                        'device_type': "CPU",
                                        'precision': 'ACCURACY',
                                        'model_priority': 'HIGH',
                                        'num_of_threads': parallel_threads,
                                        'num_streams': 1,
                                        'enable_opencl_throttling': False,
                                        'enable_qdq_optimizer': False,
                                        'disable_dynamic_shapes': True
                                    }]
                                    ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=openvino_providers, provider_options=openvino_options)

                                elif 'CoreMLExecutionProvider' in usable_providers:
                                    # CoreML CPU configuration
                                    def setup_coreml_cpu():
                                        provider_options[0]['MLComputeUnits'] = 'CPUOnly'

                                    def cleanup_coreml_cpu():
                                        provider_options[0]['MLComputeUnits'] = 'ALL'

                                    ort_session_A, _ = try_create_session_with_config(ORT_Accelerate_Providers, provider_options, 'Apple-CoreML-CPU', onnx_model_A, setup_coreml_cpu, cleanup_coreml_cpu)
                        else:
                            # Non-CPU logic
                            if 'OpenVINOExecutionProvider' in ORT_Accelerate_Providers[0]:
                                # OpenVINO GPU/NPU configuration
                                def setup_openvino():
                                    provider_options[0]['disable_dynamic_shapes'] = True

                                def cleanup_openvino():
                                    provider_options[0]['disable_dynamic_shapes'] = False
                                    provider_options[0]['device_type'] = 'GPU'  # Reset to GPU

                                setup_openvino()

                                # Try NPU first if available
                                if has_npu:
                                    provider_options[0]['device_type'] = 'NPU'
                                    ort_session_A, success = try_create_session_with_config(ORT_Accelerate_Providers, provider_options, 'OpenVINO-NPU', onnx_model_A)
                                    if success:
                                        device_type_A = 'npu'

                                # Try GPU if NPU failed or not available
                                if ort_session_A is None:
                                    provider_options[0]['device_type'] = 'GPU'
                                    ort_session_A, success = try_create_session_with_config(ORT_Accelerate_Providers, provider_options, 'OpenVINO-GPU', onnx_model_A)
                                    if success:
                                        device_type_A = device_type

                                cleanup_openvino()

                            elif 'CUDAExecutionProvider' in ORT_Accelerate_Providers:
                                # CUDA configuration
                                def setup_cuda():
                                    if is_special_model():
                                        provider_options[0]['cudnn_conv_algo_search'] = 'DEFAULT'

                                def cleanup_cuda():
                                    provider_options[0]['cudnn_conv_algo_search'] = 'EXHAUSTIVE'

                                ort_session_A, _ = try_create_session_with_config(ORT_Accelerate_Providers, provider_options, 'NVIDUA-GPU', onnx_model_A, setup_cuda, cleanup_cuda)

                            elif 'CoreMLExecutionProvider' in ORT_Accelerate_Providers[0]:
                                # CoreML GPU/NPU configuration
                                def setup_coreml_gpu():
                                    provider_options[0]['RequireStaticInputShapes'] = '1'

                                def cleanup_coreml_gpu():
                                    provider_options[0]['RequireStaticInputShapes'] = '0'

                                ort_session_A, _ = try_create_session_with_config(ORT_Accelerate_Providers, provider_options,'Apple-CoreML-GPU_NPU', onnx_model_A, setup_coreml_gpu, cleanup_coreml_gpu)

                            else:
                                # Other providers
                                ort_session_A, _ = try_create_session_with_config(ORT_Accelerate_Providers, provider_options, 'GPU_NPU', onnx_model_A)

                        # Fallback to CPU if all attempts failed
                        if ort_session_A is None:
                            onnx_model_A = onnx_model_A.replace('FP16', 'FP32')
                            ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'], provider_options=None)

                        # Setup input/output metadata
                        in_name_A = ort_session_A.get_inputs()
                        out_name_A = ort_session_A.get_outputs()
                        in_name_A0 = in_name_A[0].name
                        out_name_A0 = [out_name_A[0].name]
                        INPUT_AUDIO_LENGTH_A = ort_session_A._inputs_meta[0].shape[-1]
                        stride_step_A = INPUT_AUDIO_LENGTH_A

                        # Temporary fallback to 'cpu', due to the onnxruntime doesn't update yet.
                        if device_type_A in ['gpu', 'npu']:
                            device_type_A = 'cpu'

                        print(f'\n降噪可用的硬件 Denoise-Usable Providers: {ort_session_A.get_providers()}')

                        return ort_session_A, device_type_A, in_name_A0, out_name_A0, INPUT_AUDIO_LENGTH_A, stride_step_A
                    # Call the function
                    ort_session_A, device_type_A, in_name_A0, out_name_A0, INPUT_AUDIO_LENGTH_A, stride_step_A = setup_denoiser_session(onnx_model_A)
        else:
            audio = np.array(AudioSegment.from_file(input_audio).set_channels(1).set_frame_rate(SAMPLE_RATE_16K).get_array_of_samples(), dtype=np.float32)

        if FIRST_RUN:
            print(f'\n所有模型已成功加载。All Models have been successfully loaded.')
            print('----------------------------------------------------------------------------------------------------------')

        # Process audio
        audio = normalize_to_int16(audio)
        audio_len = audio.shape[-1]
        if switcher_run_test:
            audio_len = audio_len // 10
            audio = audio[:audio_len]
        audio = audio.reshape(1, 1, -1)
        inv_audio_len = 100.0 / audio_len
        if USE_DENOISED:
            if audio_len > INPUT_AUDIO_LENGTH_A:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH_A) / stride_step_A)) + 1
                total_length_needed = (num_windows - 1) * stride_step_A + INPUT_AUDIO_LENGTH_A
                pad_amount = total_length_needed - audio_len
                final_slice = audio[..., -pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, pad_amount))).astype(audio.dtype)
                audio = np.concatenate((audio, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH_A:
                audio_float = audio.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(1, 1, INPUT_AUDIO_LENGTH_A - audio_len))).astype(audio.dtype)
                audio = np.concatenate((audio, white_noise), axis=-1)
            aligned_len = audio.shape[-1]
            print('----------------------------------------------------------------------------------------------------------')
            print('\n对音频进行降噪。Denoising the audio.')
            args_list = []
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH_A
            start_time = time.time()
            while slice_end <= aligned_len:
                args_list.append((inv_audio_len, slice_start, slice_end, audio))
                slice_start += stride_step_A
                slice_end = slice_start + INPUT_AUDIO_LENGTH_A
            results = run_inference_x(inference_A, args_list, progress_prefix='Denoising')
            end_time = time.time()
            saved = [r[1][0].numpy() for r in results]
            de_audio = np.concatenate(saved, axis=-1)
            audio_len = audio_len // 3
            inv_audio_len = 100.0 / audio_len
            de_audio = de_audio[..., :audio_len]
            audio_len_3 = audio_len + audio_len + audio_len
            audio = np.mean(audio[..., :audio_len_3].reshape(-1, 3), axis=-1, dtype=np.float32)
            audio = audio * slider_denoise_factor_minus + de_audio.astype(np.float32) * slider_denoise_factor
            audio = normalize_to_int16(audio.clip(min=-32768.0, max=32767.0))
            sf.write(f'./Cache/{file_name}_{model_denoiser}.wav', de_audio.reshape(-1), SAMPLE_RATE_16K, format='WAVEX')
            print(f'Denoising: 100.00%\n降噪完成。Complete.\nTime Cost: {(end_time - start_time):.3f} Seconds.')
            del saved
            del results
            del de_audio

        # VAD parts.
        print('----------------------------------------------------------------------------------------------------------')
        print('\n接下来利用VAD模型提取语音片段。Next, use the VAD model to extract speech segments.')
        start_time = time.time()
        if vad_type != -1:
            if USE_DENOISED or HAS_CACHE:
                waveform = np.array(AudioSegment.from_file(f'./Cache/{file_name}_{model_denoiser}.wav').set_channels(1).set_frame_rate(SAMPLE_RATE_16K).get_array_of_samples(), dtype=np.float32)
                waveform = normalize_to_int16(waveform)
                waveform = waveform.reshape(1, 1, -1)
            else:
                waveform = audio
        if vad_type == 1:
            print('\nVAD-Faster_Whisper-Silero 不提供可视化的运行进度。\nThe VAD-Faster_Whisper-Silero does not provide the running progress for visualization.\n')
            vad_options = {
                'threshold': slider_vad_SPEAKING_SCORE,
                'neg_threshold': slider_vad_SILENCE_SCORE,
                'max_speech_duration_s': slider_vad_MAX_SPEECH_DURATION,
                'min_speech_duration_ms': slider_vad_MIN_SPEECH_DURATION_ms,
                'min_silence_duration_ms': slider_vad_MIN_SILENCE_DURATION,
                'speech_pad_ms': slider_vad_pad
            }
            timestamps = get_speech_timestamps_FW(
                (waveform.reshape(-1).astype(np.float32) * inv_int16),
                vad_options=VadOptions(**vad_options),
                sampling_rate=SAMPLE_RATE_16K
            )
            timestamps = [(item['start'] * inv_16k, item['end'] * inv_16k) for item in timestamps]
            del waveform
        elif vad_type == 2:
            print('\nVAD-Official-Silero 不提供可视化的运行进度。\nThe VAD-Official-Silero does not provide the running progress for visualization.\n')
            with torch.inference_mode():
                timestamps = get_speech_timestamps(
                    torch.from_numpy(waveform.reshape(-1).astype(np.float32) * inv_16k),
                    model=silero_vad,
                    threshold=slider_vad_SPEAKING_SCORE,
                    neg_threshold=slider_vad_SILENCE_SCORE,
                    max_speech_duration_s=slider_vad_MAX_SPEECH_DURATION,
                    min_speech_duration_ms=slider_vad_MIN_SPEECH_DURATION_ms,
                    min_silence_duration_ms=slider_vad_MIN_SILENCE_DURATION,
                    speech_pad_ms=slider_vad_pad,
                    return_seconds=True
                )
                timestamps = [(item['start'], item['end']) for item in timestamps]
                del waveform
        elif vad_type == 3:
            print('\nVAD-Pyannote_Segmentation 不提供可视化的运行进度。\nThe VAD-Pyannote_Segmentation does not provide the running progress for visualization.\n')
            with torch.inference_mode():
                timestamps = pyannote_vad_pipeline(f'./Cache/{file_name}_{model_denoiser}.wav')
                segments = list(timestamps._tracks.keys())
                total_seconds = audio_len * inv_16k
                timestamps = []
                for segment in segments:
                    segment_start = segment.start - slider_vad_pad
                    segment_end = segment.end + slider_vad_pad
                    if segment_start < 0:
                        segment_start = 0
                    if segment_end > total_seconds:
                        segment_end = total_seconds
                    timestamps.append((segment_start, segment_end))
                del waveform
        elif vad_type == 4:
            print('\nVAD-HumAware 不提供可视化的运行进度。\nThe VAD-HumAware does not provide the running progress for visualization.\n')
            with torch.inference_mode():
                waveform = torch.from_numpy(waveform.reshape(-1).astype(np.float32) * inv_16k)
                waveform_len = len(waveform)
                if waveform_len > INPUT_AUDIO_LENGTH_B:
                    num_windows = int(torch.ceil(torch.tensor((waveform_len - INPUT_AUDIO_LENGTH_B) / stride_step_B))) + 1
                    total_length_needed = (num_windows - 1) * stride_step_B + INPUT_AUDIO_LENGTH_B
                    pad_amount = total_length_needed - waveform_len
                    final_slice = waveform[-pad_amount:]
                    rms = torch.sqrt(torch.mean(final_slice * final_slice))
                    white_noise = rms * torch.randn(pad_amount, device=waveform.device, dtype=waveform.dtype)
                    waveform = torch.cat((waveform, white_noise), dim=-1)
                    waveform_len = len(waveform)
                elif waveform_len < INPUT_AUDIO_LENGTH_B:
                    rms = torch.sqrt(torch.mean(waveform * waveform))
                    white_noise = rms * torch.randn(INPUT_AUDIO_LENGTH_B - audio_len, device=waveform.device, dtype=waveform.dtype)
                    waveform = torch.cat((waveform, white_noise), dim=-1)
                    waveform_len = len(waveform)
                silence = True
                saved = []
                for i in range(0, waveform_len, INPUT_AUDIO_LENGTH_B):
                    score = humaware_vad(waveform[i: i + INPUT_AUDIO_LENGTH_B], SAMPLE_RATE_16K)
                    if silence:
                        if score >= slider_vad_SPEAKING_SCORE:
                            silence = False
                    else:
                        if score <= slider_vad_SILENCE_SCORE:
                            silence = True
                    saved.append(silence)
                timestamps = vad_to_timestamps(saved, HumAware_param, slider_vad_pad)
                del saved
                del waveform
                gc.collect()
        elif vad_type == 5:
            print('\nVAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet 不提供可视化的运行进度。\nThe VAD-NVIDIA_Frame_VAD_Multilingual_MarbleNet does not provide the running progress for visualization.\n')
            waveform = onnxruntime.OrtValue.ortvalue_from_numpy(waveform, device_type_B, DEVICE_ID)
            all_outpus_B = ort_session_B._sess.run_with_ort_values({in_name_B0: waveform._ortvalue}, out_name_B, run_options)
            score_silence = all_outpus_B[0].numpy()
            score_active = all_outpus_B[1].numpy()
            signal_len = all_outpus_B[2].numpy()
            silence = True
            saved = []
            for i in range(signal_len[0]):
                if silence:
                    if score_active[:, i] >= slider_vad_SPEAKING_SCORE:
                        silence = False
                else:
                    if score_silence[:, i] >= slider_vad_SILENCE_SCORE:
                        silence = True
                saved.append(silence)
            timestamps = vad_to_timestamps(saved, NVIDIA_VAD_param, slider_vad_pad)
            del saved
            del waveform
            del score_silence
            del score_active
            del signal_len
            gc.collect()
        elif vad_type == 6:
            waveform = waveform.reshape(-1)
            if audio_len > INPUT_AUDIO_LENGTH_B:
                num_windows = int(np.ceil((audio_len - INPUT_AUDIO_LENGTH_B) / stride_step_B)) + 1
                total_length_needed = (num_windows - 1) * stride_step_B + INPUT_AUDIO_LENGTH_B
                pad_amount = total_length_needed - audio_len
                final_slice = waveform[-pad_amount:].astype(np.float32)
                white_noise = (np.sqrt(np.mean(final_slice * final_slice, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(pad_amount))).astype(waveform.dtype)
                waveform = np.concatenate((waveform, white_noise), axis=-1)
            elif audio_len < INPUT_AUDIO_LENGTH_B:
                audio_float = waveform.astype(np.float32)
                white_noise = (np.sqrt(np.mean(audio_float * audio_float, dtype=np.float32), dtype=np.float32) * np.random.normal(loc=0.0, scale=1.0, size=(INPUT_AUDIO_LENGTH_B - audio_len))).astype(waveform.dtype)
                waveform = np.concatenate((waveform, white_noise), axis=-1)
            audio_len = waveform.shape[-1]
            inv_audio_len = 100.0 / audio_len
            silence = True
            saved = []
            slice_start = 0
            slice_end = INPUT_AUDIO_LENGTH_B
            while slice_end <= audio_len:
                score, _ = ten_vad.process(waveform[slice_start: slice_end])
                if silence:
                    if score >= slider_vad_SPEAKING_SCORE:
                        silence = False
                else:
                    if score <= slider_vad_SILENCE_SCORE:
                        silence = True
                saved.append(silence)
                print(f'VAD: {slice_start * inv_audio_len:.3f}%')
                slice_start += stride_step_B
                slice_end = slice_start + INPUT_AUDIO_LENGTH_B
            timestamps = vad_to_timestamps(saved, TEN_VAD_param, slider_vad_pad)
        else:
            print('\n这个任务不使用 VAD。This task does not use VAD.\n')
        if vad_type != -1:
            timestamps = process_timestamps(timestamps, slider_vad_FUSION_THRESHOLD, slider_vad_MIN_SPEECH_DURATION)
            print(f'VAD: 100.00%\n完成提取语音片段。Complete.\nTime Cost: {(time.time() - start_time):.3f} Seconds.')
        else:
            timestamps = [(0.0, audio_len * inv_16k)]
        print('----------------------------------------------------------------------------------------------------------')

        # -------------------------
        # ASR (transcription)
        # -------------------------
        print('\nStart to transcribe task.')
        start_time = time.time()
        # Build tasks for the chosen inference function per asr_type
        if asr_type == 0:
            args_list = [(start, end, inv_audio_len, audio, SAMPLE_RATE_16K, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, init_save_id_beam, init_repeat_penality, init_batch_size, init_penality_reset_count_beam, init_save_id_greedy, True) for start, end in timestamps]
            results = run_inference_x(inference_CD_beam_search, args_list, progress_prefix='ASR')
        elif asr_type == 1:
            args_list = [(start, end, inv_audio_len, audio, SAMPLE_RATE_16K, language_idx) for start, end in timestamps]
            results = run_inference_x(inference_C_sensevoice, args_list, progress_prefix='ASR')
        elif asr_type == 2:
            args_list = [(start, end, inv_audio_len, audio, SAMPLE_RATE_16K, is_english) for start, end in timestamps]
            results = run_inference_x(inference_C_paraformer, args_list, progress_prefix='ASR')
        elif asr_type == 3:
            args_list = [(start, end, inv_audio_len, audio, SAMPLE_RATE_16K, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, init_save_id_beam, init_repeat_penality, init_batch_size, init_penality_reset_count_beam, init_save_id_greedy, False) for start, end in timestamps]
            results = run_inference_x(inference_CD_beam_search, args_list, progress_prefix='ASR')
        elif asr_type == 4:
            args_list = [(start, end, inv_audio_len, audio, SAMPLE_RATE_16K, init_input_ids, init_history_len, init_ids_len, init_ids_len_1, init_ids_len_2, init_ids_0, init_ids_len_5, init_ids_7, init_ids_145, init_ids_324, init_ids_39999, init_ids_vocab_size, init_attention_mask_D_0, init_attention_mask_D_1, init_past_keys_D, init_past_values_D, init_save_id_beam, init_repeat_penality, init_batch_size, init_penality_reset_count_beam, init_save_id_greedy, lang_id, region_id) for start, end in timestamps]
            results = run_inference_x(inference_CD_dolphin, args_list, progress_prefix='ASR')
        else:
            results = []
        save_text = [result[1] for result in results]
        save_timestamps = [result[2] for result in results]
        print(f'ASR: 100.00%\nComplete. Time Cost: {time.time() - start_time:.3f} Seconds')
        del audio
        del timestamps
        gc.collect()
        print('----------------------------------------------------------------------------------------------------------')

        print(f'\n保存转录结果。Saving ASR Results.')
        with open(f'./Results/Timestamps/{file_name}.txt', 'w', encoding='UTF-8') as time_file, \
                open(f'./Results/Text/{file_name}.txt', 'w', encoding='UTF-8') as text_file, \
                open(f'./Results/Subtitles/{file_name}.vtt', 'w', encoding='UTF-8') as subtitles_file:

            subtitles_file.write('WEBVTT\n\n')
            idx = 0
            for text, t_stamp in zip(save_text, save_timestamps):
                text = text.replace('\n', '')
                if asr_type == 1:
                    parts = text.split('<|withitn|>')
                    transcription = ''
                    for i in range(1, len(parts)):
                        seg = parts[i]
                        # strip language tags if present
                        for tag in ('<|zh|>', '<|en|>', '<|yue|>', '<|ja|>', '<|ko|>'):
                            if tag in seg:
                                seg = seg.split(tag)[0]
                                break
                        transcription += seg
                else:
                    transcription = text

                start_sec = t_stamp[0]
                if t_stamp[1] - start_sec > 10.0:
                    markers = re.split(r'([。、，！？；：,.!?:;])', transcription)  # Keep markers in results
                    text_chunks = [''.join(markers[i:i + 2]) for i in range(0, len(markers), 2)]
                    time_per_chunk = (t_stamp[1] - start_sec) / len(text_chunks)
                    if len(text_chunks) > 3:
                        for i, chunk in enumerate(text_chunks):
                            chunk_start = start_sec + i * time_per_chunk
                            chunk_end = chunk_start + time_per_chunk
                            chunk = chunk.replace(';', '')
                            if chunk and chunk != '。' and chunk != '.':
                                start_time = format_time(chunk_start)
                                end_time = format_time(chunk_end)
                                timestamp = f'{start_time} --> {end_time}\n'
                                time_file.write(timestamp)
                                text_file.write(f'{chunk}\n')
                                subtitles_file.write(f'{idx}\n{timestamp}{chunk}\n\n')
                                idx += 1
                    else:
                        transcription = transcription.replace(';', '')
                        if transcription and transcription != '。' and transcription != '.':
                            start_time = format_time(start_sec)
                            end_time = format_time(t_stamp[1])
                            timestamp = f'{start_time} --> {end_time}\n'
                            time_file.write(timestamp)
                            text_file.write(f'{transcription}\n')
                            subtitles_file.write(f'{idx}\n{timestamp}{transcription}\n\n')
                            idx += 1
                else:
                    transcription = transcription.replace(';', '')
                    if transcription and transcription != '。' and transcription != '.':
                        start_time = format_time(start_sec)
                        end_time = format_time(t_stamp[1])
                        timestamp = f'{start_time} --> {end_time}\n'
                        time_file.write(timestamp)
                        text_file.write(f'{transcription}\n')
                        subtitles_file.write(f'{idx}\n{timestamp}{transcription}\n\n')
                        idx += 1
            del save_text
            del save_timestamps
        print(f'\n转录任务完成。Transcribe Tasks Complete.\n\n原文字幕保存在文件夹: ./Result/Subtitles\nThe original subtitles are saved in the folder: ./Result/Subtitles\n\nTranscribe Time: {(time.time() - total_process_time):.3f} Seconds.')
        print('----------------------------------------------------------------------------------------------------------')

        if 'Translate' not in task:
            continue
        else:
            print('\n开始 LLM 翻译任务。Start to LLM Translate.')
            start_time = time.time()
            if FIRST_RUN:
                print('\n加载 LLM 模型。Loading the LLM model.')
                if model_llm == 'Whisper':
                    print('\n翻译任务完成。Translate tasks completed.')
                    continue
                else:
                    llm_path = f'./LLM/{model_llm}'

                if any(tag in model_llm for tag in ('3B', '4B')):
                    MAX_TRANSLATE_LINES = 8
                    TRANSLATE_OVERLAP = 2
                elif any(tag in model_llm for tag in ('7B', '8B', '9B')):
                    MAX_TRANSLATE_LINES = 16
                    TRANSLATE_OVERLAP = 4
                elif any(tag in model_llm for tag in ('12B', '13B', '14B')):
                    MAX_TRANSLATE_LINES = 32
                    TRANSLATE_OVERLAP = 6
                else:
                    MAX_TRANSLATE_LINES = 4
                    TRANSLATE_OVERLAP = 1
                MAX_TOKENS_PER_CHUNK = MAX_TRANSLATE_LINES * MAX_SEQ_LEN_LLM

                is_seed_x = False
                if 'Qwen' in model_llm:
                    system_prompt = (
                        '### INSTRUCTIONS ###\n'
                        f'1. Translate the provided text from {transcribe_language} to {translate_language}.\n'
                        '2. The input format is: ID-TEXT\n'
                        '3. Your output format MUST be exactly: ID-TRANSLATION\n'
                        '4. Do not add any extra content, commentary, explanations, chain of thought.\n'
                        f'5. {llm_prompt}\n'
                    )
                    LLM_STOP_TOKEN = [151643, 151645]
                    prompt_head = f'<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n'
                    prompt_tail = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
                elif 'Hunyuan-MT' in model_llm:
                    LLM_STOP_TOKEN = [127960]
                    en_target_language, zh_target_language = get_language_hunyuan(translate_language)
                    if (translate_language == 'Chinese') or (transcribe_language == 'Chinese'):
                        prompt_head = f'<|startoftext|>将下面的文本翻译成{zh_target_language}，不要额外解释。\n\n'
                    else:
                        prompt_head = f'<|startoftext|>Translate the following segment into {en_target_language}, without additional explanation.\n\n'
                    prompt_tail = '<|extra_0|>'
                elif 'Seed-X' in model_llm:
                    LLM_STOP_TOKEN = [2]
                    abbr = get_language_seedx(translate_language)
                    prompt_head = f'Translate the following {transcribe_language} sentence into {translate_language}:\n'
                    prompt_tail = f' {abbr} <s> \n'
                    is_seed_x = True
                else:
                    error = '\n未找到指定的 LLM 模型。The specified LLM model was not found.'
                    print(error)
                    return error
                tokenizer_llm = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True, use_fast=True)
                if not is_seed_x:
                    prompt_head = tokenizer_llm(prompt_head, return_tensors='np')['input_ids'].astype(np.int32)
                    prompt_tail = tokenizer_llm(prompt_tail, return_tensors='np')['input_ids'].astype(np.int32)

                # Load the LLM
                ort_session_E, use_sync_operations_E, device_type, _, _ = create_ort_session(device_type, False, llm_path + '/llm.onnx', ORT_Accelerate_Providers, session_opts, provider_options, 'LLM')
                print(f"\nLLM 可用的硬件 LLM-Usable Providers: {ort_session_E.get_providers()}")
                device_type_ort = get_ort_device(device_type, DEVICE_ID)
                io_binding_E = ort_session_E.io_binding()._iobinding
                in_name_E = ort_session_E.get_inputs()
                output_metas_E = ort_session_E.get_outputs()
                amount_of_inputs_E = len(in_name_E)
                amount_of_outputs_E = len(output_metas_E)
                in_name_E = [in_name_E[i].name for i in range(amount_of_inputs_E)]
                out_name_E = [output_metas_E[i].name for i in range(amount_of_outputs_E)]
                num_layers = (amount_of_outputs_E - 2) // 2
                num_keys_values = num_layers + num_layers
                init_attention_mask_E_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
                init_attention_mask_E_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
                init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
                init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
                if device_type != 'dml':
                    init_past_keys_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas_E[0].shape[0], 1, output_metas_E[0].shape[2], 0), dtype=np.float32), device_type, DEVICE_ID)
                    init_past_values_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas_E[num_layers].shape[0], 1, 0, output_metas_E[num_layers].shape[-1]), dtype=np.float32), device_type, DEVICE_ID)
                else:
                    init_past_keys_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas_E[0].shape[0], 1, output_metas_E[0].shape[2], 0), dtype=np.float32), 'cpu', DEVICE_ID)
                    init_past_values_E = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((output_metas_E[num_layers].shape[0], 1, 0, output_metas_E[num_layers].shape[-1]), dtype=np.float32), 'cpu', DEVICE_ID)

                # Do not use X = X._ortvalue, it will error out. Must create a new one.
                init_past_keys_E_ort = init_past_keys_E._ortvalue
                init_past_values_E_ort = init_past_values_E._ortvalue
                init_attention_mask_E_0_ort = init_attention_mask_E_0._ortvalue
                init_ids_len_1_ort = init_ids_len_1._ortvalue

                # Initial input binding setup
                init_input_feed_E = [None] * amount_of_inputs_E
                init_input_feed_E[-3] = init_history_len._ortvalue
                init_input_feed_E[-1] = init_attention_mask_E_1._ortvalue
                for i in range(num_layers):
                    init_input_feed_E[i] = init_past_keys_E_ort
                for i in range(num_layers, num_keys_values):
                    init_input_feed_E[i] = init_past_values_E_ort
                print('\nLLM 模型加载完成。LLM loading completed')
                FIRST_RUN = False

            with open(f'./Results/Text/{file_name}.txt', 'r', encoding='utf-8') as asr_file:
                asr_lines = asr_file.readlines()

            with open(f'./Results/Timestamps/{file_name}.txt', 'r', encoding='utf-8') as timestamp_file:
                timestamp_lines = timestamp_file.readlines()

            for line_index in range(len(asr_lines)):
                asr_lines[line_index] = f'{line_index}-{asr_lines[line_index].strip()}\n'

            total_lines = len(asr_lines)
            if total_lines < 1:
                print('\n翻译内容为空。Empty content for translation task.')
                continue

            print('----------------------------------------------------------------------------------------------------------')
            inv_total_lines = 100.0 / total_lines
            step_size = MAX_TRANSLATE_LINES - TRANSLATE_OVERLAP
            translated_responses = []
            for chunk_start in range(0, total_lines, step_size):
                chunk_end = min(total_lines, chunk_start + MAX_TRANSLATE_LINES)
                translation_prompt = (''.join(asr_lines[chunk_start: chunk_end]))
                print('\n' + translation_prompt)
                if is_seed_x:
                    input_ids = tokenizer_llm(prompt_head + translation_prompt + prompt_tail, return_tensors='np')['input_ids'].astype(np.int32)
                else:
                    input_ids = np.concatenate((prompt_head, tokenizer_llm(translation_prompt, return_tensors='np')['input_ids'].astype(np.int32), prompt_tail), axis=1)
                ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([input_ids.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
                input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device_type, DEVICE_ID)
                init_input_feed_E[-4] = input_ids._ortvalue
                init_input_feed_E[-2] = ids_len._ortvalue
                num_decode = 0
                save_text = ''
                bind_inputs_to_device(io_binding_E, in_name_E, init_input_feed_E, amount_of_inputs_E)
                start_time = time.time()
                while num_decode < MAX_TOKENS_PER_CHUNK:
                    bind_outputs_to_device(io_binding_E, out_name_E, device_type_ort, amount_of_outputs_E)
                    if use_sync_operations_E:
                        io_binding_E.synchronize_inputs()
                        ort_session_E._sess.run_with_iobinding(io_binding_E, run_options)
                        io_binding_E.synchronize_outputs()
                    else:
                        ort_session_E._sess.run_with_iobinding(io_binding_E, run_options)
                    all_outputs_E = io_binding_E.get_outputs()
                    max_logit_ids = all_outputs_E[num_keys_values].numpy()[0, 0]
                    num_decode += 1
                    if max_logit_ids in LLM_STOP_TOKEN:
                        break
                    if num_decode < 2:
                        io_binding_E.bind_ortvalue_input(in_name_E[-1], init_attention_mask_E_0_ort)
                        io_binding_E.bind_ortvalue_input(in_name_E[-2], init_ids_len_1_ort)
                    bind_inputs_to_device(io_binding_E, in_name_E, all_outputs_E, amount_of_outputs_E)
                    text = tokenizer_llm.decode(max_logit_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    save_text += text
                    print(text, end='', flush=True)
                print(f'\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s')
                translated_responses.append(save_text)
                print(f'Translating: - {chunk_end * inv_total_lines:.3f}%')
                print('----------------------------------------------------------------------------------------------------------')
                if chunk_end == total_lines:
                    break
            merged_responses = '\n'.join(translated_responses).split('\n')
            with open(f'./Results/Subtitles/{file_name}_translated.vtt', 'w', encoding='UTF-8') as subtitles_file:
                subtitles_file.write('WEBVTT\n\n')
                timestamp_len = len(timestamp_lines)
                save_line_index = []
                for i in range(len(merged_responses)):
                    response_line = merged_responses[i]
                    if response_line:
                        if is_seed_x:
                            dot_split = False
                            for j in range(10):
                                if f'{j}. ' in response_line[:4]:
                                    dot_split = True
                                    break
                            if dot_split:
                                parts = response_line.split('. ')
                            else:
                                parts = response_line.split('-')
                        else:
                            parts = response_line.split('-')
                        if len(parts) > 1:
                            line_index = parts[0]
                            if line_index.isdigit():
                                line_index = int(line_index)
                                if line_index in save_line_index:
                                    continue
                                if line_index < timestamp_len:
                                    text = ''.join(parts[1:])
                                    subtitles_file.write(f'{line_index}\n{timestamp_lines[line_index]}{text.strip()}\n\n')
                                    save_line_index.append(line_index)
            print(f'\n翻译完成。LLM Translate Complete.\nTime Cost: {time.time() - start_time:.3f} Seconds')
            print('----------------------------------------------------------------------------------------------------------')
    success = f'\n所有任务已完成。翻译字幕保存在文件夹: ./Result/Subtitles\nAll tasks complete. The translated subtitles are saved in the folder: ./Result/Subtitles\n\nTotal Time: {(time.time() - total_process_time):.3f} Seconds.\n'
    print(success)
    return success


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
CUSTOM_CSS = """
/* ===== Base Styling ======================================================= */
html, body, .gradio-container {
    background: #0d0d0d;
    color: #f4f4f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 22px; /* increased from 18px */
    line-height: 1.4;
}

h1, h2, h3, h4, h5, h6, .markdown {
    color: #f4f4f4;
}

/* Make general markdown/body text larger */
.markdown {
    font-size: 22px; /* base markdown text bigger */
}

/* ===== Input Elements ===================================================== */
input, textarea, select, .input-container {
    background: #111;
    border: 1px solid #333;
    color: #f4f4f4;
    border-radius: 6px;
    font-size: 22px; /* increased from 18px */
}

label {
    color: #1e90ff !important;
    font-size: 24px !important; /* increased from 20px */
    font-weight: 600 !important;
}

.dropdown, .slider, .checkbox, .radio {
    background: #111;
    border: 1px solid #333;
    border-radius: 6px;
    font-size: 22px; /* ensure control text is larger */
}

/* Optional: make slider tooltip/ticks a bit larger */
.slider .noUi-tooltip,
.slider .noUi-value {
    font-size: 18px;
}

.slider .noUi-connect {
    background: #1e90ff;
}

.slider .noUi-handle {
    background: #0d8bfd;
    border: 1px solid #80d0ff;
}

/* ===== Button Styling ===================================================== */
button, .button-primary {
    font-size: 24px; /* increased from 20px */
    font-weight: 700;
    background: linear-gradient(90deg, #0d8bfd 0%, #31d2ff 100%);
    border: none;
    color: #0d0d0d;
    box-shadow: 0 0 12px #0d8bfd, 0 0 6px #31d2ff inset;
    transition: all 0.15s;
}

button:hover {
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 0 20px #31d2ff;
}

/* ===== Page Title ========================================================= */
.big-title {
    font-size: 52px; /* increased from 40px */
    font-weight: 900;
    text-align: center;
    margin: 16px 0 30px 0;
    background: linear-gradient(90deg, #31d2ff 0%, #ffffff 50%, #31d2ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 20px rgba(49, 210, 255, 0.5);
}

/* ===== Section Blocks ===================================================== */
.section {
    padding: 18px 18px 12px 18px;
    border-radius: 14px;
    margin-bottom: 20px;
    border: 1px solid transparent;
    box-shadow: 0 0 8px rgba(0, 0, 0, 0.35);
}

.section-title {
    font-size: 32px; /* increased from 26px */
    font-weight: 800;
    margin: 0 0 16px 0;
}

/* Section Color Themes */
.section-sys   { background: rgba(30, 144, 255, 0.12); border-color: #1e90ff55; }
.section-model { background: rgba(255, 165, 0, 0.12);  border-color: #ffa50055; }
.section-tgt   { background: rgba(50, 205, 50, 0.12);  border-color: #32cd3255; }
.section-aud   { background: rgba(255, 20, 147, 0.12); border-color: #ff149355; }
.section-vad   { background: rgba(75, 0, 130, 0.12);   border-color: #4b008255; }

/* ===== Special Text Areas ================================================= */
.task-state textarea {
    background: #000 !important;
    color: #00c853 !important;
    font-family: Consolas, monospace;
    font-size: 24px; /* increased from 20px */
}

.llm-prompt textarea {
    background: #000 !important;
    color: #fce303 !important;
    font-family: Consolas, monospace;
    font-size: 24px; /* increased from 20px */
}

.media-path textarea {
    background: #000 !important;
    color: #03d3fc !important;
    font-family: Consolas, monospace;
    font-size: 24px; /* increased from 20px */
}

/* Larger placeholder text for inputs */
input::placeholder,
textarea::placeholder {
    font-size: 20px;
}
"""

# ============================================================================
# MAIN GRADIO INTERFACE
# ============================================================================


def create_interface():
    """Create and configure the main Gradio interface."""

    with gr.Blocks(css=CUSTOM_CSS, title='Subtitles is All You Need') as GUI:
        # ====================================================================
        # HEADER SECTION
        # ====================================================================
        gr.Markdown("<div class='big-title'>Subtitles is All You Need</div>")

        # Main controls row
        with gr.Row():
            # Logo/Image column
            with gr.Column(scale=1):
                gr.Image(
                    r'./Icon/psyduck.jpg',
                    type='filepath',
                    show_download_button=False,
                    show_fullscreen_button=False
                )

            # Primary controls column
            with gr.Column(scale=6):
                # Task selection and test mode
                with gr.Row():
                    task = gr.Dropdown(
                        choices=TASK_LIST,
                        label='任务 / Task',
                        info='选择操作\nSelect an operation for the audio.',
                        value=TASK_LIST[1],
                        interactive=True,
                    )
                    switcher_run_test = gr.Checkbox(
                        label='简短测试 / Run Test',
                        info='对音频的 10% 时长运行简短测试。\nRun a short test on 10% of the audio length.',
                        value=False,
                        interactive=True,
                    )

                # File path input
                file_path_input = gr.Textbox(
                    label='视频 / 音频文件路径 Video / Audio File Path',
                    info='输入要转录的视频/音频文件或文件夹的路径。\nEnter the path of the video / audio file or folder you want to transcribe.',
                    value='./Media',
                    interactive=True,
                    elem_classes='media-path'
                )

        # ====================================================================
        # SYSTEM SETTINGS SECTION
        # ====================================================================
        with gr.Column(elem_classes=['section', 'section-sys']):
            gr.Markdown("<div class='section-title'>🖥️  系统设置 / System Settings</div>")

            with gr.Row():
                parallel_threads = gr.Slider(
                    minimum=1,
                    maximum=72,
                    step=1,
                    label='并行处理 / Parallel Threads',
                    info='用于并行处理的 CPU 内核数。\nNumber of CPU cores.',
                    value=physical_cores,
                    interactive=True
                )
                hardware = gr.Dropdown(
                    choices=HARDWARE_LIST,
                    label='硬件设备 / Hardware Device',
                    info='选择用于运行任务的设备。\nSelect the device for running the task.',
                    value=HARDWARE_LIST[0],
                    visible=True,
                    interactive=True
                )

        # ====================================================================
        # MODEL SELECTION SECTION
        # ====================================================================
        with gr.Column(elem_classes=['section', 'section-model']):
            gr.Markdown("<div class='section-title'>🧠  模型选择 / Model Selection</div>")

            # Model selection row
            with gr.Row():
                model_llm = gr.Dropdown(
                    choices=LLM_LIST,
                    label='大型语言模型 / LLM Model',
                    info='用于翻译的模型。\nModel used for translation.',
                    value=LLM_LIST[3],
                    visible=True,
                    interactive=True
                )
                model_asr = gr.Dropdown(
                    choices=ASR_LIST,
                    label='ASR模型 / ASR Model',
                    info='用于转录的模型。\nModel used for transcription.',
                    value=ASR_LIST[0],
                    visible=True,
                    interactive=True
                )

            # LLM prompt configuration
            llm_prompt = gr.Textbox(
                label='指导翻译任务 / Translate Prompt',
                value='Using full context, write a fluent, idiomatic translation that reads native, not literal. Keep the original person name and do not translate it.',
                interactive=True,
                elem_classes='llm-prompt',
                visible=False
            )

            # Advanced ASR settings (initially hidden)
            slide_top_k_asr = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                label='ASR解码候选 Top_K / ASR Decode Top_K Candidate',
                info='在前K个候选中进行解码，设定1则是贪心搜索。\nHigher = More decode options; Setting 1 is greedy search.',
                value=3,
                visible=False,
                interactive=True
            )

            slide_beam_size_asr = gr.Slider(
                minimum=1,
                maximum=10,
                step=1,
                label='ASR线束搜索量 / ASR Beam Search Size',
                info='多线路并行解码，返回总分最高的那条。设定1则是贪心搜索。\nDecode multiple beams in parallel and return the one with the highest score. Setting 1 is greedy search.',
                value=3,
                visible=False,
                interactive=True
            )

            slide_repeat_penality_value_asr = gr.Slider(
                minimum=0.7,
                maximum=1.0,
                step=0.025,
                label='ASR重复处罚 / ASR Repeat Penalty',
                info='防止重复输出解码，设定1则无处罚。\nPrevent duplicate decoding. Setting 1 is disable.',
                value=0.95,
                visible=False,
                interactive=True
            )

        # ====================================================================
        # TARGET LANGUAGE SECTION
        # ====================================================================
        with gr.Column(elem_classes=['section', 'section-tgt']):
            gr.Markdown("<div class='section-title'>🌐  目标语言 / Target Language</div>")

            with gr.Row():
                transcribe_language = gr.Dropdown(
                    choices=SENSEVOICE_LANGUAGE_LIST,
                    label='转录语言 / Transcription Language',
                    info='源媒体的语言。\nLanguage of the input media.',
                    value=SENSEVOICE_LANGUAGE_LIST[4],
                    visible=True,
                    interactive=True
                )
                translate_language = gr.Dropdown(
                    choices=WHISPER_LANGUAGE_LIST,
                    label='翻译语言 / Translation Language',
                    info='要翻译成的语言。\nLanguage to translate into.',
                    value=WHISPER_LANGUAGE_LIST[0],
                    visible=True,
                    interactive=True
                )

        # ====================================================================
        # AUDIO PROCESSING SECTION
        # ====================================================================
        with gr.Column(elem_classes=['section', 'section-aud']):
            gr.Markdown("<div class='section-title'>🎙️  音频处理 / Audio Processor</div>")

            with gr.Row():
                model_denoiser = gr.Dropdown(
                    choices=DENOISER_LIST,
                    label='降噪器 / Denoiser',
                    info='选择用于增强人声的降噪器。\nChoose a denoiser for audio processing.',
                    value=DENOISER_LIST[3],
                    visible=True,
                    interactive=True
                )
                slider_denoise_factor = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    step=0.05,
                    label='降噪系数 / Denoise Factor',
                    info='较大的值可增强降噪效果。\nHigher = stronger denoise.',
                    value=0.7,
                    visible=True,
                    interactive=True
                )
                switcher_denoiser_cache = gr.Checkbox(
                    label='使用缓存 / Use Cache',
                    info='使用以前的降噪结果以节省时间。\nUse previous results.',
                    value=True,
                    visible=True,
                    interactive=True
                )

        # ====================================================================
        # VAD CONFIGURATION SECTION
        # ====================================================================
        with gr.Column(elem_classes=['section', 'section-vad']):
            gr.Markdown("<div class='section-title'>🔊  VAD 配置 / VAD Configurations</div>")

            # Main VAD model selection
            model_vad = gr.Dropdown(
                choices=VAD_LIST,
                label='语音活动检测 / Voice Activity Detection - VAD',
                info='选择用于音频处理的 VAD。\nSelect the VAD used for audio processing.',
                value=VAD_LIST[2],
                visible=True,
                interactive=True
            )

            # VAD padding setting
            slider_vad_pad = gr.Slider(
                minimum=0,
                maximum=1000,
                step=25,
                label='VAD 填充 / VAD Padding',
                info='在时间戳的开头和结尾添加填充。单位：毫秒。\nAdd padding to the start and end of the timestamps. Unit: Milliseconds.',
                value=400,
                visible=True,
                interactive=True
            )

            # Voice detection scores
            with gr.Row():
                slider_vad_SPEAKING_SCORE = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.025,
                    label='语音状态分数 / Voice State Score',
                    info='值越大，判断语音状态越困难。\nThe higher the value, the more difficult it is to determine the state of the speech',
                    value=0.4,
                    visible=True,
                    interactive=True
                )
                slider_vad_SILENCE_SCORE = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.025,
                    label='静音状态分数 / Silence State Score',
                    info='值越大，越容易截断语音。\nA larger value makes it easier to cut off speaking.',
                    value=0.4,
                    visible=True,
                    interactive=True
                )

            # Duration and timing controls
            with gr.Row():
                slider_vad_FUSION_THRESHOLD = gr.Slider(
                    minimum=0,
                    maximum=5,
                    step=0.025,
                    label='合并时间戳 / Merge Timestamps',
                    info='如果两个语音段间隔太近，它们会被合并成一个。 单位：秒。\nIf two voice segments are too close, they will be merged into one. Unit: Seconds.',
                    value=0.2,
                    visible=True,
                    interactive=True
                )
                slider_vad_MIN_SPEECH_DURATION = gr.Slider(
                    minimum=0,
                    maximum=2,
                    step=0.025,
                    label='过滤短语音段 / Filter Short Voice Segment',
                    info='最短语音时长。单位：秒。\nMinimum duration for voice filtering. Unit: Seconds.',
                    value=0.05,
                    visible=True,
                    interactive=True
                )
                slider_vad_MAX_SPEECH_DURATION = gr.Slider(
                    minimum=1,
                    maximum=20,
                    step=1,
                    label='过滤长语音段 / Filter Long Voice Segment',
                    info='最大语音时长。单位：秒。\nMaximum voice duration. Unit: Seconds.',
                    value=20,
                    visible=True,
                    interactive=True
                )
                slider_vad_MIN_SILENCE_DURATION = gr.Slider(
                    minimum=100,
                    maximum=3000,
                    step=25,
                    label='静音时长判断 / Silence Duration Judgment',
                    info='最短静音时长。单位：毫秒。\nMinimum silence duration. Unit: Milliseconds.',
                    value=1000,
                    visible=True,
                    interactive=True
                )

        # ====================================================================
        # STATUS AND EXECUTION SECTION
        # ====================================================================

        # Task status display
        task_state = gr.Textbox(
            label='任务状态 / Task State',
            value='点击运行任务并稍等片刻。/ Click the Run button and wait a moment.',
            interactive=False,
            elem_classes='task-state'
        )

        # Main execution button
        submit_button = gr.Button(
            '🚀  运行任务  |  Run Task',
            variant='primary'
        )

        # ====================================================================
        # EVENT HANDLERS
        # ====================================================================

        # Main task execution
        submit_button.click(
            fn=MAIN_PROCESS,
            inputs=[
                task,
                hardware,
                parallel_threads,
                file_path_input,
                translate_language,
                transcribe_language,
                slide_top_k_asr,
                slide_beam_size_asr,
                slide_repeat_penality_value_asr,
                model_asr,
                model_vad,
                model_denoiser,
                model_llm,
                llm_prompt,
                switcher_run_test,
                switcher_denoiser_cache,
                slider_vad_pad,
                slider_denoise_factor,
                slider_vad_SPEAKING_SCORE,
                slider_vad_SILENCE_SCORE,
                slider_vad_FUSION_THRESHOLD,
                slider_vad_MIN_SPEECH_DURATION,
                slider_vad_MAX_SPEECH_DURATION,
                slider_vad_MIN_SILENCE_DURATION
            ],
            outputs=task_state
        )

        # Dynamic UI updates based on user selections
        task.change(
            fn=update_task,
            inputs=task,
            outputs=[model_llm, translate_language]
        )

        model_llm.change(
            fn=update_translate_language,
            inputs=model_llm,
            outputs=[translate_language, model_asr, llm_prompt]
        )

        model_asr.change(
            fn=update_transcribe_language,
            inputs=model_asr,
            outputs=[transcribe_language, slide_top_k_asr, slide_beam_size_asr, slide_repeat_penality_value_asr]
        )

        model_denoiser.change(
            fn=update_denoiser,
            inputs=model_denoiser,
            outputs=[switcher_denoiser_cache, slider_denoise_factor]
        )

        model_vad.change(
            fn=update_vad,
            inputs=model_vad,
            outputs=[
                slider_vad_SPEAKING_SCORE,
                slider_vad_SILENCE_SCORE,
                slider_vad_MAX_SPEECH_DURATION,
                slider_vad_MIN_SILENCE_DURATION,
                slider_vad_FUSION_THRESHOLD,
                slider_vad_MIN_SPEECH_DURATION,
                slider_vad_pad
            ]
        )

    return GUI


# Launch the app
if __name__ == "__main__":
    inv_16k = 1.0 / 16000.0
    inv_int16 = 1.0 / 32768.0
    HumAware_param = 512.0 * inv_16k
    NVIDIA_VAD_param = 320.0 * inv_16k
    TEN_VAD_param = 256 * inv_16k

    GUI = create_interface()
    GUI.launch()
    
