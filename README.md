<div align="center">

# 🎙️ Automatic Speech Recognition · ONNX

**SOTA speech-recognition families. One ONNX Runtime pipeline. Audio in → text out.**  
**顶尖语音识别家族，一套 ONNX Runtime 流程 —— 音频进，文字出。**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-powered-005CED?logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![Stars](https://img.shields.io/github/stars/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX?style=social)](https://github.com/DakeQQ/Automatic-Speech-Recognition-ASR-ONNX)

</div>

---

## ✨ Highlights · 亮点

- 🔌 **True end-to-end** — Raw audio in, text out; `STFT` / Kaldi-FBank is baked into the graph, no external feature extractor. 
- 🔌 **真·端到端** — 音频进、文字出，`STFT` / FBank 已内置计算图。



## 🧩 Supported Models · 支持模型

**🗣️ Single models · 单模型**

| **Model · 模型** | **Variants & Links · 变体与链接** | **Highlights · 亮点** |
|:---|:---|:---|
| **ARK-ASR** | [Hugging Face](https://huggingface.co/Audio8/ARK-ASR-0.6B) | - 0.6B multilingual autoregressive ASR <br> - 0.6B 多语种自回归语音识别 |
| **Audio8-ASR** | [Hugging Face](https://huggingface.co/Audio8/Audio8-ASR-0.1B) | - Automatic recognition: Chinese, English, French, German, Japanese, Korean & Cantonese <br> - 自动识别中文、英语、法语、德语、日语、韩语和粤语 |
| **Dolphin** | [GitHub](https://github.com/DataoceanAI/Dolphin/tree/main) | - V1 · CN-Dialect 方言 · Streaming 流式 |
| **FireRedASR** | [GitHub](https://github.com/FireRedTeam/FireRedASR) · [GitHub 2S](https://github.com/FireRedTeam/FireRedASR2S) | - Attention encoder–decoder <br> - 注意力编码器-解码器 |
| **Fun-ASR-Nano-2512** | [ModelScope](https://www.modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512) | - CTC fast head + AR decoder <br> - CTC 快速头 + 自回归解码 |
| **Mega-ASR** | [GitHub](https://github.com/xzf-thu/Mega-ASR) | - Routes between Qwen3-ASR Base and Mega paths for degraded audio <br> - 根据音频质量在 Qwen3-ASR Base 和 Mega 路径间路由 |
| **Nemotron** | [ModelScope](https://modelscope.cn/models/nv-community/nemotron-3.5-asr-streaming-0.6b) | - Streaming Multilingual <br> - 流式多语种 |
| **Paraformer** | [ModelScope Small-ZH](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) · [ModelScope Large-ZH](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) · [ModelScope Large-EN](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch) · [ModelScope Streaming-ZH 流式](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online) | - Chinese & English, streaming option <br> - 中英文，支持流式 |
| **Parakeet** | [Hugging Face](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | - Multilingual FastConformer + TDT decoder <br> - 多语种 FastConformer + TDT 解码器 |
| **Qwen3-ASR** | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B) | - Hot-words + language prompt, greedy / beam search <br> - 热词 + 语言提示，贪心/束搜索 |
| **Qwen3 ForcedAligner** | [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B) | - Non-autoregressive word-level timestamps <br> - 非自回归词级时间戳 |
| **SenseVoice** | [GitHub](https://github.com/FunAudioLLM/SenseVoice) | - Multilingual, punctuation & emotion <br> - 多语种、标点与情感 |
| **Whisper** | [Hugging Face V3](https://huggingface.co/openai/whisper-large-v3) · [Hugging Face V2](https://huggingface.co/openai/whisper-large-v2) · - Turbo / fine-tunes | Set a target language <br> - 需指定目标语言 |
| **X-ASR** | [GitHub](https://github.com/Gilgamesh-J/X-ASR) | - Streaming Zipformer transducer (ZH-EN) <br> - 流式 Zipformer 转录器（中英） |

**👥 Multi-talker models · 多人说话模型**

| **Model · 模型** | **Variants & Links · 变体与链接** | **Highlights · 亮点** |
|:---|:---|:---|
| **Parakeet MultiTalker Streaming** | [Hugging Face](https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1) | - Four-speaker streaming ASR + diarization <br> - 四人流式语音识别 + 说话人分离 |

---

## ⚡ Performance · 性能

> CPU real-time factor (RTF), lower is faster. / CPU 实时率（RTF），数值越低越快。

| **OS**          | **Device** | **Backend**           | **Model**                                      | **Real-Time Factor**<br>(Chunk Size: 128000 or 8s) |
|:----------------:|:----------:|:---------------------:|:---------------------------------------------:|:--------------------------------------------------:|
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | SenseVoiceSmall<br>f32                           | 0.037                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | SenseVoiceSmall<br>q8f32                         | 0.075                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall<br>f32                           | 0.019                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall<br>q8f32                         | 0.022                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Whisper-Large-v3-en<br>q8f32                     | 0.15                                               |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Whisper-Large-v3-Turbo-en<br>q8f32               | 0.073                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | Paraformer-Small-Chinese<br>f32                  | 0.04                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | Paraformer-Large-English<br>q8f32                | 0.14                                               |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Paraformer-Large-Streaming-Chinese<br>f32        | 0.06 <br> Chunk Size: 8000                         |
| Ubuntu 24.04     | Desktop     | CPU<br>i3-12300     | FireRedASR-AED-L-Chinese<br>q8f32                | 0.17                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | Dolphin-Small<br>q8f32                           | 0.14                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | Fun-ASR-Nano<br>q4f32                            | 0.11                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | Qwen3-ASR-0.6B<br>q4f32                          | 0.12                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | Nemotron-ASR-0.6B<br>q8f32                       | 0.1                                                |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | Parakeet-ASR-0.6B<br>q8f32                       | 0.08                                               |



---

## 🚀 Quick Start · 快速开始

```bash
# 1) Export the model to ONNX / 导出为 ONNX
python <Model>/Export_<Model>.py
# 2) Quantize · slim / 量化 · 精简
python <Model>/Optimize_ONNX.py
# 3) Transcribe / 推理转录
python <Model>/Inference_<Model>_ONNX.py
```

> Replace `<Model>` with any folder above, e.g. `ARK_ASR`, `Audio8_ASR`, `Mega_ASR`, `SenseVoice`. / 将 `<Model>` 替换为上方任一目录，如 `ARK_ASR`、`Audio8_ASR`、`Mega_ASR`、`SenseVoice`。

### ARK-ASR · ARK 语音识别

> Download [ARK-ASR-0.6B](https://huggingface.co/Audio8/ARK-ASR-0.6B) to `~/Downloads/ARK-ASR-0.6B`, the default path configured by the exporter. / 将 [ARK-ASR-0.6B](https://huggingface.co/Audio8/ARK-ASR-0.6B) 下载到导出脚本默认配置路径 `~/Downloads/ARK-ASR-0.6B`。

```bash
python ARK_ASR/Export_ARK_ASR.py
python ARK_ASR/Optimize_ONNX.py
python ARK_ASR/Inference_ARK_ASR_ONNX.py
```

### Parakeet MultiTalker Streaming · Parakeet 流式多人语音识别

> Download `multitalker-parakeet-streaming-0.6b-v1.nemo` to `~/Downloads/multitalker-parakeet-streaming-0.6b-v1/`. This pipeline also uses `~/Downloads/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo` for diarization. / 将 `multitalker-parakeet-streaming-0.6b-v1.nemo` 下载到 `~/Downloads/multitalker-parakeet-streaming-0.6b-v1/`；该流程还使用 `~/Downloads/diar_streaming_sortformer_4spk-v2.1/diar_streaming_sortformer_4spk-v2.1.nemo` 进行说话人分离。

```bash
python Parakeet/MultiTalker-Streaming/Export_MultiTalker_Streaming_Parakeet_ASR.py
python Parakeet/MultiTalker-Streaming/Optimize_ONNX.py
python Parakeet/MultiTalker-Streaming/Inference_MultiTalker_Streaming_Parakeet_ASR_ONNX.py
```

## 🧰 Pairs Well With · 推荐搭配

- 🎚️ [Voice Activity Detection (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX) — 语音活动检测
- 🧹 [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX) — 音频降噪

## 📄 License · 许可证

[Apache-2.0](LICENSE)

## 🔗 Learn More · 了解更多

⭐ **Star & follow for more on-device AI** → **[DakeQQ's projects](https://github.com/DakeQQ?tab=repositories)**  
喜欢就点个 **Star** 关注，获取更多端侧 AI 项目 → **[DakeQQ 项目主页](https://github.com/DakeQQ?tab=repositories)**
