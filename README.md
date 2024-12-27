---

## Automatic-Speech-Recognition-ASR-ONNX  
Harness the power of ONNX Runtime to transcribe audio into text effortlessly.

### Supported Models  
1. **Single Model**:  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)
   - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3)
   - [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
   - [Custom Fine tune Whisper, such as: kotoba-Japanese](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)

2. **Combined Models (ASR + Speaker Identify)**:  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) + [ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) + [ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)  

### Features  
- End-to-end speech recognition with built-in `STFT` processing.  
  **Input**: Audio file  
  **Output**: Transcription result  
- Seamlessly integrate with these additional tools for improved performance:  
  - [Voice Activity Detection (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX)  
  - [Audio Denoiser](https://github.com/DakeQQ/Audio-Denoiser-ONNX)  

### Learn More  
- Visit the [project overview](https://dakeqq.github.io/overview/) for further details.

---

## 性能 Performance  

| **OS**          | **Device** | **Backend**           | **Model**                                      | **Real-Time Factor**<br>(Chunk Size: 128000 or 8s) |
|:----------------:|:----------:|:---------------------:|:---------------------------------------------:|:--------------------------------------------------:|
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | SenseVoiceSmall<br>f32                           | 0.037                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | SenseVoiceSmall<br>q8f32                         | 0.075                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall<br>f32                           | 0.019                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall<br>q8f32                         | 0.022                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall + <br>ERes2NetV2_w24s4ep4<br>f32 | 0.1                                                |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Whisper-Large-v3-en<br>q8f32                     | 0.15                                               |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Whisper-Large-v3-Turbo-en<br>q8f32               | 0.073                                              |
---

## Coming Soon 🚀  
- [Paraformer-small-zh](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)  
- [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)

---

### 自动语音识别（ASR）ONNX  
利用 ONNX Runtime 实现音频到文本的高效转录。

### 支持模型  
1. **单模型**：  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)  

2. **组合模型 (ASR + 讲话者识别)**：  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) + [ERes2NetV2](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice) + [ERes2NetV2_w24s4ep4](https://modelscope.cn/models/iic/speech_eres2netv2w24s4ep4_sv_zh-cn_16k-common)  

### 功能特点  
- 端到端语音识别，内置 `STFT` 处理。  
  **输入**：音频文件  
  **输出**：转录结果  
- 推荐搭配以下工具，提升性能：  
  - [语音活动检测 (VAD)](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX)  
  - [音频去噪](https://github.com/DakeQQ/Audio-Denoiser-ONNX)  

### 了解更多  
- 访问[项目概览](https://dakeqq.github.io/overview/)获取更多信息。

---
