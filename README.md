---

## Automatic-Speech-Recognition-ASR-ONNX  
Harness the power of ONNX Runtime to transcribe audio into text effortlessly.

### Supported Models  
1. **Single Model**:  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)
   - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3) / [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) / [Whisper-V3-Japanese](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) / [Whisper-V3-Turbo-Japanese](https://huggingface.co/hhim8826/whisper-large-v3-turbo-ja) / [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) / [Distil-Whisper-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) / [anime-whisper](https://huggingface.co/litagin/anime-whisper) / [whisper-ja-anime](https://huggingface.co/efwkjn/whisper-ja-anime-v0.1)...
   - [Whisper-Large-V2](https://huggingface.co/openai/whisper-large-v2) / [Whisper-V2-Japanese](https://huggingface.co/clu-ling/whisper-large-v2-japanese-5k-steps) ...
   - [Paraformer-Small-Chinese](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) / [Paraformer-Large-Chinese](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) / [Paraformer-Large-English](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)
   - [Paraformer-Online-Streaming-Chinese](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)
   - [FireRedASR-AED](https://github.com/FireRedTeam/FireRedASR)
   - [Dolphin](https://github.com/DataoceanAI/Dolphin/tree/main) / [link](https://drive.google.com/drive/folders/1B4ucWr-zos0dUVDqVFLbw-mY7vbYIOip?usp=drive_link)
     
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
- This Whisper does not support automatic language detection. Please specify a target language.

### Learn More  
- Visit the [project overview](https://github.com/DakeQQ?tab=repositories) for further details.

---

## 性能 Performance  

| **OS**          | **Device** | **Backend**           | **Model**                                      | **Real-Time Factor**<br>(Chunk Size: 128000 or 8s) |
|:----------------:|:----------:|:---------------------:|:---------------------------------------------:|:--------------------------------------------------:|
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | SenseVoiceSmall<br>f32                           | 0.037                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | SenseVoiceSmall<br>q8f32                         | 0.075                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall<br>f32                           | 0.019                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall<br>q8f32                         | 0.022                                              |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | SenseVoiceSmall + <br>ERes2NetV2_w24s4ep4<br>f32 | 0.10                                               |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Whisper-Large-v3-en<br>q8f32                     | 0.15                                               |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Whisper-Large-v3-Turbo-en<br>q8f32               | 0.073                                              |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | Paraformer-Small-Chinese<br>f32                  | 0.04                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i5-7300HQ     | Paraformer-Large-English<br>q8f32                | 0.14                                               |
| Ubuntu 24.04     | Desktop    | CPU<br>i3-12300      | Paraformer-Large-Streaming-Chinese<br>f32        | 0.06 <br> Chunk Size: 8800                         |
| Ubuntu 24.04     | Laptop     | CPU<br>i3-12300      | FireRedASR-AED-L-Chinese<br>q8f32                | 0.17                                               |
| Ubuntu 24.04     | Laptop     | CPU<br>i7-1165G7     | Dolphin-Small<br>q8f32                           | 0.13                                               |

---

## Coming Soon 🚀  
- None


---

### 自动语音识别（ASR）ONNX  
利用 ONNX Runtime 实现音频到文本的高效转录。

### 支持模型  
1. **单模型**：  
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)
   - [Whisper-Large-V3](https://huggingface.co/openai/whisper-large-v3) / [Whisper-Large-V3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) / [Whisper-V3-Japanese](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0) / [Whisper-V3-Turbo-Japanese](https://huggingface.co/hhim8826/whisper-large-v3-turbo-ja) / [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) / [Distil-Whisper-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) / [anime-whisper](https://huggingface.co/litagin/anime-whisper) / [whisper-ja-anime](https://huggingface.co/efwkjn/whisper-ja-anime-v0.1)...
   - [Whisper-Large-V2](https://huggingface.co/openai/whisper-large-v2) / [Whisper-V2-Japanese](https://huggingface.co/clu-ling/whisper-large-v2-japanese-5k-steps) ...
   - [Paraformer-Small-中文](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1) / [Paraformer-Large-中文](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch) / [Paraformer-Large-英文](https://modelscope.cn/models/iic/speech_paraformer_asr-en-16k-vocab4199-pytorch)
   - [Paraformer-实时-流式-中文](https://modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online)
   - [FireRedASR-AED](https://github.com/FireRedTeam/FireRedASR)
   - [Dolphin](https://github.com/DataoceanAI/Dolphin/tree/main) / [link](https://drive.google.com/drive/folders/1B4ucWr-zos0dUVDqVFLbw-mY7vbYIOip?usp=drive_link)

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
- 此 Whisper 不支持自动语言检测。请指定目标语言。

### 了解更多  
- 访问[项目概览](https://github.com/DakeQQ?tab=repositories)获取更多信息。

---
