# Automatic-Speech-Recognition-ASR-ONNX
Utilizes ONNX Runtime to transcribe audio into text.
1. Now support:
   - [SenseVoiceSmall](https://github.com/FunAudioLLM/SenseVoice)
2. This end-to-end version includes internal `STFT` processing. Input audio; output is ASR result.
3. It is recommended to work with the [VAD](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX) and the [denoised](https://github.com/DakeQQ/Audio-Denoiser-ONNX) model.
4. See more -> https://dakeqq.github.io/overview/

# Automatic-Speech-Recognition-ASR-ONNX
1. 现在支持:
   - [SenseVoiceSmall](https://modelscope.cn/models/iic/SenseVoiceSmall)
2. 这个端到端版本包括内部的 `STFT` 处理。输入为音频，输出为 ASR 结果。
3. 建议与 [VAD](https://github.com/DakeQQ/Voice-Activity-Detection-VAD-ONNX) 和 [去噪模型](https://github.com/DakeQQ/Audio-Denoiser-ONNX) 一起使用。.
4. See more -> https://dakeqq.github.io/overview/

# 性能 Performance
| OS | Device | Backend | Model | Real-Time Factor<br>( Chunk_Size: 128000 or 8s ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | SenseVoiceSmall<br>f32 | 0.037 |
| Ubuntu-24.04 | Laptop | CPU<br>i5-7300HQ | SenseVoiceSmall<br>q8f32 | 0.075 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | SenseVoiceSmall<br>f32 | 0.019 |
| Ubuntu-24.04 | Desktop | CPU<br>i3-12300 | SenseVoiceSmall<br>q8f32 | 0.022 |


# Coming Up ...
 - [Paraformer-small-zh](https://modelscope.cn/models/iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358-tensorflow1)
 - [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
