import gc
import os
import subprocess

import torch
import onnx.version_converter
import onnxruntime
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxslim import slim
from transformers import AutoModelForSpeechSeq2Seq


# Path Setting
download_path = "/home/DakeQQ/Downloads/whisper-large-v2"                               # The whisper model download path.
original_folder_path = "/home/DakeQQ/Downloads/Whisper_ONNX"                            # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/Whisper_Optimized"                      # The optimized folder.
# model_path = os.path.join(original_folder_path, "Whisper_Encoder.onnx")               # The original fp32 model name.
# optimized_model_path = os.path.join(optimized_folder_path, "Whisper_Encoder.onnx")    # The optimized model name.
model_path = os.path.join(original_folder_path, "Whisper_Decoder.onnx")                 # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "Whisper_Decoder.onnx")      # The optimized model name.
do_quantize = True                                                                      # Use dynamic quant the model to int8 format.
use_gpu_fp16 = False                                                                    # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                       # ['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider']
target_platform = "amd64"                                                               # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.


# Check model
DYNAMIC_AXES = True  # Currently, whisper model only support dynamic_axes = True.
# if isinstance(onnxruntime.InferenceSession(model_path)._inputs_meta[0].shape[-1], str):
#     DYNAMIC_AXES = True
# else:
#     DYNAMIC_AXES = False


if do_quantize:
    quantize_dynamic(
        model_input=model_path,
        model_output=optimized_model_path,
        per_channel=True,                                   # True for model accuracy but cost a lot of time during quanting process.
        reduce_range=False,                                 # True for some x86_64 platform.
        weight_type=QuantType.QUInt8,                       # It is recommended using uint8 + Symmetric False
        extra_options={'ActivationSymmetric': False,        # True for inference speed. False may keep more accuracy.
                       'WeightSymmetric': False,            # True for inference speed. False may keep more accuracy.
                       'EnableSubgraph': True,              # True for more quant.
                       'ForceQuantizeNoInputCheck': False,  # True for more quant.
                       'MatMulConstBOnly': True             # False for more quant. Sometime, the inference speed may get worse.
                       },
        nodes_to_exclude=None,                              # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
        use_external_data_format=False                      # Save the model into two parts.
    )


# ONNX Model Optimizer
slim(
    model=optimized_model_path if do_quantize else model_path,
    output_model=optimized_model_path,
    no_shape_infer=True if DYNAMIC_AXES else False,         # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


try:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(download_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=False).eval()
except:
    model = AutoModelForSpeechSeq2Seq.from_pretrained(download_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True).eval()
HIDDEN_SIZE = model.config.d_model
NUM_HEAD = model.model.config.decoder_attention_heads
del model

# transformers.optimizer
model = optimize_model(optimized_model_path,
                       use_gpu=True,                               # Set to True because the model uses float16.
                       opt_level=99 if (target_platform == "amd64") and not use_gpu_fp16 else 2,
                       num_heads=NUM_HEAD,
                       hidden_size=HIDDEN_SIZE,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
if use_gpu_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=False if DYNAMIC_AXES else True,  # True for more optimize but may get errors.
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(optimized_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=True if DYNAMIC_AXES else False,   # True for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)


# Upgrade the Opset version. (optional process)
model = onnx.load(optimized_model_path)
model = onnx.version_converter.convert_version(model, 21)
onnx.save(model, optimized_model_path, save_as_external_data=False)
del model
gc.collect()


if not use_gpu_fp16:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"  # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {optimized_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {optimized_folder_path}'], shell=True)
