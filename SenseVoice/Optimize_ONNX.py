import gc
import os
import subprocess

import onnx.version_converter
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxslim import slim


# Path Setting
original_folder_path = "/home/DakeQQ/Downloads/SenseVoice_ONNX"                           # The fp32 saved folder.
optimized_folder_path = "/home/DakeQQ/Downloads/SenseVoice_Optimized"                     # The optimized folder.
model_path = os.path.join(original_folder_path, "SenseVoiceSmall.onnx")                   # The original fp32 model name.
optimized_model_path = os.path.join(optimized_folder_path, "SenseVoiceSmall.onnx")        # The optimized model name.
do_quantize = True                                                                        # Use dynamic quant the model to int8 format.
use_fp16 = False                                                                          # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                         # ['CPUExecutionProvider', 'CUDAExecutionProvider', 'CoreMLExecutionProvider', 'DmlExecutionProvider']

# ONNX Model Optimizer
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


# transformers.optimizer
# Use this function for float16 quantization will get errors.
model = optimize_model(optimized_model_path if do_quantize else model_path,
                       use_gpu=False,        # Set to True because the model uses float16.
                       opt_level=2,
                       num_heads=4,          # The SenseVoiceSmall model parameter.
                       hidden_size=512,      # The SenseVoiceSmall model parameter.
                       provider=provider,
                       verbose=False,
                       model_type='bert')
if use_fp16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=True,      # True for more optimize but may get errors.
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(optimized_model_path, use_external_data_format=False)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=optimized_model_path,
    output_model=optimized_model_path,
    no_shape_infer=False,                        # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=False,
    verbose=False
)
