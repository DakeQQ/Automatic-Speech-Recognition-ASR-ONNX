import os
import gc
import glob
import sys
import onnx
import subprocess
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.quantization import QuantType, quantize_dynamic, quant_utils
from onnxruntime.transformers.optimizer import optimize_model


# Path Setting
original_folder_path = r"/home/DakeQQ/Downloads/FireRedASR_ONNX"                              # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/FireRedASR_Optimized"                          # The optimized folder.
project_path = "/home/DakeQQ/Downloads/FireRedASR-main"                                       # The FireRedASR Github project path.
model_path = os.path.join(original_folder_path, "FireRedASR_AED_L-Encoder.onnx")              # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "FireRedASR_AED_L-Encoder.onnx")       # The optimized model stored path.
# model_path = os.path.join(original_folder_path, "FireRedASR_AED_L-Decoder.onnx")            # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "FireRedASR_AED_L-Decoder.onnx")     # The optimized model stored path.

quant_int8 = True                                                                             # Quant the model to int8 format.
quant_float16 = False                                                                         # Quant the model to float16 format.
use_gpu = False                                                                               # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                             # ['CPUExecutionProvider', 'CUDAExecutionProvider']
upgrade_opset = 17                                                                            # Optional process. Set 0 for close.
use_low_memory_mode_in_Android = False                                                        # If True, save the model into 2 parts.


# Start Quantize
if quant_int8:
    quantize_dynamic(
        model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
        model_output=quanted_model_path,
        per_channel=True,                                        # True for model accuracy but cost a lot of time during quanting process.
        reduce_range=False,                                      # True for some x86_64 platform.
        weight_type=QuantType.QUInt8,                            # It is recommended using uint8 + Symmetric False
        extra_options={'ActivationSymmetric': False,             # True for inference speed. False may keep more accuracy.
                       'WeightSymmetric': False,                 # True for inference speed. False may keep more accuracy.
                       'EnableSubgraph': True,                   # True for more quant.
                       'ForceQuantizeNoInputCheck': False,       # True for more quant.
                       'MatMulConstBOnly': True                  # False for more quant. Sometime, the inference speed may get worse.
                       },
        nodes_to_exclude=None,                                   # Specify the node names to exclude quant process. Example: nodes_to_exclude={'/Gather'}
        use_external_data_format=True                            # Save the model into two parts.
    )
    # ONNX Model Optimizer
    slim(
        model=quanted_model_path,
        output_model=quanted_model_path,
        no_shape_infer=True if "Encoder" in model_path else False,  # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False
    )
else:
    # ONNX Model Optimizer
    slim(
        model=quant_utils.load_model_with_shape_infer(Path(model_path)),
        output_model=quanted_model_path,
        no_shape_infer=True if "Encoder" in model_path else False,  # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False
    )

if project_path == "NONE" or project_path is None or project_path == "":
    num_heads = 0    # default
    hidden_size = 0  # default
else:
    try:
        if project_path not in sys.path:
            sys.path.append(project_path)
        from fireredasr.models.fireredasr import FireRedAsr
        model = FireRedAsr.from_pretrained("aed", model_path).model.half()
        hidden_size = model.encoder.odim
        num_heads = model.encoder.layer_stack._modules['0'].mhsa.n_head
        del model
        gc.collect()
        if project_path in sys.path:
            sys.path.remove(project_path)
    except:
        num_heads = 0
        hidden_size = 0
# transformers.optimizer
model = optimize_model(quanted_model_path,
                       use_gpu=use_gpu,
                       opt_level=2,
                       num_heads=num_heads,
                       hidden_size=hidden_size,
                       provider=provider,
                       verbose=False,
                       model_type='bert')
if quant_float16:
    model.convert_float_to_float16(
        keep_io_types=False,
        force_fp16_initializers=True,
        use_symbolic_shape_infer=True,  # True for more optimize but may get errors.
        max_finite_val=65504.0,
        op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
    )
model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=True if "Encoder" in model_path else False,  # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=use_low_memory_mode_in_Android,
    verbose=False
)


# Upgrade the Opset version. (optional process)
if upgrade_opset > 0:
    try:
        model = onnx.load(quanted_model_path)
        model = onnx.version_converter.convert_version(model, upgrade_opset)
        onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
        del model
        gc.collect()
    except:
        model = onnx.load(quanted_model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
        del model
        gc.collect()
else:
    model = onnx.load(quanted_model_path)
    onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
    del model
    gc.collect()

pattern = os.path.join(quanted_folder_path, '*.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

if not use_low_memory_mode_in_Android:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"      # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    target_platform = "arm"                 # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)
