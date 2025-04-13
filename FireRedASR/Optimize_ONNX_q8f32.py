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


use_gpu = False                                                                             # If true, the transformers.optimizer will remain the FP16 processes.
provider = 'CPUExecutionProvider'                                                           # ['CPUExecutionProvider', 'CUDAExecutionProvider']
use_low_memory_mode_in_Android = False                                                      # If you need to use low memory mode on Android, please set it to True.
upgrade_opset = 21                                                                          # Optional process. Set 0 for close.


# Start Quantize
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


model_size_bytes = sys.getsizeof(onnx.load(quanted_model_path).SerializeToString())
model_size_gb = model_size_bytes * 9.31322575e-10  # 1 / (1024 * 1024 * 1024)
if model_size_gb > 2.0:
    is_large_model = True
else:
    is_large_model = True if use_low_memory_mode_in_Android else False
    

# ONNX Model Optimizer
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=True if "Encoder" in model_path else False,  # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
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
model.save_model_to_file(quanted_model_path, use_external_data_format=is_large_model)
del model
gc.collect()


# onnxslim 2nd
slim(
    model=quanted_model_path,
    output_model=quanted_model_path,
    no_shape_infer=True if "Encoder" in model_path else False,  # False for more optimize but may get errors.
    skip_fusion_patterns=False,
    no_constant_folding=False,
    save_as_external_data=is_large_model,
    verbose=False
)


# Upgrade the Opset version. (optional process)
if upgrade_opset > 0:
    try:
        model = onnx.load(quanted_model_path)
        model = onnx.version_converter.convert_version(model, upgrade_opset)
        onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
        del model
        gc.collect()
    except:
        model = onnx.load(quanted_model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
        del model
        gc.collect()
else:
    model = onnx.load(quanted_model_path)
    onnx.save(model, quanted_model_path, save_as_external_data=is_large_model)
    del model
    gc.collect()

pattern = os.path.join(quanted_folder_path, '*.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")
        
if not is_large_model:
    # Convert the simplified model to ORT format.
    if provider == 'CPUExecutionProvider':
        optimization_style = "Fixed"
    else:
        optimization_style = "Runtime"      # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    target_platform = "arm"                 # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.
    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)
