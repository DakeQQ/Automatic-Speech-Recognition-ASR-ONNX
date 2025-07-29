import os
import gc
import glob
import sys
import onnx
import torch
import subprocess
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.quantization import QuantType, quantize_dynamic, quant_utils
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM


# Path Setting
download_path = r'/home/DakeQQ/Downloads/whisper-large-v3-turbo'                          # Set the folder path where the whole project downloaded, otherwise set "NONE".
original_folder_path = r"/home/DakeQQ/Downloads/Whisper_ONNX"                             # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/Whisper_Optimized"                         # The optimized folder.

model_path = os.path.join(original_folder_path, "Whisper_Encoder.onnx")               # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Whisper_Encoder.onnx")        # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "Whisper_Decoder.onnx")               # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "Whisper_Decoder.onnx")        # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "Greedy_Search.onnx")                 # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "Greedy_Search.onnx")          # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "First_Beam_Search.onnx")             # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "First_Beam_Search.onnx")      # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "Second_Beam_Search.onnx")            # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "Second_Beam_Search.onnx")     # The optimized model stored path.

# model_path = os.path.join(original_folder_path, "Reset_Penality.onnx")                # The original fp32 model path.
# quanted_model_path = os.path.join(quanted_folder_path, "Reset_Penality.onnx")         # The optimized model stored path.


quant_int8 = True                                                                # Quant the model to int8 format.
quant_float16 = False                                                            # Quant the model to float16 format.
use_gpu = False                                                                  # If true, the transformers.optimizer will save the model with GPU device only format.
use_low_memory_mode_in_Android = False                                           # # If True, save the model into 2 parts.
upgrade_opset = 17                                                               # Optional process. Set 0 for close.
target_platform = "arm"                                                          # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.


# Start Quantize
if quant_int8 and "Reset_Penality" not in model_path:
    quantize_dynamic(
        model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
        model_output=quanted_model_path,
        per_channel=True,                                       # True for model accuracy but cost a lot of time during quanting process.
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
        no_shape_infer=True,  # False for more optimize but may get errors.
        skip_fusion_patterns=False,
        no_constant_folding=False,
        save_as_external_data=use_low_memory_mode_in_Android,
        verbose=False
    )
else:
    # ONNX Model Optimizer
    if "Reset_Penality" in model_path:
        model = optimize_model(model_path,
                               use_gpu=use_gpu,
                               opt_level=2,
                               num_heads=0,
                               hidden_size=0,
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
    else:
        slim(
            model=quant_utils.load_model_with_shape_infer(Path(model_path)),
            output_model=quanted_model_path,
            no_shape_infer=True,                # False for more optimize but may get errors.
            skip_fusion_patterns=False,
            no_constant_folding=False,
            save_as_external_data=use_low_memory_mode_in_Android,
            verbose=False
        )

# transformers.optimizer
if "Reset_Penality" not in model_path or "First_Beam_Search" not in model_path:
    if download_path.lower() == "none" or download_path is None:
        num_heads = 0    # default
        hidden_size = 0  # default
    else:
        model = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        try:
            num_heads = model.config.num_attention_heads
            hidden_size = model.config.hidden_size
        except:
            num_heads = 0
            hidden_size = 0
        del model
        gc.collect()

    model = optimize_model(quanted_model_path,
                           use_gpu=use_gpu,
                           opt_level=2,
                           num_heads=num_heads,
                           hidden_size=hidden_size,
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
        no_shape_infer=False,                                     # False for more optimize but may get errors.
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

    pattern = os.path.join(quanted_folder_path, '*.onnx.data')
    files_to_delete = glob.glob(pattern)
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

if not use_low_memory_mode_in_Android and not quant_float16:
    # Convert the simplified model to ORT format.
    if use_gpu:
        optimization_style = "Runtime"      # ['Runtime', 'Fixed']; Runtime for XNNPACK/NNAPI/QNN/CoreML..., Fixed for CPU provider
    else:
        optimization_style = "Fixed"

    # Call subprocess may get permission failed on Windows system.
    subprocess.run([f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} --enable_type_reduction {quanted_folder_path}'], shell=True)
