import os
import gc
import glob
import torch
import subprocess
import onnx.version_converter
from pathlib import Path
from onnxslim import slim
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModelForCausalLM
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)

# Path Setting
original_folder_path = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_ONNX"                   # The original folder.
quanted_folder_path = r"/home/DakeQQ/Downloads/Fun_ASR_Nano_Optimized"               # The optimized folder.

# Create the output directory if it doesn't exist
os.makedirs(quanted_folder_path, exist_ok=True)

# List of models to process
model_names = [
    "FunASR_Nano_Encoder",              
    "FunASR_Nano_Decoder_Embed",
    "FunASR_Nano_Decoder_Main",
    "FunASR_Nano_Greedy_Search",
    "FunASR_Nano_First_Beam_Search",
    "FunASR_Nano_Second_Beam_Search",
    "FunASR_Nano_Reset_Penality"
]

# Settings
quant_int4 = True                        # Quant the model to int4 format.
quant_int8 = False                       # Quant the model to int8 format.
quant_float16 = False                    # Quant the model to float16 format.
use_openvino = False                     # Set true for OpenVINO optimization.
use_low_memory_mode_in_Android = False   # If True, save the model into 2 parts.
upgrade_opset = 0                        # Optional process. Set 0 for close.
target_platform = "amd64"                # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.

# Int4 matmul_nbits_quantizer Settings
algorithm = "k_quant"                    # ["DEFAULT", "RTN", "HQQ", "k_quant"]
bits = 4                                 # [4, 8]; It is not recommended to use 8.
block_size = 32                          # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
accuracy_level = 4                       # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric = False                  # False may get more accuracy.
nodes_to_exclude = None                  # Set the node names here. Such as: ["/layers.0/mlp/down_proj/MatMul"]


# --- Main Processing Loop ---
algorithm_copy = algorithm
for model_name in model_names:
    print(f"--- Processing model: {model_name} ---")

    # Dynamically set model paths for the current iteration
    model_path = os.path.join(original_folder_path, f"{model_name}.onnx")
    quanted_model_path = os.path.join(quanted_folder_path, f"{model_name}.onnx")
    
    # Check if the original model file exists before processing
    if not os.path.exists(model_path):
        print(f"Warning: Model file not found at {model_path}. Skipping.")
        continue

    # Start Quantize
    if quant_int4 and ("Embed" in model_path or "Main" in model_path or "Encoder" in model_path):
        if "Embed" in model_path:
            op_types = ["Gather"]
            quant_axes = [1]
            algorithm = "DEFAULT"  # Fallback to DEFAULT
        else:
            op_types = ["MatMul"]
            quant_axes = [0]
            algorithm = algorithm_copy

        # Start Weight-Only Quantize
        model = quant_utils.load_model_with_shape_infer(Path(model_path))

        if algorithm == "RTN":
            quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types)
            )
        elif algorithm == "HQQ":
            quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
                bits=bits,
                block_size=block_size,
                axis=quant_axes[0],
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
            )
        elif algorithm == "k_quant":
            quant_config = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types)
            )
        else:
            quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=block_size,
                is_symmetric=quant_symmetric,
                accuracy_level=accuracy_level,
                quant_format=quant_utils.QuantFormat.QOperator,
                op_types_to_quantize=tuple(op_types),
                quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
            )
        quant_config.bits = bits
        quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
            model,
            block_size=block_size,
            is_symmetric=quant_symmetric,
            accuracy_level=accuracy_level,
            quant_format=quant_utils.QuantFormat.QOperator,
            op_types_to_quantize=tuple(op_types),
            quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types))),
            algo_config=quant_config,
            nodes_to_exclude=nodes_to_exclude
        )
        quant.process()
        quant.model.save_model_to_file(
            quanted_model_path,
            True                                         # save_as_external_data
        )

    elif quant_int8 and ("Reset_Penality" not in model_path):
        print("Applying UINT8 quantization...")
        quantize_dynamic(
            model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
            model_output=quanted_model_path,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
            extra_options={'ActivationSymmetric': False,
                           'WeightSymmetric': False,
                           'EnableSubgraph': True,
                           'ForceQuantizeNoInputCheck': False,
                           'MatMulConstBOnly': True
                           },
            nodes_to_exclude=None,
            use_external_data_format=True
        )
        # ONNX Model Optimizer
        print("Slimming the quantized model...")
        slim(
            model=quanted_model_path,
            output_model=quanted_model_path,
            no_shape_infer=False,
            skip_fusion_patterns=False,
            no_constant_folding=False,
            save_as_external_data=use_low_memory_mode_in_Android,
            verbose=False
        )
    else:
        # ONNX Model Optimizer for non-INT8 or Reset_Penality model
        print("Optimizing model (non-UINT8 path)...")
        if "Reset_Penality" in model_path:
            model = optimize_model(model_path,
                                       use_gpu=False,
                                       opt_level=2,
                                       num_heads=0,
                                       hidden_size=0,
                                       verbose=False,
                                       model_type='bert',
                                       only_onnxruntime=False)
            if quant_float16:
                print("Converting model to Float16...")
                model.convert_float_to_float16(
                    keep_io_types=False,
                    force_fp16_initializers=True,
                    use_symbolic_shape_infer=True,
                    max_finite_val=65504.0,
                    op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
                )
            model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)
        else:
            slim(
                model=quant_utils.load_model_with_shape_infer(Path(model_path)),
                output_model=quanted_model_path,
                no_shape_infer=True,
                skip_fusion_patterns=False,
                no_constant_folding=False,
                save_as_external_data=use_low_memory_mode_in_Android,
                verbose=False,
                dtype='fp16' if quant_float16 and "First_Beam_Search" in model_path else 'fp32'
            )

    # transformers.optimizer
    if ("Reset_Penality" not in model_path) and ("First_Beam_Search" not in model_path):
        print("Applying transformers.optimizer...")
        model = optimize_model(quanted_model_path,
                               use_gpu=False,
                               opt_level=1 if use_openvino else 2,
                               num_heads=8,
                               hidden_size=1024,
                               verbose=False,
                               model_type='bert',
                               only_onnxruntime=use_openvino)
        if quant_float16:
            print("Converting model to Float16...")
            model.convert_float_to_float16(
                keep_io_types=False,
                force_fp16_initializers=True,
                use_symbolic_shape_infer=True,
                max_finite_val=65504.0,
                op_block_list=['DynamicQuantizeLinear', 'DequantizeLinear', 'DynamicQuantizeMatMul', 'Range', 'MatMulIntegerToFloat']
            )
        model.save_model_to_file(quanted_model_path, use_external_data_format=use_low_memory_mode_in_Android)
        del model
        gc.collect()

        # onnxslim 2nd pass
        print("Applying second onnxslim pass...")
        slim(
            model=quanted_model_path,
            output_model=quanted_model_path,
            no_shape_infer=True,
            skip_fusion_patterns=False,
            no_constant_folding=False,
            save_as_external_data=use_low_memory_mode_in_Android,
            verbose=False
        )

    # Upgrade the Opset version. (optional process)
    if upgrade_opset > 0:
        print(f"Upgrading Opset to {upgrade_opset}...")
        try:
            model = onnx.load(quanted_model_path)
            converted_model = onnx.version_converter.convert_version(model, upgrade_opset)
            onnx.save(converted_model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
            del model, converted_model
            gc.collect()
        except Exception as e:
            print(f"Could not upgrade opset due to an error: {e}. Saving model with original opset.")
            model = onnx.load(quanted_model_path)
            onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
            del model
            gc.collect()
    else:
        model = onnx.load(quanted_model_path)
        onnx.save(model, quanted_model_path, save_as_external_data=use_low_memory_mode_in_Android)
        del model
        gc.collect()

    # This check is outside the main processing block in the original script.
    # It should be inside the loop if you want to convert each model to ORT format right after processing.
    # if not use_low_memory_mode_in_Android and not quant_float16:
    #     print(f"Converting {model_name} to ORT format...")
    #     if quant_float16:
    #         optimization_style = "Runtime"
    #     else:
    #         optimization_style = "Fixed"
    #
    #     subprocess.run(
    #         f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} {quanted_model_path}',
    #         shell=True
    #     )
    #
    # print(f"--- Finished processing {model_name} ---\n")


# Clean up external data files at the very end
print("Cleaning up temporary *.onnx.data files...")
pattern = os.path.join(quanted_folder_path, '*.onnx.data')
files_to_delete = glob.glob(pattern)
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted {file_path}")
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

print("--- All models processed successfully! ---")
