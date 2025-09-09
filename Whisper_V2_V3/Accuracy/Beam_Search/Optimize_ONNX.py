import os
import gc
import glob
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

# Create the output directory if it doesn't exist
os.makedirs(quanted_folder_path, exist_ok=True)

# List of models to process
model_names = [
    "Whisper_Encoder",
    "Whisper_Decoder",
    "Greedy_Search",
    "First_Beam_Search",
    "Second_Beam_Search",
    "Reset_Penality"
]

# Settings
quant_int8 = True                        # Quant the model to int8 format.
quant_float16 = False                    # Quant the model to float16 format.
use_openvino = False                     # Set true for OpenVINO optimization.
use_low_memory_mode_in_Android = False   # If True, save the model into 2 parts.
upgrade_opset = 17                       # Optional process. Set 0 for close.
target_platform = "arm"                  # ['arm', 'amd64']; The 'amd64' means x86_64 desktop, not means the AMD chip.


# --- Main Processing Loop ---
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
    if quant_int8 and "Reset_Penality" not in model_path:
        print("Applying UINT8 quantization...")
        quantize_dynamic(
            model_input=quant_utils.load_model_with_shape_infer(Path(model_path)),
            model_output=quanted_model_path,
            per_channel=True,
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
            no_shape_infer=True,
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
                                   model_type='bert')
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
                verbose=False
            )

    # transformers.optimizer
    if "Reset_Penality" not in model_path and "First_Beam_Search" not in model_path:
        print("Applying transformers.optimizer...")
        if download_path.lower() == "none" or download_path is None:
            num_heads = 0
            hidden_size = 0
        else:
            model_for_config = AutoModelForCausalLM.from_pretrained(download_path, torch_dtype=torch.float16, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
            try:
                num_heads = model_for_config.config.num_attention_heads
                hidden_size = model_for_config.config.hidden_size
            except:
                num_heads = 0
                hidden_size = 0
            del model_for_config
            gc.collect()

        model = optimize_model(quanted_model_path,
                               use_gpu=False,
                               opt_level=1 if use_openvino else 2,
                               num_heads=num_heads,
                               hidden_size=hidden_size,
                               verbose=False,
                               model_type='bert')
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
            no_shape_infer=False,
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
    if not use_low_memory_mode_in_Android and not quant_float16:
        print(f"Converting {model_name} to ORT format...")
        if quant_float16:
            optimization_style = "Runtime"
        else:
            optimization_style = "Fixed"

        subprocess.run(
            f'python -m onnxruntime.tools.convert_onnx_models_to_ort --output_dir {quanted_folder_path} --optimization_style {optimization_style} --target_platform {target_platform} {quanted_model_path}',
            shell=True
        )
    
    print(f"--- Finished processing {model_name} ---\n")


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

