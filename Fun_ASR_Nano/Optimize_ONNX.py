"""Optimize & quantize the exported FunASR Nano ONNX modules."""

from pathlib import Path
import shutil
import sys


# ============================== USER CONFIG ==============================

_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "Fun_ASR_Nano_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Fun_ASR_Nano_Optimized")

USE_OPENVINO = False
FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

FUN_ASR_NUM_HEADS = 16
FUN_ASR_HIDDEN_SIZE = 1024

WEIGHT_ONLY_ALGORITHM = "k_quant"
BLOCK_SIZE = 32
ACCURACY_LEVEL = 4
QUANT_SYMMETRIC = False

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
]


# ============================== MODEL PLANS ==============================

TRANSFORMER_PLAN = dict(num_heads=FUN_ASR_NUM_HEADS, hidden_size=FUN_ASR_HIDDEN_SIZE)
Q4_MATMUL = dict(method="Q4", op_types=("MatMul",), axes=(0,), **TRANSFORMER_PLAN)
Q4_GATHER = dict(method="Q4", algo="DEFAULT", op_types=("Gather",), axes=(1,), **TRANSFORMER_PLAN)
F32_HELPER = dict(method="F32", transformer=False)

MODEL_PLANS = {
    "FunASR_Nano_Metadata":                 Plan(**F32_HELPER),
    "FunASR_Nano_Encoder":                  Plan(**Q4_MATMUL, opt_level=2),
    "FunASR_Nano_CTC_Decoder":              Plan(**Q4_MATMUL, opt_level=2),
    "FunASR_Nano_Decoder_Embed":            Plan(**Q4_GATHER),
    "FunASR_Nano_Decoder_Main":             Plan(**Q4_MATMUL),
    "FunASR_Nano_Rotary_Mask_Text_Prefill": Plan(**F32_HELPER),
    "FunASR_Nano_Rotary_Mask_Text_Decode":  Plan(**F32_HELPER),
    "FunASR_Nano_Greedy_Search":            Plan(**F32_HELPER),
    "FunASR_Nano_First_Beam_Search":        Plan(**F32_HELPER),
    "FunASR_Nano_Second_Beam_Search":       Plan(**F32_HELPER),
    "FunASR_Nano_Apply_Penalty":            Plan(**F32_HELPER),
    "FunASR_Nano_Argmax":                   Plan(**F32_HELPER),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    block_size=BLOCK_SIZE,
    accuracy_level=ACCURACY_LEVEL,
    quant_symmetric=QUANT_SYMMETRIC,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=2,
    optimizer_only_onnxruntime=USE_OPENVINO,
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)

    # Mirror the bundled tokenizer assets from the exported folder into the optimized folder so
    # the optimized model set also runs stand-alone (Inference_Fun_ASR_Nano_ONNX.py loads the
    # tokenizer from its selected ONNX folder).
    _src_folder = Path(ORIGINAL_FOLDER_PATH)
    _dst_folder = Path(OPTIMIZED_FOLDER_PATH)
    for _asset in ("Qwen3-0.6B", "multilingual.tiktoken"):
        _asset_src = _src_folder / _asset
        try:
            if _asset_src.is_dir():
                shutil.copytree(_asset_src, _dst_folder / _asset, dirs_exist_ok=True)
                print(f"[Tokenizer] Copied {_asset} -> {_dst_folder / _asset}")
            elif _asset_src.is_file():
                shutil.copyfile(_asset_src, _dst_folder / _asset)
                print(f"[Tokenizer] Copied {_asset} -> {_dst_folder / _asset}")
        except Exception as _exc:  # noqa: BLE001 - a failed asset copy must not fail optimization
            print(f"[Tokenizer] Skipped {_asset} ({_exc}).")
