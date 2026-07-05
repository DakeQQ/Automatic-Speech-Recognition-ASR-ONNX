"""Optimize & quantize the exported Qwen ForcedAligner ONNX modules."""

from pathlib import Path
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


MODEL_SIZE = "0.6B"
ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "Qwen_ForcedAligner_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Qwen_ForcedAligner_Optimized")

USE_OPENVINO = False
FORCE_EXTERNAL_DATA = True
UPGRADE_OPSET = 0

FORCED_ALIGNER_NUM_HEADS = 16
FORCED_ALIGNER_HIDDEN_SIZE = 1024 if "0.6B" in MODEL_SIZE else 2048

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

TRANSFORMER_PLAN = dict(num_heads=FORCED_ALIGNER_NUM_HEADS, hidden_size=FORCED_ALIGNER_HIDDEN_SIZE)
Q4_MATMUL = dict(method="Q4", op_types=("MatMul",), axes=(0,), **TRANSFORMER_PLAN)
Q4_GATHER = dict(method="Q4", algo="DEFAULT", op_types=("Gather",), axes=(1,), block_size=16, **TRANSFORMER_PLAN)

MODEL_PLANS = {
    "ForcedAligner_Metadata":       Plan(method="F32", transformer=False),
    "ForcedAligner_Encoder":        Plan(**Q4_MATMUL, opt_level=2),
    "ForcedAligner_Embed":          Plan(**Q4_GATHER),
    "ForcedAligner_Decoder_Main":   Plan(**Q4_MATMUL),
    "ForcedAligner_Rotary_Mask":    Plan(method="F32", transformer=False),
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
