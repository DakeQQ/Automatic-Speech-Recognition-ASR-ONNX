"""Optimize & quantize the exported FireRedASR-AED ONNX modules."""

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


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "FireRedASR_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "FireRedASR_Optimized")

USE_OPENVINO = False
FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

FIRERED_NUM_HEADS = 20
FIRERED_HIDDEN_SIZE = 1280

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


# ============================== MODEL PLANS ==============================

TRANSFORMER_PLAN = dict(num_heads=FIRERED_NUM_HEADS, hidden_size=FIRERED_HIDDEN_SIZE)

MODEL_PLANS = {
    "FireRedASR_Metadata":                Plan(method="F32", transformer=False),
    "FireRedASR_Encoder":                 Plan(method="DYNAMIC", **TRANSFORMER_PLAN),
    "FireRedASR_Decoder":                 Plan(method="DYNAMIC", **TRANSFORMER_PLAN),
    "FireRedASR_Decoder_Embed":           Plan(method="DYNAMIC", transformer=False),
    "FireRedASR_Position_Mask_Prefill":   Plan(method="F32", transformer=False),
    "FireRedASR_Position_Mask_Decode":    Plan(method="F32", transformer=False),
    "FireRedASR_Greedy_Search":           Plan(method="F32", transformer=False),
    "FireRedASR_Argmax":                  Plan(method="F32", transformer=False),
    "FireRedASR_First_Beam_Search":       Plan(method="F32", transformer=False),
    "FireRedASR_Second_Beam_Search":      Plan(method="F32", transformer=False),
    "FireRedASR_Apply_Penality":          Plan(method="F32", transformer=False),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    dynamic_weight_type="QInt8",
    # Keep per-tensor quantization: rank-3 constant-weight MatMuls reject a 1-D B zero-point at runtime.
    dynamic_per_channel=False,
    dynamic_reduce_range=False,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=2,
    optimizer_only_onnxruntime=USE_OPENVINO,
    f16_max_finite_val=65504.0,
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
