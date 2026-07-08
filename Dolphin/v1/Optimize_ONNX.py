"""Optimize & quantize the exported Dolphin ONNX modules."""

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


MODEL_PATH = r"/home/DakeQQ/Downloads/dolphin-small"
ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "Dolphin_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Dolphin_Optimized")

USE_OPENVINO = False
FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

DOLPHIN_NUM_HEADS = 12 if "small" in MODEL_PATH.lower() else 8
DOLPHIN_HIDDEN_SIZE = 768 if "small" in MODEL_PATH.lower() else 512

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


# ============================== MODEL PLANS ==============================

TRANSFORMER_PLAN = dict(num_heads=DOLPHIN_NUM_HEADS, hidden_size=DOLPHIN_HIDDEN_SIZE)

MODEL_PLANS = {
    "Dolphin_Metadata":                 Plan(method="F32", transformer=False),
    "Dolphin_Encoder":                  Plan(method="DYNAMIC", **TRANSFORMER_PLAN),
    "Dolphin_Decoder":                  Plan(method="DYNAMIC", **TRANSFORMER_PLAN),
    "Dolphin_Decoder_Embed":            Plan(method="DYNAMIC", transformer=False),
    "Dolphin_Position_Mask_Prefill":    Plan(method="F32", transformer=False),
    "Dolphin_Position_Mask_Decode":     Plan(method="F32", transformer=False),
    "Dolphin_Greedy_Search":            Plan(method="F32", transformer=False),
    "Dolphin_Argmax":                   Plan(method="F32", transformer=False),
    "Dolphin_First_Beam_Search":        Plan(method="F32", transformer=False),
    "Dolphin_Second_Beam_Search":       Plan(method="F32", transformer=False),
    "Dolphin_Apply_Penality":           Plan(method="F32", transformer=False),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    dynamic_weight_type="QInt8",
    dynamic_per_channel=True,
    dynamic_reduce_range=False,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=2,
    optimizer_only_onnxruntime=USE_OPENVINO,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    copy_artifacts=("vocab_Dolphin.txt", "bpe.model"),
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
