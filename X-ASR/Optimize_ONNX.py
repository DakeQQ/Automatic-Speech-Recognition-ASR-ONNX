"""Optimize & quantize the exported X-ASR streaming ONNX modules."""

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

from Optimize_ONNX_Common import OptimizerConfig, Plan, collect_quant_unsafe_nodes, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "X_ASR_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "X_ASR_Optimized")

FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
    # "Softmax",
    # "Sqrt",
    # "Pow",
    # "ReduceMean",
    # "ReduceSum",
    # "Exp",
    # "Log",
]


# ============================== MODEL PLANS ==============================

X_ASR_PLAN = dict(
    method="DYNAMIC",
    num_heads=0,
    hidden_size=0,
    nodes_to_exclude=collect_quant_unsafe_nodes,
)

MODEL_PLANS = {
    "X_ASR_Metadata": Plan(method="F32", transformer=False),
    "X_ASR_Encoder": Plan(**X_ASR_PLAN),
    "X_ASR_Decoder": Plan(**X_ASR_PLAN),
    "X_ASR_Joiner": Plan(**X_ASR_PLAN),
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
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)

    # ── Move the tokenizer vocab (tokens.txt) from the export folder to the optimized folder
    # so the optimized folder (the default inference target) runs inference stand-alone. ──
    _optimized_dir = Path(OPTIMIZED_FOLDER_PATH)
    for _asset in ("tokens.txt",):
        _src = Path(ORIGINAL_FOLDER_PATH) / _asset
        if not _src.exists():
            print(f"[Tokenizer] Skipped {_asset} (not found in {ORIGINAL_FOLDER_PATH})")
            continue
        try:
            _optimized_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(_src), str(_optimized_dir / _asset))
            print(f"[Tokenizer] Moved {_asset} -> {OPTIMIZED_FOLDER_PATH}")
        except Exception as _exc:  # noqa: BLE001
            print(f"[Tokenizer] Skipped {_asset} ({_exc})")
