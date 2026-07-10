"""Optimize & quantize the exported Whisper ONNX modules."""

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

from Optimize_ONNX_Common import OptimizerConfig, Plan, metadata_int_for_model, run_optimizer


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "Whisper_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Whisper_Optimized")

USE_OPENVINO = False
FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


def whisper_num_heads(model_path: str) -> int:
    key = "num_encoder_heads" if "Encoder" in Path(model_path).stem else "num_decoder_heads"
    return metadata_int_for_model(model_path, key)


def whisper_hidden_size(model_path: str) -> int:
    return metadata_int_for_model(model_path, "hidden_size")


# ============================== MODEL PLANS ==============================

TRANSFORMER_PLAN = dict(num_heads=whisper_num_heads, hidden_size=whisper_hidden_size)

MODEL_PLANS = {
    "Whisper_Encoder":                  Plan(method="DYNAMIC", **TRANSFORMER_PLAN),
    "Whisper_Decoder":                  Plan(method="DYNAMIC", **TRANSFORMER_PLAN),
    "Whisper_Decoder_Embed":            Plan(method="DYNAMIC", transformer=False),
    "Whisper_Position_Mask_Prefill":    Plan(method="F32", transformer=False),
    "Whisper_Position_Mask_Decode":     Plan(method="F32", transformer=False),
    "Whisper_Greedy_Search":            Plan(method="F32", transformer=False),
    "Whisper_Argmax":                   Plan(method="F32", transformer=False),
    "Whisper_First_Beam_Search":        Plan(method="F32", transformer=False),
    "Whisper_Second_Beam_Search":       Plan(method="F32", transformer=False),
    "Whisper_Apply_Penality":           Plan(method="F32", transformer=False),
    "Whisper_No_Speech_Detection":      Plan(method="F32", transformer=False),
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
    f16_max_finite_val=65504.0,
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)

    # ── Move the bundled tokenizer folder from the export folder to the optimized folder so the
    # optimized folder (the default inference target) runs inference stand-alone. ──
    _optimized_dir = Path(OPTIMIZED_FOLDER_PATH)
    for _asset in ("tokenizer",):
        _src = Path(ORIGINAL_FOLDER_PATH) / _asset
        if not _src.exists():
            print(f"[Tokenizer] Skipped {_asset} (not found in {ORIGINAL_FOLDER_PATH})")
            continue
        try:
            _optimized_dir.mkdir(parents=True, exist_ok=True)
            _dst = _optimized_dir / _asset
            if _dst.exists():
                shutil.rmtree(_dst) if _dst.is_dir() else _dst.unlink()
            shutil.move(str(_src), str(_dst))
            print(f"[Tokenizer] Moved {_asset} -> {OPTIMIZED_FOLDER_PATH}")
        except Exception as _exc:  # noqa: BLE001
            print(f"[Tokenizer] Skipped {_asset} ({_exc})")
