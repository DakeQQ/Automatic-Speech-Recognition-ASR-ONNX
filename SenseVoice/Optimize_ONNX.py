"""Optimize & quantize the exported SenseVoice ONNX module."""

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


ORIGINAL_FOLDER_PATH = str(_SCRIPT_DIR / "SenseVoice_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "SenseVoice_Optimized")

FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

SENSEVOICE_NUM_HEADS = 4
SENSEVOICE_HIDDEN_SIZE = 512

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]


# ============================== MODEL PLANS ==============================

MODEL_PLANS = {
    "SenseVoice_Metadata": Plan(method="F32", transformer=False),
    "SenseVoiceSmall": Plan(
        method="DYNAMIC",
        opt_level=2,
        num_heads=SENSEVOICE_NUM_HEADS,
        hidden_size=SENSEVOICE_HIDDEN_SIZE,
        nodes_to_exclude=collect_quant_unsafe_nodes,
    ),
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
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)

    # ── Move the tokenizer model (SentencePiece BPE) from the export folder to the optimized folder
    # so the optimized folder (the default inference target) runs inference stand-alone. ──
    _optimized_dir = Path(OPTIMIZED_FOLDER_PATH)
    for _asset in ("chn_jpn_yue_eng_ko_spectok.bpe.model",):
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
