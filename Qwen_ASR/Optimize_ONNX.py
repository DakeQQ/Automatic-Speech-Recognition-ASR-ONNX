"""Optimize & quantize the exported Qwen3-ASR ONNX modules.

Config-only front-end: this script only defines the per-module Plans and the shared
OptimizerConfig, then delegates the whole quantize/optimize pipeline to
``Optimize_ONNX_Common.py`` (the same structure the other ASR export scripts use).

Each module in MODEL_PLANS picks a method; a Plan field left ``None`` inherits the
matching OptimizerConfig default. Plans are resolved and validated up front, so an
incompatible combination fails before any file is written.

    Method       Backend                   Result
    "Q2/Q4/Q8"   matmul_nbits_quantizer    2/4/8-bit weight-only (MatMulNBits)
    "DYNAMIC"    quantize_dynamic          INT8 dynamic (DynamicQuantizeLinear)
    "F16"        convert_float_to_float16  float16 weights & activations
    "F32"        -                         keep float32 (optimize only)
"""

from pathlib import Path
import shutil
import sys

from onnx import TensorProto


# ============================== SHARED PIPELINE =========================

# Reuse the shared optimizer pipeline: walk up to the repo root that holds it.
_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, metadata_int_for_model, run_optimizer


# ============================== USER CONFIG ==============================

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Qwen_ASR_ONNX")       # Source *.onnx modules.
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Qwen_ASR_Optimized")  # Destination folder.

FORCE_EXTERNAL_DATA   = False   # two-part storage (*.onnx.data); auto-forced when >2GB
UPGRADE_OPSET         = 0       # target ONNX opset (0 = keep current)

F16_OP_BLOCK_LIST = [           # op types kept out of any float16 conversion
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
]


# ========================== METADATA GEOMETRY ===========================

def qwen_asr_num_heads(model_path: str) -> int:
    if "Encoder" in Path(model_path).stem:
        return metadata_int_for_model(model_path, "audio_encoder_attention_heads")
    return metadata_int_for_model(model_path, "num_attention_heads")


def qwen_asr_hidden_size(model_path: str) -> int:
    if "Encoder" in Path(model_path).stem:
        return metadata_int_for_model(model_path, "audio_encoder_d_model")
    return metadata_int_for_model(model_path, "hidden_size")


# ============================== MODEL PLANS =============================

# Per-module plan. Comment out a line to skip that module. Heavy graphs are weight-quantized
# (embedding Gather, decoder/encoder MatMul); helper graphs stay float32, optimize-only.
# num_heads/hidden_size feed the attention-fusion optimizer from ASR_Matadata.onnx.
MODEL_PLANS: dict[str, Plan] = {
    "Qwen3_ASR_Encoder":                  Plan(method="Q4", external=True, num_heads=qwen_asr_num_heads, hidden_size=qwen_asr_hidden_size),
    "Qwen3_ASR_Decoder_Embed":            Plan(method="Q4", algo="DEFAULT", block_size=16, op_types=("Gather",), axes=(1,), external=True),
    "Qwen3_ASR_Decoder_Main":             Plan(method="Q4", external=True, num_heads=qwen_asr_num_heads, hidden_size=qwen_asr_hidden_size),
    "Qwen3_ASR_Rotary_Mask_Text_Prefill": Plan(method="F32", transformer=False),
    "Qwen3_ASR_Rotary_Mask_Text_Decode":  Plan(method="F32", transformer=False),
    "Qwen3_ASR_Greedy_Search":            Plan(method="F32", transformer=False),
    "Qwen3_ASR_First_Beam_Search":        Plan(method="F32", transformer=False),
    "Qwen3_ASR_Second_Beam_Search":       Plan(method="F32", transformer=False),
    "Qwen3_ASR_Apply_Penalty":            Plan(method="F32", transformer=False),
    "Qwen3_ASR_Argmax":                   Plan(method="F32", transformer=False),
    "Qwen3_ASR_Concat_Embed":             Plan(method="F32", transformer=False),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=2,
    # Export emits ORT's fused SimplifiedLayerNormalization in the default domain, which
    # onnx.shape_inference can't type-propagate; this fallback keeps quantize_dynamic from
    # aborting should any heavy module be switched to the DYNAMIC method.
    dynamic_default_tensor_type=TensorProto.FLOAT,
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
