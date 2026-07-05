"""Optimize & quantize the exported Whisper ONNX modules."""

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


MODEL_PATH = r"/home/DakeQQ/Downloads/whisper-large-v3-turbo"
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


def _load_whisper_config():
    if MODEL_PATH is None or MODEL_PATH.lower() == "none":
        return None
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)


def whisper_num_heads(model_path: str) -> int:
    config = _load_whisper_config()
    if config is None:
        return 0
    if "Encoder" in model_path:
        return int(getattr(config, "encoder_attention_heads", 0) or getattr(config, "num_attention_heads", 0))
    return int(getattr(config, "decoder_attention_heads", 0) or getattr(config, "num_attention_heads", 0))


def whisper_hidden_size(model_path: str) -> int:
    config = _load_whisper_config()
    if config is None:
        return 0
    return int(getattr(config, "d_model", 0) or getattr(config, "hidden_size", 0))


# ============================== MODEL PLANS ==============================

TRANSFORMER_PLAN = dict(num_heads=whisper_num_heads, hidden_size=whisper_hidden_size)

MODEL_PLANS = {
    "Whisper_Metadata":                 Plan(method="F32", transformer=False),
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
