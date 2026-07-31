"""Optimize & quantize the exported SenseVoice ONNX module."""

from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import (
    consolidate_optimized_model_weights,
    DEFAULT_F16_OP_BLOCK_LIST,
    OptimizerConfig,
    Plan,
    run_optimizer,
)


# SenseVoice receives int16 PCM directly. On the real normalized demo, the
# folded DFT Conv reaches about 5.6e5 and its square reaches about 3.1e11, so a
# literal F16 front end overflows before the Mel projection. Keep only the exact
# Conv -> square/Split/Add -> Mel MatMul -> Clip -> Log chain in F32. The F16
# converter inserts one int16->F32 input Cast and one F32->F16 Cast after Log;
# all encoder/CTC compute remains F16.
_F16_FRONTEND_GUARD_OPS = {
    "/Conv": "Conv",
    "/Mul": "Mul",
    "/Split": "Split",
    "/Add": "Add",
    "/Transpose": "Transpose",
    "/MatMul": "MatMul",
    "/Clip": "Clip",
    "/Log": "Log",
}
SENSEVOICE_F16_GUARD_NODE_NAMES = tuple(_F16_FRONTEND_GUARD_OPS)


# ============================== USER CONFIG ==============================
# Edit this section only.
# F16 is validated with the fixed F32 front-end guard above. F32 remains the
# safest fallback; low-bit plans require separate representative validation.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "SenseVoice_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "SenseVoice_Optimized")

WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET       = 0
OPTIMIZER_LEVEL     = 2
OPTIMIZER_ONLY_ONNXRUNTIME = False

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_OP_BLOCK_LIST = DEFAULT_F16_OP_BLOCK_LIST


MODEL_PLANS = {
    "SenseVoiceSmall": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
    ),
    "ASR_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
    ),
}


CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=DYNAMIC_REDUCE_RANGE,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_force_initializers=F16_FORCE_INITIALIZERS,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    f16_node_block_list=list(SENSEVOICE_F16_GUARD_NODE_NAMES),
    copy_artifacts=("chn_jpn_yue_eng_ko_spectok.bpe.model",),
)

# ============================ END USER CONFIG ============================


def main() -> None:
    run_optimizer(CONFIG, model_names=("SenseVoiceSmall",))
    storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER_PATH,
        "SenseVoice_SharedInitializers.onnx",
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
