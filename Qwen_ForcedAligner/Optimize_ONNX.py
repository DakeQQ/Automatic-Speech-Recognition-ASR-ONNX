"""Optimize the merged Qwen ForcedAligner NAR graph."""

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
    QUANTIZATION_F16_OP_BLOCK_LIST,
    consolidate_optimized_model_weights,
    copy_artifact,
    OptimizerConfig,
    Plan,
    run_optimizer,
)

# ============================== USER CONFIG ==============================
# Edit this section only.
# Q8 is the direct-script default. Set USE_FLOAT16 only for the validated
# mixed-F16 plan with frontend and residual precision repairs.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Qwen_ForcedAligner_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Qwen_ForcedAligner_Optimized")

OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = True
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 2
USE_FLOAT16                = False

WEIGHT_ONLY_ALGORITHM      = "AFFINE_REFINE_V2"
WEIGHT_ONLY_BLOCK_SIZE     = 32
WEIGHT_ONLY_ACCURACY_LEVEL = 4
WEIGHT_ONLY_SYMMETRIC      = False

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False

F16_OP_BLOCK_LIST = QUANTIZATION_F16_OP_BLOCK_LIST


# ============================== MODEL PLANS =============================

MODEL_PLANS = {
    "ForcedAligner_Merged": Plan(
        method="F16" if USE_FLOAT16 else "Q8",
        algo=None if USE_FLOAT16 else WEIGHT_ONLY_ALGORITHM,
        opt_level=2,
        num_heads=0,
        hidden_size=0,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
        run_second_slim=not USE_FLOAT16,
    ),
    "ForcedAligner_SharedInitializers": Plan(
        method="F32",
        optimize=False,
        transformer=False,
    ),
    "ASR_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
    ),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    block_size=WEIGHT_ONLY_BLOCK_SIZE,
    accuracy_level=WEIGHT_ONLY_ACCURACY_LEVEL,
    quant_symmetric=WEIGHT_ONLY_SYMMETRIC,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_fusion_options=(
        {
            "enable_skip_layer_norm": False,
            "enable_bias_skip_layer_norm": False,
        }
        if USE_FLOAT16
        else None
    ),
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
)

# ============================ END USER CONFIG ============================


def main() -> None:
    run_optimizer(
        CONFIG,
        model_names=("ForcedAligner_Merged",),
        reset_output_folder=True,
    )
    copy_artifact(
        Path(ORIGINAL_FOLDER_PATH) / "tokenizer",
        Path(OPTIMIZED_FOLDER_PATH) / "tokenizer",
        required=True,
    )
    storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER_PATH,
        "ForcedAligner_SharedInitializers.onnx",
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
