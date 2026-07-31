"""Optimize & quantize the exported X-ASR streaming ONNX modules."""

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
    collect_quant_unsafe_nodes,
    run_optimizer,
)

# ============================== USER CONFIG ==============================
# Edit this section only.
# Q8 is the direct-script default. Keep collect_quant_unsafe_nodes for every
# low-precision profile of the encoder, decoder, and joiner.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "X_ASR_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "X_ASR_Optimized")

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
    "X_ASR_Encoder": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
        nodes_to_exclude=collect_quant_unsafe_nodes,
    ),
    "X_ASR_Decoder": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
        nodes_to_exclude=collect_quant_unsafe_nodes,
    ),
    "X_ASR_Joiner": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
        nodes_to_exclude=collect_quant_unsafe_nodes,
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
    copy_artifacts=("tokens.txt",),
)

# ============================ END USER CONFIG ============================


def main() -> None:
    run_optimizer(
        CONFIG,
        model_names=("X_ASR_Encoder", "X_ASR_Decoder", "X_ASR_Joiner"),
    )
    storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER_PATH,
        "X_ASR_SharedInitializers.onnx",
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
