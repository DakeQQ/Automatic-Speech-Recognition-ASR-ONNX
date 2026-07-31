"""Optimize & quantize the exported Parakeet TDT ASR ONNX modules (offline)."""

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

# ============================== USER CONFIG ==============================
# Edit this section only.
# Q8 is the direct-script default for both independent runtime graphs.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Parakeet_ASR_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Parakeet_ASR_Optimized")
WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 2

F16_KEEP_IO_TYPES = False
F16_FORCE_INITIALIZERS = False
F16_MAX_FINITE_VALUE = 32767.0
F16_OP_BLOCK_LIST     = DEFAULT_F16_OP_BLOCK_LIST
SLIM_SKIP_FUSION_PATTERNS = ["FusionGemm"]
COPY_ARTIFACTS            = ("tokenizer.json", "tokenizer_config.json")

MODEL_PLANS = {
    # mel front-end + Fast-Conformer + encoder projector
    "Parakeet_ASR_Encoder": Plan(
        method="Q8",
        external=False,
        num_heads=0,
        hidden_size=0,
    ),
    # folded TDT decoder/joint + greedy step
    "Parakeet_ASR_Decoder": Plan(
        method="Q8",
        external=False,
        transformer=False,
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
    f16_max_finite_val=F16_MAX_FINITE_VALUE,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    slim_skip_fusion_patterns=SLIM_SKIP_FUSION_PATTERNS,
    copy_artifacts=COPY_ARTIFACTS,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_force_initializers=F16_FORCE_INITIALIZERS,
)

# ============================ END USER CONFIG ============================


def main() -> None:
    original_folder = Path(ORIGINAL_FOLDER_PATH)
    optimized_folder = Path(OPTIMIZED_FOLDER_PATH)
    print(
        f"\n########## Optimizing offline models: {original_folder.name} "
        f"-> {optimized_folder.name} ##########"
    )
    run_optimizer(
        CONFIG,
        model_names=("Parakeet_ASR_Encoder", "Parakeet_ASR_Decoder"),
    )
    storage = consolidate_optimized_model_weights(
        optimized_folder,
        "Parakeet_ASR_SharedInitializers.onnx",
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
