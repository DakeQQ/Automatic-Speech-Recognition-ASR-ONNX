"""Optimize & quantize the exported Paraformer ONNX module."""

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


# Paraformer's int16-range Kaldi front end cannot square its unscaled DFT output
# in F16 (the real demo reaches about 8e11). The encoder residual also crosses
# 65504 between Add_96 and Add_99, while FFN projections 36..38 independently
# exceed the F16 range. Blocking these exact nodes makes the converter insert
# small F16->F32->F16 precision islands; the remaining MatMuls stay F16.
#
# The residual names below are post-transformer-fusion names. Source and final
# topology are both checked fail-closed so optimizer/name drift cannot silently
# remove a required Cast boundary.
_F16_FRONTEND_GUARD_OPS = {
    "/fbank_model/Conv": "Conv",
    "/fbank_model/Mul": "Mul",
    "/fbank_model/Split": "Split",
    "/fbank_model/Add": "Add",
    "/fbank_model/MatMul": "MatMul",
    "/fbank_model/Max": "Max",
    "/fbank_model/Log": "Log",
}
_F16_RESIDUAL_GUARD_NAMES = (
    *(f"SkipLayerNorm_AddBias_{index}" for index in range(32, 51)),
    *(f"SkipLayerNorm_{index}" for index in range(65, 100, 2)),
)
_F16_FFN_GUARD_NAMES = tuple(
    f"/w_2_{index}/MatMul" for index in range(36, 39)
)
PARAFORMER_F16_GUARD_NODE_NAMES = (
    *_F16_FRONTEND_GUARD_OPS,
    *_F16_RESIDUAL_GUARD_NAMES,
    *_F16_FFN_GUARD_NAMES,
)


# ============================== USER CONFIG ==============================
# Edit this section only.
# Q8 is the direct-script default; campaign profiles select the other methods.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Paraformer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Paraformer_Optimized")

WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

F16_KEEP_IO_TYPES = False
F16_FORCE_INITIALIZERS = False

FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET       = 0
OPTIMIZER_LEVEL     = 1
OPTIMIZER_ONLY_ONNXRUNTIME = False
F16_OP_BLOCK_LIST = DEFAULT_F16_OP_BLOCK_LIST

MODEL_PLANS = {
    "Paraformer": Plan(
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
    f16_op_block_list=F16_OP_BLOCK_LIST,
    f16_node_block_list=list(PARAFORMER_F16_GUARD_NODE_NAMES),
    copy_artifacts=("Vocab_Paraformer.txt",),
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_force_initializers=F16_FORCE_INITIALIZERS,
)

# ============================ END USER CONFIG ============================


def main() -> None:
    run_optimizer(CONFIG, model_names=("Paraformer",))
    storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER_PATH,
        "Paraformer_SharedInitializers.onnx",
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
