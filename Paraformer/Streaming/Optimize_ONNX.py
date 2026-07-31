"""Optimize & quantize the exported Paraformer streaming ONNX modules."""

from pathlib import Path
import copy
import sys

import onnx

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
# Q8 is the direct-script default. Campaign profiles can select Q4, dynamic
# INT8, or the validated literal-F16 exporter graph without changing its I/O.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Paraformer_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Paraformer_Optimized")

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
    "Paraformer_Streaming_Encoder": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
    ),
    "Paraformer_Streaming_Decoder": Plan(
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
    copy_artifacts=("Vocab_Paraformer.txt",),
)

# ============================ END USER CONFIG ============================


def _expose_f16_encoder_bridge_outputs(model_path: Path) -> int:
    """Keep encoder internals F32 while exposing F16 decoder bridge tensors."""
    model = onnx.load(str(model_path), load_external_data=False)
    graph_outputs = {value.name: value for value in model.graph.output}
    converted = 0
    for output_name in ("encoder_out", "list_frame"):
        output = graph_outputs[output_name]
        internal_name = output_name + "_internal_f32"
        producer = next(
            node for node in model.graph.node if output_name in node.output
        )
        producer.output[:] = [
            internal_name if name == output_name else name
            for name in producer.output
        ]
        for node in model.graph.node:
            node.input[:] = [
                internal_name if name == output_name else name
                for name in node.input
            ]
        internal_info = copy.deepcopy(output)
        internal_info.name = internal_name
        model.graph.value_info.append(internal_info)
        output.type.tensor_type.elem_type = onnx.TensorProto.FLOAT16
        model.graph.node.append(
            onnx.helper.make_node(
                "Cast",
                [internal_name],
                [output_name],
                name=f"Paraformer_Streaming_{output_name}_To_Float16",
                to=onnx.TensorProto.FLOAT16,
            )
        )
        converted += 1
    onnx.save_model(model, str(model_path), save_as_external_data=False)
    return converted


def postprocess_model(name, _plan, output_path: Path) -> None:
    if (
        name == "Paraformer_Streaming_Encoder"
        and MODEL_PLANS["Paraformer_Streaming_Encoder"].method.upper() == "F32"
        and MODEL_PLANS["Paraformer_Streaming_Decoder"].method.upper() == "F16"
    ):
        converted = _expose_f16_encoder_bridge_outputs(output_path)
        print(f"  Exposed {converted} F16 encoder-to-decoder bridge output(s).")


def main() -> None:
    run_optimizer(
        CONFIG,
        model_names=(
            "Paraformer_Streaming_Encoder",
            "Paraformer_Streaming_Decoder",
        ),
        after_model=postprocess_model,
    )
    storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER_PATH,
        "Paraformer_Streaming_SharedInitializers.onnx",
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
