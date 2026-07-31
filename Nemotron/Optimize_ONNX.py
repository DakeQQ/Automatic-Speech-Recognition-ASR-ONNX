"""Optimize & quantize the exported Nemotron ASR ONNX modules (auto-adapts to offline & streaming)."""

import copy
from pathlib import Path
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
# Edit this section only. Every present target is optimized.
# Q8 is the direct-script default for every present graph.

WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 1
OPTIMIZER_FUSION_OPTIONS   = {
    "enable_skip_layer_norm": False,
    "enable_bias_skip_layer_norm": False,
}

F16_KEEP_IO_TYPES = False
F16_FORCE_INITIALIZERS = False
F16_MAX_FINITE_VALUE = 32767.0
F16_OP_BLOCK_LIST     = DEFAULT_F16_OP_BLOCK_LIST
SLIM_SKIP_FUSION_PATTERNS = ["FusionGemm"]
COPY_ARTIFACTS            = ("tokenizer.model", "vocab.txt", "model_config.yaml")

OFFLINE_MODEL_PLANS = {
    "Nemotron_ASR_Encoder": Plan(
        method="Q8",
        external=False,
        num_heads=0,
        hidden_size=0,
        transformer=True,
    ),
    "Nemotron_ASR_Decoder": Plan(
        method="Q8",
        external=False,
        transformer=True,
    ),
    "ASR_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
    ),
}

STREAMING_MODEL_PLANS = {
    "Nemotron_ASR_Streaming_Encoder": Plan(
        method="Q8",
        external=False,
        num_heads=0,
        hidden_size=0,
        transformer=True,
    ),
    "Nemotron_ASR_Streaming_Decoder": Plan(
        method="Q8",
        external=False,
        transformer=True,
    ),
    "ASR_Metadata": Plan(
        method="F32",
        optimize=False,
        transformer=False,
    ),
}

OFFLINE_OPTIMIZER_MODEL_NAMES = (
    "Nemotron_ASR_Encoder",
    "Nemotron_ASR_Decoder",
)
STREAMING_OPTIMIZER_MODEL_NAMES = (
    "Nemotron_ASR_Streaming_Encoder",
    "Nemotron_ASR_Streaming_Decoder",
)

OFFLINE_CONFIG = OptimizerConfig(
    original_folder_path=str(_SCRIPT_DIR / "Nemotron_ASR_ONNX"),
    optimized_folder_path=str(_SCRIPT_DIR / "Nemotron_ASR_Optimized"),
    model_plans=OFFLINE_MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=DYNAMIC_REDUCE_RANGE,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    optimizer_fusion_options=OPTIMIZER_FUSION_OPTIONS,
    f16_max_finite_val=F16_MAX_FINITE_VALUE,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    slim_skip_fusion_patterns=SLIM_SKIP_FUSION_PATTERNS,
    copy_artifacts=COPY_ARTIFACTS,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_force_initializers=F16_FORCE_INITIALIZERS,
)

STREAMING_CONFIG = OptimizerConfig(
    original_folder_path=str(_SCRIPT_DIR / "Nemotron_ASR_Streaming_ONNX"),
    optimized_folder_path=str(_SCRIPT_DIR / "Nemotron_ASR_Streaming_Optimized"),
    model_plans=STREAMING_MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=DYNAMIC_REDUCE_RANGE,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    optimizer_fusion_options=OPTIMIZER_FUSION_OPTIONS,
    f16_max_finite_val=F16_MAX_FINITE_VALUE,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    slim_skip_fusion_patterns=SLIM_SKIP_FUSION_PATTERNS,
    copy_artifacts=COPY_ARTIFACTS,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_force_initializers=F16_FORCE_INITIALIZERS,
)

TARGETS = (
    ("offline", OFFLINE_CONFIG, OFFLINE_OPTIMIZER_MODEL_NAMES),
    ("streaming", STREAMING_CONFIG, STREAMING_OPTIMIZER_MODEL_NAMES),
)

# ============================ END USER CONFIG ============================


def _is_present(config: OptimizerConfig) -> bool:
    return (Path(config.original_folder_path) / "ASR_Metadata.onnx").exists()


def restore_dynamic_relative_shift(source_path: Path, output_path: Path) -> int:
    """Restore 24 dynamic relative-position Reshapes removed as false no-ops."""
    source = onnx.load(str(source_path), load_external_data=False)
    model = onnx.load(str(output_path), load_external_data=False)
    source_producers = {
        output: node
        for node in source.graph.node
        for output in node.output
        if output
    }
    source_consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in source.graph.node:
        for input_name in node.input:
            if input_name:
                source_consumers.setdefault(input_name, []).append(node)

    candidates = []
    for reshape in source.graph.node:
        if reshape.op_type != "Reshape" or len(reshape.input) != 2 or len(reshape.output) != 1:
            continue
        producer = source_producers.get(reshape.input[0])
        consumers = source_consumers.get(reshape.output[0], ())
        if (
            producer is not None
            and producer.op_type == "Slice"
            and len(consumers) == 1
            and consumers[0].op_type == "Slice"
        ):
            candidates.append((reshape, consumers[0]))
    if not candidates:
        return 0
    if len(candidates) != 24:
        raise RuntimeError(
            f"Expected 24 Nemotron relative-shift Reshapes, found {len(candidates)}."
        )

    nodes = list(model.graph.node)
    by_name = {node.name: node for node in nodes if node.name}
    existing_outputs = {
        value.name for value in (*model.graph.input, *model.graph.output)
    }
    existing_outputs.update(initializer.name for initializer in model.graph.initializer)
    existing_outputs.update(output for node in nodes for output in node.output if output)
    source_initializers = {
        initializer.name: initializer for initializer in source.graph.initializer
    }
    model_initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    allowed_shape_ops = {
        "Add", "Cast", "Concat", "Constant", "Div", "Gather", "Mul",
        "Shape", "Squeeze", "Sub", "Unsqueeze",
    }
    required_outputs: set[str] = set()

    def require_shape_tensor(tensor_name: str) -> None:
        if tensor_name in existing_outputs or tensor_name in required_outputs:
            return
        initializer = source_initializers.get(tensor_name)
        if initializer is not None:
            model.graph.initializer.append(copy.deepcopy(initializer))
            model_initializers[tensor_name] = model.graph.initializer[-1]
            existing_outputs.add(tensor_name)
            return
        producer = source_producers.get(tensor_name)
        if producer is None or producer.op_type not in allowed_shape_ops:
            raise RuntimeError(
                f"Nemotron relative-shift shape tensor {tensor_name!r} is unavailable."
            )
        for input_name in producer.input:
            if input_name:
                require_shape_tensor(input_name)
        required_outputs.update(output for output in producer.output if output)

    for reshape, _ in candidates:
        require_shape_tensor(reshape.input[1])

    source_order = {
        id(node): index for index, node in enumerate(source.graph.node)
    }
    required_nodes = {
        id(source_producers[output]): source_producers[output]
        for output in required_outputs
        if output in source_producers
    }
    shape_nodes = sorted(
        required_nodes.values(),
        key=lambda node: source_order[id(node)],
    )
    existing_names = {node.name for node in nodes if node.name}
    for node in shape_nodes:
        copied = copy.deepcopy(node)
        if copied.name in existing_names:
            copied.name += "/RestoredRelativeShift"
        existing_names.add(copied.name)

    first_consumer_index = min(
        nodes.index(by_name[consumer.name])
        for _, consumer in candidates
    )
    copied_shape_nodes = []
    for node in shape_nodes:
        copied = copy.deepcopy(node)
        if copied.name in existing_names:
            copied.name += "/RestoredRelativeShift"
        existing_names.add(copied.name)
        copied_shape_nodes.append(copied)
    nodes[first_consumer_index:first_consumer_index] = copied_shape_nodes

    restored = 0
    for source_reshape, source_consumer in candidates:
        consumer = next(node for node in nodes if node.name == source_consumer.name)
        if consumer.input[0] == source_reshape.output[0]:
            continue
        if consumer.input[0] != source_reshape.input[0]:
            raise RuntimeError(
                f"Unexpected Nemotron relative-shift bypass at {consumer.name!r}."
            )
        insert_index = nodes.index(consumer)
        nodes.insert(insert_index, copy.deepcopy(source_reshape))
        consumer.input[0] = source_reshape.output[0]
        restored += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    del model.graph.value_info[:]
    onnx.save_model(model, str(output_path), save_as_external_data=False)
    return restored


def postprocess_model(name: str, _plan, output_path: Path, config: OptimizerConfig) -> None:
    if name.endswith("_Encoder"):
        source_path = Path(config.original_folder_path) / f"{name}.onnx"
        restored = restore_dynamic_relative_shift(source_path, output_path)
        print(f"  Restored {restored} Nemotron dynamic relative-shift Reshape(s).")


def main() -> None:
    for kind, config, model_names in TARGETS:
        if not _is_present(config):
            continue
        original = Path(config.original_folder_path)
        optimized = Path(config.optimized_folder_path)
        print(f"\n########## Optimizing {kind} models: {original.name} "
              f"-> {optimized.name} ##########")
        run_optimizer(
            config,
            model_names=model_names,
            after_model=lambda name, plan, output_path, config=config: postprocess_model(
                name, plan, output_path, config
            ),
        )
        storage = consolidate_optimized_model_weights(
            optimized,
            f"{original.name.removesuffix('_ONNX')}_SharedInitializers.onnx",
        )
        print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")


if __name__ == "__main__":
    main()
