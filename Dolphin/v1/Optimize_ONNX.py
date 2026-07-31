"""Optimize Dolphin v1 and rebuild its selected-method shared/merged bundle.

The v1 and CN-Dialect decoder artifacts use the same graph ABI. Reuse the
canonical optimizer implementation while supplying v1 paths and tokenizer
artifacts; only one merged Main donor is optimized and transplanted. Q4 is the
default; F16 conversion controls and precision repairs apply only when F16 is
explicitly selected.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
IMPLEMENTATION = SCRIPT_DIR.parent / "CN-Dialect" / "Optimize_ONNX.py"
SPEC = importlib.util.spec_from_file_location("_dolphin_merged_optimizer", IMPLEMENTATION)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load Dolphin merged optimizer: {IMPLEMENTATION}")
optimizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimizer)

from Optimize_ONNX_Common import OptimizerConfig, Plan, producer_ancestry_node_names


def exclude_v1_encoder_frontend_nodes(model_path: str) -> list[str]:
    """Keep the v1 frontend in float32 when the selected method converts activations."""
    return producer_ancestry_node_names(
        model_path,
        "/embed/Gemm_output_0",
        graph_label="Dolphin v1 encoder F32 frontend",
    )

# ============================== USER CONFIG ==============================
# Edit this section only. These values override the canonical CN optimizer.
# Q4 is selected below; other optimizer defaults/plans are inherited from
# CN-Dialect. The dynamic relative-shift repair is always retained, while the
# v1 F16 frontend/precision repairs are active only if F16 is selected.

ORIGINAL_FOLDER_PATH  = str(SCRIPT_DIR / "Dolphin_ONNX")
OPTIMIZED_FOLDER_PATH = str(SCRIPT_DIR / "Dolphin_Optimized")
DEFAULT_METHOD        = "Q8"

WEIGHT_ONLY_ALGORITHM      = "AFFINE_REFINE_V2"
OPTIMIZER_ONLY_ONNXRUNTIME = optimizer.OPTIMIZER_ONLY_ONNXRUNTIME
FORCE_EXTERNAL_DATA        = optimizer.FORCE_EXTERNAL_DATA
UPGRADE_OPSET              = optimizer.UPGRADE_OPSET
OPTIMIZER_LEVEL            = 1

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
COPY_ARTIFACTS = (
    "vocab_Dolphin.txt",
    "bpe.model",
)

# Each v1 graph owns its optimization plan. The CN module contributes only the
# shared implementation functions and common tuning values.
MODEL_PLANS = {
    "Dolphin_Encoder": Plan(
        method=DEFAULT_METHOD,
        num_heads=0,
        hidden_size=0,
        external=True,
        nodes_to_exclude=exclude_v1_encoder_frontend_nodes,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "Dolphin_Decoder": Plan(
        method=DEFAULT_METHOD,
        optimize=True,
        transformer=False,
        num_heads=0,
        hidden_size=0,
        external=True,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "Dolphin_ProbePrefillGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_ProbePrefillPenaltyGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_ProbePrefillSampling": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_PrefillGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_PrefillPenaltyGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_PrefillSampling": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_DecodeGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_DecodePenaltyGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_DecodeSampling": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "Dolphin_SharedInitializers": Plan(method=DEFAULT_METHOD, process=False, optimize=False, transformer=False),
    "ASR_Metadata": Plan(method="F32", process=False, optimize=False, transformer=False),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    dynamic_weight_type=optimizer.DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=optimizer.DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=optimizer.DYNAMIC_REDUCE_RANGE,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_op_block_list=optimizer.F16_OP_BLOCK_LIST,
    copy_artifacts=COPY_ARTIFACTS,
)

# ============================ END USER CONFIG ============================

optimizer.ORIGINAL_FOLDER_PATH = ORIGINAL_FOLDER_PATH
optimizer.OPTIMIZED_FOLDER_PATH = OPTIMIZED_FOLDER_PATH
optimizer.MODEL_PLANS = MODEL_PLANS
optimizer.CONFIG = CONFIG
optimizer.PROBE_AWARE = True


def restore_v1_dynamic_relative_shift() -> int:
    """Restore dynamic rel-shift reshapes removed as false no-ops.

    Every encoder layer reshapes its padded relative logits back to the dynamic
    MatMul shape before the final Slice. Generic optimization incorrectly
    removes that Reshape, leaving $2T-1$ logits to be added to a $T$-wide score.
    Copy only the exact source Shape/Reshape pairs after strict topology checks.
    """
    source_path = Path(optimizer.ORIGINAL_FOLDER_PATH) / "Dolphin_Encoder.onnx"
    output_path = Path(optimizer.OPTIMIZED_FOLDER_PATH) / "Dolphin_Encoder.onnx"
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

    candidates: list[tuple[onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]] = []
    for reshape in source.graph.node:
        if (
            reshape.domain not in ("", "ai.onnx")
            or reshape.op_type != "Reshape"
            or len(reshape.input) != 2
            or len(reshape.output) != 1
        ):
            continue
        data_slice = source_producers.get(reshape.input[0])
        shape = source_producers.get(reshape.input[1])
        consumers = source_consumers.get(reshape.output[0], [])
        if not (
            data_slice is not None
            and data_slice.op_type == "Slice"
            and shape is not None
            and shape.domain in ("", "ai.onnx")
            and shape.op_type == "Shape"
            and len(shape.input) == 1
            and len(shape.output) == 1
            and len(consumers) == 1
            and consumers[0].domain in ("", "ai.onnx")
            and consumers[0].op_type == "Slice"
            and consumers[0].input[0] == reshape.output[0]
        ):
            continue
        candidates.append((shape, reshape, consumers[0]))

    expected = len(
        [
            value
            for value in source.graph.output
            if value.name.startswith("en_key_layer_")
        ]
    )
    if expected <= 0:
        raise RuntimeError("Dolphin v1 Encoder exposes no cross-key layer outputs.")
    if len(candidates) != expected:
        raise RuntimeError(
            f"Expected {expected} v1 relative-shift reshapes, found "
            f"{len(candidates)}."
        )

    nodes = list(model.graph.node)
    by_name: dict[str, onnx.NodeProto] = {}
    for node in nodes:
        if node.name:
            if node.name in by_name:
                raise RuntimeError(
                    f"Optimized v1 encoder has duplicate node name {node.name!r}."
                )
            by_name[node.name] = node

    restored = 0
    existing = 0
    for source_shape, source_reshape, source_consumer in candidates:
        shape = by_name.get(source_shape.name)
        reshape = by_name.get(source_reshape.name)
        consumer = by_name.get(source_consumer.name)
        if consumer is None or consumer.op_type != "Slice":
            raise RuntimeError(
                f"Optimized v1 encoder is missing rel-shift consumer "
                f"{source_consumer.name!r}."
            )

        if shape is not None or reshape is not None:
            if not (
                shape is not None
                and reshape is not None
                and list(shape.input) == list(source_shape.input)
                and list(shape.output) == list(source_shape.output)
                and list(reshape.input) == list(source_reshape.input)
                and list(reshape.output) == list(source_reshape.output)
                and consumer.input[0] == source_reshape.output[0]
            ):
                raise RuntimeError(
                    f"Partial or incompatible v1 rel-shift repair near "
                    f"{source_reshape.name!r}."
                )
            existing += 1
            continue

        if consumer.input[0] != source_reshape.input[0]:
            raise RuntimeError(
                f"v1 rel-shift consumer {consumer.name!r} is not directly "
                f"bypassing {source_reshape.name!r}."
            )
        insert_index = nodes.index(consumer)
        shape = copy.deepcopy(source_shape)
        reshape = copy.deepcopy(source_reshape)
        nodes[insert_index:insert_index] = [shape, reshape]
        consumer.input[0] = reshape.output[0]
        by_name[shape.name] = shape
        by_name[reshape.name] = reshape
        restored += 1

    if restored and existing:
        raise RuntimeError(
            f"v1 rel-shift graph was only partially repaired: "
            f"restored={restored}, existing={existing}."
        )
    if restored:
        del model.graph.node[:]
        model.graph.node.extend(nodes)

    # Incorrect inferred shapes caused the no-op classification. Remove internal
    # declarations so ORT infers through the restored dynamic Shape/Reshape path.
    pruned = len(model.graph.value_info)
    if pruned:
        del model.graph.value_info[:]
    onnx.save(model, str(output_path))
    return restored


def main() -> None:
    original_builder = optimizer.build_quantized_merged_bundle

    def repaired_builder():
        restored = restore_v1_dynamic_relative_shift()
        print(
            f"  Restored {restored} dynamic relative-shift reshape(s) "
            "before Dolphin v1 Encoder transplantation."
        )
        return original_builder()

    optimizer.build_quantized_merged_bundle = repaired_builder
    optimizer.main()


if __name__ == "__main__":
    main()
