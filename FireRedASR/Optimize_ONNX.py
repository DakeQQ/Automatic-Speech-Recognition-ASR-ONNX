"""Convert FireRedASR Encoder/Main donors and rebuild Encoder+prefill graphs."""

from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path

import onnx
from onnx import numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

import Shared_Merged
from Optimize_ONNX_Common import (
    collect_target_only_shared_shell_initializers,
    consolidate_optimized_model_weights,
    copy_artifact,
    DEFAULT_F16_OP_BLOCK_LIST,
    normalize_float16_output_bridge,
    OptimizerConfig,
    Plan,
    producer_ancestry_node_names,
    remove_redundant_casts,
    run_optimizer,
    share_external_initializers_if_identical,
)


MAIN_STEM = Path(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["main"]).stem


# ============================== USER CONFIG ==============================
# Edit this section only.
# F32 is the numerical baseline, F16 is the validated deployment mode, DYNAMIC
# is portable INT8, and Q2/Q4/Q8 are block weight-only methods. Keep the
# FireRed frontend/relative-shift/logits protections when changing a method.

ORIGINAL_FOLDER_PATH  = str(SCRIPT_DIR / "FireRedASR_ONNX")
OPTIMIZED_FOLDER_PATH = str(SCRIPT_DIR / "FireRedASR_Optimized")

OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 2

WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = False
DYNAMIC_REDUCE_RANGE = False

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_MAX_FINITE_VALUE   = 32767.0
F16_OP_BLOCK_LIST = DEFAULT_F16_OP_BLOCK_LIST

MODEL_PLANS = {
    "FireRedASR_Encoder": Plan(
        method="Q8",
        num_heads=lambda path: _num_heads(path),
        hidden_size=lambda path: _hidden_size(path),
        optimize=True,
        transformer=True,
        external=False,
        nodes_to_exclude=lambda path: _exclude_encoder_frontend_nodes(path),
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "FireRedASR_Decoder": Plan(
        method="Q8",
        optimize=True,
        transformer=True,
        external=False,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "FireRedASR_PrefillGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FireRedASR_PrefillPenaltyGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FireRedASR_PrefillSampling": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FireRedASR_DecodeGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FireRedASR_DecodePenaltyGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FireRedASR_DecodeSampling": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FireRedASR_SharedInitializers": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "ASR_Metadata": Plan(method="F32", process=False, optimize=False, transformer=False),
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
    f16_max_finite_val=F16_MAX_FINITE_VALUE,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    copy_artifacts=(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["metadata"],),
)

# ============================ END USER CONFIG ============================


def _num_heads(model_path: str) -> int:
    model = onnx.load(model_path, load_external_data=False)
    if "Encoder" in Path(model_path).stem:
        outputs = list(model.graph.output)
        if not outputs:
            raise RuntimeError("FireRed Encoder has no cross-KV outputs.")
        value = outputs[0]
        axis = 0
        initializer = next(
            (
                item for item in model.graph.initializer
                if item.name.endswith("cross_kv_split_sizes")
            ),
            None,
        )
        if initializer is not None:
            split_sizes = numpy_helper.to_array(initializer).reshape(-1)
            if split_sizes.size == 2 and split_sizes[0] == split_sizes[1] > 0:
                return int(split_sizes[0])
    else:
        values = [
            item for item in model.graph.input
            if item.name.startswith("in_de_key_layer_")
        ]
        if not values:
            raise RuntimeError("FireRed Main has no self-KV key inputs.")
        value = values[0]
        axis = 1
    dim = value.type.tensor_type.shape.dim[axis]
    if not dim.HasField("dim_value") or dim.dim_value < 1:
        raise RuntimeError(f"Cannot derive attention heads from {value.name!r}.")
    return int(dim.dim_value)


def _hidden_size(model_path: str) -> int:
    model = onnx.load(model_path, load_external_data=False)
    names = ("hidden_states", "audio")
    value = next(
        (item for item in model.graph.input if item.name in names),
        None,
    )
    if value is not None and value.name == "hidden_states":
        dim = value.type.tensor_type.shape.dim[-1]
        if dim.HasField("dim_value") and dim.dim_value > 0:
            return int(dim.dim_value)
    if "Encoder" in Path(model_path).stem:
        shape_initializer = next(
            (
                item for item in model.graph.initializer
                if item.name.endswith("cross_kv_shape")
            ),
            None,
        )
        split_initializer = next(
            (
                item for item in model.graph.initializer
                if item.name.endswith("cross_kv_split_sizes")
            ),
            None,
        )
        if shape_initializer is not None and split_initializer is not None:
            shape = numpy_helper.to_array(shape_initializer).reshape(-1)
            split = numpy_helper.to_array(split_initializer).reshape(-1)
            if shape.size == 3 and split.size == 2 and shape[-1] > 0 and split[0] > 0:
                return int(shape[-1] * split[0])
        outputs = list(model.graph.output)
        if outputs:
            tensor_type = outputs[0].type.tensor_type
            heads = tensor_type.shape.dim[0]
            head_dim = tensor_type.shape.dim[1]
            if (
                heads.HasField("dim_value")
                and head_dim.HasField("dim_value")
                and heads.dim_value > 0
                and head_dim.dim_value > 0
            ):
                return int(heads.dim_value * head_dim.dim_value)
    raise RuntimeError(f"Cannot derive hidden size from {Path(model_path).name} I/O.")


def _exclude_encoder_frontend_nodes(model_path: str) -> list[str]:
    """Keep int16-range fbank and the subsampling projection in float32."""
    # F16 squaring of the Conv-STFT output overflows for int16-range PCM. Keep
    # its exact producer ancestry, including CMVN and Conv2d subsampling, in F32.
    return [
        *producer_ancestry_node_names(
        model_path,
        "/encoder/input_preprocessor/out/Add_output_0",
        graph_label="FireRedASR encoder F32 frontend",
        ),
        "/encoder/layer_stack.0/ffn1/net/net.0/LayerNormalization",
    ]


def _persist(model, path: Path) -> None:
    simplified = remove_redundant_casts(model)
    if simplified:
        print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
    Shared_Merged.prune_unreachable_nodes(model)
    Shared_Merged.save_model(model, path)
    print(f"  {path.name} ({path.stat().st_size} bytes)")


def _copy_file(source: Path, target: Path) -> None:
    copy_artifact(source, target)


def _constant_tensor_for_value(
    value_name: str,
    initializers: dict[str, onnx.TensorProto],
    producers: dict[str, onnx.NodeProto],
) -> onnx.TensorProto | None:
    initializer = initializers.get(value_name)
    if initializer is not None:
        return initializer
    producer = producers.get(value_name)
    if not (
        producer is not None
        and producer.domain in ("", "ai.onnx")
        and producer.op_type == "Constant"
        and not producer.input
    ):
        return None
    values = [
        attribute.t
        for attribute in producer.attribute
        if attribute.name == "value"
        and attribute.type == onnx.AttributeProto.TENSOR
    ]
    return values[0] if len(values) == 1 else None


def _same_tensor_value(left: onnx.TensorProto, right: onnx.TensorProto) -> bool:
    if left.data_type != right.data_type or tuple(left.dims) != tuple(right.dims):
        return False
    try:
        left_array = numpy_helper.to_array(left)
        right_array = numpy_helper.to_array(right)
    except (OSError, RuntimeError, ValueError):
        return False
    return (
        left_array.dtype == right_array.dtype
        and left_array.shape == right_array.shape
        and left_array.tobytes() == right_array.tobytes()
    )


def _relative_shift_shape_is_compatible(
    source_shape: onnx.NodeProto,
    optimized_shape: onnx.NodeProto,
    source_initializers: dict[str, onnx.TensorProto],
    optimized_initializers: dict[str, onnx.TensorProto],
    source_producers: dict[str, onnx.NodeProto],
    optimized_producers: dict[str, onnx.NodeProto],
) -> bool:
    """Allow only value-identical constant renames in the optimized shape node."""
    source_attributes = {
        attribute.name: attribute.SerializeToString()
        for attribute in source_shape.attribute
    }
    optimized_attributes = {
        attribute.name: attribute.SerializeToString()
        for attribute in optimized_shape.attribute
    }
    if not (
        source_shape.domain in ("", "ai.onnx")
        and optimized_shape.domain in ("", "ai.onnx")
        and source_shape.op_type == optimized_shape.op_type == "Concat"
        and list(source_shape.output) == list(optimized_shape.output)
        and len(source_shape.attribute) == len(source_attributes)
        and len(optimized_shape.attribute) == len(optimized_attributes)
        and source_attributes == optimized_attributes
        and len(source_shape.input) == len(optimized_shape.input)
    ):
        return False
    for source_input, optimized_input in zip(
        source_shape.input, optimized_shape.input
    ):
        if source_input == optimized_input:
            continue
        source_tensor = _constant_tensor_for_value(
            source_input, source_initializers, source_producers
        )
        optimized_tensor = _constant_tensor_for_value(
            optimized_input, optimized_initializers, optimized_producers
        )
        if (
            source_tensor is None
            or optimized_tensor is None
            or not _same_tensor_value(source_tensor, optimized_tensor)
        ):
            return False
    return True


def _restore_encoder_dynamic_relative_shift() -> int:
    """Restore dynamic relative-shift Reshapes removed as false no-ops."""
    source_path = Path(ORIGINAL_FOLDER_PATH) / "FireRedASR_Encoder.onnx"
    output_path = Path(OPTIMIZED_FOLDER_PATH) / "FireRedASR_Encoder.onnx"
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

    candidates: list[tuple[onnx.NodeProto, onnx.NodeProto]] = []
    shape_nodes: dict[str, onnx.NodeProto] = {}
    for reshape in source.graph.node:
        if not (
            reshape.domain in ("", "ai.onnx")
            and reshape.op_type == "Reshape"
            and len(reshape.input) == 2
            and len(reshape.output) == 1
            and reshape.name.startswith("/encoder/layer_stack.")
            and "/mhsa/" in reshape.name
        ):
            continue
        data_slice = source_producers.get(reshape.input[0])
        shape = source_producers.get(reshape.input[1])
        consumers = source_consumers.get(reshape.output[0], [])
        if not (
            data_slice is not None
            and data_slice.domain in ("", "ai.onnx")
            and data_slice.op_type == "Slice"
            and shape is not None
            and shape.domain in ("", "ai.onnx")
            and shape.op_type == "Concat"
            and len(shape.output) == 1
            and len(consumers) == 1
            and consumers[0].domain in ("", "ai.onnx")
            and consumers[0].op_type == "Slice"
            and consumers[0].input[0] == reshape.output[0]
        ):
            continue
        candidates.append((reshape, consumers[0]))
        shape_nodes[shape.output[0]] = shape

    main = onnx.load(
        str(
            Path(ORIGINAL_FOLDER_PATH)
            / Shared_Merged.DEFAULT_MODEL_FILE_NAMES["main"]
        ),
        load_external_data=False,
    )
    expected = len(
        [value for value in main.graph.input if value.name.startswith("en_key_layer_")]
    )
    if expected < 1:
        raise RuntimeError("Cannot derive FireRed layer count from Main cross-KV I/O.")
    if len(candidates) != expected or len(shape_nodes) != 1:
        raise RuntimeError(
            "Expected one shared shape and "
            f"{expected} FireRedASR relative-shift Reshapes, found "
            f"{len(shape_nodes)} and {len(candidates)}."
        )
    source_shape = next(iter(shape_nodes.values()))
    if any(reshape.input[1] != source_shape.output[0] for reshape, _ in candidates):
        raise RuntimeError("FireRedASR relative-shift Reshapes do not share one shape.")

    nodes = list(model.graph.node)
    by_name: dict[str, onnx.NodeProto] = {}
    for node in nodes:
        if not node.name:
            continue
        if node.name in by_name:
            raise RuntimeError(
                f"Optimized FireRedASR encoder has duplicate node name {node.name!r}."
            )
        by_name[node.name] = node

    optimized_producers = {
        output: node
        for node in nodes
        for output in node.output
        if output
    }
    source_initializers = {
        initializer.name: initializer for initializer in source.graph.initializer
    }
    model_initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    existing_shape = by_name.get(source_shape.name)
    existing_reshapes = [by_name.get(reshape.name) for reshape, _ in candidates]
    existing_count = sum(node is not None for node in existing_reshapes)
    if existing_count not in (0, expected):
        raise RuntimeError(
            "FireRedASR relative-shift graph is only partially repaired: "
            f"{existing_count}/{expected} Reshapes exist."
        )
    if existing_count == expected:
        if existing_shape is None or not _relative_shift_shape_is_compatible(
            source_shape,
            existing_shape,
            source_initializers,
            model_initializers,
            source_producers,
            optimized_producers,
        ):
            raise RuntimeError("Existing FireRedASR relative-shift shape is incompatible.")
        for (source_reshape, source_consumer), reshape in zip(
            candidates, existing_reshapes
        ):
            consumer = by_name.get(source_consumer.name)
            if not (
                reshape is not None
                and reshape.SerializeToString() == source_reshape.SerializeToString()
                and consumer is not None
                and consumer.op_type == "Slice"
                and consumer.input[0] == source_reshape.output[0]
            ):
                raise RuntimeError(
                    f"Existing relative-shift repair near {source_reshape.name!r} "
                    "is incompatible."
                )
        restored = 0
    else:
        if existing_shape is not None:
            raise RuntimeError(
                "FireRedASR relative-shift shape exists without all Reshapes."
            )

        graph_inputs = {value.name for value in model.graph.input}
        for input_name in source_shape.input:
            source_initializer = source_initializers.get(input_name)
            if source_initializer is not None:
                existing_initializer = model_initializers.get(input_name)
                if existing_initializer is None:
                    copied = copy.deepcopy(source_initializer)
                    model.graph.initializer.append(copied)
                    model_initializers[input_name] = copied
                elif (
                    existing_initializer.data_type != source_initializer.data_type
                    or list(existing_initializer.dims) != list(source_initializer.dims)
                ):
                    raise RuntimeError(
                        f"Relative-shift initializer {input_name!r} is incompatible."
                    )
            elif input_name not in graph_inputs and input_name not in optimized_producers:
                raise RuntimeError(
                    f"Relative-shift shape input {input_name!r} is unavailable."
                )

        consumers: list[onnx.NodeProto] = []
        for source_reshape, source_consumer in candidates:
            consumer = by_name.get(source_consumer.name)
            if consumer is None or consumer.op_type != "Slice":
                raise RuntimeError(
                    f"Optimized encoder is missing relative-shift consumer "
                    f"{source_consumer.name!r}."
                )
            if consumer.input[0] != source_reshape.input[0]:
                raise RuntimeError(
                    f"Relative-shift consumer {consumer.name!r} is not directly "
                    f"bypassing {source_reshape.name!r}."
                )
            if source_reshape.output[0] in optimized_producers:
                raise RuntimeError(
                    f"Relative-shift output {source_reshape.output[0]!r} is occupied."
                )
            consumers.append(consumer)

        shape_insert_index = min(nodes.index(consumer) for consumer in consumers)
        for input_name in source_shape.input:
            producer = optimized_producers.get(input_name)
            if producer is not None and nodes.index(producer) >= shape_insert_index:
                raise RuntimeError(
                    f"Relative-shift shape input {input_name!r} is not topologically ready."
                )
        shape_copy = copy.deepcopy(source_shape)
        nodes.insert(shape_insert_index, shape_copy)
        for (source_reshape, _), consumer in zip(candidates, consumers):
            insert_index = nodes.index(consumer)
            reshape_copy = copy.deepcopy(source_reshape)
            nodes.insert(insert_index, reshape_copy)
            consumer.input[0] = source_reshape.output[0]

        del model.graph.node[:]
        model.graph.node.extend(nodes)
        restored = expected

    pruned = len(model.graph.value_info)
    if pruned:
        del model.graph.value_info[:]
    Shared_Merged.set_metadata(
        model,
        "fireredasr_f16_dynamic_rel_shift_restored",
        str(restored or expected),
    )
    Shared_Merged.set_metadata(
        model,
        "fireredasr_f16_encoder_stale_value_info_pruned",
        str(pruned),
    )
    onnx.save(model, str(output_path))
    return restored


def _repair_encoder_frontend_boundary() -> bool:
    """Insert the F32-to-F16 adapter omitted after the blocked frontend norm."""
    output_path = Path(OPTIMIZED_FOLDER_PATH) / "FireRedASR_Encoder.onnx"
    model = onnx.load(str(output_path), load_external_data=False)
    frontend_output = "/encoder/input_preprocessor/out/Add_output_0"
    node_name = "/encoder/layer_stack.0/ffn1/net/net.0/LayerNormalization"
    node_index, layer_norm = next(
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.name == node_name
    )
    public_output = layer_norm.output[0]
    private_output = public_output + "_frontend_f32"
    layer_norm.output[0] = private_output
    adapter = onnx.helper.make_node(
        "Cast",
        [private_output],
        [public_output],
        name="FireRedASR_Encoder_Frontend_To_Float16",
        to=onnx.TensorProto.FLOAT16,
    )
    model.graph.node.insert(node_index + 1, adapter)
    residual_input = frontend_output + "_residual_f16"
    residual_adapter = onnx.helper.make_node(
        "Cast",
        [frontend_output],
        [residual_input],
        name="FireRedASR_Encoder_Frontend_Residual_To_Float16",
        to=onnx.TensorProto.FLOAT16,
    )
    model.graph.node.insert(node_index + 2, residual_adapter)
    for node in model.graph.node:
        if node.name in (node_name, residual_adapter.name):
            continue
        node.input[:] = [
            residual_input if name == frontend_output else name
            for name in node.input
        ]
    onnx.save(model, str(output_path))
    return True


def _restore_standalone_main_boundaries(
    model: onnx.ModelProto,
    *,
    float16: bool,
) -> None:
    """Restore Main aliases and, for F16 donors, its FP32 ``logits`` bridge."""
    Shared_Merged.restore_precision_free_graph_outputs(model)
    if not float16:
        return
    normalize_float16_output_bridge(
        model,
        producer_op_type="Gemm",
        producer_name_contains="/tgt_word_prj/",
        private_output_name="fireredasr_main_f16_logits",
        bridge_node_name="FireRedASR_Main_Logits_To_Float32",
        metadata_key="fireredasr_f16_logits_cast_normalized",
        graph_label="optimized FireRed standalone Main",
    )


def build_quantized_merged_bundle(
    source_folder: Path,
    output_folder: Path,
    model_file_names: dict[str, str],
    *,
    main_is_float16: bool,
) -> None:
    available = Shared_Merged.make_merged_build_plan(model_file_names)

    main_path = output_folder / model_file_names["main"]
    encoder_path = output_folder / model_file_names["encoder"]

    shared_name = model_file_names["shared_initializers"]
    shared_data_name = model_file_names["shared_initializers_data"]

    # Materialize both private converter results before replacing the raw blob.
    optimized_main = Shared_Merged.load_model(main_path)
    optimized_encoder = Shared_Merged.load_model(encoder_path)
    Shared_Merged.namespace_encoder_initializers(optimized_encoder)
    namespaced = Shared_Merged.namespace_internal_tensors(
        optimized_main,
        marker="_inlfunc_",
        namespace="main_",
    )
    if namespaced:
        print(
            f"  Namespaced {namespaced} Main function-inlining tensor(s)."
        )
    _restore_standalone_main_boundaries(
        optimized_main,
        float16=main_is_float16,
    )
    Shared_Merged.restore_precision_free_graph_outputs(optimized_encoder)

    (output_folder / shared_name).unlink(missing_ok=True)
    (output_folder / shared_data_name).unlink(missing_ok=True)
    target_shell_prefixes = tuple(
        prefix
        for prefix in Shared_Merged.SHELL_PREFIXES
        if prefix != "encoder_"
    )
    additional_shared = collect_target_only_shared_shell_initializers(
        source_folder,
        [source_folder / file_name for file_name, _, _ in available],
        optimized_main,
        target_shell_prefixes,
    )
    leaked_encoder_weights = sorted(
        name
        for name in additional_shared
        if name.startswith(Shared_Merged.ENCODER_INITIALIZER_PREFIX)
    )
    if leaked_encoder_weights:
        raise RuntimeError(
            "Target-only shell collection retained replaced raw Encoder weights: "
            f"{leaked_encoder_weights[:8]}."
        )
    if additional_shared:
        print(
            "  Preserving target-only shared shell initializers: "
            + ", ".join(sorted(additional_shared))
        )
    print(
        f"\n{'=' * 60}\n"
        "Transplanting optimized Encoder/Main into all FireRedASR strategy shells\n"
        f"{'=' * 60}"
    )
    external_by_name = None
    for file_name, _, _ in available:
        source_path = source_folder / file_name
        # Target-only shell tensors were materialized above. Keep raw Encoder/Main
        # references structure-only so a projected >4 GiB graph is never loaded six times.
        target = Shared_Merged.load_model(source_path, load_external_data=False)
        merged = Shared_Merged.transplant_quantized_main(target, optimized_main)
        del target
        merged = Shared_Merged.transplant_optimized_encoder(
            merged,
            optimized_encoder,
        )
        if external_by_name is None:
            if file_name != model_file_names["prefill_greedy"]:
                raise RuntimeError("Shared extraction did not start from PrefillGreedy.")
            external_by_name = Shared_Merged.extract_and_write_shared(
                [merged],
                output_folder / shared_name,
                primary_model=merged,
                additional_shared=additional_shared,
            )
        else:
            Shared_Merged.redirect_shared_initializers_to_external(
                merged,
                external_by_name,
            )
        _persist(merged, output_folder / file_name)
        del merged
        gc.collect()

    shared_data_path = output_folder / shared_data_name
    print(f"  {shared_data_name} ({shared_data_path.stat().st_size} bytes)")



def _remove_optimizer_donors(
    output_folder: Path,
    model_file_names: dict[str, str],
) -> None:
    """Delete private Encoder/Main sidecars after the final bundle is written."""
    shared_data_name = model_file_names["shared_initializers_data"]
    for role in ("main", "encoder"):
        donor_path = output_folder / model_file_names[role]
        if not donor_path.exists():
            continue
        locations = Shared_Merged._external_locations(donor_path)
        donor_path.unlink(missing_ok=True)
        donor_path.with_name(donor_path.name + ".data").unlink(missing_ok=True)
        for location in locations:
            if location != shared_data_name:
                (output_folder / location).unlink(missing_ok=True)


def main() -> None:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    model_file_names = dict(Shared_Merged.DEFAULT_MODEL_FILE_NAMES)

    for folder in (source_folder, output_folder):
        removed = Shared_Merged.delete_obsolete_strategy_artifacts(
            folder,
            model_file_names,
        )
        if removed:
            print(
                f"[Cleanup] Removed {len(removed)} obsolete strategy artifact(s) "
                f"from {folder}."
            )

    # Optimize Encoder and Main independently so no cross-component fusion can
    # invalidate their stable transplant ABIs.
    resolved_plans = run_optimizer(
        CONFIG,
        model_names=("FireRedASR_Encoder", "FireRedASR_Decoder"),
        copy_configured_artifacts=False,
        print_completion=False,
        reset_output_folder=True,
    )
    _copy_file(
        source_folder / model_file_names["metadata"],
        output_folder / model_file_names["metadata"],
    )
    restored = _restore_encoder_dynamic_relative_shift()
    print(
        f"  Restored {restored} dynamic relative-shift Reshape(s) "
        "in FireRedASR_Encoder.onnx."
    )
    if resolved_plans["FireRedASR_Encoder"].uses_float16:
        _repair_encoder_frontend_boundary()
        print("  Inserted the FireRedASR encoder F32-to-F16 frontend boundary.")

    build_quantized_merged_bundle(
        source_folder,
        output_folder,
        model_file_names,
        main_is_float16=resolved_plans[MAIN_STEM].uses_float16,
    )
    shared_model_path = output_folder / model_file_names["shared_initializers"]
    for standalone_path in sorted(output_folder.glob("*.onnx")):
        if standalone_path != shared_model_path:
            share_external_initializers_if_identical(
                standalone_path,
                shared_model_path,
                require_all_external=False,
            )

    _remove_optimizer_donors(output_folder, model_file_names)

    for artifact in (
        "dict.txt",
        "train_bpe1000.model",
    ):
        _copy_file(source_folder / artifact, output_folder / artifact)
    storage = consolidate_optimized_model_weights(
        output_folder,
        model_file_names["shared_initializers"],
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")
    print("\n--- All standalone and shared/merged FireRedASR models processed successfully! ---")


if __name__ == "__main__":
    main()
