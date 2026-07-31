"""Build Dolphin CN-Dialect streaming merged graphs around one shared weight blob.

The streaming encoder, metadata carrier, and vocabulary remain standalone.  The
split decoder-side Embed, position/mask shell, Main, and Argmax graphs are
composed into exactly two runtime graphs:

* ``Dolphin_PrefillGreedy.onnx``
* ``Dolphin_DecodeGreedy.onnx``

Large numeric initializers from Main, Embed, and the position/mask shells are
streamed into ``Dolphin_SharedInitializers.onnx.data``. Byte-identical tensors
reuse one verified data range. Merged ORT sessions mmap that blob and inject its
tensors through ``SessionOptions.add_initializer``.
"""

from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


MIN_SHARED_INITIALIZER_ELEMENTS = 1024
_PRECISION_FREE_CAST_PREFIX = "InsertedPrecisionFreeCast_"
_EXTERNAL_DATA_TYPE_KEY = "__tensor_data_type"
_EXTERNAL_DIMS_KEY = "__tensor_dims"
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

# Every non-Main graph is prefixed.  Unprefixed nodes are the decoder Main and
# form the stable ABI used by quantized-Main transplantation.
SHELL_PREFIXES = ("embed_", "prefill_", "decode_", "argmax_")

PREFILL_GREEDY_MODEL_NAME = "Dolphin_PrefillGreedy.onnx"
DECODE_GREEDY_MODEL_NAME = "Dolphin_DecodeGreedy.onnx"
SHARED_MODEL_NAME = "Dolphin_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"

DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Dolphin_Encoder.onnx",
    "main": "Dolphin_Decoder.onnx",
    "embed": "Dolphin_Decoder_Embed.onnx",
    "position_prefill": "Dolphin_Position_Mask_Prefill.onnx",
    "position_decode": "Dolphin_Position_Mask_Decode.onnx",
    "argmax": "Dolphin_Argmax.onnx",
    "prefill_greedy": PREFILL_GREEDY_MODEL_NAME,
    "decode_greedy": DECODE_GREEDY_MODEL_NAME,
    "shared_initializers": SHARED_MODEL_NAME,
    "shared_initializers_data": SHARED_DATA_NAME,
    "vocab": "vocab_Dolphin_CN_Dialect.txt",
}

RUNTIME_STANDALONE_MODEL_KEYS = ("metadata", "encoder")
REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS = frozenset(RUNTIME_STANDALONE_MODEL_KEYS)
MERGED_CONSTITUENT_MODEL_KEYS = (
    "main",
    "embed",
    "position_prefill",
    "position_decode",
    "argmax",
)


def _model_file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = (
        DEFAULT_MODEL_FILE_NAMES
        if model_file_names is None
        else {**DEFAULT_MODEL_FILE_NAMES, **model_file_names}
    )
    return names[key]


def load_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def _node_attribute(node: onnx.NodeProto, name: str):
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return None


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a data-light merged graph without a private external sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save(model, str(path))


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def _initializer_num_elements(initializer: TensorProto) -> int:
    total = 1
    for dim in initializer.dims:
        total *= int(dim)
    return total


def _is_shareable_initializer(initializer: TensorProto, min_elements: int) -> bool:
    if initializer.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
        return False
    if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
        return False
    return _initializer_num_elements(initializer) >= min_elements


def save_shared_initializers_from_tensors(
    shared: dict[str, TensorProto], path: Path
) -> dict[str, int]:
    """Stream tensors into one mmap-friendly file, reusing byte-identical ranges."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    references: list[TensorProto] = []
    canonical_ranges: dict[tuple, tuple[int, int, str]] = {}
    offset = 0
    deduplicated_tensors = 0
    deduplicated_bytes = 0
    with open(data_path, "wb") as data_file:
        for name, tensor in sorted(shared.items()):
            raw = tensor.raw_data
            if not raw:
                raw = numpy_helper.to_array(tensor).tobytes()
            length = len(raw)
            fingerprint = (
                int(tensor.data_type),
                tuple(int(dim) for dim in tensor.dims),
                length,
                hashlib.sha256(raw).digest(),
            )
            canonical = canonical_ranges.get(fingerprint)
            if canonical is None:
                tensor_offset = offset
                data_file.write(raw)
                canonical_ranges[fingerprint] = (tensor_offset, length, name)
                offset += length
            else:
                tensor_offset, canonical_length, canonical_name = canonical
                canonical_tensor = shared[canonical_name]
                canonical_raw = canonical_tensor.raw_data
                if not canonical_raw:
                    canonical_raw = numpy_helper.to_array(canonical_tensor).tobytes()
                if canonical_length != length or canonical_raw != raw:
                    raise RuntimeError(
                        "Shared-initializer digest collision between "
                        f"{canonical_name!r} and {name!r}."
                    )
                deduplicated_tensors += 1
                deduplicated_bytes += length

            reference = TensorProto()
            reference.name = name
            reference.data_type = tensor.data_type
            reference.dims.extend(tensor.dims)
            reference.data_location = TensorProto.EXTERNAL
            for key, value in (
                ("location", data_name),
                ("offset", str(tensor_offset)),
                ("length", str(length)),
            ):
                entry = reference.external_data.add()
                entry.key = key
                entry.value = value
            references.append(reference)

    graph = onnx.helper.make_graph(
        [], "dolphin_shared_initializers", [], [], initializer=references
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="Dolphin/CN-Dialect-Streaming/Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    onnx.save_model(model, str(path))
    return {
        "initializer_count": len(references),
        "unique_data_ranges": len(canonical_ranges),
        "content_deduplicated_tensors": deduplicated_tensors,
        "content_deduplicated_bytes": deduplicated_bytes,
        "data_bytes": offset,
    }


def shared_external_data_map(
    shared_model_path: Path,
) -> dict[str, dict[str, str]]:
    model = onnx.load(str(shared_model_path), load_external_data=False)
    result: dict[str, dict[str, str]] = {}
    for initializer in model.graph.initializer:
        external = _external_data_map(initializer)
        external[_EXTERNAL_DATA_TYPE_KEY] = str(initializer.data_type)
        external[_EXTERNAL_DIMS_KEY] = ",".join(str(dim) for dim in initializer.dims)
        result[initializer.name] = external
    return result


def make_external_initializer_ref(
    initializer: TensorProto, external_data: dict[str, str]
) -> TensorProto:
    if "location" not in external_data:
        raise RuntimeError(
            f"Shared initializer {initializer.name!r} has no external-data location."
        )
    reference = TensorProto()
    reference.name = initializer.name
    reference.data_type = initializer.data_type
    reference.dims.extend(initializer.dims)
    reference.data_location = TensorProto.EXTERNAL
    for key in ("location", "offset", "length", "checksum", "basepath"):
        value = external_data.get(key)
        if value is not None:
            entry = reference.external_data.add()
            entry.key = key
            entry.value = value
    return reference


def redirect_shared_initializers_to_external(
    model: onnx.ModelProto,
    external_by_name: dict[str, dict[str, str]],
) -> int:
    rewritten: list[TensorProto] = []
    count = 0
    for initializer in model.graph.initializer:
        external = external_by_name.get(initializer.name)
        if external is None:
            rewritten.append(initializer)
        else:
            expected_data_type = int(external[_EXTERNAL_DATA_TYPE_KEY])
            expected_dims = tuple(
                int(dim)
                for dim in external[_EXTERNAL_DIMS_KEY].split(",")
                if dim
            )
            actual_dims = tuple(int(dim) for dim in initializer.dims)
            if initializer.data_type != expected_data_type or actual_dims != expected_dims:
                raise RuntimeError(
                    f"Shared initializer ABI mismatch for {initializer.name!r}: "
                    f"target=(dtype={initializer.data_type}, dims={actual_dims}), "
                    f"shared=(dtype={expected_data_type}, dims={expected_dims})."
                )
            rewritten.append(make_external_initializer_ref(initializer, external))
            count += 1
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    return count


def prefixed(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    import onnx.compose

    return onnx.compose.add_prefix(
        model,
        prefix,
        rename_nodes=True,
        rename_edges=True,
        rename_inputs=True,
        rename_outputs=True,
        rename_initializers=True,
        rename_value_infos=True,
    )


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(
        model.graph.value_info
    )
    return {value.name: value for value in values}


def _ensure_value_info_from(
    target: onnx.ModelProto,
    source: onnx.ModelProto,
    names: tuple[str, ...] | list[str],
) -> None:
    target_names = value_info_by_name(target)
    source_names = value_info_by_name(source)
    for name in names:
        if name in target_names:
            continue
        value = source_names.get(name)
        if value is None:
            raise RuntimeError(
                f"Cannot restore value_info for merged tensor {name!r}."
            )
        target.graph.value_info.append(value)
        target_names[name] = value


def set_graph_outputs(model: onnx.ModelProto, output_names: list[str]) -> None:
    by_name = value_info_by_name(model)
    del model.graph.output[:]
    model.graph.output.extend(by_name[name] for name in output_names)


def prune_unreachable_nodes(model: onnx.ModelProto) -> int:
    """Delete pure ONNX nodes made dead by the minimized public output contract."""
    producer: dict[str, int] = {}
    for index, node in enumerate(model.graph.node):
        for output in node.output:
            if not output:
                continue
            producer[output] = index

    required: set[int] = set()
    pending = [value.name for value in model.graph.output]
    while pending:
        index = producer.get(pending.pop())
        if index is None or index in required:
            continue
        required.add(index)
        pending.extend(name for name in model.graph.node[index].input if name)

    retained = [node for index, node in enumerate(model.graph.node) if index in required]
    removed = len(model.graph.node) - len(retained)
    required_values = {name for node in retained for name in node.input if name}
    interface_names = {value.name for value in model.graph.input}
    interface_names.update(value.name for value in model.graph.output)
    retained_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name in required_values or initializer.name in interface_names
    ]
    live_values = {value.name for value in model.graph.input}
    live_values.update(value.name for value in model.graph.output)
    live_values.update(initializer.name for initializer in retained_initializers)
    live_values.update(name for node in retained for name in (*node.input, *node.output) if name)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    retained_info = [value for value in model.graph.value_info if value.name in live_values]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    retained_annotations = [
        annotation
        for annotation in model.graph.quantization_annotation
        if annotation.tensor_name in live_values
    ]
    del model.graph.quantization_annotation[:]
    model.graph.quantization_annotation.extend(retained_annotations)
    return removed


def simplify_argmax_logits_cast(model: onnx.ModelProto) -> int:
    """Bypass an exact F16-to-F32 widening Cast used only by ArgMax."""
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)
    public_outputs = {value.name for value in model.graph.output}
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
        if value.type.HasField("tensor_type")
    }
    initializer_types = {
        initializer.name: initializer.data_type
        for initializer in model.graph.initializer
    }

    changed = 0
    for argmax in model.graph.node:
        if argmax.op_type != "ArgMax" or len(argmax.input) != 1:
            continue
        cast = producers.get(argmax.input[0])
        if not (
            cast is not None
            and cast.op_type == "Cast"
            and len(cast.input) == 1
            and len(cast.output) == 1
            and _node_attribute(cast, "to") == TensorProto.FLOAT
            and cast.output[0] not in public_outputs
            and consumers.get(cast.output[0]) == [argmax]
        ):
            continue
        source = cast.input[0]
        source_type = value_types.get(source, initializer_types.get(source))
        source_producer = producers.get(source)
        if (
            source_type is None
            and source_producer is not None
            and source_producer.op_type in ("Gemm", "MatMul")
        ):
            source_type = next(
                (
                    initializer_types[name]
                    for name in source_producer.input[1:]
                    if name in initializer_types
                ),
                None,
            )
        if source_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
            continue
        argmax.input[0] = source
        changed += 1

    if changed:
        prune_unreachable_nodes(model)
    return changed


def copy_metadata(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    existing = {prop.key: prop for prop in destination.metadata_props}
    for source in sources:
        for prop in source.metadata_props:
            if prop.key in existing:
                existing[prop.key].value = prop.value
            else:
                existing[prop.key] = destination.metadata_props.add(
                    key=prop.key, value=prop.value
                )


def restore_float16_public_output_names(
    model: onnx.ModelProto,
    label: str = "float16 model",
) -> dict[str, str]:
    """Reconnect public outputs whose precision-free Cast aliases were removed."""
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            if not output:
                continue
            if output in producers:
                raise RuntimeError(
                    f"{label} has duplicate producer for tensor {output!r}."
                )
            producers[output] = node

    graph_inputs = {value.name for value in model.graph.input}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_outputs = {value.name: value for value in model.graph.output}
    value_infos = {value.name: value for value in model.graph.value_info}
    aliases: dict[str, str] = {}
    for name, output_info in graph_outputs.items():
        if name in producers or name in graph_inputs or name in initializer_names:
            continue
        alias = _PRECISION_FREE_CAST_PREFIX + name
        alias_node = producers.get(alias)
        if alias_node is None:
            raise RuntimeError(
                f"{label} output {name!r} has neither a producer nor the "
                f"expected precision-free alias {alias!r}."
            )
        alias_info = value_infos.get(alias)
        if alias_info is not None:
            alias_type = alias_info.type.tensor_type
            output_type = output_info.type.tensor_type
            alias_dims = alias_type.shape.dim
            output_dims = output_type.shape.dim
            incompatible_dims = any(
                alias_dim.HasField("dim_value")
                and output_dim.HasField("dim_value")
                and alias_dim.dim_value != output_dim.dim_value
                for alias_dim, output_dim in zip(alias_dims, output_dims)
            )
            if (
                not alias_info.type.HasField("tensor_type")
                or not output_info.type.HasField("tensor_type")
                or alias_type.elem_type != output_type.elem_type
                or len(alias_dims) != len(output_dims)
                or incompatible_dims
            ):
                raise RuntimeError(
                    f"Cannot restore {label} output {name!r}: precision alias "
                    "has an incompatible dtype, rank, or concrete shape."
                )
        aliases[alias] = name

    if aliases:
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = aliases.get(name, name)
            for index, name in enumerate(node.output):
                node.output[index] = aliases.get(name, name)

        retained_value_infos = [
            value for value in model.graph.value_info if value.name not in aliases
        ]
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_infos)

        for annotation in model.graph.quantization_annotation:
            annotation.tensor_name = aliases.get(
                annotation.tensor_name,
                annotation.tensor_name,
            )
            for parameter in annotation.quant_parameter_tensor_names:
                parameter.value = aliases.get(parameter.value, parameter.value)

    return aliases


def restore_float16_merged_boundary_names(
    model: onnx.ModelProto,
) -> dict[str, str]:
    """Repair F16 aliases and adapt source-precision shells to the converted Main."""
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            if not output:
                continue
            if output in producers:
                raise RuntimeError(
                    f"Float16 donor has duplicate producer for tensor {output!r}."
                )
            producers[output] = node

    graph_inputs = {value.name for value in model.graph.input}
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    graph_outputs = {value.name: value for value in model.graph.output}
    value_infos = {value.name: value for value in model.graph.value_info}
    aliases: dict[str, str] = {}

    # Public graph outputs must remain addressable by their declared names.
    for name in graph_outputs:
        if name in producers or name in graph_inputs or name in initializer_names:
            continue
        alias = _PRECISION_FREE_CAST_PREFIX + name
        if alias not in producers:
            raise RuntimeError(
                f"Float16 donor graph output {name!r} has neither a producer nor "
                f"the expected precision-free alias {alias!r}."
            )
        aliases[alias] = name

    # These values are the float shell/Main boundaries needed by transplantation.
    for name in (
        "embed_hidden_states",
        "prefill_position_embed",
        "prefill_attention_mask",
    ):
        alias = _PRECISION_FREE_CAST_PREFIX + name
        if alias in producers and name not in producers:
            if name in graph_inputs or name in initializer_names:
                raise RuntimeError(
                    f"Cannot restore float16 boundary {alias!r}: {name!r} is reserved."
                )
            aliases[alias] = name

    for alias, name in aliases.items():
        alias_info = value_infos.get(alias)
        output_info = graph_outputs.get(name)
        if alias_info is None or output_info is None:
            continue
        alias_type = alias_info.type.tensor_type
        output_type = output_info.type.tensor_type
        alias_dims = alias_type.shape.dim
        output_dims = output_type.shape.dim
        incompatible_dims = any(
            alias_dim.HasField("dim_value")
            and output_dim.HasField("dim_value")
            and alias_dim.dim_value != output_dim.dim_value
            for alias_dim, output_dim in zip(alias_dims, output_dims)
        )
        if (
            not alias_info.type.HasField("tensor_type")
            or not output_info.type.HasField("tensor_type")
            or alias_type.elem_type != output_type.elem_type
            or len(alias_dims) != len(output_dims)
            or incompatible_dims
        ):
            raise RuntimeError(
                f"Cannot restore float16 graph output {name!r}: precision alias "
                "has an incompatible dtype, rank, or concrete shape."
            )

    if aliases:
        for node in model.graph.node:
            for index, name in enumerate(node.input):
                node.input[index] = aliases.get(name, name)
            for index, name in enumerate(node.output):
                node.output[index] = aliases.get(name, name)

        interface_names = graph_inputs | initializer_names | set(graph_outputs)
        retained_value_infos: list[onnx.ValueInfoProto] = []
        retained_names = {
            value.name
            for value in model.graph.value_info
            if value.name not in aliases
        }
        for value in model.graph.value_info:
            target = aliases.get(value.name)
            if target is None:
                retained_value_infos.append(value)
            elif target not in interface_names and target not in retained_names:
                copied = copy.deepcopy(value)
                copied.name = target
                retained_value_infos.append(copied)
                retained_names.add(target)
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_infos)

        for annotation in model.graph.quantization_annotation:
            annotation.tensor_name = aliases.get(
                annotation.tensor_name,
                annotation.tensor_name,
            )
            for parameter in annotation.quant_parameter_tensor_names:
                parameter.value = aliases.get(parameter.value, parameter.value)

    # Keep reusable strategy shells in source precision and adapt only at the
    # Main boundary. These unprefixed Casts are transplanted with Main and have
    # their prefill position input remapped to the decode position input.
    type_by_name = {
        value.name: value
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
    }
    nodes = list(model.graph.node)
    casts_by_index: dict[int, list[onnx.NodeProto]] = {}
    cast_value_infos: list[onnx.ValueInfoProto] = []
    cast_count = 0
    for boundary_name in ("embed_hidden_states", "prefill_position_embed"):
        value_info = type_by_name.get(boundary_name)
        if value_info is None or not value_info.type.HasField("tensor_type"):
            raise RuntimeError(
                f"Float16 donor is missing tensor metadata for {boundary_name!r}."
            )
        elem_type = value_info.type.tensor_type.elem_type
        if elem_type == TensorProto.FLOAT16:
            continue
        if elem_type != TensorProto.FLOAT:
            raise RuntimeError(
                f"Float16 donor boundary {boundary_name!r} has unsupported "
                f"element type {elem_type}."
            )

        cast_output = f"dolphin_main_f16_{boundary_name}"
        existing_cast = producers.get(cast_output)
        if existing_cast is not None:
            if not (
                existing_cast.op_type == "Cast"
                and list(existing_cast.input) == [boundary_name]
                and _node_attribute(existing_cast, "to") == TensorProto.FLOAT16
                and any(
                    not _node_is_shell(node) and cast_output in node.input
                    for node in nodes
                )
            ):
                raise RuntimeError(f"Float16 Main adapter collision: {cast_output!r}.")
            continue

        consumer_indices = [
            index
            for index, node in enumerate(nodes)
            if not _node_is_shell(node) and boundary_name in node.input
        ]
        if not consumer_indices:
            raise RuntimeError(
                f"Float16 donor Main does not consume boundary {boundary_name!r}."
            )
        insert_index = min(consumer_indices)
        producer_indices = [
            index
            for index, node in enumerate(nodes)
            if boundary_name in node.output
        ]
        if producer_indices and max(producer_indices) >= insert_index:
            raise RuntimeError(
                f"Float16 donor boundary {boundary_name!r} is not topologically "
                "before its Main consumer."
            )

        if cast_output in type_by_name:
            raise RuntimeError(f"Float16 Main adapter collision: {cast_output!r}.")
        for index in consumer_indices:
            node = nodes[index]
            for input_index, input_name in enumerate(node.input):
                if input_name == boundary_name:
                    node.input[input_index] = cast_output
        cast = onnx.helper.make_node(
            "Cast",
            inputs=[boundary_name],
            outputs=[cast_output],
            name=f"DolphinMainF16Cast_{boundary_name}",
            to=TensorProto.FLOAT16,
        )
        casts_by_index.setdefault(insert_index, []).append(cast)
        cast_info = copy.deepcopy(value_info)
        cast_info.name = cast_output
        cast_info.type.tensor_type.elem_type = TensorProto.FLOAT16
        cast_value_infos.append(cast_info)
        cast_count += 1

    if cast_count:
        rebuilt_nodes: list[onnx.NodeProto] = []
        for index, node in enumerate(nodes):
            rebuilt_nodes.extend(casts_by_index.get(index, ()))
            rebuilt_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rebuilt_nodes)
        model.graph.value_info.extend(cast_value_infos)

    # Failed symbolic inference can leave stale FLOAT value_info on F16 Main
    # intermediates. Let ORT infer Main tensors while retaining shell/adaptor ABI.
    producer_after_repair = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    boundary_names = {
        "embed_hidden_states",
        "prefill_position_embed",
        "prefill_attention_mask",
    }
    retained_value_infos = [
        value
        for value in model.graph.value_info
        if value.name in boundary_names
        or value.name.startswith("dolphin_main_f16_")
        or (
            value.name in producer_after_repair
            and _node_is_shell(producer_after_repair[value.name])
        )
    ]
    pruned_count = len(model.graph.value_info) - len(retained_value_infos)
    if pruned_count:
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_infos)

    return aliases


def merge_models_no_check(
    first: onnx.ModelProto,
    second: onnx.ModelProto,
    io_map: list[tuple[str, str]],
) -> onnx.ModelProto:
    """Compose two graphs without the memory-heavy ONNX checker path."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "Dolphin/CN-Dialect-Streaming/Shared_Merged.py"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"

    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    seen_inputs: set[str] = set()
    candidates = list(first.graph.input) + [
        value for value in second.graph.input if value.name not in mapped_targets
    ]
    for value in candidates:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializers: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(
        second.graph.initializer
    ):
        existing = initializers.get(initializer.name)
        if existing is None:
            initializers[initializer.name] = initializer
        elif existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"Initializer name collision with different data: {initializer.name}"
            )
    merged.graph.initializer.extend(initializers.values())

    merged.graph.node.extend(first.graph.node)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    for node in merged.graph.node[second_start:]:
        for index, name in enumerate(node.input):
            replacement = source_by_target.get(name)
            if replacement is not None:
                node.input[index] = replacement

    seen_value_info = {value.name for value in merged.graph.input}
    seen_value_info.update(initializers)
    for value in list(first.graph.value_info) + list(second.graph.value_info):
        if value.name not in seen_value_info:
            merged.graph.value_info.append(value)
            seen_value_info.add(value.name)

    seen_outputs: set[str] = set()
    output_candidates = [
        value for value in first.graph.output if value.name not in mapped_sources
    ] + list(second.graph.output)
    for value in output_candidates:
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)

    copy_metadata(merged, first, second)
    return merged


def load_main_with_shared_initializers(
    source_folder: Path,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
) -> tuple[onnx.ModelProto, dict[str, TensorProto]]:
    main_path = source_folder / _model_file_name(model_file_names, "main")
    main = load_model(main_path)
    shared = {
        initializer.name: initializer
        for initializer in main.graph.initializer
        if _is_shareable_initializer(initializer, min_elements)
    }
    return main, shared


def _main_kv_output_names(main: onnx.ModelProto) -> list[str]:
    return [
        output.name
        for output in main.graph.output
        if output.name.startswith("out_de_")
    ]


def _order_decoder_inputs(model: onnx.ModelProto) -> None:
    """Keep recurrent self-KV first, fixed cross-KV second, controls last."""
    self_kv = [
        value for value in model.graph.input if value.name.startswith("in_de_")
    ]
    cross_kv = [
        value
        for value in model.graph.input
        if value.name.startswith(("en_key_", "en_value_"))
    ]
    other = [
        value
        for value in model.graph.input
        if not value.name.startswith(("in_de_", "en_key_", "en_value_"))
    ]
    del model.graph.input[:]
    model.graph.input.extend(self_kv + cross_kv + other)


def _merge_position_embed_into_main(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
) -> tuple[onnx.ModelProto, str, onnx.ModelProto]:
    if kind == "prefill":
        position = prefixed(
            load_model(
                source_folder
                / _model_file_name(model_file_names, "position_prefill")
            ),
            "prefill_",
        )
        position_and_embed = merge_models_no_check(position, embed, io_map=[])
        merged = merge_models_no_check(
            position_and_embed,
            main,
            io_map=[
                ("prefill_position_embed", "position_embed"),
                ("prefill_attention_mask", "attention_mask"),
                ("embed_hidden_states", "hidden_states"),
            ],
        )
        return merged, "prefill_kv_seq_len", position

    if kind != "decode":
        raise ValueError(f"Unknown Dolphin position-shell kind: {kind!r}")
    position = prefixed(
        load_model(
            source_folder / _model_file_name(model_file_names, "position_decode")
        ),
        "decode_",
    )
    mask_info = next(
        value for value in main.graph.input if value.name == "attention_mask"
    )
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(
        mask_info.type.tensor_type.elem_type
    )
    position.graph.initializer.append(
        numpy_helper.from_array(
            np.zeros((1, 1, 1), dtype=mask_dtype),
            name="decode_zero_attention_mask",
        )
    )
    position_and_embed = merge_models_no_check(position, embed, io_map=[])
    merged = merge_models_no_check(
        position_and_embed,
        main,
        io_map=[
            ("decode_position_embed", "position_embed"),
            ("decode_zero_attention_mask", "attention_mask"),
            ("embed_hidden_states", "hidden_states"),
        ],
    )
    return merged, "decode_kv_seq_len_next", position


def _finalize(
    merged: onnx.ModelProto,
    main: onnx.ModelProto,
    position: onnx.ModelProto,
    output_names: list[str],
) -> onnx.ModelProto:
    # Preserve logits as value_info for the validator without exposing/copying the
    # full vector in the production runtime graph.
    _ensure_value_info_from(merged, main, ("logits",))
    set_graph_outputs(merged, output_names)
    prune_unreachable_nodes(merged)
    _order_decoder_inputs(merged)
    copy_metadata(merged, main, position)
    merged.producer_name = "Dolphin/CN-Dialect-Streaming/Shared_Merged.py"
    return merged


def eliminate_decode_zero_attention_mask(model: onnx.ModelProto) -> int:
    """Remove only the decode-only ``score + broadcast_zero`` subgraphs.

    The shared Main must retain its dynamic mask input for prefill, while the
    decode shell always supplies one FLOAT16 zero. The legacy exporter cannot
    specialize that composed branch, and ORT does not eliminate its dynamic
    broadcast Adds. Every structural, dtype, rank, producer, and consumer
    precondition is checked before the model is mutated.
    """
    initializer_name = "decode_zero_attention_mask"
    initializers = {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }
    initializer = initializers.get(initializer_name)
    if initializer is None:
        raise RuntimeError("Decode zero-mask rewrite found no mask initializer.")
    if (
        initializer.data_type != TensorProto.FLOAT16
        or tuple(initializer.dims) != (1, 1, 1)
        or not np.all(numpy_helper.to_array(initializer) == 0)
    ):
        raise RuntimeError(
            "Decode zero-mask initializer failed dtype/shape/value preconditions."
        )

    consumers: dict[str, list[onnx.NodeProto]] = {}
    producers: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)
        for name in node.output:
            if name:
                if name in producers:
                    raise RuntimeError(f"Multiple producers for {name!r}.")
                producers[name] = node

    mask_value = initializer_name
    removable_cast = None
    direct_consumers = consumers.get(mask_value, [])
    if len(direct_consumers) == 1 and direct_consumers[0].op_type == "Cast":
        removable_cast = direct_consumers[0]
        if removable_cast.domain not in ("", "ai.onnx") or len(removable_cast.output) != 1:
            raise RuntimeError("Decode zero-mask Cast failed domain/output preconditions.")
        to_attributes = [
            attribute.i
            for attribute in removable_cast.attribute
            if attribute.name == "to"
        ]
        if to_attributes != [TensorProto.FLOAT]:
            raise RuntimeError("Decode zero-mask Cast has an unexpected target dtype.")
        mask_value = removable_cast.output[0]

    additions = consumers.get(mask_value, [])

    def is_private_softmax_input(value: str) -> bool:
        """Accept a standard Softmax or its onnxslim function expansion."""
        direct = consumers.get(value, [])
        if len(direct) == 1:
            node = direct[0]
            return node.domain in ("", "ai.onnx") and node.op_type == "Softmax"
        if len(direct) != 2:
            return False

        reduce_nodes = [node for node in direct if node.op_type == "ReduceMax"]
        sub_nodes = [node for node in direct if node.op_type == "Sub"]
        if len(reduce_nodes) != 1 or len(sub_nodes) != 1:
            return False
        reduce_max, subtract = reduce_nodes[0], sub_nodes[0]
        if (
            reduce_max.domain not in ("", "ai.onnx")
            or subtract.domain not in ("", "ai.onnx")
            or len(reduce_max.input) < 1
            or reduce_max.input[0] != value
            or len(reduce_max.output) != 1
            or list(subtract.input) != [value, reduce_max.output[0]]
            or len(subtract.output) != 1
            or consumers.get(reduce_max.output[0], []) != [subtract]
        ):
            return False

        subtract_consumers = consumers.get(subtract.output[0], [])
        if len(subtract_consumers) != 1:
            return False
        exponential = subtract_consumers[0]
        if (
            exponential.domain not in ("", "ai.onnx")
            or exponential.op_type != "Exp"
            or list(exponential.input) != [subtract.output[0]]
            or len(exponential.output) != 1
        ):
            return False

        exp_output = exponential.output[0]
        exp_consumers = consumers.get(exp_output, [])
        reduce_sums = [node for node in exp_consumers if node.op_type == "ReduceSum"]
        divisions = [node for node in exp_consumers if node.op_type == "Div"]
        if len(reduce_sums) != 1 or len(divisions) != 1 or len(exp_consumers) != 2:
            return False
        reduce_sum, division = reduce_sums[0], divisions[0]
        return (
            reduce_sum.domain in ("", "ai.onnx")
            and division.domain in ("", "ai.onnx")
            and len(reduce_sum.input) >= 1
            and reduce_sum.input[0] == exp_output
            and list(reduce_max.input[1:]) == list(reduce_sum.input[1:])
            and len(reduce_sum.output) == 1
            and list(division.input) == [exp_output, reduce_sum.output[0]]
            and len(division.output) == 1
            and consumers.get(reduce_sum.output[0], []) == [division]
        )

    replacements: dict[str, str] = {}
    removed_outputs = {initializer_name}
    if removable_cast is not None:
        removed_outputs.add(mask_value)
    graph_outputs = {value.name for value in model.graph.output}
    for node in additions:
        if node.op_type != "Add" or node.domain not in ("", "ai.onnx"):
            raise RuntimeError("Decode zero-mask consumer is not a standard Add.")
        if (
            len(node.input) != 2
            or sum(name == mask_value for name in node.input) != 1
            or len(node.output) != 1
        ):
            raise RuntimeError("Decode zero-mask Add failed arity preconditions.")
        score = node.input[0] if node.input[1] == mask_value else node.input[1]
        score_producer = producers.get(score)
        if score_producer is None or score_producer.op_type != "MatMul":
            raise RuntimeError("Decode zero-mask Add is not fed by a MatMul score.")
        output = node.output[0]
        if output in graph_outputs or not is_private_softmax_input(output):
            raise RuntimeError(
                "Decode zero-mask Add output is not a private Softmax input "
                "or recognized Softmax expansion."
            )
        replacements[output] = score
        removed_outputs.add(output)

    remove_node_outputs = {node.output[0] for node in additions}
    if removable_cast is not None:
        remove_node_outputs.add(removable_cast.output[0])
    rewritten_nodes = []
    for node in model.graph.node:
        if any(output in remove_node_outputs for output in node.output):
            continue
        copied = copy.deepcopy(node)
        for index, name in enumerate(copied.input):
            copied.input[index] = replacements.get(name, name)
        rewritten_nodes.append(copied)
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(
        value
        for value in initializers.values()
        if value.name != initializer_name
    )
    retained_value_info = [
        value
        for value in model.graph.value_info
        if value.name not in removed_outputs
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    return len(additions)


def _merge_greedy(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
) -> onnx.ModelProto:
    merged, kv_seq_len, position = _merge_position_embed_into_main(
        source_folder, main, embed, kind, model_file_names
    )
    argmax = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "argmax")),
        "argmax_",
    )
    merged = merge_models_no_check(
        merged, argmax, io_map=[("logits", "argmax_logits")]
    )
    merged = _finalize(
        merged,
        main,
        position,
        _main_kv_output_names(main) + ["argmax_max_logits_idx", kv_seq_len],
    )
    if kind == "decode":
        eliminate_decode_zero_attention_mask(merged)
    return merged


def merge_prefill_greedy(source_folder, main, embed, model_file_names=None):
    return _merge_greedy(
        Path(source_folder), main, embed, "prefill", model_file_names
    )


def merge_decode_greedy(source_folder, main, embed, model_file_names=None):
    return _merge_greedy(
        Path(source_folder), main, embed, "decode", model_file_names
    )


def make_merged_build_plan(model_file_names: dict[str, str] | None = None):
    prefill = _model_file_name(model_file_names, "position_prefill")
    decode = _model_file_name(model_file_names, "position_decode")
    argmax = _model_file_name(model_file_names, "argmax")
    return [
        (
            _model_file_name(model_file_names, "prefill_greedy"),
            merge_prefill_greedy,
            [prefill, argmax],
        ),
        (
            _model_file_name(model_file_names, "decode_greedy"),
            merge_decode_greedy,
            [decode, argmax],
        ),
    ]


MERGED_BUILD_PLAN = make_merged_build_plan()


def _add_shareable_initializers(
    shared: dict[str, TensorProto],
    model: onnx.ModelProto,
    min_elements: int,
) -> None:
    for initializer in model.graph.initializer:
        if not _is_shareable_initializer(initializer, min_elements):
            continue
        existing = shared.get(initializer.name)
        if existing is not None:
            if existing.SerializeToString() != initializer.SerializeToString():
                raise RuntimeError(
                    f"Shared initializer collision: {initializer.name}"
                )
            continue
        shared[initializer.name] = initializer


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
) -> dict:
    """Build Dolphin's greedy prefill/decode pair around one shared blob."""
    source_folder = Path(source_folder)
    out_folder = source_folder if out_folder is None else Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    embed_name = _model_file_name(model_file_names, "embed")
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(
        model_file_names, "shared_initializers_data"
    )
    for file_name, _, _ in make_merged_build_plan(model_file_names):
        (out_folder / file_name).unlink(missing_ok=True)
        (out_folder / (file_name + ".data")).unlink(missing_ok=True)

    shared_model_path = out_folder / shared_name
    shared_data_path = out_folder / shared_data_name
    shared_model_path.unlink(missing_ok=True)
    shared_data_path.unlink(missing_ok=True)

    main, shared = load_main_with_shared_initializers(
        source_folder, min_shared_elements, model_file_names
    )
    embed = prefixed(load_model(source_folder / embed_name), "embed_")
    _add_shareable_initializers(shared, embed, min_shared_elements)
    # Prefixes keep graph-local names distinct while byte-level range deduplication lets the
    # identical prefill/decode position tables occupy one physical range in the shared blob.
    for key, prefix in (
        ("position_prefill", "prefill_"),
        ("position_decode", "decode_"),
    ):
        shell = prefixed(
            load_model(source_folder / _model_file_name(model_file_names, key)),
            prefix,
        )
        _add_shareable_initializers(shared, shell, min_shared_elements)
        del shell

    shared_storage = save_shared_initializers_from_tensors(
        shared, shared_model_path
    )
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    redirect_shared_initializers_to_external(main, external_by_name)
    redirect_shared_initializers_to_external(embed, external_by_name)

    graphs: dict[str, Path] = {}
    for file_name, recipe, _ in make_merged_build_plan(
        model_file_names
    ):
        merged = recipe(source_folder, main, embed, model_file_names)
        redirect_shared_initializers_to_external(merged, external_by_name)
        output_path = out_folder / file_name
        save_model(merged, output_path)
        graphs[file_name] = output_path
        del merged

    result = {
        "graphs": graphs,
        "skipped": {},
        "shared_model": shared_model_path,
        "shared_data": shared_data_path,
        "shared_storage": shared_storage,
    }
    if out_folder.resolve() == source_folder.resolve():
        result["removed_constituents"] = delete_merged_constituents(
            source_folder,
            model_file_names=model_file_names,
            protected_names=(shared_name, shared_data_name),
        )
    return result


def _iter_all_data_tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("g"):
                yield from _iter_all_data_tensors(attribute.g)
            for subgraph in attribute.graphs:
                yield from _iter_all_data_tensors(subgraph)


def _external_locations(onnx_path: Path) -> set[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    locations: set[str] = set()
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        location = _external_data_map(tensor).get("location")
        if location:
            locations.add(location)
    return locations


def copy_runtime_standalones(
    source_folder: Path,
    target_folder: Path,
    model_file_names: dict[str, str] | None = None,
) -> list[Path]:
    """Copy only the standalone graphs Dolphin streaming actually emits."""
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for key in RUNTIME_STANDALONE_MODEL_KEYS:
        name = _model_file_name(model_file_names, key)
        source = source_folder / name
        target = target_folder / name
        target.unlink(missing_ok=True)
        target.with_name(target.name + ".data").unlink(missing_ok=True)
        if not source.exists():
            if key in REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS:
                raise FileNotFoundError(
                    f"Required Dolphin standalone graph was not exported: {source}"
                )
            continue
        shutil.copy2(source, target)
        for location in _external_locations(source):
            relative = Path(location)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"Unsafe external-data location {location!r} in {source.name}."
                )
            source_data = source_folder / relative
            target_data = target_folder / relative
            target_data.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_data, target_data)
        copied.append(target)
    return copied


def delete_merged_constituents(
    folder: Path,
    model_file_names: dict[str, str] | None = None,
    protected_names: tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    folder = Path(folder)
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(
        model_file_names, "shared_initializers_data"
    )
    protected = set(protected_names or (shared_name, shared_data_name))
    removed: list[str] = []
    for key in MERGED_CONSTITUENT_MODEL_KEYS:
        path = folder / _model_file_name(model_file_names, key)
        if not path.exists():
            continue
        for location in _external_locations(path):
            if location in protected:
                continue
            external_path = folder / location
            if external_path.exists():
                external_path.unlink()
                removed.append(external_path.name)
        path.unlink()
        removed.append(path.name)
        sidecar = path.with_name(path.name + ".data")
        if sidecar.exists() and sidecar.name not in protected:
            sidecar.unlink()
            removed.append(sidecar.name)
    return removed


# ---------------------------------------------------------------------------
# Optimized-Main transplantation used by Optimize_ONNX.py.
# ---------------------------------------------------------------------------


def _node_is_shell(node: onnx.NodeProto) -> bool:
    return any(
        output.removeprefix(_PRECISION_FREE_CAST_PREFIX).startswith(SHELL_PREFIXES)
        for output in node.output
    )


def _used_inputs(nodes) -> set[str]:
    return {name for node in nodes for name in node.input if name}


def _copy_node_with_input_remap(
    node: onnx.NodeProto, remap: dict[str, str]
) -> onnx.NodeProto:
    copied = copy.deepcopy(node)
    for index, name in enumerate(copied.input):
        copied.input[index] = remap.get(name, name)
    return copied


def _copy_value_info_with_name(
    value_info: onnx.ValueInfoProto, name: str
) -> onnx.ValueInfoProto:
    copied = copy.deepcopy(value_info)
    copied.name = name
    return copied


def _merge_opsets(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    opsets: dict[str, int] = {}
    for model in (destination, *sources):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    del destination.opset_import[:]
    for domain, version in sorted(opsets.items()):
        destination.opset_import.add(domain=domain, version=version)


def _target_position_remap(target: onnx.ModelProto) -> dict[str, str]:
    names = {value.name for value in target.graph.input}
    names.update(initializer.name for initializer in target.graph.initializer)
    for node in target.graph.node:
        names.update(node.output)
    if "decode_position_embed" not in names:
        return {}
    return {
        "prefill_position_embed": "decode_position_embed",
        "prefill_attention_mask": "decode_zero_attention_mask",
    }


def transplant_quantized_main(
    target: onnx.ModelProto,
    quantized_primary: onnx.ModelProto,
) -> onnx.ModelProto:
    """Replace a target shell's source Main with one optimized donor Main."""
    specialize_decode_zero_mask = (
        "decode_zero_attention_mask" not in {
            initializer.name for initializer in target.graph.initializer
        }
        and any(
            "decode_position_embed" in node.output for node in target.graph.node
        )
    )
    remap = _target_position_remap(target)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in quantized_primary.graph.node
        if not _node_is_shell(node)
    ]
    if not primary_main_nodes:
        raise RuntimeError("Optimized primary graph contains no decoder Main block.")

    merged = copy.deepcopy(target)
    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in target.graph.node:
        if _node_is_shell(node):
            new_nodes.append(copy.deepcopy(node))
        elif not inserted:
            new_nodes.extend(copy.deepcopy(primary_main_nodes))
            inserted = True
    if not inserted:
        new_nodes.extend(copy.deepcopy(primary_main_nodes))

    primary_initializers = {
        initializer.name: initializer
        for initializer in quantized_primary.graph.initializer
    }
    target_initializers = {
        initializer.name: initializer for initializer in target.graph.initializer
    }
    used = _used_inputs(new_nodes)
    produced = {
        output
        for node in new_nodes
        for output in node.output
        if output
    }
    required_initializers = used - produced
    main_initializer_names = (
        _used_inputs(primary_main_nodes)
        & set(primary_initializers)
        & required_initializers
    )
    new_initializers: list[TensorProto] = []
    seen: set[str] = set()

    def add(initializer: TensorProto) -> None:
        if initializer.name not in seen:
            new_initializers.append(copy.deepcopy(initializer))
            seen.add(initializer.name)

    for initializer in target.graph.initializer:
        if (
            initializer.name in required_initializers
            and initializer.name not in main_initializer_names
        ):
            add(initializer)
    for initializer in quantized_primary.graph.initializer:
        if initializer.name in main_initializer_names:
            add(initializer)
    for name in sorted(required_initializers):
        if name not in seen and name in target_initializers:
            add(target_initializers[name])
        if name not in seen and name in primary_initializers:
            add(primary_initializers[name])

    del merged.graph.node[:]
    merged.graph.node.extend(new_nodes)
    del merged.graph.initializer[:]
    merged.graph.initializer.extend(new_initializers)

    existing = {value.name for value in merged.graph.input}
    existing.update(value.name for value in merged.graph.output)
    existing.update(initializer.name for initializer in merged.graph.initializer)
    value_infos: list[onnx.ValueInfoProto] = []

    def add_value_info(
        value_info: onnx.ValueInfoProto, name: str | None = None
    ) -> None:
        value_name = name or value_info.name
        if value_name not in existing:
            value_infos.append(
                _copy_value_info_with_name(value_info, value_name)
            )
            existing.add(value_name)

    for value_info in quantized_primary.graph.value_info:
        add_value_info(value_info, remap.get(value_info.name, value_info.name))
    for value_info in target.graph.value_info:
        if value_info.name.startswith(SHELL_PREFIXES):
            add_value_info(value_info)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(value_infos)
    _merge_opsets(merged, quantized_primary)
    _order_decoder_inputs(merged)
    if specialize_decode_zero_mask:
        if "decode_zero_attention_mask" in {
            initializer.name for initializer in merged.graph.initializer
        }:
            raise RuntimeError(
                "Quantized Main transplant unexpectedly retained a decode zero mask."
            )
        merged.graph.initializer.append(
            numpy_helper.from_array(
                np.zeros((1, 1, 1), dtype=np.float16),
                name="decode_zero_attention_mask",
            )
        )
        eliminate_decode_zero_attention_mask(merged)
    simplify_argmax_logits_cast(merged)
    prune_unreachable_nodes(merged)
    return merged


def extract_and_write_shared(
    models: dict[str, onnx.ModelProto] | list[onnx.ModelProto],
    shared_model_path: Path,
    primary_model: onnx.ModelProto | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    additional_shared: dict[str, TensorProto] | None = None,
) -> dict[str, dict[str, str]]:
    """Extract all large numeric donor tensors and redirect supplied models."""
    values = list(models.values()) if isinstance(models, dict) else list(models)
    if not values:
        raise RuntimeError("No merged model was supplied for shared extraction.")
    source = primary_model or values[0]
    shared = {
        initializer.name: initializer
        for initializer in source.graph.initializer
        if _is_shareable_initializer(initializer, min_shared_elements)
    }
    for name, initializer in (additional_shared or {}).items():
        if name != initializer.name:
            raise RuntimeError(
                f"Additional shared initializer key/name mismatch: "
                f"{name!r} != {initializer.name!r}."
            )
        existing = shared.get(name)
        if existing is not None and existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"Additional shared initializer collision for {name!r}."
            )
        if not _is_shareable_initializer(initializer, min_shared_elements):
            raise RuntimeError(
                f"Additional shared initializer {name!r} is not shareable."
            )
        shared[name] = initializer
    save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    for model in values:
        redirect_shared_initializers_to_external(model, external_by_name)
    return external_by_name


# ---------------------------------------------------------------------------
# Runtime-side shared-initializer attachment.
# ---------------------------------------------------------------------------


def attach_shared_initializers(session_options, shared_model_path: Path):
    """Mmap and inject shared tensors; returned references must remain alive."""
    import onnxruntime as ort

    shared_model_path = Path(shared_model_path)
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays: dict[str, np.ndarray] = {}
    ort_values: list = []
    for initializer in shared_model.graph.initializer:
        if initializer.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        external = _external_data_map(initializer)
        location = external.get("location")
        if not location:
            raise RuntimeError(
                f"Shared initializer {initializer.name!r} is not external."
            )
        data_path = shared_model_path.parent / location
        offset = int(external.get("offset", "0"))
        length = int(external.get("length", "0"))
        dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        shape = tuple(int(dim) for dim in initializer.dims)
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if length and length != expected:
            raise RuntimeError(
                f"Shared initializer {initializer.name!r} length mismatch: "
                f"{length} != {expected}."
            )
        array = np.memmap(
            data_path, dtype=dtype, mode="r", offset=offset, shape=shape
        )
        value = ort.OrtValue.ortvalue_from_numpy(array)
        arrays[initializer.name] = array
        ort_values.append(value)
        session_options.add_initializer(initializer.name, value)
    return arrays, ort_values
