"""Build FireRedASR-AED Encoder+prefill graphs backed by one shared blob.

Each prefill graph owns Encoder, decoder Embed, absolute position/mask, Main, and
the selected first-token head. Encoder cross-KV remains public for the Encoder-free
decode graph. Main names stay unprefixed as the stable transplant ABI; Encoder
weights use a stable private namespace and are shared across all prefill strategies.
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
ENCODER_INITIALIZER_PREFIX = "encoder_weight/"
_PRECISION_FREE_CAST_PREFIX = "InsertedPrecisionFreeCast_"
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

# Every non-Main graph is prefixed. Keep this synchronized with the recipes.
SHELL_PREFIXES = (
    "encoder_",
    "embed_",
    "prefill_",
    "decode_",
    "argmax_",
    "greedy_",
    "penalty_",
    "sampling_",
)

PREFILL_GREEDY_MODEL_NAME = "FireRedASR_PrefillGreedy.onnx"
PREFILL_PENALTY_GREEDY_MODEL_NAME = "FireRedASR_PrefillPenaltyGreedy.onnx"
PREFILL_SAMPLING_MODEL_NAME = "FireRedASR_PrefillSampling.onnx"
DECODE_GREEDY_MODEL_NAME = "FireRedASR_DecodeGreedy.onnx"
DECODE_PENALTY_GREEDY_MODEL_NAME = "FireRedASR_DecodePenaltyGreedy.onnx"
DECODE_SAMPLING_MODEL_NAME = "FireRedASR_DecodeSampling.onnx"
SHARED_MODEL_NAME = "FireRedASR_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"

DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "FireRedASR_Encoder.onnx",
    "main": "FireRedASR_Decoder.onnx",
    "embed": "FireRedASR_Decoder_Embed.onnx",
    "position_prefill": "FireRedASR_Position_Mask_Prefill.onnx",
    "position_decode": "FireRedASR_Position_Mask_Decode.onnx",
    "greedy": "FireRedASR_Argmax.onnx",
    "penalty_greedy": "FireRedASR_Greedy_Search.onnx",
    "penalty": "FireRedASR_Apply_Penality.onnx",
    "sampling": "FireRedASR_TopKTopPSampling.onnx",
    "prefill_greedy": PREFILL_GREEDY_MODEL_NAME,
    "prefill_penalty_greedy": PREFILL_PENALTY_GREEDY_MODEL_NAME,
    "prefill_sampling": PREFILL_SAMPLING_MODEL_NAME,
    "decode_greedy": DECODE_GREEDY_MODEL_NAME,
    "decode_penalty_greedy": DECODE_PENALTY_GREEDY_MODEL_NAME,
    "decode_sampling": DECODE_SAMPLING_MODEL_NAME,
    "shared_initializers": SHARED_MODEL_NAME,
    "shared_initializers_data": SHARED_DATA_NAME,
}

RUNTIME_STANDALONE_MODEL_KEYS = ("metadata",)
REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS = frozenset(RUNTIME_STANDALONE_MODEL_KEYS)
MERGED_CONSTITUENT_MODEL_KEYS = (
    "encoder",
    "main",
    "embed",
    "position_prefill",
    "position_decode",
    "greedy",
    "penalty_greedy",
    "penalty",
    "sampling",
)


def _model_file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = (
        DEFAULT_MODEL_FILE_NAMES
        if model_file_names is None
        else {**DEFAULT_MODEL_FILE_NAMES, **model_file_names}
    )
    return names[key]


def load_model(
    path: Path,
    load_external_data: bool = True,
) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=load_external_data)


def _node_attribute(node: onnx.NodeProto, name: str):
    for attribute in node.attribute:
        if attribute.name == name:
            return onnx.helper.get_attribute_value(attribute)
    return None


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a data-light merged graph without a private sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save(model, str(path))


def replace_onnx_metadata(path: Path | str, metadata: dict[str, str]) -> None:
    """Replace the dedicated metadata carrier's keys exactly.

    Other ONNX files keep their graph-local metadata; callers must use this only
    for ``ASR_Metadata.onnx``.
    """
    model = onnx.load(str(path), load_external_data=False)
    del model.metadata_props[:]
    for key, value in sorted(metadata.items()):
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save_model(model, str(path), save_as_external_data=False)


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
) -> None:
    """Stream tensors directly into one external-data file with low peak memory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    references: list[TensorProto] = []
    regions_by_digest: dict[tuple[int, bytes], tuple[int, int]] = {}
    offset = 0
    with open(data_path, "wb") as data_file:
        for name, tensor in sorted(shared.items()):
            raw = tensor.raw_data
            if not raw:
                raw = numpy_helper.to_array(tensor).tobytes()
            length = len(raw)
            digest_key = (length, hashlib.sha256(raw).digest())
            region = regions_by_digest.get(digest_key)
            if region is None:
                region = (offset, length)
                regions_by_digest[digest_key] = region
                data_file.write(raw)
                offset += length

            reference = TensorProto()
            reference.name = name
            reference.data_type = tensor.data_type
            reference.dims.extend(tensor.dims)
            reference.data_location = TensorProto.EXTERNAL
            for key, value in (
                ("location", data_name),
                ("offset", str(region[0])),
                ("length", str(region[1])),
            ):
                entry = reference.external_data.add()
                entry.key = key
                entry.value = value
            references.append(reference)

    graph = onnx.helper.make_graph(
        [], "fireredasr_shared_initializers", [], [], initializer=references
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="FireRedASR/Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    model.metadata_props.add(key="fireredasr_shared_initializers", value="1")
    model.metadata_props.add(key="initializer_count", value=str(len(references)))
    onnx.save_model(model, str(path))


def shared_external_data_map(
    shared_model_path: Path,
) -> dict[str, dict[str, str]]:
    model = onnx.load(str(shared_model_path), load_external_data=False)
    result = {}
    for initializer in model.graph.initializer:
        external = _external_data_map(initializer)
        external["__tensor_data_type"] = str(initializer.data_type)
        external["__tensor_dims"] = ",".join(str(dim) for dim in initializer.dims)
        result[initializer.name] = external
    return result


def make_external_initializer_ref(
    initializer: TensorProto, external_data: dict[str, str]
) -> TensorProto:
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
            expected_type = int(external["__tensor_data_type"])
            expected_dims = tuple(
                int(dim) for dim in external["__tensor_dims"].split(",") if dim
            )
            if initializer.data_type != expected_type or tuple(initializer.dims) != expected_dims:
                raise RuntimeError(
                    f"Shared initializer ABI mismatch for {initializer.name!r}."
                )
            rewritten.append(make_external_initializer_ref(initializer, external))
            count += 1
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    return count


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
    if not shared:
        raise RuntimeError(f"{main_path.name} has no shareable initializer.")
    return main, shared


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


def prefixed_preserving_initializers(
    model: onnx.ModelProto,
    prefix: str,
) -> onnx.ModelProto:
    """Prefix Encoder graph values while retaining stable namespaced weights."""
    stable_names = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
    }
    result = prefixed(model, prefix)
    restore = {f"{prefix}{name}": name for name in stable_names}
    for initializer in result.graph.initializer:
        initializer.name = restore.get(initializer.name, initializer.name)
    for node in result.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = restore.get(name, name)
    for annotation in result.graph.quantization_annotation:
        annotation.tensor_name = restore.get(
            annotation.tensor_name, annotation.tensor_name
        )
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = restore.get(parameter.value, parameter.value)
    return result


def namespace_encoder_initializers(model: onnx.ModelProto) -> None:
    """Move every Encoder initializer into one stable private namespace."""
    remap = {
        initializer.name: f"{ENCODER_INITIALIZER_PREFIX}{initializer.name}"
        for initializer in model.graph.initializer
        if not initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
    }
    for initializer in model.graph.initializer:
        initializer.name = remap.get(initializer.name, initializer.name)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = remap.get(
            annotation.tensor_name, annotation.tensor_name
        )
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)


def namespace_internal_tensors(
    model: onnx.ModelProto,
    *,
    marker: str,
    namespace: str,
) -> int:
    """Namespace optimizer-generated internal tensors matching ``marker``."""
    names = {value.name for value in model.graph.input}
    names.update(value.name for value in model.graph.output)
    names.update(value.name for value in model.graph.value_info)
    names.update(initializer.name for initializer in model.graph.initializer)
    names.update(
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    )
    remap = {
        name: f"{namespace}{name}"
        for name in names
        if marker in name and not name.startswith(namespace)
    }
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value.name = remap.get(value.name, value.name)
    for initializer in model.graph.initializer:
        initializer.name = remap.get(initializer.name, initializer.name)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = remap.get(name, name)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = remap.get(
            annotation.tensor_name, annotation.tensor_name
        )
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)
    return len(remap)


def rename_tensor(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    """Rename one top-level tensor everywhere with collision checks."""
    if old_name == new_name:
        return
    names = {value.name for value in model.graph.input}
    names.update(value.name for value in model.graph.output)
    names.update(value.name for value in model.graph.value_info)
    names.update(initializer.name for initializer in model.graph.initializer)
    names.update(
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    )
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.name == old_name:
            value.name = new_name
    for initializer in model.graph.initializer:
        if initializer.name == old_name:
            initializer.name = new_name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
        for index, name in enumerate(node.output):
            if name == old_name:
                node.output[index] = new_name
    for annotation in model.graph.quantization_annotation:
        if annotation.tensor_name == old_name:
            annotation.tensor_name = new_name
        for parameter in annotation.quant_parameter_tensor_names:
            if parameter.value == old_name:
                parameter.value = new_name


def _cross_kv_names(num_layers: int, prefix: str = "") -> list[str]:
    return [
        *(f"{prefix}en_key_layer_{index}" for index in range(num_layers)),
        *(f"{prefix}en_value_layer_{index}" for index in range(num_layers)),
    ]


def _main_cross_kv_values(
    main: onnx.ModelProto,
    *,
    num_layers: int | None = None,
) -> tuple[int, list[onnx.ValueInfoProto]]:
    values = [
        value
        for value in main.graph.input
        if value.name.startswith(("en_key_layer_", "en_value_layer_"))
    ]
    if num_layers is None:
        num_layers = len(values) // 2
    return num_layers, values


def build_prefill_frontend(
    source_folder: Path,
    encoder: onnx.ModelProto | None = None,
    model_file_names: dict[str, str] | None = None,
    *,
    num_layers: int | None = None,
    expected_num_heads: int | None = None,
    expected_head_dim: int | None = None,
    reference_cross_values: list[onnx.ValueInfoProto] | None = None,
) -> tuple[onnx.ModelProto, list[str]]:
    """Prepare a prefixed FireRed Encoder and exact public cross-KV ABI."""
    if encoder is None:
        encoder = load_model(
            source_folder / _model_file_name(model_file_names, "encoder")
        )
    if num_layers is None:
        num_layers = len(encoder.graph.output) // 2
    expected = _cross_kv_names(num_layers)
    namespace_encoder_initializers(encoder)
    component = prefixed_preserving_initializers(encoder, "encoder_")
    rename_tensor(component, "encoder_audio", "audio")
    return component, [f"encoder_{name}" for name in expected]


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


def _ensure_value_info_from(
    target: onnx.ModelProto,
    source: onnx.ModelProto,
    names: tuple[str, ...] | list[str],
) -> None:
    target_values = value_info_by_name(target)
    source_values = value_info_by_name(source)
    for name in names:
        if name in target_values:
            continue
        value = source_values.get(name)
        if value is None:
            raise RuntimeError(f"Cannot restore value_info for {name!r}.")
        target.graph.value_info.append(value)
        target_values[name] = value


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
        set_metadata(model, "argmax_logits_widening_cast_removed", str(changed))
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


def set_metadata(destination: onnx.ModelProto, key: str, value: str) -> None:
    for prop in destination.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    destination.metadata_props.add(key=key, value=value)


def restore_precision_free_graph_outputs(
    model: onnx.ModelProto,
    *,
    alias_prefix: str = _PRECISION_FREE_CAST_PREFIX,
) -> dict[str, str]:
    """Restore public names orphaned when a precision-free Cast is removed."""
    graph_inputs = {value.name for value in model.graph.input}
    initializers = {initializer.name for initializer in model.graph.initializer}
    producers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in model.graph.node:
        for output_index, output in enumerate(node.output):
            if output:
                producers.setdefault(output, []).append((node, output_index))
    missing = [
        value.name
        for value in model.graph.output
        if value.name not in graph_inputs
        and value.name not in initializers
        and value.name not in producers
    ]
    remap: dict[str, str] = {}
    for public_name in missing:
        alias = f"{alias_prefix}{public_name}"
        owners = producers.get(alias, [])
        if len(owners) != 1:
            raise RuntimeError(
                f"Cannot restore {public_name!r}: alias {alias!r} has "
                f"{len(owners)} producers."
            )
        remap[alias] = public_name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = remap.get(name, name)
    retained = [value for value in model.graph.value_info if value.name not in remap]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = remap.get(
            annotation.tensor_name, annotation.tensor_name
        )
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)
    return remap


def restore_float16_merged_boundary_names(
    model: onnx.ModelProto,
) -> dict[str, str]:
    """Repair float16-converter aliases and shell-to-Main precision boundaries."""
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

    # Public output names are an ABI. onnxslim may remove a precision-free Cast
    # while leaving its producer under the converter's temporary alias.
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

    # Restore the exact shell values that form the stable Main transplant ABI.
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
        if not alias_info.type.HasField("tensor_type") or not output_info.type.HasField(
            "tensor_type"
        ):
            raise RuntimeError(
                f"Cannot restore float16 graph output {name!r}: missing tensor type."
            )
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
            alias_type.elem_type != output_type.elem_type
            or len(alias_dims) != len(output_dims)
            or incompatible_dims
        ):
            raise RuntimeError(
                f"Cannot restore float16 graph output {name!r}: precision "
                "alias has an incompatible dtype, rank, or concrete shape."
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
            value.name for value in model.graph.value_info if value.name not in aliases
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

        set_metadata(
            model,
            "fireredasr_f16_boundary_name_restore_count",
            str(len(aliases)),
        )

    # The converter widens F16 Main logits before the excluded float32 shell,
    # but gives the Cast output a private name such as ``logits_cast_to_fp32``.
    # Transplanted source shells still consume the stable ``logits`` ABI and
    # would otherwise receive the pre-Cast F16 Gemm output.  Normalize that
    # bridge so every strategy shell receives float32 logits.
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    logits_producer = producers.get("logits")
    if logits_producer is None:
        raise RuntimeError("Float16 donor Main has no producer for 'logits'.")
    logits_cast_normalized = (
        logits_producer.op_type == "Cast"
        and _node_attribute(logits_producer, "to") == TensorProto.FLOAT
    )
    if not logits_cast_normalized:
        logits_casts = [
            node
            for node in model.graph.node
            if node.op_type == "Cast"
            and list(node.input) == ["logits"]
            and len(node.output) == 1
            and _node_attribute(node, "to") == TensorProto.FLOAT
        ]
        if not logits_casts:
            logits_info = value_infos.get("logits")
            if (
                logits_info is not None
                and logits_info.type.HasField("tensor_type")
                and logits_info.type.tensor_type.elem_type != TensorProto.FLOAT16
            ):
                raise RuntimeError(
                    "Cannot synthesize the float32 logits bridge: the donor's "
                    "stable logits value is not float16."
                )
            legacy_cast_output = "fireredasr_logits_cast_to_fp32"
            occupied = (
                graph_inputs
                | initializer_names
                | set(graph_outputs)
                | set(producers)
            )
            if legacy_cast_output in occupied:
                raise RuntimeError(
                    f"Cannot synthesize logits bridge output {legacy_cast_output!r}."
                )
            logits_cast = onnx.helper.make_node(
                "Cast",
                ["logits"],
                [legacy_cast_output],
                name="FireRedASR_Main_Logits_To_Float32",
                to=TensorProto.FLOAT,
            )
            producer_index = next(
                index
                for index, node in enumerate(model.graph.node)
                if node is logits_producer
            )
            model.graph.node.insert(producer_index + 1, logits_cast)
            logits_casts = [model.graph.node[producer_index + 1]]
        if len(logits_casts) != 1:
            raise RuntimeError(
                "Expected exactly one float16-to-float32 logits boundary Cast, "
                f"found {len(logits_casts)}."
            )
        logits_cast = logits_casts[0]
        legacy_cast_output = logits_cast.output[0]
        internal_logits = "fireredasr_main_f16_logits"
        occupied = graph_inputs | initializer_names | set(graph_outputs) | set(producers)
        if internal_logits in occupied or not legacy_cast_output:
            raise RuntimeError(
                f"Cannot normalize float16 logits boundary to {internal_logits!r}."
            )
        if legacy_cast_output in graph_inputs or legacy_cast_output in initializer_names:
            raise RuntimeError(
                f"Float16 logits Cast output {legacy_cast_output!r} is reserved."
            )

        for index, output in enumerate(logits_producer.output):
            if output == "logits":
                logits_producer.output[index] = internal_logits
        logits_cast.input[0] = internal_logits
        logits_cast.output[0] = "logits"
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == legacy_cast_output:
                    node.input[index] = "logits"

        impacted_names = {"logits", legacy_cast_output, internal_logits}
        shape_source = next(
            (
                value
                for value in model.graph.value_info
                if value.name in (legacy_cast_output, "logits")
                and value.type.HasField("tensor_type")
            ),
            None,
        )
        retained_value_infos = [
            value
            for value in model.graph.value_info
            if value.name not in impacted_names
        ]
        if shape_source is None:
            retained_value_infos.extend(
                (
                    onnx.helper.make_tensor_value_info(
                        internal_logits, TensorProto.FLOAT16, None
                    ),
                    onnx.helper.make_tensor_value_info(
                        "logits", TensorProto.FLOAT, None
                    ),
                )
            )
        else:
            internal_info = copy.deepcopy(shape_source)
            internal_info.name = internal_logits
            internal_info.type.tensor_type.elem_type = TensorProto.FLOAT16
            logits_info = copy.deepcopy(shape_source)
            logits_info.name = "logits"
            logits_info.type.tensor_type.elem_type = TensorProto.FLOAT
            retained_value_infos.extend((internal_info, logits_info))
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_infos)

        for annotation in model.graph.quantization_annotation:
            if annotation.tensor_name == "logits":
                annotation.tensor_name = internal_logits
            elif annotation.tensor_name == legacy_cast_output:
                annotation.tensor_name = "logits"
            for parameter in annotation.quant_parameter_tensor_names:
                if parameter.value == "logits":
                    parameter.value = internal_logits
                elif parameter.value == legacy_cast_output:
                    parameter.value = "logits"
        set_metadata(model, "fireredasr_f16_logits_cast_normalized", "1")

    # Embed and position shells stay float32 so all strategy graphs can reuse
    # them. Insert explicit adapters only where the converted Main consumes them.
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

        cast_output = f"fireredasr_main_f16_{boundary_name}"
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
            name=f"FireRedASRMainF16Cast_{boundary_name}",
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
        set_metadata(
            model,
            "fireredasr_f16_main_input_cast_count",
            str(cast_count),
        )

    # Failed symbolic inference can leave stale FLOAT declarations on F16 Main
    # intermediates. Let schemas infer Main values and retain only shell/adapters.
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
        "logits",
    }
    retained_value_infos = [
        value
        for value in model.graph.value_info
        if value.name in boundary_names
        or value.name.startswith("fireredasr_main_f16_")
        or (
            value.name in producer_after_repair
            and _node_is_shell(producer_after_repair[value.name])
        )
    ]
    pruned_count = len(model.graph.value_info) - len(retained_value_infos)
    if pruned_count:
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_infos)
        set_metadata(
            model,
            "fireredasr_f16_stale_value_info_pruned",
            str(pruned_count),
        )

    return aliases


def merge_models_no_check(
    first: onnx.ModelProto,
    second: onnx.ModelProto,
    io_map: list[tuple[str, str]],
) -> onnx.ModelProto:
    """Compose models without checker/shape inference materializing a huge proto."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "FireRedASR/Shared_Merged.py"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"

    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    functions: dict[tuple[str, str], onnx.FunctionProto] = {}
    for model in (first, second):
        for function in model.functions:
            key = (function.domain, function.name)
            existing = functions.get(key)
            if existing is None:
                functions[key] = function
            elif existing.SerializeToString() != function.SerializeToString():
                raise RuntimeError(
                    "Local FunctionProto collision with different bodies: "
                    f"domain={function.domain!r}, name={function.name!r}."
                )
    merged.functions.extend(functions.values())

    seen_inputs: set[str] = set()
    candidates = list(first.graph.input) + [
        value for value in second.graph.input if value.name not in mapped_targets
    ]
    for value in candidates:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializers: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
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
    values = list(first.graph.value_info) + list(second.graph.value_info)
    values += [value for value in first.graph.output if value.name in mapped_sources]
    values += [value for value in second.graph.input if value.name in mapped_targets]
    for value in values:
        name = source_by_target.get(value.name, value.name)
        if name in seen_value_info:
            continue
        copied = copy.deepcopy(value)
        copied.name = name
        merged.graph.value_info.append(copied)
        seen_value_info.add(name)

    seen_outputs: set[str] = set()
    outputs = [
        value for value in first.graph.output if value.name not in mapped_sources
    ] + list(second.graph.output)
    for value in outputs:
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)

    copy_metadata(merged, first, second)
    return merged


def _main_kv_output_names(main: onnx.ModelProto) -> list[str]:
    return [value.name for value in main.graph.output if value.name.startswith("out_de_")]


def _order_kv_inputs_first(model: onnx.ModelProto) -> None:
    kv_inputs = [value for value in model.graph.input if value.name.startswith("in_de_")]
    other_inputs = [
        value for value in model.graph.input if not value.name.startswith("in_de_")
    ]
    del model.graph.input[:]
    model.graph.input.extend(kv_inputs + other_inputs)


def _merge_position_into_main(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
) -> tuple[onnx.ModelProto, str, onnx.ModelProto, list[str]]:
    num_layers, main_cross_values = _main_cross_kv_values(main)
    if kind == "prefill":
        position = prefixed(
            load_model(
                source_folder / _model_file_name(model_file_names, "position_prefill")
            ),
            "prefill_",
        )
        position_and_embed = merge_models_no_check(position, embed, [])
        merged = merge_models_no_check(
            position_and_embed,
            main,
            [
                ("prefill_position_embed", "position_embed"),
                ("prefill_attention_mask", "attention_mask"),
                ("embed_hidden_states", "hidden_states"),
            ],
        )
        if prefill_frontend is None:
            prefill_frontend = build_prefill_frontend(
                source_folder,
                model_file_names=model_file_names,
                num_layers=num_layers,
                reference_cross_values=main_cross_values,
            )
        frontend, cross_outputs = prefill_frontend
        io_map = [
            (encoder_name, main_value.name)
            for encoder_name, main_value in zip(cross_outputs, main_cross_values)
        ]
        merged = merge_models_no_check(frontend, merged, io_map)
        _ensure_value_info_from(merged, frontend, cross_outputs)
        return merged, "prefill_kv_seq_len", position, cross_outputs

    position = prefixed(
        load_model(
            source_folder / _model_file_name(model_file_names, "position_decode")
        ),
        "decode_",
    )
    mask_info = next(value for value in main.graph.input if value.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(
        mask_info.type.tensor_type.elem_type
    )
    position.graph.initializer.append(
        numpy_helper.from_array(
            np.zeros((1, 1, 1), dtype=mask_dtype),
            name="decode_zero_attention_mask",
        )
    )
    position_and_embed = merge_models_no_check(position, embed, [])
    merged = merge_models_no_check(
        position_and_embed,
        main,
        [
            ("decode_position_embed", "position_embed"),
            ("decode_zero_attention_mask", "attention_mask"),
            ("embed_hidden_states", "hidden_states"),
        ],
    )
    return merged, "decode_kv_seq_len_next", position, []


def _finalize(
    merged: onnx.ModelProto,
    main: onnx.ModelProto,
    position: onnx.ModelProto,
    output_names: list[str],
) -> onnx.ModelProto:
    _ensure_value_info_from(merged, main, ("logits",))
    set_graph_outputs(merged, output_names)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    copy_metadata(merged, main, position)
    merged.producer_name = "FireRedASR/Shared_Merged.py"
    return merged


def _merge_greedy(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
) -> onnx.ModelProto:
    merged, kv_seq_len, position, cross_outputs = _merge_position_into_main(
        source_folder, main, embed, kind, model_file_names, prefill_frontend
    )
    argmax = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "greedy")),
        "argmax_",
    )
    merged = merge_models_no_check(merged, argmax, [("logits", "argmax_logits")])
    return _finalize(
        merged,
        main,
        position,
        _main_kv_output_names(main)
        + cross_outputs
        + ["argmax_max_logits_idx", kv_seq_len],
    )


def _merge_penalty_greedy_prefill(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
) -> onnx.ModelProto:
    merged, kv_seq_len, position, cross_outputs = _merge_position_into_main(
        source_folder,
        main,
        embed,
        "prefill",
        model_file_names,
        prefill_frontend,
    )
    greedy = prefixed(
        load_model(
            source_folder / _model_file_name(model_file_names, "penalty_greedy")
        ),
        "greedy_",
    )
    merged = merge_models_no_check(merged, greedy, [("logits", "greedy_logits")])
    return _finalize(
        merged,
        main,
        position,
        _main_kv_output_names(main)
        + cross_outputs
        + ["greedy_max_logits_idx", "greedy_save_id_out", kv_seq_len],
    )


def _merge_sampling(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
) -> onnx.ModelProto:
    merged, kv_seq_len, position, cross_outputs = _merge_position_into_main(
        source_folder, main, embed, kind, model_file_names, prefill_frontend
    )
    sampling = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "sampling")),
        "sampling_",
    )
    merged = merge_models_no_check(
        merged, sampling, [("logits", "sampling_logits")]
    )
    return _finalize(
        merged,
        main,
        position,
        _main_kv_output_names(main)
        + cross_outputs
        + ["sampling_sampled_id", "sampling_save_id_out", kv_seq_len],
    )


def merge_prefill_greedy(
    source_folder,
    main,
    embed,
    model_file_names=None,
    prefill_frontend=None,
):
    return _merge_greedy(
        source_folder,
        main,
        embed,
        "prefill",
        model_file_names,
        prefill_frontend,
    )


def merge_decode_greedy(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    if prefill_frontend is not None:
        raise RuntimeError("Decode graph must not receive an Encoder frontend.")
    return _merge_greedy(source_folder, main, embed, "decode", model_file_names)


def merge_prefill_penalty_greedy(
    source_folder,
    main,
    embed,
    model_file_names=None,
    prefill_frontend=None,
):
    return _merge_penalty_greedy_prefill(
        source_folder, main, embed, model_file_names, prefill_frontend
    )


def merge_decode_penalty_greedy(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    if prefill_frontend is not None:
        raise RuntimeError("Decode graph must not receive an Encoder frontend.")
    merged, kv_seq_len, position, cross_outputs = _merge_position_into_main(
        source_folder, main, embed, "decode", model_file_names
    )
    if cross_outputs:
        raise RuntimeError("Decode graph unexpectedly acquired Encoder outputs.")
    penalty = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty")),
        "penalty_",
    )
    greedy = prefixed(
        load_model(
            source_folder / _model_file_name(model_file_names, "penalty_greedy")
        ),
        "greedy_",
    )
    merged = merge_models_no_check(
        merged, penalty, [("logits", "penalty_logits_in")]
    )
    merged = merge_models_no_check(
        merged, greedy, [("penalty_logits_out", "greedy_logits")]
    )
    return _finalize(
        merged,
        main,
        position,
        _main_kv_output_names(main)
        + ["greedy_max_logits_idx", "greedy_save_id_out", kv_seq_len],
    )


def merge_prefill_sampling(
    source_folder,
    main,
    embed,
    model_file_names=None,
    prefill_frontend=None,
):
    return _merge_sampling(
        source_folder,
        main,
        embed,
        "prefill",
        model_file_names,
        prefill_frontend,
    )


def merge_decode_sampling(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    if prefill_frontend is not None:
        raise RuntimeError("Decode graph must not receive an Encoder frontend.")
    return _merge_sampling(
        source_folder, main, embed, "decode", model_file_names
    )


def _recipe_with_frontend(recipe, prefill_frontend):
    def wrapped(source_folder, main, embed, model_file_names=None):
        return recipe(
            source_folder,
            main,
            embed,
            model_file_names,
            prefill_frontend=prefill_frontend,
        )

    wrapped.__name__ = recipe.__name__
    return wrapped


def make_merged_build_plan(
    model_file_names: dict[str, str] | None = None,
    prefill_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
):
    name = lambda role: _model_file_name(model_file_names, role)
    return [
        (
            name("prefill_greedy"),
            _recipe_with_frontend(merge_prefill_greedy, prefill_frontend),
            [name("encoder"), name("position_prefill"), name("greedy")],
        ),
        (
            name("prefill_penalty_greedy"),
            _recipe_with_frontend(
                merge_prefill_penalty_greedy, prefill_frontend
            ),
            [name("encoder"), name("position_prefill"), name("penalty_greedy")],
        ),
        (
            name("prefill_sampling"),
            _recipe_with_frontend(merge_prefill_sampling, prefill_frontend),
            [name("encoder"), name("position_prefill"), name("sampling")],
        ),
        (
            name("decode_greedy"),
            merge_decode_greedy,
            [name("position_decode"), name("greedy")],
        ),
        (
            name("decode_penalty_greedy"),
            merge_decode_penalty_greedy,
            [name("position_decode"), name("penalty"), name("penalty_greedy")],
        ),
        (
            name("decode_sampling"),
            merge_decode_sampling,
            [name("position_decode"), name("sampling")],
        ),
    ]


MERGED_BUILD_PLAN = make_merged_build_plan()


def dedup_tied_embed_into_head(
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
) -> dict[str, object]:
    """Rewrite a byte-verified row Embed over Main's transposed tied table."""
    embed_initializers = {
        initializer.name: initializer for initializer in embed.graph.initializer
    }
    main_initializers = {
        initializer.name: initializer for initializer in main.graph.initializer
    }
    gather_matches = [
        (index, node, embed_initializers[node.input[0]])
        for index, node in enumerate(embed.graph.node)
        if node.op_type == "Gather"
        and len(node.input) >= 2
        and node.input[0] in embed_initializers
        and len(embed_initializers[node.input[0]].dims) == 2
        and int(_node_attribute(node, "axis") or 0) == 0
    ]
    head_matches = [
        main_initializers[node.input[1]]
        for node in main.graph.node
        if node.op_type == "MatMul"
        and "/tgt_word_prj/" in node.name
        and len(node.input) >= 2
        and node.input[1] in main_initializers
    ]
    if len(gather_matches) != 1 or len(head_matches) != 1:
        raise RuntimeError(
            "Expected one FireRedASR Embed Gather and one vocabulary-head "
            f"MatMul; found {len(gather_matches)} and {len(head_matches)}."
        )
    gather_index, gather, embed_table = gather_matches[0]
    head_table = head_matches[0]
    embed_array = numpy_helper.to_array(embed_table)
    head_array = numpy_helper.to_array(head_table)
    if (
        embed_array.dtype != head_array.dtype
        or embed_array.shape != head_array.T.shape
        or embed_array.tobytes() != np.ascontiguousarray(head_array.T).tobytes()
    ):
        raise RuntimeError("FireRedASR Embed is not the exact transpose of its head.")

    occupied = {
        name
        for node in embed.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    temporary = gather.output[0] + "__from_tied_head"
    suffix = 1
    while temporary in occupied:
        temporary = gather.output[0] + f"__from_tied_head_{suffix}"
        suffix += 1
    public_output = gather.output[0]
    gather.input[0] = head_table.name
    gather.output[0] = temporary
    retained_attributes = [
        attribute for attribute in gather.attribute if attribute.name != "axis"
    ]
    del gather.attribute[:]
    gather.attribute.extend(retained_attributes)
    gather.attribute.append(onnx.helper.make_attribute("axis", 1))
    transpose = onnx.helper.make_node(
        "Transpose",
        [temporary],
        [public_output],
        name="embed_TiedHeadTranspose",
        perm=[1, 2, 0],
    )
    nodes = list(embed.graph.node)
    del embed.graph.node[:]
    embed.graph.node.extend(
        nodes[:gather_index + 1] + [transpose] + nodes[gather_index + 1:]
    )
    retained = [
        initializer
        for initializer in embed.graph.initializer
        if initializer.name != embed_table.name
    ]
    retained.append(copy.deepcopy(head_table))
    del embed.graph.initializer[:]
    embed.graph.initializer.extend(retained)
    return {
        "embed_initializer": embed_table.name,
        "head_initializer": head_table.name,
        "bytes_eliminated": int(embed_array.nbytes),
    }


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
) -> dict:
    """Build six strategies around one shared Encoder/Main/Embed blob."""
    source_folder = Path(source_folder)
    out_folder = source_folder if out_folder is None else Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    delete_obsolete_strategy_artifacts(source_folder, model_file_names)
    if out_folder.resolve() != source_folder.resolve():
        delete_obsolete_strategy_artifacts(out_folder, model_file_names)

    main_name = _model_file_name(model_file_names, "main")
    encoder_name = _model_file_name(model_file_names, "encoder")
    embed_name = _model_file_name(model_file_names, "embed")
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    shared_model_path = out_folder / shared_name
    shared_data_path = out_folder / shared_data_name

    for file_name, _, _ in make_merged_build_plan(model_file_names):
        (out_folder / file_name).unlink(missing_ok=True)
        (out_folder / (file_name + ".data")).unlink(missing_ok=True)
    shared_model_path.unlink(missing_ok=True)
    shared_data_path.unlink(missing_ok=True)

    main, shared = load_main_with_shared_initializers(
        source_folder, min_shared_elements, model_file_names
    )
    num_layers, main_cross_values = _main_cross_kv_values(main)
    encoder = load_model(source_folder / encoder_name)
    namespace_encoder_initializers(encoder)
    # Share only expensive Encoder weights. Small shape/control initializers must
    # remain inline because ORT shape inference reads them before add_initializer().
    for initializer in encoder.graph.initializer:
        if not _is_shareable_initializer(initializer, min_shared_elements):
            continue
        existing = shared.get(initializer.name)
        if existing is not None and existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"Encoder shared initializer collision: {initializer.name!r}."
            )
        shared[initializer.name] = initializer
    embed = prefixed(load_model(source_folder / embed_name), "embed_")
    embed_dedup = dedup_tied_embed_into_head(main, embed)
    print(
        "  Reused the tied vocabulary-head table for Embed; eliminated "
        f"{embed_dedup['bytes_eliminated']} bytes."
    )
    for initializer in embed.graph.initializer:
        if not _is_shareable_initializer(initializer, min_shared_elements):
            continue
        shared[initializer.name] = initializer

    save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    redirect_shared_initializers_to_external(main, external_by_name)
    redirect_shared_initializers_to_external(encoder, external_by_name)
    redirect_shared_initializers_to_external(embed, external_by_name)

    standalone_main = out_folder / main_name
    standalone_encoder = out_folder / encoder_name
    save_model(main, standalone_main)
    save_model(encoder, standalone_encoder)
    prefill_frontend = build_prefill_frontend(
        source_folder,
        encoder=encoder,
        model_file_names=model_file_names,
        num_layers=num_layers,
        reference_cross_values=main_cross_values,
    )

    graphs: dict[str, Path] = {}
    for file_name, recipe, _ in make_merged_build_plan(
        model_file_names,
        prefill_frontend=prefill_frontend,
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
        "standalone_main": standalone_main,
        "standalone_encoder": standalone_encoder,
    }
    if out_folder.resolve() == source_folder.resolve():
        result["removed_constituents"] = delete_merged_constituents(
            source_folder,
            model_file_names=model_file_names,
            protected_names=(
                shared_name,
                shared_data_name,
                main_name,
                encoder_name,
            ),
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


def _delete_named_graph_artifacts(
    folder: Path,
    file_names: tuple[str, ...],
    protected_names: set[str],
) -> list[str]:
    folder = Path(folder)
    if not folder.exists():
        return []
    root = folder.resolve()
    removed: list[str] = []
    for file_name in dict.fromkeys(file_names):
        onnx_path = folder / file_name
        locations: set[str] = set()
        if onnx_path.exists():
            locations = _external_locations(onnx_path)
        for location in locations:
            relative = Path(location)
            if relative.is_absolute() or location in protected_names:
                continue
            external_path = (folder / relative).resolve()
            if not external_path.is_relative_to(root):
                continue
            if external_path.exists() and external_path.is_file():
                external_path.unlink()
                removed.append(str(external_path.relative_to(root)))
        for candidate in (onnx_path, onnx_path.with_name(onnx_path.name + ".data")):
            if candidate.name in protected_names:
                continue
            if candidate.exists() or candidate.is_symlink():
                candidate.unlink()
                removed.append(candidate.name)
    return removed


def delete_obsolete_strategy_artifacts(
    folder: Path,
    model_file_names: dict[str, str] | None = None,
) -> list[str]:
    """Remove superseded split/merged strategy graphs and private sidecars."""
    old_family = "Be" + "am"
    obsolete_names = (
        f"FireRedASR_First_{old_family}_Search.onnx",
        f"FireRedASR_Second_{old_family}_Search.onnx",
        f"FireRedASR_Prefill{old_family}First.onnx",
        f"FireRedASR_Decode{old_family}Next.onnx",
        f"FireRedASR_DecodePenalty{old_family}Next.onnx",
    )
    protected = {
        _model_file_name(model_file_names, "shared_initializers"),
        _model_file_name(model_file_names, "shared_initializers_data"),
    }
    return _delete_named_graph_artifacts(Path(folder), obsolete_names, protected)


def _consolidate_external_data(
    source_onnx: Path,
    source_folder: Path,
    target_onnx: Path,
    target_folder: Path,
) -> Path:
    """Stream scattered initializer/Constant files into one ``<model>.onnx.data``."""
    model = onnx.load(str(source_onnx), load_external_data=False)
    sidecar_name = target_onnx.name + ".data"
    data_path = target_folder / sidecar_name
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.unlink(missing_ok=True)
    offset = 0
    with open(data_path, "wb") as data_file:
        for tensor in _iter_all_data_tensors(model.graph):
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            external = _external_data_map(tensor)
            location = external.get("location")
            if not location:
                raise RuntimeError(
                    f"External tensor {tensor.name!r} in {source_onnx.name} has no location."
                )
            relative = Path(location)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"Unsafe external-data location {location!r} in {source_onnx.name}."
                )
            source_data = source_folder / relative
            source_offset = int(external.get("offset", "0") or "0")
            declared_length = external.get("length")
            with open(source_data, "rb") as handle:
                handle.seek(source_offset)
                raw = (
                    handle.read(int(declared_length))
                    if declared_length
                    else handle.read()
                )
            data_file.write(raw)
            written = len(raw)
            del tensor.external_data[:]
            for key, value in (
                ("location", sidecar_name),
                ("offset", str(offset)),
                ("length", str(written)),
            ):
                entry = tensor.external_data.add()
                entry.key = key
                entry.value = value
            offset += written
    onnx.save(model, str(target_onnx))
    return data_path


def copy_runtime_standalones(
    source_folder: Path,
    target_folder: Path,
    model_file_names: dict[str, str] | None = None,
) -> list[Path]:
    """Copy only Encoder and Metadata; FireRed emits no Qwen KV helpers."""
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for role in RUNTIME_STANDALONE_MODEL_KEYS:
        file_name = _model_file_name(model_file_names, role)
        source = source_folder / file_name
        if not source.exists():
            if role in REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS:
                raise FileNotFoundError(source)
            continue
        target = target_folder / file_name
        target.unlink(missing_ok=True)
        target.with_name(target.name + ".data").unlink(missing_ok=True)
        locations = _external_locations(source)
        sidecar_name = file_name + ".data"
        if not locations:
            shutil.copy2(source, target)
        elif locations == {sidecar_name}:
            shutil.copy2(source, target)
            shutil.copy2(
                source_folder / sidecar_name,
                target_folder / sidecar_name,
            )
        else:
            _consolidate_external_data(
                source,
                source_folder,
                target,
                target_folder,
            )
        copied.append(target)
    return copied


def delete_merged_constituents(
    folder: Path,
    model_file_names: dict[str, str] | None = None,
    protected_names: tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    folder = Path(folder)
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    protected = set(protected_names or (shared_name, shared_data_name))
    removed: list[str] = []
    for role in MERGED_CONSTITUENT_MODEL_KEYS:
        path = folder / _model_file_name(model_file_names, role)
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
# Quantized-Main transplant: quantize one merged donor and reuse its Main.
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


def _tensor_element_types(model: onnx.ModelProto) -> dict[str, int]:
    """Collect and conservatively propagate top-level tensor element types."""
    types: dict[str, int] = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        if value.type.HasField("tensor_type"):
            elem_type = int(value.type.tensor_type.elem_type)
            if elem_type != TensorProto.UNDEFINED:
                types[value.name] = elem_type
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.UNDEFINED:
            types[initializer.name] = int(initializer.data_type)
    same_as_first = {
        "Abs", "Add", "Clip", "Concat", "Div", "Dropout", "Exp", "Expand",
        "Gather", "GatherElements", "Gemm", "LayerNormalization", "MatMul",
        "Mul", "Neg", "Pad", "Pow", "ReduceMean", "ReduceSum", "Relu",
        "Reshape", "ScatterElements", "Sigmoid", "Slice", "Softmax", "Split",
        "Sqrt", "Squeeze", "Sub", "Tanh", "Transpose", "Unsqueeze", "Where",
    }
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            inferred = None
            if node.op_type == "Cast":
                inferred = _node_attribute(node, "to")
            elif node.op_type == "Constant":
                inferred = next(
                    (
                        int(attribute.t.data_type)
                        for attribute in node.attribute
                        if attribute.name == "value" and attribute.HasField("t")
                    ),
                    None,
                )
            elif node.op_type in ("Shape", "Size", "NonZero", "ArgMax", "ArgMin"):
                inferred = TensorProto.INT64
            elif node.op_type in same_as_first and node.input:
                inferred = types.get(node.input[0])
                if inferred is None and len(node.input) > 1:
                    inferred = types.get(node.input[1])
            if inferred in (None, TensorProto.UNDEFINED):
                continue
            for output in node.output:
                if output and output not in types:
                    types[output] = int(inferred)
                    changed = True
    return types


def _remap_element_types(
    types: dict[str, int],
    remap: dict[str, str],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, elem_type in types.items():
        target = remap.get(name, name)
        existing = result.get(target)
        if existing is not None and existing != elem_type:
            raise RuntimeError(
                f"Conflicting element types after remapping {target!r}."
            )
        result[target] = elem_type
    return result


def _unique_tensor_name(base: str, reserved: set[str]) -> str:
    name = base
    suffix = 1
    while name in reserved:
        name = f"{base}_{suffix}"
        suffix += 1
    reserved.add(name)
    return name


def _merge_opsets(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    opsets: dict[str, int] = {}
    for model in (destination, *sources):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    del destination.opset_import[:]
    for domain, version in sorted(opsets.items()):
        destination.opset_import.add(domain=domain, version=version)

    functions: dict[tuple[str, str], onnx.FunctionProto] = {}
    for model in (destination, *sources):
        for function in model.functions:
            key = (function.domain, function.name)
            existing = functions.get(key)
            if existing is None:
                functions[key] = function
            elif existing.SerializeToString() != function.SerializeToString():
                raise RuntimeError(
                    "Local FunctionProto collision with different bodies: "
                    f"domain={function.domain!r}, name={function.name!r}."
                )
    del destination.functions[:]
    destination.functions.extend(functions.values())


def _target_position_remap(
    target: onnx.ModelProto,
    donor_main_nodes: list[onnx.NodeProto],
) -> dict[str, str]:
    names = {value.name for value in target.graph.input}
    names.update(initializer.name for initializer in target.graph.initializer)
    for node in target.graph.node:
        names.update(node.output)
    donor_inputs = _used_inputs(donor_main_nodes)
    decode = "decode_position_embed" in names
    remap = {
        "hidden_states": "embed_hidden_states",
        "position_embed": (
            "decode_position_embed" if decode else "prefill_position_embed"
        ),
        "attention_mask": (
            "decode_zero_attention_mask" if decode else "prefill_attention_mask"
        ),
    }
    if not decode:
        encoder_outputs = {
            output
            for node in target.graph.node
            for output in node.output
            if output.startswith("encoder_en_")
        }
        expected = {
            name for name in donor_inputs if name.startswith(("en_key_", "en_value_"))
        }
        mapped = {f"encoder_{name}" for name in expected}
        if mapped - encoder_outputs:
            raise RuntimeError(
                "Prefill template is missing Encoder cross-KV boundary values: "
                f"{sorted(mapped - encoder_outputs)}."
            )
        remap.update({name: f"encoder_{name}" for name in expected})
    return {name: value for name, value in remap.items() if name in donor_inputs}


def transplant_quantized_main(
    target: onnx.ModelProto, quantized_primary: onnx.ModelProto
) -> onnx.ModelProto:
    """Replace a target shell's Main block with the quantized donor Main block."""
    donor_main_nodes = [
        node for node in quantized_primary.graph.node if not _node_is_shell(node)
    ]
    remap = _target_position_remap(target, donor_main_nodes)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in donor_main_nodes
    ]
    if not primary_main_nodes:
        raise RuntimeError("Quantized primary graph has no Main node block.")

    primary_types = _remap_element_types(
        _tensor_element_types(quantized_primary), remap
    )
    target_types = _tensor_element_types(target)
    main_outputs = {
        output for node in primary_main_nodes for output in node.output if output
    }
    target_shell_nodes = [node for node in target.graph.node if _node_is_shell(node)]
    shell_outputs = {
        output for node in target_shell_nodes for output in node.output if output
    }
    reserved = {value.name for value in target.graph.input}
    reserved.update(initializer.name for initializer in target.graph.initializer)
    reserved.update(output for node in target.graph.node for output in node.output if output)
    reserved.update(main_outputs)

    main_input_remap: dict[str, str] = {}
    pre_main_casts: list[onnx.NodeProto] = []
    external_inputs = _used_inputs(primary_main_nodes) - main_outputs
    for name in sorted(external_inputs & shell_outputs):
        source_type = target_types.get(name)
        target_type = primary_types.get(name)
        if source_type is None or target_type is None or source_type == target_type:
            continue
        adapted = _unique_tensor_name(f"{name}_cast_to_main", reserved)
        pre_main_casts.append(
            onnx.helper.make_node(
                "Cast",
                [name],
                [adapted],
                name=f"BoundaryCast/{name}/to_main",
                to=target_type,
            )
        )
        main_input_remap[name] = adapted
    if main_input_remap:
        primary_main_nodes = [
            _copy_node_with_input_remap(node, main_input_remap)
            for node in primary_main_nodes
        ]

    shell_input_remap: dict[str, str] = {}
    post_main_casts: list[onnx.NodeProto] = []
    downstream_inputs = _used_inputs(target_shell_nodes)
    for name in sorted(main_outputs & downstream_inputs):
        source_type = primary_types.get(name)
        target_type = target_types.get(name)
        if source_type is None or target_type is None or source_type == target_type:
            continue
        adapted = _unique_tensor_name(f"{name}_cast_to_shell", reserved)
        post_main_casts.append(
            onnx.helper.make_node(
                "Cast",
                [name],
                [adapted],
                name=f"BoundaryCast/{name}/to_shell",
                to=target_type,
            )
        )
        shell_input_remap[name] = adapted

    merged = copy.deepcopy(target)
    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in target.graph.node:
        if _node_is_shell(node):
            new_nodes.append(_copy_node_with_input_remap(node, shell_input_remap))
        elif not inserted:
            new_nodes.extend(copy.deepcopy(pre_main_casts))
            new_nodes.extend(copy.deepcopy(primary_main_nodes))
            new_nodes.extend(copy.deepcopy(post_main_casts))
            inserted = True
    if not inserted:
        new_nodes.extend(copy.deepcopy(pre_main_casts))
        new_nodes.extend(copy.deepcopy(primary_main_nodes))
        new_nodes.extend(copy.deepcopy(post_main_casts))

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

    def add_value_info(value_info: onnx.ValueInfoProto, name: str | None = None) -> None:
        value_name = value_info.name if name is None else name
        if value_name not in existing:
            value_infos.append(_copy_value_info_with_name(value_info, value_name))
            existing.add(value_name)

    for value_info in quantized_primary.graph.value_info:
        name = remap.get(value_info.name, value_info.name)
        add_value_info(value_info, main_input_remap.get(name, name))
    for value_info in target.graph.value_info:
        if value_info.name.startswith(SHELL_PREFIXES):
            add_value_info(value_info)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(value_infos)
    _merge_opsets(merged, quantized_primary)
    simplify_argmax_logits_cast(merged)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    return merged


def _node_is_encoder_component(node: onnx.NodeProto) -> bool:
    if node.name.startswith("BoundaryCast/"):
        return False
    return any(
        output.removeprefix(_PRECISION_FREE_CAST_PREFIX).startswith("encoder_")
        for output in node.output
        if output
    )


def transplant_optimized_encoder(
    target: onnx.ModelProto,
    optimized_encoder: onnx.ModelProto,
) -> onnx.ModelProto:
    """Replace a prefill template's raw Encoder with one optimized donor."""
    target_encoder_nodes = [
        node for node in target.graph.node if _node_is_encoder_component(node)
    ]
    if not target_encoder_nodes:
        public_names = {
            value.name for value in (*target.graph.input, *target.graph.output)
        }
        if "audio" in public_names or any(
            name.startswith("encoder_") for name in public_names
        ):
            raise RuntimeError("Encoder-prefill graph has no classifiable Encoder nodes.")
        return target

    donor_source = optimized_encoder
    if any(
        not initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
        for initializer in optimized_encoder.graph.initializer
    ):
        donor_source = copy.deepcopy(optimized_encoder)
        namespace_encoder_initializers(donor_source)
    donor = prefixed_preserving_initializers(donor_source, "encoder_")
    donor_nodes = [
        _copy_node_with_input_remap(node, {"encoder_audio": "audio"})
        for node in donor.graph.node
    ]

    target_public_outputs = {value.name for value in target.graph.output}
    required_boundary = {
        output
        for node in target_encoder_nodes
        for output in node.output
        if output
        and (
            output in target_public_outputs
            or any(
                output in consumer.input
                for consumer in target.graph.node
                if not _node_is_encoder_component(consumer)
            )
        )
    }
    donor_outputs = {
        output for node in donor_nodes for output in node.output if output
    }
    missing = required_boundary - donor_outputs
    if missing:
        raise RuntimeError(
            f"Optimized Encoder does not reproduce boundary values: {sorted(missing)}."
        )

    target_initializers = {
        initializer.name: initializer for initializer in target.graph.initializer
    }
    donor_initializers = {
        initializer.name: initializer for initializer in donor.graph.initializer
    }
    old_encoder_initializers = _used_inputs(target_encoder_nodes) & set(
        target_initializers
    )
    donor_initializer_names = _used_inputs(donor_nodes) & set(donor_initializers)

    merged = copy.deepcopy(target)
    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in target.graph.node:
        if _node_is_encoder_component(node):
            if not inserted:
                new_nodes.extend(copy.deepcopy(donor_nodes))
                inserted = True
            continue
        new_nodes.append(copy.deepcopy(node))
    if not inserted:
        raise RuntimeError("Prefill template lost its Encoder insertion point.")

    new_initializers: list[TensorProto] = []
    seen: set[str] = set()
    for initializer in target.graph.initializer:
        if initializer.name in old_encoder_initializers:
            continue
        new_initializers.append(copy.deepcopy(initializer))
        seen.add(initializer.name)
    for name in sorted(donor_initializer_names):
        initializer = donor_initializers[name]
        if name in seen:
            existing = next(item for item in new_initializers if item.name == name)
            if existing.SerializeToString() != initializer.SerializeToString():
                raise RuntimeError(
                    f"Optimized Encoder initializer {name!r} collides with Main."
                )
            continue
        new_initializers.append(copy.deepcopy(initializer))
        seen.add(name)

    del merged.graph.node[:]
    merged.graph.node.extend(new_nodes)
    del merged.graph.initializer[:]
    merged.graph.initializer.extend(new_initializers)

    retained_info = [
        value
        for value in target.graph.value_info
        if not value.name.startswith(("encoder_", _PRECISION_FREE_CAST_PREFIX + "encoder_"))
    ]
    existing_info = {value.name for value in merged.graph.input}
    existing_info.update(value.name for value in merged.graph.output)
    existing_info.update(initializer.name for initializer in merged.graph.initializer)
    existing_info.update(value.name for value in retained_info)
    for value in list(donor.graph.output) + list(donor.graph.value_info):
        name = "audio" if value.name == "encoder_audio" else value.name
        if name not in existing_info:
            retained_info.append(_copy_value_info_with_name(value, name))
            existing_info.add(name)
    del merged.graph.value_info[:]
    merged.graph.value_info.extend(retained_info)
    _merge_opsets(merged, optimized_encoder)
    copy_metadata(merged, optimized_encoder)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    return merged


def extract_and_write_shared(
    models: dict[str, onnx.ModelProto] | list[onnx.ModelProto],
    shared_model_path: Path,
    primary_model: onnx.ModelProto | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    additional_shared: dict[str, TensorProto] | None = None,
) -> dict[str, dict[str, str]]:
    values = list(models.values()) if isinstance(models, dict) else list(models)
    if not values:
        raise RuntimeError("No merged model was supplied for shared extraction.")
    source = values[0] if primary_model is None else primary_model
    main_nodes = [node for node in source.graph.node if not _node_is_shell(node)]
    encoder_nodes = [
        node for node in source.graph.node if _node_is_encoder_component(node)
    ]
    component_inputs = _used_inputs(main_nodes + encoder_nodes)
    shared = {
        initializer.name: initializer
        for initializer in source.graph.initializer
        if initializer.name in component_inputs
        and _is_shareable_initializer(initializer, min_shared_elements)
    }
    for name, initializer in (additional_shared or {}).items():
        if name != initializer.name:
            raise RuntimeError(
                f"Additional shared initializer key/name mismatch: "
                f"{name!r} != {initializer.name!r}."
            )
        if (
            initializer.data_type in (TensorProto.UNDEFINED, TensorProto.STRING)
            or initializer.data_type in _UNSHAREABLE_INIT_TYPES
        ):
            raise RuntimeError(
                f"Additional shared initializer {name!r} has an unsupported type."
            )
        existing = shared.get(name)
        if existing is not None:
            left = existing.raw_data or numpy_helper.to_array(existing).tobytes()
            right = initializer.raw_data or numpy_helper.to_array(initializer).tobytes()
            if (
                existing.data_type != initializer.data_type
                or tuple(existing.dims) != tuple(initializer.dims)
                or left != right
            ):
                raise RuntimeError(f"Additional shared initializer collision for {name!r}.")
        shared[name] = initializer
    if not shared:
        raise RuntimeError("Optimized Encoder/Main graph has no shareable initializer.")

    save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    for model in values:
        redirect_shared_initializers_to_external(model, external_by_name)
    return external_by_name


# ---------------------------------------------------------------------------
# Runtime attachment: mmap shared bytes and retain returned references.
# ---------------------------------------------------------------------------

def attach_shared_initializers(session_options, shared_model_path: Path):
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
