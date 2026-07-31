"""Build Dolphin merged graphs backed by one shared initializer blob.

The canonical implementation supports two explicit layouts: legacy decoder-only
composition for probe-aware v1, and direct Encoder+prefill composition for the
current ``small.cn.prompt`` deployment. Every graph keeps Main names unprefixed;
Encoder weights use a stable private namespace when direct mode is selected.
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
_EXTERNAL_DATA_TYPE_KEY = "__tensor_data_type"
_EXTERNAL_DIMS_KEY = "__tensor_dims"
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

# Every non-Main graph is prefixed.  This list is also the quantized-Main
# transplantation boundary, so it must contain every prefix introduced below.
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

PREFILL_GREEDY_MODEL_NAME = "Dolphin_PrefillGreedy.onnx"
PREFILL_PENALTY_GREEDY_MODEL_NAME = "Dolphin_PrefillPenaltyGreedy.onnx"
PREFILL_SAMPLING_MODEL_NAME = "Dolphin_PrefillSampling.onnx"
PROBE_PREFILL_GREEDY_MODEL_NAME = "Dolphin_ProbePrefillGreedy.onnx"
PROBE_PREFILL_PENALTY_GREEDY_MODEL_NAME = "Dolphin_ProbePrefillPenaltyGreedy.onnx"
PROBE_PREFILL_SAMPLING_MODEL_NAME = "Dolphin_ProbePrefillSampling.onnx"
DECODE_GREEDY_MODEL_NAME = "Dolphin_DecodeGreedy.onnx"
DECODE_PENALTY_GREEDY_MODEL_NAME = "Dolphin_DecodePenaltyGreedy.onnx"
DECODE_SAMPLING_MODEL_NAME = "Dolphin_DecodeSampling.onnx"
SHARED_MODEL_NAME = "Dolphin_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"

DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Dolphin_Encoder.onnx",
    "main": "Dolphin_Decoder.onnx",
    "embed": "Dolphin_Decoder_Embed.onnx",
    "position_prefill": "Dolphin_Position_Mask_Prefill.onnx",
    "position_decode": "Dolphin_Position_Mask_Decode.onnx",
    "greedy": "Dolphin_Greedy_Search.onnx",
    "argmax": "Dolphin_Argmax.onnx",
    "sampling": "Dolphin_TopKTopPSampling.onnx",
    "penalty": "Dolphin_Apply_Penalty.onnx",
    "prefill_greedy": PREFILL_GREEDY_MODEL_NAME,
    "prefill_penalty_greedy": PREFILL_PENALTY_GREEDY_MODEL_NAME,
    "prefill_sampling": PREFILL_SAMPLING_MODEL_NAME,
    "probe_prefill_greedy": PROBE_PREFILL_GREEDY_MODEL_NAME,
    "probe_prefill_penalty_greedy": PROBE_PREFILL_PENALTY_GREEDY_MODEL_NAME,
    "probe_prefill_sampling": PROBE_PREFILL_SAMPLING_MODEL_NAME,
    "decode_greedy": DECODE_GREEDY_MODEL_NAME,
    "decode_penalty_greedy": DECODE_PENALTY_GREEDY_MODEL_NAME,
    "decode_sampling": DECODE_SAMPLING_MODEL_NAME,
    "shared_initializers": SHARED_MODEL_NAME,
    "shared_initializers_data": SHARED_DATA_NAME,
}

# Dolphin emits no Qwen-v3 KV helper graphs.  Only these run-time standalones
# survive the merge; vocab_Dolphin_CN_Dialect.txt is copied by the exporter.
RUNTIME_STANDALONE_MODEL_KEYS = ("metadata", "encoder")
REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS = frozenset(RUNTIME_STANDALONE_MODEL_KEYS)
MERGED_CONSTITUENT_MODEL_KEYS = (
    "main",
    "embed",
    "position_prefill",
    "position_decode",
    "greedy",
    "argmax",
    "sampling",
    "penalty",
)


def _model_file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = DEFAULT_MODEL_FILE_NAMES if model_file_names is None else {
        **DEFAULT_MODEL_FILE_NAMES,
        **model_file_names,
    }
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
    """Save a data-light merged graph without creating a private sidecar."""
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
    shared: dict[str, TensorProto],
    path: Path,
) -> None:
    """Stream initializer bytes into one external-data file with low peak memory."""
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
        nodes=[],
        name="dolphin_shared_initializers",
        inputs=[],
        outputs=[],
        initializer=references,
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="Dolphin/Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    onnx.save_model(model, str(path))


def shared_external_data_map(shared_model_path: Path) -> dict[str, dict[str, str]]:
    model = onnx.load(str(shared_model_path), load_external_data=False)
    result: dict[str, dict[str, str]] = {}
    for initializer in model.graph.initializer:
        external = _external_data_map(initializer)
        external[_EXTERNAL_DATA_TYPE_KEY] = str(initializer.data_type)
        external[_EXTERNAL_DIMS_KEY] = ",".join(str(dim) for dim in initializer.dims)
        result[initializer.name] = external
    return result


def make_external_initializer_ref(
    initializer: TensorProto,
    external_data: dict[str, str],
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


def prefixed_preserving_initializers(
    model: onnx.ModelProto,
    prefix: str,
) -> onnx.ModelProto:
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


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


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
    if encoder is None:
        encoder = load_model(
            source_folder / _model_file_name(model_file_names, "encoder")
        )
    if num_layers is None:
        num_layers = len(encoder.graph.output) // 2
    names = _cross_kv_names(num_layers)
    namespace_encoder_initializers(encoder)
    component = prefixed_preserving_initializers(encoder, "encoder_")
    rename_tensor(component, "encoder_audio", "audio")
    return component, [f"encoder_{name}" for name in names]


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
            raise RuntimeError(f"Cannot restore value_info for merged tensor {name!r}.")
        target.graph.value_info.append(value)
        target_names[name] = value


def set_graph_outputs(model: onnx.ModelProto, output_names: list[str]) -> None:
    by_name = value_info_by_name(model)
    del model.graph.output[:]
    model.graph.output.extend([by_name[name] for name in output_names])


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


def minimize_public_outputs(model: onnx.ModelProto) -> list[str]:
    """Remove outputs not consumed by the matching Dolphin runtime stage."""
    inputs = {value.name for value in model.graph.input}
    is_decode = "decode_kv_seq_len" in inputs
    removable: set[str] = set()
    if is_decode:
        removable.add("logits")
    removed = [value.name for value in model.graph.output if value.name in removable]
    if removed:
        retained = [
            value for value in model.graph.output if value.name not in removable
        ]
        del model.graph.output[:]
        model.graph.output.extend(retained)
        prune_unreachable_nodes(model)
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


def copy_metadata(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    existing = {property_.key: property_ for property_ in dst.metadata_props}
    for source in sources:
        for property_ in source.metadata_props:
            if property_.key in existing:
                existing[property_.key].value = property_.value
            else:
                existing[property_.key] = dst.metadata_props.add(
                    key=property_.key,
                    value=property_.value,
                )


def restore_precision_free_graph_outputs(
    model: onnx.ModelProto,
    *,
    alias_prefix: str = _PRECISION_FREE_CAST_PREFIX,
) -> dict[str, str]:
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
                f"Cannot restore Dolphin output {public_name!r}: "
                f"alias {alias!r} has {len(owners)} producers."
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
        annotation.tensor_name = remap.get(annotation.tensor_name, annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)
    return remap


def restore_float16_merged_boundary_names(
    model: onnx.ModelProto,
) -> dict[str, str]:
    """Repair precision-free aliases and shell/Main dtypes at merge boundaries.

    The ORT float16 converter may insert a no-op Cast and rename its producer to
    ``InsertedPrecisionFreeCast_<name>``.  onnxslim can remove that Cast without
    reconnecting the original graph output.  It can do the same to the prefill
    mask at the shell/Main boundary, which also defeats Main transplantation.
    Restore only those explicit ABI values and reject dtype/name collisions.  A
    blocked float32 Embed/position shell also needs an explicit Cast before the
    converted F16 Main; otherwise its first LayerNormalization receives float32
    while its scale is float16.
    """
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

    # Public graph outputs must remain exactly addressable by their declared names.
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

    # These are the only float shell values consumed by the unprefixed Main.
    for name in (
        "embed_hidden_states",
        "prefill_position_embed",
        "decode_position_embed",
        "prefill_attention_mask",
        "decode_zero_attention_mask",
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
        if alias_info is not None and output_info is not None:
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

    # Keep reusable strategy shells in their source precision, and adapt only at
    # the Main boundary.  The generated Casts are unprefixed Main nodes, so they
    # are transplanted and position-name-remapped together with the F16 decoder.
    type_by_name = {
        value.name: value
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
    }
    for initializer in model.graph.initializer:
        if initializer.name in type_by_name:
            continue
        type_by_name[initializer.name] = onnx.helper.make_tensor_value_info(
            initializer.name,
            initializer.data_type,
            list(initializer.dims),
        )

    initializer_types = {
        initializer.name: initializer.data_type
        for initializer in model.graph.initializer
    }

    def concrete_producer_type(name: str) -> int | None:
        producer = producers.get(name)
        if producer is None:
            return None
        if producer.op_type == "Cast":
            return _node_attribute(producer, "to")
        if producer.op_type == "Gather" and producer.input:
            return initializer_types.get(producer.input[0])
        if producer.op_type == "Constant":
            for attribute in producer.attribute:
                if attribute.name == "value" and attribute.HasField("t"):
                    return attribute.t.data_type
        return None
    nodes = list(model.graph.node)
    casts_by_index: dict[int, list[onnx.NodeProto]] = {}
    cast_value_infos: list[onnx.ValueInfoProto] = []
    cast_count = 0
    boundary_names = (
        "embed_hidden_states",
        "prefill_position_embed",
        "decode_position_embed",
        "prefill_attention_mask",
        "decode_zero_attention_mask",
    )
    for boundary_name in boundary_names:
        concrete_type = concrete_producer_type(boundary_name)
        value_info = type_by_name.get(boundary_name)
        if value_info is None and concrete_type is not None:
            value_info = onnx.helper.make_tensor_value_info(
                boundary_name,
                concrete_type,
                None,
            )
        if value_info is None:
            continue
        if not value_info.type.HasField("tensor_type"):
            raise RuntimeError(
                f"Float16 donor is missing tensor metadata for {boundary_name!r}."
            )
        elem_type = concrete_type
        if elem_type is None:
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

    # Symbolic inference can fail during conversion and leave stale FLOAT
    # value_info on F16 Main tensors.  Those declarations override schema
    # inference and make an otherwise-correct LayerNormalization look mixed-type.
    # Keep only trustworthy shell/adapter metadata; Main intermediates are
    # inferred from their nodes by ONNX Runtime.
    producer_after_repair = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    boundary_names = set(boundary_names)
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
    pruned_value_info_count = len(model.graph.value_info) - len(retained_value_infos)
    if pruned_value_info_count:
        del model.graph.value_info[:]
        model.graph.value_info.extend(retained_value_infos)

    return aliases


def merge_models_no_check(
    first: onnx.ModelProto,
    second: onnx.ModelProto,
    io_map: list[tuple[str, str]],
) -> onnx.ModelProto:
    """Compose two models without invoking ONNX's >2 GiB checker path."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "Dolphin/Shared_Merged.py"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"

    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    functions: dict[tuple[str, str, str], onnx.FunctionProto] = {}
    for model in (first, second):
        for function in model.functions:
            key = (function.domain, function.name, function.overload)
            existing = functions.get(key)
            if existing is None:
                functions[key] = function
            elif existing.SerializeToString() != function.SerializeToString():
                raise RuntimeError(
                    "Local FunctionProto collision with different bodies: "
                    f"domain={function.domain!r}, name={function.name!r}, "
                    f"overload={function.overload!r}."
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

    initializer_by_name: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
        existing = initializer_by_name.get(initializer.name)
        if existing is None:
            initializer_by_name[initializer.name] = initializer
        elif existing is not initializer and existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"Initializer name collision with different data: {initializer.name}"
            )
    merged.graph.initializer.extend(initializer_by_name.values())

    merged.graph.node.extend(first.graph.node)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    for node in merged.graph.node[second_start:]:
        for index, name in enumerate(node.input):
            replacement = source_by_target.get(name)
            if replacement is not None:
                node.input[index] = replacement

    seen_value_info = {value.name for value in merged.graph.input}
    seen_value_info.update(initializer_by_name)
    for value in list(first.graph.value_info) + list(second.graph.value_info):
        if value.name not in seen_value_info:
            merged.graph.value_info.append(value)
            seen_value_info.add(value.name)

    seen_outputs: set[str] = set()
    candidates = [
        value for value in first.graph.output if value.name not in mapped_sources
    ] + list(second.graph.output)
    for value in candidates:
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
    if not shared:
        raise RuntimeError(f"{main_path.name} has no shareable initializer.")
    return main, shared


def _main_kv_output_names(main: onnx.ModelProto) -> list[str]:
    return [output.name for output in main.graph.output if output.name.startswith("out_de_")]


def _order_kv_inputs_first(model: onnx.ModelProto) -> None:
    kv_inputs = [value for value in model.graph.input if value.name.startswith("in_de_")]
    other_inputs = [value for value in model.graph.input if not value.name.startswith("in_de_")]
    del model.graph.input[:]
    model.graph.input.extend(kv_inputs + other_inputs)


def _merge_position_embed_into_main(
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
            load_model(source_folder / _model_file_name(model_file_names, "position_prefill")),
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
        if prefill_frontend is None:
            return merged, "prefill_kv_seq_len", position, []
        frontend, cross_outputs = prefill_frontend
        io_map = [
            (encoder_name, main_value.name)
            for encoder_name, main_value in zip(cross_outputs, main_cross_values)
        ]
        merged = merge_models_no_check(frontend, merged, io_map)
        _ensure_value_info_from(merged, frontend, cross_outputs)
        return merged, "prefill_kv_seq_len", position, cross_outputs

    position = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "position_decode")),
        "decode_",
    )
    mask_info = next(value for value in main.graph.input if value.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(mask_info.type.tensor_type.elem_type)
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
    return merged, "decode_kv_seq_len_next", position, []


def _finalize(
    merged: onnx.ModelProto,
    main: onnx.ModelProto,
    position: onnx.ModelProto,
    output_names: list[str],
) -> onnx.ModelProto:
    _ensure_value_info_from(merged, main, ("logits",))
    set_graph_outputs(merged, output_names)
    minimize_public_outputs(merged)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    copy_metadata(merged, main, position)
    merged.producer_name = "Dolphin/Shared_Merged.py"
    return merged


def merge_prefill_greedy(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "prefill", model_file_names, prefill_frontend
    )
    argmax = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "argmax")),
        "argmax_",
    )
    merged = merge_models_no_check(merged, argmax, io_map=[("logits", "argmax_logits")])
    outputs = _main_kv_output_names(main) + cross_outputs + [
        "argmax_max_logits_idx", "logits", kv_seq_len
    ]
    return _finalize(merged, main, position, outputs)


def merge_decode_greedy(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    if prefill_frontend is not None:
        raise RuntimeError("Decode graph must not receive an Encoder frontend.")
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "decode", model_file_names
    )
    if cross_outputs:
        raise RuntimeError("Decode graph unexpectedly acquired Encoder outputs.")
    argmax = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "argmax")),
        "argmax_",
    )
    merged = merge_models_no_check(merged, argmax, io_map=[("logits", "argmax_logits")])
    outputs = _main_kv_output_names(main) + ["argmax_max_logits_idx", kv_seq_len]
    return _finalize(merged, main, position, outputs)


def merge_prefill_penalty_greedy(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "prefill", model_file_names, prefill_frontend
    )
    greedy = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "greedy")),
        "greedy_",
    )
    merged = merge_models_no_check(merged, greedy, io_map=[("logits", "greedy_logits")])
    outputs = _main_kv_output_names(main) + cross_outputs + [
        "greedy_max_logits_idx",
        "greedy_save_id_out",
        "logits",
        kv_seq_len,
    ]
    return _finalize(merged, main, position, outputs)


def merge_decode_penalty_greedy(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    if prefill_frontend is not None:
        raise RuntimeError("Decode graph must not receive an Encoder frontend.")
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "decode", model_file_names
    )
    if cross_outputs:
        raise RuntimeError("Decode graph unexpectedly acquired Encoder outputs.")
    penalty = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty")),
        "penalty_",
    )
    greedy = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "greedy")),
        "greedy_",
    )
    merged = merge_models_no_check(merged, penalty, io_map=[("logits", "penalty_logits_in")])
    merged = merge_models_no_check(
        merged,
        greedy,
        io_map=[("penalty_logits_out", "greedy_logits")],
    )
    outputs = _main_kv_output_names(main) + [
        "greedy_max_logits_idx",
        "greedy_save_id_out",
        kv_seq_len,
    ]
    return _finalize(merged, main, position, outputs)


def _merge_sampling(
    source_folder,
    main,
    embed,
    kind,
    model_file_names=None,
    prefill_frontend=None,
):
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, kind, model_file_names, prefill_frontend
    )
    sampling = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "sampling")),
        "sampling_",
    )
    merged = merge_models_no_check(
        merged,
        sampling,
        io_map=[("logits", "sampling_logits")],
    )
    outputs = _main_kv_output_names(main) + cross_outputs + [
        "sampling_sampled_id",
        "sampling_save_id_out",
    ]
    if kind == "prefill":
        outputs.append("logits")
    outputs.append(kv_seq_len)
    return _finalize(merged, main, position, outputs)


def merge_prefill_sampling(
    source_folder, main, embed, model_file_names=None, prefill_frontend=None
):
    return _merge_sampling(
        source_folder, main, embed, "prefill", model_file_names, prefill_frontend
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
    *,
    merge_encoder_into_prefill: bool = False,
    prefill_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
):
    position_prefill = _model_file_name(model_file_names, "position_prefill")
    position_decode = _model_file_name(model_file_names, "position_decode")
    argmax = _model_file_name(model_file_names, "argmax")
    greedy = _model_file_name(model_file_names, "greedy")
    sampling = _model_file_name(model_file_names, "sampling")
    penalty = _model_file_name(model_file_names, "penalty")
    encoder_dependencies = (
        [_model_file_name(model_file_names, "encoder")]
        if merge_encoder_into_prefill
        else []
    )
    return [
        (
            _model_file_name(model_file_names, "prefill_greedy"),
            _recipe_with_frontend(merge_prefill_greedy, prefill_frontend),
            [*encoder_dependencies, position_prefill, argmax],
        ),
        (
            _model_file_name(model_file_names, "prefill_penalty_greedy"),
            _recipe_with_frontend(merge_prefill_penalty_greedy, prefill_frontend),
            [*encoder_dependencies, position_prefill, greedy],
        ),
        (
            _model_file_name(model_file_names, "prefill_sampling"),
            _recipe_with_frontend(merge_prefill_sampling, prefill_frontend),
            [*encoder_dependencies, position_prefill, sampling],
        ),
        (
            _model_file_name(model_file_names, "decode_greedy"),
            merge_decode_greedy,
            [position_decode, argmax],
        ),
        (
            _model_file_name(model_file_names, "decode_penalty_greedy"),
            merge_decode_penalty_greedy,
            [position_decode, penalty, greedy],
        ),
        (
            _model_file_name(model_file_names, "decode_sampling"),
            merge_decode_sampling,
            [position_decode, sampling],
        ),
    ]


def make_probe_aware_build_plan(
    model_file_names: dict[str, str] | None = None,
    *,
    probe_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
):
    cached_plan = make_merged_build_plan(
        model_file_names,
        merge_encoder_into_prefill=False,
    )
    name = lambda role: _model_file_name(model_file_names, role)
    probe_plan = [
        (
            name("probe_prefill_greedy"),
            _recipe_with_frontend(merge_prefill_greedy, probe_frontend),
            [name("encoder"), name("position_prefill"), name("argmax")],
        ),
        (
            name("probe_prefill_penalty_greedy"),
            _recipe_with_frontend(merge_prefill_penalty_greedy, probe_frontend),
            [name("encoder"), name("position_prefill"), name("greedy")],
        ),
        (
            name("probe_prefill_sampling"),
            _recipe_with_frontend(merge_prefill_sampling, probe_frontend),
            [name("encoder"), name("position_prefill"), name("sampling")],
        ),
    ]
    return [*probe_plan, *cached_plan]


MERGED_BUILD_PLAN = make_merged_build_plan()


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
    retain_prefill_logits: bool = True,
    *,
    merge_encoder_into_prefill: bool = False,
    probe_aware: bool = False,
) -> dict:
    """Build six Dolphin strategies in legacy or direct Encoder+prefill layout."""
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    delete_obsolete_strategy_artifacts(source_folder, model_file_names)
    if out_folder.resolve() != source_folder.resolve():
        delete_obsolete_strategy_artifacts(out_folder, model_file_names)

    main_name = _model_file_name(model_file_names, "main")
    encoder_name = _model_file_name(model_file_names, "encoder")
    embed_name = _model_file_name(model_file_names, "embed")
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    if shared_data_name != shared_name + ".data":
        raise RuntimeError(
            f"Shared data must be named {shared_name + '.data'!r}, got {shared_data_name!r}."
        )
    if merge_encoder_into_prefill and probe_aware:
        raise ValueError("Dolphin direct and probe-aware modes are mutually exclusive.")
    has_encoder_frontend = merge_encoder_into_prefill or probe_aware
    shared_model_path = out_folder / shared_name
    shared_data_path = out_folder / shared_data_name
    for legacy in (
        "shared_initializers.npz",
        "shared_initializers.manifest.json",
        "shared_initializers.onnx",
        "shared_initializers.onnx.data",
    ):
        (out_folder / legacy).unlink(missing_ok=True)

    main, shared = load_main_with_shared_initializers(
        source_folder,
        min_shared_elements,
        model_file_names,
    )
    encoder = None
    prefill_frontend = None
    standalone_main = None
    standalone_encoder = None
    if has_encoder_frontend:
        num_layers, main_cross_values = _main_cross_kv_values(main)
        encoder = load_model(source_folder / encoder_name)
        namespace_encoder_initializers(encoder)
        for initializer in encoder.graph.initializer:
            if not _is_shareable_initializer(initializer, min_shared_elements):
                continue
            existing = shared.get(initializer.name)
            if existing is not None and existing.SerializeToString() != initializer.SerializeToString():
                raise RuntimeError(
                    f"Dolphin Encoder shared initializer collision: {initializer.name!r}."
                )
            shared[initializer.name] = initializer
        prefill_frontend = build_prefill_frontend(
            source_folder,
            encoder=encoder,
            model_file_names=model_file_names,
            num_layers=num_layers,
            reference_cross_values=main_cross_values,
        )
    embed = prefixed(load_model(source_folder / embed_name), "embed_")
    for initializer in embed.graph.initializer:
        if _is_shareable_initializer(initializer, min_shared_elements):
            shared[initializer.name] = initializer

    save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    redirect_shared_initializers_to_external(main, external_by_name)
    if encoder is not None:
        redirect_shared_initializers_to_external(encoder, external_by_name)
    redirect_shared_initializers_to_external(embed, external_by_name)

    if has_encoder_frontend:
        standalone_main = out_folder / main_name
        standalone_encoder = out_folder / encoder_name
        save_model(main, standalone_main)
        save_model(encoder, standalone_encoder)

    graphs: dict[str, Path] = {}
    build_plan = (
        make_probe_aware_build_plan(
            model_file_names,
            probe_frontend=prefill_frontend,
        )
        if probe_aware
        else make_merged_build_plan(
            model_file_names,
            merge_encoder_into_prefill=merge_encoder_into_prefill,
            prefill_frontend=prefill_frontend,
        )
    )
    for name, recipe, _ in build_plan:
        merged = recipe(source_folder, main, embed, model_file_names)
        if not retain_prefill_logits and name in {
            _model_file_name(model_file_names, "prefill_greedy"),
            _model_file_name(model_file_names, "prefill_penalty_greedy"),
            _model_file_name(model_file_names, "prefill_sampling"),
        }:
            retained_outputs = [
                value for value in merged.graph.output if value.name != "logits"
            ]
            del merged.graph.output[:]
            merged.graph.output.extend(retained_outputs)
            prune_unreachable_nodes(merged)
        redirect_shared_initializers_to_external(merged, external_by_name)
        output_path = out_folder / name
        save_model(merged, output_path)
        graphs[name] = output_path
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
        if tensor.data_location == TensorProto.EXTERNAL:
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
    """Remove superseded selection graphs and private external-data files."""
    old_family = "Be" + "am"
    obsolete_names = (
        f"Dolphin_First_{old_family}_Search.onnx",
        f"Dolphin_Second_{old_family}_Search.onnx",
        f"Dolphin_Prefill{old_family}First.onnx",
        f"Dolphin_Decode{old_family}Next.onnx",
        f"Dolphin_DecodePenalty{old_family}Next.onnx",
    )
    protected = {
        _model_file_name(model_file_names, "shared_initializers"),
        _model_file_name(model_file_names, "shared_initializers_data"),
    }
    return _delete_named_graph_artifacts(Path(folder), obsolete_names, protected)


def _copy_declared_external_data(source: Path, target_folder: Path) -> None:
    for location in _external_locations(source):
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Unsafe external-data location {location!r} in {source.name}."
            )
        source_data = source.parent / relative
        target_data = target_folder / relative
        target_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_data, target_data)


def copy_runtime_standalones(
    source_folder: Path,
    target_folder: Path,
    model_file_names: dict[str, str] | None = None,
    *,
    include_encoder: bool = True,
) -> list[Path]:
    """Copy only the standalone graphs Dolphin actually emits and still needs."""
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    roles = (
        RUNTIME_STANDALONE_MODEL_KEYS
        if include_encoder
        else tuple(
            key for key in RUNTIME_STANDALONE_MODEL_KEYS if key != "encoder"
        )
    )
    for key in roles:
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
        _copy_declared_external_data(source, target_folder)
        copied.append(target)
    return copied


def delete_merged_constituents(
    folder: Path,
    model_file_names: dict[str, str] | None = None,
    protected_names: tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    folder = Path(folder)
    protected = set(protected_names or (SHARED_MODEL_NAME, SHARED_DATA_NAME))
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
    node: onnx.NodeProto,
    remap: dict[str, str],
) -> onnx.NodeProto:
    copied = copy.deepcopy(node)
    for index, name in enumerate(copied.input):
        copied.input[index] = remap.get(name, name)
    return copied


def _copy_value_info_with_name(
    value_info: onnx.ValueInfoProto,
    name: str,
) -> onnx.ValueInfoProto:
    copied = copy.deepcopy(value_info)
    copied.name = name
    return copied


def _merge_opsets(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    opsets: dict[str, int] = {}
    for model in (dst, *sources):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    del dst.opset_import[:]
    for domain, version in sorted(opsets.items()):
        dst.opset_import.add(domain=domain, version=version)

    functions: dict[tuple[str, str, str], onnx.FunctionProto] = {}
    for model in (dst, *sources):
        for function in model.functions:
            key = (function.domain, function.name, function.overload)
            existing = functions.get(key)
            if existing is None:
                functions[key] = function
            elif existing.SerializeToString() != function.SerializeToString():
                raise RuntimeError(
                    "Local FunctionProto collision with different bodies: "
                    f"domain={function.domain!r}, name={function.name!r}, "
                    f"overload={function.overload!r}."
                )
    del dst.functions[:]
    dst.functions.extend(functions.values())


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
        expected = {
            name
            for name in donor_inputs
            if name.startswith(("en_key_layer_", "en_value_layer_"))
        }
        encoder_outputs = {
            output
            for node in target.graph.node
            for output in node.output
            if output.startswith("encoder_en_")
        }
        if encoder_outputs:
            mapped = {f"encoder_{name}" for name in expected}
            if mapped - encoder_outputs:
                raise RuntimeError(
                    "Dolphin probe-prefill template is missing Encoder cross-KV values: "
                    f"{sorted(mapped - encoder_outputs)}."
                )
            remap.update({name: f"encoder_{name}" for name in expected})
        else:
            public_inputs = {value.name for value in target.graph.input}
            if expected - public_inputs:
                raise RuntimeError(
                    "Dolphin cached-cross-KV prefill is missing public inputs: "
                    f"{sorted(expected - public_inputs)}."
                )
    return {name: value for name, value in remap.items() if name in donor_inputs}


def transplant_quantized_main(
    target: onnx.ModelProto,
    quantized_primary: onnx.ModelProto,
) -> onnx.ModelProto:
    """Replace target's source Main block with the optimized donor Main."""
    donor_main_nodes = [
        node for node in quantized_primary.graph.node if not _node_is_shell(node)
    ]
    remap = _target_position_remap(target, donor_main_nodes)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in donor_main_nodes
    ]
    if not primary_main_nodes:
        raise RuntimeError("Optimized primary graph contains no Dolphin Main node block.")

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
        initializer.name: initializer
        for initializer in target.graph.initializer
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

    donor_inputs_by_target = {
        remap.get(value.name, value.name): value
        for value in quantized_primary.graph.input
    }
    shell_boundary_names = {
        "embed_hidden_states",
        "prefill_position_embed",
        "decode_position_embed",
        "prefill_attention_mask",
        "decode_zero_attention_mask",
    }
    for index, value in enumerate(merged.graph.input):
        authoritative = donor_inputs_by_target.get(value.name)
        if authoritative is not None and value.name not in shell_boundary_names:
            merged.graph.input[index].CopyFrom(
                _copy_value_info_with_name(authoritative, value.name)
            )
    donor_outputs = {value.name: value for value in quantized_primary.graph.output}
    for index, value in enumerate(merged.graph.output):
        authoritative = donor_outputs.get(value.name)
        if authoritative is not None:
            merged.graph.output[index].CopyFrom(authoritative)

    existing = {value.name for value in merged.graph.input}
    existing.update(value.name for value in merged.graph.output)
    existing.update(initializer.name for initializer in merged.graph.initializer)
    value_infos: list[onnx.ValueInfoProto] = []

    def add_value_info(value_info: onnx.ValueInfoProto, name: str | None = None) -> None:
        value_name = name or value_info.name
        if value_name not in existing:
            value_infos.append(_copy_value_info_with_name(value_info, value_name))
            existing.add(value_name)

    for value_info in quantized_primary.graph.input:
        target_name = remap.get(value_info.name, value_info.name)
        if target_name not in shell_boundary_names:
            add_value_info(value_info, target_name)
    for value_info in quantized_primary.graph.value_info:
        add_value_info(value_info, remap.get(value_info.name, value_info.name))
    for value_info in target.graph.value_info:
        if value_info.name.startswith(SHELL_PREFIXES):
            add_value_info(value_info)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(value_infos)
    _merge_opsets(merged, quantized_primary)
    minimize_public_outputs(merged)
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
    target_encoder_nodes = [
        node for node in target.graph.node if _node_is_encoder_component(node)
    ]
    if not target_encoder_nodes:
        public_names = {
            value.name for value in (*target.graph.input, *target.graph.output)
        }
        if "audio" in public_names or any(name.startswith("encoder_") for name in public_names):
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
    target_initializers = {
        initializer.name: initializer for initializer in target.graph.initializer
    }
    donor_initializers = {
        initializer.name: initializer for initializer in donor.graph.initializer
    }
    old_encoder_initializers = _used_inputs(target_encoder_nodes) & set(target_initializers)
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
        raise RuntimeError("Dolphin prefill lost its Encoder insertion point.")

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
                    f"Optimized Dolphin Encoder initializer {name!r} collides."
                )
            continue
        new_initializers.append(copy.deepcopy(initializer))
        seen.add(name)
    del merged.graph.node[:]
    merged.graph.node.extend(new_nodes)
    del merged.graph.initializer[:]
    merged.graph.initializer.extend(new_initializers)

    donor_outputs = {value.name: value for value in donor.graph.output}
    for index, value in enumerate(merged.graph.output):
        authoritative = donor_outputs.get(value.name)
        if authoritative is not None:
            merged.graph.output[index].CopyFrom(authoritative)

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
    """Extract all large numeric donor initializers and redirect supplied models."""
    values = list(models.values()) if isinstance(models, dict) else list(models)
    if not values:
        raise RuntimeError("No merged model was supplied for shared-initializer extraction.")
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
        if not _is_shareable_initializer(initializer, min_shared_elements):
            raise RuntimeError(f"Additional shared initializer {name!r} is not shareable.")
        existing = shared.get(name)
        if existing is not None:
            if existing.data_location == TensorProto.EXTERNAL and not existing.raw_data:
                if (
                    existing.data_type != initializer.data_type
                    or tuple(existing.dims) != tuple(initializer.dims)
                ):
                    raise RuntimeError(
                        f"Materialized shell initializer ABI mismatch for {name!r}."
                    )
                # ``collect_target_only_shared_shell_initializers`` loaded and
                # byte-verified this tensor across every raw target before the
                # old shared blob was replaced. Discard its stale offset now.
                shared[name] = initializer
                continue
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
        raise RuntimeError("Optimized primary graph contains no shareable initializer.")
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
    """mmap and inject shared initializers; returned references must stay alive."""
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
            raise RuntimeError(f"Shared initializer {initializer.name!r} is not external.")
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(
                f"Unsafe shared initializer location {location!r} for {initializer.name!r}."
            )
        data_path = shared_model_path.parent / relative
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
            data_path,
            dtype=dtype,
            mode="r",
            offset=offset,
            shape=shape,
        )
        arrays[initializer.name] = array
        ort_value = ort.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(initializer.name, ort_value)
    return arrays, ort_values
