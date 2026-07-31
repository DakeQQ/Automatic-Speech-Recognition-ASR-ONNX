"""Build Whisper probe-aware Encoder-once graphs around one shared blob.

Graph A merges Encoder + SOT/full-prompt prefill, Graph B retains the existing
decoder-only cached-cross-KV full-prefix prefill, and Graph C remains decode.
NoSpeech stays standalone and consumes Graph A's raw pre-begin-suppression logits.
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

# Position/embed/head nodes are shells; unprefixed nodes are the decoder Main.
# Keep this list synchronized with every prefix introduced by the recipes below.
SHELL_PREFIXES = (
    "encoder_",
    "embed_",
    "prefill_",
    "decode_",
    "begin_",
    "argmax_",
    "greedy_",
    "penalty_",
    "sampling_",
)

PREFILL_GREEDY_MODEL_NAME = "Whisper_PrefillGreedy.onnx"
PREFILL_PENALTY_GREEDY_MODEL_NAME = "Whisper_PrefillPenaltyGreedy.onnx"
PREFILL_SAMPLING_MODEL_NAME = "Whisper_PrefillSampling.onnx"
PROBE_PREFILL_GREEDY_MODEL_NAME = "Whisper_ProbePrefillGreedy.onnx"
PROBE_PREFILL_PENALTY_GREEDY_MODEL_NAME = "Whisper_ProbePrefillPenaltyGreedy.onnx"
PROBE_PREFILL_SAMPLING_MODEL_NAME = "Whisper_ProbePrefillSampling.onnx"
DECODE_GREEDY_MODEL_NAME = "Whisper_DecodeGreedy.onnx"
DECODE_PENALTY_GREEDY_MODEL_NAME = "Whisper_DecodePenaltyGreedy.onnx"
DECODE_SAMPLING_MODEL_NAME = "Whisper_DecodeSampling.onnx"
SHARED_MODEL_NAME = "Whisper_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"

DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "Whisper_Encoder.onnx",
    "main": "Whisper_Decoder.onnx",
    "embed": "Whisper_Decoder_Embed.onnx",
    "position_prefill": "Whisper_Position_Mask_Prefill.onnx",
    "position_decode": "Whisper_Position_Mask_Decode.onnx",
    "begin_suppress": "Whisper_Begin_Suppress.onnx",
    "greedy": "Whisper_Greedy_Search.onnx",
    "argmax": "Whisper_Argmax.onnx",
    "sampling": "Whisper_TopKTopPSampling.onnx",
    "penalty": "Whisper_Apply_Penalty.onnx",
    "no_speech": "Whisper_No_Speech_Detection.onnx",
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

RUNTIME_STANDALONE_MODEL_KEYS = ("metadata", "no_speech")
REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS = frozenset(RUNTIME_STANDALONE_MODEL_KEYS)
MERGED_CONSTITUENT_MODEL_KEYS = (
    "main",
    "embed",
    "position_prefill",
    "position_decode",
    "begin_suppress",
    "greedy",
    "argmax",
    "sampling",
    "penalty",
)


def _model_file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = DEFAULT_MODEL_FILE_NAMES if model_file_names is None else {**DEFAULT_MODEL_FILE_NAMES, **model_file_names}
    return names[key]


def load_model(
    path: Path,
    load_external_data: bool = True,
) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=load_external_data)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a data-light merged graph without a per-graph external sidecar."""
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


def _tensor_raw_bytes(tensor: TensorProto) -> bytes:
    raw = tensor.raw_data
    return raw if raw else numpy_helper.to_array(tensor).tobytes()


def save_shared_initializers_from_tensors(shared: dict[str, TensorProto], path: Path) -> dict[str, int]:
    """Stream tensors into one file and alias byte-identical logical initializers.

    Tensor names remain distinct because ONNX graphs reference names, but equal dtype/shape/content
    tensors share one physical ``(offset, length)``. A digest match is byte-verified before aliasing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    refs: list[TensorProto] = []
    offset = 0
    logical_bytes = 0
    duplicate_count = 0
    unique_by_fingerprint: dict[tuple[int, tuple[int, ...], int, bytes], tuple[TensorProto, int, int]] = {}
    with open(data_path, "wb") as data_file:
        for name, tensor in sorted(shared.items()):
            raw = _tensor_raw_bytes(tensor)
            length = len(raw)
            logical_bytes += length
            fingerprint = (
                int(tensor.data_type),
                tuple(int(dim) for dim in tensor.dims),
                length,
                hashlib.sha256(raw).digest(),
            )
            existing = unique_by_fingerprint.get(fingerprint)
            if existing is None:
                tensor_offset = offset
                data_file.write(raw)
                unique_by_fingerprint[fingerprint] = (tensor, tensor_offset, length)
                offset += length
            else:
                source_tensor, tensor_offset, source_length = existing
                if source_length != length or _tensor_raw_bytes(source_tensor) != raw:
                    raise RuntimeError(f"Initializer digest collision while deduplicating {name!r}.")
                duplicate_count += 1

            ref = TensorProto()
            ref.name = name
            ref.data_type = tensor.data_type
            ref.dims.extend(tensor.dims)
            ref.data_location = TensorProto.EXTERNAL
            for key, value in (("location", data_name), ("offset", str(tensor_offset)), ("length", str(length))):
                entry = ref.external_data.add()
                entry.key = key
                entry.value = value
            refs.append(ref)

    graph = onnx.helper.make_graph([], "whisper_shared_initializers", [], [], initializer=refs)
    model = onnx.helper.make_model(
        graph,
        producer_name="Whisper/Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    model.metadata_props.add(key="whisper_shared_initializers", value="1")
    model.metadata_props.add(key="initializer_count", value=str(len(refs)))
    model.metadata_props.add(key="unique_data_count", value=str(len(unique_by_fingerprint)))
    model.metadata_props.add(key="deduplicated_initializer_count", value=str(duplicate_count))
    model.metadata_props.add(key="logical_data_bytes", value=str(logical_bytes))
    model.metadata_props.add(key="physical_data_bytes", value=str(offset))
    onnx.save_model(model, str(path))
    return {
        "initializer_count": len(refs),
        "unique_data_count": len(unique_by_fingerprint),
        "deduplicated_initializer_count": duplicate_count,
        "logical_data_bytes": logical_bytes,
        "physical_data_bytes": offset,
    }


def shared_external_data_map(shared_model_path: Path) -> dict[str, dict[str, str]]:
    model = onnx.load(str(shared_model_path), load_external_data=False)
    result = {}
    for initializer in model.graph.initializer:
        external = _external_data_map(initializer)
        external["__tensor_data_type"] = str(initializer.data_type)
        external["__tensor_dims"] = ",".join(str(dim) for dim in initializer.dims)
        result[initializer.name] = external
    return result


def make_external_initializer_ref(initializer: TensorProto, external_data: dict[str, str]) -> TensorProto:
    ref = TensorProto()
    ref.name = initializer.name
    ref.data_type = initializer.data_type
    ref.dims.extend(initializer.dims)
    ref.data_location = TensorProto.EXTERNAL
    for key in ("location", "offset", "length", "checksum", "basepath"):
        value = external_data.get(key)
        if value is not None:
            entry = ref.external_data.add()
            entry.key = key
            entry.value = value
    return ref


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


def _node_int_attribute(node: onnx.NodeProto, name: str, default: int) -> int:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(onnx.helper.get_attribute_value(attribute))
    return default


def dedup_tied_embed_into_lm_head(
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    compare_rows: int = 1024,
) -> dict[str, object]:
    """Make Embed gather the verified transposed LM-head initializer owned by Main.

    The legacy exporter materializes tied PyTorch storage twice with opposite layouts:
    Embed owns ``[vocab, hidden]`` while the final MatMul owns ``[hidden, vocab]``.
    ONNX cannot express cross-model storage aliasing at export time. After exact chunked byte
    verification, Gather(axis=1) from the MatMul table yields ``[hidden, batch, ids]`` and one
    small Transpose restores the public ``[batch, ids, hidden]`` output.

    Structural ambiguity raises before mutation. A structurally valid but untied checkpoint is
    left unchanged and reported as not applied.
    """
    embed_initializers = {initializer.name: initializer for initializer in embed.graph.initializer}
    gather_matches: list[tuple[int, onnx.NodeProto, TensorProto]] = []
    for index, node in enumerate(embed.graph.node):
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        initializer = embed_initializers.get(node.input[0])
        if initializer is None or len(initializer.dims) != 2:
            continue
        if _node_int_attribute(node, "axis", 0) == 0:
            gather_matches.append((index, node, initializer))
    if len(gather_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one axis-0 Embed Gather with a rank-2 initializer, found {len(gather_matches)}."
        )
    gather_index, gather, embed_table = gather_matches[0]

    value_info = {
        value.name: value
        for value in (*embed.graph.input, *embed.graph.output, *embed.graph.value_info)
    }
    ids_info = value_info.get(gather.input[1])
    output_info = value_info.get(gather.output[0])
    ids_rank = len(ids_info.type.tensor_type.shape.dim) if ids_info is not None else None
    output_rank = len(output_info.type.tensor_type.shape.dim) if output_info is not None else None
    if ids_rank != 2 or output_rank != 3:
        raise RuntimeError(
            f"Tied Embed rewrite requires rank-2 ids and rank-3 output, got {ids_rank} and {output_rank}."
        )
    if len(embed.graph.output) != 1 or embed.graph.output[0].name != gather.output[0]:
        raise RuntimeError("Tied Embed rewrite requires the Gather result to be the sole Embed output.")

    main_initializers = {initializer.name: initializer for initializer in main.graph.initializer}
    candidate_names = {
        node.input[1]
        for node in main.graph.node
        if node.op_type == "MatMul" and len(node.input) == 2 and node.input[1] in main_initializers
    }
    candidates = [
        main_initializers[name]
        for name in candidate_names
        if list(main_initializers[name].dims) == [int(embed_table.dims[1]), int(embed_table.dims[0])]
        and main_initializers[name].data_type == embed_table.data_type
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one transposed LM-head MatMul initializer, found {len(candidates)}."
        )
    lm_head = candidates[0]

    embed_array = numpy_helper.to_array(embed_table)
    head_array = numpy_helper.to_array(lm_head)
    rows = int(embed_table.dims[0])
    for start in range(0, rows, compare_rows):
        stop = min(start + compare_rows, rows)
        expected = np.ascontiguousarray(head_array[:, start:stop].T)
        if embed_array[start:stop].tobytes(order="C") != expected.tobytes(order="C"):
            return {
                "applied": False,
                "reason": "embedding and LM-head tensors are not exact transposes",
                "embed_initializer": embed_table.name,
                "lm_head_initializer": lm_head.name,
            }

    occupied = {value.name for value in (*embed.graph.input, *embed.graph.output, *embed.graph.value_info)}
    occupied.update(initializer.name for initializer in embed.graph.initializer)
    occupied.update(name for node in embed.graph.node for name in (*node.input, *node.output) if name)
    temporary = gather.output[0] + "__from_tied_lm_head"
    suffix = 0
    while temporary in occupied:
        suffix += 1
        temporary = gather.output[0] + f"__from_tied_lm_head_{suffix}"

    public_output = gather.output[0]
    gather.input[0] = lm_head.name
    gather.output[0] = temporary
    retained_attributes = [attribute for attribute in gather.attribute if attribute.name != "axis"]
    del gather.attribute[:]
    gather.attribute.extend(retained_attributes)
    gather.attribute.append(onnx.helper.make_attribute("axis", 1))
    transpose = onnx.helper.make_node(
        "Transpose",
        [temporary],
        [public_output],
        name="embed_TiedLMHeadTranspose",
        perm=[1, 2, 0],
    )
    nodes = list(embed.graph.node)
    del embed.graph.node[:]
    embed.graph.node.extend(nodes[:gather_index + 1] + [transpose] + nodes[gather_index + 1:])
    retained_initializers = [
        initializer for initializer in embed.graph.initializer if initializer.name != embed_table.name
    ]
    retained_initializers.append(copy.deepcopy(lm_head))
    del embed.graph.initializer[:]
    embed.graph.initializer.extend(retained_initializers)
    return {
        "applied": True,
        "embed_initializer": embed_table.name,
        "lm_head_initializer": lm_head.name,
        "bytes_eliminated": int(embed_array.nbytes),
        "inserted_node": transpose.name,
    }


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
    stable = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
    }
    result = prefixed(model, prefix)
    restore = {f"{prefix}{name}": name for name in stable}
    for initializer in result.graph.initializer:
        initializer.name = restore.get(initializer.name, initializer.name)
    for node in result.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = restore.get(name, name)
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


def hoist_constant_tensors_to_initializers(model: onnx.ModelProto) -> int:
    """Hoist exact top-level Constant tensors for streaming/shared extraction."""
    occupied = {initializer.name for initializer in model.graph.initializer}
    hoisted: list[TensorProto] = []
    retained_nodes: list[onnx.NodeProto] = []
    count = 0
    for node in model.graph.node:
        tensor_attributes = [
            attribute
            for attribute in node.attribute
            if attribute.name == "value" and attribute.HasField("t")
        ]
        if node.op_type != "Constant" or len(node.output) != 1 or len(tensor_attributes) != 1:
            retained_nodes.append(node)
            continue
        output = node.output[0]
        if not output or output in occupied:
            raise RuntimeError(f"Cannot hoist Whisper Constant output {output!r}.")
        tensor = copy.deepcopy(tensor_attributes[0].t)
        if tensor.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
            retained_nodes.append(node)
            continue
        tensor.name = output
        hoisted.append(tensor)
        occupied.add(output)
        count += 1
    if count:
        del model.graph.node[:]
        model.graph.node.extend(retained_nodes)
        model.graph.initializer.extend(hoisted)
    return count


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
    names.update(name for node in model.graph.node for name in (*node.input, *node.output) if name)
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
    return len(remap)


def rename_tensor(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    names = {value.name for value in model.graph.input}
    names.update(value.name for value in model.graph.output)
    names.update(value.name for value in model.graph.value_info)
    names.update(initializer.name for initializer in model.graph.initializer)
    names.update(name for node in model.graph.node for name in (*node.input, *node.output) if name)
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


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


def _cross_kv_names(num_layers: int, prefix: str = "") -> list[str]:
    return [
        *(f"{prefix}en_key_layer_{index}" for index in range(num_layers)),
        *(f"{prefix}en_value_layer_{index}" for index in range(num_layers)),
    ]


def _main_cross_values(main: onnx.ModelProto) -> tuple[int, list[onnx.ValueInfoProto]]:
    values = [
        value
        for value in main.graph.input
        if value.name.startswith(("en_key_layer_", "en_value_layer_"))
    ]
    num_layers = len(values) // 2
    return num_layers, values


def build_probe_prefill_frontend(
    source_folder: Path,
    encoder: onnx.ModelProto | None = None,
    model_file_names: dict[str, str] | None = None,
    *,
    reference_cross_values: list[onnx.ValueInfoProto] | None = None,
) -> tuple[onnx.ModelProto, list[str]]:
    if encoder is None:
        encoder = load_model(source_folder / _model_file_name(model_file_names, "encoder"))
    num_layers = len(encoder.graph.output) // 2
    names = _cross_kv_names(num_layers)
    hoist_constant_tensors_to_initializers(encoder)
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
        if name not in target_names:
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
    """Remove decode-only raw-logit outputs after selection-head composition."""
    is_decode = any(
        value.name == "decode_kv_seq_len" for value in model.graph.input
    )
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


def copy_metadata(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    existing = {prop.key: prop for prop in dst.metadata_props}
    for source in sources:
        for prop in source.metadata_props:
            if prop.key in existing:
                existing[prop.key].value = prop.value
            else:
                existing[prop.key] = dst.metadata_props.add(key=prop.key, value=prop.value)


def merge_models_no_check(
    first: onnx.ModelProto,
    second: onnx.ModelProto,
    io_map: list[tuple[str, str]],
) -> onnx.ModelProto:
    """Compose two models without invoking the >2 GiB ONNX checker path."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "Whisper/Shared_Merged.py"
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
                raise RuntimeError(f"Whisper FunctionProto collision: {key}.")
    merged.functions.extend(functions.values())

    seen_inputs: set[str] = set()
    values = list(first.graph.input) + [value for value in second.graph.input if value.name not in mapped_targets]
    for value in values:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializer_by_name: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
        existing = initializer_by_name.get(initializer.name)
        if existing is None:
            initializer_by_name[initializer.name] = initializer
        elif existing is not initializer and existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(f"Initializer name collision with different data: {initializer.name}")
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
    outputs = [value for value in first.graph.output if value.name not in mapped_sources] + list(second.graph.output)
    for value in outputs:
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
    probe_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
) -> tuple[onnx.ModelProto, str, onnx.ModelProto, list[str]]:
    num_layers, main_cross_values = _main_cross_values(main)
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
        if probe_frontend is None:
            return merged, "prefill_kv_seq_len", position, []
        frontend, cross_outputs = probe_frontend
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
        numpy_helper.from_array(np.zeros((1, 1, 1), dtype=mask_dtype), name="decode_zero_attention_mask")
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


def _merge_begin_suppress(
    source_folder: Path,
    merged: onnx.ModelProto,
    model_file_names: dict[str, str] | None,
) -> tuple[onnx.ModelProto, str]:
    begin = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "begin_suppress")),
        "begin_",
    )
    merged = merge_models_no_check(merged, begin, io_map=[("logits", "begin_logits_in")])
    return merged, "begin_logits_out"


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
    merged.producer_name = "Whisper/Shared_Merged.py"
    return merged


def merge_prefill_greedy(
    source_folder, main, embed, model_file_names=None, probe_frontend=None
):
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "prefill", model_file_names, probe_frontend
    )
    merged, head_logits = _merge_begin_suppress(source_folder, merged, model_file_names)
    argmax = prefixed(load_model(source_folder / _model_file_name(model_file_names, "argmax")), "argmax_")
    merged = merge_models_no_check(merged, argmax, io_map=[(head_logits, "argmax_logits")])
    outputs = _main_kv_output_names(main) + cross_outputs + ["argmax_max_logits_idx", "logits", kv_seq_len]
    return _finalize(merged, main, position, outputs)


def merge_decode_greedy(source_folder, main, embed, model_file_names=None, probe_frontend=None):
    if probe_frontend is not None:
        raise RuntimeError("Whisper decode must not receive an Encoder frontend.")
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "decode", model_file_names
    )
    if cross_outputs:
        raise RuntimeError("Whisper decode unexpectedly acquired Encoder outputs.")
    argmax = prefixed(load_model(source_folder / _model_file_name(model_file_names, "argmax")), "argmax_")
    merged = merge_models_no_check(merged, argmax, io_map=[("logits", "argmax_logits")])
    outputs = _main_kv_output_names(main) + ["argmax_max_logits_idx", kv_seq_len]
    return _finalize(merged, main, position, outputs)


def merge_prefill_penalty_greedy(source_folder, main, embed, model_file_names=None, probe_frontend=None):
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "prefill", model_file_names, probe_frontend
    )
    merged, head_logits = _merge_begin_suppress(source_folder, merged, model_file_names)
    greedy = prefixed(load_model(source_folder / _model_file_name(model_file_names, "greedy")), "greedy_")
    merged = merge_models_no_check(merged, greedy, io_map=[(head_logits, "greedy_logits")])
    outputs = _main_kv_output_names(main) + cross_outputs + [
        "greedy_max_logits_idx", "greedy_save_id_out", "logits", kv_seq_len
    ]
    return _finalize(merged, main, position, outputs)


def merge_decode_penalty_greedy(source_folder, main, embed, model_file_names=None, probe_frontend=None):
    if probe_frontend is not None:
        raise RuntimeError("Whisper decode must not receive an Encoder frontend.")
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, "decode", model_file_names
    )
    if cross_outputs:
        raise RuntimeError("Whisper decode unexpectedly acquired Encoder outputs.")
    penalty = prefixed(load_model(source_folder / _model_file_name(model_file_names, "penalty")), "penalty_")
    greedy = prefixed(load_model(source_folder / _model_file_name(model_file_names, "greedy")), "greedy_")
    merged = merge_models_no_check(merged, penalty, io_map=[("logits", "penalty_logits_in")])
    merged = merge_models_no_check(
        merged, greedy, io_map=[("penalty_logits_out", "greedy_logits")]
    )
    outputs = _main_kv_output_names(main) + [
        "greedy_max_logits_idx", "greedy_save_id_out", kv_seq_len
    ]
    return _finalize(merged, main, position, outputs)


def _merge_sampling(source_folder, main, embed, kind, model_file_names=None, probe_frontend=None):
    merged, kv_seq_len, position, cross_outputs = _merge_position_embed_into_main(
        source_folder, main, embed, kind, model_file_names, probe_frontend
    )
    head_logits = "logits"
    if kind == "prefill":
        # Keep Main's logits public for language/no-speech detection, but route
        # first-token selection through Whisper's begin-only suppression shell.
        merged, head_logits = _merge_begin_suppress(
            source_folder, merged, model_file_names
        )
    sampling = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "sampling")),
        "sampling_",
    )
    merged = merge_models_no_check(
        merged,
        sampling,
        io_map=[(head_logits, "sampling_logits")],
    )
    outputs = _main_kv_output_names(main) + cross_outputs + [
        "sampling_sampled_id",
        "sampling_save_id_out",
    ]
    if kind == "prefill":
        outputs.append("logits")
    outputs.append(kv_seq_len)
    return _finalize(merged, main, position, outputs)


def merge_prefill_sampling(source_folder, main, embed, model_file_names=None, probe_frontend=None):
    return _merge_sampling(
        source_folder, main, embed, "prefill", model_file_names, probe_frontend
    )


def merge_decode_sampling(source_folder, main, embed, model_file_names=None, probe_frontend=None):
    if probe_frontend is not None:
        raise RuntimeError("Whisper decode must not receive an Encoder frontend.")
    return _merge_sampling(
        source_folder, main, embed, "decode", model_file_names
    )


def _recipe_with_probe(recipe, probe_frontend):
    def wrapped(source_folder, main, embed, model_file_names=None):
        return recipe(
            source_folder,
            main,
            embed,
            model_file_names,
            probe_frontend=probe_frontend,
        )
    wrapped.__name__ = recipe.__name__
    return wrapped


def make_merged_build_plan(
    model_file_names: dict[str, str] | None = None,
    *,
    probe_frontend: tuple[onnx.ModelProto, list[str]] | None = None,
):
    position_prefill = _model_file_name(model_file_names, "position_prefill")
    position_decode = _model_file_name(model_file_names, "position_decode")
    begin = _model_file_name(model_file_names, "begin_suppress")
    argmax = _model_file_name(model_file_names, "argmax")
    greedy = _model_file_name(model_file_names, "greedy")
    sampling = _model_file_name(model_file_names, "sampling")
    penalty = _model_file_name(model_file_names, "penalty")
    return [
        (
            _model_file_name(model_file_names, "probe_prefill_greedy"),
            _recipe_with_probe(merge_prefill_greedy, probe_frontend),
            [_model_file_name(model_file_names, "encoder"), position_prefill, begin, argmax],
        ),
        (
            _model_file_name(model_file_names, "probe_prefill_penalty_greedy"),
            _recipe_with_probe(merge_prefill_penalty_greedy, probe_frontend),
            [_model_file_name(model_file_names, "encoder"), position_prefill, begin, greedy],
        ),
        (
            _model_file_name(model_file_names, "probe_prefill_sampling"),
            _recipe_with_probe(merge_prefill_sampling, probe_frontend),
            [_model_file_name(model_file_names, "encoder"), position_prefill, begin, sampling],
        ),
        (
            _model_file_name(model_file_names, "prefill_greedy"),
            merge_prefill_greedy,
            [position_prefill, begin, argmax],
        ),
        (
            _model_file_name(model_file_names, "prefill_penalty_greedy"),
            merge_prefill_penalty_greedy,
            [position_prefill, begin, greedy],
        ),
        (
            _model_file_name(model_file_names, "prefill_sampling"),
            merge_prefill_sampling,
            [position_prefill, begin, sampling],
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


MERGED_BUILD_PLAN = make_merged_build_plan()


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
) -> dict:
    """Build exactly three Whisper strategy pairs around one shared weight blob."""
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    delete_obsolete_strategy_artifacts(source_folder, model_file_names)
    if out_folder.resolve() != source_folder.resolve():
        delete_obsolete_strategy_artifacts(out_folder, model_file_names)

    build_plan = make_merged_build_plan(model_file_names)

    main_name = _model_file_name(model_file_names, "main")
    encoder_name = _model_file_name(model_file_names, "encoder")
    embed_name = _model_file_name(model_file_names, "embed")
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    shared_model_path = out_folder / shared_name
    shared_data_path = out_folder / shared_data_name
    for graph_name, _, _ in build_plan:
        (out_folder / graph_name).unlink(missing_ok=True)
        (out_folder / (graph_name + ".data")).unlink(missing_ok=True)
    for legacy in (
        "shared_initializers.npz",
        "shared_initializers.manifest.json",
        "shared_initializers.onnx",
        "shared_initializers.onnx.data",
    ):
        (out_folder / legacy).unlink(missing_ok=True)

    main, shared = load_main_with_shared_initializers(
        source_folder, min_shared_elements, model_file_names
    )
    _, main_cross_values = _main_cross_values(main)
    encoder = load_model(source_folder / encoder_name)
    hoist_constant_tensors_to_initializers(encoder)
    namespace_encoder_initializers(encoder)
    for initializer in encoder.graph.initializer:
        if not _is_shareable_initializer(initializer, min_shared_elements):
            continue
        existing = shared.get(initializer.name)
        if existing is not None and existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(f"Whisper Encoder initializer collision: {initializer.name!r}.")
        shared[initializer.name] = initializer
    embed = prefixed(load_model(source_folder / embed_name), "embed_")
    embed_dedup = dedup_tied_embed_into_lm_head(main, embed)
    for initializer in embed.graph.initializer:
        if _is_shareable_initializer(initializer, min_shared_elements):
            existing = shared.get(initializer.name)
            if existing is not None:
                if (
                    existing.data_type != initializer.data_type
                    or list(existing.dims) != list(initializer.dims)
                    or _tensor_raw_bytes(existing) != _tensor_raw_bytes(initializer)
                ):
                    raise RuntimeError(
                        f"Shared initializer collision with different data: {initializer.name}"
                    )
            else:
                shared[initializer.name] = initializer

    # These immutable shell buffers recur in several strategy graphs. The prefixes exactly
    # match the composition recipes, so redirect_shared_initializers_to_external can replace
    # every copy after merge. The writer also aliases equal prefill/decode position tables.
    for role, prefix in (
        ("position_prefill", "prefill_"),
        ("position_decode", "decode_"),
        ("begin_suppress", "begin_"),
    ):
        shell_path = source_folder / _model_file_name(model_file_names, role)
        if not shell_path.exists():
            continue
        shell = prefixed(load_model(shell_path), prefix)
        for initializer in shell.graph.initializer:
            if not _is_shareable_initializer(initializer, min_shared_elements):
                continue
            existing = shared.get(initializer.name)
            if existing is not None:
                if (
                    existing.data_type != initializer.data_type
                    or list(existing.dims) != list(initializer.dims)
                    or _tensor_raw_bytes(existing) != _tensor_raw_bytes(initializer)
                ):
                    raise RuntimeError(
                        f"Shared shell initializer collision with different data: {initializer.name}"
                    )
            else:
                shared[initializer.name] = initializer
        del shell

    shared_stats = save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    redirect_shared_initializers_to_external(main, external_by_name)
    redirect_shared_initializers_to_external(encoder, external_by_name)
    redirect_shared_initializers_to_external(embed, external_by_name)
    standalone_main = out_folder / main_name
    standalone_encoder = out_folder / encoder_name
    save_model(main, standalone_main)
    save_model(encoder, standalone_encoder)
    probe_frontend = build_probe_prefill_frontend(
        source_folder,
        encoder=encoder,
        model_file_names=model_file_names,
        reference_cross_values=main_cross_values,
    )

    graphs: dict[str, Path] = {}
    build_plan = make_merged_build_plan(
        model_file_names,
        probe_frontend=probe_frontend,
    )
    for name, recipe, _ in build_plan:
        merged = recipe(source_folder, main, embed, model_file_names)
        redirect_shared_initializers_to_external(merged, external_by_name)
        output_path = out_folder / name
        save_model(merged, output_path)
        graphs[name] = output_path
        del merged

    result = {
        "graphs": graphs,
        "shared_model": shared_model_path,
        "shared_data": shared_data_path,
        "shared_stats": shared_stats,
        "embed_dedup": embed_dedup,
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


def copy_runtime_standalones(
    source_folder: Path,
    target_folder: Path,
    model_file_names: dict[str, str] | None = None,
) -> list[Path]:
    """Copy only the standalone graphs Whisper actually emits and still needs."""
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
                raise FileNotFoundError(f"Required Whisper standalone graph was not exported: {source}")
            continue
        # torch.onnx.export can externalize a >2GB standalone graph (e.g. the Whisper encoder)
        # as hundreds of loose per-tensor files -- one per initializer plus one per externalized
        # Constant attribute. Ship the deployed graph self-contained instead: keep an already
        # clean layout as-is, otherwise stream every scattered file into one <model>.onnx.data
        # sidecar so the standalone graph matches the single-blob layout of the merged graphs.
        locations = _external_locations(source)
        sidecar_name = name + ".data"
        if not locations:
            shutil.copy2(source, target)                                       # weights embedded in the proto
        elif locations == {sidecar_name}:
            shutil.copy2(source, target)                                       # already one clean sidecar
            shutil.copy2(source_folder / sidecar_name, target_folder / sidecar_name)
        else:
            _consolidate_external_data(source, source_folder, target, target_folder)
        copied.append(target)
    return copied


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
    """Remove superseded selection graphs and their private external data."""
    old_family = "Be" + "am"
    obsolete_names = (
        f"Whisper_First_{old_family}_Search.onnx",
        f"Whisper_Second_{old_family}_Search.onnx",
        f"Whisper_Prefill{old_family}First.onnx",
        f"Whisper_Decode{old_family}Next.onnx",
        f"Whisper_DecodePenalty{old_family}Next.onnx",
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
    """Stream a graph's scattered per-tensor external files into one ``<model>.onnx.data``.

    torch.onnx.export writes every large initializer -- and every externalized Constant
    attribute tensor -- to its own file named after the tensor, so a single >2GB standalone
    graph ships as hundreds of loose weight files. This reloads only the graph structure
    (``load_external_data=False``), copies each external tensor's bytes sequentially into one
    sidecar, and rewrites its ``(location, offset, length)`` to point into it. Peak memory
    stays at a single tensor's bytes; the multi-GB weights never all live in memory at once.
    Offsets are packed back-to-back (no alignment), matching
    ``save_shared_initializers_from_tensors`` and the ``np.memmap`` reader in
    ``attach_shared_initializers``.

    The caller must not consolidate in place: ``target_folder`` must differ from
    ``source_folder`` (or use a distinct name) so no source loose file is overwritten mid-stream.
    """
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
                raw = handle.read(int(declared_length)) if declared_length else handle.read()
            data_file.write(raw)
            written = len(raw)
            del tensor.external_data[:]
            for entry_key, entry_value in (
                ("location", sidecar_name),
                ("offset", str(offset)),
                ("length", str(written)),
            ):
                entry = tensor.external_data.add()
                entry.key = entry_key
                entry.value = entry_value
            offset += written
    onnx.save(model, str(target_onnx))
    return data_path


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
# Quantized-Main transplantation used by Optimize_ONNX.py.
# ---------------------------------------------------------------------------

def _node_is_shell(node: onnx.NodeProto) -> bool:
    return any(output.startswith(SHELL_PREFIXES) for output in node.output)


def _used_inputs(nodes) -> set[str]:
    return {name for node in nodes for name in node.input if name}


def _copy_node_with_input_remap(node: onnx.NodeProto, remap: dict[str, str]) -> onnx.NodeProto:
    copied = copy.deepcopy(node)
    for index, name in enumerate(copied.input):
        copied.input[index] = remap.get(name, name)
    return copied


def _copy_value_info_with_name(value_info: onnx.ValueInfoProto, name: str) -> onnx.ValueInfoProto:
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
                raise RuntimeError(f"Whisper FunctionProto collision: {key}.")
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
        "position_embed": "decode_position_embed" if decode else "prefill_position_embed",
        "attention_mask": "decode_zero_attention_mask" if decode else "prefill_attention_mask",
    }
    if not decode:
        expected = {
            name for name in donor_inputs if name.startswith(("en_key_layer_", "en_value_layer_"))
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
                    f"Whisper probe template misses Encoder values: {sorted(mapped - encoder_outputs)}."
                )
            remap.update({name: f"encoder_{name}" for name in expected})
    return {name: value for name, value in remap.items() if name in donor_inputs}


def transplant_quantized_main(
    target: onnx.ModelProto,
    quantized_primary: onnx.ModelProto,
) -> onnx.ModelProto:
    """Replace a target graph's unquantized decoder Main with a quantized donor Main."""
    donor_main_nodes = [
        node for node in quantized_primary.graph.node if not _node_is_shell(node)
    ]
    remap = _target_position_remap(target, donor_main_nodes)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in donor_main_nodes
    ]
    if not primary_main_nodes:
        raise RuntimeError("Quantized primary graph contains no decoder Main node block.")

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

    primary_initializers = {initializer.name: initializer for initializer in quantized_primary.graph.initializer}
    target_initializers = {initializer.name: initializer for initializer in target.graph.initializer}
    used = _used_inputs(new_nodes)
    produced = {
        output for node in new_nodes for output in node.output if output
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
        value_name = name or value_info.name
        if value_name not in existing:
            value_infos.append(_copy_value_info_with_name(value_info, value_name))
            existing.add(value_name)

    for value_info in quantized_primary.graph.value_info:
        add_value_info(value_info, remap.get(value_info.name, value_info.name))
    for value_info in target.graph.value_info:
        if value_info.name.startswith(SHELL_PREFIXES):
            add_value_info(value_info)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(value_infos)
    _merge_opsets(merged, quantized_primary)
    minimize_public_outputs(merged)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    return merged


def _node_is_encoder_component(node: onnx.NodeProto) -> bool:
    return any(
        output.removeprefix(_PRECISION_FREE_CAST_PREFIX).startswith("encoder_")
        for output in node.output
        if output
    )


def restore_precision_free_graph_outputs(model: onnx.ModelProto) -> dict[str, str]:
    inputs = {value.name for value in model.graph.input}
    initializers = {initializer.name for initializer in model.graph.initializer}
    producers = {output for node in model.graph.node for output in node.output if output}
    remap = {}
    for value in model.graph.output:
        if value.name in inputs or value.name in initializers or value.name in producers:
            continue
        alias = _PRECISION_FREE_CAST_PREFIX + value.name
        if alias not in producers:
            raise RuntimeError(f"Cannot restore Whisper output {value.name!r}.")
        remap[alias] = value.name
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = remap.get(name, name)
    retained = [value for value in model.graph.value_info if value.name not in remap]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    return remap


def transplant_optimized_encoder(target: onnx.ModelProto, optimized_encoder: onnx.ModelProto) -> onnx.ModelProto:
    target_nodes = [node for node in target.graph.node if _node_is_encoder_component(node)]
    if not target_nodes:
        if any(value.name == "audio" for value in target.graph.input):
            raise RuntimeError("Whisper probe graph has no classifiable Encoder nodes.")
        return target
    donor_source = copy.deepcopy(optimized_encoder)
    hoist_constant_tensors_to_initializers(donor_source)
    namespace_encoder_initializers(donor_source)
    donor = prefixed_preserving_initializers(donor_source, "encoder_")
    donor_nodes = [
        _copy_node_with_input_remap(node, {"encoder_audio": "audio"})
        for node in donor.graph.node
    ]
    target_initializers = {initializer.name: initializer for initializer in target.graph.initializer}
    donor_initializers = {initializer.name: initializer for initializer in donor.graph.initializer}
    old_names = _used_inputs(target_nodes) & set(target_initializers)
    donor_names = _used_inputs(donor_nodes) & set(donor_initializers)
    merged = copy.deepcopy(target)
    nodes = []
    inserted = False
    for node in target.graph.node:
        if _node_is_encoder_component(node):
            if not inserted:
                nodes.extend(copy.deepcopy(donor_nodes))
                inserted = True
            continue
        nodes.append(copy.deepcopy(node))
    initializers = [copy.deepcopy(i) for i in target.graph.initializer if i.name not in old_names]
    seen = {i.name for i in initializers}
    for name in sorted(donor_names):
        if name not in seen:
            initializers.append(copy.deepcopy(donor_initializers[name]))
            seen.add(name)
    del merged.graph.node[:]
    merged.graph.node.extend(nodes)
    del merged.graph.initializer[:]
    merged.graph.initializer.extend(initializers)
    retained = [value for value in target.graph.value_info if not value.name.startswith("encoder_")]
    del merged.graph.value_info[:]
    merged.graph.value_info.extend(retained)
    _merge_opsets(merged, optimized_encoder)
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
    """Extract all large numeric donor initializers and redirect the supplied models."""
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
            raise RuntimeError(
                f"Additional shared initializer {name!r} is not shareable."
            )
        existing = shared.get(name)
        if existing is not None:
            if existing.data_location == TensorProto.EXTERNAL and not existing.raw_data:
                if (
                    existing.data_type != initializer.data_type
                    or tuple(existing.dims) != tuple(initializer.dims)
                ):
                    raise RuntimeError(
                        f"Materialized Whisper shell ABI mismatch for {name!r}."
                    )
                shared[name] = initializer
                continue
            if (
                existing.data_type != initializer.data_type
                or tuple(existing.dims) != tuple(initializer.dims)
                or _tensor_raw_bytes(existing) != _tensor_raw_bytes(initializer)
            ):
                raise RuntimeError(
                    f"Additional shared initializer collision for {name!r}."
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
    """mmap and inject the shared initializers; returned references must stay alive."""
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
        data_path = shared_model_path.parent / location
        offset = int(external.get("offset", "0"))
        length = int(external.get("length", "0"))
        dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        shape = tuple(int(dim) for dim in initializer.dims)
        expected = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        if length and length != expected:
            raise RuntimeError(
                f"Shared initializer {initializer.name!r} length mismatch: {length} != {expected}."
            )
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        arrays[initializer.name] = array
        ort_value = ort.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(initializer.name, ort_value)
    return arrays, ort_values
