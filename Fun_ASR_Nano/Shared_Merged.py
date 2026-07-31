"""Build FunASR-Nano Encoder+prefill graphs backed by one initializer bundle.

The target keeps optional CTC, Metadata, and a standalone prompt Embed graph.
Every token-generation strategy uses one Encoder+prefill graph and one decode graph.
Prefill folds Encoder + rotary + Main + selection head; decode folds token Embed +
rotary + Main + selection head. Encoder and Main are retained only as data-light
optimizer donors and are never runtime compute sessions.

Large Encoder, Main, and Embed initializers are streamed once to
``FunASR_Nano_SharedInitializers.onnx.data``. Merged graphs and the standalone Embed
carry references to that blob; the runtime mmaps it and replaces those initializers with
``SessionOptions.add_initializer()`` before ORT reads private copies. The six merged
graphs cover greedy, penalty-greedy, and Top-K/Top-P sampling.
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
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

# Shell outputs are always prefixed; Main outputs and tensor names remain canonical.
SHELL_PREFIXES = (
    "encoder_",
    "prefill_",
    "decode_",
    "decode_embed_",
    "greedy_",
    "penalty_greedy_",
    "penalty_",
    "sampling_",
)

DEFAULT_MODEL_FILE_NAMES = {
    "metadata":               "ASR_Metadata.onnx",
    "encoder":                "FunASR_Nano_Encoder.onnx",
    "ctc_decoder":            "FunASR_Nano_CTC_Decoder.onnx",
    "embed":                  "FunASR_Nano_Decoder_Embed.onnx",
    "main":                   "FunASR_Nano_Decoder_Main.onnx",
    "rotary_prefill":         "FunASR_Nano_Rotary_Mask_Text_Prefill.onnx",
    "rotary_decode":          "FunASR_Nano_Rotary_Mask_Text_Decode.onnx",
    # Logical greedy is the stateless Argmax graph. Penalty-greedy is the
    # history-appending Greedy_Search graph used with Apply_Penalty.
    "greedy":                 "FunASR_Nano_Argmax.onnx",
    "penalty_greedy":         "FunASR_Nano_Greedy_Search.onnx",
    "penalty":                "FunASR_Nano_Apply_Penalty.onnx",
    "sampling":               "FunASR_Nano_TopKTopPSampling.onnx",
    "prefill_greedy":         "FunASR_Nano_TextPrefillGreedy.onnx",
    "prefill_penalty_greedy": "FunASR_Nano_TextPrefillPenaltyGreedy.onnx",
    "prefill_sampling":       "FunASR_Nano_TextPrefillSampling.onnx",
    "decode_greedy":          "FunASR_Nano_DecodeGreedy.onnx",
    "decode_penalty_greedy":  "FunASR_Nano_DecodePenaltyGreedy.onnx",
    "decode_sampling":        "FunASR_Nano_DecodeSampling.onnx",
    "shared_initializers":    "FunASR_Nano_SharedInitializers.onnx",
}
DEFAULT_MODEL_FILE_NAMES["shared_initializers_data"] = (
    DEFAULT_MODEL_FILE_NAMES["shared_initializers"] + ".data"
)

MERGED_ROLE_KEYS = (
    "prefill_greedy",
    "prefill_penalty_greedy",
    "prefill_sampling",
    "decode_greedy",
    "decode_penalty_greedy",
    "decode_sampling",
)
MERGED_CONSTITUENT_ROLE_KEYS = (
    "main",
    "rotary_prefill",
    "rotary_decode",
    "greedy",
    "penalty_greedy",
    "penalty",
    "sampling",
)
RUNTIME_STANDALONE_ROLE_KEYS = ("metadata", "ctc_decoder")


def model_file_names(overrides: dict[str, str] | None = None) -> dict[str, str]:
    names = dict(DEFAULT_MODEL_FILE_NAMES)
    if overrides:
        names.update(overrides)
    names["shared_initializers_data"] = names["shared_initializers"] + ".data"
    return names


def load_model(path: Path, load_external_data: bool = True) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=load_external_data)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a light graph containing only small constants and shared placeholders."""
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


def _tensor_bytes(tensor: TensorProto) -> bytes:
    if tensor.raw_data:
        return tensor.raw_data
    return numpy_helper.to_array(tensor).tobytes()


def _same_tensor(left: TensorProto, right: TensorProto) -> bool:
    return (
        left.data_type == right.data_type
        and tuple(left.dims) == tuple(right.dims)
        and _tensor_bytes(left) == _tensor_bytes(right)
    )


def save_shared_initializers_from_tensors(shared: dict[str, TensorProto], path: Path) -> None:
    """Stream initializer bytes directly into one mmap-friendly external-data file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    refs: list[TensorProto] = []
    regions_by_digest: dict[tuple[int, bytes], tuple[int, int]] = {}
    offset = 0
    with open(data_path, "wb") as data_file:
        for name, tensor in sorted(shared.items()):
            raw = _tensor_bytes(tensor)
            length = len(raw)
            digest_key = (length, hashlib.sha256(raw).digest())
            region = regions_by_digest.get(digest_key)
            if region is None:
                region = (offset, length)
                regions_by_digest[digest_key] = region
                data_file.write(raw)
                offset += length
            ref = TensorProto()
            ref.name = name
            ref.data_type = tensor.data_type
            ref.dims.extend(tensor.dims)
            ref.data_location = TensorProto.EXTERNAL
            for key, value in (
                ("location", data_name),
                ("offset", str(region[0])),
                ("length", str(region[1])),
            ):
                entry = ref.external_data.add()
                entry.key = key
                entry.value = value
            refs.append(ref)

    graph = onnx.helper.make_graph([], "funasr_nano_shared_initializers", [], [], initializer=refs)
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    onnx.save_model(model, str(path))


def shared_external_data_map(shared_model_path: Path) -> dict[str, dict[str, str]]:
    model = load_model(shared_model_path, load_external_data=False)
    result = {}
    for initializer in model.graph.initializer:
        external = _external_data_map(initializer)
        external["__tensor_data_type"] = str(initializer.data_type)
        external["__tensor_dims"] = ",".join(str(dim) for dim in initializer.dims)
        result[initializer.name] = external
    return result


def make_external_initializer_ref(
    initializer: TensorProto,
    external_data: dict[str, str],
) -> TensorProto:
    """Create a valid external initializer overridden by ORT ``add_initializer``."""
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
    """Replace shared tensors with lightweight external references for runtime injection."""
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


def restore_precision_free_graph_outputs(
    model: onnx.ModelProto,
    *,
    alias_prefix: str = "InsertedPrecisionFreeCast_",
) -> dict[str, str]:
    """Restore public names orphaned when a precision-boundary Cast is removed.

    ORT's float16 converter names the private producer feeding an output Cast as
    ``InsertedPrecisionFreeCast_<public-name>``. A later simplifier can erase the
    no-op Cast without transferring the public name to that producer, leaving a
    syntactically declared but unproduced graph output. Rename only exact aliases
    for missing outputs, update all internal consumers, and fail closed for every
    other disconnected output.
    """
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
    if not missing:
        return {}

    remap: dict[str, str] = {}
    for public_name in missing:
        alias = f"{alias_prefix}{public_name}"
        owners = producers.get(alias, [])
        if len(owners) != 1:
            raise RuntimeError(
                f"Cannot restore public output {public_name!r}: expected one producer "
                f"for exact precision-free alias {alias!r}, found {len(owners)}."
            )
        remap[alias] = public_name

    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            node.input[input_index] = remap.get(name, name)
        for output_index, name in enumerate(node.output):
            node.output[output_index] = remap.get(name, name)

    # The public graph output already carries authoritative type/shape metadata.
    # Drop now-redundant private alias entries instead of creating duplicate names.
    kept_value_info = [
        value for value in model.graph.value_info if value.name not in remap
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)

    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = remap.get(annotation.tensor_name, annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)

    return remap


def namespace_conflicting_initializers(
    primary_model: onnx.ModelProto,
    embed_models: list[onnx.ModelProto],
    *,
    prefix: str = "quantized_embed_",
) -> dict[str, str]:
    """Give independently quantized Embed tensors a collision-free namespace.

    Weight-only quantizers derive auxiliary names from the source initializer. A
    tied lm-head/Embed table therefore produces identically named packed weights,
    scales, and zero-points even though MatMulNBits and GatherBlockQuantized use
    different layouts. Those tensors must not share an ONNX identifier after the
    two node blocks are composed.

    ``embed_models`` may contain structure-only and materialized views of the same
    graph. Every view is rewritten with the exact same mapping.
    """
    if not embed_models:
        return {}

    primary_names = {initializer.name for initializer in primary_model.graph.initializer}
    embed_name_sets = [
        {initializer.name for initializer in model.graph.initializer}
        for model in embed_models
    ]
    collisions = sorted(primary_names & set().union(*embed_name_sets))
    if not collisions:
        return {}

    reserved = primary_names | set().union(*embed_name_sets)
    remap: dict[str, str] = {}
    for name in collisions:
        base = f"{prefix}{name}"
        replacement = base
        suffix = 1
        while replacement in reserved:
            replacement = f"{base}_{suffix}"
            suffix += 1
        remap[name] = replacement
        reserved.add(replacement)

    for model in embed_models:
        for initializer in model.graph.initializer:
            initializer.name = remap.get(initializer.name, initializer.name)
        for node in model.graph.node:
            for input_index, name in enumerate(node.input):
                node.input[input_index] = remap.get(name, name)
        for value in model.graph.value_info:
            value.name = remap.get(value.name, value.name)
        for annotation in model.graph.quantization_annotation:
            annotation.tensor_name = remap.get(annotation.tensor_name, annotation.tensor_name)
            for parameter in annotation.quant_parameter_tensor_names:
                parameter.value = remap.get(parameter.value, parameter.value)

    return remap


def prefixed(
    model: onnx.ModelProto,
    prefix: str,
    *,
    preserve_initializer_names: bool = False,
) -> onnx.ModelProto:
    import onnx.compose

    return onnx.compose.add_prefix(
        model,
        prefix,
        rename_nodes=True,
        rename_edges=True,
        rename_inputs=True,
        rename_outputs=True,
        rename_initializers=not preserve_initializer_names,
        rename_value_infos=True,
    )


def prefixed_preserving_initializers(
    model: onnx.ModelProto,
    prefix: str,
) -> onnx.ModelProto:
    """Prefix Encoder graph values while retaining stable namespaced weights."""
    original_initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
    }
    result = prefixed(model, prefix)
    restore = {f"{prefix}{name}": name for name in original_initializer_names}
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
    """Move every standalone Encoder initializer into a stable private namespace."""
    remap = {
        initializer.name: f"{ENCODER_INITIALIZER_PREFIX}{initializer.name}"
        for initializer in model.graph.initializer
        if not initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
    }
    if not remap:
        return
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
    if not remap:
        return 0
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
    """Rename one top-level tensor everywhere."""
    if old_name == new_name:
        return
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


def build_prefill_frontend(
    source_folder: Path,
    encoder: onnx.ModelProto | None = None,
    names: dict[str, str] | None = None,
) -> tuple[onnx.ModelProto, str, str, str | None]:
    """Prepare one prefixed Encoder component for every prefill strategy."""
    if encoder is None:
        encoder = load_model(source_folder / _name(names, "encoder"))
    output_names = [value.name for value in encoder.graph.output]
    namespace_encoder_initializers(encoder)
    component = prefixed_preserving_initializers(encoder, "encoder_")
    rename_tensor(component, f"encoder_{encoder.graph.input[0].name}", "audio")
    rename_tensor(component, f"encoder_{encoder.graph.input[1].name}", "query_embed")
    ctc_output = "encoder_enc_normed" if len(output_names) == 3 else None
    return component, "encoder_concat_embed", "encoder_ids_len", ctc_output


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


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


def copy_metadata(destination: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    existing = {prop.key: prop for prop in destination.metadata_props}
    for source in sources:
        for prop in source.metadata_props:
            if prop.key in existing:
                existing[prop.key].value = prop.value
            else:
                existing[prop.key] = destination.metadata_props.add(key=prop.key, value=prop.value)


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
            functions.setdefault(key, function)
    merged.functions.extend(functions.values())

    seen_inputs: set[str] = set()
    inputs = list(first.graph.input) + [value for value in second.graph.input if value.name not in mapped_targets]
    for value in inputs:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializer_by_name: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
        initializer_by_name.setdefault(initializer.name, initializer)
    merged.graph.initializer.extend(initializer_by_name.values())

    merged.graph.node.extend(first.graph.node)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    for node in merged.graph.node[second_start:]:
        for index, name in enumerate(node.input):
            replacement = source_by_target.get(name)
            if replacement is not None:
                node.input[index] = replacement

    seen_values = {value.name for value in merged.graph.input}
    seen_values.update(initializer_by_name)
    values = list(first.graph.value_info) + list(second.graph.value_info)
    # Preserve mapped graph-output type information as value_info. This permits a validation
    # copy to expose Main's internal logits without changing the production output contract.
    values += [value for value in first.graph.output if value.name in mapped_sources]
    values += [value for value in second.graph.input if value.name in mapped_targets]
    for value in values:
        name = source_by_target.get(value.name, value.name)
        if name in seen_values:
            continue
        copied = copy.deepcopy(value)
        copied.name = name
        merged.graph.value_info.append(copied)
        seen_values.add(name)

    seen_outputs: set[str] = set()
    outputs = [value for value in first.graph.output if value.name not in mapped_sources]
    outputs += list(second.graph.output)
    for value in outputs:
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)

    copy_metadata(merged, first, second)
    return merged


def _name(names: dict[str, str] | None, role: str) -> str:
    return model_file_names(names)[role]


def _main_kv_output_names(main: onnx.ModelProto) -> list[str]:
    return [output.name for output in main.graph.output if output.name.startswith("out_")]


def _order_kv_inputs_first(model: onnx.ModelProto) -> None:
    kv_inputs = [value for value in model.graph.input if value.name.startswith("in_")]
    other_inputs = [value for value in model.graph.input if not value.name.startswith("in_")]
    del model.graph.input[:]
    model.graph.input.extend(kv_inputs + other_inputs)


def _merge_shell_into_main(
    source_folder: Path,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    kind: str,
    names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, str, str, str | None] | None = None,
) -> tuple[onnx.ModelProto, str, onnx.ModelProto, list[str]]:
    if kind == "prefill":
        rotary = prefixed(
            load_model(source_folder / _name(names, "rotary_prefill")),
            "prefill_",
            preserve_initializer_names=True,
        )
        merged = merge_models_no_check(
            rotary,
            main,
            [
                ("prefill_rotary_cos", "rotary_cos"),
                ("prefill_rotary_sin", "rotary_sin"),
                ("prefill_attention_mask", "attention_mask"),
            ],
        )
        if prefill_frontend is None:
            prefill_frontend = build_prefill_frontend(source_folder, names=names)
        frontend, hidden_name, ids_len_name, ctc_output = prefill_frontend
        merged = merge_models_no_check(
            frontend,
            merged,
            [
                (hidden_name, "hidden_states"),
                (ids_len_name, "prefill_ids_len"),
            ],
        )
        return merged, "prefill_kv_seq_len", rotary, [ctc_output] if ctc_output else []

    decode_embed = prefixed(embed, "decode_embed_", preserve_initializer_names=True)
    rotary = prefixed(
        load_model(source_folder / _name(names, "rotary_decode")),
        "decode_",
        preserve_initializer_names=True,
    )
    mask_info = next(value for value in main.graph.input if value.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(mask_info.type.tensor_type.elem_type)
    rotary.graph.initializer.append(
        numpy_helper.from_array(
            np.zeros((1, 1, 1, 1, 1), dtype=mask_dtype),
            name="decode_zero_attention_mask",
        )
    )
    shell = merge_models_no_check(decode_embed, rotary, [])
    merged = merge_models_no_check(
        shell,
        main,
        [
            ("decode_embed_hidden_states", "hidden_states"),
            ("decode_rotary_cos", "rotary_cos"),
            ("decode_rotary_sin", "rotary_sin"),
            ("decode_zero_attention_mask", "attention_mask"),
        ],
    )
    return merged, "decode_kv_seq_len", rotary, []


def _finalize(
    merged: onnx.ModelProto,
    main: onnx.ModelProto,
    rotary: onnx.ModelProto,
    output_names: list[str],
) -> onnx.ModelProto:
    set_graph_outputs(merged, output_names)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    copy_metadata(merged, main, rotary)
    return merged


def _merge_greedy(
    source_folder,
    main,
    embed,
    kind,
    names=None,
    prefill_frontend=None,
):
    merged, kv_seq_len, rotary, aux_outputs = _merge_shell_into_main(
        source_folder, main, embed, kind, names, prefill_frontend
    )
    greedy = prefixed(load_model(source_folder / _name(names, "greedy")), "greedy_")
    merged = merge_models_no_check(merged, greedy, [("logits", "greedy_logits")])
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main)
        + ["greedy_max_logits_idx", kv_seq_len]
        + aux_outputs,
    )


def _merge_penalty_greedy_prefill(
    source_folder,
    main,
    embed,
    names=None,
    prefill_frontend=None,
):
    merged, kv_seq_len, rotary, aux_outputs = _merge_shell_into_main(
        source_folder, main, embed, "prefill", names, prefill_frontend
    )
    greedy = prefixed(load_model(source_folder / _name(names, "penalty_greedy")), "penalty_greedy_")
    merged = merge_models_no_check(merged, greedy, [("logits", "penalty_greedy_logits")])
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main)
        + ["penalty_greedy_max_logits_idx", "penalty_greedy_save_id_out", kv_seq_len]
        + aux_outputs,
    )


def _merge_penalty_greedy_decode(source_folder, main, embed, names=None):
    merged, kv_seq_len, rotary, aux_outputs = _merge_shell_into_main(
        source_folder, main, embed, "decode", names
    )
    penalty = prefixed(load_model(source_folder / _name(names, "penalty")), "penalty_")
    greedy = prefixed(load_model(source_folder / _name(names, "penalty_greedy")), "penalty_greedy_")
    merged = merge_models_no_check(merged, penalty, [("logits", "penalty_logits_in")])
    merged = merge_models_no_check(
        merged,
        greedy,
        [("penalty_logits_out", "penalty_greedy_logits")],
    )
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main)
        + ["penalty_greedy_max_logits_idx", "penalty_greedy_save_id_out", kv_seq_len],
    )


def _merge_sampling(
    source_folder,
    main,
    embed,
    kind,
    names=None,
    prefill_frontend=None,
):
    merged, kv_seq_len, rotary, aux_outputs = _merge_shell_into_main(
        source_folder, main, embed, kind, names, prefill_frontend
    )
    sampling = prefixed(load_model(source_folder / _name(names, "sampling")), "sampling_")
    merged = merge_models_no_check(merged, sampling, [("logits", "sampling_logits")])
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main)
        + ["sampling_sampled_id", "sampling_save_id_out", kv_seq_len]
        + aux_outputs,
    )


def merge_prefill_greedy(
    source_folder, main, embed, names=None, prefill_frontend=None
):
    return _merge_greedy(
        source_folder, main, embed, "prefill", names, prefill_frontend
    )


def merge_prefill_penalty_greedy(
    source_folder, main, embed, names=None, prefill_frontend=None
):
    return _merge_penalty_greedy_prefill(
        source_folder, main, embed, names, prefill_frontend
    )


def merge_prefill_sampling(
    source_folder, main, embed, names=None, prefill_frontend=None
):
    return _merge_sampling(
        source_folder, main, embed, "prefill", names, prefill_frontend
    )


def merge_decode_greedy(
    source_folder, main, embed, names=None, prefill_frontend=None
):
    return _merge_greedy(source_folder, main, embed, "decode", names)


def merge_decode_penalty_greedy(
    source_folder, main, embed, names=None, prefill_frontend=None
):
    return _merge_penalty_greedy_decode(source_folder, main, embed, names)


def merge_decode_sampling(
    source_folder, main, embed, names=None, prefill_frontend=None
):
    return _merge_sampling(source_folder, main, embed, "decode", names)


def _recipe_with_names(recipe, names, prefill_frontend=None):
    def wrapped(source_folder, main, embed):
        return recipe(
            source_folder,
            main,
            embed,
            names,
            prefill_frontend=prefill_frontend,
        )

    wrapped.__name__ = recipe.__name__
    return wrapped


def make_merged_build_plan(
    names: dict[str, str] | None = None,
    prefill_frontend: tuple[onnx.ModelProto, str, str, str | None] | None = None,
):
    n = model_file_names(names)
    return [
        (n["prefill_greedy"], _recipe_with_names(merge_prefill_greedy, n, prefill_frontend),
         [n["encoder"], n["rotary_prefill"], n["greedy"]]),
        (n["prefill_penalty_greedy"], _recipe_with_names(merge_prefill_penalty_greedy, n, prefill_frontend),
         [n["encoder"], n["rotary_prefill"], n["penalty_greedy"]]),
        (n["prefill_sampling"], _recipe_with_names(merge_prefill_sampling, n, prefill_frontend),
         [n["encoder"], n["rotary_prefill"], n["sampling"]]),
        (n["decode_greedy"], _recipe_with_names(merge_decode_greedy, n),
         [n["embed"], n["rotary_decode"], n["greedy"]]),
        (n["decode_penalty_greedy"], _recipe_with_names(merge_decode_penalty_greedy, n),
         [n["embed"], n["rotary_decode"], n["penalty"], n["penalty_greedy"]]),
        (n["decode_sampling"], _recipe_with_names(merge_decode_sampling, n),
         [n["embed"], n["rotary_decode"], n["sampling"]]),
    ]


MERGED_BUILD_PLAN = make_merged_build_plan()


def _collect_shareable_initializers(
    models: list[onnx.ModelProto],
    min_elements: int,
) -> dict[str, TensorProto]:
    shared: dict[str, TensorProto] = {}
    for model in models:
        for initializer in model.graph.initializer:
            if not _is_shareable_initializer(initializer, min_elements):
                continue
            existing = shared.get(initializer.name)
            if existing is not None and not _same_tensor(existing, initializer):
                raise RuntimeError(
                    f"Shareable initializer {initializer.name!r} has different data across donors."
                )
            shared.setdefault(initializer.name, initializer)
    return shared


def _find_single_gather(model: onnx.ModelProto) -> onnx.NodeProto | None:
    gathers = [node for node in model.graph.node if node.op_type == "Gather"]
    return gathers[0] if len(gathers) == 1 else None


def prepare_embed_for_sharing(
    embed: onnx.ModelProto,
    shared: dict[str, TensorProto],
    min_elements: int,
) -> dict:
    """Classify Embed as tied-to-lm_head or add its table to the shared set.

    The tied path is accepted only after shape, dtype, gather-axis, and full-value
    equality checks. Any mismatch falls back to sharing the original row-major
    embedding table without changing its graph semantics.
    """
    gather = _find_single_gather(embed)
    initializers = {initializer.name: initializer for initializer in embed.graph.initializer}
    if gather is None or len(gather.input) < 2 or gather.input[0] not in initializers:
        raise RuntimeError("Embed graph must contain exactly one initializer-backed Gather")
    gather_axis = next((attribute.i for attribute in gather.attribute if attribute.name == "axis"), 0)
    if gather_axis != 0:
        raise RuntimeError(f"Expected row Gather(axis=0), got axis={gather_axis}")

    table = initializers[gather.input[0]]
    if len(table.dims) != 2 or not _is_shareable_initializer(table, min_elements):
        raise RuntimeError("Embed table is not a shareable rank-2 tensor")
    vocab, hidden = map(int, table.dims)

    embed_array = numpy_helper.to_array(table)
    for name, candidate in shared.items():
        if (
            len(candidate.dims) == 2
            and tuple(map(int, candidate.dims)) == (hidden, vocab)
            and candidate.data_type == table.data_type
        ):
            candidate_array = numpy_helper.to_array(candidate)
            tied = np.array_equal(embed_array.T, candidate_array)
            del candidate_array
            if tied:
                del embed_array
                return {
                    "mode": "tied_lm_head",
                    "gather": gather,
                    "table_name": table.name,
                    "lm_head_name": name,
                    "vocab": vocab,
                    "hidden": hidden,
                }
    del embed_array

    existing = shared.get(table.name)
    if existing is not None and not _same_tensor(existing, table):
        raise RuntimeError(f"Embed initializer name collision: {table.name!r}")
    shared[table.name] = table
    return {
        "mode": "shared_embed_table",
        "gather": gather,
        "table_name": table.name,
        "vocab": vocab,
        "hidden": hidden,
    }


def finalize_embed_sharing(
    embed: onnx.ModelProto,
    prepared: dict,
    shared: dict[str, TensorProto],
    external_by_name: dict[str, dict[str, str]],
) -> dict:
    """Redirect Embed to the shared bundle, rewriting only a verified tied table."""
    if prepared["mode"] == "shared_embed_table":
        redirect_shared_initializers_to_external(embed, external_by_name)
        return {**prepared, "initializer_name": prepared["table_name"]}

    gather = prepared["gather"]
    lm_name = prepared["lm_head_name"]
    lm_tensor = shared[lm_name]
    gathered_name = gather.output[0] + "_gathered_hbs"
    replacement = [
        onnx.helper.make_node(
            "Gather",
            [lm_name, gather.input[1]],
            [gathered_name],
            name="tied_embed_gather",
            axis=1,
        ),
        onnx.helper.make_node(
            "Transpose",
            [gathered_name],
            [gather.output[0]],
            name="tied_embed_transpose",
            perm=[1, 2, 0],
        ),
    ]
    nodes: list[onnx.NodeProto] = []
    for node in embed.graph.node:
        nodes.extend(replacement if node is gather else [node])
    del embed.graph.node[:]
    embed.graph.node.extend(nodes)

    kept = [initializer for initializer in embed.graph.initializer if initializer.name != prepared["table_name"]]
    del embed.graph.initializer[:]
    embed.graph.initializer.extend(kept)
    embed.graph.initializer.append(
        make_external_initializer_ref(lm_tensor, external_by_name[lm_name])
    )
    return {**prepared, "initializer_name": lm_name}


def _external_locations(onnx_path: Path) -> set[str]:
    model = load_model(onnx_path, load_external_data=False)
    locations: set[str] = set()
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        location = _external_data_map(initializer).get("location")
        if location:
            locations.add(location)
    return locations


def _copy_onnx_with_external_data(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    for location in _external_locations(source):
        src_data = source.parent / location
        dst_data = destination.parent / location
        if src_data.resolve() == dst_data.resolve():
            continue
        dst_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_data, dst_data)


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
    names: dict[str, str] | None = None,
) -> list[str]:
    """Remove superseded split/merged strategy graphs and private sidecars."""
    n = model_file_names(names)
    old_family = "Be" + "am"
    obsolete_names = (
        f"FunASR_Nano_First_{old_family}_Search.onnx",
        f"FunASR_Nano_Second_{old_family}_Search.onnx",
        f"FunASR_Nano_TextPrefill{old_family}First.onnx",
        f"FunASR_Nano_Decode{old_family}Next.onnx",
        f"FunASR_Nano_DecodePenalty{old_family}Next.onnx",
    )
    protected = {n["shared_initializers"], n["shared_initializers_data"]}
    return _delete_named_graph_artifacts(Path(folder), obsolete_names, protected)


def copy_runtime_standalones(
    source_folder: Path,
    out_folder: Path,
    names: dict[str, str] | None = None,
) -> list[Path]:
    n = model_file_names(names)
    copied: list[Path] = []
    for role in RUNTIME_STANDALONE_ROLE_KEYS:
        source = Path(source_folder) / n[role]
        if not source.exists():
            if role == "metadata":
                raise FileNotFoundError(source)
            continue
        destination = Path(out_folder) / n[role]
        if source.resolve() != destination.resolve():
            _copy_onnx_with_external_data(source, destination)
        copied.append(destination)
    return copied


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names_override: dict[str, str] | None = None,
    *,
    copy_standalones: bool = True,
    delete_constituents: bool | None = None,
) -> dict:
    """Build six strategies and one shared Encoder/Main/Embed weight blob."""
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    names = model_file_names(model_file_names_override)
    delete_obsolete_strategy_artifacts(source_folder, names)
    if out_folder.resolve() != source_folder.resolve():
        delete_obsolete_strategy_artifacts(out_folder, names)

    main_path = source_folder / names["main"]
    encoder_path = source_folder / names["encoder"]
    embed_path = source_folder / names["embed"]
    main = load_model(main_path)
    encoder = load_model(encoder_path)
    namespace_encoder_initializers(encoder)
    embed = load_model(embed_path)
    rotary_models = [
        load_model(source_folder / names[role])
        for role in ("rotary_prefill", "rotary_decode")
        if (source_folder / names[role]).exists()
    ]
    shared = _collect_shareable_initializers(
        [main, encoder, *rotary_models], min_shared_elements
    )
    embed_prepared = prepare_embed_for_sharing(embed, shared, min_shared_elements)
    shared_model_path = out_folder / names["shared_initializers"]
    save_shared_initializers_from_tensors(shared, shared_model_path)
    external_by_name = shared_external_data_map(shared_model_path)
    redirect_shared_initializers_to_external(main, external_by_name)
    redirect_shared_initializers_to_external(encoder, external_by_name)
    embed_sharing = finalize_embed_sharing(embed, embed_prepared, shared, external_by_name)
    del shared, rotary_models

    # Keep data-light component donors for Optimize_ONNX.py. Runtime never loads them.
    main_out = out_folder / names["main"]
    encoder_out = out_folder / names["encoder"]
    save_model(main, main_out)
    save_model(encoder, encoder_out)
    prefill_frontend = build_prefill_frontend(
        source_folder,
        encoder=encoder,
        names=names,
    )

    # Standalone Embed is required for task-prompt embedding and shares the exact same blob
    # as the copy folded into every decode graph.
    embed_out = out_folder / names["embed"]
    save_model(embed, embed_out)

    graphs: dict[str, Path] = {}
    for file_name, recipe, _ in make_merged_build_plan(
        names, prefill_frontend=prefill_frontend
    ):
        merged = recipe(source_folder, main, embed)
        redirect_shared_initializers_to_external(merged, external_by_name)
        out_path = out_folder / file_name
        save_model(merged, out_path)
        graphs[file_name] = out_path
        del merged

    copied = copy_runtime_standalones(source_folder, out_folder, names) if copy_standalones else []
    result = {
        "graphs": graphs,
        "skipped": {},
        "shared_model": shared_model_path,
        "shared_data": out_folder / names["shared_initializers_data"],
        "embed": embed_out,
        "standalone_main": main_out,
        "standalone_encoder": encoder_out,
        "embed_sharing": embed_sharing,
        "standalones": copied,
    }

    if delete_constituents is None:
        delete_constituents = out_folder.resolve() == source_folder.resolve()
    if delete_constituents:
        result["removed_constituents"] = delete_merged_constituents(
            out_folder,
            names,
            protected_names=(
                names["shared_initializers"],
                names["shared_initializers_data"],
                names["embed"],
                names["main"],
                names["encoder"],
            ),
        )
    return result


def delete_merged_constituents(
    folder: Path,
    names: dict[str, str] | None = None,
    protected_names: tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    n = model_file_names(names)
    protected = set(protected_names or ())
    protected.update((n["shared_initializers"], n["shared_initializers_data"], n["embed"]))
    removed: list[str] = []
    for role in MERGED_CONSTITUENT_ROLE_KEYS:
        onnx_path = Path(folder) / n[role]
        if not onnx_path.exists() or onnx_path.name in protected:
            continue
        for location in _external_locations(onnx_path):
            if location in protected:
                continue
            data_path = onnx_path.parent / location
            if data_path.exists():
                data_path.unlink()
                removed.append(data_path.name)
        onnx_path.unlink()
        removed.append(onnx_path.name)
        sidecar = onnx_path.with_name(onnx_path.name + ".data")
        if sidecar.exists() and sidecar.name not in protected:
            sidecar.unlink()
            removed.append(sidecar.name)
    return removed


# ---------------------------------------------------------------------------
# Quantized bundle support: transplant one quantized Main and one quantized Embed.
# ---------------------------------------------------------------------------
def _node_is_shell(node: onnx.NodeProto) -> bool:
    precision_free_prefix = "InsertedPrecisionFreeCast_"
    return any(
        output.startswith(SHELL_PREFIXES)
        or (
            output.startswith(precision_free_prefix)
            and output[len(precision_free_prefix):].startswith(SHELL_PREFIXES)
        )
        for output in node.output
    )


def _node_is_decode_embed(node: onnx.NodeProto) -> bool:
    return any(output.startswith("decode_embed_") for output in node.output)


def _used_inputs(nodes) -> set[str]:
    return {name for node in nodes for name in node.input if name}


def _copy_node_with_input_remap(node: onnx.NodeProto, remap: dict[str, str]) -> onnx.NodeProto:
    copied = copy.deepcopy(node)
    for index, name in enumerate(copied.input):
        copied.input[index] = remap.get(name, name)
    return copied


def _copy_value_info_with_name(value: onnx.ValueInfoProto, name: str) -> onnx.ValueInfoProto:
    copied = copy.deepcopy(value)
    copied.name = name
    return copied


def _tensor_element_types(model: onnx.ModelProto) -> dict[str, int]:
    """Collect known top-level tensor element types without running shape inference."""
    types: dict[str, int] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.type.HasField("tensor_type"):
            elem_type = int(value.type.tensor_type.elem_type)
            if elem_type != TensorProto.UNDEFINED:
                types[value.name] = elem_type
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.UNDEFINED:
            types[initializer.name] = int(initializer.data_type)
    return types


def _remap_element_types(
    types: dict[str, int],
    remap: dict[str, str],
) -> dict[str, int]:
    remapped: dict[str, int] = {}
    for name, elem_type in types.items():
        target_name = remap.get(name, name)
        existing = remapped.get(target_name)
        if existing is not None and existing != elem_type:
            raise RuntimeError(
                f"Conflicting element types after remapping {target_name!r}: "
                f"{existing} != {elem_type}."
            )
        remapped[target_name] = elem_type
    return remapped


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


def _target_main_remap(
    target: onnx.ModelProto,
    primary_main_nodes: list[onnx.NodeProto],
) -> dict[str, str]:
    names = {value.name for value in target.graph.input}
    names.update(initializer.name for initializer in target.graph.initializer)
    for node in target.graph.node:
        names.update(node.output)
    donor_inputs = _used_inputs(primary_main_nodes)
    target_is_decode = "decode_rotary_cos" in names
    remap: dict[str, str] = {}
    if not target_is_decode and "hidden_states" in donor_inputs:
        target_main_nodes = [
            node for node in target.graph.node if not _node_is_shell(node)
        ]
        target_main_inputs = _used_inputs(target_main_nodes)
        encoder_outputs = {
            output
            for node in target.graph.node
            for output in node.output
            if output.startswith("encoder_")
        }
        hidden_candidates = sorted(encoder_outputs & target_main_inputs)
        if hidden_candidates != ["encoder_concat_embed"]:
            raise RuntimeError(
                "Cannot identify the Fun-ASR Encoder-to-Main hidden-state boundary: "
                f"found {hidden_candidates}."
            )
        remap["hidden_states"] = hidden_candidates[0]
    if "prefill_rotary_cos" in donor_inputs:
        if not target_is_decode:
            remap["InsertedPrecisionFreeCast_prefill_attention_mask"] = (
                "prefill_attention_mask"
            )
            return remap
        remap.update(
            {
                "prefill_rotary_cos": "decode_rotary_cos",
                "prefill_rotary_sin": "decode_rotary_sin",
                "prefill_attention_mask": "decode_zero_attention_mask",
                "InsertedPrecisionFreeCast_prefill_attention_mask": "decode_zero_attention_mask",
                "hidden_states": "decode_embed_hidden_states",
            }
        )
        return remap
    if target_is_decode:
        remap.update(
            {
                "rotary_cos": "decode_rotary_cos",
                "rotary_sin": "decode_rotary_sin",
                "attention_mask": "decode_zero_attention_mask",
                "hidden_states": "decode_embed_hidden_states",
            }
        )
        return remap
    remap.update(
        {
            "rotary_cos": "prefill_rotary_cos",
            "rotary_sin": "prefill_rotary_sin",
            "attention_mask": "prefill_attention_mask",
        }
    )
    return remap


def transplant_quantized_components(
    target: onnx.ModelProto,
    quantized_primary: onnx.ModelProto,
    quantized_embed: onnx.ModelProto | None = None,
) -> onnx.ModelProto:
    """Replace a merged target's Main and decode-Embed blocks with canonical donors."""
    donor_main_nodes = [
        node for node in quantized_primary.graph.node if not _node_is_shell(node)
    ]
    remap = _target_main_remap(target, donor_main_nodes)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in donor_main_nodes
    ]
    if not primary_main_nodes:
        raise RuntimeError("Quantized primary graph contains no Main node block.")

    primary_output_info = {value.name: value for value in quantized_primary.graph.output}
    primary_types = _remap_element_types(_tensor_element_types(quantized_primary), remap)
    target_types = _tensor_element_types(target)

    has_decode_embed = any(_node_is_decode_embed(node) for node in target.graph.node)
    embed_shell = None
    embed_nodes: list[onnx.NodeProto] = []
    if has_decode_embed:
        if quantized_embed is None:
            raise RuntimeError("Decode target requires a quantized Embed donor.")
        embed_shell = prefixed(
            quantized_embed,
            "decode_embed_",
            preserve_initializer_names=True,
        )
        embed_nodes = [copy.deepcopy(node) for node in embed_shell.graph.node]

    # Source strategy shells remain in their exported dtypes while the canonical
    # F16 donor converts Main activations. Add narrow boundary adapters rather than
    # converting or duplicating every strategy's multi-gigabyte Main weights.
    # Typical F16 adapters are rotary float32->float16 before Main and logits
    # float16->float32 before selection heads; Q4/F32 donors need none.
    upstream_types = dict(target_types)
    if embed_shell is not None:
        upstream_types.update(_tensor_element_types(embed_shell))

    main_outputs = {
        output for node in primary_main_nodes for output in node.output if output
    }
    target_shell_nodes = [
        node
        for node in target.graph.node
        if _node_is_shell(node) and not _node_is_decode_embed(node)
    ]
    upstream_outputs = {
        output for node in target_shell_nodes for output in node.output if output
    }
    upstream_outputs.update(
        output for node in embed_nodes for output in node.output if output
    )

    reserved_names = {value.name for value in target.graph.input}
    reserved_names.update(initializer.name for initializer in target.graph.initializer)
    reserved_names.update(
        output for node in target.graph.node for output in node.output if output
    )
    reserved_names.update(
        output for node in primary_main_nodes for output in node.output if output
    )
    reserved_names.update(
        output for node in embed_nodes for output in node.output if output
    )

    main_input_remap: dict[str, str] = {}
    pre_main_casts: list[onnx.NodeProto] = []
    main_external_inputs = _used_inputs(primary_main_nodes) - main_outputs
    for name in sorted(main_external_inputs & upstream_outputs):
        source_type = upstream_types.get(name)
        target_type = primary_types.get(name)
        if source_type is None or target_type is None or source_type == target_type:
            continue
        adapted = _unique_tensor_name(f"{name}_cast_to_main", reserved_names)
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
        adapted = _unique_tensor_name(f"{name}_cast_to_shell", reserved_names)
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
    inserted_main = False
    inserted_embed = False
    for node in target.graph.node:
        if _node_is_decode_embed(node):
            if not inserted_embed:
                new_nodes.extend(copy.deepcopy(embed_nodes))
                inserted_embed = True
            continue
        if _node_is_shell(node):
            new_nodes.append(_copy_node_with_input_remap(node, shell_input_remap))
            continue
        if not inserted_main:
            new_nodes.extend(copy.deepcopy(pre_main_casts))
            new_nodes.extend(copy.deepcopy(primary_main_nodes))
            new_nodes.extend(copy.deepcopy(post_main_casts))
            inserted_main = True
    if not inserted_main:
        new_nodes.extend(copy.deepcopy(pre_main_casts))
        new_nodes.extend(copy.deepcopy(primary_main_nodes))
        new_nodes.extend(copy.deepcopy(post_main_casts))

    primary_initializers = {initializer.name: initializer for initializer in quantized_primary.graph.initializer}
    target_initializers = {initializer.name: initializer for initializer in target.graph.initializer}
    embed_initializers = (
        {initializer.name: initializer for initializer in quantized_embed.graph.initializer}
        if quantized_embed is not None
        else {}
    )
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
    embed_initializer_names = (
        _used_inputs(embed_nodes)
        & set(embed_initializers)
        & required_initializers
    )

    new_initializers: list[TensorProto] = []
    for name in sorted(required_initializers):
        source = None
        if name in main_initializer_names:
            source = primary_initializers.get(name)
        elif name in embed_initializer_names:
            source = embed_initializers.get(name)
        if source is None:
            source = target_initializers.get(name)
        if source is None:
            source = primary_initializers.get(name)
        if source is None:
            source = embed_initializers.get(name)
        if source is not None:
            new_initializers.append(copy.deepcopy(source))

    del merged.graph.node[:]
    merged.graph.node.extend(new_nodes)
    del merged.graph.initializer[:]
    merged.graph.initializer.extend(new_initializers)

    # Rebuild the Main-facing graph inputs from the quantized donor while retaining target
    # shell/head inputs. This keeps quantizer-added graph inputs and changed I/O dtypes exact.
    shell_input_names = {
        name
        for node in target.graph.node
        if _node_is_shell(node)
        for name in node.input
        if name
    }
    target_inputs = {value.name: value for value in target.graph.input}
    rebuilt_inputs: list[onnx.ValueInfoProto] = []
    seen_inputs: set[str] = set()
    donor_main_external_inputs = _used_inputs(donor_main_nodes) & {
        value.name for value in quantized_primary.graph.input
    }
    for value in quantized_primary.graph.input:
        if value.name not in donor_main_external_inputs:
            continue
        name = remap.get(value.name, value.name)
        if value.name in remap or name in seen_inputs:
            continue
        rebuilt_inputs.append(_copy_value_info_with_name(value, name))
        seen_inputs.add(name)
    for name, value in target_inputs.items():
        if name in seen_inputs or name not in shell_input_names:
            continue
        rebuilt_inputs.append(copy.deepcopy(value))
        seen_inputs.add(name)
    del merged.graph.input[:]
    merged.graph.input.extend(rebuilt_inputs)

    # Keep target strategy outputs, but source plain Main KV value_info from the quantized
    # donor. Strategy outputs remain defined by the shell graph and retain target metadata.
    rebuilt_outputs: list[onnx.ValueInfoProto] = []
    for value in target.graph.output:
        donor = primary_output_info.get(value.name)
        rebuilt_outputs.append(copy.deepcopy(donor if donor is not None else value))
    del merged.graph.output[:]
    merged.graph.output.extend(rebuilt_outputs)

    existing = {value.name for value in merged.graph.input}
    existing.update(value.name for value in merged.graph.output)
    existing.update(initializer.name for initializer in merged.graph.initializer)
    new_value_infos: list[onnx.ValueInfoProto] = []

    def add_value(value: onnx.ValueInfoProto, name: str | None = None) -> None:
        value_name = name or value.name
        if value_name in existing:
            return
        new_value_infos.append(_copy_value_info_with_name(value, value_name))
        existing.add(value_name)

    for value in quantized_primary.graph.value_info:
        name = remap.get(value.name, value.name)
        add_value(value, main_input_remap.get(name, name))
    if embed_shell is not None:
        for value in list(embed_shell.graph.value_info) + list(embed_shell.graph.output):
            add_value(value)
    for value in target.graph.value_info:
        if value.name.startswith(SHELL_PREFIXES):
            add_value(value)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(new_value_infos)
    _merge_opsets(merged, quantized_primary, *(tuple([quantized_embed]) if quantized_embed else ()))
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    return merged


def _node_is_encoder_component(node: onnx.NodeProto) -> bool:
    if node.name.startswith("BoundaryCast/"):
        return False
    alias_prefix = "InsertedPrecisionFreeCast_"
    return any(
        output.startswith("encoder_")
        or (
            output.startswith(alias_prefix)
            and output[len(alias_prefix):].startswith("encoder_")
        )
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
        if any(
            name == "audio" or name.startswith("encoder_")
            for value in (*target.graph.input, *target.graph.output)
            for name in (value.name,)
        ):
            raise RuntimeError("Encoder-prefill graph has no classifiable Encoder nodes.")
        return target

    donor_input_names = [value.name for value in optimized_encoder.graph.input]
    donor_output_names = [value.name for value in optimized_encoder.graph.output]
    if donor_input_names != ["audio", "query_embed"]:
        raise RuntimeError(
            "Optimized Fun-ASR Encoder input ABI changed: "
            f"expected ['audio', 'query_embed'], got {donor_input_names}."
        )
    if donor_output_names[:2] != ["concat_embed", "ids_len"] or len(
        donor_output_names
    ) not in (2, 3):
        raise RuntimeError(
            "Optimized Fun-ASR Encoder output ABI changed: "
            f"got {donor_output_names}."
        )
    if len(donor_output_names) == 3 and donor_output_names[2] != "enc_normed":
        raise RuntimeError(
            f"Unexpected optimized Encoder auxiliary output {donor_output_names[2]!r}."
        )
    query_shape = optimized_encoder.graph.input[1].type.tensor_type.shape
    concat_shape = optimized_encoder.graph.output[0].type.tensor_type.shape
    if (
        len(query_shape.dim) != 3
        or len(concat_shape.dim) != 3
        or (
            query_shape.dim[1].dim_param
            and query_shape.dim[1].dim_param == concat_shape.dim[1].dim_param
        )
    ):
        raise RuntimeError(
            "Optimized Encoder has an unsafe query/concat symbolic sequence ABI."
        )

    donor_source = optimized_encoder
    if any(
        not initializer.name.startswith(ENCODER_INITIALIZER_PREFIX)
        for initializer in optimized_encoder.graph.initializer
    ):
        donor_source = copy.deepcopy(optimized_encoder)
        namespace_encoder_initializers(donor_source)
    donor = prefixed_preserving_initializers(donor_source, "encoder_")
    input_remap = {
        "encoder_audio": "audio",
        "encoder_query_embed": "query_embed",
    }
    donor_nodes = [
        _copy_node_with_input_remap(node, input_remap)
        for node in donor.graph.node
    ]
    donor_input_info = {
        input_remap.get(value.name, value.name): _copy_value_info_with_name(
            value, input_remap.get(value.name, value.name)
        )
        for value in donor.graph.input
    }
    if set(donor_input_info) != {"audio", "query_embed"}:
        raise RuntimeError(
            "Prefixed optimized Encoder inputs do not map to the public prefill ABI: "
            f"{sorted(donor_input_info)}."
        )
    target_input_names = {value.name for value in target.graph.input}
    missing_target_inputs = set(donor_input_info) - target_input_names
    if missing_target_inputs:
        raise RuntimeError(
            "Prefill template is missing optimized Encoder input(s): "
            f"{sorted(missing_target_inputs)}."
        )
    donor_output_info = {value.name: value for value in donor.graph.output}

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
    missing_boundary = required_boundary - donor_outputs
    if missing_boundary:
        raise RuntimeError(
            "Optimized Encoder does not reproduce prefill boundary tensor(s): "
            f"{sorted(missing_boundary)}."
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

    # The raw strategy template owns the shell topology, but the optimized donor owns
    # the Encoder tensor ABI. In particular, an F16 donor exposes F16 audio/query inputs
    # and F16 concat/CTC outputs. Retaining the raw template's F32 declarations makes
    # otherwise-correct donor nodes internally contradictory and also causes the runtime
    # to construct the wrong OrtValue dtype. Replace only the Encoder-facing public I/O;
    # all KV, history, strategy, and shell interfaces remain target-owned.
    rebuilt_inputs = [
        copy.deepcopy(donor_input_info.get(value.name, value))
        for value in target.graph.input
    ]
    del merged.graph.input[:]
    merged.graph.input.extend(rebuilt_inputs)

    rebuilt_outputs: list[onnx.ValueInfoProto] = []
    for value in target.graph.output:
        donor_value = donor_output_info.get(value.name)
        rebuilt_outputs.append(copy.deepcopy(donor_value if donor_value is not None else value))
    del merged.graph.output[:]
    merged.graph.output.extend(rebuilt_outputs)

    retained_info = [
        value
        for value in target.graph.value_info
        if not value.name.startswith("encoder_")
        and not value.name.startswith("InsertedPrecisionFreeCast_encoder_")
    ]
    existing_info = {value.name for value in merged.graph.input}
    existing_info.update(value.name for value in merged.graph.output)
    existing_info.update(initializer.name for initializer in merged.graph.initializer)
    existing_info.update(value.name for value in retained_info)
    for value in list(donor.graph.output) + list(donor.graph.value_info):
        name = input_remap.get(value.name, value.name)
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
    models: list[onnx.ModelProto],
    shared_model_path: Path,
    *,
    primary_model: onnx.ModelProto,
    embed_model: onnx.ModelProto | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
) -> dict[str, dict[str, str]]:
    """Extract large optimized Encoder/Main/Embed tensors into one bundle."""
    main_nodes = [node for node in primary_model.graph.node if not _node_is_shell(node)]
    encoder_nodes = [
        node for node in primary_model.graph.node if _node_is_encoder_component(node)
    ]
    component_inputs = _used_inputs(main_nodes + encoder_nodes)
    shared: dict[str, TensorProto] = {}

    for initializer in primary_model.graph.initializer:
        if (
            initializer.name in component_inputs
            and _is_shareable_initializer(initializer, min_shared_elements)
        ):
            shared[initializer.name] = initializer

    if embed_model is not None:
        embed_inputs = _used_inputs(embed_model.graph.node)
        for initializer in embed_model.graph.initializer:
            if initializer.name not in embed_inputs or not _is_shareable_initializer(initializer, min_shared_elements):
                continue
            existing = shared.get(initializer.name)
            if existing is not None and not _same_tensor(existing, initializer):
                raise RuntimeError(
                    "Encoder/Main/Embed shared initializer collision: "
                    f"{initializer.name!r}"
                )
            shared[initializer.name] = initializer

    if not shared:
        raise RuntimeError(
            "Optimized Encoder/Main/Embed donors contain no shareable initializer."
        )
    save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    for model in models:
        redirect_shared_initializers_to_external(model, external_by_name)
    return external_by_name


# ---------------------------------------------------------------------------
# Runtime attachment: one mmap and add_initializer set per ORT session.
# ---------------------------------------------------------------------------
def attach_shared_initializers(session_options, shared_model_path: Path):
    """Attach mmap-backed shared initializers and return refs that must stay alive."""
    import onnxruntime as ort

    shared_model_path = Path(shared_model_path)
    shared_model = load_model(shared_model_path, load_external_data=False)
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
        dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        shape = tuple(int(dim) for dim in initializer.dims)
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        arrays[initializer.name] = array
        ort_value = ort.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(initializer.name, ort_value)
    return arrays, ort_values
