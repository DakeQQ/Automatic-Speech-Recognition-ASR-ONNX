"""Build the Qwen ForcedAligner merged NAR graph with one shared weight blob.

Qwen3-ForcedAligner is deliberately non-autoregressive: it has no KV cache,
decode loop, autoregressive search, or prefill/decode split. Its architecture-appropriate
merged deployment is therefore one graph:

    Embed -> Encoder -> Rotary+Mask -> Decoder Main (timestamp ArgMax)

All large numeric initializers from those four constituents are streamed once to
``ForcedAligner_SharedInitializers.onnx.data``.  The merged graph keeps lightweight
external references, and inference injects mmap-backed OrtValues with
``SessionOptions.add_initializer``.
"""

from __future__ import annotations

import copy
import gc
import hashlib
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


MIN_SHARED_INITIALIZER_ELEMENTS = 1024
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

SHELL_PREFIXES = ("embed_", "encoder_", "rotary_")
DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "embed": "ForcedAligner_Embed.onnx",
    "encoder": "ForcedAligner_Encoder.onnx",
    "rotary_mask": "ForcedAligner_Rotary_Mask.onnx",
    "main": "ForcedAligner_Decoder_Main.onnx",
    "merged": "ForcedAligner_Merged.onnx",
    "shared_initializers": "ForcedAligner_SharedInitializers.onnx",
    "shared_initializers_data": "ForcedAligner_SharedInitializers.onnx.data",
}
MERGED_CONSTITUENT_KEYS = ("embed", "encoder", "rotary_mask", "main")
MERGED_INPUT_NAMES = ("audio", "input_ids")
MERGED_OUTPUT_NAMES = ("output_ids",)


def _model_file_names(overrides: Mapping[str, str] | None) -> dict[str, str]:
    names = dict(DEFAULT_MODEL_FILE_NAMES)
    if overrides:
        names.update({str(key): str(value) for key, value in overrides.items()})
    names["shared_initializers_data"] = names["shared_initializers"] + ".data"
    return names


def load_model(path: Path, *, load_external_data: bool = True) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=load_external_data)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a data-light graph without creating a per-graph external sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
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


def _tensor_bytes(initializer: TensorProto) -> bytes:
    if initializer.raw_data:
        return initializer.raw_data
    return numpy_helper.to_array(initializer).tobytes(order="C")


def make_external_initializer_ref(
    initializer: TensorProto,
    external_data: Mapping[str, str],
) -> TensorProto:
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
            entry.value = str(value)
    return ref


def save_shared_initializers_from_tensors(
    shared: Mapping[str, TensorProto],
    path: Path,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Stream a tensor mapping to one shared carrier and one external data file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    refs: list[TensorProto] = []
    regions_by_digest: dict[tuple[int, bytes], tuple[int, int]] = {}
    offset = 0
    with open(data_path, "wb") as stream:
        for name, initializer in sorted(shared.items()):
            raw = _tensor_bytes(initializer)
            length = len(raw)
            digest_key = (length, hashlib.sha256(raw).digest())
            region = regions_by_digest.get(digest_key)
            if region is None:
                region = (offset, length)
                regions_by_digest[digest_key] = region
                stream.write(raw)
                offset += length
            external = {
                "location": data_name,
                "offset": str(region[0]),
                "length": str(region[1]),
            }
            ref = make_external_initializer_ref(initializer, external)
            ref.name = name
            refs.append(ref)
    _save_shared_carrier(path, refs, metadata)


def _save_shared_carrier(
    path: Path,
    refs: Iterable[TensorProto],
    metadata: Mapping[str, object] | None,
) -> None:
    refs = list(refs)
    graph = onnx.helper.make_graph([], "forced_aligner_shared_initializers", [], [], initializer=refs)
    model = onnx.helper.make_model(
        graph,
        producer_name="Qwen_ForcedAligner/Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    _apply_metadata(model, metadata)
    _set_metadata_value(model, "qwen_forcedaligner_shared_initializers", "1")
    _set_metadata_value(model, "initializer_count", str(len(refs)))
    onnx.save_model(model, str(path))


def shared_external_data_map(shared_model_path: Path) -> dict[str, dict[str, str]]:
    model = load_model(Path(shared_model_path), load_external_data=False)
    result = {}
    for initializer in model.graph.initializer:
        external = _external_data_map(initializer)
        external["__tensor_data_type"] = str(initializer.data_type)
        external["__tensor_dims"] = ",".join(str(dim) for dim in initializer.dims)
        result[initializer.name] = external
    return result


def redirect_shared_initializers_to_external(
    model: onnx.ModelProto,
    external_by_name: Mapping[str, Mapping[str, str]],
) -> int:
    rewritten: list[TensorProto] = []
    count = 0
    for initializer in model.graph.initializer:
        external = external_by_name.get(initializer.name)
        if external is None:
            rewritten.append(initializer)
            continue
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
    """Restore public names orphaned when a precision-boundary Cast is removed."""
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
                f"Cannot restore public output {public_name!r}: expected one producer "
                f"for {alias!r}, found {len(owners)}."
            )
        remap[alias] = public_name
    if not remap:
        return {}
    for node in model.graph.node:
        for input_index, name in enumerate(node.input):
            node.input[input_index] = remap.get(name, name)
        for output_index, name in enumerate(node.output):
            node.output[output_index] = remap.get(name, name)
    kept = [value for value in model.graph.value_info if value.name not in remap]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    return remap


def _tensor_element_types(model: onnx.ModelProto) -> dict[str, int]:
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


def prefixed(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    """Prefix every private name in a constituent; Main intentionally stays unprefixed."""
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
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


def set_graph_outputs(model: onnx.ModelProto, output_names: list[str]) -> None:
    by_name = value_info_by_name(model)
    del model.graph.output[:]
    model.graph.output.extend(by_name[name] for name in output_names)


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
    """Compose two potentially >2-GiB models without invoking ``onnx.checker``."""
    first_types = _tensor_element_types(first)
    second_types = _tensor_element_types(second)
    reserved = {value.name for value in first.graph.input}
    reserved.update(value.name for value in second.graph.input)
    reserved.update(initializer.name for initializer in first.graph.initializer)
    reserved.update(initializer.name for initializer in second.graph.initializer)
    reserved.update(
        output
        for model in (first, second)
        for node in model.graph.node
        for output in node.output
        if output
    )
    float_types = {
        TensorProto.FLOAT16,
        TensorProto.FLOAT,
        TensorProto.DOUBLE,
        TensorProto.BFLOAT16,
    }
    source_by_target: dict[str, str] = {}
    mapped_sources: set[str] = set()
    connection_casts: list[onnx.NodeProto] = []
    cast_value_infos: list[onnx.ValueInfoProto] = []
    for connection_index, (source, target) in enumerate(io_map):
        source_type = first_types.get(source)
        target_type = second_types.get(target)
        if source_type is None or target_type is None:
            raise RuntimeError(
                f"Cannot determine component interface types for {source!r} -> {target!r}."
            )
        effective_source = source
        if source_type != target_type:
            if source_type not in float_types or target_type not in float_types:
                raise RuntimeError(
                    f"Unsafe component type mismatch {source!r} ({source_type}) -> "
                    f"{target!r} ({target_type})."
                )
            base = f"{source}_cast_for_{target}"
            effective_source = base
            suffix = 1
            while effective_source in reserved:
                effective_source = f"{base}_{suffix}"
                suffix += 1
            reserved.add(effective_source)
            connection_casts.append(
                onnx.helper.make_node(
                    "Cast",
                    [source],
                    [effective_source],
                    name=f"ComponentBoundaryCast/{connection_index}",
                    to=target_type,
                )
            )
            cast_value_infos.append(
                onnx.helper.make_tensor_value_info(effective_source, target_type, None)
            )
        source_by_target[target] = effective_source
        mapped_sources.add(source)
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "Qwen_ForcedAligner/Shared_Merged.py"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"

    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    seen_inputs: set[str] = set()
    candidates = list(first.graph.input) + [value for value in second.graph.input if value.name not in mapped_targets]
    for value in candidates:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    initializers: dict[str, TensorProto] = {}
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
        initializers.setdefault(initializer.name, initializer)
    merged.graph.initializer.extend(initializers.values())

    merged.graph.node.extend(first.graph.node)
    merged.graph.node.extend(connection_casts)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    for node in merged.graph.node[second_start:]:
        for index, name in enumerate(node.input):
            node.input[index] = source_by_target.get(name, name)

    seen_value_info = {value.name for value in merged.graph.input}
    seen_value_info.update(initializers)
    values = (
        list(first.graph.value_info)
        + list(first.graph.output)
        + cast_value_infos
        + list(second.graph.value_info)
        + list(second.graph.output)
    )
    for value in values:
        if value.name not in seen_value_info:
            merged.graph.value_info.append(value)
            seen_value_info.add(value.name)

    seen_outputs: set[str] = set()
    candidates = [value for value in first.graph.output if value.name not in mapped_sources] + list(second.graph.output)
    for value in candidates:
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)

    copy_metadata(merged, first, second)
    return merged


def _rename_value(model: onnx.ModelProto, old_name: str, new_name: str) -> None:
    if old_name == new_name:
        return
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
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


def _order_public_inputs(model: onnx.ModelProto, names: tuple[str, ...]) -> None:
    by_name = {value.name: value for value in model.graph.input}
    del model.graph.input[:]
    model.graph.input.extend(by_name[name] for name in names)


def _set_metadata_value(model: onnx.ModelProto, key: str, value: str) -> None:
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    model.metadata_props.add(key=key, value=value)


def _apply_metadata(model: onnx.ModelProto, metadata: Mapping[str, object] | None) -> None:
    if not metadata:
        return
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, bool):
            text = "1" if value else "0"
        elif isinstance(value, (list, tuple)):
            text = ",".join(str(item) for item in value)
        else:
            text = str(value)
        _set_metadata_value(model, str(key), text)


def _externalize_model_initializers(
    model: onnx.ModelProto,
    stream,
    data_name: str,
    refs: list[TensorProto],
    offset: int,
    min_elements: int,
    regions_by_digest: dict[tuple[int, bytes], tuple[int, int]],
) -> tuple[int, int]:
    rewritten: list[TensorProto] = []
    shared_count = 0
    for initializer in model.graph.initializer:
        if not _is_shareable_initializer(initializer, min_elements):
            rewritten.append(initializer)
            continue
        raw = _tensor_bytes(initializer)
        length = len(raw)
        digest_key = (length, hashlib.sha256(raw).digest())
        region = regions_by_digest.get(digest_key)
        if region is None:
            region = (offset, length)
            regions_by_digest[digest_key] = region
            stream.write(raw)
            offset += length
        external = {
            "location": data_name,
            "offset": str(region[0]),
            "length": str(region[1]),
        }
        ref = make_external_initializer_ref(initializer, external)
        rewritten.append(ref)
        refs.append(copy.deepcopy(ref))
        shared_count += 1

    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    return offset, shared_count


def _load_and_externalize_components(
    source_folder: Path,
    names: Mapping[str, str],
    shared_model_path: Path,
    min_shared_elements: int,
    metadata: Mapping[str, object] | None,
) -> tuple[dict[str, onnx.ModelProto], int]:
    shared_model_path.parent.mkdir(parents=True, exist_ok=True)
    data_name = shared_model_path.name + ".data"
    data_path = shared_model_path.with_name(data_name)
    shared_model_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    components: dict[str, onnx.ModelProto] = {}
    refs: list[TensorProto] = []
    offset = 0
    shared_count = 0
    regions_by_digest: dict[tuple[int, bytes], tuple[int, int]] = {}
    component_prefixes = {
        "embed": "embed_",
        "encoder": "encoder_",
        "rotary_mask": "rotary_",
        "main": "",
    }

    try:
        with open(data_path, "wb") as stream:
            for role in ("embed", "encoder", "rotary_mask", "main"):
                path = source_folder / names[role]
                loaded = load_model(path, load_external_data=True)
                prefix = component_prefixes[role]
                model = prefixed(loaded, prefix) if prefix else loaded
                if model is not loaded:
                    del loaded
                    gc.collect()
                offset, count = _externalize_model_initializers(
                    model,
                    stream,
                    data_name,
                    refs,
                    offset,
                    min_shared_elements,
                    regions_by_digest,
                )
                shared_count += count
                components[role] = model
    except BaseException:
        shared_model_path.unlink(missing_ok=True)
        data_path.unlink(missing_ok=True)
        raise

    _save_shared_carrier(shared_model_path, refs, metadata)
    return components, shared_count


def _compose_forced_aligner(components: Mapping[str, onnx.ModelProto]) -> onnx.ModelProto:
    merged = merge_models_no_check(
        components["embed"],
        components["encoder"],
        io_map=[("embed_text_embed", "encoder_text_embed")],
    )
    merged = merge_models_no_check(
        merged,
        components["rotary_mask"],
        io_map=[("encoder_ids_len", "rotary_ids_len")],
    )
    merged = merge_models_no_check(
        merged,
        components["main"],
        io_map=[
            ("encoder_concat_embed", "hidden_states"),
            ("rotary_rotary_cos", "rotary_cos"),
            ("rotary_rotary_sin", "rotary_sin"),
            ("rotary_attention_mask", "attention_mask"),
        ],
    )
    _rename_value(merged, "encoder_audio", "audio")
    _rename_value(merged, "embed_input_ids", "input_ids")
    set_graph_outputs(merged, ["output_ids"])
    _order_public_inputs(merged, MERGED_INPUT_NAMES)
    merged.producer_name = "Qwen_ForcedAligner/Shared_Merged.py"
    return merged


def _external_locations(path: Path) -> set[str]:
    if not path.exists():
        return set()
    model = load_model(path, load_external_data=False)
    locations: set[str] = set()
    for initializer in model.graph.initializer:
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        location = _external_data_map(initializer).get("location")
        if location:
            locations.add(location)
    return locations


def delete_merged_constituents(
    folder: Path,
    model_file_names: Mapping[str, str] | None = None,
    protected_names: Iterable[str] | None = None,
) -> list[str]:
    """Delete only the four split graphs absorbed by this NAR target."""
    folder = Path(folder)
    names = _model_file_names(model_file_names)
    protected = set(protected_names or ())
    protected.update((names["shared_initializers"], names["shared_initializers_data"], names["merged"]))
    removed: list[str] = []
    for role in MERGED_CONSTITUENT_KEYS:
        path = folder / names[role]
        if not path.exists():
            continue
        for location in _external_locations(path):
            external_path = folder / location
            if external_path.name not in protected and external_path.exists():
                external_path.unlink()
                removed.append(external_path.name)
        path.unlink()
        removed.append(path.name)
        sidecar = path.with_name(path.name + ".data")
        if sidecar.name not in protected and sidecar.exists():
            sidecar.unlink()
            removed.append(sidecar.name)
    return removed


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    *,
    delete_constituents: bool | None = None,
) -> dict[str, object]:
    """Build the one-session NAR graph and its mmap-able shared initializer carrier."""
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)
    names = _model_file_names(model_file_names)
    same_folder = source_folder.resolve() == out_folder.resolve()
    if delete_constituents is None:
        delete_constituents = same_folder

    merged_path = out_folder / names["merged"]
    shared_model_path = out_folder / names["shared_initializers"]
    shared_data_path = out_folder / names["shared_initializers_data"]
    for path in (merged_path, merged_path.with_name(merged_path.name + ".data"), shared_model_path, shared_data_path):
        path.unlink(missing_ok=True)

    # Remove stale split-era production files without touching the source currently being read.
    stale_removed: list[str] = []
    if not same_folder:
        stale_removed = delete_merged_constituents(
            out_folder,
            names,
            protected_names=(merged_path.name, shared_model_path.name, shared_data_path.name),
        )

    components, shared_count = _load_and_externalize_components(
        source_folder,
        names,
        shared_model_path,
        min_shared_elements,
        metadata,
    )
    try:
        merged = _compose_forced_aligner(components)
        _apply_metadata(merged, metadata)
        external_by_name = shared_external_data_map(shared_model_path)
        redirect_shared_initializers_to_external(merged, external_by_name)
        save_model(merged, merged_path)
    finally:
        components.clear()
        gc.collect()

    removed = stale_removed
    if delete_constituents:
        removed.extend(
            delete_merged_constituents(
                source_folder,
                names,
                protected_names=(merged_path.name, shared_model_path.name, shared_data_path.name),
            )
        )
    return {
        "graphs": {names["merged"]: merged_path},
        "merged_model": merged_path,
        "shared_model": shared_model_path,
        "shared_data": shared_data_path,
        "shared_initializer_count": shared_count,
        # Kept for the current exporter report; no graph reload/postflight is performed.
        "validation": {
            "shared_initializer_count": shared_count,
            "shared_data_bytes": shared_data_path.stat().st_size,
        },
        "removed_constituents": removed,
    }


def attach_shared_initializers(session_options, shared_model_path: Path):
    """mmap shared bytes, inject OrtValues, and return references that must stay alive."""
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
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        shape = tuple(int(dim) for dim in initializer.dims)
        array = np.memmap(data_path, dtype=np_dtype, mode="r", offset=offset, shape=shape)
        arrays[initializer.name] = array
        ort_value = ort.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(initializer.name, ort_value)
    return arrays, ort_values
