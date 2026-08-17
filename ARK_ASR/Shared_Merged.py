"""Build ARK-ASR merged ONNX graphs backed by one shared initializer bundle.

Each prefill graph composes the audio encoder, rotary/mask shell, decoder Main,
and selected decode head. Decode graphs compose rotary, Main, and the selected
head. Encoder and Main weight names remain stable so their large initializers are
stored once and optimized standalone donors can be transplanted back into every
strategy graph.
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

SHELL_PREFIXES = (
    "encoder_",
    "prefill_",
    "decode_",
    "greedy_",
    "penalty_greedy_",
    "penalty_",
    "sampling_",
)

PREFILL_GREEDY_MODEL_NAME = "ARK_ASR_Prefill_Greedy.onnx"
PREFILL_PENALTY_GREEDY_MODEL_NAME = "ARK_ASR_Prefill_Penalty_Greedy.onnx"
PREFILL_SAMPLING_MODEL_NAME = "ARK_ASR_PrefillSampling.onnx"
DECODE_GREEDY_MODEL_NAME = "ARK_ASR_Decode_Greedy.onnx"
DECODE_PENALTY_GREEDY_MODEL_NAME = "ARK_ASR_Decode_Penalty_Greedy.onnx"
DECODE_SAMPLING_MODEL_NAME = "ARK_ASR_DecodeSampling.onnx"
SHARED_MODEL_NAME = "ARK_ASR_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"

DEFAULT_MODEL_FILE_NAMES = {
    "metadata": "ASR_Metadata.onnx",
    "encoder": "ARK_ASR_Encoder.onnx",
    "embed": "ARK_ASR_Decoder_Embed.onnx",
    "main": "ARK_ASR_Decoder_Main.onnx",
    "rotary_prefill": "ARK_ASR_Rotary_Mask_Text_Prefill.onnx",
    "rotary_decode": "ARK_ASR_Rotary_Mask_Text_Decode.onnx",
    # Generic role -> ASR artifact.  ASR's Argmax is plain greedy; ASR's
    # Greedy_Search is the history-tracking head used after Apply_Penalty.
    "greedy": "ARK_ASR_Argmax.onnx",
    "penalty_greedy": "ARK_ASR_Greedy_Search.onnx",
    "penalty": "ARK_ASR_Apply_Penalty.onnx",
    "sampling": "ARK_ASR_TopKTopPSampling.onnx",
    "prefill_greedy": PREFILL_GREEDY_MODEL_NAME,
    "prefill_penalty_greedy": PREFILL_PENALTY_GREEDY_MODEL_NAME,
    "prefill_sampling": PREFILL_SAMPLING_MODEL_NAME,
    "decode_greedy": DECODE_GREEDY_MODEL_NAME,
    "decode_penalty_greedy": DECODE_PENALTY_GREEDY_MODEL_NAME,
    "decode_sampling": DECODE_SAMPLING_MODEL_NAME,
    "shared_initializers": SHARED_MODEL_NAME,
    "shared_initializers_data": SHARED_DATA_NAME,
}

RUNTIME_STANDALONE_MODEL_KEYS = ("metadata", "embed")
REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS = frozenset(RUNTIME_STANDALONE_MODEL_KEYS)
MERGED_CONSTITUENT_MODEL_KEYS = (
    "encoder",
    "main",
    "rotary_prefill",
    "rotary_decode",
    "greedy",
    "penalty_greedy",
    "penalty",
    "sampling",
)


def _model_file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = DEFAULT_MODEL_FILE_NAMES if model_file_names is None else model_file_names
    return names[key]


def load_model(path: Path, load_external_data: bool = True) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=load_external_data)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a data-light merged graph without creating a per-graph sidecar."""
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


def save_shared_initializers_from_tensors(shared: dict[str, TensorProto], path: Path) -> None:
    """Stream tensors into one external-data blob with near-source peak memory."""
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
            references.append(ref)

    graph = onnx.helper.make_graph([], "ark_asr_shared_initializers", [], [], initializer=references)
    model = onnx.helper.make_model(
        graph,
        producer_name="ARK_ASR/Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", 20)],
    )
    model.ir_version = 10
    model.metadata_props.add(key="ark_asr_shared_initializers", value="1")
    model.metadata_props.add(key="initializer_count", value=str(len(references)))
    onnx.save_model(model, str(path))


def shared_external_data_map(shared_model_path: Path) -> dict[str, dict[str, str]]:
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
    model: onnx.ModelProto, external_by_name: dict[str, dict[str, str]]
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

    kept_value_info = [value for value in model.graph.value_info if value.name not in remap]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = remap.get(annotation.tensor_name, annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)

    return remap


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


def extend_shared_initializers(
    shared: dict[str, TensorProto],
    model: onnx.ModelProto,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    *,
    component_label: str,
) -> int:
    """Add one component's large tensors to ``shared`` with fail-closed dedup."""
    added = 0
    for initializer in model.graph.initializer:
        if not _is_shareable_initializer(initializer, min_elements):
            continue
        existing = shared.get(initializer.name)
        if existing is None:
            shared[initializer.name] = initializer
            added += 1
        elif existing.SerializeToString() != initializer.SerializeToString():
            raise RuntimeError(
                f"{component_label} initializer {initializer.name!r} collides "
                "with a different shared tensor."
            )
    return added


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
    model: onnx.ModelProto, prefix: str
) -> onnx.ModelProto:
    """Prefix Encoder graph edges while preserving namespaced donor weights.

    Optimizers may materialize generic Constant tensors such as
    ``/Constant_1_output_0``. Those must receive the component prefix or they can
    collide with Main constants; only explicitly namespaced Encoder weights stay
    unchanged for cross-graph sharing.
    """
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
    model: onnx.ModelProto, *, marker: str, namespace: str
) -> int:
    """Namespace internal tensors matching ``marker`` throughout one graph."""
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
    model_file_names: dict[str, str] | None = None,
) -> tuple[onnx.ModelProto, str, str]:
    """Expose ARK's encoder output as the prefill hidden-state boundary."""
    if encoder is None:
        encoder = load_model(
            source_folder / _model_file_name(model_file_names, "encoder")
        )
    namespace_encoder_initializers(encoder)
    encoder_component = prefixed_preserving_initializers(encoder, "encoder_")
    encoder_hidden = f"encoder_{encoder.graph.output[0].name}"
    encoder_ids_len = f"encoder_{encoder.graph.output[1].name}"
    rename_tensor(
        encoder_component,
        f"encoder_{encoder.graph.input[0].name}",
        "audio",
    )
    rename_tensor(
        encoder_component,
        f"encoder_{encoder.graph.input[1].name}",
        "prompt_tail_embed",
    )
    return encoder_component, encoder_hidden, encoder_ids_len


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {value.name: value for value in values}


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
    """Compose models without checker/shape-inference materializing a >2 GiB proto."""
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "ARK_ASR/Shared_Merged.py"
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
    for initializer in list(first.graph.initializer) + list(second.graph.initializer):
        initializers.setdefault(initializer.name, initializer)
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


def _main_kv_output_names(main: onnx.ModelProto) -> list[str]:
    return [value.name for value in main.graph.output if value.name.startswith("present_")]


def _order_kv_inputs_first(model: onnx.ModelProto) -> None:
    kv_inputs = [value for value in model.graph.input if value.name.startswith("past_")]
    other_inputs = [value for value in model.graph.input if not value.name.startswith("past_")]
    del model.graph.input[:]
    model.graph.input.extend(kv_inputs + other_inputs)


def _merge_rotary_into_main(
    source_folder: Path,
    main: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, str, str] | None = None,
) -> tuple[onnx.ModelProto, str, onnx.ModelProto]:
    if kind == "prefill":
        rotary = prefixed(
            load_model(source_folder / _model_file_name(model_file_names, "rotary_prefill")),
            "prefill_",
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
            prefill_frontend = build_prefill_frontend(
                source_folder, model_file_names=model_file_names
            )
        frontend, hidden_name, ids_len_name = prefill_frontend
        merged = merge_models_no_check(
            frontend,
            merged,
            [
                (hidden_name, "hidden_states"),
                (ids_len_name, "prefill_ids_len"),
            ],
        )
        return merged, "prefill_kv_seq_len", rotary

    rotary = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "rotary_decode")),
        "decode_",
    )
    mask_info = next(value for value in main.graph.input if value.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(mask_info.type.tensor_type.elem_type)
    rotary.graph.initializer.append(
        numpy_helper.from_array(
            np.zeros((1, 1, 1, 1, 1), dtype=mask_dtype),
            name="decode_zero_attention_mask",
        )
    )
    merged = merge_models_no_check(
        rotary,
        main,
        [
            ("decode_rotary_cos", "rotary_cos"),
            ("decode_rotary_sin", "rotary_sin"),
            ("decode_zero_attention_mask", "attention_mask"),
        ],
    )
    return merged, "decode_kv_seq_len_next", rotary


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
    merged.producer_name = "ARK_ASR/Shared_Merged.py"
    return merged


def _merge_greedy(
    source_folder: Path,
    main: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, str, str] | None = None,
) -> onnx.ModelProto:
    merged, kv_seq_len, rotary = _merge_rotary_into_main(
        source_folder, main, kind, model_file_names, prefill_frontend
    )
    greedy = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "greedy")),
        "greedy_",
    )
    merged = merge_models_no_check(merged, greedy, [("logits", "greedy_logits")])
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main) + ["greedy_max_logits_idx", kv_seq_len],
    )


def _merge_sampling(
    source_folder: Path,
    main: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, str, str] | None = None,
) -> onnx.ModelProto:
    merged, kv_seq_len, rotary = _merge_rotary_into_main(
        source_folder, main, kind, model_file_names, prefill_frontend
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
        rotary,
        _main_kv_output_names(main)
        + ["sampling_sampled_id", "sampling_save_id_out", kv_seq_len],
    )


def _merge_penalty_greedy_prefill(
    source_folder: Path,
    main: onnx.ModelProto,
    model_file_names: dict[str, str] | None,
    prefill_frontend: tuple[onnx.ModelProto, str, str] | None = None,
) -> onnx.ModelProto:
    merged, kv_seq_len, rotary = _merge_rotary_into_main(
        source_folder, main, "prefill", model_file_names, prefill_frontend
    )
    head = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty_greedy")),
        "penalty_greedy_",
    )
    merged = merge_models_no_check(
        merged, head, [("logits", "penalty_greedy_logits")]
    )
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main)
        + [
            "penalty_greedy_max_logits_idx",
            "penalty_greedy_save_id_out",
            kv_seq_len,
        ],
    )


def merge_prefill_greedy(
    source_folder, main, model_file_names=None, prefill_frontend=None
):
    return _merge_greedy(
        source_folder, main, "prefill", model_file_names, prefill_frontend
    )


def merge_decode_greedy(
    source_folder, main, model_file_names=None, prefill_frontend=None
):
    return _merge_greedy(source_folder, main, "decode", model_file_names)


def merge_prefill_penalty_greedy(
    source_folder, main, model_file_names=None, prefill_frontend=None
):
    return _merge_penalty_greedy_prefill(
        source_folder, main, model_file_names, prefill_frontend
    )


def merge_decode_penalty_greedy(
    source_folder, main, model_file_names=None, prefill_frontend=None
):
    merged, kv_seq_len, rotary = _merge_rotary_into_main(
        source_folder, main, "decode", model_file_names
    )
    penalty = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty")),
        "penalty_",
    )
    head = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty_greedy")),
        "penalty_greedy_",
    )
    merged = merge_models_no_check(
        merged, penalty, [("logits", "penalty_logits_in")]
    )
    merged = merge_models_no_check(
        merged,
        head,
        [("penalty_logits_out", "penalty_greedy_logits")],
    )
    return _finalize(
        merged,
        main,
        rotary,
        _main_kv_output_names(main)
        + [
            "penalty_greedy_max_logits_idx",
            "penalty_greedy_save_id_out",
            kv_seq_len,
        ],
    )


def merge_prefill_sampling(
    source_folder, main, model_file_names=None, prefill_frontend=None
):
    return _merge_sampling(
        source_folder, main, "prefill", model_file_names, prefill_frontend
    )


def merge_decode_sampling(
    source_folder, main, model_file_names=None, prefill_frontend=None
):
    return _merge_sampling(source_folder, main, "decode", model_file_names)


def _recipe_with_names(recipe, model_file_names, prefill_frontend=None):
    def wrapped(source_folder, main):
        return recipe(
            source_folder,
            main,
            model_file_names,
            prefill_frontend=prefill_frontend,
        )

    wrapped.__name__ = recipe.__name__
    return wrapped


def make_merged_build_plan(
    model_file_names: dict[str, str] | None = None,
    prefill_frontend: tuple[onnx.ModelProto, str, str] | None = None,
):
    name = lambda role: _model_file_name(model_file_names, role)
    return [
        (
            name("prefill_greedy"),
            _recipe_with_names(
                merge_prefill_greedy, model_file_names, prefill_frontend
            ),
            [name("encoder"), name("rotary_prefill"), name("greedy")],
        ),
        (
            name("prefill_penalty_greedy"),
            _recipe_with_names(
                merge_prefill_penalty_greedy, model_file_names, prefill_frontend
            ),
            [name("encoder"), name("rotary_prefill"), name("penalty_greedy")],
        ),
        (
            name("prefill_sampling"),
            _recipe_with_names(
                merge_prefill_sampling, model_file_names, prefill_frontend
            ),
            [name("encoder"), name("rotary_prefill"), name("sampling")],
        ),
        (
            name("decode_greedy"),
            _recipe_with_names(merge_decode_greedy, model_file_names),
            [name("rotary_decode"), name("greedy")],
        ),
        (
            name("decode_penalty_greedy"),
            _recipe_with_names(merge_decode_penalty_greedy, model_file_names),
            [name("rotary_decode"), name("penalty"), name("penalty_greedy")],
        ),
        (
            name("decode_sampling"),
            _recipe_with_names(merge_decode_sampling, model_file_names),
            [name("rotary_decode"), name("sampling")],
        ),
    ]


MERGED_BUILD_PLAN = make_merged_build_plan()


def _external_locations(onnx_path: Path) -> set[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    locations = set()
    for initializer in model.graph.initializer:
        if initializer.data_location == TensorProto.EXTERNAL:
            location = _external_data_map(initializer).get("location")
            if location:
                locations.add(location)
    return locations


def references_shared_bundle(model_path: Path, shared_data_name: str) -> bool:
    """True if the graph reads any initializer from the shared bundle blob."""
    return shared_data_name in _external_locations(Path(model_path))


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
    legacy_names = (
        f"ARK_ASR_First_{old_family}_Search.onnx",
        f"ARK_ASR_Second_{old_family}_Search.onnx",
        f"ARK_ASR_Prefill_{old_family}.onnx",
        f"ARK_ASR_Decode_{old_family}.onnx",
        f"ARK_ASR_Decode_Penalty_{old_family}.onnx",
    )
    protected = {
        _model_file_name(model_file_names, "shared_initializers"),
        _model_file_name(model_file_names, "shared_initializers_data"),
    }
    return _delete_named_graph_artifacts(Path(folder), legacy_names, protected)


def delete_merged_constituents(
    folder: Path,
    protected_names: tuple[str, ...] | set[str] | None = None,
    model_file_names: dict[str, str] | None = None,
) -> list[str]:
    folder = Path(folder)
    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    protected = set(protected_names or (shared_name, shared_data_name))
    removed: list[str] = []
    for role in MERGED_CONSTITUENT_MODEL_KEYS:
        onnx_path = folder / _model_file_name(model_file_names, role)
        if not onnx_path.exists():
            continue
        for location in _external_locations(onnx_path):
            external_path = folder / location
            if location not in protected and external_path.exists():
                external_path.unlink()
                removed.append(external_path.name)
        onnx_path.unlink()
        removed.append(onnx_path.name)
        sidecar = onnx_path.with_name(onnx_path.name + ".data")
        if sidecar.name not in protected and sidecar.exists():
            sidecar.unlink()
            removed.append(sidecar.name)
    return removed


def _find_single_gather(model: onnx.ModelProto) -> onnx.NodeProto | None:
    gathers = [node for node in model.graph.node if node.op_type == "Gather"]
    return gathers[0] if len(gathers) == 1 else None


def dedup_embed_into_shared_lm_head(
    source_folder: Path,
    out_folder: Path,
    shared: dict[str, TensorProto],
    external_by_name: dict[str, dict[str, str]],
    model_file_names: dict[str, str] | None = None,
) -> dict | None:
    """Share the tied token-embedding table with Main's pristine ``lm_head``.

    ``lm_head`` is exported transposed to the row-major embedding table
    (``[hidden, vocab]`` vs ``[vocab, hidden]``) and, because ARK-ASR ties
    ``embed_tokens`` and ``lm_head``, it is byte-identical to ``embed_tokens.T``.
    Rewrite the standalone Embed graph so its ``Gather`` reads *columns* from the
    already-shared ``lm_head`` tensor (``gather_axis=1``) and transposes the small
    gathered result back to ``[batch, ids_len, hidden]``.  The heavy embedding
    table is then stored exactly once (inside the shared bundle), removing its
    duplicate copy from the Embed graph.  Returns ``None`` (leaving the Embed graph
    untouched) if the tie/layout cannot be verified byte-for-byte.
    """
    embed_path = source_folder / _model_file_name(model_file_names, "embed")
    if not embed_path.exists():
        return None
    embed_model = load_model(embed_path)
    gather = _find_single_gather(embed_model)
    if gather is None or gather.input[0] not in {init.name for init in embed_model.graph.initializer}:
        return None
    table_name, index_name = gather.input[0], gather.input[1]
    gather_axis = next((attr.i for attr in gather.attribute if attr.name == "axis"), 0)
    if gather_axis != 0:
        return None
    table = next(init for init in embed_model.graph.initializer if init.name == table_name)
    if len(table.dims) != 2:
        return None
    vocab, hidden = int(table.dims[0]), int(table.dims[1])

    lm_name = None
    for name, tensor in shared.items():
        if (
            len(tensor.dims) == 2
            and int(tensor.dims[0]) == hidden
            and int(tensor.dims[1]) == vocab
            and tensor.data_type == table.data_type
        ):
            lm_name = name
            lm_tensor = tensor
            break
    if lm_name is None or lm_name not in external_by_name:
        return None

    embed_array = numpy_helper.to_array(table)
    lm_array = numpy_helper.to_array(lm_tensor)
    if embed_array.shape != (vocab, hidden) or not np.array_equal(embed_array.T, lm_array):
        return None
    del embed_array, lm_array

    gathered_name = gather.output[0] + "_gathered_hbs"
    new_gather = onnx.helper.make_node(
        "Gather", [lm_name, index_name], [gathered_name],
        name="dedup_embed_gather", axis=1,
    )
    transpose = onnx.helper.make_node(
        "Transpose", [gathered_name], [gather.output[0]],
        name="dedup_embed_transpose", perm=[1, 2, 0],
    )
    new_nodes: list[onnx.NodeProto] = []
    for node in embed_model.graph.node:
        if node is gather:
            new_nodes.extend([new_gather, transpose])
        else:
            new_nodes.append(node)
    del embed_model.graph.node[:]
    embed_model.graph.node.extend(new_nodes)
    kept_initializers = [
        init for init in embed_model.graph.initializer if init.name != table_name
    ]
    del embed_model.graph.initializer[:]
    embed_model.graph.initializer.extend(kept_initializers)
    embed_model.graph.initializer.append(
        make_external_initializer_ref(lm_tensor, external_by_name[lm_name])
    )

    out_embed = out_folder / _model_file_name(model_file_names, "embed")
    save_model(embed_model, out_embed)
    return {"lm_head_name": lm_name, "path": out_embed, "vocab": vocab, "hidden": hidden}


def _embed_table_tied_to_shared(table: TensorProto, shared: dict[str, TensorProto]) -> bool:
    """True when a shared ``[hidden, vocab]`` tensor is byte-identical to ``table.T``.

    Only shape/dtype-compatible shared tensors are materialised for the comparison, so an
    untied checkpoint pays for at most one ``[hidden, vocab]`` lm_head decode.
    """
    vocab, hidden = int(table.dims[0]), int(table.dims[1])
    for tensor in shared.values():
        if (
            len(tensor.dims) == 2
            and int(tensor.dims[0]) == hidden
            and int(tensor.dims[1]) == vocab
            and tensor.data_type == table.data_type
        ):
            embed_array = numpy_helper.to_array(table)
            lm_array = numpy_helper.to_array(tensor)
            tied = embed_array.shape == (vocab, hidden) and np.array_equal(embed_array.T, lm_array)
            del embed_array, lm_array
            if tied:
                return True
    return False


def prepare_embed_table_for_sharing(
    source_folder: Path,
    shared: dict[str, TensorProto],
    min_elements: int,
    model_file_names: dict[str, str] | None = None,
) -> dict | None:
    """Fold the standalone embedding table into the shared-initializer set.

    ``dedup_embed_into_shared_lm_head`` can only remove the embedding table when it is
    tied to Main's ``lm_head``.  ARK-ASR unties them, so the heavy ``[vocab, hidden]``
    table would otherwise ship as a private ``Decoder_Embed.onnx.data`` sidecar.  When the
    table is *not* tied, register it in ``shared`` (mutated in place) so it is streamed once
    into the single mmap-friendly bundle alongside Main; the standalone Embed graph is later
    redirected to read it from there by :func:`finalize_embed_table_consolidation`.

    Returns the loaded Embed graph plus table geometry for that later redirect, or ``None``
    when the table is tied (handled by dedup), missing, unshareable, or name-colliding.
    """
    embed_path = source_folder / _model_file_name(model_file_names, "embed")
    if not embed_path.exists():
        return None
    embed_model = load_model(embed_path)
    gather = _find_single_gather(embed_model)
    if gather is None or gather.input[0] not in {init.name for init in embed_model.graph.initializer}:
        return None
    table_name = gather.input[0]
    gather_axis = next((attr.i for attr in gather.attribute if attr.name == "axis"), 0)
    table = next(init for init in embed_model.graph.initializer if init.name == table_name)
    if gather_axis != 0 or len(table.dims) != 2 or not _is_shareable_initializer(table, min_elements):
        return None
    if table_name in shared:
        # A Main initializer already owns this name; leave the Embed graph untouched.
        return None
    if _embed_table_tied_to_shared(table, shared):
        # Tie path: dedup_embed_into_shared_lm_head removes the table entirely.
        return None
    shared[table_name] = table
    return {
        "embed_model": embed_model,
        "table_name": table_name,
        "vocab": int(table.dims[0]),
        "hidden": int(table.dims[1]),
    }


def finalize_embed_table_consolidation(
    prepared: dict,
    out_folder: Path,
    external_by_name: dict[str, dict[str, str]],
    model_file_names: dict[str, str] | None = None,
) -> dict:
    """Redirect the standalone Embed graph's table to the shared bundle and save it.

    Pairs with :func:`prepare_embed_table_for_sharing`: the table is already inside the
    shared blob, so the Embed graph keeps its plain ``Gather`` but reads the ``[vocab,
    hidden]`` weight from the shared external-data file (mmapped once, next to Main).
    """
    embed_model = prepared["embed_model"]
    redirect_shared_initializers_to_external(embed_model, external_by_name)
    out_embed = out_folder / _model_file_name(model_file_names, "embed")
    save_model(embed_model, out_embed)
    return {
        "table_name": prepared["table_name"],
        "path": out_embed,
        "vocab": prepared["vocab"],
        "hidden": prepared["hidden"],
    }


def copy_runtime_standalones(
    source_folder: Path,
    target_folder: Path,
    model_file_names: dict[str, str] | None = None,
    skip_roles: tuple[str, ...] | frozenset[str] = (),
) -> list[Path]:
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)
    skip_roles = frozenset(skip_roles)
    copied: list[Path] = []
    for role in RUNTIME_STANDALONE_MODEL_KEYS:
        if role in skip_roles:
            continue
        file_name = _model_file_name(model_file_names, role)
        source = source_folder / file_name
        if not source.exists():
            if role in REQUIRED_RUNTIME_STANDALONE_MODEL_KEYS:
                raise FileNotFoundError(source)
            continue
        target = target_folder / file_name
        target.unlink(missing_ok=True)
        target.with_name(target.name + ".data").unlink(missing_ok=True)
        shutil.copy2(source, target)
        source_data = source.with_name(source.name + ".data")
        if source_data.exists():
            shutil.copy2(source_data, target.with_name(target.name + ".data"))
        copied.append(target)
    return copied


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
    dedup_embed: bool = True,
) -> dict:
    source_folder = Path(source_folder)
    out_folder = source_folder if out_folder is None else Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    delete_obsolete_strategy_artifacts(source_folder, model_file_names)
    if out_folder.resolve() != source_folder.resolve():
        delete_obsolete_strategy_artifacts(out_folder, model_file_names)

    shared_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    shared_model_path = out_folder / shared_name
    shared_data_path = out_folder / shared_data_name

    for file_name, _, _ in make_merged_build_plan(model_file_names):
        (out_folder / file_name).unlink(missing_ok=True)
        (out_folder / (file_name + ".data")).unlink(missing_ok=True)
    shared_model_path.unlink(missing_ok=True)
    shared_data_path.unlink(missing_ok=True)

    main_for_merge, shared = load_main_with_shared_initializers(
        source_folder, min_shared_elements, model_file_names
    )
    encoder_path = source_folder / _model_file_name(model_file_names, "encoder")
    encoder_for_merge = load_model(encoder_path)
    namespace_encoder_initializers(encoder_for_merge)
    extend_shared_initializers(
        shared,
        encoder_for_merge,
        min_shared_elements,
        component_label=encoder_path.name,
    )
    for rotary_role, prefix in (
        ("rotary_prefill", "prefill_"),
        ("rotary_decode", "decode_"),
    ):
        rotary_path = source_folder / _model_file_name(model_file_names, rotary_role)
        rotary_for_merge = prefixed(load_model(rotary_path), prefix)
        extend_shared_initializers(
            shared,
            rotary_for_merge,
            min_shared_elements,
            component_label=rotary_path.name,
        )
    # Bring the standalone token-embedding table into the single shared bundle.  When it is
    # tied to Main's lm_head the table is dropped entirely (the Embed graph reads lm_head);
    # otherwise (ARK-ASR unties them) it is registered here so the heavy [vocab, hidden]
    # weight streams once into the shared blob instead of a private Decoder_Embed.onnx.data
    # sidecar.  Prepared before the bundle is written so the table lands inside it.
    embed_prepared = None
    if dedup_embed:
        embed_prepared = prepare_embed_table_for_sharing(
            source_folder, shared, min_shared_elements, model_file_names
        )
    save_shared_initializers_from_tensors(shared, shared_model_path)
    external_by_name = shared_external_data_map(shared_model_path)
    embed_dedup = None
    embed_consolidated = None
    if dedup_embed:
        if embed_prepared is not None:
            # Untied table already streamed into the bundle: redirect the Embed graph to it.
            embed_consolidated = finalize_embed_table_consolidation(
                embed_prepared, out_folder, external_by_name, model_file_names
            )
        else:
            # Tied table (or no eligible Embed graph): share Main's pristine lm_head instead.
            embed_dedup = dedup_embed_into_shared_lm_head(
                source_folder, out_folder, shared, external_by_name, model_file_names
            )
    del shared
    redirect_shared_initializers_to_external(main_for_merge, external_by_name)
    redirect_shared_initializers_to_external(encoder_for_merge, external_by_name)
    # Keep lightweight standalone donors for Optimize_ONNX.py. Runtime prefill
    # uses neither graph separately; both donors point at the same shared blob.
    standalone_main = out_folder / _model_file_name(model_file_names, "main")
    standalone_encoder = out_folder / _model_file_name(model_file_names, "encoder")
    save_model(main_for_merge, standalone_main)
    save_model(encoder_for_merge, standalone_encoder)
    prefill_frontend = build_prefill_frontend(
        source_folder,
        encoder=encoder_for_merge,
        model_file_names=model_file_names,
    )
    build_plan = make_merged_build_plan(
        model_file_names, prefill_frontend=prefill_frontend
    )

    graphs: dict[str, Path] = {}
    for file_name, recipe, _ in build_plan:
        merged = recipe(source_folder, main_for_merge)
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
        "embed_dedup": embed_dedup,
        "embed_consolidated": embed_consolidated,
    }
    if out_folder.resolve() == source_folder.resolve():
        result["removed_constituents"] = delete_merged_constituents(
            source_folder,
            protected_names=(shared_name, shared_data_name),
            model_file_names=model_file_names,
        )
    return result


# ---------------------------------------------------------------------------
# Quantized-Main transplant: quantize one merged graph, reuse it in all shells.
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


def _used_inputs(nodes: list[onnx.NodeProto]) -> set[str]:
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
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.type.HasField("tensor_type"):
            elem_type = int(value.type.tensor_type.elem_type)
            if elem_type != TensorProto.UNDEFINED:
                types[value.name] = elem_type
    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.UNDEFINED:
            types[initializer.name] = int(initializer.data_type)

    same_as_first_input = {
        "Abs", "Add", "Clip", "Concat", "Div", "Dropout", "Exp", "Expand",
        "Gather", "GatherElements", "Gelu", "Identity", "LayerNormalization",
        "Log", "MatMul", "Max", "Min", "Mul", "Neg", "Pad", "Pow", "QuickGelu",
        "ReduceMax", "ReduceMean", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape", "ScatterElements",
        "Sigmoid", "SimplifiedLayerNormalization", "Slice", "Softmax", "Split", "Sqrt",
        "Squeeze", "Sub", "Tanh", "Transpose", "Unsqueeze", "Where",
    }
    changed = True
    while changed:
        changed = False
        for node in model.graph.node:
            inferred = None
            if node.op_type == "Cast":
                inferred = next(
                    (attribute.i for attribute in node.attribute if attribute.name == "to"),
                    None,
                )
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
            elif node.op_type in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Not"):
                inferred = TensorProto.BOOL
            elif node.op_type in same_as_first_input and node.input:
                inferred = types.get(node.input[0])
                if inferred is None and node.op_type == "MatMul" and len(node.input) > 1:
                    inferred = types.get(node.input[1])
            if inferred is None or inferred == TensorProto.UNDEFINED:
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


def _target_rotary_remap(
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
        target_main_nodes = [node for node in target.graph.node if not _node_is_shell(node)]
        target_main_inputs = _used_inputs(target_main_nodes)
        encoder_outputs = {
            output
            for node in target.graph.node
            for output in node.output
            if output.startswith("encoder_")
        }
        hidden_candidates = sorted(encoder_outputs & target_main_inputs)
        if len(hidden_candidates) != 1:
            raise RuntimeError(
                "Cannot identify the encoder-prefill hidden-state boundary: "
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
            }
        )
        return remap
    if "rotary_cos" in donor_inputs:
        prefix = "decode" if target_is_decode else "prefill"
        remap.update(
            {
                "rotary_cos": f"{prefix}_rotary_cos",
                "rotary_sin": f"{prefix}_rotary_sin",
                "attention_mask": (
                    "decode_zero_attention_mask"
                    if target_is_decode
                    else "prefill_attention_mask"
                ),
            }
        )
    return remap


def transplant_optimized_main(
    target: onnx.ModelProto, optimized_primary: onnx.ModelProto
) -> onnx.ModelProto:
    donor_main_nodes = [
        node for node in optimized_primary.graph.node if not _node_is_shell(node)
    ]
    remap = _target_rotary_remap(target, donor_main_nodes)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in donor_main_nodes
    ]
    if not primary_main_nodes:
        raise RuntimeError("Quantized primary graph has no Main node block to transplant.")

    primary_types = _remap_element_types(_tensor_element_types(optimized_primary), remap)
    target_types = _tensor_element_types(target)
    main_outputs = {
        output for node in primary_main_nodes for output in node.output if output
    }
    target_shell_nodes = [node for node in target.graph.node if _node_is_shell(node)]
    shell_outputs = {
        output for node in target_shell_nodes for output in node.output if output
    }

    reserved_names = {value.name for value in target.graph.input}
    reserved_names.update(initializer.name for initializer in target.graph.initializer)
    reserved_names.update(
        output for node in target.graph.node for output in node.output if output
    )
    reserved_names.update(main_outputs)

    main_input_remap: dict[str, str] = {}
    pre_main_casts: list[onnx.NodeProto] = []
    main_external_inputs = _used_inputs(primary_main_nodes) - main_outputs
    for name in sorted(main_external_inputs & shell_outputs):
        source_type = target_types.get(name)
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
        for initializer in optimized_primary.graph.initializer
    }
    target_initializers = {
        initializer.name: initializer for initializer in target.graph.initializer
    }
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

    def add_initializer(initializer: TensorProto) -> None:
        if initializer.name not in seen:
            new_initializers.append(copy.deepcopy(initializer))
            seen.add(initializer.name)

    for initializer in target.graph.initializer:
        if (
            initializer.name in required_initializers
            and initializer.name not in main_initializer_names
        ):
            add_initializer(initializer)
    for initializer in optimized_primary.graph.initializer:
        if initializer.name in main_initializer_names:
            add_initializer(initializer)
    for name in sorted(required_initializers):
        if name not in seen and name in target_initializers:
            add_initializer(target_initializers[name])
        if (
            name not in seen
            and name in main_initializer_names
            and name in primary_initializers
        ):
            add_initializer(primary_initializers[name])

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

    for value_info in optimized_primary.graph.value_info:
        name = remap.get(value_info.name, value_info.name)
        add_value_info(value_info, main_input_remap.get(name, name))
    for value_info in target.graph.value_info:
        if value_info.name.startswith(SHELL_PREFIXES):
            add_value_info(value_info)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(value_infos)
    _merge_opsets(merged, optimized_primary)
    prune_unreachable_nodes(merged)
    _order_kv_inputs_first(merged)
    return merged


# Backward-compatible name for sibling scripts that still use quantized donors.
transplant_quantized_main = transplant_optimized_main


def _node_is_encoder_component(node: onnx.NodeProto) -> bool:
    return any(output.startswith("encoder_") for output in node.output if output)


def transplant_optimized_encoder(
    target: onnx.ModelProto, optimized_encoder: onnx.ModelProto
) -> onnx.ModelProto:
    """Replace a prefill template's raw Encoder with an optimized donor.

    Decode templates contain no ``encoder_`` component and are returned unchanged.
    Encoder initializers keep their standalone names so they can share one bundle
    with Main without duplicating weights in each prefill strategy.
    """
    target_encoder_nodes = [
        node for node in target.graph.node if _node_is_encoder_component(node)
    ]
    if not target_encoder_nodes:
        return target
    if len(optimized_encoder.graph.input) != 2 or not optimized_encoder.graph.output:
        raise RuntimeError(
            "Optimized ARK-ASR Encoder must expose audio/prompt-tail inputs and hidden output."
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
        f"encoder_{optimized_encoder.graph.input[0].name}": "audio",
        f"encoder_{optimized_encoder.graph.input[1].name}": "prompt_tail_embed",
    }
    donor_nodes = [
        _copy_node_with_input_remap(node, input_remap)
        for node in donor.graph.node
    ]
    required_boundary = {
        output
        for node in target_encoder_nodes
        for output in node.output
        if output and any(
            output in consumer.input
            for consumer in target.graph.node
            if not _node_is_encoder_component(consumer)
        )
    }
    donor_outputs = {
        output for node in donor_nodes for output in node.output if output
    }
    missing_boundary = required_boundary - donor_outputs
    if missing_boundary:
        raise RuntimeError(
            "Optimized Encoder does not reproduce prefill boundary tensor(s): "
            f"{sorted(missing_boundary)}"
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
        if not value.name.startswith("encoder_")
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
    models: dict[str, onnx.ModelProto] | list[onnx.ModelProto],
    shared_model_path: Path,
    primary_model: onnx.ModelProto | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
) -> dict[str, dict[str, str]]:
    values = list(models.values()) if isinstance(models, dict) else list(models)
    if not values:
        raise RuntimeError("No merged models were supplied for shared extraction.")
    source = values[0] if primary_model is None else primary_model
    main_nodes = [node for node in source.graph.node if not _node_is_shell(node)]
    encoder_nodes = [
        node for node in source.graph.node if _node_is_encoder_component(node)
    ]
    shared_inputs = _used_inputs(main_nodes + encoder_nodes)
    shared = {
        initializer.name: initializer
        for initializer in source.graph.initializer
        if initializer.name in shared_inputs
        and _is_shareable_initializer(initializer, min_shared_elements)
    }
    if not shared:
        raise RuntimeError(
            "Optimized Encoder/Main components have no shareable initializer."
        )

    save_shared_initializers_from_tensors(shared, shared_model_path)
    del shared
    external_by_name = shared_external_data_map(shared_model_path)
    for model in values:
        redirect_shared_initializers_to_external(model, external_by_name)
    return external_by_name


# ---------------------------------------------------------------------------
# Runtime attachment: mmap once per process and keep returned references alive.
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
        dtype = onnx.helper.tensor_dtype_to_np_dtype(initializer.data_type)
        shape = tuple(int(dim) for dim in initializer.dims)
        array = np.memmap(data_path, dtype=dtype, mode="r", offset=offset, shape=shape)
        value = ort.OrtValue.ortvalue_from_numpy(array)
        arrays[initializer.name] = array
        ort_values.append(value)
        session_options.add_initializer(initializer.name, value)
    return arrays, ort_values
