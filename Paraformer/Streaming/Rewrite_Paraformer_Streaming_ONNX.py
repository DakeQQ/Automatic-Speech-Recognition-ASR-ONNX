"""Strictly hoist and deduplicate tensor-valued Constant nodes in raw Paraformer graphs.

The legacy TorchScript ONNX exporter emits hundreds of repeated Constant nodes for identical
Slice controls and Split sizes. PyTorch cannot register those exporter-generated controls as
shared module buffers. This narrow rewrite changes only standard-domain Constant(value=Tensor)
nodes into shared graph initializers; every compute node and graph interface remains unchanged.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import onnx
from onnx import AttributeProto, TensorProto


_MODEL_FILES = (
    "Paraformer_Streaming_Encoder.onnx",
    "Paraformer_Streaming_Decoder.onnx",
)
_COPY_FILES = ("ASR_Metadata.onnx", "Vocab_Paraformer.txt")
def _tensor_key(tensor: TensorProto) -> bytes:
    value = TensorProto()
    value.CopyFrom(tensor)
    value.ClearField("name")
    value.ClearField("doc_string")
    return value.SerializeToString(deterministic=True)

def rewrite_constant_nodes(raw_model_path: str | Path, final_model_path: str | Path) -> dict[str, int]:
    """Write a constant-hoisted copy of ``raw_model_path`` to ``final_model_path``."""

    raw_path = Path(raw_model_path).expanduser().resolve()
    final_path = Path(final_model_path).expanduser().resolve()
    model = onnx.load(str(raw_path), load_external_data=False)

    constants = []
    retained_nodes = []
    for node in model.graph.node:
        is_tensor_constant = (
            node.op_type == "Constant"
            and node.domain in ("", "ai.onnx")
            and not node.input
            and len(node.output) == 1
            and bool(node.output[0])
            and len(node.attribute) == 1
            and node.attribute[0].name == "value"
            and node.attribute[0].type == AttributeProto.TENSOR
        )
        if not is_tensor_constant:
            retained_nodes.append(node)
            continue
        constants.append(node)

    canonical_by_value: dict[bytes, str] = {}
    aliases: dict[str, str] = {}
    hoisted = []
    for node in constants:
        tensor = node.attribute[0].t
        key = _tensor_key(tensor)
        output_name = node.output[0]
        canonical_name = canonical_by_value.get(key)
        if canonical_name is None:
            canonical_name = output_name
            canonical_by_value[key] = canonical_name
            initializer = TensorProto()
            initializer.CopyFrom(tensor)
            initializer.name = canonical_name
            hoisted.append(initializer)
        else:
            aliases[output_name] = canonical_name
    rewired_inputs = 0
    for node in retained_nodes:
        for index, input_name in enumerate(node.input):
            canonical_name = aliases.get(input_name)
            if canonical_name is not None:
                node.input[index] = canonical_name
                rewired_inputs += 1
    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    model.graph.initializer.extend(hoisted)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.with_name(f".{final_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        onnx.save_model(model, str(temporary_path), save_as_external_data=False)
        os.replace(temporary_path, final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return {
        "raw_nodes": len(retained_nodes) + len(constants),
        "final_nodes": len(retained_nodes),
        "constant_nodes_removed": len(constants),
        "unique_initializers_added": len(hoisted),
        "duplicate_constants_merged": len(aliases),
        "consumer_inputs_rewired": rewired_inputs,
    }


def rewrite_folder(raw_folder: str | Path, final_folder: str | Path) -> list[tuple[str, dict[str, int]]]:
    raw_root = Path(raw_folder).expanduser().resolve()
    final_root = Path(final_folder).expanduser().resolve()
    final_root.mkdir(parents=True, exist_ok=True)

    staging_root = final_root / f".paraformer-rewrite-{uuid.uuid4().hex}"
    staging_root.mkdir()
    try:
        reports = []
        for name in _MODEL_FILES:
            reports.append((name, rewrite_constant_nodes(raw_root / name, staging_root / name)))
        for name in _COPY_FILES:
            source = raw_root / name
            shutil.copy2(source, staging_root / name)
        for name in (*_MODEL_FILES, *_COPY_FILES):
            os.replace(staging_root / name, final_root / name)
        return reports
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)

