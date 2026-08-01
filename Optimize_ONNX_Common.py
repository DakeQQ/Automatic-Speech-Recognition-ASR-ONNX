"""Shared, fail-closed ONNX optimization pipeline for every ASR exporter.

Model scripts should contain only their user configuration, topology selectors,
and model-specific repair/transplant hooks. Quantization, generic optimization,
artifact copying, and plan validation belong here so all scripts use the same
option semantics and stage ordering.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import onnx
import onnx.version_converter
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, matmul_nbits_quantizer, quant_utils, quantize_dynamic
from onnxslim import slim


NodeSelector = list[str] | Callable[[str], list[str] | None] | None
IntValue = int | Callable[[str], int]

_WEIGHT_ONLY_BITS = {"Q2": 2, "Q4": 4, "Q8": 8}
_QUANT_FORMATS = {
    "QOPERATOR": quant_utils.QuantFormat.QOperator,
    "QDQ": quant_utils.QuantFormat.QDQ,
}
_DYNAMIC_WEIGHT_TYPES = {"QUINT8": QuantType.QUInt8, "QINT8": QuantType.QInt8}
_WEIGHT_ONLY_ALGO_BITS = {
    "DEFAULT": frozenset(_WEIGHT_ONLY_BITS.values()),
    "HQQ": frozenset(_WEIGHT_ONLY_BITS.values()),
    "AFFINE_REFINE_V2": frozenset({4, 8}),
    "RTN": frozenset({4}),
    "k_quant": frozenset({4}),
}
_VALID_ALGOS = frozenset(_WEIGHT_ONLY_ALGO_BITS)
ASR_METADATA_ARTIFACT = "ASR_Metadata.onnx"
QUANTIZED_METHODS = frozenset((*_WEIGHT_ONLY_BITS, "DYNAMIC"))

# Keep quantization/contrib operators in F32 when a graph is converted to F16.
# ``Range`` is also blocked by the broad default because many models use it for
# shape/index construction. Models with a validated literal-F16 ABI may use the
# narrower QUANTIZATION_F16_OP_BLOCK_LIST instead.
QUANTIZATION_F16_OP_BLOCK_LIST = (
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
)
DEFAULT_F16_OP_BLOCK_LIST = (*QUANTIZATION_F16_OP_BLOCK_LIST, "Range")

# Configuration quick guide for every model script:
#   method="F32"      - optimize only; safest numerical baseline.
#   method="F16"      - mixed precision; keep I/O types for separately composed
#                       graphs and protect known overflow/residual boundaries.
#   method="DYNAMIC"  - dynamic INT8 constant weights; portable CPU default.
#                       Set algo="AFFINE_REFINE_V2" for refined QInt8/QUInt8.
#   method="Q4"       - best-supported low-bit block path. DEFAULT is portable;
#                       k_quant/RTN are specialized Q4 MatMul-only algorithms;
#                       AFFINE_REFINE_V2 supports MatMul and portable Gather.
#   method="Q2"       - use DEFAULT/HQQ; unsupported k_quant/RTN plans fall back
#                       to DEFAULT before any output file is touched.
#   method="Q8"       - DEFAULT or AFFINE_REFINE_V2.
#   Gather weights    - DEFAULT or AFFINE_REFINE_V2 with gather axis 0 and
#                       quantization on the final, block-divisible axis.
#   optimize=False    - preserve a donor/shell tensor-name ABI; F16 conversion
#                       still runs the required conversion pipeline.
#   transformer=False - disable Python transformer-pattern fusions for a graph
#                       that is not a transformer or has a strict transplant ABI.
# Always retain model-specific node exclusions and post-pass repairs. They guard
# validated numerical/provider bugs and are not generic cleanup candidates.


@dataclass
class Plan:
    """Per-module recipe; ``None`` fields inherit :class:`OptimizerConfig`.

        Method guide:
            * ``F32``: retain float32 weights and run configured graph cleanup.
            * ``F16``: convert floating weights/activations; protect sensitive nodes
                with ``nodes_to_exclude`` and/or the global F16 block lists.
            * ``DYNAMIC``: dynamic INT8 weight quantization, generally the most
                portable quantized choice for CPU execution providers.
            * ``Q2``/``Q4``/``Q8``: block weight-only quantization. ``DEFAULT`` is the
                portable algorithm; ``k_quant`` and ``RTN`` are specialized Q4 paths.

        ``optimize=False`` skips generic cleanup except when F16 conversion itself
        requires it. ``transformer=False`` skips Python transformer fusions while
        retaining onnxslim cleanup. Both are important ABI guards for merged donors.
    """

    method: str = "Q8"  # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32
    process: bool = True
    # weight-only (Q2/Q4/Q8)
    algo: str | None = None
    op_types: tuple[str, ...] | None = None
    axes: tuple[int, ...] | None = None
    block_size: int | None = None
    accuracy_level: int | None = None
    symmetric: bool | None = None
    quant_format: str | None = None
    # dynamic INT8
    dynamic_weight_type: str | None = None
    per_channel: bool | None = None
    reduce_range: bool | None = None
    default_tensor_type: int | None = None
    # node selection
    nodes_to_exclude: NodeSelector = None
    nodes_to_include: NodeSelector = None
    # optimize / precision
    optimize: bool = True
    transformer: bool = True
    opt_level: int | None = None
    fp16: bool = False
    f16_force_initializers: bool | None = None
    num_heads: IntValue = 0
    hidden_size: IntValue = 0
    # storage
    external: bool | None = None
    # onnxslim shape inference knobs
    first_slim_no_shape_infer: bool = True
    run_second_slim: bool = True
    second_slim_no_shape_infer: bool | None = None


@dataclass
class OptimizerConfig:
    """Global defaults shared by every module in one ``Optimize_ONNX.py``.

    Prefer per-model exceptions in :class:`Plan`. The field
    ``optimizer_only_onnxruntime`` is unrelated to OpenVINO: when true, only
    ORT's generic optimizer runs and Python transformer fusions are bypassed.
    """

    original_folder_path: str
    optimized_folder_path: str
    model_plans: dict[str, Plan]
    # weight-only defaults
    weight_only_algorithm: str = "k_quant"
    block_size: int = 64
    accuracy_level: int = 4
    quant_symmetric: bool = True
    quant_format: str = "QOperator"
    affine_v2_seed_iterations: int = 4
    affine_v2_seed_zp_radius: int = 2
    affine_v2_iterations: int = 6
    affine_v2_clip_ratios: tuple[float, ...] = (1.0, 0.94, 0.82, 0.70, 0.55)
    affine_v2_chunk_blocks: int = 8192
    affine_v2_weighted_tolerance: float = 0.15
    affine_v2_asym_zp_sweep_limit: int = 32
    affine_v2_numba_threads: int = 4
    # dynamic INT8 defaults
    dynamic_weight_type: str = "QInt8"
    dynamic_per_channel: bool = True
    dynamic_reduce_range: bool = False
    dynamic_default_tensor_type: int | None = TensorProto.FLOAT
    # node selection defaults
    nodes_to_exclude: NodeSelector = None
    nodes_to_include: NodeSelector = None
    # storage / opset
    force_external_data: bool = False
    upgrade_opset: int = 0
    # graph optimizer
    optimizer_level: int = 2
    optimizer_model_type: str = "bert"
    optimizer_only_onnxruntime: bool = False
    optimizer_fusion_options: dict | None = None
    shape_infer: bool = True
    # onnxslim
    slim_skip_fusion_patterns: list[str] | None = None
    slim_skip_optimizations: list[str] | None = None
    slim_size_threshold: int | None = None
    second_slim_no_shape_infer: bool | None = None
    safe_reshape_fusion: bool = True
    # float16
    f16_keep_io_types: bool | None = None
    f16_force_initializers: bool = False
    f16_min_positive_val: float = 1e-7
    f16_max_finite_val: float = 32767.0
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None
    # optional side artifacts copied after all models are processed
    copy_artifacts: tuple[str, ...] = ()
    metadata_artifact: str = ASR_METADATA_ARTIFACT


@dataclass
class ResolvedPlan:
    method: str
    process: bool
    algo: str
    op_types: tuple[str, ...]
    axes: tuple[int, ...]
    block_size: int
    accuracy_level: int
    symmetric: bool
    quant_format: str
    dynamic_weight_type: str
    per_channel: bool
    reduce_range: bool
    default_tensor_type: int | None
    nodes_to_exclude: NodeSelector
    nodes_to_include: NodeSelector
    optimize: bool
    transformer: bool
    opt_level: int | None
    fp16: bool
    f16_force_initializers: bool
    num_heads: IntValue
    hidden_size: IntValue
    external: bool
    first_slim_no_shape_infer: bool
    run_second_slim: bool
    second_slim_no_shape_infer: bool | None

    @property
    def uses_float16(self) -> bool:
        return self.fp16 or self.method == "F16"


def _pick(value, default):
    return default if value is None else value


def plan_uses_float16(plan: Plan | ResolvedPlan) -> bool:
    """Return whether a configured or resolved plan converts to float16."""
    return plan.fp16 or plan.method.upper() == "F16"


def method_is_quantized(method: str) -> bool:
    return method.upper() in QUANTIZED_METHODS


def _normalize_algorithm(value: str) -> str:
    normalized = value.strip()
    return "k_quant" if normalized.casefold() == "k_quant" else normalized.upper()


def resolve_plan(
    plan: Plan,
    config: OptimizerConfig,
    model_name: str | None = None,
) -> ResolvedPlan:
    method = plan.method.strip().upper()
    algorithm = _normalize_algorithm(_pick(plan.algo, config.weight_only_algorithm))
    op_types = _pick(plan.op_types, ("MatMul",))
    is_embed = model_name is not None and "embed" in model_name.casefold()
    if algorithm != "AFFINE_REFINE_V2" and (
        method == "Q8" or "Gather" in op_types or (
            method in _WEIGHT_ONLY_BITS and is_embed
        )
    ):
        algorithm = "DEFAULT"
    elif method == "Q2" and algorithm in {"k_quant", "RTN"}:
        algorithm = "DEFAULT"
    return ResolvedPlan(
        method=method,
        process=plan.process,
        algo=algorithm,
        op_types=op_types,
        axes=_pick(plan.axes, (0,)),
        block_size=_pick(plan.block_size, config.block_size),
        accuracy_level=_pick(plan.accuracy_level, config.accuracy_level),
        symmetric=_pick(plan.symmetric, config.quant_symmetric),
        quant_format=_pick(plan.quant_format, config.quant_format).upper(),
        dynamic_weight_type=_pick(plan.dynamic_weight_type, config.dynamic_weight_type).upper(),
        per_channel=_pick(plan.per_channel, config.dynamic_per_channel),
        reduce_range=_pick(plan.reduce_range, config.dynamic_reduce_range),
        default_tensor_type=_pick(plan.default_tensor_type, config.dynamic_default_tensor_type),
        nodes_to_exclude=_pick(plan.nodes_to_exclude, config.nodes_to_exclude),
        nodes_to_include=_pick(plan.nodes_to_include, config.nodes_to_include),
        optimize=plan.optimize,
        transformer=plan.transformer,
        opt_level=plan.opt_level,
        fp16=plan.fp16,
        f16_force_initializers=_pick(
            plan.f16_force_initializers,
            config.f16_force_initializers,
        ),
        num_heads=plan.num_heads,
        hidden_size=plan.hidden_size,
        external=_pick(plan.external, config.force_external_data),
        first_slim_no_shape_infer=plan.first_slim_no_shape_infer,
        run_second_slim=plan.run_second_slim,
        second_slim_no_shape_infer=_pick(plan.second_slim_no_shape_infer, config.second_slim_no_shape_infer),
    )


def resolve_plans(
    config: OptimizerConfig,
    model_names: Sequence[str] | None = None,
) -> dict[str, ResolvedPlan]:
    """Resolve selected plans once."""
    names = tuple(
        name
        for name in (config.model_plans if model_names is None else model_names)
        if config.model_plans[name].process
    )
    resolved = {
        name: resolve_plan(config.model_plans[name], config, model_name=name)
        for name in names
    }
    for name, plan in resolved.items():
        validate_plan(name, plan)
    return resolved


def validate_plan(name: str, plan: ResolvedPlan) -> None:
    valid_methods = set(_WEIGHT_ONLY_BITS) | {"DYNAMIC", "F16", "F32"}
    if plan.method not in valid_methods:
        raise ValueError(f"[{name}] unknown method {plan.method!r}; choose one of {sorted(valid_methods)}.")
    if plan.method in _WEIGHT_ONLY_BITS:
        bits = _WEIGHT_ONLY_BITS[plan.method]
        if plan.algo not in _VALID_ALGOS:
            raise ValueError(f"[{name}] unknown algo {plan.algo!r}; choose one of {sorted(_VALID_ALGOS)}.")
        if bits not in _WEIGHT_ONLY_ALGO_BITS[plan.algo]:
            compatible = sorted(
                algo for algo, supported_bits in _WEIGHT_ONLY_ALGO_BITS.items()
                if bits in supported_bits
            )
            raise ValueError(
                f"[{name}] algo={plan.algo!r} cannot produce {bits}-bit weights; "
                f"use one of {compatible} for method={plan.method!r}."
            )
        if plan.quant_format not in _QUANT_FORMATS:
            raise ValueError(f"[{name}] unknown quant_format; choose 'QOperator' or 'QDQ'.")
        if len(plan.op_types) != len(plan.axes):
            raise ValueError(f"[{name}] op_types {plan.op_types} and axes {plan.axes} must have equal length.")
        if "Gather" in plan.op_types and plan.algo not in {"DEFAULT", "AFFINE_REFINE_V2"}:
            raise ValueError(f"[{name}] Gather quantization does not support algo={plan.algo!r}.")
        if plan.quant_format == "QDQ" and (plan.algo != "DEFAULT" or bits != 4):
            raise ValueError(f"[{name}] QDQ supports only DEFAULT Q4 quantization.")
    if plan.method == "DYNAMIC":
        if plan.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
            raise ValueError(f"[{name}] unknown dynamic_weight_type {plan.dynamic_weight_type!r}.")
        if plan.algo == "AFFINE_REFINE_V2" and any(op != "MatMul" for op in plan.op_types):
            raise ValueError(f"[{name}] AFFINE_REFINE_V2 dynamic quantization supports MatMul only.")
        if plan.algo == "AFFINE_REFINE_V2" and plan.reduce_range:
            raise ValueError(f"[{name}] AFFINE_REFINE_V2 dynamic quantization does not support reduce_range.")


def model_exceeds_2gb(model_path: str) -> bool:
    """Return whether materializing this model inline would cross 2 GiB.

    External tensors may reference a shared blob with any declared ``location``;
    checking only the conventional ``<model>.onnx.data`` path misses those
    payloads and can produce an unparseable oversized protobuf. Count each
    referenced tensor because loading external data and saving inline serializes
    one raw payload per TensorProto, even when multiple tensors alias one range.
    """
    limit = 2 * 1024**3
    source_path = Path(model_path)
    total = source_path.stat().st_size
    if total >= limit:
        return True

    model = onnx.load(str(source_path), load_external_data=False)
    try:
        _recover_zero_length_external_ranges(model, source_path)
        for tensor in _iter_all_data_tensors(model.graph):
            if tensor.data_location != TensorProto.EXTERNAL:
                continue
            fields = {entry.key: entry.value for entry in tensor.external_data}
            declared_length = fields.get("length")
            length = (
                int(declared_length)
                if declared_length not in (None, "", "0")
                else _tensor_raw_data_length(tensor)
            )
            total += length
            if total >= limit:
                return True
        return False
    finally:
        del model
        gc.collect()


def model_size_mb(model_path: str) -> float:
    total = os.path.getsize(model_path)
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total / (1024 * 1024)


def remove_model_files(model_path: str | Path) -> None:
    """Remove one ONNX proto and its conventional private sidecar, if present."""
    model_path = Path(model_path)
    model_path.unlink(missing_ok=True)
    model_path.with_name(model_path.name + ".data").unlink(missing_ok=True)


# Backward-compatible internal name.
_remove_external_files = remove_model_files


def _save_model(model, model_path: str, external: bool) -> None:
    _remove_external_files(model_path)
    if external:
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(model_path) + ".data",
            size_threshold=1024,
            convert_attribute=True,
        )
    else:
        onnx.save(model, model_path)


def _iter_all_data_tensors(graph):
    yield from graph.initializer
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                yield attr.t
            yield from attr.tensors
            if attr.HasField("g"):
                yield from _iter_all_data_tensors(attr.g)
            for subgraph in attr.graphs:
                yield from _iter_all_data_tensors(subgraph)


def _node_int_attribute(node: onnx.NodeProto, name: str) -> int | None:
    return next((int(attr.i) for attr in node.attribute if attr.name == name), None)


_FIRST_INPUT_TYPE_OPS = frozenset(
    {
        "Abs", "Attention", "AveragePool", "BatchNormalization", "Ceil",
        "Clip", "Compress", "Conv", "ConvTranspose", "CumSum", "Dropout",
        "Elu", "Erf", "Exp", "Expand", "Flatten", "Floor", "Gather",
        "GatherElements", "GatherND", "Gelu",
        "GlobalAveragePool", "GlobalMaxPool", "GridSample", "GroupQueryAttention",
        "HardSigmoid", "HardSwish", "Identity", "LayerNormalization",
        "LeakyRelu", "Log", "LogSoftmax", "LpNormalization", "MatMulNBits",
        "MaxPool", "Neg", "Pad", "Pow", "PRelu", "Reciprocal", "ReduceL1",
        "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean",
        "ReduceMin", "ReduceProd", "ReduceSum", "ReduceSumSquare", "Relu",
        "Reshape", "Resize", "RoiAlign", "Round", "ScatterElements", "ScatterND",
        "Selu", "Shrink", "Sigmoid", "Sign", "SimplifiedLayerNormalization",
        "Sin", "Slice", "Softmax", "Softplus", "Softsign", "Split", "Sqrt",
        "Squeeze", "Swish", "Tan", "Tanh", "ThresholdedRelu", "Tile",
        "Transpose", "Trilu", "Unsqueeze",
    }
)
_SAME_TYPE_INPUT_OPS = frozenset(
    {"Add", "Concat", "Div", "Einsum", "Gemm", "MatMul", "Max", "Min", "Mod", "Mul", "Sub", "Sum"}
)
_INDEX_TYPE_FLEXIBLE_OPS = frozenset(
    {"Gather", "GatherElements", "GatherND", "ScatterElements", "ScatterND"}
)
_EXACT_WIDENING_CASTS = frozenset(
    {
        (TensorProto.FLOAT16, TensorProto.FLOAT),
        (TensorProto.FLOAT16, TensorProto.DOUBLE),
        (TensorProto.BFLOAT16, TensorProto.FLOAT),
        (TensorProto.BFLOAT16, TensorProto.DOUBLE),
        (TensorProto.FLOAT, TensorProto.DOUBLE),
        (TensorProto.INT8, TensorProto.INT16),
        (TensorProto.INT8, TensorProto.INT32),
        (TensorProto.INT8, TensorProto.INT64),
        (TensorProto.INT16, TensorProto.INT32),
        (TensorProto.INT16, TensorProto.INT64),
        (TensorProto.INT32, TensorProto.INT64),
        (TensorProto.UINT8, TensorProto.UINT16),
        (TensorProto.UINT8, TensorProto.UINT32),
        (TensorProto.UINT8, TensorProto.UINT64),
        (TensorProto.UINT16, TensorProto.UINT32),
        (TensorProto.UINT16, TensorProto.UINT64),
        (TensorProto.UINT32, TensorProto.UINT64),
    }
)


def _constant_node_type(node: onnx.NodeProto) -> int | None:
    tensor = next(
        (
            attr.t
            for attr in node.attribute
            if attr.name == "value" and attr.HasField("t")
        ),
        None,
    )
    if tensor is not None:
        return int(tensor.data_type)
    scalar_types = {
        "value_float": TensorProto.FLOAT,
        "value_floats": TensorProto.FLOAT,
        "value_int": TensorProto.INT64,
        "value_ints": TensorProto.INT64,
        "value_string": TensorProto.STRING,
        "value_strings": TensorProto.STRING,
    }
    return next(
        (data_type for name, data_type in scalar_types.items() if any(attr.name == name for attr in node.attribute)),
        None,
    )


def _authoritative_tensor_types(graph: onnx.GraphProto) -> dict[str, int]:
    """Propagate types from graph inputs, initializers, and operator contracts.

    Deliberately ignore ``value_info`` because failed symbolic inference can leave
    stale F32 annotations in mixed-F16 graphs. Unknown paths remain unknown.
    """
    tensor_types = {
        value.name: int(value.type.tensor_type.elem_type)
        for value in graph.input
        if value.type.HasField("tensor_type")
        and value.type.tensor_type.elem_type != TensorProto.UNDEFINED
    }
    tensor_types.update({
        initializer.name: int(initializer.data_type)
        for initializer in graph.initializer
    })

    changed = True
    while changed:
        changed = False
        for node in graph.node:
            if not node.output:
                continue
            inferred: list[int | None] = [None] * len(node.output)
            if node.op_type == "Cast":
                inferred[0] = _node_int_attribute(node, "to")
            elif node.op_type == "Constant":
                inferred[0] = _constant_node_type(node)
            elif node.op_type in ("Shape", "Size", "ArgMax", "ArgMin", "NonZero"):
                inferred[0] = TensorProto.INT64
            elif node.op_type in ("Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "And", "Or", "Not"):
                inferred[0] = TensorProto.BOOL
            elif node.op_type in ("MatMulInteger", "ConvInteger"):
                inferred[0] = TensorProto.INT32
            elif node.op_type == "TopK":
                inferred[0] = tensor_types.get(node.input[0]) if node.input else None
                if len(inferred) > 1:
                    inferred[1] = TensorProto.INT64
            elif node.op_type == "Where" and len(node.input) >= 3:
                value_types = {
                    tensor_types[name]
                    for name in node.input[1:3]
                    if name in tensor_types
                }
                if len(value_types) == 1:
                    inferred[0] = next(iter(value_types))
            elif node.op_type in _FIRST_INPUT_TYPE_OPS and node.input:
                inferred[0] = tensor_types.get(node.input[0])
                if node.op_type == "Split":
                    inferred = [inferred[0]] * len(node.output)
                elif node.op_type == "MaxPool" and len(inferred) > 1:
                    inferred[1] = TensorProto.INT64
                elif node.op_type == "Dropout" and len(inferred) > 1:
                    inferred[1] = TensorProto.BOOL
            elif node.op_type in _SAME_TYPE_INPUT_OPS:
                value_types = {
                    tensor_types[name]
                    for name in node.input
                    if name and name in tensor_types
                }
                if len(value_types) == 1:
                    inferred[0] = next(iter(value_types))
            for output_name, data_type in zip(node.output, inferred):
                if (
                    output_name
                    and data_type is not None
                    and output_name not in tensor_types
                ):
                    tensor_types[output_name] = int(data_type)
                    changed = True
    return tensor_types


def restore_precision_free_graph_outputs(
    model: onnx.ModelProto,
    *,
    alias_prefix: str = "InsertedPrecisionFreeCast_",
) -> dict[str, str]:
    """Restore exact public names orphaned when an output Cast is removed."""
    available = {value.name for value in model.graph.input}
    available.update(initializer.name for initializer in model.graph.initializer)
    producers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in model.graph.node:
        for output_index, output in enumerate(node.output):
            if output:
                producers.setdefault(output, []).append((node, output_index))
                available.add(output)
    missing = [value.name for value in model.graph.output if value.name not in available]
    remap: dict[str, str] = {}
    for public_name in missing:
        alias = f"{alias_prefix}{public_name}"
        owners = producers.get(alias, ())
        if len(owners) != 1:
            raise RuntimeError(
                f"Cannot restore public output {public_name!r}: expected one exact "
                f"precision-free producer {alias!r}, found {len(owners)}."
            )
        remap[alias] = public_name
    if not remap:
        return {}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = remap.get(name, name)
        for index, name in enumerate(node.output):
            node.output[index] = remap.get(name, name)
    retained_value_info = [
        value for value in model.graph.value_info if value.name not in remap
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    for annotation in model.graph.quantization_annotation:
        annotation.tensor_name = remap.get(annotation.tensor_name, annotation.tensor_name)
        for parameter in annotation.quant_parameter_tensor_names:
            parameter.value = remap.get(parameter.value, parameter.value)
    return remap


def set_model_metadata(model: onnx.ModelProto, key: str, value: object) -> None:
    """Set one in-memory metadata value without creating duplicate keys."""
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = str(value)
            return
    model.metadata_props.add(key=key, value=str(value))


def normalize_float16_output_bridge(
    model: onnx.ModelProto,
    *,
    producer_op_type: str,
    producer_name_contains: str,
    private_output_name: str,
    bridge_node_name: str,
    metadata_key: str,
    output_name: str = "logits",
    graph_label: str = "float16 model",
) -> None:
    """Normalize one F16 producer to a stable public F32 output.

    Generic mixed-precision cleanup can leave the public name on the F16
    producer and move the F32 value to a private Cast output. Merged selection
    shells require the opposite contract. Matching is strict and intentionally
    model-specific through the producer operation/name arguments.
    """
    restore_precision_free_graph_outputs(model)
    producers = [
        node
        for node in model.graph.node
        if node.op_type == producer_op_type
        and producer_name_contains in node.name
    ]
    if len(producers) != 1:
        raise RuntimeError(
            f"{graph_label} expected one {producer_op_type} matching "
            f"{producer_name_contains!r}, found {len(producers)}."
        )
    producer = producers[0]
    initializer_types = {
        initializer.name: int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    if not any(
        initializer_types.get(name) == TensorProto.FLOAT16
        for name in producer.input[1:]
    ):
        raise RuntimeError(f"{graph_label} output producer has no FLOAT16 weight/bias.")

    raw_name = producer.output[0]
    bridges = [
        node
        for node in model.graph.node
        if node.op_type == "Cast"
        and list(node.input) == [raw_name]
        and _node_int_attribute(node, "to") == TensorProto.FLOAT
    ]
    if len(bridges) > 1:
        raise RuntimeError(f"{graph_label} has multiple FLOAT32 output bridges.")
    bridge = bridges[0] if bridges else None

    if raw_name == output_name:
        occupied = {value.name for value in (*model.graph.input, *model.graph.output)}
        occupied.update(initializer.name for initializer in model.graph.initializer)
        occupied.update(
            output
            for node in model.graph.node
            for output in node.output
            if output
        )
        if private_output_name in occupied:
            raise RuntimeError(
                f"{graph_label} private output collision: {private_output_name!r}."
            )
        producer.output[0] = private_output_name
    else:
        private_output_name = raw_name

    old_bridge_output = bridge.output[0] if bridge is not None else None
    if bridge is None:
        new_bridge = onnx.helper.make_node(
            "Cast",
            [private_output_name],
            [output_name],
            name=bridge_node_name,
            to=TensorProto.FLOAT,
        )
        producer_index = next(
            index for index, node in enumerate(model.graph.node) if node is producer
        )
        model.graph.node.insert(producer_index + 1, new_bridge)
        bridge = model.graph.node[producer_index + 1]
    else:
        bridge.input[0] = private_output_name
        bridge.output[0] = output_name

    aliases = {
        name
        for name in (raw_name, old_bridge_output)
        if name and name != output_name
    }
    for node in model.graph.node:
        if node is bridge:
            continue
        for index, name in enumerate(node.input):
            if name in aliases:
                node.input[index] = output_name

    retained = [
        value
        for value in model.graph.value_info
        if value.name not in {private_output_name, old_bridge_output, output_name}
    ]
    retained.extend(
        (
            onnx.helper.make_tensor_value_info(
                private_output_name,
                TensorProto.FLOAT16,
                None,
            ),
            onnx.helper.make_tensor_value_info(
                output_name,
                TensorProto.FLOAT,
                None,
            ),
        )
    )
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained)
    for value in model.graph.output:
        if value.name == output_name:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
    set_model_metadata(model, metadata_key, "1")
def repair_model_file(model_path: str | Path) -> int:
    """Restore exact public aliases in a data-light graph."""
    model_path = Path(model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    restored = restore_precision_free_graph_outputs(model)
    if restored:
        onnx.save(model, str(model_path))
    return len(restored)


def _safe_external_data_path(model_path: Path, location: str) -> Path:
    relative = Path(location)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(
            f"Unsafe external-data location {location!r} in {model_path.name}."
        )
    target = model_path.parent / relative
    target.resolve(strict=True).relative_to(model_path.parent.resolve())
    return target


def _tensor_raw_data_length(tensor: TensorProto) -> int:
    """Return the exact packed byte length implied by a numeric TensorProto."""
    if tensor.data_type == TensorProto.STRING:
        raise RuntimeError(
            f"Cannot infer a raw byte length for string tensor {tensor.name!r}."
        )
    element_count = 1
    for dim in tensor.dims:
        if dim < 0:
            raise RuntimeError(
                f"Tensor {tensor.name!r} has invalid negative dimension {dim}."
            )
        element_count *= int(dim)
    packed_4bit_types = {
        getattr(TensorProto, name)
        for name in ("UINT4", "INT4", "FLOAT4E2M1")
        if hasattr(TensorProto, name)
    }
    packed_2bit_types = {
        getattr(TensorProto, name)
        for name in ("UINT2", "INT2")
        if hasattr(TensorProto, name)
    }
    if tensor.data_type in packed_2bit_types:
        return (element_count + 3) // 4
    if tensor.data_type in packed_4bit_types:
        return (element_count + 1) // 2
    itemsize = np.dtype(
        onnx.helper.tensor_dtype_to_np_dtype(tensor.data_type)
    ).itemsize
    return element_count * itemsize


def _external_initializer_region(
    initializer: TensorProto,
    model_path: Path,
) -> tuple[Path, int, int]:
    """Return and bounds-check one initializer's exact external byte region."""
    if initializer.data_location != TensorProto.EXTERNAL:
        raise RuntimeError(
            f"Initializer {initializer.name!r} in {model_path.name} is not external."
        )
    external = {entry.key: entry.value for entry in initializer.external_data}
    location = external.get("location")
    if not location:
        raise RuntimeError(
            f"Initializer {initializer.name!r} in {model_path.name} has no "
            "external-data location."
        )
    data_path = _safe_external_data_path(model_path, location)
    expected_length = _tensor_raw_data_length(initializer)
    offset = int(external.get("offset", "0") or "0")
    length = int(external.get("length", str(expected_length)) or "0")
    if length != expected_length:
        raise RuntimeError(
            f"Initializer {initializer.name!r} in {model_path.name} declares "
            f"{length} bytes; expected {expected_length}."
        )
    if offset < 0 or offset + length > data_path.stat().st_size:
        raise RuntimeError(
            f"Initializer {initializer.name!r} in {model_path.name} has an "
            "invalid or truncated external payload."
        )
    return data_path, offset, length


def _external_region_digest(region: tuple[Path, int, int]) -> bytes:
    """Hash an external-data region without materializing it in memory."""
    data_path, offset, length = region
    digest = hashlib.sha256()
    remaining = length
    with data_path.open("rb") as data_file:
        data_file.seek(offset)
        while remaining:
            chunk = data_file.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError(
                    f"External payload became truncated while reading {data_path}."
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.digest()


def _inline_initializer_signature(initializer: TensorProto) -> tuple[int, bytes]:
    """Return the canonical byte length and SHA-256 of an inline initializer."""
    if initializer.data_location == TensorProto.EXTERNAL:
        raise RuntimeError(
            f"Initializer {initializer.name!r} is external, not inline."
        )
    if initializer.data_type == TensorProto.STRING:
        raise RuntimeError(
            f"Cannot byte-share string initializer {initializer.name!r}."
        )
    raw = (
        initializer.raw_data
        if initializer.raw_data
        else numpy_helper.to_array(initializer).tobytes()
    )
    expected_length = _tensor_raw_data_length(initializer)
    if len(raw) != expected_length:
        raise RuntimeError(
            f"Inline initializer {initializer.name!r} contains {len(raw)} bytes; "
            f"expected {expected_length}."
        )
    return expected_length, hashlib.sha256(raw).digest()


def _external_reference_paths(
    model: onnx.ModelProto,
    model_path: Path,
) -> tuple[set[Path], set[Path]]:
    """Return lexical and canonical paths referenced by all external tensors."""
    lexical_paths: set[Path] = set()
    canonical_paths: set[Path] = set()
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        external = {entry.key: entry.value for entry in tensor.external_data}
        location = external.get("location")
        if not location:
            raise RuntimeError(
                f"External tensor {tensor.name!r} in {model_path.name} has no "
                "external-data location."
            )
        data_path = _safe_external_data_path(model_path, location)
        lexical_paths.add(data_path.absolute())
        canonical_paths.add(data_path.resolve())
    return lexical_paths, canonical_paths


_TENSOR_PAYLOAD_FIELDS = (
    "raw_data",
    "float_data",
    "int32_data",
    "string_data",
    "int64_data",
    "double_data",
    "uint64_data",
)


def _iter_all_storage_tensors(graph: onnx.GraphProto):
    """Yield dense tensor payloads, including sparse and nested graph storage."""
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("sparse_tensor"):
                yield attribute.sparse_tensor.values
                yield attribute.sparse_tensor.indices
            for sparse in attribute.sparse_tensors:
                yield sparse.values
                yield sparse.indices
            if attribute.HasField("g"):
                yield from _iter_all_storage_tensors(attribute.g)
            for subgraph in attribute.graphs:
                yield from _iter_all_storage_tensors(subgraph)


_STRUCTURAL_INPUTS: dict[str, frozenset[int] | None] = {
    "ConstantOfShape": frozenset({0}),
    "CumSum": frozenset({1}),
    "Expand": frozenset({1}),
    "Gather": frozenset({1}),
    "GatherElements": frozenset({1}),
    "GatherND": frozenset({1}),
    "NonMaxSuppression": frozenset({1, 2, 3, 4, 5}),
    "OneHot": frozenset({1}),
    "Pad": frozenset({1}),
    "Range": None,
    "Reshape": frozenset({1}),
    "Resize": frozenset({1, 2, 3}),
    "ScatterElements": frozenset({1}),
    "ScatterND": frozenset({1}),
    "Slice": frozenset({1, 2, 3, 4}),
    "Split": frozenset({1}),
    "Squeeze": frozenset({1}),
    "Tile": frozenset({1}),
    "TopK": frozenset({1}),
    "Trilu": frozenset({1}),
    "Unsqueeze": frozenset({1}),
}


def _is_structural_input(node: onnx.NodeProto, input_index: int) -> bool:
    positions = _STRUCTURAL_INPUTS.get(node.op_type, ())
    if positions is None:
        return True
    if input_index in positions:
        return True
    return node.op_type.startswith("Reduce") and input_index == 1


_STRUCTURAL_EXPRESSION_OPS = frozenset({
    "Abs",
    "Add",
    "Cast",
    "Ceil",
    "Clip",
    "Concat",
    "Div",
    "Expand",
    "Floor",
    "Gather",
    "GatherElements",
    "GatherND",
    "Identity",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "Neg",
    "Range",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "Reshape",
    "Round",
    "ScatterElements",
    "ScatterND",
    "Slice",
    "Split",
    "Squeeze",
    "Sub",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "Where",
})

_INLINE_CONTROL_DATA_TYPES = frozenset({
    TensorProto.BOOL,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
})


def _structural_value_dependencies(graph: onnx.GraphProto) -> set[str]:
    """Return values whose payload contributes to a structural operator input."""
    producers = {
        output: node
        for node in graph.node
        for output in node.output
        if output
    }
    pending = [
        input_name
        for node in graph.node
        for input_index, input_name in enumerate(node.input)
        if input_name and _is_structural_input(node, input_index)
    ]
    dependencies: set[str] = set()
    while pending:
        value_name = pending.pop()
        if value_name in dependencies:
            continue
        dependencies.add(value_name)
        producer = producers.get(value_name)
        if producer is None or producer.op_type not in _STRUCTURAL_EXPRESSION_OPS:
            continue
        pending.extend(input_name for input_name in producer.input if input_name)
    return dependencies


def _iter_structural_storage_tensors(graph: onnx.GraphProto):
    structural_values = _structural_value_dependencies(graph)
    for initializer in graph.initializer:
        if initializer.name in structural_values:
            yield initializer
    for node in graph.node:
        if node.op_type == "Constant" and node.output and node.output[0] in structural_values:
            for attribute in node.attribute:
                if attribute.HasField("t"):
                    yield attribute.t
                yield from attribute.tensors
                if attribute.HasField("sparse_tensor"):
                    yield attribute.sparse_tensor.values
                    yield attribute.sparse_tensor.indices
                for sparse in attribute.sparse_tensors:
                    yield sparse.values
                    yield sparse.indices
        for attribute in node.attribute:
            if attribute.HasField("g"):
                yield from _iter_structural_storage_tensors(attribute.g)
            for subgraph in attribute.graphs:
                yield from _iter_structural_storage_tensors(subgraph)


def _iter_inline_storage_tensors(graph: onnx.GraphProto):
    for tensor in _iter_structural_storage_tensors(graph):
        yield tensor
    for tensor in _iter_all_storage_tensors(graph):
        if tensor.data_type in _INLINE_CONTROL_DATA_TYPES:
            yield tensor


def _iter_shared_storage_tensors(
    graph: onnx.GraphProto,
    *,
    keep_control_data_inline: bool = True,
):
    """Yield model data/parameter tensors while keeping shape controls inline.

    ORT shape inference must read values such as Reshape targets and Split sizes
    before external data is materialized. Tensors that contribute to structural
    values through small expression chains such as Concat -> Reshape also remain
    in the protobuf.
    """
    structural_values = _structural_value_dependencies(graph)
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if input_name:
                consumers.setdefault(input_name, []).append((node, input_index))

    def should_share(tensor: TensorProto, value_name: str | None) -> bool:
        if value_name in structural_values:
            return False
        if keep_control_data_inline and tensor.data_type in _INLINE_CONTROL_DATA_TYPES:
            return False
        if tensor.data_location == TensorProto.EXTERNAL:
            return True
        uses = consumers.get(value_name or "", ())
        return not uses or not any(
            _is_structural_input(node, input_index)
            for node, input_index in uses
        )

    for initializer in graph.initializer:
        if should_share(initializer, initializer.name):
            yield initializer
    for sparse in graph.sparse_initializer:
        if should_share(sparse.values, sparse.values.name):
            yield sparse.values
        if should_share(sparse.indices, sparse.indices.name):
            yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            is_constant = node.op_type == "Constant" and bool(node.output)
            value_name = node.output[0] if is_constant else None
            if attribute.HasField("t") and should_share(attribute.t, value_name) and (
                attribute.t.data_location == TensorProto.EXTERNAL or is_constant
            ):
                yield attribute.t
            for tensor in attribute.tensors:
                if should_share(tensor, value_name) and (
                    tensor.data_location == TensorProto.EXTERNAL or is_constant
                ):
                    yield tensor
            if attribute.HasField("sparse_tensor"):
                sparse = attribute.sparse_tensor
                if should_share(sparse.values, value_name) and (
                    sparse.values.data_location == TensorProto.EXTERNAL or is_constant
                ):
                    yield sparse.values
                if should_share(sparse.indices, value_name):
                    yield sparse.indices
            for sparse in attribute.sparse_tensors:
                if should_share(sparse.values, value_name) and (
                    sparse.values.data_location == TensorProto.EXTERNAL or is_constant
                ):
                    yield sparse.values
                if should_share(sparse.indices, value_name):
                    yield sparse.indices
            if attribute.HasField("g"):
                yield from _iter_shared_storage_tensors(
                    attribute.g,
                    keep_control_data_inline=keep_control_data_inline,
                )
            for subgraph in attribute.graphs:
                yield from _iter_shared_storage_tensors(
                    subgraph,
                    keep_control_data_inline=keep_control_data_inline,
                )


def _inline_tensor_bytes(tensor: TensorProto) -> bytes:
    """Return the canonical raw payload for one inline numeric tensor."""
    if tensor.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
        raise RuntimeError(f"Tensor {tensor.name!r} is not numeric storage.")
    raw = bytes(tensor.raw_data)
    if not raw:
        raw = np.ascontiguousarray(numpy_helper.to_array(tensor)).tobytes()
    expected_length = _tensor_raw_data_length(tensor)
    if len(raw) != expected_length:
        raise RuntimeError(
            f"Tensor {tensor.name!r} contains {len(raw)} bytes; expected "
            f"{expected_length}."
        )
    return raw


def _replace_tensor_payload_with_external_ref(
    tensor: TensorProto,
    data_name: str,
    offset: int,
    length: int,
) -> None:
    for field in _TENSOR_PAYLOAD_FIELDS:
        tensor.ClearField(field)
    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for key, value in (
        ("location", data_name),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def _is_initializer_manifest(model: onnx.ModelProto) -> bool:
    return (
        not model.graph.node
        and not model.graph.input
        and not model.graph.output
        and bool(model.graph.initializer)
    )


def _default_shared_model_name(folder: Path) -> str:
    stem = folder.name.removesuffix("_Optimized") or folder.name
    return f"{stem}_SharedInitializers.onnx"


def consolidate_optimized_model_weights(
    folder: str | Path,
    shared_model_name: str | None = None,
) -> dict[str, int | str]:
    """Store every final model-data tensor in one deduplicated shared sidecar.

    Existing initializer manifests are retained so runtimes that mmap and attach
    them keep the same API. Direct model families receive a lightweight manifest.
    Graph tensor names and shapes are unchanged; only their payload locations are
    rewritten. Byte-identical payloads alias one physical ``(offset, length)``.
    """
    folder = Path(folder)
    graph_paths = sorted(folder.glob("*.onnx"))
    if not graph_paths:
        raise FileNotFoundError(f"No ONNX graphs found in {folder}.")

    discovered_manifests: list[Path] = []
    generated_manifest_path: Path | None = None
    for graph_path in graph_paths:
        if "SharedInitializers" not in graph_path.name:
            continue
        candidate = onnx.load(str(graph_path), load_external_data=False)
        if _is_initializer_manifest(candidate):
            if candidate.producer_name == "Optimize_ONNX_Common.py":
                if generated_manifest_path is not None:
                    raise RuntimeError(
                        f"{folder} contains multiple common-generated manifests."
                    )
                generated_manifest_path = graph_path
            else:
                discovered_manifests.append(graph_path)
    if len(discovered_manifests) > 1:
        raise RuntimeError(
            f"{folder} contains multiple initializer manifests: "
            f"{[path.name for path in discovered_manifests]}."
        )
    if discovered_manifests:
        shared_model_path = discovered_manifests[0]
        if shared_model_name and shared_model_path.name != shared_model_name:
            raise RuntimeError(
                f"Existing shared manifest {shared_model_path.name!r} does not "
                f"match requested name {shared_model_name!r}."
            )
    else:
        shared_model_path = folder / (
            shared_model_name
            or (
                generated_manifest_path.name
                if generated_manifest_path is not None
                else _default_shared_model_name(folder)
            )
        )
        if (
            generated_manifest_path is not None
            and generated_manifest_path != shared_model_path
        ):
            raise RuntimeError(
                f"Existing generated manifest {generated_manifest_path.name!r} "
                f"does not match requested name {shared_model_path.name!r}."
            )
        if generated_manifest_path is not None:
            graph_paths.remove(generated_manifest_path)
    data_name = shared_model_path.name + ".data"
    data_path = folder / data_name

    original_external_paths: set[Path] = set()
    temporary_models: dict[Path, Path] = {}
    manifest_candidates: dict[
        str, tuple[int, tuple[int, ...], tuple[int, int]] | None
    ] = {}
    source_region_cache: dict[tuple[Path, int, int], tuple[int, int]] = {}
    target_by_digest: dict[tuple[int, bytes], tuple[int, int]] = {}
    logical_tensors = 0
    logical_bytes = 0

    with tempfile.NamedTemporaryFile(
        prefix=data_name + ".",
        suffix=".tmp",
        dir=folder,
        delete=False,
    ) as target_file:
        temporary_data_path = Path(target_file.name)
        try:
            for graph_path in graph_paths:
                model = onnx.load(str(graph_path), load_external_data=False)
                _recover_zero_length_external_ranges(model, graph_path)
                initializer_manifest = _is_initializer_manifest(model)
                for tensor in (() if initializer_manifest else _iter_inline_storage_tensors(model.graph)):
                    if tensor.data_location != TensorProto.EXTERNAL:
                        continue
                    source_path, source_offset, source_length = _external_initializer_region(
                        tensor,
                        graph_path,
                    )
                    original_external_paths.add(source_path.absolute())
                    with source_path.open("rb") as source_file:
                        source_file.seek(source_offset)
                        raw = source_file.read(source_length)
                    if len(raw) != source_length:
                        raise RuntimeError(
                            f"Structural tensor {tensor.name!r} became truncated while "
                            f"reading {source_path}."
                        )
                    for field in _TENSOR_PAYLOAD_FIELDS:
                        tensor.ClearField(field)
                    del tensor.external_data[:]
                    tensor.data_location = TensorProto.DEFAULT
                    tensor.raw_data = raw
                for tensor in _iter_shared_storage_tensors(
                    model.graph,
                    keep_control_data_inline=not initializer_manifest,
                ):
                    if tensor.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
                        continue
                    expected_length = _tensor_raw_data_length(tensor)
                    if expected_length == 0:
                        continue
                    logical_tensors += 1
                    logical_bytes += expected_length

                    source_region = None
                    inline_payload = None
                    if tensor.data_location == TensorProto.EXTERNAL:
                        source_region = _external_initializer_region(
                            tensor,
                            graph_path,
                        )
                        original_external_paths.add(source_region[0].absolute())
                        source_key = (
                            source_region[0].resolve(),
                            source_region[1],
                            source_region[2],
                        )
                        target_region = source_region_cache.get(source_key)
                    else:
                        source_key = None
                        target_region = None
                        inline_payload = _inline_tensor_bytes(tensor)

                    if target_region is None:
                        candidate_offset = target_file.tell()
                        digest = hashlib.sha256()
                        written = 0
                        if source_region is not None:
                            source_path, source_offset, source_length = source_region
                            with source_path.open("rb") as source_file:
                                source_file.seek(source_offset)
                                remaining = source_length
                                while remaining:
                                    chunk = source_file.read(min(8 * 1024 * 1024, remaining))
                                    if not chunk:
                                        raise RuntimeError(
                                            f"External tensor {tensor.name!r} became "
                                            f"truncated while reading {source_path}."
                                        )
                                    target_file.write(chunk)
                                    digest.update(chunk)
                                    written += len(chunk)
                                    remaining -= len(chunk)
                        else:
                            target_file.write(inline_payload)
                            digest.update(inline_payload)
                            written = len(inline_payload)
                        if written != expected_length:
                            raise RuntimeError(
                                f"Tensor {tensor.name!r} streamed {written} bytes; "
                                f"expected {expected_length}."
                            )
                        digest_key = (written, digest.digest())
                        target_region = target_by_digest.get(digest_key)
                        if target_region is None:
                            target_region = (candidate_offset, written)
                            target_by_digest[digest_key] = target_region
                        else:
                            target_file.seek(candidate_offset)
                            target_file.truncate()
                        if source_key is not None:
                            source_region_cache[source_key] = target_region

                    _replace_tensor_payload_with_external_ref(
                        tensor,
                        data_name,
                        target_region[0],
                        target_region[1],
                    )
                    if tensor.name:
                        contract = (
                            int(tensor.data_type),
                            tuple(int(dim) for dim in tensor.dims),
                            target_region,
                        )
                        previous = manifest_candidates.get(tensor.name, contract)
                        manifest_candidates[tensor.name] = (
                            contract if previous == contract else None
                        )

                temporary_model = graph_path.with_name(
                    graph_path.name + ".consolidating.tmp"
                )
                temporary_model.unlink(missing_ok=True)
                onnx.save_model(model, str(temporary_model), save_as_external_data=False)
                temporary_models[graph_path] = temporary_model

            target_file.flush()
            os.fsync(target_file.fileno())
        except BaseException:
            for temporary_model in temporary_models.values():
                temporary_model.unlink(missing_ok=True)
            temporary_data_path.unlink(missing_ok=True)
            raise

    if not discovered_manifests:
        references = []
        for name, contract in sorted(manifest_candidates.items()):
            if contract is None:
                continue
            data_type, dims, (offset, length) = contract
            reference = TensorProto()
            reference.name = name
            reference.data_type = data_type
            reference.dims.extend(dims)
            _replace_tensor_payload_with_external_ref(
                reference, data_name, offset, length
            )
            references.append(reference)
        manifest = onnx.helper.make_model(
            onnx.helper.make_graph(
                [],
                "asr_shared_initializers",
                [],
                [],
                initializer=references,
            ),
            producer_name="Optimize_ONNX_Common.py",
            opset_imports=[onnx.helper.make_opsetid("", 20)],
        )
        manifest.ir_version = 10
        manifest.metadata_props.add(key="initializer_count", value=str(len(references)))
        temporary_manifest = shared_model_path.with_name(
            shared_model_path.name + ".consolidating.tmp"
        )
        onnx.save_model(
            manifest,
            str(temporary_manifest),
            save_as_external_data=False,
        )
        temporary_models[shared_model_path] = temporary_manifest

    os.replace(temporary_data_path, data_path)
    for graph_path, temporary_model in temporary_models.items():
        os.replace(temporary_model, graph_path)

    protected = {data_path.absolute(), data_path.resolve()}
    removed_payloads = 0
    cleanup_paths = original_external_paths | {
        path.absolute() for path in folder.glob("*.data")
    }
    for original_path in sorted(cleanup_paths, key=str):
        if (
            original_path in protected
            or original_path.resolve() in protected
            or not original_path.exists()
        ):
            continue
        original_path.unlink()
        removed_payloads += 1

    report = validate_consolidated_model_weights(folder, shared_model_path.name)
    report.update(
        {
            "logical_tensor_count": logical_tensors,
            "logical_data_bytes": logical_bytes,
            "removed_payload_file_count": removed_payloads,
        }
    )
    return report


def validate_consolidated_model_weights(
    folder: str | Path,
    shared_model_name: str,
) -> dict[str, int | str]:
    """Validate one-sidecar storage and absence of duplicate physical payloads."""
    folder = Path(folder)
    data_name = shared_model_name + ".data"
    data_path = folder / data_name
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    payload_files = sorted(folder.glob("*.data"))
    if payload_files != [data_path]:
        raise RuntimeError(
            f"Expected only {data_name!r}; found {[path.name for path in payload_files]}."
        )

    referenced_regions: set[tuple[int, int]] = set()
    logical_tensors = 0
    for graph_path in sorted(folder.glob("*.onnx")):
        model = onnx.load(str(graph_path), load_external_data=False)
        initializer_manifest = _is_initializer_manifest(model)
        external_inline_required = [
            tensor.name
            for tensor in (() if initializer_manifest else _iter_inline_storage_tensors(model.graph))
            if tensor.data_location == TensorProto.EXTERNAL
        ]
        if external_inline_required:
            raise RuntimeError(
                f"{graph_path.name} retains external control tensor(s): "
                f"{external_inline_required[:8]}."
            )
        for tensor in _iter_shared_storage_tensors(
            model.graph,
            keep_control_data_inline=not initializer_manifest,
        ):
            if tensor.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
                continue
            expected_length = _tensor_raw_data_length(tensor)
            if expected_length == 0:
                continue
            logical_tensors += 1
            if tensor.data_location != TensorProto.EXTERNAL:
                raise RuntimeError(
                    f"{graph_path.name}:{tensor.name} retains inline numeric data."
                )
            fields = {entry.key: entry.value for entry in tensor.external_data}
            if fields.get("location") != data_name:
                raise RuntimeError(
                    f"{graph_path.name}:{tensor.name} references "
                    f"{fields.get('location')!r}, expected {data_name!r}."
                )
            offset = int(fields.get("offset", "0"))
            length = int(fields.get("length", "0"))
            if (
                length != expected_length
                or offset < 0
                or offset + length > data_path.stat().st_size
            ):
                raise RuntimeError(
                    f"{graph_path.name}:{tensor.name} has invalid shared range."
                )
            referenced_regions.add((offset, length))

    digest_regions: dict[tuple[int, bytes], tuple[int, int]] = {}
    with data_path.open("rb") as data_file:
        for offset, length in sorted(referenced_regions):
            data_file.seek(offset)
            digest = hashlib.sha256()
            remaining = length
            while remaining:
                chunk = data_file.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError(f"Truncated shared range at offset {offset}.")
                digest.update(chunk)
                remaining -= len(chunk)
            key = (length, digest.digest())
            previous = digest_regions.get(key)
            if previous is not None and previous != (offset, length):
                raise RuntimeError(
                    f"Duplicate payload stored at {previous} and {(offset, length)}."
                )
            digest_regions[key] = (offset, length)

    return {
        "shared_model": shared_model_name,
        "shared_data": data_name,
        "shared_data_bytes": data_path.stat().st_size,
        "logical_tensor_count": logical_tensors,
        "unique_data_ranges": len(referenced_regions),
    }


def share_external_initializers_if_identical(
    model_path: str | Path,
    shared_model_path: str | Path,
    *,
    require_all_external: bool = False,
) -> dict[str, int]:
    """Redirect private or inline initializers to exact shared-bundle matches.

    A tensor is shared only when name, dtype, dimensions, byte length, and SHA-256
    all match. This deliberately rejects independently quantized or transposed
    representations even when they originate from the same source weight. Inline
    tensors are hashed from the same canonical bytes used by the shared writer.
    Private files are removed only after the rewritten graph has no references to
    them. Set ``require_all_external`` for a standalone whose complete external
    payload is expected to live in the shared bundle.
    """
    model_path = Path(model_path)
    shared_model_path = Path(shared_model_path)
    model = onnx.load(str(model_path), load_external_data=False)
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    shared_by_name = {
        initializer.name: initializer
        for initializer in shared_model.graph.initializer
    }
    original_external_names = {
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == TensorProto.EXTERNAL
    }
    original_locations = {
        entry.value
        for initializer in model.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    digest_cache: dict[tuple[Path, int, int], bytes] = {}

    def digest(region: tuple[Path, int, int]) -> bytes:
        if region not in digest_cache:
            digest_cache[region] = _external_region_digest(region)
        return digest_cache[region]

    rewritten: list[TensorProto] = []
    shared_names: set[str] = set()
    shared_bytes = 0
    for initializer in model.graph.initializer:
        candidate = shared_by_name.get(initializer.name)
        if (
            candidate is None
            or candidate.data_location != TensorProto.EXTERNAL
            or initializer.data_type != candidate.data_type
            or tuple(initializer.dims) != tuple(candidate.dims)
        ):
            rewritten.append(initializer)
            continue
        shared_region = _external_initializer_region(candidate, shared_model_path)
        if initializer.data_location == TensorProto.EXTERNAL:
            source_region = _external_initializer_region(initializer, model_path)
            source_length = source_region[2]
            source_digest = digest(source_region)
        else:
            source_length, source_digest = _inline_initializer_signature(initializer)
        if source_length != shared_region[2] or source_digest != digest(shared_region):
            rewritten.append(initializer)
            continue

        reference = TensorProto()
        reference.name = initializer.name
        reference.data_type = initializer.data_type
        reference.dims.extend(initializer.dims)
        reference.data_location = TensorProto.EXTERNAL
        reference.external_data.extend(candidate.external_data)
        rewritten.append(reference)
        shared_names.add(initializer.name)
        shared_bytes += source_length

    required_names = set(original_external_names)
    if require_all_external:
        required_names.update(
            initializer.name
            for initializer in model.graph.initializer
            if initializer.name in shared_by_name
        )
    unmatched = sorted(required_names - shared_names)
    if require_all_external and unmatched:
        raise RuntimeError(
            f"{model_path.name} has external initializer(s) without an exact "
            f"shared match: {unmatched}."
        )
    if not shared_names:
        return {
            "shared_initializer_count": 0,
            "shared_data_bytes": 0,
            "removed_external_file_count": 0,
        }

    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    onnx.save(model, str(model_path))

    protected_lexical, protected_canonical = _external_reference_paths(
        model, model_path
    )
    shared_lexical, shared_canonical = _external_reference_paths(
        shared_model, shared_model_path
    )
    protected_lexical.update(shared_lexical)
    protected_canonical.update(shared_canonical)
    current_path = model_path.absolute()
    for sibling_path in model_path.parent.glob("*.onnx"):
        if sibling_path.absolute() == current_path:
            continue
        try:
            sibling = onnx.load(str(sibling_path), load_external_data=False)
            sibling_lexical, sibling_canonical = _external_reference_paths(
                sibling, sibling_path
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot safely remove private external data because sibling "
                f"model {sibling_path.name!r} could not be inspected."
            ) from exc
        protected_lexical.update(sibling_lexical)
        protected_canonical.update(sibling_canonical)

    removed = 0
    removed_paths: set[Path] = set()
    candidates: dict[Path, tuple[Path, bool]] = {}
    for location in original_locations:
        data_path = _safe_external_data_path(model_path, location)
        candidates[data_path.absolute()] = (data_path.resolve(), data_path.is_symlink())
    for data_path, (canonical_path, is_symlink) in sorted(
        candidates.items(), key=lambda item: str(item[0])
    ):
        if data_path in protected_lexical:
            continue
        if not is_symlink and canonical_path in protected_canonical:
            continue
        if data_path in removed_paths:
            continue
        data_path.unlink()
        removed_paths.add(data_path)
        removed += 1
    return {
        "shared_initializer_count": len(shared_names),
        "shared_data_bytes": shared_bytes,
        "removed_external_file_count": removed,
    }


def rewrite_tied_embed_from_matmul_nbits(
    embed_model: onnx.ModelProto,
    quantized_main: onnx.ModelProto,
    *,
    alias_prefix: str = "tied_embed_quantized_",
    allow_row_gather: bool = False,
) -> dict[str, object]:
    """Make a verified tied Embed reuse Main's packed lm-head tuple.

    Exporters first byte-verify the source tie and represent Embed either as a
    row ``Gather(axis=0)`` over ``[vocab, hidden]`` or as
    ``Gather(axis=1) -> Transpose[1,2,0]`` over ``[hidden, vocab]``. Replace that
    lookup with one row ``GatherBlockQuantized`` whose three aliases expose
    Main's exact packed weight, scales, and zero-point bytes under
    Gather-compatible shapes. No second quantization is performed.
    """
    embed_initializers = {
        initializer.name: initializer
        for initializer in embed_model.graph.initializer
    }
    gather_matches = []
    for gather in embed_model.graph.node:
        if (
            gather.domain not in ("", "ai.onnx")
            or gather.op_type != "Gather"
            or len(gather.input) < 2
            or len(gather.output) != 1
        ):
            continue
        table = embed_initializers.get(gather.input[0])
        if table is None or len(table.dims) != 2:
            continue
        axis = _node_int_attribute(gather, "axis")
        axis = 0 if axis is None else axis
        if axis == 0 and allow_row_gather:
            vocab, hidden = (int(dim) for dim in table.dims)
            gather_matches.append((gather, None, table, hidden, vocab))
            continue
        if axis == 1:
            consumers = [
                node
                for node in embed_model.graph.node
                if gather.output[0] in node.input
            ]
            if len(consumers) != 1:
                continue
            transpose = consumers[0]
            permutation = tuple(
                int(value)
                for value in next(
                    (
                        attribute.ints
                        for attribute in transpose.attribute
                        if attribute.name == "perm"
                    ),
                    (),
                )
            )
            if (
                transpose.domain not in ("", "ai.onnx")
                or transpose.op_type != "Transpose"
                or list(transpose.input) != [gather.output[0]]
                or len(transpose.output) != 1
                or permutation != (1, 2, 0)
            ):
                continue
            hidden, vocab = (int(dim) for dim in table.dims)
            gather_matches.append((gather, transpose, table, hidden, vocab))
    if len(gather_matches) != 1:
        raise RuntimeError(
            "Expected exactly one tied row Gather or column Gather->Transpose "
            f"pattern; found {len(gather_matches)}."
        )
    gather, transpose, table, hidden, vocab = gather_matches[0]

    main_initializers = {
        initializer.name: initializer
        for initializer in quantized_main.graph.initializer
    }
    tuple_matches = []
    for node in quantized_main.graph.node:
        if (
            node.domain != "com.microsoft"
            or node.op_type != "MatMulNBits"
            or len(node.input) < 4
        ):
            continue
        attributes = {
            attribute.name: int(attribute.i)
            for attribute in node.attribute
            if attribute.type == onnx.AttributeProto.INT
        }
        if attributes.get("K") != hidden or attributes.get("N") != vocab:
            continue
        tensors = tuple(main_initializers.get(node.input[index]) for index in (1, 2, 3))
        if any(tensor is None for tensor in tensors):
            continue
        tuple_matches.append((node, tensors, attributes))
    if len(tuple_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one MatMulNBits tied-table tuple with K={hidden}, "
            f"N={vocab}; found {len(tuple_matches)}."
        )
    _, packed_tuple, attributes = tuple_matches[0]
    bits = attributes.get("bits", 4)
    block_size = attributes.get("block_size", 0)
    if bits not in (2, 4, 8) or block_size <= 0 or hidden % block_size:
        raise RuntimeError(
            f"Unsupported tied MatMulNBits geometry: bits={bits}, "
            f"block_size={block_size}, hidden={hidden}."
        )
    k_blocks = hidden // block_size
    packed_block_bytes = (block_size * bits + 7) // 8
    packed_zero_point_columns = (k_blocks * bits + 7) // 8
    expected_contracts = (
        (TensorProto.UINT8, [vocab, k_blocks, packed_block_bytes]),
        (packed_tuple[1].data_type, [vocab, k_blocks]),
        (TensorProto.UINT8, [vocab, packed_zero_point_columns]),
    )
    for tensor, (data_type, dims) in zip(packed_tuple, expected_contracts):
        if tensor.data_type != data_type or list(tensor.dims) != dims:
            raise RuntimeError(
                f"Packed tied tensor {tensor.name!r} has type/shape "
                f"{tensor.data_type}/{list(tensor.dims)}; expected "
                f"{data_type}/{dims}."
            )

    occupied = {initializer.name for initializer in embed_model.graph.initializer}
    occupied.update(
        output
        for node in embed_model.graph.node
        for output in node.output
        if output
    )

    def unique_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in occupied:
            candidate = f"{base}_{suffix}"
            suffix += 1
        occupied.add(candidate)
        return candidate

    alias_dims = (
        [vocab, (hidden * bits + 7) // 8],
        [vocab, k_blocks],
        [vocab, packed_zero_point_columns],
    )
    aliases = []
    for role, source, dims in zip(("weight", "scales", "zero_points"), packed_tuple, alias_dims):
        alias = copy.deepcopy(source)
        alias.name = unique_name(f"{alias_prefix}{role}")
        del alias.dims[:]
        alias.dims.extend(dims)
        if _tensor_raw_data_length(alias) != _tensor_raw_data_length(source):
            raise RuntimeError(
                f"Tied Embed alias {alias.name!r} changes the packed byte length."
            )
        aliases.append(alias)

    replacement = onnx.helper.make_node(
        "GatherBlockQuantized",
        [aliases[0].name, gather.input[1], aliases[1].name, aliases[2].name],
        [transpose.output[0] if transpose is not None else gather.output[0]],
        name=unique_name(f"{alias_prefix}gather"),
        domain="com.microsoft",
        gather_axis=0,
        quantize_axis=1,
        block_size=block_size,
        bits=bits,
    )
    rewritten_nodes = []
    for node in embed_model.graph.node:
        if node is gather:
            rewritten_nodes.append(replacement)
        elif transpose is None or node is not transpose:
            rewritten_nodes.append(node)
    del embed_model.graph.node[:]
    embed_model.graph.node.extend(rewritten_nodes)

    remaining_inputs = {
        input_name
        for node in embed_model.graph.node
        for input_name in node.input
        if input_name
    }
    retained_initializers = [
        initializer
        for initializer in embed_model.graph.initializer
        if initializer.name != table.name or table.name in remaining_inputs
    ]
    del embed_model.graph.initializer[:]
    embed_model.graph.initializer.extend(retained_initializers)
    embed_model.graph.initializer.extend(aliases)
    retained_info = [
        value
        for value in embed_model.graph.value_info
        if transpose is None or value.name != gather.output[0]
    ]
    del embed_model.graph.value_info[:]
    embed_model.graph.value_info.extend(retained_info)
    if not any(opset.domain == "com.microsoft" for opset in embed_model.opset_import):
        embed_model.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    return {
        "bits": bits,
        "block_size": block_size,
        "hidden": hidden,
        "vocab": vocab,
        "removed_table": table.name if table.name not in remaining_inputs else None,
        "alias_names": tuple(alias.name for alias in aliases),
        "shared_data_bytes": sum(_tensor_raw_data_length(tensor) for tensor in packed_tuple),
    }


def collect_target_only_shared_shell_initializers(
    source_folder: str | Path,
    model_paths: Iterable[str | Path],
    primary_model: onnx.ModelProto,
    shell_prefixes: tuple[str, ...],
    *,
    min_shared_elements: int = 1024,
    precision_free_prefix: str = "InsertedPrecisionFreeCast_",
) -> dict[str, TensorProto]:
    """Materialize external shell tensors absent from the replacement donor.

    A structure-only target can otherwise retain an old shared-file offset that
    remains in bounds but points to unrelated bytes in the rebuilt blob.
    """
    source_folder = Path(source_folder)
    primary_names = {
        initializer.name for initializer in primary_model.graph.initializer
    }
    blocked_types = {
        TensorProto.UNDEFINED,
        TensorProto.STRING,
        *(
            getattr(TensorProto, name)
            for name in ("UINT4", "INT4", "FLOAT4E2M1")
            if hasattr(TensorProto, name)
        ),
    }

    def is_shell(node: onnx.NodeProto) -> bool:
        return any(
            output.startswith(shell_prefixes)
            or (
                output.startswith(precision_free_prefix)
                and output[len(precision_free_prefix):].startswith(shell_prefixes)
            )
            for output in node.output
        )

    additional: dict[str, TensorProto] = {}
    for raw_model_path in model_paths:
        model_path = Path(raw_model_path)
        target = onnx.load(str(model_path), load_external_data=False)
        shell_inputs = {
            name
            for node in target.graph.node
            if is_shell(node)
            for name in node.input
            if name
        }
        for initializer in target.graph.initializer:
            if (
                initializer.name not in shell_inputs
                or initializer.name in primary_names
                or initializer.data_location != TensorProto.EXTERNAL
                or initializer.data_type in blocked_types
            ):
                continue
            elements = 1
            for dim in initializer.dims:
                elements *= int(dim)
            if elements < min_shared_elements:
                raise RuntimeError(
                    f"Target-only external shell initializer {initializer.name!r} "
                    f"in {model_path.name} has only {elements} elements and cannot "
                    "be silently left at an old shared offset."
                )
            array = numpy_helper.to_array(initializer, base_dir=str(source_folder))
            materialized = numpy_helper.from_array(array, name=initializer.name)
            existing = additional.get(initializer.name)
            if existing is not None and (
                existing.data_type != materialized.data_type
                or tuple(existing.dims) != tuple(materialized.dims)
                or numpy_helper.to_array(existing).tobytes()
                != numpy_helper.to_array(materialized).tobytes()
            ):
                raise RuntimeError(
                    f"Target shells disagree on initializer {initializer.name!r}."
                )
            additional[initializer.name] = materialized
        del target
    return additional


def _is_registered_static_buffer_cast(
    cast: onnx.NodeProto,
    producers: dict[str, onnx.NodeProto],
    static_values: set[str],
) -> bool:
    producer = producers.get(cast.input[0]) if cast.input else None
    return bool(
        cast.input
        and (
            cast.input[0] in static_values
            or (
                producer is not None
                and producer.op_type in ("Slice", "Gather")
                and producer.input
                and producer.input[0] in static_values
            )
        )
    )


def _is_flexible_index_consumer(
    consumer: onnx.NodeProto,
    tensor_name: str,
    source_type: int | None,
    target_type: int | None,
) -> bool:
    if (source_type, target_type) != (TensorProto.INT32, TensorProto.INT64):
        return False
    if consumer.op_type not in _INDEX_TYPE_FLEXIBLE_OPS:
        return False
    return all(
        index == 1
        for index, input_name in enumerate(consumer.input)
        if input_name == tensor_name
    )


def _remove_redundant_casts_from_graph(
    graph: onnx.GraphProto,
) -> tuple[int, int, int]:
    simplified = 0
    removed = 0
    protected = 0
    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                nested_simplified, nested_removed, nested_protected = (
                    _remove_redundant_casts_from_graph(attr.g)
                )
                simplified += nested_simplified
                removed += nested_removed
                protected += nested_protected
            for subgraph in attr.graphs:
                nested_simplified, nested_removed, nested_protected = (
                    _remove_redundant_casts_from_graph(subgraph)
                )
                simplified += nested_simplified
                removed += nested_removed
                protected += nested_protected

    consumers: dict[str, list[onnx.NodeProto]] = {}
    producers = {
        output: node
        for node in graph.node
        for output in node.output
        if output
    }
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    public_outputs = {value.name for value in graph.output}
    tensor_types = _authoritative_tensor_types(graph)
    static_values = {initializer.name for initializer in graph.initializer}
    static_values.update(
        output
        for node in graph.node
        if node.op_type == "Constant"
        for output in node.output
        if output
    )
    removable_outputs: set[str] = set()

    for cast in graph.node:
        if not (
            cast.domain in ("", "ai.onnx")
            and cast.op_type == "Cast"
            and len(cast.input) == 1
            and len(cast.output) == 1
        ):
            continue
        if _is_registered_static_buffer_cast(cast, producers, static_values):
            protected += 1
            continue

        source_type = tensor_types.get(cast.input[0])
        target_type = _node_int_attribute(cast, "to")
        cast_consumers = consumers.get(cast.output[0], [])
        rewired = False
        identity_cast = source_type is not None and source_type == target_type
        exact_widening = (source_type, target_type) in _EXACT_WIDENING_CASTS
        for consumer in cast_consumers:
            bypass = identity_cast
            if consumer.op_type in ("Shape", "Size"):
                bypass = True
            elif consumer.op_type in ("ArgMax", "ArgMin") and exact_widening:
                bypass = True
            elif _is_flexible_index_consumer(
                consumer, cast.output[0], source_type, target_type
            ):
                bypass = True
            if not bypass:
                continue
            for index, input_name in enumerate(consumer.input):
                if input_name == cast.output[0]:
                    consumer.input[index] = cast.input[0]
                    rewired = True

        remaining_consumers = [
            consumer
            for consumer in cast_consumers
            if cast.output[0] in consumer.input
        ]
        if (
            identity_cast
            and cast.output[0] in public_outputs
            and cast.input[0] not in public_outputs
            and not remaining_consumers
        ):
            source_producer = producers.get(cast.input[0])
            if source_producer is not None and sum(
                output_name == cast.input[0]
                for node in graph.node
                for output_name in node.output
            ) == 1:
                for index, output_name in enumerate(source_producer.output):
                    if output_name == cast.input[0]:
                        source_producer.output[index] = cast.output[0]
                for node in consumers.get(cast.input[0], ()):
                    if cast.output[0] in node.output:
                        continue
                    for index, input_name in enumerate(node.input):
                        if input_name == cast.input[0]:
                            node.input[index] = cast.output[0]
                for value in graph.value_info:
                    if value.name == cast.input[0]:
                        value.name = cast.output[0]
                for annotation in graph.quantization_annotation:
                    if annotation.tensor_name == cast.input[0]:
                        annotation.tensor_name = cast.output[0]
                removable_outputs.add(cast.output[0])
        if (
            cast.output[0] not in public_outputs
            and not remaining_consumers
            and (rewired or not cast_consumers)
        ):
            removable_outputs.add(cast.output[0])
        if rewired or cast.output[0] in removable_outputs:
            simplified += 1

    if not removable_outputs:
        return simplified, removed, protected

    retained_nodes = [
        node
        for node in graph.node
        if not (
            node.domain in ("", "ai.onnx")
            and node.op_type == "Cast"
            and len(node.output) == 1
            and node.output[0] in removable_outputs
        )
    ]
    del graph.node[:]
    graph.node.extend(retained_nodes)
    retained_info = [
        value for value in graph.value_info if value.name not in removable_outputs
    ]
    del graph.value_info[:]
    graph.value_info.extend(retained_info)
    retained_annotations = [
        annotation
        for annotation in graph.quantization_annotation
        if annotation.tensor_name not in removable_outputs
    ]
    del graph.quantization_annotation[:]
    graph.quantization_annotation.extend(retained_annotations)
    return simplified, removed + len(removable_outputs), protected


def remove_redundant_casts(model: onnx.ModelProto) -> int:
    """Remove provable no-op Casts and bypass type-insensitive Cast edges."""
    simplified, removed, protected = _remove_redundant_casts_from_graph(model.graph)
    if not simplified:
        return 0
    metadata = {prop.key: prop for prop in model.metadata_props}
    updates = {
        "optimizer_redundant_casts_simplified": str(simplified),
        "optimizer_redundant_cast_nodes_removed": str(removed),
        "optimizer_static_buffer_casts_preserved": str(protected),
    }
    for key, value in updates.items():
        if key in metadata:
            metadata[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    return simplified


def remove_redundant_casts_from_file(model_path: str) -> int:
    """Apply :func:`remove_redundant_casts` without loading sidecars."""
    model = onnx.load(model_path, load_external_data=False)
    simplified = remove_redundant_casts(model)
    if simplified:
        onnx.save(model, model_path)
    del model
    gc.collect()
    return simplified


# Backward-compatible names for external callers of the earlier narrow pass.
remove_redundant_argmax_casts = remove_redundant_casts
remove_redundant_argmax_casts_from_file = remove_redundant_casts_from_file


def _retarget_external_location(model_path: str, new_location: str) -> None:
    model = onnx.load(model_path, load_external_data=False)
    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location == TensorProto.EXTERNAL:
            for entry in tensor.external_data:
                if entry.key == "location":
                    entry.value = new_location
    onnx.save(model, model_path)
    del model
    gc.collect()


def _materialize_constant_tensors_as_initializers(graph) -> int:
    existing_initializers = {initializer.name for initializer in graph.initializer}
    nodes_to_remove = []
    converted = 0

    for node in graph.node:
        for attr in node.attribute:
            if attr.HasField("g"):
                converted += _materialize_constant_tensors_as_initializers(attr.g)
            for subgraph in attr.graphs:
                converted += _materialize_constant_tensors_as_initializers(subgraph)

        if node.op_type != "Constant" or len(node.output) != 1:
            continue

        tensor = None
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                tensor = TensorProto()
                tensor.CopyFrom(attr.t)
                break
        if tensor is None:
            continue

        output_name = node.output[0]
        if output_name in existing_initializers:
            nodes_to_remove.append(node)
            continue

        tensor.name = output_name
        graph.initializer.append(tensor)
        existing_initializers.add(output_name)
        nodes_to_remove.append(node)
        converted += 1

    for node in nodes_to_remove:
        graph.node.remove(node)

    return converted


def _recover_zero_length_external_ranges(
    model: onnx.ModelProto,
    model_path: Path,
) -> int:
    """Recover legacy external ranges whose payload length was serialized as zero.

    The recovery is deliberately fail-closed: only non-empty numeric tensors with
    an explicit zero length are changed, and every inferred range must be in bounds
    without partially overlapping any other declared or inferred tensor range.
    Exact aliases of the same byte range remain valid.
    """
    records_by_path: dict[Path, list[tuple[int, int, TensorProto, bool]]] = {}
    repairs: list[tuple[TensorProto, int]] = []

    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        fields: dict[str, str] = {}
        for entry in tensor.external_data:
            if entry.key in fields:
                raise RuntimeError(
                    f"External tensor {tensor.name!r} in {model_path.name} has "
                    f"duplicate {entry.key!r} fields."
                )
            fields[entry.key] = entry.value
        location = fields.get("location")
        if not location:
            raise RuntimeError(
                f"External tensor {tensor.name!r} in {model_path.name} has no "
                "external-data location."
            )
        offset = int(fields.get("offset", "0"))
        declared_length = (
            None if "length" not in fields else int(fields["length"])
        )
        if offset < 0 or (declared_length is not None and declared_length < 0):
            raise RuntimeError(
                f"External tensor {tensor.name!r} in {model_path.name} has a "
                "negative offset or length."
            )

        needs_inferred_length = declared_length is None or declared_length == 0
        expected_length = (
            _tensor_raw_data_length(tensor)
            if needs_inferred_length
            else declared_length
        )
        recover = declared_length == 0 and expected_length > 0
        effective_length = expected_length if needs_inferred_length else declared_length
        data_path = _safe_external_data_path(model_path, location)
        if offset + effective_length > data_path.stat().st_size:
            raise RuntimeError(
                f"External tensor {tensor.name!r} in {model_path.name} has a "
                "truncated payload."
            )
        records_by_path.setdefault(data_path.resolve(), []).append(
            (offset, effective_length, tensor, recover)
        )
        if recover:
            repairs.append((tensor, expected_length))

    for data_path, records in records_by_path.items():
        aliases: dict[tuple[int, int], list[tuple[TensorProto, bool]]] = {}
        for offset, length, tensor, recover in records:
            aliases.setdefault((offset, length), []).append((tensor, recover))
        intervals = sorted(
            (
                offset,
                length,
                group[0][0],
                any(recover for _, recover in group),
            )
            for (offset, length), group in aliases.items()
        )
        furthest_end = -1
        furthest_tensor = None
        furthest_recovered_end = -1
        furthest_recovered_tensor = None
        for offset, length, tensor, recover in intervals:
            end = offset + length
            other_tensor = None
            if recover and offset < furthest_end:
                other_tensor = furthest_tensor
            elif offset < furthest_recovered_end:
                other_tensor = furthest_recovered_tensor
            if other_tensor is not None:
                raise RuntimeError(
                    f"Cannot recover zero-length external tensor {tensor.name!r} "
                    f"in {model_path.name}: inferred range [{offset}, {end}) "
                    f"partially overlaps {other_tensor.name!r} in {data_path.name}."
                )
            if end > furthest_end:
                furthest_end = end
                furthest_tensor = tensor
            if recover and end > furthest_recovered_end:
                furthest_recovered_end = end
                furthest_recovered_tensor = tensor

    for tensor, expected_length in repairs:
        length_entries = [entry for entry in tensor.external_data if entry.key == "length"]
        if len(length_entries) != 1:
            raise RuntimeError(
                f"External tensor {tensor.name!r} in {model_path.name} does not "
                "have exactly one recoverable length field."
            )
        length_entries[0].value = str(expected_length)
    return len(repairs)


def resave(src_path: str, dst_path: str, external: bool) -> None:
    source_path = Path(src_path)
    model = onnx.load(src_path, load_external_data=False)
    recovered_ranges = _recover_zero_length_external_ranges(model, source_path)
    if recovered_ranges:
        print(
            f"  Recovered {recovered_ranges} legacy zero-length external "
            "tensor range(s) from dtype and shape."
        )
    onnx.external_data_helper.load_external_data_for_model(
        model,
        str(source_path.resolve().parent),
    )
    converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
    if converted_constants:
        print(f"  Materialized {converted_constants} Constant tensor nodes as initializers before save.")
    _save_model(model, dst_path, external)
    del model
    gc.collect()


def read_onnx_metadata(model_path: str) -> dict[str, str]:
    """Return a model's ``metadata_props`` as a plain dict (external weights left on disk)."""
    model = onnx.load(model_path, load_external_data=False)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    del model
    gc.collect()
    return metadata


def write_onnx_metadata(model_path: str, metadata: dict[str, str]) -> None:
    """Add/overwrite ``metadata_props`` on an ONNX file in place, preserving external-weight sidecars.

    ``load_external_data=False`` keeps any ``*.data`` sidecar untouched (only the graph proto + metadata
    are rewritten), so updating the metadata carrier is safe for inline and external-data models. A no-op
    when the source model carried no metadata.
    """
    if not metadata:
        return
    model = onnx.load(model_path, load_external_data=False)
    existing = {prop.key: prop for prop in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = value
        else:
            model.metadata_props.add(key=key, value=value)
    onnx.save(model, model_path)
    del model
    gc.collect()


def producer_ancestry_node_names(
    model_path: str | Path,
    boundary_tensor: str,
    *,
    graph_label: str = "ONNX graph",
) -> list[str]:
    """Return the ordered producer ancestry of a strict tensor boundary."""
    model = onnx.load(str(model_path), load_external_data=False)
    producers: dict[str, onnx.NodeProto] = {}
    node_order: dict[int, int] = {}
    for index, node in enumerate(model.graph.node):
        node_order[id(node)] = index
        for output in node.output:
            if not output:
                continue
            if output in producers:
                raise RuntimeError(
                    f"{graph_label} has duplicate producer for {output!r}."
                )
            producers[output] = node
    if boundary_tensor not in producers:
        raise RuntimeError(
            f"Cannot find {graph_label} boundary {boundary_tensor!r}."
        )

    selected: dict[int, onnx.NodeProto] = {}
    pending = [boundary_tensor]
    while pending:
        node = producers.get(pending.pop())
        if node is None or id(node) in selected:
            continue
        if not node.name:
            raise RuntimeError(
                f"Unnamed {node.op_type} node found in {graph_label} ancestry."
            )
        selected[id(node)] = node
        pending.extend(name for name in node.input if name)
    names = [
        node.name
        for node in sorted(selected.values(), key=lambda item: node_order[id(item)])
    ]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{graph_label} ancestry contains duplicate node names.")
    return names


def _multiply_shape_terms(left, right):
    coefficient = left[0] * right[0]
    powers = dict(left[1])
    for symbol, exponent in right[1].items():
        powers[symbol] = powers.get(symbol, 0) + exponent
        if powers[symbol] == 0:
            del powers[symbol]
    return coefficient, powers


def _resolve_reshape_shape(shape: tuple[int, ...], input_terms: list):
    result, inferred_index = [], None
    known_product = (Fraction(1), {})
    for index, dimension in enumerate(shape):
        if dimension == -1:
            if inferred_index is not None:
                return None
            inferred_index = index
            result.append(None)
            continue
        if dimension == 0:
            if index >= len(input_terms):
                return None
            term = input_terms[index]
        elif dimension > 0:
            term = (Fraction(dimension), {})
        else:
            return None
        result.append(term)
        known_product = _multiply_shape_terms(known_product, term)

    if inferred_index is not None:
        inverse = (
            1 / known_product[0],
            {symbol: -exponent for symbol, exponent in known_product[1].items()},
        )
        result[inferred_index] = _multiply_shape_terms(
            (Fraction(1), {"size": 1}), inverse
        )
    return result


def _compose_reshape_shapes(
    first_shape: tuple[int, ...],
    second_shape: tuple[int, ...],
) -> tuple[int, ...] | None:
    input_terms = [
        (Fraction(1), {f"dim_{index}": 1})
        for index in range(max(len(first_shape), len(second_shape)))
    ]
    middle_terms = _resolve_reshape_shape(first_shape, input_terms)
    final_terms = (
        _resolve_reshape_shape(second_shape, middle_terms)
        if middle_terms is not None
        else None
    )
    if final_terms is None:
        return None

    composed, unresolved = [], []
    for index, (coefficient, powers) in enumerate(final_terms):
        if not powers and coefficient.denominator == 1 and coefficient > 0:
            composed.append(coefficient.numerator)
        elif coefficient == 1 and powers == {f"dim_{index}": 1}:
            composed.append(0)
        else:
            unresolved.append(index)
            composed.append(None)
    if len(unresolved) > 1:
        return None
    if unresolved:
        composed[unresolved[0]] = -1

    candidate = tuple(composed)
    return (
        candidate
        if _resolve_reshape_shape(candidate, input_terms) == final_terms
        else None
    )


def _constant_int_values(
    name: str,
    producer: dict[str, onnx.NodeProto],
    initializers: dict[str, TensorProto],
) -> tuple[int, ...] | None:
    tensor = initializers.get(name)
    if tensor is None:
        node = producer.get(name)
        if node is None or node.op_type != "Constant":
            return None
        tensor = next(
            (attribute.t for attribute in node.attribute if attribute.name == "value"),
            None,
        )
    if tensor is None:
        return None
    try:
        values = numpy_helper.to_array(tensor)
    except Exception:
        return None
    if values.dtype.kind not in "iu":
        return None
    return tuple(int(value) for value in values.reshape(-1))


def _fusion_name_factory(graph: onnx.GraphProto, prefix: str):
    used = {value.name for value in (*graph.input, *graph.output, *graph.value_info)}
    used.update(initializer.name for initializer in graph.initializer)
    for node in graph.node:
        if node.name:
            used.add(node.name)
        used.update(name for name in (*node.input, *node.output) if name)

    def make(suffix: str) -> str:
        base = f"{prefix}{suffix}"
        candidate = base
        index = 1
        while candidate in used:
            candidate = f"{base}_{index}"
            index += 1
        used.add(candidate)
        return candidate

    return make


def _dead_code_elimination(graph: onnx.GraphProto) -> None:
    graph_outputs = {value.name for value in graph.output}
    changed = True
    while changed:
        used = set(graph_outputs)
        for node in graph.node:
            used.update(name for name in node.input if name)
        kept = [
            node
            for node in graph.node
            if not node.output or any(output in used for output in node.output)
        ]
        changed = len(kept) != len(graph.node)
        if changed:
            del graph.node[:]
            graph.node.extend(kept)


def fuse_consecutive_reshapes_graph(graph: onnx.GraphProto) -> int:
    """Fuse constant-shape Reshape pairs only when composition is provably exact."""
    graph_outputs = {value.name for value in graph.output}
    make_name = _fusion_name_factory(graph, "reshape_fusion_")
    removed_values: set[str] = set()
    fused = 0

    while True:
        producer = {
            output: node
            for node in graph.node
            for output in node.output
            if output
        }
        consumers: dict[str, list[onnx.NodeProto]] = {}
        for node in graph.node:
            for name in node.input:
                if name:
                    consumers.setdefault(name, []).append(node)
        initializers = {
            initializer.name: initializer for initializer in graph.initializer
        }
        replacement = None

        for second in graph.node:
            if second.op_type != "Reshape" or len(second.input) < 2:
                continue
            first = producer.get(second.input[0])
            if first is None or first.op_type != "Reshape" or len(first.input) < 2:
                continue
            middle = first.output[0]
            if middle in graph_outputs or len(consumers.get(middle, ())) != 1:
                continue
            if any(
                next(
                    (
                        attribute.i
                        for attribute in node.attribute
                        if attribute.name == "allowzero"
                    ),
                    0,
                )
                for node in (first, second)
            ):
                continue
            first_shape = _constant_int_values(
                first.input[1], producer, initializers
            )
            second_shape = _constant_int_values(
                second.input[1], producer, initializers
            )
            if first_shape is None or second_shape is None:
                continue
            composed_shape = _compose_reshape_shapes(first_shape, second_shape)
            if composed_shape is None:
                continue
            replacement = first, second, composed_shape, second_shape
            break

        if replacement is None:
            break
        first, second, composed_shape, second_shape = replacement
        second.input[0] = first.input[0]
        if composed_shape != second_shape:
            shape_name = make_name(f"shape_{fused}")
            graph.initializer.append(
                numpy_helper.from_array(
                    np.asarray(composed_shape, dtype=np.int64), name=shape_name
                )
            )
            second.input[1] = shape_name
        removed_values.update(first.output)
        kept = [node for node in graph.node if id(node) != id(first)]
        del graph.node[:]
        graph.node.extend(kept)
        fused += 1

    if fused:
        _dead_code_elimination(graph)
        _drop_unused_graph_initializers(graph)
        kept_info = [
            value for value in graph.value_info if value.name not in removed_values
        ]
        del graph.value_info[:]
        graph.value_info.extend(kept_info)
    return fused


def fuse_consecutive_reshapes(model_path: str | Path) -> int:
    """Apply the semantics-safe Reshape fusion without loading external weights."""
    model = onnx.load(str(model_path), load_external_data=False)
    fused = fuse_consecutive_reshapes_graph(model.graph)
    if fused:
        onnx.save(model, str(model_path))
    del model
    gc.collect()
    return fused


def _slim_skip_fusion_patterns(config: OptimizerConfig) -> list[str] | None:
    patterns = list(config.slim_skip_fusion_patterns or ())
    if config.safe_reshape_fusion and "EliminationReshape" not in patterns:
        patterns.append("EliminationReshape")
    return patterns or None


def run_onnxslim(model_path: str, external: bool, config: OptimizerConfig, no_shape_infer: bool) -> None:
    def _slim() -> None:
        slim(
            model=model_path,
            output_model=model_path,
            no_shape_infer=no_shape_infer,
            skip_fusion_patterns=_slim_skip_fusion_patterns(config),
            skip_optimizations=config.slim_skip_optimizations,
            size_threshold=config.slim_size_threshold,
            save_as_external_data=external,
            verbose=False,
        )
        if config.safe_reshape_fusion:
            fused = fuse_consecutive_reshapes(model_path)
            if fused:
                print(
                    f"  Fused {fused} semantics-safe consecutive Reshape pair(s)."
                )

    data_path = model_path + ".data"
    if not external or not os.path.exists(data_path):
        _slim()
        return

    data_stash_path = model_path + ".stash.data"
    model_stash_path = model_path + ".stash.onnx"
    for stash_path in (data_stash_path, model_stash_path):
        if os.path.exists(stash_path):
            os.remove(stash_path)
    shutil.copy2(model_path, model_stash_path)
    os.replace(data_path, data_stash_path)
    try:
        _retarget_external_location(model_path, os.path.basename(data_stash_path))
        _slim()
    except BaseException:
        Path(model_path).unlink(missing_ok=True)
        Path(data_path).unlink(missing_ok=True)
        os.replace(model_stash_path, model_path)
        os.replace(data_stash_path, data_path)
        raise
    else:
        Path(model_stash_path).unlink(missing_ok=True)
        Path(data_stash_path).unlink(missing_ok=True)


def build_fusion_options(config: OptimizerConfig):
    if not config.optimizer_fusion_options:
        return None
    from onnxruntime.transformers.fusion_options import FusionOptions

    options = FusionOptions(config.optimizer_model_type)
    for key, value in config.optimizer_fusion_options.items():
        setattr(options, key, value)
    return options


def _deduplicate_node_names(graph) -> int:
    used_names, next_name_suffix, used_values, next_value_suffix, remap, renamed = set(), {}, set(), {}, {}, 0
    used_values.update(i.name for i in graph.input)
    used_values.update(i.name for i in graph.initializer)
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]

        name = node.name
        if name:
            if name not in used_names:
                used_names.add(name)
            else:
                suffix = next_name_suffix.get(name, 1)
                while f"{name}_{suffix}" in used_names:
                    suffix += 1
                node.name = f"{name}_{suffix}"
                used_names.add(node.name)
                next_name_suffix[name] = suffix + 1
                renamed += 1

        for i, output in enumerate(node.output):
            if not output:
                continue
            if output not in used_values:
                used_values.add(output)
                continue
            suffix = next_value_suffix.get(output, 1)
            while f"{output}_{suffix}" in used_values:
                suffix += 1
            new_output = f"{output}_{suffix}"
            node.output[i] = new_output
            used_values.add(new_output)
            next_value_suffix[output] = suffix + 1
            remap[output] = new_output
            renamed += 1
    return renamed


def _resolve_int(value: IntValue, src_path: str) -> int:
    return int(value(src_path)) if callable(value) else int(value)


def _resolve_nodes(selector: NodeSelector, src_path: str) -> list[str] | None:
    nodes = selector(src_path) if callable(selector) else selector
    return nodes or None


def optimize_onnx_model(model_path: str, rp: ResolvedPlan, config: OptimizerConfig, src_path: str,
                        use_fp16: bool, external: bool, keep_io_types: bool) -> None:
    from onnxruntime.transformers.optimizer import optimize_model

    model = optimize_model(
        model_path,
        use_gpu=False,
        opt_level=config.optimizer_level if rp.opt_level is None else rp.opt_level,
        num_heads=_resolve_int(rp.num_heads, src_path),
        hidden_size=_resolve_int(rp.hidden_size, src_path),
        optimization_options=build_fusion_options(config),
        model_type=config.optimizer_model_type,
        only_onnxruntime=config.optimizer_only_onnxruntime,
        verbose=False,
    )
    if use_fp16:
        # ``nodes_to_exclude`` is the per-plan conversion boundary for F16 too.
        # Resolve it against the current post-slim graph so callable selectors see
        # the exact node names that the converter is about to process.
        node_block_list = list(_resolve_nodes(rp.nodes_to_exclude, model_path) or ())
        node_block_list.extend(config.f16_node_block_list or ())
        node_block_list = list(dict.fromkeys(node_block_list)) or None
        model.convert_float_to_float16(
            keep_io_types=keep_io_types,
            force_fp16_initializers=rp.f16_force_initializers,
            use_symbolic_shape_infer=config.shape_infer,
            max_finite_val=config.f16_max_finite_val,
            min_positive_val=config.f16_min_positive_val,
            op_block_list=config.f16_op_block_list,
            node_block_list=node_block_list,
        )
        renamed = _deduplicate_node_names(model.model.graph)
        if renamed:
            print(f"  Renamed {renamed} duplicate node names after float16 conversion.")
    model.save_model_to_file(model_path, use_external_data_format=external, convert_attribute=True)
    del model
    gc.collect()


def upgrade_opset_version(model_path: str, version: int, external: bool) -> None:
    print(f"  Upgrading opset to {version}...")
    model = onnx.version_converter.convert_version(onnx.load(model_path), version)
    _save_model(model, model_path, external)
    del model
    gc.collect()


def build_weight_only_config(rp: ResolvedPlan, bits: int):
    algo = rp.algo
    op_types, axes = list(rp.op_types), list(rp.axes)
    quant_axes = tuple(zip(op_types, axes))
    quant_format = _QUANT_FORMATS[rp.quant_format]
    common = {
        "quant_format": quant_format,
        "op_types_to_quantize": tuple(op_types),
    }
    if algo == "AFFINE_REFINE_V2":
        return None, quant_axes, algo
    if algo == "RTN":
        cfg = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(**common)
    elif algo == "HQQ":
        cfg = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits, block_size=rp.block_size, axis=axes[0], quant_axes=quant_axes, **common,
        )
    elif algo == "k_quant":
        cfg = matmul_nbits_quantizer.KQuantWeightOnlyQuantConfig(**common)
    else:
        cfg = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
            block_size=rp.block_size,
            is_symmetric=rp.symmetric,
            accuracy_level=rp.accuracy_level,
            quant_axes=quant_axes,
            **common,
        )
    cfg.bits = bits
    return cfg, quant_axes, algo


@dataclass
class AffineRefineStats:
    blocks: int = 0
    improved_blocks: int = 0
    seed_error: float = 0.0
    refined_error: float = 0.0

    def add(self, other: "AffineRefineStats") -> None:
        self.blocks += other.blocks
        self.improved_blocks += other.improved_blocks
        self.seed_error += other.seed_error
        self.refined_error += other.refined_error


_K_QUANT_SEARCH_OFFSETS = np.asarray(
    tuple(-1.0 + 0.1 * index for index in range(20)), dtype=np.float32
)
_K_QUANT_FINAL_CHUNK_VALUES = 262144


def quant_tensor_k_quant_cpu(
    data: np.ndarray,
    num_bits: int = 4,
    group_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize rows with ORT's k-quant objective using bounded CPU buffers."""
    if num_bits < 1 or group_size < 1:
        raise ValueError("num_bits and group_size must be positive.")
    values = np.ascontiguousarray(np.asarray(data).reshape(-1, group_size), dtype=np.float32)
    block_count = values.shape[0]
    maxq = (1 << num_bits) - 1
    maxq_float = np.float32(maxq)
    quantized = np.empty_like(values)
    scratch = np.empty_like(values)
    weighted_quantized = np.empty_like(values)
    np.multiply(values, values, out=scratch)
    rms = np.sqrt(np.sum(scratch, axis=1, dtype=np.float32) / np.float32(group_size))
    weights = np.abs(values)
    weights += rms[:, None]
    minimum = np.min(values, axis=1)
    maximum = np.max(values, axis=1)
    span = maximum - minimum
    constant = span == 0.0
    sum_weights = np.sum(weights, axis=1, dtype=np.float32)
    sum_weighted_values = np.sum(weights * values, axis=1, dtype=np.float32)
    inverse_scale = np.ones(block_count, dtype=np.float32)
    np.divide(maxq_float, span, out=inverse_scale, where=span != 0.0)
    best_scale = np.reciprocal(inverse_scale)
    best_minimum = minimum.copy()
    quantized[:] = np.clip(np.rint((values - best_minimum[:, None]) * inverse_scale[:, None]), 0, maxq_float)
    best_error = np.sum(weights * (quantized * best_scale[:, None] + best_minimum[:, None] - values) ** 2, axis=1)
    for offset in _K_QUANT_SEARCH_OFFSETS:
        span = maximum - best_minimum
        candidate_inverse = np.ones(block_count, dtype=np.float32)
        np.divide(maxq_float + offset, span, out=candidate_inverse, where=span != 0.0)
        quantized[:] = np.clip(
            np.rint((values - best_minimum[:, None]) * candidate_inverse[:, None]), 0, maxq_float
        )
        np.multiply(weights, quantized, out=weighted_quantized)
        sum_l = np.sum(weighted_quantized, axis=1, dtype=np.float32)
        sum_l2 = np.sum(weighted_quantized * quantized, axis=1, dtype=np.float32)
        sum_xl = np.sum(weighted_quantized * values, axis=1, dtype=np.float32)
        determinant = sum_weights * sum_l2 - sum_l * sum_l
        valid = (determinant != 0.0) & np.isfinite(determinant)
        candidate_scale = np.divide(
            sum_weights * sum_xl - sum_weighted_values * sum_l,
            determinant,
            out=np.zeros_like(determinant),
            where=valid,
        )
        candidate_minimum = np.divide(
            sum_l2 * sum_weighted_values - sum_l * sum_xl,
            determinant,
            out=np.zeros_like(determinant),
            where=valid,
        )
        valid &= (candidate_scale > 0.0) & np.isfinite(candidate_scale) & np.isfinite(candidate_minimum)
        candidate_error = np.sum(
            weights * (quantized * candidate_scale[:, None] + candidate_minimum[:, None] - values) ** 2,
            axis=1,
        )
        improved = valid & (candidate_error < best_error)
        best_error[improved] = candidate_error[improved]
        best_scale[improved] = candidate_scale[improved]
        best_minimum[improved] = candidate_minimum[improved]
    if np.any(constant):
        constant_value = values[constant, 0]
        positive = constant_value > 0.0
        negative = constant_value < 0.0
        best_scale[constant] = np.where(
            positive,
            constant_value / maxq_float,
            np.where(negative, -constant_value / maxq_float, np.float32(1.0)),
        )
        best_minimum[constant] = np.where(
            negative, constant_value, np.float32(0.0)
        )
    zero_point = np.clip(np.rint(-best_minimum / best_scale), 0, maxq).astype(np.uint8)
    rows_per_chunk = max(1, _K_QUANT_FINAL_CHUNK_VALUES // group_size)
    for start in range(0, block_count, rows_per_chunk):
        end = min(start + rows_per_chunk, block_count)
        quantized[start:end] = np.clip(
            np.rint(values[start:end] / best_scale[start:end, None] + zero_point[start:end, None]),
            0,
            maxq,
        )
    return quantized, best_scale.reshape(-1, 1), zero_point.reshape(-1, 1)


def _iter_affine_row_chunks(values: np.ndarray, block_size: int, max_blocks: int):
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    padded_columns = block_count * block_size
    rows_per_chunk = max(1, max_blocks // block_count)
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = np.ascontiguousarray(values[row_start:row_end], dtype=np.float32)
        if padded_columns != columns:
            chunk = np.pad(chunk, ((0, 0), (0, padded_columns - columns)))
        yield row_start * block_count, row_end * block_count, chunk.reshape(-1, block_size)


@lru_cache(maxsize=None)
def _affine_v2_numba_kernel(thread_count: int):
    """Build the optional parallel refinement kernel lazily."""
    try:
        from numba import njit, prange, set_num_threads
    except ImportError:
        return None
    if thread_count < 1:
        raise ValueError("AFFINE_REFINE_V2 Numba thread count must be positive.")
    set_num_threads(thread_count)

    @njit(parallel=True, nogil=True, cache=True)
    def refine(
        weight,
        quantized,
        scales,
        zero_points,
        clip_ratios,
        iterations,
        tolerance,
        tiny,
        symmetric,
        maxq,
        midpoint,
        sweep_limit,
    ):
        block_count, width = weight.shape
        baseline_errors = np.empty(block_count, dtype=np.float32)
        refined_errors = np.empty(block_count, dtype=np.float32)
        improved = np.zeros(block_count, dtype=np.bool_)
        candidate_codes = np.empty((block_count, width), dtype=np.uint8)
        max_code = int(maxq)
        for block in prange(block_count):
            sum_squares = np.float32(0.0)
            positive_max = np.float32(0.0)
            negative_max = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block, column])
                sum_squares += value * value
                positive_max = max(positive_max, value)
                negative_max = max(negative_max, -value)
            rms = np.float32(np.sqrt(sum_squares / np.float32(width)))
            seed_scale = np.float32(scales[block])
            seed_zp = int(zero_points[block])
            baseline_plain = np.float32(0.0)
            baseline_weighted = np.float32(0.0)
            for column in range(width):
                value = np.float32(weight[block, column])
                residual = value - seed_scale * (
                    np.float32(quantized[block, column]) - np.float32(seed_zp)
                )
                squared = residual * residual
                baseline_plain += squared
                baseline_weighted += (rms + np.abs(value)) * squared
            local_plain = baseline_plain
            weighted_bound = tolerance * baseline_weighted
            if symmetric:
                zp_low = midpoint
                zp_high = midpoint
            elif max_code + 1 <= sweep_limit:
                zp_low = 0
                zp_high = max_code
            else:
                zp_low = max(0, seed_zp - sweep_limit // 2)
                zp_high = zp_low + sweep_limit - 1
                if zp_high > max_code:
                    zp_high = max_code
                    zp_low = max(0, zp_high - sweep_limit + 1)
            for zero_point_int in range(zp_low, zp_high + 1):
                positive_scale = np.float32(0.0)
                negative_scale = np.float32(0.0)
                if zero_point_int < max_code:
                    positive_scale = positive_max / np.float32(max_code - zero_point_int)
                if zero_point_int > 0:
                    negative_scale = negative_max / np.float32(zero_point_int)
                coverage_scale = max(positive_scale, negative_scale)
                if coverage_scale <= tiny:
                    coverage_scale = np.float32(1.0)
                for start_index in range(clip_ratios.size + 1):
                    candidate_scale = (
                        seed_scale
                        if start_index == 0
                        else coverage_scale * clip_ratios[start_index - 1]
                    )
                    zero_point = np.float32(zero_point_int)
                    for _ in range(iterations):
                        denominator = np.float32(0.0)
                        numerator = np.float32(0.0)
                        for column in range(width):
                            value = np.float32(weight[block, column])
                            code = np.rint(value / candidate_scale + zero_point)
                            code = min(np.float32(maxq), max(np.float32(0.0), code))
                            centered = code - zero_point
                            denominator += centered * centered
                            numerator += centered * value
                        if denominator <= tiny:
                            break
                        fitted = numerator / denominator
                        if not np.isfinite(fitted) or fitted <= tiny or fitted == candidate_scale:
                            break
                        candidate_scale = fitted
                    candidate_plain = np.float32(0.0)
                    candidate_weighted = np.float32(0.0)
                    for column in range(width):
                        value = np.float32(weight[block, column])
                        code = np.rint(value / candidate_scale + zero_point)
                        code = min(np.float32(maxq), max(np.float32(0.0), code))
                        candidate_codes[block, column] = np.uint8(code)
                        residual = value - candidate_scale * (code - zero_point)
                        squared = residual * residual
                        candidate_plain += squared
                        candidate_weighted += (rms + np.abs(value)) * squared
                    if candidate_plain < local_plain and candidate_weighted <= weighted_bound:
                        local_plain = candidate_plain
                        scales[block] = candidate_scale
                        zero_points[block] = np.uint8(zero_point_int)
                        for column in range(width):
                            quantized[block, column] = candidate_codes[block, column]
            baseline_errors[block] = baseline_plain
            refined_errors[block] = local_plain
            improved[block] = local_plain < baseline_plain
        return baseline_errors, refined_errors, improved

    return refine


def _affine_refine_v2_rows(
    data: np.ndarray,
    block_size: int,
    symmetric: bool,
    bits: int,
    config: OptimizerConfig,
    *,
    allow_arbitrary_block_size: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AffineRefineStats]:
    """Refine a k-quant seed against plain block MSE with a weighted Pareto guard."""
    values = np.asarray(data)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("AFFINE_REFINE_V2 expects a finite 2-D row matrix.")
    if not allow_arbitrary_block_size and (
        block_size < 16 or block_size > 256 or block_size & (block_size - 1)
    ):
        raise ValueError("AFFINE_REFINE_V2 block_size must be a power of two in [16, 256].")
    if bits not in (4, 8):
        raise ValueError("AFFINE_REFINE_V2 supports 4- or 8-bit weights.")
    if (
        config.affine_v2_seed_iterations < 1
        or config.affine_v2_seed_zp_radius < 0
        or config.affine_v2_iterations < 1
        or config.affine_v2_chunk_blocks < 1
        or config.affine_v2_weighted_tolerance < 0.0
    ):
        raise ValueError("AFFINE_REFINE_V2 iteration/chunk/tolerance settings are invalid.")
    ratios = np.asarray(config.affine_v2_clip_ratios, dtype=np.float32)
    if ratios.ndim != 1 or not ratios.size or np.any((ratios <= 0.0) | (ratios > 1.0)):
        raise ValueError("AFFINE_REFINE_V2 clip ratios must be in (0, 1].")
    maxq = (1 << bits) - 1
    midpoint = 1 << (bits - 1)
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    total_blocks = rows * block_count
    all_q = np.empty((total_blocks, block_size), dtype=np.uint8)
    all_scales = np.empty(total_blocks, dtype=np.float32)
    all_zp = np.empty(total_blocks, dtype=np.uint8)
    stats = AffineRefineStats(blocks=total_blocks)
    tiny = np.finfo(np.float32).tiny
    numba_kernel = _affine_v2_numba_kernel(config.affine_v2_numba_threads)
    for start, end, weight in _iter_affine_row_chunks(values, block_size, config.affine_v2_chunk_blocks):
        if symmetric:
            positive = np.maximum(weight.max(axis=1, keepdims=True), 0.0)
            negative = np.maximum(-weight.min(axis=1, keepdims=True), 0.0)
            seed_scale = np.maximum(positive / (maxq - midpoint), negative / midpoint)
            seed_scale = np.where(seed_scale > tiny, seed_scale, 1.0).astype(np.float32)
            seed_zp = np.full((weight.shape[0], 1), midpoint, dtype=np.uint8)
            seed_q = np.clip(np.rint(weight / seed_scale + midpoint), 0, maxq)
        else:
            seed_q, seed_scale, seed_zp = quant_tensor_k_quant_cpu(weight, bits, block_size)
        seed_q = np.asarray(seed_q, dtype=np.float32)
        seed_scale = np.asarray(seed_scale, dtype=np.float32).reshape(-1, 1)
        seed_zp = np.asarray(seed_zp, dtype=np.uint8).reshape(-1, 1)
        importance = np.sqrt(np.mean(weight * weight, axis=1, keepdims=True)) + np.abs(weight)
        importance = np.where(np.sum(importance, axis=1, keepdims=True) > 0.0, importance, 1.0)
        seed_dequant = seed_scale * (seed_q - seed_zp.astype(np.float32))
        seed_weighted = np.sum(importance * (weight - seed_dequant) ** 2, axis=1)
        for delta in ((0,) if symmetric else range(
            -config.affine_v2_seed_zp_radius,
            config.affine_v2_seed_zp_radius + 1,
        )):
            candidate_zp = np.clip(seed_zp.astype(np.int16) + delta, 0, maxq).astype(np.float32)
            candidate_scale = seed_scale.copy()
            for _ in range(config.affine_v2_seed_iterations):
                candidate_q = np.clip(np.rint(weight / candidate_scale + candidate_zp), 0, maxq)
                centered = candidate_q - candidate_zp
                denominator = np.sum(importance * centered * centered, axis=1, keepdims=True)
                fitted = np.divide(
                    np.sum(importance * centered * weight, axis=1, keepdims=True),
                    denominator,
                    out=candidate_scale.copy(),
                    where=denominator > tiny,
                )
                candidate_scale = np.where(
                    np.isfinite(fitted) & (fitted > tiny), fitted, candidate_scale
                )
            candidate_q = np.clip(np.rint(weight / candidate_scale + candidate_zp), 0, maxq)
            candidate_weighted = np.sum(
                importance * (weight - candidate_scale * (candidate_q - candidate_zp)) ** 2,
                axis=1,
            )
            take = candidate_weighted < seed_weighted
            seed_q[take] = candidate_q[take]
            seed_scale[take] = candidate_scale[take]
            seed_zp[take] = candidate_zp[take].astype(np.uint8)
            seed_weighted[take] = candidate_weighted[take]
        seed_q = seed_q.astype(np.uint8)
        seed_residual = weight - seed_scale * (seed_q.astype(np.float32) - seed_zp)
        seed_plain = np.sum(seed_residual * seed_residual, axis=1)
        weighted_bound = (1.0 + config.affine_v2_weighted_tolerance) * np.sum(
            importance * seed_residual * seed_residual, axis=1
        )
        best_q = seed_q.copy()
        best_scale = seed_scale[:, 0].copy()
        best_zp = seed_zp[:, 0].copy()
        best_plain = seed_plain.copy()
        if numba_kernel is not None:
            baseline_plain, refined_plain, improved = numba_kernel(
                weight,
                best_q,
                best_scale,
                best_zp,
                ratios,
                config.affine_v2_iterations,
                np.float32(1.0 + config.affine_v2_weighted_tolerance),
                tiny,
                symmetric,
                np.float32(maxq),
                np.int64(midpoint),
                np.int64(config.affine_v2_asym_zp_sweep_limit),
            )
            all_q[start:end] = best_q
            all_scales[start:end] = best_scale
            all_zp[start:end] = best_zp
            stats.improved_blocks += int(np.count_nonzero(improved))
            stats.seed_error += float(baseline_plain.sum(dtype=np.float64))
            stats.refined_error += float(refined_plain.sum(dtype=np.float64))
            continue
        positive = np.maximum(weight.max(axis=1, keepdims=True), 0.0)
        negative = np.maximum(-weight.min(axis=1, keepdims=True), 0.0)
        if bits == 8 and not symmetric:
            limit = config.affine_v2_asym_zp_sweep_limit
            if limit < 16:
                raise ValueError("AFFINE_REFINE_V2 asymmetric sweep limit must be at least 16.")
            window_low = np.clip(seed_zp.astype(np.int16) - limit // 2, 0, maxq)
            window_low = np.clip(
                window_low - np.maximum(window_low + limit - 1 - maxq, 0),
                0,
                maxq,
            )
            zero_points = (window_low + offset for offset in range(limit))
        elif symmetric:
            zero_points = (np.full_like(seed_zp, midpoint, dtype=np.float32),)
        else:
            zero_points = (np.full_like(seed_zp, value, dtype=np.float32) for value in range(maxq + 1))
        for zero_point_value in zero_points:
            zero_point = np.asarray(zero_point_value, dtype=np.float32)
            positive_denominator = np.float32(maxq) - zero_point
            pos_scale = np.where(
                positive_denominator > 0.0,
                positive / np.where(positive_denominator > 0.0, positive_denominator, 1.0),
                0.0,
            )
            neg_scale = np.where(
                zero_point > 0.0,
                negative / np.where(zero_point > 0.0, zero_point, 1.0),
                0.0,
            )
            coverage = np.where(np.maximum(pos_scale, neg_scale) > tiny, np.maximum(pos_scale, neg_scale), 1.0)
            for initial in (seed_scale, *(coverage * ratio for ratio in ratios)):
                candidate_scale = initial.copy()
                for _ in range(config.affine_v2_iterations):
                    candidate_q = np.clip(np.rint(weight / candidate_scale + zero_point), 0, maxq)
                    centered = candidate_q - zero_point
                    denominator = np.sum(centered * centered, axis=1, keepdims=True)
                    fitted = np.divide(
                        np.sum(centered * weight, axis=1, keepdims=True),
                        denominator,
                        out=candidate_scale.copy(),
                        where=denominator > tiny,
                    )
                    candidate_scale = np.where(np.isfinite(fitted) & (fitted > tiny), fitted, candidate_scale)
                candidate_q = np.clip(np.rint(weight / candidate_scale + zero_point), 0, maxq)
                residual = weight - candidate_scale * (candidate_q - zero_point)
                plain = np.sum(residual * residual, axis=1)
                weighted = np.sum(importance * residual * residual, axis=1)
                take = (plain < best_plain) & (weighted <= weighted_bound)
                best_q[take] = candidate_q[take].astype(np.uint8)
                best_scale[take] = candidate_scale[take, 0]
                best_zp[take] = zero_point[take, 0].astype(np.uint8)
                best_plain[take] = plain[take]
        all_q[start:end] = best_q
        all_scales[start:end] = best_scale
        all_zp[start:end] = best_zp
        stats.improved_blocks += int(np.count_nonzero(best_plain < seed_plain))
        stats.seed_error += float(seed_plain.sum(dtype=np.float64))
        stats.refined_error += float(best_plain.sum(dtype=np.float64))
    return (
        all_q.reshape(rows, block_count, block_size),
        all_scales.reshape(rows, block_count),
        all_zp.reshape(rows, block_count),
        stats,
    )


def _pack_codes(values: np.ndarray, bits: int, pad_value: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint8)
    if bits == 8:
        return np.ascontiguousarray(values)
    if values.shape[-1] & 1:
        values = np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, 1)], constant_values=pad_value)
    return (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)


def _unique_quant_name(graph: onnx.GraphProto, prefix: str):
    used = {initializer.name for initializer in graph.initializer}
    used.update(name for node in graph.node for name in (*node.input, *node.output) if name)
    counter = 0

    def make(suffix: str) -> str:
        nonlocal counter
        while True:
            candidate = f"{prefix}{counter}_{suffix}"
            counter += 1
            if candidate not in used:
                used.add(candidate)
                return candidate
    return make


def _drop_unused_graph_initializers(graph: onnx.GraphProto) -> None:
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    kept = [initializer for initializer in graph.initializer if initializer.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)


def _ensure_ms_opset(model: onnx.ModelProto) -> None:
    for opset in model.opset_import:
        if opset.domain == "com.microsoft":
            opset.version = max(opset.version, 1)
            return
    model.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def _k_quant_q4_rows(
    data: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the CPU k-quant objective in bounded chunks, including padded K blocks."""
    values = np.asarray(data)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("k_quant expects a finite 2-D row matrix.")
    rows, columns = values.shape
    block_count = (columns + block_size - 1) // block_size
    quantized = np.empty((rows * block_count, block_size), dtype=np.uint8)
    scales = np.empty(rows * block_count, dtype=np.float32)
    zero_points = np.empty(rows * block_count, dtype=np.uint8)
    for start, end, weight in _iter_affine_row_chunks(values, block_size, 65536):
        with np.errstate(divide="ignore", invalid="ignore"):
            chunk_q, chunk_scales, chunk_zero_points = quant_tensor_k_quant_cpu(
                weight, 4, block_size
            )
        quantized[start:end] = np.clip(chunk_q, 0, 15).astype(np.uint8)
        scales[start:end] = np.asarray(chunk_scales, dtype=np.float32).reshape(-1)
        zero_points[start:end] = np.clip(
            np.asarray(chunk_zero_points, dtype=np.int16).reshape(-1), 0, 15
        ).astype(np.uint8)
    return (
        quantized.reshape(rows, block_count, block_size),
        scales.reshape(rows, block_count),
        zero_points.reshape(rows, block_count),
    )


def quantize_k_quant_model(model: onnx.ModelProto, rp: ResolvedPlan) -> int:
    """Replace selected constant MatMuls with portable CPU-generated k-quant Q4 ops."""
    includes = set(rp.nodes_to_include or ())
    excludes = set(rp.nodes_to_exclude or ())
    quantized_matmuls = 0

    def rewrite(graph: onnx.GraphProto) -> None:
        nonlocal quantized_matmuls
        init_map = {initializer.name: initializer for initializer in graph.initializer}
        make = _unique_quant_name(graph, "k_quant_q4_")
        rewritten = []
        replaced = set()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite(subgraph)
            selected = (
                node.op_type == "MatMul"
                and "MatMul" in rp.op_types
                and (not includes or node.name in includes)
                and node.name not in excludes
            )
            weight = init_map.get(node.input[1]) if selected and len(node.input) >= 2 else None
            if weight is None:
                rewritten.append(node)
                continue
            array = numpy_helper.to_array(weight)
            if array.ndim != 2 or array.dtype.kind != "f":
                rewritten.append(node)
                continue
            input_features, output_features = array.shape
            q, scales, zp = _k_quant_q4_rows(array.T, rp.block_size)
            q_name, scale_name, zp_name = make("weight"), make("scales"), make("zero_points")
            graph.initializer.extend([
                numpy_helper.from_array(_pack_codes(q, 4), q_name),
                numpy_helper.from_array(scales.astype(array.dtype), scale_name),
                numpy_helper.from_array(_pack_codes(zp, 4, 8), zp_name),
            ])
            attributes = {
                "K": input_features,
                "N": output_features,
                "bits": 4,
                "block_size": rp.block_size,
            }
            if rp.accuracy_level:
                attributes["accuracy_level"] = rp.accuracy_level
            rewritten.append(helper.make_node(
                "MatMulNBits",
                (node.input[0], q_name, scale_name, zp_name),
                node.output,
                name=(node.name or make("matmul")) + "_K_QUANT_Q4",
                domain="com.microsoft",
                **attributes,
            ))
            replaced.add(weight.name)
            quantized_matmuls += 1
        del graph.node[:]
        graph.node.extend(rewritten)
        _drop_unused_graph_initializers(graph)
        remaining = {initializer.name for initializer in graph.initializer}
        if replaced - remaining:
            kept_inputs = [value for value in graph.input if value.name not in replaced - remaining]
            del graph.input[:]
            graph.input.extend(kept_inputs)

    rewrite(model.graph)
    if quantized_matmuls:
        _ensure_ms_opset(model)
    print(f"  k_quant CPU surgery: {quantized_matmuls} MatMul -> MatMulNBits.")
    return quantized_matmuls


def quantize_affine_v2_model(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
    bits: int,
    config: OptimizerConfig,
) -> AffineRefineStats:
    """Rewrite selected constant MatMul/Gather nodes as V2-refined QOperators."""
    if rp.quant_format != "QOPERATOR" or bits not in (4, 8):
        raise ValueError("AFFINE_REFINE_V2 weight-only quantization requires QOperator Q4/Q8.")
    total = AffineRefineStats()
    quant_axes = dict(zip(rp.op_types, rp.axes))
    counts = {"MatMul": 0, "Gather": 0}
    includes = set(rp.nodes_to_include or ())
    excludes = set(rp.nodes_to_exclude or ())

    def rewrite(graph: onnx.GraphProto) -> None:
        init_map = {initializer.name: initializer for initializer in graph.initializer}
        make = _unique_quant_name(graph, f"affine_refine_v2_q{bits}_")
        rewritten = []
        replaced = set()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite(subgraph)
            selected = node.op_type in rp.op_types and (not includes or node.name in includes) and node.name not in excludes
            weight = None
            if selected and node.op_type == "MatMul" and len(node.input) >= 2:
                weight = init_map.get(node.input[1])
            elif selected and node.op_type == "Gather" and len(node.input) >= 2:
                weight = init_map.get(node.input[0])
            if weight is None:
                rewritten.append(node)
                continue
            array = numpy_helper.to_array(weight)
            if array.ndim != 2 or array.dtype.kind != "f":
                rewritten.append(node)
                continue
            if node.op_type == "MatMul":
                input_features, output_features = array.shape
                q, scales, zp, stats = _affine_refine_v2_rows(
                    array.T, rp.block_size, rp.symmetric, bits, config
                )
                q_name, scale_name, zp_name = make("weight"), make("scales"), make("zero_points")
                graph.initializer.extend([
                    numpy_helper.from_array(_pack_codes(q, bits), q_name),
                    numpy_helper.from_array(scales.astype(array.dtype), scale_name),
                    numpy_helper.from_array(_pack_codes(zp, bits, 1 << (bits - 1)), zp_name),
                ])
                attrs = {"K": input_features, "N": output_features, "bits": bits, "block_size": rp.block_size}
                if rp.accuracy_level:
                    attrs["accuracy_level"] = rp.accuracy_level
                replacement = helper.make_node(
                    "MatMulNBits", (node.input[0], q_name, scale_name, zp_name), node.output,
                    name=(node.name or make("matmul")) + f"_AFFINE_REFINE_V2_Q{bits}",
                    domain="com.microsoft", **attrs,
                )
            else:
                gather_axis = int(helper.get_node_attr_value(node, "axis")) if any(a.name == "axis" for a in node.attribute) else 0
                quant_axis = quant_axes.get("Gather", 1) % array.ndim
                if gather_axis % array.ndim != 0 or quant_axis != array.ndim - 1 or array.shape[-1] % rp.block_size:
                    rewritten.append(node)
                    continue
                q, scales, zp, stats = _affine_refine_v2_rows(
                    array, rp.block_size, rp.symmetric, bits, config
                )
                logical_q = q.reshape(array.shape[0], -1)[:, :array.shape[1]]
                q_name, scale_name, zp_name = make("weight"), make("scales"), make("zero_points")
                if bits == 4:
                    graph.initializer.extend([
                        helper.make_tensor(q_name, TensorProto.UINT4, array.shape, _pack_codes(logical_q.reshape(-1), 4).tobytes(), raw=True),
                        numpy_helper.from_array(scales.astype(array.dtype), scale_name),
                        helper.make_tensor(zp_name, TensorProto.UINT4, zp.shape, _pack_codes(zp.reshape(-1), 4).tobytes(), raw=True),
                    ])
                else:
                    graph.initializer.extend([
                        numpy_helper.from_array(logical_q.astype(np.uint8), q_name),
                        numpy_helper.from_array(scales.astype(array.dtype), scale_name),
                        numpy_helper.from_array(zp.astype(np.uint8), zp_name),
                    ])
                replacement = helper.make_node(
                    "GatherBlockQuantized", (q_name, node.input[1], scale_name, zp_name), node.output,
                    name=(node.name or make("gather")) + f"_AFFINE_REFINE_V2_Q{bits}",
                    domain="com.microsoft", gather_axis=0, quantize_axis=quant_axis,
                    block_size=rp.block_size, bits=bits,
                )
            rewritten.append(replacement)
            replaced.add(weight.name)
            counts[node.op_type] += 1
            total.add(stats)
        del graph.node[:]
        graph.node.extend(rewritten)
        _drop_unused_graph_initializers(graph)
        remaining = {initializer.name for initializer in graph.initializer}
        if replaced - remaining:
            kept_inputs = [value for value in graph.input if value.name not in replaced - remaining]
            del graph.input[:]
            graph.input.extend(kept_inputs)

    rewrite(model.graph)
    if any(counts.values()):
        _ensure_ms_opset(model)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(
        f"  AFFINE_REFINE_V2 surgery: {counts['MatMul']} MatMul, {counts['Gather']} Gather; "
        f"improved {total.improved_blocks}/{total.blocks} blocks, MSE ratio={ratio:.6f}."
    )
    return total


def _partition_k_quant_matmuls(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
    src_path: str,
) -> tuple[list[str], list[str], list[str]]:
    """Partition selected constant MatMuls by k_quant packing support.

    ORT's 4-bit k_quant zero-point packer packs pairs of K blocks and then
    reshapes by output channel. An odd K-block count cannot be represented by
    that layout, so those nodes must use the portable DEFAULT quantizer.
    """
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    included = set(_resolve_nodes(rp.nodes_to_include, src_path) or ())
    excluded = set(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
    all_constant_matmuls = []
    compatible = []
    fallback = []
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2:
            continue
        weight = initializers.get(node.input[1])
        if weight is None or len(weight.dims) != 2:
            continue
        all_constant_matmuls.append(node.name)
        if node.name in excluded or (included and node.name not in included):
            continue
        k_blocks = (int(weight.dims[0]) + rp.block_size - 1) // rp.block_size
        (compatible if k_blocks % 2 == 0 else fallback).append(node.name)
    return all_constant_matmuls, compatible, fallback


def _lower_eligible_gemm_to_matmul_add(graph: onnx.GraphProto) -> int:
    """Lower exact affine Gemms so block/dynamic MatMul quantizers can see them."""
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    eligible = []
    for node in graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Gemm" or len(node.input) < 2:
            continue
        attributes = {attribute.name: attribute for attribute in node.attribute}
        alpha = float(attributes["alpha"].f) if "alpha" in attributes else 1.0
        beta = float(attributes["beta"].f) if "beta" in attributes else 1.0
        trans_a = int(attributes["transA"].i) if "transA" in attributes else 0
        trans_b = int(attributes["transB"].i) if "transB" in attributes else 0
        weight = initializers.get(node.input[1])
        if (
            alpha == 1.0
            and beta == 1.0
            and trans_a == 0
            and trans_b == 1
            and weight is not None
            and len(weight.dims) == 2
        ):
            eligible.append(node)
    if not eligible:
        return 0

    eligible_ids = {id(node) for node in eligible}
    existing_tensor_names = {
        name
        for node in graph.node
        for name in (*node.input, *node.output)
        if name
    } | set(initializers)
    transposed_weights: dict[str, str] = {}

    def unique_name(base: str) -> str:
        candidate = base
        suffix = 1
        while candidate in existing_tensor_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        existing_tensor_names.add(candidate)
        return candidate

    for node in eligible:
        source_name = node.input[1]
        if source_name in transposed_weights:
            continue
        source = initializers[source_name]
        exclusive = all(id(consumer) in eligible_ids for consumer in consumers.get(source_name, ()))
        target_name = source_name if exclusive else unique_name(source_name + "_gemm_transposed")
        array = np.ascontiguousarray(numpy_helper.to_array(source).T)
        transposed = numpy_helper.from_array(array, name=target_name)
        if exclusive:
            source.CopyFrom(transposed)
        else:
            graph.initializer.append(transposed)
            initializers[target_name] = transposed
        transposed_weights[source_name] = target_name

    replacements = {id(node): node for node in eligible}
    rewritten = []
    for index, node in enumerate(graph.node):
        if id(node) not in replacements:
            rewritten.append(node)
            continue
        output_name = node.output[0]
        has_bias = len(node.input) >= 3 and bool(node.input[2])
        matmul_output = unique_name(output_name + "_gemm_matmul") if has_bias else output_name
        node_name = node.name or f"Gemm_{index}"
        rewritten.append(
            onnx.helper.make_node(
                "MatMul",
                (node.input[0], transposed_weights[node.input[1]]),
                (matmul_output,),
                name=node_name,
            )
        )
        if has_bias:
            rewritten.append(
                onnx.helper.make_node(
                    "Add",
                    (matmul_output, node.input[2]),
                    (output_name,),
                    name=node_name + "_Bias",
                )
            )
    del graph.node[:]
    graph.node.extend(rewritten)
    return len(eligible)


def _make_weight_only_quantizer(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
    cfg,
    quant_axes,
    *,
    nodes_to_exclude: list[str] | None,
    nodes_to_include: list[str] | None,
):
    return matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=rp.block_size,
        is_symmetric=rp.symmetric,
        accuracy_level=rp.accuracy_level,
        quant_format=_QUANT_FORMATS[rp.quant_format],
        op_types_to_quantize=tuple(rp.op_types),
        quant_axes=quant_axes,
        algo_config=cfg,
        nodes_to_exclude=nodes_to_exclude,
        nodes_to_include=nodes_to_include,
    )


def _restrict_weight_only_includes(
    model: onnx.ModelProto,
    op_types: tuple[str, ...],
    included: list[str],
    excluded: list[str],
) -> list[str]:
    """Convert ORT's additive include list into a restrictive selection."""
    if not included:
        return excluded
    selected = set(included)
    return list(dict.fromkeys((
        *excluded,
        *(
            node.name
            for node in model.graph.node
            if node.name
            and node.op_type in op_types
            and node.name not in selected
        ),
    )))


def quantize_weight_only(
    src_path: str,
    dst_path: str,
    rp: ResolvedPlan,
    bits: int,
    external: bool,
    config: OptimizerConfig | None = None,
) -> None:
    config = config or OptimizerConfig("", "", {})
    cfg, quant_axes, algo = build_weight_only_config(rp, bits)
    print(
        f"  Quantizing weights ({algo}, {bits}-bit, block={rp.block_size}, "
        f"format={rp.quant_format}, ops={list(rp.op_types)})..."
    )
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
    if converted_constants:
        print(f"  Materialized {converted_constants} Constant tensor nodes as initializers for weight quantization.")
    lowered_gemms = _lower_eligible_gemm_to_matmul_add(model.graph) if "MatMul" in rp.op_types else 0
    if lowered_gemms:
        print(f"  Lowered {lowered_gemms} affine Gemm node(s) to MatMul + Add for weight quantization.")
    if algo == "k_quant":
        rp = copy.copy(rp)
        rp.nodes_to_exclude = _resolve_nodes(rp.nodes_to_exclude, src_path)
        rp.nodes_to_include = _resolve_nodes(rp.nodes_to_include, src_path)
        quantize_k_quant_model(model, rp)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return
    if algo == "AFFINE_REFINE_V2":
        rp = copy.copy(rp)
        rp.nodes_to_exclude = _resolve_nodes(rp.nodes_to_exclude, src_path)
        rp.nodes_to_include = _resolve_nodes(rp.nodes_to_include, src_path)
        quantize_affine_v2_model(model, rp, bits, config)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return
    resolved_excludes = list(_resolve_nodes(rp.nodes_to_exclude, src_path) or ())
    resolved_includes = list(_resolve_nodes(rp.nodes_to_include, src_path) or ())
    resolved_excludes = _restrict_weight_only_includes(
        model,
        tuple(rp.op_types),
        resolved_includes,
        resolved_excludes,
    )
    if algo == "k_quant" and bits == 4 and "MatMul" in rp.op_types:
        all_matmuls, compatible, fallback = _partition_k_quant_matmuls(model, rp, src_path)
        print(
            "  Q4 routing: "
            f"{len(compatible)} k_quant-compatible MatMul(s), "
            f"{len(fallback)} DEFAULT fallback MatMul(s)."
        )
        current_model = model
        quant = None
        if compatible:
            k_quant_excludes = list(dict.fromkeys((
                *resolved_excludes,
                *(name for name in all_matmuls if name not in compatible),
            )))
            quant = _make_weight_only_quantizer(
                current_model,
                rp,
                cfg,
                quant_axes,
                nodes_to_exclude=k_quant_excludes or None,
                nodes_to_include=None,
            )
            quant.process()
            current_model = quant.model.model
        if fallback:
            default_cfg = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
                block_size=rp.block_size,
                is_symmetric=rp.symmetric,
                accuracy_level=rp.accuracy_level,
                quant_axes=quant_axes,
                quant_format=_QUANT_FORMATS[rp.quant_format],
                op_types_to_quantize=tuple(rp.op_types),
                bits=bits,
            )
            quant = _make_weight_only_quantizer(
                current_model,
                rp,
                default_cfg,
                quant_axes,
                nodes_to_exclude=resolved_excludes or None,
                nodes_to_include=fallback,
            )
            quant.process()
        if quant is None:
            _save_model(model, dst_path, external)
            del model
            gc.collect()
            return
    else:
        quant = _make_weight_only_quantizer(
            model,
            rp,
            cfg,
            quant_axes,
            nodes_to_exclude=resolved_excludes or None,
            nodes_to_include=resolved_includes or None,
        )
        quant.process()
    quant.model.topological_sort()
    _save_model(quant.model.model, dst_path, external)
    del model, quant
    gc.collect()


def _quantize_affine_v2_dynamic_model(
    model: onnx.ModelProto,
    rp: ResolvedPlan,
    config: OptimizerConfig,
) -> AffineRefineStats:
    total = AffineRefineStats()
    includes = set(rp.nodes_to_include or ())
    excludes = set(rp.nodes_to_exclude or ())
    count = 0

    def rewrite(graph: onnx.GraphProto) -> None:
        nonlocal count
        init_map = {initializer.name: initializer for initializer in graph.initializer}
        make = _unique_quant_name(graph, "affine_refine_v2_dynamic_")
        rewritten = []
        replaced = set()
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    rewrite(attribute.g)
                for subgraph in attribute.graphs:
                    rewrite(subgraph)
            selected = node.op_type == "MatMul" and (not includes or node.name in includes) and node.name not in excludes
            weight = init_map.get(node.input[1]) if selected and len(node.input) >= 2 else None
            if weight is None:
                rewritten.append(node)
                continue
            array = numpy_helper.to_array(weight)
            if array.ndim != 2 or array.dtype.kind != "f":
                rewritten.append(node)
                continue
            rows = np.ascontiguousarray(array.T if rp.per_channel else array.reshape(1, -1), dtype=np.float32)
            q, scales, zp, stats = _affine_refine_v2_rows(
                rows, rows.shape[1], rp.symmetric, 8, config, allow_arbitrary_block_size=True
            )
            q = q.reshape(rows.shape)
            q = q.T if rp.per_channel else q.reshape(array.shape)
            scales = scales.reshape(-1)
            zp = zp.reshape(-1)
            if rp.dynamic_weight_type == "QINT8":
                q = (q.astype(np.int16) - 128).astype(np.int8)
                zp = (zp.astype(np.int16) - 128).astype(np.int8)
            else:
                q = q.astype(np.uint8)
                zp = zp.astype(np.uint8)
            if not rp.per_channel:
                scales, zp = scales[0], zp[0]
            q_name, scale_name, zp_name = make("weight"), make("scale"), make("zero_point")
            graph.initializer.extend([
                numpy_helper.from_array(q, q_name),
                numpy_helper.from_array(np.asarray(scales, dtype=np.float32), scale_name),
                numpy_helper.from_array(np.asarray(zp), zp_name),
            ])
            rewritten.append(helper.make_node(
                "DynamicQuantizeMatMul", (node.input[0], q_name, scale_name, zp_name), node.output,
                name=(node.name or make("matmul")) + "_AFFINE_REFINE_V2_DYNAMIC",
                domain="com.microsoft",
            ))
            replaced.add(weight.name)
            total.add(stats)
            count += 1
        del graph.node[:]
        graph.node.extend(rewritten)
        _drop_unused_graph_initializers(graph)
        remaining = {initializer.name for initializer in graph.initializer}
        if replaced - remaining:
            kept_inputs = [value for value in graph.input if value.name not in replaced - remaining]
            del graph.input[:]
            graph.input.extend(kept_inputs)

    rewrite(model.graph)
    if count:
        _ensure_ms_opset(model)
    ratio = total.refined_error / total.seed_error if total.seed_error else 1.0
    print(f"  AFFINE_REFINE_V2 dynamic surgery: {count} MatMul; MSE ratio={ratio:.6f}.")
    return total


def quantize_dynamic_int8(
    src_path: str,
    dst_path: str,
    rp: ResolvedPlan,
    external: bool,
    config: OptimizerConfig | None = None,
) -> None:
    config = config or OptimizerConfig("", "", {})
    weight_type = _DYNAMIC_WEIGHT_TYPES[rp.dynamic_weight_type]
    extra_options = {
        "ActivationSymmetric": rp.symmetric,
        "WeightSymmetric": rp.symmetric,
        "EnableSubgraph": True,
        "ForceQuantizeNoInputCheck": False,
        "MatMulConstBOnly": True,
    }
    if rp.default_tensor_type is not None:
        extra_options["DefaultTensorType"] = rp.default_tensor_type
    print(
        f"  Quantizing weights (dynamic INT8, {rp.dynamic_weight_type}, "
        f"per_channel={rp.per_channel}, reduce_range={rp.reduce_range})..."
    )
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    converted_constants = _materialize_constant_tensors_as_initializers(model.graph)
    if converted_constants:
        print(f"  Materialized {converted_constants} Constant tensor nodes as initializers for weight quantization.")
    lowered_gemms = _lower_eligible_gemm_to_matmul_add(model.graph)
    if lowered_gemms:
        print(f"  Lowered {lowered_gemms} affine Gemm node(s) to MatMul + Add for dynamic quantization.")
    if rp.algo == "AFFINE_REFINE_V2":
        rp = copy.copy(rp)
        rp.nodes_to_exclude = _resolve_nodes(rp.nodes_to_exclude, src_path)
        rp.nodes_to_include = _resolve_nodes(rp.nodes_to_include, src_path)
        _quantize_affine_v2_dynamic_model(model, rp, config)
        _save_model(model, dst_path, external)
        del model
        gc.collect()
        return
    quantize_dynamic(
        model_input=model,
        model_output=dst_path,
        per_channel=rp.per_channel,
        reduce_range=rp.reduce_range,
        weight_type=weight_type,
        extra_options=extra_options,
        nodes_to_quantize=_resolve_nodes(rp.nodes_to_include, src_path),
        nodes_to_exclude=_resolve_nodes(rp.nodes_to_exclude, src_path),
        use_external_data_format=external,
    )
    del model
    gc.collect()


def collect_quant_unsafe_nodes(model_path: str) -> list[str]:
    """Collect MatMul/Gemm/Gather nodes that dynamic quantization should skip.

    Skips MatMul/Gemm fed by float16 or rank>2 constant weights and Gather fed by float16
    weights. This covers frontend filterbanks, relative-position tables, and fp16 embeddings.
    """
    model = onnx.load(model_path)
    fp16_weights: set[str] = set()
    high_rank_weights: set[str] = set()

    def _register(name: str, data_type: int, dims) -> None:
        if data_type == TensorProto.FLOAT16:
            fp16_weights.add(name)
        if len(dims) > 2:
            high_rank_weights.add(name)

    for tensor in _iter_all_data_tensors(model.graph):
        if tensor.name:
            _register(tensor.name, tensor.data_type, tensor.dims)

    for node in model.graph.node:
        if node.op_type == "Constant" and node.output:
            for attr in node.attribute:
                if attr.HasField("t"):
                    _register(node.output[0], attr.t.data_type, attr.t.dims)
                for tensor in attr.tensors:
                    _register(node.output[0], tensor.data_type, tensor.dims)

    nodes_to_exclude = []
    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            if any(name in fp16_weights or name in high_rank_weights for name in node.input):
                nodes_to_exclude.append(node.name)
        elif node.op_type == "Gather":
            if any(name in fp16_weights for name in node.input):
                nodes_to_exclude.append(node.name)
    del model
    gc.collect()
    return nodes_to_exclude


def get_model_paths(config: OptimizerConfig, name: str) -> tuple[str, str]:
    return (
        os.path.join(config.original_folder_path, f"{name}.onnx"),
        os.path.join(config.optimized_folder_path, f"{name}.onnx"),
    )


def process_model(
    name: str,
    rp: ResolvedPlan,
    config: OptimizerConfig,
    mixed_precision: bool,
) -> Path:
    src_path, dst_path = get_model_paths(config, name)
    _remove_external_files(dst_path)

    external = rp.external or model_exceeds_2gb(src_path)
    use_fp16 = rp.fp16 or rp.method == "F16"
    keep_io_types = mixed_precision if config.f16_keep_io_types is None else config.f16_keep_io_types

    if rp.method in _WEIGHT_ONLY_BITS:
        quantize_weight_only(src_path, dst_path, rp, _WEIGHT_ONLY_BITS[rp.method], external, config)
    elif rp.method == "DYNAMIC":
        quantize_dynamic_int8(src_path, dst_path, rp, external, config)
    else:
        resave(src_path, dst_path, external)

    if rp.optimize or use_fp16:
        print("  Optimizing (onnxslim -> transformers optimizer -> onnxslim)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=rp.first_slim_no_shape_infer)
        if rp.transformer or use_fp16:
            optimize_onnx_model(dst_path, rp, config, src_path, use_fp16, external, keep_io_types)
            if rp.run_second_slim:
                second_no_shape = not config.shape_infer if rp.second_slim_no_shape_infer is None else rp.second_slim_no_shape_infer
                run_onnxslim(dst_path, external, config, no_shape_infer=second_no_shape)

    if config.upgrade_opset > 0:
        upgrade_opset_version(dst_path, config.upgrade_opset, external)

    simplified_casts = remove_redundant_casts_from_file(dst_path)
    if simplified_casts:
        print(
            f"  Simplified {simplified_casts} provably redundant Cast node/path(s)."
        )

    restored_outputs = repair_model_file(dst_path)
    if restored_outputs:
        print(
            f"  Restored {restored_outputs} precision-free public output name(s)."
        )

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")
    return Path(dst_path)


def copy_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    required: bool = False,
) -> bool:
    """Copy one file or directory without mutating the raw export bundle."""
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        if required:
            raise FileNotFoundError(source)
        print(f"Skipped optional artifact: {source}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
        shutil.copytree(source, destination)
    else:
        if destination.exists() and destination.is_dir():
            shutil.rmtree(destination)
        shutil.copy2(source, destination)
    print(f"Copied {source.name} -> {destination}")
    return True


def copy_artifacts(config: OptimizerConfig) -> None:
    artifacts = (*config.copy_artifacts, config.metadata_artifact)
    seen = set()
    for artifact in artifacts:
        if not artifact or artifact in seen:
            continue
        seen.add(artifact)
        src_path = os.path.join(config.original_folder_path, artifact)
        dst_path = os.path.join(config.optimized_folder_path, artifact)
        copy_artifact(src_path, dst_path)


def format_plan_summary(plan: ResolvedPlan) -> str:
    """Return a concise summary of the effective plan configuration."""
    if plan.method in _WEIGHT_ONLY_BITS:
        return (
            f"{plan.method}, {plan.algo}, block={plan.block_size}, "
            f"format={plan.quant_format}"
        )
    if plan.method == "DYNAMIC":
        return (
            f"DYNAMIC/{plan.algo}/{plan.dynamic_weight_type}, per_channel={plan.per_channel}, "
            f"reduce_range={plan.reduce_range}"
        )
    if plan.uses_float16:
        return f"F16, force_initializers={plan.f16_force_initializers}"
    return "F32"


def run_optimizer(
    config: OptimizerConfig,
    *,
    model_names: Sequence[str] | None = None,
    after_model: Callable[[str, ResolvedPlan, Path], None] | None = None,
    copy_configured_artifacts: bool = True,
    print_completion: bool = True,
    reset_output_folder: bool = False,
) -> dict[str, ResolvedPlan]:
    """Process and optionally post-process selected model plans."""
    resolved = resolve_plans(config, model_names)
    output_folder = Path(config.optimized_folder_path)
    if reset_output_folder and output_folder.exists():
        source_folder = Path(config.original_folder_path).resolve()
        if output_folder.resolve() == source_folder:
            raise RuntimeError("Refusing to reset the source model directory.")
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for name in resolved:
        _, dst_path = get_model_paths(config, name)
        _remove_external_files(dst_path)

    mixed_precision = (
        any(plan.uses_float16 for plan in resolved.values())
        and not all(plan.uses_float16 for plan in resolved.values())
    )
    if mixed_precision and config.f16_keep_io_types is None:
        print(
            "TIP: mixed float16/float32 modules detected - forcing keep_io_types=True on "
            "float16 conversions so shared graph I/O stays float32-compatible."
        )

    processed = 0
    for name, rp in resolved.items():
        print(
            f"\n{'=' * 60}\nProcessing: {name}  "
            f"[{format_plan_summary(rp)}]\n{'=' * 60}"
        )
        output_path = process_model(name, rp, config, mixed_precision)
        processed += 1
        if after_model is not None:
            after_model(name, rp, output_path)

    if copy_configured_artifacts:
        copy_artifacts(config)
    if print_completion:
        print(
            f"\n--- Processed {processed}/{len(resolved)} configured model(s) "
            "successfully! ---"
        )
    return resolved
