"""Shared Qwen-style ONNX optimization pipeline for the ASR export scripts."""

from __future__ import annotations

import gc
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import onnx
import onnx.version_converter
from onnx import TensorProto
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
_VALID_ALGOS = {"DEFAULT", "RTN", "HQQ", "k_quant"}


@dataclass
class Plan:
    """Per-module optimization recipe; ``None`` fields inherit OptimizerConfig defaults."""

    method: str = "DYNAMIC"  # Q2 | Q4 | Q8 | DYNAMIC | F16 | F32
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
    num_heads: IntValue = 0
    hidden_size: IntValue = 0
    # storage
    external: bool | None = None
    # onnxslim shape inference knobs
    first_slim_no_shape_infer: bool = True
    second_slim_no_shape_infer: bool | None = None


@dataclass
class OptimizerConfig:
    """Global defaults shared by every module in one Optimize_ONNX.py script."""

    original_folder_path: str
    optimized_folder_path: str
    model_plans: dict[str, Plan]
    # weight-only defaults
    weight_only_algorithm: str = "k_quant"
    block_size: int = 32
    accuracy_level: int = 4
    quant_symmetric: bool = False
    quant_format: str = "QOperator"
    # dynamic INT8 defaults
    dynamic_weight_type: str = "QUInt8"
    dynamic_per_channel: bool = True
    dynamic_reduce_range: bool = False
    dynamic_default_tensor_type: int | None = None
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
    # float16
    f16_keep_io_types: bool | None = None
    f16_force_initializers: bool = True
    f16_min_positive_val: float = 1e-7
    f16_max_finite_val: float = 32767.0
    f16_node_block_list: list[str] | None = None
    f16_op_block_list: list[str] | None = None
    # optional side artifacts copied after all models are processed
    copy_artifacts: tuple[str, ...] = ()


@dataclass
class ResolvedPlan:
    method: str
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
    num_heads: IntValue
    hidden_size: IntValue
    external: bool
    first_slim_no_shape_infer: bool
    second_slim_no_shape_infer: bool | None


def _pick(value, default):
    return default if value is None else value


def _uses_fp16(plan: Plan) -> bool:
    return plan.fp16 or plan.method.upper() == "F16"


def resolve_plan(plan: Plan, config: OptimizerConfig) -> ResolvedPlan:
    return ResolvedPlan(
        method=plan.method.upper(),
        algo=_pick(plan.algo, config.weight_only_algorithm),
        op_types=_pick(plan.op_types, ("MatMul",)),
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
        num_heads=plan.num_heads,
        hidden_size=plan.hidden_size,
        external=_pick(plan.external, config.force_external_data),
        first_slim_no_shape_infer=plan.first_slim_no_shape_infer,
        second_slim_no_shape_infer=_pick(plan.second_slim_no_shape_infer, config.second_slim_no_shape_infer),
    )


def validate_plan(name: str, rp: ResolvedPlan) -> None:
    valid_methods = set(_WEIGHT_ONLY_BITS) | {"DYNAMIC", "F16", "F32"}
    if rp.method not in valid_methods:
        raise ValueError(f"[{name}] unknown method {rp.method!r}; choose one of {sorted(valid_methods)}.")

    if rp.method in _WEIGHT_ONLY_BITS:
        bits = _WEIGHT_ONLY_BITS[rp.method]
        if rp.algo not in _VALID_ALGOS:
            raise ValueError(f"[{name}] unknown algo {rp.algo!r}; choose one of {sorted(_VALID_ALGOS)}.")
        if rp.quant_format not in _QUANT_FORMATS:
            raise ValueError(f"[{name}] unknown quant_format; choose 'QOperator' or 'QDQ'.")
        if len(rp.op_types) != len(rp.axes):
            raise ValueError(f"[{name}] op_types {rp.op_types} and axes {rp.axes} must have equal length.")
        if "Gather" in rp.op_types and rp.algo != "DEFAULT":
            raise ValueError(f"[{name}] Gather quantization requires algo='DEFAULT' (got {rp.algo!r}).")
        if rp.quant_format == "QDQ" and (rp.algo != "DEFAULT" or bits != 4):
            raise ValueError(
                f"[{name}] QDQ format supports only algo='DEFAULT' with 4-bit (got {rp.algo!r}, {bits}-bit)."
            )

    if rp.method == "DYNAMIC" and rp.dynamic_weight_type not in _DYNAMIC_WEIGHT_TYPES:
        raise ValueError(f"[{name}] unknown dynamic_weight_type; choose 'QUInt8' or 'QInt8'.")


def model_exceeds_2gb(model_path: str) -> bool:
    total = os.path.getsize(model_path)
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total > 2 * 1024**3


def model_size_mb(model_path: str) -> float:
    total = os.path.getsize(model_path)
    data_path = model_path + ".data"
    if os.path.exists(data_path):
        total += os.path.getsize(data_path)
    return total / (1024 * 1024)


def _remove_external_files(model_path: str) -> None:
    for path in (model_path, model_path + ".data"):
        if os.path.exists(path):
            os.remove(path)


def _save_model(model, model_path: str, external: bool) -> None:
    _remove_external_files(model_path)
    if external:
        onnx.save(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(model_path) + ".data",
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


def resave(src_path: str, dst_path: str, external: bool) -> None:
    model = onnx.load(src_path)
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
    are rewritten), so restamping is safe for both inline and external-data models. A no-op when the
    source model carried no metadata.
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


def run_onnxslim(model_path: str, external: bool, config: OptimizerConfig, no_shape_infer: bool) -> None:
    def _slim() -> None:
        slim(
            model=model_path,
            output_model=model_path,
            no_shape_infer=no_shape_infer,
            skip_fusion_patterns=config.slim_skip_fusion_patterns,
            skip_optimizations=config.slim_skip_optimizations,
            size_threshold=config.slim_size_threshold,
            save_as_external_data=external,
            verbose=False,
        )

    data_path = model_path + ".data"
    if not external or not os.path.exists(data_path):
        _slim()
        return

    stash_path = model_path + ".stash.data"
    if os.path.exists(stash_path):
        os.remove(stash_path)
    os.replace(data_path, stash_path)
    _retarget_external_location(model_path, os.path.basename(stash_path))
    try:
        _slim()
    except BaseException:
        if not os.path.exists(data_path):
            os.replace(stash_path, data_path)
            _retarget_external_location(model_path, os.path.basename(data_path))
        raise
    finally:
        if os.path.exists(stash_path):
            os.remove(stash_path)


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
        model.convert_float_to_float16(
            keep_io_types=keep_io_types,
            force_fp16_initializers=config.f16_force_initializers,
            use_symbolic_shape_infer=config.shape_infer,
            max_finite_val=config.f16_max_finite_val,
            min_positive_val=config.f16_min_positive_val,
            op_block_list=config.f16_op_block_list,
            node_block_list=config.f16_node_block_list,
        )
        renamed = _deduplicate_node_names(model.model.graph)
        if renamed:
            print(f"  Renamed {renamed} duplicate node names after float16 conversion.")
    model.save_model_to_file(model_path, use_external_data_format=external)
    del model
    gc.collect()


def upgrade_opset_version(model_path: str, version: int, external: bool) -> None:
    print(f"  Upgrading opset to {version}...")
    try:
        model = onnx.version_converter.convert_version(onnx.load(model_path), version)
        _save_model(model, model_path, external)
        del model
        gc.collect()
    except Exception as exc:
        print(f"  Opset upgrade failed: {exc}. Keeping current version.")
        resave(model_path, model_path, external)


def build_weight_only_config(rp: ResolvedPlan, bits: int):
    op_types, axes = list(rp.op_types), list(rp.axes)
    quant_axes = tuple(zip(op_types, axes))
    quant_format = _QUANT_FORMATS[rp.quant_format]
    common = {
        "quant_format": quant_format,
        "op_types_to_quantize": tuple(op_types),
    }
    if rp.algo == "RTN":
        cfg = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(**common)
    elif rp.algo == "HQQ":
        cfg = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
            bits=bits, block_size=rp.block_size, axis=axes[0], quant_axes=quant_axes, **common,
        )
    elif rp.algo == "k_quant":
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
    return cfg, quant_axes


def quantize_weight_only(src_path: str, dst_path: str, rp: ResolvedPlan, bits: int, external: bool) -> None:
    cfg, quant_axes = build_weight_only_config(rp, bits)
    print(
        f"  Quantizing weights ({rp.algo}, {bits}-bit, block={rp.block_size}, "
        f"format={rp.quant_format}, ops={list(rp.op_types)})..."
    )
    model = quant_utils.load_model_with_shape_infer(Path(src_path))
    quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
        model,
        block_size=rp.block_size,
        is_symmetric=rp.symmetric,
        accuracy_level=rp.accuracy_level,
        quant_format=_QUANT_FORMATS[rp.quant_format],
        op_types_to_quantize=tuple(rp.op_types),
        quant_axes=quant_axes,
        algo_config=cfg,
        nodes_to_exclude=_resolve_nodes(rp.nodes_to_exclude, src_path),
        nodes_to_include=_resolve_nodes(rp.nodes_to_include, src_path),
    )
    quant.process()
    quant.model.save_model_to_file(dst_path, external)
    del model, quant
    gc.collect()


def quantize_dynamic_int8(src_path: str, dst_path: str, rp: ResolvedPlan, external: bool) -> None:
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


def process_model(name: str, rp: ResolvedPlan, config: OptimizerConfig, mixed_precision: bool) -> None:
    src_path, dst_path = get_model_paths(config, name)
    if not os.path.exists(src_path):
        print(f"  Skipping - file not found: {src_path}")
        return

    source_metadata = read_onnx_metadata(src_path)
    _remove_external_files(dst_path)

    external = rp.external or model_exceeds_2gb(src_path)
    use_fp16 = rp.fp16 or rp.method == "F16"
    keep_io_types = mixed_precision if config.f16_keep_io_types is None else config.f16_keep_io_types

    if rp.method in _WEIGHT_ONLY_BITS:
        quantize_weight_only(src_path, dst_path, rp, _WEIGHT_ONLY_BITS[rp.method], external)
    elif rp.method == "DYNAMIC":
        quantize_dynamic_int8(src_path, dst_path, rp, external)
    else:
        resave(src_path, dst_path, external)

    if rp.optimize or use_fp16:
        print("  Optimizing (onnxslim -> transformers optimizer -> onnxslim)...")
        run_onnxslim(dst_path, external, config, no_shape_infer=rp.first_slim_no_shape_infer)
        if rp.transformer or use_fp16:
            optimize_onnx_model(dst_path, rp, config, src_path, use_fp16, external, keep_io_types)
            second_no_shape = not config.shape_infer if rp.second_slim_no_shape_infer is None else rp.second_slim_no_shape_infer
            run_onnxslim(dst_path, external, config, no_shape_infer=second_no_shape)

    if config.upgrade_opset > 0:
        upgrade_opset_version(dst_path, config.upgrade_opset, external)

    if not external and os.path.exists(dst_path + ".data"):
        os.remove(dst_path + ".data")

    # Restamp the source model's metadata_props onto the optimized output. Quantization / onnxslim /
    # the transformers optimizer can drop custom metadata; the geometry / token / max_seq_len facts are
    # invariant through those passes, so copying them across keeps the runtime's metadata reads working.
    # Only the activation dtype is allowed to change here, and only for plans that actually run fp16 conversion.
    output_metadata = dict(source_metadata)
    if use_fp16:
        output_metadata["activations_fp16"] = "1"
    write_onnx_metadata(dst_path, output_metadata)


def copy_artifacts(config: OptimizerConfig) -> None:
    for artifact in config.copy_artifacts:
        src_path = os.path.join(config.original_folder_path, artifact)
        dst_path = os.path.join(config.optimized_folder_path, artifact)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)
            print(f"Copied {artifact} -> {dst_path}")


def run_optimizer(config: OptimizerConfig) -> None:
    os.makedirs(config.optimized_folder_path, exist_ok=True)

    resolved = {name: resolve_plan(plan, config) for name, plan in config.model_plans.items()}
    for name, rp in resolved.items():
        validate_plan(name, rp)

    for name in resolved:
        _, dst_path = get_model_paths(config, name)
        _remove_external_files(dst_path)

    mixed_precision = (
        any(_uses_fp16(plan) for plan in config.model_plans.values())
        and not all(_uses_fp16(plan) for plan in config.model_plans.values())
    )
    if mixed_precision and config.f16_keep_io_types is None:
        print(
            "TIP: mixed float16/float32 modules detected - forcing keep_io_types=True on "
            "float16 conversions so shared graph I/O stays float32-compatible."
        )

    for name, rp in resolved.items():
        print(f"\n{'=' * 60}\nProcessing: {name}  [{rp.method}]\n{'=' * 60}")
        process_model(name, rp, config, mixed_precision)

    copy_artifacts(config)
    print("\n--- All models processed successfully! ---")
