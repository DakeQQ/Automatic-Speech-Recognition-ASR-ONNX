"""Optimize Audio8-ASR donors and rebuild its merged Encoder+Prefill bundle.

Encoder and Main are optimized once as standalone data-light donors and then
transplanted into each raw strategy shell. Embed is always rebuilt as a view of
Main's verified tied lm-head representation, so the table is quantized once and
stored in one shared range without a private Embed sidecar.
"""

from __future__ import annotations

import copy
import gc
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

import Shared_Merged
from Optimize_ONNX_Common import (
    QUANTIZATION_F16_OP_BLOCK_LIST,
    consolidate_optimized_model_weights,
    copy_artifact,
    OptimizerConfig,
    Plan,
    read_onnx_metadata,
    remove_model_files,
    remove_redundant_casts,
    resolve_plan,
    run_optimizer,
    share_external_initializers_if_identical,
)


_ENCODER_MODEL = Path(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["encoder"]).stem
_MAIN_MODEL = Path(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["main"]).stem
_EMBED_MODEL = Path(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["embed"]).stem
# ============================== USER CONFIG ==============================
# Edit this section only. The implementation below derives its behavior from
# these values; component methods are not selected by hidden conditionals.

# Input/output folders.
ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Audio8_ASR_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Audio8_ASR_Optimized")

# Runtime composition policy.
#   "MAIN_TIED": synthesize Embed from Main's tied lm-head representation. This
#                 supports Q2/Q4/Q8 and floating F32/F16 Main plans without a
#                 private table. The Embed plan below is inactive.
#   "INDEPENDENT": optimize Embed with its own plan below, then share only exact
#                  byte-identical initializers with the merged bundle.
EMBED_WEIGHT_SOURCE               = "MAIN_TIED"
REQUIRE_EMBED_SHARED_INITIALIZERS = True

# Enable this only when Encoder, Main, and independent Embed all use F16. It
# removes converter ABI casts after composition and exposes literal F16 runtime
# interfaces/constants. Mixed module methods should leave this False.
FULL_FLOAT16_RUNTIME = False

# Split optional MatMulNBits bias inputs into MatMulNBits + Add for ORT CUDA.
# Leave False for the smaller CPU-native fused-bias representation.
ENABLE_Q4_CUDA_COMPATIBILITY = False

# Global defaults inherited by any Plan field left as None.
WEIGHT_ONLY_ALGORITHM      = "AFFINE_REFINE_V2"  # DEFAULT | RTN | HQQ | k_quant | AFFINE_REFINE_V2
WEIGHT_ONLY_BLOCK_SIZE     = 64
WEIGHT_ONLY_ACCURACY_LEVEL = 4
WEIGHT_ONLY_SYMMETRIC      = True
WEIGHT_ONLY_QUANT_FORMAT   = "QOperator"  # QOperator | QDQ

DYNAMIC_WEIGHT_TYPE         = "QInt8"  # QInt8 | QUInt8
DYNAMIC_PER_CHANNEL         = True
DYNAMIC_REDUCE_RANGE        = False

FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET       = 0

OPTIMIZER_LEVEL              = 2
OPTIMIZER_MODEL_TYPE         = "bert"
OPTIMIZER_ONLY_ONNXRUNTIME   = False
OPTIMIZER_FUSION_OPTIONS     = (
    {
        "enable_skip_layer_norm":      False,
        "enable_bias_skip_layer_norm": False,
    }
    if FULL_FLOAT16_RUNTIME
    else None
)
ENABLE_SHAPE_INFERENCE     = True

SLIM_SKIP_FUSION_PATTERNS  = None
SLIM_SKIP_OPTIMIZATIONS    = None
SLIM_SIZE_THRESHOLD        = None
SECOND_SLIM_NO_SHAPE_INFER = None

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_MIN_POSITIVE_VALUE = 1e-7
F16_MAX_FINITE_VALUE   = 32767.0
F16_NODE_BLOCK_LIST    = None
F16_OP_BLOCK_LIST      = QUANTIZATION_F16_OP_BLOCK_LIST


def _main_nodes_to_exclude(model_path: str) -> list[str] | None:
    """Keep the tied lm-head floating only for dynamic INT8 Main plans."""
    if MODEL_PLANS[_MAIN_MODEL].method.upper() != "DYNAMIC":
        return None
    model = onnx.load(model_path, load_external_data=False)
    matches = [
        node.name
        for node in model.graph.node
        if node.op_type in ("MatMul", "Gemm") and "/lm_head" in node.name
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(
            f"Expected one named Audio8-ASR lm-head projection; found {matches}."
        )
    return matches

# Configure every reusable component independently. Valid methods are Q2, Q4,
# Q8, DYNAMIC, F16, and F32. Any omitted Plan option inherits the global default
# above. Useful Plan options include algo, op_types, axes, block_size,
# accuracy_level, symmetric, quant_format, dynamic_weight_type, per_channel,
# reduce_range, nodes_to_exclude/include, optimize, transformer, opt_level, fp16,
# f16_force_initializers, external, first_slim_no_shape_infer, run_second_slim,
# and second_slim_no_shape_infer. Encoder/Main are transplanted into every
# strategy graph; Embed is used only when EMBED_WEIGHT_SOURCE == "INDEPENDENT".
MODEL_PLANS: dict[str, Plan] = {
    "Audio8_ASR_Encoder": Plan(
        method="Q8",
        algo=WEIGHT_ONLY_ALGORITHM,
        external=FORCE_EXTERNAL_DATA,
        num_heads=0,
        hidden_size=0,
    ),
    "Audio8_ASR_Decoder_Main": Plan(
        method="Q8",
        algo=WEIGHT_ONLY_ALGORITHM,
        external=FORCE_EXTERNAL_DATA,
        optimize=True,
        nodes_to_exclude=_main_nodes_to_exclude,
        num_heads=0,
        hidden_size=0,
    ),
    "Audio8_ASR_Decoder_Embed": Plan(
        method="Q8",
        op_types=("Gather",),
        axes=(0,),
        transformer=False,
        external=FORCE_EXTERNAL_DATA,
        run_second_slim=False,
    ),
    "Audio8_ASR_Prefill_Greedy":         Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Audio8_ASR_Prefill_Penalty_Greedy": Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Audio8_ASR_PrefillSampling":        Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Audio8_ASR_Decode_Greedy":          Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Audio8_ASR_Decode_Penalty_Greedy":  Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Audio8_ASR_DecodeSampling":         Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Audio8_ASR_SharedInitializers":     Plan(method="Q8", process=False, optimize=True, transformer=True),
    "ASR_Metadata":                     Plan(method="F32", process=False, optimize=True, transformer=True),
}

# Full-F16 starts with no guards. If validation finds a reproducible NaN/Inf,
# add exactly one printed candidate node at a time, rebuild, and retest.
F16_OVERFLOW_GUARD_NODE_NAMES: dict[str, tuple[str, ...]] = {
    _ENCODER_MODEL: (),
    _MAIN_MODEL: (),
}
F16_OVERFLOW_CANDIDATE_OPS = frozenset(
    ("Pow", "ReduceMean", "ReduceSum", "ReduceSumSquare")
)

# Embed is synthesized rather than independently optimized in MAIN_TIED mode.
ACTIVE_MODEL_NAMES = tuple(
    name
    for name, plan in MODEL_PLANS.items()
    if plan.process
    if name != _EMBED_MODEL or EMBED_WEIGHT_SOURCE == "INDEPENDENT"
)

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    block_size=WEIGHT_ONLY_BLOCK_SIZE,
    accuracy_level=WEIGHT_ONLY_ACCURACY_LEVEL,
    quant_symmetric=WEIGHT_ONLY_SYMMETRIC,
    quant_format=WEIGHT_ONLY_QUANT_FORMAT,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=DYNAMIC_REDUCE_RANGE,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_model_type=OPTIMIZER_MODEL_TYPE,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    optimizer_fusion_options=OPTIMIZER_FUSION_OPTIONS,
    shape_infer=ENABLE_SHAPE_INFERENCE,
    slim_skip_fusion_patterns=SLIM_SKIP_FUSION_PATTERNS,
    slim_skip_optimizations=SLIM_SKIP_OPTIMIZATIONS,
    slim_size_threshold=SLIM_SIZE_THRESHOLD,
    second_slim_no_shape_infer=SECOND_SLIM_NO_SHAPE_INFER,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_force_initializers=F16_FORCE_INITIALIZERS,
    f16_min_positive_val=F16_MIN_POSITIVE_VALUE,
    f16_max_finite_val=F16_MAX_FINITE_VALUE,
    f16_node_block_list=F16_NODE_BLOCK_LIST,
    f16_op_block_list=F16_OP_BLOCK_LIST,
)

# ============================ END USER CONFIG ============================


def _node_int_attribute(node: onnx.NodeProto, name: str, default: int) -> int:
    return int(next((attr.i for attr in node.attribute if attr.name == name), default))


def _inspect_embed_layout(
    model: onnx.ModelProto,
) -> tuple[int, onnx.NodeProto, int, TensorProto, int | None]:
    """Return the single Embed Gather/table and validate its supported layout."""
    gather_indices = [
        index
        for index, node in enumerate(model.graph.node)
        if node.domain in ("", "ai.onnx") and node.op_type == "Gather"
    ]
    if len(gather_indices) != 1:
        raise RuntimeError(
            "Audio8-ASR Embed must contain exactly one standard Gather; "
            f"found {len(gather_indices)}."
        )
    gather_index = gather_indices[0]
    gather = model.graph.node[gather_index]
    if len(gather.input) < 2 or len(gather.output) != 1:
        raise RuntimeError("Audio8-ASR Embed Gather has an unexpected input/output contract.")

    initializer_indices = {
        initializer.name: index
        for index, initializer in enumerate(model.graph.initializer)
    }
    table_index = initializer_indices.get(gather.input[0])
    if table_index is None:
        raise RuntimeError("Audio8-ASR Embed Gather does not read a constant table.")
    table = model.graph.initializer[table_index]
    if len(table.dims) != 2:
        raise RuntimeError(
            f"Audio8-ASR Embed table must be rank 2, got {list(table.dims)}."
        )

    gather_axis = _node_int_attribute(gather, "axis", 0)
    if gather_axis == 0:
        return gather_index, gather, table_index, table, None
    if gather_axis != 1:
        raise RuntimeError(
            f"Audio8-ASR Embed Gather axis must be 0 or 1, got {gather_axis}."
        )

    consumers = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if gather.output[0] in node.input
    ]
    if len(consumers) != 1:
        raise RuntimeError(
            "The tied column-Gather Embed must have exactly one consumer; "
            f"found {len(consumers)}."
        )
    transpose_index, transpose = consumers[0]
    permutation = tuple(
        int(value)
        for value in next(
            (
                attr.ints
                for attr in transpose.attribute
                if attr.name == "perm"
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
        raise RuntimeError(
            "The tied column-Gather Embed must be followed by Transpose[1,2,0]."
        )
    return gather_index, gather, table_index, table, transpose_index


def _run_optimizer_pipeline() -> None:
    """Optimize every selected donor once."""
    if EMBED_WEIGHT_SOURCE in {"MAIN_TIED", "MAIN_Q4"}:
        target = (
            "CUDA-compatible"
            if ENABLE_Q4_CUDA_COMPATIBILITY
            else "CPU-native encoder"
        )
        print(f"[Tied Embed] {target}; Embed reuses Main's lm-head representation")
    elif EMBED_WEIGHT_SOURCE == "INDEPENDENT":
        print("[Embed Target] Independently optimized module plan")
    else:
        raise ValueError(f"Unsupported EMBED_WEIGHT_SOURCE: {EMBED_WEIGHT_SOURCE!r}")
    run_optimizer(
        CONFIG,
        model_names=ACTIVE_MODEL_NAMES,
        reset_output_folder=True,
    )


def _resolved_module_plan(name: str):
    return resolve_plan(MODEL_PLANS[name], CONFIG, model_name=name)


def _module_uses_f16(name: str) -> bool:
    plan = _resolved_module_plan(name)
    return plan.fp16 or plan.method == "F16"


def _external_data_map(initializer: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in initializer.external_data}


def _block_quantized_tuple_for_node(
    model: onnx.ModelProto,
    *,
    op_type: str,
    required_k: int,
    required_n: int,
    required_block_size: int,
    required_bits: int,
) -> tuple[onnx.NodeProto, tuple[TensorProto, TensorProto, TensorProto]]:
    """Find one asymmetric block-quantized tuple with the required geometry."""
    initializers = {
        initializer.name: initializer for initializer in model.graph.initializer
    }
    matches: list[
        tuple[onnx.NodeProto, tuple[TensorProto, TensorProto, TensorProto]]
    ] = []
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != op_type or len(node.input) < 4:
            continue
        initializer_inputs = (
            (node.input[0], node.input[2], node.input[3])
            if op_type == "GatherBlockQuantized"
            else tuple(node.input[1:4])
        )
        tensors = tuple(initializers.get(name) for name in initializer_inputs)
        if any(tensor is None for tensor in tensors):
            continue
        block_size = _node_int_attribute(node, "block_size", 0)
        bits = _node_int_attribute(node, "bits", 4)
        if op_type == "MatMulNBits":
            geometry_matches = (
                _node_int_attribute(node, "K", 0) == required_k
                and _node_int_attribute(node, "N", 0) == required_n
            )
        else:
            data = tensors[0]
            geometry_matches = (
                _node_int_attribute(node, "gather_axis", 0) == 0
                and _node_int_attribute(node, "quantize_axis", -1) == 1
                and list(data.dims) == [required_n, required_k]
            )
        if (
            block_size == required_block_size
            and bits == required_bits
            and geometry_matches
        ):
            matches.append((node, tensors))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {op_type} tied-table tuple with "
            f"K={required_k}, N={required_n}, block={required_block_size}, "
            f"bits={required_bits}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _share_block_quantized_embed_with_main(
    embed_path: Path,
    shared_path: Path,
    main_graph_path: Path,
    block_size: int,
    bits: int,
) -> dict[str, int]:
    """Build Embed directly over Main's shared block-quantized lm-head tuple.

    The source Embed and lm-head tensors are exactly tied, but independently
    quantizing Gather with DEFAULT produces a different Q4 approximation. Avoid
    that second quantization entirely: expose Main's packed uint8 weight, scales,
    and packed zero points under Gather-compatible shapes, then create one
    row-Gather node over those shared byte ranges.
    """
    source_embed_path = Path(ORIGINAL_FOLDER_PATH) / f"{_EMBED_MODEL}.onnx"
    source_main_path = Path(ORIGINAL_FOLDER_PATH) / f"{_MAIN_MODEL}.onnx"
    embed = onnx.load(str(source_embed_path), load_external_data=False)
    shared = onnx.load(str(shared_path), load_external_data=False)
    main_graph = onnx.load(str(main_graph_path), load_external_data=False)
    raw_main = onnx.load(str(source_main_path), load_external_data=False)
    _, raw_gather, _, raw_table, _ = _inspect_embed_layout(embed)
    hidden, vocab = (int(dim) for dim in raw_table.dims)
    lm_head_consumers = [
        node
        for node in raw_main.graph.node
        if node.domain in ("", "ai.onnx")
        and node.op_type == "MatMul"
        and "/lm_head" in node.name
        and len(node.input) >= 2
    ]
    raw_main_initializers = {
        initializer.name: initializer for initializer in raw_main.graph.initializer
    }
    if len(lm_head_consumers) != 1:
        raise RuntimeError(
            f"Expected one raw lm-head MatMul; found {len(lm_head_consumers)}."
        )
    raw_lm_head = raw_main_initializers.get(lm_head_consumers[0].input[1])
    if (
        raw_lm_head is None
        or raw_lm_head.data_type != raw_table.data_type
        or list(raw_lm_head.dims) != [hidden, vocab]
    ):
        raise RuntimeError("Raw Audio8-ASR Embed and lm-head tables are not exactly tied.")
    if hidden % block_size != 0 or hidden % 2 != 0:
        raise RuntimeError(
            f"Q4 Embed sharing requires an even hidden size divisible by block "
            f"size {block_size}; got {hidden}."
        )
    raw_embed_external = _external_data_map(raw_table)
    raw_main_external = _external_data_map(raw_lm_head)
    if raw_embed_external or raw_main_external:
        if raw_embed_external != raw_main_external:
            raise RuntimeError(
                "Raw Audio8-ASR Embed and lm-head tables reference different bytes."
            )
    elif raw_table.raw_data != raw_lm_head.raw_data:
        raise RuntimeError("Raw Audio8-ASR Embed and lm-head tables are not byte-identical.")

    _, shared_tuple = _block_quantized_tuple_for_node(
        main_graph,
        op_type="MatMulNBits",
        required_k=hidden,
        required_n=vocab,
        required_block_size=block_size,
        required_bits=bits,
    )
    manifest_by_name = {
        initializer.name: initializer for initializer in shared.graph.initializer
    }
    manifest_tuple = tuple(
        manifest_by_name.get(initializer.name) for initializer in shared_tuple
    )
    if any(initializer is None for initializer in manifest_tuple):
        raise RuntimeError("Main's tied Q4 tuple is incomplete in the shared manifest.")
    for graph_ref, manifest_ref in zip(shared_tuple, manifest_tuple):
        if (
            graph_ref.data_type != manifest_ref.data_type
            or list(graph_ref.dims) != list(manifest_ref.dims)
            or _external_data_map(graph_ref) != _external_data_map(manifest_ref)
        ):
            raise RuntimeError(
                f"Shared manifest contract mismatch for {graph_ref.name!r}."
            )
    shared_tuple = manifest_tuple

    k_blocks = (hidden + block_size - 1) // block_size
    packed_block_bytes = (block_size * bits + 7) // 8
    packed_zero_point_columns = (k_blocks * bits + 7) // 8
    expected_main = (
        (TensorProto.UINT8, [vocab, k_blocks, packed_block_bytes]),
        (raw_table.data_type, [vocab, k_blocks]),
        (TensorProto.UINT8, [vocab, packed_zero_point_columns]),
    )
    for initializer, (data_type, dims) in zip(shared_tuple, expected_main):
        if initializer.data_type != data_type or list(initializer.dims) != dims:
            raise RuntimeError(
                f"Main Q4 tensor {initializer.name!r} has unexpected type/shape: "
                f"{initializer.data_type}, {list(initializer.dims)}."
            )
    expected_lengths = (
        vocab * k_blocks * packed_block_bytes,
        vocab * k_blocks * np.dtype(
            onnx.helper.tensor_dtype_to_np_dtype(raw_table.data_type)
        ).itemsize,
        vocab * packed_zero_point_columns,
    )
    actual_lengths = tuple(
        int(_external_data_map(initializer).get("length", "0"))
        for initializer in shared_tuple
    )
    if actual_lengths != expected_lengths:
        raise RuntimeError(
            f"Main Q4 tuple has unexpected byte lengths: {actual_lengths}; "
            f"expected {expected_lengths}."
        )

    alias_specs = (
        (
            f"{shared_tuple[0].name}_embed_view",
            TensorProto.UINT8,
            [vocab, (hidden * bits + 7) // 8],
        ),
        (f"{shared_tuple[1].name}_embed_view", raw_table.data_type, [vocab, k_blocks]),
        (
            f"{shared_tuple[2].name}_embed_view",
            TensorProto.UINT8,
            [vocab, packed_zero_point_columns],
        ),
    )
    aliases: list[TensorProto] = []
    for source, (name, data_type, dims) in zip(shared_tuple, alias_specs):
        alias = TensorProto()
        alias.name = name
        alias.data_type = data_type
        alias.dims.extend(dims)
        alias.data_location = TensorProto.EXTERNAL
        alias.external_data.extend(source.external_data)
        aliases.append(alias)

    manifest_initializers = {
        initializer.name: initializer for initializer in shared.graph.initializer
    }
    for alias in aliases:
        existing = manifest_initializers.get(alias.name)
        if existing is not None:
            if existing.SerializeToString() != alias.SerializeToString():
                raise RuntimeError(
                    f"Shared Q4 Embed alias {alias.name!r} already exists with "
                    "a different contract."
                )
            continue
        shared.graph.initializer.append(copy.deepcopy(alias))
        manifest_initializers[alias.name] = alias
    for prop in shared.metadata_props:
        if prop.key == "initializer_count":
            prop.value = str(len(shared.graph.initializer))
    onnx.save(shared, str(shared_path))

    gather = onnx.helper.make_node(
        "GatherBlockQuantized",
        [aliases[0].name, raw_gather.input[1], aliases[1].name, aliases[2].name],
        [embed.graph.output[0].name],
        name="shared_kquant_embed_gather",
        domain="com.microsoft",
        gather_axis=0,
        quantize_axis=1,
        block_size=block_size,
        bits=bits,
    )
    del embed.graph.node[:]
    embed.graph.node.append(gather)
    del embed.graph.initializer[:]
    embed.graph.initializer.extend(copy.deepcopy(initializer) for initializer in aliases)
    del embed.graph.value_info[:]
    if not any(opset.domain == "com.microsoft" for opset in embed.opset_import):
        embed.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    old_private_sidecar = embed_path.with_name(embed_path.name + ".data")
    had_private_sidecar = old_private_sidecar.exists()
    Shared_Merged.save_model(embed, embed_path)
    return {
        "shared_initializer_count": len(aliases),
        "shared_data_bytes": sum(actual_lengths),
        "removed_external_file_count": int(
            had_private_sidecar and not old_private_sidecar.exists()
        ),
    }


def _share_float_embed_with_main(
    embed_path: Path,
    shared_path: Path,
    main_graph_path: Path,
) -> dict[str, int]:
    """Build Embed as a column Gather over Main's shared F32/F16 lm-head."""
    source_embed_path = Path(ORIGINAL_FOLDER_PATH) / f"{_EMBED_MODEL}.onnx"
    source_main_path = Path(ORIGINAL_FOLDER_PATH) / f"{_MAIN_MODEL}.onnx"
    source_embed = onnx.load(str(source_embed_path), load_external_data=False)
    source_main = onnx.load(str(source_main_path), load_external_data=False)
    _, raw_gather, _, raw_table, _ = _inspect_embed_layout(source_embed)
    hidden, vocab = (int(dim) for dim in raw_table.dims)
    raw_lm_nodes = [
        node
        for node in source_main.graph.node
        if node.domain in ("", "ai.onnx")
        and node.op_type == "MatMul"
        and "/lm_head" in node.name
        and len(node.input) >= 2
    ]
    raw_initializers = {
        initializer.name: initializer
        for initializer in source_main.graph.initializer
    }
    if len(raw_lm_nodes) != 1:
        raise RuntimeError(f"Expected one raw lm-head MatMul; found {len(raw_lm_nodes)}.")
    raw_lm_head = raw_initializers.get(raw_lm_nodes[0].input[1])
    if (
        raw_lm_head is None
        or raw_lm_head.data_type != raw_table.data_type
        or list(raw_lm_head.dims) != [hidden, vocab]
    ):
        raise RuntimeError("Raw Audio8-ASR Embed and lm-head tables are not tied.")
    raw_embed_external = _external_data_map(raw_table)
    raw_main_external = _external_data_map(raw_lm_head)
    if raw_embed_external or raw_main_external:
        if raw_embed_external != raw_main_external:
            raise RuntimeError("Raw Audio8-ASR Embed and lm-head reference different bytes.")
    elif raw_table.raw_data != raw_lm_head.raw_data:
        raise RuntimeError("Raw Audio8-ASR Embed and lm-head are not byte-identical.")

    main_graph = onnx.load(str(main_graph_path), load_external_data=False)
    shared = onnx.load(str(shared_path), load_external_data=False)
    main_initializers = {
        initializer.name: initializer for initializer in main_graph.graph.initializer
    }
    candidates = []
    for node in main_graph.graph.node:
        if (
            node.domain in ("", "ai.onnx")
            and node.op_type == "MatMul"
            and "/lm_head" in node.name
            and len(node.input) >= 2
        ):
            initializer = main_initializers.get(node.input[1])
            if (
                initializer is not None
                and list(initializer.dims) == [hidden, vocab]
                and initializer.data_type in (TensorProto.FLOAT, TensorProto.FLOAT16)
            ):
                candidates.append(initializer)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one optimized floating lm-head initializer; found {len(candidates)}."
        )
    lm_head = candidates[0]
    shared_by_name = {
        initializer.name: initializer for initializer in shared.graph.initializer
    }
    manifest_ref = shared_by_name.get(lm_head.name)
    if (
        manifest_ref is None
        or manifest_ref.data_type != lm_head.data_type
        or list(manifest_ref.dims) != list(lm_head.dims)
        or _external_data_map(manifest_ref) != _external_data_map(lm_head)
    ):
        raise RuntimeError("Optimized lm-head is missing from the shared manifest.")

    gathered_name = "shared_lm_head_embed_gathered"
    gather = onnx.helper.make_node(
        "Gather",
        [lm_head.name, raw_gather.input[1]],
        [gathered_name],
        name="shared_lm_head_embed_gather",
        axis=1,
    )
    transpose = onnx.helper.make_node(
        "Transpose",
        [gathered_name],
        [source_embed.graph.output[0].name],
        name="shared_lm_head_embed_transpose",
        perm=[1, 2, 0],
    )
    del source_embed.graph.node[:]
    source_embed.graph.node.extend((gather, transpose))
    del source_embed.graph.initializer[:]
    source_embed.graph.initializer.append(copy.deepcopy(manifest_ref))
    del source_embed.graph.value_info[:]
    source_embed.graph.output[0].type.tensor_type.elem_type = lm_head.data_type
    old_private_sidecar = embed_path.with_name(embed_path.name + ".data")
    had_private_sidecar = old_private_sidecar.exists()
    Shared_Merged.save_model(source_embed, embed_path)
    external = _external_data_map(manifest_ref)
    return {
        "shared_initializer_count": 1,
        "shared_data_bytes": int(external["length"]),
        "removed_external_file_count": int(
            had_private_sidecar and not old_private_sidecar.exists()
        ),
    }


def _share_tied_embed_with_main(
    embed_path: Path,
    shared_path: Path,
    main_graph_path: Path,
    main_plan,
) -> dict[str, int]:
    if main_plan.method in {"Q2", "Q4", "Q8"}:
        return _share_block_quantized_embed_with_main(
            embed_path,
            shared_path,
            main_graph_path,
            main_plan.block_size,
            int(main_plan.method[1:]),
        )
    if main_plan.method in {"F32", "F16", "DYNAMIC"} or main_plan.uses_float16:
        return _share_float_embed_with_main(embed_path, shared_path, main_graph_path)
    raise RuntimeError(
        f"Tied Audio8-ASR Embed does not support Main method {main_plan.method!r}."
    )


def _f16_overflow_candidates(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    """Return named F16 overflow candidates in deterministic widening order."""
    reserved_names = {node.name for node in model.graph.node if node.name}
    tensor_types = Shared_Merged._tensor_element_types(model)
    graph_positions: dict[str, int] = {}
    candidates: list[onnx.NodeProto] = []
    for position, node in enumerate(model.graph.node):
        if (
            node.domain not in ("", "ai.onnx")
            or node.op_type not in F16_OVERFLOW_CANDIDATE_OPS
            or not any(
                tensor_types.get(output) == TensorProto.FLOAT16
                for output in node.output
                if output
            )
        ):
            continue
        if not node.name:
            output = next((name for name in node.output if name), str(position))
            base = f"F16OverflowCandidate/{node.op_type}/{output}"
            node.name = base
            suffix = 1
            while node.name in reserved_names:
                node.name = f"{base}_{suffix}"
                suffix += 1
            reserved_names.add(node.name)
        candidates.append(node)
        graph_positions[node.name] = position
    priority = {"Pow": 0, "ReduceMean": 1, "ReduceSum": 2, "ReduceSumSquare": 3}
    return sorted(
        candidates,
        key=lambda node: (priority[node.op_type], graph_positions[node.name]),
    )


def _apply_f16_overflow_guards(
    model: onnx.ModelProto,
    graph_label: str,
    requested_names: tuple[str, ...],
) -> int:
    """Widen only explicitly selected Pow/reduction nodes to F32."""
    if not requested_names:
        return 0
    candidates = _f16_overflow_candidates(model)
    by_name = {node.name: node for node in candidates}
    requested = list(dict.fromkeys(requested_names))
    missing = [name for name in requested if name not in by_name]
    if missing:
        available = [node.name for node in candidates]
        raise RuntimeError(
            f"{graph_label}: unknown F16 overflow guard node(s) {missing}; "
            f"available candidates are {available}."
        )
    tensor_types = Shared_Merged._tensor_element_types(model)
    selected = set(requested)
    reserved_nodes = {node.name for node in model.graph.node if node.name}
    reserved_tensors = {initializer.name for initializer in model.graph.initializer}
    reserved_tensors.update(
        output for node in model.graph.node for output in node.output if output
    )

    def unique_tensor(base: str) -> str:
        name = base
        suffix = 1
        while name in reserved_tensors:
            name = f"{base}_{suffix}"
            suffix += 1
        reserved_tensors.add(name)
        return name

    def unique_node(base: str) -> str:
        name = base
        suffix = 1
        while name in reserved_nodes:
            name = f"{base}_{suffix}"
            suffix += 1
        reserved_nodes.add(name)
        return name

    widened_outputs: dict[str, str] = {}
    for source_node in model.graph.node:
        if source_node.name not in selected:
            continue
        for output_name in source_node.output:
            if output_name and tensor_types.get(output_name) == TensorProto.FLOAT16:
                widened_outputs[output_name] = unique_tensor(
                    f"{output_name}_overflow_guard_f32"
                )

    rewritten: list[onnx.NodeProto] = []
    for source_node in model.graph.node:
        if source_node.name not in selected:
            rewritten.append(copy.deepcopy(source_node))
            continue
        node = copy.deepcopy(source_node)
        role = node.name.replace("/", "_").strip("_") or node.op_type
        widened_inputs = 0
        for input_index, input_name in enumerate(node.input):
            if input_name in widened_outputs:
                node.input[input_index] = widened_outputs[input_name]
                widened_inputs += 1
                continue
            if not input_name or tensor_types.get(input_name) != TensorProto.FLOAT16:
                continue
            adapted = unique_tensor(f"{input_name}_overflow_guard_f32")
            rewritten.append(
                onnx.helper.make_node(
                    "Cast",
                    [input_name],
                    [adapted],
                    name=unique_node(f"F16OverflowGuard/{role}/input_{input_index}"),
                    to=TensorProto.FLOAT,
                )
            )
            node.input[input_index] = adapted
            widened_inputs += 1

        output_casts: list[onnx.NodeProto] = []
        for output_index, output_name in enumerate(node.output):
            widened = widened_outputs.get(output_name)
            if widened is None:
                continue
            node.output[output_index] = widened
            output_casts.append(
                onnx.helper.make_node(
                    "Cast",
                    [widened],
                    [output_name],
                    name=unique_node(f"F16OverflowGuard/{role}/output_{output_index}"),
                    to=TensorProto.FLOAT16,
                )
            )
        if not widened_inputs:
            raise RuntimeError(
                f"{graph_label}: selected overflow node {node.name!r} has no F16 input."
            )
        if not output_casts:
            raise RuntimeError(
                f"{graph_label}: selected overflow node {node.name!r} has no F16 output."
            )
        rewritten.append(node)
        rewritten.extend(output_casts)

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    return len(selected)

def _source_model_file_names(folder: Path) -> dict[str, str]:
    del folder
    return dict(Shared_Merged.DEFAULT_MODEL_FILE_NAMES)


def _replace_metadata(path: Path, metadata: dict[str, str]) -> None:
    model = onnx.load(str(path), load_external_data=False)
    model.producer_name = ""
    model.producer_version = ""
    del model.metadata_props[:]
    for key, value in metadata.items():
        model.metadata_props.add(key=str(key), value=str(value))
    onnx.save_model(model, str(path), save_as_external_data=False)


def _remove_obsolete_artifacts(*folders: Path) -> None:
    for folder in folders:
        removed = Shared_Merged.delete_obsolete_strategy_artifacts(folder)
        if removed:
            print(
                f"[Cleanup] Removed {len(removed)} obsolete strategy artifact(s) "
                f"from {folder}."
            )


def _copy_tokenizer(source_folder: Path, target_folder: Path) -> None:
    copy_artifact(source_folder / "tokenizer", target_folder / "tokenizer")


def _repair_processed_graph(name: str) -> None:
    """Restore stable ABI names in an optimized graph."""
    path = Path(OPTIMIZED_FOLDER_PATH) / f"{name}.onnx"
    model = Shared_Merged.load_model(path, load_external_data=False)
    uses_f16 = _module_uses_f16(name)
    restored = Shared_Merged.restore_precision_free_graph_outputs(model)
    static_identity_casts = _remove_static_identity_casts(model)
    split_nbits_biases = (
        _split_cuda_unsupported_matmul_nbits_bias(model)
        if ENABLE_Q4_CUDA_COMPATIBILITY and name in (_ENCODER_MODEL, _MAIN_MODEL)
        else 0
    )
    overflow_guards = (
        _apply_f16_overflow_guards(
            model,
            path.name,
            F16_OVERFLOW_GUARD_NODE_NAMES.get(name, ()),
        )
        if uses_f16
        else 0
    )
    full_f16_rewrites = (
        _finalize_runtime_graph_full_f16(model)
        if FULL_FLOAT16_RUNTIME and name == _EMBED_MODEL
        else 0
    )
    if (
        restored
        or static_identity_casts
        or split_nbits_biases
        or overflow_guards
        or full_f16_rewrites
    ):
        onnx.save(model, str(path))
    if restored:
        print(f"  Restored {len(restored)} precision-free public output name(s).")
    if static_identity_casts:
        print(f"  Removed {static_identity_casts} static identity Cast node(s).")
    if split_nbits_biases:
        print(
            f"  Split {split_nbits_biases} CUDA-unsupported MatMulNBits "
            "bias input(s)."
        )
    if overflow_guards:
        print(f"  Applied {overflow_guards} incremental F16 overflow guard(s).")
    if full_f16_rewrites:
        print(
            f"  Finalized full-F16 runtime ABI/constants with "
            f"{full_f16_rewrites} targeted rewrite(s)."
        )
    elif uses_f16 and name in (_ENCODER_MODEL, _MAIN_MODEL):
        candidates = [node.name for node in _f16_overflow_candidates(model)]
        print(
            "  Full-F16 baseline retained: no overflow guards requested; "
            f"{len(candidates)} Pow/Mean/Sum candidate(s) available."
        )
        for candidate in candidates:
            print(f"    {candidate}")


def _remove_static_identity_casts(model: onnx.ModelProto) -> int:
    """Remove Casts whose initializer input already has the requested dtype."""
    initializer_types = {
        initializer.name: int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    public_outputs = {value.name for value in model.graph.output}
    aliases: dict[str, str] = {}
    for node in model.graph.node:
        if (
            node.domain in ("", "ai.onnx")
            and node.op_type == "Cast"
            and len(node.input) == 1
            and len(node.output) == 1
            and node.output[0] not in public_outputs
        ):
            target_type = next(
                (attribute.i for attribute in node.attribute if attribute.name == "to"),
                None,
            )
            if initializer_types.get(node.input[0]) == target_type:
                aliases[node.output[0]] = node.input[0]
    if not aliases:
        return 0
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = aliases.get(name, name)
    retained = [
        node
        for node in model.graph.node
        if not (
            node.op_type == "Cast"
            and len(node.output) == 1
            and node.output[0] in aliases
        )
    ]
    del model.graph.node[:]
    model.graph.node.extend(retained)
    retained_info = [
        value for value in model.graph.value_info if value.name not in aliases
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    return len(aliases)


def _finalize_runtime_graph_full_f16(model: onnx.ModelProto) -> int:
    """Remove component ABI casts after composition and expose literal F16 I/O."""
    removable_prefixes = (
        "graph_input_cast",
        "graph_output_cast",
        "encoder_graph_input_cast",
        "encoder_graph_output_cast",
    )
    removable_names = {
        "prefill_/Cast",
        "decode_/Cast",
        "BoundaryCast/logits/to_shell",
    }
    graph_outputs = {value.name for value in model.graph.output}
    aliases: dict[str, str] = {}
    output_renames: dict[str, str] = {}
    remove_indices: set[int] = set()
    for index, node in enumerate(model.graph.node):
        target_type = (
            next((attr.i for attr in node.attribute if attr.name == "to"), None)
            if node.op_type == "Cast"
            else None
        )
        if not (
            node.op_type == "Cast"
            and len(node.input) == 1
            and len(node.output) == 1
            and target_type in (TensorProto.FLOAT, TensorProto.FLOAT16)
            and (
                node.name.startswith(removable_prefixes)
                or node.name in removable_names
            )
        ):
            continue
        remove_indices.add(index)
        source, target = node.input[0], node.output[0]
        if target in graph_outputs:
            output_renames[source] = target
        else:
            aliases[target] = source

    def resolve(name: str) -> str:
        visited: set[str] = set()
        while name in aliases:
            if name in visited:
                raise RuntimeError(f"Cycle in F16 ABI cast aliases at {name!r}.")
            visited.add(name)
            name = aliases[name]
        return output_renames.get(name, name)

    rewritten: list[onnx.NodeProto] = []
    for index, source_node in enumerate(model.graph.node):
        if index in remove_indices:
            continue
        node = copy.deepcopy(source_node)
        for input_index, name in enumerate(node.input):
            node.input[input_index] = resolve(name)
        for output_index, name in enumerate(node.output):
            node.output[output_index] = output_renames.get(name, name)
        if node.op_type == "RandomUniformLike":
            dtype = next(
                (attr for attr in node.attribute if attr.name == "dtype"),
                None,
            )
            if dtype is not None and dtype.i == TensorProto.FLOAT:
                dtype.i = TensorProto.FLOAT16
        rewritten.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rewritten)

    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in values:
            value.name = resolve(value.name)
            if (
                value.type.HasField("tensor_type")
                and value.type.tensor_type.elem_type == TensorProto.FLOAT
            ):
                value.type.tensor_type.elem_type = TensorProto.FLOAT16

    converted_constants = 0
    for initializer in model.graph.initializer:
        if (
            initializer.data_type == TensorProto.FLOAT
            and initializer.data_location != TensorProto.EXTERNAL
        ):
            array = onnx.numpy_helper.to_array(initializer).astype(np.float16)
            initializer.CopyFrom(
                onnx.numpy_helper.from_array(array, initializer.name)
            )
            converted_constants += 1
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t") and attribute.t.data_type == TensorProto.FLOAT:
                array = onnx.numpy_helper.to_array(attribute.t).astype(np.float16)
                attribute.t.CopyFrom(
                    onnx.numpy_helper.from_array(array, attribute.t.name)
                )
                converted_constants += 1

    interface_names = {value.name for value in model.graph.input}
    interface_names.update(value.name for value in model.graph.output)
    stale_names = set(aliases)
    seen_info: set[str] = set()
    retained_info: list[onnx.ValueInfoProto] = []
    for value in model.graph.value_info:
        if (
            value.name in stale_names
            or value.name in interface_names
            or value.name in seen_info
        ):
            continue
        retained_info.append(value)
        seen_info.add(value.name)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    return len(remove_indices) + converted_constants


def _split_cuda_unsupported_matmul_nbits_bias(model: onnx.ModelProto) -> int:
    """Move MatMulNBits bias inputs to Add nodes for ORT CUDA compatibility.

    ORT's CPU kernel accepts the optional sixth ``bias`` input, while the CUDA
    kernel in ORT 1.27 rejects it. Keeping the same public output name on a
    following standard Add preserves the graph ABI and exact arithmetic.
    """
    candidate_indices = [
        index
        for index, node in enumerate(model.graph.node)
        if node.domain == "com.microsoft"
        and node.op_type == "MatMulNBits"
        and len(node.input) >= 6
        and bool(node.input[5])
    ]
    if not candidate_indices:
        return 0
    candidates = [model.graph.node[index] for index in candidate_indices]
    malformed = [
        (node.name, len(node.input), len(node.output))
        for node in candidates
        if len(node.input) != 6 or len(node.output) != 1
    ]
    if malformed:
        raise RuntimeError(
            "Unexpected biased MatMulNBits input/output contract: "
            f"{malformed[:5]}."
        )

    reserved_tensors = {value.name for value in model.graph.input}
    reserved_tensors.update(initializer.name for initializer in model.graph.initializer)
    reserved_tensors.update(
        output
        for node in model.graph.node
        for output in node.output
        if output
    )
    reserved_nodes = {node.name for node in model.graph.node if node.name}
    value_info = Shared_Merged.value_info_by_name(model)
    private_value_info: list[onnx.ValueInfoProto] = []
    candidate_indices = set(candidate_indices)
    rewritten: list[onnx.NodeProto] = []

    for node_index, node in enumerate(model.graph.node):
        if node_index not in candidate_indices:
            rewritten.append(copy.deepcopy(node))
            continue

        public_output = node.output[0]
        private_output = f"{public_output}_without_bias"
        suffix = 1
        while private_output in reserved_tensors:
            private_output = f"{public_output}_without_bias_{suffix}"
            suffix += 1
        reserved_tensors.add(private_output)

        matmul = copy.deepcopy(node)
        del matmul.input[5:]
        while matmul.input and not matmul.input[-1]:
            del matmul.input[-1]
        matmul.output[0] = private_output

        add_name = f"{node.name or public_output}/CudaBiasAdd"
        base_name = add_name
        suffix = 1
        while add_name in reserved_nodes:
            add_name = f"{base_name}_{suffix}"
            suffix += 1
        reserved_nodes.add(add_name)
        bias_add = onnx.helper.make_node(
            "Add",
            [private_output, node.input[5]],
            [public_output],
            name=add_name,
        )
        rewritten.extend([matmul, bias_add])

        public_info = value_info.get(public_output)
        if public_info is not None and private_output not in value_info:
            private_info = copy.deepcopy(public_info)
            private_info.name = private_output
            private_value_info.append(private_info)
            value_info[private_output] = private_info

    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    model.graph.value_info.extend(private_value_info)
    return len(candidates)


def build_optimized_merged_bundle() -> dict[str, Path]:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    output_folder.mkdir(parents=True, exist_ok=True)
    model_file_names = _source_model_file_names(source_folder)

    main_name = model_file_names["main"]
    encoder_name = model_file_names["encoder"]
    optimized_main_path = output_folder / main_name
    optimized_encoder_path = output_folder / encoder_name
    # Materialize both donors before replacing an older shared blob. Their private
    # sidecars are deleted after all six deployment graphs have been rebuilt.
    optimized_main = Shared_Merged.load_model(optimized_main_path)
    optimized_encoder = Shared_Merged.load_model(optimized_encoder_path)
    Shared_Merged.namespace_encoder_initializers(optimized_encoder)
    namespaced_main_tensors = Shared_Merged.namespace_internal_tensors(
        optimized_main, marker="_inlfunc_", namespace="main_"
    )
    if namespaced_main_tensors:
        print(
            f"  Namespaced {namespaced_main_tensors} Main function-inlining "
            "tensor(s)."
        )
    Shared_Merged.restore_precision_free_graph_outputs(optimized_main)
    Shared_Merged.restore_precision_free_graph_outputs(optimized_encoder)

    build_plan = Shared_Merged.make_merged_build_plan(model_file_names)
    primary_name = model_file_names["prefill_greedy"]
    for file_name, _, _ in build_plan:
        remove_model_files(output_folder / file_name)
    shared_name = model_file_names["shared_initializers"]
    shared_data_name = model_file_names["shared_initializers_data"]
    remove_model_files(output_folder / shared_name)

    main_plan = _resolved_module_plan(_MAIN_MODEL)
    source_metadata = read_onnx_metadata(
        str(source_folder / model_file_names["metadata"])
    )
    metadata = dict(source_metadata)

    print("\n" + "=" * 60)
    print("Transplanting optimized Encoder/Main into every ASR strategy graph")
    print("=" * 60)

    external_by_name = None
    generated: dict[str, Path] = {}
    for file_name, _, _ in build_plan:
        source_path = source_folder / file_name
        # Materialize raw shell constants before replacing the exporter's shared
        # blob; otherwise their external offsets would refer to the new bundle.
        target = Shared_Merged.load_model(source_path)
        merged = Shared_Merged.transplant_optimized_main(target, optimized_main)
        del target
        merged = Shared_Merged.transplant_optimized_encoder(
            merged, optimized_encoder
        )
        simplified = remove_redundant_casts(merged)
        static_identity_casts = _remove_static_identity_casts(merged)
        split_nbits_biases = (
            _split_cuda_unsupported_matmul_nbits_bias(merged)
            if ENABLE_Q4_CUDA_COMPATIBILITY
            else 0
        )
        full_f16_rewrites = (
            _finalize_runtime_graph_full_f16(merged)
            if FULL_FLOAT16_RUNTIME
            else 0
        )
        if simplified:
            print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
        if static_identity_casts:
            print(f"  Removed {static_identity_casts} static identity Cast node(s).")
        if split_nbits_biases:
            print(
                f"  Split {split_nbits_biases} CUDA-unsupported MatMulNBits "
                "bias input(s)."
            )
        if full_f16_rewrites:
            print(
                f"  Finalized full-F16 runtime ABI/constants with "
                f"{full_f16_rewrites} targeted rewrite(s)."
            )
        if external_by_name is None:
            if file_name != primary_name:
                raise RuntimeError("Shared extraction did not start from PrefillGreedy.")
            external_by_name = Shared_Merged.extract_and_write_shared(
                [merged],
                output_folder / shared_name,
                primary_model=merged,
            )
        else:
            Shared_Merged.redirect_shared_initializers_to_external(
                merged, external_by_name
            )

        output_path = output_folder / file_name
        Shared_Merged.save_model(merged, output_path)
        print(f"  {file_name} ({output_path.stat().st_size} bytes)")
        generated[file_name] = output_path
        del merged
        gc.collect()

    shared_data_path = output_folder / shared_data_name
    print(f"  {shared_data_name} ({shared_data_path.stat().st_size} bytes)")

    if EMBED_WEIGHT_SOURCE == "INDEPENDENT":
        embed_share = share_external_initializers_if_identical(
            output_folder / model_file_names["embed"],
            output_folder / shared_name,
            require_all_external=REQUIRE_EMBED_SHARED_INITIALIZERS,
        )
    else:
        embed_share = _share_tied_embed_with_main(
            output_folder / model_file_names["embed"],
            output_folder / shared_name,
            output_folder / primary_name,
            main_plan,
        )
    shared_embed_initializers = embed_share["shared_initializer_count"]
    print(
        f"  Shared {shared_embed_initializers} optimized Embed initializer(s); "
        f"reused {embed_share['shared_data_bytes']} bytes and removed "
        f"{embed_share['removed_external_file_count']} private sidecar(s)."
    )
    metadata_path = output_folder / model_file_names["metadata"]
    if not metadata_path.exists():
        copy_artifact(
            source_folder / model_file_names["metadata"],
            metadata_path,
            required=True,
        )
    _replace_metadata(metadata_path, metadata)

    # Encoder/Main are optimization donors only. Their private optimized sidecars
    # would duplicate the tensors now stored in the shared deployment blob.
    remove_model_files(optimized_main_path)
    remove_model_files(optimized_encoder_path)
    remove_model_files(output_folder / model_file_names["concat_embed"])

    _copy_tokenizer(source_folder, output_folder)
    return generated


if __name__ == "__main__":
    _remove_obsolete_artifacts(
        Path(ORIGINAL_FOLDER_PATH), Path(OPTIMIZED_FOLDER_PATH)
    )
    _run_optimizer_pipeline()
    for _name in ACTIVE_MODEL_NAMES:
        _repair_processed_graph(_name)
    build_optimized_merged_bundle()
    _storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER_PATH,
        Shared_Merged.DEFAULT_MODEL_FILE_NAMES["shared_initializers"],
    )
    print(f"  Consolidated {_storage['unique_data_ranges']} unique shared range(s).")
    print("\n--- All standalone and merged ASR models processed successfully! ---")
