"""Optimize FunASR-Nano donors and rebuild its Encoder+prefill strategy bundle.

Encoder and Main are optimized once as component donors and transplanted into
every applicable raw strategy shell. Block-quantized Embed is rebuilt directly
over Main's tied lm-head tuple; floating plans exact-share the same table. All
weights are then consolidated into one deployment blob. Optional CTC remains a
standalone auxiliary head.
"""

from __future__ import annotations

import copy
import gc
import shutil
import sys
from pathlib import Path

import onnx


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

import Shared_Merged
from Optimize_ONNX_Common import (
    consolidate_optimized_model_weights,
    QUANTIZATION_F16_OP_BLOCK_LIST,
    OptimizerConfig,
    Plan,
    copy_artifacts,
    remove_redundant_casts,
    resolve_plan,
    rewrite_tied_embed_from_matmul_nbits,
    run_optimizer,
    share_external_initializers_if_identical,
)


# ============================== USER CONFIG ==============================
# Edit this section only.
# F32 is the numerical baseline, F16 is mixed precision, DYNAMIC is portable
# INT8, and Q2/Q4/Q8 are block weight-only. Keep the frontend/residual/logits
# precision selectors and merged-component ABI guards below.

ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Fun_ASR_Nano_ONNX")
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Fun_ASR_Nano_Optimized")

MODEL_FILE_NAMES = Shared_Merged.model_file_names()
MAIN_STEM = Path(MODEL_FILE_NAMES["main"]).stem
EMBED_STEM = Path(MODEL_FILE_NAMES["embed"]).stem
ENCODER_STEM = Path(MODEL_FILE_NAMES["encoder"]).stem
CTC_STEM = Path(MODEL_FILE_NAMES["ctc_decoder"]).stem

WEIGHT_ONLY_ALGORITHM      = "AFFINE_REFINE_V2"
WEIGHT_ONLY_BLOCK_SIZE     = 64
WEIGHT_ONLY_ACCURACY_LEVEL = 4
WEIGHT_ONLY_SYMMETRIC      = True

OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 2

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_OP_BLOCK_LIST = QUANTIZATION_F16_OP_BLOCK_LIST


def _main_nodes_to_exclude(model_path: str) -> list[str]:
    method = MODEL_PLANS[MAIN_STEM].method.upper()
    if method == "F16":
        return list(fun_asr_main_fp32_residual_nodes(model_path))
    if method != "DYNAMIC":
        return []
    model = onnx.load(model_path, load_external_data=False)
    matches = [
        node.name
        for node in model.graph.node
        if node.op_type in ("MatMul", "Gemm") and "/lm_head" in node.name
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(
            f"Expected one named Fun-ASR-Nano lm-head projection; found {matches}."
        )
    return [matches[0]]

MODEL_PLANS = {
    "FunASR_Nano_Encoder": Plan(
        method="Q8",
        opt_level=OPTIMIZER_LEVEL,
        num_heads=0,
        hidden_size=0,
        nodes_to_exclude=lambda path: fun_asr_encoder_fp32_frontend_nodes(path),
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "FunASR_Nano_CTC_Decoder": Plan(
        method="Q8",
        opt_level=OPTIMIZER_LEVEL,
        num_heads=0,
        hidden_size=0,
    ),
    "FunASR_Nano_Decoder_Embed": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
        external=FORCE_EXTERNAL_DATA,
    ),
    "FunASR_Nano_Decoder_Main": Plan(
        method="Q8",
        num_heads=0,
        hidden_size=0,
        external=FORCE_EXTERNAL_DATA,
        optimize=True,
        nodes_to_exclude=_main_nodes_to_exclude,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "FunASR_Nano_TextPrefillGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FunASR_Nano_TextPrefillPenaltyGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FunASR_Nano_TextPrefillSampling": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FunASR_Nano_DecodeGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FunASR_Nano_DecodePenaltyGreedy": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FunASR_Nano_DecodeSampling": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "FunASR_Nano_SharedInitializers": Plan(method="Q8", process=False, optimize=False, transformer=False),
    "ASR_Metadata": Plan(method="F32", process=False, optimize=False, transformer=False),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    block_size=WEIGHT_ONLY_BLOCK_SIZE,
    accuracy_level=WEIGHT_ONLY_ACCURACY_LEVEL,
    quant_symmetric=WEIGHT_ONLY_SYMMETRIC,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    copy_artifacts=(MODEL_FILE_NAMES["metadata"],),
)

# ============================ END USER CONFIG ============================


def _derive_main_num_layers(
    model: onnx.ModelProto,
    *,
    graph_label: str,
) -> int:
    """Derive and validate the decoder layer count from the public KV ABI."""

    def _indices(values, prefix: str) -> tuple[int, ...]:
        names = [value.name for value in values if value.name.startswith(prefix)]
        if not names:
            raise RuntimeError(f"{graph_label} has no {prefix}* tensors.")
        malformed = [name for name in names if not name[len(prefix):].isdigit()]
        if malformed:
            raise RuntimeError(
                f"{graph_label} has malformed {prefix}* tensor names: {malformed}."
            )
        parsed = [int(name[len(prefix):]) for name in names]
        expected = tuple(range(len(parsed)))
        if len(set(parsed)) != len(parsed) or tuple(sorted(parsed)) != expected:
            raise RuntimeError(
                f"{graph_label} has non-contiguous {prefix}* indices: {parsed}."
            )
        return expected

    families = {
        "input keys": _indices(model.graph.input, "in_key_"),
        "input values": _indices(model.graph.input, "in_value_"),
        "output keys": _indices(model.graph.output, "out_key_"),
        "output values": _indices(model.graph.output, "out_value_"),
    }
    layer_indices = families["input keys"]
    mismatches = {
        label: indices
        for label, indices in families.items()
        if indices != layer_indices
    }
    if mismatches:
        raise RuntimeError(
            f"{graph_label} has inconsistent KV layer families: {mismatches}."
        )

    num_layers = len(layer_indices)
    expected_inputs = [
        *(f"in_key_{index}" for index in layer_indices),
        *(f"in_value_{index}" for index in layer_indices),
    ]
    expected_outputs = [
        *(f"out_key_{index}" for index in layer_indices),
        *(f"out_value_{index}" for index in layer_indices),
    ]
    actual_inputs = [
        value.name
        for value in model.graph.input
        if value.name.startswith(("in_key_", "in_value_"))
    ]
    actual_outputs = [
        value.name
        for value in model.graph.output
        if value.name.startswith(("out_key_", "out_value_"))
    ]
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise RuntimeError(
            f"{graph_label} KV tensors are not in the required key-then-value order: "
            f"inputs={actual_inputs}, outputs={actual_outputs}."
        )
    return num_layers


def fun_asr_encoder_fp32_frontend_nodes(model_path: str) -> list[str]:
    """Keep Kaldi feature extraction in F32 through its position-add boundary.

    Public float audio still carries int16-range PCM. Converting the DFT Conv and
    its power-spectrum square to float16 overflows before log compression. Select
    the complete producer ancestry of the first encoder LayerNormalization so the
    converter inserts exactly one F32->F16 adapter at the SANM stack boundary.
    """
    model = onnx.load(model_path, load_external_data=False)
    producer = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    first_norm = next(
        (
            node
            for node in model.graph.node
            if node.op_type == "LayerNormalization" and node.input
        ),
        None,
    )
    if first_norm is None:
        raise RuntimeError("Could not locate the first encoder LayerNormalization boundary.")

    selected: dict[str, onnx.NodeProto] = {}
    pending = [first_norm.input[0]]
    while pending:
        node = producer.get(pending.pop())
        if node is None or node.name in selected:
            continue
        if not node.name:
            raise RuntimeError(
                f"Unnamed {node.op_type} node found in the encoder F32 frontend ancestry."
            )
        selected[node.name] = node
        pending.extend(name for name in node.input if name)

    selected_ops = {node.op_type for node in selected.values()}
    required_ops = {"Conv", "Mul", "Log", "Gather", "Add"}
    if not required_ops.issubset(selected_ops):
        raise RuntimeError(
            "Encoder F32 frontend selection did not cover the expected Kaldi path: "
            f"missing ops {sorted(required_ops - selected_ops)}."
        )
    return list(selected)


def fun_asr_main_fp32_residual_nodes(model_path: str) -> list[str]:
    """Keep overflow-prone decoder residual paths in F32 around F16 branches.

    Real audio embeddings contain values above 1,000. By layer two, one F16
    ``down_proj`` result exceeds 65,504. CPU happens to carry that infinity through
    RMSNorm, while CUDA correctly propagates it to NaN. Keep the residual Adds,
    hidden/final RMSNorms, and down projections in float32; qkv, attention, gate/up,
    o_proj, and lm_head remain float16.
    """
    model = onnx.load(model_path, load_external_data=False)
    producer = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    num_layers = _derive_main_num_layers(
        model,
        graph_label="Fun-ASR-Nano source Main",
    )

    down_projections = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and "/down_proj" in node.name
    ]
    hidden_norms = [
        node
        for node in model.graph.node
        if node.op_type == "SimplifiedLayerNormalization"
        and len(node.input) >= 2
        and node.input[1] in ("hidden_norm_scale", "final_norm_scale")
    ]
    residual_adds = []
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        branch_producers = [producer.get(name) for name in node.input]
        if any(
            branch is not None
            and ("/o_proj" in branch.name or "/down_proj" in branch.name)
            for branch in branch_producers
        ):
            residual_adds.append(node)

    expected = {
        "down projections": (len(down_projections), num_layers),
        "hidden/final norms": (len(hidden_norms), 2 * num_layers + 1),
        "residual adds": (len(residual_adds), 2 * num_layers),
    }
    mismatches = {
        label: (actual, required)
        for label, (actual, required) in expected.items()
        if actual != required
    }
    if mismatches:
        raise RuntimeError(f"Unexpected decoder residual topology for F16 conversion: {mismatches}")

    selected = [*down_projections, *hidden_norms, *residual_adds]
    if any(not node.name for node in selected):
        raise RuntimeError("Unnamed node found in the decoder F32 residual selection.")
    return [node.name for node in selected]


def _embed_share_requires_complete_match() -> bool:
    """Require complete sharing only when Embed weights remain plain F16."""
    return (
        resolve_plan(
            MODEL_PLANS[EMBED_STEM],
            CONFIG,
            model_name=EMBED_STEM,
        ).method
        == "F16"
    )


def _copy_tokenizer_assets(source: Path, destination: Path) -> None:
    assets = ["Qwen3-0.6B"]
    if (source / MODEL_FILE_NAMES["ctc_decoder"]).is_file():
        assets.append("multilingual.tiktoken")
    for asset in assets:
        src = source / asset
        dst = destination / asset
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"[Tokenizer] Copied {asset} -> {dst}")
        elif src.is_file():
            shutil.copyfile(src, dst)
            print(f"[Tokenizer] Copied {asset} -> {dst}")
        else:
            raise FileNotFoundError(src)


def _remove_optimizer_metadata(model: onnx.ModelProto) -> None:
    retained = [
        (prop.key, prop.value)
        for prop in model.metadata_props
        if not prop.key.startswith("optimizer_")
    ]
    del model.metadata_props[:]
    for key, value in retained:
        model.metadata_props.add(key=key, value=value)


def _cleanup_obsolete_artifacts(*folders: Path) -> None:
    for folder in folders:
        removed = Shared_Merged.delete_obsolete_strategy_artifacts(folder, MODEL_FILE_NAMES)
        if removed:
            print(f"[Cleanup] Removed {len(removed)} obsolete strategy artifact(s) from {folder}.")


def _save_and_stamp(model: onnx.ModelProto, path: Path) -> None:
    simplified = remove_redundant_casts(model)
    _remove_optimizer_metadata(model)
    if simplified:
        print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
    logits_bridge = (
        _restore_fp32_logits_bridge(model)
        if MODEL_PLANS[MAIN_STEM].method.upper() == "F16"
        and any(node.op_type == "MatMul" and "/lm_head" in node.name for node in model.graph.node)
        else 0
    )
    if logits_bridge:
        print("  Restored the float32 Main-to-selection logits bridge.")
    logits_annotations = _normalize_fp32_logits_annotations(model)
    if logits_annotations:
        print(
            f"  Normalized {logits_annotations} stable FP32 logits annotation(s)."
        )
    Shared_Merged.save_model(model, path)
    print(f"  {path.name} ({path.stat().st_size} bytes)")


def _save_structure_and_stamp(model: onnx.ModelProto, path: Path) -> None:
    """Rewrite only the graph proto, preserving referenced packed-type sidecars."""
    simplified = remove_redundant_casts(model)
    _remove_optimizer_metadata(model)
    if simplified:
        print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
    onnx.save(model, str(path))
    sidecar = path.with_name(path.name + ".data")
    referenced_locations = Shared_Merged._external_locations(path)
    removed_unreferenced_sidecar = sidecar.exists() and sidecar.name not in referenced_locations
    if removed_unreferenced_sidecar:
        sidecar.unlink()
    suffix = (
        " + preserved packed sidecar"
        if sidecar.name in referenced_locations
        else " + removed unreferenced sidecar"
        if removed_unreferenced_sidecar
        else ""
    )
    print(f"  {path.name} ({path.stat().st_size} bytes{suffix})")


def _restore_main_residual_precision_casts(model: onnx.ModelProto) -> int:
    """Reinsert required casts around the F32 residual islands in an F16 Main.

    The F16 converter creates these adapters, but the trailing simplifier can
    remove them after failed/stale type inference. Rebuild them from the strict
    model-specific topology after all generic optimization has finished.
    """
    producer = {
        output: node
        for node in model.graph.node
        for output in node.output
        if output
    }
    initializer_types = {
        initializer.name: int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    num_layers = _derive_main_num_layers(
        model,
        graph_label="Fun-ASR-Nano optimized Main",
    )

    qkv = [node for node in model.graph.node if node.op_type == "MatMul" and "/qkv" in node.name]
    gate_up = [
        node for node in model.graph.node
        if node.op_type == "MatMul" and "/gate_up_proj" in node.name
    ]
    down = [
        node for node in model.graph.node
        if node.op_type == "MatMul" and "/down_proj" in node.name
    ]
    lm_head = [
        node for node in model.graph.node
        if node.op_type == "MatMul" and "/lm_head" in node.name
    ]
    residual_adds = []
    for node in model.graph.node:
        if node.op_type != "Add":
            continue
        if any(
            (branch := producer.get(name)) is not None and "/o_proj" in branch.name
            for name in node.input
        ):
            residual_adds.append(node)

    expected = {
        "qkv": (len(qkv), num_layers),
        "gate_up": (len(gate_up), num_layers),
        "down": (len(down), num_layers),
        "lm_head": (len(lm_head), 1),
        "attention residual adds": (len(residual_adds), num_layers),
    }
    mismatches = {
        label: (actual, required)
        for label, (actual, required) in expected.items()
        if actual != required
    }
    if mismatches:
        raise RuntimeError(f"Cannot restore decoder precision adapters: {mismatches}")

    cast_specs: dict[int, tuple[int, int, str]] = {}
    for node in (*qkv, *gate_up, *lm_head):
        if initializer_types.get(node.input[1]) != onnx.TensorProto.FLOAT16:
            raise RuntimeError(f"Expected F16 weight for {node.name!r}.")
        cast_specs[id(node)] = (0, onnx.TensorProto.FLOAT16, "to_f16_branch")
    for node in down:
        if initializer_types.get(node.input[1]) != onnx.TensorProto.FLOAT:
            raise RuntimeError(f"Expected F32 down-projection weight for {node.name!r}.")
        cast_specs[id(node)] = (0, onnx.TensorProto.FLOAT, "to_f32_down_proj")

    residual_input_specs: dict[int, tuple[int, int, str]] = {}
    for node in residual_adds:
        branch_indices = [
            index
            for index, name in enumerate(node.input)
            if (branch := producer.get(name)) is not None and "/o_proj" in branch.name
        ]
        if len(branch_indices) != 1:
            raise RuntimeError(
                f"Expected one o_proj branch input for residual node {node.name!r}."
            )
        residual_input_specs[id(node)] = (
            branch_indices[0], onnx.TensorProto.FLOAT, "to_f32_residual"
        )

    reserved = {value.name for value in model.graph.input}
    reserved.update(initializer.name for initializer in model.graph.initializer)
    reserved.update(output for node in model.graph.node for output in node.output if output)
    new_nodes: list[onnx.NodeProto] = []
    inserted = 0
    for node in model.graph.node:
        spec = cast_specs.get(id(node)) or residual_input_specs.get(id(node))
        if spec is not None:
            input_index, target_type, role = spec
            source = node.input[input_index]
            source_node = producer.get(source)
            already_cast = False
            if source_node is not None and source_node.op_type == "Cast":
                cast_to = next(
                    (attribute.i for attribute in source_node.attribute if attribute.name == "to"),
                    None,
                )
                already_cast = cast_to == target_type
            if not already_cast:
                base = f"{source}_{role}"
                adapted = base
                suffix = 1
                while adapted in reserved:
                    adapted = f"{base}_{suffix}"
                    suffix += 1
                reserved.add(adapted)
                new_nodes.append(
                    onnx.helper.make_node(
                        "Cast",
                        [source],
                        [adapted],
                        name=f"RestoredPrecisionCast/{role}/{inserted}",
                        to=target_type,
                    )
                )
                node.input[input_index] = adapted
                inserted += 1
        new_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return inserted


def _restore_fp32_logits_bridge(model: onnx.ModelProto) -> int:
    """Normalize the F16 lm-head boundary to one stable float32 ``logits`` tensor."""
    lm_heads = [
        node
        for node in model.graph.node
        if node.op_type == "MatMul" and "/lm_head" in node.name
    ]
    if len(lm_heads) != 1:
        raise RuntimeError(f"Expected one decoder lm_head MatMul, found {len(lm_heads)}.")
    lm_head = lm_heads[0]
    initializer_types = {
        initializer.name: int(initializer.data_type)
        for initializer in model.graph.initializer
    }
    if initializer_types.get(lm_head.input[1]) != onnx.TensorProto.FLOAT16:
        raise RuntimeError(f"Expected F16 lm_head weight for {lm_head.name!r}.")

    raw_name = lm_head.output[0]
    float_casts = []
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.input or node.input[0] != raw_name:
            continue
        cast_to = next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )
        if cast_to == onnx.TensorProto.FLOAT:
            float_casts.append(node)
    if len(float_casts) > 1:
        raise RuntimeError(
            f"Expected at most one float32 consumer Cast for lm_head output {raw_name!r}."
        )

    reserved = {value.name for value in model.graph.input}
    reserved.update(initializer.name for initializer in model.graph.initializer)
    reserved.update(output for node in model.graph.node for output in node.output if output)
    bridge = float_casts[0] if float_casts else None
    bridge_output = bridge.output[0] if bridge is not None else None
    if bridge is not None and raw_name != "logits" and bridge_output == "logits":
        # The graph already has the right Cast topology, but failed/stale shape
        # inference can leave an internal ``logits`` value_info declared F16.
        # ORT rejects that contradiction even though the Cast's ``to`` is FLOAT.
        for values in (model.graph.input, model.graph.output, model.graph.value_info):
            for value in values:
                if value.name == "logits" and value.type.HasField("tensor_type"):
                    value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        return 0

    private_name = raw_name
    if private_name == "logits":
        base = "logits_main_f16"
        private_name = base
        suffix = 1
        while private_name in reserved:
            private_name = f"{base}_{suffix}"
            suffix += 1
        reserved.add(private_name)
        lm_head.output[0] = private_name

    old_bridge_output = bridge_output
    if bridge is None:
        bridge = onnx.helper.make_node(
            "Cast",
            [private_name],
            ["logits"],
            name="RestoredPrecisionCast/logits/to_f32_shell",
            to=onnx.TensorProto.FLOAT,
        )
        index = next(index for index, node in enumerate(model.graph.node) if node is lm_head)
        model.graph.node.insert(index + 1, bridge)
        bridge = model.graph.node[index + 1]
        source_to_rewire = raw_name
    else:
        bridge.input[0] = private_name
        bridge.output[0] = "logits"
        source_to_rewire = old_bridge_output

    sources_to_rewire = {
        name
        for name in (raw_name, source_to_rewire)
        if name and name != "logits"
    }
    if sources_to_rewire:
        for node in model.graph.node:
            if node is bridge:
                continue
            for input_index, name in enumerate(node.input):
                if name in sources_to_rewire:
                    node.input[input_index] = "logits"

    info_by_name = {
        value.name: value
        for value in list(model.graph.input)
        + list(model.graph.output)
        + list(model.graph.value_info)
    }
    logits_info = info_by_name.get("logits")
    if logits_info is None:
        template = info_by_name.get(old_bridge_output) or info_by_name.get(raw_name)
        if template is None:
            logits_info = onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, None
            )
        else:
            logits_info = onnx.ValueInfoProto()
            logits_info.CopyFrom(template)
            logits_info.name = "logits"
        model.graph.value_info.append(logits_info)
    logits_info.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    kept_info = [
        value
        for value in model.graph.value_info
        if value.name == "logits" or value.name not in {private_name, old_bridge_output}
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_info)
    for annotation in model.graph.quantization_annotation:
        if annotation.tensor_name == old_bridge_output:
            annotation.tensor_name = "logits"
        for parameter in annotation.quant_parameter_tensor_names:
            if parameter.value == old_bridge_output:
                parameter.value = "logits"
    return 1


def _float_logits_casts(model: onnx.ModelProto) -> list[onnx.NodeProto]:
    result = []
    for node in model.graph.node:
        if node.op_type != "Cast" or "logits" not in node.output:
            continue
        cast_to = next(
            (attribute.i for attribute in node.attribute if attribute.name == "to"),
            None,
        )
        if cast_to == onnx.TensorProto.FLOAT:
            result.append(node)
    return result


def _normalize_fp32_logits_annotations(model: onnx.ModelProto) -> int:
    """Make the one stable ``logits`` Cast and every declaration agree on F32."""
    casts = _float_logits_casts(model)
    if not casts:
        return 0
    if len(casts) != 1:
        raise RuntimeError(
            f"Expected one Cast producing FP32 logits, found {len(casts)}."
        )
    changed = 0
    declarations = 0
    for values in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in values:
            if value.name != "logits" or not value.type.HasField("tensor_type"):
                continue
            declarations += 1
            if value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
                value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
                changed += 1
    if declarations == 0:
        model.graph.value_info.append(
            onnx.helper.make_tensor_value_info(
                "logits",
                onnx.TensorProto.FLOAT,
                None,
            )
        )
        changed += 1
    return changed


def _repair_processed_graph_outputs(name: str) -> None:
    """Restore stable public names immediately after the shared F16 pipeline."""
    path = Path(OPTIMIZED_FOLDER_PATH) / f"{name}.onnx"
    model = Shared_Merged.load_model(path, load_external_data=False)
    restored = Shared_Merged.restore_precision_free_graph_outputs(model)
    precision_casts = (
        _restore_main_residual_precision_casts(model)
        if name == MAIN_STEM
        and MODEL_PLANS[name].method.upper() == "F16"
        else 0
    )
    logits_bridge = (
        _restore_fp32_logits_bridge(model)
        if name == MAIN_STEM
        and MODEL_PLANS[name].method.upper() == "F16"
        else 0
    )
    logits_annotations = _normalize_fp32_logits_annotations(model)
    _remove_optimizer_metadata(model)
    onnx.save(model, str(path))
    if restored:
        print(f"  Restored {len(restored)} precision-free public output name(s).")
    if precision_casts:
        print(f"  Restored {precision_casts} F32/F16 residual-boundary Cast node(s).")
    if logits_bridge:
        print("  Restored the float32 Main-to-selection logits bridge.")
    if logits_annotations:
        print(
            f"  Normalized {logits_annotations} stable FP32 logits annotation(s)."
        )


def build_quantized_merged_bundle() -> dict[str, Path]:
    source = Path(ORIGINAL_FOLDER_PATH)
    destination = Path(OPTIMIZED_FOLDER_PATH)
    main_path = destination / MODEL_FILE_NAMES["main"]
    encoder_path = destination / MODEL_FILE_NAMES["encoder"]
    embed_path = destination / MODEL_FILE_NAMES["embed"]
    print(
        f"\n{'=' * 60}\n"
        "Transplanting optimized Encoder + Main + Embed into merged shells\n"
        f"{'=' * 60}"
    )
    build_plan = Shared_Merged.make_merged_build_plan(MODEL_FILE_NAMES)
    for file_name, _, _ in build_plan:
        for candidate in (
            destination / file_name,
            destination / (file_name + ".data"),
        ):
            candidate.unlink(missing_ok=True)

    optimized_main = Shared_Merged.load_model(main_path)
    optimized_encoder = Shared_Merged.load_model(encoder_path)
    Shared_Merged.namespace_encoder_initializers(optimized_encoder)
    namespaced_main_tensors = Shared_Merged.namespace_internal_tensors(
        optimized_main,
        marker="_inlfunc_",
        namespace="main_",
    )
    if namespaced_main_tensors:
        print(
            f"  Namespaced {namespaced_main_tensors} Main function-inlining tensor(s)."
        )
    for label, donor in (
        (MODEL_FILE_NAMES["main"], optimized_main),
        (MODEL_FILE_NAMES["encoder"], optimized_encoder),
    ):
        restored_outputs = Shared_Merged.restore_precision_free_graph_outputs(donor)
        if restored_outputs:
            print(
                f"  Restored {len(restored_outputs)} precision-free public output(s) "
                f"in {label}."
            )

    main_plan = resolve_plan(MODEL_PLANS[MAIN_STEM], CONFIG, model_name=MAIN_STEM)
    packed_tied_embed = main_plan.method in {"Q2", "Q4", "Q8"}
    if packed_tied_embed:
        quantized_embed_full = Shared_Merged.load_model(
            source / MODEL_FILE_NAMES["embed"]
        )
        tied_report = rewrite_tied_embed_from_matmul_nbits(
            quantized_embed_full,
            optimized_main,
            alias_prefix="funasr_tied_embed_",
        )
        quantized_embed = copy.deepcopy(quantized_embed_full)
        print(
            f"  Rebuilt tied Embed from Main's Q{tied_report['bits']} tuple; "
            f"reused {tied_report['shared_data_bytes']} packed bytes."
        )
    elif main_plan.method == "DYNAMIC":
        quantized_embed_full = Shared_Merged.load_model(
            source / MODEL_FILE_NAMES["embed"]
        )
        quantized_embed = copy.deepcopy(quantized_embed_full)
        print("  Reused Main's floating tied lm-head table for dynamic Embed.")
    else:
        # Floating plans retain the separately converted Embed donor. Exact
        # transpose sharing below collapses it with lm-head when dtypes match.
        quantized_embed = Shared_Merged.load_model(embed_path, load_external_data=False)
        quantized_embed_full = Shared_Merged.load_model(embed_path)
        embed_initializer_remap = Shared_Merged.namespace_conflicting_initializers(
            optimized_main,
            [quantized_embed, quantized_embed_full],
        )
        if embed_initializer_remap:
            print(
                "  Namespaced independently optimized Embed initializers: "
                + ", ".join(sorted(embed_initializer_remap))
            )

    shared_path = destination / MODEL_FILE_NAMES["shared_initializers"]
    shared_path.unlink(missing_ok=True)
    (destination / MODEL_FILE_NAMES["shared_initializers_data"]).unlink(missing_ok=True)
    external_by_name = None
    generated: dict[str, Path] = {}
    for file_name, _, _ in build_plan:
        source_path = source / file_name
        # Materialize the source target before its Main is replaced. Small rotary/mask
        # shell constants currently live in the exporter's shared blob; carrying their
        # old external offsets into the newly rebuilt quantized blob would make every
        # non-primary strategy invalid. Main is momentarily loaded too, but only its
        # shell initializers survive transplant and save.
        target = Shared_Merged.load_model(source_path)
        model = Shared_Merged.transplant_quantized_components(
            target,
            optimized_main,
            quantized_embed,
        )
        del target
        model = Shared_Merged.transplant_optimized_encoder(
            model,
            optimized_encoder,
        )
        if external_by_name is None:
            if file_name != MODEL_FILE_NAMES["prefill_greedy"]:
                raise RuntimeError("Shared extraction did not start from PrefillGreedy.")
            external_by_name = Shared_Merged.extract_and_write_shared(
                [model, quantized_embed_full],
                shared_path,
                primary_model=model,
                embed_model=quantized_embed_full,
            )
        else:
            Shared_Merged.redirect_shared_initializers_to_external(
                model,
                external_by_name,
            )
        output_path = destination / file_name
        _save_and_stamp(model, output_path)
        generated[file_name] = output_path
        del model
        gc.collect()

    if packed_tied_embed:
        Shared_Merged.redirect_shared_initializers_to_external(
            quantized_embed,
            external_by_name,
        )
    del quantized_embed_full
    _save_structure_and_stamp(quantized_embed, embed_path)
    shared_data = destination / MODEL_FILE_NAMES["shared_initializers_data"]
    print(f"  {shared_data.name} ({shared_data.stat().st_size} bytes)")
    require_complete_embed_share = _embed_share_requires_complete_match()
    embed_share = share_external_initializers_if_identical(
        embed_path,
        shared_path,
        require_all_external=require_complete_embed_share,
    )
    retained_private_embed_data = embed_path.with_name(embed_path.name + ".data")
    if not require_complete_embed_share and retained_private_embed_data.exists():
        print(
            "  Standalone Embed retained its independently optimized private "
            "sidecar."
        )
    print(
        "  Standalone Embed exact-match sharing: "
        f"{embed_share['shared_initializer_count']} initializer(s), "
        f"{embed_share['shared_data_bytes']} reused bytes, "
        f"{embed_share['removed_external_file_count']} private sidecar(s) removed."
    )

    # Encoder/Main were optimization donors only. Their private sidecars would
    # duplicate tensors now stored in the shared deployment blob.
    for donor_path in (main_path, encoder_path):
        for location in Shared_Merged._external_locations(donor_path):
            candidate = donor_path.parent / location
            if candidate.name != MODEL_FILE_NAMES["shared_initializers_data"]:
                candidate.unlink(missing_ok=True)
        donor_path.unlink(missing_ok=True)
        donor_path.with_name(donor_path.name + ".data").unlink(missing_ok=True)

    return generated


def main() -> None:
    source = Path(ORIGINAL_FOLDER_PATH)
    destination = Path(OPTIMIZED_FOLDER_PATH)
    _cleanup_obsolete_artifacts(
        source,
        destination,
    )
    active_plans = dict(MODEL_PLANS)
    if not (source / MODEL_FILE_NAMES["ctc_decoder"]).exists():
        active_plans.pop(CTC_STEM, None)
    main_plan = resolve_plan(active_plans[MAIN_STEM], CONFIG, model_name=MAIN_STEM)
    if main_plan.method in {"Q2", "Q4", "Q8", "DYNAMIC"}:
        active_plans.pop(EMBED_STEM, None)
    CONFIG.model_plans = active_plans
    def repair_processed_model(name, plan, path) -> None:
        del plan, path
        _repair_processed_graph_outputs(name)

    run_optimizer(
        CONFIG,
        after_model=repair_processed_model,
        copy_configured_artifacts=False,
        print_completion=False,
        reset_output_folder=True,
    )
    copy_artifacts(CONFIG)
    build_quantized_merged_bundle()
    _copy_tokenizer_assets(source, destination)
    storage = consolidate_optimized_model_weights(
        destination,
        MODEL_FILE_NAMES["shared_initializers"],
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")

    print("\n--- All models processed successfully! ---")


if __name__ == "__main__":
    main()
