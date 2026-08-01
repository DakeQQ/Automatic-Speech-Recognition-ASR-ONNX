"""Optimize Dolphin CN-Dialect streaming and rebuild one shared merged pair.

Only PrefillGreedy is converted directly. Its optimized, unprefixed decoder Main
is transplanted into DecodeGreedy, then all large decoder/embed initializers are
streamed once into Dolphin_SharedInitializers.onnx.data.
"""

from __future__ import annotations

import gc
import shutil
import sys
from pathlib import Path

import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (SCRIPT_DIR, *SCRIPT_DIR.parents):
    if (candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

import Shared_Merged
from Optimize_ONNX_Common import (
    assert_no_large_unquantized_linear_weights,
    collect_target_only_shared_shell_initializers,
    consolidate_optimized_model_weights,
    DEFAULT_F16_OP_BLOCK_LIST,
    OptimizerConfig,
    Plan,
    producer_ancestry_node_names,
    remove_model_files,
    remove_redundant_casts,
    resolve_plan,
    run_optimizer,
    share_external_initializers_if_identical,
)


def exclude_merged_shell_and_gather_nodes(model_path: str) -> list[str]:
    """Keep reusable shells and all Gather nodes stable across both graphs."""
    model = onnx.load(model_path, load_external_data=False)
    excluded = [
        node.name
        for node in model.graph.node
        if node.name
        and (
            node.op_type == "Gather"
            or any(
                output.startswith(Shared_Merged.SHELL_PREFIXES)
                for output in node.output
            )
        )
    ]
    del model
    return excluded


def exclude_encoder_frontend_nodes(model_path: str) -> list[str]:
    """Keep raw-PCM fbank through the subsampling projection in float32.

    F16 squaring in the Kaldi power-spectrum path overflows for int16-range PCM
    and turns the recurrent encoder state into NaN. Walk the exact producer
    ancestry of the projection output instead of relying on name prefixes.
    """
    return producer_ancestry_node_names(
        model_path,
        "/embed/Add_output_0",
        graph_label="Dolphin streaming encoder F32 frontend",
    )


PRIMARY_MERGED_STEM = Path(Shared_Merged.PREFILL_GREEDY_MODEL_NAME).stem

# ============================== USER CONFIG ==============================
# Edit this section only.
# Keep the validated frontend overflow selector, shell/Gather exclusions, and
# post-conversion boundary repair when changing precision or quantization.

ORIGINAL_FOLDER_PATH  = str(SCRIPT_DIR / "Dolphin_CN_Dialect_Streaming_ONNX")
OPTIMIZED_FOLDER_PATH = str(
    SCRIPT_DIR / "Dolphin_CN_Dialect_Streaming_Optimized"
)

# F32: no precision change; F16: mixed precision; DYNAMIC: portable INT8;
# Q4/Q8: AFFINE_REFINE_V2 block weight-only. Q2 requires another algorithm.
ENCODER_METHOD      = "Q8"
DECODER_MAIN_METHOD = "Q8"

WEIGHT_ONLY_ALGORITHM      = "AFFINE_REFINE_V2"
OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 1

DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_OP_BLOCK_LIST = DEFAULT_F16_OP_BLOCK_LIST

COPY_ARTIFACTS = (
    "vocab_Dolphin_CN_Dialect.txt",
)

# Encoder remains standalone. Decoder Main is converted exactly once inside
# PrefillGreedy. optimize=False preserves the shell/Main tensor-name ABI used
# when that Main is transplanted into DecodeGreedy.
MODEL_PLANS = {
    "Dolphin_Encoder": Plan(
        method=ENCODER_METHOD,
        num_heads=0,
        hidden_size=0,
        external=True,
        nodes_to_exclude=(
            exclude_encoder_frontend_nodes
            if ENCODER_METHOD.upper() == "F16"
            else None
        ),
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "Dolphin_PrefillGreedy": Plan(
        method=DECODER_MAIN_METHOD,
        optimize=True,
        transformer=True,
        num_heads=0,
        hidden_size=0,
        external=True,
        nodes_to_exclude=exclude_merged_shell_and_gather_nodes,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "Dolphin_DecodeGreedy":         Plan(method=DECODER_MAIN_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_SharedInitializers":   Plan(method=DECODER_MAIN_METHOD, process=False, optimize=True, transformer=True),
    "ASR_Metadata":                 Plan(method="F32", process=False, optimize=True, transformer=True),
}

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    weight_only_algorithm=WEIGHT_ONLY_ALGORITHM,
    dynamic_weight_type=DYNAMIC_WEIGHT_TYPE,
    dynamic_per_channel=DYNAMIC_PER_CHANNEL,
    dynamic_reduce_range=DYNAMIC_REDUCE_RANGE,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=OPTIMIZER_LEVEL,
    optimizer_only_onnxruntime=OPTIMIZER_ONLY_ONNXRUNTIME,
    f16_keep_io_types=F16_KEEP_IO_TYPES,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    copy_artifacts=COPY_ARTIFACTS,
)

# ============================ END USER CONFIG ============================


def code_model_file_names() -> dict[str, str]:
    return dict(Shared_Merged.DEFAULT_MODEL_FILE_NAMES)


def write_metadata(path: Path, metadata: dict[str, str], *, replace: bool = True) -> None:
    model = onnx.load(str(path), load_external_data=False)
    values = {} if replace else {prop.key: prop.value for prop in model.metadata_props}
    values.update({str(key): str(value) for key, value in metadata.items()})
    del model.metadata_props[:]
    for key in sorted(values):
        model.metadata_props.add(key=key, value=values[key])
    onnx.save(model, str(path))


def persist(
    model: onnx.ModelProto,
    path: Path,
) -> None:
    simplified = remove_redundant_casts(model)
    if simplified:
        print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
    Shared_Merged.prune_unreachable_nodes(model)
    Shared_Merged.save_model(model, path)
    print(f"  {path.name} ({path.stat().st_size} bytes)")


def repair_float16_encoder_frontend_boundary(
    model: onnx.ModelProto,
) -> bool:
    """Insert the adapter missed by conversion at the blocked frontend edge."""
    boundary = "/embed/Add_output_0"
    internal = boundary + "_before_f16"
    nodes = list(model.graph.node)
    producers = {
        output: (index, node)
        for index, node in enumerate(nodes)
        for output in node.output
        if output
    }
    existing = producers.get(boundary)
    if existing is None:
        raise RuntimeError(
            f"Converted streaming encoder has no producer for {boundary!r}."
        )
    _, boundary_producer = existing
    if (
        boundary_producer.op_type == "Cast"
        and boundary_producer.name == "DolphinEncoderF16FrontendCast"
        and list(boundary_producer.input) == [internal]
    ):
        inserted = False
    else:
        if internal in producers:
            raise RuntimeError(f"Streaming encoder adapter collision: {internal!r}.")
        producer_index, producer = existing
        output_index = list(producer.output).index(boundary)
        producer.output[output_index] = internal
        adapter = onnx.helper.make_node(
            "Cast",
            inputs=[internal],
            outputs=[boundary],
            name="DolphinEncoderF16FrontendCast",
            to=onnx.TensorProto.FLOAT16,
        )
        nodes.insert(producer_index + 1, adapter)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        inserted = True

    # Failed symbolic inference leaves FLOAT declarations on converted encoder
    # intermediates. Graph inputs/outputs retain the public ABI; internal types
    # are safely inferred from nodes after the explicit adapter is present.
    pruned = len(model.graph.value_info)
    if pruned:
        del model.graph.value_info[:]
    return inserted or bool(pruned)


def repair_float16_standalone_outputs() -> None:
    """Repair public output aliases without materializing external weights."""
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    for stem, configured_plan in MODEL_PLANS.items():
        if stem == PRIMARY_MERGED_STEM:
            continue
        plan = resolve_plan(configured_plan, CONFIG)
        if not (plan.fp16 or plan.method == "F16"):
            continue
        path = output_folder / f"{stem}.onnx"
        if not path.exists():
            continue
        model = onnx.load(str(path), load_external_data=False)
        restored = Shared_Merged.restore_float16_public_output_names(
            model,
            path.name,
        )
        frontend_repaired = False
        if stem == "Dolphin_Encoder":
            frontend_repaired = repair_float16_encoder_frontend_boundary(model)
        if restored or frontend_repaired:
            onnx.save(model, str(path))
            if restored:
                print(
                    f"  Restored {len(restored)} float16 public output name(s) "
                    f"in {path.name}."
                )
            if frontend_repaired:
                print(
                    "  Repaired the float32 frontend-to-F16 encoder boundary "
                    f"in {path.name}."
                )
        del model
        gc.collect()


def build_optimized_merged_bundle() -> dict[str, object]:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    output_folder.mkdir(parents=True, exist_ok=True)
    model_file_names = code_model_file_names()

    available = list(Shared_Merged.make_merged_build_plan(model_file_names))
    primary_name = model_file_names["prefill_greedy"]
    primary_path = output_folder / primary_name

    primary_plan = resolve_plan(MODEL_PLANS[PRIMARY_MERGED_STEM], CONFIG)
    converted_method = primary_plan.fp16 or primary_plan.method == "F16"

    # Materialize the private optimizer sidecar once, then repair aliases that
    # onnxslim may leave after removing precision-free float16 Cast nodes.
    optimized_primary = Shared_Merged.load_model(primary_path)
    if converted_method:
        restored_boundaries = Shared_Merged.restore_float16_merged_boundary_names(
            optimized_primary
        )
    else:
        restored_boundaries = {}
    if restored_boundaries:
        print(
            "  Restored float16 merge-boundary names: "
            + ", ".join(sorted(restored_boundaries.values()))
        )
    for file_name, _, _ in available:
        if file_name != primary_name:
            remove_model_files(output_folder / file_name)

    shared_name = model_file_names["shared_initializers"]
    shared_data_name = model_file_names["shared_initializers_data"]
    remove_model_files(output_folder / shared_name)

    print("\n" + "=" * 60)
    print("Transplanting one optimized Dolphin Main into DecodeGreedy")
    print("=" * 60)

    additional_shared = collect_target_only_shared_shell_initializers(
        source_folder,
        [
            source_folder / file_name
            for file_name, _, _ in available
            if file_name != primary_name
        ],
        optimized_primary,
        Shared_Merged.SHELL_PREFIXES,
    )
    if additional_shared:
        print(
            "  Preserving target-only shared shell initializers: "
            + ", ".join(sorted(additional_shared))
        )
    external_by_name = Shared_Merged.extract_and_write_shared(
        [optimized_primary],
        output_folder / shared_name,
        primary_model=optimized_primary,
        additional_shared=additional_shared,
    )

    generated = {primary_name: primary_path}
    for file_name, _, _ in available:
        if file_name == primary_name:
            continue
        source_path = source_folder / file_name
        # Structure-only load: Main is replaced before save, so the source shared
        # weights are never materialized for the decode target.
        target = onnx.load(str(source_path), load_external_data=False)
        merged = Shared_Merged.transplant_quantized_main(target, optimized_primary)
        del target
        Shared_Merged.redirect_shared_initializers_to_external(
            merged, external_by_name
        )
        output_path = output_folder / file_name
        persist(merged, output_path)
        generated[file_name] = output_path
        del merged
        gc.collect()

    # Transplant the canonical Main before removing an ArgMax-only precision
    # bridge from the completed primary graph.
    persist(optimized_primary, primary_path)

    shared_model_path = output_folder / shared_name
    shared_data_path = output_folder / shared_data_name
    print(f"  {shared_data_name} ({shared_data_path.stat().st_size} bytes)")

    metadata_path = output_folder / model_file_names["metadata"]
    if not metadata_path.exists():
        shutil.copy2(source_folder / model_file_names["metadata"], metadata_path)
    return {
        "graphs": generated,
        "shared_model": shared_model_path,
        "shared_data": shared_data_path,
    }


def main() -> None:
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    run_optimizer(CONFIG, reset_output_folder=True)
    repair_float16_standalone_outputs()
    for stem in ("Dolphin_Encoder", PRIMARY_MERGED_STEM):
        plan = resolve_plan(MODEL_PLANS[stem], CONFIG, model_name=stem)
        if plan.method not in {"Q2", "Q4", "Q8"}:
            continue
        model = onnx.load(str(output_folder / f"{stem}.onnx"), load_external_data=False)
        assert_no_large_unquantized_linear_weights(
            model,
            graph_label=f"optimized streaming {stem}",
        )
        del model
    bundle = build_optimized_merged_bundle()
    for standalone_path in sorted(output_folder.glob("*.onnx")):
        if standalone_path != bundle["shared_model"]:
            share_external_initializers_if_identical(
                standalone_path,
                bundle["shared_model"],
                require_all_external=False,
            )
    storage = consolidate_optimized_model_weights(
        output_folder,
        Shared_Merged.DEFAULT_MODEL_FILE_NAMES["shared_initializers"],
    )
    print(
        f"\n--- Optimized Dolphin streaming bundle complete: "
        f"{len(bundle['graphs'])} graph(s), "
        f"{storage['unique_data_ranges']} unique shared range(s). ---"
    )


if __name__ == "__main__":
    main()

