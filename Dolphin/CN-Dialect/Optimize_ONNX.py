"""Optimize Dolphin CN Encoder/Main donors and rebuild direct prefill graphs.

Encoder and Main are optimized independently, transplanted into all six raw
strategy shells, and extracted into one deployment initializer blob.
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
    normalize_float16_output_bridge,
    OptimizerConfig,
    Plan,
    producer_ancestry_node_names,
    remove_model_files,
    remove_redundant_casts,
    resolve_plan,
    run_optimizer,
    share_external_initializers_if_identical,
)


MAIN_STEM = Path(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["main"]).stem

# ============================== USER CONFIG ==============================
# Edit this section only.

ORIGINAL_FOLDER_PATH  = str(SCRIPT_DIR / "Dolphin_CN_Dialect_ONNX")
OPTIMIZED_FOLDER_PATH = str(SCRIPT_DIR / "Dolphin_CN_Dialect_Optimized")

# F32: no precision change; F16: mixed precision; DYNAMIC: portable INT8;
# Q4/Q8: AFFINE_REFINE_V2 block weight-only. Q2 requires another algorithm.
DEFAULT_METHOD = "Q8"

WEIGHT_ONLY_ALGORITHM      = "AFFINE_REFINE_V2"
OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 1

DYNAMIC_WEIGHT_TYPE  = "QUInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_OP_BLOCK_LIST = DEFAULT_F16_OP_BLOCK_LIST

COPY_ARTIFACTS = (
    "vocab_Dolphin_CN_Dialect.txt",
)
PROBE_AWARE = False


def exclude_merged_shell_nodes(model_path: str) -> list[str]:
    """Keep Embed/position/head shells unchanged; convert only unprefixed Main."""
    model = onnx.load(model_path, load_external_data=False)
    excluded = [
        node.name
        for node in model.graph.node
        if any(output.startswith(Shared_Merged.SHELL_PREFIXES) for output in node.output)
    ]
    del model
    return excluded


def exclude_encoder_frontend_nodes(model_path: str) -> list[str]:
    """Keep the raw-PCM fbank through the subsampling projection in float32.

    F16 squaring in the Kaldi power-spectrum path overflows for int16-range PCM
    and turns every encoder cache into NaN.  Walk the exact producer ancestry of
    the projection output instead of relying on fragile node-name prefixes.
    """
    return producer_ancestry_node_names(
        model_path,
        "/embed/Add_output_0",
        graph_label="Dolphin encoder F32 frontend",
    )


# Encoder/Main are independent data-light optimizer donors.
MODEL_PLANS = {
    "Dolphin_Encoder": Plan(
        method=DEFAULT_METHOD,
        num_heads=0,
        hidden_size=0,
        external=FORCE_EXTERNAL_DATA,
        nodes_to_exclude=(
            exclude_encoder_frontend_nodes
            if DEFAULT_METHOD.upper() == "F16"
            else None
        ),
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "Dolphin_Decoder": Plan(
        method=DEFAULT_METHOD,
        optimize=True,
        transformer=True,
        num_heads=0,
        hidden_size=0,
        external=FORCE_EXTERNAL_DATA,
        f16_force_initializers=F16_FORCE_INITIALIZERS,
    ),
    "Dolphin_PrefillGreedy":        Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_PrefillPenaltyGreedy": Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_PrefillSampling":      Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_DecodeGreedy":         Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_DecodePenaltyGreedy":  Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_DecodeSampling":       Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "Dolphin_SharedInitializers":   Plan(method=DEFAULT_METHOD, process=False, optimize=True, transformer=True),
    "ASR_Metadata":                 Plan(method="F32", process=False, optimize=False, transformer=False),
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


def persist(model: onnx.ModelProto, path: Path) -> None:
    Shared_Merged.minimize_public_outputs(model)
    simplified = remove_redundant_casts(model)
    if simplified:
        print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
    Shared_Merged.prune_unreachable_nodes(model)
    Shared_Merged.save_model(model, path)
    print(f"  {path.name} ({path.stat().st_size} bytes)")


def restore_standalone_main_boundaries(
    model: onnx.ModelProto,
    *,
    float16: bool,
) -> None:
    """Restore Main aliases and, for F16 donors, its FP32 ``logits`` bridge."""
    Shared_Merged.restore_precision_free_graph_outputs(model)
    if not float16:
        return
    normalize_float16_output_bridge(
        model,
        producer_op_type="Gemm",
        producer_name_contains="/output_layer/Gemm",
        private_output_name="dolphin_main_f16_logits",
        bridge_node_name="Dolphin_Main_Logits_To_Float32",
        metadata_key="dolphin_f16_logits_cast_normalized",
        graph_label="optimized Dolphin standalone Main",
    )


def build_quantized_merged_bundle() -> dict[str, object]:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    output_folder.mkdir(parents=True, exist_ok=True)
    main_plan = resolve_plan(MODEL_PLANS[MAIN_STEM], CONFIG)
    encoder_plan = resolve_plan(MODEL_PLANS["Dolphin_Encoder"], CONFIG)
    main_is_float16 = main_plan.fp16 or main_plan.method == "F16"
    model_file_names = code_model_file_names()
    available = (
        Shared_Merged.make_probe_aware_build_plan(model_file_names)
        if PROBE_AWARE
        else Shared_Merged.make_merged_build_plan(
            model_file_names,
            merge_encoder_into_prefill=True,
        )
    )
    main_path = output_folder / model_file_names["main"]
    encoder_path = output_folder / model_file_names["encoder"]
    optimized_main = Shared_Merged.load_model(main_path)
    optimized_encoder = Shared_Merged.load_model(encoder_path)
    Shared_Merged.namespace_encoder_initializers(optimized_encoder)
    namespaced = Shared_Merged.namespace_internal_tensors(
        optimized_main,
        marker="_inlfunc_",
        namespace="main_",
    )
    if namespaced:
        print(f"  Namespaced {namespaced} Main function-inlining tensor(s).")
    restore_standalone_main_boundaries(
        optimized_main,
        float16=main_is_float16,
    )
    Shared_Merged.restore_precision_free_graph_outputs(optimized_encoder)
    if main_plan.method in {"Q2", "Q4", "Q8"}:
        assert_no_large_unquantized_linear_weights(
            optimized_main,
            graph_label="optimized Dolphin Main",
        )
    if encoder_plan.method in {"Q2", "Q4", "Q8"}:
        assert_no_large_unquantized_linear_weights(
            optimized_encoder,
            graph_label="optimized Dolphin Encoder",
        )

    shared_name = model_file_names["shared_initializers"]
    shared_data_name = model_file_names["shared_initializers_data"]
    remove_model_files(output_folder / shared_name)
    print("\n" + "=" * 60)
    print(
        f"Transplanting {encoder_plan.method} Dolphin Encoder and "
        f"{main_plan.method} Main into every strategy shell"
    )
    print("=" * 60)

    target_shell_prefixes = tuple(
        prefix for prefix in Shared_Merged.SHELL_PREFIXES if prefix != "encoder_"
    )
    additional_shared = collect_target_only_shared_shell_initializers(
        source_folder,
        [source_folder / file_name for file_name, _, _ in available],
        optimized_main,
        target_shell_prefixes,
    )
    if additional_shared:
        print(
            "  Preserving target-only shared shell initializers: "
            + ", ".join(sorted(additional_shared))
        )
    external_by_name = None
    generated: dict[str, Path] = {}
    primary_name = model_file_names[
        "probe_prefill_greedy" if PROBE_AWARE else "prefill_greedy"
    ]
    for file_name, _, _ in available:
        source_path = source_folder / file_name
        target = onnx.load(str(source_path), load_external_data=False)
        merged = Shared_Merged.transplant_quantized_main(target, optimized_main)
        del target
        if main_is_float16:
            Shared_Merged.restore_float16_merged_boundary_names(merged)
        merged = Shared_Merged.transplant_optimized_encoder(
            merged,
            optimized_encoder,
        )
        if external_by_name is None:
            if file_name != primary_name:
                raise RuntimeError("Dolphin shared extraction did not start from PrefillGreedy.")
            external_by_name = Shared_Merged.extract_and_write_shared(
                [merged],
                output_folder / shared_name,
                primary_model=merged,
                additional_shared=additional_shared,
            )
        else:
            Shared_Merged.redirect_shared_initializers_to_external(
                merged, external_by_name
            )
        output_path = output_folder / file_name
        persist(merged, output_path)
        generated[file_name] = output_path
        del merged
        gc.collect()

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
        "donors": (main_path, encoder_path),
    }


def main() -> None:
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    run_optimizer(CONFIG, reset_output_folder=True)
    bundle = build_quantized_merged_bundle()
    model_file_names = code_model_file_names()
    for standalone_path in sorted(output_folder.glob("*.onnx")):
        if standalone_path != bundle["shared_model"]:
            share_external_initializers_if_identical(
                standalone_path,
                bundle["shared_model"],
                require_all_external=False,
            )
    for donor_path in bundle["donors"]:
        locations = Shared_Merged._external_locations(donor_path)
        donor_path.unlink(missing_ok=True)
        donor_path.with_name(donor_path.name + ".data").unlink(missing_ok=True)
        for location in locations:
            if location != model_file_names["shared_initializers_data"]:
                (output_folder / location).unlink(missing_ok=True)
    storage = consolidate_optimized_model_weights(
        output_folder,
        model_file_names["shared_initializers"],
    )
    print(
        f"\n--- Optimized Dolphin merged bundle complete: "
        f"{len(bundle['graphs'])} strategy graph(s), "
        f"{storage['unique_data_ranges']} unique shared range(s). ---"
    )


if __name__ == "__main__":
    main()
