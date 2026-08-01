"""Optimize Whisper probe-aware Encoder/Main donors and rebuild nine graphs."""

from __future__ import annotations

import gc
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
    copy_artifact,
    DEFAULT_F16_OP_BLOCK_LIST,
    OptimizerConfig,
    Plan,
    remove_redundant_casts,
    normalize_float16_output_bridge,
    resolve_plan,
    rewrite_tied_embed_from_matmul_nbits,
    run_optimizer,
    share_external_initializers_if_identical,
)


# ============================== USER CONFIG ==============================
# Edit this section only.
# Q8 is the direct-script default. Encoder/Main are optimized once and
# transplanted; preserve Main's optimizer ABI and its FP32 logits bridge.

ORIGINAL_FOLDER_PATH  = str(SCRIPT_DIR / "Whisper_ONNX")
OPTIMIZED_FOLDER_PATH = str(SCRIPT_DIR / "Whisper_Optimized")

OPTIMIZER_ONLY_ONNXRUNTIME = False
FORCE_EXTERNAL_DATA        = False
UPGRADE_OPSET              = 0
OPTIMIZER_LEVEL            = 2

WEIGHT_ONLY_ALGORITHM = "AFFINE_REFINE_V2"
DYNAMIC_WEIGHT_TYPE  = "QInt8"
DYNAMIC_PER_CHANNEL  = True
DYNAMIC_REDUCE_RANGE = False

F16_KEEP_IO_TYPES      = False
F16_FORCE_INITIALIZERS = False
F16_MAX_FINITE_VALUE = 32767.0
F16_OP_BLOCK_LIST = DEFAULT_F16_OP_BLOCK_LIST

MAIN_STEM = Path(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["main"]).stem


def _main_nodes_to_exclude(model_path: str) -> list[str] | None:
    """Keep Whisper's tied vocabulary projection floating under dynamic INT8."""
    if MODEL_PLANS[MAIN_STEM].method.upper() != "DYNAMIC":
        return None
    model = onnx.load(model_path, load_external_data=False)
    matches = [
        node.name
        for node in model.graph.node
        if node.op_type in ("MatMul", "Gemm") and "/proj_out" in node.name
    ]
    if len(matches) != 1 or not matches[0]:
        raise RuntimeError(
            f"Expected one named Whisper vocabulary projection; found {matches}."
        )
    return matches


MODEL_PLANS = {
    "Whisper_Encoder":                   Plan(method="Q8", transformer=True, external=True),
    "Whisper_Decoder":                   Plan(method="Q8", optimize=True, transformer=True, external=True, nodes_to_exclude=_main_nodes_to_exclude),
    "Whisper_No_Speech_Detection":       Plan(method="F32", transformer=False),
    "Whisper_ProbePrefillGreedy":        Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_ProbePrefillPenaltyGreedy": Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_ProbePrefillSampling":      Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_PrefillGreedy":             Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_PrefillPenaltyGreedy":      Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_PrefillSampling":           Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_DecodeGreedy":              Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_DecodePenaltyGreedy":       Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_DecodeSampling":            Plan(method="Q8", process=False, optimize=True, transformer=True),
    "Whisper_SharedInitializers":        Plan(method="Q8", process=False, optimize=True, transformer=True),
    "ASR_Metadata":                      Plan(method="F32", process=False, optimize=True, transformer=True),
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
    f16_force_initializers=F16_FORCE_INITIALIZERS,
    f16_max_finite_val=F16_MAX_FINITE_VALUE,
    f16_op_block_list=F16_OP_BLOCK_LIST,
    copy_artifacts=(Shared_Merged.DEFAULT_MODEL_FILE_NAMES["metadata"],),
)

# ============================ END USER CONFIG ============================


def _persist(model: onnx.ModelProto, path: Path) -> None:
    simplified = remove_redundant_casts(model)
    if simplified:
        print(f"  Simplified {simplified} provably redundant Cast node/path(s).")
    Shared_Merged.save_model(model, path)
    print(f"  {path.name} ({path.stat().st_size} bytes)")


def _insert_probe_audio_f16_bridge(model: onnx.ModelProto) -> int:
    audio_input = next(
        (value for value in model.graph.input if value.name == "audio"),
        None,
    )
    if audio_input is None:
        return 0
    if audio_input.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise RuntimeError("Whisper probe audio input no longer has the expected FLOAT ABI.")
    consumers = [
        (index, node)
        for index, node in enumerate(model.graph.node)
        if node.name.startswith("encoder_") and "audio" in node.input
    ]
    if not consumers:
        raise RuntimeError("Whisper probe graph has no Encoder consumer of public audio.")
    bridge_output = "whisper_encoder_f16_audio"
    occupied = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    if bridge_output in occupied:
        raise RuntimeError(f"Whisper probe audio bridge collision: {bridge_output!r}.")
    for _, node in consumers:
        node.input[:] = [bridge_output if name == "audio" else name for name in node.input]
    bridge = onnx.helper.make_node(
        "Cast",
        ["audio"],
        [bridge_output],
        name="Whisper_Encoder_Audio_To_Float16",
        to=onnx.TensorProto.FLOAT16,
    )
    model.graph.node.insert(min(index for index, _ in consumers), bridge)
    model.graph.value_info.append(
        onnx.helper.make_tensor_value_info(
            bridge_output,
            onnx.TensorProto.FLOAT16,
            None,
        )
    )
    return 1


def _finalize_full_f16_shell(model: onnx.ModelProto) -> int:
    removable = {"prefill_/Cast", "decode_/Cast"}
    aliases = {
        node.output[0]: node.input[0]
        for node in model.graph.node
        if node.name in removable
        and node.op_type == "Cast"
        and len(node.input) == len(node.output) == 1
    }
    for node in model.graph.node:
        node.input[:] = [aliases.get(name, name) for name in node.input]
    rewritten = [node for node in model.graph.node if node.name not in removable]
    del model.graph.node[:]
    model.graph.node.extend(rewritten)
    retained_info = [value for value in model.graph.value_info if value.name not in aliases]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_info)
    probe_audio_bridge = _insert_probe_audio_f16_bridge(model)
    normalize_float16_output_bridge(
        model,
        producer_op_type="Add",
        producer_name_contains="/Add_17",
        private_output_name="whisper_main_f16_logits",
        bridge_node_name="Whisper_Main_Logits_To_Float32",
        metadata_key="whisper_f16_logits_cast_normalized",
        graph_label="optimized Whisper merged shell",
    )
    return len(aliases) + probe_audio_bridge + 1


def build_quantized_merged_bundle(
    source_folder: Path,
    output_folder: Path,
    model_file_names: dict[str, str],
) -> None:
    available = Shared_Merged.make_merged_build_plan(model_file_names)
    main_path = output_folder / model_file_names["main"]
    encoder_path = output_folder / model_file_names["encoder"]
    optimized_main = Shared_Merged.load_model(main_path)
    optimized_encoder = Shared_Merged.load_model(encoder_path)
    Shared_Merged.namespace_internal_tensors(optimized_main, marker="_inlfunc_", namespace="main_")
    Shared_Merged.restore_precision_free_graph_outputs(optimized_main)
    Shared_Merged.restore_precision_free_graph_outputs(optimized_encoder)
    main_plan = resolve_plan(MODEL_PLANS[MAIN_STEM], CONFIG, model_name=MAIN_STEM)
    encoder_plan = resolve_plan(
        MODEL_PLANS["Whisper_Encoder"],
        CONFIG,
        model_name="Whisper_Encoder",
    )
    if main_plan.method in {"Q2", "Q4", "Q8"}:
        assert_no_large_unquantized_linear_weights(
            optimized_main,
            graph_label="optimized Whisper Main",
        )
    if encoder_plan.method in {"Q2", "Q4", "Q8"}:
        assert_no_large_unquantized_linear_weights(
            optimized_encoder,
            graph_label="optimized Whisper Encoder",
        )

    shared_name = model_file_names["shared_initializers"]
    shared_data_name = model_file_names["shared_initializers_data"]

    additional_shared = collect_target_only_shared_shell_initializers(
        source_folder,
        [source_folder / file_name for file_name, _, _ in available],
        optimized_main,
        tuple(
            prefix
            for prefix in Shared_Merged.SHELL_PREFIXES
            if prefix != "encoder_"
        ),
    )
    if additional_shared:
        print(
            "  Preserving target-only shared shell initializers: "
            + ", ".join(sorted(additional_shared))
        )
    packed_tied_embed = main_plan.method in {"Q2", "Q4", "Q8"}
    tied_report = None
    external_by_name = None
    print(f"\n{'=' * 60}\nTransplanting optimized Encoder/Main into nine Whisper shells\n{'=' * 60}")
    for file_name, _, _ in available:
        target = onnx.load(str(source_folder / file_name), load_external_data=False)
        merged = Shared_Merged.transplant_quantized_main(target, optimized_main)
        del target
        merged = Shared_Merged.transplant_optimized_encoder(merged, optimized_encoder)
        if packed_tied_embed:
            report = rewrite_tied_embed_from_matmul_nbits(
                merged,
                optimized_main,
                alias_prefix="whisper_tied_embed_",
            )
            if tied_report is None:
                tied_report = report
                removed_table = report["removed_table"]
                if removed_table is not None:
                    additional_shared.pop(removed_table, None)
                print(
                    f"  Rebuilt tied Embed from Main's Q{report['bits']} tuple; "
                    f"reused {report['shared_data_bytes']} packed bytes."
                )
            elif report != tied_report:
                raise RuntimeError("Whisper tied Embed rewrite changed across shells.")
        if (
            MODEL_PLANS["Whisper_Encoder"].method.upper() == "F16"
            and MODEL_PLANS[MAIN_STEM].method.upper() == "F16"
        ):
            rewrites = _finalize_full_f16_shell(merged)
            print(f"  Finalized full-F16 shell with {rewrites} targeted rewrite(s).")
        if external_by_name is None:
            external_by_name = Shared_Merged.extract_and_write_shared(
                [merged], output_folder / shared_name, primary_model=merged,
                additional_shared=additional_shared,
            )
        else:
            Shared_Merged.redirect_shared_initializers_to_external(merged, external_by_name)
        output_path = output_folder / file_name
        _persist(merged, output_path)
        del merged
        gc.collect()

    shared_data_path = output_folder / shared_data_name
    print(f"  {shared_data_name} ({shared_data_path.stat().st_size} bytes)")
    for path in (main_path, encoder_path):
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".data").unlink(missing_ok=True)


def main() -> None:
    source_folder = Path(ORIGINAL_FOLDER_PATH)
    output_folder = Path(OPTIMIZED_FOLDER_PATH)
    model_file_names = dict(Shared_Merged.DEFAULT_MODEL_FILE_NAMES)

    # Standalone graphs are optimized independently. The decoder donor is handled below so its
    # quantized Main can be transplanted rather than quantizing seven copies of the same weights.
    run_optimizer(
        CONFIG,
        model_names=(
            "Whisper_Encoder",
            "Whisper_Decoder",
            "Whisper_No_Speech_Detection",
        ),
        copy_configured_artifacts=False,
        print_completion=False,
        reset_output_folder=True,
    )
    copy_artifact(
        source_folder / model_file_names["metadata"],
        output_folder / model_file_names["metadata"],
    )
    build_quantized_merged_bundle(
        source_folder,
        output_folder,
        model_file_names,
    )
    shared_model_path = output_folder / model_file_names["shared_initializers"]
    for standalone_path in sorted(output_folder.glob("*.onnx")):
        if standalone_path != shared_model_path:
            share_external_initializers_if_identical(
                standalone_path,
                shared_model_path,
                require_all_external=False,
            )

    # Keep the optimized folder inference-standalone without destructively moving source assets.
    copy_artifact(source_folder / "tokenizer", output_folder / "tokenizer")
    storage = consolidate_optimized_model_weights(
        output_folder,
        model_file_names["shared_initializers"],
    )
    print(f"  Consolidated {storage['unique_data_ranges']} unique shared range(s).")
    print("\n--- All merged Whisper models processed successfully! ---")


if __name__ == "__main__":
    main()