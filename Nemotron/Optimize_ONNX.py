"""Optimize & quantize the exported Nemotron ASR ONNX modules (auto-adapts to offline & streaming)."""

from pathlib import Path
import sys


# ============================== USER CONFIG ==============================

_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, metadata_int_for_model, run_optimizer


USE_OPENVINO = False
FORCE_EXTERNAL_DATA = False
UPGRADE_OPSET = 0

F16_OP_BLOCK_LIST = [
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "Range",
    "MatMulIntegerToFloat",
]

def nemotron_encoder_heads(model_path: str) -> int:
    return metadata_int_for_model(model_path, "encoder_heads")


def nemotron_encoder_hidden_size(model_path: str) -> int:
    return metadata_int_for_model(model_path, "encoder_d_model")

# Both possible graph sets; each is optimized only if its exported ONNX folder is present, so the same
# script auto-adapts to whatever the fused Export_Nemotron_ASR.py produced (offline, streaming, or both).
_TARGETS = (
    {"prefix": "Nemotron_ASR",
     "original": _SCRIPT_DIR / "Nemotron_ASR_ONNX",
     "optimized": _SCRIPT_DIR / "Nemotron_ASR_Optimized"},
    {"prefix": "Nemotron_ASR_Streaming",
     "original": _SCRIPT_DIR / "Nemotron_ASR_Streaming_ONNX",
     "optimized": _SCRIPT_DIR / "Nemotron_ASR_Streaming_Optimized"},
)


# ============================== MODEL PLANS ==============================

def _model_plans(prefix: str) -> dict:
    return {
        f"{prefix}_Encoder":  Plan(method="DYNAMIC", external=False, num_heads=nemotron_encoder_heads, hidden_size=nemotron_encoder_hidden_size),  # mel front-end + Conformer + prompt head
        f"{prefix}_Decoder":  Plan(method="F32", external=False, transformer=False),  # folded RNN-T decoder/joint + greedy search
    }


# ============================== PIPELINE ================================

def _make_config(target: dict) -> OptimizerConfig:
    return OptimizerConfig(
        original_folder_path=str(target["original"]),
        optimized_folder_path=str(target["optimized"]),
        model_plans=_model_plans(target["prefix"]),
        dynamic_weight_type="QInt8",
        dynamic_per_channel=False,
        dynamic_reduce_range=False,
        force_external_data=FORCE_EXTERNAL_DATA,
        upgrade_opset=UPGRADE_OPSET,
        optimizer_level=2,
        optimizer_only_onnxruntime=USE_OPENVINO,
        f16_max_finite_val=32767.0,
        f16_op_block_list=F16_OP_BLOCK_LIST,
        slim_skip_fusion_patterns=["FusionGemm"],
        copy_artifacts=("tokenizer.model", "vocab.txt", "model_config.yaml"),
    )


def _is_present(target: dict) -> bool:
    return (target["original"] / "ASR_Matadata.onnx").exists()


def main() -> None:
    targets = [t for t in _TARGETS if _is_present(t)]
    if not targets:
        searched = "\n  ".join(str(t["original"]) for t in _TARGETS)
        raise FileNotFoundError(
            "No exported Nemotron ASR ONNX graphs found to optimize. Looked in:\n  " + searched
            + "\nRun Export_Nemotron_ASR.py first (CHUNK_MS=0 for offline, CHUNK_MS>0 for streaming)."
        )
    for target in targets:
        kind = "streaming" if target["prefix"].endswith("Streaming") else "offline"
        print(f"\n########## Optimizing {kind} models: {target['original'].name} "
              f"-> {target['optimized'].name} ##########")
        run_optimizer(_make_config(target))


if __name__ == "__main__":
    main()
