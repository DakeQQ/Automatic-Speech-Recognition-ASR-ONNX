"""Optimize & quantize the exported Parakeet TDT ASR ONNX modules (offline)."""

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

def parakeet_encoder_heads(model_path: str) -> int:
    return metadata_int_for_model(model_path, "encoder_heads")


def parakeet_encoder_hidden_size(model_path: str) -> int:
    return metadata_int_for_model(model_path, "encoder_d_model")


_ORIGINAL = _SCRIPT_DIR / "Parakeet_ASR_ONNX"
_OPTIMIZED = _SCRIPT_DIR / "Parakeet_ASR_Optimized"
_PREFIX = "Parakeet_ASR"


# ============================== MODEL PLANS ==============================

def _model_plans() -> dict:
    return {
        # mel front-end + Fast-Conformer + encoder projector
        f"{_PREFIX}_Encoder": Plan(method="DYNAMIC", external=False, num_heads=parakeet_encoder_heads, hidden_size=parakeet_encoder_hidden_size),
        # folded TDT decoder/joint + greedy step
        f"{_PREFIX}_Decoder": Plan(method="F32", external=False, transformer=False),
    }


# ============================== PIPELINE ================================

def _make_config() -> OptimizerConfig:
    return OptimizerConfig(
        original_folder_path=str(_ORIGINAL),
        optimized_folder_path=str(_OPTIMIZED),
        model_plans=_model_plans(),
        dynamic_weight_type="QInt8",
        dynamic_per_channel=True,
        dynamic_reduce_range=False,
        force_external_data=FORCE_EXTERNAL_DATA,
        upgrade_opset=UPGRADE_OPSET,
        optimizer_level=2,
        optimizer_only_onnxruntime=USE_OPENVINO,
        f16_max_finite_val=32767.0,
        f16_op_block_list=F16_OP_BLOCK_LIST,
        slim_skip_fusion_patterns=["FusionGemm"],
        copy_artifacts=("tokenizer.json", "tokenizer_config.json"),
    )


def main() -> None:
    if not (_ORIGINAL / "ASR_Matadata.onnx").exists():
        raise FileNotFoundError(
            f"No exported Parakeet ASR ONNX graphs found to optimize. Looked in:\n  {_ORIGINAL}"
            "\nRun Export_Parakeet_ASR.py first."
        )
    print(f"\n########## Optimizing offline models: {_ORIGINAL.name} -> {_OPTIMIZED.name} ##########")
    run_optimizer(_make_config())


if __name__ == "__main__":
    main()
