"""Optimize & quantize the exported Qwen3-ASR ONNX modules.

Config-only front-end: this script only defines the per-module Plans and the shared
OptimizerConfig, then delegates the whole quantize/optimize pipeline to
``Optimize_ONNX_Common.py`` (the same structure the other ASR export scripts use).

Each module in MODEL_PLANS picks a method; a Plan field left ``None`` inherits the
matching OptimizerConfig default. Plans are resolved and validated up front, so an
incompatible combination fails before any file is written.

    Method       Backend                   Result
    "Q2/Q4/Q8"   matmul_nbits_quantizer    2/4/8-bit weight-only (MatMulNBits)
    "DYNAMIC"    quantize_dynamic          INT8 dynamic (DynamicQuantizeLinear)
    "F16"        convert_float_to_float16  float16 weights & activations
    "F32"        -                         keep float32 (optimize only)
"""

from pathlib import Path
import sys

from onnx import TensorProto
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


# ============================== SHARED PIPELINE =========================

# Reuse the shared optimizer pipeline: walk up to the repo root that holds it.
_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import OptimizerConfig, Plan, run_optimizer


# ============================== USER CONFIG ==============================

MODEL_PATH            = r"/home/DakeQQ/Downloads/Qwen3-ASR-0.6B"   # Same Hugging Face checkpoint folder used for export.
ORIGINAL_FOLDER_PATH  = str(_SCRIPT_DIR / "Qwen_ASR_ONNX")       # Source *.onnx modules.
OPTIMIZED_FOLDER_PATH = str(_SCRIPT_DIR / "Qwen_ASR_Optimized")  # Destination folder.

FORCE_EXTERNAL_DATA   = False   # two-part storage (*.onnx.data); auto-forced when >2GB
UPGRADE_OPSET         = 0       # target ONNX opset (0 = keep current)

F16_OP_BLOCK_LIST = [           # op types kept out of any float16 conversion
    "DynamicQuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
]


# ========================== QWEN CONFIG LOADER ==========================

class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_asr_audio_encoder"

    def __init__(self, d_model=1280, encoder_attention_heads=20, **kwargs):
        self.d_model = d_model
        self.encoder_attention_heads = encoder_attention_heads
        super().__init__(**kwargs)


class Qwen3ASRTextConfig(PretrainedConfig):
    model_type = "qwen3_asr_text"

    def __init__(self, hidden_size=4096, num_attention_heads=32, rope_scaling=None, **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        super().__init__(**kwargs)


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"

    def __init__(self, audio_config=None, text_config=None, **kwargs):
        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config
        if isinstance(text_config, dict):
            text_config = Qwen3ASRTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3ASRTextConfig()
        self.text_config = text_config
        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        return getattr(self, "text_config", self)


class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"

    def __init__(self, thinker_config=None, support_languages=None, **kwargs):
        if isinstance(thinker_config, dict):
            thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        elif thinker_config is None:
            thinker_config = Qwen3ASRThinkerConfig()
        self.thinker_config = thinker_config
        self.support_languages = support_languages
        super().__init__(**kwargs)

    def get_text_config(self, decoder=None, encoder=None) -> PretrainedConfig:
        thinker_config = getattr(self, "thinker_config", None)
        return self if thinker_config is None else thinker_config.get_text_config(decoder=decoder, encoder=encoder)


try:
    AutoConfig.register(Qwen3ASRConfig.model_type, Qwen3ASRConfig)
except ValueError as exc:
    if "already registered" not in str(exc):
        raise


def get_attention_configs(model_path: str) -> tuple[tuple[int, int], tuple[int, int]]:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    audio_config = config.thinker_config.audio_config
    text_config = config.thinker_config.text_config
    encoder_config = int(audio_config.encoder_attention_heads), int(audio_config.d_model)
    decoder_config = int(text_config.num_attention_heads), int(text_config.hidden_size)
    return encoder_config, decoder_config


(ENCODER_NUM_HEADS, ENCODER_HIDDEN_SIZE), (DECODER_NUM_HEADS, DECODER_HIDDEN_SIZE) = get_attention_configs(MODEL_PATH)


# ============================== MODEL PLANS =============================

# Per-module plan. Comment out a line to skip that module. Heavy graphs are weight-quantized
# (embedding Gather, decoder/encoder MatMul); helper graphs stay float32, optimize-only.
# num_heads/hidden_size feed the attention-fusion optimizer from the AutoConfig audio/text configs.
MODEL_PLANS: dict[str, Plan] = {
    "Qwen3_ASR_Metadata":                 Plan(method="F32", transformer=False),
    "Qwen3_ASR_Encoder":                  Plan(method="Q4", external=True, num_heads=ENCODER_NUM_HEADS, hidden_size=ENCODER_HIDDEN_SIZE),
    "Qwen3_ASR_Decoder_Embed":            Plan(method="Q4", algo="DEFAULT", block_size=16, op_types=("Gather",), axes=(1,), external=True),
    "Qwen3_ASR_Decoder_Main":             Plan(method="Q4", external=True, num_heads=DECODER_NUM_HEADS, hidden_size=DECODER_HIDDEN_SIZE),
    "Qwen3_ASR_Rotary_Mask_Text_Prefill": Plan(method="F32", transformer=False),
    "Qwen3_ASR_Rotary_Mask_Text_Decode":  Plan(method="F32", transformer=False),
    "Qwen3_ASR_Greedy_Search":            Plan(method="F32", transformer=False),
    "Qwen3_ASR_First_Beam_Search":        Plan(method="F32", transformer=False),
    "Qwen3_ASR_Second_Beam_Search":       Plan(method="F32", transformer=False),
    "Qwen3_ASR_Apply_Penalty":            Plan(method="F32", transformer=False),
    "Qwen3_ASR_Argmax":                   Plan(method="F32", transformer=False),
    "Qwen3_ASR_Concat_Embed":             Plan(method="F32", transformer=False),
}


# ============================== PIPELINE ================================

CONFIG = OptimizerConfig(
    original_folder_path=ORIGINAL_FOLDER_PATH,
    optimized_folder_path=OPTIMIZED_FOLDER_PATH,
    model_plans=MODEL_PLANS,
    force_external_data=FORCE_EXTERNAL_DATA,
    upgrade_opset=UPGRADE_OPSET,
    optimizer_level=2,
    # Export emits ORT's fused SimplifiedLayerNormalization in the default domain, which
    # onnx.shape_inference can't type-propagate; this fallback keeps quantize_dynamic from
    # aborting should any heavy module be switched to the DYNAMIC method.
    dynamic_default_tensor_type=TensorProto.FLOAT,
    f16_op_block_list=F16_OP_BLOCK_LIST,
)


if __name__ == "__main__":
    run_optimizer(CONFIG)
