from pathlib import Path


EXAMPLE_AUDIO_ROOT = Path(__file__).resolve().parent / "Test_Examples"

_MODEL_AUDIO_FILES = {
    "dolphin": (("zh", None), ("ja", None), ("ko", None)),
    "dolphin_cn_dialect": (("zh", None), ("zh", "zh-Shanghai.wav")),
    "fireredasr": (("zh", None), ("zh", "zh_1.wav"), ("zh", "zh_2.wav")),
    "fun_asr_nano": (("zh", None), ("en", None), ("yue", None), ("ja", None)),
    "fun_asr_nano_mlt": (("zh", None), ("en", None), ("yue", None), ("ja", None), ("ko", None)),
    "paraformer": (("zh", None),),
    "qwen_asr": (("zh", None), ("en", None), ("yue", None), ("ja", None), ("ko", None)),
    "qwen_forced_aligner": (("zh", None), ("en", None), ("yue", None), ("ja", None), ("ko", None)),
    "sensevoice": (("en", "test_sample.wav"),),
    "whisper": (("zh", None), ("en", None), ("ja", None), ("ko", None)),
    "x_asr": (("zh", None), ("en", None)),
}


def example_audio_path(language, filename=None):
    if filename is None:
        filename = f"{language}.mp3"
    return str(EXAMPLE_AUDIO_ROOT / language / filename)


def model_audio_cases(model_name):
    try:
        audio_files = _MODEL_AUDIO_FILES[model_name]
    except KeyError as exc:
        names = ", ".join(sorted(_MODEL_AUDIO_FILES))
        raise ValueError(f"Unknown demo audio model {model_name!r}. Available models: {names}") from exc
    return [(example_audio_path(language, filename), language) for language, filename in audio_files]


def model_audio_paths(model_name):
    return [path for path, _language in model_audio_cases(model_name)]