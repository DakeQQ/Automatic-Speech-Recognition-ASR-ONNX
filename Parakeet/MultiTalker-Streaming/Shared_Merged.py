#!/usr/bin/env python
"""Package MultiTalker Streaming Parakeet runtime graphs with shared weights.

The cache-aware encoder, metadata carrier, and one-step RNN-T decoder remain
standalone runtime graphs. The host owns the greedy symbol loop, so no static
encoder/decoder merge is semantically valid. This script only consolidates
identical optimized initializers into the established shared data artifact.
"""

from pathlib import Path
import sys


_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
    if (_candidate / "Optimize_ONNX_Common.py").exists():
        sys.path.insert(0, str(_candidate))
        break
else:
    raise RuntimeError("Could not locate Optimize_ONNX_Common.py")

from Optimize_ONNX_Common import consolidate_optimized_model_weights


OPTIMIZED_FOLDER = _SCRIPT_DIR / "MultiTalker_Streaming_Parakeet_ASR_Optimized"
SHARED_INITIALIZERS_NAME = "MultiTalker_Streaming_Parakeet_ASR_SharedInitializers.onnx"
RUNTIME_GRAPHS = (
    "ASR_Metadata.onnx",
    "MultiTalker_Streaming_Parakeet_ASR_Encoder.onnx",
    "MultiTalker_Streaming_Parakeet_ASR_Decoder.onnx",
)


def main() -> None:
    storage = consolidate_optimized_model_weights(
        OPTIMIZED_FOLDER,
        SHARED_INITIALIZERS_NAME,
    )
    print(
        f"Consolidated {storage['unique_data_ranges']} unique shared range(s) for "
        f"{len(RUNTIME_GRAPHS)} standalone runtime graph(s)."
    )


if __name__ == "__main__":
    main()