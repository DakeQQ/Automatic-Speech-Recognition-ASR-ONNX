#!/usr/bin/env python
"""Create a two-speaker overlap fixture for MultiTalker Streaming Parakeet."""

import argparse
import math
from pathlib import Path

import numpy as np
import onnxruntime
from pydub import AudioSegment


_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_TEST_AUDIO_DIR = _REPO_ROOT / "Test_Examples" / "en"
_DEFAULT_OUTPUT = _TEST_AUDIO_DIR / "test_sample_multitalker_overlap.wav"
_DEFAULT_MODEL_FOLDERS = (
    _SCRIPT_DIR / "MultiTalker_Streaming_Parakeet_ASR_Optimized",
    _SCRIPT_DIR / "MultiTalker_Streaming_Parakeet_ASR_ONNX",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix the two English test clips with a speaker overlap."
    )
    parser.add_argument(
        "--first-audio",
        type=Path,
        default=_TEST_AUDIO_DIR / "test_sample.wav",
        help="Audio assigned to speaker_0.",
    )
    parser.add_argument(
        "--second-audio",
        type=Path,
        default=_TEST_AUDIO_DIR / "en.mp3",
        help="Audio assigned to speaker_1.",
    )
    parser.add_argument(
        "--output-audio",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Destination WAV file.",
    )
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=None,
        help="Folder with ASR_Metadata.onnx; defaults to the exported model folders.",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=2.0,
        help="Duration for which speaker_0 and speaker_1 overlap (default: 2.0).",
    )
    return parser.parse_args()


def _resolve_metadata_path(model_folder: Path | None) -> Path:
    candidates = []
    if model_folder is not None:
        candidates.append(model_folder.expanduser().resolve())
    candidates.extend(folder.resolve() for folder in _DEFAULT_MODEL_FOLDERS)
    for folder in candidates:
        metadata_path = folder / "ASR_Metadata.onnx"
        if metadata_path.exists():
            return metadata_path
    searched = ", ".join(str(folder) for folder in candidates)
    raise FileNotFoundError(f"Could not find ASR_Metadata.onnx in: {searched}")


def _load_geometry(metadata_path: Path) -> tuple[int, int, int, int]:
    session = onnxruntime.InferenceSession(
        str(metadata_path),
        providers=["CPUExecutionProvider"],
    )
    metadata = session.get_modelmeta().custom_metadata_map or {}
    return (
        int(metadata["sample_rate"]),
        int(metadata["stream_stride_samples"]),
        int(metadata["stream_valid_output_frames"]),
        int(metadata["num_speakers"]),
    )


def _load_mono_samples(path: Path, sample_rate: int) -> np.ndarray:
    segment = (
        AudioSegment.from_file(path)
        .set_channels(1)
        .set_frame_rate(sample_rate)
        .set_sample_width(2)
    )
    return np.asarray(segment.get_array_of_samples(), dtype=np.float32) / np.float32(32768)


def _speaker_activity(
    total_samples: int,
    first_end: int,
    second_start: int,
    stride_samples: int,
    valid_output_frames: int,
    num_speakers: int,
) -> np.ndarray:
    num_chunks = math.ceil(total_samples / stride_samples)
    activity = np.zeros(
        (num_chunks, valid_output_frames, num_speakers),
        dtype=np.float32,
    )
    chunk_offsets = np.arange(num_chunks, dtype=np.float64)[:, None] * stride_samples
    frame_offsets = (
        np.arange(valid_output_frames, dtype=np.float64) + 0.5
    ) * (stride_samples / valid_output_frames)
    frame_centers = chunk_offsets + frame_offsets
    activity[:, :, 0] = (frame_centers < first_end).astype(np.float32)
    activity[:, :, 1] = (
        (frame_centers >= second_start) & (frame_centers < total_samples)
    ).astype(np.float32)
    return activity


def _write_wav(samples: np.ndarray, path: Path, sample_rate: int) -> None:
    pcm = np.rint(
        np.clip(samples, -1.0, 1.0 - (1.0 / 32768.0)) * np.float32(32768)
    ).astype(np.int16)
    audio = AudioSegment(
        pcm.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(path, format="wav")


def main() -> None:
    args = _parse_args()
    if args.overlap_seconds <= 0:
        raise ValueError("--overlap-seconds must be greater than zero.")

    metadata_path = _resolve_metadata_path(args.model_folder)
    sample_rate, stride_samples, valid_output_frames, num_speakers = _load_geometry(
        metadata_path
    )
    if num_speakers < 2:
        raise ValueError(f"The model supports only {num_speakers} speaker(s).")

    first_samples = _load_mono_samples(args.first_audio, sample_rate)
    second_samples = _load_mono_samples(args.second_audio, sample_rate)
    overlap_samples = round(args.overlap_seconds * sample_rate)
    if overlap_samples >= min(len(first_samples), len(second_samples)):
        raise ValueError("The overlap must be shorter than both input clips.")

    second_start = len(first_samples) - overlap_samples
    total_samples = second_start + len(second_samples)
    mixed_samples = np.zeros(total_samples, dtype=np.float32)
    mix_gain = np.float32(10 ** (-6.0 / 20.0))
    mixed_samples[: len(first_samples)] += first_samples * mix_gain
    mixed_samples[second_start:] += second_samples * mix_gain

    output_audio = args.output_audio.expanduser().resolve()
    output_activity = output_audio.with_name(
        output_audio.stem + "_diarization.npy"
    )
    _write_wav(mixed_samples, output_audio, sample_rate)
    activity = _speaker_activity(
        total_samples,
        len(first_samples),
        second_start,
        stride_samples,
        valid_output_frames,
        num_speakers,
    )
    np.save(output_activity, activity)

    print(f"Wrote audio: {output_audio}")
    print(
        f"  duration={total_samples / sample_rate:.3f}s sample_rate={sample_rate} "
        f"overlap={overlap_samples / sample_rate:.3f}s"
    )
    print(
        f"  speaker_0=[0.000, {len(first_samples) / sample_rate:.3f}) "
        f"speaker_1=[{second_start / sample_rate:.3f}, "
        f"{total_samples / sample_rate:.3f})"
    )
    print(f"Wrote activity: {output_activity}  shape={activity.shape}")


if __name__ == "__main__":
    main()