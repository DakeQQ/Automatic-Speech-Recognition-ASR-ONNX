"""Helpers for constructing ONNX Runtime values from model I/O metadata."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np


_NUMPY_DTYPES = {
    "tensor(bool)": np.dtype(np.bool_),
    "tensor(double)": np.dtype(np.float64),
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
}


def numpy_dtype(value_or_type: Any) -> np.dtype:
    """Return the NumPy dtype declared by an ORT ``NodeArg`` or type string."""
    type_name = value_or_type if isinstance(value_or_type, str) else value_or_type.type
    return _NUMPY_DTYPES[type_name]


def is_dynamic_dim(dim: Any) -> bool:
    return not isinstance(dim, (int, np.integer))


def resolve_shape(
    value_meta: Any,
    *,
    symbols: Mapping[str, int] | None = None,
    axes: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    """Resolve a ``NodeArg`` shape from axis and symbol overrides."""
    symbols = symbols or {}
    axes = axes or {}
    result: list[int] = []
    for axis, dim in enumerate(value_meta.shape):
        axis_value = axes.get(axis)
        if isinstance(dim, (int, np.integer)):
            value = int(dim)
        elif axis_value is not None:
            value = int(axis_value)
        elif isinstance(dim, str) and dim in symbols:
            value = int(symbols[dim])
        else:
            value = int(dim)
        result.append(value)
    return tuple(result)


def array_for(
    value_meta: Any,
    value: Any,
    *,
    symbols: Mapping[str, int] | None = None,
    axes: Mapping[int, int] | None = None,
) -> np.ndarray:
    """Materialize a contiguous array with the model-declared dtype and shape."""
    array = np.asarray(value, dtype=numpy_dtype(value_meta))
    explicit_axes = axes or {}
    runtime_axes = {
        axis: int(array.shape[axis])
        for axis, dim in enumerate(value_meta.shape)
        if is_dynamic_dim(dim) and axis < array.ndim
    }
    missing_axes = [
        axis
        for axis, dim in enumerate(value_meta.shape)
        if is_dynamic_dim(dim)
        and axis >= array.ndim
        and axis not in explicit_axes
    ]
    if missing_axes:
        raise ValueError(
            f"Value for {value_meta.name!r} has rank {array.ndim}; provide "
            f"axes for dynamic dimensions {missing_axes!r}."
        )
    runtime_axes.update(explicit_axes)
    shape = resolve_shape(value_meta, symbols=symbols, axes=runtime_axes)
    return np.ascontiguousarray(
        array.reshape(shape)
    )


def filled_for(
    value_meta: Any,
    fill_value: Any = 0,
    *,
    symbols: Mapping[str, int] | None = None,
    axes: Mapping[int, int] | None = None,
) -> np.ndarray:
    return np.full(
        resolve_shape(value_meta, symbols=symbols, axes=axes),
        fill_value,
        dtype=numpy_dtype(value_meta),
    )


def scalar_for(value_meta: Any, value: Any) -> np.ndarray:
    """Construct either a scalar ``[]`` or one-element ``[1]`` model input."""
    shape = tuple(value_meta.shape)
    if shape == ():
        return np.asarray(value, dtype=numpy_dtype(value_meta)).reshape(())
    return np.asarray([value], dtype=numpy_dtype(value_meta))


def metadata_by_name(values: Sequence[Any]) -> dict[str, Any]:
    return {value.name: value for value in values}


def metadata_int(
    metadata: Mapping[str, str],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    """Read one required integer from an ONNX metadata map."""
    return int(metadata[key])


def metadata_int_list(metadata: Mapping[str, str], key: str) -> list[int]:
    """Read a required comma-separated integer list from ONNX metadata."""
    return [int(item) for item in metadata[key].split(",") if item]


def metadata_json_object(
    metadata: Mapping[str, str],
    key: str,
) -> dict[str, Any]:
    """Parse one required JSON object from ONNX metadata."""
    return json.loads(metadata[key])


def load_special_token_ids(metadata: Mapping[str, str]) -> dict[str, Any]:
    """Load the model-owned special-token contract from ``special_token_ids``."""
    return metadata_json_object(metadata, "special_token_ids")


def load_supported_languages(metadata: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Load the common ASR ``supported_languages`` catalog."""
    raw_catalog = metadata_json_object(metadata, "supported_languages")
    catalog: dict[str, dict[str, Any]] = {}
    for code, raw_entry in raw_catalog.items():
        canonical = code.strip()
        entry = dict(raw_entry)
        entry["name"] = entry.get("name", canonical).strip()
        entry["aliases"] = [alias.strip() for alias in entry.get("aliases", [])]
        entry["prompt_token_ids"] = entry.get("prompt_token_ids", [])
        catalog[canonical] = entry
    return catalog


def resolve_supported_language(
    catalog: Mapping[str, Mapping[str, Any]],
    language: str,
) -> tuple[str, Mapping[str, Any]]:
    """Resolve a canonical code or alias, prioritizing canonical codes."""
    normalized = language.strip().casefold()
    for code, entry in catalog.items():
        if code.casefold() == normalized:
            return code, entry
    matches = [
        (code, entry)
        for code, entry in catalog.items()
        if any(str(alias).casefold() == normalized for alias in entry.get("aliases", ()))
    ]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Unsupported language {language!r}; choose one of {sorted(catalog)}."
    )