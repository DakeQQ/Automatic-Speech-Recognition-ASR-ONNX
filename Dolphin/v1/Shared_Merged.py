"""Dolphin v1 adapter for the shared/merged decoder implementation.

Dolphin v1 and CN-Dialect export the same decoder graph ABI (self-KV, fixed
cross-KV, token Embed, position/mask shells, and token-selection heads). Reuse
the single canonical implementation rather than maintaining a divergent copy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_IMPLEMENTATION = Path(__file__).resolve().parent.parent / "CN-Dialect" / "Shared_Merged.py"
_SPEC = importlib.util.spec_from_file_location("_dolphin_shared_merged_common", _IMPLEMENTATION)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_MODULE, _name)


def make_merged_build_plan(model_file_names=None, **kwargs):
    if kwargs.pop("probe_aware", False):
        return _MODULE.make_probe_aware_build_plan(model_file_names)
    return _MODULE.make_merged_build_plan(
        model_file_names,
        merge_encoder_into_prefill=False,
    )


def build_shared_merged_bundle(*args, **kwargs):
    kwargs["merge_encoder_into_prefill"] = False
    return _MODULE.build_shared_merged_bundle(*args, **kwargs)


def copy_runtime_standalones(*args, **kwargs):
    # Probe-aware v1 intentionally removes the runtime Encoder; legacy callers
    # still default to including it through the canonical function.
    return _MODULE.copy_runtime_standalones(*args, **kwargs)


__all__ = [name for name in globals() if not name.startswith("_")]
