"""Family runner helpers (thin wrapper).

Exports run_family_for_k by forwarding to `sufes.core`.
"""
from importlib import import_module

_core = import_module(".core", __package__)

run_family_for_k = getattr(_core, "run_family_for_k", None)

if run_family_for_k is None:
    raise ImportError(
        "sufes.core doesn't provide 'run_family_for_k'.\n"
        "If you recently refactored core, ensure that function exists."
    )

__all__ = ["run_family_for_k"]
