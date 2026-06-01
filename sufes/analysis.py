"""Higher-level analysis helpers (wrapper around core).

Provides: analyze_range, analyze_chunk, merge_summaries, save_summary_json
All objects are forwarded from `sufes.core` to keep a small module surface.
"""
from importlib import import_module

_core = import_module(".core", __package__)

analyze_range = getattr(_core, "analyze_range", None)
analyze_chunk = getattr(_core, "analyze_chunk", None)
merge_summaries = getattr(_core, "merge_summaries", None)
save_summary_json = getattr(_core, "save_summary_json", None)

missing = [name for name in ("analyze_range", "analyze_chunk", "merge_summaries", "save_summary_json") if globals().get(name) is None]
if missing:
    raise ImportError(
        "sufes.core is missing the following analysis helpers: %s\n"
        "Use sufes.core directly or restore these names in core." % ", ".join(missing)
    )

__all__ = ["analyze_range", "analyze_chunk", "merge_summaries", "save_summary_json"]
