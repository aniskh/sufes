"""sufes package - public, minimal package surface.

This module intentionally avoids importing submodules at package import
time to prevent circular import issues. Import submodules explicitly
where needed, for example:

    from sufes import core

Expose a lightweight public surface via ``__all__`` and a package
``__version__``.
"""

__all__ = [
    "core",
    "algorithms",
    "analysis",
    "family",
    "plotting",
]

__version__ = "0.2.0"
