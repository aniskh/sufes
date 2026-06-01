"""Algorithm primitives.

This module is the *source of truth* for the low-level recurrence and
cycle-detection helpers used across features.

We intentionally keep it independent from `sufes.core` so that:
- features can import `next_term_ji` without pulling the full CLI machinery,
- refactors of `core.py` don't break algorithm imports.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def next_term_ji(
    n: int,
    k: int,
    j: int,
    i: int,
    alternated: bool = False,
    alt_m: int = 1,
) -> int:
    """Generalized sufes step with parameters (k, j, i).

    - If n % k == 0: divide by k.
    - Else r = n % k and:
        * normal:        (k+i)*n + (j*k-i)*r
        * alternated:    (k+i*f)*n + (j*k-i*f)*r with f = (-alt_m)**n (parity fallback)

    Note: i is expected in 1..k-1.
    """

    if n % k == 0:
        return n // k
    r = n % k
    if not alternated:
        return (k + i) * n + (j * k - i) * r
    try:
        factor = (-alt_m) ** n
    except OverflowError:
        factor = 1 if (n % 2 == 0) else -1
    return (k + i * factor) * n + (j * k - i * factor) * r


def find_cycle(
    n: int,
    base: int = 3,
    k: Optional[int] = None,
    j_param: int = 0,
    i_param: int = 1,
    max_iters: int = 200_000,
    alternated: bool = False,
    alt_m: int = 1,
) -> Optional[Dict[str, object]]:
    """Detect a cycle using a seen-map; returns preperiod/cycle/sequence or None."""

    seen: Dict[int, int] = {}
    seq: List[int] = []
    t = n
    div = k if k is not None else base
    for it in range(max_iters):
        if t in seen:
            start = seen[t]
            cycle = seq[start:]
            return {"preperiod": start, "cycle": cycle, "sequence": seq}
        seen[t] = it
        seq.append(t)
        t = next_term_ji(t, div, j_param, i_param, alternated=alternated, alt_m=alt_m)
    return None


__all__ = ["next_term_ji", "find_cycle"]
