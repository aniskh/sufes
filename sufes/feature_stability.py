"""Stability feature.

This feature computes a simple critical valuation quantity:

  v_crit = ceil(log_k(k + i))

where k >= 2 and i is an integer (typically 0 <= i < k in the rest of the
project, but this function does not require it).

CLI wiring lives in `sufes.core` via:
  --stability-k, --stability-i

Output written to run_dir:
  - stability_k{k}_i{i}.json

Notes
-----
We compute ceil(log_k(x)) robustly for integer x using integer
arithmetic (no floating-point rounding issues):

  v = min{ v in N : k^v >= x }

"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass


def stability_vcrit(k: int, i: int) -> int:
    """Return v_crit = ceil(log_k(k+i)).

    Raises
    ------
    ValueError
        If k < 2 or if k + i <= 0.
    """
    k = int(k)
    i = int(i)
    if k < 2:
        raise ValueError(f"k must be >= 2 (got {k})")

    x = k + i
    if x <= 0:
        raise ValueError(f"k + i must be > 0 (got k+i={x} for k={k}, i={i})")

    # Find minimal v >= 0 such that k**v >= x.
    v = 0
    p = 1
    while p < x:
        p *= k
        v += 1
    return v


def stability_lambda_crit(k: int, i: int) -> float:
    """Return lambda_crit = log(k+i) - log(k) * (k-1)/k.

    Notes
    -----
    Uses natural logarithms (math.log). The base choice is irrelevant up to a
    constant factor for comparisons; this follows the formula as provided.

    Raises
    ------
    ValueError
        If k < 2 or if k+i <= 0.
    """
    k = int(k)
    i = int(i)
    if k < 2:
        raise ValueError(f"k must be >= 2 (got {k})")
    x = k + i
    if x <= 0:
        raise ValueError(f"k + i must be > 0 (got k+i={x} for k={k}, i={i})")
    return float(math.log(x) - math.log(k) * (k - 1) / k)


@dataclass(frozen=True)
class StabilityResult:
    k: int
    i: int
    v_crit: int
    lambda_crit: float


def run_stability(k: int, i: int, run_dir: str) -> StabilityResult:
    """Compute stability v_crit and write a JSON summary to `run_dir`."""
    os.makedirs(run_dir, exist_ok=True)

    v = stability_vcrit(k, i)
    lam = stability_lambda_crit(k, i)
    result = StabilityResult(k=int(k), i=int(i), v_crit=int(v), lambda_crit=float(lam))

    payload = {"k": result.k, "i": result.i, "v_crit": result.v_crit, "lambda_crit": result.lambda_crit}
    out_path = os.path.join(run_dir, f"stability_k{result.k}_i{result.i}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return result
