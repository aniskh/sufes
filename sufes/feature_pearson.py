from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

from .algorithms import next_term_ji


def _pearson_from_lists(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient for two numeric lists.

    Returns math.nan if correlation is undefined (length < 2 or zero variance).
    """
    n = len(x)
    if n == 0 or n != len(y):
        return math.nan
    if n < 2:
        return math.nan
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    # population covariance; divide by n
    cov /= n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n
    if var_x <= 0 or var_y <= 0:
        return math.nan
    return cov / math.sqrt(var_x * var_y)


def pearson(
    n: int,
    k: int,
    i: int,
    j: int,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    workers: int = 4,
) -> Dict:
    """Compute Pearson correlation for all start values n' = 1..n (inclusive).

    For backwards compatibility the function signature is unchanged, but the
    behavior is: for each n0 in 1..n simulate the trajectory for (n0,k,i,j)
    and compute the Pearson correlation of successive residues as in the
    previous single-n implementation. The returned dict contains:
      - 'n': the provided upper bound N
      - 'k','i','j'
      - 'per_n': list of per-start summaries (each has keys 'n','pearson','reason','steps','peak')
      - 'mean_pearson': arithmetic mean of defined (non-NaN) pearson values, or None

    This keeps output files like `pearson_upto_n{N}_...` meaningful: they now
    contain the per-n breakdown used for plotting/analysis.
    """

    N = int(n)

    def _pearson_for_single(n0: int) -> Dict:
        # reuse the old single-n logic
        t = int(n0)
        pos = {}
        path: List[int] = []
        peak = int(n0)
        for step in range(int(max_iters)):
            if t in pos:
                reason = "cycle"
                break
            if abs(t) > float(divergence_threshold):
                reason = "divergence"
                break
            pos[t] = len(path)
            path.append(int(t))
            if t > peak:
                peak = int(t)
            t = int(next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m))
        else:
            reason = "max_iters"

        L = len(path)
        if L < 3:
            pearson_val = math.nan
        else:
            residues = [int(tt % int(k)) for tt in path]
            xs = [float(r) for r in residues[1:-1]]
            ys = [float(r) for r in residues[2:]]
            pearson_val = _pearson_from_lists(xs, ys)

        return {
            "n": int(n0),
            "pearson": float(pearson_val) if not (pearson_val is None) else math.nan,
            "reason": reason,
            "steps": int(L),
            "peak": int(peak),
        }

    per_list: List[Dict] = []
    ns = list(range(1, N + 1))
    if int(workers) <= 1:
        for r in map(_pearson_for_single, ns):
            per_list.append(r)
    else:
        try:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for r in executor.map(_pearson_for_single, ns):
                    per_list.append(r)
        except Exception:
            for r in map(_pearson_for_single, ns):
                per_list.append(r)

    # compute mean over defined pearson values
    vals = [p.get("pearson") for p in per_list if p.get("pearson") is not None and not math.isnan(float(p.get("pearson")))]
    mean_pearson = (sum(vals) / len(vals)) if vals else None

    return {
        "n": N,
        "k": int(k),
        "i": int(i),
        "j": int(j),
        "per_n": per_list,
        "mean_pearson": float(mean_pearson) if mean_pearson is not None else None,
    }


__all__ = ["pearson"]
