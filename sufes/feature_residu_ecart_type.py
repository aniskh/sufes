"""Feature: residu-ecart-type

Re-introduces the historical CLI feature `--residu-ecart-type-*` that was temporarily disabled
while splitting the monolithic `residu.py`.

Goal
----
For each prime k <= p and each j in a configurable range, simulate the (k,i,j) map and
compute the standard deviation (ecart-type) of residues r_t = t mod k.

We compute the standard deviation on *non-zero residues* by default (for better comparability
with other residue-focused features in this repo). If there are zero non-zero residues,
std is recorded as null.

Outputs
-------
- CSV + JSON with one row per (k,j) containing count_non_zero, mean_non_zero, std_non_zero, reason.
- Optional PNG per k (std vs j) if matplotlib is installed.

This module is designed to be called from `sufes.core`.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from .algorithms import next_term_ji


def _sieve_primes(pmax: int) -> List[int]:
    """Local prime sieve (kept local to avoid importing core)."""
    if pmax < 2:
        return []
    sieve = [True] * (pmax + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(pmax**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, pmax + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


@dataclass
class ResiduEcartTypeRow:
    k: int
    j: int
    i: int
    count_non_zero: int
    mean_non_zero: Optional[float]
    std_non_zero: Optional[float]
    reason: str


def _simulate_residues(
    n0: int,
    k: int,
    i: int,
    j: int,
    *,
    max_iters: int,
    divergence_threshold: float,
    alternated: bool,
    alt_m: int,
) -> Tuple[List[int], str]:
    """Return list of residues (including zeros) and stop reason."""

    residues: List[int] = []
    t = n0

    # Simple cycle detection via visited set; good enough for this feature.
    seen: set[int] = set()

    for _ in range(max_iters):
        if abs(t) > divergence_threshold:
            return residues, "divergence_threshold"

        if t in seen:
            return residues, "cycle"
        seen.add(t)

        residues.append(int(t % k))
        t = next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m)

    return residues, "max_iters"


def _mean_std_non_zero(residues: List[int]) -> Tuple[int, Optional[float], Optional[float]]:
    non_zero = [r for r in residues if r != 0]
    n = len(non_zero)
    if n == 0:
        return 0, None, None

    mean = sum(non_zero) / n
    var = sum((r - mean) ** 2 for r in non_zero) / n
    std = math.sqrt(var)
    return n, float(mean), float(std)


def residu_ecart_type(
    n_max: Optional[int],
    p: int,
    run_dir: str,
    *,
    i_val: int = 1,
    j_multiple: int = 1,
    single_n: Optional[int] = None,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    plot: bool = True,
) -> None:
    """Compute residue std-dev over many (k,j) and optionally over n=1..N.

    Contract
    --------
    - If `single_n` is provided: compute only for that n.
    - Else: `n_max` must be provided and we aggregate over n in 1..n_max.

    Aggregation
    -----------
    We aggregate by concatenating residue samples across n (same k,j) and then
    computing std of non-zero residues.
    """

    if single_n is None and n_max is None:
        raise ValueError("residu_ecart_type: require either single_n or n_max")

    os.makedirs(run_dir, exist_ok=True)

    ks = _sieve_primes(int(p))

    n_values: List[int]
    if single_n is not None:
        n_values = [single_n]
        label = f"n{single_n}"
    else:
        assert n_max is not None
        n_values = list(range(1, n_max + 1))
        label = f"N{n_max}"

    rows: List[ResiduEcartTypeRow] = []

    # Keep per-k arrays for plotting.
    per_kj_std: Dict[int, Dict[int, Optional[float]]] = {}

    for k in ks:
        max_j = j_multiple * k
        per_kj_std[k] = {}

        for j in range(max_j):
            all_residues: List[int] = []
            worst_reason = "cycle"

            for n0 in n_values:
                residues, reason = _simulate_residues(
                    n0,
                    k,
                    i_val,
                    j,
                    max_iters=max_iters,
                    divergence_threshold=divergence_threshold,
                    alternated=alternated,
                    alt_m=alt_m,
                )
                all_residues.extend(residues)

                # Keep a conservative "worst" reason to reflect non-convergence.
                if reason != "cycle":
                    worst_reason = reason

            count_nz, mean_nz, std_nz = _mean_std_non_zero(all_residues)
            per_kj_std[k][j] = std_nz

            rows.append(
                ResiduEcartTypeRow(
                    k=k,
                    j=j,
                    i=i_val,
                    count_non_zero=count_nz,
                    mean_non_zero=mean_nz,
                    std_non_zero=std_nz,
                    reason=worst_reason,
                )
            )

    base = f"residu_ecart_type_{label}_p{p}_i{i_val}_jmult{j_multiple}"

    json_path = os.path.join(run_dir, f"{base}.json")
    csv_path = os.path.join(run_dir, f"{base}.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "k",
                "j",
                "i",
                "count_non_zero",
                "mean_non_zero",
                "std_non_zero",
                "reason",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    if plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return

        # One figure with one subplot per k.
        ncols = 3
        nrows = (len(ks) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

        for idx, k in enumerate(ks):
            ax = axes[idx // ncols][idx % ncols]
            xs = list(range(len(per_kj_std[k])))
            ys = [per_kj_std[k][j] if per_kj_std[k][j] is not None else float("nan") for j in xs]
            ax.plot(xs, ys, marker=".", linewidth=1)
            ax.set_title(f"k={k}")
            ax.set_xlabel("j")
            ax.set_ylabel("std(non-zero residues)")
            ax.grid(True, alpha=0.25)

        # Hide unused axes
        for idx in range(len(ks), nrows * ncols):
            axes[idx // ncols][idx % ncols].axis("off")

        fig.suptitle(f"Residue std-dev (non-zero) — {label}, primes k<=p={p}, i={i_val}", y=0.98)
        fig.tight_layout()

        png_path = os.path.join(run_dir, f"{base}.png")
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
