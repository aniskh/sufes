"""Feature: trajectory-minimum

Compute the minimal value reached by a single trajectory starting at n for parameters (k,i,j)
and compare it to the threshold (n*(k+i)^k)/(k^(k-2)).

This file exposes a function `trajectory_minimum(n, k, i, j, run_dir, **options)` that mirrors the
call pattern used by other features and can be dispatched from `sufes.core`.
"""
from __future__ import annotations

import os
import json
import csv
import glob
from typing import Optional, Dict
from fractions import Fraction

from .algorithms import next_term_ji


def _compute_min_for_n(n_val: int, k: int, i: int, j: int, *, max_iters: int, divergence_threshold: float, alternated: bool, alt_m: int) -> Dict:
    n0 = int(n_val)
    kk = int(k)
    ii = int(i)
    jj = int(j)

    seen: Dict[int, int] = {}
    seq: list[int] = []
    t = int(n0)
    reason = "max_iters"
    preperiod: Optional[int] = None
    iterations_nbr = 0
    for step in range(int(max_iters)):
        if abs(t) > float(divergence_threshold):
            reason = "divergence_threshold"
            break
        if t in seen:
            preperiod = seen[t]
            reason = "cycle"
            break
        seen[t] = step
        seq.append(int(t))

        # compute next term and detect whether the step was a multiplication
        if t % kk == 0:
            next_t = t // kk
            was_mult = False
        else:
            next_t = next_term_ji(t, kk, jj, ii, alternated=bool(alternated), alt_m=int(alt_m))
            was_mult = True

        # If it was a multiplication step, check if next_t is divisible by k^m for m>=2
        if was_mult:
            # only consider positive integers for exponent counting
            try:
                tmp = abs(int(next_t))
            except Exception:
                tmp = 0
            if tmp != 0:
                e = 0
                while tmp % kk == 0:
                    tmp //= kk
                    e += 1
                if e >= 2:
                    iterations_nbr += 1

        t = next_t

    # Include cycle states in the minimum: use the full recorded sequence
    # (seq contains transient values followed by the first occurrence of the cycle elements).
    min_val = int(min(seq)) if seq else None

    try:
        num = int(n0) * pow(int(kk) + int(ii), int(kk))
        den = pow(int(kk), int(kk) - 2) if int(kk) - 2 >= 0 else 1
        threshold = Fraction(num, den)
        below = (Fraction(int(min_val), 1) < threshold) if min_val is not None else None
    except Exception:
        threshold = None
        below = None

    return {
        "n": int(n0),
        "k": int(kk),
        "i": int(ii),
        "j": int(jj),
        "min_value": int(min_val) if min_val is not None else None,
    "iterations_nbr": int(iterations_nbr),
        "threshold": str(threshold) if threshold is not None else None,
        "min_below_threshold": bool(below) if below is not None else None,
        "reason": reason,
        "preperiod": int(preperiod) if preperiod is not None else None,
    }


def trajectory_minimum(
    n_val: int,
    k: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    # Single-n wrapper that writes individual JSON/CSV using helper
    os.makedirs(run_dir, exist_ok=True)
    res = _compute_min_for_n(n_val, k, i, j, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m)

    base = f"trajectory_min_n{res['n']}_k{res['k']}_i{res['i']}_j{res['j']}"
    out_json = os.path.join(run_dir, f"{base}.json")
    out_csv = os.path.join(run_dir, f"{base}.csv")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "k", "i", "j", "min_value", "iterations_nbr", "threshold", "min_below_threshold", "reason", "preperiod"])
            w.writerow([res["n"], res["k"], res["i"], res["j"], res["min_value"], res.get("iterations_nbr", 0), res["threshold"], res["min_below_threshold"], res["reason"], res.get("preperiod")])
    except Exception:
        pass

    try:
        thr_repr = str(res.get("threshold")) if res.get("threshold") is not None else "?"
        below_str = str(bool(res.get("min_below_threshold"))) if res.get("min_below_threshold") is not None else "?"
        print(f"trajectory-minimum: n={res['n']},k={res['k']},i={res['i']},j={res['j']},min={res['min_value']},iterations_nbr={res.get('iterations_nbr',0)},threshold={thr_repr},below={below_str},reason={res['reason']}")
    except Exception:
        pass


def trajectory_minimum_range(
    n_val: int,
    k: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    workers: int = 4,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Compute trajectory minima for all n=1..N and write aggregated CSV/JSON.

    Uses ThreadPoolExecutor when workers>1 and preserves deterministic ordering
    using executor.map.
    """
    N = int(n_val)
    os.makedirs(run_dir, exist_ok=True)

    args_list = [(n0, k, i, j) for n0 in range(1, N + 1)]

    results = []
    if int(workers) <= 1:
        for n0 in range(1, N + 1):
            res = _compute_min_for_n(n0, k, i, j, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m)
            results.append(res)
    else:
        try:
            from concurrent.futures import ThreadPoolExecutor

            def _task(arg_tpl):
                (n0, kk, ii, jj) = arg_tpl
                return _compute_min_for_n(n0, kk, ii, jj, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m)

            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for r in executor.map(_task, args_list):
                    results.append(r)
        except Exception:
            # fallback sequential
            for n0 in range(1, N + 1):
                res = _compute_min_for_n(n0, k, i, j, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m)
                results.append(res)

    # write aggregated outputs
    out_csv = os.path.join(run_dir, f"trajectory_min_all_N{N}_k{k}_i{i}_j{j}.csv")
    out_json = os.path.join(run_dir, f"trajectory_min_all_N{N}_k{k}_i{i}_j{j}.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "i", "j", "n", "min_value", "iterations_nbr", "threshold", "min_below_threshold", "reason", "preperiod"])
            w.writeheader()
            for r in results:
                w.writerow({"k": r["k"], "i": r["i"], "j": r["j"], "n": r["n"], "min_value": r["min_value"], "iterations_nbr": r.get("iterations_nbr", 0), "threshold": r["threshold"], "min_below_threshold": r["min_below_threshold"], "reason": r["reason"], "preperiod": r.get("preperiod")})
    except Exception:
        pass

    # compute mean iterations_nbr for this k and write a small CSV summary
    try:
        total = 0
        count = 0
        for r in results:
            if r.get("iterations_nbr") is not None:
                total += int(r.get("iterations_nbr"))
                count += 1
        mean_val = (total / count) if count > 0 else 0.0
        mean_csv = os.path.join(run_dir, f"trajectory_min_mean_N{N}_k{k}_i{i}_j{j}.csv")
        with open(mean_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "i", "j", "mean_iterations_nbr"])
            w.writerow([int(k), int(i), int(j), float(mean_val)])
    except Exception:
        pass

    # Also write a compact table with columns k,i,j,n,min_trajectory as requested by users.
    compact_csv = os.path.join(run_dir, f"trajectory_min_compact_N{N}_k{k}_i{i}_j{j}.csv")
    try:
        with open(compact_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "i", "j", "n", "min_trajectory", "iterations_nbr"])
            for r in results:
                w.writerow([r["k"], r["i"], r["j"], r["n"], r["min_value"], r.get("iterations_nbr", 0)])
    except Exception:
        pass

    # Attempt to plot iterations_nbr for multiple k values present in the same run dir.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pattern = os.path.join(run_dir, f"trajectory_min_compact_N{N}_k*_i{i}_j{j}.csv")
        files = sorted(glob.glob(pattern))
        if files:
            series = []
            for fn in files:
                basename = os.path.basename(fn)
                try:
                    part = basename.split("_k")[1]
                    kpart = part.split("_")[0]
                    kk = int(kpart)
                except Exception:
                    continue
                ns = []
                iters = []
                with open(fn, "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        try:
                            ns.append(int(row.get("n", "0")))
                            iters.append(int(row.get("iterations_nbr", "0") or 0))
                        except Exception:
                            ns.append(int(row.get("n", "0")))
                            iters.append(0)
                series.append({"k": kk, "n": ns, "iters": iters})

            if series:
                series.sort(key=lambda s: s["k"])
                n_k = len(series)
                cols = min(4, n_k)
                rows = (n_k + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
                for idx, s in enumerate(series):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r][c]
                    ax.plot(s["n"], s["iters"], marker=".", linestyle="-", markersize=2)
                    ax.set_title(f"k={s['k']}")
                    ax.set_xlim(1, N)
                    if r == rows - 1:
                        ax.set_xlabel("n")
                    ax.set_ylabel("iterations_nbr")
                total = rows * cols
                for idx in range(n_k, total):
                    r = idx // cols
                    c = idx % cols
                    axes[r][c].set_visible(False)

                fig.tight_layout()
                out_png = os.path.join(run_dir, f"trajectory_min_iterations_by_k_N{N}_i{i}_j{j}.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
    except Exception:
        # plotting is optional
        pass

    # Return the result list so callers (range-mode) can aggregate without re-reading files.
    return results
