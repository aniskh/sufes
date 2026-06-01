from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

from .algorithms import next_term_ji


def _compute_one_local(n0: int, k: int, i: int, j: int, max_iters: int, divergence_threshold: float, alternated: bool, alt_m: int) -> dict:
    """Simulate trajectory starting at ``n0`` until a cycle, divergence or max iters.

    Returns a small dict with keys:
    - n: int, starting value
    - stopping_time: Optional[int], preperiod length (None if not observed)
    - total_stopping_time: Optional[int], total steps performed until cycle/divergence or max_iters
    - reason: str in {"cycle","divergence","max_iters"}
    - steps: int, same as total_stopping_time (for convenience)
    - peak: int, maximum value encountered
    """
    t = int(n0)
    pos = {}  # map term -> index in path
    path = []
    peak = int(n0)
    min_val = int(n0)

    stopping_time_val: Optional[int] = None

    for step in range(int(max_iters)):
        # cycle detected
        if t in pos:
            return {
                "n": int(n0),
                "stopping_time": None if stopping_time_val is None else int(stopping_time_val),
                "total_stopping_time": int(step),
                "reason": "cycle",
                "steps": int(step),
                "peak": int(peak),
                "min": int(min_val),
                "never_below_start": bool(int(min_val) >= int(n0)),
            }

        # divergence check
        if abs(t) > float(divergence_threshold):
            return {
                "n": int(n0),
                "stopping_time": None if stopping_time_val is None else int(stopping_time_val),
                "total_stopping_time": int(step),
                "reason": "divergence",
                "steps": int(step),
                "peak": int(peak),
                "min": int(min_val),
                "never_below_start": bool(int(min_val) >= int(n0)),
            }

        pos[t] = len(path)
        path.append(t)
        if t > peak:
            peak = int(t)
        if t < min_val:
            min_val = int(t)

        # compute next term using user's algorithm
        nxt = int(next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m))

        # stopping_time: first time we see a value <= the
        # original starting value n0. We record the step count (1-based step
        # after applying the transition).
        if stopping_time_val is None and int(nxt) <= int(n0):
            stopping_time_val = step + 1

        t = nxt

    # reached max_iters
    return {
        "n": int(n0),
        "stopping_time": None if stopping_time_val is None else int(stopping_time_val),
        "total_stopping_time": int(max_iters),
        "reason": "max_iters",
        "steps": int(max_iters),
        "peak": int(peak),
        "min": int(min_val),
        "never_below_start": bool(int(min_val) >= int(n0)),
    }


def _stopping_worker_tuple(args: Tuple) -> dict:
    """Worker wrapper used by multiprocessing: unpack tuple and delegate."""
    return _compute_one_local(*args)


def stopping_time(
    n: int,
    k: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    workers: int = 4,
) -> dict:
    """Compute stopping and total stopping times for start values 1..n.

    If ``workers > 1`` a ThreadPoolExecutor is used. Results are written to
    ``run_dir`` as JSON (summary and full results).

    Output note: we serialize missing values as ``null`` (JSON ``null``) rather
    than NaN, because NaN isn't valid JSON and causes headaches when loading
    results in other environments.
    """
    os.makedirs(run_dir, exist_ok=True)
    n = int(n)
    k = int(k)
    i = int(i)
    j = int(j)

    results: List[dict] = []

    if int(workers) <= 1:
        for n0 in range(1, n + 1):
            results.append(_compute_one_local(n0, k, i, j, max_iters, divergence_threshold, alternated, alt_m))
    else:
        args_list = [
            (n0, int(k), int(i), int(j), int(max_iters), float(divergence_threshold), bool(alternated), int(alt_m))
            for n0 in range(1, n + 1)
        ]
        # Use thread pool executor for per-start parallelism. Preserve input order
        # by mapping over args_list and collecting results in the same order.
        try:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for res in executor.map(_stopping_worker_tuple, args_list):
                    results.append(res)
        except Exception:
            # fallback sequential on any failure
            results = []
            for n0 in range(1, n + 1):
                results.append(_compute_one_local(n0, k, i, j, max_iters, divergence_threshold, alternated, alt_m))

    # Augment results with the "never below start" property.
    # Some trajectories converge (cycle) without ever descending below n0.
    for r in results:
        if "min" not in r:
            # Backward compatibility if any code path omitted it.
            r["min"] = None
        if "never_below_start" not in r:
            try:
                r["never_below_start"] = bool(r.get("min") is not None and int(r.get("min")) >= int(r.get("n")))
            except Exception:
                r["never_below_start"] = False

    # Top 10 (largest n0) among those that never went below the start.
    never_below = [r for r in results if bool(r.get("never_below_start"))]
    never_below_sorted = sorted(never_below, key=lambda rr: int(rr.get("n", 0)), reverse=True)
    top10_never_below = never_below_sorted[:10]

    # Aggregate numerical arrays using JSON-compatible nulls for missing values.
    stopping_vals = [float(r["stopping_time"]) if r.get("stopping_time") is not None else None for r in results]
    total_vals = [float(r["total_stopping_time"]) if r.get("total_stopping_time") is not None else None for r in results]

    def _mean(xs: List[float]) -> Optional[float]:
        return (sum(xs) / len(xs)) if xs else None

    def _median(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        ys = sorted(xs)
        mid = len(ys) // 2
        return ys[mid] if (len(ys) % 2 == 1) else 0.5 * (ys[mid - 1] + ys[mid])

    stopping_known = [float(v) for v in stopping_vals if v is not None]
    total_known = [float(v) for v in total_vals if v is not None]

    summary = {
        "n": int(n),
        "k": int(k),
        "i": int(i),
        "j": int(j),
        "counts": len(results),
        "never_below_start_count": int(len(never_below)),
        "top10_never_below_start": top10_never_below,
        "stopping_times": stopping_vals,
        "total_stopping_times": total_vals,
        "stats": {
            "stopping_time": {
                "null_count": int(len(stopping_vals) - len(stopping_known)),
                "mean": _mean(stopping_known),
                "median": _median(stopping_known),
                "max": (max(stopping_known) if stopping_known else None),
            },
            "total_stopping_time": {
                "null_count": int(len(total_vals) - len(total_known)),
                "mean": _mean(total_known),
                "median": _median(total_known),
                "max": (max(total_known) if total_known else None),
            },
        },
    }

    base = f"stopping_upto_n{n}_k{k}_i{i}_j{j}"
    out_summary = os.path.join(run_dir, f"{base}_summary.json")
    try:
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    out_results = os.path.join(run_dir, f"{base}_results.json")
    try:
        with open(out_results, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)
    except Exception:
        pass

    # Console output: top 10 for this (k,i,j)
    if top10_never_below:
        print(f"[stopping_time] Top 10 n0 where trajectory never goes below start for (k={k}, i={i}, j={j}):")
        for r in top10_never_below:
            n0 = r.get("n")
            reason = r.get("reason")
            steps = r.get("steps")
            peak = r.get("peak")
            mn = r.get("min")
            st = r.get("stopping_time")
            print(f"  n0={n0}  reason={reason}  steps={steps}  stopping_time={st}  min={mn}  peak={peak}")

    return summary


__all__ = ["stopping_time"]
