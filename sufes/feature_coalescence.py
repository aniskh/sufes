"""Coalescence feature.

New definition (2026-05):

For a fixed (k,i,j), we iterate N' from 1..N and test whether the trajectory
starting at N' intersects the union of visited states from *any* starting value
<= N'-1. In other words, we maintain a growing set:

    visited_prefix(N') = ⋃_{m=1..N'} O(m)

and for each N' we test if O(N') ∩ visited_prefix(N'-1) ≠ ∅.

For each N' we record whether it coalesced, and we report aggregated counts:
    - number of coalescences up to N
    - coalescence rate = count / N

Outputs:
    - JSON summary
    - CSV summary table: k,i,j,N,coalescence_count,coalescence_rate
    - optionally a detailed per-n CSV/JSON when write_verbose is enabled.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

from .core import _log_start, _log_end
from .algorithms import next_term_ji


def coalescence(
    n_val: int,
    k: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    workers: int = 4,
    write_csv: bool = True,
    write_png: bool = True,
    write_verbose: bool = False,
) -> None:
    """Run prefix coalescence detection (new definition).

    For each N' in 1..n_val, detect whether the orbit of N' intersects the
    union of all previously visited states (from 1..N'-1). We count how many
    N' coalesce and write an aggregate summary.
    """

    if alt_m >= k or i_val >= k or i_val < 1:
        raise SystemExit(f"Invalid parameters: must have 0 < i < k and alt_m < k (k={k})")

    os.makedirs(run_dir, exist_ok=True)
    run_start = _log_start(f"coalescence prefix k={k} i={i_val} j={j_val} upto_n={n_val}")

    # memoize next_term_ji to avoid recomputing transitions repeatedly
    next_cache: Dict[int, int] = {}

    def _next(n: int) -> int:
        nn = next_cache.get(n)
        if nn is None:
            nn = next_term_ji(n, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
            next_cache[n] = nn
        return nn

    def _trace_until_hit(n0: int, visited_prefix: Set[int]) -> Dict[str, object]:
        """Trace trajectory starting at n0 until it hits visited_prefix or stops.

        Returns a dict with:
          - n
          - coalesced (bool)
          - hit_value (first value in visited_prefix reached, if any)
          - capture_time (steps to reach hit_value)
          - reason: coalesced|divergence|max_iters|none
        Also returns the list of visited states for this trajectory in `path`.
        """
        t0 = time.perf_counter()
        cur = int(n0)
        path: List[int] = [cur]
        if cur in visited_prefix:
            return {
                "n": int(n0),
                "coalesced": True,
                "hit_value": int(cur),
                "capture_time": 0,
                "reason": "coalesced",
                "elapsed_sec": time.perf_counter() - t0,
                "path": path,
            }

        for step in range(1, int(max_iters) + 1):
            cur = _next(cur)
            path.append(cur)
            if abs(cur) > divergence_threshold:
                return {
                    "n": int(n0),
                    "coalesced": False,
                    "hit_value": None,
                    "capture_time": None,
                    "reason": "divergence",
                    "elapsed_sec": time.perf_counter() - t0,
                    "path": path,
                }
            if cur in visited_prefix:
                return {
                    "n": int(n0),
                    "coalesced": True,
                    "hit_value": int(cur),
                    "capture_time": int(step),
                    "reason": "coalesced",
                    "elapsed_sec": time.perf_counter() - t0,
                    "path": path,
                }

        return {
            "n": int(n0),
            "coalesced": False,
            "hit_value": None,
            "capture_time": None,
            "reason": "max_iters",
            "elapsed_sec": time.perf_counter() - t0,
            "path": path,
        }

    visited_prefix: Set[int] = set()
    details: List[Dict[str, object]] = []
    capture_series: List[Tuple[int, Optional[int]]] = []  # (n, capture_time)
    # We don't count n=1 in the rate, because it has no previous prefix
    # trajectory to coalesce with.
    coalescence_count = 0
    capture_time_sum = 0.0
    capture_time_count = 0
    # intentionally quiet: no per-n logging (this feature can be very chatty)

    # We keep this feature sequential by default because it has a sequential
    # dependency (the prefix visited set grows every iteration). We will only
    # parallelize within a single trajectory if needed in the future, but the
    # user-level contract for bulk features is `workers` which is already used
    # elsewhere in the repository.
    if int(workers) != 1:
        # keep deterministic sequential behavior; workers is accepted for CLI
        # consistency but does not change the algorithm.
        pass

    for n0 in range(1, int(n_val) + 1):
        r = _trace_until_hit(n0, visited_prefix=visited_prefix)
        try:
            capture_series.append((int(n0), r.get("capture_time")))
        except Exception:
            pass
        # record detail (optionally)
        if bool(write_verbose):
            details.append({k: v for k, v in r.items() if k != "path"})

        # update counts (exclude n0=1 from the metric)
        if n0 != 1 and bool(r.get("coalesced")):
            coalescence_count += 1
            ct = r.get("capture_time")
            if ct is not None:
                try:
                    capture_time_sum += float(ct)
                    capture_time_count += 1
                except Exception:
                    pass

        # update visited_prefix with all visited states of this new trajectory
        try:
            for v in r.get("path", []):
                visited_prefix.add(int(v))
        except Exception:
            # ensure we still add at least n0
            visited_prefix.add(int(n0))

        # no per-n progress logging

    # aggregate summary
    denom = max(0, int(n_val) - 1)
    coalescence_rate = float(coalescence_count) / float(denom) if denom > 0 else 0.0
    mean_capture_time = (capture_time_sum / float(capture_time_count)) if capture_time_count > 0 else None
    summary = {
        "params": {"k": int(k), "i": int(i_val), "j": int(j_val), "n": int(n_val)},
        "summary": {
            "coalescence_count": int(coalescence_count),
            "coalescence_rate": float(coalescence_rate),
            "mean_capture_time": mean_capture_time,
            "denominator": int(denom),
            "note": "Rate is computed over n=2..N (n=1 excluded).",
        },
    }
    if bool(write_verbose):
        summary["details"] = details

    out_json = os.path.join(run_dir, f"coalescence_prefix_upto_n{n_val}_k{k}_i{i_val}_j{j_val}.json")
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(summary, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if bool(write_csv):
        out_csv = os.path.join(run_dir, "coalescence_prefix_summary.csv")
        try:
            # append-friendly global summary table per run_dir
            write_header = not os.path.exists(out_csv)
            with open(out_csv, "a", encoding="utf-8") as cf:
                if write_header:
                    cf.write("k,i,j,N,coalescence_count,coalescence_rate,mean_capture_time\n")
                cf.write(
                    f"{int(k)},{int(i_val)},{int(j_val)},{int(n_val)},{int(coalescence_count)},{coalescence_rate},{mean_capture_time}\n"
                )
        except Exception:
            pass

        if bool(write_verbose):
            out_details = os.path.join(run_dir, f"coalescence_prefix_upto_n{n_val}_k{k}_i{i_val}_j{j_val}_details.csv")
            try:
                with open(out_details, "w", encoding="utf-8") as cf:
                    cf.write("n,coalesced,hit_value,capture_time,reason,elapsed_sec\n")
                    for d in details:
                        cf.write(
                            f"{d.get('n')},{d.get('coalesced')},{d.get('hit_value')},{d.get('capture_time')},{d.get('reason')},{d.get('elapsed_sec')}\n"
                        )
            except Exception:
                pass

    if bool(write_png):
        # Plot: capture_time (or NaN) vs n
        try:
            import matplotlib.pyplot as plt

            xs: List[int] = []
            ys: List[float] = []
            for n0, ct in capture_series:
                xs.append(int(n0))
                if ct is None:
                    ys.append(float("nan"))
                else:
                    try:
                        ys.append(float(ct))
                    except Exception:
                        ys.append(float("nan"))

            plt.figure(figsize=(8, 3.6))
            plt.plot(xs, ys, marker='o', linestyle='-', linewidth=0.7)
            plt.xlabel("n")
            plt.ylabel("capture_time")
            plt.title(f"Prefix coalescence capture_time (k={k}, i={i_val}, j={j_val})")
            plt.grid(True, alpha=0.25)
            out_png = os.path.join(run_dir, f"coalescence_prefix_upto_n{n_val}_k{k}_i{i_val}_j{j_val}_capture_time.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
        except Exception:
            pass

    try:
        _ = _log_end(f"coalescence prefix k={k} i={i_val} j={j_val} upto_n={n_val}", run_start)
    except Exception:
        pass

    # One final human-friendly line for this (k,i,j)
    try:
        print(
            f"coalescence done: k={int(k)} i={int(i_val)} j={int(j_val)} N={int(n_val)} "
            f"count={int(coalescence_count)} denom={int(denom)} rate={float(coalescence_rate):.6g} "
            f"mean_capture_time={mean_capture_time}"
        )
    except Exception:
        pass
