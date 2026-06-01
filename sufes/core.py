"""Core implementation moved from the legacy top-level script.

This module exposes the main algorithmic functions and the CLI entry
point `main()` so the project can be used either as a package
(`python -m sufes`) or via the old script `sufes_general.py`.

The content was migrated from the original `sufes_general.py`.
"""
import argparse
import json
import os
import sys
import csv
from concurrent.futures import ThreadPoolExecutor
import time
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Normalized defaults used across all features (can always be overridden via CLI)
DEFAULT_MAX_ITERS = 500_000
DEFAULT_DIVERGENCE_THRESHOLD = 1e18

# Extracted feature implementations
from .feature_proof import run_proof as _run_proof
from . import feature_cycle as cycle_mod

# Algorithm primitives live in sufes.algorithms (not in core) to prevent
# circular imports when features import algorithms while core imports features.
from .algorithms import find_cycle, next_term_ji
from . import feature_stopping_time as stopping_time_mod


# Simple timing helpers used to print start/end and compute elapsed seconds
def _log_start(name: str) -> float:
    ts = time.perf_counter()
    try:
        print(f"[TIME] START {name} at {datetime.now().isoformat()}")
    except Exception:
        print(f"[TIME] START {name}")
    sys.stdout.flush()
    return ts


def _log_end(name: str, start_ts: float) -> float:
    elapsed = time.perf_counter() - start_ts
    try:
        print(f"[TIME] END   {name} elapsed {elapsed:.3f} s")
    except Exception:
        print(f"[TIME] END   {name} elapsed {elapsed:.3f} s")
    sys.stdout.flush()
    return elapsed


## next_term_ji and find_cycle are imported from sufes.algorithms


def analyze_range(start: int, end: int, base: int = 3, k: Optional[int] = None, j_param: Optional[int] = 0, i_param: int = 1, compact: bool = False, divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD, alternated: bool = False, alt_m: int = 1, max_iters: int = DEFAULT_MAX_ITERS) -> Dict[str, object]:
    cache: Dict[int, Dict[str, object]] = {}

    def canonical_cycle(c: List[int]) -> tuple:
        if not c:
            return tuple()
        rotations = [tuple(c[i:] + c[:i]) for i in range(len(c))]
        return min(rotations)

    div = k if k is not None else base
    j_effective = j_param if j_param is not None else 0

    def exp_div(x: int) -> int:
        kk = 0
        while x % div == 0 and x != 0:
            kk += 1
            x //= div
        return kk

    cycles: Dict[tuple, Dict[str, object]] = {}
    failures: List[int] = []
    max_preperiod = 0
    max_cycle_len = 0
    total_div_events = 0
    total_mul_events = 0
    stopping_times: List[int] = []
    peak_values: List[int] = []

    total = end - start + 1
    _start_ts = _log_start(f"analyze_range {start}-{end} div={div} j={j_effective} i={i_param}")

    for idx, n in enumerate(range(start, end + 1), 1):
        if idx % 100_000 == 0:
            print(f"Progress: {idx}/{total} starting values processed")

        path: List[int] = []
        pos_map: Dict[int, int] = {}
        t = n
        for _ in range(max_iters):
            if t in pos_map:
                start_idx = pos_map[t]
                cycle_list = path[start_idx:]
                cycle = canonical_cycle(cycle_list)

                div_from_here_local: Dict[int, int] = {}
                power_from_here_local: Dict[int, Dict[int, int]] = {}
                for v in reversed(path):
                    nxt = next_term_ji(v, div, j_effective, i_param, alternated=alternated, alt_m=alt_m)
                    base_div = 1 if v % div == 0 else 0

                    kexp = exp_div(v)
                    if nxt in cache:
                        next_div = cache[nxt]["div_from_here"]
                        next_pow = cache[nxt]["power_from_here"]
                    else:
                        next_div = div_from_here_local.get(nxt, 0)
                        next_pow = power_from_here_local.get(nxt, {})

                    div_from_here_local[v] = base_div + next_div
                    pw = dict(next_pow)
                    if kexp >= 1:
                        pw[kexp] = pw.get(kexp, 0) + 1
                    power_from_here_local[v] = pw

                for i, v in enumerate(path):
                    cache[v] = {
                        "cycle": cycle,
                        "dist_to_cycle": max(0, pos_map[t] - pos_map[v]) if t in pos_map else 0,
                        "div_from_here": div_from_here_local[v],
                        "power_from_here": power_from_here_local[v],
                    }

                preperiod = start_idx
                if compact:
                    cdata = cycles.setdefault(cycle, {"num_origins": 0, "div_total": 0, "power_counts": {}, "ge2_total": 0})
                    cdata["num_origins"] += 1
                else:
                    cdata = cycles.setdefault(cycle, {"origins": [], "div_total": 0, "power_counts": {}, "ge2_total": 0})
                    cdata["origins"].append(n)
                total_div = div_from_here_local.get(n, cache.get(n, {}).get("div_from_here", 0))
                cdata["div_total"] += total_div
                ge2 = 0
                power_map_n = power_from_here_local.get(n, cache.get(n, {}).get("power_from_here", {}))
                for e, cnt in power_map_n.items():
                    cdata["power_counts"][e] = cdata["power_counts"].get(e, 0) + cnt
                    if e >= 2:
                        ge2 += cnt
                cdata["ge2_total"] += ge2

                if preperiod > max_preperiod:
                    max_preperiod = preperiod
                if len(cycle) > max_cycle_len:
                    max_cycle_len = len(cycle)
                peak = max(path) if path else n
                stopping_times.append(preperiod)
                peak_values.append(int(peak))
                break

            if t in cache:
                known = cache[t]
                known_cycle = known["cycle"]
                d_t = known["dist_to_cycle"]

                div_from_here_local: Dict[int, int] = {}
                power_from_here_local: Dict[int, Dict[int, int]] = {}

                for v in reversed(path):
                    nxt = next_term_ji(v, div, j_effective, i_param, alternated=alternated, alt_m=alt_m)
                    base_div = 1 if v % div == 0 else 0

                    kexp = exp_div(v)
                    if nxt in cache:
                        next_div = cache[nxt]["div_from_here"]
                        next_pow = cache[nxt]["power_from_here"]
                    else:
                        next_div = div_from_here_local.get(nxt, 0)
                        next_pow = power_from_here_local.get(nxt, {})

                    div_from_here_local[v] = base_div + next_div
                    pw = dict(next_pow)
                    if kexp >= 1:
                        pw[kexp] = pw.get(kexp, 0) + 1
                    power_from_here_local[v] = pw

                for i, v in enumerate(path):
                    cache[v] = {
                        "cycle": known_cycle,
                        "dist_to_cycle": len(path) - i + d_t,
                        "div_from_here": div_from_here_local[v],
                        "power_from_here": power_from_here_local[v],
                    }

                preperiod = len(path) + d_t
                if compact:
                    cdata = cycles.setdefault(known_cycle, {"num_origins": 0, "div_total": 0, "power_counts": {}, "ge2_total": 0})
                    cdata["num_origins"] += 1
                else:
                    cdata = cycles.setdefault(known_cycle, {"origins": [], "div_total": 0, "power_counts": {}, "ge2_total": 0})
                    cdata["origins"].append(n)
                total_div = div_from_here_local.get(n, cache.get(n, {}).get("div_from_here", 0))
                cdata["div_total"] += total_div
                ge2 = 0
                power_map_n = power_from_here_local.get(n, cache.get(n, {}).get("power_from_here", {}))
                for e, cnt in power_map_n.items():
                    cdata["power_counts"][e] = cdata["power_counts"].get(e, 0) + cnt
                    if e >= 2:
                        ge2 += cnt
                cdata["ge2_total"] += ge2

                if preperiod > max_preperiod:
                    max_preperiod = preperiod
                if len(known_cycle) > max_cycle_len:
                    max_cycle_len = len(known_cycle)
                peak = max(path) if path else n
                stopping_times.append(preperiod)
                peak_values.append(int(peak))
                break

            pos_map[t] = len(path)
            path.append(t)
            if t % div == 0:
                total_div_events += 1
            else:
                total_mul_events += 1
            t = next_term_ji(t, div, j_effective, i_param, alternated=alternated, alt_m=alt_m)
            if abs(t) > divergence_threshold:
                failures.append(n)
                try:
                    peak = max(path) if path else n
                    peak_values.append(int(peak))
                except Exception:
                    pass
                break

        else:
            failures.append(n)
            try:
                peak = max(path) if path else n
                peak_values.append(int(peak))
            except Exception:
                pass

    _elapsed = _log_end(f"analyze_range {start}-{end} div={div} j={j_effective} i={i_param}", _start_ts)
    return {
        "start": start,
        "end": end,
        "total": total,
        "j": j_effective,
        "i": i_param,
        "divisor": div,
        "total_div_events": total_div_events,
        "total_mul_events": total_mul_events,
        "distinct_cycles": len(cycles),
        "cycles": cycles,
        "failures": failures,
        "max_preperiod": max_preperiod,
        "max_cycle_len": max_cycle_len,
        "avg_stopping_time": (sum(stopping_times) / len(stopping_times)) if stopping_times else None,
        "median_stopping_time": (sorted(stopping_times)[len(stopping_times)//2] if stopping_times else None),
        "max_stopping_time": (max(stopping_times) if stopping_times else None),
        "max_peak": (max(peak_values) if peak_values else None),
        "mean_peak": (sum(peak_values) / len(peak_values) if peak_values else None),
        "elapsed_sec": _elapsed,
    }


def save_summary_json(path: str, summary: Dict[str, object]) -> None:
    serializable = dict(summary)
    serial_cycles = {}
    for k, v in summary["cycles"].items():
        serial_cycles[json.dumps(list(k))] = v
    serializable["cycles"] = serial_cycles
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def _serialize_cycles(cycles: Dict[tuple, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for kcycle, v in cycles.items():
        out[json.dumps(list(kcycle))] = v
    return out


def analyze_chunk(args_tuple: tuple) -> Dict[str, object]:
    (start, end, base, k, j_param, i_param, compact, divergence_threshold, alternated, alt_m, max_iters) = args_tuple
    _ts = _log_start(f"analyze_chunk {start}-{end} k={k} j={j_param} i={i_param}")
    s = analyze_range(start, end, base=base, k=k, j_param=j_param, i_param=i_param, compact=compact, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m, max_iters=max_iters)
    serial = dict(s)
    serial_cycles = _serialize_cycles(s.get("cycles", {}))
    serial["cycles"] = serial_cycles
    try:
        serial_elapsed = s.get("elapsed_sec", None)
        if serial_elapsed is None:
            serial_elapsed = _log_end(f"analyze_chunk {start}-{end} k={k} j={j_param} i={i_param}", _ts)
        serial["elapsed_sec"] = serial_elapsed
    except Exception:
        pass
    return serial


def _prove_combo_persist(args: Tuple) -> Tuple[Tuple[int,int,int], Dict[str, object]]:
    start_time = time.perf_counter()
    (k, i_val, j_val, proof_max_n, alternated, alt_m, max_iters, divergence_threshold, run_dir) = args
    fname = os.path.join(run_dir, f"proof_k{k}_i{i_val}_j{j_val}_maxproved.txt")
    max_proved = 0
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as rf:
                txt = rf.read().strip()
                if txt:
                    max_proved = int(txt)
        except Exception:
            max_proved = 0

    # compact proved flags to reduce memory usage
    proven_flag = bytearray(proof_max_n + 1)
    for v in range(1, max_proved + 1):
        if v <= proof_max_n:
            proven_flag[v] = 1

    def prove_for_combo_local(start_n: int):
        nonlocal max_proved
        # reuse seen set across iterations to avoid many allocations
        seen = set()
        for n in range(start_n, proof_max_n + 1):
            seen.clear()
            t = n
            for step in range(max_iters):
                if abs(t) > divergence_threshold:
                    try:
                        with open(fname, "w", encoding="utf-8") as wf:
                            wf.write(str(max_proved))
                    except Exception:
                        pass
                    return max_proved, {"failed_n": n, "reason": "divergence_threshold", "steps": step}
                if t in seen:
                    if n <= proof_max_n:
                        proven_flag[n] = 1
                    max_proved = n
                    break
                if t < n or (0 <= t <= proof_max_n and proven_flag[int(t)]):
                    if n <= proof_max_n:
                        proven_flag[n] = 1
                    max_proved = n
                    break
                seen.add(t)
                t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
            else:
                try:
                    with open(fname, "w", encoding="utf-8") as wf:
                        wf.write(str(max_proved))
                except Exception:
                    pass
                return max_proved, {"failed_n": n, "reason": "max_iters", "steps": max_iters}
        try:
            with open(fname, "w", encoding="utf-8") as wf:
                wf.write(str(max_proved))
        except Exception:
            pass
        return max_proved, None

    start_n = max_proved + 1
    if start_n > proof_max_n:
        elapsed = time.perf_counter() - start_time
        return ((k, i_val, j_val), {"max_proved": max_proved, "elapsed_sec": elapsed})

    maxp, info = prove_for_combo_local(start_n)
    elapsed = time.perf_counter() - start_time
    res = {"max_proved": maxp, "elapsed_sec": elapsed}
    if info is not None:
        res.update(info)
    return ((k, i_val, j_val), res)


def _plot_residu_distribution_rows(rows: List[Dict[str, object]], out_path: str, title: str):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping residu_distribution plot")
        return

    if not rows:
        print("No residu_distribution rows to plot")
        return

    by_k = {}
    for r in rows:
        k = int(r.get('k'))
        by_k.setdefault(k, []).append(r)

    ks = sorted(by_k.keys())
    n_k = len(ks)
    ncols = min(3, n_k)
    nrows = int(math.ceil(n_k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

    for idx, k in enumerate(ks):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_title(f"k={k}")
        ax.set_xlabel("j")
        ax.set_ylabel("mean residue (non-zero)")

        rows_k = by_k[k]
        js = [int(r['j']) for r in rows_k]
        order = np.argsort(js)
        js_sorted = np.array(js)[order]

        mean_vals = [r['mean_residue'] if r['mean_residue'] is not None else float('nan') for r in rows_k]
        mean_sorted = np.array(mean_vals, dtype=float)[order]
        counts = np.array([int(r.get('count_non_zero', 0)) for r in rows_k])[order]

        # optional skewness series (non-zero residues)
        skew_vals = [r.get('skew_non_zero') if r.get('skew_non_zero') is not None else float('nan') for r in rows_k]
        skew_sorted = np.array(skew_vals, dtype=float)[order]

        ax.plot(js_sorted, mean_sorted, marker='o', linestyle='-', color='C0', label='mean')
        # plot skewness on a secondary axis (if present)
        ax_sk = ax.twinx()
        ax_sk.plot(js_sorted, skew_sorted, marker='x', linestyle='--', color='C2', alpha=0.9, label='skew')
        ax_sk.set_ylabel('skew_non_zero', color='C2')

        # count_non_zero on a third axis (offset) for readability
        ax2 = ax.twinx()
        try:
            ax2.spines['right'].set_position(('outward', 45))
        except Exception:
            pass
        ax2.bar(js_sorted, counts, alpha=0.18, color='C1', width=0.6)
        ax2.set_ylabel('count_non_zero', color='C1')

        try:
            ax.legend(loc='upper left', fontsize=8)
        except Exception:
            pass

        ax.set_xticks(list(js_sorted))
        if all(np.isnan(mean_sorted)):
            ax.text(0.5, 0.5, 'no non-zero residues', ha='center', va='center', transform=ax.transAxes)

    for j in range(n_k, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        fig.savefig(out_path, bbox_inches='tight')
        print(f"Residue distribution plot saved to {out_path}")
    except Exception:
        print(f"Could not save residu_distribution plot to {out_path}")
    plt.close(fig)


def _serialize_cycles(cycles: Dict[tuple, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for kcycle, v in cycles.items():
        out[json.dumps(list(kcycle))] = v
    return out


def merge_summaries(summaries: List[Dict[str, object]]) -> Dict[str, object]:
    if not summaries:
        return {}
    agg = {
        "start": summaries[0].get("start"),
        "end": summaries[-1].get("end"),
        "total": sum(s.get("total", 0) for s in summaries),
        "j": summaries[0].get("j"),
        "i": summaries[0].get("i"),
        "divisor": summaries[0].get("divisor"),
        "total_div_events": sum(s.get("total_div_events", 0) for s in summaries),
        "total_mul_events": sum(s.get("total_mul_events", 0) for s in summaries),
        "failures": [],
        "cycles": {},
    }
    for s in summaries:
        agg["failures"].extend(s.get("failures", []))

    cycles_map: Dict[str, Dict[str, object]] = {}
    for s in summaries:
        for kstr, v in s.get("cycles", {}).items():
            if kstr not in cycles_map:
                cycles_map[kstr] = dict(v)
                cycles_map[kstr]["power_counts"] = dict(v.get("power_counts", {}))
            else:
                existing = cycles_map[kstr]
                existing["div_total"] = existing.get("div_total", 0) + v.get("div_total", 0)
                existing["ge2_total"] = existing.get("ge2_total", 0) + v.get("ge2_total", 0)
                for e, cnt in v.get("power_counts", {}).items():
                    existing["power_counts"][e] = existing["power_counts"].get(e, 0) + cnt
                if "origins" in existing or "origins" in v:
                    existing_origins = existing.get("origins", [])
                    existing["origins"] = existing_origins + v.get("origins", [])
                else:
                    existing["num_origins"] = existing.get("num_origins", 0) + v.get("num_origins", 0)

    agg["cycles"] = cycles_map
    agg["distinct_cycles"] = len(cycles_map)
    try:
        agg["avg_stopping_time"] = None
        agg["median_stopping_time"] = None
        agg["max_stopping_time"] = max((s.get("max_stopping_time") or 0) for s in summaries) or None
        agg["max_peak"] = max((s.get("max_peak") or 0) for s in summaries) or None
        agg["mean_peak"] = None
    except Exception:
        agg["avg_stopping_time"] = None
        agg["median_stopping_time"] = None
        agg["max_stopping_time"] = None
        agg["max_peak"] = None
        agg["mean_peak"] = None

    return agg


def run_family_for_k(start: int, end: int, k_div: int, out: Optional[str] = None, compact: bool = False, alternated: bool = False, all_i: bool = False, alt_m: int = 1, max_iters: int = DEFAULT_MAX_ITERS, divergence_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD) -> Dict[int, Dict[str, object]]:
    per_i_results: Dict[int, Dict[int, Dict[str, object]]] = {}
    if all_i:
        max_i = k_div - 1
    else:
        max_i = (k_div - 1) // 2
    run_start_ts = _log_start(f"run_family_for_k k={k_div} start={start} end={end}")
    for i in range(1, max_i + 1):
        per_j_results: Dict[int, Dict[str, object]] = {}
        print(f"\n=== Running i={i} (k={k_div}) ===")
        for j in range(0, k_div):
            print(f"\n=== Running j={j} (k={k_div}, i={i}) ===")
            j_start_ts = _log_start(f"run_family_for_k k={k_div} i={i} j={j}")
            s = analyze_range(start, end, base=k_div, k=k_div, j_param=j, i_param=i, compact=compact, alternated=alternated, alt_m=alt_m, max_iters=max_iters, divergence_threshold=divergence_threshold)
            total_divs = sum(c.get("div_total", 0) for c in s["cycles"].values())
            total_ge2 = sum(c.get("ge2_total", 0) for c in s["cycles"].values())
            global_lambda = (total_ge2 / total_divs) if total_divs > 0 else None
            cycle_lambdas = [ (c.get("ge2_total",0)/c.get("div_total")) for c in s["cycles"].values() if c.get("div_total",0)>0 ]
            mean_cycle_lambda = (sum(cycle_lambdas) / len(cycle_lambdas)) if cycle_lambdas else None
            per_j_results[j] = {
                "summary": s,
                "global_lambda": global_lambda,
                "mean_cycle_lambda": mean_cycle_lambda,
                "num_cycles": s.get("distinct_cycles"),
                "max_cycle_len": s.get("max_cycle_len"),
                "num_failures": len(s.get("failures", [])),
                "sample_failures": s.get("failures", [])[:10],
                "avg_stopping_time": s.get("avg_stopping_time"),
                "median_stopping_time": s.get("median_stopping_time"),
                "max_stopping_time": s.get("max_stopping_time"),
                "max_peak": s.get("max_peak"),
                "mean_peak": s.get("mean_peak"),
            }
            try:
                elapsed_j = s.get("elapsed_sec", None)
                if elapsed_j is None:
                    elapsed_j = _log_end(f"run_family_for_k k={k_div} i={i} j={j}", j_start_ts)
                else:
                    _ = _log_end(f"run_family_for_k k={k_div} i={i} j={j}", j_start_ts)
                per_j_results[j]["elapsed_sec"] = elapsed_j
            except Exception:
                pass
        per_i_results[i] = per_j_results

        print(f"\nFinal table for i={i} (per j):")
        print(" j | global_lambda | mean_cycle_lambda | num_cycles | num_failures | max_cycle_len")
        print("---|---------------|-------------------|------------|--------------|---------------")
        for j in range(0, k_div):
            r = per_j_results[j]
            gl = f"{r['global_lambda']:.4f}" if r["global_lambda"] is not None else "N/A"
            ml = f"{r['mean_cycle_lambda']:.4f}" if r["mean_cycle_lambda"] is not None else "N/A"
            print(f" {j:2d} | {gl:13s} | {ml:17s} | {r['num_cycles']:10d} | {r['num_failures']:12d} | {r['max_cycle_len']:13d}")

    try:
        total_elapsed = _log_end(f"run_family_for_k k={k_div} start={start} end={end}", run_start_ts)
    except Exception:
        total_elapsed = None

    if out:
        def serialize_summary(s: Dict[str, object]) -> Dict[str, object]:
            outd = dict(s)
            cycles = s.get("cycles", {})
            serial_cycles = {}
            for kcycle, v in cycles.items():
                serial_cycles[json.dumps(list(kcycle))] = v
            outd["cycles"] = serial_cycles
            return outd

        serial = {"k": k_div, "start": start, "end": end, "per_i": {}}
        for i, per_j in per_i_results.items():
            serial["per_i"][str(i)] = {}
            for j, r in per_j.items():
                serial_r = {
                    "summary": serialize_summary(r["summary"]),
                    "global_lambda": r["global_lambda"],
                    "mean_cycle_lambda": r["mean_cycle_lambda"],
                    "num_cycles": r["num_cycles"],
                    "max_cycle_len": r["max_cycle_len"],
                    "num_failures": r["num_failures"],
                    "sample_failures": r["sample_failures"],
                    "avg_stopping_time": r.get("avg_stopping_time"),
                    "median_stopping_time": r.get("median_stopping_time"),
                    "max_stopping_time": r.get("max_stopping_time"),
                    "max_peak": r.get("max_peak"),
                    "mean_peak": r.get("mean_peak"),
                }
                serial["per_i"][str(i)][str(j)] = serial_r

        out_dir = os.path.dirname(out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False, indent=2)
        print(f"\nFamily summary saved to {out}")

    return per_i_results


def _sieve_primes(pmax: int) -> List[int]:
    sieve = [True] * (pmax + 1)
    sieve[0:2] = [False, False]
    for i in range(2, int(pmax ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, pmax + 1, i):
                sieve[j] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _simulate_single_n(n_val: int, k: int, j: int, i: int, max_iters: int, divergence_threshold: float, alternated: bool, alt_m: int) -> Dict[str, object]:
    seen: Dict[int, int] = {}
    seq: List[int] = []
    t = n_val
    preperiod = None
    reason = None
    for step in range(max_iters):
        if t in seen:
            preperiod = seen[t]
            reason = "cycle"
            break
        seen[t] = step
        seq.append(int(t))
        if abs(t) > divergence_threshold:
            reason = "divergence_threshold"
            break
        t = next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m)
    else:
        reason = "max_iters"
    peak = max(seq) if seq else int(n_val)
    return {
        "sequence": seq,
        "reason": reason,
        "steps": len(seq),
        "preperiod": preperiod,
        "peak": peak,
    }


def _plot_single_trajectory(seq: List[int], out_png: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not seq:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(range(len(seq))), seq, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel("value")
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
    except Exception:
        pass
    plt.close(fig)


def _plot_single_p_trajectories(run_dir: str, n_val: int, p_val: int, per_k: Dict[int, Dict[int, Dict[str, object]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ks = sorted(per_k.keys())
    n = len(ks)
    if n == 0:
        return
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
    for idx, k in enumerate(ks):
        ax = axes[idx // cols][idx % cols]
        ax.set_title(f"k={k}")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        for j, r in per_k[k].items():
            seq = r.get("sequence", [])
            if not seq:
                continue
            ax.plot(range(len(seq)), seq, linewidth=0.8, alpha=0.5)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"single-n trajectories (n={n_val})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        fig.savefig(os.path.join(run_dir, f"single_n_{n_val}_p{p_val}_trajectories.png"), dpi=150)
    except Exception:
        pass
    plt.close(fig)


def _plot_single_p_metric(run_dir: str, n_val: int, p_val: int, per_k: Dict[int, Dict[int, Dict[str, object]]], metric: str, suffix: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return
    ks = sorted(per_k.keys())
    n = len(ks)
    if n == 0:
        return
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
    for idx, k in enumerate(ks):
        ax = axes[idx // cols][idx % cols]
        ax.set_title(f"k={k}")
        ax.set_xlabel("j")
        ax.set_ylabel(metric)
        js = sorted(per_k[k].keys())
        vals = []
        for j in js:
            r = per_k[k][j]
            if r.get("reason") == "cycle":
                vals.append(r.get(metric))
            else:
                vals.append(np.nan)
        ax.plot(js, vals, marker="o", linestyle="-")
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"single-n {metric} per k (n={n_val})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        fig.savefig(os.path.join(run_dir, f"single_n_{n_val}_p{p_val}_{suffix}.png"), dpi=150)
    except Exception:
        pass
    plt.close(fig)


def _detect_output_feature_prefix(args: argparse.Namespace) -> str:
    """Choose an output directory prefix based on the active feature/mode."""
    if args.proof_persist:
        return "proof-persist"
    if args.proof:
        return "proof"
    if args.spirale_n is not None:
        return "spirale"

    s_n = args.single_overall_n if getattr(args, "single_overall_n", None) is not None else args.residu_single_overall_n
    if s_n is not None:
        return "single-overall"

    if args.residu_distribution_n is not None and args.residu_distribution_p is not None:
        return "residu-distribution"
    # divisions feature (new name) — support both new flags and legacy epsilon aliases
    if getattr(args, "divisions_n", None) is not None and getattr(args, "divisions_p", None) is not None:
        return "divisions"
    if getattr(args, "epsilon_n", None) is not None and getattr(args, "epsilon_p", None) is not None:
        return "divisions"
    if args.gamma_n is not None and args.gamma_p is not None:
        return "gamma"
    # shannon-entropy: dedicated prefix (requested)
    if args.shannon_entropy_n is not None and (args.shannon_entropy_k is not None or args.shannon_entropy_p is not None):
        return "shannon_entropy"
    if args.mixing_property_n is not None and (args.mixing_property_k is not None or args.mixing_property_p is not None):
        return "mixing_property"
    if getattr(args, "resistance_n", None) is not None and (
        getattr(args, "resistance_k", None) is not None or getattr(args, "resistance_p", None) is not None
    ):
        return "resistance"
    if args.residu_ecart_type_n is not None and args.residu_ecart_type_p is not None:
        return "residu-ecart-type"
    # NOTE: residu-lambda removed; use residu-distribution (A2) instead.

    if getattr(args, "cycle_n", None) is not None and (
        getattr(args, "cycle_k", None) is not None or getattr(args, "cycle_p", None) is not None
    ):
        return "cycle"

    # coalescence feature (compare trajectories of n and n+1)
    # support both single-k (--coalescence-k) and batch (--coalescence-p)
    if getattr(args, "coalescence_n", None) is not None and (
        getattr(args, "coalescence_k", None) is not None or getattr(args, "coalescence_p", None) is not None
    ):
        return "coalescence"

    if args.single_n is not None:
        return "single-n"
    if args.p is not None:
        return "family-p"
    if args.kmax is not None:
        return "family-kmax"
    if args.family:
        return "family"
    # footprint feature
    if getattr(args, "footprint_n", None) is not None or getattr(args, "footprint_n_multiple_k", None) is not None or getattr(args, "footprint_p", None) is not None:
        return "footprint"
    # altitude feature (peak of trajectories): use a dedicated prefix when requested
    if getattr(args, "altitude_n", None) is not None:
        return "altitude"
    # pearson feature: dedicated prefix when requested
    if getattr(args, "pearson_n", None) is not None:
        return "pearson"
    # dirichlet feature
    if getattr(args, "dirichlet_n", None) is not None:
        return "dirichlet"
    # hamming feature
    if getattr(args, "hamming_n", None) is not None:
        return "hamming"
    # lyapunov feature
    if getattr(args, "lyapunov_n", None) is not None:
        return "lyapunov"
    # stopping-time single run uses its own prefix
    if getattr(args, "stopping_n", None) is not None:
        return "stopping"
    # kernel feature: compute for all n<k
    if getattr(args, "kernel", False) or getattr(args, "kernel_k", None) is not None:
        return "kernel"
    return "run"


def _detect_output_suffix(args: argparse.Namespace) -> str:
    """Choose an informative output suffix, preferring feature-specific N/n values.

    This avoids generic endings like `_1000` (coming from default --end) when a
    feature uses its own input size, e.g. --proof-max-n or --spirale-n.
    """
    if args.proof or args.proof_persist:
        if args.proof_max_n is not None:
            return f"N{args.proof_max_n}"

    if args.spirale_n is not None:
        return f"n{args.spirale_n}"

    s_n = args.single_overall_n if getattr(args, "single_overall_n", None) is not None else args.residu_single_overall_n
    if s_n is not None:
        return f"n{s_n}"

    if args.single_n is not None:
        return f"n{args.single_n}"

    if args.residu_distribution_n is not None:
        return f"N{args.residu_distribution_n}"

    if args.mixing_property_n is not None:
        return f"n{args.mixing_property_n}"

    if getattr(args, "resistance_n", None) is not None:
        return f"n{getattr(args, 'resistance_n')}"

    if args.residu_ecart_type_single_n is not None:
        return f"n{args.residu_ecart_type_single_n}"
    if args.residu_ecart_type_n is not None:
        return f"N{args.residu_ecart_type_n}"

    # NOTE: residu-lambda removed; use residu-distribution instead.

    if getattr(args, "cycle_n", None) is not None:
        return f"N{getattr(args, 'cycle_n')}"

    if getattr(args, "kernel", False) or getattr(args, "kernel_k", None) is not None or getattr(args, "kernel_p", None) is not None:
        if getattr(args, "kernel_p", None) is not None:
            return f"p{int(getattr(args, 'kernel_p'))}"
        kk = getattr(args, "kernel_k", None)
        if kk is None:
            kk = args.k if args.k is not None else args.base
        if kk is not None:
            return f"k{int(kk)}"

    # Prefer footprint suffix if footprint args provided
    if getattr(args, "footprint_n", None) is not None:
        return f"N{getattr(args, 'footprint_n')}"
    if getattr(args, "footprint_n_multiple_k", None) is not None:
        return f"nmult{getattr(args, 'footprint_n_multiple_k')}"
    if getattr(args, "footprint_p", None) is not None:
        return f"p{getattr(args, 'footprint_p')}"

    return f"{args.start}_{args.end}"


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Analyse générale du système (paramètre base)", formatter_class=argparse.RawDescriptionHelpFormatter)

    general_grp = parser.add_argument_group('General options')
    family_grp = parser.add_argument_group('Family / k selection')
    alternation_grp = parser.add_argument_group('Alternation / variant')
    parallel_grp = parser.add_argument_group('Parallel / performance')
    limits_grp = parser.add_argument_group('Iteration limits')
    single_grp = parser.add_argument_group('Single-n diagnostics')
    proof_grp = parser.add_argument_group('Proof modes')
    residu_grp = parser.add_argument_group('Residue features')
    output_grp = parser.add_argument_group('Output / serialization')
    datalake_grp = parser.add_argument_group('Data lake (export / resume)')

    general_grp.add_argument("--start", type=int, default=1, help="valeur de départ (incluse)")
    general_grp.add_argument("--end", type=int, default=1000, help="valeur de fin (incluse)")
    general_grp.add_argument("--out", type=str, default=None, help="fichier JSON de sortie (optionnel)")
    general_grp.add_argument("--base", type=int, default=3, help="diviseur à utiliser (par défaut 3). Par ex. 5 pour le système demandé.")
    general_grp.add_argument("--k", type=int, default=None, help="diviseur k (utilisé pour la famille j=0..k-1). Si omis, on utilisera --base comme k par défaut.")

    family_grp.add_argument("--family", action="store_true", help="si vrai, exécute la famille de règles pour j=0..k-1 et affiche un tableau par j")
    family_grp.add_argument("--j", type=int, default=None, help="paramètre j à utiliser (0..k-1). Si omis, utilise le mode simple/plain/plus.")
    family_grp.add_argument("--i", type=int, default=1, help="paramètre i à utiliser pour les runs non-famille (1..k-1)")
    family_grp.add_argument("--all-i", action="store_true", help="si vrai, considère tous les i dans 1..k-1 au lieu de i < k/2")
    family_grp.add_argument("--p", type=int, default=None, help="si fourni, exécute la famille pour tous les k premiers <= p (chaque k produit un JSON results_family_k{K}_{start}_{end}.json)")
    family_grp.add_argument("--kmax", type=int, default=None, help="si fourni, exécute la famille pour tous les k = 2..kmax (chaque k produit un JSON results_family_k{K}_{start}_{end}.json)")

    alternation_grp.add_argument("--alternated", action="store_true", help="active la variante alternée (k + i*(-1)^n) dans la formule de récursion")
    alternation_grp.add_argument("--alt-m", type=int, default=1, help="paramètre m pour la variante alternée (remplace -1 par -m). Doit être < k. Par défaut 1 (comportement original)")

    parallel_grp.add_argument("--workers", type=int, default=4, help="nombre de processus travailleurs pour l'analyse du range (par défaut 4)")
    parallel_grp.add_argument("--chunk-size", type=int, default=100000, help="taille des chunks pour découper l'intervalle (par défaut 100000)")
    parallel_grp.add_argument("--use-gmpy", action="store_true", help="utiliser gmpy2/GMP pour l'arithmétique sur grands entiers si installé")
    parallel_grp.add_argument("--use-numba", action="store_true", help="utiliser numba JIT pour la voie int64 rapide (si installé)")

    limits_grp.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS, help="nombre maximal d'itérations par trajectoire")
    limits_grp.add_argument("--divergence-threshold", type=float, default=DEFAULT_DIVERGENCE_THRESHOLD, help="seuil de divergence numérique (par défaut 1e18)")

    single_grp.add_argument("--single-n", type=int, default=None, help="exécuter le calcul pour une seule valeur n donnée (mode unitaire)")
    single_grp.add_argument("--single-k", type=int, default=None, help="diviseur k à utiliser pour --single-n (par défaut reprend --k/base)")
    single_grp.add_argument("--single-i", type=int, default=None, help="paramètre i à utiliser pour --single-n (doit être fourni)")
    single_grp.add_argument("--single-j", type=int, default=None, help="paramètre j à utiliser pour --single-n (doit être fourni)")
    single_grp.add_argument("--single-p", type=int, default=None, help="si fourni avec --single-n, exécute le diagnostic pour tous les k premiers <= p (et pour tous i,j)")
    single_grp.add_argument("--stopping-n", type=int, default=None, help="valeur n pour exécuter le stopping-time (single mode)")
    single_grp.add_argument("--stopping-k", type=int, default=None, help="diviseur k (premier) pour --stopping-n")
    single_grp.add_argument("--stopping-i", type=int, default=None, help="paramètre i pour --stopping-n (doit être fourni)")
    single_grp.add_argument("--stopping-j", type=int, default=None, help="paramètre j pour --stopping-n (doit être fourni)")
    single_grp.add_argument("--stopping-p", type=int, default=None, help="valeur p: exécute le stopping-time pour tous les k premiers <= p (requiert --stopping-n et --stopping-i/--stopping-j)")
    single_grp.add_argument("--stopping-all-j", dest="stopping_all_j", action="store_true", help="si vrai avec --stopping-p: boucle sur tous les j=0..k-1 pour chaque k")

    proof_grp.add_argument("--proof", action="store_true", help="activer la vérification rapide de convergence pour tous les k premiers <= p jusqu'à --proof-max-n")
    proof_grp.add_argument("--proof-p", type=int, default=None, help="valeur p: tester tous les k premiers <= p (nécessaire avec --proof)")
    proof_grp.add_argument("--proof-all", action="store_true", help="avec --proof-p: teste tous les k <= p (premiers et non premiers)")
    proof_grp.add_argument("--proof-max-n", type=int, default=None, help="preuve de convergence pour tous les n <= PROOF_MAX_N (nécessaire avec --proof)")
    proof_grp.add_argument("--proof-persist", action="store_true", help="mode proof persistant: sauvegarde max_proved par combinaison (utilise --workers)")
    proof_grp.add_argument(
        "--proof-lake",
        action="store_true",
        help=(
            "(avec --proof-persist) active le mode 'lake': conserve un ensemble persistant de points visités "
            "par les trajectoires des n déjà prouvés (<= max_proved) et considère qu'un n converge si sa "
            "trajectoire coalesce avec ce lake."
        ),
    )
    proof_grp.add_argument("--proof-j-mult", type=int, default=1, help="multiplie la borne supérieure de j pour les modes --proof/--proof-persist: j range sera 0..(proof_j_mult*k-1). Par défaut 1 (j dans 0..k-1)")
    proof_grp.add_argument("--proof-i", type=int, default=None, help="(optionnel) si fourni, calcule la preuve uniquement pour cette valeur de i (au lieu de boucler sur i)")
    proof_grp.add_argument("--plot-proof", action="store_true", help="génère une image de heatmaps (par k) pour les résultats du mode proof/proof-persist")
    proof_grp.add_argument("--proof-k", type=int, default=None, help="(optionnel) si fourni, exécute la preuve seulement pour ce k premier au lieu de --proof-p")

    residu_grp.add_argument("--residu-distribution-n", type=int, default=None, help="valeur n pour calculer la distribution des résidus (utilisé avec --residu-distribution-p)")
    residu_grp.add_argument("--residu-distribution-p", type=int, default=None, help="si fourni avec --residu-distribution-n, exécute la distribution des résidus pour tous les k premiers <= p (i est fixé à 1, j dans 0..2k-1)")
    residu_grp.add_argument("--residu-distribution-i", type=int, default=1, help="paramètre i à utiliser pour la distribution des résidus (par défaut 1)")
    residu_grp.add_argument("--residu-distribution-j", type=int, default=None, help="(optionnel) si fourni, calcule la distribution des résidus uniquement pour ce j (au lieu de boucler sur j dans 0..j_multiple*k-1)")
    residu_grp.add_argument(
        "--residu-distribution-j-mult",
        dest="residu_distribution_j_mult",
        type=int,
        default=2,
        help="multiplieur m pour la plage de j dans residu-distribution: j dans 0..(m*k-1) quand --residu-distribution-j n'est pas fourni (par défaut 2)",
    )
    residu_grp.add_argument("--residu-distribution-include-zero", action="store_true", default=False, help="si présent, calcule aussi la moyenne des résidus en incluant les zéros (mean of all residues)")
    residu_grp.add_argument("--residu-distribution-all-j", dest="residu_distribution_all_j", action="store_true", help="si présent avec --residu-distribution-p: boucle sur tous les j=0..k-1 pour chaque k (remplace --residu-distribution-j et --residu-j-multiple)")
    residu_grp.add_argument("--residu-distribution-all-n", dest="residu_distribution_all_n", action="store_true", help="si présent avec --residu-distribution-p: boucle sur tous les nombres n0=1..n et affiche la moyenne agrégée des métriques pour tout n0<=n")
    residu_grp.add_argument("--spirale-n", type=int, default=None, help="valeur n pour générer une spirale (avec --spirale-k)")
    residu_grp.add_argument("--spirale-k", type=int, default=None, help="diviseur k (doit être premier) pour --spirale-n")
    residu_grp.add_argument("--spirale-p", type=int, default=None, help="valeur p: exécute la spirale pour tous les k premiers <= p (utiliser avec --spirale-n)")
    residu_grp.add_argument("--spirale-all", action="store_true", help="avec --spirale-p: exécute la spirale pour tous les k <= p (premiers et non premiers)")
    residu_grp.add_argument("--spirale-i", type=int, default=1, help="paramètre i à utiliser pour la spirale (par défaut 1)")
    residu_grp.add_argument("--spirale-j", type=int, default=0, help="paramètre j à utiliser pour la spirale (par défaut 0)")
    residu_grp.add_argument("--spirale-angle-mode", type=str, default="residue", help="mode d'angle pour la spirale: 'residue' (angle=2π*(t mod k)/k) or 'step' (angle = step*(2π/k))")
    residu_grp.add_argument("--residu-single-overall-n", type=int, default=None, help="valeur n pour exécuter le mode single overall (avec --residu-single-overall-k/i/j)")
    residu_grp.add_argument("--residu-single-overall-k", type=int, default=None, help="diviseur k pour --residu-single-overall-n")
    residu_grp.add_argument("--residu-single-overall-i", type=int, default=None, help="paramètre i pour --residu-single-overall-n")
    residu_grp.add_argument("--residu-single-overall-j", type=int, default=None, help="paramètre j pour --residu-single-overall-n")
    # New, shorter flags for the single-overall feature (preferred)
    residu_grp.add_argument("--single-overall-n", dest="single_overall_n", type=int, default=None, help="valeur n pour exécuter le mode single overall (avec --single-overall-k/i/j)")
    residu_grp.add_argument("--single-overall-k", dest="single_overall_k", type=int, default=None, help="diviseur k pour --single-overall-n")
    residu_grp.add_argument("--single-overall-i", dest="single_overall_i", type=int, default=None, help="paramètre i pour --single-overall-n")
    residu_grp.add_argument("--single-overall-j", dest="single_overall_j", type=int, default=None, help="paramètre j pour --single-overall-n")
    residu_grp.add_argument("--residu-j-multiple", type=int, default=2, help="multiplicateur pour la borne supérieure de j dans certaines features legacy (conservé pour compatibilité). Par défaut 2")
    residu_grp.add_argument("--residu-ecart-n", type=int, default=None, help="valeur N: borne supérieure (incluse) pour les valeurs de départ n=1..N à utiliser dans le calcul de l'écart-type des résidus")
    residu_grp.add_argument("--residu-ecart-p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour le calcul de l'écart-type des résidus")
    residu_grp.add_argument("--residu-ecart-type-n", type=int, default=None, help="valeur N: pour calculer l'écart-type des résidus en agrégeant les trajectoires de tous les n=1..N")
    residu_grp.add_argument("--residu-ecart-type-p", type=int, default=None, help="valeur p: calculer l'écart-type pour tous les k premiers <= p (nécessaire avec --residu-ecart-type-n)")
    residu_grp.add_argument("--residu-ecart-type-single-n", type=int, default=None, help="si fourni, calcule l'écart-type uniquement pour ce n donné (ignore --residu-ecart-type-n range)")
    residu_grp.add_argument("--residu-ecart-type-i", type=int, default=1, help="paramètre i à utiliser pour le calcul de l'écart-type (par défaut 1)")
    # Coalescence feature
    residu_grp.add_argument("--coalescence-n", type=int, default=None, help="valeur n pour exécuter la coalescence (compare trajectoire de n et n+1)")
    residu_grp.add_argument("--coalescence-k", type=int, default=None, help="diviseur k (premier) pour --coalescence-n")

    # kernel feature: compute the usual run summary for all starting values 1..(2k-1)
    residu_grp.add_argument("--kernel", action="store_true", help="active la feature kernel: calcule pour tous les n avec 1 <= n <= 2*k-1")
    residu_grp.add_argument("--kernel-k", type=int, default=None, help="diviseur k (premier) pour la feature kernel (par défaut reprend --k/base)")
    residu_grp.add_argument("--kernel-p", type=int, default=None, help="valeur p: exécute kernel pour tous les k premiers <= p")
    residu_grp.add_argument("--kernel-i", type=int, default=None, help="paramètre i pour la feature kernel (par défaut reprend --i)")
    residu_grp.add_argument("--kernel-j", type=int, default=None, help="paramètre j pour la feature kernel (par défaut reprend --j ou 0)")
    residu_grp.add_argument("--coalescence-i", type=int, default=1, help="paramètre i pour la coalescence (par défaut 1)")
    residu_grp.add_argument("--coalescence-j", type=int, default=0, help="paramètre j pour la coalescence (par défaut 0)")
    residu_grp.add_argument("--coalescence-j-multi", dest="coalescence_j_multi", type=int, default=1, help="si >1, boucle sur j=0..(k*j_multi-1) pour coalescence (par défaut 1 => j unique)")
    residu_grp.add_argument("--coalescence-verbose", dest="coalescence_verbose", action="store_true", help="si présent, écrit aussi les CSV/PNG détaillés par (k,i,j). Par défaut, mode moins verbeux en batch.")
    residu_grp.add_argument("--coalescence-p", type=int, default=None, help="valeur p: exécute coalescence pour tous les k premiers <= p (utilise --coalescence-n as upper n and loops k primes)")
    residu_grp.add_argument("--k-adique-n", type=int, default=None, help="valeur n pour exécuter la distribution k-adique (avec --k-adique-k/i/j)")
    residu_grp.add_argument("--k-adique-k", type=int, default=None, help="diviseur k pour --k-adique-n (doit être premier)")
    residu_grp.add_argument("--k-adique-i", type=int, default=1, help="paramètre i pour --k-adique-n (par défaut 1)")
    residu_grp.add_argument("--k-adique-j", type=int, default=0, help="paramètre j pour --k-adique-n (par défaut 0)")

    # lyapunov feature
    residu_grp.add_argument("--lyapunov-n", type=int, default=None, help="valeur n pour exécuter le calcul Lyapunov (single trajectory)")
    residu_grp.add_argument("--lyapunov-k", type=int, default=None, help="diviseur k (premier) pour --lyapunov-n")
    residu_grp.add_argument("--lyapunov-p", type=int, default=None, help="valeur p: exécute lyapunov pour tous les k premiers <= p (requiert --lyapunov-n)")
    residu_grp.add_argument("--lyapunov-i", type=int, default=None, help="paramètre i pour --lyapunov-n (doit être fourni)")
    residu_grp.add_argument("--lyapunov-j", type=int, default=None, help="paramètre j pour --lyapunov-n (doit être fourni)")

    # cycle length feature (bulk over all start values 1..N)
    residu_grp.add_argument("--cycle-n", type=int, default=None, help="valeur N: calcule la taille du cycle pour tous les n=1..N (avec --cycle-k/i/j)")
    residu_grp.add_argument("--cycle-k", type=int, default=None, help="diviseur k (premier) pour --cycle-n")
    residu_grp.add_argument("--cycle-p", type=int, default=None, help="valeur p: boucle sur tous les k premiers <= p (avec --cycle-n/i/j) et génère une seule figure avec un subplot par k")
    residu_grp.add_argument("--cycle-i", type=int, default=1, help="paramètre i pour --cycle-n (par défaut 1)")
    residu_grp.add_argument("--cycle-j", type=int, default=0, help="paramètre j pour --cycle-n (par défaut 0)")
    residu_grp.add_argument("--cycle-all-j", action="store_true", help="si vrai, boucle sur tous les j=0..k-1; calcule des moyennes par (k,j) et génère des subplots")
    residu_grp.add_argument("--cycle-cardinality", action="store_true", help="si vrai, compte la cardinalité (nb d'occurrences) de chaque cycle canonique détecté et écrit un CSV/JSON de résumé")
    residu_grp.add_argument("--special-cycles", action="store_true", help="si vrai, détecte et affiche (k,i,j) pour lesquels tous les n=1..N ont la même taille de cycle")
    residu_grp.add_argument("--extra-special-cycles", action="store_true", help="si vrai, détecte et affiche (k,i,j) pour lesquels tous les n=1..N partagent exactement le même cycle canonique (pas seulement la même longueur)")
    residu_grp.add_argument("--fst-appearance", action="store_true", help="(cycle) si vrai, détecte pour chaque cycle canonique sa première apparition et le n0 correspondant")
    residu_grp.add_argument("--cycle-j-multiple", "--cycle-j-multi", dest="cycle_j_multiple", type=int, default=1, help="étend la plage de j à 0..(jm*k-1) quand utilisée avec --cycle-all-j (par défaut 1). Alias: --cycle-j-multi")
    residu_grp.add_argument("--card-top-cycles", type=int, default=5, help="nombre de cycles à tracer par k dans le graphe de cardinalités par j (<=0 pour afficher tous les cycles). Par défaut 5")
    # footprint feature args 
    residu_grp.add_argument("--footprint-n", dest="footprint_n", type=int, default=None, help="valeur N: exécute la feature footprint pour tous les n=1..N")
    residu_grp.add_argument("--footprint-k", dest="footprint_k", type=int, default=None, help="diviseur k (premier) pour la feature footprint")
    residu_grp.add_argument("--footprint-p", dest="footprint_p", type=int, default=None, help="valeur p: exécute la feature footprint pour tous les k premiers <= p (utiliser avec --footprint-n)")
    residu_grp.add_argument("--footprint-i", dest="footprint_i", type=int, default=1, help="paramètre i pour la feature footprint (par défaut 1)")
    residu_grp.add_argument("--footprint-j", dest="footprint_j", type=int, default=0, help="paramètre j pour la feature footprint (par défaut 0)")
    residu_grp.add_argument(
        "--footprint-n-multiple-k",
        dest="footprint_n_multiple_k",
        type=int,
        default=None,
        help="si fourni, calcule pour tous les n <= footprint_n_multiple_k * k (override N)",
    )
    residu_grp.add_argument(
        "--footprint-j-multi",
        dest="footprint_j_multi",
        type=int,
        default=1,
        help="si >1, boucle sur j=0..(k*j_multi-1) et superpose les résultats sur un même subplot par k. Par défaut 1 (utilise --footprint-j)",
    )
    residu_grp.add_argument(
        "--footprint-prefixes",
        dest="footprint_prefixes",
        action="store_true",
        help="si présent, calcule footprint pour tous les préfixes 1..N (i.e. N'=1..N) au lieu d'un seul N; produit un JSON agrégé et une figure (subplot par k)",
    )
    residu_grp.add_argument(
        "--footprint-compact",
        dest="footprint_compact",
        action="store_true",
        help="si présent avec footprint (surtout --footprint-prefixes), n'écrit pas les fichiers détaillés par N (visited/summary/origins) et produit seulement un résumé JSON agrégé (+ PNG éventuels)",
    )
    residu_grp.add_argument(
        "--footprint-verbose",
        dest="footprint_verbose",
        action="store_true",
        help="si présent, force l'écriture des fichiers détaillés footprint (visited/summary/origins). Par défaut en mode --footprint-prefixes on évite ces fichiers pour réduire la verbosité.",
    )
    residu_grp.add_argument(
        "--footprint-check-parity",
        dest="footprint_check_parity",
        action="store_true",
        help=(
            "si présent, vérifie une propriété de parité sur S(N) et enregistre les résultats dans le JSON résumé: "
            "pour k=2: S(N)=N+1 si N pair sinon S(N)=N; pour k>=3: S(N)=N si N pair sinon S(N)=N+1"
        ),
    )

    residu_grp.add_argument(
        "--footprint-n-delta",
        dest="footprint_n_delta",
        action="store_true",
        help=(
            "si présent (surtout avec --footprint-prefixes), calcule la distribution de Δ(N)=|S(N)-N|: "
            "cardinalités et pourcentages par valeur de Δ, et l'inclut dans le JSON résumé"
        ),
    )

    residu_grp.add_argument(
        "--footprint-total",
        dest="footprint_total",
        action="store_true",
        help=(
            "si présent (avec --footprint-prefixes), enregistre aussi la série total_unique_visited(N) = "
            "nombre d'entiers distincts visités par l'union des trajectoires de départ 1..N"
        ),
    )
    # divisions flags (new name) — kept alongside epsilon aliases for backward compatibility
    residu_grp.add_argument("--divisions-n", dest="divisions_n", type=int, default=None, help="valeur n pour exécuter le calcul divisions (alias --epsilon-n)")
    residu_grp.add_argument("--divisions-p", dest="divisions_p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour le calcul divisions (alias --epsilon-p)")
    residu_grp.add_argument("--divisions-i", dest="divisions_i", type=int, default=1, help="paramètre i pour le calcul divisions (par défaut 1, alias --epsilon-i)")
    residu_grp.add_argument("--divisions-j", dest="divisions_j", type=int, default=None, help="paramètre j pour le calcul divisions (par défaut None, alias --epsilon-j)")
    residu_grp.add_argument("--divisions-find-best-j", dest="divisions_find_best_j", action="store_true", help="si vrai, cherche pour chaque k le j (0..k-1) qui minimise |epsilonv*100/R| et produit un tableau (k,j,ratio) et un graphe (alias --epsilon-find-best-j)")
    residu_grp.add_argument("--divisions-ordre-multiplicatif-j", dest="divisions_ordre_multiplicatif_j", action="store_true", help="si vrai, calcule pour chaque k et chaque j in 0..k-1 l'ordre multiplicatif de j+1 mod k et inclut le j qui a l'ordre maximum dans le tableau de sortie (alias --epsilon-ordre-multiplicatif-j)")
    residu_grp.add_argument("--divisions-table", dest="divisions_table", action="store_true", help="si vrai, génère un tableau CSV détaillé (k, v, count_ge_m for m=2..v, count_ge1) et l'affiche en console (alias --epsilon-table)")
    residu_grp.add_argument("--divisions-j-multi", dest="divisions_j_multi", type=int, default=1, help="multiplieur m pour la plage de j: j dans 0..(m*k-1) quand --divisions-find-best-j n'est pas fourni (par défaut 1) (alias --epsilon-j-multi)")
    residu_grp.add_argument("--divisions-all-n", dest="divisions_all_n", action="store_true", help="si présent avec --divisions-p: boucle sur tous les n0=1..n (où n provient de --divisions-n) et écrit les résultats agrégés (alias --epsilon-all-n)")

    # epsilon aliases (kept for compatibility)
    residu_grp.add_argument("--epsilon-n", dest="epsilon_n", type=int, default=None, help=argparse.SUPPRESS if False else "valeur n pour exécuter le calcul d'epsilon (deprecated; use --divisions-n)")
    residu_grp.add_argument("--epsilon-p", dest="epsilon_p", type=int, default=None, help=argparse.SUPPRESS if False else "valeur p: considérer tous les k premiers <= p pour le calcul d'epsilon (deprecated; use --divisions-p)")
    residu_grp.add_argument("--epsilon-i", dest="epsilon_i", type=int, default=1, help=argparse.SUPPRESS if False else "paramètre i pour le calcul d'epsilon (par défaut 1) (deprecated; use --divisions-i)")
    residu_grp.add_argument("--epsilon-j", dest="epsilon_j", type=int, default=None, help=argparse.SUPPRESS if False else "paramètre j pour le calcul d'epsilon (par défaut 0) (deprecated; use --divisions-j)")
    residu_grp.add_argument("--epsilon-find-best-j", dest="epsilon_find_best_j", action="store_true", help=argparse.SUPPRESS if False else "deprecated; use --divisions-find-best-j")
    residu_grp.add_argument("--epsilon-ordre-multiplicatif-j", dest="epsilon_ordre_multiplicatif_j", action="store_true", help=argparse.SUPPRESS if False else "deprecated; use --divisions-ordre-multiplicatif-j")
    residu_grp.add_argument("--epsilon-table", dest="epsilon_table", action="store_true", help=argparse.SUPPRESS if False else "deprecated; use --divisions-table")
    residu_grp.add_argument("--epsilon-j-multi", dest="epsilon_j_multi", type=int, default=1, help=argparse.SUPPRESS if False else "deprecated; use --divisions-j-multi")
    residu_grp.add_argument("--epsilon-all-n", dest="epsilon_all_n", action="store_true", help=argparse.SUPPRESS if False else "deprecated; use --divisions-all-n")

    # trajectory-minimum feature
    residu_grp.add_argument("--trajectory-minimum-n", type=int, default=None, help="valeur n: exécute la feature trajectory-minimum pour un seul n (avec --trajectory-minimum-k/i/j)")
    residu_grp.add_argument("--trajectory-minimum-k", type=int, default=None, help="diviseur k (premier) pour --trajectory-minimum-n")
    residu_grp.add_argument("--trajectory-minimum-p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour la feature trajectory-minimum (alternative à --trajectory-minimum-k)")
    residu_grp.add_argument("--trajectory-minimum-i", type=int, default=1, help="paramètre i pour --trajectory-minimum-n (par défaut 1)")
    residu_grp.add_argument("--trajectory-minimum-j", type=int, default=0, help="paramètre j pour --trajectory-minimum-n (par défaut 0)")
    residu_grp.add_argument("--trajectory-minimum-j-multi", type=int, default=None, help="quand donné, boucle sur tous les j=0..k*j_multi-1 pour la feature trajectory-minimum")
    residu_grp.add_argument("--trajectory-minimum-all", dest="trajectory_minimum_all", action="store_true", help="si vrai, exécute la feature trajectory-minimum pour tous les n=1..N (où N provient de --trajectory-minimum-n)")
    residu_grp.add_argument("--trajectory-minimum-single", dest="trajectory_minimum_single", action="store_true", help="si présent, exécute la feature trajectory-minimum seulement pour le n fourni (par défaut, on exécute pour tous les n=1..N)")

    # gamma feature
    residu_grp.add_argument("--gamma-n", type=int, default=None, help="valeur n pour exécuter le calcul de gamma (avec --gamma-p/i/j)")
    residu_grp.add_argument("--gamma-p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour le calcul de gamma")
    residu_grp.add_argument("--gamma-i", type=int, default=1, help="paramètre i pour le calcul de gamma (par défaut 1)")
    residu_grp.add_argument("--gamma-j", type=int, default=0, help="paramètre j pour le calcul de gamma (par défaut 0)")
    residu_grp.add_argument("--gamma-all-i", action="store_true", help="si vrai, pour chaque k on calcule gamma pour tous les i avec 0 < i < k et on génère un graphe gamma(i) par k")
    residu_grp.add_argument("--gamma-all-j", action="store_true", help="si vrai, pour chaque k on calcule gamma pour tous les j avec 0 <= j < k et on génère un graphe gamma(j) par k")
    residu_grp.add_argument("--plot-gamma", action="store_true", help="si vrai, génère un PNG pour gamma en fonction de k")

    # shannon entropy feature (standalone)
    residu_grp.add_argument("--shannon-entropy-n", type=int, default=None, help="valeur n pour calculer l'entropie de Shannon des résidus non nuls (avec --shannon-entropy-k/i/j)")
    residu_grp.add_argument("--shannon-entropy-k", type=int, default=None, help="diviseur k (premier) pour --shannon-entropy-n")
    residu_grp.add_argument("--shannon-entropy-p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour --shannon-entropy-n (alternative à --shannon-entropy-k)")
    residu_grp.add_argument("--shannon-entropy-i", type=int, default=1, help="paramètre i pour --shannon-entropy-n (par défaut 1)")
    residu_grp.add_argument("--shannon-entropy-j", type=int, default=0, help="paramètre j pour --shannon-entropy-n (par défaut 0)")
    residu_grp.add_argument("--shannon-entropy-all-j", action="store_true", help="si vrai (avec --shannon-entropy-p), boucle sur tous les j=0..k-1 et trace min/max/mean dans les schémas")

    # mixing-property feature (lag plot of residues)
    residu_grp.add_argument("--mixing-property-n", type=int, default=None, help="valeur n pour générer le lag-plot des résidus (mixing property) (avec --mixing-property-k/i/j)")
    residu_grp.add_argument("--mixing-property-k", type=int, default=None, help="diviseur k (premier) pour --mixing-property-n")
    residu_grp.add_argument("--mixing-property-p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour --mixing-property-n (alternative à --mixing-property-k)")
    residu_grp.add_argument("--mixing-property-i", type=int, default=1, help="paramètre i pour --mixing-property-n (par défaut 1)")
    residu_grp.add_argument("--mixing-property-j", type=int, default=0, help="paramètre j pour --mixing-property-n (par défaut 0)")
    residu_grp.add_argument("--mixing-property-all-j", action="store_true", help="si vrai, boucle sur tous les j=0..k-1 et superpose les nuages dans une seule image")
    residu_grp.add_argument("--mixing-property-lag", type=int, default=1, help="lag ℓ pour le lag-plot: scatter (r_t, r_{t+ℓ}). Par défaut 1")
    residu_grp.add_argument("--mixing-property-max-points", type=int, default=20000, help="nombre maximal de points affichés dans le scatter (par défaut 20000)")

    # resistance feature (alternating M->D->M.. prefix until D->D)
    residu_grp.add_argument("--resistance-n", dest="resistance_n", type=int, default=None, help="valeur n pour calculer la résistance (avec --resistance-k ou --resistance-p, et --resistance-i/j)")
    residu_grp.add_argument("--resistance-k", dest="resistance_k", type=int, default=None, help="diviseur k (premier) pour --resistance-n")
    residu_grp.add_argument("--resistance-p", dest="resistance_p", type=int, default=None, help="valeur p: considérer tous les k premiers <= p pour --resistance-n (alternative à --resistance-k)")
    residu_grp.add_argument("--resistance-i", dest="resistance_i", type=int, default=1, help="paramètre i pour --resistance-n (par défaut 1)")
    residu_grp.add_argument("--resistance-j", dest="resistance_j", type=int, default=0, help="paramètre j pour --resistance-n (par défaut 0)")
    residu_grp.add_argument("--resistance-all-j", dest="resistance_all_j", action="store_true", help="si vrai, boucle sur tous les j=0..k-1 et affiche resistance(j) (un subplot par k)")
    residu_grp.add_argument("--resistance-all-n", dest="resistance_all_n", action="store_true", help="si vrai, boucle sur tous les n=1..N où N est --resistance-n (avec --resistance-p)")
    # NOTE: residu-lambda removed; A2/A_v moved into --residu-distribution.
    # altitude feature (peak of trajectories)
    residu_grp.add_argument("--altitude-n", type=int, default=None, help="valeur n: calcule le pic (peak) pour tous les n'=1..n")
    residu_grp.add_argument("--altitude-k", type=int, default=None, help="diviseur k (premier) pour --altitude-n")
    residu_grp.add_argument("--altitude-p", type=int, default=None, help="valeur p: exécute altitude pour tous les k premiers <= p (requiert --altitude-n)")
    residu_grp.add_argument("--altitude-i", type=int, default=1, help="paramètre i pour --altitude-n (par défaut 1)")
    residu_grp.add_argument("--altitude-j", type=int, default=0, help="paramètre j pour --altitude-n (par défaut 0)")
    residu_grp.add_argument("--altitude-partitionning", action="store_true", help="si vrai, partitionne les n' par n' %% k et calcule peak/distance séparément par partition (k sous-groupes)")
    # pearson feature (new): Pearson correlation of successive residue vectors along a trajectory
    residu_grp.add_argument("--pearson-n", type=int, default=None, help="valeur n pour calculer le coefficient de Pearson (pour tous les n'=1..N)")
    residu_grp.add_argument("--pearson-k", type=int, default=None, help="diviseur k (premier) pour --pearson-n")
    residu_grp.add_argument("--pearson-p", type=int, default=None, help="valeur p: exécute pearson pour tous les k premiers <= p (requiert --pearson-n)")
    residu_grp.add_argument("--pearson-i", type=int, default=1, help="paramètre i pour --pearson-n (par défaut 1)")
    residu_grp.add_argument("--pearson-j", type=int, default=0, help="paramètre j pour --pearson-n (par défaut 0)")
    residu_grp.add_argument("--pearson-all-j", action="store_true", help="si vrai (avec --pearson-p): boucle sur tous les j=0..k-1 pour chaque k")
    # dirichlet feature: Pearson between residue sequences of n and n+1
    residu_grp.add_argument("--dirichlet-n", type=int, default=None, help="valeur n pour exécuter la feature dirichlet (pour tous les n'=1..N)")
    residu_grp.add_argument("--dirichlet-k", type=int, default=None, help="diviseur k (premier) pour --dirichlet-n")
    residu_grp.add_argument("--dirichlet-p", type=int, default=None, help="valeur p: exécute dirichlet pour tous les k premiers <= p (requiert --dirichlet-n)")
    residu_grp.add_argument("--dirichlet-i", type=int, default=1, help="paramètre i pour --dirichlet-n (par défaut 1)")
    residu_grp.add_argument("--dirichlet-j", type=int, default=0, help="paramètre j pour --dirichlet-n (par défaut 0)")
    residu_grp.add_argument("--dirichlet-plot-3d", action="store_true", help="si vrai, génère des PNG 3D en spirale des restes pour un n0 échantillon (n0 et n0+1)")
    # hamming feature (distance de Hamming entre restes de n et n+1)
    residu_grp.add_argument("--hamming-n", type=int, default=None, help="valeur n pour exécuter la feature hamming (analyse pour n et n+1)")
    residu_grp.add_argument("--hamming-k", type=int, default=None, help="diviseur k (premier) pour --hamming-n")
    residu_grp.add_argument("--hamming-i", type=int, default=1, help="paramètre i pour --hamming-n (par défaut 1)")
    residu_grp.add_argument("--hamming-j", type=int, default=0, help="paramètre j pour --hamming-n (par défaut 0)")
    residu_grp.add_argument("--hamming-all-j", dest="hamming_all_j", action="store_true", help="si vrai, boucle sur tous les j=0..k-1 pour chaque k (génère une image par k avec un subplot par j)")
    residu_grp.add_argument("--hamming-p", type=int, default=None, help="valeur p: exécute hamming pour tous les k premiers <= p (requiert --hamming-n)")
    # Note: all features use the global --workers flag for parallelism.

    output_grp.add_argument("--compact-json", action="store_true", help="si vrai, n'enregistre pas les listes d'origines dans les JSON (garde uniquement des agrégats) pour économiser de l'espace disque")

    # datalake feature (new)
    datalake_grp.add_argument(
        "--datalake-path",
        dest="datalake_path",
        type=str,
        default=None,
        help="chemin racine pour écrire les fichiers datalake (arborescence k{k}/i{i}/chunk_...)",
    )
    datalake_grp.add_argument("--datalake-k", dest="datalake_k", type=int, default=None, help="k (premier) pour la feature datalake")
    datalake_grp.add_argument(
        "--datalake-p",
        dest="datalake_p",
        type=int,
        default=None,
        help="valeur p: exécute datalake pour tous les k premiers <= p (requiert --datalake-n)",
    )
    datalake_grp.add_argument(
        "--datalake-i-max",
        dest="datalake_i_max",
        type=int,
        default=None,
        help="borne supérieure incluse pour i (1<=i<=i_max). Par défaut i_max=k (donc i=1..k).",
    )
    datalake_grp.add_argument("--datalake-n", dest="datalake_n", type=int, default=None, help="N (borne supérieure) pour la feature datalake")
    datalake_grp.add_argument(
        "--datalake-j-mult",
        dest="datalake_j_mult",
        type=int,
        default=2,
        help="multiplicateur pour la plage de j: j dans 0..(j_mult*k-1). Par défaut 2",
    )
    datalake_grp.add_argument(
        "--datalake-base-chunk",
        dest="datalake_base_chunk",
        type=int,
        default=10_000,
        help="taille de chunk de base (la taille effective grandit lentement avec n, de type logarithmique). Par défaut 10000",
    )
    datalake_grp.add_argument(
        "--datalake-trajectory-limit",
        dest="datalake_trajectory_limit",
        type=int,
        default=200,
        help="nombre maximal de termes de trajectoire stockés par (n,k,i,j). 0 => ne stocke pas la trajectoire. Par défaut 200",
    )
    datalake_grp.add_argument(
        "--datalake-trajectory-hash",
        dest="datalake_trajectory_hash",
        action="store_true",
        help="si présent, ajoute un hash SHA256 de la trajectoire (tronquée) ou d'une signature si trajectory-limit=0",
    )

    args = parser.parse_args(argv)
    # Validate prime constraints for any provided k-like arguments.
    def _ensure_prime_param(name: str, kval: Optional[int]) -> None:
        if kval is None:
            return
        if kval < 2:
            raise SystemExit(f"{name} must be an integer >= 2")
        primes_check = _sieve_primes(kval)
        if kval not in primes_check:
            raise SystemExit(f"{name}={kval} must be prime")

    # Global/default k (args.k or args.base) must be prime
    if args.k is not None:
        _ensure_prime_param("--k", args.k)
    else:
        _ensure_prime_param("--base", args.base)

    # Validate feature-specific k parameters if provided
    _ensure_prime_param("--single-k", args.single_k)
    _ensure_prime_param("--single-overall-k", args.single_overall_k)
    _ensure_prime_param("--spirale-k", args.spirale_k)
    _ensure_prime_param("--dirichlet-k", getattr(args, "dirichlet_k", None))
    _ensure_prime_param("--residu-single-overall-k", args.residu_single_overall_k)
    _ensure_prime_param("--shannon-entropy-k", args.shannon_entropy_k)
    _ensure_prime_param("--mixing-property-k", args.mixing_property_k)
    _ensure_prime_param("--resistance-k", getattr(args, "resistance_k", None))
    _ensure_prime_param("--pearson-k", getattr(args, "pearson_k", None))
    _ensure_prime_param("--lyapunov-k", getattr(args, "lyapunov_k", None))
    # kernel-k: only validate as prime when explicitly provided (kernel-p will be handled separately)
    if getattr(args, "kernel_k", None) is not None:
        _ensure_prime_param("--kernel-k", int(getattr(args, "kernel_k")))

    # datalake feature: validate if requested
    if getattr(args, "datalake_path", None):
        _ensure_prime_param("--datalake-k", getattr(args, "datalake_k", None))
        if getattr(args, "datalake_p", None) is not None and int(getattr(args, "datalake_p")) < 2:
            raise SystemExit("--datalake-p must be >= 2")

    k_div = args.k if args.k is not None else args.base

    # datalake dispatch (runs independently of the normal run_dir feature routing)
    if getattr(args, "datalake_path", None) is not None:
        if getattr(args, "datalake_n", None) is None:
            raise SystemExit("--datalake-path requires --datalake-n")
        if getattr(args, "datalake_k", None) is None and getattr(args, "datalake_p", None) is None:
            raise SystemExit("--datalake-path requires either --datalake-k or --datalake-p")
        if getattr(args, "datalake_k", None) is not None and getattr(args, "datalake_p", None) is not None:
            raise SystemExit("Use only one of --datalake-k or --datalake-p")
        from . import feature_datalake as datalake_mod

        if getattr(args, "datalake_p", None) is not None:
            ks = _sieve_primes(int(getattr(args, "datalake_p")))
        else:
            ks = [int(getattr(args, "datalake_k"))]

        for dk in ks:
            i_max = getattr(args, "datalake_i_max", None)
            if i_max is None:
                i_max = dk
            i_max = int(i_max)
            if i_max < 1:
                raise SystemExit("--datalake-i-max must be >= 1")

            # Run i = 1..i_max (inclusive). Default is i_max=k.
            for i_val in range(1, i_max + 1):
                cfg = datalake_mod.DataLakeConfig(
                    datalake_path=str(getattr(args, "datalake_path")),
                    k=dk,
                    i=int(i_val),
                    n_max=int(getattr(args, "datalake_n")),
                    j_mult=int(getattr(args, "datalake_j_mult", 2)),
                    max_iters=int(getattr(args, "max_iters", DEFAULT_MAX_ITERS)),
                    divergence_threshold=float(getattr(args, "divergence_threshold", DEFAULT_DIVERGENCE_THRESHOLD)),
                    alternated=bool(getattr(args, "alternated", False)),
                    alt_m=int(getattr(args, "alt_m", 1)),
                    trajectory_limit=int(getattr(args, "datalake_trajectory_limit", 200)),
                    trajectory_hash=bool(getattr(args, "datalake_trajectory_hash", False)),
                    workers=int(getattr(args, "workers", 4)),
                )
                datalake_mod.run_datalake(cfg, base_chunk=int(getattr(args, "datalake_base_chunk", 10_000)))
        return

    base_output = "output"
    os.makedirs(base_output, exist_ok=True)

    # If Shannon entropy feature is requested, use a dedicated run prefix.
    # We rely on the common run_dir creation below, and override it early.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_prefix = _detect_output_feature_prefix(args)
    output_suffix = _detect_output_suffix(args)
    # If the user requested the trajectory-minimum range mode (default behavior
    # when --trajectory-minimum-n is provided without --trajectory-minimum-single),
    # prefer a dedicated run prefix so output directories start with
    # `trajectory_minimum_...`.
    if getattr(args, "trajectory_minimum_n", None) is not None and not getattr(args, "trajectory_minimum_single", False):
        output_prefix = "trajectory_minimum"
    run_dir = os.path.join(base_output, f"{output_prefix}_{timestamp}_{output_suffix}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run output directory: {run_dir}")
    info_path = os.path.join(run_dir, "run_info.txt")
    with open(info_path, "w", encoding="utf-8") as infof:
        infof.write("sufes package run\n")
        infof.write("feature: " + output_prefix + "\n")
        infof.write("output_suffix: " + output_suffix + "\n")
        infof.write("timestamp: " + timestamp + "\n")
        infof.write("start=%s, end=%s, base=%s, k=%s, j=%s, p=%s, kmax=%s, compact_json=%s, alternated=%s, alt_m=%s, all_i=%s, workers=%s, chunk_size=%s, max_iters=%s, divergence_threshold=%s\n" % (args.start, args.end, args.base, args.k, args.j, args.p, args.kmax, args.compact_json, args.alternated, args.alt_m, args.all_i, args.workers, args.chunk_size, args.max_iters, args.divergence_threshold))

    log_path = os.path.join(run_dir, "run.log")
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    try:
        logf = open(log_path, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.__stdout__, logf)
        sys.stderr = _Tee(sys.__stderr__, logf)
        print(f"Logging stdout/stderr into {log_path}")
    except Exception:
        print("Warning: could not open log file for writing; continuing without file logging")

    if args.use_gmpy:
        try:
            import gmpy2  # noqa: F401
            print("gmpy2 available (note: core algorithms currently use Python ints)")
        except Exception:
            print("gmpy2 not available — continuing with Python ints")

    if args.use_numba:
        try:
            import numba  # noqa: F401
            print("numba available (note: no JIT path implemented in core)")
        except Exception:
            print("numba not available — continuing without JIT")

    from . import feature_gamma as gamma_mod
    from . import feature_shannon_entropy as shannon_entropy_mod
    from . import feature_mixing_property as mixing_property_mod
    from . import feature_resistance as resistance_mod
    from . import feature_single_overall as single_overall_mod
    from . import feature_residu_distribution as residu_distribution_mod
    from . import feature_divisions as divisions_mod
    from . import feature_lyapunov as lyapunov_mod
    from . import feature_k_adique as k_adique_mod
    from . import feature_altitude as altitude_mod

    # spirale: plot trajectory in polar/spiral coordinates for a single n/k/i/j
    if args.spirale_n is not None:
        if args.spirale_all and args.spirale_p is None:
            raise SystemExit("--spirale-all requires --spirale-p when using --spirale-n")
        if args.spirale_k is not None and args.spirale_p is not None:
            raise SystemExit("Provide exactly one of --spirale-k or --spirale-p when using --spirale-n")
        from . import feature_spirale as spirale_mod
        if args.spirale_k is not None:
            spirale_mod.spirale(
                args.spirale_n,
                args.spirale_k,
                args.spirale_i,
                args.spirale_j,
                run_dir,
                angle_mode=args.spirale_angle_mode,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
            return
        if args.spirale_p is not None:
            if bool(getattr(args, "spirale_all", False)):
                spirale_mod.spirale_all(
                    args.spirale_n,
                    args.spirale_p,
                    args.spirale_i,
                    args.spirale_j,
                    run_dir,
                    angle_mode=args.spirale_angle_mode,
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                )
            else:
                spirale_mod.spirale_p(
                    args.spirale_n,
                    args.spirale_p,
                    args.spirale_i,
                    args.spirale_j,
                    run_dir,
                    angle_mode=args.spirale_angle_mode,
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                )
            return

    # gamma: compute gamma for primes k <= p with a single (n,i,j)
    if args.gamma_n is not None:
        if args.gamma_p is None:
            raise SystemExit("--gamma-n requires --gamma-p")
        gamma_mod.gamma(
            args.gamma_n,
            args.gamma_p,
            args.gamma_i,
            args.gamma_j,
            run_dir,
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
            do_plot=bool(args.plot_gamma),
            all_i=bool(args.gamma_all_i),
            all_j=bool(args.gamma_all_j),
        )
        return

    # trajectory-minimum: single-trajectory minimum and comparison
    if getattr(args, "trajectory_minimum_n", None) is not None:
        if getattr(args, "trajectory_minimum_k", None) is None and getattr(args, "trajectory_minimum_p", None) is None:
            raise SystemExit("--trajectory-minimum-n requires --trajectory-minimum-k or --trajectory-minimum-p")
        from . import feature_trajectory_minimum as traj_min_mod

        # Determine mode: default is range (all n = 1..N) unless user requests single.
        N = int(getattr(args, "trajectory_minimum_n"))
        # support either a single prime k or batch primes <= p
        kk = getattr(args, "trajectory_minimum_k")
        pp = getattr(args, "trajectory_minimum_p")
        ii = int(getattr(args, "trajectory_minimum_i"))
        jj = int(getattr(args, "trajectory_minimum_j"))

        # If user requested explicit single-run for a single k, call the single wrapper.
        if getattr(args, "trajectory_minimum_single", False):
            if kk is None and pp is not None:
                # when single requested with p, it's ambiguous — require a k
                raise SystemExit("--trajectory-minimum-single requires --trajectory-minimum-k (not --trajectory-minimum-p)")
            traj_min_mod.trajectory_minimum(
                getattr(args, "trajectory_minimum_n"),
                int(kk),
                getattr(args, "trajectory_minimum_i"),
                getattr(args, "trajectory_minimum_j"),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
            return

        # Range mode: compute for all n = 1..N. Support batch over primes when --trajectory-minimum-p is provided.
        workers = int(getattr(args, "workers", 4) or 1)
        if pp is not None:
            primes = _sieve_primes(int(pp))
            for k_val in primes:
                print(f"trajectory-minimum: running for prime k={k_val}")
                j_multi = getattr(args, "trajectory_minimum_j_multi", None)
                if j_multi is None:
                    results = traj_min_mod.trajectory_minimum_range(
                        N,
                        int(k_val),
                        ii,
                        jj,
                        run_dir,
                        workers=workers,
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                    )
                else:
                    # loop over j values 0..k*j_multi-1
                    for j_val in range(0, int(k_val) * int(j_multi)):
                        print(f"trajectory-minimum: running for prime k={k_val} j={j_val}")
                        results = traj_min_mod.trajectory_minimum_range(
                            N,
                            int(k_val),
                            ii,
                            int(j_val),
                            run_dir,
                            workers=workers,
                            max_iters=args.max_iters,
                            divergence_threshold=args.divergence_threshold,
                            alternated=args.alternated,
                            alt_m=args.alt_m,
                        )
            print(f"trajectory-minimum: wrote consolidated results to {os.path.join(run_dir, 'trajectory_minimum')}")
            return

        # Single-k range mode
        kk = int(kk)
        j_multi = getattr(args, "trajectory_minimum_j_multi", None)
        if j_multi is None:
            results = traj_min_mod.trajectory_minimum_range(
                N,
                kk,
                ii,
                jj,
                run_dir,
                workers=workers,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
        else:
            for j_val in range(0, kk * int(j_multi)):
                print(f"trajectory-minimum: running for k={kk} j={j_val}")
                results = traj_min_mod.trajectory_minimum_range(
                    N,
                    kk,
                    ii,
                    int(j_val),
                    run_dir,
                    workers=workers,
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                )
        print(f"trajectory-minimum: wrote consolidated results to {os.path.join(run_dir, 'trajectory_minimum')}")
        return
    # Shannon entropy: standalone computation for either a single k or all primes k <= p
    if args.shannon_entropy_n is not None or args.shannon_entropy_k is not None or args.shannon_entropy_p is not None:
        if args.shannon_entropy_n is None:
            raise SystemExit("--shannon-entropy-n is required when using shannon entropy feature")
        if args.shannon_entropy_k is not None and args.shannon_entropy_p is not None:
            raise SystemExit("Provide exactly one of --shannon-entropy-k or --shannon-entropy-p")
        if args.shannon_entropy_k is None and args.shannon_entropy_p is None:
            raise SystemExit("--shannon-entropy-n requires --shannon-entropy-k or --shannon-entropy-p")

        if args.shannon_entropy_k is not None:
            shannon_entropy_mod.shannon_entropy_run(
                args.shannon_entropy_n,
                args.shannon_entropy_k,
                args.shannon_entropy_i,
                args.shannon_entropy_j,
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
        else:
            shannon_entropy_mod.shannon_entropy_p_run(
                args.shannon_entropy_n,
                args.shannon_entropy_p,
                args.shannon_entropy_i,
                args.shannon_entropy_j,
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                all_j=bool(args.shannon_entropy_all_j),
            )
        return

    # mixing-property: lag plot of residues for a single (n,k,i,j) or for all primes k <= p
    if args.mixing_property_n is not None:
        if args.mixing_property_k is not None and args.mixing_property_p is not None:
            raise SystemExit("Provide exactly one of --mixing-property-k or --mixing-property-p")
        if args.mixing_property_k is None and args.mixing_property_p is None:
            raise SystemExit("--mixing-property-n requires --mixing-property-k or --mixing-property-p")
        if args.mixing_property_k is not None:
            mixing_property_mod.mixing_property(
                args.mixing_property_n,
                args.mixing_property_k,
                args.mixing_property_i,
                args.mixing_property_j,
                run_dir,
                all_j=bool(args.mixing_property_all_j),
                lag=int(args.mixing_property_lag),
                max_points=int(args.mixing_property_max_points),
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
        else:
            mixing_property_mod.mixing_property_p(
                args.mixing_property_n,
                args.mixing_property_p,
                args.mixing_property_i,
                args.mixing_property_j,
                run_dir,
                all_j=bool(args.mixing_property_all_j),
                lag=int(args.mixing_property_lag),
                max_points=int(args.mixing_property_max_points),
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
        return

    # resistance feature
    if getattr(args, "resistance_n", None) is not None:
        if getattr(args, "resistance_k", None) is not None and getattr(args, "resistance_p", None) is not None:
            raise SystemExit("Provide exactly one of --resistance-k or --resistance-p")
        if getattr(args, "resistance_k", None) is None and getattr(args, "resistance_p", None) is None:
            raise SystemExit("--resistance-n requires --resistance-k or --resistance-p")

        if getattr(args, "resistance_k", None) is not None:
            resistance_mod.resistance(
                getattr(args, "resistance_n"),
                getattr(args, "resistance_k"),
                getattr(args, "resistance_i"),
                getattr(args, "resistance_j"),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
        else:
            resistance_mod.resistance_p(
                getattr(args, "resistance_n"),
                getattr(args, "resistance_p"),
                getattr(args, "resistance_i"),
                getattr(args, "resistance_j"),
                run_dir,
                all_j=bool(getattr(args, "resistance_all_j", False)),
                all_n=bool(getattr(args, "resistance_all_n", False)),
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
            )
        return

    # kernel feature
    # - single-k mode: --kernel (or --kernel-k)
    # - batch mode: --kernel-p (all primes k<=p)
    if getattr(args, "kernel_p", None) is not None:
        from . import feature_kernel as kernel_mod

        p_val = int(getattr(args, "kernel_p"))
        primes = _sieve_primes(p_val)

        i_for = getattr(args, "kernel_i", None)
        if i_for is None:
            i_for = getattr(args, "i", None)
        if i_for is None:
            raise SystemExit("--kernel-p requires --kernel-i (or global --i)")
        j_for = getattr(args, "kernel_j", None)
        if j_for is None:
            j_for = args.j if args.j is not None else 0

        index = {"p": p_val, "i": int(i_for), "j": int(j_for), "runs": []}
        for k_val in primes:
            if args.alt_m >= k_val:
                continue
            print(f"Running kernel for k={k_val}")
            summ = kernel_mod.kernel(
                int(k_val),
                int(i_for),
                int(j_for),
                run_dir,
                max_iters=int(args.max_iters),
                divergence_threshold=float(args.divergence_threshold),
                alternated=bool(args.alternated),
                alt_m=int(args.alt_m),
                workers=int(getattr(args, "workers", 4)),
            )
            # record outputs for easy discovery
            out_base = f"kernel_k{int(k_val)}_i{int(i_for)}_j{int(j_for)}"
            index["runs"].append(
                {
                    "k": int(k_val),
                    "csv": f"{out_base}.csv",
                    "json": f"{out_base}.json",
                    "png": f"{out_base}_subplots.png",
                    "summary": summ.get("summary", {}),
                }
            )

        try:
            out_index = os.path.join(run_dir, f"kernel_p{p_val}_i{int(i_for)}_j{int(j_for)}_index.json")
            with open(out_index, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            print(f"Wrote: {out_index}")
        except Exception:
            pass
        return

    if getattr(args, "kernel", False) or getattr(args, "kernel_k", None) is not None:
        from . import feature_kernel as kernel_mod

        k_for = getattr(args, "kernel_k", None)
        if k_for is None:
            k_for = args.k if args.k is not None else args.base
        if k_for is None:
            raise SystemExit("--kernel requires --kernel-k or a global --k/base")
        i_for = getattr(args, "kernel_i", None)
        if i_for is None:
            i_for = getattr(args, "i", None)
        if i_for is None:
            raise SystemExit("--kernel requires --kernel-i (or global --i)")
        j_for = getattr(args, "kernel_j", None)
        if j_for is None:
            j_for = args.j if args.j is not None else 0

        kernel_mod.kernel(
            int(k_for),
            int(i_for),
            int(j_for),
            run_dir,
            max_iters=int(args.max_iters),
            divergence_threshold=float(args.divergence_threshold),
            alternated=bool(args.alternated),
            alt_m=int(args.alt_m),
            workers=int(getattr(args, "workers", 4)),
        )
        return

    # Prefer the new --single-overall-* args; fall back to legacy --residu-single-overall-* for compatibility
    s_n = args.single_overall_n if getattr(args, 'single_overall_n', None) is not None else args.residu_single_overall_n
    s_k = args.single_overall_k if getattr(args, 'single_overall_k', None) is not None else args.residu_single_overall_k
    s_i = args.single_overall_i if getattr(args, 'single_overall_i', None) is not None else args.residu_single_overall_i
    s_j = args.single_overall_j if getattr(args, 'single_overall_j', None) is not None else args.residu_single_overall_j
    if s_n is not None and s_k is not None and s_i is not None and s_j is not None:
        single_overall_mod.single_n_overall(
            s_n,
            s_k,
            s_i,
            s_j,
            run_dir,
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
        )
        return

    if args.residu_distribution_n is not None and args.residu_distribution_p is not None:
        # If the user requested all j in 0..k-1, override j and j_multiple
        j_val_local = args.residu_distribution_j
        j_mult_local = int(getattr(args, "residu_distribution_j_mult", 2))
        if getattr(args, "residu_distribution_all_j", False):
            j_val_local = None
            j_mult_local = int(getattr(args, "residu_distribution_j_mult", 1))

        residu_distribution_mod.residu_distribution(
            args.residu_distribution_n,
            args.residu_distribution_p,
            run_dir,
            i_val=args.residu_distribution_i,
            j_val=j_val_local,
            j_multiple=int(j_mult_local),
            # By default, aggregate across n0=1..N unless user overrides with flag
            all_n=bool(getattr(args, "residu_distribution_all_n", True)),
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
            include_zero_mean=args.residu_distribution_include_zero,
        )
        return

    # NOTE: removed legacy/duplicated --residu-ecart handling to avoid
    # confusion with --residu-ecart-type which is the supported flag set.

    # NOTE: legacy --residu-ecart-type / --residu-lambda entrypoints were
    # residu-ecart-type (reintroduced as a dedicated feature module)
    if args.residu_ecart_type_n is not None and args.residu_ecart_type_p is not None:
        from . import feature_residu_ecart_type as residu_ecart_type_mod

        residu_ecart_type_mod.residu_ecart_type(
            n_max=int(args.residu_ecart_type_n),
            p=int(args.residu_ecart_type_p),
            run_dir=run_dir,
            i_val=int(args.residu_ecart_type_i),
            j_multiple=int(args.residu_j_multiple),
            single_n=args.residu_ecart_type_single_n,
            max_iters=int(args.max_iters),
            divergence_threshold=float(args.divergence_threshold),
            alternated=bool(args.alternated),
            alt_m=int(args.alt_m),
            plot=True,
        )
        return

    # NOTE: residu-lambda removed; A2/A_v moved into residu-distribution.

    # k-adique distribution for a single (n,k,i,j)
    if args.k_adique_n is not None and args.k_adique_k is not None:
        k_adique_mod.k_adique_distribution(
            args.k_adique_n,
            args.k_adique_k,
            args.k_adique_i,
            args.k_adique_j,
            run_dir,
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
        )
        return

    # coalescence feature: prefix coalescence (trajectory of n intersects any previous <= n-1)
    # If --coalescence-p is provided we intend to run batch mode for primes <= p.
    # Only run the single-k coalescence path when --coalescence-n is provided and
    # --coalescence-p is NOT supplied.
    if getattr(args, "coalescence_n", None) is not None and getattr(args, "coalescence_p", None) is None:
        if getattr(args, "coalescence_k", None) is None:
            raise SystemExit("--coalescence-n requires --coalescence-k when --coalescence-p is not provided")
        from . import feature_coalescence as coalescence_mod

        k_val = int(args.coalescence_k)
        j_multi = int(getattr(args, "coalescence_j_multi", 1))
        if j_multi < 1:
            raise SystemExit("--coalescence-j-multi must be >= 1")
        js = list(range(0, k_val * j_multi))

        print(f"Running coalescence (prefix) for k={k_val}, i={int(args.coalescence_i)}, {len(js)} j values")
        for j_val in js:
            print(f"  coalescence: j={int(j_val)}")
            coalescence_mod.coalescence(
                int(args.coalescence_n),
                int(k_val),
                int(args.coalescence_i),
                int(j_val),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                workers=int(getattr(args, "workers", 4)),
                write_csv=True,
                write_png=bool(getattr(args, "coalescence_verbose", False)),
                write_verbose=bool(getattr(args, "coalescence_verbose", False)),
            )

            # brief printed summary from produced JSON
            json_path = os.path.join(
                run_dir,
                f"coalescence_prefix_upto_n{int(args.coalescence_n)}_k{int(k_val)}_i{int(args.coalescence_i)}_j{int(j_val)}.json",
            )
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                s = data.get("summary", {})
                print(
                    f"    summary: N={int(args.coalescence_n)} coalescence_count={s.get('coalescence_count')} coalescence_rate={s.get('coalescence_rate')} mean_capture_time={s.get('mean_capture_time')}"
                )
            except Exception:
                pass
        return

    # coalescence batch over primes p: run coalescence for every prime k <= p
    if getattr(args, "coalescence_p", None) is not None:
        if getattr(args, "coalescence_n", None) is None:
            raise SystemExit("--coalescence-p requires --coalescence-n to specify upper bound N")
        from . import feature_coalescence as coalescence_mod

        p_val = int(getattr(args, "coalescence_p"))
        primes = _sieve_primes(p_val)
        j_multi = int(getattr(args, "coalescence_j_multi", 1))
        if j_multi < 1:
            raise SystemExit("--coalescence-j-multi must be >= 1")

        per_k_results = {}  # k -> j -> {counts...}
        for k_val in primes:
            if args.alt_m >= k_val:
                continue
            js = list(range(0, int(k_val) * j_multi))
            print(f"Running coalescence (prefix) for k={k_val} (j_multi={j_multi}, {len(js)} js)")

            per_k_results[str(k_val)] = {}
            for j_val in js:
                coalescence_mod.coalescence(
                    int(args.coalescence_n),
                    int(k_val),
                    int(args.coalescence_i),
                    int(j_val),
                    run_dir,
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                    workers=int(getattr(args, "workers", 4)),
                    write_csv=True,
                    write_png=bool(getattr(args, "coalescence_verbose", False)),
                    write_verbose=bool(getattr(args, "coalescence_verbose", False)),
                )

                json_path = os.path.join(
                    run_dir,
                    f"coalescence_prefix_upto_n{int(args.coalescence_n)}_k{int(k_val)}_i{int(args.coalescence_i)}_j{int(j_val)}.json",
                )
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    s = data.get("summary", {})
                except Exception:
                    s = {}

                n_total = int(args.coalescence_n)
                n_coal = int(s.get("coalescence_count") or 0)
                mean_ct = s.get("mean_capture_time")
                n_div = 0
                n_err = 0
                per_k_results[str(k_val)][str(j_val)] = {
                    "total": int(n_total),
                    "coalesced": int(n_coal),
                    "mean_capture_time": mean_ct,
                    "divergence": int(n_div),
                    "error": int(n_err),
                }
                print(
                    f"  j={int(j_val)}: total={n_total} coalesced={n_coal} mean_capture_time={mean_ct} divergence={n_div} error={n_err}"
                )

        # One global aggregated JSON summary for this run
        out_global = os.path.join(
            run_dir,
            f"coalescence_n{int(args.coalescence_n)}_p{int(p_val)}_i{int(args.coalescence_i)}_jmulti{int(j_multi)}_summary.json",
        )
        try:
            with open(out_global, "w", encoding="utf-8") as gf:
                json.dump(
                    {
                        "n": int(args.coalescence_n),
                        "p": int(p_val),
                        "i": int(args.coalescence_i),
                        "j_multi": int(j_multi),
                        "k_values": [int(k) for k in primes],
                        "results": per_k_results,
                    },
                    gf,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"Wrote global coalescence summary: {out_global}")
        except Exception:
            pass

        # Note: old combined figures were based on (steps_a,steps_b) for n and n+1.
        # With the new definition, per-n plots are only generated in verbose mode
        # by feature_coalescence itself.

        # If verbose is enabled, also build a combined capture_time figure:
        # subplot per k, overlay capture_time(n) for different j.
        if bool(getattr(args, "coalescence_verbose", False)):
            try:
                import math as _math
                import matplotlib.pyplot as plt

                def _j_color(idx: int) -> str:
                    return f"C{idx % 10}"

                ks = [int(k) for k in primes if str(k) in per_k_results]
                if ks:
                    cols = 2 if len(ks) > 1 else 1
                    rows = int(_math.ceil(len(ks) / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.2 * rows), squeeze=False)

                    for idx_k, k_val in enumerate(ks):
                        ax = axes[idx_k // cols][idx_k % cols]
                        j_map = per_k_results.get(str(k_val), {})
                        js_sorted = sorted([int(j) for j in j_map.keys()])

                        for idx_j, j_val in enumerate(js_sorted):
                            # Prefer the detailed CSV written by the feature.
                            # It contains capture_time per n.
                            details_csv = os.path.join(
                                run_dir,
                                f"coalescence_prefix_upto_n{int(args.coalescence_n)}_k{int(k_val)}_i{int(args.coalescence_i)}_j{int(j_val)}_details.csv",
                            )
                            if not os.path.exists(details_csv):
                                continue

                            xs = []
                            ys = []
                            try:
                                with open(details_csv, "r", encoding="utf-8") as f:
                                    header = f.readline()
                                    for line in f:
                                        parts = line.strip().split(",")
                                        if len(parts) < 6:
                                            continue
                                        try:
                                            n0 = int(parts[0])
                                        except Exception:
                                            continue
                                        # capture_time is 4th column
                                        ct_raw = parts[3]
                                        try:
                                            ct = float(ct_raw) if ct_raw not in ("", "None", "nan") else float("nan")
                                        except Exception:
                                            ct = float("nan")
                                        xs.append(n0)
                                        ys.append(ct)
                            except Exception:
                                continue

                            if xs:
                                ax.plot(xs, ys, linewidth=0.8, label=f"j={j_val}", color=_j_color(idx_j))

                        ax.set_title(f"k={k_val}")
                        ax.set_xlabel("n")
                        ax.set_ylabel("capture_time")
                        ax.grid(True, alpha=0.25)
                        if js_sorted:
                            ax.legend(fontsize=8)

                    for j in range(len(ks), rows * cols):
                        axes[j // cols][j % cols].axis("off")

                    fig.suptitle(
                        f"Coalescence capture_time (n={int(args.coalescence_n)}, i={int(args.coalescence_i)}, p={int(p_val)})"
                    )
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_cap = os.path.join(run_dir, f"coalescence_n{int(args.coalescence_n)}_p{int(p_val)}_capture_time_by_k.png")
                    fig.savefig(out_cap, dpi=150)
                    plt.close(fig)
                    print(f"Wrote: {out_cap}")
            except Exception:
                pass

            # Also build mean_capture_time vs j: subplot per k, x=j.
            try:
                import math as _math
                import matplotlib.pyplot as plt

                ks = [int(k) for k in primes if str(k) in per_k_results]
                if ks:
                    cols = 2 if len(ks) > 1 else 1
                    rows = int(_math.ceil(len(ks) / cols))
                    fig_m, axes_m = plt.subplots(rows, cols, figsize=(6.0 * cols, 3.2 * rows), squeeze=False)

                    for idx_k, k_val in enumerate(ks):
                        ax = axes_m[idx_k // cols][idx_k % cols]
                        j_map = per_k_results.get(str(k_val), {})
                        js_sorted = sorted([int(j) for j in j_map.keys()])

                        xs = []
                        ys = []
                        for j_val in js_sorted:
                            rec = j_map.get(str(j_val), {})
                            xs.append(int(j_val))
                            mct = rec.get("mean_capture_time")
                            if mct is None:
                                ys.append(float("nan"))
                            else:
                                try:
                                    ys.append(float(mct))
                                except Exception:
                                    ys.append(float("nan"))

                        ax.plot(xs, ys, marker='o', linestyle='-', linewidth=0.8)
                        ax.set_title(f"k={int(k_val)}")
                        ax.set_xlabel("j")
                        ax.set_ylabel("mean_capture_time")
                        ax.grid(True, alpha=0.25)

                    for j in range(len(ks), rows * cols):
                        axes_m[j // cols][j % cols].axis("off")

                    fig_m.suptitle(
                        f"Coalescence mean_capture_time by j (n={int(args.coalescence_n)}, i={int(args.coalescence_i)}, p={int(p_val)})"
                    )
                    fig_m.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_mct = os.path.join(
                        run_dir,
                        f"coalescence_n{int(args.coalescence_n)}_p{int(p_val)}_mean_capture_time_by_k.png",
                    )
                    fig_m.savefig(out_mct, dpi=150)
                    plt.close(fig_m)
                    print(f"Wrote: {out_mct}")
            except Exception:
                pass

        return

    # cycle length for all start values n=1..N
    if getattr(args, "cycle_n", None) is not None and getattr(args, "cycle_p", None) is not None:
        if bool(getattr(args, "cycle_all_j", False)):
            cycle_mod.run_cycle_feature_p_all_j(
                int(getattr(args, "cycle_n")),
                int(getattr(args, "cycle_p")),
                int(getattr(args, "cycle_i")),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                cycle_cardinality=bool(getattr(args, "cycle_cardinality", False)),
                        special_cycles=bool(getattr(args, "special_cycles", False)),
                        extra_special_cycles=bool(getattr(args, "extra_special_cycles", False)),
                        j_multiple=int(getattr(args, "cycle_j_multiple", 1)),
                        card_top_cycles=int(getattr(args, "card_top_cycles", 5)),
                        workers=int(getattr(args, "workers", 4)),
                        fst_appearance=bool(getattr(args, "fst_appearance", False)),
            )
        else:
            cycle_mod.run_cycle_feature_p(
                int(getattr(args, "cycle_n")),
                int(getattr(args, "cycle_p")),
                int(getattr(args, "cycle_i")),
                int(getattr(args, "cycle_j")),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                cycle_cardinality=bool(getattr(args, "cycle_cardinality", False)),
                        special_cycles=bool(getattr(args, "special_cycles", False)),
                        extra_special_cycles=bool(getattr(args, "extra_special_cycles", False)),
                        j_multiple=int(getattr(args, "cycle_j_multiple", 1)),
                        workers=int(getattr(args, "workers", 4)),
                        fst_appearance=bool(getattr(args, "fst_appearance", False)),
            )
        return

    # footprint feature
    if getattr(args, "footprint_n", None) is not None or getattr(args, "footprint_n_multiple_k", None) is not None:
        # Support either --footprint-k (single prime k) or --footprint-p (all primes <= p)
        if getattr(args, "footprint_k", None) is None and getattr(args, "footprint_p", None) is None:
            raise SystemExit("--footprint-n or --footprint-n-multiple-k requires --footprint-k or --footprint-p")
        from . import feature_footprint as footprint_mod

        # Use a dedicated output folder for footprint runs so outputs are organised separately
        footprint_prefix = "footprint"
        footprint_run_dir = os.path.join(base_output, f"{footprint_prefix}_{timestamp}_{output_suffix}")
        os.makedirs(footprint_run_dir, exist_ok=True)
        print(f"Footprint run output directory: {footprint_run_dir}")

        def _effective_n_for_k(k_val: int) -> int:
            """Return the effective N used for footprint.

            Contract: footprint processes all start values 1..N (inclusive).
            When --footprint-n-multiple-k is set to m, we use N = m*k (no +1).
            """

            if getattr(args, "footprint_n_multiple_k", None) is not None:
                mult = int(getattr(args, "footprint_n_multiple_k"))
                return mult * int(k_val)
            return int(getattr(args, "footprint_n"))

        def _run_footprint_prefixes_for_k(k_val: int, j_val: int, run_dir_local: str) -> dict:
            """Compute footprint for all prefix sizes N'=1..N_eff for one (k,j).

            Returns an object:
              {k,i,j,N_eff, per_n:[{N,S,max_seen,total_unique_visited,visited_ratio_N_2N,visited_count_N_2N}]}
            """

            N_eff = _effective_n_for_k(int(k_val))
            per_n = []
            collect_delta = bool(getattr(args, "footprint_n_delta", False))
            collect_total = bool(getattr(args, "footprint_total", False))
            delta_counts: dict[int, int] = {}  # delta -> count
            # In prefixes mode, default to compact output unless the user explicitly requests verbose.
            compact_mode = bool(getattr(args, "footprint_compact", False)) or (
                bool(getattr(args, "footprint_prefixes", False)) and not bool(getattr(args, "footprint_verbose", False))
            )

            parity_enabled = bool(getattr(args, "footprint_check_parity", False)) and int(k_val) >= 2
            parity_failures = []  # list of small dicts {N, expected_S, got_S}
            for n_eff in range(1, int(N_eff) + 1):
                if compact_mode:
                    # Compute footprint in-memory without writing per-N files.
                    # This is intentionally a small, self-contained version of feature_footprint.footprint
                    # that only returns the summary scalars needed for the aggregated output.
                    from .algorithms import next_term_ji as _footprint_next_term_ji

                    N_local = int(n_eff)
                    k_local = int(k_val)
                    i_local = int(getattr(args, "footprint_i"))
                    j_local = int(j_val)
                    visited = set()
                    max_seen = 0
                    ns_local = list(range(1, N_local + 1))

                    def _walk_one(n0: int):
                        local_visited = set()
                        local_max = int(n0)
                        t = int(n0)
                        local_seen = set()
                        for _ in range(int(args.max_iters)):
                            if abs(t) > float(args.divergence_threshold):
                                break
                            local_visited.add(int(t))
                            if int(t) > local_max:
                                local_max = int(t)
                            if int(t) in local_seen:
                                break
                            local_seen.add(int(t))
                            t = _footprint_next_term_ji(t, k_local, j_local, i_local, alternated=bool(args.alternated), alt_m=int(args.alt_m))
                        return local_visited, int(local_max)

                    if int(getattr(args, "workers", 4)) <= 1:
                        for loc_set, loc_max in map(_walk_one, ns_local):
                            visited.update(loc_set)
                            if loc_max > max_seen:
                                max_seen = int(loc_max)
                    else:
                        try:
                            from concurrent.futures import ThreadPoolExecutor as _FootprintThreadPoolExecutor

                            with _FootprintThreadPoolExecutor(max_workers=int(getattr(args, "workers", 4))) as executor:
                                for loc_set, loc_max in executor.map(_walk_one, ns_local):
                                    visited.update(loc_set)
                                    if loc_max > max_seen:
                                        max_seen = int(loc_max)
                        except Exception:
                            for loc_set, loc_max in map(_walk_one, ns_local):
                                visited.update(loc_set)
                                if loc_max > max_seen:
                                    max_seen = int(loc_max)

                    max_visited = max(visited) if visited else 0
                    S = 0
                    for s in range(1, int(max_visited) + 1):
                        if s in visited:
                            S = s
                            continue
                        break

                    if N_local > 0 and visited:
                        count_in_band = sum(1 for t in visited if (t > N_local and t <= 2 * N_local))
                        ratio = float(count_in_band) / float(N_local)
                    else:
                        count_in_band = 0
                        ratio = 0.0

                    summ_n = {
                        "N": int(N_local),
                        "S": int(S),
                        "max_seen": int(max_seen),
                        "total_unique_visited": int(len(visited)),
                        "visited_ratio_N_2N": float(ratio),
                        "visited_count_N_2N": int(count_in_band),
                    }
                else:
                    summ_n = footprint_mod.footprint(
                        int(n_eff),
                        int(k_val),
                        int(getattr(args, "footprint_i")),
                        int(j_val),
                        run_dir_local,
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                        do_plot=False,
                        workers=int(getattr(args, "workers", 4)),
                    )
                # keep only the stable/interesting scalars for the aggregated payload
                row = {
                    "N": int(summ_n.get("N", n_eff)),
                    # New naming: use F (max covered prefix). Keep S for backward compat.
                    "F": int(summ_n.get("F", summ_n.get("S", 0))),
                    "S": int(summ_n.get("F", summ_n.get("S", 0))),
                    "max_seen": int(summ_n.get("max_seen", 0)),
                    "visited_ratio_N_2N": float(summ_n.get("visited_ratio_N_2N", 0.0)),
                    "visited_count_N_2N": int(summ_n.get("visited_count_N_2N", 0)),
                }
                # This is the metric requested by --footprint-total.
                if collect_total:
                    row["total_unique_visited"] = int(summ_n.get("total_unique_visited", 0))
                per_n.append(row)

                if collect_delta:
                    try:
                        N_now = int(summ_n.get("N", n_eff))
                        F_now = int(summ_n.get("F", summ_n.get("S", 0)))
                        d = int(abs(int(F_now) - int(N_now)))
                        delta_counts[d] = int(delta_counts.get(d, 0) + 1)
                    except Exception:
                        pass

                if parity_enabled:
                    N_now = int(summ_n.get("N", n_eff))
                    S_now = int(summ_n.get("F", summ_n.get("S", 0)))
                    if int(k_val) == 2:
                        # k=2 special rule requested: if N even => S(N)=N+1, if N odd => S(N)=N
                        expected = int(N_now + 1) if (N_now % 2 == 0) else int(N_now)
                        rule = "k=2: S(N)=N+1 if N even else N"
                    else:
                        # k>=3 rule: if N even => S(N)=N, if N odd => S(N)=N+1
                        expected = int(N_now) if (N_now % 2 == 0) else int(N_now + 1)
                        rule = "k>=3: S(N)=N if N even else N+1"

                    if int(S_now) != int(expected):
                        # Keep only a few failures to avoid huge JSONs.
                        if len(parity_failures) < 50:
                            parity_failures.append({"N": int(N_now), "expected_S": int(expected), "got_S": int(S_now), "rule": rule})

            payload = {
                "k": int(k_val),
                "i": int(getattr(args, "footprint_i")),
                "j": int(j_val),
                "N_eff": int(N_eff),
                "per_n": per_n,
            }

            # --- Asymptotic/tail diagnostics for ratios ---
            # We can't do a formal limit, but we can summarize tail behavior.
            def _tail_stats(vals: list[float], tail: int = 50) -> dict:
                v = [
                    float(x)
                    for x in vals
                    if x is not None
                    and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
                ]
                if not v:
                    return {"n": 0}
                t = v[-int(tail) :] if len(v) > int(tail) else v
                return {
                    "n": int(len(v)),
                    "tail": int(len(t)),
                    "last": float(v[-1]),
                    "mean_tail": float(sum(t) / len(t)) if t else float(v[-1]),
                    "min_tail": float(min(t)) if t else float(v[-1]),
                    "max_tail": float(max(t)) if t else float(v[-1]),
                }

            try:
                xs_n = [int(r.get("N")) for r in per_n if r.get("N") is not None]
                fs = [int(r.get("F")) for r in per_n if r.get("F") is not None]
                ss = [float(f) / float(n) if int(n) != 0 else 0.0 for n, f in zip(xs_n, fs)]
                payload["ratio_F_over_N_tail"] = {
                    "definition": "F(N')/N'",
                    **_tail_stats(ss, tail=50),
                }
            except Exception:
                pass

            if collect_total:
                try:
                    tv = [
                        int(r.get("total_unique_visited"))
                        for r in per_n
                        if r.get("total_unique_visited") is not None
                    ]
                    xs_n2 = [int(r.get("N")) for r in per_n if r.get("N") is not None]
                    tr = [float(v) / float(n) if int(n) != 0 else 0.0 for n, v in zip(xs_n2, tv)]
                    payload["ratio_total_unique_visited_over_N_tail"] = {
                        "definition": "total_unique_visited(N')/N'",
                        **_tail_stats(tr, tail=50),
                    }
                except Exception:
                    pass

            if collect_delta:
                total = int(len(per_n))
                deltas_sorted = sorted(delta_counts.items(), key=lambda kv: kv[0])
                payload["delta_abs_S_minus_N"] = {
                    "enabled": True,
                    "definition": "delta(N)=|S(N)-N|",
                    "total": total,
                    "counts": {str(int(d)): int(c) for d, c in deltas_sorted},
                    "percent": {str(int(d)): (float(c) / float(total) if total > 0 else 0.0) for d, c in deltas_sorted},
                }
            if parity_enabled:
                # Compute pass rate using the appropriate rule per k
                total = int(len(per_n))
                if int(k_val) == 2:
                    n_fail = (
                        int(
                            len(
                                [
                                    1
                                    for r in per_n
                                    if (
                                        (int(r.get("N", 0)) % 2 == 0 and int(r.get("S", -1)) != int(r.get("N", 0)) + 1)
                                        or (int(r.get("N", 0)) % 2 == 1 and int(r.get("S", -1)) != int(r.get("N", 0)))
                                    )
                                ]
                            )
                        )
                        if total > 0
                        else 0
                    )
                    rule = "k=2: S(N)=N+1 if N even else N"
                else:
                    n_fail = (
                        int(
                            len(
                                [
                                    1
                                    for r in per_n
                                    if (
                                        (int(r.get("N", 0)) % 2 == 0 and int(r.get("S", -1)) != int(r.get("N", 0)))
                                        or (int(r.get("N", 0)) % 2 == 1 and int(r.get("S", -1)) != int(r.get("N", 0)) + 1)
                                    )
                                ]
                            )
                        )
                        if total > 0
                        else 0
                    )
                    rule = "k>=3: S(N)=N if N even else N+1"

                payload["parity_check"] = {
                    "enabled": True,
                    "rule": rule,
                    "total": total,
                    "failures": int(n_fail),
                    "pass_rate": float((total - int(n_fail)) / total) if total > 0 else 0.0,
                    "failures_sample": parity_failures,
                }
            return payload

        # Prepare j values for a given k
        def _js_for_k(k_val: int) -> list[int]:
            j_multi = int(getattr(args, "footprint_j_multi", 1))
            if j_multi < 1:
                raise SystemExit("--footprint-j-multi must be >= 1")
            if j_multi == 1:
                return [int(getattr(args, "footprint_j"))]
            return list(range(0, int(k_val) * int(j_multi)))

        # If --footprint-p provided, loop primes <= p
        if getattr(args, "footprint_p", None) is not None:
            p_val = int(getattr(args, "footprint_p"))
            primes = _sieve_primes(p_val)
            all_summaries = []
            for k_val in primes:
                if args.alt_m >= k_val:
                    continue
                js = _js_for_k(int(k_val))
                if bool(getattr(args, "footprint_prefixes", False)):
                    print(f"Running footprint prefixes for k={int(k_val)} ({len(js)} j values)")
                    for j_val in js:
                        all_summaries.append(_run_footprint_prefixes_for_k(int(k_val), int(j_val), footprint_run_dir))
                else:
                    N_eff = _effective_n_for_k(int(k_val))
                    for j_val in js:
                        summ = footprint_mod.footprint(
                            int(N_eff),
                            int(k_val),
                            int(getattr(args, "footprint_i")),
                            int(j_val),
                            footprint_run_dir,
                            max_iters=args.max_iters,
                            divergence_threshold=args.divergence_threshold,
                            alternated=args.alternated,
                            alt_m=args.alt_m,
                            do_plot=False,
                            workers=int(getattr(args, "workers", 4)),
                        )
                        # optional parity check annotation
                        if bool(getattr(args, "footprint_check_parity", False)) and int(k_val) >= 2:
                            try:
                                N_now = int(summ.get("N", N_eff))
                                S_now = int(summ.get("S", 0))
                                if int(k_val) == 2:
                                    expected = int(N_now + 1) if (N_now % 2 == 0) else int(N_now)
                                    rule = "k=2: S(N)=N+1 if N even else N"
                                else:
                                    expected = int(N_now) if (N_now % 2 == 0) else int(N_now + 1)
                                    rule = "k>=3: S(N)=N if N even else N+1"
                                summ["parity_check"] = {
                                    "enabled": True,
                                    "rule": rule,
                                    "expected_S": int(expected),
                                    "got_S": int(S_now),
                                    "passed": bool(int(S_now) == int(expected)),
                                }
                            except Exception:
                                pass
                        summ["k"] = int(k_val)
                        summ["j"] = int(j_val)
                        all_summaries.append(summ)

            # save combined summaries (include effective N in the filename)
            n_label = f"nmult{int(getattr(args, 'footprint_n_multiple_k'))}" if getattr(args, "footprint_n_multiple_k", None) is not None else f"N{int(args.footprint_n)}"
            mode_label = "prefixes" if bool(getattr(args, "footprint_prefixes", False)) else "single"
            jmulti = int(getattr(args, "footprint_j_multi", 1))
            out_all = os.path.join(footprint_run_dir, f"footprint_{mode_label}_{n_label}_p{p_val}_jmulti{jmulti}_summaries.json")
            try:
                with open(out_all, "w", encoding="utf-8") as f:
                    json.dump(all_summaries, f, ensure_ascii=False, indent=2)
                print(f"Wrote: {out_all}")
            except Exception:
                pass

            # If prefixes mode, plot S(N') vs N' with one subplot per k
            if bool(getattr(args, "footprint_prefixes", False)):
                try:
                    import matplotlib.pyplot as plt

                    # group payloads by k then overlay different j curves in same subplot
                    by_k = {}
                    for item in all_summaries:
                        try:
                            kk = int(item.get("k"))
                        except Exception:
                            continue
                        by_k.setdefault(kk, []).append(item)

                    ks = sorted(by_k.keys())
                    nplots = len(ks)
                    cols = int(math.ceil(math.sqrt(nplots))) if nplots > 0 else 1
                    rows = int(math.ceil(nplots / cols)) if nplots > 0 else 1
                    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)
                    make_total = bool(getattr(args, "footprint_total", False))
                    if make_total:
                        fig_total, axes_total = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)
                        fig_max, axes_max = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)
                        fig_total_ratio, axes_total_ratio = plt.subplots(
                            rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False
                        )

                    # Always produce a ratio plot F(N')/N' in prefixes mode: one subplot per k.
                    fig_ratio, axes_ratio = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)
                    make_delta = bool(getattr(args, "footprint_n_delta", False))
                    if make_delta:
                        fig_delta, axes_delta = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False)

                    def _j_color(idx: int) -> str:
                        return f"C{idx % 10}"

                    for idx_k, k_ in enumerate(ks):
                        ax = axes[idx_k // cols][idx_k % cols]
                        ax_t = axes_total[idx_k // cols][idx_k % cols] if make_total else None
                        ax_m = axes_max[idx_k // cols][idx_k % cols] if make_total else None
                        ax_tr = axes_total_ratio[idx_k // cols][idx_k % cols] if make_total else None
                        ax_r = axes_ratio[idx_k // cols][idx_k % cols]
                        ax_d = axes_delta[idx_k // cols][idx_k % cols] if make_delta else None
                        items_k = by_k.get(k_, [])
                        # sort by j for stable coloring
                        items_k_sorted = sorted(items_k, key=lambda it: int(it.get("j", 0)))
                        any_curve = False
                        any_curve_t = False
                        any_curve_m = False
                        any_curve_tr = False
                        any_curve_r = False
                        any_curve_d = False
                        for idx_j, item in enumerate(items_k_sorted):
                            per_n = item.get("per_n", []) if isinstance(item.get("per_n"), list) else []
                            xs = [int(r.get("N")) for r in per_n if r.get("N") is not None]
                            ys = [int(r.get("F", r.get("S"))) for r in per_n if (r.get("F") is not None or r.get("S") is not None)]
                            if xs and ys and len(xs) == len(ys):
                                ax.plot(xs, ys, marker=".", markersize=2, linewidth=0.8, color=_j_color(idx_j), label=f"j={int(item.get('j', 0))}")
                                any_curve = True

                                # ratio F(N')/N'
                                ys_r = [float(s) / float(n) if int(n) != 0 else 0.0 for n, s in zip(xs, ys)]
                                ax_r.plot(
                                    xs,
                                    ys_r,
                                    marker=".",
                                    markersize=2,
                                    linewidth=0.8,
                                    color=_j_color(idx_j),
                                    label=f"j={int(item.get('j', 0))}",
                                )
                                any_curve_r = True

                            if make_total and ax_t is not None:
                                ys_t = [
                                    int(r.get("total_unique_visited"))
                                    for r in per_n
                                    if r.get("total_unique_visited") is not None
                                ]
                                if xs and ys_t and len(xs) == len(ys_t):
                                    ax_t.plot(
                                        xs,
                                        ys_t,
                                        marker=".",
                                        markersize=2,
                                        linewidth=0.8,
                                        color=_j_color(idx_j),
                                        label=f"j={int(item.get('j', 0))}",
                                    )
                                    any_curve_t = True

                                if ax_tr is not None:
                                    ys_tr = [float(v) / float(n) if int(n) != 0 else 0.0 for n, v in zip(xs, ys_t)]
                                    ax_tr.plot(
                                        xs,
                                        ys_tr,
                                        marker=".",
                                        markersize=2,
                                        linewidth=0.8,
                                        color=_j_color(idx_j),
                                        label=f"j={int(item.get('j', 0))}",
                                    )
                                    any_curve_tr = True

                            if make_total and ax_m is not None:
                                ys_m = [int(r.get("max_seen")) for r in per_n if r.get("max_seen") is not None]
                                if xs and ys_m and len(xs) == len(ys_m):
                                    ax_m.plot(
                                        xs,
                                        ys_m,
                                        marker=".",
                                        markersize=2,
                                        linewidth=0.8,
                                        color=_j_color(idx_j),
                                        label=f"j={int(item.get('j', 0))}",
                                    )
                                    any_curve_m = True

                            if make_delta and ax_d is not None:
                                ys_d = []
                                try:
                                    for r in per_n:
                                        if r.get("N") is None or r.get("S") is None:
                                            continue
                                        ys_d.append(int(abs(int(r.get("S")) - int(r.get("N")))))
                                except Exception:
                                    ys_d = []
                                if xs and ys_d and len(xs) == len(ys_d):
                                    ax_d.plot(
                                        xs,
                                        ys_d,
                                        marker=".",
                                        markersize=2,
                                        linewidth=0.8,
                                        color=_j_color(idx_j),
                                        label=f"j={int(item.get('j', 0))}",
                                    )
                                    any_curve_d = True
                        if any_curve:
                            ax.set_axis_on()
                            if len(items_k_sorted) > 1:
                                ax.legend(fontsize=7)
                        else:
                            ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                            ax.set_axis_off()
                        ax.set_title(f"k={k_}")
                        ax.set_xlabel("N'")
                        ax.set_ylabel("F(N')")
                        ax.grid(True, alpha=0.25)

                        if make_total and ax_t is not None:
                            if any_curve_t:
                                ax_t.set_axis_on()
                                if len(items_k_sorted) > 1:
                                    ax_t.legend(fontsize=7)
                            else:
                                ax_t.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                                ax_t.set_axis_off()
                            ax_t.set_title(f"k={k_}")
                            ax_t.set_xlabel("N'")
                            ax_t.set_ylabel("total_unique_visited(N')")
                            ax_t.grid(True, alpha=0.25)

                        if make_total and ax_m is not None:
                            if any_curve_m:
                                ax_m.set_axis_on()
                                if len(items_k_sorted) > 1:
                                    ax_m.legend(fontsize=7)
                            else:
                                ax_m.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                                ax_m.set_axis_off()
                            ax_m.set_title(f"k={k_}")
                            ax_m.set_xlabel("N'")
                            ax_m.set_ylabel("max_seen(N')")
                            ax_m.grid(True, alpha=0.25)

                        if make_total and ax_tr is not None:
                            if any_curve_tr:
                                ax_tr.set_axis_on()
                                if len(items_k_sorted) > 1:
                                    ax_tr.legend(fontsize=7)
                            else:
                                ax_tr.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                                ax_tr.set_axis_off()
                            ax_tr.set_title(f"k={k_}")
                            ax_tr.set_xlabel("N'")
                            ax_tr.set_ylabel("total_unique_visited(N')/N'")
                            ax_tr.grid(True, alpha=0.25)

                        if make_delta and ax_d is not None:
                            if any_curve_d:
                                ax_d.set_axis_on()
                                if len(items_k_sorted) > 1:
                                    ax_d.legend(fontsize=7)
                            else:
                                ax_d.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                                ax_d.set_axis_off()
                            ax_d.set_title(f"k={k_}")
                            ax_d.set_xlabel("N'")
                            ax_d.set_ylabel("|F(N')-N'|")
                            ax_d.grid(True, alpha=0.25)

                        # Ratio plot formatting (always enabled)
                        if any_curve_r:
                            ax_r.set_axis_on()
                            if len(items_k_sorted) > 1:
                                ax_r.legend(fontsize=7)
                        else:
                            ax_r.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                            ax_r.set_axis_off()
                        ax_r.set_title(f"k={k_}")
                        ax_r.set_xlabel("N'")
                        ax_r.set_ylabel("F(N')/N'")
                        ax_r.grid(True, alpha=0.25)

                    for idx_off in range(nplots, rows * cols):
                        axes[idx_off // cols][idx_off % cols].axis("off")
                        if make_total:
                            axes_total[idx_off // cols][idx_off % cols].axis("off")
                            axes_max[idx_off // cols][idx_off % cols].axis("off")
                            axes_total_ratio[idx_off // cols][idx_off % cols].axis("off")
                        axes_ratio[idx_off // cols][idx_off % cols].axis("off")
                        if make_delta:
                            axes_delta[idx_off // cols][idx_off % cols].axis("off")

                    fig.tight_layout()
                    out_png = os.path.join(footprint_run_dir, f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_F_by_k.png")
                    fig.savefig(out_png, dpi=150)
                    plt.close(fig)
                    print(f"Wrote: {out_png}")

                    fig_ratio.tight_layout()
                    out_png_r = os.path.join(
                        footprint_run_dir,
                        f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_F_over_N_by_k.png",
                    )
                    fig_ratio.savefig(out_png_r, dpi=150)
                    plt.close(fig_ratio)
                    print(f"Wrote: {out_png_r}")

                    # If --footprint-total is enabled, also plot the tail mean ratio
                    # total_unique_visited(N')/N' as a function of j (one subplot per k).
                    if make_total:
                        try:
                            fig_tail, axes_tail = plt.subplots(
                                rows, cols, figsize=(5.6 * cols, 3.6 * rows), squeeze=False
                            )
                            for idx_k, k_ in enumerate(ks):
                                ax = axes_tail[idx_k // cols][idx_k % cols]
                                items_k = by_k.get(k_, [])
                                # sort by j for stable x-axis
                                items_k_sorted = sorted(items_k, key=lambda it: int(it.get("j", 0)))
                                xs_j = []
                                ys_mt = []
                                for item in items_k_sorted:
                                    j_val = int(item.get("j", 0))
                                    tail = item.get("ratio_total_unique_visited_over_N_tail") or {}
                                    mt = tail.get("mean_tail")
                                    if mt is None:
                                        continue
                                    xs_j.append(j_val)
                                    ys_mt.append(float(mt))

                                if xs_j and ys_mt and len(xs_j) == len(ys_mt):
                                    ax.plot(xs_j, ys_mt, marker=".", markersize=3, linewidth=0.8)
                                    ax.set_axis_on()
                                else:
                                    ax.text(0.5, 0.5, "no tail stats", ha="center", va="center", color="gray")
                                    ax.set_axis_off()

                                ax.set_title(f"k={k_}")
                                ax.set_xlabel("j")
                                ax.set_ylabel("mean_tail(total_unique_visited/N')")
                                ax.grid(True, alpha=0.25)

                            for idx_off in range(nplots, rows * cols):
                                axes_tail[idx_off // cols][idx_off % cols].axis("off")

                            fig_tail.tight_layout()
                            out_png_tail = os.path.join(
                                footprint_run_dir,
                                f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_total_unique_visited_over_N_tail_mean_by_k.png",
                            )
                            fig_tail.savefig(out_png_tail, dpi=150)
                            plt.close(fig_tail)
                            print(f"Wrote: {out_png_tail}")
                        except Exception as e:
                            try:
                                print(f"[footprint] Warning: failed to plot tail mean ratio by j: {e}")
                            except Exception:
                                pass

                    if make_total:
                        fig_total.tight_layout()
                        out_png_t = os.path.join(
                            footprint_run_dir,
                            f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_total_unique_visited_by_k.png",
                        )
                        fig_total.savefig(out_png_t, dpi=150)
                        plt.close(fig_total)
                        print(f"Wrote: {out_png_t}")

                        fig_max.tight_layout()
                        out_png_m = os.path.join(
                            footprint_run_dir,
                            f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_max_seen_by_k.png",
                        )
                        fig_max.savefig(out_png_m, dpi=150)
                        plt.close(fig_max)
                        print(f"Wrote: {out_png_m}")

                        fig_total_ratio.tight_layout()
                        out_png_tr = os.path.join(
                            footprint_run_dir,
                            f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_total_unique_visited_over_N_by_k.png",
                        )
                        fig_total_ratio.savefig(out_png_tr, dpi=150)
                        plt.close(fig_total_ratio)
                        print(f"Wrote: {out_png_tr}")

                    if make_delta:
                        fig_delta.tight_layout()
                        out_png_d = os.path.join(
                            footprint_run_dir,
                            f"footprint_prefixes_{n_label}_p{p_val}_jmulti{jmulti}_delta_abs_S_minus_N_by_k.png",
                        )
                        fig_delta.savefig(out_png_d, dpi=150)
                        plt.close(fig_delta)
                        print(f"Wrote: {out_png_d}")
                except Exception as e:
                    try:
                        print(f"[footprint] Warning: failed to plot prefixes S_by_k: {e}")
                    except Exception:
                        pass
                return

            # create a subplot per k showing histogram of visited node values
            # (only valid in single-N mode, since it expects per-k visited.json files)
            try:
                import matplotlib.pyplot as plt

                n = len(all_summaries)
                cols = int(math.ceil(math.sqrt(n))) if n > 0 else 1
                rows = int(math.ceil(n / cols)) if n > 0 else 1
                fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                for idx, s in enumerate(all_summaries):
                    ax = axes[idx // cols][idx % cols]
                    base_k = f"footprint_N{int(s.get('N'))}_k{s.get('k')}_i{s.get('i')}_j{s.get('j')}"
                    visited_file = os.path.join(footprint_run_dir, f"{base_k}_visited.json")
                    try:
                        with open(visited_file, "r", encoding="utf-8") as vf:
                            xs = json.load(vf)
                    except Exception:
                        xs = []
                    if xs:
                        ax.hist(xs, bins=100)
                        ax.set_axis_on()
                    else:
                        ax.text(0.5, 0.5, "no data", ha="center", va="center", color="gray")
                        ax.set_axis_off()
                    ax.set_title(f"k={s.get('k')} (N={s.get('N')}) S={s.get('S')}")

                # turn off unused axes
                for idx_off in range(n, rows * cols):
                    axes[idx_off // cols][idx_off % cols].axis("off")

                fig.tight_layout()
                out_png = os.path.join(footprint_run_dir, f"footprint_{n_label}_p{p_val}_jmulti{jmulti}_visited_by_k.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
                print(f"Wrote: {out_png}")
            except Exception as e:
                # Don't fail the whole run if matplotlib isn't available or a per-k file is missing,
                # but do print the reason so the user can diagnose why the PNG isn't created.
                try:
                        print(f"[footprint] Warning: failed to create visited_by_k subplot PNG: {e}")
                except Exception:
                    pass

            # plot visited_ratio_N_2N as a function of k
            try:
                import matplotlib.pyplot as plt

                ks = []
                ratios = []
                for s in all_summaries:
                    k_ = s.get("k")
                    r_ = s.get("visited_ratio_N_2N")
                    if k_ is None or r_ is None:
                        continue
                    ks.append(int(k_))
                    ratios.append(float(r_))

                if ks:
                    pairs = sorted(zip(ks, ratios), key=lambda t: t[0])
                    ks_sorted = [p[0] for p in pairs]
                    ratios_sorted = [p[1] for p in pairs]

                    fig, ax = plt.subplots(figsize=(6.5, 3.8))
                    ax.plot(ks_sorted, ratios_sorted, marker="o", linewidth=1.5)
                    ax.set_title(f"Footprint ratio |{{t: N < t ≤ 2N}}| / N vs k — {n_label}, p={p_val}")
                    ax.set_xlabel("k")
                    ax.set_ylabel("visited_ratio_N_2N")
                    ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    out_ratio = os.path.join(footprint_run_dir, f"footprint_{n_label}_p{p_val}_jmulti{jmulti}_ratio_by_k.png")
                    fig.savefig(out_ratio, dpi=180)
                    plt.close(fig)
                    print(f"Wrote: {out_ratio}")
            except Exception:
                pass

            # write a CSV table k,i,j,N,S,visited_ratio_N_2N,S_origins
            out_csv = os.path.join(footprint_run_dir, f"footprint_{n_label}_p{p_val}_jmulti{jmulti}_table.csv")
            try:
                with open(out_csv, "w", encoding="utf-8") as f:
                    f.write("k,i,j,N,S,visited_ratio_N_2N,S_origins\n")
                    for s in all_summaries:
                        k_ = s.get("k")
                        i_ = s.get("i")
                        j_ = s.get("j")
                        N_ = s.get("N")
                        S_ = s.get("S")
                        ratio_ = s.get("visited_ratio_N_2N")
                        origins = s.get("S_origins", []) or []
                        origins_str = ";".join(str(v) for v in origins)
                        f.write(f"{k_},{i_},{j_},{N_},{S_},{ratio_},\"{origins_str}\"\n")
                print(f"Wrote: {out_csv}")
            except Exception:
                pass
        else:
            # Single k path
            k_val = int(getattr(args, "footprint_k"))
            js = _js_for_k(int(k_val))
            if bool(getattr(args, "footprint_prefixes", False)):
                # single-k prefixes: write one aggregated JSON per (k,j)
                n_label = f"nmult{int(getattr(args, 'footprint_n_multiple_k'))}" if getattr(args, "footprint_n_multiple_k", None) is not None else f"N{int(args.footprint_n)}"
                for j_val in js:
                    payload = _run_footprint_prefixes_for_k(int(k_val), int(j_val), footprint_run_dir)
                    out_all = os.path.join(
                        footprint_run_dir,
                        f"footprint_prefixes_{n_label}_k{k_val}_i{int(getattr(args, 'footprint_i'))}_j{int(j_val)}.json",
                    )
                try:
                    with open(out_all, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    print(f"Wrote: {out_all}")
                except Exception:
                    pass
            else:
                N_eff = _effective_n_for_k(k_val)
                for j_val in js:
                    footprint_mod.footprint(
                        int(N_eff),
                        int(k_val),
                        int(getattr(args, "footprint_i")),
                        int(j_val),
                        footprint_run_dir,
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                        do_plot=True,
                        workers=int(getattr(args, "workers", 4)),
                    )
        return

    if getattr(args, "cycle_n", None) is not None and getattr(args, "cycle_k", None) is not None:
        # Support optional multi-j looping for single-k mode. If the user
        # provided --cycle-all-j or a j-multiple > 1 (via --cycle-j-multiple or
        # its alias --cycle-j-multi), then iterate j in 0..(k*j_multiple-1)
        # and call run_cycle_feature for each j. Otherwise run the single j.
        k_for = int(getattr(args, "cycle_k"))
        j_mult = int(getattr(args, "cycle_j_multiple", 1))
        do_all_j = bool(getattr(args, "cycle_all_j", False)) or (j_mult is not None and j_mult > 1)
        if do_all_j:
            j_end = k_for * j_mult
            for j_val in range(0, j_end):
                print(f"Running cycle: k={k_for} i={int(getattr(args, 'cycle_i'))} j={j_val}")
                cycle_mod.run_cycle_feature(
                    int(getattr(args, "cycle_n")),
                    k_for,
                    int(getattr(args, "cycle_i")),
                    int(j_val),
                    run_dir,
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                    cycle_cardinality=bool(getattr(args, "cycle_cardinality", False)),
                    special_cycles=bool(getattr(args, "special_cycles", False)),
                    extra_special_cycles=bool(getattr(args, "extra_special_cycles", False)),
                    j_multiple=j_mult,
                    workers=int(getattr(args, "workers", 4)),
                    fst_appearance=bool(getattr(args, "fst_appearance", False)),
                )
        else:
            cycle_mod.run_cycle_feature(
                int(getattr(args, "cycle_n")),
                int(getattr(args, "cycle_k")),
                int(getattr(args, "cycle_i")),
                int(getattr(args, "cycle_j")),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                cycle_cardinality=bool(getattr(args, "cycle_cardinality", False)),
                special_cycles=bool(getattr(args, "special_cycles", False)),
                extra_special_cycles=bool(getattr(args, "extra_special_cycles", False)),
                j_multiple=int(getattr(args, "cycle_j_multiple", 1)),
                workers=int(getattr(args, "workers", 4)),
                fst_appearance=bool(getattr(args, "fst_appearance", False)),
            )
        return

    # epsilon calculation across primes k <= p
    # divisions: prefer new flags, fall back to epsilon aliases for backward compatibility
    divisions_n = getattr(args, "divisions_n", None) if hasattr(args, "divisions_n") else None
    divisions_p = getattr(args, "divisions_p", None) if hasattr(args, "divisions_p") else None
    if divisions_n is None and getattr(args, "epsilon_n", None) is not None:
        divisions_n = args.epsilon_n
    if divisions_p is None and getattr(args, "epsilon_p", None) is not None:
        divisions_p = args.epsilon_p

    if divisions_n is not None and divisions_p is not None:
        i_arg = getattr(args, "divisions_i", None) or getattr(args, "epsilon_i", 1)
        j_arg = getattr(args, "divisions_j", None) if getattr(args, "divisions_j", None) is not None else getattr(args, "epsilon_j", None)
        divisions_mod.residu_divisions(
            divisions_n,
            divisions_p,
            i_arg,
            j_arg,
            run_dir,
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
            find_best_j=bool(getattr(args, "divisions_find_best_j", False)) or bool(getattr(args, "epsilon_find_best_j", False)),
            ordre_multiplicatif_j=bool(getattr(args, "divisions_ordre_multiplicatif_j", False)) or bool(getattr(args, "epsilon_ordre_multiplicatif_j", False)),
            table_mode=bool(getattr(args, "divisions_table", False)) or bool(getattr(args, "epsilon_table", False)),
            j_multiple=int(getattr(args, "divisions_j_multi", getattr(args, "epsilon_j_multi", 1))),
            all_n=bool(getattr(args, "divisions_all_n", getattr(args, "epsilon_all_n", False))),
        )
        return

    if args.single_n is not None:
        if args.single_p is not None:
            primes = _sieve_primes(args.single_p)
            i_val = args.single_i if args.single_i is not None else 1
            per_k: Dict[int, Dict[int, Dict[str, object]]] = {}
            for k in primes:
                if args.alt_m >= k:
                    continue
                per_j: Dict[int, Dict[str, object]] = {}
                for j in range(0, k):
                    res = _simulate_single_n(args.single_n, k, j, i_val, args.max_iters, args.divergence_threshold, args.alternated, args.alt_m)
                    per_j[j] = {
                        "steps": res["steps"],
                        "peak": res["peak"],
                        "reason": res["reason"],
                        "sequence": res["sequence"],
                        "sequence_sample": res["sequence"][:200],
                    }
                per_k[k] = per_j
                out_path = os.path.join(run_dir, f"single_n_{args.single_n}_k{k}_i{i_val}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"n": args.single_n, "k": k, "i": i_val, "per_j": per_j}, f, ensure_ascii=False, indent=2)
            _plot_single_p_trajectories(run_dir, args.single_n, args.single_p, per_k)
            _plot_single_p_metric(run_dir, args.single_n, args.single_p, per_k, "steps", "steps_perk")
            _plot_single_p_metric(run_dir, args.single_n, args.single_p, per_k, "peak", "peak_perk")
        else:
            if args.single_k is None or args.single_i is None or args.single_j is None:
                raise SystemExit("--single-n requires --single-k, --single-i, --single-j (or use --single-p)")
            res = _simulate_single_n(args.single_n, args.single_k, args.single_j, args.single_i, args.max_iters, args.divergence_threshold, args.alternated, args.alt_m)
            out_path = args.out if args.out is not None else os.path.join(run_dir, f"single_n_{args.single_n}_k{args.single_k}_i{args.single_i}_j{args.single_j}.json")
            payload = {
                "n": args.single_n,
                "k": args.single_k,
                "i": args.single_i,
                "j": args.single_j,
                **{k: v for k, v in res.items() if k != "sequence"},
                "sequence": res["sequence"],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            _plot_single_trajectory(res["sequence"], out_path + ".png", f"single-n n={args.single_n} k={args.single_k} i={args.single_i} j={args.single_j}")
        return

    # stopping-time (single)
    if args.stopping_n is not None:
        # If --stopping-p is provided we run for all primes k <= p and don't
        # require --stopping-k. Otherwise require --stopping-k/i/j for single-k mode.
        if getattr(args, "stopping_p", None) is None:
            if args.stopping_k is None or args.stopping_i is None or args.stopping_j is None:
                raise SystemExit("--stopping-n requires --stopping-k, --stopping-i, --stopping-j (or use --stopping-p to run over many k)")
        # If --stopping-p provided, run for all primes k <= p and create subplots
        if getattr(args, "stopping_p", None) is not None:
            p_val = int(getattr(args, "stopping_p"))
            primes = _sieve_primes(p_val)
            all_summaries = []
            perfect_stopping = []
            for k_val in primes:
                if args.alt_m >= k_val:
                    continue
                # If --stopping-all-j: compute for all j in 0..k-1, else use the provided j
                if bool(getattr(args, "stopping_all_j", False)):
                    for j_val in range(0, int(k_val)):
                        try:
                            summ = stopping_time_mod.stopping_time(
                                int(args.stopping_n),
                                int(k_val),
                                int(args.stopping_i),
                                int(j_val),
                                run_dir,
                                max_iters=args.max_iters,
                                divergence_threshold=args.divergence_threshold,
                                alternated=args.alternated,
                                alt_m=args.alt_m,
                                workers=int(getattr(args, "workers", 4)),
                            )
                        except Exception:
                            summ = {"error": True, "k": k_val, "j": j_val}
                        summ["k"] = k_val
                        summ["j"] = j_val
                        all_summaries.append(summ)

                        # perfect stopping: no missing stopping_time for any n0>=2.
                        # We ignore the (expected) null at n0=1 since you can't go below 1.
                        try:
                            if not summ.get("error"):
                                sts = summ.get("stopping_times")
                                if isinstance(sts, list):
                                    tail = sts[1:]  # ignore n0=1
                                    if all(v is not None for v in tail):
                                        perfect_stopping.append({"k": int(k_val), "i": int(args.stopping_i), "j": int(j_val)})
                        except Exception:
                            pass
                else:
                    try:
                        summ = stopping_time_mod.stopping_time(
                            int(args.stopping_n),
                            int(k_val),
                            int(args.stopping_i),
                            int(args.stopping_j),
                            run_dir,
                            max_iters=args.max_iters,
                            divergence_threshold=args.divergence_threshold,
                            alternated=args.alternated,
                            alt_m=args.alt_m,
                            workers=int(getattr(args, "workers", 4)),
                        )
                    except Exception:
                        summ = {"error": True, "k": k_val}
                    summ["k"] = k_val
                    summ["j"] = int(args.stopping_j)
                    all_summaries.append(summ)

                    try:
                        if not summ.get("error"):
                            sts = summ.get("stopping_times")
                            if isinstance(sts, list):
                                tail = sts[1:]
                                if all(v is not None for v in tail):
                                    perfect_stopping.append({"k": int(k_val), "i": int(args.stopping_i), "j": int(args.stopping_j)})
                    except Exception:
                        pass

            # Save combined summaries
            out_all = os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_summaries.json")
            try:
                with open(out_all, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "n": int(args.stopping_n),
                            "p": int(p_val),
                            "i": int(args.stopping_i),
                            "all_j": bool(getattr(args, "stopping_all_j", False)),
                            "alternated": bool(getattr(args, "alternated", False)),
                            "alt_m": int(getattr(args, "alt_m", 1)),
                            "summaries": all_summaries,
                            "perfect_stopping": perfect_stopping,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
            except Exception:
                pass

            # Also write a dedicated file for perfect_stopping for quick discovery.
            try:
                out_perf = os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_perfect_stopping.json")
                with open(out_perf, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "n": int(args.stopping_n),
                            "p": int(p_val),
                            "i": int(args.stopping_i),
                            "all_j": bool(getattr(args, "stopping_all_j", False)),
                            "alternated": bool(getattr(args, "alternated", False)),
                            "alt_m": int(getattr(args, "alt_m", 1)),
                            "perfect_stopping": perfect_stopping,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                out_perf_csv = os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_perfect_stopping.csv")
                with open(out_perf_csv, "w", encoding="utf-8") as f:
                    f.write("k,i,j\n")
                    for row in perfect_stopping:
                        f.write(f"{row.get('k')},{row.get('i')},{row.get('j')}\n")
            except Exception:
                pass

            # Compute mean stopping times per k and write CSV table k,i,j,n,mean_stopping_time,mean_total_stopping_time
            out_csv = os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_means_by_kj.csv")
            try:
                def _is_nan(x: object) -> bool:
                    return isinstance(x, float) and math.isnan(x)

                with open(out_csv, "w", encoding="utf-8") as f:
                    f.write("k,i,j,mean_stopping_time,mean_total_stopping_time\n")
                    # prepare data grouped by k for plotting
                    grouped_by_k = {}
                    for s in all_summaries:
                        k_ = s.get("k")
                        j_ = s.get("j")
                        if k_ is None or s.get("error"):
                            continue
                        stopping_vals = s.get("stopping_times") if isinstance(s.get("stopping_times"), list) else []
                        total_vals = s.get("total_stopping_times") if isinstance(s.get("total_stopping_times"), list) else []
                        # Ignore missing/nulls (and any legacy NaNs if old runs are mixed in).
                        s_nums = [float(v) for v in stopping_vals if isinstance(v, (int, float)) and not _is_nan(v)]
                        t_nums = [float(v) for v in total_vals if isinstance(v, (int, float)) and not _is_nan(v)]
                        mean_s = (sum(s_nums) / len(s_nums)) if s_nums else ""
                        mean_t = (sum(t_nums) / len(t_nums)) if t_nums else ""
                        f.write(f"{k_},{int(args.stopping_i)},{j_},{mean_s},{mean_t}\n")
                        grouped_by_k.setdefault(k_, []).append((j_, mean_s, mean_t))
            except Exception as e:
                print(f"[stopping] Failed to write means_by_kj.csv: {e}")

            # Explicit mean plots: one subplot per k, mean_stopping_time vs j and mean_total_stopping_time vs j
            try:
                import matplotlib.pyplot as plt

                # Ensure we have grouped_by_k built; if not, build it from summaries
                if 'grouped_by_k' not in locals():
                    grouped_by_k = {}
                    for s in all_summaries:
                        k_ = s.get("k")
                        j_ = s.get("j")
                        if k_ is None or s.get("error"):
                            continue
                        stopping_vals = s.get("stopping_times") if isinstance(s.get("stopping_times"), list) else []
                        total_vals = s.get("total_stopping_times") if isinstance(s.get("total_stopping_times"), list) else []
                        s_nums = [float(v) for v in stopping_vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
                        t_nums = [float(v) for v in total_vals if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))]
                        mean_s = (sum(s_nums) / len(s_nums)) if s_nums else float('nan')
                        mean_t = (sum(t_nums) / len(t_nums)) if t_nums else float('nan')
                        grouped_by_k.setdefault(k_, []).append((j_, mean_s, mean_t))

                ks = sorted(grouped_by_k.keys())
                nplots = len(ks)
                if nplots > 0:
                    cols = int(math.ceil(math.sqrt(nplots)))
                    rows = int(math.ceil(nplots / cols))

                    fig_s, axes_s = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                    fig_t, axes_t = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)

                    any_plotted_s = False
                    any_plotted_t = False
                    for idx, k_ in enumerate(ks):
                        items = sorted(grouped_by_k.get(k_, []), key=lambda t: t[0])
                        js = [it[0] for it in items]
                        mean_ss = [it[1] for it in items]
                        mean_ts = [it[2] for it in items]
                        row = idx // cols
                        col = idx % cols
                        ax_s = axes_s[row][col]
                        ax_t = axes_t[row][col]
                        # plot if there's any numeric data (not all NaN)
                        plot_s = any([isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) for v in mean_ss])
                        plot_t = any([isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) for v in mean_ts])
                        if plot_s and js:
                            ax_s.plot(js, mean_ss, marker='o', linewidth=1)
                            any_plotted_s = True
                        else:
                            ax_s.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                            ax_s.set_axis_off()
                        if plot_t and js:
                            ax_t.plot(js, mean_ts, marker='o', linewidth=1)
                            any_plotted_t = True
                        else:
                            ax_t.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                            ax_t.set_axis_off()
                        ax_s.set_title(f"k={k_}")
                        ax_s.set_xlabel('j')
                        ax_s.set_ylabel('mean_stopping_time')
                        ax_s.grid(True, alpha=0.25)

                        ax_t.set_title(f"k={k_}")
                        ax_t.set_xlabel('j')
                        ax_t.set_ylabel('mean_total_stopping_time')
                        ax_t.grid(True, alpha=0.25)

                    fig_s.tight_layout()
                    fig_t.tight_layout()
                    if any_plotted_s:
                        fig_s.savefig(os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_mean_stopping_time_by_k.png"), dpi=150)
                    if any_plotted_t:
                        fig_t.savefig(os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_mean_total_stopping_time_by_k.png"), dpi=150)
                    plt.close(fig_s)
                    plt.close(fig_t)
            except Exception:
                pass

            # Build subplot figures: one subplot per k for stopping_time and total_stopping_time
            try:
                import matplotlib.pyplot as plt

                # build a list of distinct k values to determine subplot grid size
                if 'grouped_by_k' in locals():
                    ks = sorted(grouped_by_k.keys())
                else:
                    ks = sorted({s.get("k") for s in all_summaries if s.get("k") is not None})
                nplots = len(ks)
                if nplots == 0:
                    return
                cols = int(math.ceil(math.sqrt(nplots)))
                rows = int(math.ceil(nplots / cols))

                # stopping_time subplot grid
                fig1, axes1 = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                fig2, axes2 = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)

                # If we grouped by k (--stopping-all-j), plot mean vs j per k. Otherwise
                # fall back to plotting the single-j summaries already in all_summaries.
                if 'grouped_by_k' in locals():
                    # iterate over the distinct ks so axes layout matches the grid size
                    for idx, k_ in enumerate(ks):
                        items = grouped_by_k.get(k_, [])
                        row = idx // cols
                        col = idx % cols
                        ax1 = axes1[row][col]
                        ax2 = axes2[row][col]
                        # sort items by j
                        items_sorted = sorted(items, key=lambda t: t[0])
                        js = [it[0] for it in items_sorted]
                        mean_ss = [it[1] if it[1] != "" else float('nan') for it in items_sorted]
                        mean_ts = [it[2] if it[2] != "" else float('nan') for it in items_sorted]
                        if js:
                            ax1.plot(js, mean_ss, marker='o', linewidth=1)
                        else:
                            ax1.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                            ax1.set_axis_off()
                        ax1.set_title(f"k={k_}")
                        ax1.set_xlabel('j')
                        ax1.set_ylabel('mean_stopping_time')
                        ax1.grid(True, alpha=0.25)

                        if js:
                            ax2.plot(js, mean_ts, marker='o', linewidth=1)
                        else:
                            ax2.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                            ax2.set_axis_off()
                        ax2.set_title(f"k={k_}")
                        ax2.set_xlabel('j')
                        ax2.set_ylabel('mean_total_stopping_time')
                        ax2.grid(True, alpha=0.25)
                else:
                    for idx, s in enumerate(all_summaries):
                        k_ = s.get("k")
                        row = idx // cols
                        col = idx % cols
                        ax1 = axes1[row][col]
                        ax2 = axes2[row][col]
                        stopping_vals = s.get("stopping_times") if isinstance(s.get("stopping_times"), list) else None
                        total_vals = s.get("total_stopping_times") if isinstance(s.get("total_stopping_times"), list) else None
                        ns = list(range(1, int(args.stopping_n) + 1))
                        if stopping_vals:
                            ax1.plot(ns, [float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float('nan') for v in stopping_vals], marker='.', markersize=2, linewidth=0.7)
                        ax1.set_title(f"k={k_}")
                        ax1.set_xlabel('n')
                        ax1.set_ylabel('stopping_time')
                        ax1.grid(True, alpha=0.25)

                        if total_vals:
                            ax2.plot(ns, [float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else float('nan') for v in total_vals], marker='.', markersize=2, linewidth=0.7)
                        ax2.set_title(f"k={k_}")
                        ax2.set_xlabel('n')
                        ax2.set_ylabel('total_stopping_time')
                        ax2.grid(True, alpha=0.25)

                fig1.tight_layout()
                fig2.tight_layout()
                fig1.savefig(os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_stopping_time_by_k.png"), dpi=150)
                fig2.savefig(os.path.join(run_dir, f"stopping_n{args.stopping_n}_p{p_val}_total_stopping_time_by_k.png"), dpi=150)
                plt.close(fig1)
                plt.close(fig2)
            except Exception:
                pass

            return
        else:
            summary = stopping_time_mod.stopping_time(
                int(args.stopping_n),
                int(args.stopping_k),
                int(args.stopping_i),
                int(args.stopping_j),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                workers=int(getattr(args, "workers", 4)),
            )
            out_path = os.path.join(run_dir, f"stopping_n{args.stopping_n}_k{args.stopping_k}_i{args.stopping_i}_j{args.stopping_j}_summary.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            return

    # altitude feature: compute peak for all n' <= N for single k or primes k <= p
    if getattr(args, "altitude_n", None) is not None:
        if getattr(args, "altitude_p", None) is not None and getattr(args, "altitude_k", None) is not None:
            raise SystemExit("Provide exactly one of --altitude-k or --altitude-p when using --altitude-n")
        if getattr(args, "altitude_p", None) is None:
            # single k mode: require altitude_k
            if getattr(args, "altitude_k", None) is None:
                raise SystemExit("--altitude-n requires --altitude-k or --altitude-p")
            summary = altitude_mod.altitude(
                int(args.altitude_n),
                int(args.altitude_k),
                int(getattr(args, "altitude_i", 1)),
                int(getattr(args, "altitude_j", 0)),
                run_dir,
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                partitionning=bool(getattr(args, "altitude_partitionning", False)),
                workers=int(getattr(args, "workers", 4)),
            )
            out_path = os.path.join(run_dir, f"altitude_upto_n{args.altitude_n}_k{args.altitude_k}_i{args.altitude_i}_j{args.altitude_j}_summary.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return

    # (pearson handling moved later to avoid interfering with altitude's if/else)
        else:
            # altitude-p: loop primes <= p and collect per-k summaries
            p_val = int(getattr(args, "altitude_p"))
            primes = _sieve_primes(p_val)
            all_summaries = []
            for k_val in primes:
                if args.alt_m >= k_val:
                    continue
                try:
                    summ = altitude_mod.altitude(
                        int(args.altitude_n),
                        int(k_val),
                        int(getattr(args, "altitude_i", 1)),
                        int(getattr(args, "altitude_j", 0)),
                        run_dir,
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                        partitionning=bool(getattr(args, "altitude_partitionning", False)),
                        workers=int(getattr(args, "workers", 4)),
                    )
                except Exception:
                    summ = {"error": True, "k": k_val}
                summ["k"] = k_val
                all_summaries.append(summ)

            out_all = os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_summaries.json")
            try:
                with open(out_all, "w", encoding="utf-8") as f:
                    json.dump(all_summaries, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

            # write CSV of mean peaks per k
            out_csv = os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_peaks_by_k.csv")
            try:
                import math as _math
                with open(out_csv, "w", encoding="utf-8") as f:
                    f.write("k,i,j,n,mean_peak\n")
                    for s in all_summaries:
                        k_ = s.get("k")
                        if k_ is None or s.get("error"):
                            continue
                        peaks = s.get("peaks") if isinstance(s.get("peaks"), list) else []
                        nums = [float(v) for v in peaks if v is not None and not (_math.isnan(v))]
                        mean_p = (sum(nums) / len(nums)) if nums else ""
                        f.write(f"{k_},{int(args.altitude_i)},{int(args.altitude_j)},{int(args.altitude_n)},{mean_p}\n")
            except Exception:
                pass

            # optional plot of mean peaks per k (subplot per k with x=n')
            try:
                import matplotlib.pyplot as plt
                ks = [s.get("k") for s in all_summaries if s.get("k") is not None]
                nplots = len(ks)
                if nplots > 0:
                    cols = int(math.ceil(math.sqrt(nplots)))
                    rows = int(math.ceil(nplots / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                    for idx, s in enumerate(all_summaries):
                        k_ = s.get("k")
                        row = idx // cols
                        col = idx % cols
                        ax = axes[row][col]
                        peaks = s.get("peaks") if isinstance(s.get("peaks"), list) else []
                        ns = list(range(1, int(args.altitude_n) + 1))
                        if peaks:
                            ax.plot(ns, [float(v) if v is not None else float('nan') for v in peaks], marker='.', markersize=2, linewidth=0.7)
                        ax.set_title(f"k={k_}")
                        ax.set_xlabel('n')
                        ax.set_ylabel('peak')
                        ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_peaks_by_k.png"), dpi=150)
                    plt.close(fig)
            except Exception:
                pass

            # optional plot of distance_to_altitude per k (subplot per k)
            try:
                import matplotlib.pyplot as plt
                ks = [s.get("k") for s in all_summaries if s.get("k") is not None]
                nplots = len(ks)
                if nplots > 0:
                    cols = int(math.ceil(math.sqrt(nplots)))
                    rows = int(math.ceil(nplots / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                    for idx, s in enumerate(all_summaries):
                        k_ = s.get("k")
                        row = idx // cols
                        col = idx % cols
                        ax = axes[row][col]
                        dists = s.get("distances_to_altitude") if isinstance(s.get("distances_to_altitude"), list) else []
                        ns = list(range(1, int(args.altitude_n) + 1))
                        if dists:
                            ax.plot(ns, [float(v) if v is not None else float('nan') for v in dists], marker='.', markersize=2, linewidth=0.7)
                        ax.set_title(f"k={k_}")
                        ax.set_xlabel('n')
                        ax.set_ylabel('distance_to_altitude')
                        ax.grid(True, alpha=0.25)
                    fig.tight_layout()
                    fig.savefig(os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_distance_to_altitude_by_k.png"), dpi=150)
                    plt.close(fig)
            except Exception:
                pass

            # optional plot: peak value distribution per k (one subplot per k)
            try:
                import matplotlib.pyplot as plt
                from collections import Counter
                import math as _math

                ks = [s.get("k") for s in all_summaries if s.get("k") is not None]
                nplots = len(ks)
                if nplots > 0:
                    cols = int(_math.ceil(_math.sqrt(nplots)))
                    rows = int(_math.ceil(nplots / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                    for idx, s in enumerate(all_summaries):
                        k_ = s.get("k")
                        row = idx // cols
                        col = idx % cols
                        ax = axes[row][col]
                        peaks = s.get("peaks") if isinstance(s.get("peaks"), list) else []
                        if peaks:
                            counts = Counter(peaks)
                            items = sorted(counts.items(), key=lambda t: int(t[0]))
                            xs = [int(t[0]) for t in items]
                            ys = [int(t[1]) for t in items]
                            ax.bar(xs, ys, width=1.0)
                            # reduce xticks if too many unique peak values
                            if len(xs) > 12:
                                step = max(1, len(xs) // 12)
                                ax.set_xticks(xs[::step])
                                for label in ax.get_xticklabels():
                                    label.set_rotation(45)
                            else:
                                ax.set_xticks(xs)
                                for label in ax.get_xticklabels():
                                    label.set_rotation(45)

                            # compute and display top-10 peaks with percentage of total peaks
                            try:
                                total = sum(ys) if ys else 0
                                top10 = counts.most_common(10)
                                top_lines = []
                                for val, cnt in top10:
                                    pct = (cnt / total * 100.0) if total else 0.0
                                    top_lines.append(f"{int(val)}: {cnt} ({pct:.1f}%)")
                                txt = "\n".join(top_lines)
                                # place text inside subplot (top-left area)
                                ax.text(0.01, 0.98, txt, transform=ax.transAxes, fontsize=6, va='top', ha='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                            except Exception:
                                pass
                        else:
                            ax.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                            ax.set_axis_off()
                        ax.set_title(f"k={k_}")
                        ax.set_xlabel('peak value')
                        ax.set_ylabel('count')
                        ax.grid(True, alpha=0.25)
                    # turn off unused axes
                    for j in range(nplots, rows * cols):
                        axes[j // cols][j % cols].axis('off')
                    fig.tight_layout()
                    fig.savefig(os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_peak_distribution_by_k.png"), dpi=150)
                    plt.close(fig)
                    # Print top-10 peaks per k into run.log as a table
                    try:
                        print("\nTop-10 peaks per k (value: count (percent)):\n")
                        for s in all_summaries:
                            if s.get('error'):
                                continue
                            k_ = s.get('k')
                            peaks = s.get('peaks') if isinstance(s.get('peaks'), list) else []
                            if not peaks:
                                continue
                            counts = Counter(peaks)
                            total = sum(counts.values()) if counts else 0
                            top10 = counts.most_common(10)
                            print(f"k={k_}")
                            print(" peak_value | count | percent")
                            print("------------|-------|--------")
                            for val, cnt in top10:
                                pct = (cnt / total * 100.0) if total else 0.0
                                print(f" {int(val):10d} | {int(cnt):5d} | {pct:6.2f}%")
                            print("")
                    except Exception:
                        pass
            except Exception:
                pass

            # mean vs k plots: mean peak and mean distance_to_altitude as functions of k
            try:
                import matplotlib.pyplot as plt
                mean_rows = []
                for s in all_summaries:
                    if s.get('error'):
                        continue
                    k_ = s.get('k')
                    if k_ is None:
                        continue
                    # prefer precomputed means if available
                    mean_p = s.get('mean_peak') if s.get('mean_peak') is not None else None
                    if mean_p is None:
                        peaks = s.get('peaks') if isinstance(s.get('peaks'), list) else []
                        nums = [float(v) for v in peaks if v is not None]
                        mean_p = (sum(nums) / len(nums)) if nums else None
                    mean_d = s.get('mean_distance_to_altitude') if s.get('mean_distance_to_altitude') is not None else None
                    if mean_d is None:
                        dists = s.get('distances_to_altitude') if isinstance(s.get('distances_to_altitude'), list) else []
                        numsd = [float(v) for v in dists if v is not None]
                        mean_d = (sum(numsd) / len(numsd)) if numsd else None
                    mean_rows.append((int(k_), mean_p, mean_d))

                if mean_rows:
                    mean_rows.sort(key=lambda x: x[0])
                    ks_sorted = [r[0] for r in mean_rows]
                    mean_peaks_sorted = [r[1] if r[1] is not None else float('nan') for r in mean_rows]
                    mean_dists_sorted = [r[2] if r[2] is not None else float('nan') for r in mean_rows]

                    # mean peak vs k
                    plt.figure(figsize=(8, 4))
                    plt.plot(ks_sorted, mean_peaks_sorted, marker='o', linewidth=0.8)
                    plt.xlabel('k')
                    plt.ylabel('mean_peak')
                    plt.title(f'Mean peak vs k (n={args.altitude_n}, p={p_val})')
                    plt.grid(True)
                    out_plot = os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_mean_peak_vs_k.png")
                    plt.tight_layout()
                    plt.savefig(out_plot)
                    plt.close()

                    # mean distance vs k
                    plt.figure(figsize=(8, 4))
                    plt.plot(ks_sorted, mean_dists_sorted, marker='o', linewidth=0.8)
                    plt.xlabel('k')
                    plt.ylabel('mean_distance_to_altitude')
                    plt.title(f'Mean distance_to_altitude vs k (n={args.altitude_n}, p={p_val})')
                    plt.grid(True)
                    out_plot = os.path.join(run_dir, f"altitude_n{args.altitude_n}_p{p_val}_mean_distance_vs_k.png")
                    plt.tight_layout()
                    plt.savefig(out_plot)
                    plt.close()
            except Exception:
                pass

            return

    # pearson feature handling: single-k or p-mode over primes <= p
    from . import feature_pearson as pearson_mod
    from . import feature_dirichlet as dirichlet_mod
    from . import feature_hamming as hamming_mod

    if getattr(args, "pearson_n", None) is not None:
        # single-k mode: require pearson-k unless pearson-p provided
        if getattr(args, "pearson_p", None) is None and getattr(args, "pearson_k", None) is None:
            raise SystemExit("--pearson-n requires --pearson-k or --pearson-p")

        if getattr(args, "pearson_p", None) is None:
            # single k path
            pk = int(getattr(args, "pearson_k"))
            if args.alt_m >= pk:
                raise SystemExit(f"--alt-m must be < k (k={pk}, alt_m={args.alt_m})")
            try:
                summ = pearson_mod.pearson(
                    int(args.pearson_n),
                    pk,
                    int(getattr(args, "pearson_i", 1)),
                    int(getattr(args, "pearson_j", 0)),
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                    workers=int(getattr(args, "workers", 4)),
                )
            except Exception:
                summ = {"error": True, "k": pk}
            out_path = os.path.join(run_dir, f"pearson_upto_n{args.pearson_n}_k{pk}_i{args.pearson_i}_j{args.pearson_j}_summary.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summ, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return

    # dirichlet feature handling: Pearson between residues of n and n+1
    if getattr(args, "dirichlet_n", None) is not None:
        # require dirichlet-k or dirichlet-p
        if getattr(args, "dirichlet_p", None) is None and getattr(args, "dirichlet_k", None) is None:
            raise SystemExit("--dirichlet-n requires --dirichlet-k or --dirichlet-p")

        if getattr(args, "dirichlet_p", None) is None:
            dk = int(getattr(args, "dirichlet_k"))
            if args.alt_m >= dk:
                raise SystemExit(f"--alt-m must be < k (k={dk}, alt_m={args.alt_m})")
            try:
                if getattr(args, "dirichlet_plot_3d", False):
                    summ = dirichlet_mod.dirichlet_with_plots(
                        int(args.dirichlet_n),
                        dk,
                        int(getattr(args, "dirichlet_i", 1)),
                        int(getattr(args, "dirichlet_j", 0)),
                        out_dir=run_dir,
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                        save_3d=True,
                    )
                else:
                    summ = dirichlet_mod.dirichlet(
                        int(args.dirichlet_n),
                        dk,
                        int(getattr(args, "dirichlet_i", 1)),
                        int(getattr(args, "dirichlet_j", 0)),
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                    )
            except Exception:
                summ = {"error": True, "k": dk}
            out_path = os.path.join(run_dir, f"dirichlet_upto_n{args.dirichlet_n}_k{dk}_i{args.dirichlet_i}_j{args.dirichlet_j}_summary.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(summ, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return

    # hamming feature handling (single n) — only when not using --hamming-p
    if getattr(args, "hamming_n", None) is not None and getattr(args, "hamming_p", None) is None:
        if getattr(args, "hamming_k", None) is None:
            raise SystemExit("--hamming-n requires --hamming-k when not using --hamming-p")
        hk = int(getattr(args, "hamming_k"))
        if args.alt_m >= hk:
            raise SystemExit(f"--alt-m must be < k (k={hk}, alt_m={args.alt_m})")
        try:
            # Ask the hamming feature to write its PNG inside the run_dir
            png_prefix = os.path.join(run_dir, "hamming")
            summ = hamming_mod.hamming(
                int(args.hamming_n),
                hk,
                int(getattr(args, "hamming_i", 1)),
                int(getattr(args, "hamming_j", 0)),
                max_iters=args.max_iters,
                divergence_threshold=args.divergence_threshold,
                alternated=args.alternated,
                alt_m=args.alt_m,
                write_png=True,
                png_prefix=png_prefix,
                workers=int(getattr(args, "workers", 4)),
            )
        except Exception:
            summ = {"error": True, "k": hk}
        out_path = os.path.join(run_dir, f"hamming_upto_n{args.hamming_n}_k{hk}_i{args.hamming_i}_j{args.hamming_j}_summary.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summ, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return

    # hamming feature across primes k <= p
    if getattr(args, "hamming_p", None) is not None:
        if getattr(args, "hamming_n", None) is None:
            raise SystemExit("--hamming-p requires --hamming-n")
        hamming_mod.hamming_p(
            int(args.hamming_n),
            int(args.hamming_p),
            int(getattr(args, "hamming_i", 1)),
            int(getattr(args, "hamming_j", 0)),
            run_dir,
            all_j=bool(getattr(args, "hamming_all_j", False)),
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
        )
        # write a small placeholder summary (the function already wrote the PNG)
        out_p = os.path.join(run_dir, f"hamming_p{args.hamming_p}_n{args.hamming_n}_summaries.json")
        try:
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump({"n": int(args.hamming_n), "p": int(args.hamming_p)}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return

        # dirichlet-p mode: loop primes <= p
        p_val = int(getattr(args, "dirichlet_p"))
        primes = _sieve_primes(p_val)
        all_summaries = []
        for k_val in primes:
            if args.alt_m >= k_val:
                continue
            try:
                if getattr(args, "dirichlet_plot_3d", False):
                    summ = dirichlet_mod.dirichlet_with_plots(
                        int(args.dirichlet_n),
                        int(k_val),
                        int(getattr(args, "dirichlet_i", 1)),
                        int(getattr(args, "dirichlet_j", 0)),
                        out_dir=run_dir,
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                        save_3d=True,
                    )
                else:
                    summ = dirichlet_mod.dirichlet(
                        int(args.dirichlet_n),
                        int(k_val),
                        int(getattr(args, "dirichlet_i", 1)),
                        int(getattr(args, "dirichlet_j", 0)),
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                    )
            except Exception:
                summ = {"error": True, "k": k_val}
            summ["k"] = k_val
            summ["j"] = int(args.dirichlet_j)
            all_summaries.append(summ)

        # save combined summaries
        out_all = os.path.join(run_dir, f"dirichlet_n{args.dirichlet_n}_p{p_val}_summaries.json")
        try:
            with open(out_all, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # write CSV table k,i,j,mean_pearson
        out_csv = os.path.join(run_dir, f"dirichlet_n{args.dirichlet_n}_p{p_val}_by_kj.csv")
        try:
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("k,i,j,mean_pearson\n")
                for s in all_summaries:
                    k_ = s.get("k")
                    i_ = s.get("i")
                    j_ = s.get("j")
                    if s.get("error"):
                        mean_p = ""
                    else:
                        mean_p = s.get("mean_pearson") if s.get("mean_pearson") is not None else ""
                    f.write(f"{k_},{i_},{j_},{mean_p}\n")
        except Exception:
            pass

        # create subplot per k showing dirichlet pearson vs n
        try:
            import matplotlib.pyplot as plt

            by_k = {}
            for r in all_summaries:
                k_ = int(r.get("k"))
                by_k.setdefault(k_, []).append(r)

            ks = sorted(by_k.keys())
            n_k = len(ks)
            if n_k > 0:
                ncols = min(3, n_k)
                nrows = int(math.ceil(n_k / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)
                for idx, k_ in enumerate(ks):
                    ax = axes[idx // ncols][idx % ncols]
                    ax.set_title(f"k={k_}")
                    ax.set_xlabel("n")
                    ax.set_ylabel("pearson")
                    rows_k = by_k.get(k_, [])

                    # If per-n data exists, plot pearson vs n
                    has_per_n = any(isinstance(r.get('per_n'), list) and r.get('per_n') for r in rows_k)
                    if has_per_n:
                        for r in sorted(rows_k, key=lambda t: int(t.get('j', 0))):
                            if r.get('error'):
                                continue
                            j_val = r.get('j')
                            per_n = r.get('per_n', [])
                            ns = [int(p.get('n')) for p in per_n]
                            pears = [float(p.get('pearson')) if (p.get('pearson') is not None and not (isinstance(p.get('pearson'), float) and math.isnan(p.get('pearson')))) else float('nan') for p in per_n]
                            label = f"j={j_val}" if len(rows_k) > 1 else None
                            ax.plot(ns, pears, marker='.', linewidth=0.8, label=label)
                        if len(rows_k) > 1:
                            try:
                                ax.legend(fontsize=8)
                            except Exception:
                                pass
                    else:
                        for r in sorted(rows_k, key=lambda t: int(t.get('j', 0))):
                            if r.get('error'):
                                continue
                            val = r.get('mean_pearson') if r.get('mean_pearson') is not None else r.get('pearson')
                            if val is None:
                                continue
                            j_val = r.get('j')
                            label = f"j={j_val}" if len(rows_k) > 1 else None
                            ax.plot([int(args.dirichlet_n)], [float(val)], marker='o', label=label)
                        if len(rows_k) > 1:
                            try:
                                ax.legend(fontsize=8)
                            except Exception:
                                pass

                for j in range(len(ks), nrows * ncols):
                    axes[j // ncols][j % ncols].axis('off')
                fig.suptitle(f'Dirichlet Pearson per k (n=1..{args.dirichlet_n})')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                out_plot = os.path.join(run_dir, f"dirichlet_n{args.dirichlet_n}_p{p_val}_pearson_by_k.png")
                fig.savefig(out_plot)
                plt.close(fig)
        except Exception:
            pass

        return

        # pearson-p mode: loop primes <= p and optionally all j
        p_val = int(getattr(args, "pearson_p"))
        primes = _sieve_primes(p_val)
        all_summaries = []
        for k_val in primes:
            if args.alt_m >= k_val:
                continue
            if bool(getattr(args, "pearson_all_j", False)):
                for j_val in range(0, int(k_val)):
                    try:
                        summ = pearson_mod.pearson(
                            int(args.pearson_n),
                            int(k_val),
                            int(getattr(args, "pearson_i", 1)),
                            int(j_val),
                            max_iters=args.max_iters,
                            divergence_threshold=args.divergence_threshold,
                            alternated=args.alternated,
                            alt_m=args.alt_m,
                        )
                    except Exception:
                        summ = {"error": True, "k": k_val, "j": j_val}
                    summ["k"] = k_val
                    summ["j"] = j_val
                    all_summaries.append(summ)
            else:
                try:
                    summ = pearson_mod.pearson(
                        int(args.pearson_n),
                        int(k_val),
                        int(getattr(args, "pearson_i", 1)),
                        int(getattr(args, "pearson_j", 0)),
                        max_iters=args.max_iters,
                        divergence_threshold=args.divergence_threshold,
                        alternated=args.alternated,
                        alt_m=args.alt_m,
                    )
                except Exception:
                    summ = {"error": True, "k": k_val}
                summ["k"] = k_val
                summ["j"] = int(args.pearson_j)
                all_summaries.append(summ)

        # save combined summaries
        out_all = os.path.join(run_dir, f"pearson_n{args.pearson_n}_p{p_val}_summaries.json")
        try:
            with open(out_all, "w", encoding="utf-8") as f:
                json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        # write CSV table k,i,j,mean_pearson (or empty if error),reason,steps,peak
        out_csv = os.path.join(run_dir, f"pearson_n{args.pearson_n}_p{p_val}_by_kj.csv")
        try:
            with open(out_csv, "w", encoding="utf-8") as f:
                f.write("k,i,j,mean_pearson,reason,steps,peak\n")
                for s in all_summaries:
                    k_ = s.get("k")
                    i_ = s.get("i")
                    j_ = s.get("j")
                    if s.get("error"):
                        mean_p = ""
                        reason = "error"
                        steps = ""
                        peak = ""
                    else:
                        mean_p = s.get("mean_pearson") if s.get("mean_pearson") is not None else ""
                        # when pearson_mod returns per_n shaped dict, we may not have reason/steps/peak at top-level
                        reason = s.get("reason", "")
                        steps = s.get("steps", "")
                        peak = s.get("peak", "")
                    f.write(f"{k_},{i_},{j_},{mean_p},{reason},{steps},{peak}\n")
        except Exception:
            pass

        # create subplot per k showing pearson vs j
        try:
            import matplotlib.pyplot as plt

            by_k = {}
            for r in all_summaries:
                k_ = int(r.get("k"))
                by_k.setdefault(k_, []).append(r)

            ks = sorted(by_k.keys())
            n_k = len(ks)
            if n_k > 0:
                ncols = min(3, n_k)
                nrows = int(math.ceil(n_k / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)
                for idx, k_ in enumerate(ks):
                    ax = axes[idx // ncols][idx % ncols]
                    ax.set_title(f"k={k_}")
                    ax.set_xlabel("n")
                    ax.set_ylabel("pearson")
                    rows_k = by_k.get(k_, [])

                    # If per-n data exists, plot pearson vs n; overlay multiple j as separate lines
                    has_per_n = any(isinstance(r.get('per_n'), list) and r.get('per_n') for r in rows_k)
                    if has_per_n:
                        for r in sorted(rows_k, key=lambda t: int(t.get('j', 0))):
                            if r.get('error'):
                                continue
                            j_val = r.get('j')
                            per_n = r.get('per_n', [])
                            ns = [int(p.get('n')) for p in per_n]
                            pears = [float(p.get('pearson')) if (p.get('pearson') is not None and not (isinstance(p.get('pearson'), float) and math.isnan(p.get('pearson')))) else float('nan') for p in per_n]
                            label = f"j={j_val}" if len(rows_k) > 1 else None
                            ax.plot(ns, pears, marker='.', linewidth=0.8, label=label)
                        if len(rows_k) > 1:
                            try:
                                ax.legend(fontsize=8)
                            except Exception:
                                pass
                    else:
                        # Fallback: plot mean_pearson as a single point at x=N for each j
                        for r in sorted(rows_k, key=lambda t: int(t.get('j', 0))):
                            if r.get('error'):
                                continue
                            val = r.get('mean_pearson') if r.get('mean_pearson') is not None else r.get('pearson')
                            if val is None:
                                continue
                            j_val = r.get('j')
                            label = f"j={j_val}" if len(rows_k) > 1 else None
                            ax.plot([int(args.pearson_n)], [float(val)], marker='o', label=label)
                        if len(rows_k) > 1:
                            try:
                                ax.legend(fontsize=8)
                            except Exception:
                                pass

                for j in range(n_k, nrows * ncols):
                    axes[j // ncols][j % ncols].axis('off')
                fig.suptitle(f"Pearson correlation per k (n={args.pearson_n}, p={p_val})")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                fig.savefig(os.path.join(run_dir, f"pearson_n{args.pearson_n}_p{p_val}_pearson_by_k.png"), dpi=150)
                plt.close(fig)
        except Exception:
            pass

        # Additionally, build a subplot figure: one subplot per k with x-axis = j and y-axis = mean_pearson
        try:
            import matplotlib.pyplot as plt
            import math as _math

            ks = sorted(by_k.keys())
            nplots = len(ks)
            if nplots > 0:
                cols = int(math.ceil(math.sqrt(nplots)))
                rows = int(math.ceil(nplots / cols))
                figm, axesm = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.2 * rows), squeeze=False)
                for idx, k_ in enumerate(ks):
                    row = idx // cols
                    col = idx % cols
                    ax = axesm[row][col]
                    rows_k = by_k.get(k_, [])
                    # collect (j, mean) pairs
                    pairs = []
                    for r in rows_k:
                        if r.get('error'):
                            continue
                        jv = r.get('j')
                        meanp = r.get('mean_pearson') if r.get('mean_pearson') is not None else (r.get('pearson') if r.get('pearson') is not None else None)
                        if meanp is None:
                            continue
                        pairs.append((int(jv), float(meanp)))
                    if pairs:
                        pairs.sort(key=lambda t: t[0])
                        js = [p[0] for p in pairs]
                        ms = [p[1] for p in pairs]
                        ax.plot(js, ms, marker='o', linestyle='-')
                        ax.set_xticks(js)
                    else:
                        ax.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                        ax.set_axis_off()
                    ax.set_title(f"k={k_}")
                    ax.set_xlabel('j')
                    ax.set_ylabel('mean_pearson')

                # turn off unused axes
                for j in range(nplots, rows * cols):
                    axesm[j // cols][j % cols].axis('off')
                figm.tight_layout()
                figm.savefig(os.path.join(run_dir, f"pearson_n{args.pearson_n}_p{p_val}_mean_by_k.png"), dpi=150)
                plt.close(figm)
        except Exception:
            pass

        return

    if args.proof or args.proof_persist:
        # Require proof_max_n and exactly one of proof_p or proof_k
        if args.proof_max_n is None:
            raise SystemExit("--proof/--proof-persist requires --proof-max-n and either --proof-p or --proof-k")
        # allow any integer for --proof-i (including negative); run_proof reduces
        # the value modulo k when computing effective behaviour
        if args.proof_all and args.proof_p is None:
            raise SystemExit("--proof-all requires --proof-p when using --proof/--proof-persist")
        if args.proof_all and args.proof_k is not None:
            raise SystemExit("--proof-all cannot be combined with --proof-k (use --proof-p)")
        if (args.proof_p is None) == (args.proof_k is None):
            # both None or both provided -> error
            raise SystemExit("Provide exactly one of --proof-p or --proof-k when using --proof/--proof-persist")
        if args.proof_all:
            k_values = list(range(2, args.proof_p + 1))
            proof_scope = f"all_to_p{args.proof_p}"
        elif args.proof_k is not None:
            # validate proof_k is prime
            if args.proof_k < 2:
                raise SystemExit("--proof-k must be a prime integer >= 2")
            primes_check = _sieve_primes(args.proof_k)
            if args.proof_k not in primes_check:
                raise SystemExit(f"--proof-k={args.proof_k} does not appear to be prime")
            k_values = [args.proof_k]
            proof_scope = f"k{args.proof_k}"
        else:
            k_values = _sieve_primes(args.proof_p)
            proof_scope = f"p{args.proof_p}"
        try:
            rows = _run_proof(
                k_values,
                args.proof_max_n,
                args.proof_j_mult,
                args.all_i,
                args.alternated,
                args.alt_m,
                args.max_iters,
                args.divergence_threshold,
                run_dir,
                args.proof_persist,
                int(getattr(args, "workers", 4)),
                proof_i=args.proof_i,
                proof_lake=bool(getattr(args, "proof_lake", False)),
            )
        except ValueError as e:
            raise SystemExit(str(e))
        out_json = os.path.join(run_dir, f"proof_{proof_scope}_maxn{args.proof_max_n}.json")
        out_csv = os.path.join(run_dir, f"proof_{proof_scope}_maxn{args.proof_max_n}.csv")
        # Ensure output directory exists (can be missing if run_dir changed or was deleted)
        os.makedirs(run_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "p": args.proof_p,
                    "proof_k": args.proof_k,
                    "proof_all": args.proof_all,
                    "proof_i": args.proof_i,
                    "k_values": k_values,
                    "max_n": args.proof_max_n,
                    "alternated": bool(args.alternated),
                    "alt_m": int(args.alt_m),
                    "rows": rows,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(out_csv, "w", encoding="utf-8") as f:
            f.write("k,i,j,alternated,alt_m,max_proved,status,reason\n")
            for r in rows:
                f.write(
                    f"{r.get('k')},{r.get('i')},{r.get('j')},{r.get('alternated')},{r.get('alt_m')},{r.get('max_proved')},{r.get('status')},{r.get('reason')}\n"
                )
        if args.plot_proof and args.proof_j_mult == 1:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                for k in k_values:
                    rows_k = [r for r in rows if r.get("k") == k and r.get("status") != "SKIPPED"]
                    if not rows_k:
                        continue
                    max_i = max(r.get("i") for r in rows_k)
                    arr = np.full((max_i, k), np.nan)
                    for r in rows_k:
                        i_idx = int(r.get("i")) - 1
                        j_idx = int(r.get("j"))
                        if 0 <= j_idx < k:
                            arr[i_idx, j_idx] = float(r.get("max_proved") or 0)
                    fig, ax = plt.subplots(figsize=(6, 4))
                    im = ax.imshow(arr, aspect="auto")
                    ax.set_title(f"proof k={k}")
                    ax.set_xlabel("j")
                    ax.set_ylabel("i")
                    fig.colorbar(im, ax=ax)
                    fig.tight_layout()
                    fig.savefig(os.path.join(run_dir, f"proof_k{k}_heatmap.png"), dpi=150)
                    plt.close(fig)
            except Exception:
                pass
        return

    # If --p provided: run families for primes <= p
    if args.p is not None:
        primes = _sieve_primes(args.p)
        print(f"Running family for primes k <= {args.p}: {primes}")
        for k in primes:
            out_path = os.path.join(run_dir, f"results_family_k{k}_{args.start}_{args.end}.json")
            if args.alt_m >= k:
                raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={args.alt_m})")
            run_family_for_k(args.start, args.end, k, out=out_path, compact=args.compact_json, alternated=args.alternated, all_i=args.all_i, alt_m=args.alt_m, max_iters=args.max_iters, divergence_threshold=args.divergence_threshold)
        return

    if args.kmax is not None:
        ks = list(range(2, args.kmax + 1))
        print(f"Running family for all k in 2..{args.kmax}: {ks}")
        for k in ks:
            out_path = os.path.join(run_dir, f"results_family_k{k}_{args.start}_{args.end}.json")
            if args.alt_m >= k:
                raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={args.alt_m})")
            run_family_for_k(args.start, args.end, k, out=out_path, compact=args.compact_json, alternated=args.alternated, all_i=args.all_i, alt_m=args.alt_m, max_iters=args.max_iters, divergence_threshold=args.divergence_threshold)
        return

    if args.family:
        out_path = args.out if args.out is not None else os.path.join(run_dir, f"results_family_k{k_div}_{args.start}_{args.end}.json")
        if args.alt_m >= k_div:
            raise SystemExit(f"--alt-m must be < k (k={k_div}, alt_m={args.alt_m})")
        run_family_for_k(args.start, args.end, k_div, out=out_path, compact=args.compact_json, alternated=args.alternated, all_i=args.all_i, alt_m=args.alt_m, max_iters=args.max_iters, divergence_threshold=args.divergence_threshold)
        return

    # Lyapunov single-run or p-mode dispatch: must run before the general family/analyze_range path
    if getattr(args, "lyapunov_n", None) is not None:
        # p-mode: compute for all primes k <= p and produce a combined subplot image
        if getattr(args, "lyapunov_p", None) is not None:
            if args.lyapunov_i is None or args.lyapunov_j is None:
                raise SystemExit("--lyapunov-p requires --lyapunov-i and --lyapunov-j to be provided")
            primes = _sieve_primes(args.lyapunov_p)
            try:
                import matplotlib.pyplot as plt
            except Exception:
                plt = None
            ks = primes
            n_val = args.lyapunov_n
            per_k_lambdas = {}
            for k in ks:
                if args.alt_m >= k:
                    raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={args.alt_m})")
                # compute without per-k plotting (plot=False) and avoid writing huge per-k results if not needed
                summary = lyapunov_mod.lyapunov_run(
                    n_val,
                    k,
                    args.lyapunov_i,
                    args.lyapunov_j,
                    run_dir,
                    max_iters=args.max_iters,
                    divergence_threshold=args.divergence_threshold,
                    alternated=args.alternated,
                    alt_m=args.alt_m,
                    plot=False,
                    write_results=True,
                    workers=int(getattr(args, "workers", 4)),
                )
                per_k_lambdas[k] = summary.get("lambdas", [])

            # create combined subplots if matplotlib available
            if plt is not None and ks:
                try:
                    n = int(n_val)
                    cols = min(3, len(ks))
                    rows = int(math.ceil(len(ks) / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.0 * rows), squeeze=False)
                    for idx, k in enumerate(ks):
                        ax = axes[idx // cols][idx % cols]
                        xs = list(range(1, n + 1))
                        ys = [float(v) if v is not None else float('nan') for v in per_k_lambdas.get(k, [])]
                        ax.plot(xs, ys, marker='.', linewidth=0.6, markersize=2)
                        ax.set_title(f'k={k}')
                        ax.set_xlabel('n')
                        ax.set_ylabel('lambda')
                        ax.grid(True)
                    for j in range(len(ks), rows * cols):
                        axes[j // cols][j % cols].axis('off')
                    fig.suptitle(f'Lyapunov lambda per k (n=1..{n_val})')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    out_plot = os.path.join(run_dir, f"lyapunov_upto_n{n_val}_p{args.lyapunov_p}_lambda_by_k.png")
                    fig.savefig(out_plot)
                    plt.close(fig)
                    print(f"Wrote combined lyapunov plot to {out_plot}")
                except Exception:
                    pass
            return

        # single k mode
        k_for = args.lyapunov_k if getattr(args, "lyapunov_k", None) is not None else (args.k if args.k is not None else args.base)
        if k_for is None:
            raise SystemExit("--lyapunov-n requires --lyapunov-k or a global --k/base")
        if args.lyapunov_i is None or args.lyapunov_j is None:
            raise SystemExit("--lyapunov-n requires --lyapunov-i and --lyapunov-j to be provided")
        lyapunov_mod.lyapunov_run(
            args.lyapunov_n,
            k_for,
            args.lyapunov_i,
            args.lyapunov_j,
            run_dir,
            max_iters=args.max_iters,
            divergence_threshold=args.divergence_threshold,
            alternated=args.alternated,
            alt_m=args.alt_m,
            plot=True,
            write_results=True,
            workers=int(getattr(args, "workers", 4)),
        )
        return

    j_use = args.j if args.j is not None else 0
    if args.alt_m >= k_div:
        raise SystemExit(f"--alt-m must be < k (k={k_div}, alt_m={args.alt_m})")

    if args.workers > 1:
        chunks = []
        for s in range(args.start, args.end + 1, args.chunk_size):
            e = min(args.end, s + args.chunk_size - 1)
            chunks.append((s, e, args.base, k_div, j_use, args.i, args.compact_json, args.divergence_threshold, args.alternated, args.alt_m, args.max_iters))
        # Use thread pool executor to parallelize chunk processing (user requested thread pools)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            summaries = list(executor.map(lambda c: analyze_chunk(c), chunks))
        summary = merge_summaries(summaries)
    else:
        summary = analyze_range(args.start, args.end, base=args.base, k=k_div, j_param=j_use, i_param=args.i, compact=args.compact_json, divergence_threshold=args.divergence_threshold, alternated=args.alternated, alt_m=args.alt_m, max_iters=args.max_iters)

    out_path = args.out if args.out is not None else os.path.join(run_dir, f"summary_k{k_div}_{args.start}_{args.end}.json")
    save_summary_json(out_path, summary)
    print(f"Summary saved to {out_path}")


__all__ = [name for name in globals().keys() if not name.startswith("_")]
