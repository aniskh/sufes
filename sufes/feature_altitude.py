"""Feature: altitude

Compute the maximum value (peak) reached by the trajectory starting from n
under the (k,i,j) recurrence. The function simulates the sequence until a
cycle is detected, divergence threshold is exceeded, or max_iters is reached.

Outputs written in ``run_dir``:
- ``altitude_n{n}_k{k}_i{i}_j{j}_summary.json`` with fields {n,k,i,j,peak,steps,reason}
- ``altitude_n{n}_k{k}_i{i}_j{j}_sequence.json`` the visited sequence (may be truncated)

The function returns the summary dict.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .algorithms import next_term_ji

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def altitude(
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
    partitionning: bool = False,
    # When called in p-mode (many k), callers can disable writing the
    # per-k detailed results JSON and individual per-k plots to save time
    # and disk. By default keep previous behavior (write & plot).
    write_results: bool = True,
    plot_individual: bool = False,
    workers: int = 4,
) -> dict:
    """Compute the peak (max) value for every start value n0 in 1..n.

    Writes a summary JSON with arrays of peaks and a detailed results file.
    Returns a summary dict with peaks array.
    """
    os.makedirs(run_dir, exist_ok=True)
    n = int(n)
    k = int(k)
    i = int(i)
    j = int(j)

    results: List[dict] = []

    def _compute_one(n0: int):
        t = int(n0)
        pos = {}
        peak = int(n0)
        # distance_to_altitude: number of steps before first reaching the peak
        distance_to_alt = 0
        reason = "max_iters"
        steps = 0
        path: List[int] = []

        for step in range(int(max_iters)):
            if t in pos:
                reason = "cycle"
                steps = step
                break
            if abs(t) > float(divergence_threshold):
                reason = "divergence_threshold"
                steps = step
                break
            pos[t] = step
            path.append(int(t))
            if int(t) > peak:
                peak = int(t)
                # record the first step index where the peak was observed
                distance_to_alt = int(step)

            nxt = next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m)
            t = int(nxt)
        else:
            steps = int(max_iters)

        return {
            "n": int(n0),
            "peak": int(peak),
            "steps": int(steps),
            "reason": reason,
            "sequence": path,
            "distance_to_altitude": int(distance_to_alt),
        }

    ns = list(range(1, n + 1))
    if int(workers) <= 1:
        for r in map(_compute_one, ns):
            results.append(r)
    else:
        try:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for r in executor.map(_compute_one, ns):
                    results.append(r)
        except Exception:
            for r in map(_compute_one, ns):
                results.append(r)

    peaks = [r.get("peak") for r in results]
    distances = [r.get("distance_to_altitude", 0) for r in results]
    mean_peak = (sum(peaks) / len(peaks)) if peaks else None
    mean_distance = (sum(distances) / len(distances)) if distances else None

    # per-n combined average: (peak + distance_to_altitude) / 2 for each n'
    combined = []
    for p, d in zip(peaks, distances):
        try:
            combined.append((float(p) + float(d)) / 2.0)
        except Exception:
            combined.append(None)
    mean_combined = (sum([v for v in combined if v is not None]) / len([v for v in combined if v is not None])) if combined else None

    summary = {
        "n": int(n),
        "k": int(k),
        "i": int(i),
        "j": int(j),
        "counts": len(results),
        "peaks": peaks,
        "mean_peak": mean_peak,
        "distances_to_altitude": distances,
        "mean_distance_to_altitude": mean_distance,
        "combined_peak_distance": combined,
        "mean_peak_and_distance": mean_combined,
    }

    # If partitionning requested: group n' by remainder mod k and compute per-partition stats
    if partitionning:
        partitions = {}
        for rem in range(0, k):
            partitions[rem] = {"peaks": [], "distances": [], "mean_peak": None, "mean_distance": None}
        for r in results:
            n0 = int(r.get("n"))
            rem = int(n0 % k)
            partitions[rem]["peaks"].append(int(r.get("peak", 0)))
            partitions[rem]["distances"].append(int(r.get("distance_to_altitude", 0)))
        for rem in partitions:
            ps = partitions[rem]["peaks"]
            ds = partitions[rem]["distances"]
            partitions[rem]["mean_peak"] = (sum(ps) / len(ps)) if ps else None
            partitions[rem]["mean_distance"] = (sum(ds) / len(ds)) if ds else None
        summary["partitions"] = partitions

    base = f"altitude_upto_n{n}_k{k}_i{i}_j{j}"
    out_summary = os.path.join(run_dir, f"{base}_summary.json")
    out_results = os.path.join(run_dir, f"{base}_results.json")
    try:
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    # Optionally skip writing the large detailed results.json (sequences)
    if write_results:
        try:
            with open(out_results, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        except Exception:
            pass

    # Optional plotting: peaks vs n
    if plt is not None and plot_individual:
        try:
            xs = list(range(1, n + 1))
            ys = [float(v) for v in peaks]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, ys, marker='.', linewidth=0.7, markersize=3)
            plt.xlabel('n')
            plt.ylabel('peak')
            plt.title(f'Peak values for k={k}, i={i}, j={j} (n=1..{n})')
            plt.grid(True)
            out_plot = os.path.join(run_dir, f"{base}_peaks.png")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
        except Exception:
            # plotting optional
            pass

    # Optional plotting: peak value distribution (counts per peak) for this k
    if plt is not None and plot_individual:
        try:
            from collections import Counter
            peaks_vals = [int(v) for v in peaks if v is not None]
            if peaks_vals:
                counts = Counter(peaks_vals)
                items = sorted(counts.items(), key=lambda t: int(t[0]))
                xs = [int(t[0]) for t in items]
                ys = [int(t[1]) for t in items]
                plt.figure(figsize=(10, 4))
                plt.bar(xs, ys, width=1.0)
                if len(xs) > 20:
                    step = max(1, len(xs) // 20)
                    plt.xticks(xs[::step], rotation=45)
                else:
                    plt.xticks(xs, rotation=45)
                plt.xlabel('peak value')
                plt.ylabel('count')
                plt.title(f'Peak value distribution for k={k}, i={i}, j={j} (n=1..{n})')
                plt.tight_layout()
                out_plot = os.path.join(run_dir, f"{base}_peak_distribution.png")
                plt.savefig(out_plot)
                plt.close()
        except Exception:
            pass

    # Optional plotting: distance_to_altitude vs n
    if plt is not None and plot_individual:
        try:
            xs = list(range(1, n + 1))
            ys = [float(v) for v in distances]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, ys, marker='.', linewidth=0.7, markersize=3)
            plt.xlabel('n')
            plt.ylabel('distance_to_altitude')
            plt.title(f'Distance to altitude for k={k}, i={i}, j={j} (n=1..{n})')
            plt.grid(True)
            out_plot = os.path.join(run_dir, f"{base}_distance_to_altitude.png")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
        except Exception:
            # plotting optional
            pass

    # Optional plotting: per-partition line plots (if partitionning requested)
    if partitionning and plt is not None and plot_individual:
        try:
            # Peaks by partition: overlay one line per remainder (x = index within partition)
            plt.figure(figsize=(10, 5))
            for rem in range(0, k):
                ps = summary.get("partitions", {}).get(rem, {}).get("peaks", [])
                if not ps:
                    continue
                xs = list(range(1, len(ps) + 1))
                ys = [float(v) for v in ps]
                plt.plot(xs, ys, label=f"rem={rem}", linewidth=0.8)
            plt.xlabel('index in partition (ordered by n)')
            plt.ylabel('peak')
            plt.title(f'Peaks by partition (k={k}, i={i}, j={j})')
            plt.legend(fontsize=8, ncol=min(4, k))
            plt.grid(True)
            out_plot = os.path.join(run_dir, f"{base}_peaks_by_partition.png")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
        except Exception:
            pass

        try:
            # Distances by partition
            plt.figure(figsize=(10, 5))
            for rem in range(0, k):
                ds = summary.get("partitions", {}).get(rem, {}).get("distances", [])
                if not ds:
                    continue
                xs = list(range(1, len(ds) + 1))
                ys = [float(v) for v in ds]
                plt.plot(xs, ys, label=f"rem={rem}", linewidth=0.8)
            plt.xlabel('index in partition (ordered by n)')
            plt.ylabel('distance_to_altitude')
            plt.title(f'Distance to altitude by partition (k={k}, i={i}, j={j})')
            plt.legend(fontsize=8, ncol=min(4, k))
            plt.grid(True)
            out_plot = os.path.join(run_dir, f"{base}_distance_by_partition.png")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
        except Exception:
            pass

    return summary


__all__ = ["altitude"]
