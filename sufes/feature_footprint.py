"""Feature: footprint

Runs the generalized recurrence for all start values n=1..N with given (k,i,j)
and records all visited nodes across all trajectories. Produces a report indicating
whether the interval 1..N^2 is fully covered by the union of visited nodes, and
if not, computes the maximal prefix S such that all integers 1..S have been seen
at least once in some trajectory.

Outputs:
- `{base}_visited.json` : list of visited integers (may be large)
- `{base}_visited_hist.png` : histogram of visited node distribution
- `{base}_summary.json` : summary with fields {N,k,i,j,S,max_seen,total_unique_visited,...}


"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Set

from .algorithms import next_term_ji


def footprint(
    N: int,
    k: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    do_plot: bool = True,
    p: int | None = None,
    n_multiple_k: int | None = None,
    workers: int = 4,
) -> dict:
    os.makedirs(run_dir, exist_ok=True)
    N = int(N)
    k = int(k)
    i = int(i)
    j = int(j)

    # If the caller provided n_multiple_k, compute the effective N as
    # n_multiple_k * k (inclusive).
    if n_multiple_k is not None:
        try:
            mult = int(n_multiple_k)
            N = mult * k
        except Exception:
            pass

    def _walk_one(n0: int):
        local_visited: Set[int] = set()
        local_max = int(n0)

        t = int(n0)
        local_seen = set()
        for _ in range(int(max_iters)):
            if abs(t) > float(divergence_threshold):
                break
            local_visited.add(int(t))
            if int(t) > local_max:
                local_max = int(t)
            if int(t) in local_seen:
                break
            local_seen.add(int(t))
            t = next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m)

        return local_visited, int(local_max)

    visited: Set[int] = set()
    max_seen = 0
    ns = list(range(1, N + 1))
    if int(workers) <= 1:
        for loc_set, loc_max in map(_walk_one, ns):
            visited.update(loc_set)
            if loc_max > max_seen:
                max_seen = int(loc_max)
    else:
        try:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for loc_set, loc_max in executor.map(_walk_one, ns):
                    visited.update(loc_set)
                    if loc_max > max_seen:
                        max_seen = int(loc_max)
        except Exception:
            for loc_set, loc_max in map(_walk_one, ns):
                visited.update(loc_set)
                if loc_max > max_seen:
                    max_seen = int(loc_max)

    # compute maximal prefix F such that all 1..F are visited
    if visited:
        max_visited = max(visited)
    else:
        max_visited = 0
    F = 0
    for s in range(1, int(max_visited) + 1):
        if s in visited:
            F = s
            continue
        break

    summary = {
        "N": N,
        "k": k,
        "i": i,
        "j": j,
        # New naming: F(N) is the maximal covered prefix.
        "F": int(F),
        # Backward compat (older outputs used "S").
        "S": int(F),
        "max_seen": int(max_seen),
        "total_unique_visited": int(len(visited)),
    }

    # Ratio of unique visited nodes t such that N < t <= 2N, divided by N.
    if N > 0 and visited:
        count_in_band = sum(1 for t in visited if (t > N and t <= 2 * N))
        summary["visited_ratio_N_2N"] = float(count_in_band) / float(N)
        summary["visited_count_N_2N"] = int(count_in_band)
    else:
        summary["visited_ratio_N_2N"] = 0.0
        summary["visited_count_N_2N"] = 0

    base = f"footprint_N{N}_k{k}_i{i}_j{j}"
    out_summary = os.path.join(run_dir, f"{base}_summary.json")
    out_visited = os.path.join(run_dir, f"{base}_visited.json")
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # If F > 0, find which starting n in 1..N had F appear in their trajectory.
    origins = []
    if F > 0:

        def _contains_F(n0: int) -> int | None:
            t = int(n0)
            local_seen = set()
            for _ in range(int(max_iters)):
                if abs(t) > float(divergence_threshold):
                    break
                if int(t) == int(F):
                    return int(n0)
                if int(t) in local_seen:
                    break
                local_seen.add(int(t))
                t = next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m)
            return None

        if int(workers) <= 1:
            for v in map(_contains_F, ns):
                if v is not None:
                    origins.append(int(v))
        else:
            try:
                with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                    for v in executor.map(_contains_F, ns):
                        if v is not None:
                            origins.append(int(v))
            except Exception:
                for v in map(_contains_F, ns):
                    if v is not None:
                        origins.append(int(v))

        out_F_origins = os.path.join(run_dir, f"{base}_F_origins.json")
        # Backward compat: keep writing the legacy filename too.
        out_S_origins = os.path.join(run_dir, f"{base}_S_origins.json")
        try:
            with open(out_F_origins, "w", encoding="utf-8") as f:
                json.dump(origins, f, ensure_ascii=False)
            with open(out_S_origins, "w", encoding="utf-8") as f:
                json.dump(origins, f, ensure_ascii=False)
        except Exception:
            pass

        summary["F_origins_count"] = int(len(origins))
        summary["F_origins"] = list(origins)
        # Backward compat aliases
        summary["S_origins_count"] = int(len(origins))
        summary["S_origins"] = list(origins)
        try:
            with open(out_summary, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    try:
        with open(out_visited, "w", encoding="utf-8") as f:
            json.dump(sorted(list(visited)), f, ensure_ascii=False)
    except Exception:
        pass

    if do_plot:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            xs = sorted(list(visited))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(xs, bins=200)
            ax.set_title(f"Footprint visited nodes — N={N}, k={k}, i={i}, j={j}")
            ax.set_xlabel("node value")
            ax.set_ylabel("count in bin")

            # Overlay N/x
            try:
                import numpy as np  # type: ignore

                if xs:
                    x_min = max(1, xs[0])
                    x_max = xs[-1]
                else:
                    x_min = 1
                    x_max = max(1, N)
                x_vals = np.linspace(x_min, x_max, 800)
                y_vals = float(N) / x_vals
                ax2 = ax.twinx()
                ax2.plot(x_vals, y_vals, color="crimson", linewidth=1.2, label="N/x")
                ax2.set_ylabel("N/x", color="crimson")
                ax2.tick_params(axis="y", labelcolor="crimson")
                try:
                    ax2.legend(loc="upper right")
                except Exception:
                    pass
            except Exception:
                pass

            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"{base}_visited_hist.png"), dpi=200)
            plt.close(fig)
        except Exception:
            pass

    return summary


__all__ = ["footprint"]
