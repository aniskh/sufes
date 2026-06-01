"""kernel feature.

Given a prime k and parameters (i,j), compute the usual trajectory/run summary
for all starting values n with 1 <= n <= 2*k-1.

This uses the same simulation logic as other features (next_term_ji loop with
cycle detection) and respects the global iteration limits.

Outputs (written into run_dir):
- kernel_k{K}_i{I}_j{J}.csv : one row per n with summary fields
- kernel_k{K}_i{I}_j{J}.json: aggregate summary + per-n summaries

Parallelism:
- workers <= 1: sequential
- workers > 1: ThreadPoolExecutor(max_workers=workers)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from .algorithms import next_term_ji


@dataclass(frozen=True)
class KernelRow:
    n: int
    reason: str
    steps: int
    preperiod: Optional[int]
    cycle_length: Optional[int]
    peak: int
    # lightweight per-run samples for plotting/annotation
    seq_sample: List[int]
    cycle_values: Optional[List[int]]


def _simulate_kernel_single(
    n_val: int,
    k: int,
    i_val: int,
    j_val: int,
    *,
    max_iters: int,
    divergence_threshold: float,
    alternated: bool,
    alt_m: int,
) -> KernelRow:
    seen: Dict[int, int] = {}
    t = int(n_val)
    peak = int(t)
    seq: List[int] = []

    for step in range(int(max_iters)):
        seq.append(int(t))
        if int(t) > peak:
            peak = int(t)

        if t in seen:
            start = int(seen[t])
            # We just appended the current t to seq above.
            # If we see t again, the repeating cycle is the slice seq[start:step]
            # (excluding the repeated value at position 'step').
            cycle_vals = seq[start:step]
            cycle_len = int(len(cycle_vals))
            # canonicalize by rotation for stable string/compare
            if cycle_vals:
                rots = [cycle_vals[ii:] + cycle_vals[:ii] for ii in range(len(cycle_vals))]
                cycle_vals = min(rots)
            return KernelRow(
                n=int(n_val),
                reason="cycle",
                steps=int(step),
                preperiod=int(start),
                cycle_length=int(cycle_len),
                peak=int(peak),
                seq_sample=seq,
                cycle_values=cycle_vals,
            )
        seen[t] = int(step)

        if abs(int(t)) > float(divergence_threshold):
            return KernelRow(
                n=int(n_val),
                reason="divergence_threshold",
                steps=int(step),
                preperiod=None,
                cycle_length=None,
                peak=int(peak),
                seq_sample=seq,
                cycle_values=None,
            )

        t = next_term_ji(int(t), int(k), int(j_val), int(i_val), alternated=bool(alternated), alt_m=int(alt_m))

    return KernelRow(
        n=int(n_val),
        reason="max_iters",
        steps=int(max_iters),
        preperiod=None,
        cycle_length=None,
        peak=int(peak),
        seq_sample=seq,
        cycle_values=None,
    )


def kernel(
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
) -> Dict[str, object]:
    """Run the kernel feature for all 1<=n<=2*k-1.

    Returns a JSON-serializable summary dict (also written to disk).
    """

    if int(k) < 2:
        raise SystemExit(f"k must be >= 2 (got {k})")
    if int(alt_m) >= int(k):
        raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={alt_m})")
    if int(i_val) < 1 or int(i_val) >= int(k):
        raise SystemExit(f"i must be in 1..k-1 (got i={i_val}, k={k})")

    os.makedirs(run_dir, exist_ok=True)

    n_values = list(range(1, 2 * int(k)))

    def _task(n0: int) -> KernelRow:
        return _simulate_kernel_single(
            n0,
            int(k),
            int(i_val),
            int(j_val),
            max_iters=int(max_iters),
            divergence_threshold=float(divergence_threshold),
            alternated=bool(alternated),
            alt_m=int(alt_m),
        )

    rows: List[KernelRow] = []
    if int(workers) <= 1:
        rows = [_task(n0) for n0 in n_values]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(workers)) as ex:
            rows = list(ex.map(_task, n_values))

    # Aggregate stats
    total = len(rows)
    count_cycle = sum(1 for r in rows if r.reason == "cycle")
    count_fail = total - count_cycle
    max_steps = max((r.steps for r in rows), default=0)
    max_peak = max((r.peak for r in rows), default=0)
    cycle_lengths = [int(r.cycle_length) for r in rows if r.cycle_length is not None]
    unique_cycle_lengths = sorted(set(cycle_lengths))

    payload: Dict[str, object] = {
        "k": int(k),
        "i": int(i_val),
        "j": int(j_val),
    "range": {"start": 1, "end": 2 * int(k) - 1, "count": int(total)},
        "params": {
            "max_iters": int(max_iters),
            "divergence_threshold": float(divergence_threshold),
            "alternated": bool(alternated),
            "alt_m": int(alt_m),
            "workers": int(workers),
        },
        "summary": {
            "count_cycle": int(count_cycle),
            "count_failed": int(count_fail),
            "max_steps": int(max_steps),
            "max_peak": int(max_peak),
            "unique_cycle_lengths": unique_cycle_lengths,
        },
        "rows": [
            {
                "n": int(r.n),
                "reason": str(r.reason),
                "steps": int(r.steps),
                "preperiod": int(r.preperiod) if r.preperiod is not None else None,
                "cycle_length": int(r.cycle_length) if r.cycle_length is not None else None,
                "cycle": r.cycle_values,
                "peak": int(r.peak),
            }
            for r in rows
        ],
    }

    out_base = f"kernel_k{int(k)}_i{int(i_val)}_j{int(j_val)}"
    out_csv = os.path.join(run_dir, f"{out_base}.csv")
    out_json = os.path.join(run_dir, f"{out_base}.json")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "reason", "steps", "preperiod", "cycle_length", "cycle", "peak"])
        for r in rows:
            w.writerow([
                int(r.n),
                str(r.reason),
                int(r.steps),
                int(r.preperiod) if r.preperiod is not None else "",
                int(r.cycle_length) if r.cycle_length is not None else "",
                json.dumps(r.cycle_values, ensure_ascii=False) if r.cycle_values is not None else "",
                int(r.peak),
            ])

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Optional: combined subplot image (one subplot per n)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    if plt is not None:
        try:
            nplots = len(rows)
            if nplots > 0:
                # Grid close to square
                cols = int(max(1, round(nplots ** 0.5)))
                rows_grid = int((nplots + cols - 1) // cols)

                fig, axes = plt.subplots(rows_grid, cols, figsize=(4.5 * cols, 2.6 * rows_grid), squeeze=False)
                axes_list = [axes[r][c] for r in range(rows_grid) for c in range(cols)]

                for ax, rr in zip(axes_list, rows):
                    xs = list(range(len(rr.seq_sample)))
                    ax.plot(xs, rr.seq_sample, linewidth=1)
                    title = f"n={rr.n} steps={rr.steps}"
                    ax.set_title(title)
                    ax.set_xlabel("step")
                    ax.set_ylabel("value")

                    cycle_txt = ""
                    if rr.reason == "cycle" and rr.preperiod is not None and rr.cycle_length is not None:
                        # keep subplot annotations compact
                        cyc_show = None
                        if rr.cycle_values is not None:
                            if len(rr.cycle_values) <= 12:
                                cyc_show = str(rr.cycle_values)
                            else:
                                cyc_show = str(rr.cycle_values[:12])[:-1] + ", ...]"
                        if cyc_show is None:
                            cycle_txt = f"cycle_start={rr.preperiod}  cycle_len={rr.cycle_length}"
                        else:
                            cycle_txt = f"cycle_start={rr.preperiod}  cycle_len={rr.cycle_length}\n{cyc_show}"
                        # mark cycle start
                        if 0 <= int(rr.preperiod) < len(rr.seq_sample):
                            ax.axvline(int(rr.preperiod), color="C3", linestyle="--", linewidth=1)
                    else:
                        cycle_txt = rr.reason

                    ax.text(
                        0.02,
                        0.98,
                        f"{cycle_txt}\npeak={rr.peak}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, linewidth=0.0),
                    )

                # hide unused axes
                for ax in axes_list[len(rows) :]:
                    ax.axis("off")

                fig.suptitle(f"kernel trajectories (k={int(k)} i={int(i_val)} j={int(j_val)})")
                fig.tight_layout(rect=[0, 0.02, 1, 0.97])

                out_png = os.path.join(run_dir, f"{out_base}_subplots.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
                print(f"Wrote: {out_png}")
        except Exception:
            # plotting should never break the feature
            pass

    print(
        f"kernel: k={int(k)} i={int(i_val)} j={int(j_val)} "
        f"n=1..{2*int(k)-1} cycles={int(count_cycle)}/{int(total)} "
        f"unique_cycle_lengths={len(unique_cycle_lengths)}"
    )
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")

    return payload


__all__ = ["kernel", "KernelRow"]
