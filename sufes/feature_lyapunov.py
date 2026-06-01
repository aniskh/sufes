"""Lyapunov feature: compute a Lyapunov-like exponent for a single trajectory.

The feature follows the user's specification:
- For a division step (n % k == 0) we add g_t = ln(1/k) = -ln(k)
- For a multiplication step we add g_t = ln(|multiplier|) where multiplier is
  the effective multiplicative coefficient applied to n (k + i or k + i*factor
  for alternated mode). We support the alternated flag and alt_m parameter.

The run writes a JSON summary file into the provided run_dir and prints a short
report to stdout.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from .algorithms import next_term_ji

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def lyapunov_run(
    n: int,
    k: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    plot: bool = True,
    write_results: bool = True,
    workers: int = 4,
) -> dict:
    """Compute lambda for every starting value n0 in 1..n and write results.

    Returns a summary dict and writes JSON and optional PNG to run_dir.
    """
    os.makedirs(run_dir, exist_ok=True)
    n = int(n)
    k = int(k)
    i = int(i_val)
    j = int(j_val)

    results: List[dict] = []

    def _compute_one(n0: int) -> dict:
        seen = {}
        path = []
        g_list = []
        t = int(n0)
        for step in range(int(max_iters)):
            if t in seen:
                start_idx = seen[t]
                preperiod = start_idx
                sum_g = sum(g_list[:preperiod]) if preperiod > 0 else 0.0
                lam = (sum_g / preperiod) if preperiod > 0 else None
                return {
                    "n": int(n0),
                    "preperiod": preperiod,
                    "lambda": lam,
                    "sum_g": sum_g,
                    "steps": len(path),
                    "reason": "cycle",
                }

            if abs(t) > float(divergence_threshold):
                return {"n": int(n0), "preperiod": None, "lambda": None, "sum_g": None, "steps": len(path), "reason": "divergence_threshold"}

            seen[t] = len(path)
            path.append(int(t))

            if t % k == 0:
                g = -math.log(k)
            else:
                if not alternated:
                    mult = k + i
                else:
                    try:
                        factor = (-alt_m) ** t
                    except OverflowError:
                        factor = 1 if (t % 2 == 0) else -1
                    mult = k + i * factor
                g = math.log(abs(mult)) if mult != 0 else float("-inf")

            g_list.append(g)
            t = next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m)

        return {"n": int(n0), "preperiod": None, "lambda": None, "sum_g": None, "steps": len(path), "reason": "max_iters"}

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

    lambdas = [r.get("lambda") for r in results]

    summary = {
        "n": n,
        "k": k,
        "i": i,
        "j": j,
        "counts": len(results),
        "lambdas": lambdas,
    }

    base = f"lyapunov_upto_n{n}_k{k}_i{i}_j{j}"
    out_summary = os.path.join(run_dir, f"{base}_summary.json")
    out_results = os.path.join(run_dir, f"{base}_results.json")
    try:
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if write_results:
        try:
            with open(out_results, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)
        except Exception:
            pass

    # Plot lambdas vs n if requested and matplotlib available
    if plot and plt is not None:
        try:
            xs = list(range(1, n + 1))
            ys = [float(v) if v is not None else float('nan') for v in lambdas]
            plt.figure(figsize=(10, 4))
            plt.plot(xs, ys, marker='.', linewidth=0.6, markersize=3)
            plt.xlabel('n')
            plt.ylabel('lambda')
            plt.title(f'Lyapunov-like lambda for k={k}, i={i}, j={j} (n=1..{n})')
            plt.grid(True)
            out_plot = os.path.join(run_dir, f"{base}_lambda.png")
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()
        except Exception:
            pass

    print(f"Wrote lyapunov summary to {out_summary}")
    return summary
