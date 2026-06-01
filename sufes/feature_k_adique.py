"""k-adic distribution feature.

This feature computes distributions derived from the *multiplication steps* of a
trajectory for a single (n, k, i, j) combination.

It is exposed in the CLI via:
  --k-adique-n, --k-adique-k, --k-adique-i, --k-adique-j

Outputs written to run_dir:
  - k_adique_n{n}_k{k}_i{i}_j{j}.json
  - k_adique_n{n}_k{k}_i{i}_j{j}.csv  (one row per multiplication event)
  - k_adique_n{n}_k{k}_i{i}_j{j}_residue_percentages.png
  - k_adique_n{n}_k{k}_i{i}_j{j}_valuation_percentages.png
  - k_adique_n{n}_k{k}_i{i}_j{j}_residue_ge2.png

Notes
-----
- We record events only when the rule applies the "multiplication" transform
  (i.e. when t % k != 0). For those events we compute the k-adic valuation of
  the *next* term.
- This is a diagnostic / exploratory tool (plots are best-effort if matplotlib
  is available).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

from .algorithms import next_term_ji


def k_adique_distribution(
    n_val: int,
    k: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    if alt_m >= k:
        raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={alt_m})")
    if i_val < 1 or i_val >= k:
        raise SystemExit(f"i must be in 1..k-1 (got i={i_val}, k={k})")

    def _k_adic_valuation_abs(v: int, base_k: int) -> int:
        av = abs(int(v))
        if av == 0:
            return 0
        e = 0
        while av % base_k == 0:
            av //= base_k
            e += 1
        return e

    os.makedirs(run_dir, exist_ok=True)

    seq: List[int] = []
    seen: Dict[int, int] = {}
    t = int(n_val)
    reason = None
    peak = t
    preperiod = None

    # Records only for multiplication events
    mult_steps: List[Dict[str, int]] = []
    residue_to_vals: Dict[int, List[int]] = {}

    for step in range(max_iters):
        if int(t) > peak:
            peak = int(t)
        if t in seen:
            preperiod = seen[t]
            reason = "cycle"
            break
        seen[t] = len(seq)
        seq.append(int(t))
        if abs(t) > divergence_threshold:
            reason = "divergence_threshold"
            break

        r = int(abs(t) % k)
        if r != 0:
            next_t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
            nu_next = _k_adic_valuation_abs(next_t, k)
            mult_steps.append(
                {
                    "step": int(step),
                    "t_before": int(t),
                    "residue": int(r),
                    "t_after": int(next_t),
                    "valuation": int(nu_next),
                }
            )
            residue_to_vals.setdefault(int(r), []).append(int(nu_next))
            t = int(next_t)
        else:
            t = int(t) // int(k)
    else:
        reason = "max_iters"

    total_mult = len(mult_steps)

    residue_distribution: List[Dict[str, object]] = []
    valuation_distribution: List[Dict[str, object]] = []
    if total_mult:
        residue_counts = {rr: len(residue_to_vals.get(rr, [])) for rr in range(1, k)}
        for rr in range(1, k):
            residue_distribution.append(
                {
                    "residue": int(rr),
                    "count": int(residue_counts.get(rr, 0)),
                    "percentage": float(residue_counts.get(rr, 0)) * 100.0 / float(total_mult),
                }
            )

        val_counts: Dict[int, int] = {}
        for ev in mult_steps:
            vv = int(ev.get("valuation", 0))
            val_counts[vv] = val_counts.get(vv, 0) + 1
        for vv in sorted(val_counts.keys()):
            valuation_distribution.append(
                {
                    "valuation": int(vv),
                    "count": int(val_counts[vv]),
                    "percentage": float(val_counts[vv]) * 100.0 / float(total_mult),
                }
            )

    residue_ge2_rows = []
    for rkey in range(1, k):
        vals = residue_to_vals.get(rkey, [])
        cnt = len(vals)
        ge2 = sum(1 for v in vals if v >= 2)
        pct_ge2 = (float(ge2) * 100.0 / float(cnt)) if cnt else 0.0
        residue_ge2_rows.append(
            {
                "residue": int(rkey),
                "count": int(cnt),
                "ge2": int(ge2),
                "pct_ge2": float(pct_ge2),
            }
        )

    best_residue = None
    if residue_ge2_rows:
        best_residue = max(residue_ge2_rows, key=lambda row: float(row.get("pct_ge2", 0.0)))

    payload = {
        "n": int(n_val),
        "k": int(k),
        "i": int(i_val),
        "j": int(j_val),
        "reason": str(reason),
        "steps": int(len(seq)),
        "preperiod": None if preperiod is None else int(preperiod),
        "peak": int(peak),
        "total_multiplication_events": int(total_mult),
        "sample_multiplication_events": mult_steps[:200],
        "residue_distribution": residue_distribution,
        "valuation_distribution": valuation_distribution,
        "residue_ge2": residue_ge2_rows,
        "best_residue_ge2": best_residue,
    }

    json_path = os.path.join(run_dir, f"k_adique_n{n_val}_k{k}_i{i_val}_j{j_val}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)

    csv_path = os.path.join(run_dir, f"k_adique_n{n_val}_k{k}_i{i_val}_j{j_val}.csv")
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write(f"# n={n_val},k={k},i={i_val},j={j_val},reason={reason}\n")
        cf.write("step,t_before,residue,t_after,valuation\n")
        for ev in mult_steps:
            cf.write(
                f"{ev['step']},{ev['t_before']},{ev['residue']},{ev['t_after']},{ev['valuation']}\n"
            )

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None

    if plt is None or not total_mult:
        return

    # Residue distribution bar
    try:
        fig, ax = plt.subplots(figsize=(9, 4))
        x_all = [int(r["residue"]) for r in residue_distribution]
        y_all = [float(r["percentage"]) for r in residue_distribution]
        ax.bar(x_all, y_all, color="C0", alpha=0.85)
        ax.set_xticks(x_all)
        ax.set_xlabel("residue")
        ax.set_ylabel("percentage among multiplication events (%)")
        ax.set_title(f"k-adic: residue percentages (multiplication steps) n={n_val} k={k} i={i_val} j={j_val}")
        fig.tight_layout()
        res_png = os.path.join(
            run_dir, f"k_adique_n{n_val}_k{k}_i{i_val}_j{j_val}_residue_percentages.png"
        )
        fig.savefig(res_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass

    # Valuation distribution bar
    try:
        if valuation_distribution:
            fig, ax = plt.subplots(figsize=(9, 4))
            x_val = [int(v["valuation"]) for v in valuation_distribution]
            y_val = [float(v["percentage"]) for v in valuation_distribution]
            ax.bar(x_val, y_val, color="C2", alpha=0.85)
            ax.set_xticks(x_val)
            ax.set_xlabel("k-adic valuation (e)")
            ax.set_ylabel("percentage among multiplication events (%)")
            ax.set_title(f"k-adic: valuation percentages n={n_val} k={k} i={i_val} j={j_val}")
            fig.tight_layout()
            val_png = os.path.join(
                run_dir, f"k_adique_n{n_val}_k{k}_i{i_val}_j{j_val}_valuation_percentages.png"
            )
            fig.savefig(val_png, dpi=150)
            plt.close(fig)
    except Exception:
        pass

    # Residue >= 2 percentage
    try:
        fig, ax = plt.subplots(figsize=(9, 4))
        x_r = [int(r["residue"]) for r in residue_ge2_rows]
        y_r = [float(r["pct_ge2"]) for r in residue_ge2_rows]
        ax.bar(x_r, y_r, color="C3", alpha=0.85)
        ax.set_xticks(x_r)
        ax.set_xlabel("residue")
        ax.set_ylabel("P(valuation >= 2 | residue) among mult steps (%)")
        ax.set_title(f"k-adic: residue -> valuation>=2 (multiplication steps) n={n_val} k={k} i={i_val} j={j_val}")
        fig.tight_layout()
        ge2_png = os.path.join(
            run_dir, f"k_adique_n{n_val}_k{k}_i{i_val}_j{j_val}_residue_ge2.png"
        )
        fig.savefig(ge2_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass
