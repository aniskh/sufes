"""single_n_overall feature extracted from the legacy monolithic residue implementation.

This module is intentionally self-contained to avoid circular imports.

Public API:
- single_n_overall(...)

It simulates one (n,k,i,j) trajectory, computes residue/valuation stats, and
writes CSV+JSON+PNGs into the run directory.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .algorithms import next_term_ji
from .stats import shannon_entropy_non_zero_residues, skewness_non_zero_residues


def single_n_overall(
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

    # Historically we restricted i to 1..k-1. For exploratory runs we allow any
    # integer i and reduce it mod k for the actual iteration rule.
    #
    # We map into 1..k-1 (never 0) so the rule stays non-degenerate.
    i_requested = int(i_val)
    i_eff = int(i_requested) % int(k)
    if i_eff == 0:
        i_eff = 1

    def _k_adic_valuation_abs(v: int, base_k: int) -> int:
        av = abs(int(v))
        if av == 0:
            return 0
        e = 0
        while av % base_k == 0:
            av //= base_k
            e += 1
        return e

    def _percent_distribution(
        keys: List[int], counts_map: Dict[int, int], total_count: int
    ) -> Tuple[List[Dict[str, object]], float]:
        rows: List[Dict[str, object]] = []
        if total_count <= 0:
            for key in keys:
                rows.append({"residue": int(key), "count": 0, "percentage": 0.0})
            return rows, 0.0

        running = 0.0
        for idx, key in enumerate(keys):
            cnt = int(counts_map.get(key, 0))
            if idx < len(keys) - 1:
                pct = float(cnt) * 100.0 / float(total_count)
                running += float(pct)
            else:
                pct = 100.0 - running
            rows.append({"residue": int(key), "count": cnt, "percentage": float(pct)})
        return rows, float(sum(float(r["percentage"]) for r in rows))

    os.makedirs(run_dir, exist_ok=True)
    seq: List[int] = []
    residues: List[int] = []
    seen: Dict[int, int] = {}
    t = int(n_val)
    reason: Optional[str] = None
    peak = t
    preperiod: Optional[int] = None

    for step in range(max_iters):
        seq.append(int(t))
        r = int(abs(t) % k)
        residues.append(int(r))
        if int(t) > peak:
            peak = int(t)
        if t in seen:
            reason = "cycle"
            preperiod = int(seen[t])
            break
        seen[t] = int(step)
        if abs(t) > divergence_threshold:
            reason = "divergence_threshold"
            preperiod = None
            break
        t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
    else:
        reason = "max_iters"
        preperiod = None

    is_divisible_by_k: List[int] = [1 if (abs(v) % k == 0) else 0 for v in seq]
    k_adic_valuations: List[int] = [_k_adic_valuation_abs(v, k) for v in seq]

    count_total = int(len(residues))
    count_divisible = int(sum(is_divisible_by_k))
    count_divisible_k2_or_more = int(sum(1 for nu in k_adic_valuations if int(nu) >= 2))
    count_non_zero = int(sum(1 for r in residues if int(r) != 0))

    lambda_divisible = (float(sum(k_adic_valuations)) / float(count_total)) if count_total else None
    valuations_non_zero = [val for val, r in zip(k_adic_valuations, residues) if int(r) != 0]
    lambda_divisible_non_zero = (
        float(sum(valuations_non_zero)) / float(len(valuations_non_zero)) if valuations_non_zero else None
    )
    gamma = (float(count_divisible_k2_or_more) / float(count_divisible)) if count_divisible > 0 else None

    sum_non_zero = int(sum(int(r) for r in residues if int(r) != 0))
    sum_total = int(sum(int(r) for r in residues))
    mean_non_zero = (float(sum_non_zero) / float(count_non_zero)) if count_non_zero else None
    mean_total = (float(sum_total) / float(count_total)) if count_total else None

    var_non_zero = None
    std_non_zero = None
    if count_non_zero and mean_non_zero is not None:
        ssd = sum((float(r) - float(mean_non_zero)) ** 2 for r in residues if int(r) != 0)
        var_non_zero = float(ssd) / float(count_non_zero)
        std_non_zero = math.sqrt(var_non_zero)

    var_total = None
    std_total = None
    if count_total and mean_total is not None:
        ssd_total = sum((float(r) - float(mean_total)) ** 2 for r in residues)
        var_total = float(ssd_total) / float(count_total)
        std_total = math.sqrt(var_total)

    counts_all: Dict[int, int] = {rr: 0 for rr in range(k)}
    for r in residues:
        counts_all[int(r)] = counts_all.get(int(r), 0) + 1
    residue_distribution, residue_percent_sum = _percent_distribution(list(range(k)), counts_all, count_total)

    counts_non_zero_map: Dict[int, int] = {rr: 0 for rr in range(1, k)}
    for r in residues:
        if int(r) != 0:
            counts_non_zero_map[int(r)] = counts_non_zero_map.get(int(r), 0) + 1
    non_zero_residue_distribution, non_zero_residue_percent_sum = _percent_distribution(
        list(range(1, k)), counts_non_zero_map, count_non_zero
    )

    H, Hmax, Hratio = shannon_entropy_non_zero_residues(counts_non_zero_map, int(k))
    Hdelta = (float(Hmax) - float(H)) if (H is not None and Hmax is not None) else None

    skew_non_zero = skewness_non_zero_residues(counts_non_zero_map, int(k))

    cycle_info = None
    if reason == "cycle" and preperiod is not None and preperiod < len(seq):
        cycle_vals = seq[preperiod:]
        cycle_res = residues[preperiod:]
        cycle_info = {
            "start_index": preperiod,
            "length": len(cycle_vals),
            "values": cycle_vals,
            "residues": cycle_res,
        }

    payload = {
        "n": n_val,
        "k": k,
        "i": int(i_eff),
        "i_requested": int(i_requested),
        "i_effective": int(i_eff),
        "j": j_val,
        "reason": reason,
        "steps": len(seq),
        "preperiod": preperiod,
        "peak": peak,
        "sequence_sample": seq[:200],
        "residues_sample": residues[:200],
        "cycle": cycle_info,
        "residue_stats": {
            "count_total": count_total,
            "count_divisible": count_divisible,
            "count_divisible_k2_or_more": count_divisible_k2_or_more,
            "gamma": gamma,
            "count_non_zero": count_non_zero,
            "lambda_divisible": lambda_divisible,
            "lambda_divisible_non_zero": lambda_divisible_non_zero,
            "mean_non_zero": mean_non_zero,
            "mean_total": mean_total,
            "var_non_zero": var_non_zero,
            "std_non_zero": std_non_zero,
            "skew_non_zero": skew_non_zero,
            "var_total": var_total,
            "std_total": std_total,
            "residue_distribution": residue_distribution,
            "residue_percent_sum": residue_percent_sum,
            "non_zero_residue_distribution": non_zero_residue_distribution,
            "non_zero_residue_percent_sum": non_zero_residue_percent_sum,
            "shannon_entropy_non_zero": H,
            "shannon_entropy_non_zero_max": Hmax,
            "shannon_entropy_non_zero_ratio": Hratio,
            "shannon_entropy_non_zero_delta": Hdelta,
        },
    }

    json_path = os.path.join(run_dir, f"single_overall_n{n_val}_k{k}_i{i_eff}_j{j_val}.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, ensure_ascii=False, indent=2)

    # Detailed CSV
    csv_path = os.path.join(run_dir, f"single_overall_n{n_val}_k{k}_i{i_eff}_j{j_val}.csv")
    with open(csv_path, "w", encoding="utf-8") as cf:
        cf.write(f"# n={n_val},k={k},i_requested={i_requested},i_effective={i_eff},j={j_val},reason={reason}\n")
        cf.write(f"# count_total={count_total},count_divisible={count_divisible},count_non_zero={count_non_zero}\n")
        cf.write(f"# count_divisible_k2_or_more={count_divisible_k2_or_more},gamma={gamma}\n")
        cf.write(f"# mean_non_zero={mean_non_zero},var_non_zero={var_non_zero},std_non_zero={std_non_zero},skew_non_zero={skew_non_zero}\n")
        cf.write(f"# shannon_entropy_non_zero={H},Hmax={Hmax},ratio={Hratio},delta={Hdelta}\n")
        if cycle_info is not None:
            cf.write(f"# cycle_start={cycle_info.get('start_index')} cycle_length={cycle_info.get('length')}\n")
        cf.write("step,value,remainder,is_divisible_by_k,k_adic_valuation\n")
        for step, (val, rem, div_flag, nu) in enumerate(zip(seq, residues, is_divisible_by_k, k_adic_valuations)):
            cf.write(f"{step},{val},{rem},{div_flag},{nu}\n")

    # print summary (captured in run.log)
    print("Single overall summary:")
    print(f" n={n_val} k={k} i={i_val} j={j_val} reason={reason}")
    print(f" count_total={count_total} count_divisible={count_divisible} count_non_zero={count_non_zero}")
    print(f" mean_non_zero={mean_non_zero} std_non_zero={std_non_zero} skew_non_zero={skew_non_zero}")
    print(f" shannon_entropy_non_zero={H} Hmax={Hmax} ratio={Hratio} delta={Hdelta}")

    # optional plots
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is None:
        return

    try:
        # trajectory
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(list(range(len(seq))), seq, linewidth=1)
        ax.set_title(f"trajectory n={n_val} k={k} i={i_val} j={j_val}")
        ax.set_xlabel("step")
        ax.set_ylabel("value")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"single_overall_n{n_val}_k{k}_i{i_val}_j{j_val}_trajectory.png"), dpi=150)
        plt.close(fig)

        # residues time series
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(list(range(len(residues))), residues, marker=".", linestyle="-", markersize=2)
        ax.set_ylim(-0.5, k - 0.5)
        ax.set_yticks(list(range(0, k)))
        ax.set_title(f"residues n={n_val} k={k} i={i_val} j={j_val}")
        ax.set_xlabel("step")
        ax.set_ylabel("residue (mod k)")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"single_overall_n{n_val}_k{k}_i{i_val}_j{j_val}_residues.png"), dpi=150)
        plt.close(fig)

        # residue percentages (all)
        fig, ax = plt.subplots(figsize=(9, 4))
        x_all = [int(r["residue"]) for r in residue_distribution]
        y_all = [float(r["percentage"]) for r in residue_distribution]
        ax.bar(x_all, y_all, color="C0", alpha=0.85)
        ax.set_xticks(x_all)
        ax.set_xlabel("residue")
        ax.set_ylabel("percentage (%)")
        ax.set_title(f"Residue percentages (all) n={n_val} k={k} i={i_val} j={j_val}")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"single_overall_n{n_val}_k{k}_i{i_val}_j{j_val}_residue_percentages.png"), dpi=150)
        plt.close(fig)

        # residue percentages (non-zero)
        fig, ax = plt.subplots(figsize=(9, 4))
        x_nz = [int(r["residue"]) for r in non_zero_residue_distribution]
        y_nz = [float(r["percentage"]) for r in non_zero_residue_distribution]
        ax.bar(x_nz, y_nz, color="C2", alpha=0.85)
        ax.set_xticks(x_nz)
        ax.set_xlabel("residue (non-zero)")
        ax.set_ylabel("percentage among non-zero (%)")
        ax.set_title(f"Residue percentages (non-zero) n={n_val} k={k} i={i_val} j={j_val}")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"single_overall_n{n_val}_k{k}_i{i_val}_j{j_val}_residue_percentages_non_zero.png"), dpi=150)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
