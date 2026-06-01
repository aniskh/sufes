"""Residue distribution feature (k<=p, loop on j).

Extracted from the legacy monolithic residue implementation to keep the codebase maintainable.

Public API mirrors the historical function so :mod:`sufes.core` can keep calling
`residu_distribution(...)`.

This feature simulates a trajectory for each prime k<=p and each j in a range,
then aggregates residue statistics (mean on non-zero residues, skewness, etc.).
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .core import _log_end, _log_start, next_term_ji


def _plot_residu_distribution_rows(
    rows: List[Dict[str, object]],
    out_path: str,
    title: str,
    *,
    y_key: str = "mean_residue",
    y_label: str = "mean_residue",
) -> None:
    """Plot one subplot per k: y(j) with count_non_zero as secondary bars."""

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return

    if not rows:
        return

    by_k: Dict[int, List[Dict[str, object]]] = {}
    for r in rows:
        kk = int(r.get("k"))
        by_k.setdefault(kk, []).append(r)

    ks = sorted(by_k.keys())
    n_k = len(ks)
    ncols = min(3, max(1, n_k))
    nrows = int((n_k + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

    for idx, kk in enumerate(ks):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_title(f"k={kk}")
        ax.set_xlabel("j")
        ax.set_ylabel(y_label)

        rows_k = by_k[kk]
        js = [int(r["j"]) for r in rows_k]
        order = np.argsort(js)
        js_sorted = np.array(js)[order]
        y_vals = [r.get(y_key) if r.get(y_key) is not None else float("nan") for r in rows_k]
        y_sorted = np.array(y_vals, dtype=float)[order]
        counts = np.array([int(r.get("count_non_zero", 0)) for r in rows_k])[order]

        ax.plot(js_sorted, y_sorted, marker="o", linestyle="-", color="C0")
        ax2 = ax.twinx()
        ax2.bar(js_sorted, counts, alpha=0.25, color="C1", width=0.6)
        ax2.set_ylabel("count_non_zero", color="C1")
        ax.set_xticks(list(js_sorted))

    for j in range(n_k, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        fig.savefig(out_path, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def _plot_residue_percentage_rows(
    rows: List[Dict[str, object]],
    out_path: str,
    title: str,
) -> None:
    """Plot one subplot per k: percentage of each residue r in 1..k-1.

    Expects rows with keys: k, residue, percentage.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return

    if not rows:
        return

    by_k: Dict[int, List[Dict[str, object]]] = {}
    for r in rows:
        kk = int(r.get("k"))
        by_k.setdefault(kk, []).append(r)

    ks = sorted(by_k.keys())
    n_k = len(ks)
    ncols = min(3, max(1, n_k))
    nrows = int((n_k + ncols - 1) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

    for idx, kk in enumerate(ks):
        ax = axes[idx // ncols][idx % ncols]
        ax.set_title(f"k={kk}")
        ax.set_xlabel("residue r")
        ax.set_ylabel("percentage (%)")

        rows_k = by_k[kk]
        rs = [int(r.get("residue")) for r in rows_k]
        order = np.argsort(rs)
        rs_sorted = np.array(rs)[order]
        perc = [float(r.get("percentage") or 0.0) for r in rows_k]
        perc_sorted = np.array(perc, dtype=float)[order]

        ax.bar(rs_sorted, perc_sorted, alpha=0.8, color="C0", width=0.8)
        ax.set_xticks(list(rs_sorted))
        ax.grid(True, axis="y", alpha=0.25)

    for j in range(n_k, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    try:
        fig.savefig(out_path, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def residu_distribution(
    dist_n: int,
    pmax: int,
    run_dir: str,
    i_val: int = 1,
    j_val: Optional[int] = None,
    j_multiple: int = 2,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    include_zero_mean: bool = False,
    all_n: bool = False,
) -> None:
    """Compute residue distribution for a single start value N=dist_n.

    For each prime k <= pmax and each j in 0..(j_multiple*k-1) (or a single j
    if provided), simulate the trajectory starting at N and compute:
      - count_non_zero
      - mean_residue of non-zero residues (or including zeros when requested)
      - skewness on non-zero residues

    Outputs in run_dir:
      - residu_distribution_n{N}_p{P}_{jtag}.csv
      - residu_distribution_n{N}_p{P}_{jtag}.json
      - residu_distribution_n{N}_p{P}_{jtag}.png
      - residu_distribution_n{N}_p{P}_{jtag}_skew_non_zero.png
    """

    sieve = [True] * (pmax + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(pmax ** 0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, pmax + 1, ii):
                sieve[jj] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]

    # rows collects results for a single dist_n run (or aggregated later)
    rows: List[Dict[str, object]] = []

    # If all_n is requested, we'll aggregate across n0 = 1..dist_n
    if all_n:
        agg_map: Dict[tuple, List[Dict[str, object]]] = {}
        # (k,j) -> residue r -> count across all n0<=N and all steps
        agg_residue_counts: Dict[Tuple[int, int], Dict[int, int]] = {}
        n_range = range(1, int(dist_n) + 1)
    else:
        n_range = [int(dist_n)]

    # Iterate per-k so we can log a single START/END per k (much less verbose)
    for k in primes:
        if alt_m >= k or i_val >= k or i_val < 1:
            continue

        # Reference std for a uniform distribution over residues {1, ..., k-1}:
        # Std = (k-1) / (2*sqrt(3))  == sqrt((k^2 - 2k + 1)/12)
        theory_std_non_zero = (float(k - 1) / (2.0 * math.sqrt(3.0))) if k > 1 else None

        k_start_ts = _log_start(f"residu_distribution k={k} i={i_val}")

        # determine j values once per k
        if j_val is None:
            j_values = range(0, j_multiple * k)
        else:
            j_values = range(int(j_val), int(j_val) + 1)

        for n0 in n_range:
            for j_cur in j_values:
                t = n0
                seen = set()

                # Track moments on non-zero residues: sum(r), sum(r^2), sum(r^3)
                sum_non_zero = 0
                sumsq_non_zero = 0
                sumcube_non_zero = 0
                count_non_zero = 0
                count_total = 0

                # A2: probability to hit v_k(next) >= 2 right after a multiplication step.
                # Multiplication step iff current t is NOT divisible by k.
                mul_ops = 0
                hit_ge2 = 0
                residue_counts: Dict[int, int] = {}
                reason_local = None
                for _ in range(max_iters):
                    r = int(abs(t) % k)
                    count_total += 1
                    if r != 0:
                        residue_counts[r] = residue_counts.get(r, 0) + 1
                    if r != 0:
                        sum_non_zero += int(r)
                        sumsq_non_zero += int(r) * int(r)
                        sumcube_non_zero += int(r) * int(r) * int(r)
                        count_non_zero += 1
                    if t in seen:
                        reason_local = "cycle"
                        break
                    seen.add(t)
                    if abs(t) > divergence_threshold:
                        reason_local = "divergence_threshold"
                        break

                    prev = t
                    nxt = next_term_ji(prev, k, j_cur, i_val, alternated=alternated, alt_m=alt_m)

                    # Count multiplication operations and valuation>=2 hits on the result.
                    if abs(int(prev)) % int(k) != 0:
                        mul_ops += 1
                        try:
                            av = abs(int(nxt))
                            e = 0
                            while av != 0 and av % int(k) == 0:
                                av //= int(k)
                                e += 1
                            if e >= 2:
                                hit_ge2 += 1
                        except Exception:
                            pass

                    t = nxt
                else:
                    reason_local = "max_iters"

                A2 = (float(hit_ge2) / float(mul_ops)) if mul_ops > 0 else None

                mean_residue = (float(sum_non_zero) / count_non_zero) if count_non_zero else None
                mean_residue_including_zero = (float(sum_non_zero) / count_total) if count_total else None

                # Skewness of non-zero residues (population skewness)
                skew_non_zero = None
                std_non_zero = None
                if count_non_zero and mean_residue is not None:
                    try:
                        mu = float(mean_residue)
                        e2 = float(sumsq_non_zero) / float(count_non_zero)
                        var = e2 - mu * mu
                        if var <= 0:
                            std_non_zero = 0.0
                            skew_non_zero = 0.0
                        else:
                            sigma = math.sqrt(var)
                            std_non_zero = float(sigma)
                            e3 = float(sumcube_non_zero) / float(count_non_zero)
                            m3 = e3 - 3.0 * mu * e2 + 2.0 * (mu ** 3)
                            skew_non_zero = float(m3) / float(sigma ** 3) if sigma != 0 else 0.0
                    except Exception:
                        skew_non_zero = None
                        std_non_zero = None

                if include_zero_mean:
                    mean_residue = mean_residue_including_zero

                row = {
                    "k": k,
                    "j": j_cur,
                    "i": i_val,
                    "n0": n0,
                    "A2": A2,
                    "mul_ops": int(mul_ops),
                    "hit_ge2": int(hit_ge2),
                    "count_non_zero": count_non_zero,
                    "count_total": count_total,
                    "mean_residue": mean_residue,
                    "mean_residue_including_zero": mean_residue_including_zero,
                    "std_non_zero": std_non_zero,
                    "theory_std_non_zero": theory_std_non_zero,
                    "epsilon_std": (
                        (100.0 * (float(std_non_zero) - float(theory_std_non_zero)) / float(theory_std_non_zero))
                        if (std_non_zero is not None and theory_std_non_zero is not None and theory_std_non_zero != 0)
                        else None
                    ),
                    "skew_non_zero": skew_non_zero,
                    "reason": reason_local,
                    "elapsed_sec": None,
                }

                if all_n:
                    agg_map.setdefault((k, j_cur), []).append(row)
                    # Aggregate residue counts (non-zero residues only)
                    key = (int(k), int(j_cur))
                    bucket = agg_residue_counts.setdefault(key, {})
                    for rr, cc in residue_counts.items():
                        bucket[int(rr)] = int(bucket.get(int(rr), 0)) + int(cc)
                else:
                    rows.append(row)

        try:
            _log_end(f"residu_distribution k={k} i={i_val}", k_start_ts)
        except Exception:
            pass

    # If all_n True, compute aggregated rows (averages over n0)
    if all_n:
        rows = []
        for (k, j), lst in agg_map.items():
            # average numeric metrics across n0 (skip None values)
            def _avg(key: str):
                vals = [v.get(key) for v in lst if v.get(key) is not None]
                return float(sum(vals)) / float(len(vals)) if vals else None

            def _sum_int(key: str) -> int:
                out = 0
                for v in lst:
                    try:
                        out += int(v.get(key) or 0)
                    except Exception:
                        pass
                return int(out)

            rows.append(
                {
                    "k": k,
                    "j": j,
                    "i": i_val,
                    "n0": None,
                    "A2": _avg("A2"),
                    "mul_ops": _sum_int("mul_ops"),
                    "hit_ge2": _sum_int("hit_ge2"),
                    "count_non_zero": int(_avg("count_non_zero") or 0),
                    "count_total": int(_avg("count_total") or 0),
                    "mean_residue": _avg("mean_residue"),
                    "mean_residue_including_zero": _avg("mean_residue_including_zero"),
                    "std_non_zero": _avg("std_non_zero"),
                    "theory_std_non_zero": (float(int(k) - 1) / (2.0 * math.sqrt(3.0))) if int(k) > 1 else None,
                    "epsilon_std": None,
                    "skew_non_zero": _avg("skew_non_zero"),
                    "reason": "aggregated",
                    "elapsed_sec": _avg("elapsed_sec"),
                }
            )

        # Fill epsilon_std for aggregated rows (needs std and theory)
        for r in rows:
            try:
                s = r.get("std_non_zero")
                th = r.get("theory_std_non_zero")
                if s is None or th is None:
                    r["epsilon_std"] = None
                else:
                    thf = float(th)
                    r["epsilon_std"] = (100.0 * (float(s) - thf) / thf) if thf != 0 else None
            except Exception:
                r["epsilon_std"] = None

    else:
        # rows already filled for single-run path
        pass
        # rows already filled for single-run path

    os.makedirs(run_dir, exist_ok=True)
    j_tag = f"_j{j_val}" if j_val is not None else f"_jmult{j_multiple}"
    out_csv = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}.csv")
    out_json = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}.json")

    # Additional output (all_n only): residue percentage distribution for each (k,j)
    out_pct_csv = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_residue_percentages.csv")
    out_pct_json = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_residue_percentages.json")

    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            cf.write(
                "k,j,i,count_non_zero,count_total,mean_residue,mean_residue_including_zero,std_non_zero,theory_std_non_zero,epsilon_std,skew_non_zero,reason\n"
            )
            for r in rows:
                mean_val = "" if r["mean_residue"] is None else f"{r['mean_residue']:.6f}"
                mean_incl = (
                    ""
                    if r.get("mean_residue_including_zero") is None
                    else f"{r['mean_residue_including_zero']:.6f}"
                )
                std_val = "" if r.get("std_non_zero") is None else f"{float(r.get('std_non_zero')):.6f}"
                th_val = "" if r.get("theory_std_non_zero") is None else f"{float(r.get('theory_std_non_zero')):.6f}"
                eps_val = "" if r.get("epsilon_std") is None else f"{float(r.get('epsilon_std')):.6f}"
                skew_val = "" if r.get("skew_non_zero") is None else f"{float(r.get('skew_non_zero')):.6f}"
                cf.write(
                    f"{r['k']},{r['j']},{r['i']},{r['count_non_zero']},{r.get('count_total', '')},{mean_val},{mean_incl},{std_val},{th_val},{eps_val},{skew_val},{r['reason']}\n"
                )
    except Exception:
        pass

    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(
                {"n": dist_n, "p": pmax, "j": j_val, "j_multiple": j_multiple, "rows": rows},
                jf,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    # Residue percentage outputs (non-zero residues 1..k-1), aggregated across n0<=N.
    if all_n:
        pct_rows: List[Dict[str, object]] = []
        try:
            for (k, j), cmap in sorted(agg_residue_counts.items(), key=lambda t: (t[0][0], t[0][1])):
                total_non_zero = int(sum(int(v) for v in cmap.values()))
                # emit rows for residues 1..k-1 even if absent (0%)
                for rr in range(1, int(k)):
                    c = int(cmap.get(int(rr), 0))
                    pct = (100.0 * float(c) / float(total_non_zero)) if total_non_zero > 0 else 0.0
                    pct_rows.append(
                        {
                            "k": int(k),
                            "j": int(j),
                            "i": int(i_val),
                            "residue": int(rr),
                            "count": int(c),
                            "total_non_zero": int(total_non_zero),
                            "percentage": float(pct),
                        }
                    )

            with open(out_pct_csv, "w", encoding="utf-8") as f:
                f.write("k,j,i,residue,count,total_non_zero,percentage\n")
                for r in pct_rows:
                    f.write(
                        f"{r['k']},{r['j']},{r['i']},{r['residue']},{r['count']},{r['total_non_zero']},{float(r['percentage']):.6f}\n"
                    )
        except Exception:
            pct_rows = []

        try:
            with open(out_pct_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "n": int(dist_n),
                        "p": int(pmax),
                        "j": j_val,
                        "j_multiple": int(j_multiple),
                        "i": int(i_val),
                        "rows": pct_rows,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

        # Plot (we aggregate across j too by default: sum over j for each k)
        try:
            # k -> residue -> count
            by_k_res: Dict[int, Dict[int, int]] = {}
            for pr in pct_rows:
                kk = int(pr["k"])
                rr = int(pr["residue"])
                by_k_res.setdefault(kk, {})
                by_k_res[kk][rr] = by_k_res[kk].get(rr, 0) + int(pr.get("count", 0))

            plot_rows: List[Dict[str, object]] = []
            for kk in sorted(by_k_res.keys()):
                total = sum(int(v) for v in by_k_res[kk].values())
                for rr in range(1, int(kk)):
                    c = int(by_k_res[kk].get(rr, 0))
                    pct = (100.0 * float(c) / float(total)) if total > 0 else 0.0
                    plot_rows.append({"k": int(kk), "residue": int(rr), "percentage": float(pct)})

            out_png = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_residue_percentages.png")
            title = f"Residue percentages (non-zero) aggregated over n0<=N (N={dist_n}, p={pmax})"
            _plot_residue_percentage_rows(plot_rows, out_png, title)
        except Exception:
            pass

    # Mean-residue plot
    try:
        png_out = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}.png")
        title = f"Residue distribution (n={dist_n}, p={pmax})"
        title += f", j={j_val}" if j_val is not None else f", j_mult={j_multiple}"
        _plot_residu_distribution_rows(rows, png_out, title)
    except Exception:
        pass

    # A2 plot
    try:
        if rows:
            a2_png = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_A2.png")
            a2_title = f"A2 = P(v_k(next)≥2 | multiplication) (n={dist_n}, p={pmax})"
            a2_title += f", j={j_val}" if j_val is not None else f", j_mult={j_multiple}"
            _plot_residu_distribution_rows(rows, a2_png, a2_title, y_key="A2", y_label="A2")
    except Exception:
        pass

    # Skewness plot
    try:
        if rows:
            skew_png = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_skew_non_zero.png")
            skew_title = f"Residue skewness (non-zero) (n={dist_n}, p={pmax})"
            skew_title += f", j={j_val}" if j_val is not None else f", j_mult={j_multiple}"
            _plot_residu_distribution_rows(rows, skew_png, skew_title, y_key="skew_non_zero", y_label="skew_non_zero")
    except Exception:
        pass

    # Std deviation plot (non-zero residues)
    try:
        if rows:
            std_png = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_std_non_zero.png")
            std_title = f"Residue std (non-zero) (n={dist_n}, p={pmax})"
            std_title += f", j={j_val}" if j_val is not None else f", j_mult={j_multiple}"
            _plot_residu_distribution_rows(rows, std_png, std_title, y_key="std_non_zero", y_label="std_non_zero")
    except Exception:
        pass

    # epsilon_std plot (percentage difference vs reference std)
    try:
        if rows:
            eps_png = os.path.join(run_dir, f"residu_distribution_n{dist_n}_p{pmax}{j_tag}_epsilon_std.png")
            eps_title = f"epsilon_std (%) vs ref (k-1)/(2*sqrt(3)) (n={dist_n}, p={pmax})"
            eps_title += f", j={j_val}" if j_val is not None else f", j_mult={j_multiple}"
            _plot_residu_distribution_rows(rows, eps_png, eps_title, y_key="epsilon_std", y_label="epsilon_std (%)")
    except Exception:
        pass
