"""Shannon entropy feature extracted from the legacy monolithic residue implementation.

Public API:
- shannon_entropy_run(...)
- shannon_entropy_p_run(...)

The feature simulates a single (n,k,i,j) trajectory (or sweeps primes k<=p)
and computes Shannon entropy of the residue distribution restricted to
non-zero residues r in 1..k-1.

This module is intentionally self-contained.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .core import next_term_ji
from .stats import shannon_entropy_non_zero_residues


def _percent_distribution(keys: List[int], counts_map: Dict[int, int], total_count: int) -> Tuple[List[Dict[str, object]], float]:
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
            running += pct
        else:
            pct = 100.0 - running
        rows.append({"residue": int(key), "count": cnt, "percentage": float(pct)})
    return rows, float(sum(float(r["percentage"]) for r in rows))


def shannon_entropy_run(
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
) -> None:
    """Run Shannon entropy for one (n,k,i,j)."""

    if alt_m >= k:
        raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={alt_m})")
    if i_val < 1 or i_val >= k:
        raise SystemExit(f"i must be in 1..k-1 (got i={i_val}, k={k})")

    os.makedirs(run_dir, exist_ok=True)

    seq: List[int] = []
    residues: List[int] = []
    seen: Dict[int, int] = {}
    t = int(n_val)
    reason: Optional[str] = None
    peak = int(t)

    for _ in range(max_iters):
        seq.append(int(t))
        residues.append(int(abs(t) % k))
        if int(t) > peak:
            peak = int(t)
        if t in seen:
            reason = "cycle"
            break
        seen[t] = len(seq) - 1
        if abs(t) > divergence_threshold:
            reason = "divergence_threshold"
            break
        t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
    else:
        reason = "max_iters"

    count_total = int(len(seq))
    count_non_zero = int(sum(1 for r in residues if int(r) != 0))

    counts_all: Dict[int, int] = {rr: 0 for rr in range(k)}
    for r in residues:
        counts_all[int(r)] = counts_all.get(int(r), 0) + 1
    residue_distribution, residue_percent_sum = _percent_distribution(list(range(k)), counts_all, count_total)

    counts_non_zero_map: Dict[int, int] = {rr: 0 for rr in range(1, k)}
    for r in residues:
        if int(r) != 0:
            counts_non_zero_map[int(r)] = counts_non_zero_map.get(int(r), 0) + 1
    non_zero_residue_distribution, non_zero_residue_percent_sum = _percent_distribution(list(range(1, k)), counts_non_zero_map, count_non_zero)

    H, Hmax, Hratio = shannon_entropy_non_zero_residues(counts_non_zero_map, int(k))
    Hdelta = (float(Hmax) - float(H)) if Hmax is not None else None

    payload = {
        "n": int(n_val),
        "k": int(k),
        "i": int(i_val),
        "j": int(j_val),
        "reason": reason,
        "steps": int(count_total),
        "peak": int(peak),
        "residue_stats": {
            "count_total": int(count_total),
            "count_non_zero": int(count_non_zero),
            "residue_distribution": residue_distribution,
            "residue_percent_sum": float(residue_percent_sum),
            "non_zero_residue_distribution": non_zero_residue_distribution,
            "non_zero_residue_percent_sum": float(non_zero_residue_percent_sum),
            "shannon_entropy_non_zero": H,
            "shannon_entropy_non_zero_max": Hmax,
            "shannon_entropy_non_zero_ratio": Hratio,
            "shannon_entropy_non_zero_delta": Hdelta,
        },
    }

    out_json = os.path.join(run_dir, f"shannon_entropy_n{n_val}_k{k}_i{i_val}_j{j_val}.json")
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    out_csv = os.path.join(run_dir, f"shannon_entropy_n{n_val}_k{k}_i{i_val}_j{j_val}.csv")
    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            cf.write("k,i,j,reason,steps,count_non_zero,peak,H,Hmax,Hratio,Hdelta\n")
            cf.write(
                f"{k},{i_val},{j_val},{reason},{count_total},{count_non_zero},{peak},"
                f"{'' if H is None else H},{'' if Hmax is None else Hmax},{'' if Hratio is None else Hratio},{'' if Hdelta is None else Hdelta}\n"
            )
    except Exception:
        pass

    try:
        print(
            f"Shannon entropy (non-zero residues, base2) n={n_val} k={k} i={i_val} j={j_val}: "
            f"{H} (Hmax={Hmax}, ratio={Hratio}, delta={Hdelta})"
        )
    except Exception:
        pass


def shannon_entropy_p_run(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    all_j: bool = False,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Run Shannon entropy for all primes k <= pmax."""

    if pmax is None or int(pmax) < 2:
        raise SystemExit("--shannon-entropy-p must be >= 2")

    os.makedirs(run_dir, exist_ok=True)

    sieve = [True] * (int(pmax) + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(int(pmax) ** 0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, int(pmax) + 1, ii):
                sieve[jj] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]

    rows: List[Dict[str, object]] = []

    for k in primes:
        if alt_m >= k:
            continue
        if i_val < 1 or i_val >= k:
            continue

        def _one_j(j_cur: int) -> Dict[str, object]:
            seq: List[int] = []
            residues: List[int] = []
            seen: Dict[int, int] = {}
            t = int(n_val)
            reason: Optional[str] = None
            peak = int(t)

            for _ in range(max_iters):
                seq.append(int(t))
                residues.append(int(abs(t) % k))
                if int(t) > peak:
                    peak = int(t)
                if t in seen:
                    reason = "cycle"
                    break
                seen[t] = len(seq) - 1
                if abs(t) > divergence_threshold:
                    reason = "divergence_threshold"
                    break
                t = next_term_ji(t, k, j_cur, i_val, alternated=alternated, alt_m=alt_m)
            else:
                reason = "max_iters"

            count_total = int(len(seq))
            count_non_zero = int(sum(1 for r in residues if int(r) != 0))

            counts_non_zero_map: Dict[int, int] = {rr: 0 for rr in range(1, k)}
            for r in residues:
                if int(r) != 0:
                    counts_non_zero_map[int(r)] = counts_non_zero_map.get(int(r), 0) + 1

            H, Hmax, Hratio = shannon_entropy_non_zero_residues(counts_non_zero_map, int(k))
            Hdelta = (float(Hmax) - float(H)) if Hmax is not None else None
            R0 = (float(count_non_zero) / float(count_total)) if count_total else None

            return {
                "k": int(k),
                "i": int(i_val),
                "j": int(j_cur),
                "reason": reason,
                "steps": int(count_total),
                "count_non_zero": int(count_non_zero),
                "R0": R0,
                "peak": int(peak),
                "H": H,
                "Hmax": Hmax,
                "Hratio": Hratio,
                "Hdelta": Hdelta,
            }

        if not all_j:
            rows.append(_one_j(int(j_val)))
        else:
            per_j = [_one_j(int(jj)) for jj in range(0, int(k))]
            H_vals = [float(r["H"]) for r in per_j if r.get("H") is not None]
            Hr_vals = [float(r["Hratio"]) for r in per_j if r.get("Hratio") is not None]
            R0_vals = [float(r["R0"]) for r in per_j if r.get("R0") is not None]

            def _stats(vals: List[float]) -> Dict[str, Optional[float]]:
                if not vals:
                    return {"min": None, "max": None, "mean": None}
                return {"min": float(min(vals)), "max": float(max(vals)), "mean": float(sum(vals) / float(len(vals)))}

            Hs = _stats(H_vals)
            Hrs = _stats(Hr_vals)
            R0s = _stats(R0_vals)

            steps_mean = float(sum(float(r.get("steps", 0) or 0) for r in per_j)) / float(len(per_j))
            count_nz_mean = float(sum(float(r.get("count_non_zero", 0) or 0) for r in per_j)) / float(len(per_j))
            peak_max = max(int(r.get("peak", 0) or 0) for r in per_j)

            rows.append(
                {
                    "k": int(k),
                    "i": int(i_val),
                    "j": "all",
                    "reason": "mixed",
                    "steps": steps_mean,
                    "count_non_zero": count_nz_mean,
                    "R0_min": R0s["min"],
                    "R0_max": R0s["max"],
                    "R0_mean": R0s["mean"],
                    "peak": int(peak_max),
                    "H_min": Hs["min"],
                    "H_max": Hs["max"],
                    "H_mean": Hs["mean"],
                    "Hratio_min": Hrs["min"],
                    "Hratio_max": Hrs["max"],
                    "Hratio_mean": Hrs["mean"],
                    "Hmax": per_j[0].get("Hmax"),
                }
            )

    payload = {"n": int(n_val), "p": int(pmax), "i": int(i_val), "j": ("all" if all_j else int(j_val)), "all_j": bool(all_j), "rows": rows}

    j_tag = "all" if all_j else str(int(j_val))
    out_json = os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}.json")
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    out_csv = os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}.csv")
    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            if not all_j:
                cf.write("k,i,j,reason,steps,count_non_zero,R0,peak,H,Hmax,Hratio,Hdelta\n")
                for r in rows:
                    cf.write(
                        f"{r.get('k','')},{r.get('i','')},{r.get('j','')},{r.get('reason','')},{r.get('steps','')},{r.get('count_non_zero','')},"
                        f"{'' if r.get('R0') is None else r.get('R0')},{r.get('peak','')},{'' if r.get('H') is None else r.get('H')},"
                        f"{'' if r.get('Hmax') is None else r.get('Hmax')},{'' if r.get('Hratio') is None else r.get('Hratio')},{'' if r.get('Hdelta') is None else r.get('Hdelta')}\n"
                    )
            else:
                cf.write("k,i,j,reason,steps_mean,count_non_zero_mean,R0_min,R0_max,R0_mean,peak,Hmin,Hmax,Hmean,Hratio_min,Hratio_max,Hratio_mean\n")
                for r in rows:
                    cf.write(
                        f"{r.get('k','')},{r.get('i','')},{r.get('j','')},{r.get('reason','')},{r.get('steps','')},{r.get('count_non_zero','')},"
                        f"{'' if r.get('R0_min') is None else r.get('R0_min')},{'' if r.get('R0_max') is None else r.get('R0_max')},{'' if r.get('R0_mean') is None else r.get('R0_mean')},{r.get('peak','')},"
                        f"{'' if r.get('H_min') is None else r.get('H_min')},{'' if r.get('H_max') is None else r.get('H_max')},{'' if r.get('H_mean') is None else r.get('H_mean')},"
                        f"{'' if r.get('Hratio_min') is None else r.get('Hratio_min')},{'' if r.get('Hratio_max') is None else r.get('Hratio_max')},{'' if r.get('Hratio_mean') is None else r.get('Hratio_mean')}\n"
                    )
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is None or not rows:
        return

    try:
        ks = [int(r["k"]) for r in rows]
        if not all_j:
            Hs = [float(r["H"]) if r.get("H") is not None else float("nan") for r in rows]
            Hratios = [float(r["Hratio"]) if r.get("Hratio") is not None else float("nan") for r in rows]
            R0s = [float(r["R0"]) if r.get("R0") is not None else float("nan") for r in rows]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ks, Hs, marker="o", linestyle="-")
            ax.set_xlabel("k")
            ax.set_ylabel("H (bits)")
            ax.set_title(f"Shannon entropy (non-zero residues) n={n_val} i={i_val} j={j_tag}")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}_H.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ks, [hr * 100.0 for hr in Hratios], marker="o", linestyle="-")
            ax.set_xlabel("k")
            ax.set_ylabel("100 * H/Hmax")
            ax.set_title(f"Shannon entropy ratio (non-zero) n={n_val} i={i_val} j={j_tag}")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}_Hratio.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ks, [r0 * 100.0 for r0 in R0s], marker="o", linestyle="-")
            ax.set_xlabel("k")
            ax.set_ylabel("100 * count_non_zero / steps")
            ax.set_title(f"Non-zero residue rate n={n_val} i={i_val} j={j_tag}")
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}_R0.png"), dpi=150)
            plt.close(fig)
        else:
            Hmin = [float(r["H_min"]) if r.get("H_min") is not None else float("nan") for r in rows]
            Hmax = [float(r["H_max"]) if r.get("H_max") is not None else float("nan") for r in rows]
            Hmean = [float(r["H_mean"]) if r.get("H_mean") is not None else float("nan") for r in rows]

            Hrmin = [float(r["Hratio_min"]) if r.get("Hratio_min") is not None else float("nan") for r in rows]
            Hrmax = [float(r["Hratio_max"]) if r.get("Hratio_max") is not None else float("nan") for r in rows]
            Hrmean = [float(r["Hratio_mean"]) if r.get("Hratio_mean") is not None else float("nan") for r in rows]

            R0min = [float(r["R0_min"]) if r.get("R0_min") is not None else float("nan") for r in rows]
            R0max = [float(r["R0_max"]) if r.get("R0_max") is not None else float("nan") for r in rows]
            R0mean = [float(r["R0_mean"]) if r.get("R0_mean") is not None else float("nan") for r in rows]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(ks, Hmin, Hmax, alpha=0.2, label="min..max over j")
            ax.plot(ks, Hmean, marker="o", linestyle="-", label="mean over j")
            ax.set_xlabel("k")
            ax.set_ylabel("H (bits)")
            ax.set_title(f"Shannon entropy (non-zero) n={n_val} i={i_val} j=0..k-1")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}_H.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(ks, [v * 100.0 for v in Hrmin], [v * 100.0 for v in Hrmax], alpha=0.2, label="min..max over j")
            ax.plot(ks, [v * 100.0 for v in Hrmean], marker="o", linestyle="-", label="mean over j")
            ax.set_xlabel("k")
            ax.set_ylabel("100 * H/Hmax")
            ax.set_title(f"Shannon entropy ratio (non-zero) n={n_val} i={i_val} j=0..k-1")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}_Hratio.png"), dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.fill_between(ks, [v * 100.0 for v in R0min], [v * 100.0 for v in R0max], alpha=0.2, label="min..max over j")
            ax.plot(ks, [v * 100.0 for v in R0mean], marker="o", linestyle="-", label="mean over j")
            ax.set_xlabel("k")
            ax.set_ylabel("100 * count_non_zero / steps")
            ax.set_title(f"Non-zero residue rate n={n_val} i={i_val} j=0..k-1")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"shannon_entropy_n{n_val}_p{pmax}_i{i_val}_j{j_tag}_R0.png"), dpi=150)
            plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
