"""Resistance feature.

This module hosts the resistance-related computations extracted from
`sufes/residu.py`.

Contract:
- Inputs: (n,k,i,j) parameters, and an output folder `run_dir`.
- Outputs: CSV/JSON and (when matplotlib is available) PNG plots.
- Errors: invalid parameters raise SystemExit with a helpful message.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

from .core import next_term_ji


def resistance(
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
    """Compute the 'resistance' of a trajectory.

    Definition (as requested): simulate the usual (n,k,i,j) map and classify each
    step as:
      - D (division) if t % k == 0
      - M (multiplication) otherwise

    We count the length of the alternating pattern "M -> D -> M -> ..." starting
    from the first multiplication step, and stop at the first occurrence of
    "D -> D" (two consecutive divisions). The reported value is the number of
    operations in that alternating prefix.

    Outputs written to run_dir:
      - resistance_n{n}_k{k}_i{i}_j{j}.json
      - resistance_n{n}_k{k}_i{i}_j{j}.csv
    """
    if int(k) < 2:
        raise SystemExit(f"k must be >= 2 (got k={k})")
    if alternated and int(alt_m) >= int(k):
        raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={alt_m})")
    if int(i_val) < 1 or int(i_val) >= int(k):
        raise SystemExit(f"i must be in 1..k-1 (got i={i_val}, k={k})")

    os.makedirs(run_dir, exist_ok=True)

    t = int(n_val)
    seen: Dict[int, int] = {}
    reason: Optional[str] = None
    peak = int(t)

    resistance_len = 0
    started = False
    prev_op: Optional[str] = None  # 'M' or 'D'
    ops_sample: List[str] = []

    for step in range(int(max_iters)):
        if int(t) > peak:
            peak = int(t)
        if t in seen:
            reason = "cycle"
            break
        seen[t] = int(step)
        if abs(int(t)) > float(divergence_threshold):
            reason = "divergence_threshold"
            break

        op = "D" if (int(t) % int(k) == 0) else "M"
        if len(ops_sample) < 2000:
            ops_sample.append(op)

        if not started:
            if op == "M":
                started = True
                resistance_len = 1
                prev_op = op
        else:
            if prev_op == "D" and op == "D":
                reason = "dd_stop"
                break

            resistance_len += 1
            prev_op = op

        t = next_term_ji(t, int(k), int(j_val), int(i_val), alternated=alternated, alt_m=int(alt_m))
    else:
        reason = "max_iters"

    payload = {
        "n": int(n_val),
        "k": int(k),
        "i": int(i_val),
        "j": int(j_val),
        "reason": reason,
        "peak": int(peak),
        "steps": int(len(seen)),
        "resistance": int(resistance_len),
        "started": bool(started),
        "ops_sample": ops_sample,
    }

    out_json = os.path.join(run_dir, f"resistance_n{n_val}_k{k}_i{i_val}_j{j_val}.json")
    out_csv = os.path.join(run_dir, f"resistance_n{n_val}_k{k}_i{i_val}_j{j_val}.csv")
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            cf.write("key,value\n")
            for k2 in ["n", "k", "i", "j", "reason", "peak", "steps", "resistance", "started"]:
                cf.write(f"{k2},{payload.get(k2,'')}\n")
    except Exception:
        pass


def resistance_p(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    all_j: bool = False,
    all_n: bool = False,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Compute resistance for all primes k <= pmax and save a single figure."""
    if int(pmax) < 2:
        raise SystemExit(f"p must be >= 2 (got p={pmax})")

    sieve = [True] * (int(pmax) + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(int(pmax) ** 0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, int(pmax) + 1, ii):
                sieve[jj] = False
    primes = [ii for ii, is_p in enumerate(sieve) if is_p]

    os.makedirs(run_dir, exist_ok=True)

    def _resistance_value_for_kj(n_start: int, k_local: int, j_local: int) -> Dict[str, object]:
        if alternated and int(alt_m) >= int(k_local):
            return {
                "k": int(k_local),
                "j": int(j_local),
                "resistance": None,
                "started": False,
                "reason": "alt_m>=k",
                "peak": None,
                "steps": None,
            }
        if int(i_val) < 1 or int(i_val) >= int(k_local):
            return {
                "k": int(k_local),
                "j": int(j_local),
                "resistance": None,
                "started": False,
                "reason": "invalid_i",
                "peak": None,
                "steps": None,
            }

        t = int(n_start)
        seen_local: Dict[int, int] = {}
        reason_local: Optional[str] = None
        peak_local = int(t)

        resistance_len_local = 0
        started_local = False
        prev_op_local: Optional[str] = None

        for step in range(int(max_iters)):
            if int(t) > peak_local:
                peak_local = int(t)
            if t in seen_local:
                reason_local = "cycle"
                break
            seen_local[t] = int(step)
            if abs(int(t)) > float(divergence_threshold):
                reason_local = "divergence_threshold"
                break

            op = "D" if (int(t) % int(k_local) == 0) else "M"

            if not started_local:
                if op == "M":
                    started_local = True
                    resistance_len_local = 1
                    prev_op_local = op
            else:
                if prev_op_local == "D" and op == "D":
                    reason_local = "dd_stop"
                    break

                resistance_len_local += 1
                prev_op_local = op

            t = next_term_ji(t, int(k_local), int(j_local), int(i_val), alternated=alternated, alt_m=int(alt_m))
        else:
            reason_local = "max_iters"

        return {
            "k": int(k_local),
            "j": int(j_local),
            "resistance": int(resistance_len_local),
            "started": bool(started_local),
            "reason": reason_local,
            "peak": int(peak_local),
            "steps": int(len(seen_local)),
        }

    rows: List[Dict[str, object]] = []
    n_values = range(1, int(n_val) + 1) if bool(all_n) else (int(n_val),)

    per_k_mean: Dict[int, Optional[float]] = {}
    per_kj_mean: Dict[Tuple[int, int], Optional[float]] = {}
    per_k_skew: Dict[int, Optional[float]] = {}
    per_kj_skew: Dict[Tuple[int, int], Optional[float]] = {}

    for k in primes:
        if int(k) < 2:
            continue
        if all_j:
            sum_by_j: Dict[int, float] = {jj: 0.0 for jj in range(0, int(k))}
            cnt_by_j: Dict[int, int] = {jj: 0 for jj in range(0, int(k))}
            sum2_by_j: Dict[int, float] = {jj: 0.0 for jj in range(0, int(k))}
            sum3_by_j: Dict[int, float] = {jj: 0.0 for jj in range(0, int(k))}

            for n_cur in n_values:
                for j_cur in range(0, int(k)):
                    rr = _resistance_value_for_kj(int(n_cur), int(k), int(j_cur))
                    rr["n"] = int(n_cur)
                    rows.append(rr)
                    if bool(all_n) and rr.get("resistance") is not None:
                        x = float(rr.get("resistance"))
                        sum_by_j[int(j_cur)] += x
                        cnt_by_j[int(j_cur)] += 1
                        sum2_by_j[int(j_cur)] += x * x
                        sum3_by_j[int(j_cur)] += x * x * x

            if bool(all_n):
                for j_cur in range(0, int(k)):
                    cntv = cnt_by_j.get(int(j_cur), 0)
                    if cntv > 0:
                        mu = sum_by_j.get(int(j_cur), 0.0) / float(cntv)
                        per_kj_mean[(int(k), int(j_cur))] = mu
                        ex2 = sum2_by_j.get(int(j_cur), 0.0) / float(cntv)
                        ex3 = sum3_by_j.get(int(j_cur), 0.0) / float(cntv)
                        var = ex2 - mu * mu
                        if var <= 0:
                            per_kj_skew[(int(k), int(j_cur))] = 0.0
                        else:
                            sigma = math.sqrt(var)
                            m3 = ex3 - 3.0 * mu * ex2 + 2.0 * (mu**3)
                            per_kj_skew[(int(k), int(j_cur))] = float(m3) / float(sigma**3)
                    else:
                        per_kj_mean[(int(k), int(j_cur))] = None
                        per_kj_skew[(int(k), int(j_cur))] = None
        else:
            sum_r = 0.0
            cnt_r = 0
            sum2_r = 0.0
            sum3_r = 0.0
            for n_cur in n_values:
                rr = _resistance_value_for_kj(int(n_cur), int(k), int(j_val))
                rr["n"] = int(n_cur)
                rows.append(rr)
                if bool(all_n) and rr.get("resistance") is not None:
                    x = float(rr.get("resistance"))
                    sum_r += x
                    cnt_r += 1
                    sum2_r += x * x
                    sum3_r += x * x * x

            if bool(all_n):
                if cnt_r > 0:
                    mu = sum_r / float(cnt_r)
                    per_k_mean[int(k)] = mu
                    ex2 = sum2_r / float(cnt_r)
                    ex3 = sum3_r / float(cnt_r)
                    var = ex2 - mu * mu
                    if var <= 0:
                        per_k_skew[int(k)] = 0.0
                    else:
                        sigma = math.sqrt(var)
                        m3 = ex3 - 3.0 * mu * ex2 + 2.0 * (mu**3)
                        per_k_skew[int(k)] = float(m3) / float(sigma**3)
                else:
                    per_k_mean[int(k)] = None
                    per_k_skew[int(k)] = None

    n_tag = f"n{n_val}" if not bool(all_n) else f"N{n_val}"
    out_csv = os.path.join(run_dir, f"resistance_{n_tag}_p{pmax}_i{i_val}_j{j_val}.csv")
    out_json = os.path.join(run_dir, f"resistance_{n_tag}_p{pmax}_i{i_val}_j{j_val}.json")
    out_png = os.path.join(run_dir, f"resistance_{n_tag}_p{pmax}_i{i_val}_j{j_val}.png")
    out_png_mean_kj = os.path.join(run_dir, f"resistance_mean_{n_tag}_p{pmax}_i{i_val}_by_kj.png")
    out_png_skew_kj = os.path.join(run_dir, f"resistance_skew_{n_tag}_p{pmax}_i{i_val}_by_kj.png")

    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            if bool(all_n) and bool(all_j):
                cf.write("# mean_resistance(k,j) over n=1..N\n")
                for (kk, jj) in sorted(per_kj_mean.keys()):
                    mv = per_kj_mean.get((int(kk), int(jj)))
                    cf.write(f"# mean_resistance(k={kk},j={jj})={'' if mv is None else mv}\n")
                cf.write("k,j,mean_resistance,skew_resistance\n")
                for (kk, jj) in sorted(per_kj_mean.keys()):
                    mv = per_kj_mean.get((int(kk), int(jj)))
                    sv = per_kj_skew.get((int(kk), int(jj)))
                    cf.write(f"{kk},{jj},{'' if mv is None else mv},{'' if sv is None else sv}\n")
                cf.write("\n")
            elif bool(all_n) and (not bool(all_j)):
                for kk in sorted(per_k_mean.keys()):
                    mv = per_k_mean.get(int(kk))
                    sv = per_k_skew.get(int(kk))
                    cf.write(f"# mean_resistance(k={kk})={'' if mv is None else mv}\n")
                    cf.write(f"# skew_resistance(k={kk})={'' if sv is None else sv}\n")

            if all_j:
                if bool(all_n):
                    cf.write("n,k,j,resistance,started,reason,peak,steps\n")
                else:
                    cf.write("k,j,resistance,started,reason,peak,steps\n")
                for r in rows:
                    if bool(all_n):
                        cf.write(
                            f"{r.get('n','')},{r.get('k','')},{r.get('j','')},{'' if r.get('resistance') is None else r.get('resistance')},{1 if r.get('started') else 0},{'' if r.get('reason') is None else r.get('reason')},{'' if r.get('peak') is None else r.get('peak')},{'' if r.get('steps') is None else r.get('steps')}\n"
                        )
                    else:
                        cf.write(
                            f"{r.get('k','')},{r.get('j','')},{'' if r.get('resistance') is None else r.get('resistance')},{1 if r.get('started') else 0},{'' if r.get('reason') is None else r.get('reason')},{'' if r.get('peak') is None else r.get('peak')},{'' if r.get('steps') is None else r.get('steps')}\n"
                        )
            else:
                if bool(all_n):
                    cf.write("n,k,resistance,started,reason,peak,steps\n")
                else:
                    cf.write("k,resistance,started,reason,peak,steps\n")
                for r in rows:
                    if bool(all_n):
                        cf.write(
                            f"{r.get('n','')},{r.get('k','')},{'' if r.get('resistance') is None else r.get('resistance')},{1 if r.get('started') else 0},{'' if r.get('reason') is None else r.get('reason')},{'' if r.get('peak') is None else r.get('peak')},{'' if r.get('steps') is None else r.get('steps')}\n"
                        )
                    else:
                        cf.write(
                            f"{r.get('k','')},{'' if r.get('resistance') is None else r.get('resistance')},{1 if r.get('started') else 0},{'' if r.get('reason') is None else r.get('reason')},{'' if r.get('peak') is None else r.get('peak')},{'' if r.get('steps') is None else r.get('steps')}\n"
                        )
    except Exception:
        pass

    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            payload: Dict[str, object] = {
                "n": int(n_val),
                "p": int(pmax),
                "i": int(i_val),
                "j": int(j_val),
                "all_j": bool(all_j),
                "all_n": bool(all_n),
                "rows": rows,
            }
            if bool(all_n) and bool(all_j):
                payload["mean_resistance_by_kj"] = {f"{k},{j}": per_kj_mean.get((int(k), int(j))) for (k, j) in sorted(per_kj_mean.keys())}
                payload["skew_resistance_by_kj"] = {f"{k},{j}": per_kj_skew.get((int(k), int(j))) for (k, j) in sorted(per_kj_skew.keys())}
            elif bool(all_n) and (not bool(all_j)):
                payload["mean_resistance_by_k"] = {str(k): per_k_mean.get(int(k)) for k in sorted(per_k_mean.keys())}
                payload["skew_resistance_by_k"] = {str(k): per_k_skew.get(int(k)) for k in sorted(per_k_skew.keys())}
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Plots (optional)
    try:
        import matplotlib.pyplot as plt
        import math as _math
    except Exception:
        plt = None

    if plt is not None and rows:
        try:
            ks = sorted({int(r.get("k")) for r in rows if r.get("k") is not None})
            n_k = len(ks)
            ncols = min(4, n_k) if n_k else 1
            nrows = int(_math.ceil(n_k / ncols)) if n_k else 1
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)

            by_k: Dict[int, List[Dict[str, object]]] = {}
            for r in rows:
                kk = r.get("k")
                if kk is None:
                    continue
                by_k.setdefault(int(kk), []).append(r)

            for idx, k in enumerate(ks):
                ax = axes[idx // ncols][idx % ncols]
                ax.set_title(f"k={k}")
                rows_k = by_k.get(int(k), [])

                if bool(all_n) and bool(all_j):
                    ax.remove()
                    ax3 = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
                    ax3.set_title(f"k={k}")
                    ax3.set_xlabel("n")
                    ax3.set_ylabel("j")
                    ax3.set_zlabel("resistance")
                    xs = [int(r.get("n", 0)) for r in rows_k]
                    ys = [int(r.get("j", 0)) for r in rows_k]
                    zs = [float(r.get("resistance") or 0) for r in rows_k]
                    ax3.scatter(xs, ys, zs, s=6, alpha=0.65)
                elif all_j:
                    ax.set_xlabel("j")
                    ax.set_ylabel("resistance")
                    js = [int(r.get("j", 0)) for r in rows_k]
                    vals = [r.get("resistance") if r.get("resistance") is not None else float("nan") for r in rows_k]
                    order = sorted(range(len(js)), key=lambda ii: js[ii])
                    js_sorted = [js[ii] for ii in order]
                    vals_sorted = [vals[ii] for ii in order]
                    ax.plot(js_sorted, vals_sorted, marker="o", linestyle="-", linewidth=0.8)
                    ax.set_xticks(js_sorted)
                elif bool(all_n):
                    ax.set_xlabel("n")
                    ax.set_ylabel("resistance")
                    ns = [int(r.get("n", 0)) for r in rows_k]
                    vals = [r.get("resistance") if r.get("resistance") is not None else float("nan") for r in rows_k]
                    order = sorted(range(len(ns)), key=lambda ii: ns[ii])
                    ns_sorted = [ns[ii] for ii in order]
                    vals_sorted = [vals[ii] for ii in order]
                    ax.plot(ns_sorted, vals_sorted, marker=".", linestyle="-", linewidth=0.8)
                else:
                    ax.set_xlabel("metric")
                    ax.set_ylabel("value")
                    val = rows_k[0].get("resistance") if rows_k else None
                    if val is None:
                        ax.text(0.5, 0.5, str(rows_k[0].get("reason")) if rows_k else "", ha="center", va="center", transform=ax.transAxes)
                    else:
                        ax.bar(["resistance"], [int(val)], color="C0", alpha=0.8)
                        ax.set_ylim(0, max(1, int(val) + 1))

            for j in range(n_k, nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")

            title = f"Resistance (M/D alternation) {n_tag} p={pmax} i={i_val}"
            if bool(all_n) and bool(all_j):
                title += " (all n, all j)"
            elif bool(all_n):
                title += f" (all n, j={j_val})"
            elif bool(all_j):
                title += " (all j)"
            else:
                title += f" j={j_val}"
            fig.suptitle(title)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

    if plt is not None and bool(all_n) and bool(all_j) and per_kj_mean:
        try:
            ks_mean = sorted({int(kk) for (kk, _jj) in per_kj_mean.keys()})
            n_k = len(ks_mean)
            ncols = min(4, n_k) if n_k else 1
            nrows = int(_math.ceil(n_k / ncols)) if n_k else 1
            fig2, axes2 = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)
            for idx, kk in enumerate(ks_mean):
                ax = axes2[idx // ncols][idx % ncols]
                ax.set_title(f"k={kk}")
                ax.set_xlabel("j")
                ax.set_ylabel("mean_resistance")
                js = sorted([int(jj) for (k2, jj) in per_kj_mean.keys() if int(k2) == int(kk)])
                ys = [per_kj_mean.get((int(kk), int(jj))) for jj in js]
                ys_plot = [float(v) if v is not None else float("nan") for v in ys]
                ax.plot(js, ys_plot, marker="o", linestyle="-", linewidth=0.9)
                ax.set_xticks(js)
            for j in range(n_k, nrows * ncols):
                axes2[j // ncols][j % ncols].axis("off")
            fig2.suptitle(f"Mean resistance over n=1..N (N={n_val}) vs j (p={pmax}, i={i_val})")
            fig2.tight_layout(rect=(0, 0, 1, 0.95))
            fig2.savefig(out_png_mean_kj, dpi=160)
            plt.close(fig2)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass

    if plt is not None and bool(all_n) and bool(all_j) and per_kj_skew:
        try:
            ks_skew = sorted({int(kk) for (kk, _jj) in per_kj_skew.keys()})
            n_k = len(ks_skew)
            ncols = min(4, n_k) if n_k else 1
            nrows = int(_math.ceil(n_k / ncols)) if n_k else 1
            fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows), squeeze=False)
            for idx, kk in enumerate(ks_skew):
                ax = axes3[idx // ncols][idx % ncols]
                ax.set_title(f"k={kk}")
                ax.set_xlabel("j")
                ax.set_ylabel("skew_resistance")
                js = sorted([int(jj) for (k2, jj) in per_kj_skew.keys() if int(k2) == int(kk)])
                ys = [per_kj_skew.get((int(kk), int(jj))) for jj in js]
                ys_plot = [float(v) if v is not None else float("nan") for v in ys]
                ax.plot(js, ys_plot, marker="o", linestyle="-", linewidth=0.9, color="C3")
                ax.set_xticks(js)
            for j in range(n_k, nrows * ncols):
                axes3[j // ncols][j % ncols].axis("off")
            fig3.suptitle(f"Skewness of resistance over n=1..N (N={n_val}) vs j (p={pmax}, i={i_val})")
            fig3.tight_layout(rect=(0, 0, 1, 0.95))
            fig3.savefig(out_png_skew_kj, dpi=160)
            plt.close(fig3)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
