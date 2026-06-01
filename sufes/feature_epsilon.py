"""Epsilon feature (A0 diagnostic across primes).

This feature computes, for a fixed start value ``n`` and fixed parameters
``(i,j)``, an empirical estimate

        A0(k) := R0 = (# visited nodes divisible by k) / (# visited nodes),

for each prime ``k <= p``.

It then compares it to the reference value

        ref1(k) = k / (2k - 1)

and outputs the difference.

Notes
- We still compute the k-adic valuation nu_k(t) internally, but we only keep
    the information needed to compute A0.
- The implementation keeps the public function signature for CLI compatibility.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .core import next_term_ji


def residu_epsilon(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    find_best_j: bool = False,
    ordre_multiplicatif_j: bool = False,
    table_mode: bool = False,
    j_multiple: int = 1,
    all_n: bool = False,
) -> None:
    """Compute A0(k)=R0 and its deviation to ref1(k)=k/(2k-1) for primes k<=pmax."""

    # sieve primes up to pmax
    sieve = [True] * (pmax + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(pmax**0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, pmax + 1, ii):
                sieve[jj] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]

    def _k_adic_valuation_abs(v: int, base_k: int) -> int:
        av = abs(int(v))
        if av == 0:
            return 0
        e = 0
        while av % base_k == 0:
            av //= base_k
            e += 1
        return e

    results: List[Dict[str, object]] = []
    best_rows: List[Dict[str, object]] = []
    order_rows: List[Dict[str, object]] = []

    # Determine n values to process: single n or all n in 1..n_val when all_n True
    if all_n:
        n_values = range(1, int(n_val) + 1)
    else:
        n_values = [int(n_val)]

    for k in primes:
        if alt_m >= k or i_val >= k or i_val < 1:
            # skip invalid combinations
            continue

        # if requested, we'll evaluate all j in 0..k-1 (find_best_j/ordre),
        # otherwise support an extended j range when j_multiple>1.
        if find_best_j or ordre_multiplicatif_j:
            j_values = range(0, k)
        else:
            try:
                jm = int(j_multiple or 1)
            except Exception:
                jm = 1
            # If the caller supplied a concrete j_val, respect it and only run that j.
            if j_val is not None:
                j_values = range(j_val, j_val + 1)
            elif jm <= 1:
                j_values = range(0, k)
            else:
                j_values = range(0, k * jm)
        per_j_rows: List[Dict[str, object]] = []

        for j_cur in j_values:
            # process each requested starting n (single or many)
            for n_start in n_values:
                t = int(n_start)
                seen = set()
                nu_list: List[int] = []
                reason: Optional[str] = None
                peak = t
                for _step in range(max_iters):
                    if int(t) > peak:
                        peak = int(t)
                    if t in seen:
                        reason = "cycle"
                        break
                    seen.add(t)
                    # compute valuation for this node
                    nu = _k_adic_valuation_abs(t, k)
                    nu_list.append(nu)
                    # advance
                    t = next_term_ji(t, k, j_cur, i_val, alternated=alternated, alt_m=alt_m)
                    if abs(t) > divergence_threshold:
                        reason = "divergence_threshold"
                        break
                else:
                    reason = "max_iters"

                # raw count: number of nodes divisible by k
                count_ge1_raw = int(sum(1 for v in nu_list if v >= 1))
                # A0 = (# visited nodes divisible by k) / (# visited nodes)
                A0 = (float(count_ge1_raw) / float(len(nu_list))) if len(nu_list) > 0 else None

                # ref1 = k/(2k-1) and deviations
                try:
                    a0_ref1 = float(k) / float((2 * k) - 1)
                except Exception:
                    a0_ref1 = None

                if A0 is None:
                    a0_ref1_delta = None
                    a0_ref1_abs_err = None
                    a0_ref1_delta_percent = None
                else:
                    a0_ref1_delta = (float(A0) - float(a0_ref1)) if a0_ref1 is not None else None
                    a0_ref1_abs_err = abs(float(a0_ref1_delta)) if a0_ref1_delta is not None else None
                    if A0 == 0 or a0_ref1_delta is None:
                        a0_ref1_delta_percent = None
                    else:
                        a0_ref1_delta_percent = float(a0_ref1_delta) * 100.0 / float(A0)

                row = {
                    "n": int(n_start),
                    "k": k,
                    "j": j_cur,
                    "count_steps": len(nu_list),
                    "count_ge1_raw": count_ge1_raw,
                    "A0": A0,
                    "a0_ref1": a0_ref1,
                    "a0_ref1_delta": a0_ref1_delta,
                    "a0_ref1_delta_percent": a0_ref1_delta_percent,
                    "a0_ref1_abs_err": a0_ref1_abs_err,
                    "reason": reason,
                    "peak": peak,
                }
                results.append(row)
                per_j_rows.append(row)

        if find_best_j:
            candidates = [r for r in per_j_rows if r.get("epsilonv_ratio_percent") is not None]
            if candidates:
                best = min(
                    candidates,
                    key=lambda rr: (abs(float(rr["epsilonv_ratio_percent"])), int(rr.get("j", 0))),
                )
                best_rows.append({"k": k, "j": int(best["j"]), "epsilonv_ratio_percent": best["epsilonv_ratio_percent"]})
            else:
                best_rows.append({"k": k, "j": None, "epsilonv_ratio_percent": None})

        if ordre_multiplicatif_j:
            order_for_js: List[Tuple[int, Optional[int]]] = []
            for jcur in range(0, k):
                a = (jcur + 1) % k
                if a == 0:
                    order_for_js.append((jcur, None))
                    continue
                found = None
                for f in range(1, k):
                    if pow(a, f, k) == 1:
                        found = f
                        break
                order_for_js.append((jcur, found))

            defined = [(jcur, ordv) for (jcur, ordv) in order_for_js if ordv is not None]
            if defined:
                _jmax, ordmax = max(defined, key=lambda x: (int(x[1]), -int(x[0])))
                ties = [j for (j, o) in defined if o == ordmax]
                j_pref = min(ties)
                order_rows.append({"k": k, "j_order_max": int(j_pref), "order_max": int(ordmax)})
            else:
                order_rows.append({"k": k, "j_order_max": None, "order_max": None})

    os.makedirs(run_dir, exist_ok=True)

    # Build filename parts: omit j part when j_val is None to avoid 'jNone' in filenames
    j_part = f"_j{j_val}" if j_val is not None else ""
    jmult_part = f"_jmult{j_multiple}" if j_multiple is not None else ""
    base_name = f"epsilon_n{n_val}_p{pmax}_i{i_val}{j_part}{jmult_part}"
    out_json = os.path.join(run_dir, f"{base_name}.json")
    try:
        meta: Dict[str, object] = {"n": n_val, "p": pmax, "i": i_val, "j": j_val, "rows": results}
        if find_best_j:
            meta["best_rows"] = best_rows
        if ordre_multiplicatif_j:
            meta["order_rows"] = order_rows
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    out_csv = os.path.join(run_dir, f"{base_name}.csv")
    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            cf.write(
                "k,j,count_steps,count_ge1_raw,A0,a0_ref1,delta,abs_err,delta_percent,reason,peak\n"
            )
            for r in results:
                count_ge1_raw_val = "" if r.get("count_ge1_raw") is None else r.get("count_ge1_raw")
                A0_val = "" if r.get("A0") is None else r.get("A0")
                a0_ref1_val = "" if r.get("a0_ref1") is None else r.get("a0_ref1")
                a0_ref1_delta_val = "" if r.get("a0_ref1_delta") is None else r.get("a0_ref1_delta")
                a0_ref1_delta_pct_val = (
                    "" if r.get("a0_ref1_delta_percent") is None else r.get("a0_ref1_delta_percent")
                )
                abs_err_val = "" if r.get("a0_ref1_abs_err") is None else r.get("a0_ref1_abs_err")
                cf.write(
                    f"{r.get('k','')},{r.get('j','')},{r.get('count_steps','')},{count_ge1_raw_val},{A0_val},{a0_ref1_val},{a0_ref1_delta_val},{abs_err_val},{a0_ref1_delta_pct_val},{r.get('reason','')},{r.get('peak','')}\n"
                )
    except Exception:
        pass

    # --- Summary CSV: mean A0 per (k,i,j) ---
    try:
        # group by (k,i,j) and collect A0 values (ignore None)
        group: Dict[Tuple[int, int, int], List[float]] = {}
        for r in results:
            k_ = r.get("k")
            j_ = r.get("j")
            a0 = r.get("A0")
            if k_ is None or j_ is None:
                continue
            key = (int(k_), int(i_val), int(j_))
            if a0 is None:
                continue
            try:
                group.setdefault(key, []).append(float(a0))
            except Exception:
                continue

        out_mean_csv = os.path.join(run_dir, f"epsilon_meanA0_{base_name}.csv")
        with open(out_mean_csv, "w", encoding="utf-8") as mf:
            mf.write("k,i,j,mean_A0\n")
            # sort by k then j for stable ordering
            for (k_, i_, j_) in sorted(group.keys(), key=lambda t: (t[0], t[2])):
                vals = group.get((k_, i_, j_), [])
                if vals:
                    mean_val = float(sum(vals)) / float(len(vals))
                    mf.write(f"{k_},{i_},{j_},{mean_val}\n")
                else:
                    mf.write(f"{k_},{i_},{j_},\n")
    except Exception:
        pass

    # --- Plot mean_A0: one subplot per k (x=j, y=mean_A0) ---
    try:
        import math
        import matplotlib.pyplot as plt

        # Build by_k mapping k -> list of (j, mean)
        by_k = {}
        for (k_, i_, j_), vals in group.items():
            if not vals:
                continue
            mean_val = float(sum(vals)) / float(len(vals))
            by_k.setdefault(int(k_), []).append((int(j_), float(mean_val)))

        ks = sorted(by_k.keys())
        if ks:
            n_k = len(ks)
            ncols = min(3, n_k)
            nrows = int(math.ceil(n_k / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False)

            for idx, k_ in enumerate(ks):
                ax = axes[idx // ncols][idx % ncols]
                data = by_k.get(k_, [])
                if not data:
                    ax.text(0.5, 0.5, 'no data', ha='center', va='center', color='gray')
                    ax.set_axis_off()
                    continue
                # sort by j
                data_sorted = sorted(data, key=lambda t: t[0])
                js = [t[0] for t in data_sorted]
                means = [t[1] for t in data_sorted]
                ax.plot(js, means, marker='o', linestyle='-', linewidth=1)
                ax.set_title(f'k={k_}')
                ax.set_xlabel('j')
                ax.set_ylabel('mean_A0')
                ax.grid(True, alpha=0.25)

            # turn off unused axes
            for idx_off in range(len(ks), nrows * ncols):
                axes[idx_off // ncols][idx_off % ncols].axis('off')

            fig.tight_layout()
            out_png = os.path.join(run_dir, f"epsilon_meanA0_{base_name}.png")
            try:
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
            except Exception:
                plt.close(fig)
    except Exception:
        # plotting is optional; ignore failures (matplotlib may be absent)
        pass

    # Legacy options are intentionally ignored after the epsilon cleanup.
    # They remain in the function signature for CLI/backward compatibility.
