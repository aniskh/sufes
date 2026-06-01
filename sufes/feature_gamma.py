"""Gamma feature.

This module contains the implementation previously in `sufes/gamma.py`.

We keep `sufes/gamma.py` as a small compatibility shim so existing imports
(`from sufes import gamma`) keep working.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

from .core import next_term_ji


def _primes_up_to(pmax: int) -> List[int]:
    if pmax < 2:
        return []
    sieve = [True] * (pmax + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(pmax**0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, pmax + 1, ii):
                sieve[jj] = False
    return [i for i, is_p in enumerate(sieve) if is_p]


def _compute_gamma_from_trajectory(traj: List[int]) -> Optional[float]:
    S = len(traj)
    if S <= 1:
        return None
    acc = 0.0
    for a, b in zip(traj, traj[1:]):
        aa = abs(int(a))
        bb = abs(int(b))
        if aa == 0 or bb == 0:
            return None
        acc += math.log(float(bb) / float(aa))
    return acc / float(S)


def _simulate_trajectory(
    n_val: int,
    k: int,
    i_val: int,
    j_val: int,
    *,
    max_iters: int,
    divergence_threshold: float,
    alternated: bool,
    alt_m: int,
) -> Dict[str, object]:
    """Simule la trajectoire U_t(n) jusqu'au cycle/divergence/max_iters.

    En cas de divergence ou max_iters, on tronque la trajectoire aux 1000
    premières itérations pour le calcul de gamma.
    """

    t = int(n_val)
    seen = set()
    trajectory: List[int] = []
    reason: Optional[str] = None
    peak = t

    for _step in range(max_iters):
        if int(t) > peak:
            peak = int(t)
        if t in seen:
            reason = "cycle"
            break
        seen.add(t)
        trajectory.append(int(t))

        t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
        if abs(t) > divergence_threshold:
            reason = "divergence_threshold"
            break
    else:
        reason = "max_iters"

    traj_for_gamma = trajectory
    if reason in ("divergence_threshold", "max_iters") and len(traj_for_gamma) > 1000:
        traj_for_gamma = traj_for_gamma[:1000]

    gamma_val = _compute_gamma_from_trajectory(traj_for_gamma)
    return {
        "count_steps": len(trajectory),
        "gamma": gamma_val,
        "reason": reason,
        "peak": peak,
        "traj_len_gamma": len(traj_for_gamma),
    }


def gamma(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    do_plot: bool = True,
    all_i: bool = False,
    all_j: bool = False,
) -> None:
    """Compute gamma for primes k <= pmax for given (n,i,j)."""

    primes = _primes_up_to(pmax)

    results: List[Dict[str, object]] = []
    per_k_i_rows: Dict[int, List[Dict[str, object]]] = {}
    per_k_j_rows: Dict[int, List[Dict[str, object]]] = {}

    for k in primes:
        if alt_m >= k:
            continue

        if all_j:
            j_values = range(0, k)
        else:
            j_values = range(j_val, j_val + 1)

        if all_i:
            i_values = range(1, k)
        else:
            i_values = range(i_val, i_val + 1)

        per_i: List[Dict[str, object]] = []
        per_j: List[Dict[str, object]] = []

        for j_cur in j_values:
            for i_cur in i_values:
                if i_cur >= k or i_cur < 1:
                    continue
                if j_cur < 0 or j_cur >= k:
                    continue

                sim = _simulate_trajectory(
                    n_val,
                    k,
                    i_cur,
                    j_cur,
                    max_iters=max_iters,
                    divergence_threshold=divergence_threshold,
                    alternated=alternated,
                    alt_m=alt_m,
                )

                row = {
                    "k": k,
                    "n": n_val,
                    "i": i_cur,
                    "j": j_cur,
                    "count_steps": sim["count_steps"],
                    "traj_len_gamma": sim.get("traj_len_gamma"),
                    "gamma": sim["gamma"],
                    "reason": sim["reason"],
                    "peak": sim["peak"],
                }

                results.append(row)

                if all_i and not all_j:
                    per_i.append(row)
                if all_j and not all_i:
                    per_j.append(row)

        if all_i and not all_j:
            per_k_i_rows[k] = per_i
        if all_j and not all_i:
            per_k_j_rows[k] = per_j

    os.makedirs(run_dir, exist_ok=True)
    i_label = "all" if all_i else str(i_val)
    j_label = "all" if all_j else str(j_val)
    out_csv = os.path.join(run_dir, f"gamma_n{n_val}_p{pmax}_i{i_label}_j{j_label}.csv")
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("k,n,i,j,count_steps,traj_len_gamma,gamma,reason,peak\n")
        for r in results:
            f.write(
                f"{r['k']},{r['n']},{r['i']},{r['j']},{r['count_steps']},{r.get('traj_len_gamma','')},"
                f"{'' if r['gamma'] is None else r['gamma']},{r['reason']},{r['peak']}\n"
            )

    if not do_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    try:
        ks = [r["k"] for r in results]
        gvals = [float("nan") if r["gamma"] is None else float(r["gamma"]) for r in results]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ks, gvals, marker="o", linewidth=1)
        ax.set_xlabel("k")
        ax.set_ylabel("gamma")
        ax.set_title(f"gamma(n={n_val}, i={i_label}, j={j_label})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"gamma_n{n_val}_p{pmax}_i{i_label}_j{j_label}.png"), dpi=150)
        plt.close(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass

    if do_plot and all_i and not all_j:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            plt = None

        if plt is not None:
            try:
                k_items = sorted(per_k_i_rows.items(), key=lambda kv: kv[0])
                k_items = [(k, rows) for (k, rows) in k_items if rows]
                if not k_items:
                    return

                n_plots = len(k_items)
                ncols = 3
                nrows = (n_plots + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

                for idx, (k, rows) in enumerate(k_items):
                    rr = idx // ncols
                    cc = idx % ncols
                    ax = axes[rr][cc]
                    xs = [int(r["i"]) for r in rows]
                    ys = [float("nan") if r.get("gamma") is None else float(r["gamma"]) for r in rows]
                    ax.plot(xs, ys, marker="o", linewidth=1)
                    ax.set_title(f"k={k}")
                    ax.set_xlabel("i")
                    ax.set_ylabel("gamma")
                    ax.grid(True, alpha=0.3)

                for idx in range(n_plots, nrows * ncols):
                    rr = idx // ncols
                    cc = idx % ncols
                    axes[rr][cc].axis("off")

                fig.suptitle(f"gamma vs i (n={n_val}, j={j_val}, k primes <= {pmax})", y=1.02)
                fig.tight_layout()
                fig.savefig(os.path.join(run_dir, f"gamma_by_i_grid_n{n_val}_p{pmax}_j{j_val}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                try:
                    plt.close("all")
                except Exception:
                    pass

    if do_plot and all_j and not all_i:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            plt = None

        if plt is not None:
            try:
                k_items = sorted(per_k_j_rows.items(), key=lambda kv: kv[0])
                k_items = [(k, rows) for (k, rows) in k_items if rows]
                if not k_items:
                    return

                n_plots = len(k_items)
                ncols = 3
                nrows = (n_plots + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

                for idx, (k, rows) in enumerate(k_items):
                    rr = idx // ncols
                    cc = idx % ncols
                    ax = axes[rr][cc]
                    xs = [int(r["j"]) for r in rows]
                    ys = [float("nan") if r.get("gamma") is None else float(r["gamma"]) for r in rows]
                    ax.plot(xs, ys, marker="o", linewidth=1)
                    ax.set_title(f"k={k}")
                    ax.set_xlabel("j")
                    ax.set_ylabel("gamma")
                    ax.grid(True, alpha=0.3)

                for idx in range(n_plots, nrows * ncols):
                    rr = idx // ncols
                    cc = idx % ncols
                    axes[rr][cc].axis("off")

                fig.suptitle(f"gamma vs j (n={n_val}, i={i_val}, k primes <= {pmax})", y=1.02)
                fig.tight_layout()
                fig.savefig(os.path.join(run_dir, f"gamma_by_j_grid_n{n_val}_p{pmax}_i{i_val}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                try:
                    plt.close("all")
                except Exception:
                    pass

    if do_plot and all_i and all_j:
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            plt = None

        if plt is not None:
            try:
                per_k: Dict[int, List[Dict[str, object]]] = {}
                for r in results:
                    kk = int(r["k"])
                    per_k.setdefault(kk, []).append(r)

                k_items = sorted(((k, rows) for k, rows in per_k.items() if rows), key=lambda kv: kv[0])
                if not k_items:
                    return

                n_plots = len(k_items)
                ncols = 3
                nrows = (n_plots + ncols - 1) // ncols
                fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 3.2 * nrows), squeeze=False)

                for idx, (k, rows) in enumerate(k_items):
                    rr = idx // ncols
                    cc = idx % ncols
                    ax = axes[rr][cc]

                    gamma_by_i: Dict[int, List[float]] = {}
                    for row in rows:
                        ii = int(row["i"])
                        gv = row.get("gamma")
                        if gv is None:
                            continue
                        try:
                            gamma_by_i.setdefault(ii, []).append(float(gv))
                        except Exception:
                            continue

                    xs = sorted(gamma_by_i.keys())
                    ys_mean = [(sum(gamma_by_i[ii]) / float(len(gamma_by_i[ii]))) if gamma_by_i[ii] else float("nan") for ii in xs]
                    ys_min = [min(gamma_by_i[ii]) if gamma_by_i[ii] else float("nan") for ii in xs]
                    ys_max = [max(gamma_by_i[ii]) if gamma_by_i[ii] else float("nan") for ii in xs]

                    ax.plot(xs, ys_mean, marker="o", linewidth=1, label="mean_j")
                    ax.fill_between(xs, ys_min, ys_max, alpha=0.2, label="min/max_j")
                    ax.set_title(f"k={k}")
                    ax.set_xlabel("i")
                    ax.set_ylabel("gamma")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8, loc="best")

                for idx in range(n_plots, nrows * ncols):
                    rr = idx // ncols
                    cc = idx % ncols
                    axes[rr][cc].axis("off")

                fig.suptitle(f"gamma vs i aggregated over j (n={n_val}, k primes <= {pmax})", y=1.02)
                fig.tight_layout()
                fig.savefig(os.path.join(run_dir, f"gamma_by_i_grid_meanj_n{n_val}_p{pmax}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                try:
                    plt.close("all")
                except Exception:
                    pass
