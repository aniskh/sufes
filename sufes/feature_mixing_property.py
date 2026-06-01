"""Mixing property feature (lag-plot of residues).

Extracted from the legacy monolithic residue implementation to keep the codebase maintainable.

Public API mirrors the legacy functions so :mod:`sufes.core` can keep calling
`residu_mod.mixing_property(...)` and `residu_mod.mixing_property_p(...)` via thin
wrappers.

Contract:
- Inputs: (n,k,i,j) parameters, output folder `run_dir`, and optional controls.
- Outputs: JSON/CSV exports and a PNG plot (when matplotlib is available).
- Errors: invalid parameters raise SystemExit with a helpful message.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from .core import next_term_ji


def mixing_property(
    n_val: int,
    k: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    all_j: bool = False,
    lag: int = 1,
    max_points: int = 20000,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Generate a lag-plot (mixing property) for residues r_t = t mod k.

    We simulate the usual (n,k,i,j) rule until reaching a cycle, divergence_threshold,
    or max_iters. We collect residues r_t = |t| mod k, then create point pairs:
      (x_t, y_t) = (r_t, r_{t+lag})

    Outputs written to run_dir:
      - mixing_property_n{n}_k{k}_i{i}_j{j}_lag{lag}.json
      - mixing_property_n{n}_k{k}_i{i}_j{j}_lag{lag}.csv
      - mixing_property_n{n}_k{k}_i{i}_j{j}_lag{lag}.png
    """
    if int(k) < 2:
        raise SystemExit(f"k must be >= 2 (got k={k})")
    if int(lag) < 1:
        raise SystemExit(f"lag must be >= 1 (got lag={lag})")
    if alternated and int(alt_m) >= int(k):
        raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={alt_m})")
    if int(i_val) < 1 or int(i_val) >= int(k):
        raise SystemExit(f"i must be in 1..k-1 (got i={i_val}, k={k})")

    os.makedirs(run_dir, exist_ok=True)

    def _simulate_residues_for_j(j: int) -> Tuple[List[int], Optional[str], int, Optional[int]]:
        t_local = int(n_val)
        seen_local: Dict[int, int] = {}
        residues_local: List[int] = []
        reason_local: Optional[str] = None
        peak_local = int(t_local)
        preperiod_local: Optional[int] = None

        for step_local in range(int(max_iters)):
            rr = int(abs(int(t_local)) % int(k))
            residues_local.append(int(rr))

            if int(t_local) > peak_local:
                peak_local = int(t_local)
            if t_local in seen_local:
                reason_local = "cycle"
                preperiod_local = int(seen_local[t_local])
                break
            seen_local[t_local] = int(step_local)
            if abs(int(t_local)) > float(divergence_threshold):
                reason_local = "divergence_threshold"
                preperiod_local = None
                break
            t_local = next_term_ji(
                t_local,
                int(k),
                int(j),
                int(i_val),
                alternated=alternated,
                alt_m=int(alt_m),
            )
        else:
            reason_local = "max_iters"
            preperiod_local = None
        return residues_local, reason_local, int(peak_local), preperiod_local

    L = int(lag)
    j_values = list(range(0, int(k))) if bool(all_j) else [int(j_val)]

    per_j: List[Dict[str, object]] = []
    all_pairs_for_csv: List[Tuple[int, int, int]] = []
    plot_data: List[Tuple[int, List[Tuple[int, int]]]] = []

    global_peak = int(n_val)
    for j_cur in j_values:
        residues, reason, peak, preperiod = _simulate_residues_for_j(int(j_cur))
        global_peak = max(global_peak, int(peak))

        # build pairs for this j
        pairs_j: List[Tuple[int, int]] = []
        n_pairs_j = max(0, len(residues) - L)
        for idx in range(n_pairs_j):
            x = int(residues[idx])
            y = int(residues[idx + L])
            pairs_j.append((x, y))
            all_pairs_for_csv.append((int(j_cur), x, y))

        # subsample for plot
        plot_pairs = pairs_j
        if int(max_points) > 0 and len(plot_pairs) > int(max_points):
            step_s = max(1, int(len(plot_pairs) // int(max_points)))
            plot_pairs = plot_pairs[::step_s]
        plot_data.append((int(j_cur), plot_pairs))

        per_j.append(
            {
                "j": int(j_cur),
                "reason": reason,
                "peak": int(peak),
                "count_steps": int(len(residues)),
                "count_pairs": int(len(pairs_j)),
                "preperiod": preperiod,
                "residues_sample": residues[:200],
                "pairs_sample": [{"x": int(x), "y": int(y)} for (x, y) in pairs_j[:200]],
            }
        )

    j_tag = "jall" if bool(all_j) else f"j{int(j_val)}"
    base = f"mixing_property_n{n_val}_k{k}_i{i_val}_{j_tag}_lag{lag}"
    out_json = os.path.join(run_dir, base + ".json")
    out_csv = os.path.join(run_dir, base + ".csv")

    payload = {
        "n": int(n_val),
        "k": int(k),
        "i": int(i_val),
        "lag": int(lag),
        "all_j": bool(all_j),
        "j_values": j_values,
        "peak": int(global_peak),
        "rows": per_j,
    }
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            cf.write("j,idx,x,y\n")
            for idx, (j_cur, x, y) in enumerate(all_pairs_for_csv):
                cf.write(f"{j_cur},{idx},{x},{y}\n")
    except Exception:
        pass

    # plot
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is not None:
        try:
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            # use a categorical map: up to 20 distinct colors, then wrap
            cmap = plt.get_cmap("tab20")
            for idx, (j_cur, pts) in enumerate(plot_data):
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                color = cmap(idx % 20)
                label = f"j={j_cur}" if bool(all_j) else None
                ax.scatter(xs, ys, s=7, alpha=0.28, color=color, label=label)

            ax.set_xlim(-0.5, int(k) - 0.5)
            ax.set_ylim(-0.5, int(k) - 0.5)
            ax.set_xticks(list(range(0, int(k))))
            ax.set_yticks(list(range(0, int(k))))
            ax.set_xlabel(r"$r_t$ (mod k)")
            ax.set_ylabel(rf"$r_{{t+{L}}}$ (mod k)")
            title = f"Lag-plot residues (lag={L}) n={n_val} k={k} i={i_val}"
            if bool(all_j):
                title += " (all j)"
            else:
                title += f" j={int(j_val)}"
            ax.set_title(title)
            ax.grid(True, alpha=0.15)
            if bool(all_j) and len(j_values) <= 12:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
            fig.tight_layout()
            out_png = os.path.join(run_dir, base + ".png")
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass


def mixing_property_p(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    *,
    all_j: bool = False,
    lag: int = 1,
    max_points: int = 20000,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Lag-plot residues for all primes k <= pmax.

    For each prime k <= pmax, we compute the lag-plot data (optionally for all j=0..k-1)
    and render all k-plots in a single PNG (one subplot per k).

    Outputs written to run_dir:
      - mixing_property_n{n}_p{p}_i{i}_{jtag}_lag{lag}.json
      - mixing_property_n{n}_p{p}_i{i}_{jtag}_lag{lag}.csv
      - mixing_property_n{n}_p{p}_i{i}_{jtag}_lag{lag}.png
    """
    if int(pmax) < 2:
        raise SystemExit(f"p must be >= 2 (got p={pmax})")
    if int(lag) < 1:
        raise SystemExit(f"lag must be >= 1 (got lag={lag})")

    os.makedirs(run_dir, exist_ok=True)

    # sieve primes up to pmax
    sieve = [True] * (int(pmax) + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(int(pmax) ** 0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, int(pmax) + 1, ii):
                sieve[jj] = False
    primes = [kk for kk, ok in enumerate(sieve) if ok]

    L = int(lag)
    j_tag = "jall" if bool(all_j) else f"j{int(j_val)}"
    base = f"mixing_property_n{n_val}_p{pmax}_i{i_val}_{j_tag}_lag{lag}"
    out_json = os.path.join(run_dir, base + ".json")
    out_csv = os.path.join(run_dir, base + ".csv")
    out_png = os.path.join(run_dir, base + ".png")

    rows: List[Dict[str, object]] = []
    # store plot data per k: list of (label, points)
    plot_per_k: List[Tuple[int, List[Tuple[int, List[Tuple[int, int]]]]]] = []
    # CSV rows: k,j,idx,x,y
    csv_rows: List[Tuple[int, int, int, int, int]] = []

    for k in primes:
        if alternated and int(alt_m) >= int(k):
            # skip invalid alt-m for this k
            continue
        if int(i_val) < 1 or int(i_val) >= int(k):
            # skip invalid i for this k
            continue

        def _simulate_residues_for_j(j: int) -> Tuple[List[int], Optional[str], int, Optional[int]]:
            t_local = int(n_val)
            seen_local: Dict[int, int] = {}
            residues_local: List[int] = []
            reason_local: Optional[str] = None
            peak_local = int(t_local)
            preperiod_local: Optional[int] = None
            for step_local in range(int(max_iters)):
                rr = int(abs(int(t_local)) % int(k))
                residues_local.append(int(rr))
                if int(t_local) > peak_local:
                    peak_local = int(t_local)
                if t_local in seen_local:
                    reason_local = "cycle"
                    preperiod_local = int(seen_local[t_local])
                    break
                seen_local[t_local] = int(step_local)
                if abs(int(t_local)) > float(divergence_threshold):
                    reason_local = "divergence_threshold"
                    preperiod_local = None
                    break
                t_local = next_term_ji(
                    t_local,
                    int(k),
                    int(j),
                    int(i_val),
                    alternated=alternated,
                    alt_m=int(alt_m),
                )
            else:
                reason_local = "max_iters"
                preperiod_local = None
            return residues_local, reason_local, int(peak_local), preperiod_local

        j_values = list(range(0, int(k))) if bool(all_j) else [int(j_val)]
        plot_data_k: List[Tuple[int, List[Tuple[int, int]]]] = []
        per_j_k: List[Dict[str, object]] = []

        for j_cur in j_values:
            residues, reason, peak, preperiod = _simulate_residues_for_j(int(j_cur))
            pairs_j: List[Tuple[int, int]] = []
            n_pairs_j = max(0, len(residues) - L)
            for idx in range(n_pairs_j):
                x = int(residues[idx])
                y = int(residues[idx + L])
                pairs_j.append((x, y))
                csv_rows.append((int(k), int(j_cur), int(len(csv_rows)), x, y))

            # subsample for plot
            plot_pairs = pairs_j
            if int(max_points) > 0 and len(plot_pairs) > int(max_points):
                step_s = max(1, int(len(plot_pairs) // int(max_points)))
                plot_pairs = plot_pairs[::step_s]
            plot_data_k.append((int(j_cur), plot_pairs))
            per_j_k.append(
                {
                    "j": int(j_cur),
                    "reason": reason,
                    "peak": int(peak),
                    "count_steps": int(len(residues)),
                    "count_pairs": int(len(pairs_j)),
                    "preperiod": preperiod,
                }
            )

        rows.append({"k": int(k), "j_values": j_values, "all_j": bool(all_j), "lag": int(lag), "rows": per_j_k})
        plot_per_k.append((int(k), [(int(jj), pts) for (jj, pts) in plot_data_k]))

    # write JSON
    payload = {
        "n": int(n_val),
        "p": int(pmax),
        "i": int(i_val),
        "lag": int(lag),
        "all_j": bool(all_j),
        "j": int(j_val),
        "rows": rows,
    }
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # write CSV pairs
    try:
        with open(out_csv, "w", encoding="utf-8") as cf:
            cf.write("k,j,idx,x,y\n")
            for (k2, j2, idx, x, y) in csv_rows:
                cf.write(f"{k2},{j2},{idx},{x},{y}\n")
    except Exception:
        pass

    # plot grid (one subplot per k)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is not None:
        try:
            n_k = len(plot_per_k)
            if n_k == 0:
                return
            ncols = min(3, max(1, n_k))
            nrows = int((n_k + ncols - 1) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.8 * nrows), squeeze=False)
            cmap = plt.get_cmap("tab20")

            for idx_k, (k2, jplots) in enumerate(plot_per_k):
                ax = axes[idx_k // ncols][idx_k % ncols]
                ax.set_title(f"k={k2}")
                for idx_j, (j2, pts) in enumerate(jplots):
                    if not pts:
                        continue
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    color = cmap(idx_j % 20)
                    label = f"j={j2}" if bool(all_j) else None
                    ax.scatter(xs, ys, s=6, alpha=0.25, color=color, label=label)

                ax.set_xlim(-0.5, int(k2) - 0.5)
                ax.set_ylim(-0.5, int(k2) - 0.5)
                ax.set_xticks(list(range(0, int(k2))))
                ax.set_yticks(list(range(0, int(k2))))
                ax.set_xlabel(r"$r_t$ (mod k)")
                ax.set_ylabel(rf"$r_{{t+{L}}}$")
                ax.grid(True, alpha=0.15)
                if bool(all_j) and len(jplots) <= 8:
                    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)

            for extra in range(n_k, nrows * ncols):
                axes[extra // ncols][extra % ncols].axis("off")

            fig.suptitle(f"Mixing property lag-plot (n={n_val}, p={pmax}, i={i_val}, lag={L})")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
