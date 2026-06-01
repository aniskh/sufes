"""Spiral plotting feature.

This module contains the implementation previously living in the legacy monolithic residue implementation.

Public API:
- spirale(...)
- spirale_p(...)
- spirale_all(...)

This module is self-contained.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

from .core import next_term_ji


def spirale(
    n_val: int,
    k: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    angle_mode: str = "residue",
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Plot the trajectory in a spiral layout.

    angle_mode:
      - 'residue' -> angle = $2\pi (t \bmod k) / k$
      - 'step'    -> angle = step * (2\pi / k)

    The radius is computed as $\log(|t|+1)$.

    Outputs in run_dir:
      - spirale_n{n}_k{k}_i{i}_j{j}.json
      - spirale_n{n}_k{k}_i{i}_j{j}.png
    """

    if alt_m >= k:
        raise SystemExit(f"--alt-m must be < k (k={k}, alt_m={alt_m})")
    if i_val < 1 or i_val >= k:
        raise SystemExit(f"i must be in 1..k-1 (got i={i_val}, k={k})")

    os.makedirs(run_dir, exist_ok=True)
    seq: List[int] = []
    residues: List[int] = []
    seen = set()
    t = int(n_val)
    reason = None
    peak = t

    for step in range(max_iters):
        seq.append(int(t))
        r = int(abs(t) % k)
        residues.append(int(r))
        if int(t) > peak:
            peak = int(t)
        if t in seen:
            reason = "cycle"
            break
        seen.add(t)
        if abs(t) > divergence_threshold:
            reason = "divergence_threshold"
            break
        t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
    else:
        reason = "max_iters"

    # polar -> cartesian
    points: List[Dict[str, object]] = []
    two_pi = 2.0 * math.pi
    for idx, (val, r) in enumerate(zip(seq, residues)):
        if angle_mode == "residue":
            angle = two_pi * (r % k) / float(k)
        else:
            angle = idx * (two_pi / float(k))
        radius = math.log(abs(val) + 1.0)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        points.append(
            {
                "step": idx,
                "t": int(val),
                "residue": int(r),
                "angle": float(angle),
                "radius": float(radius),
                "x": float(x),
                "y": float(y),
            }
        )

    # JSON
    json_path = os.path.join(run_dir, f"spirale_n{n_val}_k{k}_i{i_val}_j{j_val}.json")
    payload = {
        "n": int(n_val),
        "k": int(k),
        "i": int(i_val),
        "j": int(j_val),
        "reason": reason,
        "steps": int(len(seq)),
        "peak": int(peak),
        "points": points,
    }
    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # Plot (optional)
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        plt = None

    if plt is not None and points:
        try:
            xs = np.array([p["x"] for p in points], dtype=float)
            ys = np.array([p["y"] for p in points], dtype=float)
            steps = np.array([p["step"] for p in points], dtype=float)
            cmap = plt.get_cmap("viridis")

            fig, ax = plt.subplots(figsize=(6, 6))
            sc = ax.scatter(xs, ys, c=steps, cmap=cmap, s=8)
            ax.plot(xs, ys, linewidth=0.6, alpha=0.6, color="gray")
            ax.set_title(f"Spirale n={n_val} k={k} i={i_val} j={j_val} (angle={angle_mode})")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.axis("equal")
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label("step")
            fig.tight_layout()
            png_path = os.path.join(run_dir, f"spirale_n{n_val}_k{k}_i{i_val}_j{j_val}.png")
            try:
                fig.savefig(png_path, dpi=150)
            except Exception:
                pass
            plt.close(fig)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass


def _spirale_multi_k(
    *,
    n_val: int,
    k_values: List[int],
    run_dir: str,
    i_val: int,
    j_val: int,
    angle_mode: str,
    max_iters: int,
    divergence_threshold: float,
    alternated: bool,
    alt_m: int,
    out_label: str,
    title_label: str,
) -> None:
    os.makedirs(run_dir, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    panels: List[Tuple[int, List[Dict[str, object]]]] = []

    for k in k_values:
        if alt_m >= k or i_val >= k or i_val < 1:
            summary_rows.append({"k": int(k), "skipped": True, "reason": "alt_m>=k or invalid i"})
            continue

        seq: List[int] = []
        residues: List[int] = []
        seen = set()
        t = int(n_val)
        reason = None
        peak = t
        for step in range(max_iters):
            seq.append(int(t))
            r = int(abs(t) % k)
            residues.append(int(r))
            if int(t) > peak:
                peak = int(t)
            if t in seen:
                reason = "cycle"
                break
            seen.add(t)
            if abs(t) > divergence_threshold:
                reason = "divergence_threshold"
                break
            t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
        else:
            reason = "max_iters"

        two_pi = 2.0 * math.pi
        points: List[Dict[str, object]] = []
        for idx, (val, r) in enumerate(zip(seq, residues)):
            if angle_mode == "residue":
                angle = two_pi * (r % k) / float(k)
            else:
                angle = idx * (two_pi / float(k))
            radius = math.log(abs(val) + 1.0)
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append(
                {
                    "step": idx,
                    "t": int(val),
                    "residue": int(r),
                    "x": float(x),
                    "y": float(y),
                    "angle": float(angle),
                    "radius": float(radius),
                }
            )

        summary_rows.append(
            {
                "k": int(k),
                "n": int(n_val),
                "i": int(i_val),
                "j": int(j_val),
                "reason": reason,
                "steps": int(len(seq)),
                "peak": int(peak),
                "points": points,
            }
        )
        panels.append((int(k), points))

    out_json = os.path.join(run_dir, f"spirale_n{n_val}_{out_label}_i{i_val}_j{j_val}.json")
    try:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(
                {
                    "n": int(n_val),
                    "i": int(i_val),
                    "j": int(j_val),
                    "angle_mode": angle_mode,
                    "k_values": [int(k) for k in k_values],
                    "rows": summary_rows,
                },
                jf,
                ensure_ascii=False,
                indent=2,
            )
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        plt = None

    if plt is not None and panels:
        try:
            n_k = len(panels)
            ncols = min(3, max(1, n_k))
            nrows = int((n_k + ncols - 1) / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), squeeze=False)
            for idx, (k, points) in enumerate(panels):
                ax = axes[idx // ncols][idx % ncols]
                if not points:
                    ax.text(0.5, 0.5, "no points", ha="center", va="center", transform=ax.transAxes)
                    continue
                xs = np.array([p["x"] for p in points], dtype=float)
                ys = np.array([p["y"] for p in points], dtype=float)
                steps = np.array([p["step"] for p in points], dtype=float)
                cmap = plt.get_cmap("viridis")
                ax.scatter(xs, ys, c=steps, cmap=cmap, s=6)
                ax.plot(xs, ys, linewidth=0.5, alpha=0.6, color="gray")
                ax.set_title(f"k={k}")
                ax.axis("equal")
            for j in range(len(panels), nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            fig.suptitle(f"Spirale n={n_val} {title_label} i={i_val} j={j_val} (angle={angle_mode})")
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            png_out = os.path.join(run_dir, f"spirale_n{n_val}_{out_label}_i{i_val}_j{j_val}.png")
            try:
                fig.savefig(png_out, dpi=150)
            except Exception:
                pass
            plt.close(fig)
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass


def spirale_p(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    angle_mode: str = "residue",
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Run spirale for all primes k <= pmax and produce a combined PNG + JSON summary."""

    if pmax < 2:
        return

    sieve = [True] * (int(pmax) + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(int(pmax) ** 0.5) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, int(pmax) + 1, ii):
                sieve[jj] = False
    primes = [i for i, is_p in enumerate(sieve) if is_p]

    _spirale_multi_k(
        n_val=int(n_val),
        k_values=primes,
        run_dir=run_dir,
        i_val=int(i_val),
        j_val=int(j_val),
        angle_mode=angle_mode,
        max_iters=int(max_iters),
        divergence_threshold=float(divergence_threshold),
        alternated=bool(alternated),
        alt_m=int(alt_m),
        out_label=f"p{int(pmax)}",
        title_label=f"p={int(pmax)}",
    )


def spirale_all(
    n_val: int,
    pmax: int,
    i_val: int,
    j_val: int,
    run_dir: str,
    angle_mode: str = "residue",
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
) -> None:
    """Run spirale for all integer k in [2, pmax] and produce a combined PNG + JSON summary."""

    if pmax < 2:
        return

    all_k = list(range(2, int(pmax) + 1))
    _spirale_multi_k(
        n_val=int(n_val),
        k_values=all_k,
        run_dir=run_dir,
        i_val=int(i_val),
        j_val=int(j_val),
        angle_mode=angle_mode,
        max_iters=int(max_iters),
        divergence_threshold=float(divergence_threshold),
        alternated=bool(alternated),
        alt_m=int(alt_m),
        out_label=f"all{int(pmax)}",
        title_label=f"all<= {int(pmax)}",
    )
