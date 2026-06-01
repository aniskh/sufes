from __future__ import annotations

import math
import os
from typing import List, Dict

from .algorithms import next_term_ji


def _plot_residues_3d(residues: List[int], k: int, out_png: str, title: str = "residues 3D spiral") -> None:
    """Plot residues as angles on a 3D spiral (spring-like)."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import numpy as np
    except Exception:
        # matplotlib not available — skip plotting
        return

    if not residues:
        return

    # Map residues r in [0, k-1] to angle theta = 2*pi*r/k
    thetas = np.array([2.0 * math.pi * (int(r) % k) / float(k) for r in residues])

    # Steps along the x-axis (helix axis)
    steps = np.arange(len(residues))
    step_width = 0.2
    x = steps * step_width

    # radius may be constant or vary slowly to emphasize the spring
    base_radius = 1.0
    radius = base_radius + 0.05 * np.sin(0.2 * steps)

    # Helix around x-axis: y = R*cos(theta), z = R*sin(theta)
    y = radius * np.cos(thetas)
    z = radius * np.sin(thetas)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    # Draw connecting line to emphasize spring; use lighter alpha
    ax.plot(x, y, z, linewidth=1.2, alpha=0.7, color='C0')
    sc = ax.scatter(x, y, z, c=residues, cmap='viridis', s=18)

    ax.set_title(title)
    ax.set_xlabel('step (x)')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    try:
        fig.colorbar(sc, ax=ax, label='residue')
    except Exception:
        pass

    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
        print(f"3D residues spiral saved to {out_png}")
    except Exception:
        print(f"Could not save 3D residues spiral to {out_png}")
    plt.close(fig)


def _plot_pearson_vs_n(summary: Dict, out_png: str, title: str = "pearson vs n") -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return

    per_n = summary.get('per_n', [])
    if not per_n:
        return

    ns = [int(p.get('n')) for p in per_n]
    pears = []
    for p in per_n:
        v = p.get('pearson')
        try:
            if v is None:
                pears.append(float('nan'))
            else:
                pears.append(float(v))
        except Exception:
            pears.append(float('nan'))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(ns, pears, marker='.', linestyle='-', linewidth=0.8)
    ax.set_xlabel('n')
    ax.set_ylabel('pearson')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
    except Exception:
        pass
    plt.close(fig)


def _pearson_from_lists(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y) or n < 2:
        return math.nan
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n
    if var_x <= 0 or var_y <= 0:
        return math.nan
    return cov / math.sqrt(var_x * var_y)


def _simulate_residues(n0: int, k: int, i: int, j: int, max_iters: int, divergence_threshold: float, alternated: bool, alt_m: int) -> List[int]:
    t = int(n0)
    seen = set()
    path: List[int] = []
    for step in range(int(max_iters)):
        if t in seen:
            break
        seen.add(t)
        path.append(int(t % k))
        if abs(t) > float(divergence_threshold):
            break
        t = int(next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m))
    return path


def _simulate_full_sequence(n0: int, k: int, i: int, j: int, max_iters: int, divergence_threshold: float, alternated: bool, alt_m: int) -> List[int]:
    """Simulate full trajectory values t (not only residues) until cycle/divergence."""
    t = int(n0)
    seen = set()
    seq: List[int] = []
    for step in range(int(max_iters)):
        if t in seen:
            break
        seen.add(t)
        seq.append(int(t))
        if abs(t) > float(divergence_threshold):
            break
        t = int(next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m))
    return seq


def _plot_multipliers(seq: List[int], k: int, out_png: str, title: str = "multipliers vs step") -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not seq:
        return

    # multiplier q at each step: q = t // k (integer division)
    qs = [int(t // k) for t in seq]
    steps = list(range(len(qs)))

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, qs, marker='.', linestyle='-', linewidth=0.9)
    ax.set_xlabel('step')
    ax.set_ylabel(f'multiplier q (floor(t / {k}))')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.4)
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
        print(f"Multiplier plot saved to {out_png}")
    except Exception:
        pass
    plt.close(fig)


def _plot_multipliers_both(seq_a: List[int], seq_b: List[int], k: int, out_png: str, label_a: str = 'n', label_b: str = 'n+1', title: str = 'multipliers vs step') -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    qs_a = [int(t // k) for t in seq_a] if seq_a else []
    qs_b = [int(t // k) for t in seq_b] if seq_b else []
    steps_a = list(range(len(qs_a)))
    steps_b = list(range(len(qs_b)))

    fig, ax = plt.subplots(figsize=(8, 3))
    if qs_a:
        ax.plot(steps_a, qs_a, marker='o', linestyle='-', linewidth=0.9, markersize=4, label=label_a, color='C0')
    if qs_b:
        ax.plot(steps_b, qs_b, marker='x', linestyle='-', linewidth=0.9, markersize=4, label=label_b, color='C1')

    ax.set_xlabel('step')
    ax.set_ylabel(f'multiplier q (floor(t / {k}))')
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.4)
    try:
        ax.legend()
    except Exception:
        pass
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
        print(f"Combined multiplier plot saved to {out_png}")
    except Exception:
        pass
    plt.close(fig)


def _plot_residues_both(res_a: List[int], res_b: List[int], k: int, out_png: str, label_a: str = 'n', label_b: str = 'n+1', title: str = 'residues vs step') -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps_a = list(range(len(res_a)))
    steps_b = list(range(len(res_b)))

    fig, ax = plt.subplots(figsize=(8, 3))
    if res_a:
        ax.plot(steps_a, res_a, marker='o', linestyle='-', linewidth=0.9, markersize=4, label=label_a, color='C0')
    if res_b:
        ax.plot(steps_b, res_b, marker='x', linestyle='-', linewidth=0.9, markersize=4, label=label_b, color='C1')

    ax.set_xlabel('step')
    ax.set_ylabel(f'residue (mod {k})')
    ax.set_title(title)
    ax.set_ylim(-0.5, max(k - 0.5, 1.5))
    ax.set_yticks(list(range(0, k)))
    ax.grid(True, linestyle=':', alpha=0.4)
    try:
        ax.legend()
    except Exception:
        pass
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=150)
        print(f"Combined residues plot saved to {out_png}")
    except Exception:
        pass
    plt.close(fig)


def dirichlet(n: int, k: int, i: int, j: int, *, max_iters: int = 500_000, divergence_threshold: float = 1e18, alternated: bool = False, alt_m: int = 1) -> Dict:
    """Compute Pearson between residue sequences of n and n+1 (single n only).

    Older behaviour looped over n0=1..N; this function now treats the
    argument `n` as the specific starting value to analyse. Returns a
    summary dict with `per_n` containing a single entry for the provided n.
    """
    n0 = int(n)
    seq_a = _simulate_residues(n0, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
    seq_b = _simulate_residues(n0 + 1, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
    L = min(len(seq_a), len(seq_b))
    if L < 2:
        corr = math.nan
    else:
        xs = [float(v) for v in seq_a[:L]]
        ys = [float(v) for v in seq_b[:L]]
        corr = _pearson_from_lists(xs, ys)
    # Hamming distance: proportion of positions t in 0..L-1 where residues differ
    if L <= 0:
        hamming = math.nan
    else:
        diffs = sum(1 for t in range(L) if int(seq_a[t]) != int(seq_b[t]))
        hamming = float(diffs) / float(L)

    per_n = [{
        "n": n0,
        "pearson": float(corr) if not (corr is None) else math.nan,
        "len_a": len(seq_a),
        "len_b": len(seq_b),
        "hamming": hamming,
    }]
    vals = [p.get("pearson") for p in per_n if p.get("pearson") is not None and not math.isnan(float(p.get("pearson")))]
    mean_pearson = (sum(vals) / len(vals)) if vals else None
    # mean hamming across per_n entries (here only one entry)
    hvals = [p.get("hamming") for p in per_n if p.get("hamming") is not None and not math.isnan(float(p.get("hamming")))]
    mean_hamming = (sum(hvals) / len(hvals)) if hvals else None

    return {
        "n": n0,
        "k": int(k),
        "i": int(i),
        "j": int(j),
        "per_n": per_n,
        "mean_pearson": float(mean_pearson) if mean_pearson is not None else None,
        "mean_hamming": float(mean_hamming) if mean_hamming is not None else None,
    }


def dirichlet_with_plots(n: int, k: int, i: int, j: int, out_dir: str | None = None, *, max_iters: int = 500_000, divergence_threshold: float = 1e18, alternated: bool = False, alt_m: int = 1, save_3d: bool = False, sample_n0: int | None = None) -> Dict:
    """Run dirichlet analysis and optionally save 3D spiral plots of residues.

    If save_3d is True and sample_n0 is provided, a 3D spiral PNG for that n0
    and for n0+1 will be written into out_dir with informative filenames.
    """
    summary = dirichlet(n, k, i, j, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m)

    if not save_3d or out_dir is None:
        return summary

    # Ensure output dir exists
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    # If sample_n0 not provided, use the input n (we analyse n and n+1)
    if sample_n0 is None:
        sample_n0 = int(n)

    seq_a = _simulate_residues(sample_n0, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
    seq_b = _simulate_residues(sample_n0 + 1, k, i, j, max_iters, divergence_threshold, alternated, alt_m)

    out_a = os.path.join(out_dir, f"dirichlet_n{sample_n0}_k{k}_i{i}_j{j}_residues3d.png")
    out_b = os.path.join(out_dir, f"dirichlet_n{sample_n0+1}_k{k}_i{i}_j{j}_residues3d.png")

    _plot_residues_3d(seq_a, k, out_a, title=f"residues (n={sample_n0}) k={k}")
    _plot_residues_3d(seq_b, k, out_b, title=f"residues (n={sample_n0+1}) k={k}")

    # Combined 2D residues plot (n vs n+1) with different colors
    try:
        out_res_both = os.path.join(out_dir, f"dirichlet_n{sample_n0}_k{k}_i{i}_j{j}_residues_both.png")
        _plot_residues_both(seq_a, seq_b, k, out_res_both, label_a=f"n={sample_n0}", label_b=f"n={sample_n0+1}", title=f"residues (n vs n+1) k={k}")
    except Exception:
        pass

    # Generate a combined multiplier plot (q = floor(t / k)) for n and n+1
    try:
        seq_full_a = _simulate_full_sequence(sample_n0, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
        seq_full_b = _simulate_full_sequence(sample_n0 + 1, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
        out_q = os.path.join(out_dir, f"dirichlet_n{sample_n0}_k{k}_i{i}_j{j}_multipliers_both.png")
        _plot_multipliers_both(seq_full_a, seq_full_b, k, out_q, label_a=f"n={sample_n0}", label_b=f"n={sample_n0+1}", title=f"multipliers (n={sample_n0} vs n={sample_n0+1}) k={k}")
    except Exception:
        pass

    # Also generate a pearson vs n plot
    try:
        out_pearson = os.path.join(out_dir, f"dirichlet_n{n}_k{k}_pearson_vs_n.png")
        _plot_pearson_vs_n(summary, out_pearson, title=f"Dirichlet Pearson (k={k}) n=1..{n}")
    except Exception:
        pass

    return summary


__all__ = ["dirichlet"]
