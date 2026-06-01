from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .algorithms import next_term_ji

try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:  # pragma: no cover - plotting optional
    plt = None
    np = None


def _simulate_residues(n0: int, k: int, i: int, j: int, max_iters: int, divergence_threshold: float, alternated: bool, alt_m: int) -> Tuple[List[int], int]:
    """Simulate residues starting from n0 and return (path, preperiod_index).

    preperiod_index is the index in path where the cycle starts (0-based). If no
    repeat is found within max_iters, preperiod_index == len(path).
    """
    t = int(n0)
    seen_idx: Dict[int, int] = {}
    path: List[int] = []
    for step in range(int(max_iters)):
        if t in seen_idx:
            # cycle detected; preperiod is the first index where this value occurred
            preperiod = seen_idx[t]
            return path, preperiod
        # record index before appending
        seen_idx[t] = len(path)
        path.append(int(t % k))
        if abs(t) > float(divergence_threshold):
            break
        t = int(next_term_ji(t, k, j, i, alternated=alternated, alt_m=alt_m))
    # no repeat detected within max_iters: treat preperiod as full path length
    return path, len(path)


def _plot_hamming(per_n: List[Dict], out_path: Path, title: Optional[str] = None) -> None:
    """Save a Hamming vs n plot to out_path (PNG)."""
    if plt is None or np is None:
        return

    xs = [p["n"] for p in per_n]
    ys = [float(p["hamming"]) if p.get("hamming") is not None and not math.isnan(float(p.get("hamming"))) else np.nan for p in per_n]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, ys, marker="o", linestyle="-", color="#1f77b4")
    ax.set_xlabel("n0")
    ax.set_ylabel("Hamming distance")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def hamming(
    n: int,
    k: int,
    i: int,
    j: int,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    write_png: bool = True,
    png_prefix: Optional[str] = None,
    workers: int = 4,
) -> Dict:
    """Compute Hamming distance between residue sequences of n0 and n0+1 for all n0 in 1..n.

    Returns dict: {n, k, i, j, per_n: [{n, len_a, len_b, hamming}], mean_hamming}

    By default this function will save a PNG named
    `hamming_n{n}_k{k}_i{i}_j{j}.png` in the current working directory. You can disable
    saving by passing write_png=False or customize the filename prefix with png_prefix.
    """
    N = int(n)
    def _one(n0: int) -> Dict:
        seq_a, pre_a = _simulate_residues(n0, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
        seq_b, pre_b = _simulate_residues(n0 + 1, k, i, j, max_iters, divergence_threshold, alternated, alt_m)
        s = min(pre_a, pre_b)
        if s <= 0:
            hval = math.nan
        else:
            La = len(seq_a)
            Lb = len(seq_b)
            effective_s = min(s, La, Lb)
            if effective_s <= 0:
                hval = math.nan
            else:
                diffs = sum(1 for t in range(effective_s) if int(seq_a[t]) != int(seq_b[t]))
                hval = float(diffs) / float(effective_s)

        return {
            "n": n0,
            "len_a": len(seq_a),
            "len_b": len(seq_b),
            "pre_a": int(pre_a),
            "pre_b": int(pre_b),
            "used_len": int(s),
            "hamming": hval,
        }

    per_n: List[Dict] = []
    ns = list(range(1, N + 1))
    if int(workers) <= 1:
        for r in map(_one, ns):
            per_n.append(r)
    else:
        try:
            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for r in executor.map(_one, ns):
                    per_n.append(r)
        except Exception:
            for r in map(_one, ns):
                per_n.append(r)

    hvals = [p.get("hamming") for p in per_n if p.get("hamming") is not None and not math.isnan(float(p.get("hamming")))]
    mean_hamming = (sum(hvals) / len(hvals)) if hvals else None

    result = {"n": N, "k": int(k), "i": int(i), "j": int(j), "per_n": per_n, "mean_hamming": float(mean_hamming) if mean_hamming is not None else None}

    # Save PNG by default (user requested an image presentation)
    if write_png and plt is not None:
        fname = (png_prefix + "_" if png_prefix else "") + f"hamming_n{N}_k{int(k)}_i{int(i)}_j{int(j)}.png"
        out_path = Path.cwd() / fname
        title = f"Hamming distance (n0 vs n0+1) — n<={N}, k={int(k)}, i={int(i)}, j={int(j)} — mean={result['mean_hamming']:.4f}" if result["mean_hamming"] is not None else f"Hamming (n<={N}, k={int(k)})"
        try:
            _plot_hamming(per_n, out_path, title=title)
        except Exception:
            # Don't fail the computation if plotting fails
            pass

    return result


__all__ = ["hamming"]


def _sieve_primes(upto: int) -> List[int]:
    if upto < 2:
        return []
    sieve = bytearray(b"\x01") * (upto + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(upto ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start: upto + 1: step] = b"\x00" * (((upto - start) // step) + 1)
    return [i for i, isprime in enumerate(sieve) if isprime]


def hamming_p(n: int, p: int, i: int, j: int, run_dir: str, *, all_j: bool = False, max_iters: int = 500_000, divergence_threshold: float = 1e18, alternated: bool = False, alt_m: int = 1) -> Dict:
    """Compute Hamming for all primes k <= p and save a single figure with subplots (one per k).

    Returns a combined summary dict with per-k summaries and writes a PNG inside run_dir.
    """
    primes = _sieve_primes(p)
    summaries = []
    per_k_data = []
    for k in primes:
        if not all_j:
            try:
                summ = hamming(n, k, i, j, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m, write_png=False)
            except Exception:
                summ = {"error": True, "k": k}
            summaries.append(summ)
            per_k_data.append((k, summ))
        else:
            # loop on all j for this k
            per_j = []
            for jj in range(0, k):
                try:
                    summj = hamming(n, k, i, jj, max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m, write_png=False)
                except Exception:
                    summj = {"error": True, "k": k, "j": jj}
                per_j.append((jj, summj))
            summaries.append({"k": k, "per_j": [s for (_, s) in per_j]})
            per_k_data.append((k, per_j))

    # Plotting
    if plt is not None and np is not None and per_k_data:
        if not all_j:
            # one subplot per k in a single figure (as before)
            nplots = len(per_k_data)
            cols = min(4, nplots)
            rows = (nplots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
            for idx, (k, summ) in enumerate(per_k_data):
                r = idx // cols
                c = idx % cols
                ax = axes[r][c]
                per_n = summ.get("per_n", [])
                xs = [p0["n"] for p0 in per_n]
                ys = [float(p0["hamming"]) if p0.get("hamming") is not None and not math.isnan(float(p0.get("hamming"))) else np.nan for p0 in per_n]
                ax.plot(xs, ys, marker=".", linestyle="-", ms=3)
                ax.set_ylim(-0.02, 1.02)
                ax.set_title(f"k={k} mean={summ.get('mean_hamming'):.3f}" if summ.get("mean_hamming") is not None else f"k={k}")
                ax.grid(True, linestyle="--", alpha=0.3)
                if r == rows - 1:
                    ax.set_xlabel("n0")
            # hide unused axes
            for idx in range(len(per_k_data), rows * cols):
                r = idx // cols
                c = idx % cols
                axes[r][c].axis('off')
            fig.tight_layout()
            out_png = Path(run_dir) / f"hamming_p{p}_n{n}.png"
            try:
                out_png.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(str(out_png), dpi=150)
            except Exception:
                pass
            plt.close(fig)
        else:
            # produce one image per k, with a subplot per j
            for k, per_j in per_k_data:
                nplots = len(per_j)
                cols = min(4, nplots) if nplots > 0 else 1
                rows = (nplots + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows), squeeze=False)
                for idx, (jj, summj) in enumerate(per_j):
                    r = idx // cols
                    c = idx % cols
                    ax = axes[r][c]
                    per_n = summj.get("per_n", [])
                    xs = [p0["n"] for p0 in per_n]
                    ys = [float(p0["hamming"]) if p0.get("hamming") is not None and not math.isnan(float(p0.get("hamming"))) else np.nan for p0 in per_n]
                    ax.plot(xs, ys, marker=".", linestyle="-", ms=3)
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_title(f"j={jj} mean={summj.get('mean_hamming'):.3f}" if summj.get("mean_hamming") is not None else f"j={jj}")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    if r == rows - 1:
                        ax.set_xlabel("n0")
                # hide unused axes
                for idx in range(len(per_j), rows * cols):
                    r = idx // cols
                    c = idx % cols
                    axes[r][c].axis('off')
                fig.tight_layout()
                out_png = Path(run_dir) / f"hamming_k{k}_p{p}_n{n}.png"
                try:
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(str(out_png), dpi=150)
                except Exception:
                    pass
                plt.close(fig)

    combined = {"n": int(n), "p": int(p), "i": int(i), "j": int(j), "summaries": summaries, "all_j": bool(all_j)}
    return combined
