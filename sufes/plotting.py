"""Plotting helpers moved from `plot_typology_3d_grid.py`.

This module provides `plot_3d_grid(run_dir, out_prefix)` which creates
combined 3D surface plots from per-k JSON results produced by the analysis.
"""
import glob
import json
import math
import os

try:
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    plt = None
    np = None


def latest_run_dir(base: str = "output"):
    if not os.path.exists(base):
        return None
    runs = [
        os.path.join(base, d)
        for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d))
    ]
    return max(runs, key=os.path.getmtime) if runs else None


def read_per_k_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_grid_axes(n: int):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols


def build_Zs_for_k(data):
    per_i = data.get("per_i", {})
    if not per_i:
        return None, None, None, None, None, None, None, None, None
    i_keys = sorted([int(x) for x in per_i.keys()])
    k = int(data.get("k", 0))
    if k == 0:
        maxj = 0
        for per_j in per_i.values():
            maxj = max(maxj, max(int(x) for x in per_j.keys()))
        k = maxj + 1
    max_i = max(i_keys)
    Z_lambda = np.full((max_i, k), np.nan)
    Z_divnorm = np.full((max_i, k), np.nan)
    Z_fail = np.full((max_i, k), np.nan)
    Z_stopping_avg = np.full((max_i, k), np.nan)
    Z_stopping_max = np.full((max_i, k), np.nan)
    Z_peak_mean = np.full((max_i, k), np.nan)
    Z_peak_max = np.full((max_i, k), np.nan)
    for i_str, per_j in per_i.items():
        i = int(i_str) - 1
        for j_str, r in per_j.items():
            j = int(j_str)
            gl = r.get("global_lambda")
            if gl is not None:
                Z_lambda[i, j] = gl
            summary = r.get("summary", {})
            total_divs = summary.get("total_div_events", None)
            total = summary.get("total", None)
            if total_divs is not None and total is not None and total > 0:
                Z_divnorm[i, j] = total_divs / total
            num_fail = r.get("num_failures", None)
            if num_fail is not None:
                Z_fail[i, j] = num_fail
            avg_stop = summary.get("avg_stopping_time", None)
            max_stop = summary.get("max_stopping_time", None)
            mean_peak = summary.get("mean_peak", None)
            max_peak = summary.get("max_peak", None)
            if avg_stop is not None:
                Z_stopping_avg[i, j] = avg_stop
            if max_stop is not None:
                Z_stopping_max[i, j] = max_stop
            if mean_peak is not None:
                Z_peak_mean[i, j] = mean_peak
            if max_peak is not None:
                Z_peak_max[i, j] = max_peak
    return Z_lambda, Z_divnorm, Z_fail, Z_stopping_avg, Z_stopping_max, Z_peak_mean, Z_peak_max, max_i, k


def plot_3d_grid(run_dir: str, out_prefix: str = "combined_3d"):
    if plt is None or np is None:
        raise SystemExit("matplotlib/numpy not available — plotting requires these packages")
    files = sorted(glob.glob(os.path.join(run_dir, "results_family_k*_*.json")))
    if not files:
        raise SystemExit(f"no results JSON found in {run_dir}")
    data_list = []
    for fn in files:
        try:
            d = read_per_k_json(fn)
        except Exception:
            continue
        k = int(d.get("k", 0))
        data_list.append((k, d))
    data_list.sort()
    n = len(data_list)
    rows, cols = make_grid_axes(n)

    Zs_lambda = []
    Zs_div = []
    Zs_fail = []
    Zs_stop_avg = []
    Zs_stop_max = []
    Zs_peak_mean = []
    Zs_peak_max = []
    for k, d in data_list:
        Zl, Zd, Zf, Zs_avg, Zs_max, Zp_mean, Zp_max, max_i, k_val = build_Zs_for_k(d)
        if Zl is None:
            continue
        Zs_lambda.append((k, Zl))
        Zs_div.append((k, Zd))
        Zs_fail.append((k, Zf))
        Zs_stop_avg.append((k, Zs_avg))
        Zs_stop_max.append((k, Zs_max))
        Zs_peak_mean.append((k, Zp_mean))
        Zs_peak_max.append((k, Zp_max))

    def plot_surface_grid(Zlist, title, cmap, out_file, zlim=None):
        fig = plt.figure(figsize=(cols * 4, rows * 3.5))
        axes = []
        for idx, (k, Z) in enumerate(Zlist):
            ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
            axes.append(ax)
            ni, nj = Z.shape
            J, I = np.meshgrid(np.arange(nj), np.arange(ni))
            Zmask = np.ma.masked_invalid(Z)
            try:
                ax.plot_surface(J, I, Zmask, cmap=cmap, linewidth=0, antialiased=False)
            except Exception:
                ax.plot_wireframe(J, I, np.nan_to_num(Zmask, nan=0.0))
            ax.set_title(f"k={k}")
            ax.set_xlabel("j")
            ax.set_ylabel("i (1-based)")
            ax.view_init(elev=30, azim=-60)
            if zlim is not None:
                ax.set_zlim(zlim)
        try:
            from matplotlib import cm
            mappable = cm.ScalarMappable(cmap=cmap)
            if zlim is not None:
                mappable.set_clim(*zlim)
            fig.colorbar(mappable, ax=axes, shrink=0.6, pad=0.1)
        except Exception:
            pass
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join(run_dir, out_file), dpi=150)
        plt.close(fig)

    def global_range(Zlist):
        vals = np.concatenate([Z.flatten() for (_, Z) in Zlist if Z.size > 0])
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return None
        return float(np.nanmin(vals)), float(np.nanmax(vals))

    if Zs_lambda:
        rng = global_range(Zs_lambda)
        plot_surface_grid(Zs_lambda, "Global lambda (3D)", "viridis", f"{out_prefix}_global_lambda.png", zlim=rng)
    if Zs_div:
        rng = global_range(Zs_div)
        plot_surface_grid(Zs_div, "Divisions per start (normalized) (3D)", "plasma", f"{out_prefix}_divisions_norm.png", zlim=rng)
    if Zs_fail:
        rng = global_range(Zs_fail)
        plot_surface_grid(Zs_fail, "Num failures (3D)", "inferno", f"{out_prefix}_num_failures.png", zlim=rng)
    if Zs_stop_avg:
        rng = global_range(Zs_stop_avg)
        plot_surface_grid(Zs_stop_avg, "Avg stopping time (3D)", "magma", f"{out_prefix}_stopping_time_avg.png", zlim=rng)
    if Zs_stop_max:
        rng = global_range(Zs_stop_max)
        plot_surface_grid(Zs_stop_max, "Max stopping time (3D)", "cividis", f"{out_prefix}_stopping_time_max.png", zlim=rng)
    if Zs_peak_mean:
        rng = global_range(Zs_peak_mean)
        plot_surface_grid(Zs_peak_mean, "Mean peak value (3D)", "YlOrRd", f"{out_prefix}_peak_mean.png", zlim=rng)
    if Zs_peak_max:
        rng = global_range(Zs_peak_max)
        plot_surface_grid(Zs_peak_max, "Max peak value (3D)", "hot", f"{out_prefix}_peak_max.png", zlim=rng)


__all__ = ["latest_run_dir", "read_per_k_json", "make_grid_axes", "build_Zs_for_k", "plot_3d_grid"]
