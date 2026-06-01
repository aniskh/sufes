"""Plot helper for residu_distribution CSV files.

This module was moved from ``tools/plot_residu_distribution_from_csv.py``
so it can be imported and reused from the package.
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


def plot_residu_distribution_from_csv(in_csv: str, out_png: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise SystemExit(f"Error importing matplotlib: {e}")

    if not os.path.exists(in_csv):
        raise SystemExit(f"Input CSV not found: {in_csv}")

    rows_by_k: Dict[int, List[Tuple[int, float, int]]] = defaultdict(list)
    with open(in_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                k = int(r.get("k") or r.get("K") or r.get("prime_k"))
            except Exception:
                continue
            try:
                j = int(r.get("j") or r.get("J") or r.get("step_j"))
            except Exception:
                continue
            mean_str = r.get("mean_residue") or r.get("mean") or r.get("mean_res")
            try:
                mean = float(mean_str) if mean_str not in (None, "") else float("nan")
            except Exception:
                mean = float("nan")
            try:
                cnt = int(r.get("count_non_zero") or r.get("count") or 0)
            except Exception:
                cnt = 0
            rows_by_k[k].append((j, mean, cnt))

    if not rows_by_k:
        raise SystemExit("No data found in CSV")

    ks = sorted(rows_by_k.keys())
    num = len(ks)
    cols = 4
    rows = (num + cols - 1) // cols
    fig_w = min(20, 4 * cols)
    fig_h = max(3, 2.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, k in enumerate(ks):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]
        data = sorted(rows_by_k[k], key=lambda x: x[0])
        js = [t[0] for t in data]
        means = [t[1] for t in data]
        cnts = [t[2] for t in data]
        ax.plot(js, means, marker="o", linestyle="-")
        ax.set_title(f"k={k}")
        ax.set_xlabel("j")
        ax.set_ylabel("mean residue")
        for jv, mv, cv in zip(js, means, cnts):
            if cv < 5:
                ax.annotate(str(cv), (jv, mv), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=6)
        ax.grid(True)

    total = rows * cols
    for idx in range(num, total):
        r = idx // cols
        c = idx % cols
        axes[r][c].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print("Wrote", out_png)


def main(argv: List[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 2:
        print("Usage: python3 -m sufes.plot_residu_distribution_from_csv <input_csv> <output_png>")
        return 1
    in_csv, out_png = args[0], args[1]
    plot_residu_distribution_from_csv(in_csv, out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
