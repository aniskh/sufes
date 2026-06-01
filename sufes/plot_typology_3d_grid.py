"""CLI entrypoint for combined 3D typology plots.

This module lives in the package and can be called with:

    python3 -m sufes.plot_typology_3d_grid
"""

from __future__ import annotations

import argparse

from .plotting import latest_run_dir, plot_3d_grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create combined 3D typology plots")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="path to run directory (default: latest in ./output/)",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="combined_3d",
        help="prefix for output images",
    )
    args = parser.parse_args(argv)

    run_dir = args.run_dir if args.run_dir is not None else latest_run_dir()
    if not run_dir:
        raise SystemExit("no run directory found under ./output")

    print("Using run dir:", run_dir)
    plot_3d_grid(run_dir, out_prefix=args.out_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
