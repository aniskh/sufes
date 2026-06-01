"""Feature: cycle lengths for all start values 1..N for a fixed (k,i,j).

This feature runs the usual (k,i,j) recurrence for every start value n in 1..N and
reports the cycle length if the iteration converges to a cycle within the given limits.

CLI wiring is done in `sufes.core`.

Outputs
-------
Writes:

- `cycle_N{N}_k{k}_i{i}_j{j}.json`  (list of rows)
- `cycle_N{N}_k{k}_i{i}_j{j}.csv`   (one row per n)
- optional `cycle_N{N}_k{k}_i{i}_j{j}.png` histogram of cycle lengths (if matplotlib)

Row fields:

- n,k,i,j
- steps: number of simulated steps until stop
- reason: one of {cycle, divergence_threshold, max_iters}
- preperiod: index where cycle starts (if cycle)
- cycle_length: len(cycle) (if cycle)
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Dict, List, Optional

from .algorithms import next_term_ji


def _egcd(a: int, b: int):
    """Extended Euclid: returns (g, x, y) such that a*x + b*y = g = gcd(a,b)."""
    a = int(a)
    b = int(b)
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = _egcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return (g, x, y)


### NOTE: the cycle feature used to compute an `alpha` helper derived from (k+i)/j.
### This was removed by request; keep imports minimal.


@dataclass
class CycleResult:
    n: int
    k: int
    i: int
    j: int
    steps: int
    reason: str
    preperiod: Optional[int]
    cycle_length: Optional[int]
    cycle_sample: Optional[List[int]]
    cycle_key: Optional[str]


def _canonical_cycle(c: List[int]) -> List[int]:
    """Canonicalize a cycle up to rotation (minimal rotation).

    We keep it as a list to make it JSON-friendly.
    """

    if not c:
        return []
    # Remove possible closing element if present (a,b,c,a)
    if len(c) >= 2 and c[0] == c[-1]:
        c = c[:-1]
    if not c:
        return []
    rotations = [c[i:] + c[:i] for i in range(len(c))]
    return min(rotations)


def cycle_length_for(
    n_val: int,
    k: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    max_cycle_sample: int = 128,
) -> CycleResult:
    """Compute the cycle length for a single start value.

    Uses a seen-map (value -> first step index). This is consistent with other
    features and is deterministic.
    """

    seen: Dict[int, int] = {}
    seq: List[int] = []

    t = int(n_val)
    reason = "max_iters"
    preperiod: Optional[int] = None
    cycle_length: Optional[int] = None
    cycle_sample: Optional[List[int]] = None
    cycle_key: Optional[str] = None

    for step in range(int(max_iters)):
        if abs(t) > float(divergence_threshold):
            reason = "divergence_threshold"
            break

        if t in seen:
            preperiod = seen[t]
            reason = "cycle"
            # `t` is the first repeated state; build the explicit cycle as:
            #   seq[preperiod:] followed by the repeated state t, so the cycle
            #   representation is closed (useful for debugging/plots).
            cyc_open = seq[preperiod:]
            cycle_length = int(len(cyc_open))
            cyc_closed = list(cyc_open)
            cyc_closed.append(int(t))

            # Canonical key (rotation-invariant) for cardinality counting.
            # We canonicalize using the *open* cycle (no repeated closing elem).
            cyc_can = _canonical_cycle([int(x) for x in cyc_open])
            cycle_key = json.dumps(cyc_can, ensure_ascii=False)

            if int(max_cycle_sample) > 0:
                cycle_sample = [int(x) for x in cyc_closed[: int(max_cycle_sample)]]
            break

        seen[t] = step
        seq.append(int(t))
        t = next_term_ji(t, int(k), int(j), int(i), alternated=bool(alternated), alt_m=int(alt_m))

    return CycleResult(
        n=int(n_val),
        k=int(k),
        i=int(i),
        j=int(j),
        # Steps is the number of generated states in `seq`.
        steps=int(len(seq)),
        reason=str(reason),
        preperiod=preperiod,
        cycle_length=cycle_length,
        cycle_sample=cycle_sample,
        cycle_key=cycle_key,
    )


def run_cycle_feature(
    n_val: int,
    k: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    cycle_cardinality: bool = False,
    special_cycles: bool = False,
    extra_special_cycles: bool = False,
    j_multiple: int = 1,
    fst_appearance: bool = False,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    workers: int = 4,
) -> None:
    """Run the cycle feature for all start values 1..N.

    Note: The parameter is named `n_val` for consistency with other features, but it
    represents N (upper bound) here.
    """

    os.makedirs(run_dir, exist_ok=True)

    N = int(n_val)
    if N < 1:
        raise SystemExit(f"--cycle-n must be >= 1 (got {N})")

    rows: List[CycleResult] = []
    cycle_counts: Dict[str, int] = {}
    first_appears: Dict[str, int] = {}

    def _compute(n0: int) -> CycleResult:
        return cycle_length_for(
            n0,
            int(k),
            int(i),
            int(j),
            run_dir,
            max_iters=max_iters,
            divergence_threshold=divergence_threshold,
            alternated=alternated,
            alt_m=alt_m,
            max_cycle_sample=8,
        )

    # If j_multiple > 1 the caller may want to run multiple j values. However
    # run_cycle_feature is defined for a fixed j: the responsibility to call it
    # multiple times for different j values is handled by the caller (core.py).
    # Here we only compute for the provided j value across n=1..N.
    n_list = list(range(1, N + 1))
    if int(workers) <= 1:
        for r in map(_compute, n_list):
            rows.append(r)
    else:
        try:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                for r in executor.map(_compute, n_list):
                    rows.append(r)
        except Exception:
            for r in map(_compute, n_list):
                rows.append(r)

    if cycle_cardinality:
        for r in rows:
            if r.reason == "cycle" and r.cycle_key is not None:
                ck = str(r.cycle_key)
                cycle_counts[ck] = int(cycle_counts.get(ck, 0)) + 1

    if fst_appearance:
        # For each distinct canonical cycle, record the smallest n that produced it.
        for r in rows:
            if r.reason != "cycle" or r.cycle_key is None:
                continue
            ck = str(r.cycle_key)
            prev = first_appears.get(ck)
            if prev is None or int(r.n) < int(prev):
                first_appears[ck] = int(r.n)

    base = f"cycle_N{N}_k{int(k)}_i{int(i)}_j{int(j)}"
    out_json = os.path.join(run_dir, f"{base}.json")
    out_csv = os.path.join(run_dir, f"{base}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Write number of distinct canonical cycles for this (k,i,j)
    try:
        distinct_keys = {str(r.cycle_key) for r in rows if r.reason == "cycle" and r.cycle_key is not None}
        out_dist_csv = os.path.join(run_dir, f"{base}_distinct_cycles.csv")
        with open(out_dist_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "i", "j", "nbre_cycles_distincts"])
            w.writerow([int(k), int(i), int(j), int(len(distinct_keys))])
    except Exception:
        pass

    if cycle_cardinality:
        card_rows = [{"k": int(k), "i": int(i), "j": int(j), "N": int(N), "cycle": cyc, "count": int(cnt)} for cyc, cnt in cycle_counts.items()]
        with open(os.path.join(run_dir, f"{base}_cardinality.json"), "w", encoding="utf-8") as f:
            json.dump(card_rows, f, ensure_ascii=False, indent=2)
        with open(os.path.join(run_dir, f"{base}_cardinality.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "i", "j", "N", "cycle", "count"])
            w.writeheader()
            for row in card_rows:
                w.writerow(row)
        # Special-cycles: check whether all n in 1..N produced a cycle and all
        # cycles have the same length. If so print a one-line table and write a
        # tiny CSV.
        if special_cycles:
            try:
                def _uniform(rows: List[CycleResult]):
                    if not rows:
                        return False, None
                    # must all be cycle and have a non-None integer cycle_length
                    if not all((r.reason == "cycle" and r.cycle_length is not None) for r in rows):
                        return False, None
                    vals = {int(r.cycle_length) for r in rows}
                    if len(vals) == 1:
                        return True, vals.pop()
                    return False, None

                ok, size = _uniform(rows)
                if ok:
                    # Even if all cycle lengths are identical, there may be
                    # multiple distinct canonical cycles of the same length.
                    cycles = sorted({str(r.cycle_key) for r in rows if r.reason == "cycle" and r.cycle_key is not None})
                    print("special-cycles: k,i,j,cycle_size,cycles")
                    print(f"{int(k)},{int(i)},{int(j)},{int(size)},{json.dumps(cycles, ensure_ascii=False)}")
                    try:
                        out_special = os.path.join(run_dir, f"{base}_special_cycles.csv")
                        with open(out_special, "w", newline="", encoding="utf-8") as f:
                            w = csv.writer(f)
                            w.writerow(["k", "i", "j", "cycle_size", "cycles"])
                            w.writerow([int(k), int(i), int(j), int(size), json.dumps(cycles, ensure_ascii=False)])
                    except Exception:
                        pass
                else:
                    print(f"special-cycles: none for k={k}, i={i}, j={j}")
            except Exception:
                pass
        # Print a concise cardinality summary to the console
        try:
            TOP_M = 10

            def _short_cycle_label(cycle_key: str) -> str:
                try:
                    vals = json.loads(cycle_key)
                    if isinstance(vals, list):
                        if len(vals) > 6:
                            return f"{vals[:6]}…"
                        return str(vals)
                except Exception:
                    pass
                return cycle_key[:120] + ("…" if len(cycle_key) > 120 else "")

            distinct = len(card_rows)
            print(f"cycle: cardinality: distinct_cycles={distinct}")
            for cyc, cnt in sorted(cycle_counts.items(), key=lambda kv: kv[1], reverse=True)[:TOP_M]:
                print(f"  {int(cnt):5d}  {_short_cycle_label(str(cyc))}")
        except Exception:
            pass

    if fst_appearance:
        base_fst = f"{base}_fst_appearance"
        fst_rows = [
            {"k": int(k), "i": int(i), "j": int(j), "N": int(N), "cycle": str(ck), "first_n": int(n0)}
            for ck, n0 in sorted(first_appears.items(), key=lambda t: int(t[1]))
        ]
        try:
            with open(os.path.join(run_dir, f"{base_fst}.json"), "w", encoding="utf-8") as f:
                json.dump(fst_rows, f, ensure_ascii=False, indent=2)
            with open(os.path.join(run_dir, f"{base_fst}.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["k", "i", "j", "N", "cycle", "first_n"])
                w.writeheader()
                for row in fst_rows:
                    w.writerow(row)
        except Exception:
            pass
        try:
            print(f"cycle: fst-appearance: distinct_cycles={len(fst_rows)}")
            for row in fst_rows[:15]:
                print(f"  first_n={row['first_n']} cycle={row['cycle']}")
            if len(fst_rows) > 15:
                print(f"  ... (+{len(fst_rows)-15} more)")
        except Exception:
            pass

    # Extra-special-cycles: detect whether all n=1..N have the exact same canonical
    # cycle (not only the same length). This is stricter than `special_cycles` and
    # reports the canonical cycle if uniform. This check runs independently of
    # --cycle-cardinality.
    if extra_special_cycles:
        try:
            if not rows:
                print(f"extra-special-cycles: none for k={k}, i={i}, j={j} (no rows)")
            else:
                # require all to be cycles with a non-null cycle_key
                if all((r.reason == "cycle" and r.cycle_key is not None) for r in rows):
                    keys = {str(r.cycle_key) for r in rows}
                    if len(keys) == 1:
                        ck = keys.pop()
                        try:
                            cyc_list = json.loads(ck)
                            size = len(cyc_list) if isinstance(cyc_list, list) else None
                        except Exception:
                            size = None
                        print("extra-special-cycles: k,i,j,cycle")
                        print(f"{int(k)},{int(i)},{int(j)},{ck}")
                        try:
                            out_extra = os.path.join(run_dir, f"{base}_extra_special_cycles.csv")
                            with open(out_extra, "w", newline="", encoding="utf-8") as f:
                                w = csv.writer(f)
                                w.writerow(["k", "i", "j", "cycle"])
                                w.writerow([int(k), int(i), int(j), ck])
                        except Exception:
                            pass
                    else:
                        print(f"extra-special-cycles: none for k={k}, i={i}, j={j} (multiple canonical cycles)")
                else:
                    print(f"extra-special-cycles: none for k={k}, i={i}, j={j} (not all n produced cycles)")
        except Exception:
            pass

    # Optional plots
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.ticker import MaxNLocator  # type: ignore
        # y-values for successes; failures are represented as NaN (so they don't
        # draw, but the x-axis stays aligned).
        xs = [r.n for r in rows]
        ys = [float(r.cycle_length) if (r.reason == "cycle" and r.cycle_length is not None) else float("nan") for r in rows]

        reason_counts = {
            "cycle": sum(1 for r in rows if r.reason == "cycle"),
            "divergence_threshold": sum(1 for r in rows if r.reason == "divergence_threshold"),
            "max_iters": sum(1 for r in rows if r.reason == "max_iters"),
        }

        fig, (ax_series, ax_reason) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4), gridspec_kw={"width_ratios": [3, 1]})

        ax_series.plot(xs, ys, marker=".", linestyle="-", linewidth=0.8, markersize=2)
        ax_series.set_title("Cycle length per n")
        ax_series.set_xlabel("n")
        ax_series.set_ylabel("cycle_length")
        # Prefer integer ticks for cycle lengths (they're integer-valued)
        try:
            ax_series.yaxis.set_major_locator(MaxNLocator(integer=True))
        except Exception:
            pass
        ax_series.grid(True, alpha=0.25)

        # If there are failures, annotate counts (useful since NaNs are invisible).
        failed = int(N) - int(reason_counts["cycle"])
        ax_series.text(
            0.99,
            0.98,
            f"cycles={reason_counts['cycle']}  failed={failed}",
            transform=ax_series.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        ax_reason.bar(list(reason_counts.keys()), list(reason_counts.values()), color=["#4c78a8", "#f58518", "#e45756"])
        ax_reason.set_title("Stop reasons")
        ax_reason.set_ylabel("count")
        ax_reason.tick_params(axis="x", rotation=20)
        ax_reason.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Cycle stats — N={N}, k={k}, i={i}, j={j}")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, f"{base}.png"), dpi=200)
        plt.close(fig)
    except Exception:
        pass

    # Additionally compute the cumulative maximum of cycle lengths for n<=N
    # and save a small plot showing max_cycle_up_to_n vs n for this (k,i).
    try:
        # build map n -> cycle_length (only consider successful cycles)
        max_by_n = [None] * (N + 1)
        for r in rows:
            if r.reason == "cycle" and r.cycle_length is not None:
                max_by_n[int(r.n)] = int(r.cycle_length)

        cum_max = []
        cur = None
        for n0 in range(1, N + 1):
            val = max_by_n[n0]
            if val is not None:
                cur = val if (cur is None or val > cur) else cur
            cum_max.append(float(cur) if cur is not None else float('nan'))

        try:
            import matplotlib.pyplot as plt  # type: ignore

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(list(range(1, N + 1)), cum_max, linewidth=1.0)
            ax2.set_title(f"Max cycle length up to n (N={N}) — k={k}, i={i}, j={j}")
            ax2.set_xlabel("n")
            ax2.set_ylabel("max_cycle_length_up_to_n")
            ax2.grid(True, alpha=0.25)
            out_png2 = os.path.join(run_dir, f"{base}_max_cycle_up_to_n.png")
            try:
                fig2.tight_layout()
                fig2.savefig(out_png2, dpi=180)
                plt.close(fig2)
            except Exception:
                pass
        except Exception:
            # matplotlib not available: skip this plot
            pass
    except Exception:
        pass

    # compute max cycle length for this (k,i,j)
    max_cycle_len = None
    try:
        vals = [int(r.cycle_length) for r in rows if r.reason == "cycle" and r.cycle_length is not None]
        if vals:
            max_cycle_len = int(max(vals))
    except Exception:
        max_cycle_len = None

    # write small table with the max for this (k,i,j)
    try:
        out_max_json = os.path.join(run_dir, f"{base}_max_cycle.json")
        out_max_csv = os.path.join(run_dir, f"{base}_max_cycle.csv")
        with open(out_max_json, "w", encoding="utf-8") as f:
            json.dump({"k": int(k), "i": int(i), "j": int(j), "N": int(N), "max_cycle_length": max_cycle_len}, f, ensure_ascii=False, indent=2)
        with open(out_max_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["k", "i", "j", "N", "max_cycle_length"])
            w.writerow([int(k), int(i), int(j), int(N), max_cycle_len])
        # Also print a compact one-line CSV to stdout for visibility in run logs
        try:
            print("max-cycle: k,i,j,N,max_cycle_length")
            print(f"{int(k)},{int(i)},{int(j)},{int(N)},{max_cycle_len}")
        except Exception:
            pass
    except Exception:
        pass

    count_cycle = sum(1 for r in rows if r.reason == "cycle")
    print(f"cycle: computed n=1..{N} (cycle={count_cycle}, failed={N - count_cycle})")

    # return max value for potential callers that aggregate across j
    return {"k": int(k), "i": int(i), "j": int(j), "N": int(N), "max_cycle_length": max_cycle_len}


def run_cycle_feature_p(
    n_val: int,
    pmax: int,
    i: int,
    j: int,
    run_dir: str,
    *,
    cycle_cardinality: bool = False,
    special_cycles: bool = False,
    extra_special_cycles: bool = False,
    j_multiple: int = 1,
    fst_appearance: bool = False,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    workers: int = 4,
) -> None:
    """Run the cycle feature for all primes k <= pmax.

    Produces:
    - CSV/JSON concatenating all rows (k,n)
    - one PNG with one subplot per k: cycle_length(n)
    """
    if int(pmax) < 2:
        raise SystemExit(f"--cycle-p must be >= 2 (got {pmax})")

    os.makedirs(run_dir, exist_ok=True)

    N = int(n_val)
    if N < 1:
        raise SystemExit(f"--cycle-n must be >= 1 (got {N})")

    # sieve primes up to pmax
    sieve = [True] * (int(pmax) + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(int(pmax) ** 0.8) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, int(pmax) + 1, ii):
                sieve[jj] = False
    primes = [kk for kk, ok in enumerate(sieve) if ok]

    all_rows: List[CycleResult] = []
    per_k_rows: Dict[int, List[CycleResult]] = {}

    # summary stats per k (cycles only)
    per_k_summary: List[Dict[str, object]] = []

    per_k_cycle_counts: Dict[int, Dict[str, int]] = {}
    per_k_first: Dict[int, Dict[str, int]] = {}
    for k in primes:
        if int(alt_m) >= int(k):
            # skip invalid alternated parameter for this k
            continue
        if cycle_cardinality:
            per_k_cycle_counts[int(k)] = {}
        if fst_appearance:
            per_k_first[int(k)] = {}
        rows_k: List[CycleResult] = []

        if int(workers) <= 1:
            for n0 in range(1, N + 1):
                r = cycle_length_for(
                    n0,
                    k,
                    i,
                    j,
                    run_dir,
                    max_iters=max_iters,
                    divergence_threshold=divergence_threshold,
                    alternated=alternated,
                    alt_m=alt_m,
                    max_cycle_sample=8,
                )
                rows_k.append(r)
                if cycle_cardinality and r.reason == "cycle" and r.cycle_key is not None:
                    ck = str(r.cycle_key)
                    per_k_cycle_counts[int(k)][ck] = int(per_k_cycle_counts[int(k)].get(ck, 0)) + 1
                if fst_appearance and r.reason == "cycle" and r.cycle_key is not None:
                    ck = str(r.cycle_key)
                    prev = per_k_first[int(k)].get(ck)
                    if prev is None or int(r.n) < int(prev):
                        per_k_first[int(k)][ck] = int(r.n)
        else:
            try:
                from concurrent.futures import ThreadPoolExecutor

                args_list = [
                    (
                        n0,
                        int(k),
                        int(i),
                        int(j),
                        run_dir,
                        dict(
                            max_iters=max_iters,
                            divergence_threshold=divergence_threshold,
                            alternated=alternated,
                            alt_m=alt_m,
                            max_cycle_sample=8,
                        ),
                    )
                    for n0 in range(1, N + 1)
                ]

                def _task(tpl):
                    (n0, kk, ii, jj, rd, extras) = tpl
                    return cycle_length_for(
                        n0,
                        kk,
                        ii,
                        jj,
                        rd,
                        max_iters=extras["max_iters"],
                        divergence_threshold=extras["divergence_threshold"],
                        alternated=extras["alternated"],
                        alt_m=extras["alt_m"],
                        max_cycle_sample=extras["max_cycle_sample"],
                    )

                with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                    for r in executor.map(_task, args_list):
                        rows_k.append(r)
                        if cycle_cardinality and r.reason == "cycle" and r.cycle_key is not None:
                            ck = str(r.cycle_key)
                            per_k_cycle_counts[int(k)][ck] = int(per_k_cycle_counts[int(k)].get(ck, 0)) + 1
                        if fst_appearance and r.reason == "cycle" and r.cycle_key is not None:
                            ck = str(r.cycle_key)
                            prev = per_k_first[int(k)].get(ck)
                            if prev is None or int(r.n) < int(prev):
                                per_k_first[int(k)][ck] = int(r.n)
            except Exception:
                for n0 in range(1, N + 1):
                    r = cycle_length_for(
                        n0,
                        k,
                        i,
                        j,
                        run_dir,
                        max_iters=max_iters,
                        divergence_threshold=divergence_threshold,
                        alternated=alternated,
                        alt_m=alt_m,
                        max_cycle_sample=8,
                    )
                    rows_k.append(r)
                    if cycle_cardinality and r.reason == "cycle" and r.cycle_key is not None:
                        ck = str(r.cycle_key)
                        per_k_cycle_counts[int(k)][ck] = int(per_k_cycle_counts[int(k)].get(ck, 0)) + 1
                    if fst_appearance and r.reason == "cycle" and r.cycle_key is not None:
                        ck = str(r.cycle_key)
                        prev = per_k_first[int(k)].get(ck)
                        if prev is None or int(r.n) < int(prev):
                            per_k_first[int(k)][ck] = int(r.n)

        per_k_rows[int(k)] = rows_k
        all_rows.extend(rows_k)

        cyc_lengths = [r.cycle_length for r in rows_k if r.reason == "cycle" and r.cycle_length is not None]
        mean_cycle_length = (sum(int(x) for x in cyc_lengths) / float(len(cyc_lengths))) if cyc_lengths else None
        per_k_summary.append(
            {
                "k": int(k),
                "i": int(i),
                "j": int(j),
                "N": int(N),
                "count_cycle": int(len(cyc_lengths)),
                "count_failed": int(len(rows_k) - len(cyc_lengths)),
                "mean_cycle_length": float(mean_cycle_length) if mean_cycle_length is not None else None,
            }
        )

    base = f"cycle_N{N}_p{int(pmax)}_i{int(i)}_j{int(j)}"
    out_json = os.path.join(run_dir, f"{base}.json")
    out_csv = os.path.join(run_dir, f"{base}.csv")

    out_mean_json = os.path.join(run_dir, f"{base}_mean_by_k.json")
    out_mean_csv = os.path.join(run_dir, f"{base}_mean_by_k.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_rows], f, ensure_ascii=False, indent=2)

    with open(out_mean_json, "w", encoding="utf-8") as f:
        json.dump(per_k_summary, f, ensure_ascii=False, indent=2)

    if all_rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(asdict(r))

    with open(out_mean_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["k", "i", "j", "N", "count_cycle", "count_failed", "mean_cycle_length"])
        w.writeheader()
        for row in per_k_summary:
            w.writerow(row)

    # produce k,i,j,max_cycle_length table for this p-mode run (single j)
    try:
        max_rows: List[Dict[str, object]] = []
        for k, rows_k in per_k_rows.items():
            try:
                vals = [int(r.cycle_length) for r in rows_k if r.reason == "cycle" and r.cycle_length is not None]
                mval = int(max(vals)) if vals else None
            except Exception:
                mval = None
            max_rows.append({"k": int(k), "i": int(i), "j": int(j), "N": int(N), "max_cycle_length": mval})
        out_max_json = os.path.join(run_dir, f"cycle_N{N}_p{int(pmax)}_i{int(i)}_max_by_kj.json")
        out_max_csv = os.path.join(run_dir, f"cycle_N{N}_p{int(pmax)}_i{int(i)}_max_by_kj.csv")
        with open(out_max_json, "w", encoding="utf-8") as f:
            json.dump(max_rows, f, ensure_ascii=False, indent=2)
        with open(out_max_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "i", "j", "N", "max_cycle_length"]) 
            w.writeheader()
            for row in max_rows:
                w.writerow(row)
        # write distinct-cycle counts per k for this p-mode run
        try:
            out_dist_k_csv = os.path.join(run_dir, f"cycle_N{N}_p{int(pmax)}_i{int(i)}_distinct_cycles_by_k.csv")
            with open(out_dist_k_csv, "w", newline="", encoding="utf-8") as f2:
                w2 = csv.writer(f2)
                w2.writerow(["k", "i", "j", "nbre_cycles_distincts"])
                for k, rows_k in per_k_rows.items():
                    distinct_keys = {str(r.cycle_key) for r in rows_k if r.reason == "cycle" and r.cycle_key is not None}
                    w2.writerow([int(k), int(i), int(j), int(len(distinct_keys))])
        except Exception:
            pass
        # Print compact table to stdout for run visibility
        try:
            print("max-cycle (p-mode): k,i,j,N,max_cycle_length")
            for row in max_rows:
                print(f"{int(row['k'])},{int(row['i'])},{int(row['j'])},{int(row['N'])},{row['max_cycle_length']}")
        except Exception:
            pass
    except Exception:
        pass

    # Additionally: for each k produce a subplot that shows the cumulative
    # maximum cycle length up to n (1..N). Save one combined PNG with one
    # subplot per k. This summarizes the worst seen cycle length as n grows.
    try:
        import matplotlib.pyplot as plt  # type: ignore

        ks_sorted = sorted(per_k_rows.keys())
        if ks_sorted:
            ncols = 3
            nrows = (len(ks_sorted) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

            for idx, k in enumerate(ks_sorted):
                ax = axes[idx // ncols][idx % ncols]
                rows_k = per_k_rows.get(k, [])
                # Build max_by_n for this k
                max_by_n = [None] * (N + 1)
                for r in rows_k:
                    if r.reason == "cycle" and r.cycle_length is not None:
                        max_by_n[int(r.n)] = int(r.cycle_length)
                cum = []
                cur = None
                for n0 in range(1, N + 1):
                    v = max_by_n[n0]
                    if v is not None:
                        cur = v if (cur is None or v > cur) else cur
                    cum.append(float(cur) if cur is not None else float('nan'))
                ax.plot(list(range(1, N + 1)), cum, linewidth=0.8)
                ax.set_title(f"k={k}")
                ax.set_xlabel("n")
                ax.set_ylabel("max_cycle_up_to_n")
                ax.grid(True, alpha=0.25)

            for idx in range(len(ks_sorted), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

            fig.suptitle(f"Max cycle length up to n — N={N}, primes k<=p={int(pmax)}, i={int(i)}")
            fig.tight_layout()
            out_png = os.path.join(run_dir, f"cycle_N{N}_p{int(pmax)}_i{int(i)}_max_cycle_up_to_n.png")
            try:
                fig.savefig(out_png, dpi=180)
            except Exception:
                pass
            plt.close(fig)
    except Exception:
        pass


def _plot_extra_special_counts_by_k(run_dir: str, base_allj: str, extra_rows: List[List[object]]) -> None:
    """Plot number of extra-special (k,i,j) combinations per k.

    extra_rows is expected to contain rows like: [k, i, j, cycle_key].
    """

    try:
        per_k: Dict[int, set] = {}
        for row in extra_rows:
            if not row or len(row) < 3:
                continue
            try:
                kk = int(row[0])
                ii = int(row[1])
                jj = int(row[2])
            except Exception:
                continue
            per_k.setdefault(kk, set()).add((ii, jj))

        ks = sorted(per_k.keys())
        counts = [len(per_k[k]) for k in ks]

        # CSV summary
        out_csv = os.path.join(run_dir, f"{base_allj}_extra_special_counts_by_k.csv")
        try:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["k", "extra_special_count"])
                for k, c in zip(ks, counts):
                    w.writerow([int(k), int(c)])
        except Exception:
            pass

        # PNG plot
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.ticker import MaxNLocator  # type: ignore

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([str(k) for k in ks], counts, color="#4c78a8")
        ax.set_title("Extra-special cycles: count of (i,j) combinations per k")
        ax.set_xlabel("k")
        ax.set_ylabel("count")
        try:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        except Exception:
            pass
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        out_png = os.path.join(run_dir, f"{base_allj}_extra_special_counts_by_k.png")
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        pass



def run_cycle_feature_p_all_j(
    n_val: int,
    pmax: int,
    i: int,
    run_dir: str,
    *,
    cycle_cardinality: bool = False,
    special_cycles: bool = False,
    extra_special_cycles: bool = False,
    fst_appearance: bool = False,
    j_multiple: int = 1,
    card_top_cycles: int = 5,
    max_iters: int = 500_000,
    divergence_threshold: float = 1e18,
    alternated: bool = False,
    alt_m: int = 1,
    workers: int = 4,
) -> None:
    """Run cycle feature for all primes k<=p and all j in 0..(j_multiple*k-1).

    This mode can be expensive (roughly sum_k (j_multiple*k * N)).

    Outputs:
    - series table with one row per (k,j,n)
    - mean table with one row per (k,j)
    - plots (if matplotlib) overlaying curves per j, plus mean-by-j per k
    """

    if int(pmax) < 2:
        raise SystemExit(f"--cycle-p must be >= 2 (got {pmax})")

    os.makedirs(run_dir, exist_ok=True)

    N = int(n_val)
    if N < 1:
        raise SystemExit(f"--cycle-n must be >= 1 (got {N})")

    # sieve primes up to pmax
    sieve = [True] * (int(pmax) + 1)
    sieve[0:2] = [False, False]
    for ii in range(2, int(int(pmax) ** 0.8) + 1):
        if sieve[ii]:
            for jj in range(ii * ii, int(pmax) + 1, ii):
                sieve[jj] = False
    primes = [kk for kk, ok in enumerate(sieve) if ok]

    # Full series rows: one row per (n,k,j)
    series_rows: List[Dict[str, object]] = []

    # Mean rows: one row per (k,j)
    mean_rows: List[Dict[str, object]] = []

    # Special / extra-special summaries (accumulated during scan).
    # special: all n<=N converge to cycles of identical size
    # extra-special: all n<=N converge to the exact same canonical cycle
    special_rows: List[List[int]] = []
    extra_rows: List[List[object]] = []

    # For plotting
    per_k_series: Dict[int, Dict[int, List[CycleResult]]] = {}
    per_k_means: Dict[int, List[Optional[float]]] = {}

    # Cardinality: occurrences per distinct canonical cycle.
    # - per (k,j): map cycle_key -> count
    # - per k (aggregated over all j): map cycle_key -> count
    per_kj_cycle_counts: Dict[int, Dict[int, Dict[str, int]]] = {}
    per_k_cycle_counts: Dict[int, Dict[str, int]] = {}

    # First appearance per (k,j): cycle_key -> first n
    per_kj_first: Dict[int, Dict[int, Dict[str, int]]] = {}

    for k in primes:
        if int(alt_m) >= int(k):
            continue

        per_k_series[int(k)] = {}
        per_kj_cycle_counts[int(k)] = {}
        per_k_cycle_counts[int(k)] = {}
        if fst_appearance:
            per_kj_first[int(k)] = {}
        means_for_k: List[Optional[float]] = []

        # allow j in 0..(j_multiple*k - 1)
        j_end = int(k) * int(j_multiple)
        for j in range(0, j_end):
            # Optimization for `--special-cycles`:
            # We can short-circuit the (k,j) combination as soon as we find any n0
            # whose *cycle length* differs from the reference length (taken from
            # n0=1), or any non-cycle.
            #
            # This is sequential by nature and only activated when special_cycles
            # is enabled.
            if special_cycles and not extra_special_cycles:
                ref = cycle_length_for(
                    1,
                    int(k),
                    int(i),
                    int(j),
                    run_dir,
                    max_iters=max_iters,
                    divergence_threshold=divergence_threshold,
                    alternated=alternated,
                    alt_m=alt_m,
                    max_cycle_sample=0,
                )
                if ref.reason != "cycle" or ref.cycle_length is None:
                    continue
                ref_len = int(ref.cycle_length)
                is_special = True
                for n0 in range(2, N + 1):
                    rchk = cycle_length_for(
                        n0,
                        int(k),
                        int(i),
                        int(j),
                        run_dir,
                        max_iters=max_iters,
                        divergence_threshold=divergence_threshold,
                        alternated=alternated,
                        alt_m=alt_m,
                        max_cycle_sample=0,
                    )
                    if rchk.reason != "cycle" or rchk.cycle_length is None or int(rchk.cycle_length) != ref_len:
                        is_special = False
                        break
                if not is_special:
                    continue

            # Optimization for `--extra-special-cycles`:
            # If the user only cares about extra-special cycles, we can short-circuit
            # the (k,j) combination as soon as we find any n0 whose canonical cycle
            # differs from the reference cycle (taken from n0=1), or any non-cycle.
            #
            # IMPORTANT: we still need full per-n data for outputs (series/means/
            # cardinalities) for the combinations we keep, so we only skip the heavy
            # computation when the combo is already proven NOT extra-special.
            #
            # This early check is done sequentially (it necessarily depends on the
            # reference cycle), and is only activated when extra_special_cycles is
            # enabled.
            if extra_special_cycles:
                ref = cycle_length_for(
                    1,
                    int(k),
                    int(i),
                    int(j),
                    run_dir,
                    max_iters=max_iters,
                    divergence_threshold=divergence_threshold,
                    alternated=alternated,
                    alt_m=alt_m,
                    max_cycle_sample=0,
                )
                if ref.reason != "cycle" or ref.cycle_key is None:
                    # Not extra-special: fail fast.
                    continue
                ref_key = str(ref.cycle_key)
                is_extra = True
                for n0 in range(2, N + 1):
                    rchk = cycle_length_for(
                        n0,
                        int(k),
                        int(i),
                        int(j),
                        run_dir,
                        max_iters=max_iters,
                        divergence_threshold=divergence_threshold,
                        alternated=alternated,
                        alt_m=alt_m,
                        max_cycle_sample=0,
                    )
                    if rchk.reason != "cycle" or rchk.cycle_key is None or str(rchk.cycle_key) != ref_key:
                        is_extra = False
                        break
                if not is_extra:
                    # As requested: immediately skip to next (k,j).
                    continue

            rows_kj: List[CycleResult] = []
            cyc_lengths: List[int] = []
            failed = 0
            local_cycle_counts: Dict[str, int] = {}

            # Prepare args for per-n computation
            args_list_n = [
                (n0, int(k), int(i), int(j), run_dir, dict(max_iters=max_iters, divergence_threshold=divergence_threshold, alternated=alternated, alt_m=alt_m, max_cycle_sample=0))
                for n0 in range(1, N + 1)
            ]

            if int(workers) <= 1:
                for tpl in args_list_n:
                    n0 = tpl[0]
                    r = cycle_length_for(
                        n0,
                        int(k),
                        int(i),
                        int(j),
                        run_dir,
                        max_iters=max_iters,
                        divergence_threshold=divergence_threshold,
                        alternated=alternated,
                        alt_m=alt_m,
                        max_cycle_sample=0,
                    )
                    rows_kj.append(r)
                    series_rows.append({
                        "n": int(r.n),
                        "k": int(r.k),
                        "i": int(r.i),
                        "j": int(r.j),
                        "steps": int(r.steps),
                        "reason": str(r.reason),
                        "preperiod": r.preperiod,
                        "cycle_length": r.cycle_length,
                        "cycle_key": r.cycle_key,
                    })
                    if r.reason == "cycle" and r.cycle_length is not None:
                        cyc_lengths.append(int(r.cycle_length))
                        if r.cycle_key is not None:
                            cyc_key = str(r.cycle_key)
                            local_cycle_counts[cyc_key] = int(local_cycle_counts.get(cyc_key, 0)) + 1
                            per_k_cycle_counts[int(k)][cyc_key] = int(per_k_cycle_counts[int(k)].get(cyc_key, 0)) + 1
                    else:
                        failed += 1
            else:
                try:
                    from concurrent.futures import ThreadPoolExecutor

                    def _task_n(arg_tpl):
                        (n0, kk, ii, jj, rd, extras) = arg_tpl
                        return cycle_length_for(
                            n0,
                            kk,
                            ii,
                            jj,
                            rd,
                            max_iters=extras["max_iters"],
                            divergence_threshold=extras["divergence_threshold"],
                            alternated=extras["alternated"],
                            alt_m=extras["alt_m"],
                            max_cycle_sample=extras["max_cycle_sample"],
                        )

                    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                        for r in executor.map(_task_n, args_list_n):
                            rows_kj.append(r)
                            series_rows.append({
                                "n": int(r.n),
                                "k": int(r.k),
                                "i": int(r.i),
                                "j": int(r.j),
                                "steps": int(r.steps),
                                "reason": str(r.reason),
                                "preperiod": r.preperiod,
                                "cycle_length": r.cycle_length,
                                "cycle_key": r.cycle_key,
                            })
                            if r.reason == "cycle" and r.cycle_length is not None:
                                cyc_lengths.append(int(r.cycle_length))
                                if r.cycle_key is not None:
                                    cyc_key = str(r.cycle_key)
                                    local_cycle_counts[cyc_key] = int(local_cycle_counts.get(cyc_key, 0)) + 1
                                    per_k_cycle_counts[int(k)][cyc_key] = int(per_k_cycle_counts[int(k)].get(cyc_key, 0)) + 1
                            else:
                                failed += 1
                except Exception:
                    # fallback sequential
                    for n0 in range(1, N + 1):
                        r = cycle_length_for(
                            n0,
                            int(k),
                            int(i),
                            int(j),
                            run_dir,
                            max_iters=max_iters,
                            divergence_threshold=divergence_threshold,
                            alternated=alternated,
                            alt_m=alt_m,
                            max_cycle_sample=0,
                        )
                        rows_kj.append(r)
                        series_rows.append({
                            "n": int(r.n),
                            "k": int(r.k),
                            "i": int(r.i),
                            "j": int(r.j),
                            "steps": int(r.steps),
                            "reason": str(r.reason),
                            "preperiod": r.preperiod,
                            "cycle_length": r.cycle_length,
                            "cycle_key": r.cycle_key,
                        })
                        if r.reason == "cycle" and r.cycle_length is not None:
                            cyc_lengths.append(int(r.cycle_length))
                            if r.cycle_key is not None:
                                cyc_key = str(r.cycle_key)
                                local_cycle_counts[cyc_key] = int(local_cycle_counts.get(cyc_key, 0)) + 1
                                per_k_cycle_counts[int(k)][cyc_key] = int(per_k_cycle_counts[int(k)].get(cyc_key, 0)) + 1
                        else:
                            failed += 1

            per_k_series[int(k)][int(j)] = rows_kj
            per_kj_cycle_counts[int(k)][int(j)] = local_cycle_counts

            if fst_appearance:
                first_map: Dict[str, int] = {}
                for rr in rows_kj:
                    if rr.reason != "cycle" or rr.cycle_key is None:
                        continue
                    ck = str(rr.cycle_key)
                    prev = first_map.get(ck)
                    if prev is None or int(rr.n) < int(prev):
                        first_map[ck] = int(rr.n)
                per_kj_first[int(k)][int(j)] = first_map

            # Evaluate special / extra-special *after* we computed all n for this (k,j).
            # This matches the strict definitions used by the user and keeps logic
            # independent from the early-exit shortcuts.
            if special_cycles or extra_special_cycles:
                all_cycle = all((r.reason == "cycle" and r.cycle_length is not None) for r in rows_kj)
                if all_cycle:
                    if special_cycles:
                        lens = {int(r.cycle_length) for r in rows_kj if r.cycle_length is not None}
                        if len(lens) == 1:
                            cycles = sorted({str(r.cycle_key) for r in rows_kj if r.cycle_key is not None})
                            special_rows.append([int(k), int(i), int(j), int(next(iter(lens))), json.dumps(cycles, ensure_ascii=False)])
                    if extra_special_cycles:
                        keys = [str(r.cycle_key) for r in rows_kj if r.cycle_key is not None]
                        if len(keys) == len(rows_kj) and len(set(keys)) == 1:
                            extra_rows.append([int(k), int(i), int(j), keys[0]])

            mean_cycle_length = (sum(cyc_lengths) / float(len(cyc_lengths))) if cyc_lengths else None
            means_for_k.append(float(mean_cycle_length) if mean_cycle_length is not None else None)

            mean_rows.append(
                {
                    "k": int(k),
                    "j": int(j),
                    "i": int(i),
                    "N": int(N),
                    "count_cycle": int(len(cyc_lengths)),
                    "count_failed": int(failed),
                    "mean_cycle_length": float(mean_cycle_length) if mean_cycle_length is not None else None,
                    
                }
            )

        per_k_means[int(k)] = means_for_k

    base_allj = f"cycle_allj_N{N}_p{int(pmax)}_i{int(i)}_jm{int(j_multiple)}"

    # Special / extra-special results.
    # IMPORTANT: because we now short-circuit non-matching (k,j) combinations
    # early (to speed up detection), we can't rely on per_k_series alone to
    # discover all special/extra-special combos. We therefore accumulate results
    # during the per-(k,j) scan (see loops above) and only use this block to emit
    # the final tables/files.
    
    # (Populated during the main computation loop)
    try:
        special_rows  # type: ignore[name-defined]
    except Exception:
        special_rows = []  # type: ignore[assignment]
    try:
        extra_rows  # type: ignore[name-defined]
    except Exception:
        extra_rows = []  # type: ignore[assignment]

    if special_cycles and special_rows:
        print("special-cycles: k,i,j,cycle_size,cycles")
        for row in special_rows:
            print(",".join(str(x) for x in row))
        try:
            out_special = os.path.join(run_dir, f"{base_allj}_special_cycles.csv")
            with open(out_special, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["k", "i", "j", "cycle_size", "cycles"])
                for row in special_rows:
                    w.writerow(row)
        except Exception:
            pass

    # Extra-special cycles: stricter check where canonical cycle (full list) must be
    # identical for all n in 1..N for a given (k,j).
    #
    # Semantics expected by the CLI flag:
    # - every n0 in 1..N must converge to a cycle (reason == "cycle")
    # - and the canonical cycle identity (cycle_key) must be identical for all n0
    #
    # Only then we report the (k,i,j) combination.
    if extra_special_cycles and extra_rows:
        print("extra-special-cycles: k,i,j,cycle")
        for row in extra_rows:
            print(",".join(str(x) for x in row))
        try:
            out_extra = os.path.join(run_dir, f"{base_allj}_extra_special_cycles.csv")
            with open(out_extra, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["k", "i", "j", "cycle"])
                for row in extra_rows:
                    w.writerow(row)
        except Exception:
            pass

        # Additional summary: number of extra-special (i,j) combos per k.
        _plot_extra_special_counts_by_k(run_dir, base_allj, extra_rows)

    # fst-appearance summary across all (k,j)
    if fst_appearance and per_kj_first:
        try:
            fst_rows: List[Dict[str, object]] = []
            for kk in sorted(per_kj_first.keys()):
                for jj in sorted(per_kj_first[kk].keys()):
                    fmap = per_kj_first[kk][jj]
                    for ck, n0 in fmap.items():
                        fst_rows.append({"k": int(kk), "i": int(i), "j": int(jj), "N": int(N), "cycle": str(ck), "first_n": int(n0)})

            fst_rows.sort(key=lambda r: (int(r["k"]), int(r["j"]), int(r["first_n"])))
            out_json = os.path.join(run_dir, f"{base_allj}_fst_appearance.json")
            out_csv = os.path.join(run_dir, f"{base_allj}_fst_appearance.csv")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(fst_rows, f, ensure_ascii=False, indent=2)
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["k", "i", "j", "N", "cycle", "first_n"])
                w.writeheader()
                for row in fst_rows:
                    w.writerow(row)

            print(f"cycle(all-j): fst-appearance rows={len(fst_rows)}")
        except Exception:
            pass

        # Aggregate distribution of first_n across all (k,j,cycle)
        try:
            from collections import Counter

            counts = Counter(int(r["first_n"]) for r in fst_rows if r.get("first_n") is not None)
            if counts:
                xs = sorted(counts.keys())
                ys = [int(counts[x]) for x in xs]

                total_cnt = int(sum(ys))

                def _threshold(min_coverage: float) -> int:
                    if total_cnt <= 0:
                        return int(xs[-1])
                    target = float(min_coverage) * float(total_cnt)
                    cum = 0
                    thr = int(xs[-1])
                    for x in xs:
                        cum += int(counts[int(x)])
                        if float(cum) >= target:
                            thr = int(x)
                            break
                    return int(thr)

                n0_kernel = _threshold(0.8)
                n0_critical = _threshold(0.95)

                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    fig, ax = plt.subplots(figsize=(9.0, 4.2))
                    # line plot is usually more legible than a massive bar chart
                    ax.plot(xs, ys, marker=".", linewidth=1.0, markersize=4)
                    # annotate kernel/critical thresholds
                    try:
                        ax.axvline(int(n0_kernel), color="C2", linestyle="--", linewidth=1.2, label=f"n0_kernel (90%) = {int(n0_kernel)}")
                        ax.axvline(int(n0_critical), color="C3", linestyle="--", linewidth=1.2, label=f"n0_critical (95%) = {int(n0_critical)}")
                        ax.legend(fontsize=9)
                    except Exception:
                        pass
                    ax.set_title(f"fst-appearance distribution (all k,j) — N={int(N)}, p={int(pmax)}, i={int(i)}, jm={int(j_multiple)}")
                    ax.set_xlabel("first_n (n0)")
                    ax.set_ylabel("count (combined over all (k,j,cycle))")
                    ax.grid(True, alpha=0.25)

                    out_png = os.path.join(run_dir, f"{base_allj}_fst_appearance_distribution.png")
                    fig.tight_layout()
                    fig.savefig(out_png, dpi=220)
                    plt.close(fig)
                    print(f"Wrote: {out_png}")
                except Exception:
                    pass

                # also write a small CSV for the distribution (useful for external plotting)
                try:
                    out_csv = os.path.join(run_dir, f"{base_allj}_fst_appearance_distribution.csv")
                    with open(out_csv, "w", encoding="utf-8") as f:
                        f.write("first_n,count\n")
                        for x in xs:
                            f.write(f"{int(x)},{int(counts[int(x)])}\n")
                except Exception:
                    pass

                # Write the kernel/critical thresholds (90% / 95% cumulative coverage)
                try:
                    covered_90 = int(sum(int(counts[int(x)]) for x in xs if int(x) <= int(n0_kernel)))
                    covered_95 = int(sum(int(counts[int(x)]) for x in xs if int(x) <= int(n0_critical)))
                    crit = {
                        "total_count": int(total_cnt),
                        "n0_kernel_coverage": 0.8,
                        "n0_kernel": int(n0_kernel),
                        "covered_count_90": int(covered_90),
                        "n0_critical_coverage": 0.95,
                        "n0_critical": int(n0_critical),
                        "covered_count_95": int(covered_95),
                        "N": int(N),
                        "p": int(pmax),
                        "i": int(i),
                        "jm": int(j_multiple),
                    }
                    out_json = os.path.join(run_dir, f"{base_allj}_fst_appearance_critical.json")
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(crit, f, ensure_ascii=False, indent=2)
                    # tiny CSV for convenience
                    out_csv2 = os.path.join(run_dir, f"{base_allj}_fst_appearance_critical.csv")
                    with open(out_csv2, "w", encoding="utf-8") as f:
                        f.write("coverage,n0_threshold,total_count,covered_count\n")
                        f.write(f"0.8,{int(crit['n0_kernel'])},{int(crit['total_count'])},{int(crit['covered_count_90'])}\n")
                        f.write(f"0.95,{int(crit['n0_critical'])},{int(crit['total_count'])},{int(crit['covered_count_95'])}\n")
                    print(
                        f"cycle(all-j): fst-appearance thresholds: "
                        f"n0_kernel={int(n0_kernel)} (covered={int(covered_90)}/{int(total_cnt)} >=90%), "
                        f"n0_critical={int(n0_critical)} (covered={int(covered_95)}/{int(total_cnt)} >=95%)"
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Graphical summary: one image per k, with one subplot per j.
        # Each subplot shows the sequence of first_n values (sorted) for the
        # distinct canonical cycles observed at that (k,j).
        try:
            import matplotlib.pyplot as plt  # type: ignore

            ks = sorted(per_kj_first.keys())
            for kk in ks:
                jmap = per_kj_first.get(int(kk), {})
                js = sorted(jmap.keys())
                if not js:
                    continue

                nplots = len(js)
                cols = min(4, nplots)
                rows_grid = (nplots + cols - 1) // cols

                fig, axes = plt.subplots(rows_grid, cols, figsize=(4.8 * cols, 3.0 * rows_grid), squeeze=False)
                for idx, jj in enumerate(js):
                    ax = axes[idx // cols][idx % cols]
                    fmap = jmap.get(int(jj), {})
                    if not fmap:
                        ax.text(0.8, 0.8, "no cycles", ha="center", va="center", color="gray")
                        ax.set_axis_off()
                        continue

                    first_ns = sorted(int(v) for v in fmap.values())
                    ys = list(range(1, len(first_ns) + 1))
                    ax.plot(first_ns, ys, marker=".", linestyle="-", linewidth=0.8, markersize=3)
                    ax.set_title(f"j={int(jj)} (cycles={len(first_ns)})")
                    ax.set_xlabel("first_n (n0)")
                    ax.set_ylabel("cycle index (sorted by first_n)")
                    ax.grid(True, alpha=0.25)

                # hide unused axes
                for idx in range(nplots, rows_grid * cols):
                    axes[idx // cols][idx % cols].axis("off")

                fig.suptitle(f"Cycle fst-appearance — k={int(kk)} (N={int(N)}, i={int(i)}, jm={int(j_multiple)})")
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                out_png = os.path.join(run_dir, f"{base_allj}_fst_appearance_k{int(kk)}.png")
                fig.savefig(out_png, dpi=200)

            # Plot: max cycle length per j (subplot per k)
            if mean_rows:
                try:
                    import matplotlib.pyplot as plt  # type: ignore

                    ks_sorted2 = sorted(per_k_means.keys())
                    ncols = 3
                    nrows = (len(ks_sorted2) + ncols - 1) // ncols
                    figx, axesx = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

                    for idx, k in enumerate(ks_sorted2):
                        ax = axesx[idx // ncols][idx % ncols]
                        # for this k, j range is 0..(k*j_multiple-1)
                        j_end = int(k) * int(j_multiple)
                        xs = list(range(0, j_end))
                        ys = []
                        perj_map = per_kj_cycle_counts.get(int(k), {})
                        for jj in xs:
                            # compute max cycle length for this (k,jj)
                            try:
                                cmap = perj_map.get(jj, {})
                                # perj_map stores counts per cycle_key; we need cycle lengths
                                # but we don't have lengths here — fallback: derive from per_k_series
                                rows_kj = per_k_series.get(int(k), {}).get(int(jj), [])
                                vals = [int(r.cycle_length) for r in rows_kj if r.reason == "cycle" and r.cycle_length is not None]
                                ys.append(float(max(vals)) if vals else float('nan'))
                            except Exception:
                                ys.append(float('nan'))

                        ax.plot(xs, ys, linestyle='-', linewidth=0.8)
                        ax.set_title(f"k={k}")
                        ax.set_xlabel("j")
                        ax.set_ylabel("max_cycle_length")
                        ax.grid(True, alpha=0.25)

                    for idx in range(len(ks_sorted2), nrows * ncols):
                        axesx[idx // ncols][idx % ncols].axis("off")

                    figx.suptitle(f"Max cycle length per j — N={N}, primes k<=p={int(pmax)}, i={int(i)}, jm={int(j_multiple)}")
                    figx.tight_layout()
                    out_png = os.path.join(run_dir, f"{base_allj}_max_by_j.png")
                    try:
                        figx.savefig(out_png, dpi=200)
                    except Exception:
                        pass
                    plt.close(figx)
                except Exception:
                    pass
                plt.close(fig)
        except Exception:
            pass

    # Console: compact mean table (k,i,j,mean_cycle_length)
    try:
        if mean_rows:
            print("cycle(all-j): mean table: k,i,j,mean_cycle_length")
            for row in mean_rows:
                # row keys: k,j,i,N,count_cycle,count_failed,mean_cycle_length
                k_ = row.get("k")
                j_ = row.get("j")
                i_ = row.get("i")
                mcl = row.get("mean_cycle_length")
                print(f"{k_},{i_},{j_},{mcl}")
    except Exception:
        pass

    out_series_json = os.path.join(run_dir, f"{base_allj}.json")
    out_series_csv = os.path.join(run_dir, f"{base_allj}.csv")
    out_mean_json = os.path.join(run_dir, f"{base_allj}_mean_by_kj.json")
    out_mean_csv = os.path.join(run_dir, f"{base_allj}_mean_by_kj.csv")

    # Build max table: one row per (k,j) with the maximum cycle length across n in 1..N
    max_rows: List[Dict[str, object]] = []
    for k, perj in per_k_series.items():
        for j, rows_kj in perj.items():
            max_len = None
            for r in rows_kj:
                if r.reason == "cycle" and r.cycle_length is not None:
                    v = int(r.cycle_length)
                    if max_len is None or v > max_len:
                        max_len = v
            max_rows.append({"k": int(k), "i": int(i), "j": int(j), "N": int(N), "max_cycle_length": int(max_len) if max_len is not None else None})

    out_max_json = os.path.join(run_dir, f"{base_allj}_max_by_kij.json")
    out_max_csv = os.path.join(run_dir, f"{base_allj}_max_by_kij.csv")

    out_card_kj_json = os.path.join(run_dir, f"{base_allj}_cardinality_by_kj.json")
    out_card_kj_csv = os.path.join(run_dir, f"{base_allj}_cardinality_by_kj.csv")
    out_card_k_json = os.path.join(run_dir, f"{base_allj}_cardinality_by_k.json")
    out_card_k_csv = os.path.join(run_dir, f"{base_allj}_cardinality_by_k.csv")

    with open(out_series_json, "w", encoding="utf-8") as f:
        json.dump(series_rows, f, ensure_ascii=False, indent=2)
    with open(out_mean_json, "w", encoding="utf-8") as f:
        json.dump(mean_rows, f, ensure_ascii=False, indent=2)

    # write consolidated max table
    try:
        with open(out_max_json, "w", encoding="utf-8") as f:
            json.dump(max_rows, f, ensure_ascii=False, indent=2)
        with open(out_max_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "i", "j", "N", "max_cycle_length"])
            w.writeheader()
            for row in max_rows:
                w.writerow(row)
        # write distinct-cycle counts per (k,j)
        try:
            out_dist_kj_csv = os.path.join(run_dir, f"{base_allj}_distinct_cycles_by_kj.csv")
            with open(out_dist_kj_csv, "w", newline="", encoding="utf-8") as f2:
                w2 = csv.writer(f2)
                w2.writerow(["k", "i", "j", "nbre_cycles_distincts"])
                for k, perj in per_k_series.items():
                    for j, rows_kj in perj.items():
                        distinct_keys = {str(r.cycle_key) for r in rows_kj if r.reason == "cycle" and r.cycle_key is not None}
                        w2.writerow([int(k), int(i), int(j), int(len(distinct_keys))])
        except Exception:
            pass
        # Print compact table to stdout for run visibility
        try:
            print("max-cycle (all-j): k,i,j,N,max_cycle_length")
            for row in max_rows:
                print(f"{int(row['k'])},{int(row['i'])},{int(row['j'])},{int(row['N'])},{row['max_cycle_length']}")
        except Exception:
            pass
    except Exception:
        pass

    card_rows_kj: List[Dict[str, object]] = []
    card_rows_k: List[Dict[str, object]] = []
    if cycle_cardinality:
        # Cardinality exports
        for k, perj in per_kj_cycle_counts.items():
            for j, cmap in perj.items():
                for cyc_key, cnt in cmap.items():
                    card_rows_kj.append({"k": int(k), "j": int(j), "i": int(i), "N": int(N), "cycle": cyc_key, "count": int(cnt)})

        for k, cmap in per_k_cycle_counts.items():
            for cyc_key, cnt in cmap.items():
                card_rows_k.append({"k": int(k), "i": int(i), "N": int(N), "cycle": cyc_key, "count": int(cnt)})

        with open(out_card_kj_json, "w", encoding="utf-8") as f:
            json.dump(card_rows_kj, f, ensure_ascii=False, indent=2)
        with open(out_card_k_json, "w", encoding="utf-8") as f:
            json.dump(card_rows_k, f, ensure_ascii=False, indent=2)

        # Additionally produce an aggregated JSON mapping k -> j -> list of {cycle, count}
        try:
            agg: Dict[int, Dict[int, List[Dict[str, object]]]] = {}
            for k, perj in per_kj_cycle_counts.items():
                agg[int(k)] = {}
                for j, cmap in perj.items():
                    agg[int(k)][int(j)] = []
                    for cyc_key, cnt in cmap.items():
                        agg[int(k)][int(j)].append({"cycle": cyc_key, "count": int(cnt)})
            out_agg = os.path.join(run_dir, f"{base_allj}_cardinality_by_kj_agg.json")
            with open(out_agg, "w", encoding="utf-8") as f:
                json.dump(agg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    if series_rows:
        with open(out_series_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["n", "k", "i", "j", "steps", "reason", "preperiod", "cycle_length", "cycle_key"],
            )
            w.writeheader()
            for row in series_rows:
                w.writerow(row)

    if mean_rows:
        with open(out_mean_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["k", "j", "i", "N", "count_cycle", "count_failed", "mean_cycle_length"],
            )
            w.writeheader()
            for row in mean_rows:
                w.writerow(row)

    if cycle_cardinality and card_rows_kj:
        with open(out_card_kj_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "j", "i", "N", "cycle", "count"])
            w.writeheader()
            for row in card_rows_kj:
                w.writerow(row)

    if cycle_cardinality and card_rows_k:
        with open(out_card_k_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["k", "i", "N", "cycle", "count"])
            w.writeheader()
            for row in card_rows_k:
                w.writerow(row)

    # Console summary for all-j mode
    if cycle_cardinality:
        try:
            # overall distinct cycles aggregated over k
            total_cycles = {}
            for k, cmap in per_k_cycle_counts.items():
                for cyc, cnt in cmap.items():
                    total_cycles[cyc] = int(total_cycles.get(cyc, 0)) + int(cnt)

            print(f"cycle(all-j): cardinality: distinct_cycles_total={len(total_cycles)}")
            TOP = 10
            def _short2(cycle_key: str) -> str:
                try:
                    vals = json.loads(cycle_key)
                    if isinstance(vals, list):
                        return str(vals[:6]) + ("…" if len(vals) > 6 else "")
                except Exception:
                    pass
                return cycle_key[:120] + ("…" if len(cycle_key) > 120 else "")

            for cyc, cnt in sorted(total_cycles.items(), key=lambda kv: kv[1], reverse=True)[:TOP]:
                print(f"  {int(cnt):6d}  {_short2(cyc)}")
            # Print a more detailed per-k, per-j cardinality summary (top cycles per j)
            try:
                print("\ncycle(all-j): cardinality by k/j (top cycles per j)")
                TOP_J = 5
                for kk in sorted(per_kj_cycle_counts.keys()):
                    perj_map = per_kj_cycle_counts.get(int(kk), {})
                    if not perj_map:
                        continue
                    print(f" k={kk}:")
                    for jval in sorted(perj_map.keys()):
                        cmap = perj_map.get(jval, {})
                        if not cmap:
                            continue
                        top = sorted(cmap.items(), key=lambda kv: kv[1], reverse=True)[:TOP_J]
                        top_str = ", ".join([f"{int(cnt)}:{_short2(key)}" for key, cnt in top])
                        print(f"   j={jval}: {top_str}")
            except Exception:
                pass
        except Exception:
            pass

    # Plot 1: cycle_length vs n, overlay curves for each j. One subplot per k.
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.ticker import MaxNLocator  # type: ignore

        ks_sorted = sorted(per_k_series.keys())
        if ks_sorted:
            ncols = 3
            nrows = (len(ks_sorted) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

            for idx, k in enumerate(ks_sorted):
                ax = axes[idx // ncols][idx % ncols]
                series_k = per_k_series[k]

                # Overlay one line per j. Keep it light to avoid huge legends.
                for j_val, rows_kj in series_k.items():
                    xs = [r.n for r in rows_kj]
                    ys = [float(r.cycle_length) if (r.reason == "cycle" and r.cycle_length is not None) else float("nan") for r in rows_kj]
                    ax.plot(xs, ys, linestyle="-", linewidth=0.6, alpha=0.8)

                ax.set_title(f"k={k}")
                ax.set_xlabel("n")
                ax.set_ylabel("cycle_length")
                try:
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                except Exception:
                    pass
                ax.grid(True, alpha=0.25)

            for idx in range(len(ks_sorted), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

            fig.suptitle(f"Cycle length per n (overlay j) — N={N}, primes k<=p={int(pmax)}, i={int(i)}", y=0.98)
            fig.tight_layout()
            fig.savefig(os.path.join(run_dir, f"{base_allj}_cyclelen_by_n.png"), dpi=200)
            plt.close(fig)

        # Plot 2: mean cycle length over n as a function of j. One subplot per k.
        ks_sorted2 = sorted(per_k_means.keys())
        if ks_sorted2:
            ncols = 3
            nrows = (len(ks_sorted2) + ncols - 1) // ncols
            figm, axesm = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

            for idx, k in enumerate(ks_sorted2):
                ax = axesm[idx // ncols][idx % ncols]
                ys0 = per_k_means[k]
                xs = list(range(len(ys0)))
                ys = [float(v) if v is not None else float("nan") for v in ys0]
                ax.plot(xs, ys, linestyle="-", linewidth=0.8)
                ax.set_title(f"k={k}")
                ax.set_xlabel("j")
                ax.set_ylabel("mean_cycle_length")
                ax.grid(True, alpha=0.25)

            for idx in range(len(ks_sorted2), nrows * ncols):
                axesm[idx // ncols][idx % ncols].axis("off")

            figm.suptitle(f"Mean cycle length over n — N={N}, primes k<=p={int(pmax)}, i={int(i)}", y=0.98)
            figm.tight_layout()
            figm.savefig(os.path.join(run_dir, f"{base_allj}_mean_by_j.png"), dpi=200)
            plt.close(figm)

        # Plot 3 (optional): cycle cardinalities per j.
        # For each k (subplot), x=j and we overlay the top cycles (by total count across j)
        # as line plots. Remaining cycles are aggregated as an 'other' line.
        if cycle_cardinality and per_kj_cycle_counts:
            ks_sorted3 = sorted(per_kj_cycle_counts.keys())
            if ks_sorted3:
                TOP_M = 5

                def _short_cycle_label(cycle_key: str, max_items: int = 4) -> str:
                    # cycle_key is a JSON list string like "[1,2,3]".
                    try:
                        vals = json.loads(cycle_key)
                        if isinstance(vals, list):
                            vals_i = [int(x) for x in vals[:max_items]]
                            tail = "…" if len(vals) > max_items else ""
                            return f"{vals_i}{tail}"
                    except Exception:
                        pass
                    return cycle_key[:24] + ("…" if len(cycle_key) > 24 else "")

                ncols = 3
                nrows = (len(ks_sorted3) + ncols - 1) // ncols
                figc, axesc = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)

                for idx, k in enumerate(ks_sorted3):
                    ax = axesc[idx // ncols][idx % ncols]
                    perj = per_kj_cycle_counts.get(int(k), {})
                    if not perj:
                        ax.set_title(f"k={k}")
                        ax.text(0.8, 0.8, "no cycles", ha="center", va="center", transform=ax.transAxes)
                        ax.axis("off")
                        continue
                    # Determine top cycles for this k (aggregate across all j).
                    total_by_cycle: Dict[str, int] = {}
                    for _j, cmap in perj.items():
                        for cyc_key, cnt in cmap.items():
                            total_by_cycle[cyc_key] = int(total_by_cycle.get(cyc_key, 0)) + int(cnt)

                    # Choose top cycles to plot based on card_top_cycles (<=0 => all)
                    if card_top_cycles is None or int(card_top_cycles) <= 0:
                        top_cycles = [ck for ck in sorted(total_by_cycle.keys())]
                    else:
                        top_cycles = [ck for ck, _ in sorted(total_by_cycle.items(), key=lambda kv: kv[1], reverse=True)[: int(card_top_cycles)]]

                    # x-range is 0..(j_end-1) where j_end depends on j_multiple
                    # compute j_end consistent with earlier loop
                    j_end = int(k) * int(j_multiple)
                    xs = list(range(0, j_end))

                    # Precompute y per cycle for speed.
                    used_sum = [0 for _ in xs]
                    any_labeled = False
                    for cyc_key in top_cycles:
                        ys = [int(perj.get(jj, {}).get(cyc_key, 0)) for jj in xs]
                        for ii, v in enumerate(ys):
                            used_sum[ii] += int(v)
                        if any(int(v) != 0 for v in ys):
                            ax.plot(xs, ys, linewidth=1.0, alpha=0.85, label=_short_cycle_label(cyc_key))
                            any_labeled = True

                    # Other cycles aggregated.
                    other_total = [0 for _ in xs]
                    for jj in xs:
                        cmap = perj.get(jj, {})
                        all_cnt = sum(int(v) for v in cmap.values())
                        other_total[jj] = int(all_cnt) - int(used_sum[jj])
                    if any(v != 0 for v in other_total):
                        ax.plot(xs, other_total, linewidth=1.0, alpha=0.7, linestyle="--", label="other")
                        any_labeled = True

                    ax.set_title(f"k={k}")
                    ax.set_xlabel("j")
                    ax.set_ylabel("cardinality (#n)")
                    ax.grid(True, alpha=0.25)

                    # Legends can get large; keep them inside each subplot.
                    if any_labeled:
                        ax.legend(fontsize=7, loc="upper right", frameon=False)

                for idx in range(len(ks_sorted3), nrows * ncols):
                    axesc[idx // ncols][idx % ncols].axis("off")

                figc.suptitle(
                    f"Cycle cardinalities per j — N={N}, primes k<=p={int(pmax)}, i={int(i)} (top {TOP_M} cycles per k)",
                    y=0.98,
                )
                figc.tight_layout()
                figc.savefig(os.path.join(run_dir, f"{base_allj}_cardinality_by_j.png"), dpi=200)
                plt.close(figc)
    except Exception:
        pass

    print(
        f"cycle(all-j): computed primes k<=p={int(pmax)} for n=1..{N}, j=0..k-1 "
        f"(series_rows={len(series_rows)}, mean_rows={len(mean_rows)})"
    )

    # Additionally build a combined PNG with one subplot per k showing the
    # cumulative maximum cycle length up to n, aggregated over all j (take
    # the maximum cycle length across j for each n).
    try:
        import matplotlib.pyplot as plt  # type: ignore

        ks_sorted = sorted(per_k_series.keys())
        if ks_sorted:
            ncols = 3
            nrows = (len(ks_sorted) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
            for idx, k in enumerate(ks_sorted):
                ax = axes[idx // ncols][idx % ncols]
                # per_k_series[k] is a dict j -> List[CycleResult]
                perj = per_k_series.get(k, {})
                # Build, for each n 1..N, the maximum cycle_length across j
                max_by_n = [None] * (N + 1)
                for jval, rows_kj in perj.items():
                    for r in rows_kj:
                        if r.reason == "cycle" and r.cycle_length is not None:
                            n0 = int(r.n)
                            v = int(r.cycle_length)
                            if max_by_n[n0] is None or v > max_by_n[n0]:
                                max_by_n[n0] = v
                # cumulative
                cum = []
                cur = None
                for n0 in range(1, N + 1):
                    v = max_by_n[n0]
                    if v is not None:
                        cur = v if (cur is None or v > cur) else cur
                    cum.append(float(cur) if cur is not None else float('nan'))
                ax.plot(list(range(1, N + 1)), cum, linewidth=0.8)
                ax.set_title(f"k={k}")
                ax.set_xlabel("n")
                ax.set_ylabel("max_cycle_up_to_n")
                ax.grid(True, alpha=0.25)

            for idx in range(len(ks_sorted), nrows * ncols):
                axes[idx // ncols][idx % ncols].axis("off")

            fig.suptitle(f"Max cycle length up to n (all-j) — N={N}, primes k<=p={int(pmax)}, i={int(i)}, jm={int(j_multiple)}")
            fig.tight_layout()
            out_png = os.path.join(run_dir, f"{base_allj}_max_cycle_up_to_n.png")
            try:
                fig.savefig(out_png, dpi=180)
            except Exception:
                pass
            plt.close(fig)
    except Exception:
        pass
