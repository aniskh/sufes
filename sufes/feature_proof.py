"""Feature: proof

This module hosts the implementation for the `--proof` / `--proof-persist` feature.

Historically this lived in `sufes.core`; it was extracted to keep `core.py`
small and focused on CLI parsing + dispatch.

Public contract:
- `run_proof(...) -> List[Dict[str, object]]`
- `_prove_combo_persist(...)` multiprocessing helper (kept public-ish because
  the pool needs it picklable).

No CLI parsing and no IO here besides the persist helper writing its own
progress file (`proof_k..._maxproved.txt`).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Set, Tuple

from .algorithms import next_term_ji


def _lake_compact_repr(lake: Set[int]) -> Dict[str, object]:
    """Compact representation of a lake as [1..Ns] + extras.

    We compute the largest Ns such that {1,2,...,Ns} is included in lake.
    The remainder is returned as a sorted list of integers > Ns.
    """

    if not lake:
        return {"Ns": 0, "extras": []}

    ns = 0
    # amortized OK because ns only grows during the run.
    while (ns + 1) in lake:
        ns += 1
    extras = sorted(x for x in lake if x > ns)
    return {"Ns": ns, "extras": extras}


def _prove_combo_persist_lake(
    *,
    k: int,
    i_val: int,
    j_val: int,
    proof_max_n: int,
    alternated: bool,
    alt_m: int,
    max_iters: int,
    divergence_threshold: float,
    run_dir: str,
) -> Dict[str, object]:
    """Sequential persist worker with an incremental 'lake' of visited states.

    Lake contract:
      - lake contains all points t visited by trajectories of proved n (<= max_proved)
      - when processing a new n, if its trajectory hits any t in lake, we mark it proved
        and we add all visited points from that trajectory into the lake.

    Persistence:
      - max_proved is stored in the existing progress file.
      - lake is stored in a separate JSON file.
    """

    start_time = time.perf_counter()

    alt_tag = f"_altm{alt_m}" if alternated else ""
    fname = os.path.join(run_dir, f"proof_k{k}_i{i_val}_j{j_val}{alt_tag}_maxproved.txt")
    lake_fname = os.path.join(run_dir, f"proof_k{k}_i{i_val}_j{j_val}{alt_tag}_lake.json")

    max_proved = 0
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as rf:
                txt = rf.read().strip()
                if txt:
                    max_proved = int(txt)
        except (OSError, ValueError):
            max_proved = 0

    # Load lake
    lake: Set[int] = set()
    if os.path.exists(lake_fname):
        try:
            import json

            with open(lake_fname, "r", encoding="utf-8") as rf:
                obj = json.load(rf) or {}
            ns = int(obj.get("Ns") or 0)
            extras = obj.get("extras") or []
            lake.update(range(1, ns + 1))
            for x in extras:
                try:
                    xi = int(x)
                    if xi > 0:
                        lake.add(xi)
                except Exception:
                    continue
        except Exception:
            lake = set()

    # If starting fresh, seed the lake with trajectory of n=1..max_proved if it's empty.
    # This is a fallback for older runs before lake existed.
    if max_proved > 0 and not lake:
        for n in range(1, max_proved + 1):
            t = n
            for _step in range(max_iters):
                if abs(t) > divergence_threshold:
                    break
                lake.add(int(t))
                t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)

    # Use a compact boolean flag array instead of a set to track proved n values.
    # This reduces Python object overhead: one byte per index vs many Python ints.
    proven_flag = bytearray(proof_max_n + 1)
    for v in range(1, max_proved + 1):
        if v <= proof_max_n:
            proven_flag[v] = 1

    def _persist_state():
        try:
            import json

            os.makedirs(run_dir, exist_ok=True)
            with open(fname, "w", encoding="utf-8") as wf:
                wf.write(str(max_proved))
            with open(lake_fname, "w", encoding="utf-8") as wf:
                json.dump(_lake_compact_repr(lake), wf, ensure_ascii=False)
        except OSError:
            pass

    start_n = max_proved + 1
    if start_n > proof_max_n:
        elapsed = time.perf_counter() - start_time
        return {
            "k": k,
            "i": i_val,
            "j": j_val,
            "alternated": bool(alternated),
            "alt_m": int(alt_m),
            "max_proved": max_proved,
            "status": "OK" if max_proved == proof_max_n else "FAILED",
            "reason": None,
            "elapsed_sec": elapsed,
            "lake": _lake_compact_repr(lake),
        }

    status = "OK"
    reason = None
    failed_n = None

    # Reuse the seen_local set between iterations to avoid repeated allocations.
    seen_local: Set[int] = set()
    # Avoid storing extremely large trajectory values which explode memory.
    store_limit = max(10 * proof_max_n, 1_000_000)
    for n in range(start_n, proof_max_n + 1):
        seen_local.clear()
        t = n
        for step in range(max_iters):
            if abs(t) > divergence_threshold:
                status = "FAILED"
                reason = "divergence_threshold"
                failed_n = n
                _persist_state()
                break

            ti = int(t)
            # Lake coalescence criterion
            if ti in lake:
                # mark as proved
                if n <= proof_max_n:
                    proven_flag[n] = 1
                # mark as proved in compact flag array
                max_proved = n
                # only persist bounded values to the lake to limit memory
                if store_limit is None:
                    lake.update(seen_local)
                else:
                    lake.update(x for x in seen_local if abs(x) <= store_limit)
                lake.add(ti)
                break

            # classic criteria (kept as extra safety)
            if ti in seen_local:
                if n <= proof_max_n:
                    proven_flag[n] = 1
                max_proved = n
                lake.update(seen_local)
                break
            if ti < n or (0 <= ti <= proof_max_n and proven_flag[int(ti)]):
                if n <= proof_max_n:
                    proven_flag[n] = 1
                max_proved = n
                lake.update(seen_local)
                break

            seen_local.add(ti)
            t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
        else:
            status = "FAILED"
            reason = "max_iters"
            failed_n = n
            _persist_state()
            break

        if status == "FAILED":
            break

    # Persist occasionally (every proved n) to allow resume.
        _persist_state()

    elapsed = time.perf_counter() - start_time
    row: Dict[str, object] = {
        "k": k,
        "i": i_val,
        "j": j_val,
        "alternated": bool(alternated),
        "alt_m": int(alt_m),
        "max_proved": max_proved,
        "status": "OK" if status == "OK" and max_proved == proof_max_n else "FAILED" if status == "FAILED" else "OK",
        "reason": reason,
        "elapsed_sec": elapsed,
        "failed_n": failed_n,
        "lake": _lake_compact_repr(lake),
    }
    try:
        msg_reason = f" reason={reason}" if reason else ""
        print(
            f"proof combo done (lake): k={int(k)} i={int(i_val)} j={int(j_val)} "
            f"max_proved={max_proved} status={row.get('status')}{msg_reason} elapsed_sec={elapsed:.3f}"
        )
    except Exception:
        pass
    return row


def _prove_combo_persist(args: Tuple) -> Tuple[Tuple[int, int, int], Dict[str, object]]:
    """Persisted proof worker.

    Args tuple layout (kept stable for multiprocessing pickling):
      (k, i_val, j_val, proof_max_n, alternated, alt_m, max_iters, divergence_threshold, run_dir)

    Returns:
      ((k, i_val, j_val), info)
    """

    start_time = time.perf_counter()
    (k, i_val, j_val, proof_max_n, alternated, alt_m, max_iters, divergence_threshold, run_dir) = args

    # Persist file should include alternated/m to avoid mixing progress between modes.
    alt_tag = f"_altm{alt_m}" if alternated else ""
    fname = os.path.join(run_dir, f"proof_k{k}_i{i_val}_j{j_val}{alt_tag}_maxproved.txt")
    max_proved = 0
    if os.path.exists(fname):
        try:
            with open(fname, "r", encoding="utf-8") as rf:
                txt = rf.read().strip()
                if txt:
                    max_proved = int(txt)
        except (OSError, ValueError):
            max_proved = 0

    # compact proved flags to reduce memory usage in threaded workers
    proven_flag = bytearray(proof_max_n + 1)
    for v in range(1, max_proved + 1):
        if v <= proof_max_n:
            proven_flag[v] = 1

    def prove_for_combo_local(start_n: int):
        nonlocal max_proved
        # reuse local seen set to avoid allocating many small sets
        seen = set()
        store_limit = max(10 * proof_max_n, 1_000_000)
        for n in range(start_n, proof_max_n + 1):
            seen.clear()
            t = n
            for step in range(max_iters):
                if abs(t) > divergence_threshold:
                    try:
                        os.makedirs(run_dir, exist_ok=True)
                        with open(fname, "w", encoding="utf-8") as wf:
                            wf.write(str(max_proved))
                    except OSError:
                        pass
                    return max_proved, {"failed_n": n, "reason": "divergence_threshold", "steps": step}

                if t in seen:
                    if n <= proof_max_n:
                        proven_flag[n] = 1
                    max_proved = n
                    break

                if t < n or (0 <= t <= proof_max_n and proven_flag[int(t)]):
                    if n <= proof_max_n:
                        proven_flag[n] = 1
                    max_proved = n
                    break

                if abs(t) <= store_limit:
                    if abs(t) <= store_limit:
                        seen.add(t)
                t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
            else:
                try:
                    os.makedirs(run_dir, exist_ok=True)
                    with open(fname, "w", encoding="utf-8") as wf:
                        wf.write(str(max_proved))
                except OSError:
                    pass
                return max_proved, {"failed_n": n, "reason": "max_iters", "steps": max_iters}

        try:
            os.makedirs(run_dir, exist_ok=True)
            with open(fname, "w", encoding="utf-8") as wf:
                wf.write(str(max_proved))
        except OSError:
            pass
        return max_proved, None

    start_n = max_proved + 1
    if start_n > proof_max_n:
        elapsed = time.perf_counter() - start_time
        return ((k, i_val, j_val), {"max_proved": max_proved, "elapsed_sec": elapsed})

    maxp, info = prove_for_combo_local(start_n)
    elapsed = time.perf_counter() - start_time
    res = {"max_proved": maxp, "elapsed_sec": elapsed}
    if info is not None:
        res.update(info)
    # small progress line per combination (k,i,j)
    try:
        status = "OK" if int(res.get("max_proved") or 0) == int(proof_max_n) and "failed_n" not in res else "FAILED"
        reason = res.get("reason")
        msg_reason = f" reason={reason}" if reason else ""
        print(
            f"proof combo done: k={int(k)} i={int(i_val)} j={int(j_val)} "
            f"max_proved={res.get('max_proved')} status={status}{msg_reason} elapsed_sec={elapsed:.3f}"
        )
    except Exception:
        pass
    return ((k, i_val, j_val), res)


def run_proof(
    primes: List[int],
    proof_max_n: int,
    proof_j_mult: int,
    all_i: bool,
    alternated: bool,
    alt_m: int,
    max_iters: int,
    divergence_threshold: float,
    run_dir: str,
    persist: bool,
    workers: int,
    proof_i: Optional[int] = None,
    proof_lake: bool = False,
) -> List[Dict[str, object]]:
    """Core proof routine (pure-ish, returns rows for CSV/JSON)."""

    # NOTE: `all_i` is kept for CLI/backward-compatibility, but the default
    # behaviour when --proof-i is not provided is now to scan i in {1,2,3,4,5}
    # (filtered to i<k), regardless of `all_i`.
    _ = all_i

    rows: List[Dict[str, object]] = []
    combos: List[Tuple[int, int, int]] = []

    for k in primes:
        if proof_i is not None:
            # Accept any integer --proof-i; reduce modulo k for the effective rule.
            i_range = [int(proof_i)]
        else:
            # Default: scan i in {1,2,3} when --proof-i is not provided.
            i_range = [1, 2, 3]

        j_range = range(0, proof_j_mult * k)
        for i_req in i_range:
            # Use i as requested (no modulo/remap) for exploratory runs.
            for j_val in j_range:
                combos.append((k, int(i_req), int(j_val)))

    if persist:
        if proof_lake:
            # Lake mode needs sequential processing to maintain a consistent lake.
            workers = 1

        # Ensure run_dir exists, since workers write progress files.
        os.makedirs(run_dir, exist_ok=True)

        tasks = [
            (k, i_val, j_val, proof_max_n, alternated, alt_m, max_iters, divergence_threshold, run_dir)
            for (k, i_val, j_val) in combos
        ]

        if proof_lake:
            results = []
        else:
            if int(workers) > 1:
                # Use threads and log each combo as it completes (avoid waiting for all tasks)
                from concurrent.futures import ThreadPoolExecutor, as_completed

                results = []
                with ThreadPoolExecutor(max_workers=int(workers)) as executor:
                    future_to_task = {executor.submit(_prove_combo_persist, t): t for t in tasks}
                    for fut in as_completed(future_to_task):
                        try:
                            combo, info = fut.result()
                        except Exception as e:
                            # If a worker crashed, log and continue
                            try:
                                print(f"proof combo crashed: {e}")
                            except Exception:
                                pass
                            continue

                        # print a concise progress line per completed combo (k,i,j)
                        try:
                            k, i_val, j_val = combo
                            status = "OK" if info.get("max_proved") == proof_max_n and "failed_n" not in info else "FAILED" if "failed_n" in info or info.get("max_proved") != proof_max_n else "OK"
                            reason = info.get("reason")
                            elapsed = info.get("elapsed_sec")
                            msg_reason = f" reason={reason}" if reason else ""
                            try:
                                elapsed_str = f"{elapsed:.3f}" if isinstance(elapsed, (int, float)) else "NA"
                            except Exception:
                                elapsed_str = "NA"
                            print(
                                f"proof-persist: finished k={int(k)} i={int(i_val)} j={int(j_val)} max_proved={info.get('max_proved')} status={status}{msg_reason} elapsed_sec={elapsed_str}"
                            )
                        except Exception:
                            # best-effort logging only
                            try:
                                print(f"proof-persist: finished combo {combo} info={info}")
                            except Exception:
                                pass

                        results.append((combo, info))
            else:
                # sequential execution: run each task and log immediately
                results = []
                for t in tasks:
                    try:
                        combo, info = _prove_combo_persist(t)
                    except Exception as e:
                        try:
                            print(f"proof combo crashed: {e}")
                        except Exception:
                            pass
                        continue

                    # concise per-combo logging (same format as threaded branch)
                    try:
                        k, i_val, j_val = combo
                        status = "OK" if info.get("max_proved") == proof_max_n and "failed_n" not in info else "FAILED" if "failed_n" in info or info.get("max_proved") != proof_max_n else "OK"
                        reason = info.get("reason")
                        elapsed = info.get("elapsed_sec")
                        msg_reason = f" reason={reason}" if reason else ""
                        try:
                            elapsed_str = f"{elapsed:.3f}" if isinstance(elapsed, (int, float)) else "NA"
                        except Exception:
                            elapsed_str = "NA"
                        print(
                            f"proof-persist: finished k={int(k)} i={int(i_val)} j={int(j_val)} max_proved={info.get('max_proved')} status={status}{msg_reason} elapsed_sec={elapsed_str}"
                        )
                    except Exception:
                        try:
                            print(f"proof-persist: finished combo {combo} info={info}")
                        except Exception:
                            pass

                    results.append((combo, info))

        if proof_lake:
            for k, i_val, j_val in combos:
                if alt_m >= k:
                    rows.append({"k": k, "i": i_val, "j": j_val, "max_proved": None, "status": "SKIPPED", "reason": "alt_m>=k"})
                    continue
                rows.append(
                    _prove_combo_persist_lake(
                        k=k,
                        i_val=i_val,
                        j_val=j_val,
                        proof_max_n=proof_max_n,
                        alternated=alternated,
                        alt_m=alt_m,
                        max_iters=max_iters,
                        divergence_threshold=divergence_threshold,
                        run_dir=run_dir,
                    )
                )
        else:
            for (k, i_val, j_val), info in results:
                status = "OK" if info.get("max_proved") == proof_max_n else "FAILED"
                if "failed_n" in info:
                    status = "FAILED"
                if alt_m >= k:
                    status = "SKIPPED"

                row = {
                    "k": k,
                    "i": i_val,
                    "j": j_val,
                    "alternated": bool(alternated),
                    "alt_m": int(alt_m),
                    "max_proved": info.get("max_proved"),
                    "status": status,
                    "reason": info.get("reason"),
                }
                rows.append(row)

                # Per-combo logging is handled by the worker (_prove_combo_persist)
                # which also includes elapsed time.

        return rows

    # Non-persist mode: single-process proof.
    for k, i_val, j_val in combos:
        if alt_m >= k:
            rows.append({"k": k, "i": i_val, "j": j_val, "max_proved": None, "status": "SKIPPED", "reason": "alt_m>=k"})
            continue

        # compact proved flags for the non-persist path
        proven_flag = bytearray(proof_max_n + 1)
        max_proved = 0
        status = "OK"
        reason = None
        # reuse a single seen set to avoid many small allocations
        seen = set()
        for n in range(1, proof_max_n + 1):
            seen.clear()
            store_limit = max(10 * proof_max_n, 1_000_000)
            t = n
            for _step in range(max_iters):
                if abs(t) > divergence_threshold:
                    status = "FAILED"
                    reason = "divergence_threshold"
                    break

                if t in seen:
                    if n <= proof_max_n:
                        proven_flag[n] = 1
                    max_proved = n
                    break

                if t < n or (0 <= t <= proof_max_n and proven_flag[int(t)]):
                    if n <= proof_max_n:
                        proven_flag[n] = 1
                    max_proved = n
                    break

                if abs(t) <= store_limit:
                    seen.add(t)
                t = next_term_ji(t, k, j_val, i_val, alternated=alternated, alt_m=alt_m)
            else:
                status = "FAILED"
                reason = "max_iters"

            if status == "FAILED":
                break

        rows.append(
            {
                "k": k,
                "i": i_val,
                "j": j_val,
                "alternated": bool(alternated),
                "alt_m": int(alt_m),
                "max_proved": max_proved,
                "status": status,
                "reason": reason,
            }
        )

        # small progress line per combination (k,i,j)
        try:
            msg_reason = f" reason={reason}" if reason else ""
            print(
                f"proof combo done: k={int(k)} i={int(i_val)} j={int(j_val)} "
                f"max_proved={max_proved} status={status}{msg_reason}"
            )
        except Exception:
            pass

    return rows
