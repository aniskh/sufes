"""Feature: datalake

Goal
----
Persist per-(k,i) experiment results into a deterministic on-disk "datalake"
layout so runs can be stopped and resumed without recomputing.

The user requested an output layout:

  {datalake_path}/k{k}/i{i}/<chunk_dir>/data.json

Where each chunk contains all results for n in [n_start, n_end] and for all
j in 0..(j_mult*k-1). The chosen JSON structure is option A:

  results_by_j[j] = list of per-n records.

To keep early iterations lightweight, this module supports:
- trajectory truncation (--datalake-trajectory-limit)
- trajectory hashing (--datalake-trajectory-hash)

It also writes a checkpoint file:

  {datalake_path}/k{k}/i{i}/checkpoint.json

which records the next chunk start to resume from.

Notes
-----
This is a "simple setup" implementation. It is designed to be extended
(iterative chunk sizing, more metadata, compression, cloud upload).

"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .algorithms import next_term_ji


@dataclass(frozen=True)
class DataLakeConfig:
    datalake_path: str
    k: int
    i: int
    n_max: int
    j_mult: int
    max_iters: int
    divergence_threshold: float
    alternated: bool
    alt_m: int
    trajectory_limit: int
    trajectory_hash: bool
    workers: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _checkpoint_path(root_ki: str) -> str:
    return os.path.join(root_ki, "checkpoint.json")


def load_checkpoint(root_ki: str) -> int:
    """Return next_n_start. Defaults to 1 if no checkpoint exists."""
    cp = _checkpoint_path(root_ki)
    if not os.path.exists(cp):
        return 1
    try:
        with open(cp, "r", encoding="utf-8") as f:
            data = json.load(f)
        nxt = int(data.get("next_n_start", 1))
        return max(1, nxt)
    except Exception:
        return 1


def write_checkpoint(root_ki: str, cfg: DataLakeConfig, next_n_start: int) -> None:
    """Write checkpoint atomically."""
    cp = _checkpoint_path(root_ki)
    tmp = cp + ".tmp"
    payload = {
        "k": int(cfg.k),
        "i": int(cfg.i),
        "n_max": int(cfg.n_max),
        "j_mult": int(cfg.j_mult),
        "next_n_start": int(next_n_start),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cp)


def chunk_ranges_loglike(n_start: int, n_max: int, base_chunk: int = 10_000) -> List[Tuple[int, int]]:
    """Return chunk ranges [a,b] covering [n_start,n_max] with a simple log-like growth.

    We start with base_chunk for small n, and increase chunk size slowly with n.

    Current simple heuristic (can be tuned later):
      size = base_chunk * max(1, floor(log10(n_current)))

    This grows at most by a factor of ~7 for n up to 1e7.

    The user requested a logarithmic-ish variation; this is the simplest safe
    version.
    """

    out: List[Tuple[int, int]] = []
    cur = int(n_start)
    n_max = int(n_max)
    while cur <= n_max:
        # log10 bucket: 1..9 ->1, 10..99 ->1, 100..999 ->2, ...
        # We implement a cheap digit-count based log10.
        digits = len(str(cur))
        factor = max(1, digits - 1)
        size = int(base_chunk * factor)
        end = min(n_max, cur + size - 1)
        out.append((cur, end))
        cur = end + 1
    return out


def _trajectory_record(
    n0: int,
    k: int,
    i_val: int,
    j_val: int,
    *,
    max_iters: int,
    divergence_threshold: float,
    alternated: bool,
    alt_m: int,
    trajectory_limit: int,
    trajectory_hash: bool,
) -> Tuple[Dict[str, object], Optional[Tuple[int, ...]]]:
    """Simulate trajectory for one (n0,k,i,j) and return a JSON record."""

    seen: Dict[int, int] = {}
    seq: List[int] = []
    t = int(n0)
    reason: str
    cycle: Optional[List[int]] = None

    # Storage policy:
    # - Store a truncated trajectory (first `trajectory_limit` steps) for:
    #   - divergent runs (reason == "divergence_threshold")
    #   - convergent runs (reason == "cycle")
    # - For max_iters we omit the trajectory by default to keep files smaller.
    limit = int(trajectory_limit)

    for step in range(int(max_iters)):
        if abs(t) > float(divergence_threshold):
            reason = "divergence_threshold"
            break
        if t in seen:
            reason = "cycle"
            start = seen[t]
            # We don't store the full cycle here; we'll create a canonical cycle key
            # and let the caller map it to a cycle_id.
            cycle = seq[start:]
            break
        seen[t] = len(seq)
        # For now we always build the sequence in-memory up to the limit.
        # Whether we keep it in the output depends on the final "reason".
        if limit > 0 and len(seq) < limit:
            seq.append(int(t))
        t = int(next_term_ji(t, int(k), int(j_val), int(i_val), alternated=bool(alternated), alt_m=int(alt_m)))
    else:
        reason = "max_iters"

    out: Dict[str, object] = {
        "n": int(n0),
        "k": int(k),
        "i": int(i_val),
        "j": int(j_val),
        "reason": str(reason),
        "steps": int(len(seen)),
    }

    # Keep the stored trajectory for diverging and converging runs.
    if limit > 0 and str(reason) in {"divergence_threshold", "cycle"}:
        out["trajectory"] = seq

    if trajectory_hash:
        # hash the stored trajectory sequence (truncated) OR empty if not stored
        h = hashlib.sha256()
        if limit > 0:
            h.update(",".join(str(x) for x in seq).encode("utf-8"))
        else:
            # if no trajectory stored, hash a compact signature: first 256 visited (deterministic order)
            xs = sorted(list(seen.keys()))[:256]
            h.update(",".join(str(x) for x in xs).encode("utf-8"))
        out["trajectory_hash_sha256"] = h.hexdigest()

    # Produce a canonical cycle key for convergent runs.
    cycle_key: Optional[Tuple[int, ...]] = None
    if str(reason) == "cycle" and cycle:
        # Canonicalize by rotation to make the key stable.
        c = [int(x) for x in cycle]
        rots = [tuple(c[i:] + c[:i]) for i in range(len(c))]
        cycle_key = min(rots) if rots else None

    return out, cycle_key


def _cycle_uid(cycle_key: Tuple[int, ...]) -> str:
    """Return a stable unique id for a canonical cycle.

    We hash the canonical cycle representation so the id is stable across
    chunks/runs (for the same cycle).
    """

    h = hashlib.sha256()
    h.update(",".join(str(int(x)) for x in cycle_key).encode("utf-8"))
    return h.hexdigest()


def _cycles_dir(root_ki: str) -> str:
    return os.path.join(root_ki, "cycles")


def _cycles_path(root_ki: str, j: int) -> str:
    return os.path.join(_cycles_dir(root_ki), f"j{int(j):04d}_cycles.json")


def _load_cycles_by_uid(path: str) -> Dict[str, List[int]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out: Dict[str, List[int]] = {}
        for item in payload.get("cycles", []) or []:
            uid = item.get("cycle_uid")
            cyc = item.get("cycle")
            if isinstance(uid, str) and isinstance(cyc, list):
                out[uid] = [int(x) for x in cyc]
        return out
    except Exception:
        return {}


def _write_cycles_atomically(path: str, meta: Dict[str, object], cycles_by_uid: Dict[str, List[int]]) -> None:
    _ensure_dir(os.path.dirname(path))
    payload = {
        "meta": dict(meta),
        "cycles": [
            {"cycle_uid": uid, "cycle": cyc}
            for uid, cyc in sorted(cycles_by_uid.items(), key=lambda kv: kv[0])
        ],
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def run_datalake(cfg: DataLakeConfig, *, base_chunk: int = 10_000) -> None:
    """Run datalake export for a single (k,i) and j range 0..(j_mult*k-1)."""

    root_k = os.path.join(cfg.datalake_path, f"k{int(cfg.k)}")
    root_ki = os.path.join(root_k, f"i{int(cfg.i)}")
    _ensure_dir(root_ki)
    _ensure_dir(_cycles_dir(root_ki))

    next_n = load_checkpoint(root_ki)
    if next_n > int(cfg.n_max):
        return

    j_max_excl = int(cfg.j_mult) * int(cfg.k)
    ranges = chunk_ranges_loglike(next_n, int(cfg.n_max), base_chunk=int(base_chunk))

    for a, b in ranges:
        chunk_dir = os.path.join(root_ki, f"chunk_{a:08d}_{b:08d}")
        _ensure_dir(chunk_dir)
        out_json = os.path.join(chunk_dir, "data.json")

        # skip if already present: treat as done
        if os.path.exists(out_json):
            write_checkpoint(root_ki, cfg, b + 1)
            continue

        # New output format: one JSON per j in this chunk.
        # Each file includes a cycles table and per-n rows referencing cycle_id.
        workers = int(getattr(cfg, "workers", 1))

        meta_common = {
            "k": int(cfg.k),
            "i": int(cfg.i),
            "n_start": int(a),
            "n_end": int(b),
            "max_iters": int(cfg.max_iters),
            "divergence_threshold": float(cfg.divergence_threshold),
            "alternated": bool(cfg.alternated),
            "alt_m": int(cfg.alt_m),
            "trajectory_limit": int(cfg.trajectory_limit),
            "trajectory_hash": bool(cfg.trajectory_hash),
        }

        for j in range(0, j_max_excl):
            out_json_j = os.path.join(chunk_dir, f"j{int(j):04d}.json")
            if os.path.exists(out_json_j):
                continue

            ns = list(range(int(a), int(b) + 1))

            def _one(n0: int) -> Tuple[Dict[str, object], Optional[Tuple[int, ...]]]:
                return _trajectory_record(
                    n0,
                    cfg.k,
                    cfg.i,
                    j,
                    max_iters=cfg.max_iters,
                    divergence_threshold=cfg.divergence_threshold,
                    alternated=cfg.alternated,
                    alt_m=cfg.alt_m,
                    trajectory_limit=cfg.trajectory_limit,
                    trajectory_hash=cfg.trajectory_hash,
                )

            if workers <= 1:
                recs = list(map(_one, ns))
            else:
                try:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        recs = list(executor.map(_one, ns))
                except Exception:
                    recs = list(map(_one, ns))

            # Build cycles table for this j: stable uid per canonical cycle.
            cycle_uid_by_key: Dict[Tuple[int, ...], str] = {}
            cycles_by_uid: Dict[str, List[int]] = {}

            rows: List[Dict[str, object]] = []
            for r, ckey in recs:
                if ckey is not None:
                    uid = cycle_uid_by_key.get(ckey)
                    if uid is None:
                        uid = _cycle_uid(ckey)
                        cycle_uid_by_key[ckey] = uid
                        cycles_by_uid.setdefault(uid, list(ckey))
                    r["cycle_uid"] = str(uid)
                rows.append(r)

            payload_j = {
                "meta": {**meta_common, "j": int(j)},
                "rows": rows,
            }

            # Merge cycles into stable per-(k,i,j) file under k{k}/i{i}/cycles/.
            cycles_path = _cycles_path(root_ki, int(j))
            existing = _load_cycles_by_uid(cycles_path)
            if cycles_by_uid:
                existing.update(cycles_by_uid)
            _write_cycles_atomically(cycles_path, {**meta_common, "j": int(j)}, existing)

            tmpj = out_json_j + ".tmp"
            with open(tmpj, "w", encoding="utf-8") as f:
                json.dump(payload_j, f, ensure_ascii=False)
            os.replace(tmpj, out_json_j)

        # Keep a small marker file for the chunk completion.
        tmp = out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"meta": meta_common, "j_max_exclusive": int(j_max_excl), "status": "done"}, f, ensure_ascii=False)
        os.replace(tmp, out_json)

        # commit checkpoint after writing the chunk
        write_checkpoint(root_ki, cfg, b + 1)


__all__ = [
    "DataLakeConfig",
    "run_datalake",
    "chunk_ranges_loglike",
    "load_checkpoint",
    "write_checkpoint",
]
