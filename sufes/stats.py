"""Shared statistics helpers.

These helpers were historically defined in the legacy monolithic residue implementation
but are used by multiple feature modules.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple


def shannon_entropy_non_zero_residues(counts: Dict[int, int], k: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Shannon entropy (base 2) of residue distribution restricted to r in 1..k-1.

    Args:
        counts: mapping residue -> count.
        k: modulus base.

    Returns:
        (H, Hmax, ratio) with Hmax=log2(k-1) when k>2, else 0.
    """
    total = sum(int(counts.get(r, 0)) for r in range(1, int(k)))
    if total <= 0:
        h = 0.0
    else:
        h = 0.0
        for r in range(1, int(k)):
            c = int(counts.get(r, 0))
            if c <= 0:
                continue
            p = float(c) / float(total)
            h -= p * math.log(p, 2)

    hmax = math.log(int(k) - 1, 2) if int(k) > 2 else 0.0
    ratio = (float(h) / float(hmax)) if hmax > 0 else None
    return float(h), float(hmax), ratio


def skewness_non_zero_residues(counts: Dict[int, int], k: int) -> Optional[float]:
    """Population skewness of residue distribution restricted to r in 1..k-1."""
    total = sum(int(counts.get(r, 0)) for r in range(1, int(k)))
    if total <= 0:
        return None

    mu = 0.0
    for r in range(1, int(k)):
        mu += float(r) * float(int(counts.get(r, 0)))
    mu /= float(total)

    m2 = 0.0
    m3 = 0.0
    for r in range(1, int(k)):
        c = float(int(counts.get(r, 0)))
        if c <= 0:
            continue
        d = float(r) - mu
        m2 += c * d * d
        m3 += c * d * d * d

    m2 /= float(total)
    m3 /= float(total)

    if m2 <= 0:
        return 0.0
    sigma = math.sqrt(m2)
    if sigma == 0:
        return 0.0
    return float(m3) / float(sigma ** 3)
