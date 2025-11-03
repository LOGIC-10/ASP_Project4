from __future__ import annotations

import numpy as np


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation between two 1-D arrays."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("pearsonr_np expects 1-D inputs")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)
