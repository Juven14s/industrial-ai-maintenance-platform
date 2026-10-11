"""Target-independent post-processing shared by training and serving."""

from __future__ import annotations

import numpy as np


def apply_safety_margin(base_predictions: np.ndarray, margin: float) -> np.ndarray:
    """Return non-negative conservative RUL predictions."""
    values = np.asarray(base_predictions, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("base_predictions must be a finite, non-empty vector")
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite non-negative number")
    return np.maximum(values - margin, 0.0)
