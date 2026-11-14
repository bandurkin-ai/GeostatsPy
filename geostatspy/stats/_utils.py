"""Shared helpers: input validation, weights."""
import numpy as np


def as_1d(a, name="array"):
    """Return a finite-checked 1D float array; raise ValueError otherwise."""
    a = np.asarray(a, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {a.shape}")
    return a


def check_weights(wts, n):
    """Return normalized weights (sum = n); uniform if ``wts`` is None."""
    if wts is None:
        return np.ones(n)
    wts = as_1d(wts, "wts")
    if wts.shape[0] != n:
        raise ValueError("wts and data must have equal length")
    if np.any(wts < 0):
        raise ValueError("wts must be non-negative")
    return wts * n / wts.sum()
