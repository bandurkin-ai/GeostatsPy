"""Declustering weights: cell-based and polygonal (nearest-sample area).

Vectorized NumPy/SciPy versions of GSLIB ``declus`` and the nearest-sample
(Voronoi) polygonal declustering from ``geostats.py``. All weights are
scaled to sum to ``n`` (mean weight 1), so they can be passed directly as
``wts`` to :func:`geostatspy.stats.nscore`.
"""
import numpy as np
from scipy.spatial import cKDTree

from ._utils import as_1d


def _coords(x, y, v=None):
    x, y = as_1d(x, "x"), as_1d(y, "y")
    arrays = [x, y] + ([] if v is None else [as_1d(v, "v")])
    if len({a.shape[0] for a in arrays}) != 1:
        raise ValueError("x, y, v must have equal length")
    if x.size == 0:
        raise ValueError("empty data")
    if not all(np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("inputs must be finite")
    return (x, y) if v is None else (x, y, arrays[2])


def cell_declus(x, y, v, cmin, cmax, ncell=100, noff=8, minmax="min"):
    """Cell declustering (GSLIB declus, 2D, isotropic cells).

    For each of ``ncell + 1`` cell sizes in [cmin, cmax] and ``noff`` grid
    origin offsets, every datum gets weight inversely proportional to the
    number of data in its cell. The cell size whose declustered mean is the
    minimum (``minmax="min"``, for preferentially clustered high values) or
    maximum (``"max"``) is selected. If no size improves on the naive mean,
    uniform weights are returned (as in GSLIB).

    :param cmin, cmax: smallest / largest cell size (> 0, cmax >= cmin)
    :param ncell: number of cell-size increments (``ncell + 1`` sizes tried)
    :param noff: number of origin offsets averaged per cell size
    :param minmax: "min" or "max" declustered mean criterion
    :return: (weights, cell_sizes, means); ``cell_sizes[0] = 0`` holds the
        naive mean, followed by the ``ncell + 1`` tried sizes
    """
    x, y, v = _coords(x, y, v)
    if not (cmin > 0 and cmax >= cmin):
        raise ValueError("require 0 < cmin <= cmax")
    if ncell < 1 or noff < 1:
        raise ValueError("ncell and noff must be >= 1")
    if minmax not in ("min", "max"):
        raise ValueError("minmax must be 'min' or 'max'")

    n = x.size
    xo1, yo1 = x.min() - 0.01, y.min() - 0.01   # "lower" origin as in GSLIB
    xspan, yspan = x.max() - x.min(), y.max() - y.min()
    sizes = cmin + np.arange(ncell + 1) * (cmax - cmin) / ncell

    all_wts = np.empty((ncell + 1, n))
    for k, cs in enumerate(sizes):
        xfac = min(cs / noff, 0.5 * xspan)
        yfac = min(cs / noff, 0.5 * yspan)
        wt = np.zeros(n)
        for o in range(noff):
            ix = np.floor((x - (xo1 - o * xfac)) / cs).astype(np.int64)
            iy = np.floor((y - (yo1 - o * yfac)) / cs).astype(np.int64)
            key = ix * (iy.max() + 1) + iy
            _, inv, counts = np.unique(key, return_inverse=True,
                                       return_counts=True)
            w = 1.0 / counts[inv]
            wt += w / w.sum()                    # each offset sums to 1
        all_wts[k] = wt

    means = all_wts @ v / all_wts.sum(axis=1)
    naive = v.mean()
    cand = means if minmax == "min" else -means
    best = int(np.argmin(cand))
    ref = naive if minmax == "min" else -naive
    improves = cand[best] < ref - 1e-12 * max(1.0, abs(ref))  # beat naive mean
    wopt = all_wts[best] if (improves or ncell == 1) else np.ones(n)
    wopt = wopt * n / wopt.sum()
    return wopt, np.r_[0.0, sizes], np.r_[naive, means]


def polygonal_declus(x, y, bounds=None, nx=200, ny=200):
    """Polygonal (nearest-sample / Voronoi) declustering weights.

    The area of each sample's Voronoi polygon, clipped to ``bounds``, is
    estimated by assigning the centers of an ``nx`` x ``ny`` grid to their
    nearest sample. Samples whose polygon contains no grid center get
    weight 0, so increase ``nx``/``ny`` for very small polygons.

    :param bounds: (xmin, xmax, ymin, ymax); default: data extent
    :return: weights, length n, summing to n
    """
    x, y = _coords(x, y)
    if bounds is None:
        bounds = (x.min(), x.max(), y.min(), y.max())
    x0, x1, y0, y1 = map(float, bounds)
    if not (x1 > x0 and y1 > y0):
        raise ValueError("bounds must have positive extent "
                         "(xmin < xmax, ymin < ymax)")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")

    gx = x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx
    gy = y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny
    nodes = np.column_stack([np.tile(gx, ny), np.repeat(gy, nx)])
    _, nearest = cKDTree(np.column_stack([x, y])).query(nodes)
    counts = np.bincount(nearest, minlength=x.size).astype(float)
    return counts * x.size / counts.sum()
