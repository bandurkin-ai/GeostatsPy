"""Declustering weights: cell-based and polygonal."""


def cell_declus(x, y, v, cmin, cmax, ncell=100, noff=8, minmax="min"):
    """Cell declustering (GSLIB declus). Returns (weights, cell_sizes, means)."""
    raise NotImplementedError


def polygonal_declus(x, y, v, bounds=None):
    """Voronoi-polygon declustering weights, clipped to ``bounds``."""
    raise NotImplementedError
