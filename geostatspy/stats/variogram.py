"""Experimental variograms (2D/3D), variogram maps, model evaluation."""
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class VariogramModel:
    """Nested variogram: nugget + structures (spherical/exponential/gaussian)."""
    nug: float = 0.0
    structures: Sequence[dict] = field(default_factory=list)  # it, cc, azi, hmaj, hmin[, vert...]

    def gamma(self, h):
        raise NotImplementedError

    def cov(self, h):
        raise NotImplementedError


def experimental_variogram(coords, values, lags, lag_tol, azimuth=0.0,
                           azimuth_tol=22.5, bandwidth=None, dip=0.0,
                           kind="semivariogram"):
    """Directional experimental variogram (GSLIB gamv). Returns lag, gamma, npairs."""
    raise NotImplementedError


def variogram_map(coords, values, nx, ny, xsize, ysize, nlag_min=1):
    """Variogram map from irregular data (GSLIB varmapv)."""
    raise NotImplementedError
