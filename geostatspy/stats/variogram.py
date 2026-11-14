"""Experimental variograms, variogram maps and variogram model evaluation.

Vectorized NumPy re-implementations of GSLIB ``gamv`` (2D), ``varmapv`` and
``vmodel``. Differences from ``geostats.py``:

* lag ``k`` (k = 1..nlag) is centred on ``k * lag`` (legacy index ``k + 1``
  is centred on ``(k - 1) * lag``, i.e. shifted by one);
* ``npairs`` counts unordered pairs (legacy counts each pair twice);
* the variogram map bins by rounding to the nearest lag (legacy truncates,
  making the central bin twice as wide), returns NaN (not -999) below
  ``minnp`` pairs, and has no wrong ``tv``/``hv`` accumulation or no-op
  sill standardization.
"""
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ._utils import as_1d

_TYPES = {"spherical": 1, "exponential": 2, "gaussian": 3,
          1: 1, 2: 2, 3: 3}
_BLOCK = 512  # rows per block when forming pair matrices


# --------------------------------------------------------------------------
# Variogram model
# --------------------------------------------------------------------------
@dataclass
class VariogramModel:
    """Nested 2D variogram: nugget + anisotropic structures.

    Each structure is a dict with keys

    * ``it``: "spherical" | "exponential" | "gaussian" (or GSLIB code 1/2/3)
    * ``cc``: contribution (>= 0)
    * ``hmaj``: range in the major direction; ``hmin``: range in the minor
      direction (default ``hmaj``, i.e. isotropic)
    * ``azi``: azimuth of the major direction, degrees clockwise from +y
      (default 0)

    Ranges are practical ranges (exponential/gaussian reach 95 % of the
    contribution), as in GSLIB.
    """
    nug: float = 0.0
    structures: Sequence[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.nug < 0:
            raise ValueError("nug must be >= 0")
        parsed = []
        for s in self.structures:
            it = _TYPES.get(s.get("it"))
            if it is None:
                raise ValueError(f"unknown structure type {s.get('it')!r}")
            hmaj = float(s["hmaj"])
            hmin = float(s.get("hmin", hmaj))
            cc = float(s["cc"])
            if cc < 0 or hmaj <= 0 or hmin <= 0:
                raise ValueError("require cc >= 0, hmaj > 0, hmin > 0")
            parsed.append((it, cc, float(s.get("azi", 0.0)), hmaj, hmin))
        self._parsed = parsed

    @property
    def sill(self):
        return self.nug + sum(p[1] for p in self._parsed)

    def _struct_cov(self, dx, dy):
        """Sum of structure covariances (nugget excluded) for lag vectors."""
        dx, dy = np.broadcast_arrays(np.asarray(dx, float),
                                     np.asarray(dy, float))
        out = np.zeros(dx.shape)
        for it, cc, azi, hmaj, hmin in self._parsed:
            a = np.deg2rad(90.0 - azi)
            u = dx * np.cos(a) + dy * np.sin(a)          # along major
            w = (-dx * np.sin(a) + dy * np.cos(a)) * (hmaj / hmin)
            hr = np.hypot(u, w) / hmaj                   # range-normalised
            if it == 1:
                out += cc * np.where(hr < 1.0, 1.0 - hr * (1.5 - 0.5 * hr * hr), 0.0)
            elif it == 2:
                out += cc * np.exp(-3.0 * hr)
            else:
                out += cc * np.exp(-3.0 * hr * hr)
        return out

    def cov_xy(self, dx, dy):
        """Covariance for lag vector(s) (dx, dy); C(0) = sill."""
        c = self._struct_cov(dx, dy)
        zero = (np.asarray(dx) == 0) & (np.asarray(dy) == 0)
        return np.where(zero, self.sill, c)

    def gamma_xy(self, dx, dy):
        """Semivariogram for lag vector(s) (dx, dy); gamma(0) = 0."""
        return self.sill - self.cov_xy(dx, dy)

    def cov(self, h, azimuth=0.0):
        """Covariance at distance ``h`` along ``azimuth`` (deg from +y)."""
        h = np.asarray(h, float)
        az = np.deg2rad(azimuth)
        return self.cov_xy(h * np.sin(az), h * np.cos(az))

    def gamma(self, h, azimuth=0.0):
        """Semivariogram at distance ``h`` along ``azimuth`` (deg from +y)."""
        return self.sill - self.cov(h, azimuth)


# --------------------------------------------------------------------------
# Experimental variogram
# --------------------------------------------------------------------------
def _xy(coords, values):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coords must have shape (n, 2)")
    if coords.shape[1] == 3:
        raise NotImplementedError("3D variograms are not implemented yet")
    v = as_1d(values, "values")
    if v.shape[0] != coords.shape[0]:
        raise ValueError("coords and values must have equal length")
    if coords.shape[0] < 2:
        raise ValueError("need at least 2 data")
    if not (np.all(np.isfinite(coords)) and np.all(np.isfinite(v))):
        raise ValueError("inputs must be finite")
    return coords[:, 0], coords[:, 1], v


def experimental_variogram(coords, values, lag, nlag, lag_tol=None,
                           azimuth=0.0, azimuth_tol=22.5, bandwidth=np.inf,
                           standardize=False):
    """Directional experimental semivariogram (2D, GSLIB gamv).

    Lag ``k`` (1..nlag) collects the pairs with distance within ``lag_tol``
    of ``k * lag``, direction within ``azimuth_tol`` of ``azimuth`` (either
    way) and perpendicular offset <= ``bandwidth``. Windows may overlap when
    ``lag_tol > lag / 2``.

    :param coords: (n, 2) array of x, y
    :param lag: lag distance; ``nlag``: number of lags
    :param lag_tol: lag tolerance (default ``0.5 * lag``)
    :param azimuth: degrees clockwise from +y; ``azimuth_tol >= 90`` is
        omnidirectional, ``<= 0`` is taken as 45 degrees (GSLIB)
    :param bandwidth: max offset perpendicular to the azimuth
    :param standardize: divide by the data variance (sill = 1)
    :return: (mean_pair_distance, gamma, npairs) arrays of length ``nlag``;
        lags without pairs have NaN distance/gamma and 0 pairs
    """
    x, y, v = _xy(coords, values)
    if lag <= 0 or nlag < 1:
        raise ValueError("require lag > 0 and nlag >= 1")
    tol = 0.5 * lag if lag_tol is None else float(lag_tol)
    if tol < 0:
        raise ValueError("lag_tol must be >= 0")
    csatol = np.cos(np.deg2rad(45.0 if azimuth_tol <= 0 else azimuth_tol))
    az = np.deg2rad(90.0 - azimuth)
    ux, uy = np.cos(az), np.sin(az)
    centers = lag * np.arange(1, nlag + 1)

    n = x.size
    sum_h = np.zeros(nlag)
    sum_g = np.zeros(nlag)
    npairs = np.zeros(nlag, dtype=np.int64)
    eps = 1e-20
    for i0 in range(0, n - 1, _BLOCK):
        i1 = min(i0 + _BLOCK, n - 1)
        ii = np.arange(i0, i1)[:, None]
        jj = np.arange(n)[None, :]
        keep = jj > ii                                  # unordered pairs
        dx = x[None, :] - x[i0:i1, None]
        dy = y[None, :] - y[i0:i1, None]
        h = np.hypot(dx, dy)
        keep &= h <= centers[-1] + tol
        safe = np.where(h < eps, 1.0, h)
        dcaz = np.where(h < eps, 1.0, (dx * ux + dy * uy) / safe)
        band = np.abs(ux * dy - uy * dx)
        keep &= (np.abs(dcaz) >= csatol - 1e-12) & (band <= bandwidth)
        if not keep.any():
            continue
        hk = h[keep]
        gk = (v[None, :] - v[i0:i1, None])[keep] ** 2
        for k, c in enumerate(centers):
            sel = np.abs(hk - c) <= tol
            m = int(sel.sum())
            if m:
                npairs[k] += m
                sum_h[k] += hk[sel].sum()
                sum_g[k] += gk[sel].sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        dist = np.where(npairs > 0, sum_h / npairs, np.nan)
        gamma = np.where(npairs > 0, 0.5 * sum_g / npairs, np.nan)
    if standardize:
        gamma = gamma / v.var()
    return dist, gamma, npairs


def variogram_map(coords, values, nxlag, nylag, dxlag, dylag, minnp=5,
                  standardize=False):
    """Variogram map from irregular data (2D, GSLIB varmapv).

    Cell ``[iy, ix]`` holds the semivariogram for lag vectors rounded to
    ``((ix - nxlag) * dxlag, (iy - nylag) * dylag)``; rows increase with +y.
    All ordered pairs are used, so the map is symmetric; self pairs fill the
    central cell (gamma 0).

    :return: (gamma, npairs) arrays of shape (2*nylag+1, 2*nxlag+1); gamma is
        NaN where ``npairs < minnp``
    """
    x, y, v = _xy(coords, values)
    if nxlag < 0 or nylag < 0 or dxlag <= 0 or dylag <= 0:
        raise ValueError("require nxlag, nylag >= 0 and dxlag, dylag > 0")
    ny, nx = 2 * nylag + 1, 2 * nxlag + 1
    npp = np.zeros(ny * nx)
    gsum = np.zeros(ny * nx)
    n = x.size
    for i0 in range(0, n, _BLOCK):
        i1 = min(i0 + _BLOCK, n)
        ix = np.rint((x[None, :] - x[i0:i1, None]) / dxlag).astype(np.int64) + nxlag
        iy = np.rint((y[None, :] - y[i0:i1, None]) / dylag).astype(np.int64) + nylag
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        cell = (iy * nx + ix)[ok]
        d2 = (v[None, :] - v[i0:i1, None])[ok] ** 2
        npp += np.bincount(cell, minlength=ny * nx)
        gsum += np.bincount(cell, weights=d2, minlength=ny * nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        gam = np.where(npp >= max(minnp, 1), 0.5 * gsum / npp, np.nan)
    if standardize:
        gam = gam / v.var()
    return gam.reshape(ny, nx), npp.reshape(ny, nx).astype(np.int64)
