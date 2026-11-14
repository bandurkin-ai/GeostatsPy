"""Normal score transform and back-transform (GSLIB nscore / backtr).

Vectorized NumPy/SciPy reimplementation. Differences from ``geostats.py``:

* ties in the fitting data get distinct scores by stable data order
  (GSLIB breaks ties with a random epsilon);
* the upper hyperbolic tail is evaluated per value (the legacy ``backtr``
  accidentally uses the whole input array there);
* tails are named (``"linear"``, ``"power"``, ``"hyperbolic"``, ``None``)
  instead of GSLIB integer codes.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

from ._utils import as_1d, check_weights

_LOWER_TAILS = (None, "linear", "power")
_UPPER_TAILS = (None, "linear", "power", "hyperbolic")


@dataclass
class NormalScoreTransform:
    """Fitted transform table: sorted data values <-> normal scores."""
    values: np.ndarray   # sorted original values
    scores: np.ndarray   # matching normal scores (strictly increasing)
    zmin: float          # lower trimming limit (tail extrapolation)
    zmax: float          # upper trimming limit (tail extrapolation)

    def forward(self, x):
        """Original values -> normal scores (linear interpolation in table).

        Values outside the table range are clamped to the end scores.
        """
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.values, self.scores)

    def inverse(self, y, ltail="linear", utail="linear",
                ltpar=None, utpar=None):
        """Normal scores -> original values, with tail extrapolation.

        :param ltail: lower tail: None (clamp), "linear" or "power"
        :param utail: upper tail: None (clamp), "linear", "power" or "hyperbolic"
        :param ltpar: lower power-tail exponent (> 0)
        :param utpar: upper power exponent (> 0) / hyperbolic exponent (>= 1)
        """
        return _backtr(np.asarray(y, dtype=float), self, ltail, utail,
                       ltpar, utpar)


def nscore(x, wts=None, zmin=None, zmax=None, reference=None,
           reference_wts=None):
    """Weighted normal score transform.

    :param x: 1D data to transform
    :param wts: optional declustering weights for ``x`` (or for ``reference``)
    :param zmin, zmax: trimming limits stored for tail back-transform
        (default: min/max of the table values)
    :param reference: optional 1D reference distribution used to build the
        table instead of ``x`` (GSLIB ``ismooth``); ``wts`` are then ignored
        in favour of ``reference_wts``
    :return: (scores, NormalScoreTransform)
    """
    x = as_1d(x, "x")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    use_ref = reference is not None
    src = as_1d(reference, "reference") if use_ref else x
    w = check_weights(reference_wts if use_ref else wts, src.size)
    if src.size == 0:
        raise ValueError("empty data")

    order = np.argsort(src, kind="stable")
    vals, w = src[order], w[order]
    cp = np.cumsum(w) / w.sum()
    scores_sorted = norm.ppf(cp - 0.5 * w / w.sum())  # midpoint of each step

    tr = NormalScoreTransform(
        values=vals, scores=scores_sorted,
        zmin=float(vals[0] if zmin is None else zmin),
        zmax=float(vals[-1] if zmax is None else zmax),
    )
    if use_ref:
        return tr.forward(x), tr
    # fitting data: scores by rank, so ties stay distinct
    ns = np.empty_like(x)
    ns[order] = scores_sorted
    return ns, tr


def backtr(y, transform, ltail="linear", utail="linear",
           ltpar=None, utpar=None):
    """Back-transform normal scores using a fitted ``NormalScoreTransform``."""
    return transform.inverse(y, ltail, utail, ltpar, utpar)


def _backtr(y, tr, ltail, utail, ltpar, utpar):
    if ltail not in _LOWER_TAILS:
        raise ValueError(f"ltail must be one of {_LOWER_TAILS}")
    if utail not in _UPPER_TAILS:
        raise ValueError(f"utail must be one of {_UPPER_TAILS}")
    if ltail == "power" and (ltpar is None or ltpar <= 0):
        raise ValueError("ltpar > 0 required for power lower tail")
    if utail == "power" and (utpar is None or utpar <= 0):
        raise ValueError("utpar > 0 required for power upper tail")
    if utail == "hyperbolic" and (utpar is None or utpar < 1):
        raise ValueError("utpar >= 1 required for hyperbolic upper tail")

    vr, vrg = tr.values, tr.scores
    out = np.interp(y, vrg, vr)  # interior; ends clamp to vr[0], vr[-1]

    # lower tail: cdf in [0, cdf(vrg[0])] -> [zmin, vr[0]]
    lo = y <= vrg[0]
    if ltail is not None and lo.any():
        cdflo = norm.cdf(vrg[0])
        p = 1.0 if ltail == "linear" else 1.0 / ltpar
        out[lo] = _powint(0.0, cdflo, tr.zmin, vr[0], norm.cdf(y[lo]), p)

    # upper tail: cdf in [cdf(vrg[-1]), 1] -> [vr[-1], zmax]
    hi = y >= vrg[-1]
    if utail is not None and hi.any():
        cdfhi = norm.cdf(vrg[-1])
        cdfy = norm.cdf(y[hi])
        if utail == "hyperbolic":
            lam = vr[-1] ** utpar * (1.0 - cdfhi)
            with np.errstate(divide="ignore"):
                out[hi] = (lam / (1.0 - cdfy)) ** (1.0 / utpar)
        else:
            p = 1.0 if utail == "linear" else 1.0 / utpar
            out[hi] = _powint(cdfhi, 1.0, vr[-1], tr.zmax, cdfy, p)
    return out


def _powint(xlow, xhigh, ylow, yhigh, x, power):
    """Power interpolation between (xlow, ylow) and (xhigh, yhigh)."""
    if xhigh - xlow < 1e-20:
        return np.full_like(x, 0.5 * (ylow + yhigh), dtype=float)
    return ylow + (yhigh - ylow) * ((x - xlow) / (xhigh - xlow)) ** power
