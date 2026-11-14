import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from geostatspy import stats
from geostatspy import geostats as gs

rng = np.random.default_rng(1)
X = rng.lognormal(mean=1.0, sigma=0.6, size=300)
W = rng.uniform(0.5, 2.0, size=300)


def test_roundtrip_unweighted_and_weighted():
    for w in (None, W):
        y, tr = stats.nscore(X, wts=w)
        assert np.allclose(stats.backtr(y, tr), X)


def test_scores_approximately_standard_normal():
    y, _ = stats.nscore(X)
    assert abs(y.mean()) < 0.01 and abs(y.std() - 1) < 0.05


@pytest.mark.parametrize("wts", [None, W])
def test_matches_legacy_nscore(wts):
    df = pd.DataFrame({"v": X, "w": W})
    ns_old, vr, vrg = gs.nscore(df, "v", wcol=None if wts is None else "w")
    ns_new, tr = stats.nscore(X, wts=wts)
    # legacy mis-indexes the table edge (wrong score for the sample minimum),
    # so compare every other point; ours must equal the first table score
    keep = X != X.min()
    assert np.allclose(ns_new[keep], ns_old[keep], atol=1e-4)  # gauinv approx
    assert ns_new[~keep][0] == pytest.approx(tr.scores[0])
    assert np.allclose(tr.values, vr) and np.allclose(tr.scores, vrg, atol=1e-4)


def test_matches_legacy_backtr_tails():
    _, tr = stats.nscore(X, zmin=X.min() * 0.5, zmax=X.max() * 2)
    y = np.array([-4.0, -3.0, -0.5, 0.0, 1.2, 3.0, 4.0])
    df = pd.DataFrame({"y": y})
    cases = [("linear", "linear", None, None, 1, 1),
             ("power", "power", 2.0, 1.5, 2, 2)]
    for lt, ut, lp, up, lc, uc in cases:
        old = gs.backtr(df, "y", tr.values, tr.scores, tr.zmin, tr.zmax,
                        lc, lp, uc, up)
        new = stats.backtr(y, tr, lt, ut, lp, up)
        assert np.allclose(new, old, rtol=1e-3)  # legacy gcum is an approximation


def test_hyperbolic_upper_tail_monotone_and_above_max():
    _, tr = stats.nscore(X)
    y = np.array([3.0, 4.0, 5.0])
    z = stats.backtr(y, tr, "linear", "hyperbolic", None, 1.5)
    assert np.all(np.diff(z) > 0) and z[0] >= X.max() - 1e-9


def test_clamp_and_reference_distribution():
    _, tr = stats.nscore(X)
    assert stats.backtr([-9, 9], tr, None, None).tolist() == [X.min(), X.max()]
    ref = rng.normal(size=500)
    ns, rtr = stats.nscore(X, reference=ref)
    assert ns.shape == X.shape and np.all(np.isfinite(ns))
    assert rtr.values.size == 500


def test_ties_get_distinct_scores():
    y, _ = stats.nscore(np.array([1.0, 1.0, 2.0, 2.0]))
    assert len(set(y)) == 4


def test_validation():
    with pytest.raises(ValueError):
        stats.nscore([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        stats.nscore([1.0, np.nan])
    _, tr = stats.nscore(X)
    with pytest.raises(ValueError):
        stats.backtr([0.0], tr, "power", "linear", None, None)
    with pytest.raises(ValueError):
        stats.backtr([0.0], tr, "cubic", "linear")
