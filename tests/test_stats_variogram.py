import numpy as np
import pandas as pd
import pytest

from geostatspy import stats
from geostatspy import geostats as gs
from geostatspy.stats import VariogramModel

rng = np.random.default_rng(11)
N = 250
XY = rng.uniform(0, 100, (N, 2))
V = np.sin(XY[:, 0] / 15) + 0.5 * np.cos(XY[:, 1] / 25) + rng.normal(0, 0.2, N)
DF = pd.DataFrame({"x": XY[:, 0], "y": XY[:, 1], "v": V})


# ---- model ---------------------------------------------------------------
def test_model_basic_properties():
    m = VariogramModel(0.1, [{"it": "spherical", "cc": 0.9, "hmaj": 30}])
    assert m.sill == pytest.approx(1.0)
    assert m.gamma(0.0) == 0.0 and m.cov(0.0) == pytest.approx(1.0)
    assert m.gamma(1e-9) == pytest.approx(0.1, abs=1e-6)       # nugget jump
    assert m.gamma(30.0) == pytest.approx(1.0) and m.gamma(100.0) == pytest.approx(1.0)
    h = np.linspace(0.1, 29, 50)
    assert np.all(np.diff(m.gamma(h)) > 0)


@pytest.mark.parametrize("it", ["exponential", "gaussian"])
def test_practical_range_is_95pct(it):
    m = VariogramModel(0.0, [{"it": it, "cc": 1.0, "hmaj": 20}])
    assert m.gamma(20.0) == pytest.approx(1 - np.exp(-3), rel=1e-9)


def test_model_matches_legacy_vmodel_anisotropic():
    vario = {"nug": 0.1, "nst": 2,
             "cc1": 0.5, "it1": 1, "azi1": 30, "hmaj1": 40, "hmin1": 15,
             "cc2": 0.4, "it2": 2, "azi2": 120, "hmaj2": 80, "hmin2": 50}
    m = VariogramModel(0.1, [
        {"it": 1, "cc": 0.5, "azi": 30, "hmaj": 40, "hmin": 15},
        {"it": 2, "cc": 0.4, "azi": 120, "hmaj": 80, "hmin": 50}])
    for azm in (0.0, 30.0, 75.0, 120.0):
        _, h, gam, cov, _ = gs.vmodel(10, 5.0, azm, vario)
        assert np.allclose(m.gamma(h, azm), gam, atol=1e-6)
        assert np.allclose(m.cov(h[1:], azm), cov[1:], atol=1e-6)


def test_model_validation():
    with pytest.raises(ValueError):
        VariogramModel(0, [{"it": "cubic", "cc": 1, "hmaj": 1}])
    with pytest.raises(ValueError):
        VariogramModel(0, [{"it": 1, "cc": 1, "hmaj": 0}])
    with pytest.raises(ValueError):
        VariogramModel(-1, [])


# ---- experimental --------------------------------------------------------
@pytest.mark.parametrize("azm,atol,bw", [(0.0, 22.5, 1e5), (45.0, 20.0, 15.0),
                                         (0.0, 90.0, 1e5)])
def test_matches_legacy_gamv(azm, atol, bw):
    lag, nlag = 8.0, 6
    # legacy lag index il is centred on (il-1)*lag -> ask for one more lag
    dis, gam, npp = gs.gamv(DF, "x", "y", "v", -1e9, 1e9, lag, 0.5 * lag,
                            nlag + 1, azm, atol, bw, 0)
    d, g, n = stats.experimental_variogram(XY, V, lag, nlag, azimuth=azm,
                                           azimuth_tol=atol, bandwidth=bw)
    ok = n > 0
    assert ok.sum() >= 4
    assert np.allclose(d[ok], dis[2:nlag + 2][ok])
    assert np.allclose(g[ok], gam[2:nlag + 2][ok])
    # legacy counts ordered pairs (x2) and, if omnidirectional, again (x2)
    assert np.array_equal(n, (npp[2:nlag + 2] / (4 if atol >= 90 else 2)).astype(int))


def test_unordered_pair_count_omni_no_overlap():
    d, g, n = stats.experimental_variogram(XY, V, 10.0, 5, azimuth_tol=90)
    h = np.hypot(*(XY[:, None] - XY[None]).transpose(2, 0, 1))
    iu = np.triu_indices(N, 1)
    expect = [(np.abs(h[iu] - 10.0 * k) <= 5.0).sum() for k in range(1, 6)]
    assert n.tolist() == expect


def test_recovers_known_model():
    r = np.random.default_rng(3)
    pts = r.uniform(0, 100, (400, 2))
    # white noise + nothing: pure nugget -> flat variogram at variance
    z = r.normal(0, 1, 400)
    _, g, _ = stats.experimental_variogram(pts, z, 8.0, 6, azimuth_tol=90)
    assert np.allclose(g, 1.0, atol=0.2)
    _, gs_, _ = stats.experimental_variogram(pts, z, 8.0, 6, azimuth_tol=90,
                                             standardize=True)
    assert np.allclose(gs_, g / z.var())


def test_empty_lags_are_nan_and_chunking_consistent():
    d, g, n = stats.experimental_variogram(XY, V, 8.0, 40, azimuth_tol=90)
    assert n[-1] == 0 and np.isnan(g[-1]) and np.isnan(d[-1])
    big = np.tile(XY, (3, 1)) + rng.normal(0, 1e-3, (3 * N, 2))   # > 1 block
    _, _, nb = stats.experimental_variogram(big, np.tile(V, 3), 8.0, 4, azimuth_tol=90)
    assert nb.sum() > 0


def test_experimental_validation():
    with pytest.raises(ValueError):
        stats.experimental_variogram(XY, V[:-1], 5, 3)
    with pytest.raises(ValueError):
        stats.experimental_variogram(XY, V, 0, 3)
    with pytest.raises(NotImplementedError):
        stats.experimental_variogram(np.zeros((5, 3)), np.zeros(5), 1, 2)


# ---- variogram map -------------------------------------------------------
def test_variogram_map_symmetry_center_and_counts():
    gam, npp = stats.variogram_map(XY, V, 5, 5, 10.0, 10.0, minnp=3)
    assert gam.shape == npp.shape == (11, 11)
    assert npp[5, 5] >= N and gam[5, 5] < 0.1                # self pairs + |lag|<half cell
    assert np.array_equal(npp, npp[::-1, ::-1])             # point symmetry
    ok = ~np.isnan(gam)
    assert np.allclose(gam[ok], gam[::-1, ::-1][ok])
    assert np.all(npp[np.isnan(gam)] < 3)


def test_variogram_map_east_strip_is_populated():
    # E-W strip of the map (row nylag) has finite gamma at positive lags
    gam, npp = stats.variogram_map(XY, V, 4, 4, 10.0, 10.0, minnp=1)
    assert np.isfinite(gam[4, 5:]).all() and (npp[4, 5:] > 0).all()
