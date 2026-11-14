import numpy as np
import pandas as pd
import pytest

from geostatspy import stats
from geostatspy import geostats as gs

rng = np.random.default_rng(7)
# clustered sampling: dense cluster of LOW-valued... high values sampled densely
Xc = np.r_[rng.uniform(0, 100, 60), rng.normal(30, 4, 80)]
Yc = np.r_[rng.uniform(0, 100, 60), rng.normal(30, 4, 80)]
V = 1.0 + 0.05 * (100 - np.hypot(Xc - 30, Yc - 30)) + rng.normal(0, 0.1, 140)


def test_cell_declus_weights_sum_and_reduce_bias():
    w, sizes, means = stats.cell_declus(Xc, Yc, V, 5, 60, ncell=20, noff=4)
    assert w.sum() == pytest.approx(len(V)) and np.all(w > 0)
    assert np.average(V, weights=w) < V.mean()      # clustered highs down-weighted
    assert sizes[0] == 0 and means[0] == pytest.approx(V.mean())
    assert len(sizes) == len(means) == 22


def test_cell_declus_matches_legacy():
    df = pd.DataFrame({"x": Xc, "y": Yc, "v": V})
    w_old, s_old, m_old = gs.declus(df, "x", "y", "v", iminmax=1, noff=4,
                                    ncell=20, cmin=5, cmax=60)
    w, s, m = stats.cell_declus(Xc, Yc, V, 5, 60, ncell=20, noff=4)
    assert np.allclose(s, s_old)
    assert np.allclose(m, m_old, rtol=1e-2)
    assert np.average(V, weights=w) == pytest.approx(
        np.average(V, weights=w_old), rel=1e-2)


def test_cell_declus_max_and_uniform_fallback():
    w, _, _ = stats.cell_declus(Xc, Yc, V, 5, 60, ncell=20, minmax="max")
    assert np.average(V, weights=w) >= V.mean() - 1e-12
    # constant property: declustering can't change the mean -> equal weights
    gx, gy = np.meshgrid(np.arange(10.0), np.arange(10.0))
    w, _, _ = stats.cell_declus(gx.ravel(), gy.ravel(), np.ones(100), 1, 5, ncell=4)
    assert np.allclose(w, 1.0)


def test_polygonal_matches_legacy_on_same_nodes():
    nx = ny = 50
    b = (0.0, 100.0, 0.0, 100.0)
    w = stats.polygonal_declus(Xc, Yc, bounds=b, nx=nx, ny=ny)
    df = pd.DataFrame({"x": Xc, "y": Yc, "v": V})
    # legacy nodes sit at xmn + ix*xsiz; place them on our cell centers
    xsiz = 100.0 / nx
    w_old, _ = gs.polygonal_declus(df, "x", "y", "v", -1e9, 1e9, nx,
                                   xsiz / 2, xsiz, ny, xsiz / 2, xsiz)
    assert np.allclose(w, w_old)
    assert w.sum() == pytest.approx(len(V))


def test_polygonal_isolated_sample_gets_large_weight():
    x = np.r_[np.random.default_rng(0).uniform(0, 10, 30), 90.0]
    y = np.r_[np.random.default_rng(1).uniform(0, 10, 30), 90.0]
    w = stats.polygonal_declus(x, y, bounds=(0, 100, 0, 100))
    assert w[-1] > 5 * np.median(w[:-1])


def test_validation():
    with pytest.raises(ValueError):
        stats.cell_declus(Xc, Yc[:-1], V, 5, 60)
    with pytest.raises(ValueError):
        stats.cell_declus(Xc, Yc, V, 0, 60)
    with pytest.raises(ValueError):
        stats.cell_declus(Xc, Yc, V, 5, 60, minmax="avg")
    with pytest.raises(ValueError):
        stats.polygonal_declus([1, 1], [2, 2])
