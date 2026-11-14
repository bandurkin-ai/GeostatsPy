import pytest
from geostatspy import stats

def test_public_api_importable():
    for name in stats.__all__:
        assert hasattr(stats, name)

@pytest.mark.xfail(raises=NotImplementedError, strict=True)
def test_nscore_roundtrip():
    import numpy as np
    x = np.random.default_rng(0).lognormal(size=200)
    y, tr = stats.nscore(x)
    assert np.allclose(stats.backtr(y, tr), x)
