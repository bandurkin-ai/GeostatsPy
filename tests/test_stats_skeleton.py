from geostatspy import stats


def test_public_api_importable():
    for name in stats.__all__:
        assert hasattr(stats, name)
