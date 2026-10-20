"""
GCV as R's smooth.spline, with ties and weights (#17), and the deprecated
natural engine (#20).
"""
import numpy as np
import pytest

from scatter_smooth import SplineSmoother

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, default_converter
    from rpy2.robjects.conversion import localconverter
    ro.r('1')
    R_ENABLED = True
except Exception:
    R_ENABLED = False


def _data(ties, weights):
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(0, 10, 300))
    y = np.sin(x) + rng.normal(0, .3, 300)
    w = rng.uniform(.2, 3, 300) if weights else None
    return (np.round(x, 2) if ties else x), y, w


@pytest.mark.parametrize('ties', [False, True])
@pytest.mark.parametrize('weights', [False, True])
def test_gcv_score(ties, weights):
    # the criterion: weights normalized to mean 1, the RSS over all the
    # observations (within the ties too), n the number of them
    x, y, w = _data(ties, weights)
    s = SplineSmoother(x, w=w, lamval=0.1, all_knots=True)
    f = s.smooth(y)
    ww = np.ones_like(y) if w is None else w * len(y) / w.sum()
    n = len(y)
    expected = (np.sum(ww * (y - f)**2) / n) / (1 - s.compute_df() / n)**2
    np.testing.assert_allclose(s.gcv_score(y), expected, rtol=1e-12)


@pytest.mark.skipif(not R_ENABLED, reason='R or rpy2 is not installed')
@pytest.mark.parametrize('ties', [False, True])
@pytest.mark.parametrize('weights', [False, True])
def test_gcv_matches_R(ties, weights):
    # with ties, the df chosen was 11.78 (R: 11.64), and 13.40 with weights (R: 13.73)
    x, y, w = _data(ties, weights)
    s = SplineSmoother(x, w=w, all_knots=True)
    s.solve_gcv(y)
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv['x'] = x
        ro.globalenv['y'] = y
        ro.globalenv['w'] = w if w is not None else np.ones_like(y)
        df_R = ro.r('smooth.spline(x, y, w=w, all.knots=TRUE, cv=FALSE)$df')[0]
    # R's optimizer (golden section on spar) stops at about 1e-3 in df
    np.testing.assert_allclose(s.compute_df(), df_R, atol=0.01)


@pytest.mark.parametrize('kwargs', [dict(), dict(all_knots=True), dict(n_knots=30)])
def test_gcv_is_a_minimum(kwargs):
    x, y, w = _data(True, True)
    s = SplineSmoother(x, w=w, **kwargs)
    lam = s.solve_gcv(y)
    best = s.gcv_score(y, lam)
    for factor in [0.8, 1.25]:
        assert s.gcv_score(y, lam * factor) >= best - 1e-12


def test_natural_engine_deprecated_and_df():
    # #20: df targeting failed with x outside [0, 1]; its df comes from the
    # B-spline engine with the same knots now (the same model)
    x = np.sort(np.random.default_rng(0).uniform(0, 10, 50))
    y = np.sin(x)
    with pytest.warns(DeprecationWarning, match='bspline'):
        s = SplineSmoother(x, df=5, engine='natural', all_knots=True)
    np.testing.assert_allclose(s.compute_df(), 5, rtol=1e-8)
    b = SplineSmoother(x, df=5, engine='bspline', all_knots=True)
    np.testing.assert_allclose(s.lamval, b.lamval, rtol=1e-8)
    np.testing.assert_allclose(s.smooth(y), b.smooth(y), rtol=1e-5, atol=1e-6)
    # its GCV uses the weights (it ignored them)
    x, y, w = _data(False, True)
    with pytest.warns(DeprecationWarning):
        s = SplineSmoother(x, w=w, engine='natural', n_knots=40)
    b = SplineSmoother(x, w=w, engine='bspline', n_knots=40)
    np.testing.assert_allclose(s.solve_gcv(y), b.solve_gcv(y), rtol=1e-3)
