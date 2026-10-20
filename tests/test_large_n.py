"""
Many unique x values: R's default knots, df targeting, weight updates and ties.
"""
import numpy as np
import pytest

from scatter_smooth import SplineSmoother
from scatter_smooth.splines import _nknots_smspl


def test_nknots_matches_R():
    # stats:::.nknots.smspl(c(49, 50, 51, 100, 199, 200, 500, 800, 3200, 10000, 1e6))
    n = [49, 50, 51, 100, 199, 200, 500, 800, 3200, 10000, 1000000]
    assert [_nknots_smspl(m) for m in n] == [49, 49, 50, 62, 99, 99, 118, 140, 200, 205, 215]


@pytest.mark.parametrize('n', [2000, 5000, 10000, 100000])
@pytest.mark.parametrize('df', [4, 10])
def test_df_large_n(n, df):
    # issue #11: with all the unique values as knots, df targeting failed from
    # about n = 5000 and silently fitted a line
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 10, n)
    s = SplineSmoother(x, df=df)
    assert s.engine == 'bspline'
    assert s.n_k_ == _nknots_smspl(n)
    np.testing.assert_allclose(s.compute_df(), df, rtol=1e-8)


def test_all_knots_ill_conditioned_raises():
    # all the knots at large n: an error, not a line
    x = np.random.default_rng(4).uniform(0, 10, 10000)
    with pytest.raises((RuntimeError, ValueError)):
        SplineSmoother(x, df=4, all_knots=True)


def test_all_knots_small_n_is_reinsch():
    x = np.random.default_rng(0).uniform(0, 1, 200)
    assert SplineSmoother(x, lamval=1e-3, all_knots=True).engine == 'reinsch'
    assert SplineSmoother(x, lamval=1e-3).engine == 'bspline'
    assert SplineSmoother(x[:40], lamval=1e-3).engine == 'reinsch'


@pytest.mark.parametrize('kwargs', [{}, {'all_knots': True}])
def test_smooth_returns_fitted_values_and_linear_part(kwargs):
    rng = np.random.default_rng(1)
    x = np.round(rng.uniform(0, 10, 500), 2)
    y = np.sin(x) + rng.normal(0, .3, 500)
    w = rng.uniform(.5, 2, 500)
    s = SplineSmoother(x, w=w, df=6, **kwargs)
    f = s.smooth(y)
    np.testing.assert_allclose(f, s.predict(x), rtol=1e-12, atol=1e-12)
    X = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], f * np.sqrt(w), rcond=None)[0]
    np.testing.assert_allclose([s.coef_, s.intercept_], beta, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(s.nonlinear_, f - (s.coef_ * x + s.intercept_), atol=1e-12)


@pytest.mark.parametrize('kwargs', [{}, {'all_knots': True}])
def test_update_weights_matches_fresh_smoother(kwargs):
    # issue #15: the bspline engine reused a stale Cholesky factor after the
    # weights changed; issue #16: smooth(sample_weight=) mishandled ties
    rng = np.random.default_rng(2)
    x = np.round(rng.uniform(0, 10, 400), 1)               # ties
    y = np.sin(x) + rng.normal(0, .3, 400)
    w1, w2 = rng.uniform(.2, 3, (2, 400))
    s = SplineSmoother(x, w=w1, lamval=0.5, **kwargs)
    s.compute_df()                                         # fills the cache
    s.smooth(y)
    s.update_weights(w2)
    ref = SplineSmoother(x, w=w2, lamval=0.5, **kwargs)
    np.testing.assert_allclose(s.smooth(y), ref.smooth(y), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(s.compute_df(), ref.compute_df(), rtol=1e-10)
    s = SplineSmoother(x, w=w1, lamval=0.5, **kwargs)
    np.testing.assert_allclose(s.smooth(y, sample_weight=w2), ref.smooth(y), rtol=1e-10, atol=1e-12)


def test_ties_are_aggregated():
    # the fit depends on the data through the unique x values only: with ties,
    # the same as the weighted means at the unique values with the summed weights
    rng = np.random.default_rng(3)
    xu = np.sort(rng.uniform(0, 10, 300))
    idx = rng.integers(0, 300, 2000)
    idx[:300] = np.arange(300)
    x, w = xu[idx], rng.uniform(.5, 2, 2000)
    y = np.cos(x) + rng.normal(0, .3, 2000)
    wu = np.bincount(idx, weights=w)
    yu = np.bincount(idx, weights=w * y) / wu
    knots = SplineSmoother(x, lamval=1.).knots
    s = SplineSmoother(x, w=w, lamval=0.1, knots=knots)
    su = SplineSmoother(xu, w=wu, lamval=0.1, knots=knots)
    np.testing.assert_allclose(s.smooth(y), su.smooth(yu)[idx], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(s.compute_df(), su.compute_df(), rtol=1e-10)


def test_df_out_of_range():
    x = np.random.default_rng(0).uniform(0, 10, 300)
    s = SplineSmoother(x, n_knots=10, lamval=1.)
    assert s.n_k_ == 10
    # the natural cubic splines with 10 knots have dimension 10
    np.testing.assert_allclose(s.compute_df(1e-12), 10, rtol=1e-6)


@pytest.mark.parametrize('kwargs', [{}, {'all_knots': True}])
def test_spline_coef_and_roughness(kwargs):
    rng = np.random.default_rng(5)
    x = np.round(rng.uniform(0, 10, 300), 1)
    y = np.sin(x) + rng.normal(0, .3, 300)
    s = SplineSmoother(x, lamval=0.3, **kwargs)
    f1 = s.smooth(y)
    c1, r1 = s.spline_coef_.copy(), s.roughness()
    # the roughness: the integral of f''^2 (f'' is linear between the knots)
    t = np.sort(np.unique(s.knots))
    d2 = s.predict(t, deriv=2)
    h = np.diff(t)
    np.testing.assert_allclose(r1, np.sum(h / 3 * (d2[:-1]**2 + d2[:-1] * d2[1:] + d2[1:]**2)),
                               rtol=1e-8)
    # the coefficients are linear in the fit; a constant shifts them
    f2 = s.smooth(np.cos(x))
    c2 = s.spline_coef_.copy()
    s.set_spline_coef(0.5 * (c1 + c2) - 2.)
    np.testing.assert_allclose(s.predict(x), 0.5 * (f1 + f2) - 2., rtol=1e-9, atol=1e-10)
    s.set_spline_coef(c1)
    np.testing.assert_allclose(s.predict(x), f1, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(s.roughness(c1 + 3.), r1, rtol=1e-9)


def test_df_beyond_range_raises():
    # issue #18
    x = np.random.default_rng(0).uniform(0, 10, 300)
    with pytest.raises(ValueError, match='out of range'):
        SplineSmoother(x, n_knots=10, df=11)
    with pytest.raises(ValueError, match='out of range'):
        SplineSmoother(x, df=1.5)
    np.testing.assert_allclose(SplineSmoother(x, n_knots=10, df=9.5).compute_df(), 9.5, rtol=1e-8)
    # tied quantile knots are dropped
    s = SplineSmoother(np.repeat(np.arange(5.), 10), n_knots=20, df=4)
    assert s.n_k_ == 5
    np.testing.assert_allclose(s.compute_df(), 4, rtol=1e-8)


def test_df_fit_is_exact():
    # after matching df, smooth() solved with compute_df's cached factor,
    # whose diagonal was padded: the fit was off by about 3e-7
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 5000)
    y = np.sin(6 * x) + rng.standard_normal(5000)
    s = SplineSmoother(x, df=5)
    f1 = s.smooth(y)
    # the penalized least squares solution, directly
    ref = SplineSmoother(x, lamval=s.lamval)
    np.testing.assert_allclose(f1, ref.smooth(y), rtol=1e-12, atol=1e-12)
