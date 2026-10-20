"""
LOESS with tied x and with span > 1 (#19).
"""
import numpy as np
import pytest

from scatter_smooth import LoessSmoother


@pytest.fixture
def wage():
    # integer-valued x, as ISL's Wage age
    rng = np.random.default_rng(0)
    age = rng.integers(18, 81, 3000).astype(float)
    return age, 50 + 2 * age - 0.02 * age**2 + rng.normal(0, 30, 3000)


@pytest.mark.parametrize('degree', [0, 1, 2, 3])
@pytest.mark.parametrize('span', [0.01, 0.05, 0.3])
def test_ties_small_span(wage, degree, span):
    # with all the neighbours at the boundary distance tied, the local design
    # was singular: degree 1 returned NaN, degree 2 garbage
    age, y = wage
    l = LoessSmoother(age, span=span, degree=degree)
    l.smooth(y)
    f = l.predict()
    assert np.all(np.isfinite(f))
    xg = np.linspace(18, 80, 200)
    g = l.predict(xg)
    assert np.all(np.isfinite(g))
    truth = 50 + 2 * xg - 0.02 * xg**2
    # each age has about 48 observations with sd 30: the fit is near the truth
    assert np.max(np.abs(g - truth)) < 20


def test_ties_matches_tied_means(wage):
    # a neighbourhood holding a single distinct x value: the local fit is the
    # mean of y there, for any degree
    age, y = wage
    for degree in [1, 2]:
        l = LoessSmoother(age, span=0.005, degree=degree)
        l.smooth(y)
        v = 40.
        np.testing.assert_allclose(l.predict(np.array([v])), y[age == v].mean(), rtol=1e-10)


@pytest.mark.parametrize('degree', [1, 2])
def test_span_above_one(wage, degree):
    # span > 1 widens the neighbourhood (sqrt(span) times the largest
    # distance): no longer the same fit as span = 1, and the global weighted
    # polynomial as span grows
    age, y = wage
    fits = {}
    for span in [1., 2., 1e8]:
        l = LoessSmoother(age, span=span, degree=degree)
        l.smooth(y)
        fits[span] = l.predict()
    assert np.max(np.abs(fits[2.] - fits[1.])) > 1e-3
    glob = np.polyval(np.polyfit(age, y, degree), age)
    np.testing.assert_allclose(fits[1e8], glob, rtol=1e-6)


@pytest.mark.parametrize('degree', [0, 1, 2, 3])
@pytest.mark.parametrize('span', [0.1, 0.5, 1.5])
@pytest.mark.parametrize('ties', [False, True])
def test_leverage_is_the_smoother_diagonal(degree, span, ties):
    # leverage(): the weight of y_i in the fit at x_i, in one pass; the
    # diagonal from smoothing the unit vectors, and its sum the df
    rng = np.random.default_rng(6)
    n = 120
    x = rng.uniform(0, 1, n)
    if ties:
        x = np.round(x * 15) / 15
    w = rng.uniform(.5, 2, n)
    l = LoessSmoother(x, w=w, span=span, degree=degree)
    lev = l.leverage()
    diag = np.empty(n)
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.
        l.smooth(e)
        diag[i] = l.predict()[i]
    np.testing.assert_allclose(lev, diag, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(l.compute_df(), diag.sum(), rtol=1e-8)
