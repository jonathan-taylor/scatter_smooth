"""
Bad input raises a clear error rather than crashing or returning nonsense:
#12 (predict before smooth), #13 (mismatched lengths in C++), #14 (input
validation) and #21 (deriv).
"""
import numpy as np
import pytest

from scatter_smooth import SplineSmoother, LoessSmoother
from scatter_smooth._scatter_smooth_extension import (ReinschSmoother,
                                                      BSplineSmoother,
                                                      NaturalSplineSmoother,
                                                      LoessSmootherCpp,
                                                      trace_takahashi)

rng = np.random.default_rng(0)
X = np.sort(rng.uniform(0, 10, 50))
Y = np.sin(X) + rng.normal(0, .1, 50)


# --- #12: predict before smooth

@pytest.mark.parametrize('kwargs', [dict(df=5), dict(df=5, n_knots=10), dict(lamval=1., engine='natural')])
def test_predict_before_smooth(kwargs):
    with pytest.raises(RuntimeError, match='not fitted'):
        SplineSmoother(X, **kwargs).predict(X[:3])


def test_cpp_predict_before_smooth():
    xs = (X - X.min()) / (X.max() - X.min())
    with pytest.raises(RuntimeError, match='not fitted'):
        ReinschSmoother(xs, None).predict(xs[:3], 0)
    with pytest.raises(RuntimeError, match='not fitted'):
        BSplineSmoother(xs, np.linspace(0, 1, 10), None, 4).predict(xs[:3], 0)
    with pytest.raises(RuntimeError, match='not fitted'):
        LoessSmootherCpp(X, None, 0.5, 1).predict(X[:3], 0)
    with pytest.raises(ValueError, match='not been fitted'):
        LoessSmoother(X).predict(X[:3])


# --- #13: mismatched lengths

def test_mismatched_lengths_python():
    with pytest.raises(ValueError, match='length'):
        SplineSmoother(X, w=np.ones(10), lamval=1.)
    s = SplineSmoother(X, lamval=1.)
    with pytest.raises(ValueError, match='length'):
        s.smooth(Y[:10])
    with pytest.raises(ValueError, match='length'):
        s.update_weights(np.ones(10))
    for engine in ['bspline', 'natural']:
        s = SplineSmoother(X, lamval=1., n_knots=10, engine=engine)
        with pytest.raises(ValueError, match='length'):
            s.smooth(Y[:10])
    l = LoessSmoother(X)
    with pytest.raises(ValueError, match='length'):
        l.smooth(Y[:10])
    with pytest.raises(ValueError, match='length'):
        LoessSmoother(X, w=np.ones(10))


def test_mismatched_lengths_cpp():
    xs = (X - X.min()) / (X.max() - X.min())
    with pytest.raises(ValueError, match='length'):
        ReinschSmoother(xs, np.ones(3))
    with pytest.raises(ValueError, match='length'):
        ReinschSmoother(xs, None).smooth(Y[:5], 1.)
    b = BSplineSmoother(xs, np.linspace(0, 1, 10), None, 4)
    with pytest.raises(ValueError, match='length'):
        b.compute_rhs(Y[:3])
    with pytest.raises(ValueError, match='length'):
        b.update_weights(np.ones(3))
    with pytest.raises(ValueError, match='length'):
        BSplineSmoother(xs, np.linspace(0, 1, 10), np.ones(3), 4)
    n = NaturalSplineSmoother(xs, np.linspace(0, 1, 10), None)
    with pytest.raises(ValueError, match='length'):
        n.smooth(Y[:3], 1.)
    with pytest.raises(ValueError, match='length'):
        n.update_weights(np.ones(3))
    l = LoessSmootherCpp(X, None, 0.5, 1)
    with pytest.raises(ValueError, match='length'):
        l.set_y(Y[:10])
    with pytest.raises(ValueError, match='length'):
        LoessSmootherCpp(X, np.ones(10), 0.5, 1)
    with pytest.raises(ValueError, match='same shape'):
        trace_takahashi(np.ones((4, 10)), np.ones((4, 9)))


# --- #14: input validation

@pytest.mark.parametrize('x', [np.array([1.]), np.ones(20), np.array([0., 1, 2]),
                               np.r_[X[:-1], np.nan], np.array([])])
def test_bad_x(x):
    with pytest.raises(ValueError):
        SplineSmoother(x, df=3)


def test_bad_y():
    s = SplineSmoother(X, lamval=1.)
    for bad in [np.nan, np.inf]:
        y = Y.copy()
        y[3] = bad
        with pytest.raises(ValueError, match='non-finite'):
            s.smooth(y)


def test_weights():
    with pytest.raises(ValueError, match='non-negative'):
        SplineSmoother(X, w=np.r_[-1., np.ones(49)], lamval=1.)
    with pytest.raises(ValueError, match='positive'):
        SplineSmoother(X, w=np.zeros(50), lamval=1.)
    # a zero weight is allowed, as in R: the point does not count
    w = np.ones(50)
    w[10] = 0
    s = SplineSmoother(X, w=w, lamval=1.)
    assert s.engine == 'bspline'      # Reinsch needs a positive weight at each knot
    f = s.smooth(Y)
    assert np.all(np.isfinite(f))
    y2 = Y.copy()
    y2[10] = 1e6
    np.testing.assert_allclose(s.smooth(y2), f, rtol=1e-10, atol=1e-10)
    with pytest.raises(ValueError, match='positive weight'):
        SplineSmoother(X, w=w, lamval=1., engine='reinsch', all_knots=True)
    s = SplineSmoother(X, lamval=1., all_knots=True)
    with pytest.raises(ValueError, match='positive weight'):
        s.update_weights(w)


def test_lamval_df():
    with pytest.raises(ValueError, match='non-negative'):
        SplineSmoother(X, lamval=-1.)
    with pytest.raises(ValueError, match='one of df and lamval'):
        SplineSmoother(X, lamval=1., df=5)
    with pytest.raises(ValueError, match='out of range'):
        SplineSmoother(X, df=0)
    with pytest.raises(ValueError, match='out of range'):
        SplineSmoother(X, df=60)


def test_order_and_knots():
    with pytest.raises(ValueError, match='order'):
        SplineSmoother(X, lamval=1., order=2, n_knots=10)
    with pytest.raises(ValueError, match='n_knots'):
        SplineSmoother(X, lamval=1., n_knots=3)
    # the caller's knots are not sorted in place
    knots = np.linspace(10, 0, 8)
    SplineSmoother(X, lamval=1., knots=knots)
    assert knots[0] == 10


def test_loess_validation():
    for span in [0, -1, np.nan]:
        with pytest.raises(ValueError, match='span'):
            LoessSmoother(X, span=span)
    with pytest.raises(ValueError, match='empty'):
        LoessSmoother(np.array([]))
    with pytest.raises(ValueError, match='non-finite'):
        LoessSmoother(np.r_[X[:-1], np.nan])


def test_column_vector_x():
    s1 = SplineSmoother(X[:, None], lamval=1.)
    s2 = SplineSmoother(X, lamval=1.)
    np.testing.assert_allclose(s1.smooth(Y), s2.smooth(Y))
    np.testing.assert_allclose(s1.predict(X[:, None]), s2.predict(X))
    np.testing.assert_allclose(s2.predict(1.), s2.predict(np.array([1.])))


# --- #21: deriv

@pytest.mark.parametrize('kwargs', [dict(all_knots=True), dict(n_knots=12)])
def test_third_derivative(kwargs):
    # f''' is constant on each interval between knots: central differences of f''
    s = SplineSmoother(X, lamval=0.1, **kwargs)
    s.smooth(Y)
    t = np.sort(s.knots)
    mid = 0.5 * (t[:-1] + t[1:])
    h = 1e-4 * (t[1:] - t[:-1])
    d3 = s.predict(mid, deriv=3)
    fd = (s.predict(mid + h, deriv=2) - s.predict(mid - h, deriv=2)) / (2 * h)
    np.testing.assert_allclose(d3, fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())


def test_bad_deriv():
    s = SplineSmoother(X, lamval=1.)
    s.smooth(Y)
    for d in [-1, 4]:
        with pytest.raises(ValueError, match='deriv'):
            s.predict(X, deriv=d)
    s = SplineSmoother(X, lamval=1., n_knots=10, engine='natural')
    s.smooth(Y)
    with pytest.raises(ValueError, match='deriv'):
        s.predict(X, deriv=3)
    l = LoessSmoother(X, degree=1)
    l.smooth(Y)
    for d in [-1, 2]:
        with pytest.raises(ValueError, match='deriv'):
            l.predict(X, deriv=d)
