"""
Tests for the Reinsch algorithm implementation.

This module verifies the correctness of the C++ `ReinschSmoother` against the
more general `SplineSmoother` when the number of knots equals the number of
data points. It checks:
- The fitted values from the smoother.
- The calculation of effective degrees of freedom.
- The calculation of the Generalized Cross-Validation (GCV) score.
"""
import numpy as np
import pytest
from scatter_smooth import SplineSmoother

from scatter_smooth._scatter_smooth_extension import ReinschSmoother

def test_reinsch_fitter_exact_fit():
    """
    Test that the Reinsch fitter gives the same result as the basis fitter.

    When the number of knots is equal to the number of data points, the
    smoothing spline is an exact fit, and the Reinsch algorithm provides an
    efficient O(n) solution. This test confirms that the result from the
    `ReinschSmoother` is identical to that of the general `SplineSmoother`
    in this specific case.
    """
    rng = np.random.default_rng(200)
    x = np.sort(rng.uniform(0, 10, 50))
    # Reinsch is for n_knots == n_x
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    
    # Python SplineSmoother (Basis form)
    py_fitter = SplineSmoother(x, knots=x, engine='natural')
    py_fitter.smooth(y)
    
    # Reinsch Smoother
    # Pass SCALED x to match physics if we want to use same lambda scaling
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    
    reinsch_fitter = ReinschSmoother(x_scaled, None)
    
    lamval = 0.5
    lam_scaled = lamval / scale**3
    
    # Basis fit
    py_fitter.lamval = lamval # This will use basis form inside
    py_fitter.smooth(y)
    basis_pred = py_fitter.predict(x)
    
    # Reinsch fit
    # fit returns fitted values directly
    reinsch_pred = reinsch_fitter.smooth(y, lam_scaled)
    
    np.testing.assert_allclose(reinsch_pred, basis_pred, atol=1e-5)

def test_reinsch_df():
    """
    Test the degrees of freedom calculation for the Reinsch smoother.

    This test verifies that the fast O(n) degrees of freedom calculation
    from the `ReinschSmoother` matches the result from the slower, but more
    general, trace-of-the-hat-matrix calculation in the basis smoother.
    """
    rng = np.random.default_rng(201)
    x = np.sort(rng.uniform(0, 10, 50))
    
    py_fitter = SplineSmoother(x, knots=x, engine='natural')
    py_fitter._prepare_matrices()
    
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    
    reinsch_fitter = ReinschSmoother(x_scaled, None)
    
    lamval = 0.1
    lam_scaled = lamval / scale**3
    
    # Calculate DF via Basis (slow)
    # df = trace(S)
    basis_df = py_fitter._cpp_fitter.compute_df(lam_scaled)
    
    # Calculate DF via Reinsch (fast)
    reinsch_df = reinsch_fitter.compute_df(lam_scaled)
    
    np.testing.assert_allclose(reinsch_df, basis_df, atol=1e-5)

def test_reinsch_gcv():
    """
    Test the GCV score calculation for the Reinsch smoother.

    This test ensures that the `gcv_score` method of the `ReinschSmoother`
    produces the same value as the equivalent calculation in the general
    basis smoother.
    """
    rng = np.random.default_rng(202)
    x = np.sort(rng.uniform(0, 10, 50))
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    
    py_fitter = SplineSmoother(x, knots=x, engine='natural')
    py_fitter.smooth(y)
    
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    
    reinsch_fitter = ReinschSmoother(x_scaled, None)
    
    lamval = 0.1
    lam_scaled = lamval / scale**3
    
    # Basis GCV
    basis_gcv = py_fitter._cpp_fitter.gcv_score(lam_scaled, y)
    
    # Reinsch GCV
    reinsch_gcv = reinsch_fitter.gcv_score(lam_scaled, y)
    
    np.testing.assert_allclose(reinsch_gcv, basis_gcv, atol=1e-5)
