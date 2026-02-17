"""
Tests for spline trace (degrees of freedom) calculations.

This module verifies the agreement between different algorithms for
computing the trace of the hat matrix (effective degrees of freedom) for
a smoothing spline.
"""
import numpy as np
import pytest

from scatter_smooth import SplineSmoother
from scatter_smooth._scatter_smooth_extension import CubicSplineTraceCpp, ReinschSmoother

@pytest.mark.parametrize("lam", np.logspace(-10, 2, 10))
def test_unweighted_trace_cpp_agreement(lam):
    """
    Compare O(N) trace, basis form, and sparse solve calculations.

    This test checks three different methods for calculating the effective
    degrees of freedom (trace of the hat matrix S) for an unweighted
    smoothing spline across a range of lambda values:

    1.  **O(N) Fast Trace (Takahashi's Algorithm)**: An efficient method
        specifically for the unweighted case with knots at all data points.
    2.  **Basis Form Calculation**: The general method using the spline
        basis and penalty matrices.
    3.  **Sparse Solve Calculation**: A method implemented in the Reinsch
        smoother that uses a sparse solver.

    The test asserts that all three methods produce nearly identical results.
    """
    rng = np.random.default_rng(2023)
    x = np.sort(rng.uniform(0, 10, 100))
    y = np.sin(x) + rng.normal(0, 0.1, 100)

    # Setup the fitter
    fitter_all_knots = SplineSmoother(x, knots=x)
    x_scaled_internal = (x - fitter_all_knots.x_min_) / fitter_all_knots.x_scale_
    
    # 1. O(N) trace calculation (Takahashi's algorithm)
    trace_solver_cpp = CubicSplineTraceCpp(x_scaled_internal)
    lam_scaled = lam / fitter_all_knots.x_scale_**3
    df_fast_cpp = trace_solver_cpp.compute_trace(lam_scaled)

    # 2. Basis form DF calculation
    fitter_all_knots._use_reinsch = False
    fitter_all_knots._cpp_fitter = None # Force re-initialization
    from scatter_smooth._scatter_smooth_extension import NaturalSplineSmoother
    fitter_all_knots._setup_scaling_and_knots()
    x_scaled = fitter_all_knots.x_scaled_
    knots_scaled = fitter_all_knots.knots_scaled_
    fitter_all_knots._cpp_fitter = NaturalSplineSmoother(x_scaled, knots_scaled, fitter_all_knots.w)
    fitter_all_knots.lamval = lam
    df_basis_all_knots = fitter_all_knots._cpp_fitter.compute_df(lam_scaled)
    
    # 3. Sparse solve DF calculation
    reinsch_fitter_cpp = ReinschSmoother(x_scaled_internal, weights=None)
    df_sparse_solve = reinsch_fitter_cpp.compute_df_sparse(lam_scaled)

    np.testing.assert_allclose(df_fast_cpp, df_basis_all_knots, atol=1e-4)
    np.testing.assert_allclose(df_fast_cpp, df_sparse_solve, atol=1e-4)

