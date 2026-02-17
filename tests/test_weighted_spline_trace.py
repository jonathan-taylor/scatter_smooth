"""
Tests for weighted spline trace (degrees of freedom) calculations.

This module verifies the agreement between different algorithms for
computing the trace of the hat matrix for a *weighted* smoothing spline.
"""
import numpy as np
import pytest

from scatter_smooth._scatter_smooth_extension import ReinschSmoother
from .weighted_spline_trace import WeightedCubicSplineTrace, naive_weighted_trace

@pytest.mark.parametrize("lam", np.logspace(-10, 2, 5))
@pytest.mark.parametrize("use_weights", [True, False])
def test_weighted_trace_vs_reinsch_sparse(lam, use_weights):
    """
    Compare O(N) weighted trace, sparse solve, and naive dense calculations.

    This test checks three different methods for calculating the effective
    degrees of freedom for a weighted (or unweighted) smoothing spline:

    1.  **Weighted O(N) Trace (Takahashi's Algorithm)**: An efficient
        algorithm adapted for the weighted case.
    2.  **Sparse Solve Calculation**: A method implemented in the C++ Reinsch
        smoother that uses a sparse solver.
    3.  **Naive Dense Calculation**: A slow O(N^3) method that constructs
        the dense hat matrix and takes its trace, used as a ground truth.

    The test asserts that all three methods produce nearly identical results
    across a range of lambda values, for both weighted and unweighted scenarios.
    """
    rng = np.random.default_rng(2023)
    n_points = 100
    x = np.sort(rng.uniform(0, 10, n_points))

    weights = None
    if use_weights:
        weights = rng.uniform(0.5, 2.0, n_points)

    # Scaling similar to SplineSmoother in fitter.py
    x_min, x_max = x.min(), x.max()
    x_scale = x_max - x_min if x_max > x_min else 1.0
    x_scaled = (x - x_min) / x_scale
    lam_scaled = lam / (x_scale**3)

    # 1. Weighted O(N) trace calculation (Takahashi's algorithm)
    weighted_trace_solver = WeightedCubicSplineTrace(x_scaled, weights=weights)
    df_weighted_takahashi = weighted_trace_solver.compute_trace(lam_scaled)

    # 2. Sparse solve DF calculation from Reinsch fitter
    reinsch_fitter_cpp = ReinschSmoother(x_scaled, weights)
    df_sparse_solve = reinsch_fitter_cpp.compute_df_sparse(lam_scaled)

    # 3. Naive O(N^3) weighted trace calculation
    df_naive = naive_weighted_trace(x_scaled, weights if use_weights else np.ones(n_points), lam_scaled)

    np.testing.assert_allclose(df_weighted_takahashi, df_sparse_solve, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(df_weighted_takahashi, df_naive, atol=1e-4, rtol=1e-4)
