import numpy as np
import pytest

from smoothing_spline.fitter import SplineFitter
from smoothing_spline._spline_extension import CubicSplineTraceCpp, SplineFitterReinschCpp

@pytest.mark.parametrize("lam", np.logspace(-10, 2, 10))
def test_unweighted_trace_cpp_agreement(lam):
    """
    Compare the O(N) trace calculation, the basis form calculation,
    and the sparse solve calculation across a grid of lambda values.
    """
    rng = np.random.default_rng(2023)
    x = np.sort(rng.uniform(0, 10, 100))
    y = np.sin(x) + rng.normal(0, 0.1, 100)

    # Setup the fitter
    fitter_all_knots = SplineFitter(x, knots=x)
    x_scaled_internal = (x - fitter_all_knots.x_min_) / fitter_all_knots.x_scale_
    
    # 1. O(N) trace calculation (Takahashi's algorithm)
    trace_solver_cpp = CubicSplineTraceCpp(x_scaled_internal)
    lam_scaled = lam / fitter_all_knots.x_scale_**3
    df_fast_cpp = trace_solver_cpp.compute_trace(lam_scaled)

    # 2. Basis form DF calculation
    fitter_all_knots._use_reinsch = False
    fitter_all_knots._cpp_fitter = None # Force re-initialization
    from smoothing_spline._spline_extension import SplineFitterCpp
    x_scaled, knots_scaled = fitter_all_knots._setup_scaling_and_knots()
    fitter_all_knots._cpp_fitter = SplineFitterCpp(x_scaled, knots_scaled, fitter_all_knots.w)
    fitter_all_knots.lamval = lam
    df_basis_all_knots = fitter_all_knots._cpp_fitter.compute_df(lam_scaled)
    
    # 3. Sparse solve DF calculation
    reinsch_fitter_cpp = SplineFitterReinschCpp(x_scaled_internal, weights=None)
    df_sparse_solve = reinsch_fitter_cpp.compute_df_sparse(lam_scaled)

    np.testing.assert_allclose(df_fast_cpp, df_basis_all_knots, atol=1e-4)
    np.testing.assert_allclose(df_fast_cpp, df_sparse_solve, atol=1e-4)

