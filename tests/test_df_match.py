"""
Tests for comparing different methods of df calculation.

This module verifies that the fast Takahashi algorithm for computing the
effective degrees of freedom (df) for a B-spline smoother yields the same
result as the direct, but slower, computation by taking the trace of the
hat matrix.
"""
import numpy as np
from scatter_smooth import SplineSmoother

def test_bspline_df_takahashi():
    """
    Test that the Takahashi df calculation matches the C++ trace method.

    This test initializes a B-spline smoother and computes the effective
    degrees of freedom for a given lambda using two methods:
    1. The fast Takahashi algorithm implemented in Python.
    2. The direct trace-of-hat-matrix method implemented in C++.

    It then asserts that the results from both methods are all-close.
    """
    np.random.seed(42)
    n = 100
    x = np.sort(np.random.rand(n))
    y = np.sin(10 * x) + np.random.normal(0, 0.1, n)
    
    # Init with bspline engine
    spline = SplineSmoother(x, engine='bspline', order=4, n_knots=10)
    
    lam = 1e-4
    df_takahashi = spline.compute_df(lam)
    
    # Compare with C++ implementation (which uses the slow method currently)
    # We need to call the C++ compute_df directly or use engine='natural' (if compatible)
    # but natural has different basis.
    
    # We can use the C++ fitter directly from the spline object
    lam_scaled = lam / (spline.x_scale_**3)
    df_cpp_slow = spline._cpp_fitter.compute_df(lam_scaled)
    
    print(f"DF Takahashi: {df_takahashi}")
    print(f"DF CPP Slow:  {df_cpp_slow}")
    
    assert np.allclose(df_takahashi, df_cpp_slow), f"DF mismatch: {df_takahashi} != {df_cpp_slow}"
    print("SUCCESS: DF calculation matches.")

if __name__ == "__main__":
    test_bspline_df_takahashi()
