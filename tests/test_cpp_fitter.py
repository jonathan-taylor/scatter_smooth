"""
Tests for the C++ natural spline fitter.

This module contains tests that verify the C++ implementation of the natural
spline smoother against a pure Python implementation. It checks:
- The correctness of the constructed basis (N) and penalty (Omega) matrices.
- The consistency of the fitted spline coefficients (alpha).
- The handling of weights.
- The calculation of effective degrees of freedom (df).
- The prediction on training and new data, including extrapolation.
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scatter_smooth import SplineSmoother
from tests.spline_fitter import SplineSmoother as SplineSmootherPy

from scatter_smooth._scatter_smooth_extension import NaturalSplineSmoother as ExtSplineSmootherCpp

def test_cpp_fitter_integration():
    """
    Test the integration of the C++ fitter with Python.

    This test ensures that the C++ fitter, when given properly scaled data,
    produces the same basis and penalty matrices as the Python implementation.
    It also verifies that the end-to-end fitting process (solving for alpha)
    yields the same result when accounting for the scaling of the smoothing
    parameter lambda.
    """
    rng = np.random.default_rng(99)
    x = np.sort(rng.uniform(0, 10, 50))
    knots = np.sort(rng.uniform(0, 10, 10))
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    
    # Python Matrices (using pure python implementation)
    py_fitter = SplineSmootherPy(x, knots=knots)
    py_fitter._prepare_matrices() # Computes N_, NTW_, Omega_
    
    # C++ Matrices via Class
    # Note: Python implementation scales x and knots internally.
    # C++ implementation uses raw values.
    # To compare matrices directly, we must pass scaled values to C++ 
    # OR adjust the comparison.
    
    scale = py_fitter.x_scale_
    x_min = py_fitter.x_min_
    x_scaled = (x - x_min) / scale
    knots_scaled = (knots - x_min) / scale
    
    # 1. Compare exact matrices by passing scaled data to C++
    cpp_fitter_scaled = ExtSplineSmootherCpp(x_scaled, knots_scaled)
    cpp_N = cpp_fitter_scaled.get_N()
    cpp_Omega = cpp_fitter_scaled.get_Omega()
    
    np.testing.assert_allclose(cpp_N, py_fitter.N_, atol=1e-8)
    np.testing.assert_allclose(cpp_Omega, py_fitter.Omega_, atol=1e-8)
    
    # 2. Compare Fit (End-to-End) using raw data in C++
    # Python uses scaled lambda and scaled matrices -> invariant objective
    # C++ using raw data should need RAW lambda to get same alpha
    
    cpp_fitter_raw = ExtSplineSmootherCpp(x, knots)
    
    lamval = 0.5
    # Python solve
    lam_scaled = lamval / scale**3
    LHS = py_fitter.NTW_ @ py_fitter.N_ + lam_scaled * py_fitter.Omega_
    RHS = py_fitter.NTW_ @ y
    py_alpha = np.linalg.solve(LHS, RHS)
    
    # C++ fit with raw lambda
    cpp_alpha = cpp_fitter_raw.smooth(y, lamval)
    
    np.testing.assert_allclose(cpp_alpha, py_alpha, atol=1e-5)

def test_cpp_fitter_weights():
    """
    Test that the C++ fitter handles weights correctly.

    This test compares the fitted coefficients from the C++ and Python
    implementations when observation weights are provided. It ensures that
    the C++ fitter produces the same results as the Python reference.
    """
    rng = np.random.default_rng(100)
    x = np.sort(rng.uniform(0, 10, 50))
    knots = np.sort(rng.uniform(0, 10, 10))
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    w = rng.uniform(0.5, 2.0, 50)
    
    py_fitter = SplineSmootherPy(x, knots=knots, w=w)
    py_fitter._prepare_matrices()
    
    # C++ uses raw x, knots
    cpp_fitter = ExtSplineSmootherCpp(x, knots, w)
    
    lamval = 0.5
    # Python internally scales, so we replicate the math for verification
    # But C++ fit(y, lamval) should work directly if logic holds
    
    # Verify Python's internal solve
    scale = py_fitter.x_scale_
    lam_scaled = lamval / scale**3
    LHS = py_fitter.NTW_ @ py_fitter.N_ + lam_scaled * py_fitter.Omega_
    RHS = py_fitter.NTW_ @ y
    py_alpha = np.linalg.solve(LHS, RHS)
    
    # C++ fit with raw lambda
    cpp_alpha = cpp_fitter.smooth(y, lamval)
    
    np.testing.assert_allclose(cpp_alpha, py_alpha, atol=1e-5)

def test_cpp_df():
    """
    Test the degrees of freedom calculation in C++.

    This test verifies that the C++ `compute_df` method produces the same
    effective degrees of freedom as the Python implementation, which is
    calculated by tracing the hat matrix S.
    """
    rng = np.random.default_rng(101)
    x = np.sort(rng.uniform(0, 10, 50))
    knots = np.sort(rng.uniform(0, 10, 10))
    
    py_fitter = SplineSmootherPy(x, knots=knots)
    py_fitter._prepare_matrices()
    
    cpp_fitter = ExtSplineSmootherCpp(x, knots)
    
    lamval = 0.5
    # Python DF
    scale = py_fitter.x_scale_
    lam_scaled = lamval / scale**3
    LHS = py_fitter.NTW_ @ py_fitter.N_ + lam_scaled * py_fitter.Omega_
    S_matrix = np.linalg.solve(LHS, py_fitter.NTW_ @ py_fitter.N_)
    py_df = np.trace(S_matrix)
    
    # C++ DF with raw lambda
    cpp_df = cpp_fitter.compute_df(lamval)
    
    np.testing.assert_allclose(cpp_df, py_df, atol=1e-5)

def test_cpp_prediction():
    """
    Test the prediction capabilities of the C++ fitter.

    This test compares the predicted values from the high-level C++-based
    `SplineSmoother` with the pure Python implementation. It checks for
    consistency on both the training data and new data points, including
    extrapolation outside the original data range.
    """
    # Helper already imports SplineSmoother as SplineSmootherPy
    rng = np.random.default_rng(102)
    x = np.sort(rng.uniform(0, 10, 50))
    y = np.sin(x) + rng.normal(0, 0.1, 50)
    
    # Pure Python fit
    py_fitter = SplineSmootherPy(x, lamval=0.2)
    py_fitter.smooth(y)
    py_pred = py_fitter.predict(x)
    
    # C++ fit (Main Class)
    cpp_fitter = SplineSmoother(x, lamval=0.2)
    cpp_fitter.smooth(y)
    cpp_pred = cpp_fitter.predict(x)
    
    np.testing.assert_allclose(cpp_pred, py_pred, atol=1e-5)
    
    # New data prediction (extrapolation check)
    x_new = np.linspace(-2, 12, 100)
    py_pred_new = py_fitter.predict(x_new)
    cpp_pred_new = cpp_fitter.predict(x_new)
    
    np.testing.assert_allclose(cpp_pred_new, py_pred_new, atol=1e-5)
