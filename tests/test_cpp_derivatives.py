"""
Tests for the derivatives of the C++ natural spline basis.

This module verifies the correctness of the derivatives of the C++ natural
spline basis implementation. It compares the 0th, 1st, and 2nd derivatives
against the equivalent derivatives from `scipy.interpolate.CubicSpline`.
The extrapolation behavior of the derivatives is also tested.
"""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from scatter_smooth._scatter_smooth_extension import compute_natural_spline_basis

def test_cpp_basis_derivatives():
    """
    Test that the C++ basis derivatives match scipy's implementation.

    This test computes the 0th, 1st, and 2nd derivatives of the natural
    spline basis using both the C++ function and `scipy.CubicSpline`.
    It asserts that the results are all-close for each derivative order.
    """
    rng = np.random.default_rng(300)
    x = np.sort(rng.uniform(0, 10, 50))
    knots = np.sort(rng.uniform(0, 10, 8))
    
    identity = np.eye(len(knots))
    cs = CubicSpline(knots, identity, bc_type='natural')
    
    # 0th order
    cpp_0 = compute_natural_spline_basis(x, knots, extrapolate_linear=False, derivative_order=0)
    scipy_0 = cs(x)
    np.testing.assert_allclose(cpp_0, scipy_0, atol=1e-8)
    
    # 1st order
    cpp_1 = compute_natural_spline_basis(x, knots, extrapolate_linear=False, derivative_order=1)
    scipy_1 = cs(x, nu=1)
    np.testing.assert_allclose(cpp_1, scipy_1, atol=1e-8)
    
    # 2nd order
    cpp_2 = compute_natural_spline_basis(x, knots, extrapolate_linear=False, derivative_order=2)
    scipy_2 = cs(x, nu=2)
    np.testing.assert_allclose(cpp_2, scipy_2, atol=1e-8)

def test_cpp_basis_derivatives_extrapolation():
    """
    Test the extrapolation behavior of the C++ basis derivatives.

    For linear extrapolation:
    - The 1st derivative should be constant outside the knot boundaries.
    - The 2nd derivative should be zero outside the knot boundaries.

    This test verifies both of these conditions.
    """
    # Linear extrapolation means:
    # 1st derivative constant outside
    # 2nd derivative zero outside
    
    knots = np.array([0., 1., 2.])
    x = np.array([-1.0, 3.0])
    
    # 1st order
    cpp_1 = compute_natural_spline_basis(x, knots, extrapolate_linear=True, derivative_order=1)
    
    # Check constancy by comparing with boundary values
    # Left boundary
    x_left = np.array([0.0])
    cpp_1_left = compute_natural_spline_basis(x_left, knots, extrapolate_linear=True, derivative_order=1)
    np.testing.assert_allclose(cpp_1[0], cpp_1_left[0], atol=1e-8)
    
    # Right boundary
    x_right = np.array([2.0])
    cpp_1_right = compute_natural_spline_basis(x_right, knots, extrapolate_linear=True, derivative_order=1)
    np.testing.assert_allclose(cpp_1[1], cpp_1_right[0], atol=1e-8)
    
    # 2nd order
    cpp_2 = compute_natural_spline_basis(x, knots, extrapolate_linear=True, derivative_order=2)
    np.testing.assert_allclose(cpp_2, 0.0, atol=1e-8)
