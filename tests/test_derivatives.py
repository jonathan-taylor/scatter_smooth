"""
Tests for spline derivatives.

This module contains tests for the derivatives of the fitted spline.
It verifies that the predicted derivatives of a known function (a simple
quadratic) are close to the analytical derivatives.
"""
import numpy as np
import pytest
from scatter_smooth import SplineSmoother

@pytest.mark.xfail(reason="Derivative calculation is not accurate enough at the boundaries.")
def test_derivatives():
    """
    Test the derivatives of the spline fitter.
    """
    # Create a simple quadratic function
    x = np.linspace(-5, 5, 100)
    y = x**2
    
    # Fit a spline to the function
    fitter = SplineSmoother(x, n_knots=20)
    fitter.lamval = 1e-8 # Use a small lambda to closely follow the function
    fitter.smooth(y)
    
    # Check the first derivative
    d1_true = 2 * x
    d1_pred = fitter.predict(x, deriv=1)
    np.testing.assert_allclose(d1_pred, d1_true, atol=1e-2, rtol=1e-2)
    
    # Check the second derivative
    d2_true = np.full_like(x, 2.0)
    d2_pred = fitter.predict(x, deriv=2)
    np.testing.assert_allclose(d2_pred, d2_true, atol=1e-2, rtol=1e-2)

    # Check the third derivative (should be close to zero)
    d3_true = np.zeros_like(x)
    d3_pred = fitter.predict(x, deriv=3)
    np.testing.assert_allclose(d3_pred, d3_true, atol=1e-2, rtol=1e-2)

if __name__ == '__main__':
    test_derivatives()
