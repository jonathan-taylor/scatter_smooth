"""
Tests for the LOWESS smoother.

This module contains tests for the `LoessSmoother`, comparing its
implementation against a naive Python version to ensure consistency. It
includes tests for:
- Basic fitting and prediction with different spans and polynomial degrees.
- The use of observation weights.
- Prediction on single scalar values.
"""
import numpy as np
import pytest
from scatter_smooth.loess import LoessSmoother
from .loess import LoessSmoother as LoessSmootherNaive

def test_lowess_consistency():
    """
    Test that C++ and Python implementations produce consistent results.
    """
    rng = np.random.default_rng(42)
    n = 100
    x = np.sort(rng.uniform(0, 10, n))
    y = np.sin(x) + rng.normal(0, 0.2, n)
    
    # Test for different spans and degrees
    for span in [0.3, 0.7]:
        for degree in [1, 2]:
            py_fitter = LoessSmootherNaive(x, span=span, degree=degree)
            cpp_fitter = LoessSmoother(x, span=span, degree=degree)
            
            py_fitter.smooth(y)
            cpp_fitter.smooth(y)
            
            x_new = np.linspace(0, 10, 50)
            
            y_py = py_fitter.predict(x_new)
            y_cpp = cpp_fitter.predict(x_new)
            
            # Allow small numerical differences
            np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-5, 
                                       err_msg=f"Mismatch for span={span}, degree={degree}")

def test_lowess_weights():
    """
    Test with observation weights.
    """
    rng = np.random.default_rng(123)
    n = 50
    x = np.sort(rng.uniform(0, 10, n))
    y = x * 0.5 + rng.normal(0, 0.5, n)
    w = rng.uniform(0.1, 2.0, n)
    
    py_fitter = LoessSmootherNaive(x, w=w, span=0.5, degree=1)
    cpp_fitter = LoessSmoother(x, w=w, span=0.5, degree=1)
    
    py_fitter.smooth(y)
    cpp_fitter.smooth(y)
    
    x_new = np.linspace(0, 10, 20)
    
    y_py = py_fitter.predict(x_new)
    y_cpp = cpp_fitter.predict(x_new)
    
    np.testing.assert_allclose(y_cpp, y_py, rtol=1e-5, atol=1e-5)

def test_single_prediction():
    """
    Test prediction on a single point (scalar vs array).
    """
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 20)
    y = np.sin(x)
    
    fitter = LoessSmoother(x, span=0.5)
    fitter.smooth(y)
    
    val = 5.0
    pred = fitter.predict([val])
    assert pred.shape == (1,)
    assert not np.isnan(pred[0])

if __name__ == "__main__":
    pytest.main([__file__])
