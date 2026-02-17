"""
Tests for the B-spline basis and penalty matrices.

This module verifies the correctness of the B-spline basis matrix (N), the
weighted crossproduct matrix (N^T W N), and the penalty matrix (Omega) by
comparing them against a reference implementation using `scipy.interpolate.BSpline`.
It also checks the evaluation of basis functions and their derivatives.
"""
import numpy as np
import pytest
from scipy.interpolate import BSpline
from scatter_smooth import SplineSmoother

def compute_ref_matrices(x, knots, order=4, weights=None):
    """
    Compute reference B-spline matrices using scipy.

    This function constructs the B-spline basis matrix `N` and the penalty
    matrix `Omega` using `scipy.interpolate.BSpline` as a ground truth.

    Args:
        x (np.ndarray): The x-coordinates of the data points.
        knots (np.ndarray): The interior knots of the spline.
        order (int, optional): The order of the spline. Defaults to 4 (cubic).
        weights (np.ndarray, optional): Weights for the data points.
            Defaults to None (uniform weights).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - NTWN: The matrix N^T @ W @ N.
            - Omega: The penalty matrix.
    """
    if weights is None:
        weights = np.ones(len(x))
    W = np.diag(weights)
    
    n_interior = len(knots)
    t = np.zeros(n_interior + 2 * (order - 1))
    t[:order] = knots[0]
    if n_interior > 2:
        t[order:order + n_interior - 2] = knots[1:-1]
    t[-order:] = knots[-1]
    
    n_basis = len(t) - order
    
    N = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        spl = BSpline(t, c, order - 1)
        N[:, i] = spl(x)
        
    NTWN = N.T @ W @ N
    
    Omega = np.zeros((n_basis, n_basis))
    unique_t = np.unique(t)
    
    for k in range(len(unique_t) - 1):
        t_start = unique_t[k]
        t_end = unique_t[k+1]
        mid = 0.5 * (t_start + t_end)
        half_len = 0.5 * (t_end - t_start)
        
        nodes = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
        gl_x = mid + half_len * nodes
        gl_w = half_len * np.array([1.0, 1.0])
        
        vals_d2 = np.zeros((2, n_basis))
        for i in range(n_basis):
            c = np.zeros(n_basis)
            c[i] = 1.0
            spl = BSpline(t, c, order - 1)
            vals_d2[:, i] = spl(gl_x, nu=2)
            
        for q in range(2):
            w_q = gl_w[q]
            row = vals_d2[q, :]
            Omega += w_q * np.outer(row, row)
            
    return NTWN, Omega

@pytest.mark.parametrize("n_samples", [10, 20])
@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("weighted", [False, True])
def test_matrices(n_samples, uniform, weighted):
    """
    Test that the C++ matrices match the scipy reference implementation.

    This test covers various scenarios:
    - Different numbers of samples.
    - Uniform and non-uniform knots.
    - Weighted and unweighted data.

    It checks:
    - The knot vector used in the C++ implementation.
    - The evaluation of the basis functions and their first two derivatives.
    - The NTWN and Omega matrices.
    """
    np.random.seed(123)
    if uniform:
        x = np.linspace(0, 1, n_samples)
        knots = np.linspace(0, 1, 5) 
    else:
        x = np.sort(np.random.rand(n_samples))
        knots = np.sort(np.concatenate(([0, 1], np.random.rand(3))))
        
    if weighted:
        w = np.random.uniform(0.5, 1.5, n_samples)
    else:
        w = None
        
    sf = SplineSmoother(x, w=w, knots=knots, engine='bspline')
    cpp_NTWN = sf._cpp_fitter.get_NTWN()
    cpp_Omega = sf._cpp_fitter.get_Omega()
    cpp_knots = sf._cpp_fitter.get_knots()
    
    # Use scaled values for reference
    x_scaled = (x - sf.x_min_) / sf.x_scale_
    knots_scaled = sf.knots_scaled_
    
    ref_NTWN, ref_Omega = compute_ref_matrices(x_scaled, knots_scaled, weights=w)
    
    # Check knots
    n_interior = len(knots_scaled)
    order = 4
    t = np.zeros(n_interior + 2 * (order - 1))
    t[:order] = knots_scaled[0]
    if n_interior > 2:
        t[order:order + n_interior - 2] = knots_scaled[1:-1]
    t[-order:] = knots_scaled[-1]
    
    np.testing.assert_allclose(cpp_knots, t, atol=1e-12, err_msg="Knots mismatch")
    
    # Check basis functions and derivatives
    # Pick a few points in the domain (avoiding knots to avoid C0 vs limit issues for now)
    test_x = np.linspace(t[order-1] + 1e-5, t[-order] - 1e-5, 10)
    
    for val in test_x:
        # Check value (deriv=0)
        cpp_val = sf._cpp_fitter.eval_basis(val, 0)
        ref_val = np.zeros(len(cpp_val))
        for i in range(len(ref_val)):
            c = np.zeros(len(ref_val))
            c[i] = 1.0
            spl = BSpline(t, c, order - 1)
            ref_val[i] = spl(val)
        np.testing.assert_allclose(cpp_val, ref_val, atol=1e-10, err_msg=f"Basis value mismatch at {val}")
        
        # Check 1st deriv
        cpp_d1 = sf._cpp_fitter.eval_basis(val, 1)
        ref_d1 = np.zeros(len(cpp_d1))
        for i in range(len(ref_d1)):
            c = np.zeros(len(ref_d1))
            c[i] = 1.0
            spl = BSpline(t, c, order - 1)
            ref_d1[i] = spl(val, nu=1)
        np.testing.assert_allclose(cpp_d1, ref_d1, atol=1e-10, err_msg=f"Basis d1 mismatch at {val}")
        
        # Check 2nd deriv
        cpp_d2 = sf._cpp_fitter.eval_basis(val, 2)
        ref_d2 = np.zeros(len(cpp_d2))
        for i in range(len(ref_d2)):
            c = np.zeros(len(ref_d2))
            c[i] = 1.0
            spl = BSpline(t, c, order - 1)
            ref_d2[i] = spl(val, nu=2)
        np.testing.assert_allclose(cpp_d2, ref_d2, atol=1e-10, err_msg=f"Basis d2 mismatch at {val}")

    np.testing.assert_allclose(cpp_NTWN, ref_NTWN, atol=1e-10, rtol=1e-10, err_msg="NTWN mismatch")
    np.testing.assert_allclose(cpp_Omega, ref_Omega, atol=1e-10, rtol=1e-10, err_msg="Omega mismatch")

if __name__ == "__main__":
    test_matrices(20, False, True)
