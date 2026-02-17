"""
Tests for the C++ penalty matrix implementation.

This module verifies the correctness of the C++ function
`compute_penalty_matrix` by comparing its output to a reference
implementation based on the definitions from Friedman et al. (ESL).
"""
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import linalg as splinalg

from scatter_smooth._scatter_smooth_extension import compute_penalty_matrix

def test_cpp_penalty_matches_scipy():
    """
    Verify that the C++ penalty matrix matches the reference implementation.

    This test constructs the natural spline penalty matrix Omega using both the
    C++ function and a Python implementation based on the sparse matrix
    definitions from "The Elements of Statistical Learning". It then asserts
    that the resulting matrices are all-close.
    """
    rng = np.random.default_rng(55)
    knots = np.sort(rng.uniform(0, 10, 15))
    
    # Python Implementation
    n_k = len(knots)
    hk = np.diff(knots)
    inv_hk = 1.0 / hk
    R_k = sparse.diags([hk[1:-1]/6.0, (hk[:-1]+hk[1:])/3.0, hk[1:-1]/6.0], [-1, 0, 1], shape=(n_k-2, n_k-2))
    Q_k = sparse.diags([inv_hk[:-1], -inv_hk[:-1]-inv_hk[1:], inv_hk[1:]], [0, -1, -2], shape=(n_k, n_k-2))

    if sparse.issparse(R_k):
        R_inv_QT = splinalg.spsolve(R_k.tocsc(), Q_k.T.tocsc()).toarray()
    else:
        R_inv_QT = np.linalg.solve(R_k, Q_k.T)

    expected_Omega = Q_k @ R_inv_QT
    
    # C++ Implementation
    cpp_Omega = compute_penalty_matrix(knots)
    
    np.testing.assert_allclose(cpp_Omega, expected_Omega, atol=1e-8, rtol=1e-8)
