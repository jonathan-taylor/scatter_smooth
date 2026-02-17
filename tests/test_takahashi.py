"""
Tests for the Takahashi algorithm for trace computation.

This module contains tests for the Takahashi algorithm, which is a fast
method for computing the trace of the product of the inverse of a banded
matrix and another banded matrix. This is particularly useful for calculating
the effective degrees of freedom in a smoothing spline.
"""
import numpy as np
import pytest
from scipy.linalg import cholesky_banded, inv
from .takahashi_trace import (takahashi_upper,
                              trace_product_banded)
from scatter_smooth._scatter_smooth_extension import trace_takahashi as trace_takahashi_cpp

def band_to_dense(ab, w, N):
    """
    Helper to expand banded storage to a dense matrix for verification.

    Args:
        ab (np.ndarray): The banded matrix in scipy's upper-banded format.
        w (int): The number of super-diagonals (bandwidth).
        N (int): The size of the matrix.

    Returns:
        np.ndarray: The full dense matrix.
    """
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(max(0, i-w), min(N, i+w+1)):
            if j >= i:
                val = ab[w - (j-i), j]
            else:
                val = ab[w - (i-j), i]
            d[i, j] = val
    return d

def test_takahashi_trace():
    """
    Verify the fast banded Takahashi method against a slow dense method.

    This test performs the following steps:
    1.  Generates two random symmetric positive-definite banded matrices, A and B.
    2.  Computes `trace((A+B)^-1 @ B)` using a fast method:
        a.  Computes C = A + B in banded form.
        b.  Performs a banded Cholesky decomposition of C.
        c.  Uses the Takahashi algorithm to find the bands of C^-1.
        d.  Computes the trace of the product of the two banded matrices.
    3.  Computes the same trace using a slow, dense matrix approach for reference.
    4.  Compares the results from the fast Python, fast C++, and slow dense
        methods, asserting that they are all-close.
    """
    # 1. Setup Parameters
    N = 100       # Matrix size (reduced for test speed)
    w = 5         # Bandwidth (e.g., Pentadiagonal)
    np.random.seed(42)

    # 2. Generate Random Banded Matrices A and B
    # Create random bands in Upper format
    A_band = np.random.rand(w + 1, N)
    B_band = np.random.rand(w + 1, N)

    # Make A strictly diagonally dominant to ensure it is Positive Definite
    # (The diagonal is the last row in upper format)
    A_band[-1, :] += 5.0
    # B is just a weight matrix, can be anything, but let's make it symmetric (implied by storage)

    # ---------------------------------------------------------
    # FAST METHOD (The approach to be implemented in C++)
    # ---------------------------------------------------------

    # Compute C = A + B
    C_band = A_band + B_band

    # Cholesky of C (Upper form)
    # lower=False returns U where C = U.T @ U
    U_factor = cholesky_banded(C_band, lower=False)

    # Takahashi Inverse (Get bands of C^-1)
    Z_band = takahashi_upper(U_factor)

    # Trace(Z * B)
    trace_fast = trace_product_banded(Z_band, B_band)
    trace_cpp = trace_takahashi_cpp(U_factor, B_band)
    # ---------------------------------------------------------
    # SLOW REFERENCE METHOD (Dense)
    # ---------------------------------------------------------

    A_dense = band_to_dense(A_band, w, N)
    B_dense = band_to_dense(B_band, w, N)
    C_dense = A_dense + B_dense

    # Compute Exact Inverse
    C_inv_dense = inv(C_dense)

    # Compute Exact Trace: tr(C^-1 * B)
    trace_dense = np.trace(C_inv_dense @ B_dense)

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------
    
    diff1 = abs(trace_fast - trace_dense)
    diff2 = abs(trace_fast - trace_cpp)
    # Check for match
    assert diff1 < 1e-9, f"Mismatch detected! Fast: {trace_fast}, Slow: {trace_dense}, Diff: {diff1}"
    assert diff2 < 1e-9, f"Mismatch detected! Fast: {trace_fast}, Cpp: {trace_cpp}, Diff: {diff2}"
