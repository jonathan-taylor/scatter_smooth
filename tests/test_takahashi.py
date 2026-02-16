import numpy as np
import pytest
from scipy.linalg import cholesky_banded, inv
from smoothing_spline.takahashi_trace import takahashi_upper, trace_product_banded

def band_to_dense(ab, w, N):
    """Helper to expand banded storage to dense matrix for verification."""
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
    Verification Script:
    Compares Fast Banded method vs. Slow Dense method
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
    
    diff = abs(trace_fast - trace_dense)
    
    # Check for match
    assert diff < 1e-9, f"Mismatch detected! Fast: {trace_fast}, Slow: {trace_dense}, Diff: {diff}"
