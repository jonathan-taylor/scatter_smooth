# -*- coding: utf-8 -*-
"""
Takahashi Trace Calculation

This module implements Tr((A+B)^{-1}B) where A and B are both banded.

"""

import numpy as np
from scipy.linalg import cholesky_banded

def takahashi_upper(U_banded):
    """
    Using Takahashi's algorithm, computes the upper band of the inverse of a symmetric matrix A,
    given its Cholesky factor U (where A = U.T @ U).

    Parameters:
    -----------
    U_banded : np.ndarray
        The upper Cholesky factor in SciPy upper-banded format.
        Shape (w+1, N). Row 0 is the furthest super-diagonal.
        Last row is the main diagonal.

    Returns:
    --------
    Z_banded : np.ndarray
        The upper band of A^-1 in the same format.
    """
    w = U_banded.shape[0] - 1  # Bandwidth
    N = U_banded.shape[1]

    Z = np.zeros_like(U_banded)

    # Helper to index into upper banded format
    # Corresponds to matrix element A[i, j] where j >= i
    def get(matrix, i, j):
        if j < i: i, j = j, i # Symmetry logic
        if (j - i) > w: return 0.0
        # SciPy Upper Format:
        # Row index is w - (j - i)
        # Column index is j
        return matrix[w - (j - i), j]

    def set_val(matrix, i, j, val):
        if j < i: i, j = j, i # Symmetry logic
        if (j - i) <= w:
            matrix[w - (j - i), j] = val

    # 1. Initialize corner (N-1, N-1)
    # The recurrence starts from the bottom-right
    u_nn = get(U_banded, N-1, N-1)
    set_val(Z, N-1, N-1, 1.0 / (u_nn**2))

    # 2. Backward Recurrence
    for i in range(N - 2, -1, -1):
        u_ii = get(U_banded, i, i)
        j_max = min(i + w, N - 1)

        # --- Calculate Off-Diagonals Z_ij for i < j ---
        # We iterate backwards from the edge of the band to i+1
        for j in range(j_max, i, -1):
            sum_val = 0.0
            # Inner product: sum(U_ik * Z_kj) for k > i
            for k in range(i + 1, j_max + 1):
                u_ik = get(U_banded, i, k)
                z_kj = get(Z, k, j) # Z is symmetric
                sum_val += u_ik * z_kj

            # Z_ij = - (1/U_ii) * sum(...)
            set_val(Z, i, j, -sum_val / u_ii)

        # --- Calculate Diagonal Z_ii ---
        sum_diag = 0.0
        for k in range(i + 1, j_max + 1):
            u_ik = get(U_banded, i, k)
            z_ik = get(Z, i, k)
            sum_diag += u_ik * z_ik

        # Z_ii = (1/U_ii) * [ (1/U_ii) - sum(...) ]
        z_ii = (1.0 / u_ii) * ((1.0 / u_ii) - sum_diag)
        set_val(Z, i, i, z_ii)

    return Z

def trace_product_banded(Z_banded, B_banded):
    """
    Computes tr(Z * B) where both are symmetric and stored in upper banded format.
    Formula: sum(Z_ij * B_ji) over all i,j.
    Since symmetric: sum(Z_ii * B_ii) + 2 * sum(Z_ij * B_ij for j > i)
    """
    w = Z_banded.shape[0] - 1
    N = Z_banded.shape[1]
    trace = 0.0

    for i in range(N):
        # 1. Diagonal contribution
        # In upper format, diagonal is the last row (-1)
        z_ii = Z_banded[-1, i]
        b_ii = B_banded[-1, i]
        trace += z_ii * b_ii

        # 2. Off-diagonal contribution within the band
        # We iterate up the column 'i' in the storage
        # In upper format, the column i contains elements (i-w, i) ... (i, i)
        # But for the trace sum, it's easier to iterate relative to diagonal:
        # Elements (i, i+1), (i, i+2) ...

        j_max = min(i + w, N - 1)
        for j in range(i + 1, j_max + 1):
            # Access element (i, j)
            # Row = w - (j - i)
            # Col = j
            z_val = Z_banded[w - (j - i), j]
            b_val = B_banded[w - (j - i), j]
            trace += 2.0 * z_val * b_val

    return trace

def trace_ratio_banded(A_band, B_band):
    """
    Compute Tr((A+B)^{-1}B)

    """

    # Compute C = A + B
    C_band = A_band + B_band
    
    # Cholesky of C (Upper form)
    # lower=False returns U where C = U.T @ U
    U_factor = cholesky_banded(C_band, lower=False)
    
    # Takahashi Inverse (Get bands of C^-1)
    Z_band = takahashi_upper(U_factor)
    
    # Trace(Z * B)
    return trace_product_banded(Z_band, B_band)

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
    print("--- Starting Takahashi Trace Verification ---")

    # 1. Setup Parameters
    N = 500       # Matrix size
    w = 5         # Bandwidth (e.g., Pentadiagonal)
    np.random.seed(42)

    # 2. Generate Random Banded Matrices A and B
    print(f"Generating random {N}x{N} matrices with bandwidth {w}...")

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
    print("Running Fast Banded Method...")

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
    print("Running Slow Dense Method (Reference)...")

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
    print(f"\nFast Trace: {trace_fast:.10f}")
    print(f"Slow Trace: {trace_dense:.10f}")

    diff = abs(trace_fast - trace_dense)
    print(f"Difference: {diff:.2e}")

    if diff < 1e-9:
        print("\nSUCCESS: The Takahashi implementation matches the dense reference.")
    else:
        print("\nFAILURE: Mismatch detected!")

if __name__ == "__main__":
    test_takahashi_trace()
