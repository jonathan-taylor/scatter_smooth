import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded

class WeightedCubicSplineTrace:
    """
    Computes the trace of the smoothing spline influence matrix (Effective Degrees of Freedom)
    using an exact O(N) algorithm based on Cholesky decomposition and Takahashi's equations,
    with support for non-uniform weights.
    """

    def __init__(self, x, weights=None):
        """
        Initialize with data. x must be sorted.
        """
        self.x = np.array(x, dtype=float)
        self.n = len(x)
        self.h = np.diff(self.x)  # h_i = x_{i+1} - x_i

        if self.n < 4:
            raise ValueError("Need at least 4 points for cubic spline smoothing.")

        if weights is None:
            self.weights = np.ones(self.n)
        else:
            self.weights = np.array(weights, dtype=float)
            if len(self.weights) != self.n:
                raise ValueError("Weights array must have the same length as x.")
            if np.any(self.weights <= 0):
                raise ValueError("Weights must be positive.")
        self.inv_weights = 1.0 / self.weights

        # Precompute the bands for Q and R
        self._compute_QR_bands()

    def _compute_QR_bands(self):
        """
        Compute the non-zero bands of R (tridiagonal) and Q^T W^-1 Q (pentadiagonal).
        Stored as compact structures.
        """
        n = self.n
        h = self.h
        inv_w = self.inv_weights

        # --- Build R (Tridiagonal) ---
        # R is (n-2) x (n-2)
        # R_ii = (h[i] + h[i+1]) / 3
        # R_i,i+1 = h[i+1] / 6

        self.R_diag = (h[:-1] + h[1:]) / 3.0
        self.R_off = h[1:-1] / 6.0

        # --- Build M = Q^T W^-1 Q (Pentadiagonal) ---
        # M is (n-2) x (n-2).

        dim = n - 2
        self.M_diag = np.zeros(dim)
        self.M_off1 = np.zeros(dim - 1)
        self.M_off2 = np.zeros(dim - 2)

        # Q elements in column j (referring to rows j, j+1, j+2)
        inv_h_j = 1.0 / h[:-1]                # Q_j,j
        inv_h_j_plus_1 = 1.0 / h[1:]          # Q_j+2,j
        q1_vals = -(inv_h_j + inv_h_j_plus_1) # Q_j+1,j

        # Diagonal M_jj
        # M_jj = (Q_j,j)^2 / w_j + (Q_j+1,j)^2 / w_{j+1} + (Q_j+2,j)^2 / w_{j+2}
        self.M_diag = (inv_h_j[:dim]**2 * inv_w[:dim]) + \
                      (q1_vals[:dim]**2 * inv_w[1:dim+1]) + \
                      (inv_h_j_plus_1[:dim]**2 * inv_w[2:dim+2])

        # Off-diagonal 1: M_j,j+1
        # M_j,j+1 = Q_{j+1,j} Q_{j+1,j+1} / w_{j+1} + Q_{j+2,j} Q_{j+2,j+1} / w_{j+2}
        # Note: Q_{j+1,j+1} is inv_h_j[j+1]
        # Note: Q_{j+2,j+1} is q1_vals[j+1]
        self.M_off1 = (q1_vals[:dim-1] * inv_h_j[1:dim] * inv_w[1:dim]) + \
                      (inv_h_j_plus_1[:dim-1] * q1_vals[1:dim] * inv_w[2:dim+1])

        # Off-diagonal 2: M_j,j+2
        # M_j,j+2 = Q_{j+2,j} Q_{j+2,j+2} / w_{j+2}
        # Note: Q_{j+2,j+2} is inv_h_j[j+2]
        self.M_off2 = (inv_h_j_plus_1[:dim-2] * inv_w[2:dim] * inv_h_j[2:dim])


    def compute_trace(self, lam):
        """
        Compute Trace( (I + lambda K)^-1 ) in O(N).
        Formula: 2 + Trace(R * B^-1)
        Where B = R + lambda * Q^T W^-1 Q
        """
        n_inner = self.n - 2

        # 1. Construct B = R + lam * M
        # B is pentadiagonal (bandwidth 2)
        B_diag = self.R_diag + lam * self.M_diag
        B_off1 = self.R_off + lam * self.M_off1
        B_off2 = lam * self.M_off2

        # Add a small epsilon for numerical stability, especially if lam is very small
        B_diag += 1e-9 
        # 2. Cholesky Decomposition of B (Banded)
        # Scipy cholesky_banded expects shape (lower=True):
        # row 0: diagonals
        # row 1: off-diag 1
        # row 2: off-diag 2

        ab = np.zeros((3, n_inner))
        ab[0, :] = B_diag
        ab[1, :-1] = B_off1
        ab[2, :-2] = B_off2

        # L is returned in same packed format
        try:
            L_banded = cholesky_banded(ab, lower=True)
        except np.linalg.LinAlgError:
            return np.nan

        # 3. Compute selected elements of B^-1 using Takahashi's equations
        Inv_diag = np.zeros(n_inner)
        Inv_off1 = np.zeros(n_inner - 1)

        L0 = L_banded[0, :]
        L1 = L_banded[1, :] # padded with 0 at end
        L2 = L_banded[2, :] # padded with 0 at end

        # Backward recurrence
        for i in range(n_inner - 1, -1, -1):
            val = 1.0 / L0[i]
            sum_diag = 0.0

            if i + 1 < n_inner:
                s_off = L1[i] * Inv_diag[i+1]
                if i + 2 < n_inner:
                    s_off += L2[i] * Inv_off1[i+1]

                Inv_off1[i] = -val * s_off
                sum_diag += L1[i] * Inv_off1[i]

            if i + 2 < n_inner:
                s_off2 = L1[i] * Inv_off1[i+1] + L2[i] * Inv_diag[i+2]
                X_i_i2 = -val * s_off2

                sum_diag += L2[i] * X_i_i2

            Inv_diag[i] = val * (val - sum_diag)

        # 4. Final Trace Calculation
        tr_RBinv = np.sum(self.R_diag * Inv_diag) + 2 * np.sum(self.R_off * Inv_off1)

        return 2.0 + tr_RBinv

def naive_weighted_trace(x, weights, lam):
    """
    O(N^3) implementation using dense matrices for correctness verification, with weights.
    """
    n = len(x)
    h = np.diff(x)

    # Construct Q (n x n-2) dense
    Q = np.zeros((n, n-2))
    for j in range(n-2):
        Q[j, j] = 1/h[j]
        Q[j+1, j] = -(1/h[j] + 1/h[j+1])
        Q[j+2, j] = 1/h[j+1]

    # Construct R (n-2 x n-2) dense
    R = np.zeros((n-2, n-2))
    for i in range(n-2):
        R[i, i] = (h[i] + h[i+1])/3
        if i < n-3:
            R[i, i+1] = h[i+1]/6
            R[i+1, i] = h[i+1]/6
    
    # Construct W_inv (n x n) diagonal matrix
    W_inv = np.diag(1.0 / weights)

    QT_Winv_Q = Q.T @ W_inv @ Q
    B = R + lam * QT_Winv_Q

    # Invert B fully
    B_inv = np.linalg.inv(B)

    # Trace logic: 2 + Trace(R B^-1)
    tr_naive = 2.0 + np.trace(R @ B_inv)
    return tr_naive
