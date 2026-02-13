from dataclasses import dataclass, field
from scipy.interpolate import CubicSpline
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import time
from scipy.optimize import bisect, brentq

@dataclass
class SplineFitter:
    """
    Internal class to handle the fitting logic for the smoothing spline.
    ...
    """
    x: np.ndarray
    w: np.ndarray = None
    lamval: float = None
    df: int = None
    knots: np.ndarray = None
    n_knots: int = None
    
    _cpp_fitter: object = field(init=False, default=None, repr=False)

    def _prepare_matrices(self):
        """
        Compute the scaled matrices required for both
        fitting and EDF calculation of the Natural Spline Ridge.
        """
        x = self.x
        weights = self.w
        knots = self.knots
        n_knots = self.n_knots
        
        n = len(x)
        if weights is None: weights = np.ones(n)
        
        if knots is None:
            if n_knots is not None:
                percs = np.linspace(0, 100, n_knots)
                knots = np.percentile(x, percs)
            else:
                knots = np.sort(np.unique(x))
        else:
            knots = np.asarray(knots)
            knots.sort()
            
        self.knots = knots
        n_k = len(knots)
        self.n_k_ = n_k

        # --- Standardization / Scaling ---
        x_min, x_max = x.min(), x.max()
        scale = x_max - x_min if x_max > x_min else 1.0
        self.x_min_ = x_min
        self.x_scale_ = scale

        x_scaled = (x - x_min) / scale
        knots_scaled = (knots - x_min) / scale
        self.knots_scaled_ = knots_scaled

        try:
            from smoothing_spline._spline_extension import SplineFitterCpp
            # C++ implementation available
            # Note: We pass SCALED values to C++ to match the Python logic's scaling behavior
            # effectively treating the C++ solver as working in the scaled domain.
            self._cpp_fitter = SplineFitterCpp(x_scaled, knots_scaled, self.w)
            
            # These are needed for introspection / debugging if accessed
            # We fetch them lazily or just pre-fetch if we want full compat
            self.N_ = self._cpp_fitter.get_N()
            self.Omega_ = self._cpp_fitter.get_Omega()
            
            # Reconstruct NTW_ for compatibility if needed (C++ stores it internally)
            if self.w is not None:
                self.NTW_ = self.N_.T * self.w
            else:
                self.NTW_ = self.N_.T

            return
            
        except ImportError:
            self._cpp_fitter = None

        # Fallback to Python Implementation
        # --- Design Matrix N ---
        cs_basis = CubicSpline(knots_scaled, np.eye(n_k), bc_type='natural')
        N_mat = cs_basis(x_scaled)

        # Apply linear extrapolation to the basis matrix for points outside the boundaries
        mask_lo = x_scaled < knots_scaled[0]
        mask_hi = x_scaled > knots_scaled[-1]
        
        if np.any(mask_lo) or np.any(mask_hi):
            cs_basis_d1 = cs_basis.derivative(nu=1)
            d1_left = cs_basis_d1(knots_scaled[0])
            d1_right = cs_basis_d1(knots_scaled[-1])
            
            vals_at_left_boundary = np.zeros(n_k); vals_at_left_boundary[0] = 1
            vals_at_right_boundary = np.zeros(n_k); vals_at_right_boundary[-1] = 1

            if np.any(mask_lo):
                x_lo = x_scaled[mask_lo]
                N_mat[mask_lo, :] = vals_at_left_boundary[None, :] + (x_lo - knots_scaled[0])[:, None] * d1_left[None, :]
            
            if np.any(mask_hi):
                x_hi = x_scaled[mask_hi]
                N_mat[mask_hi, :] = vals_at_right_boundary[None, :] + (x_hi - knots_scaled[-1])[:, None] * d1_right[None, :]

        # --- Penalty Matrix Omega ---
        hk = np.diff(knots_scaled)
        inv_hk = 1.0 / hk
        R_k = sparse.diags([hk[1:-1]/6.0, (hk[:-1]+hk[1:])/3.0, hk[1:-1]/6.0], [-1, 0, 1], shape=(n_k-2, n_k-2))
        Q_k = sparse.diags([inv_hk[:-1], -inv_hk[:-1]-inv_hk[1:], inv_hk[1:]], [0, -1, -2], shape=(n_k, n_k-2))

        if sparse.issparse(R_k):
            R_inv_QT = splinalg.spsolve(R_k.tocsc(), Q_k.T.tocsc()).toarray()
        else:
            R_inv_QT = np.linalg.solve(R_k, Q_k.T)

        self.Omega_ = Q_k @ R_inv_QT
        self.N_ = N_mat
        self.NTW_ = self.N_.T * weights

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-12, 12)):
        """
        Finds the exact lambda value that yields the target degrees of freedom.
        """
        # Validate target DF
        if target_df >= self.n_k_ - 0.01:
            raise ValueError(f"Target DF ({target_df}) too high. Max is roughly {self.n_k_}.")
        if target_df <= 2.01:
            raise ValueError(f"Target DF ({target_df}) too low. Min is 2 (linear).")

        def df_error_func(log_lam_scaled):
            lam_scaled = 10 ** log_lam_scaled
            if self._cpp_fitter:
                current_df = self._cpp_fitter.compute_df(lam_scaled)
            else:
                LHS = XTWX + lam_scaled * self.Omega_
                S_matrix = np.linalg.solve(LHS, XTWX)
                current_df = np.trace(S_matrix)
            return current_df - target_df

        if not self._cpp_fitter:
             XTWX = self.NTW_ @ self.N_
        
        try:
            log_lam_scaled_opt = brentq(df_error_func, log10_lam_bounds[0], log10_lam_bounds[1])
        except ValueError as e:
            raise RuntimeError(
                "Could not find root in the given bounds. This usually means "
                "the target DF is effectively unreachable or bounds need expanding."
            ) from e

        lam_scaled_opt = 10 ** log_lam_scaled_opt
        return lam_scaled_opt * (self.x_scale_ ** 3)

    def __post_init__(self):
        self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        elif self.lamval is None:
            self.lamval = 0.0

    def fit(self, y, sample_weight=None):
        """
        Fit the smoothing spline.
        """
        self.y = y
        
        # Handle weights: update if provided, otherwise use self.w
        if sample_weight is not None:
            self.w = sample_weight
            if self._cpp_fitter:
                # If C++ fitter exists, use efficient update
                # Assuming SplineFitterCpp/ReinschCpp has update_weights method (implemented in C++)
                if hasattr(self._cpp_fitter, "update_weights"):
                     self._cpp_fitter.update_weights(self.w)
                else:
                     # Fallback if method missing (shouldn't happen with updated extension)
                     self._prepare_matrices() 
            else:
                self._prepare_matrices()
        
        # Ensure lamval is not None
        if self.lamval is None:
             self.lamval = 0.0
             
        lam_scaled = self.lamval / self.x_scale_**3

        if self._cpp_fitter:
            self.alpha_ = self._cpp_fitter.fit(y, lam_scaled)
        else:
            LHS = self.NTW_ @ self.N_ + lam_scaled * self.Omega_
            RHS = self.NTW_ @ self.y
            self.alpha_ = np.linalg.solve(LHS, RHS)
        
        # Store final spline
        self.spline_ = CubicSpline(self.knots_scaled_, self.alpha_, bc_type='natural')

        # Fit linear component to the predictions
        y_hat = self.predict(self.x)
        if self.w is not None:
            X = np.vander(self.x, 2)
            Xw = X * self.w[:, None]
            yw = y_hat * self.w
            beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        else:
            beta = np.polyfit(self.x, y_hat, 1)

        self.intercept_ = beta[1]
        self.coef_ = beta[0]
        
    def update_weights(self, w):
        """
        Update the weights and refit the model.
        Parameters
        ----------
        w : np.ndarray
            New weights for the observations.
        """
        self.w = w
        self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        if hasattr(self, 'y'):
            self.fit(self.y)

    def predict(self, x):
        """
        Predict the response for a new set of predictor variables.
        Parameters
        ----------
        x : np.ndarray
            The predictor variables.
        Returns
        -------
        np.ndarray
            The predicted response.
        """
        x_scaled = (x - self.x_min_) / self.x_scale_
        
        y_pred = np.zeros_like(x_scaled, dtype=float)
        
        mask_in = (x_scaled >= self.knots_scaled_[0]) & (x_scaled <= self.knots_scaled_[-1])
        mask_lo = x_scaled < self.knots_scaled_[0]
        mask_hi = x_scaled > self.knots_scaled_[-1]

        y_pred[mask_in] = self.spline_(x_scaled[mask_in])

        # Linear extrapolation for points outside the knots
        if np.any(mask_lo):
            deriv = self.spline_.derivative(1)(self.knots_scaled_[0])
            y_pred[mask_lo] = self.alpha_[0] + (x_scaled[mask_lo] - self.knots_scaled_[0]) * deriv
        
        if np.any(mask_hi):
            deriv = self.spline_.derivative(1)(self.knots_scaled_[-1])
            y_pred[mask_hi] = self.alpha_[-1] + (x_scaled[mask_hi] - self.knots_scaled_[-1]) * deriv

        return y_pred

    @property
    def nonlinear_(self):
        """
        The non-linear component of the fitted spline.
        """
        linear_part = self.coef_ * self.x + self.intercept_
        return self.predict(self.x) - linear_part



def compute_edf_reinsch(x, lamval, weights=None):
    x = np.array(x, dtype=float)
    n = len(x)
    if weights is None:
        w_inv_vec = np.ones(n)
    else:
        weights = np.array(weights, dtype=float)
        w_inv_vec = 1.0 / (weights + 1e-12)
    W_inv = sparse.diags(w_inv_vec)
    h = np.diff(x)
    inv_h = 1.0 / h
    main_diag_R = (h[:-1] + h[1:]) / 3.0
    off_diag_R = h[1:-1] / 6.0
    R = sparse.diags([off_diag_R, main_diag_R, off_diag_R],
                     [-1, 0, 1], shape=(n-2, n-2))
    d0 = inv_h[:-1]
    d1 = -inv_h[:-1] - inv_h[1:]
    d2 = inv_h[1:]
    Q = sparse.diags([d0, d1, d2], [0, -1, -2], shape=(n, n-2))
    B = Q.T @ W_inv @ Q
    M_trace = R + lamval * B
    M_trace = M_trace.tocsc()
    B = B.tocsc()
    X = splinalg.spsolve(M_trace, B)
    if sparse.issparse(X):
        tr_val = X.diagonal().sum()
    else:
        tr_val = np.trace(X)
    return n - lamval * tr_val

def estimate_edf_hutchinson(x, lamval, weights=None, n_vectors=50):
    x = np.array(x, dtype=float)
    n = len(x)
    if weights is None: w_inv_vec = np.ones(n)
    else: w_inv_vec = 1.0 / (np.array(weights, dtype=float) + 1e-12)
    W_inv = sparse.diags(w_inv_vec)
    h = np.diff(x)
    inv_h = 1.0 / h
    R = sparse.diags([h[1:-1]/6.0, (h[:-1]+h[1:])/3.0, h[1:-1]/6.0], [-1, 0, 1], shape=(n-2, n-2))
    Q = sparse.diags([inv_h[:-1], -inv_h[:-1]-inv_hk[1:], inv_hk[1:]], [0, -1, -2], shape=(n, n-2))
    B = Q.T @ W_inv @ Q
    M_trace = R + lamval * B
    M_trace = M_trace.tocsc()
    solve_M = splinalg.factorized(M_trace)
    trace_est = 0.0
    dim = n - 2
    for i in range(n_vectors):
        z = np.random.choice([-1, 1], size=dim)
        v = B @ z
        u = solve_M(v)
        trace_est += np.dot(z, u)
    mean_trace = trace_est / n_vectors
    return n - lamval * mean_trace
