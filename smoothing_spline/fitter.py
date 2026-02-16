from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import (cholesky_banded,
                          cho_solve_banded,
                          solveh_banded)
from scipy.optimize import brentq

from ._spline_extension import (
    NaturalSplineFitter, 
    ReinschFitter, 
    BSplineFitter
)
from .takahashi_trace import trace_ratio_banded

@dataclass
class SplineFitter:
    """
    SplineFitter implementation using C++ extension for performance.
    
    Parameters
    ----------
    x : np.ndarray
        The predictor variable.
    w : np.ndarray, optional
        Weights for the observations.
    lamval : float, optional
        The smoothing parameter (lambda). If None, it will be estimated or set to 0.
    df : float, optional
        The target degrees of freedom. If provided, lambda will be solved to match this DF.
    knots : np.ndarray, optional
        The knots for the spline.
    n_knots : int, optional
        The number of knots to use if knots are not provided.
    order : int, optional
        The order of the B-spline (default is 4 for cubic B-splines). Only used if engine='bspline'.
    engine : str, optional
        The fitting engine to use: 'auto', 'natural', 'reinsch', or 'bspline'.
        'auto' (default) attempts to pick the most efficient engine.
    """

    x: np.ndarray
    w: np.ndarray = None
    lamval: float = None
    df: float = None
    knots: np.ndarray = None
    n_knots: int = None
    order: int = 4
    engine: str = 'auto'
    
    _cpp_fitter: object = field(init=False, default=None, repr=False)
    _use_reinsch: bool = field(init=False, default=False, repr=False)
    _inverse_indices: np.ndarray = field(init=False, default=None, repr=False)
    
    # Scaling parameters
    x_min_: float = field(init=False, default=0.0)
    x_scale_: float = field(init=False, default=1.0)
    knots_scaled_: np.ndarray = field(init=False, default=None)
    n_k_: int = field(init=False, default=0)

    # Exposed attributes (populated after fit/init)
    coef_: float = field(init=False, default=None)
    intercept_: float = field(init=False, default=None)

    def __post_init__(self):
        self._setup_scaling_and_knots()
        self._prepare_matrices()
        
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        elif self.lamval is None:
            self.lamval = 0.0

    def _setup_scaling_and_knots(self):
        """
        Compute the scaled values and knots required for both
        fitting and EDF calculation.
        """
        x = self.x
        weights = self.w
        knots = self.knots
        n_knots = self.n_knots
        
        n = len(x)
        if weights is None: 
            weights = np.ones(n)
            self.w = None # Keep as None internally if originally None, but handle locally if needed
        
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

        self.x_scaled_ = (x - x_min) / scale
        self.knots_scaled_ = (knots - x_min) / scale

    def _prepare_matrices(self):
        """
        Initialize the C++ extension fitter.
        """

        # Decide which engine to use
        unique_x, inverse = np.unique(self.x_scaled_, return_inverse=True)
        is_reinsch_compatible = (
            len(self.knots_scaled_) == len(unique_x) and 
            np.allclose(self.knots_scaled_, unique_x)
        )
        
        if self.engine == 'auto':
            if is_reinsch_compatible:
                self.engine = 'reinsch'
            else:
                # Default fallback, should go to bspline
                self.engine = 'bspline' 

        if self.engine == 'reinsch':
            if not is_reinsch_compatible:
                raise ValueError("Reinsch engine requires knots to be exactly the unique x values.")
            
            self._use_reinsch = True
            self._inverse_indices = inverse
            
            # Aggregate weights for Reinsch
            w_raw = self.w if self.w is not None else np.ones(len(self.x_scaled_))
            w_agg = np.bincount(inverse, weights=w_raw)
            
            self._cpp_fitter = ReinschFitter(self.knots_scaled_, w_agg)
            
        elif self.engine == 'bspline':
            self._use_reinsch = False
            self._cpp_fitter = BSplineFitter(self.x_scaled_, self.knots_scaled_, self.w, self.order)
            
        elif self.engine == 'natural':
            self._use_reinsch = False
            self._cpp_fitter = NaturalSplineFitter(self.x_scaled_, self.knots_scaled_, self.w)
            
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

    def compute_df(self, lamval=None):
        """
        Compute the degrees of freedom for a given lambda.
        """
        if lamval is None:
            lamval = self.lamval
        
        lam_scaled = lamval / (self.x_scale_ ** 3)
        
        if self.engine == 'bspline':
            NTWN = self._cpp_fitter.compute_design()
            Omega = self._cpp_fitter.compute_penalty()
            n = NTWN.shape[1]
            NTWN_sub = NTWN[:, 1:n-1]
            Omega_sub = Omega[:, 1:n-1]
            
            # Matrices are already in Upper Banded format
            
            # DF = Tr((NTWN + lam*Omega)^-1 * NTWN)
            # trace_ratio_banded(A, B) = Tr((A+B)^-1 * B)
            # So A = lam*Omega, B = NTWN
            trace, self._cholesky = trace_ratio_banded(lam_scaled * Omega_sub, NTWN_sub)
            return trace
        else:
            return self._cpp_fitter.compute_df(lam_scaled)

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-20, 20)):
        """
        Finds the exact lambda value that yields the target degrees of freedom.
        """
        # Bounds check logic
        max_df = self.n_k_ if self.engine != 'bspline' else (self.n_k_ + self.order - 2)
        if target_df >= max_df - 0.01:
             return 1e-15 * (self.x_scale_ ** 3)
        if target_df <= 2.01:
             return 1e15 * (self.x_scale_ ** 3)

        shift = 3 * np.log10(self.x_scale_)
        min_log = log10_lam_bounds[0] - shift
        max_log = log10_lam_bounds[1] - shift

        if self.engine == 'bspline':
            # Use Python root finding with Takahashi DF
            def objective(log_lam):
                return self.compute_df(10**log_lam) - target_df
            
            try:
                log_lam_opt = brentq(objective, min_log + shift, max_log + shift)
                return 10**log_lam_opt
            except ValueError:
                # Fallback or wider search?
                return self._cpp_fitter.solve_for_df(target_df, min_log, max_log) * (self.x_scale_ ** 3)
        else:
             try:
                 lam_scaled = self._cpp_fitter.solve_for_df(target_df, min_log, max_log)
                 return lam_scaled * (self.x_scale_ ** 3)
             except RuntimeError as e:
                 raise RuntimeError(f"C++ solver failed: {e}")

    def solve_gcv(self, y, sample_weight=None, log10_lam_bounds=(-20, 20)):
        """
        Find optimal lambda using GCV and fit the model using C++.
        """
        if sample_weight is not None:
            self.update_weights(sample_weight)
        y_arr = np.asarray(y)
        
        if self._use_reinsch:
             w_raw = self.w if self.w is not None else np.ones(len(y_arr))
             # w_agg is already in fitter, but we need it for y_agg
             w_agg = np.bincount(self._inverse_indices, weights=w_raw)
             y_sum = np.bincount(self._inverse_indices, weights=y_arr * w_raw)
             y_eff = y_sum / w_agg
        else:
             y_eff = y_arr
        
        if hasattr(self._cpp_fitter, "solve_gcv"):
             # Adjust bounds for C++ which works on scaled lambda
             shift = 3 * np.log10(self.x_scale_)
             min_log = log10_lam_bounds[0] - shift
             max_log = log10_lam_bounds[1] - shift
             
             try:
                 lam_scaled = self._cpp_fitter.solve_gcv(y_eff, min_log, max_log)
                 self.lamval = lam_scaled * (self.x_scale_ ** 3)
                 self.fit(y)
                 return self.lamval
             except RuntimeError as e:
                 raise RuntimeError(f"C++ GCV solver failed: {e}")
                 
        raise RuntimeError("C++ extension required for GCV optimization.")

    def fit(self, y, sample_weight=None):
        """
        Fit the smoothing spline using the C++ extension.
        """
        if sample_weight is not None:
            self.w = sample_weight
            if hasattr(self._cpp_fitter, "update_weights"):
                 self._cpp_fitter.update_weights(self.w)
            else:
                 # BSpline might need full re-init if update_weights not exposed efficiently
                 # or if implementation differs. Current BSpline C++ doesn't have update_weights exposed?
                 # Let's check. SplineFitterBSpline constructor takes weights. 
                 # C++ class doesn't have update_weights method exposed in my impl above.
                 # So we need to re-create it for BSpline if weights change.
                 self._prepare_matrices()
        
        if self.lamval is None:
             self.lamval = 0.0
        lam_scaled = self.lamval / self.x_scale_**3
        
        y_arr = np.asarray(y)
        if self._use_reinsch:
             w_raw = self.w if self.w is not None else np.ones(len(y_arr))
             w_agg = np.bincount(self._inverse_indices, weights=w_raw)
             y_sum = np.bincount(self._inverse_indices, weights=y_arr * w_raw)
             y_eff = y_sum / w_agg
             
             self._cpp_fitter.fit(y_eff, lam_scaled)
        else:
             if self.engine == 'bspline':
                 NTWN = self._cpp_fitter.compute_design()
                 Omega = self._cpp_fitter.compute_penalty()
                 b = self._cpp_fitter.compute_rhs(y_arr)
                 AB = NTWN + lam_scaled * Omega
                 # AB is (kd+1, n) in UPPER banded format.
                 # The system to solve is for indices 1 to n-2.
                 # solveh_banded expects (l+1, M).
                 n = b.shape[0]
                 # slice columns
                 ab_sub = AB[:, 1:n-1]
                 b_sub = b[1:n-1]
                 
                 if not hasattr(self, "_cholesky"):
                     self.compute_df()
                 sol1 = cho_solve_banded((self._cholesky, False), b_sub)
                 sol = solveh_banded(ab_sub, b_sub)
                 print(np.linalg.norm(sol1 - sol)/np.linalg.norm(sol))
                 self._cpp_fitter.set_solution(sol)
             else:
                 self._cpp_fitter.fit(y_arr, lam_scaled)

        # Compute intercept and coef (linear part)
        # This is strictly valid for natural spline. 
        # For B-spline, "linear part" concept is slightly different but we can approximate or compute 
        # if needed. The previous code did this for all.
        y_hat = self.predict(self.x)
        w_eff = self.w if self.w is not None else np.ones(len(self.x))
        
        # Simple weighted linear regression on the fitted values
        X = np.vander(self.x, 2)
        Xw = X * w_eff[:, None]
        yw = y_hat * w_eff
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

        self.intercept_ = beta[1]
        self.coef_ = beta[0]

    def predict(self, x, deriv=0):
        """
        Predict the response for a new set of predictor variables using C++ basis evaluation.
        Parameters
        ----------
        x : np.ndarray
            The predictor variables.
        deriv : int, optional
            The order of the derivative to compute (default is 0).
        Returns
        -------
        np.ndarray
            The predicted response or its derivative.
        """
        x_scaled = (x - self.x_min_) / self.x_scale_
        
        # Scale the derivative by the chain rule
        scale_factor = 1.0 / (self.x_scale_ ** deriv)
        
        return self._cpp_fitter.predict(x_scaled, deriv) * scale_factor

    @property
    def nonlinear_(self):
        """
        The non-linear component of the fitted spline.
        """
        if self.coef_ is None:
            return None
        linear_part = self.coef_ * self.x + self.intercept_
        return self.predict(self.x) - linear_part

    def update_weights(self, w):
        """
        Update the weights and refit the model using C++.
        """
        self.w = w
        if hasattr(self._cpp_fitter, "update_weights"):
             self._cpp_fitter.update_weights(self.w)
        else:
             # Re-create fitter if update_weights is not supported (e.g. BSpline currently)
             self._prepare_matrices()

    # Expose N and Omega if available (Natural Spline only usually)
    @property
    def N_(self):
        if hasattr(self._cpp_fitter, "get_N"):
            return self._cpp_fitter.get_N()
        elif hasattr(self._cpp_fitter, "get_NTWN"):
            # BSpline case, returns NTWN instead of N
             return None 
        return None

    @property
    def Omega_(self):
        if hasattr(self._cpp_fitter, "get_Omega"):
            return self._cpp_fitter.get_Omega()
        return None
