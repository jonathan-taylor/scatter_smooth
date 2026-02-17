from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import (cholesky_banded,
                          cho_solve_banded,
                          solveh_banded)
from scipy.optimize import brentq

from ._scatter_smooth_extension import (
    NaturalSplineSmoother, 
    ReinschSmoother, 
    BSplineSmoother,
    trace_takahashi
)

@dataclass
class SplineSmoother:
    """
    SplineSmoother implementation using C++ extension for performance.
    
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
    _NTWN: np.ndarray = field(init=False, default=None, repr=False)
    _Omega: np.ndarray = field(init=False, default=None, repr=False)
    _cholesky_cache: dict = field(init=False, default_factory=dict, repr=False)
    
    # Scaling parameters
    x_min_: float = field(init=False, default=0.0)
    x_scale_: float = field(init=False, default=1.0)
    knots_scaled_: np.ndarray = field(init=False, default=None)
    n_k_: int = field(init=False, default=0)

    # Exposed attributes (populated after smooth/init)
    coef_: float = field(init=False, default=None)
    intercept_: float = field(init=False, default=None)

    def __post_init__(self):
        self._setup_scaling_and_knots()
        self._prepare_matrices()
        
        self._cholesky_cache = {}
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
            
            self._cpp_fitter = ReinschSmoother(self.knots_scaled_, w_agg)
            
        elif self.engine == 'bspline':
            self._use_reinsch = False
            self._cpp_fitter = BSplineSmoother(self.x_scaled_, self.knots_scaled_, self.w, self.order)
            self._NTWN = None
            self._Omega = None
            
        elif self.engine == 'natural':
            self._use_reinsch = False
            self._cpp_fitter = NaturalSplineSmoother(self.x_scaled_, self.knots_scaled_, self.w)
            
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

    def compute_df(self, lamval=None, eps=1e-12):
        """
        Compute the degrees of freedom for a given lambda.
        """
        if lamval is None:
            lamval = self.lamval
        
        lam_scaled = lamval / (self.x_scale_ ** 3)
        
        if self.engine == 'bspline':
            if self._NTWN is None:
                self._NTWN = self._cpp_fitter.compute_design()
            if self._Omega is None:
                self._Omega = self._cpp_fitter.compute_penalty()
                
            NTWN = self._NTWN
            Omega = self._Omega
            n = NTWN.shape[1]
            
            NTWN_sub = NTWN[:, 1:n-1]
            Omega_sub = Omega[:, 1:n-1]
            
            # Compute Cholesky explicitly and cache it
            AB = lam_scaled * Omega_sub + NTWN_sub
            
            # Add small padding to ensure positive definiteness
            if eps > 0:
                AB[-1, :] += eps * (np.abs(AB[-1, :]).mean() + 1e-10)

            try:
                # lower=False for Upper Banded
                U = cholesky_banded(AB, lower=False)
                self._cholesky_cache[lam_scaled] = U
                return trace_takahashi(U, NTWN_sub)
            except np.linalg.LinAlgError:
                return 0.0
        else:
            return self._cpp_fitter.compute_df(lam_scaled)

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-8, 12)):
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
        Find optimal lambda using GCV and smooth the model using C++.
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
                 self.smooth(y)
                 return self.lamval
             except RuntimeError as e:
                 raise RuntimeError(f"C++ GCV solver failed: {e}")
                 
        raise RuntimeError("C++ extension required for GCV optimization.")

    def smooth(self, y, sample_weight=None):
        """
        Smooth the smoothing spline using the C++ extension.
        """
        if sample_weight is not None:
            self.w = sample_weight
            if hasattr(self._cpp_fitter, "update_weights"):
                 self._cpp_fitter.update_weights(self.w)
            else:
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
             
             self._cpp_fitter.smooth(y_eff, lam_scaled)
        else:
             if self.engine == 'bspline':
                 if self._NTWN is None:
                     self._NTWN = self._cpp_fitter.compute_design()
                 if self._Omega is None:
                     self._Omega = self._cpp_fitter.compute_penalty()
                     
                 b = self._cpp_fitter.compute_rhs(y_arr)
                 
                 n = b.shape[0]
                 b_sub = b[1:n-1]
                 
                 if lam_scaled in self._cholesky_cache:
                     U = self._cholesky_cache[lam_scaled]
                     sol = cho_solve_banded((U, False), b_sub)
                 else:
                     NTWN_sub = self._NTWN[:, 1:n-1]
                     Omega_sub = self._Omega[:, 1:n-1]
                     AB = lam_scaled * Omega_sub + NTWN_sub
                     sol = solveh_banded(AB, b_sub, lower=False)
                     
                 self._cpp_fitter.set_solution(sol)
             else:
                 self._cpp_fitter.smooth(y_arr, lam_scaled)

        # Compute intercept and coef (linear part)
        y_hat = self.predict(self.x)
        w_eff = self.w if self.w is not None else np.ones(len(self.x))
        
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
             if self.engine == 'reinsch' and hasattr(self, '_inverse_indices') and self._inverse_indices is not None:
                 w_agg = np.bincount(self._inverse_indices, weights=w)
                 self._cpp_fitter.update_weights(w_agg)
             else:
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
