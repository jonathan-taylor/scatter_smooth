from dataclasses import dataclass, field
import numpy as np

@dataclass
class SplineFitter:
    """
    SplineFitter implementation using C++ extension for performance.
    """

    x: np.ndarray
    w: np.ndarray = None
    lamval: float = None
    df: int = None
    knots: np.ndarray = None
    n_knots: int = None
    _cpp_fitter: object = field(init=False, default=None, repr=False)


    def __post_init__(self):
        self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        elif self.lamval is None:
            self.lamval = 0.0

    def _prepare_matrices(self):
        raise NotImplementedError

    def _find_lamval_for_df(self, df):
        raise NotImplementedError

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
        return x_scaled, knots_scaled

    def _prepare_matrices(self):
        """
        Initialize the C++ extension fitter.
        """
        from smoothing_spline._spline_extension import SplineFitterCpp as _ExtSplineFitterCpp
        from smoothing_spline._spline_extension import SplineFitterReinschCpp as _ExtSplineFitterReinschCpp
        
        x_scaled, knots_scaled = self._setup_scaling_and_knots()
        
        # Check if we can use the efficient Reinsch algorithm (knots == unique(x))
        # np.unique returns sorted unique elements
        unique_x, inverse = np.unique(x_scaled, return_inverse=True)
        
        if len(knots_scaled) == len(unique_x) and np.allclose(knots_scaled, unique_x):
            self._use_reinsch = True
            self._inverse_indices = inverse
            
            # Aggregate weights for Reinsch
            w_raw = self.w if self.w is not None else np.ones(len(x_scaled))
            w_agg = np.bincount(inverse, weights=w_raw)
            
            self._cpp_fitter = _ExtSplineFitterReinschCpp(knots_scaled, w_agg)
            
            # For Reinsch, N and Omega are implicit or managed differently, but we expose properties if needed
            # For now, we don't expose N_ and Omega_ for Reinsch path in python as they are not used
        else:
            self._use_reinsch = False
            self._cpp_fitter = _ExtSplineFitterCpp(x_scaled, knots_scaled, self.w)
            self.N_ = self._cpp_fitter.get_N()
            self.Omega_ = self._cpp_fitter.get_Omega()
            if self.w is not None:
                self.NTW_ = self.N_.T * self.w
            else:
                self.NTW_ = self.N_.T

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-12, 12)):
        """
        Finds the exact lambda value that yields the target degrees of freedom
        using the C++ implementation.
        """
        if target_df >= self.n_k_ - 0.01:
            raise ValueError(f"Target DF ({target_df}) too high.")
        if target_df <= 2.01:
            raise ValueError(f"Target DF ({target_df}) too low.")

        if hasattr(self._cpp_fitter, "solve_for_df"):
             try:
                 # Note: log10_lam_bounds are currently ignored by C++ solver (uses [-12, 12])
                 lam_scaled = self._cpp_fitter.solve_for_df(target_df)
                 return lam_scaled * (self.x_scale_ ** 3)
             except RuntimeError as e:
                 raise RuntimeError(f"C++ solver failed: {e}")
        
        raise RuntimeError("C++ extension required for finding lambda from DF.")

    def solve_gcv(self, y, sample_weight=None, log10_lam_bounds=(-10, 10)):
        """
        Find optimal lambda using GCV and fit the model using C++.
        """
        if sample_weight is not None:
            self.update_weights(sample_weight)
        y_arr = np.asarray(y)
        
        if hasattr(self, '_use_reinsch') and self._use_reinsch:
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
        self.y = y
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
        if hasattr(self, '_use_reinsch') and self._use_reinsch:
             w_raw = self.w if self.w is not None else np.ones(len(y_arr))
             w_agg = np.bincount(self._inverse_indices, weights=w_raw)
             y_sum = np.bincount(self._inverse_indices, weights=y_arr * w_raw)
             y_eff = y_sum / w_agg
             
             self._cpp_fitter.fit(y_eff, lam_scaled)
        else:
             self._cpp_fitter.fit(y_arr, lam_scaled)

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
        linear_part = self.coef_ * self.x + self.intercept_
        return self.predict(self.x) - linear_part

    def update_weights(self, w, update_lamval=False):
        """
        Update the weights and refit the model using C++.
        """
        self.w = w
        if hasattr(self._cpp_fitter, "update_weights"):
             self._cpp_fitter.update_weights(self.w)
        else:
             self._prepare_matrices()
        if self.df is not None and update_lamval:
            self.lamval = self._find_lamval_for_df(self.df)

@dataclass
class SplineFitterBSpline:
    """
    SplineFitter implementation using B-splines and LAPACK DBPSV.
    """

    x: np.ndarray
    w: np.ndarray = None
    lamval: float = None
    knots: np.ndarray = None
    n_knots: int = None
    order: int = 4
    _cpp_fitter: object = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self._prepare_matrices()

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
        return x_scaled, knots_scaled

    def _prepare_matrices(self):
        """
        Initialize the C++ extension fitter.
        """
        from smoothing_spline._spline_extension import SplineFitterBSpline as _ExtSplineFitterBSpline
        
        x_scaled, knots_scaled = self._setup_scaling_and_knots()
        self._cpp_fitter = _ExtSplineFitterBSpline(x_scaled, knots_scaled, self.w, self.order)

    def fit(self, y):
        """
        Fit the smoothing spline using the C++ extension.
        """
        self.y = y
        if self.lamval is None:
             self.lamval = 0.0
        lam_scaled = self.lamval / self.x_scale_**3
        
        y_arr = np.asarray(y)
        self._cpp_fitter.fit(y_arr, lam_scaled)

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
