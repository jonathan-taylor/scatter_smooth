from dataclasses import dataclass, field
import numpy as np
from scipy.interpolate import CubicSpline

@dataclass
class _BaseSplineFitter:
    """
    Base class for smoothing spline fitters.
    """
    x: np.ndarray
    w: np.ndarray = None
    lamval: float = None
    df: int = None
    knots: np.ndarray = None
    n_knots: int = None

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
