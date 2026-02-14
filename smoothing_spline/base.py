from dataclasses import dataclass, field
import numpy as np

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
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def nonlinear_(self):
        """
        The non-linear component of the fitted spline.
        """
        linear_part = self.coef_ * self.x + self.intercept_
        return self.predict(self.x) - linear_part
