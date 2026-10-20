from dataclasses import dataclass, field
import numpy as np
from ._scatter_smooth_extension import LoessSmootherCpp
from .splines import _as_vector, _check_weights

@dataclass
class LoessSmoother:
    """
    LoessSmoother implementation using C++ extension for performance.
    
    Parameters
    ----------
    x : np.ndarray
        The predictor variable.
    w : np.ndarray, optional
        Weights for the observations.
    span : float, optional
        The smoothing parameter (fraction of points to use as neighbors). Default is 0.75.
    degree : int, optional
        The degree of the local polynomial (0, 1, 2, or 3). Default is 1.

    Notes
    -----
    The local fit at a point uses its `ceil(span * n)` nearest neighbours,
    with all the points tied at the largest of their distances, and tricube
    weights; for `span > 1` all the points, with the bandwidth `sqrt(span)`
    times the largest distance (as R's `loess`). Without ties the fits agree
    with R's `loess(..., surface="direct")` (not its default interpolation)
    and with statsmodels' `lowess(it=0, delta=0)`. When the neighbourhood
    holds fewer distinct x values than the degree needs (tied x), the local
    polynomial drops its highest powers: one distinct value gives the
    (weighted) mean there.
    """

    x: np.ndarray
    w: np.ndarray = None
    span: float = 0.75
    degree: int = 1
    
    y: np.ndarray = field(init=False, default=None)
    _cpp_fitter: LoessSmootherCpp = field(init=False, default=None, repr=False)

    def __post_init__(self):
        self.x = _as_vector(self.x, 'x')
        if len(self.x) == 0:
            raise ValueError('x is empty')
        if self.w is not None:
            self.w = _check_weights(self.w, len(self.x))
        if self.degree not in [0, 1, 2, 3]:
            raise ValueError("Degree must be in range [0, 3].")
        if not (np.isfinite(self.span) and self.span > 0):
            raise ValueError(f'span should be positive and finite, got {self.span}')
        
        self._cpp_fitter = LoessSmootherCpp(self.x, self.w, self.span, self.degree)
        self._coef = self._intercept = self._y_hat_train = None

    def smooth(self, y, sample_weight=None):
        """
        Fit the Loess model.

        Parameters
        ----------
        y : np.ndarray
            Response variable.
        sample_weight : np.ndarray, optional
            Observation weights. If provided, updates the instance weights.
        """
        self.y = _as_vector(y, 'y', len(self.x))
        if sample_weight is not None:
            self.update_weights(sample_weight)
        self._cpp_fitter.set_y(self.y)
        self._coef = self._intercept = self._y_hat_train = None

    def _get_y_hat_train(self):
        if self.y is None:
            raise ValueError("Model has not been fitted yet. Call smooth(y) first.")
        if self._y_hat_train is None:
            self._y_hat_train = self._cpp_fitter.predict(self.x, 0)
        return self._y_hat_train

    def _compute_linear_part(self):
        if self._coef is None:
            y_hat = self._get_y_hat_train()
            w_eff = self.w if self.w is not None else np.ones(len(self.x))
            
            X = np.vander(self.x, 2)
            Xw = X * w_eff[:, None]
            yw = y_hat * w_eff
            beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

            self._intercept = beta[1]
            self._coef = beta[0]
            
    @property
    def intercept_(self):
        self._compute_linear_part()
        return self._intercept

    @property
    def coef_(self):
        self._compute_linear_part()
        return self._coef

    def update_weights(self, w):
        """
        Update the observation weights.

        Parameters
        ----------
        w : np.ndarray
            New weights.
        """
        self.w = _check_weights(w, len(self.x))
        self._cpp_fitter.update_weights(self.w)
        self._coef = self._intercept = self._y_hat_train = None

    def leverage(self):
        """
        The diagonal of the smoother matrix: the weight of each observation
        in its own fitted value. It doesn't depend on `y`.
        """
        return np.asarray(self._cpp_fitter.leverage())

    def compute_df(self):
        """The degrees of freedom: the trace of the smoother matrix."""
        return float(self.leverage().sum())

    @property
    def nonlinear_(self):
        """
        The non-linear component of the fitted loess curve.
        """
        if self.y is None:
            return None
        linear_part = self.coef_ * self.x + self.intercept_
        return self._get_y_hat_train() - linear_part

    def predict(self, x_new=None, deriv=0):
        """
        Predict the response for a new set of predictor variables using C++ extension.
        
        Parameters
        ----------
        x_new : np.ndarray, optional
            The predictor variables. If None, uses the initial `x` values.
        deriv : int, optional
            The order of the derivative to compute (default is 0).
            
        Returns
        -------
        np.ndarray
            The predicted response or its derivative.
        """
        if self.y is None:
            raise ValueError("Model has not been fitted yet. Call smooth(y) first.")
            
        if x_new is None:
            if deriv == 0:
                return self._get_y_hat_train()
            x_pred = self.x
        else:
            x_pred = x_new
            
        if deriv not in range(self.degree + 1):
            raise ValueError(f'deriv should be between 0 and degree={self.degree}, got {deriv}')
        x_pred = np.atleast_1d(np.asarray(x_pred, dtype=float)).ravel()
        return self._cpp_fitter.predict(x_pred, deriv)
