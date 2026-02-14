from dataclasses import dataclass, field
import numpy as np

@dataclass
class LowessFitterNaive:
    """
    Naive LowessFitter implementation using pure Python/NumPy for reference/comparison.
    
    Parameters
    ----------
    x : np.ndarray
        Predictor values (1D).
    w : np.ndarray, optional
        Observation weights.
    span : float, default=0.75
        Smoothing parameter (fraction of points to use as neighbors).
    degree : int, default=1
        Polynomial degree for local regression (0, 1, or 2).
    """

    x: np.ndarray
    w: np.ndarray = None
    span: float = 0.75
    degree: int = 1
    y: np.ndarray = field(init=False, default=None)

    def fit(self, y, sample_weight=None):
        """
        Fit the Lowess model (store data).

        Parameters
        ----------
        y : np.ndarray
            Response variable.
        sample_weight : np.ndarray, optional
            Observation weights.
        """
        self.y = np.asarray(y)
        if sample_weight is not None:
            self.w = np.asarray(sample_weight)
        
        if not isinstance(self.x, np.ndarray):
            self.x = np.asarray(self.x)
        if self.w is not None and not isinstance(self.w, np.ndarray):
            self.w = np.asarray(self.w)

    def predict(self, x_new):
        """
        Predict response for new points.

        Parameters
        ----------
        x_new : np.ndarray
            Points to predict at.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        x_new = np.atleast_1d(x_new)
        n = len(self.x)
        k = int(np.ceil(self.span * n))
        k = max(k, self.degree + 1)

        y_pred = np.zeros_like(x_new, dtype=float)
        
        obs_weights = self.w if self.w is not None else np.ones(n)

        for i, val in enumerate(x_new):
            # 1. Distances
            dists = np.abs(self.x - val)
            
            # 2. Find k nearest neighbors
            if k >= n:
                 idx = np.arange(n)
            else:
                 idx = np.argpartition(dists, k-1)[:k]
            
            # 3. Compute Max Distance (Delta)
            max_dist = dists[idx].max()
            
            if max_dist <= 0:
                weights = np.ones(len(idx))
            else:
                # 4. Tricube weights
                u = dists[idx] / max_dist
                weights = (1 - u**3)**3
                weights[u >= 1] = 0
            
            # Combine with observation weights
            total_weights = weights * obs_weights[idx]
            
            if np.sum(total_weights) == 0:
                y_pred[i] = np.nan
                continue

            # 5. Local Regression
            x_local = self.x[idx] - val
            y_local = self.y[idx]
            
            sqrt_w = np.sqrt(total_weights)
            
            # Design Matrix
            X_des = np.vander(x_local, self.degree + 1, increasing=True)
            
            X_w = X_des * sqrt_w[:, None]
            y_w = y_local * sqrt_w
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                y_pred[i] = beta[0]
            except np.linalg.LinAlgError:
                y_pred[i] = np.nan

        return y_pred


class LowessFitter:
    """
    LowessFitter implementation using C++ extension.
    """
    def __init__(self, x, w=None, span=0.75, degree=1):
        self.x = np.asarray(x, dtype=np.float64)
        self.w = np.asarray(w, dtype=np.float64) if w is not None else None
        self.span = span
        self.degree = degree
        self._cpp_fitter = None
        
        self._init_cpp()

    def _init_cpp(self):
        try:
            from smoothing_spline._spline_extension import LowessFitterCpp
            self._cpp_fitter = LowessFitterCpp(self.x, self.w, self.span, self.degree)
        except ImportError:
            raise ImportError("C++ extension `_spline_extension` could not be imported. Ensure it is built.")

    def fit(self, y, sample_weight=None):
        y = np.asarray(y, dtype=np.float64)
        if sample_weight is not None:
             self.w = np.asarray(sample_weight, dtype=np.float64)
             # Re-init because weights are constructor args in C++ currently
             self._init_cpp()
             
        self._cpp_fitter.fit(y)

    def predict(self, x_new):
        x_new = np.asarray(x_new, dtype=np.float64)
        return self._cpp_fitter.predict(x_new)
