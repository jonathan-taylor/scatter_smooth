from dataclasses import dataclass, field
import numpy as np

@dataclass
class LoessSmoother:
    """
    LoessSmoother implementation using pure Python/NumPy.
    
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
    """

    x: np.ndarray
    w: np.ndarray = None
    span: float = 0.75
    degree: int = 1
    
    y: np.ndarray = field(init=False, default=None)
    intercept_: float = field(init=False, default=None)
    coef_: float = field(init=False, default=None)

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        if self.w is not None:
            self.w = np.asarray(self.w, dtype=float)
        if self.degree not in [0, 1, 2, 3]:
            raise ValueError("Degree must be in range [0, 3].")

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
        self.y = np.asarray(y, dtype=float)
        if sample_weight is not None:
            self.w = np.asarray(sample_weight, dtype=float)

        # Compute intercept and coef (global linear part of the fit) for consistency
        y_hat = self.predict(self.x)
        w_eff = self.w if self.w is not None else np.ones(len(self.x))
        
        X = np.vander(self.x, 2)
        Xw = X * w_eff[:, None]
        yw = y_hat * w_eff
        beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]

        self.intercept_ = beta[1]
        self.coef_ = beta[0]

    def update_weights(self, w):
        """
        Update the observation weights.

        Parameters
        ----------
        w : np.ndarray
            New weights.
        """
        self.w = np.asarray(w, dtype=float)

    @property
    def nonlinear_(self):
        """
        The non-linear component of the fitted loess curve.
        """
        if self.coef_ is None:
            return None
        linear_part = self.coef_ * self.x + self.intercept_
        return self.predict(self.x) - linear_part

    def predict(self, x_new, deriv=0):
        """
        Predict the response for a new set of predictor variables.
        
        Parameters
        ----------
        x_new : np.ndarray
            The predictor variables.
        deriv : int, optional
            The order of the derivative to compute (default is 0).
            
        Returns
        -------
        np.ndarray
            The predicted response or its derivative.
        """
        if self.y is None:
            raise ValueError("Model has not been fitted yet. Call smooth(y) first.")
            
        x_new = np.atleast_1d(x_new).astype(float)
        n = len(self.x)
        k = int(np.ceil(self.span * n))
        k = max(k, self.degree + 1)
        k = min(k, n) # Ensure k doesn't exceed n

        y_pred = np.zeros_like(x_new, dtype=float)
        
        obs_weights = self.w if self.w is not None else np.ones(n)

        # Pre-compute powers for x if needed? No, local x changes.
        
        for i, val in enumerate(x_new):
            # 1. Distances
            dists = np.abs(self.x - val)
            
            # 2. Find k nearest neighbors
            # We need the k-th smallest distance.
            # partial sort is O(n)
            idx = np.argpartition(dists, k-1)[:k]
            
            # 3. Compute Max Distance (Delta) within the neighborhood
            # The window width is the distance to the k-th nearest neighbor
            max_dist = dists[idx].max()
            
            # 4. Tricube weights
            if max_dist <= 1e-14:
                # All points in neighborhood are identical to val?
                # Or k is very small?
                weights = np.ones(len(idx))
            else:
                u = dists[idx] / max_dist
                # Tricube: (1 - u^3)^3 for |u| < 1
                weights = np.clip(1 - u**3, 0, None)**3
            
            # Combine with observation weights
            total_weights = weights * obs_weights[idx]
            
            if np.sum(total_weights) < 1e-14:
                y_pred[i] = np.nan
                continue

            # 5. Local Regression
            x_local = self.x[idx] - val # Center at prediction point
            y_local = self.y[idx]
            
            # Weighted Least Squares
            # Design Matrix: [1, x, x^2, ...]
            # We want the value at x=val, which corresponds to x_local=0.
            # So we just need the intercept (beta[0]).
            # If deriv=1, we need beta[1].
            
            # Construct Vandermonde matrix
            # shape (k, degree+1)
            # x_local is shape (k,)
            
            # Use sqrt weights for lstsq
            sqrt_w = np.sqrt(total_weights)
            
            # Using vander: columns are x^(deg), x^(deg-1), ..., 1
            # We want 1, x, x^2 for easier indexing? 
            # np.vander(..., increasing=True) gives 1, x, x^2
            
            X_des = np.vander(x_local, self.degree + 1, increasing=True)
            
            X_w = X_des * sqrt_w[:, None]
            y_w = y_local * sqrt_w
            
            try:
                # Solve (X'W X) beta = X'W y
                # lstsq returns beta, residuals, rank, singular values
                beta, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
                
                # The prediction is sum(beta[j] * 0^j). 
                # Since we centered x, the value at x_new (local 0) is beta[0].
                # The first derivative is beta[1] * 1!
                # The second derivative is beta[2] * 2!
                # The d-th derivative is beta[d] * d!
                
                if deriv == 0:
                    y_pred[i] = beta[0]
                elif deriv <= self.degree:
                    import math
                    y_pred[i] = beta[deriv] * math.factorial(deriv)
                else:
                    y_pred[i] = 0.0
                    
            except np.linalg.LinAlgError:
                y_pred[i] = np.nan

        return y_pred
