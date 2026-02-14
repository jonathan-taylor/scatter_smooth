from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from .base import _BaseSplineFitter

@dataclass
class SplineFitterCpp(_BaseSplineFitter):
    """
    SplineFitter implementation using C++ extension for performance.
    """
    _cpp_fitter: object = field(init=False, default=None, repr=False)

    def _prepare_matrices(self):
        """
        Initialize the C++ extension fitter.
        """
        from smoothing_spline._spline_extension import SplineFitterCpp as _ExtSplineFitterCpp
        x_scaled, knots_scaled = self._setup_scaling_and_knots()
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

        def df_error_func(log_lam_scaled):
            lam_scaled = 10 ** log_lam_scaled
            return self._cpp_fitter.compute_df(lam_scaled) - target_df

        try:
            log_lam_scaled_opt = brentq(df_error_func, log10_lam_bounds[0], log10_lam_bounds[1])
        except ValueError as e:
            raise RuntimeError("Could not find root in the given bounds.") from e

        return (10 ** log_lam_scaled_opt) * (self.x_scale_ ** 3)

    def solve_gcv(self, y, sample_weight=None, log10_lam_bounds=(-10, 10)):
        """
        Find optimal lambda using GCV and fit the model using C++.
        """
        if sample_weight is not None:
            self.update_weights(sample_weight)
        y_arr = np.asarray(y)
        def gcv_objective(log_lam):
            lam_scaled = (10**log_lam) / (self.x_scale_**3)
            return self._cpp_fitter.gcv_score(lam_scaled, y_arr)
        res = minimize_scalar(gcv_objective, bounds=log10_lam_bounds, method='bounded')
        if not res.success:
            raise RuntimeError(f"GCV optimization failed: {res.message}")
        self.lamval = 10**res.x
        self.fit(y)
        return self.lamval

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
        self.alpha_ = self._cpp_fitter.fit(y, lam_scaled)
        # self.spline_ is no longer created here as we use C++ basis for prediction

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

    def predict(self, x):
        """
        Predict the response for a new set of predictor variables using C++ basis evaluation.
        Parameters
        ----------
        x : np.ndarray
            The predictor variables.
        Returns
        -------
        np.ndarray
            The predicted response.
        """
        from smoothing_spline._spline_extension import compute_natural_spline_basis
        x_scaled = (x - self.x_min_) / self.x_scale_
        
        # compute_natural_spline_basis returns the basis matrix N
        # We assume extrapolate_linear=True (default in C++)
        N_new = compute_natural_spline_basis(x_scaled, self.knots_scaled_)
        
        # Prediction is N * alpha
        return N_new @ self.alpha_

    def update_weights(self, w):
        """
        Update the weights and refit the model using C++.
        """
        self.w = w
        if hasattr(self._cpp_fitter, "update_weights"):
             self._cpp_fitter.update_weights(self.w)
        else:
             self._prepare_matrices()
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        if hasattr(self, 'y'):
            self.fit(self.y)
