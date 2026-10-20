import warnings
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import (cholesky_banded,
                          solveh_banded)
from scipy.optimize import brentq

from ._scatter_smooth_extension import (
    NaturalSplineSmoother, 
    ReinschSmoother, 
    BSplineSmoother,
    trace_takahashi
)

def _as_vector(v, name, n=None):
    """`v` as a finite 1-D float array (an (n, 1) array is accepted), of length `n` if given."""
    v = np.asarray(v, dtype=float)
    if v.ndim == 2 and v.shape[1] == 1:
        v = v[:, 0]
    if v.ndim != 1:
        raise ValueError(f'{name} should be 1-dimensional, got shape {v.shape}')
    if n is not None and v.shape[0] != n:
        raise ValueError(f'{name} has length {v.shape[0]}, expected {n}')
    if not np.all(np.isfinite(v)):
        raise ValueError(f'{name} has non-finite values')
    return v


def _check_weights(w, n):
    w = _as_vector(w, 'weights', n)
    if np.any(w < 0):
        raise ValueError('weights should be non-negative')
    if not np.any(w > 0):
        raise ValueError('at least one weight should be positive')
    return w


def _nknots_smspl(n):
    """The number of knots of R's `smooth.spline` (`.nknots.smspl`) for `n` unique values."""
    if n < 50:
        return n
    a1, a2, a3, a4 = np.log2([50, 100, 140, 200])
    if n < 200:
        nk = 2**(a1 + (a2 - a1) * (n - 50) / 150)
    elif n < 800:
        nk = 2**(a2 + (a3 - a2) * (n - 200) / 600)
    elif n < 3200:
        nk = 2**(a3 + (a4 - a3) * (n - 800) / 2400)
    else:
        nk = 200 + (n - 3200)**0.2
    return int(nk)


def _default_knots(unique_x, all_knots=False):
    """
    All the (sorted) unique values, or, as R's `gam::s` (`sknotl`), with 50
    or more of them, `_nknots_smspl(n)` of them evenly spaced by rank.
    """
    n = len(unique_x)
    nk = n if all_knots else _nknots_smspl(n)
    if nk == n:
        return unique_x
    return unique_x[(np.arange(nk) * (n - 1)) // (nk - 1)]


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
    all_knots : bool, optional
        If True (and neither `knots` nor `n_knots` is given) the knots are
        all the unique values of `x`. By default, as R's `smooth.spline` and
        `gam::s`, they are all the unique values only when there are fewer
        than 50 of them; otherwise `.nknots.smspl(n_unique)` of them (about
        200 for large n), evenly spaced among the sorted unique values.
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
    all_knots: bool = False
    order: int = 4
    engine: str = 'auto'
    
    _cpp_fitter: object = field(init=False, default=None, repr=False)
    _use_reinsch: bool = field(init=False, default=False, repr=False)
    _inverse_indices: np.ndarray = field(init=False, default=None, repr=False)
    _NTWN: np.ndarray = field(init=False, default=None, repr=False)
    _Omega: np.ndarray = field(init=False, default=None, repr=False)
    
    # Scaling parameters
    x_min_: float = field(init=False, default=0.0)
    x_scale_: float = field(init=False, default=1.0)
    knots_scaled_: np.ndarray = field(init=False, default=None)
    n_k_: int = field(init=False, default=0)

    # Exposed attributes (populated after smooth/init)
    coef_: float = field(init=False, default=None)
    intercept_: float = field(init=False, default=None)

    def __post_init__(self):
        self._validate()
        self._setup_scaling_and_knots()
        self._prepare_matrices()
        
        if self.df is not None:
            self.lamval = self._find_lamval_for_df(self.df)
        elif self.lamval is None:
            self.lamval = 0.0
        
    def _validate(self):
        self.x = _as_vector(self.x, 'x')
        if len(np.unique(self.x)) < 4:
            raise ValueError('a cubic smoothing spline needs at least 4 distinct x values')
        if self.w is not None:
            self.w = _check_weights(self.w, len(self.x))
        if self.df is not None and self.lamval is not None:
            raise ValueError('give one of df and lamval, not both')
        if self.lamval is not None and not (np.isfinite(self.lamval) and self.lamval >= 0):
            raise ValueError(f'lamval should be non-negative and finite, got {self.lamval}')
        if self.df is not None and not np.isfinite(self.df):
            raise ValueError(f'df should be finite, got {self.df}')
        if self.order != 4:
            raise ValueError('only cubic splines (order=4) are supported')
        if self.n_knots is not None and (int(self.n_knots) != self.n_knots or self.n_knots < 4):
            raise ValueError(f'n_knots should be an integer of at least 4, got {self.n_knots}')
        if self.knots is not None:
            knots = np.unique(_as_vector(self.knots, 'knots'))
            if len(knots) < 4:
                raise ValueError('a cubic smoothing spline needs at least 4 distinct knots')
            self.knots = knots
        if self.engine not in ('auto', 'reinsch', 'bspline', 'natural'):
            raise ValueError(f"Unknown engine: {self.engine}")
        if self.engine == 'natural':
            warnings.warn("engine='natural' is deprecated: engine='bspline' fits the same natural "
                          "cubic spline with banded O(n) solves and accurate degrees of freedom",
                          DeprecationWarning, stacklevel=3)

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
                knots = np.unique(np.percentile(x, percs))
            else:
                knots = _default_knots(np.unique(x), self.all_knots)
        else:
            knots = np.sort(np.asarray(knots, float))      # a copy
            
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
        
        if self.w is not None:
            # the Reinsch engine needs a positive total weight at every knot
            has_empty_knot = np.any(np.bincount(inverse, weights=self.w) <= 0)
        else:
            has_empty_knot = False
        if self.engine == 'auto':
            if is_reinsch_compatible and not has_empty_knot:
                self.engine = 'reinsch'
            else:
                # Default fallback, should go to bspline
                self.engine = 'bspline' 

        if self.engine == 'reinsch':
            if not is_reinsch_compatible:
                raise ValueError("Reinsch engine requires knots to be exactly the unique x values.")
            if has_empty_knot:
                raise ValueError("Reinsch engine requires a positive weight at every unique x "
                                 "value; use engine='bspline' for zero weights")
            
            self._use_reinsch = True
            self._inverse_indices = inverse
            self._unique_x_scaled = unique_x
            
            # Aggregate weights for Reinsch
            w_raw = self.w if self.w is not None else np.ones(len(self.x_scaled_))
            w_agg = np.bincount(inverse, weights=w_raw)
            
            self._cpp_fitter = ReinschSmoother(self.knots_scaled_, w_agg)
            
        elif self.engine == 'bspline':
            # the fit depends on the data only through the unique x values and
            # the sums of the weights (and of weights * y) over their ties
            self._use_reinsch = False
            self._inverse_indices = inverse
            self._unique_x_scaled = unique_x
            self._make_bspline_fitter()
            
        elif self.engine == 'natural':
            self._use_reinsch = False
            self._cpp_fitter = NaturalSplineSmoother(self.x_scaled_, self.knots_scaled_, self.w)
            # the same model with the B-spline engine, for the degrees of
            # freedom: the dense solve's df is inaccurate at large lambda
            self._df_helper = SplineSmoother(self.x, w=self.w, knots=self.knots,
                                             lamval=0., engine='bspline')
            
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

    def _aggregated_weights(self):
        w_raw = self.w if self.w is not None else np.ones(len(self.x_scaled_))
        return np.bincount(self._inverse_indices, weights=w_raw)

    def _make_bspline_fitter(self):
        self._cpp_fitter = BSplineSmoother(self._unique_x_scaled, self.knots_scaled_,
                                           self._aggregated_weights(), self.order)
        self._NTWN = None
        self._Omega = None

    def compute_df(self, lamval=None, eps=1e-12):
        """
        Compute the degrees of freedom for a given lambda.

        For the B-spline engine, `eps` times the mean diagonal of the system
        is added to its diagonal, which keeps the trace stable at very large
        lambda (where the penalty, null on linear functions, swamps the
        design). That factor is not used by `smooth`, which solves the exact
        system.
        """
        if lamval is None:
            lamval = self.lamval
        if self.engine == 'natural':
            return self._df_helper.compute_df(lamval)
        
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
            
            AB = lam_scaled * Omega_sub + NTWN_sub
            if eps > 0:
                AB[-1, :] += eps * (np.abs(AB[-1, :]).mean() + 1e-10)
            try:
                # lower=False for Upper Banded
                U = cholesky_banded(AB, lower=False)
            except np.linalg.LinAlgError as e:
                raise RuntimeError(f'compute_df: the spline system is not positive definite '
                                   f'at lamval={lamval}') from e
            df = trace_takahashi(U, NTWN_sub)
        else:
            df = self._cpp_fitter.compute_df(lam_scaled)
        if not np.isfinite(df):
            raise RuntimeError(f'compute_df: the degrees of freedom at lamval={lamval} are not '
                               'finite (the spline system is too ill-conditioned; use fewer knots)')
        return df

    def _find_lamval_for_df(self, target_df, log10_lam_bounds=(-8, 12)):
        """
        Finds the exact lambda value that yields the target degrees of freedom.
        """
        # Bounds check logic
        # the natural cubic splines with these knots have dimension n_k
        max_df = self.n_k_
        if not 2 <= target_df <= max_df:
            raise ValueError(f'df={target_df} is out of range: the degrees of freedom of a '
                             f'cubic smoothing spline with {max_df} knots are in [2, {max_df}]')
        if self.engine == 'natural':
            return self._df_helper._find_lamval_for_df(target_df, log10_lam_bounds)
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
            
            lo, hi = min_log + shift, max_log + shift
            f_lo, f_hi = objective(lo), objective(hi)
            if f_lo * f_hi > 0:
                raise ValueError(f'df={target_df} is out of the range '
                                 f'({f_hi + target_df:.3g}, {f_lo + target_df:.3g}) '
                                 f'of lamval in 10**{log10_lam_bounds}')
            log_lam_opt = brentq(objective, lo, hi, xtol=1e-12, rtol=1e-12)
            return 10**log_lam_opt
        else:
             try:
                 lam_scaled = self._cpp_fitter.solve_for_df(target_df, min_log, max_log)
                 return lam_scaled * (self.x_scale_ ** 3)
             except RuntimeError as e:
                 raise RuntimeError(f"C++ solver failed: {e}")

    def gcv_score(self, y, lamval=None):
        """
        The generalized cross-validation criterion of R's `smooth.spline`,

            (sum(w * (y - f)^2) / n) / (1 - df / n)^2,

        with the weights normalized to mean 1 over the n observations of
        positive weight, and the residual sum of squares over all the
        observations (so with tied x it includes the sum of squares within
        the ties).
        """
        if lamval is not None:
            self.lamval = lamval
        y = _as_vector(y, 'y', len(self.x))
        w = self.w if self.w is not None else np.ones(len(y))
        n = np.sum(w > 0)
        w = w * n / w.sum()
        f = self.smooth(y)
        df = self.compute_df(self.lamval)
        denom = 1 - df / n
        if denom <= 0:
            return np.inf
        return (np.sum(w * (y - f)**2) / n) / denom**2

    def solve_gcv(self, y, sample_weight=None, log10_lam_bounds=(-12, 8)):
        """
        Choose lambda by GCV (`gcv_score`, as R's `smooth.spline(cv=FALSE)`)
        and smooth `y` with it: a grid over `log10_lam_bounds` (in the units
        of x scaled to [0, 1], times the mean weight), refined around its
        minimum by Brent's method. Returns lambda.
        """
        if sample_weight is not None:
            self.update_weights(sample_weight)
        y = _as_vector(y, 'y', len(self.x))
        w = self.w if self.w is not None else np.ones(len(y))
        # lambda in the units where x is in [0, 1] and the weights have mean 1
        unit = self.x_scale_**3 * w.mean()

        def crit(log_lam):
            try:
                return self.gcv_score(y, 10**log_lam * unit)
            except (RuntimeError, ValueError):
                return np.inf

        grid = np.linspace(log10_lam_bounds[0], log10_lam_bounds[1], 81)
        values = np.array([crit(g) for g in grid])
        if not np.any(np.isfinite(values)):
            raise RuntimeError('GCV: the criterion is not finite on the grid')
        i = int(np.nanargmin(values))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(crit, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-6})
        best = res.x if res.fun <= values[i] else grid[i]
        self.lamval = 10**best * unit
        self.smooth(y)
        return self.lamval

    def _aggregated_response(self, y):
        """The weighted means of `y` over the ties of `x` (0 where the weights sum to 0)."""
        w_raw = self.w if self.w is not None else np.ones(len(y))
        w_agg = np.bincount(self._inverse_indices, weights=w_raw)
        y_sum = np.bincount(self._inverse_indices, weights=y * w_raw)
        return np.divide(y_sum, w_agg, out=np.zeros_like(y_sum), where=w_agg > 0)

    def smooth(self, y, sample_weight=None):
        """
        Smooth `y` (with weights `sample_weight`, if given: the same as
        `update_weights(sample_weight)` first).

        Returns
        -------
        np.ndarray
            The fitted values at `x`.
        """
        if sample_weight is not None:
            self.update_weights(sample_weight)
        
        if self.lamval is None:
             self.lamval = 0.0
        lam_scaled = self.lamval / self.x_scale_**3
        
        y_arr = _as_vector(y, 'y', len(self.x))
        self._fitted = True
        if self._use_reinsch:
             y_eff = self._aggregated_response(y_arr)
             fitted_unique = np.asarray(self._cpp_fitter.smooth(y_eff, lam_scaled))
             self.spline_coef_ = fitted_unique.copy()
        elif self.engine == 'bspline':
             if self._NTWN is None:
                 self._NTWN = self._cpp_fitter.compute_design()
             if self._Omega is None:
                 self._Omega = self._cpp_fitter.compute_penalty()

             b = self._cpp_fitter.compute_rhs(self._aggregated_response(y_arr))
             
             n = b.shape[0]
             b_sub = b[1:n-1]
             
             # the exact system (compute_df's factor is regularized)
             NTWN_sub = self._NTWN[:, 1:n-1]
             Omega_sub = self._Omega[:, 1:n-1]
             AB = lam_scaled * Omega_sub + NTWN_sub
             sol = solveh_banded(AB, b_sub, lower=False)
                 
             self._cpp_fitter.set_solution(sol)
             self.spline_coef_ = np.asarray(sol, float)
             fitted_unique = np.asarray(self._cpp_fitter.fitted())
        else:
             self._cpp_fitter.smooth(y_arr, lam_scaled)
             fitted = self.predict(self.x)
             self._set_linear_part(self.x, fitted,
                                   self.w if self.w is not None else np.ones(len(self.x)))
             return fitted

        # the linear part, from the fit at the unique values of x
        w_agg = self._aggregated_weights()
        self._set_linear_part(self.x_min_ + self.x_scale_ * self._unique_x_scaled,
                              fitted_unique, w_agg)
        return fitted_unique[self._inverse_indices]

    def _set_linear_part(self, x, f, w):
        """The weighted least squares line through the fitted values `f` at `x`."""
        sw = w.sum()
        xbar, fbar = (w * x).sum() / sw, (w * f).sum() / sw
        sxx = (w * (x - xbar)**2).sum()
        self.coef_ = (w * (x - xbar) * (f - fbar)).sum() / sxx if sxx > 0 else 0.
        self.intercept_ = fbar - self.coef_ * xbar

    def set_spline_coef(self, coef):
        """
        Set the fitted spline from its coefficients `coef` (as `spline_coef_`
        after a smooth): for the Reinsch engine, the values of the spline at
        the knots; for the B-spline engine, the coefficients of the natural
        B-spline basis. The coefficients are linear in the fit, and adding a
        constant to them adds that constant to the spline.
        """
        coef = np.asarray(coef, float)
        if self._use_reinsch:
            # the natural cubic spline interpolating its values at the knots
            self._cpp_fitter.smooth(coef, 0.)
        elif self.engine == 'bspline':
            self._cpp_fitter.set_solution(coef)
        else:
            raise NotImplementedError("set_spline_coef: not available for engine='natural'")
        self.spline_coef_ = coef.copy()
        self._fitted = True
        return self

    def roughness(self, coef=None):
        """
        The integral of the squared second derivative of the spline with
        coefficients `coef` (by default `spline_coef_`), in the units of `x`.
        """
        coef = np.asarray(self.spline_coef_ if coef is None else coef, float)
        if self._use_reinsch:
            # Green & Silverman: g^T Q R^{-1} Q^T g
            t = self.knots_scaled_
            h = np.diff(t)
            Qg = (coef[:-2] / h[:-1] - coef[1:-1] * (1 / h[:-1] + 1 / h[1:]) +
                  coef[2:] / h[1:])
            R = np.zeros((2, len(t) - 2))
            R[1] = (h[:-1] + h[1:]) / 3
            R[0, 1:] = h[1:-1] / 6
            value = Qg @ solveh_banded(R, Qg)
        elif self.engine == 'bspline':
            if self._Omega is None:
                self._Omega = self._cpp_fitter.compute_penalty()
            n = self._Omega.shape[1]
            Om = self._Omega[:, 1:n-1]          # upper banded, as in smooth()
            kd = Om.shape[0] - 1
            value = Om[kd] @ coef**2
            for k in range(1, kd + 1):
                value += 2 * Om[kd - k, k:] @ (coef[k:] * coef[:-k])
        else:
            raise NotImplementedError("roughness: not available for engine='natural'")
        # f(x) = g((x - x_min) / scale): f'' = g'' / scale^2, dx = scale du
        return float(value) / self.x_scale_**3

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
        if not getattr(self, '_fitted', False):
            raise RuntimeError('the smoother is not fitted yet: call smooth(y) first')
        if deriv not in (0, 1, 2, 3):
            raise ValueError(f'deriv should be 0, 1, 2 or 3, got {deriv}')
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
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
        Set new weights for the next smooths. `lamval` is kept; with a `df`
        target, call `_find_lamval_for_df` again to match it at the new weights.
        """
        w = _check_weights(w, len(self.x))
        self.w = w
        if self.engine == 'reinsch':
             w_agg = self._aggregated_weights()
             if np.any(w_agg <= 0):
                 raise ValueError("Reinsch engine requires a positive weight at every unique x "
                                  "value; use engine='bspline' for zero weights")
             self._cpp_fitter.update_weights(w_agg)
        elif self.engine == 'bspline':
             self._cpp_fitter.update_weights(self._aggregated_weights())
             self._NTWN = None
        else:
             self._cpp_fitter.update_weights(self.w)
             self._df_helper.update_weights(self.w)
        
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
