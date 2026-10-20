# Changelog

## Unreleased

### Changed
- Default knots follow R's `smooth.spline` and `gam::s`: all the unique `x` values when there are fewer than 50, otherwise `.nknots.smspl(n)` of them (about 200), evenly spaced by rank among the unique values, with the B-spline engine. `all_knots=True` keeps every unique value as a knot (and the Reinsch engine). (#11)
- `smooth()` returns the fitted values at `x`. The fit and its linear part (`coef_`, `intercept_`) are computed at the unique `x` values, not at every observation, and the B-spline basis at the data is computed once: smoothing and weight updates are several times faster for large `n`.
- The B-spline engine aggregates tied `x` values (summing their weights), as the Reinsch engine does.
- `df` targeting is tighter: the root is found to about 1e-11 for the Reinsch engine (was 1e-6).

### Added
- `spline_coef_`, `set_spline_coef()` and `roughness()` on `SplineSmoother` (Reinsch and B-spline engines): the coefficients of the fitted spline, setting a fit from them, and the integral of its squared second derivative.
- `BSplineSmoother.update_weights()` and `BSplineSmoother.fitted()` in the C++ extension.

### Fixed
- `df` targeting no longer fits a line silently with many unique `x` values: a non-finite degrees of freedom raises, `brent_root` rejects non-finite values, and the Reinsch solve checks its factorization. (#11)
- The B-spline engine no longer reuses a stale Cholesky factor after `update_weights`. (#15)
- `smooth(y, sample_weight=...)` goes through `update_weights`, so it is right with tied `x`. (#16)
- The B-spline engine's maximum df is the number of knots; a `df` outside `[2, n_knots]` raises `ValueError`; tied quantile knots (`n_knots` with tied `x`) are dropped. (#18)
