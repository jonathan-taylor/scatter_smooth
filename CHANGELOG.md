# Changelog

## Unreleased

### Changed
- The wheels workflow no longer publishes to PyPI: a tag builds the wheels and the sdist as artifacts, which are uploaded by hand with `twine`.

## 0.3.0 (2026-10-20)

This release changes default fits: with 50 or more distinct `x` values the knots are thinned as in R (`all_knots=True` keeps them all), GCV follows R's criterion, and LOESS fits change with tied `x` and for `span > 1`. Bad input now raises instead of crashing or giving silent nonsense. Python 3.10 or later is required.

### Changed
- LOESS is faster: `x` is sorted once and each neighbourhood is a window of it (O(log n + k) to find, instead of all n distances and a partial sort per point), and the local normal equations come from moments. Predicting at the training points: 8× faster at span 0.1 and 2.7× at span 0.75 (n = 10,000); the fits are unchanged (to 1e-13). (#23)
- The C++ smoothing, df and prediction calls release the GIL, so other Python threads run meanwhile (one smoother object shouldn't be used from several threads at once). (#23)
- Default knots follow R's `smooth.spline` and `gam::s`: all the unique `x` values when there are fewer than 50, otherwise `.nknots.smspl(n)` of them (about 200), evenly spaced by rank among the unique values, with the B-spline engine. `all_knots=True` keeps every unique value as a knot (and the Reinsch engine). (#11)
- `smooth()` returns the fitted values at `x`. The fit and its linear part (`coef_`, `intercept_`) are computed at the unique `x` values, not at every observation, and the B-spline basis at the data is computed once: smoothing and weight updates are several times faster for large `n`.
- The B-spline engine aggregates tied `x` values (summing their weights), as the Reinsch engine does.
- `df` targeting is tighter: the root is found to about 1e-11 for the Reinsch engine (was 1e-6).

### Added
- `LoessSmoother.leverage()` and `compute_df()`: the diagonal of the smoother matrix and its trace, in one pass (before, the df took one smooth per observation). (#23)
- `spline_coef_`, `set_spline_coef()` and `roughness()` on `SplineSmoother` (Reinsch and B-spline engines): the coefficients of the fitted spline, setting a fit from them, and the integral of its squared second derivative.
- `BSplineSmoother.update_weights()` and `BSplineSmoother.fitted()` in the C++ extension.

### Deprecated
- `engine='natural'` is deprecated (`DeprecationWarning`): `engine='bspline'` fits the same natural cubic spline with banded solves. Its degrees of freedom, df targeting and GCV now come from the B-spline engine with the same knots: its dense df wasn't monotone at large lambda, so df targeting failed on ordinary data. (#20)

### Fixed
- GCV (`solve_gcv`) uses R's `smooth.spline` criterion for every engine: weights normalized to mean 1, the residual sum of squares over all the observations (so with tied `x` it includes the sum of squares within the ties) and n the number of observations. With ties it chose df 11.78 where R chooses 11.64 (13.40 against 13.73 with weights); it now agrees with R to R's optimizer tolerance. The minimum is found on a grid refined by Brent's method. New `gcv_score(y, lamval)`. The natural engine's GCV used to ignore the weights. (#17)
- `df` targeting no longer fits a line silently with many unique `x` values: a non-finite degrees of freedom raises, `brent_root` rejects non-finite values, and the Reinsch solve checks its factorization. (#11)
- The B-spline engine no longer reuses a stale Cholesky factor after `update_weights`. (#15)
- `smooth(y, sample_weight=...)` goes through `update_weights`, so it is right with tied `x`. (#16)
- The B-spline engine's maximum df is the number of knots; a `df` outside `[2, n_knots]` raises `ValueError`; tied quantile knots (`n_knots` with tied `x`) are dropped. (#18)
- After `df` targeting, the B-spline engine's `smooth()` solved with the padded Cholesky factor cached by `compute_df`, so the fit was off by about 3e-7; it now always solves the exact system (the cache is gone). (#38)
- `predict()` before `smooth()` raises `RuntimeError` instead of crashing, in Python and in every C++ class. (#12)
- Every C++ entry point checks the lengths of `y`, the weights and banded matrices, raising `ValueError` instead of reading out of bounds. (#13)
- Input validation (#14):
  - `x`, `y` and the weights must be finite and 1-dimensional (an `(n, 1)` array is accepted), with matching lengths;
  - a spline needs at least 4 distinct `x` values (and knots);
  - weights must be non-negative, with at least one positive. Zero weights are allowed, as in R: the auto engine uses the B-spline engine when a unique `x` value has zero total weight, and the Reinsch engine raises;
  - `lamval` must be non-negative; `df` and `lamval` can't both be given; `order` must be 4; `n_knots` must be at least 4;
  - LOESS `span` must be positive and finite, and `x` non-empty;
  - the caller's `knots` are no longer sorted in place.
- LOESS with tied `x` (#19):
  - the neighbourhood holds every point tied at the boundary distance;
  - when it holds fewer distinct `x` values than the degree needs, the local polynomial drops its highest powers (one distinct value gives the weighted mean) instead of returning NaN (degree 1) or garbage (degree 2);
  - when every neighbour sits at the boundary distance (zero tricube weight), they get equal weights, the limit of a slightly larger bandwidth;
  - `span > 1` widens the bandwidth to `sqrt(span)` times the largest distance, as R does (was the same fit as `span = 1`).
- `deriv`: splines accept 0–3 and the Reinsch engine now returns the true third derivative (was 0); the natural engine accepts 0–2; LOESS accepts 0 to its degree; anything else raises `ValueError`. (#21)

### Packaging
- The version comes from `setuptools_scm` (the git tag, or PKG-INFO in an sdist), so building from the sdist gives the right version (it gave 0.0.0); development builds are `X.Y.Z.devN`, sorting before the release they lead to. `get_version.py` is removed. (#26)
- Metadata: project URLs, the PEP 639 license fields (with Eigen's MPL-2.0 licence file in the wheel), an accurate description, `requires-python >= 3.10` (Python 3.9 is no longer supported), classifiers, a `test` extra, `__version__`, and pytest `testpaths`. (#32)
- Eigen comes from the GitHub mirror (`eigen-mirror/eigen`, the same commit), meson falls back to `pybind11.get_include()`, and stale ignores and comments are gone. (#34)
- README: `pip install scatter-smooth`, badges and the docs link; building from source with a recursive clone and meson-python. (#31)
- CI: Python 3.10–3.14 on Linux, macOS and Windows, `fail-fast: false`, a concurrency group, read-only permissions and a pip cache; the package is installed normally and the tests run from an empty directory, so the installed package is tested; current action versions. (#29)
- CI: one job runs the R comparisons with rpy2, and fails (instead of skipping) if R isn't usable (`SCATTER_SMOOTH_REQUIRE_R`). (#27)
- Wheels: Linux x86_64 and aarch64, macOS arm64 and x86_64, Windows x86_64 and ARM64, each installed and tested by cibuildwheel; built on tags as well as on demand; the sdist is built and test-installed (its version checked); tags publish to PyPI by trusted publishing. No free-threaded or 32-bit builds. (#28)
- Tests: the pure-Python reference implementations are `tests/_ref_*.py`, and `tests` is no longer a package, so the tests run from any directory against the installed package. (#24, in part)
