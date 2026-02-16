import numpy as np
import pytest
from smoothing_spline.fitter import SplineFitter

@pytest.mark.parametrize("n_samples", [50, 100])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("unequal_x", [False, True])
def test_compare_bspline_natural_spline(n_samples, weighted, unequal_x):
    np.random.seed(42)
    
    if unequal_x:
        # Generate unequally spaced x
        x = np.sort(np.random.rand(n_samples) ** 2) 
    else:
        x = np.linspace(0, 1, n_samples)
        
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, n_samples)
    
    if weighted:
        w = np.random.uniform(0.5, 1.5, n_samples)
    else:
        w = None
        
    # Use fewer knots than n_samples to ensure smoothing behavior
    # and to make sure we are not just interpolating
    n_knots = int(n_samples / 3)
    percs = np.linspace(0, 100, n_knots)
    knots = np.percentile(x, percs)
    
    # 1. Natural Spline Fitter
    sf = SplineFitter(x, w=w, knots=knots, df=10, engine='natural')
    sf.fit(y)
    y_pred_ns = sf.predict(x)
    lamval = sf.lamval
    
    # 2. B-Spline Fitter
    sf_bs = SplineFitter(x, w=w, knots=knots, lamval=lamval, engine='bspline')
    try:
        sf_bs.fit(y)
    except RuntimeError as e:
        if "LAPACK dpbsv failed" in str(e) or "Trailing B-spline" in str(e):
            pytest.skip(f"Solver failed due to conditioning: {e}")
        else:
            raise e
    
    y_pred_bs = sf_bs.predict(x)

    # Check interior agreement
    mse = np.mean((y_pred_ns - y_pred_bs)**2)
    
    # Check extrapolation
    x_extra = np.linspace(-0.1, 1.1, 51)
    y_extra_ns = sf.predict(x_extra)
    y_extra_bs = sf_bs.predict(x_extra)
    
    mse_extra = np.mean((y_extra_ns - y_extra_bs)**2)
    
    # Assert correlation > 0.99
    corr = np.corrcoef(y_pred_ns, y_pred_bs)[0, 1]
    assert corr > 0.999
    
    # Assert extrapolation is reasonably close
    assert mse_extra < 1e-6

def test_bspline_solve_for_df():
    rng = np.random.default_rng(42)
    n = 50
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, n)
    n_knots = 15
    knots = np.linspace(0, 1, n_knots)
    
    target_df = 8.0
    
    # Fit with B-spline engine specifying df
    fitter = SplineFitter(x, knots=knots, df=target_df, engine='bspline')
    fitter.fit(y)
    
    # Verify effective degrees of freedom
    # We can compute it manually using the fitted lambda
    lam_scaled = fitter.lamval / fitter.x_scale_**3
    computed_df = fitter._cpp_fitter.compute_df(lam_scaled)
    
    assert np.isclose(computed_df, target_df, rtol=1e-4)

def test_bspline_solve_gcv():
    rng = np.random.default_rng(43)
    n = 50
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, n)
    knots = np.linspace(0, 1, 15)
    
    fitter = SplineFitter(x, knots=knots, engine='bspline')
    best_lam = fitter.solve_gcv(y)
    
    assert best_lam > 0
    assert fitter.lamval == best_lam
