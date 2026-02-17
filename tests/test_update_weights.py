import numpy as np
import pytest
from scatter_smooth import SplineSmoother, LoessSmoother

def test_loess_update_weights():
    rng = np.random.default_rng(42)
    n = 50
    x = np.linspace(0, 10, n)
    y = np.sin(x) + rng.normal(0, 0.2, n)
    w1 = np.ones(n)
    w2 = rng.uniform(0.1, 2.0, n)
    
    # Case 1: Initial fit with w1
    loess1 = LoessSmoother(x=x, w=w1, span=0.5)
    loess1.smooth(y)
    pred1 = loess1.predict(x)
    
    # Case 2: Update weights to w2 and refit
    loess1.update_weights(w2)
    loess1.smooth(y) 
    pred2_updated = loess1.predict(x)
    
    # Case 3: Fresh fit with w2
    loess2 = LoessSmoother(x=x, w=w2, span=0.5)
    loess2.smooth(y)
    pred2_fresh = loess2.predict(x)
    
    np.testing.assert_allclose(pred2_updated, pred2_fresh, err_msg="LoessSmoother update_weights failed to match fresh fit")
    assert not np.allclose(pred1, pred2_updated), "Updating weights should change the result"

    # Test via smooth(sample_weight=...)
    loess3 = LoessSmoother(x=x, w=w1, span=0.5)
    loess3.smooth(y, sample_weight=w2)
    pred3 = loess3.predict(x)
    np.testing.assert_allclose(pred3, pred2_fresh, err_msg="LoessSmoother smooth(sample_weight=...) failed")


@pytest.mark.parametrize("engine", ["bspline", "natural", "reinsch"])
def test_spline_update_weights(engine):
    rng = np.random.default_rng(42)
    n = 50
    x = np.sort(rng.uniform(0, 10, n))
    
    # Ensure unique x for Reinsch
    if engine == "reinsch":
        x = np.linspace(0, 10, n)
        knots = x
    else:
        knots = None # let it pick default
        
    y = np.sin(x) + rng.normal(0, 0.2, n)
    w1 = np.ones(n)
    w2 = rng.uniform(0.1, 2.0, n)
    
    # Use fixed lamval to isolate the effect of weight update.
    # If we used df, the fresh fit would calculate a different lamval for w2,
    # while update_weights keeps the original lamval.
    # Note: lamval is unscaled (dependent on x scale). x is [0, 10], so scale=10.
    kwargs = {'x': x, 'w': w1, 'lamval': 1.0, 'engine': engine}
    if engine == "reinsch":
        kwargs['knots'] = knots
        
    spline1 = SplineSmoother(**kwargs)
    spline1.smooth(y)
    pred1 = spline1.predict(x)
    
    # Update weights via update_weights()
    spline1.update_weights(w2)
    spline1.smooth(y)
    pred2_updated = spline1.predict(x)
    
    # Fresh fit
    kwargs['w'] = w2
    spline2 = SplineSmoother(**kwargs)
    spline2.smooth(y)
    pred2_fresh = spline2.predict(x)
    
    # Comparison
    np.testing.assert_allclose(pred2_updated, pred2_fresh, rtol=1e-5, atol=1e-8, 
                               err_msg=f"SplineSmoother ({engine}) update_weights failed")
    
    # Weights should matter
    assert not np.allclose(pred1, pred2_updated), f"Updating weights should change the result ({engine})"

    # Test via smooth(sample_weight=...)
    # Reset
    kwargs['w'] = w1
    spline3 = SplineSmoother(**kwargs)
    spline3.smooth(y, sample_weight=w2)
    pred3 = spline3.predict(x)
    
    np.testing.assert_allclose(pred3, pred2_fresh, rtol=1e-5, atol=1e-8,
                               err_msg=f"SplineSmoother ({engine}) smooth(sample_weight=...) failed")

def test_reinsch_ties():
    rng = np.random.default_rng(42)
    x = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
    y = x + rng.normal(0, 0.1, 6)
    w1 = np.ones(6)
    w2 = np.array([2.0, 0.5, 2.0, 0.5, 2.0, 0.5])

    # Reinsch with ties
    # Reinsch is only used if knots match unique x.
    knots = np.unique(x)
    kwargs = {'x': x, 'w': w1, 'lamval': 1.0, 'engine': 'reinsch', 'knots': knots}
    
    spline1 = SplineSmoother(**kwargs)
    spline1.smooth(y)
    pred1 = spline1.predict(x)
    
    # Update weights
    spline1.update_weights(w2)
    spline1.smooth(y)
    pred2 = spline1.predict(x)
    
    # Fresh fit
    kwargs['w'] = w2
    spline2 = SplineSmoother(**kwargs)
    spline2.smooth(y)
    pred3 = spline2.predict(x)
    
    np.testing.assert_allclose(pred2, pred3, rtol=1e-5, err_msg="ReinschSmoother with ties failed update_weights")
    assert not np.allclose(pred1, pred2), "Updating weights should change the result"
