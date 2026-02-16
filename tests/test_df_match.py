import numpy as np
from smoothing_spline.fitter import SplineFitter

def test_bspline_df_takahashi():
    np.random.seed(42)
    n = 100
    x = np.sort(np.random.rand(n))
    y = np.sin(10 * x) + np.random.normal(0, 0.1, n)
    
    # Init with bspline engine
    spline = SplineFitter(x, engine='bspline', order=4, n_knots=10)
    
    lam = 1e-4
    df_takahashi = spline.compute_df(lam)
    
    # Compare with C++ implementation (which uses the slow method currently)
    # We need to call the C++ compute_df directly or use engine='natural' (if compatible)
    # but natural has different basis.
    
    # We can use the C++ fitter directly from the spline object
    lam_scaled = lam / (spline.x_scale_**3)
    df_cpp_slow = spline._cpp_fitter.compute_df(lam_scaled)
    
    print(f"DF Takahashi: {df_takahashi}")
    print(f"DF CPP Slow:  {df_cpp_slow}")
    
    assert np.allclose(df_takahashi, df_cpp_slow), f"DF mismatch: {df_takahashi} != {df_cpp_slow}"
    print("SUCCESS: DF calculation matches.")

if __name__ == "__main__":
    test_bspline_df_takahashi()
