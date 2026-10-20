---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Performance of the Engines

This document compares the computational performance of the C++ engines of `SplineSmoother`: the Reinsch algorithm (`engine='reinsch'`, all the unique $x$ values as knots, O(N)) and the B-spline basis (`engine='bspline'`, any knots). The pure Python reference implementation that the tests compare against lives in `tests/_ref_spline_fitter.py`.

## Setup

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import time

from scatter_smooth import SplineSmoother
```

## Speed Comparison at Different Scales

We will measure the time taken to fit the smoothing spline for different numbers of observations ($N$) and different numbers of knots ($K$).

```{code-cell} ipython3
def benchmark(ns, engine, **knots):
    times = []
    for n in ns:
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 10, n))
        y = np.sin(x) + rng.normal(0, 0.1, n)
        start = time.time()
        fitter = SplineSmoother(x, df=10, engine=engine, **knots)
        fitter.smooth(y)
        times.append(time.time() - start)
    return times

# Sizes to test
ns = [100, 500, 1000, 2000, 5000]
```

### 1. All Unique X as Knots (K = N)

With `all_knots=True` every unique $x$ value is a knot, which is feasible for moderate $N$.
We compare:
1. `engine='reinsch'` (Reinsch Algorithm - O(N))
2. `engine='bspline'` (B-Spline Basis)

```{code-cell} ipython3
print(f"Benchmarking with ns={ns} (All Knots)...")
results_reinsch = benchmark(ns, 'reinsch', all_knots=True)
results_bspline = benchmark(ns, 'bspline', all_knots=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ns, results_reinsch, '^-', label="'reinsch'")
ax.plot(ns, results_bspline, 'd-', label="'bspline'")

ax.set_xlabel('Number of observations (N)')
ax.set_ylabel('Time (seconds)')
ax.set_title('Speed Comparison (K = N)')
ax.legend()
ax.grid(True)
ax.set_yscale('log')
```

### 2. Fixed Number of Knots (K=200)

In practice, for large $N$, we often limit the number of knots to a fixed $K \ll N$.
`engine='reinsch'` is not available here (requires K=N). We demonstrate time required for `bspline` as $n$ increases.

```{code-cell} ipython3
ns_large = [100, 500, 1000, 1500, 2000, 5000, 10000, 20000, 30000, 50000]
K = 200
print(f"Benchmarking with large N and K={K}...")
results_fixed_bspline = benchmark(ns_large, 'bspline', n_knots=K)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ns_large, results_fixed_bspline, 'd-', label=f"'bspline' (K={K})")

ax.set_xlabel('Number of observations (N)')
ax.set_ylabel('Time (seconds)')
ax.set_title(f'Speed Comparison (Fixed K={K})')
ax.legend()
ax.grid(True)
ax.set_yscale('log')
```

## GCV Solve Performance

Automatic tuning with GCV involves multiple fits (or an optimized path). The C++ extension provides a highly optimized `solve_gcv` method.

```{code-cell} ipython3
n = 5000
K = 200
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 10, n))
y = np.sin(x) + rng.normal(0, 0.1, n)
    
```

```{code-cell} ipython3
%%timeit
fitter = SplineSmoother(x, n_knots=K)
best_lam = fitter.solve_gcv(y)
```

```{code-cell} ipython3
%%timeit
fitter = SplineSmoother(x)
best_lam = fitter.solve_gcv(y)
```

## Conclusion

With all the unique $x$ values as knots, the Reinsch engine is O(N). With a fixed number of knots $K \ll N$, the B-spline engine's cost grows slowly with $N$: the banded system is of size $K$.
