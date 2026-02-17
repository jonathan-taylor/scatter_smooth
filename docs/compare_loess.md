---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Comparing `LoessSmoother` with R's `loess`

This document demonstrates the usage of the `LoessSmoother` class in `scatter_smooth` and compares it with the standard `loess` function in R.

## Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scatter_smooth.datasets import load_bikeshare
from scatter_smooth import LoessSmoother

%load_ext rpy2.ipython
```

## Loading the Data

We use the same `Bikeshare` dataset as in the Spline comparison.

```{code-cell} ipython3
Bike = pd.read_csv(load_bikeshare())
if 'bikers' not in Bike.columns:
    Bike['bikers'] = Bike['cnt']

hr_numeric = pd.to_numeric(Bike['hr'])
bikers = Bike['bikers']

# Subsample for performance (Naive Loess is O(N^2))
n_samples = 500
idx = np.random.choice(len(hr_numeric), n_samples, replace=False)
x_sub = hr_numeric.iloc[idx].values
y_sub = bikers.iloc[idx].values

# Sort for plotting
sort_idx = np.argsort(x_sub)
x_sub = x_sub[sort_idx]
y_sub = y_sub[sort_idx]

x_plot = np.linspace(x_sub.min(), x_sub.max(), 100)
```

## Fitting Loess Models

### 1. Using `scatter_smooth` (Python)

We fit a Loess smoother with `span=0.75` and `degree=1` (linear local regression).

```{code-cell} ipython3
# Fit model
loess_py = LoessSmoother(x=x_sub, span=0.75, degree=1)
loess_py.smooth(y_sub)

# Predict
y_py = loess_py.predict(x_plot)
```

### 2. Using `loess` (R)

We fit the same model using R. Note that R's `loess` defaults to `degree=2`. We set `degree=1` to match our Python model. We also use `family="gaussian"` to ensure least-squares fitting (no robust iterations).

```{code-cell} ipython3
%%R -i x_sub -i y_sub -i x_plot -o y_r
# Fit model in R
# surface="direct" calculates exact values (slow but accurate comparison)
fit_r <- loess(y_sub ~ x_sub, span=0.75, degree=1, family="gaussian", surface="direct")

# Predict
y_r <- predict(fit_r, newdata=x_plot)
```

### Comparison

Let's visualize the results.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_sub, y_sub, s=10, c='lightgray', alpha=0.7, label='Data')
ax.plot(x_plot, y_py, 'b-', lw=3, label='Python (LoessSmoother)', alpha=0.8)
ax.plot(x_plot, y_r, 'r--', lw=3, label='R (loess)', alpha=0.8)
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Bikers")
ax.set_title("Comparison of Loess Smoothing (span=0.75, degree=1)")
ax.legend()
plt.show()

# Numerical comparison
# NaNs might appear at boundaries if weights are 0, handle them
valid = ~np.isnan(y_py) & ~np.isnan(y_r)
diff = np.mean(np.abs(y_py[valid] - y_r[valid]))
print(f"Mean Absolute Difference: {diff:.6f}")
```

## Higher Degree Polynomials

We can also fit a local quadratic model (`degree=2`).

```{code-cell} ipython3
# Python
degree = 2
loess_py_deg2 = LoessSmoother(x=x_sub, span=0.75, degree=degree)
loess_py_deg2.smooth(y_sub)
y_py_deg2 = loess_py_deg2.predict(x_plot)

# R
```

```{code-cell} ipython3
%%R -i x_sub -i y_sub -i x_plot -o y_r_deg2 -i degree
fit_r_deg2 <- loess(y_sub ~ x_sub, span=0.75, degree=degree, family="gaussian", surface="direct")
y_r_deg2 <- predict(fit_r_deg2, newdata=x_plot)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_sub, y_sub, s=10, c='lightgray', alpha=0.7)
ax.plot(x_plot, y_py_deg2, 'b-', lw=3, label=f'Python (degree={degree})')
ax.plot(x_plot, y_r_deg2, 'r--', lw=3, label=f'R (degree={degree})')
ax.legend()
plt.title(f"Comparison of Loess (degree={degree})")
plt.show()

valid = ~np.isnan(y_py_deg2) & ~np.isnan(y_r_deg2)
diff = np.mean(np.abs(y_py_deg2[valid] - y_r_deg2[valid]))
print(f"Mean Absolute Difference (Degree 2): {diff:.6f}")
```

## Speed Comparison

We compare the execution time for fitting and predicting with the Loess model. The `scatter_smooth` implementation uses an optimized C++ backend. We compare it against R's `loess` with `surface="direct"` (exact calculation).

```{code-cell} ipython3
%%timeit
loess_py = LoessSmoother(x=x_sub, span=0.75, degree=1)
loess_py.smooth(y_sub); 
loess_py.predict(x_plot)
```

```{code-cell} ipython3
%%R -i x_sub -i y_sub -i x_plot
if (!require("microbenchmark", quietly = TRUE)) {
    install.packages("microbenchmark", repos="http://cloud.r-project.org")
}
library(microbenchmark)

cat(sprintf("R Timing (n=%d, surface='direct'):\n", length(x_sub)))
summary(microbenchmark(
  {
      fit <- loess(y_sub ~ x_sub, span=0.75, degree=1, family="gaussian", surface="direct")
      predict(fit, newdata=x_plot)
  },
  times=10), unit='milliseconds')[,-1]
```

## Larger Scale Comparison (N=5000)

We increase the sample size to 5000 to see the performance difference more clearly.

```{code-cell} ipython3
n_large = 10000
x_large = np.sort(np.random.uniform(0, 10, n_large))
y_large = np.sin(x_large) + np.random.normal(0, 1, n_large)
x_plot_large = np.linspace(0, 10, 200)
```

The implementation here doesn't compute fitted values at 
the original `x` unless needed.

```{code-cell} ipython3
%%timeit 
loess_cpp_large = LoessSmoother(x_large, span=0.75, degree=1)
loess_cpp_large.smooth(y_large)
loess_cpp_large.predict(x_plot_large)
```

If we include the cost of predicting at all of `x_large` we see the
price:

```{code-cell} ipython3
%%timeit 
loess_cpp_large = LoessSmoother(x_large, span=0.75, degree=1)
loess_cpp_large.smooth(y_large)
loess_cpp_large.predict(x_plot_large)
loess_cpp_large.predict(x_large)
```

```{code-cell} ipython3
%%R -i n_large -i y_large -i x_large -i x_plot_large

cat(sprintf("\nR Timing (N=%d):\n", n_large))
summary(microbenchmark(
  {
      fit <- loess(y_large ~ x_large, span=0.75, degree=1, family="gaussian", surface="direct")
      predict(fit, newdata=x_plot_large)
  },
  times=5), unit='milliseconds')[,-1]
```

## Let's try quadratic as well

```{code-cell} ipython3
%%timeit 
loess_cpp_large = LoessSmoother(x_large, span=0.75, degree=2)
loess_cpp_large.smooth(y_large)
loess_cpp_large.predict(x_plot_large)
```

Including all the values is noticably slower

```{code-cell} ipython3
%%timeit 
loess_cpp_large = LoessSmoother(x_large, span=0.5, degree=2)
loess_cpp_large.smooth(y_large)
loess_cpp_large.predict(x_plot_large)
loess_cpp_large.predict(x_large)
```

```{code-cell} ipython3
loess_cpp_large = LoessSmoother(x_large, span=0.5, degree=2)
loess_cpp_large.smooth(y_large)
y_plot_py = loess_cpp_large.predict(x_plot_large)
```

```{code-cell} ipython3
%%R -o y_plot_r -i degree

fit <- loess(y_large ~ x_large, span=0.5, degree=2, family="gaussian", surface="direct")
y_plot_r <- predict(fit, newdata=x_plot_large)
summary(microbenchmark(
  {
      fit <- loess(y_large ~ x_large, 
                   span=0.75, 
                   degree=2, 
                   family="gaussian", 
                   surface="direct")
      predict(fit, newdata=x_plot_large)
  },
  times=5), unit='seconds')[,-1]
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x_large, y_large, s=10, c='lightgray', alpha=0.7)
ax.plot(x_plot_large, y_plot_py, 'b-', lw=3, label=f'Python (degree=2)')
ax.plot(x_plot_large, y_plot_r, 'r--', lw=3, label=f'R (degree=2)')
ax.legend()
plt.title(f"Comparison of Loess (degree={degree})")
plt.show()

valid = ~np.isnan(y_plot_py) & ~np.isnan(y_plot_r)
diff = np.mean(np.abs(y_plot_py[valid] - y_plot_r[valid]))
print(f"Mean Absolute Difference (Degree 2): {diff:.6f}")
```

```{code-cell} ipython3

```
