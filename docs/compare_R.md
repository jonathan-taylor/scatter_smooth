---
jupytext:
  formats: md:myst,ipynb
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

# Comparing `smoothing_spline` with R's `smooth.spline`

This document demonstrates the usage of the `smoothing_spline` package and compares it with the standard `smooth.spline` function in R. We will use the `Bikeshare` dataset from the `ISLP` package.

## Setup

First, we need to import the necessary libraries and load the `rpy2` extension to run R code directly in this notebook.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from ISLP import load_data
from smoothing_spline import SplineFitter

%load_ext rpy2.ipython
```

## Loading the Data

We will use the `Bikeshare` dataset, which contains daily counts of bike rentals in Washington D.C.

```{code-cell} ipython3
Bike = load_data('Bikeshare')
Bike.head()
```

We will focus on the relationship between the hour of the day (`hr`) and the number of bikers (`bikers`). Since `hr` is categorical in the original dataset but represents time, we convert it to numeric.

```{code-cell} ipython3
df = 7
# 'bikers' is 'cnt' in the original dataset, ISLP might have renamed it or we use 'cnt'
if 'bikers' not in Bike.columns:
    Bike['bikers'] = Bike['cnt']

hr_numeric = pd.to_numeric(Bike['hr'])
bikers = Bike['bikers']

# Sort by hour for cleaner plotting lines
sorted_idx = np.argsort(hr_numeric)
x_plot = hr_numeric.iloc[sorted_idx].unique()
```

## Fitting Smoothing Splines

### 1. Using `smoothing_spline` (Python)

We fit a smoothing spline with a specified degrees of freedom ($df=5$).

```{code-cell} ipython3
# Fit model
spl_py = SplineFitter(x=hr_numeric, df=df)
spl_py.fit(bikers)

# Predict
y_py = spl_py.predict(x_plot)
```

### 2. Using `smooth.spline` (R)

We fit the same model using R. We transfer the data to R and run the `smooth.spline` function.

```{code-cell} ipython3
%%R -i hr_numeric -i bikers -o y_r -i df
# Fit model in R
fit_r <- smooth.spline(hr_numeric, bikers, df=df)

# Predict at unique hours
# unique() in R returns unsorted, but we want to match x_plot order
x_vals <- sort(unique(hr_numeric))
pred_r <- predict(fit_r, x_vals)
y_r <- pred_r$y
```

### Comparison

Let's visualize the results. They should be nearly identical.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(hr_numeric + np.random.normal(0, 0.1, len(hr_numeric)), bikers, 
            s=1, c='lightgray', alpha=0.5, label='Data')
ax.plot(x_plot, y_py, 'b-', lw=3, label='Python (smoothing_spline)', alpha=0.8)
ax.plot(x_plot, y_r, 'r--', lw=3, label='R (smooth.spline)', alpha=0.8)
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Bikers")
ax.set_title("Comparison of Smoothing Splines (df={df})")
ax.legend()
plt.show()

# Numerical comparison
# Note: R might handle repeated x values slightly differently (using weights)
# smoothing_spline handles them naturally in the basis construction.
diff = np.mean(np.abs(y_py - y_r))
print(f"Mean Absolute Difference: {diff:.6f}")
```

## Speed Comparison

We will compare the execution time for fitting the model.

```{code-cell} ipython3
# Python Timing
t_py = %timeit -o -n 10 -r 3 SplineFitter(x=hr_numeric, df=10).fit(bikers)

# R Timing
```

```{code-cell} ipython3
%%R -i hr_numeric -i bikers
library(microbenchmark)
# R Timing
microbenchmark(
  smooth.spline(hr_numeric, bikers, df=10),
  times=10
)
```

## Automatic Tuning with GCV

One of the key features of smoothing splines is the automatic selection of the smoothing parameter ($\lambda$) using Generalized Cross-Validation (GCV).

### In R

R's `smooth.spline` uses GCV (or CV) by default if `df` and `spar` are not specified.

```{code-cell} ipython3
%%R
fit_gcv <- smooth.spline(hr_numeric, bikers, cv=FALSE) # cv=FALSE implies GCV
cat("Selected df (R):", fit_gcv$df, "
")
cat("Selected lambda (R):", fit_gcv$lambda, "
")
```

### In Python (smoothing_spline)

The `smoothing_spline` package also supports finding $\lambda$ that minimizes the GCV score via the `solve_gcv` method.

```{code-cell} ipython3
# Initialize fitter with data
# Note: We use the internal C++ fitter for speed if available
fitter = SplineFitter(x=hr_numeric, knots=np.unique(hr_numeric))

# Solve for GCV
best_lam = fitter.solve_gcv(bikers)
print(f"Selected lambda (Python): {best_lam}")

# Get corresponding df
# We can access the internal fitter to compute DF for verification
if fitter._cpp_fitter:
    best_lam_scaled = best_lam / fitter.x_scale_**3
    best_df = fitter._cpp_fitter.compute_df(best_lam_scaled)
    print(f"Selected df (Python): {best_df}")
else:
    best_df = "N/A (C++ extension not available)"
```

We can now visualize the optimal fit.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(hr_numeric + np.random.normal(0, 0.1, len(hr_numeric)), bikers, 
            s=1, c='lightgray')
# fitter is already fitted with best_lam by solve_gcv
ax.plot(x_plot, fitter.predict(x_plot), 'g-', lw=3, label=f'Optimal GCV (df={best_df:.2f})')
ax.set_xlabel("Hour")
ax.set_ylabel("Number of Bikers")
ax.legend()
plt.show()
```

## Another Example: Log-Transformation

The relationship between hours and bikers might be better modeled on a log scale, as counts are non-negative and variance often increases with the mean.

```{code-cell} ipython3
# Fit model on log(bikers)
log_bikers = np.log(bikers + 1) # Add 1 to avoid log(0)
spl_log = SplineFitter(x=hr_numeric, df=df)
spl_log.fit(log_bikers)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(hr_numeric + np.random.normal(0, 0.1, len(hr_numeric)), log_bikers, 
            s=1, c='lightgray')
ax.plot(x_plot, spl_log.predict(x_plot), 'purple', lw=3, label='Log-Smoothing Spline (df={df})')
ax.set_xlabel("Hour")
ax.set_ylabel("Log(Number of Bikers + 1)")
ax.legend()
plt.show()
```

## Synthetic Speed Comparison (N=500)

Finally, we compare the speed of both implementations on a synthetic dataset with 500 unique $x$ values.

```{code-cell} ipython3
# Synthetic data
n_syn = 500
x_syn = np.sort(np.random.uniform(0, 10, n_syn))
y_syn = np.sin(x_syn) + np.random.normal(0, 0.1, n_syn)

# Python Timing (using all unique x as knots)
print("Python Timing (n=500):")
%timeit SplineFitter(x=x_syn, df=10).fit(y_syn)
```

```{code-cell} ipython3
%%R -i x_syn -i y_syn
library(microbenchmark)
cat("R Timing (n=500):\n")
microbenchmark(
  smooth.spline(x_syn, y_syn, df=10),
  times=100
)
```

