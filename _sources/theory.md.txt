# Smoothing Splines: Theory and Implementation

This document outlines the theoretical background for the `smoothing_spline` package, synthesizing concepts from *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), specifically Chapter 5. It also discusses the implementation details (Reinsch form vs. Basis form) and compares the features with R's `smooth.spline`.

## 1. The Smoothing Spline Problem

The goal of smoothing splines is to find a function $f(x)$ that fits the data $\{(x_i, y_i)\}_{i=1}^N$ well while remaining smooth. This is formulated as the minimization of a penalized residual sum of squares:

$$
\min_{f} \sum_{i=1}^N (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 dt
$$

where $\lambda \ge 0$ is a smoothing parameter.
- $\lambda = 0$: $f$ becomes an interpolating spline (passes through every point).
- $\lambda 	o \infty$: $f$ approaches the linear least squares fit.

### The Solution: Natural Cubic Spline

Remarkably, the solution to this infinite-dimensional optimization problem is a finite-dimensional **Natural Cubic Spline** with knots at the unique values of $x_i$.

A natural cubic spline is a piecewise cubic polynomial that is continuous up to its second derivative, and linear beyond the boundary knots (i.e., its second derivative is zero at the boundaries).

## 2. Matrix Formulation

Since the solution is a linear combination of basis functions, we can write:
$$
f(x) = \sum_{j=1}^N 	heta_j N_j(x)
$$
where $N_j(x)$ are the natural cubic spline basis functions.

The optimization problem reduces to:
$$
\min_{	heta} (y - N	heta)^T (y - N	heta) + \lambda 	heta^T \Omega_N 	heta
$$

where:
- $N_{ij} = N_j(x_i)$ is the basis matrix.
- $(\Omega_N)_{jk} = \int N_j''(t) N_k''(t) dt$ is the penalty matrix.

The solution is a generalized ridge regression:
$$
\hat{	heta} = (N^T N + \lambda \Omega_N)^{-1} N^T y
$$

### Reinsch Form vs. Basis Form

There are two primary ways to compute the solution.

#### A. The Basis Form (Used in this Package)
We explicitly construct the basis matrix $N$ and the penalty matrix $\Omega$.
- We use the property that a natural cubic spline is determined by its values and second derivatives at the knots.
- Our C++ implementation constructs $N$ efficiently by solving the tridiagonal system that relates second derivatives to values.
- We then solve the dense linear system $(N^T N + \lambda \Omega)	heta = N^T y$.

**Why Basis Form?**
- Allows for **Regression Splines**: We can choose fewer knots than data points ($K < N$). The math remains identical, but the matrices are smaller ($K 	imes K$ instead of $N 	imes N$).
- Easier to extend to weighted least squares or other loss functions.

#### B. The Reinsch Form (Used in R's `smooth.spline`)
If knots are placed at every data point ($K=N$), the solution vector $\mathbf{f} = (f(x_1), \dots, f(x_N))^T$ can be found directly without forming $N$.
$$
\hat{\mathbf{f}} = (I + \lambda K)^{-1} y
$$
where $K = Q R^{-1} Q^T$.
- $Q$ is an $N 	imes (N-2)$ tridiagonal matrix of second differences ($1/h_i$).
- $R$ is an $(N-2) 	imes (N-2)$ tridiagonal matrix ($h_i$).

This allows solving the system in $O(N)$ time using banded solvers, making it extremely fast for large $N$.

**Our Implementation Note:**
While we use the basis form, our construction of $\Omega$ utilizes the exact same $Q R^{-1} Q^T$ structure (computed via sparse Cholesky) to ensure mathematical equivalence to the integral penalty.

## 3. Degrees of Freedom ($df$)

The smoothing parameter $\lambda$ is abstract. A more intuitive measure of model complexity is the **effective degrees of freedom ($df$)**.

$$
df(\lambda) = 	ext{trace}(\mathbf{S}_\lambda)
$$

where $\mathbf{S}_\lambda$ is the smoother matrix such that $\hat{y} = \mathbf{S}_\lambda y$.
In our formulation:
$$
\mathbf{S}_\lambda = N (N^T N + \lambda \Omega)^{-1} N^T
$$

The relationship between $df$ and $\lambda$ is monotonic.
- $df 	o 2$ as $\lambda 	o \infty$ (Linear fit).
- $df 	o N$ as $\lambda 	o 0$ (Interpolation).

Our package allows specifying `df` directly. We use a root-finding algorithm (Brent's method) to find the unique $\lambda$ that yields the requested $df$.

## 4. Comparison with R's `smooth.spline`

| Feature | `smooth.spline` (R) | `smoothing_spline` (Python) |
| :--- | :--- | :--- |
| **Algorithm** | Reinsch Form ($O(N)$) | Basis Form ($O(NK^2)$ or $O(K^3)$) |
| **Knots** | All unique $x$ (default) or $nknots$ | All unique $x$ or specified `n_knots` |
| **Input $\lambda$** | Via `spar` or `lambda` | Via `lamval` |
| **Input $df$** | Supported | Supported (via root finding) |
| **Automatic Tuning** | GCV / CV | **Not yet implemented** |
| **Weights** | Supported | Supported |
| **Derivatives** | Built-in | **Not yet exposed** (Backend supports it) |
| **Extrapolation** | Linear (via `predict`) | Linear (explicitly handled) |

### Key Differences
1.  **Speed:** For $N=10,000$, `smooth.spline` is faster because it exploits the band structure of the full system. Our implementation allows `n_knots < N` (Regression Splines), which restores speed ($O(N K^2)$) and reduces overfitting risk, a feature `smooth.spline` supports via `nknots`.
2.  **Spar (R specific):** R uses a scaling parameter `spar` related to $\lambda$. We use raw $\lambda$ (scaled by $x$-range cubed for numerical stability).

## 5. References

1.  Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd Edition. Springer. (Chapter 5).
2.  Green, P. J., & Silverman, B. W. (1994). *Nonparametric Regression and Generalized Linear Models: A Roughness Penalty Approach*. Chapman and Hall/CRC.
