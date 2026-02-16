---
jupytext:
  main_language: python
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
---

# Smoothing Splines: Theory and Implementation

This document outlines the theoretical background for the `smoothing_spline` package, synthesizing concepts from *The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman), specifically Chapter 5. It also discusses the implementation details (Reinsch form vs. Basis form) and compares the features with R's `smooth.spline`.

## 1. The Smoothing Spline Problem

The goal of smoothing splines is to find a function $f(x)$ that fits the data $\{(x_i, y_i)\}_{i=1}^N$ well while remaining smooth. This is formulated as the minimization of a penalized residual sum of squares:

$$
\min_{f} \sum_{i=1}^N w_i (y_i - f(x_i))^2 + \lambda \int [f''(t)]^2 dt
$$

where $\lambda \ge 0$ is a smoothing parameter and $w_i > 0$ are weights.
- $\lambda = 0$: $f$ becomes an interpolating spline (passes through every point).
- $\lambda \to \infty$: $f$ approaches the weighted linear least squares fit.

### The Solution: Natural Cubic Spline

Remarkably, the solution to this infinite-dimensional optimization problem is a finite-dimensional **Natural Cubic Spline** with knots at the unique values of $x_i$.

A natural cubic spline is a piecewise cubic polynomial that is continuous up to its second derivative, and linear beyond the boundary knots (i.e., its second derivative is zero at the boundaries).

## 2. Matrix Formulation

Since the solution is a linear combination of basis functions, we can write:
$$
f(x) = \sum_{j=1}^N \theta_j N_j(x)
$$
where $N_j(x)$ are the natural cubic spline basis functions.

The optimization problem reduces to:
$$
\min_{ \theta} (y - N \theta)^T W (y - N \theta) + \lambda \theta^T \Omega_N \theta
$$

where:
- $N_{ij} = N_j(x_i)$ is the basis matrix.
- $W = \text{diag}(w_1, \dots, w_N)$ is the weight matrix.
- $(\Omega_N)_{jk} = \int N_j''(t) N_k''(t) dt$ is the penalty matrix.

The solution is a generalized ridge regression:
$$
\hat{\theta} = (N^T W N + \lambda \Omega_N)^{-1} N^T W y
$$

### Reinsch Form vs. Basis Form

There are three primary ways to compute the solution in our package:

#### A. The Natural Spline Basis Form (`engine='natural'`)
We explicitly construct the basis matrix $N$ and the penalty matrix $\Omega$.
- We use the property that a natural cubic spline is determined by its values and second derivatives at the knots.
- Our C++ implementation constructs $N$ efficiently by solving the tridiagonal system that relates second derivatives to values.
- We then solve the linear system $(N^T W N + \lambda \Omega)\theta = N^T W y$.
- Use of this is not recommended as it is much slower. It was written as part of the development process.

+++

**Why Basis Form?**
- Allows for **Regression Splines**: We can choose fewer knots than data points ($K < N$). The math remains identical, but the matrices are smaller ($K \times K$ instead of $N \times N$).
- Easier to extend to weighted least squares or other loss functions.

#### B. The Reinsch Form (`engine='reinsch'`)
If knots are placed at every data point ($K=N$), the solution vector $\mathbf{f} = (f(x_1), \dots, f(x_N))^T$ can be found directly without forming $N$.
$$
\hat{\mathbf{f}} = (W + \lambda K)^{-1} W y
$$
where $K = Q R^{-1} Q^T$.
- $Q$ is an $N \times (N-2)$ tridiagonal matrix of second differences.
- $R$ is an $(N-2) \times (N-2)$ tridiagonal matrix.

This allows solving the system in $O(N)$ time using banded solvers, making it extremely fast for large $N$. This matches the algorithm used in R's `smooth.spline`.

#### C. The B-Spline Basis Form (`engine='bspline'`)
Uses the B-spline basis with compact support.
- Constructs banded matrices for the normal equations $N^T W N$ and $\Omega_N$.
- Solves using LAPACK's `dpbsv` via `scipy`. This is the only part `scipy` is used so direct calls to LAPACK could in theory be used.
- Efficient for both $K=N$ and $K<N$.

#### D. Automatic (`engine='auto'`)
Uses Reinsch form if possible, else uses the B-Spline form.

## 3. Degrees of Freedom ($df$)

The smoothing parameter $\lambda$ is abstract. A more intuitive measure of model complexity is the **effective degrees of freedom ($df$)**.

$$
df(\lambda) = \text{trace}(\mathbf{S}_\lambda)
$$

where $\mathbf{S}_\lambda$ is the smoother matrix such that $\hat{y} = \mathbf{S}_\lambda y$.

For the Reinsch form, we compute the trace in $O(N)$ time using Takahashi's equations (calculating selected elements of the inverse of a banded matrix via Cholesky decomposition).

For the Basis form (Natural), we compute the trace of the dense matrix inverse.

For the B-spline form, we also use Takahashi's equations on the banded system to compute the trace efficiently in $O(N)$ time.

## 3.1. Details on Degrees of Freedom Computation

### Trace of Inverse via Takahashi's Algorithm

Computing the degrees of freedom requires the trace of the smoother matrix $S_\lambda$. In the basis form:
$$
\hat{y} = N \hat{\theta} = N (N^T W N + \lambda \Omega_N)^{-1} N^T W y
$$
Thus, $S_\lambda = N (N^T W N + \lambda \Omega_N)^{-1} N^T W$. Using the cyclic property of the trace:
$$
\text{df}(\lambda) = \text{trace}(S_\lambda) = \text{trace}((N^T W N + \lambda \Omega_N)^{-1} N^T W N)
$$
Let $A = N^T W N$. We need to compute $\text{trace}((A + \lambda \Omega_N)^{-1} A)$.

For banded matrices (like those in our B-spline implementation), we can compute this trace efficiently in $O(N)$ time without inverting the full matrix.

Let $H = A + \lambda \Omega_N$. We need $\text{trace}(H^{-1} A)$.
Since $A$ and $\Omega_N$ are banded, $H$ is banded.
Let $H = U^T U$ be the Cholesky decomposition of $H$, where $U$ is an upper triangular banded matrix.
We need elements of $Z = H^{-1}$ specifically those where $A$ is non-zero. Since $A$ is banded, we only need the elements of $Z$ within the bandwidth of $A$.

**Takahashi's Equations** provide a way to compute the elements of $Z$ within the band of $H$ (which includes the band of $A$) using a backward recurrence starting from the bottom-right corner.

1.  Compute $Z_{NN} = 1 / U_{NN}^2$.
2.  Iterate backwards from $i = N-1$ to $1$:
    *   Compute off-diagonal elements $Z_{ij}$ inside the band using previously computed $Z$ values and $U$.
    *   Compute diagonal element $Z_{ii}$.

Once we have the band of $Z$, we can compute the trace of the product $Z A$ efficiently:
$$
\text{trace}(Z A) = \sum_{i,j} Z_{ij} A_{ji}
$$
Since $A$ is symmetric and banded, the sum only involves indices $(i,j)$ where $|i-j| \le \text{bandwidth}$, which are exactly the elements we computed.

### Handling Natural Boundary Conditions (The Projection Matrix $P$)

The "natural" boundary conditions (zero second derivative at endpoints) imply linear constraints on the B-spline coefficients $\theta$.
$$
\theta_0 = w_{s1} \theta_1 + w_{s2} \theta_2
$$
$$
\theta_{N-1} = w_{e1} \theta_{N-2} + w_{e2} \theta_{N-3}
$$
These constraints reduce the number of free parameters from $N$ to $N-2$.
We can express the full parameter vector $\theta$ as a linear transformation of the reduced parameters $\gamma = (\theta_1, \dots, \theta_{N-2})^T$:
$$
\theta = P \gamma
$$
where $P$ is an $N \times (N-2)$ sparse projection matrix. $P$ is mostly an identity matrix (shifted), with the first row and last row having non-zero entries in the first two and last two columns respectively.

The optimization problem in terms of $\gamma$ becomes:
$$
\min_{\gamma} (y - N P \gamma)^T W (y - N P \gamma) + \lambda \gamma^T P^T \Omega_N P \gamma
$$
The normal equations are:
$$
(P^T N^T W N P + \lambda P^T \Omega_N P) \gamma = P^T N^T W y
$$
Let $\tilde{A} = P^T N^T W N P$ and $\tilde{\Omega} = P^T \Omega_N P$. These matrices remain banded (with slightly modified corners).
We solve for $\gamma$ and then reconstruct $\theta = P \gamma$.

Our `bspline` engine implements this by:
1.  Constructing the full banded matrices $N^T W N$ and $\Omega_N$.
2.  Applying the projection $P$ in-place to the bands (modifying the top-left and bottom-right blocks).
3.  Solving the reduced banded system.
4.  Using Takahashi's algorithm on the reduced system to compute the degrees of freedom efficiently.

## 4. Comparison with R's `smooth.spline`

| Feature | `smooth.spline` (R) | `smoothing_spline` (Python) |
| :--- | :--- | :--- |
| **Algorithm** | Reinsch Form ($O(N)$) | `engine='auto'` (selects best), `engine='reinsch'` ($O(N)$), `engine='natural'` ($O(NK^2)$), `engine='bspline'` (Banded) |
| **Knots** | All unique $x$ (default) or $nknots$ | All unique $x$ or specified `n_knots` |
| **Input $\lambda$** | Via `spar` or `lambda` | Via `lamval` |
| **Input $df$** | Supported | Supported (via root finding) |
| **Automatic Tuning** | GCV / CV | GCV |
| **Weights** | Supported | Supported |
| **Derivatives** | Built-in | Exposed via `predict(deriv=...)` |
| **Extrapolation** | Linear (via `predict`) | Linear (explicitly handled) |

### Key Differences
1.  **Speed:** For $N=10,000$, `smooth.spline` is faster because it exploits the band structure of the full system. Our implementation allows `n_knots < N` (Regression Splines), which restores speed ($O(N K^2)$) and reduces overfitting risk, a feature `smooth.spline` supports via `nknots`.
    - `engine='reinsch'`: Fast $O(N)$ implementation for when knots = unique $x$. Matches `smooth.spline` performance.
    - `engine='natural'`: Uses the natural cubic spline basis. Good for general use and when $K < N$.
    - `engine='bspline'`: Uses B-spline basis. Efficient and stable, good for sparse matrices.
2.  **Spar (R specific):** R uses a scaling parameter `spar` related to $\lambda$. We use raw $\lambda$ (scaled by $x$-range cubed for numerical stability).

## 5. References

1.  Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd Edition. Springer. (Chapter 5).
2.  Green, P. J., & Silverman, B. W. (1994). *Nonparametric Regression and Generalized Linear Models: A Roughness Penalty Approach*. Chapman and Hall/CRC.
