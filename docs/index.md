# Smoothing Spline Documentation

This repository provides a minimal and efficient implementation of a smoothing spline, similar to `smooth.spline` in R.

It is implemented in C++ with python bindings provided by `pybind11`, offering multiple fitting engines:
*   **Reinsch Algorithm**: $O(N)$ performance for when knots equal data points (matches R's `smooth.spline`).
*   **Natural Spline Basis**: Explicit basis construction, suitable for regression splines ($K < N$).
*   **B-Spline Basis**: Efficient banded solver implementation using LAPACK.

See the table of contents for more details on the theory and comparisons with other implementations.

