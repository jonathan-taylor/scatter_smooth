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

# Scatterplot Smoothers Documentation

This repository provides minimal and efficient implementations of key scatterplot smoothers, specifically **Smoothing Splines** and **LOESS**.

The core logic is implemented in C++ with Python bindings provided by `pybind11`.

### Key Smoothers

*   **Smoothing Splines**: Similar to `smooth.spline` in R, offering multiple fitting engines:
    *   **Reinsch Algorithm**: $O(N)$ performance for when knots equal data points (matches R's `smooth.spline`).
    *   **Natural Spline Basis**: Explicit basis construction, suitable for regression splines ($K < N$).
    *   **B-Spline Basis**: Efficient banded solver implementation using LAPACK.
*   **LOESS**: A fast C++ implementation of locally estimated scatterplot smoothing (local polynomial regression).

See the table of contents for more details on the theory and comparisons with other implementations.
