# Scatterplot Smoothers

This repository provides minimal, high-performance implementations of key scatterplot smoothers, specifically **Smoothing Splines** and **LOESS**. The core logic is implemented in C++ for speed, with convenient Python bindings provided by `pybind11`. The package is designed to be familiar to users of `scikit-learn` and R's statistical functions.

## Key Smoothers

*   **Smoothing Splines:** A flexible implementation similar to R's `smooth.spline`. It supports multiple fitting engines, including the Reinsch algorithm for $O(N)$ performance, as well as B-spline and natural spline bases.
*   **LOESS (Locally Estimated Scatterplot Smoothing):** A fast C++ implementation of local polynomial regression, providing a smooth curve through a scatterplot.

## Key Features

*   **High Performance:** Core computations are in C++, leveraging the Eigen library for efficient linear algebra.
*   **Flexible Smoothing Control:**
    *   For Splines: Specify smoothness by either degrees of freedom (`df`) or a penalty term (`lambda`). Includes automatic GCV.
    *   For LOESS: Control smoothness via the `span` and `degree` parameters.
*   **Extrapolation & Derivatives:** Supports linear extrapolation and computation of derivatives.
*   **R Compatibility:** Tested against R's `smooth.spline` and `loess` for comparable results.

## Installation

For standard usage, you can install the package directly from this repository using `pip`:

```bash
pip install .
```

For development, you can install the package in editable mode with the `--no-build-isolation` flag:

```bash
pip install -e . --no-build-isolation
```


## Usage Example

Here is a simple example of how to fit a smoothing spline and a LOESS curve to noisy data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scatter_smooth import SplineSmoother, LoessSmoother

# 1. Generate some noisy data
rng = np.random.default_rng(0)
x = np.sort(rng.uniform(0, 1, 100))
y_true = np.sin(2 * np.pi * x)
y_noisy = y_true + rng.standard_normal(100) * 0.2

# 2. Fit the smoothing spline
spline = SplineSmoother(x, df=8)
spline.smooth(y_noisy)

# 3. Fit the LOESS smoother
loess = LoessSmoother(x, span=0.3)
loess.smooth(y_noisy)

# 4. Predict on a new set of points
x_pred = np.linspace(0, 1, 200)
y_spline = spline.predict(x_pred)
y_loess = loess.predict(x_pred)

# 5. Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y_noisy, label='Noisy Data', alpha=0.4)
ax.plot(x, y_true, 'k--', label='True Function')
ax.plot(x_pred, y_spline, 'r-', label='Smoothing Spline (df=8)', linewidth=2)
ax.plot(x_pred, y_loess, 'b-', label='LOESS (span=0.3)', linewidth=2)
ax.set_title('Scatterplot Smoothers')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
plt.show()
```

## Building from Source

To build the package from source, you will need a C++ compiler compatible with C++11 and the required Python development headers. The build process is managed by `setuptools` and `pybind11`.

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/scatter_smooth.git
    cd scatter_smooth
    ```

2.  Install the dependencies and build the extension:
    ```bash
    pip install -r requirements.txt # Or install numpy, pybind11 manually
    pip install .
    ```

## Running Tests

The project uses `pytest` for testing. The tests compare the output against R's `smooth.spline` and verify internal consistency. To run the tests, you will need to install `pytest` and `rpy2`.

```bash
pip install pytest rpy2
pytest
```
*Note: An installation of R is required for the `rpy2` comparison tests to run.*

## License

This project is licensed under the BSD-3-Clause License. See the [LICENSE](LICENSE) file for details.
