# Smoothing Spline

This repository provides a minimal, high-performance implementation of a smoothing spline, similar to the `smooth.spline` function in R. The core logic is implemented in C++ for speed, with convenient Python bindings provided by `pybind11`. The package is designed to be familiar to users of `scikit-learn`.

## Key Features

*   **High Performance:** Core computations are in C++, leveraging the Eigen library for efficient linear algebra.
*   **Flexible Smoothing Control:** Specify smoothness by either degrees of freedom (`df`) or a penalty term (`lambda`).
*   **Automatic GCV:** Includes a method to find the optimal smoothing parameter via Generalized Cross-Validation (GCV).
*   **Extrapolation:** Supports linear extrapolation for points outside the training data range.
*   **R Compatibility:** Tested against R's `smooth.spline` for comparable results.
*   **Scikit-learn Style API (NOT):** The `SplineSmoother` class uses `smooth(y)`/`predict(x)`. It is not an sklearn estimator, but will be / is wrapped into one in `ISLP`. This is intentional, as there are use cases for this as a lower level object where weights may be updated and the smoother be refit to new data.

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

Here is a simple example of how to fit a smoothing spline to noisy data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scatter_smooth.fitter import SplineSmoother

# 1. Generate some noisy data
rng = np.random.default_rng(0)
x = np.linspace(0, 1, 100)
y_true = np.sin(2 * np.pi * x)
y_noisy = y_true + rng.standard_normal(100) * 0.2

# 2. Fit the smoothing spline
# Specify the desired degrees of freedom (df)
fitter = SplineSmoother(x, df=8)
fitter.smooth(y_noisy)

# 3. Predict on a new set of points
x_pred = np.linspace(0, 1, 200)
y_pred = fitter.predict(x_pred)

# 4. Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y_noisy, label='Noisy Data', alpha=0.6)
plt.plot(x, y_true, 'g--', label='True Function')
plt.plot(x_pred, y_pred, 'r-', label='Fitted Spline (df=8)', linewidth=2)
plt.legend()
plt.title('Smoothing Spline Fit')
plt.xlabel('x')
plt.ylabel('y')
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
