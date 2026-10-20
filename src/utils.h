#pragma once
#include <functional>
#include <Eigen/Dense>

namespace utils {
    // throws std::invalid_argument (a ValueError in Python) unless v has size n
    void check_size(long size, long n, const char* what);

    double brent_root(std::function<double(double)> func, double a, double b, double tol=1e-6, int max_iter=100);
    double brent_min(std::function<double(double)> func, double a, double b, double tol=1e-5, int max_iter=100);

    // Takahashi trace functions for banded matrices
    // Matrices are in Upper Banded format (SciPy/LAPACK compatible)
    Eigen::MatrixXd takahashi_upper(const Eigen::MatrixXd& U_banded);
    double trace_product_banded(const Eigen::MatrixXd& Z_banded, const Eigen::MatrixXd& B_banded);
    double trace_takahashi(const Eigen::MatrixXd& U_banded, const Eigen::MatrixXd& B_banded);
}
