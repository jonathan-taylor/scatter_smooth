#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class CubicSplineTraceCpp {
    Eigen::VectorXd x_;
    int n_;
    Eigen::VectorXd h_;
    Eigen::VectorXd weights_inv_;
    Eigen::VectorXd R_diag, R_off, M_diag, M_off1, M_off2;
    void compute_QR_bands();
public:
    CubicSplineTraceCpp(const Eigen::Ref<const Eigen::VectorXd>& x, py::object weights_obj);
    double compute_trace(double lam);
};

class ReinschFitter {
    Eigen::SparseMatrix<double> Q_, R_, M_; Eigen::VectorXd weights_inv_, x_, gamma_, f_; long n_;
public:
    ReinschFitter(const Eigen::Ref<const Eigen::VectorXd>& x, py::object weights_obj);
    void update_weights(py::object weights_obj);
    Eigen::VectorXd fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval);
    double compute_df(double lamval);
    double gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y);
    double solve_for_df(double target_df, double min_log_lam = -12.0, double max_log_lam = 12.0);
    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0);
    double compute_df_sparse(double lamval);
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv=0);
};
