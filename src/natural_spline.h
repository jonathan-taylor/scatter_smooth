#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

Eigen::MatrixXd compute_natural_spline_basis(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& knots,
    bool extrapolate_linear = true,
    int derivative_order = 0
);

Eigen::MatrixXd compute_penalty_matrix(const Eigen::Ref<const Eigen::VectorXd>& knots);

class NaturalSplineFitter {
    Eigen::MatrixXd N_;
    Eigen::MatrixXd Omega_;
    Eigen::MatrixXd NTW_; 
    Eigen::MatrixXd NTWN_; 
    Eigen::VectorXd knots_;
    Eigen::VectorXd alpha_;
public:
    NaturalSplineFitter(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& knots, py::object weights_obj);
    void update_weights(py::object weights_obj);
    Eigen::VectorXd fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval);
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv=0);
    double compute_df(double lamval);
    double gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y);
    double solve_for_df(double target_df, double min_log_lam = -12.0, double max_log_lam = 12.0);
    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0);
    Eigen::MatrixXd get_N();
    Eigen::MatrixXd get_Omega();
};
