#pragma once
#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <vector>

namespace py = pybind11;

namespace bspline {
    void eval_bspline_basis(double x, int k, const Eigen::VectorXd& knots, int& span_index, Eigen::VectorXd& N, int deriv=0);
}

class BSplineFitter {
    int order_; Eigen::VectorXd knots_, coeffs_, weights_, x_; int n_basis_; Eigen::MatrixXd AB_template_, Omega_band_;
    
    void get_boundary_weights(double& ws1, double& ws2, double& we1, double& we2);
    void apply_constraints(Eigen::MatrixXd& M);
    void apply_constraints(Eigen::VectorXd& v);

public:
    BSplineFitter(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& i_k, py::object w_obj, int order = 4);
    void initialize_design();
    void initialize_penalty();
    Eigen::VectorXd eval_basis(double x_val, int deriv=0);
    Eigen::VectorXd get_knots();
    
    // Returns constrained matrices in Upper Banded format
    Eigen::MatrixXd compute_design();
    Eigen::MatrixXd compute_penalty();
    
    // Returns unconstrained full matrices for debugging
    Eigen::MatrixXd get_NTWN();
    Eigen::MatrixXd get_Omega();

    Eigen::VectorXd compute_rhs(const Eigen::Ref<const Eigen::VectorXd>& y);
    void set_solution(const Eigen::Ref<const Eigen::VectorXd>& sol);
    Eigen::VectorXd fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval);
    double compute_df(double lamval);
    double gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y);
    double solve_for_df(double target_df, double min_log_lam = -12.0, double max_log_lam = 12.0);
    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0);
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_n, int deriv=0);
};
