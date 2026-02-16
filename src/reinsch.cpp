#include "reinsch.h"
#include "utils.h"
#include <limits>
#include <cmath>
#include <iostream>

/**
 * Computes the non-zero bands of the matrices Q and R used in the Reinsch algorithm.
 * 
 * R is a tridiagonal matrix with:
 *   R_ii = (h[i] + h[i+1]) / 3
 *   R_i,i+1 = h[i+1] / 6
 * 
 * M = Q^T W^-1 Q is a pentadiagonal matrix.
 * This function precomputes the diagonal and off-diagonal elements of R and M
 * to enable efficient O(N) operations later.
 */
void CubicSplineTraceCpp::compute_QR_bands() {
    int n_inner = n_ - 2;
    R_diag.resize(n_inner); R_off.resize(n_inner - 1);
    for (int i = 0; i < n_inner; ++i) R_diag(i) = (h_(i) + h_(i + 1)) / 3.0;
    for (int i = 0; i < n_inner - 1; ++i) R_off(i) = h_(i + 1) / 6.0;
    M_diag.resize(n_inner); M_off1.resize(n_inner - 1); M_off2.resize(n_inner - 2);
    Eigen::VectorXd inv_h_j = h_.head(n_inner).cwiseInverse();
    Eigen::VectorXd inv_h_j_plus_1 = h_.tail(n_inner).cwiseInverse();
    Eigen::VectorXd q1_vals = -(inv_h_j + inv_h_j_plus_1);
    M_diag = (inv_h_j.array().square() * weights_inv_.head(n_inner).array()) +
                (q1_vals.array().square() * weights_inv_.segment(1, n_inner).array()) +
                (inv_h_j_plus_1.array().square() * weights_inv_.segment(2, n_inner).array());
    M_off1 = (q1_vals.head(n_inner - 1).array() * inv_h_j.tail(n_inner - 1).array() * weights_inv_.segment(1, n_inner - 1).array()) +
                (inv_h_j_plus_1.head(n_inner - 1).array() * q1_vals.tail(n_inner - 1).array() * weights_inv_.segment(2, n_inner - 1).array());
    M_off2 = (inv_h_j_plus_1.head(n_inner - 2).array() * inv_h_j.tail(n_inner - 2).array() * weights_inv_.segment(2, n_inner - 2).array());
}

CubicSplineTraceCpp::CubicSplineTraceCpp(const Eigen::Ref<const Eigen::VectorXd>& x, py::object weights_obj) {
    x_ = x; n_ = x.size(); h_.resize(n_ - 1);
    for (int i = 0; i < n_ - 1; ++i) h_(i) = x_(i + 1) - x_(i);
    weights_inv_.resize(n_);
    if (weights_obj.is_none()) weights_inv_.setOnes();
    else weights_inv_ = weights_obj.cast<Eigen::VectorXd>().cwiseInverse();
    compute_QR_bands();
}

/**
 * Computes the trace of the influence matrix (Effective Degrees of Freedom) in O(N).
 * 
 * The influence matrix is S = (I + lambda * K)^-1, where K = Q R^-1 Q^T W^-1.
 * The trace is equivalent to: 2 + Trace(R * (R + lambda * M)^-1)
 * 
 * It uses the Cholesky decomposition of the banded matrix B = R + lambda * M,
 * followed by Takahashi's equations (sparse inverse subset) to compute the necessary 
 * elements of B^-1 to find the trace product with R.
 */
double CubicSplineTraceCpp::compute_trace(double lam) {
    int n_inner = n_ - 2;
    Eigen::VectorXd B_diag = R_diag + lam * M_diag; B_diag.array() += 1e-9;
    Eigen::VectorXd B_off1 = R_off + lam * M_off1; Eigen::VectorXd B_off2 = lam * M_off2;
    Eigen::MatrixXd L_banded = Eigen::MatrixXd::Zero(3, n_inner);
    for (int i = 0; i < n_inner; ++i) {
        double l_ii_sq = B_diag(i);
        if (i > 0) l_ii_sq -= std::pow(L_banded(1, i-1), 2);
        if (i > 1) l_ii_sq -= std::pow(L_banded(2, i-2), 2);
        if (l_ii_sq <= 0) return std::numeric_limits<double>::quiet_NaN();
        L_banded(0, i) = std::sqrt(l_ii_sq);
        if (i < n_inner - 1) {
            double l_i1_i = B_off1(i);
            if (i > 0) l_i1_i -= L_banded(1, i-1) * L_banded(2, i-1);
            L_banded(1, i) = l_i1_i / L_banded(0, i);
        }
        if (i < n_inner - 2) L_banded(2, i) = B_off2(i) / L_banded(0, i);
    }
    Eigen::VectorXd Inv_diag = Eigen::VectorXd::Zero(n_inner), Inv_off1 = Eigen::VectorXd::Zero(n_inner - 1);
    for (int i = n_inner - 1; i >= 0; --i) {
        double val = 1.0 / L_banded(0, i), sum_diag = 0.0;
        if (i + 1 < n_inner) {
            double s_off = L_banded(1, i) * Inv_diag(i + 1);
            if (i + 2 < n_inner) s_off += L_banded(2, i) * Inv_off1(i + 1);
            Inv_off1(i) = -val * s_off; sum_diag += L_banded(1, i) * Inv_off1(i);
        }
        if (i + 2 < n_inner) {
            double s_off2 = L_banded(1, i) * Inv_off1(i + 1) + L_banded(2, i) * Inv_diag(i + 2);
            sum_diag += L_banded(2, i) * (-val * s_off2);
        }
        Inv_diag(i) = val * (val - sum_diag);
    }
    double tr_RBinv = (R_diag.array() * Inv_diag.array()).sum();
    if (n_inner > 1) tr_RBinv += 2 * (R_off.array() * Inv_off1.array()).sum();
    return 2.0 + tr_RBinv;
}

/**
 * ReinschFitter implements the smoothing spline using the Reinsch algorithm.
 * 
 * This approach is valid when the knots are exactly the unique data points x.
 * It solves the linear system for the second derivatives gamma:
 *   (R + lambda * Q^T W^-1 Q) * gamma = Q^T * y
 * 
 * The fitted values f are then recovered via:
 *   f = y - lambda * W^-1 * Q * gamma
 */
ReinschFitter::ReinschFitter(const Eigen::Ref<const Eigen::VectorXd>& x, py::object weights_obj) {
    x_ = x; n_ = x.size(); long n_inner = n_ - 2;
    Eigen::VectorXd h = x.segment(1, n_ - 1) - x.segment(0, n_ - 1);
    Eigen::VectorXd inv_h = h.cwiseInverse();
    R_.resize(n_inner, n_inner); Q_.resize(n_, n_inner); 
    std::vector<Eigen::Triplet<double>> r_t, q_t;
    for (int i = 0; i < n_inner; ++i) {
        r_t.push_back({i, i, (h[i] + h[i+1]) / 3.0});
        if (i < n_inner - 1) { r_t.push_back({i, i+1, h[i+1] / 6.0}); r_t.push_back({i+1, i, h[i+1] / 6.0}); }
    }
    R_.setFromTriplets(r_t.begin(), r_t.end());
    for (int j = 0; j < n_inner; ++j) {
        q_t.push_back({j, j, inv_h[j]}); q_t.push_back({j+1, j, -inv_h[j] - inv_h[j+1]}); q_t.push_back({j+2, j, inv_h[j+1]});
    }
    Q_.setFromTriplets(q_t.begin(), q_t.end());
    update_weights(weights_obj);
}

void ReinschFitter::update_weights(py::object weights_obj) {
    if (weights_obj.is_none()) weights_inv_ = Eigen::VectorXd::Ones(n_);
    else weights_inv_ = weights_obj.cast<Eigen::VectorXd>().cwiseInverse();
    Eigen::SparseMatrix<double> Winv(n_, n_);
    std::vector<Eigen::Triplet<double>> w_t;
    for(int i=0; i<n_; ++i) w_t.push_back({i, i, weights_inv_[i]});
    Winv.setFromTriplets(w_t.begin(), w_t.end());
    M_ = Q_.transpose() * Winv * Q_;
}

Eigen::VectorXd ReinschFitter::fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
    Eigen::VectorXd QT_y = Q_.transpose() * y;
    Eigen::SparseMatrix<double> LHS = R_ + lamval * M_;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(LHS);
    gamma_ = solver.solve(QT_y);
    f_ = y - lamval * weights_inv_.cwiseProduct(Q_ * gamma_);
    return f_;
}

double ReinschFitter::compute_df(double lamval) {
    Eigen::VectorXd original_weights = weights_inv_.cwiseInverse();
    CubicSplineTraceCpp trace_solver(x_, py::cast(original_weights));
    return trace_solver.compute_trace(lamval);
}

double ReinschFitter::gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
    Eigen::VectorXd f = fit(y, lamval);
    double rss = ((y - f).array().square() * weights_inv_.cwiseInverse().array()).sum();
    double df = compute_df(lamval), n = (double)y.size(), denom = 1.0 - df / n;
    return (denom < 1e-6) ? 1e20 : (rss / n) / (denom * denom);
}

double ReinschFitter::solve_for_df(double target_df, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return compute_df(std::pow(10.0, log_lam)) - target_df; };
    return std::pow(10.0, utils::brent_root(func, min_log_lam, max_log_lam));
}

double ReinschFitter::solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return gcv_score(std::pow(10.0, log_lam), y); };
    return std::pow(10.0, utils::brent_min(func, min_log_lam, max_log_lam));
}

double ReinschFitter::compute_df_sparse(double lamval) {
    Eigen::SparseMatrix<double> LHS = R_ + lamval * M_;
    Eigen::MatrixXd LHS_dense = LHS;
    Eigen::MatrixXd M_dense = M_;
    Eigen::LLT<Eigen::MatrixXd> solver;
    solver.compute(LHS_dense);
    if(solver.info() != Eigen::Success) return 0.0;
    Eigen::MatrixXd Sol = solver.solve(M_dense);
    return (double)n_ - lamval * Sol.trace();
}

Eigen::VectorXd ReinschFitter::predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv) {
    Eigen::VectorXd M_c(n_); M_c[0] = 0.0; M_c[n_-1] = 0.0; M_c.segment(1, n_-2) = gamma_;
    Eigen::VectorXd y_p(x_new.size());
    for(long i=0; i<x_new.size(); ++i) {
        double val = x_new[i]; long k = 0;
        if (val < x_[0]) k = 0; else if (val >= x_[n_-1]) k = n_ - 2;
        else k = std::distance(x_.data(), std::upper_bound(x_.data(), x_.data() + n_, val)) - 1;
        double h = x_[k+1] - x_[k];
        if (val < x_[0]) {
            double d1 = (f_[1] - f_[0]) / h - h * M_c[1] / 6.0;
            y_p[i] = (deriv == 0) ? f_[0] + d1 * (val - x_[0]) : (deriv == 1 ? d1 : 0);
        } else if (val > x_[n_-1]) {
            double d1 = (f_[n_-1] - f_[n_-2]) / (x_[n_-1] - x_[n_-2]) + M_c[n_-2] * (x_[n_-1] - x_[n_-2]) / 6.0;
            y_p[i] = (deriv == 0) ? f_[n_-1] + d1 * (val - x_[n_-1]) : (deriv == 1 ? d1 : 0);
        } else {
            if (deriv == 0) {
                y_p[i] = (std::pow(x_[k+1] - val, 3) * M_c[k] + std::pow(val - x_[k], 3) * M_c[k+1]) / (6.0 * h) +
                            (f_[k] - h*h * M_c[k] / 6.0) * (x_[k+1] - val) / h + (f_[k+1] - h*h * M_c[k+1] / 6.0) * (val - x_[k]) / h;
            } else if (deriv == 1) {
                y_p[i] = (-3.0 * std::pow(x_[k+1] - val, 2) * M_c[k] + 3.0 * std::pow(val - x_[k], 2) * M_c[k+1]) / (6.0 * h) -
                            (f_[k] - h*h * M_c[k] / 6.0) / h + (f_[k+1] - h*h * M_c[k+1] / 6.0) / h;
            } else if (deriv == 2) y_p[i] = (6.0 * (x_[k+1] - val) * M_c[k] + 6.0 * (val - x_[k]) * M_c[k+1]) / (6.0 * h);
            else y_p[i] = 0;
        }
    }
    return y_p;
}
