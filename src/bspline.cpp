#include "bspline.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <Eigen/Sparse>

namespace bspline {
    // ... (eval_bspline_basis implementation unchanged) ...
    void eval_bspline_basis(double x, int k, const Eigen::VectorXd& knots, int& span_index, Eigen::VectorXd& N, int deriv) {
        int n_k = (int)knots.size();
        double d_min = knots[k-1], d_max = knots[n_k-k];
        if (x < d_min) x = d_min; if (x > d_max) x = d_max;
        int i = (x >= d_max) ? n_k - k - 1 : std::distance(knots.data(), std::upper_bound(knots.data() + k - 1, knots.data() + n_k - k, x)) - 1;
        span_index = i - k + 1;
        std::vector<double> left(k), right(k); std::vector<std::vector<double>> ndu(k, std::vector<double>(k));
        ndu[0][0] = 1.0;
        for (int j = 1; j < k; ++j) {
            left[j] = x - knots[i + 1 - j]; right[j] = knots[i + j] - x;
            double saved = 0.0;
            for (int r = 0; r < j; ++r) {
                ndu[j][r] = right[r + 1] + left[j - r];
                double temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp; saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }
        N.resize(k);
        if (deriv == 0) { for (int j = 0; j < k; ++j) N[j] = ndu[j][k - 1]; return; }
        std::vector<std::vector<double>> ders(deriv + 1, std::vector<double>(k));
        for (int j = 0; j < k; ++j) ders[0][j] = ndu[j][k - 1];
        for (int r = 0; r <= k - 1; ++r) {
            int s1 = 0, s2 = 1; std::vector<std::vector<double>> a(2, std::vector<double>(k)); a[0][0] = 1.0;
            for (int d = 1; d <= deriv; ++d) {
                double d_v = 0.0; int rk = r - d, pk = k - 1 - d;
                if (r >= d) { double den = ndu[pk + 1][rk]; a[s2][0] = (den == 0.0) ? 0.0 : a[s1][0] / den; d_v = a[s2][0] * ndu[rk][pk]; }
                for (int j = std::max(0, -rk); j <= std::min(d - 1, k - r - 1); ++j) {
                    if (j == 0 && r >= d) continue;
                    double den = ndu[pk + 1][rk + j];
                    a[s2][j] = (den == 0.0) ? 0.0 : (a[s1][j] - (j > 0 ? a[s1][j - 1] : 0.0)) / den;
                    d_v += a[s2][j] * ndu[rk + j][pk];
                }
                if (r <= pk) { double den = ndu[pk + 1][r]; a[s2][d] = (den == 0.0) ? 0.0 : -a[s1][d - 1] / den; d_v += a[s2][d] * ndu[r][pk]; }
                ders[d][r] = d_v; std::swap(s1, s2);
            }
        }
        double factor = (double)(k - 1);
        for (int d = 1; d <= deriv; ++d) { for (int j = 0; j < k; ++j) ders[d][j] *= factor; factor *= (k - 1 - d); }
        for (int j = 0; j < k; ++j) N[j] = ders[deriv][j];
    }
}

BSplineSmoother::BSplineSmoother(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& i_k, py::object w_obj, int order) : order_(order), x_(x) {
    long n_i = i_k.size(), n_a = n_i + 2 * (order - 1); knots_.resize(n_a);
    for(int i=0; i<order; ++i) knots_[i] = i_k[0];
    if (n_i > 2) knots_.segment(order, n_i - 2) = i_k.segment(1, n_i - 2);
    for(int i=0; i<order; ++i) knots_[n_a - order + i] = i_k[n_i - 1];
    n_basis_ = (int)(n_a - order);
    if (n_i < 4) throw std::invalid_argument("a cubic smoothing spline needs at least 4 knots");
    if (order != 4) throw std::invalid_argument("only cubic splines (order 4) are supported");
    weights_ = w_obj.is_none() ? Eigen::VectorXd::Ones(x.size()) : w_obj.cast<Eigen::VectorXd>();
    utils::check_size(weights_.size(), x.size(), "weights");
    basis_at_x_.resize(order_, x_.size()); span_at_x_.resize(x_.size());
    Eigen::VectorXd b_v(order_);
    for (int i = 0; i < x_.size(); ++i) {
        bspline::eval_bspline_basis(x_[i], order_, knots_, span_at_x_[i], b_v, 0);
        basis_at_x_.col(i) = b_v;
    }
    int kd = order - 1; AB_template_ = Eigen::MatrixXd::Zero(kd + 1, n_basis_); initialize_design();
    Omega_band_ = Eigen::MatrixXd::Zero(kd + 1, n_basis_); initialize_penalty();
}

void BSplineSmoother::initialize_design() {
    int kd = order_ - 1;
    AB_template_.setZero();
    for (int i = 0; i < x_.size(); ++i) {
        int s_i = span_at_x_[i];
        auto b_v = basis_at_x_.col(i);
        for (int r = 0; r < order_; ++r) {
            int rg = s_i + r; if (rg >= n_basis_) continue;
            for (int c = 0; c <= r; ++c) { 
                int cg = s_i + c; 
                AB_template_(kd - (rg - cg), rg) += weights_[i] * b_v[r] * b_v[c]; 
            }
        }
    }
}

void BSplineSmoother::initialize_penalty() {
    double pt = 1.0 / std::sqrt(3.0); double gp[2] = {-pt, pt}, gw[2] = {1.0, 1.0};
    int s_k = order_ - 1, e_k = (int)knots_.size() - order_; Eigen::VectorXd b_d(order_); int s_i;
    int kd = order_ - 1;
    for (int k = s_k; k < e_k; ++k) {
        double t_s = knots_[k], t_e = knots_[k+1], dt = t_e - t_s; if (dt <= 1e-14) continue;
        for (int g = 0; g < 2; ++g) {
            bspline::eval_bspline_basis(0.5 * (t_s + t_e) + 0.5 * dt * gp[g], order_, knots_, s_i, b_d, 2);
            double w_v = 0.5 * dt * gw[g];
            for (int r = 0; r < order_; ++r) {
                int rg = s_i + r; if (rg >= n_basis_) continue;
                for (int c = 0; c <= r; ++c) { 
                    int cg = s_i + c; 
                    Omega_band_(kd - (rg - cg), rg) += w_v * b_d[r] * b_d[c]; 
                }
            }
        }
    }
}

void BSplineSmoother::update_weights(const Eigen::Ref<const Eigen::VectorXd>& w) {
    utils::check_size(w.size(), x_.size(), "weights");
    weights_ = w;
    initialize_design();
}

// the fitted spline at the data
Eigen::VectorXd BSplineSmoother::fitted() {
    if (coeffs_.size() != n_basis_) throw std::runtime_error("fitted: no fit yet (call smooth first)");
    Eigen::VectorXd f(x_.size());
    for (int i = 0; i < x_.size(); ++i) {
        int s_i = span_at_x_[i]; double v = 0.0;
        for (int j = 0; j < order_; ++j) { int idx = s_i + j; if (idx < n_basis_) v += coeffs_[idx] * basis_at_x_(j, i); }
        f[i] = v;
    }
    return f;
}

Eigen::VectorXd BSplineSmoother::eval_basis(double x_val, int deriv) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n_basis_), b_v(order_); int s_i;
    bspline::eval_bspline_basis(x_val, order_, knots_, s_i, b_v, deriv);
    for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n_basis_) v[idx] = b_v[j]; }
    return v;
}

Eigen::VectorXd BSplineSmoother::get_knots() { return knots_; }

Eigen::MatrixXd BSplineSmoother::get_NTWN() {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis_, n_basis_); int kd = order_ - 1;
    for (int j = 0; j < n_basis_; ++j) { 
        for (int i = 0; i <= kd; ++i) { 
            int row = j - (kd - i);
            if (row >= 0 && row <= j) {
                double v = AB_template_(i, j);
                M(row, j) = v;
                M(j, row) = v; 
            }
        } 
    }
    return M;
}

Eigen::MatrixXd BSplineSmoother::get_Omega() {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis_, n_basis_); int kd = order_ - 1;
    for (int j = 0; j < n_basis_; ++j) { 
        for (int i = 0; i <= kd; ++i) { 
            int row = j - (kd - i);
            if (row >= 0 && row <= j) {
                double v = Omega_band_(i, j);
                M(row, j) = v;
                M(j, row) = v; 
            }
        } 
    }
    return M;
}

// -----------------------------------------------------------------------------------------
// Theory: Natural Boundary Conditions and the Projection Matrix P
// -----------------------------------------------------------------------------------------
// The B-spline basis functions B_1(x), ..., B_N(x) (where N = n_basis_) form a basis for 
// cubic splines with the given knots. However, a "natural" cubic spline requires the 
// second derivative to be zero at the boundary knots (x_min and x_max).
//
// The second derivative of a spline f(x) = sum(theta_j * B_j(x)) is:
// f''(x) = sum(theta_j * B''_j(x))
//
// At the boundaries x_min and x_max, we require f''(x_min) = 0 and f''(x_max) = 0.
// Due to the local support of B-splines, only the first few and last few basis functions 
// are non-zero at the boundaries. Specifically, for cubic B-splines (order 4):
// - At x_min (first knot), only B_1, B_2, B_3 have non-zero support (conceptually). 
//   In our 0-indexed implementation: coeffs_[0], coeffs_[1], coeffs_[2].
// - At x_max (last knot), only B_{N-2}, B_{N-1}, B_{N} (last 3) are active.
//
// The condition f''(x_min) = 0 implies a linear constraint on theta_0, theta_1, theta_2:
// theta_0 * B''_0(x_min) + theta_1 * B''_1(x_min) + theta_2 * B''_2(x_min) = 0
// theta_0 = - (theta_1 * B''_1(x_min) / B''_0(x_min)) - (theta_2 * B''_2(x_min) / B''_0(x_min))
// theta_0 = ws1 * theta_1 + ws2 * theta_2
//
// Similarly at the trailing boundary:
// theta_{N-1} = we1 * theta_{N-2} + we2 * theta_{N-3}
//
// These constraints reduce the dimension of the free parameters from N to N-2.
// We can define a projection matrix P of size N x (N-2) that maps the reduced parameters 
// (theta_1, ..., theta_{N-2}) back to the full parameter vector theta.
//
// The linear system for the reduced parameters gamma (size N-2) becomes:
// (P^T * (N^T W N + lambda * Omega) * P) * gamma = P^T * N^T W y
//
// The `apply_constraints` function performs the operation M_reduced = P^T * M * P
// in-place on the banded matrix M (representing N^T W N or Omega).
// Because P is extremely sparse (identity plus corner modifications), this operation 
// only affects the top-left (leading) and bottom-right (trailing) corners of the matrix.
// -----------------------------------------------------------------------------------------

void BSplineSmoother::get_boundary_weights(double& ws1, double& ws2, double& we1, double& we2) {
    int s_i;
    Eigen::VectorXd d2_v(order_); 
    
    // Evaluate second derivatives of basis functions at the first internal knot (Start)
    // Note: knots_[order_-1] is the first knot where the spline actually starts.
    bspline::eval_bspline_basis(knots_[order_-1], order_, knots_, s_i, d2_v, 2);
    double v0 = d2_v[0], v1 = d2_v[1], v2 = d2_v[2]; 
    if (std::abs(v0) < 1e-12) throw std::runtime_error("Leading zero: v0=" + std::to_string(v0));
    
    // Solve for theta_0 in terms of theta_1 and theta_2
    ws1 = -v1 / v0; 
    ws2 = -v2 / v0;

    // Evaluate at the last knot (End)
    bspline::eval_bspline_basis(knots_[knots_.size() - order_], order_, knots_, s_i, d2_v, 2);
    int iN1 = n_basis_-1-s_i, iN2 = n_basis_-2-s_i, iN3 = n_basis_-3-s_i;
    double u0 = d2_v[iN1], u1 = d2_v[iN2], u2 = d2_v[iN3]; 
    if (std::abs(u0) < 1e-12) throw std::runtime_error("Trailing zero: u0=" + std::to_string(u0));
    
    // Solve for theta_{N-1} in terms of theta_{N-2} and theta_{N-3}
    we1 = -u1 / u0; 
    we2 = -u2 / u0;
}

void BSplineSmoother::apply_constraints(Eigen::MatrixXd& M) {
    // This function computes P^T * M * P in-place.
    // M is stored in Upper Banded format.
    // P maps reduced indices 0..N-3 to full indices 0..N-1.
    // Most of P is Identity (shifted). The interesting parts are the corners.
    //
    // The transformation linearly combines rows and columns at the boundaries.
    // For the leading corner (affecting indices 0, 1, 2 of full matrix -> 0, 1 of reduced matrix):
    // New A[0,0] corresponds to full A[1,1] + weights mixed from A[0,0], A[0,1], etc.
    // Specifically, if we substitute theta_0 = ws1*theta_1 + ws2*theta_2 into the quadratic form theta^T A theta,
    // and differentiate with respect to theta_1 and theta_2, we get the new matrix entries.
    
    double ws1, ws2, we1, we2;
    get_boundary_weights(ws1, ws2, we1, we2);
    
    int n = n_basis_, kd = order_ - 1;
    
    // Helper to access Upper Banded elements: M(kd - (col - row), col) for col >= row.
    auto set_M = [&](int i, int j, double val) {
        if (j < i) std::swap(i, j);
        if (j - i <= kd) {
            M(kd - (j - i), j) = val;
        }
    };
    
    auto get_M = [&](int i, int j) -> double { 
        if (j < i) std::swap(i, j); 
        if (j - i > kd) return 0.0; 
        return M(kd - (j - i), j); 
    };
    
    // Get original top-left 4x4 block elements involved in the constraint
    double a00 = get_M(0,0), a01 = get_M(1,0), a02 = get_M(2,0), a03 = get_M(3,0);
    double a11 = get_M(1,1), a12 = get_M(2,1), a13 = get_M(3,1);
    double a22 = get_M(2,2), a23 = get_M(3,2);
    
    // Update reduced matrix entries (indices shifted by -1 relative to full matrix where identity holds)
    // Reduced index 0 corresponds to Full index 1.
    // Reduced index 1 corresponds to Full index 2.
    // The quadratic terms for theta_1 involve: A[1,1]*t1^2 + A[0,0]*(ws1*t1)^2 + 2*A[0,1]*t1*(ws1*t1) ...
    
    // A_reduced[0, 0] (Full 1,1)
    set_M(1, 1, a11 + 2*ws1*a01 + ws1*ws1*a00);
    // A_reduced[0, 1] (Full 1,2)
    set_M(1, 2, a12 + ws1*a02 + ws2*a01 + ws1*ws2*a00);
    // A_reduced[1, 1] (Full 2,2)
    set_M(2, 2, a22 + 2*ws2*a02 + ws2*ws2*a00);
    
    if (kd >= 3) { 
        // Interaction with further bands (Full index 3)
        // A_reduced[0, 2] corresponds to Full[1, 3] + contribution from Full[0, 3] via theta_0
        set_M(1, 3, a13 + ws1*a03);
        // A_reduced[1, 2] corresponds to Full[2, 3] + contribution from Full[0, 3] via theta_0
        set_M(2, 3, a23 + ws2*a03);
    }

    // Get original bottom-right block elements
    double an11 = get_M(n-1, n-1), an12 = get_M(n-1, n-2), an13 = get_M(n-1, n-3), an14 = get_M(n-1, n-4);
    double an22 = get_M(n-2, n-2), an23 = get_M(n-2, n-3), an24 = get_M(n-2, n-4);
    double an33 = get_M(n-3, n-3), an34 = get_M(n-3, n-4);
    
    // Update reduced matrix entries at the end
    // Reduced index N-3 corresponds to Full index N-2.
    // Reduced index N-4 corresponds to Full index N-3.
    
    // A_reduced[N-3, N-3] (Full N-2, N-2)
    set_M(n-2, n-2, an22 + 2*we1*an12 + we1*we1*an11);
    // A_reduced[N-3, N-4] (Full N-2, N-3)
    set_M(n-2, n-3, an23 + we1*an13 + we2*an12 + we1*we2*an11);
    // A_reduced[N-4, N-4] (Full N-3, N-3)
    set_M(n-3, n-3, an33 + 2*we2*an13 + we2*we2*an11);
    
    if (kd >= 3) { 
        // A_reduced[N-3, N-5] (Full N-2, N-4)
        set_M(n-2, n-4, an24 + we1*an14);
        // A_reduced[N-4, N-5] (Full N-3, N-4)
        set_M(n-3, n-4, an34 + we2*an14);
    }
}

void BSplineSmoother::apply_constraints(Eigen::VectorXd& v) {
    // Computes P^T * v for a vector v.
    // This maps the right-hand side vector from size N to N-2.
    // v_reduced[0] = v[1] + ws1 * v[0]
    // v_reduced[1] = v[2] + ws2 * v[0]
    // ...
    // v_reduced[N-3] = v[N-2] + we1 * v[N-1]
    // v_reduced[N-4] = v[N-3] + we2 * v[N-1]
    
    double ws1, ws2, we1, we2;
    get_boundary_weights(ws1, ws2, we1, we2);
    int n = n_basis_;
    v[1] += ws1 * v[0]; 
    v[2] += ws2 * v[0];
    v[n-2] += we1 * v[n-1]; 
    v[n-3] += we2 * v[n-1];
}

Eigen::MatrixXd BSplineSmoother::compute_design() {
    Eigen::MatrixXd NTWN = AB_template_;
    apply_constraints(NTWN);
    return NTWN;
}

Eigen::MatrixXd BSplineSmoother::compute_penalty() {
    Eigen::MatrixXd Omega = Omega_band_;
    apply_constraints(Omega);
    return Omega;
}

Eigen::VectorXd BSplineSmoother::compute_rhs(const Eigen::Ref<const Eigen::VectorXd>& y) {
    int n = n_basis_;
    utils::check_size(y.size(), x_.size(), "y");
    Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
    for(int i=0; i<x_.size(); ++i) {
        int s_i = span_at_x_[i];
        for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n) b[idx] += weights_[i] * y[i] * basis_at_x_(j, i); }
    }
    apply_constraints(b);
    return b;
}

void BSplineSmoother::set_solution(const Eigen::Ref<const Eigen::VectorXd>& sol) {
    int n = n_basis_;
    if (sol.size() != n - 2) throw std::runtime_error("Solution size mismatch. Expected " + std::to_string(n-2) + ", got " + std::to_string(sol.size()));
    
    coeffs_ = Eigen::VectorXd::Zero(n);
    coeffs_.segment(1, n - 2) = sol;

    double ws1, ws2, we1, we2;
    get_boundary_weights(ws1, ws2, we1, we2);

    coeffs_[0] = ws1 * coeffs_[1] + ws2 * coeffs_[2]; 
    coeffs_[n-1] = we1 * coeffs_[n-2] + we2 * coeffs_[n-3];
}

Eigen::VectorXd BSplineSmoother::smooth(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
    utils::check_size(y.size(), x_.size(), "y");
    Eigen::MatrixXd NTWN = compute_design();
    Eigen::MatrixXd Omega = compute_penalty();
    Eigen::VectorXd b = compute_rhs(y);
    Eigen::MatrixXd AB = NTWN + lamval * Omega;
    
    int n = n_basis_, kd = order_ - 1;
    int n_r = n - 2;
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(n_r * (kd + 1) * 2);
    
    for (int j = 1; j < n-1; ++j) {
        int col_sub = j - 1;
        int row_start = std::max(1, j - kd);
        for (int i = row_start; i <= j; ++i) {
            int row_sub = i - 1;
            double val = AB(kd - (j - i), j);
            if (std::abs(val) > 1e-14) {
                triplets.push_back({row_sub, col_sub, val});
                if (row_sub != col_sub) {
                    triplets.push_back({col_sub, row_sub, val});
                }
            }
        }
    }
    
    Eigen::SparseMatrix<double> A_sparse(n_r, n_r);
    A_sparse.setFromTriplets(triplets.begin(), triplets.end());
    
    Eigen::VectorXd b_sub = b.segment(1, n_r);
    
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A_sparse);
    if(solver.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed in BSplineSmoother::smooth");
    }
    
    Eigen::VectorXd sol = solver.solve(b_sub);
    set_solution(sol);
    return coeffs_;
}

namespace {
    Eigen::SparseMatrix<double> banded_to_sparse(const Eigen::MatrixXd& band_mat, int n, int kd) {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n * (kd + 1) * 2);
        for (int j = 0; j < n; ++j) {
            int row_start = std::max(0, j - kd);
            for (int i = row_start; i <= j; ++i) {
                double val = band_mat(kd - (j - i), j);
                if (std::abs(val) > 1e-14) {
                    triplets.push_back({i, j, val});
                    if (i != j) triplets.push_back({j, i, val});
                }
            }
        }
        Eigen::SparseMatrix<double> sparse_mat(n, n);
        sparse_mat.setFromTriplets(triplets.begin(), triplets.end());
        return sparse_mat;
    }
}

double BSplineSmoother::compute_df(double lamval) {
    int n = n_basis_, kd = order_ - 1;
    
    Eigen::SparseMatrix<double> NTWN_sparse = banded_to_sparse(AB_template_, n, kd);
    Eigen::SparseMatrix<double> Omega_sparse = banded_to_sparse(Omega_band_, n, kd);
    
    double ws1, ws2, we1, we2;
    get_boundary_weights(ws1, ws2, we1, we2);
    
    std::vector<Eigen::Triplet<double>> p_triplets;
    p_triplets.reserve((n-2) + 4);
    p_triplets.push_back({0, 0, ws1});
    p_triplets.push_back({0, 1, ws2});
    p_triplets.push_back({n-1, n-3, we1});
    p_triplets.push_back({n-1, n-4, we2});
    for(int i=0; i < n-2; ++i) {
        p_triplets.push_back({i+1, i, 1.0});
    }
    
    Eigen::SparseMatrix<double> P(n, n-2);
    P.setFromTriplets(p_triplets.begin(), p_triplets.end());
    
    Eigen::SparseMatrix<double> A = P.transpose() * NTWN_sparse * P;
    Eigen::SparseMatrix<double> B = P.transpose() * Omega_sparse * P;
    
    Eigen::SparseMatrix<double> LHS = A + lamval * B;
    
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(LHS);
    if(solver.info() != Eigen::Success) return 0.0;
    
    double trace = 0.0;
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n-2);
    Eigen::VectorXd sol;
    
    for (int j = 0; j < n-2; ++j) {
        rhs.setZero();
        for (Eigen::SparseMatrix<double>::InnerIterator it(A, j); it; ++it) {
            rhs[it.row()] = it.value();
        }
        
        sol = solver.solve(rhs);
        trace += sol[j];
    }
    
    return trace;
}

double BSplineSmoother::gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
    smooth(y, lamval);
    Eigen::VectorXd f = predict(x_, 0);
    double rss = 0.0;
    for(int i=0; i<x_.size(); ++i) {
        rss += weights_[i] * std::pow(y[i] - f[i], 2);
    }
    double df = compute_df(lamval);
    double n = (double)y.size();
    double denom = 1.0 - df / n;
    if (denom < 1e-6) return 1e20;
    return (rss / n) / (denom * denom);
}

double BSplineSmoother::solve_for_df(double target_df, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return compute_df(std::pow(10.0, log_lam)) - target_df; };
    return std::pow(10.0, utils::brent_root(func, min_log_lam, max_log_lam));
}

double BSplineSmoother::solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return gcv_score(std::pow(10.0, log_lam), y); };
    return std::pow(10.0, utils::brent_min(func, min_log_lam, max_log_lam));
}

Eigen::VectorXd BSplineSmoother::predict(const Eigen::Ref<const Eigen::VectorXd>& x_n, int deriv) {
    if (coeffs_.size() != n_basis_)
        throw std::runtime_error("predict: the smoother is not fitted yet (call smooth first)");
    if (deriv < 0 || deriv > 3) throw std::invalid_argument("deriv should be 0, 1, 2 or 3");
    Eigen::VectorXd y_p(x_n.size()), b_v(order_); int s_i; double d_min = knots_[order_-1], d_max = knots_[knots_.size() - order_];
    for (int i = 0; i < (int)x_n.size(); ++i) {
        double val = x_n[i];
        if (val < d_min || val > d_max) {
            double b_u = (val < d_min) ? d_min : d_max; bspline::eval_bspline_basis(b_u, order_, knots_, s_i, b_v, 0);
            double f_v = 0.0; for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n_basis_) f_v += coeffs_[idx] * b_v[j]; }
            bspline::eval_bspline_basis(b_u, order_, knots_, s_i, b_v, 1);
            double f_s = 0.0; for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n_basis_) f_s += coeffs_[idx] * b_v[j]; }
            y_p[i] = (deriv == 0) ? f_v + f_s * (val - b_u) : (deriv == 1 ? f_s : 0.0);
        } else {
            bspline::eval_bspline_basis(val, order_, knots_, s_i, b_v, deriv);
            double f_v = 0.0; for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n_basis_) f_v += coeffs_[idx] * b_v[j]; }
            y_p[i] = f_v;
        }
    }
    return y_p;
}
