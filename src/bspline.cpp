#include "bspline.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>
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

BSplineFitter::BSplineFitter(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& i_k, py::object w_obj, int order) : order_(order), x_(x) {
    long n_i = i_k.size(), n_a = n_i + 2 * (order - 1); knots_.resize(n_a);
    for(int i=0; i<order; ++i) knots_[i] = i_k[0];
    if (n_i > 2) knots_.segment(order, n_i - 2) = i_k.segment(1, n_i - 2);
    for(int i=0; i<order; ++i) knots_[n_a - order + i] = i_k[n_i - 1];
    n_basis_ = (int)(n_a - order);
    weights_ = w_obj.is_none() ? Eigen::VectorXd::Ones(x.size()) : w_obj.cast<Eigen::VectorXd>();
    int kd = order - 1; AB_template_ = Eigen::MatrixXd::Zero(kd + 1, n_basis_); compute_NTWN();
    Omega_band_ = Eigen::MatrixXd::Zero(kd + 1, n_basis_); compute_penalty_matrix();
}

// ... (compute_NTWN, compute_penalty_matrix, eval_basis, get_knots, get_NTWN, get_Omega, compute_system, set_solution, fit unchanged except fit needs to be kept) ...

void BSplineFitter::compute_NTWN() {
    Eigen::VectorXd b_v(order_); int s_i;
    for (int i = 0; i < x_.size(); ++i) {
        bspline::eval_bspline_basis(x_[i], order_, knots_, s_i, b_v, 0);
        for (int r = 0; r < order_; ++r) {
            int rg = s_i + r; if (rg >= n_basis_) continue;
            for (int c = 0; c <= r; ++c) { int cg = s_i + c; AB_template_(rg - cg, cg) += weights_[i] * b_v[r] * b_v[c]; }
        }
    }
}

void BSplineFitter::compute_penalty_matrix() {
    double pt = 1.0 / std::sqrt(3.0); double gp[2] = {-pt, pt}, gw[2] = {1.0, 1.0};
    int s_k = order_ - 1, e_k = (int)knots_.size() - order_; Eigen::VectorXd b_d(order_); int s_i;
    for (int k = s_k; k < e_k; ++k) {
        double t_s = knots_[k], t_e = knots_[k+1], dt = t_e - t_s; if (dt <= 1e-14) continue;
        for (int g = 0; g < 2; ++g) {
            bspline::eval_bspline_basis(0.5 * (t_s + t_e) + 0.5 * dt * gp[g], order_, knots_, s_i, b_d, 2);
            double w_v = 0.5 * dt * gw[g];
            for (int r = 0; r < order_; ++r) {
                int rg = s_i + r; if (rg >= n_basis_) continue;
                for (int c = 0; c <= r; ++c) { int cg = s_i + c; Omega_band_(rg - cg, cg) += w_v * b_d[r] * b_d[c]; }
            }
        }
    }
}

Eigen::VectorXd BSplineFitter::eval_basis(double x_val, int deriv) {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(n_basis_), b_v(order_); int s_i;
    bspline::eval_bspline_basis(x_val, order_, knots_, s_i, b_v, deriv);
    for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n_basis_) v[idx] = b_v[j]; }
    return v;
}

Eigen::VectorXd BSplineFitter::get_knots() { return knots_; }

Eigen::MatrixXd BSplineFitter::get_NTWN() {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis_, n_basis_); int kd = order_ - 1;
    for (int j = 0; j < n_basis_; ++j) { for (int i = 0; i <= kd; ++i) { int row = j + i; if (row < n_basis_) { double v = AB_template_(i, j); M(row, j) = v; M(j, row) = v; } } }
    return M;
}

Eigen::MatrixXd BSplineFitter::get_Omega() {
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis_, n_basis_); int kd = order_ - 1;
    for (int j = 0; j < n_basis_; ++j) { for (int i = 0; i <= kd; ++i) { int row = j + i; if (row < n_basis_) { double v = Omega_band_(i, j); M(row, j) = v; M(j, row) = v; } } }
    return M;
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> BSplineFitter::compute_system(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
    int n = n_basis_, kd = order_ - 1;
    Eigen::MatrixXd AB = AB_template_ + lamval * Omega_band_; Eigen::VectorXd b = Eigen::VectorXd::Zero(n), b_v(order_); int s_i;
    for(int i=0; i<x_.size(); ++i) {
        bspline::eval_bspline_basis(x_[i], order_, knots_, s_i, b_v, 0);
        for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n) b[idx] += weights_[i] * y[i] * b_v[j]; }
    }
    
    // Natural spline boundary conditions logic
    auto get_A = [&](int i, int j) -> double { if (i < j) std::swap(i, j); if (i - j > kd) return 0.0; return AB(i-j, j); };
    Eigen::VectorXd d2_v(order_); bspline::eval_bspline_basis(knots_[order_-1], order_, knots_, s_i, d2_v, 2);
    double v0 = d2_v[0], v1 = d2_v[1], v2 = d2_v[2]; if (std::abs(v0) < 1e-12) throw std::runtime_error("Leading zero: v0=" + std::to_string(v0));
    double ws1 = -v1 / v0, ws2 = -v2 / v0, a00 = get_A(0,0), a01 = get_A(1,0), a02 = get_A(2,0), a03 = get_A(3,0), a11 = get_A(1,1), a12 = get_A(2,1), a13 = get_A(3,1), a22 = get_A(2,2), a23 = get_A(3,2);
    
    AB(0, 1) = a11 + 2*ws1*a01 + ws1*ws1*a00; AB(1, 1) = a12 + ws1*a02 + ws2*a01 + ws1*ws2*a00; AB(0, 2) = a22 + 2*ws2*a02 + ws2*ws2*a00;
    if (kd >= 3) { AB(2, 1) = a13 + ws1*a03; AB(1, 2) = a23 + ws2*a03; }
    b[1] += ws1 * b[0]; b[2] += ws2 * b[0];
    bspline::eval_bspline_basis(knots_[knots_.size() - order_], order_, knots_, s_i, d2_v, 2);
    int iN1 = n-1-s_i, iN2 = n-2-s_i, iN3 = n-3-s_i;
    double u0 = d2_v[iN1], u1 = d2_v[iN2], u2 = d2_v[iN3]; if (std::abs(u0) < 1e-12) throw std::runtime_error("Trailing zero: u0=" + std::to_string(u0));
    double we1 = -u1 / u0, we2 = -u2 / u0, an11 = get_A(n-1, n-1), an12 = get_A(n-1, n-2), an13 = get_A(n-1, n-3), an14 = get_A(n-1, n-4), an22 = get_A(n-2, n-2), an23 = get_A(n-2, n-3), an24 = get_A(n-2, n-4), an33 = get_A(n-3, n-3), an34 = get_A(n-3, n-4);
    
    AB(0, n-2) = an22 + 2*we1*an12 + we1*we1*an11; AB(1, n-3) = an23 + we1*an13 + we2*an12 + we1*we2*an11; AB(0, n-3) = an33 + 2*we2*an13 + we2*we2*an11;
    if (kd >= 3) { AB(2, n-4) = an24 + we1*an14; AB(1, n-4) = an34 + we2*an14; }
    b[n-2] += we1 * b[n-1]; b[n-3] += we2 * b[n-1];

    return {AB, b};
}

void BSplineFitter::set_solution(const Eigen::Ref<const Eigen::VectorXd>& sol) {
    int n = n_basis_, s_i;
    if (sol.size() != n - 2) throw std::runtime_error("Solution size mismatch. Expected " + std::to_string(n-2) + ", got " + std::to_string(sol.size()));
    
    coeffs_ = Eigen::VectorXd::Zero(n);
    coeffs_.segment(1, n - 2) = sol;

    Eigen::VectorXd d2_v(order_); 
    bspline::eval_bspline_basis(knots_[order_-1], order_, knots_, s_i, d2_v, 2);
    double v0 = d2_v[0], v1 = d2_v[1], v2 = d2_v[2];
    double ws1 = -v1 / v0, ws2 = -v2 / v0;

    bspline::eval_bspline_basis(knots_[knots_.size() - order_], order_, knots_, s_i, d2_v, 2);
    int iN1 = n-1-s_i, iN2 = n-2-s_i, iN3 = n-3-s_i;
    double u0 = d2_v[iN1], u1 = d2_v[iN2], u2 = d2_v[iN3];
    double we1 = -u1 / u0, we2 = -u2 / u0;

    coeffs_[0] = ws1 * coeffs_[1] + ws2 * coeffs_[2]; 
    coeffs_[n-1] = we1 * coeffs_[n-2] + we2 * coeffs_[n-3];
}

Eigen::VectorXd BSplineFitter::fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
    auto system = compute_system(y, lamval);
    Eigen::MatrixXd AB = system.first;
    Eigen::VectorXd b = system.second;
    int n = n_basis_, kd = order_ - 1, ldab = kd + 1;
    int n_r = n - 2, nrhs = 1, info = 0; char u_c = 'L';
    
    dpbsv_(&u_c, &n_r, &kd, &nrhs, AB.data() + ldab * 1, &ldab, b.data() + 1, &n, &info);
    
    if (info > 0) {
        std::cerr << "dpbsv failed with info: " << info << std::endl;
        Eigen::MatrixXd Af = Eigen::MatrixXd::Zero(n_r, n_r);
        for (int j = 0; j < n_r; ++j) { for (int r = 0; r <= kd; ++r) { if (j + r < n_r) { double v = AB(r, j + 1); Af(j + r, j) = v; Af(j, j + r) = v; } } }
        Eigen::LDLT<Eigen::MatrixXd> sol; sol.compute(Af); b.segment(1, n_r) = sol.solve(b.segment(1, n_r));
    }
    
    set_solution(b.segment(1, n - 2));
    return coeffs_;
}

namespace {
    Eigen::SparseMatrix<double> banded_to_sparse(const Eigen::MatrixXd& band_mat, int n, int kd) {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(n * (kd + 1) * 2);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= kd; ++i) {
                int row = j + i;
                if (row < n) {
                    double val = band_mat(i, j);
                    if (std::abs(val) > 1e-14) {
                        triplets.push_back({row, j, val});
                        if (row != j) triplets.push_back({j, row, val});
                    }
                }
            }
        }
        Eigen::SparseMatrix<double> sparse_mat(n, n);
        sparse_mat.setFromTriplets(triplets.begin(), triplets.end());
        return sparse_mat;
    }
}

double BSplineFitter::compute_df(double lamval) {
    int n = n_basis_, kd = order_ - 1;
    
    Eigen::SparseMatrix<double> NTWN_sparse = banded_to_sparse(AB_template_, n, kd);
    Eigen::SparseMatrix<double> Omega_sparse = banded_to_sparse(Omega_band_, n, kd);
    
    int s_i;
    Eigen::VectorXd d2_v(order_); 
    bspline::eval_bspline_basis(knots_[order_-1], order_, knots_, s_i, d2_v, 2);
    double v0 = d2_v[0], v1 = d2_v[1], v2 = d2_v[2];
    double ws1 = -v1 / v0, ws2 = -v2 / v0;

    bspline::eval_bspline_basis(knots_[knots_.size() - order_], order_, knots_, s_i, d2_v, 2);
    int iN1 = n-1-s_i, iN2 = n-2-s_i, iN3 = n-3-s_i;
    double u0 = d2_v[iN1], u1 = d2_v[iN2], u2 = d2_v[iN3];
    double we1 = -u1 / u0, we2 = -u2 / u0;
    
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

double BSplineFitter::gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
    fit(y, lamval);
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

double BSplineFitter::solve_for_df(double target_df, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return compute_df(std::pow(10.0, log_lam)) - target_df; };
    return std::pow(10.0, utils::brent_root(func, min_log_lam, max_log_lam));
}

double BSplineFitter::solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return gcv_score(std::pow(10.0, log_lam), y); };
    return std::pow(10.0, utils::brent_min(func, min_log_lam, max_log_lam));
}

Eigen::VectorXd BSplineFitter::predict(const Eigen::Ref<const Eigen::VectorXd>& x_n, int deriv) {
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
