#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <stdexcept>

namespace py = pybind11;

namespace {
    double brent_root(std::function<double(double)> func, double a, double b, double tol=1e-6, int max_iter=20) {
        double fa = func(a);
        double fb = func(b);
        if (fa * fb > 0) {
             throw std::runtime_error("Root not bracketed: target DF out of feasible range.");
        }
        
        if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
        
        double c = a;
        double fc = fa;
        bool mflag = true;
        double s = 0;
        double d = 0;
        
        for (int i = 0; i < max_iter; ++i) {
            if (std::abs(b - a) < tol) return b;
            if (std::abs(fb) < tol) return b;

            if (fa != fc && fb != fc) {
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
                    (b * fa * fc) / ((fb - fa) * (fb - fc)) +
                    (c * fa * fb) / ((fc - fa) * (fc - fb));
            } else {
                s = b - fb * (b - a) / (fb - fa);
            }
            
            double tmp1 = (3 * a + b) / 4;
            bool cond1 = (s < std::min(tmp1, b) || s > std::max(tmp1, b));
            bool cond2 = mflag && (std::abs(s - b) >= (std::abs(b - c) / 2));
            bool cond3 = !mflag && (std::abs(s - b) >= (std::abs(c - d) / 2));
            bool cond4 = mflag && (std::abs(b - c) < tol);
            bool cond5 = !mflag && (std::abs(c - d) < tol);
            
            if (cond1 || cond2 || cond3 || cond4 || cond5) {
                s = (a + b) / 2;
                mflag = true;
            } else {
                mflag = false;
            }
            
            double fs = func(s);
            d = c;
            c = b;
            fc = fb;
            
            if (fa * fs < 0) {
                b = s;
                fb = fs;
            } else {
                a = s;
                fa = fs;
            }
            
            if (std::abs(fa) < std::abs(fb)) { std::swap(a, b); std::swap(fa, fb); }
        }
        return b;
    }

    double brent_min(std::function<double(double)> func, double a, double b, double tol=1e-5, int max_iter=20) {
        double x, w, v, fx, fw, fv;
        double d = 0.0, e = 0.0;
        double u, fu;
        const double gold = 0.3819660;
        
        x = w = v = a + gold * (b - a);
        fx = fw = fv = func(x);
        
        for(int iter = 0; iter < max_iter; ++iter) {
            double xm = 0.5 * (a + b);
            double tol1 = tol * std::abs(x) + 1e-10;
            double tol2 = 2.0 * tol1;
            
            if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
                return x;
            }
            
            if (std::abs(e) > tol1) {
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = std::abs(q);
                double etemp = e;
                e = d;
                
                if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
                    d = gold * (e = (x >= xm ? a - x : b - x));
                } else {
                    d = p / q;
                    u = x + d;
                    if (u - a < tol2 || b - u < tol2) {
                        d = (xm - x >= 0 ? 1 : -1) * tol1;
                    }
                }
            } else {
                d = gold * (e = (x >= xm ? a - x : b - x));
            }
            
            u = (std::abs(d) >= tol1 ? x + d : x + (d > 0 ? 1 : -1) * tol1);
            fu = func(u);
            
            if (fu <= fx) {
                if (u >= x) a = x; else b = x;
                v = w; w = x; x = u;
                fv = fw; fw = fx; fx = fu;
            } else {
                if (u < x) a = u; else b = u;
                if (fu <= fw || w == x) {
                    v = w; w = u;
                    fv = fw; fw = fu;
                } else if (fu <= fv || v == x || v == w) {
                    v = u;
                    fv = fu;
                }
            }
        }
        return x;
    }
}

Eigen::MatrixXd compute_natural_spline_basis(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& knots,
    bool extrapolate_linear = true,
    int derivative_order = 0
) {
    long n_knots = knots.size();
    long n_x = x.size();

    Eigen::VectorXd h = knots.segment(1, n_knots - 1) - knots.segment(0, n_knots - 1);
    long n_inner = n_knots - 2;
    if (n_inner < 0) {
        if (n_knots == 2) {
             Eigen::MatrixXd N(n_x, 2);
             for (long i = 0; i < n_x; ++i) {
                 if (derivative_order == 0) {
                     double t = (x[i] - knots[0]) / h[0];
                     N(i, 0) = 1.0 - t;
                     N(i, 1) = t;
                 } else if (derivative_order == 1) {
                     double dt = 1.0 / h[0];
                     N(i, 0) = -dt;
                     N(i, 1) = dt;
                 } else {
                     N(i, 0) = 0.0;
                     N(i, 1) = 0.0;
                 }
             }
             return N;
        }
        throw std::runtime_error("Need at least 2 knots.");
    }

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_inner, n_inner);
    for (long i = 0; i < n_inner; ++i) {
        A(i, i) = (h[i] + h[i+1]) / 3.0;
        if (i < n_inner - 1) {
            A(i, i+1) = h[i+1] / 6.0;
            A(i+1, i) = h[i+1] / 6.0;
        }
    }

    Eigen::LLT<Eigen::MatrixXd> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
         throw std::runtime_error("Decomposition failed for spline system.");
    }

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(n_inner, n_knots);
    for (long i = 0; i < n_inner; ++i) {
        B(i, i)     = 1.0 / h[i];
        B(i, i+1)   = -1.0 / h[i] - 1.0 / h[i+1];
        B(i, i+2)   = 1.0 / h[i+1];
    }
    
    Eigen::MatrixXd M_inner = solver.solve(B);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_knots, n_knots);
    M.block(1, 0, n_inner, n_knots) = M_inner;
    
    Eigen::RowVectorXd d1_left(n_knots);
    Eigen::RowVectorXd d1_right(n_knots);
    double inv_h0 = 1.0 / h[0];
    double h0_6 = h[0] / 6.0;
    d1_left = -h0_6 * M.row(1);
    d1_left[1] += inv_h0;
    d1_left[0] -= inv_h0;

    long last_idx = n_knots - 1;
    long pen_idx = n_knots - 2;
    double hk = h[pen_idx];
    double inv_hk = 1.0 / hk;
    double hk_6 = hk / 6.0;
    d1_right = hk_6 * M.row(pen_idx);
    d1_right[last_idx] += inv_hk;
    d1_right[pen_idx] -= inv_hk;

    Eigen::MatrixXd N(n_x, n_knots);
    for (long i = 0; i < n_x; ++i) {
        double val = x[i];
        if (extrapolate_linear && val < knots[0]) {
             if (derivative_order == 0) {
                 N.row(i) = d1_left * (val - knots[0]);
                 N(i, 0) += 1.0;
             } else if (derivative_order == 1) {
                 N.row(i) = d1_left;
             } else {
                 N.row(i).setZero();
             }
             continue;
        }
        if (extrapolate_linear && val > knots[n_knots - 1]) {
             if (derivative_order == 0) {
                 N.row(i) = d1_right * (val - knots[n_knots - 1]);
                 N(i, n_knots - 1) += 1.0;
             } else if (derivative_order == 1) {
                 N.row(i) = d1_right;
             } else {
                 N.row(i).setZero();
             }
             continue;
        }

        long k = 0;
        if (val < knots[0]) {
            k = 0; 
        } else if (val >= knots[n_knots - 1]) {
            k = n_knots - 2; 
        } else {
            auto it = std::upper_bound(knots.data(), knots.data() + n_knots, val);
            k = std::distance(knots.data(), it) - 1;
            if (k < 0) k = 0;
            if (k >= n_knots - 1) k = n_knots - 2;
        }
        
        double hk_val = h[k];
        double diff = val - knots[k];
        double t = diff / hk_val;
        
        if (derivative_order == 0) {
            double term1 = (1.0 - t);
            double term2 = t;
            double term3 = (std::pow(1.0 - t, 3) - (1.0 - t)) * hk_val * hk_val / 6.0;
            double term4 = (std::pow(t, 3) - t) * hk_val * hk_val / 6.0;
            N.row(i) = term3 * M.row(k) + term4 * M.row(k+1);
            N(i, k)   += term1;
            N(i, k+1) += term2;
        } else if (derivative_order == 1) {
            double term1 = -1.0 / hk_val;
            double term2 = 1.0 / hk_val;
            double d_term3_dt = -3.0 * std::pow(1.0 - t, 2) + 1.0;
            double term3 = d_term3_dt * (hk_val / 6.0); 
            double d_term4_dt = 3.0 * std::pow(t, 2) - 1.0;
            double term4 = d_term4_dt * (hk_val / 6.0);
            N.row(i) = term3 * M.row(k) + term4 * M.row(k+1);
            N(i, k)   += term1;
            N(i, k+1) += term2;
        } else if (derivative_order == 2) {
            double d2_term3_dt2 = 6.0 * (1.0 - t);
            double term3 = d2_term3_dt2 / 6.0; 
            double d2_term4_dt2 = 6.0 * t;
            double term4 = d2_term4_dt2 / 6.0;
            N.row(i) = term3 * M.row(k) + term4 * M.row(k+1);
        }
    }
    return N;
}

Eigen::MatrixXd compute_penalty_matrix(const Eigen::Ref<const Eigen::VectorXd>& knots) {
    long n_knots = knots.size();
    if (n_knots < 2) return Eigen::MatrixXd::Zero(n_knots, n_knots);
    Eigen::VectorXd h = knots.segment(1, n_knots - 1) - knots.segment(0, n_knots - 1);
    Eigen::VectorXd inv_h = h.cwiseInverse();
    long n_inner = n_knots - 2;
    if (n_inner <= 0) return Eigen::MatrixXd::Zero(n_knots, n_knots);
    Eigen::SparseMatrix<double> R(n_inner, n_inner);
    std::vector<Eigen::Triplet<double>> r_triplets;
    for (int i = 0; i < n_inner; ++i) {
        r_triplets.push_back({i, i, (h[i] + h[i+1]) / 3.0});
        if (i < n_inner - 1) {
            r_triplets.push_back({i, i+1, h[i+1] / 6.0});
            r_triplets.push_back({i+1, i, h[i+1] / 6.0});
        }
    }
    R.setFromTriplets(r_triplets.begin(), r_triplets.end());
    Eigen::SparseMatrix<double> Q(n_knots, n_inner);
    std::vector<Eigen::Triplet<double>> q_triplets;
    for (int j = 0; j < n_inner; ++j) {
        q_triplets.push_back({j, j, inv_h[j]});
        q_triplets.push_back({j+1, j, -inv_h[j] - inv_h[j+1]});
        q_triplets.push_back({j+2, j, inv_h[j+1]});
    }
    Q.setFromTriplets(q_triplets.begin(), q_triplets.end());
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(R);
    if(solver.info() != Eigen::Success) throw std::runtime_error("R factorization failed");
    Eigen::MatrixXd QT = Q.transpose();
    Eigen::MatrixXd X = solver.solve(QT);
    return Q * X;
}

class SplineFitterCpp {
    Eigen::MatrixXd N_;
    Eigen::MatrixXd Omega_;
    Eigen::MatrixXd NTW_; 
    Eigen::MatrixXd NTWN_; 
    Eigen::VectorXd knots_;
    Eigen::VectorXd alpha_;
public:
    SplineFitterCpp(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& knots, py::object weights_obj) {
        knots_ = knots;
        N_ = compute_natural_spline_basis(x, knots, true, 0);
        Omega_ = compute_penalty_matrix(knots);
        update_weights(weights_obj);
    }
    void update_weights(py::object weights_obj) {
        if (weights_obj.is_none()) {
            NTW_ = N_.transpose(); 
            NTWN_ = N_.transpose() * N_;
        } else {
            Eigen::VectorXd weights = weights_obj.cast<Eigen::VectorXd>();
            NTW_ = N_.transpose(); 
            for (int i = 0; i < NTW_.cols(); ++i) NTW_.col(i) *= weights[i];
            NTWN_ = NTW_ * N_;
        }
    }
    Eigen::VectorXd fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
        Eigen::MatrixXd LHS = NTWN_ + lamval * Omega_;
        Eigen::VectorXd RHS = NTW_ * y;
        Eigen::LLT<Eigen::MatrixXd> solver;
        solver.compute(LHS);
        if (solver.info() != Eigen::Success) {
             Eigen::LDLT<Eigen::MatrixXd> solver_ldlt;
             solver_ldlt.compute(LHS);
             alpha_ = solver_ldlt.solve(RHS);
        } else alpha_ = solver.solve(RHS);
        return alpha_;
    }
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv=0) {
        Eigen::MatrixXd N_new = compute_natural_spline_basis(x_new, knots_, true, deriv);
        return N_new * alpha_;
    }
    double compute_df(double lamval) {
        Eigen::MatrixXd LHS = NTWN_ + lamval * Omega_;
        Eigen::LLT<Eigen::MatrixXd> solver;
        solver.compute(LHS);
        if (solver.info() != Eigen::Success) return 0.0;
        Eigen::MatrixXd X = solver.solve(NTWN_);
        return X.trace();
    }
    double gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
        Eigen::VectorXd alpha = fit(y, lamval);
        Eigen::VectorXd f = N_ * alpha;
        double rss = (y - f).squaredNorm(); 
        double df = compute_df(lamval);
        double n = (double)y.size();
        double denom = 1.0 - df / n;
        if (denom < 1e-6) return 1e20;
        return (rss / n) / (denom * denom);
    }
    double solve_for_df(double target_df) {
        auto func = [&](double log_lam) { return compute_df(std::pow(10.0, log_lam)) - target_df; };
        return std::pow(10.0, brent_root(func, -12.0, 12.0));
    }
    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0) {
        auto func = [&](double log_lam) { return gcv_score(std::pow(10.0, log_lam), y); };
        return std::pow(10.0, brent_min(func, min_log_lam, max_log_lam));
    }
    Eigen::MatrixXd get_N() { return N_; }
    Eigen::MatrixXd get_Omega() { return Omega_; }
};

class CubicSplineTraceCpp {
    Eigen::VectorXd x_;
    int n_;
    Eigen::VectorXd h_;
    Eigen::VectorXd weights_inv_;
    Eigen::VectorXd R_diag, R_off, M_diag, M_off1, M_off2;
    void compute_QR_bands() {
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
public:
    CubicSplineTraceCpp(const Eigen::Ref<const Eigen::VectorXd>& x, py::object weights_obj) {
        x_ = x; n_ = x.size(); h_.resize(n_ - 1);
        for (int i = 0; i < n_ - 1; ++i) h_(i) = x_(i + 1) - x_(i);
        weights_inv_.resize(n_);
        if (weights_obj.is_none()) weights_inv_.setOnes();
        else weights_inv_ = weights_obj.cast<Eigen::VectorXd>().cwiseInverse();
        compute_QR_bands();
    }
    double compute_trace(double lam) {
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
};

class SplineFitterReinschCpp {
    Eigen::SparseMatrix<double> Q_, R_, M_; Eigen::VectorXd weights_inv_, x_, gamma_, f_; long n_;
public:
    SplineFitterReinschCpp(const Eigen::Ref<const Eigen::VectorXd>& x, py::object weights_obj) {
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
    void update_weights(py::object weights_obj) {
        if (weights_obj.is_none()) weights_inv_ = Eigen::VectorXd::Ones(n_);
        else weights_inv_ = weights_obj.cast<Eigen::VectorXd>().cwiseInverse();
        Eigen::SparseMatrix<double> Winv(n_, n_);
        std::vector<Eigen::Triplet<double>> w_t;
        for(int i=0; i<n_; ++i) w_t.push_back({i, i, weights_inv_[i]});
        Winv.setFromTriplets(w_t.begin(), w_t.end());
        M_ = Q_.transpose() * Winv * Q_;
    }
    Eigen::VectorXd fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
        Eigen::VectorXd QT_y = Q_.transpose() * y;
        Eigen::SparseMatrix<double> LHS = R_ + lamval * M_;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(LHS);
        gamma_ = solver.solve(QT_y);
        f_ = y - lamval * weights_inv_.cwiseProduct(Q_ * gamma_);
        return f_;
    }
    double compute_df(double lamval) {
        Eigen::VectorXd original_weights = weights_inv_.cwiseInverse();
        CubicSplineTraceCpp trace_solver(x_, py::cast(original_weights));
        return trace_solver.compute_trace(lamval);
    }
    double gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
        Eigen::VectorXd f = fit(y, lamval);
        double rss = ((y - f).array().square() * weights_inv_.cwiseInverse().array()).sum();
        double df = compute_df(lamval), n = (double)y.size(), denom = 1.0 - df / n;
        return (denom < 1e-6) ? 1e20 : (rss / n) / (denom * denom);
    }
    double solve_for_df(double target_df) {
        auto func = [&](double log_lam) { return compute_df(std::pow(10.0, log_lam)) - target_df; };
        return std::pow(10.0, brent_root(func, -12.0, 12.0));
    }
    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0) {
        auto func = [&](double log_lam) { return gcv_score(std::pow(10.0, log_lam), y); };
        return std::pow(10.0, brent_min(func, min_log_lam, max_log_lam));
    }
    double compute_df_sparse(double lamval) {
        Eigen::SparseMatrix<double> LHS = R_ + lamval * M_;
        Eigen::MatrixXd LHS_dense = LHS;
        Eigen::MatrixXd M_dense = M_;
        Eigen::LLT<Eigen::MatrixXd> solver;
        solver.compute(LHS_dense);
        if(solver.info() != Eigen::Success) return 0.0;
        Eigen::MatrixXd Sol = solver.solve(M_dense);
        return (double)n_ - lamval * Sol.trace();
    }
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv=0) {
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
};

extern "C" {
    void dpbsv_(const char *uplo, const int *n, const int *kd, const int *nrhs,
                double *ab, const int *ldab, double *b, const int *ldb, int *info);
}

namespace bspline {
    void eval_bspline_basis(double x, int k, const Eigen::VectorXd& knots, int& span_index, Eigen::VectorXd& N, int deriv=0) {
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

class SplineFitterBSpline {
    int order_; Eigen::VectorXd knots_, coeffs_, weights_, x_; int n_basis_; Eigen::MatrixXd AB_template_, Omega_band_;
public:
    SplineFitterBSpline(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& i_k, py::object w_obj, int order = 4) : order_(order), x_(x) {
        long n_i = i_k.size(), n_a = n_i + 2 * (order - 1); knots_.resize(n_a);
        for(int i=0; i<order; ++i) knots_[i] = i_k[0];
        if (n_i > 2) knots_.segment(order, n_i - 2) = i_k.segment(1, n_i - 2);
        for(int i=0; i<order; ++i) knots_[n_a - order + i] = i_k[n_i - 1];
        n_basis_ = (int)(n_a - order);
        weights_ = w_obj.is_none() ? Eigen::VectorXd::Ones(x.size()) : w_obj.cast<Eigen::VectorXd>();
        int kd = order - 1; AB_template_ = Eigen::MatrixXd::Zero(kd + 1, n_basis_); compute_NTWN();
        Omega_band_ = Eigen::MatrixXd::Zero(kd + 1, n_basis_); compute_penalty_matrix();
    }
    void compute_NTWN() {
        Eigen::VectorXd b_v(order_); int s_i;
        for (int i = 0; i < x_.size(); ++i) {
            bspline::eval_bspline_basis(x_[i], order_, knots_, s_i, b_v, 0);
            for (int r = 0; r < order_; ++r) {
                int rg = s_i + r; if (rg >= n_basis_) continue;
                for (int c = 0; c <= r; ++c) { int cg = s_i + c; AB_template_(rg - cg, cg) += weights_[i] * b_v[r] * b_v[c]; }
            }
        }
    }
    void compute_penalty_matrix() {
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
    Eigen::VectorXd eval_basis(double x_val, int deriv=0) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(n_basis_), b_v(order_); int s_i;
        bspline::eval_bspline_basis(x_val, order_, knots_, s_i, b_v, deriv);
        for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n_basis_) v[idx] = b_v[j]; }
        return v;
    }
    Eigen::VectorXd get_knots() { return knots_; }
    Eigen::MatrixXd get_NTWN() {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis_, n_basis_); int kd = order_ - 1;
        for (int j = 0; j < n_basis_; ++j) { for (int i = 0; i <= kd; ++i) { int row = j + i; if (row < n_basis_) { double v = AB_template_(i, j); M(row, j) = v; M(j, row) = v; } } }
        return M;
    }
    Eigen::MatrixXd get_Omega() {
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n_basis_, n_basis_); int kd = order_ - 1;
        for (int j = 0; j < n_basis_; ++j) { for (int i = 0; i <= kd; ++i) { int row = j + i; if (row < n_basis_) { double v = Omega_band_(i, j); M(row, j) = v; M(j, row) = v; } } }
        return M;
    }
    void fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
        int n = n_basis_, kd = order_ - 1, ldab = kd + 1;
        Eigen::MatrixXd AB = AB_template_ + lamval * Omega_band_; Eigen::VectorXd b = Eigen::VectorXd::Zero(n), b_v(order_); int s_i;
        for(int i=0; i<x_.size(); ++i) {
            bspline::eval_bspline_basis(x_[i], order_, knots_, s_i, b_v, 0);
            for(int j=0; j<order_; ++j) { int idx = s_i + j; if(idx < n) b[idx] += weights_[i] * y[i] * b_v[j]; }
        }
        auto get_A = [&](int i, int j) -> double { if (i < j) std::swap(i, j); if (i - j > kd) return 0.0; return AB(i-j, j); };
        Eigen::VectorXd d2_v(order_); bspline::eval_bspline_basis(knots_[order_-1], order_, knots_, s_i, d2_v, 2);
        double v0 = d2_v[0], v1 = d2_v[1], v2 = d2_v[2]; if (std::abs(v0) < 1e-12) throw std::runtime_error("Leading zero: v0=" + std::to_string(v0));
        double ws1 = -v1 / v0, ws2 = -v2 / v0, a00 = get_A(0,0), a01 = get_A(1,0), a02 = get_A(2,0), a03 = get_A(3,0), a11 = get_A(1,1), a12 = get_A(2,1), a13 = get_A(3,1), a22 = get_A(2,2), a23 = get_A(3,2);
        // std::cout << "ws1: " << ws1 << ", ws2: " << ws2 << ", v0: " << v0 << std::endl;
        AB(0, 1) = a11 + 2*ws1*a01 + ws1*ws1*a00; AB(1, 1) = a12 + ws1*a02 + ws2*a01 + ws1*ws2*a00; AB(0, 2) = a22 + 2*ws2*a02 + ws2*ws2*a00;
        if (kd >= 3) { AB(2, 1) = a13 + ws1*a03; AB(1, 2) = a23 + ws2*a03; }
        b[1] += ws1 * b[0]; b[2] += ws2 * b[0];
        bspline::eval_bspline_basis(knots_[knots_.size() - order_], order_, knots_, s_i, d2_v, 2);
        int iN1 = n-1-s_i, iN2 = n-2-s_i, iN3 = n-3-s_i;
        double u0 = d2_v[iN1], u1 = d2_v[iN2], u2 = d2_v[iN3]; if (std::abs(u0) < 1e-12) throw std::runtime_error("Trailing zero: u0=" + std::to_string(u0));
        double we1 = -u1 / u0, we2 = -u2 / u0, an11 = get_A(n-1, n-1), an12 = get_A(n-1, n-2), an13 = get_A(n-1, n-3), an14 = get_A(n-1, n-4), an22 = get_A(n-2, n-2), an23 = get_A(n-2, n-3), an24 = get_A(n-2, n-4), an33 = get_A(n-3, n-3), an34 = get_A(n-3, n-4);
        // std::cout << "we1: " << we1 << ", we2: " << we2 << ", u0: " << u0 << std::endl;
        AB(0, n-2) = an22 + 2*we1*an12 + we1*we1*an11; AB(1, n-3) = an23 + we1*an13 + we2*an12 + we1*we2*an11; AB(0, n-3) = an33 + 2*we2*an13 + we2*we2*an11;
        if (kd >= 3) { AB(2, n-4) = an24 + we1*an14; AB(1, n-4) = an34 + we2*an14; }
        b[n-2] += we1 * b[n-1]; b[n-3] += we2 * b[n-1];
        int n_r = n - 2, nrhs = 1, info = 0; char u_c = 'L'; dpbsv_(&u_c, &n_r, &kd, &nrhs, AB.data() + ldab * 1, &ldab, b.data() + 1, &n, &info);
        if (info > 0) {
            std::cerr << "dpbsv failed with info: " << info << std::endl;
            Eigen::MatrixXd Af = Eigen::MatrixXd::Zero(n_r, n_r);
            for (int j = 0; j < n_r; ++j) { for (int r = 0; r <= kd; ++r) { if (j + r < n_r) { double v = AB(r, j + 1); Af(j + r, j) = v; Af(j, j + r) = v; } } }
            Eigen::LDLT<Eigen::MatrixXd> sol; sol.compute(Af); b.segment(1, n_r) = sol.solve(b.segment(1, n_r));
        }
        coeffs_ = b; coeffs_[0] = ws1 * coeffs_[1] + ws2 * coeffs_[2]; coeffs_[n-1] = we1 * coeffs_[n-2] + we2 * coeffs_[n-3];
    }
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_n, int deriv=0) {
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
};

PYBIND11_MODULE(_spline_extension, m) {
    py::class_<SplineFitterCpp>(m, "SplineFitterCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("knots"), py::arg("weights_obj") = py::none())
        .def("fit", &SplineFitterCpp::fit)
        .def("update_weights", &SplineFitterCpp::update_weights)
        .def("compute_df", &SplineFitterCpp::compute_df)
        .def("gcv_score", &SplineFitterCpp::gcv_score)
        .def("solve_for_df", &SplineFitterCpp::solve_for_df)
        .def("solve_gcv", &SplineFitterCpp::solve_gcv)
        .def("predict", &SplineFitterCpp::predict)
        .def("get_N", &SplineFitterCpp::get_N)
        .def("get_Omega", &SplineFitterCpp::get_Omega);

    py::class_<SplineFitterReinschCpp>(m, "SplineFitterReinschCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("weights") = py::none())
        .def("fit", &SplineFitterReinschCpp::fit)
        .def("update_weights", &SplineFitterReinschCpp::update_weights)
        .def("compute_df", &SplineFitterReinschCpp::compute_df)
        .def("compute_df_sparse", &SplineFitterReinschCpp::compute_df_sparse)
        .def("gcv_score", &SplineFitterReinschCpp::gcv_score)
        .def("solve_for_df", &SplineFitterReinschCpp::solve_for_df)
        .def("solve_gcv", &SplineFitterReinschCpp::solve_gcv)
        .def("predict", &SplineFitterReinschCpp::predict);

    py::class_<CubicSplineTraceCpp>(m, "CubicSplineTraceCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("weights_obj") = py::none())
        .def("compute_trace", &CubicSplineTraceCpp::compute_trace);

    py::class_<SplineFitterBSpline>(m, "SplineFitterBSpline").def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&, py::object, int>()).def("fit", &SplineFitterBSpline::fit).def("predict", &SplineFitterBSpline::predict).def("get_NTWN", &SplineFitterBSpline::get_NTWN).def("get_Omega", &SplineFitterBSpline::get_Omega).def("get_knots", &SplineFitterBSpline::get_knots).def("eval_basis", &SplineFitterBSpline::eval_basis, py::arg("x_val"), py::arg("deriv")=0);
    
    m.def("compute_natural_spline_basis", &compute_natural_spline_basis,
          py::arg("x"), py::arg("knots"), py::arg("extrapolate_linear") = true, py::arg("derivative_order") = 0);
    m.def("compute_penalty_matrix", &compute_penalty_matrix, py::arg("knots"));
}
