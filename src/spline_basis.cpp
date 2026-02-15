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

namespace py = pybind11;

namespace {
    double brent_root(std::function<double(double)> func, double a, double b, double tol=1e-6, int max_iter=100) {
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

    // Brent's method for minimization
    double brent_min(std::function<double(double)> func, double a, double b, double tol=1e-5, int max_iter=100) {
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

/**
 * Computes the Natural Cubic Spline Basis matrix.
 */
Eigen::MatrixXd compute_natural_spline_basis(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& knots,
    bool extrapolate_linear = true,
    int derivative_order = 0
) {
    long n_knots = knots.size();
    long n_x = x.size();

    // 1. Setup the tridiagonal system for second derivatives (M)
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
    
    // Boundary derivatives
    Eigen::RowVectorXd d1_left(n_knots);
    Eigen::RowVectorXd d1_right(n_knots);

    // Left boundary x0
    double inv_h0 = 1.0 / h[0];
    double h0_6 = h[0] / 6.0;
    d1_left = -h0_6 * M.row(1);
    d1_left[1] += inv_h0;
    d1_left[0] -= inv_h0;

    // Right boundary x_{N-1}
    long last_idx = n_knots - 1;
    long pen_idx = n_knots - 2;
    double hk = h[pen_idx];
    double inv_hk = 1.0 / hk;
    double hk_6 = hk / 6.0;
    d1_right = hk_6 * M.row(pen_idx);
    d1_right[last_idx] += inv_hk;
    d1_right[pen_idx] -= inv_hk;

    // 3. Evaluate the splines at x
    Eigen::MatrixXd N(n_x, n_knots);
    
    for (long i = 0; i < n_x; ++i) {
        double val = x[i];
        
        // --- Extrapolation ---
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

        // --- Standard Cubic Evaluation ---
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
            // d/dx term3:
            // d/dt term3 * dt/dx = d/dt term3 * (1/h)
            // d/dt [ (1-t)^3 - (1-t) ] = -3(1-t)^2 + 1
            double d_term3_dt = -3.0 * std::pow(1.0 - t, 2) + 1.0;
            double term3 = d_term3_dt * (hk_val / 6.0); // h^2/6 * 1/h = h/6
            
            // d/dt [ t^3 - t ] = 3t^2 - 1
            double d_term4_dt = 3.0 * std::pow(t, 2) - 1.0;
            double term4 = d_term4_dt * (hk_val / 6.0);

            N.row(i) = term3 * M.row(k) + term4 * M.row(k+1);
            N(i, k)   += term1;
            N(i, k+1) += term2;
        } else if (derivative_order == 2) {
            // d2/dx2 term3:
            // d2/dt2 term3 * (1/h^2)
            // d/dt [-3(1-t)^2 + 1] = -3 * 2 * (1-t) * (-1) = 6(1-t)
            double d2_term3_dt2 = 6.0 * (1.0 - t);
            double term3 = d2_term3_dt2 / 6.0; // h^2/6 * 1/h^2 = 1/6

            // d/dt [3t^2 - 1] = 6t
            double d2_term4_dt2 = 6.0 * t;
            double term4 = d2_term4_dt2 / 6.0;

            N.row(i) = term3 * M.row(k) + term4 * M.row(k+1);
            // Linear parts vanish
        }
    }

    return N;
}


/**
 * Computes the Natural Spline Penalty Matrix Omega.
 */
Eigen::MatrixXd compute_penalty_matrix(const Eigen::Ref<const Eigen::VectorXd>& knots) {
    long n_knots = knots.size();
    if (n_knots < 2) return Eigen::MatrixXd::Zero(n_knots, n_knots);
    
    Eigen::VectorXd h = knots.segment(1, n_knots - 1) - knots.segment(0, n_knots - 1);
    Eigen::VectorXd inv_h = h.cwiseInverse();
    
    long n_inner = n_knots - 2;
    if (n_inner <= 0) return Eigen::MatrixXd::Zero(n_knots, n_knots);

    Eigen::SparseMatrix<double> R(n_inner, n_inner);
    std::vector<Eigen::Triplet<double>> r_triplets;
    r_triplets.reserve(n_inner * 3);
    
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
    q_triplets.reserve(n_inner * 3);
    
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
    
    Eigen::MatrixXd Omega = Q * X;
    return Omega;
}

class SplineFitterCpp {
    Eigen::MatrixXd N_;
    Eigen::MatrixXd Omega_;
    Eigen::MatrixXd NTW_; 
    Eigen::MatrixXd NTWN_; 
    Eigen::VectorXd knots_;
    Eigen::VectorXd alpha_;
    
public:
    SplineFitterCpp(const Eigen::Ref<const Eigen::VectorXd>& x, 
                    const Eigen::Ref<const Eigen::VectorXd>& knots,
                    py::object weights_obj) {
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
            if (weights.size() != N_.rows()) throw std::invalid_argument("Weights size mismatch");
            
            NTW_ = N_.transpose(); 
            // Multiply each COL of NTW_ (row of N) by weights[i]
            for (int i = 0; i < NTW_.cols(); ++i) {
                NTW_.col(i) *= weights[i];
            }
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
             if (solver_ldlt.info() != Eigen::Success) {
                  throw std::runtime_error("Solver failed");
             }
             alpha_ = solver_ldlt.solve(RHS);
        } else {
             alpha_ = solver.solve(RHS);
        }
        
        return alpha_;
    }

    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv=0) {
        if (alpha_.size() == 0) throw std::runtime_error("Model not fitted");
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
        // Unweighted residuals for now, as before
        Eigen::VectorXd alpha = fit(y, lamval);
        Eigen::VectorXd f = N_ * alpha;
        Eigen::VectorXd resid = y - f;
        double rss = resid.squaredNorm(); 
        
        double df = compute_df(lamval);
        double n = (double)y.size();
        double denom = 1.0 - df / n;
        if (denom < 1e-6) return 1e20; // singularity
        
        return (rss / n) / (denom * denom);
    }

    double solve_for_df(double target_df) {
        auto func = [&](double log_lam) {
            double lam = std::pow(10.0, log_lam);
            return compute_df(lam) - target_df;
        };
        
        double log_lam_opt = brent_root(func, -12.0, 12.0);
        return std::pow(10.0, log_lam_opt);
    }

    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0) {
        auto func = [&](double log_lam) {
            double lam = std::pow(10.0, log_lam);
            return gcv_score(lam, y);
        };
        double log_lam_opt = brent_min(func, min_log_lam, max_log_lam);
        return std::pow(10.0, log_lam_opt);
    }

    Eigen::MatrixXd get_N() { return N_; }
    Eigen::MatrixXd get_Omega() { return Omega_; }
};

class CubicSplineTraceCpp {
public:
    CubicSplineTraceCpp(const Eigen::VectorXd& x) {
        this->x = x;
        this->n = x.size();
        if (n < 4) {
            throw std::invalid_argument("Need at least 4 points for cubic spline smoothing.");
        }
        h.resize(n - 1);
        for (int i = 0; i < n - 1; ++i) {
            h(i) = x(i + 1) - x(i);
        }
        compute_QR_bands();
    }

    double compute_trace(double lam) {
        int n_inner = n - 2;

        // Construct B = R + lam * M
        Eigen::VectorXd B_diag = R_diag + lam * M_diag;
        B_diag.array() += 1e-9; // Add a small epsilon for numerical stability
        Eigen::VectorXd B_off1 = R_off + lam * M_off1;
        Eigen::VectorXd B_off2 = lam * M_off2;
        
        // Eigen's BandMatrix is not available in the standard library.
        // For simplicity, we can represent the pentadiagonal matrix B
        // and perform Cholesky decomposition.
        // However, a full matrix would be O(N^3).
        // To maintain O(N), we need a banded Cholesky solver.
        // Let's implement a simple banded Cholesky for a pentadiagonal matrix.

        // Packed format for banded Cholesky (lower triangular part)
        // L_banded(0, i) = L_ii
        // L_banded(1, i) = L_{i+1, i}
        // L_banded(2, i) = L_{i+2, i}
        Eigen::MatrixXd L_banded = Eigen::MatrixXd::Zero(3, n_inner);

        for (int i = 0; i < n_inner; ++i) {
            double l_ii_sq = B_diag(i);
            if (i > 0) l_ii_sq -= std::pow(L_banded(1, i-1), 2);
            if (i > 1) l_ii_sq -= std::pow(L_banded(2, i-2), 2);
            
            if (l_ii_sq <= 0) {
                 return std::numeric_limits<double>::quiet_NaN();
            }
            L_banded(0, i) = std::sqrt(l_ii_sq);

            if (i < n_inner - 1) {
                double l_i1_i = B_off1(i);
                if (i > 0) l_i1_i -= L_banded(1, i-1) * L_banded(2, i-1);
                L_banded(1, i) = l_i1_i / L_banded(0, i);
            }
            if (i < n_inner - 2) {
                L_banded(2, i) = B_off2(i) / L_banded(0, i);
            }
        }
        
        // Takahashi's equations for B^-1
        Eigen::VectorXd Inv_diag = Eigen::VectorXd::Zero(n_inner);
        Eigen::VectorXd Inv_off1 = Eigen::VectorXd::Zero(n_inner - 1);
        
        for (int i = n_inner - 1; i >= 0; --i) {
            double val = 1.0 / L_banded(0, i);
            double sum_diag = 0.0;

            if (i + 1 < n_inner) {
                double s_off = L_banded(1, i) * Inv_diag(i + 1);
                if (i + 2 < n_inner) {
                    s_off += L_banded(2, i) * Inv_off1(i + 1);
                }
                Inv_off1(i) = -val * s_off;
                sum_diag += L_banded(1, i) * Inv_off1(i);
            }

            if (i + 2 < n_inner) {
                double s_off2 = L_banded(1, i) * Inv_off1(i + 1) + L_banded(2, i) * Inv_diag(i + 2);
                double X_i_i2 = -val * s_off2;
                sum_diag += L_banded(2, i) * X_i_i2;
            }
            Inv_diag(i) = val * (val - sum_diag);
        }

        // Final Trace Calculation
        double tr_RBinv = (R_diag.array() * Inv_diag.array()).sum();
        if (n_inner > 1) {
            tr_RBinv += 2 * (R_off.array() * Inv_off1.array()).sum();
        }
        
        return 2.0 + tr_RBinv;
    }

private:
    Eigen::VectorXd x;
    int n;
    Eigen::VectorXd h;

    Eigen::VectorXd R_diag, R_off;
    Eigen::VectorXd M_diag, M_off1, M_off2;

    void compute_QR_bands() {
        int n_inner = n - 2;
        
        // R bands
        R_diag.resize(n_inner);
        R_off.resize(n_inner - 1);
        for (int i = 0; i < n_inner; ++i) {
            R_diag(i) = (h(i) + h(i + 1)) / 3.0;
        }
        for (int i = 0; i < n_inner - 1; ++i) {
            R_off(i) = h(i + 1) / 6.0;
        }

        // M = Q^T Q bands
        M_diag.resize(n_inner);
        M_off1.resize(n_inner - 1);
        M_off2.resize(n_inner - 2);

        Eigen::VectorXd q0_val = h.head(n_inner).cwiseInverse();
        Eigen::VectorXd q2_val = h.tail(n_inner).cwiseInverse();
        Eigen::VectorXd q1_val = -(h.head(n_inner).cwiseInverse() + h.tail(n_inner).cwiseInverse());

        M_diag = q0_val.array().square() + q1_val.array().square() + q2_val.array().square();
        M_off1 = q1_val.head(n_inner - 1).array() * q0_val.tail(n_inner - 1).array() + q2_val.head(n_inner - 1).array() * q1_val.tail(n_inner - 1).array();
        M_off2 = q2_val.head(n_inner - 2).array() * q0_val.tail(n_inner - 2).array();
    }
};

/**
 * Reinsch Form Fitter (O(N))
 */
class SplineFitterReinschCpp {
    Eigen::SparseMatrix<double> Q_;
    Eigen::SparseMatrix<double> R_;
    Eigen::SparseMatrix<double> M_; // Q^T W^{-1} Q
    Eigen::VectorXd weights_inv_;
    Eigen::VectorXd x_;
    Eigen::VectorXd gamma_;
    Eigen::VectorXd f_;
    long n_;
    
public:
    SplineFitterReinschCpp(const Eigen::Ref<const Eigen::VectorXd>& x,
                           py::object weights_obj) {
        x_ = x;
        n_ = x.size();
        
        // Build Q and R
        long n_inner = n_ - 2;
        Eigen::VectorXd h = x.segment(1, n_ - 1) - x.segment(0, n_ - 1);
        Eigen::VectorXd inv_h = h.cwiseInverse();
        
        R_.resize(n_inner, n_inner);
        Q_.resize(n_, n_inner); 
        
        std::vector<Eigen::Triplet<double>> r_triplets;
        std::vector<Eigen::Triplet<double>> q_triplets;
        
        // R
        for (int i = 0; i < n_inner; ++i) {
            r_triplets.push_back({i, i, (h[i] + h[i+1]) / 3.0});
            if (i < n_inner - 1) {
                r_triplets.push_back({i, i+1, h[i+1] / 6.0});
                r_triplets.push_back({i+1, i, h[i+1] / 6.0});
            }
        }
        R_.setFromTriplets(r_triplets.begin(), r_triplets.end());
        
        // Q
        for (int j = 0; j < n_inner; ++j) {
            q_triplets.push_back({j, j, inv_h[j]});
            q_triplets.push_back({j+1, j, -inv_h[j] - inv_h[j+1]});
            q_triplets.push_back({j+2, j, inv_h[j+1]});
        }
        Q_.setFromTriplets(q_triplets.begin(), q_triplets.end());
        
        update_weights(weights_obj);
    }
    
    void update_weights(py::object weights_obj) {
        weights_inv_.resize(n_);
        if (weights_obj.is_none()) {
            weights_inv_.setOnes();
        } else {
            Eigen::VectorXd w = weights_obj.cast<Eigen::VectorXd>();
            weights_inv_ = w.cwiseInverse();
        }
        
        // M = Q^T W^{-1} Q
        // Re-construct Winv sparse matrix
        Eigen::SparseMatrix<double> Winv(n_, n_);
        std::vector<Eigen::Triplet<double>> w_triplets;
        for(int i=0; i<n_; ++i) w_triplets.push_back({i, i, weights_inv_[i]});
        Winv.setFromTriplets(w_triplets.begin(), w_triplets.end());
        M_ = Q_.transpose() * Winv * Q_;
    }
    
    Eigen::VectorXd fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
        Eigen::VectorXd QT_y = Q_.transpose() * y;
        Eigen::SparseMatrix<double> LHS = R_ + lamval * M_;
        
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(LHS);
        if(solver.info() != Eigen::Success) throw std::runtime_error("Reinsch solver failed");
        
        gamma_ = solver.solve(QT_y);
        
        // f = y - lam * W^{-1} * Q * gamma
        Eigen::VectorXd Q_gamma = Q_ * gamma_;
        Eigen::VectorXd term = weights_inv_.cwiseProduct(Q_gamma);
        f_ = y - lamval * term;
        return f_;
    }
    
    double compute_df_sparse(double lamval) {
        Eigen::SparseMatrix<double> LHS = R_ + lamval * M_;
        Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(LHS);
        
        double trace_val = 0.0;
        long n_inner = M_.rows();
        
        if (n_inner > 1000) {
             return compute_df_hutchinson(solver, lamval * M_, n_inner);
        }
        
        Eigen::VectorXd rhs;
        Eigen::VectorXd sol;
        
        for (int j = 0; j < n_inner; ++j) {
            rhs = lamval * M_.col(j); 
            sol = solver.solve(rhs);
            trace_val += sol[j];
        }
        
        long n = weights_inv_.size(); 
        return n - trace_val;
    }

    double compute_df(double lamval) {
        CubicSplineTraceCpp trace_solver(x_);
        return trace_solver.compute_trace(lamval);
    }
    
    double compute_df_hutchinson(Eigen::SimplicialLLT<Eigen::SparseMatrix<double>>& solver, 
                                 const Eigen::SparseMatrix<double>& B, 
                                 long n_dim) {
        int n_samples = 30; 
        double trace_est = 0.0;
        
        Eigen::VectorXd z(n_dim);
        std::srand(12345); 
        
        for(int i=0; i<n_samples; ++i) {
            for(int k=0; k<n_dim; ++k) z[k] = (std::rand() % 2) ? 1.0 : -1.0;
            
            Eigen::VectorXd v = B * z;
            Eigen::VectorXd u = solver.solve(v);
            trace_est += z.dot(u);
        }
        
        long n = weights_inv_.size();
        return n - (trace_est / n_samples);
    }

    double gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
        Eigen::VectorXd f = fit(y, lamval);
        Eigen::VectorXd resid = y - f;
        Eigen::VectorXd w = weights_inv_.cwiseInverse();
        double rss = (resid.array().square() * w.array()).sum();
        
        double df = compute_df_sparse(lamval);
        double n = (double)y.size();
        double denom = 1.0 - df / n;
        if (denom < 1e-6) return 1e20;
        
        return (rss / n) / (denom * denom);
    }

    double solve_for_df(double target_df) {
        auto func = [&](double log_lam) {
            double lam = std::pow(10.0, log_lam);
            return compute_df_sparse(lam) - target_df;
        };
        double log_lam_opt = brent_root(func, -12.0, 12.0);
        return std::pow(10.0, log_lam_opt);
    }

    double solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam = -12.0, double max_log_lam = 12.0) {
        auto func = [&](double log_lam) {
            double lam = std::pow(10.0, log_lam);
            return gcv_score(lam, y);
        };
        double log_lam_opt = brent_min(func, min_log_lam, max_log_lam);
        return std::pow(10.0, log_lam_opt);
    }

    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv=0) {
        if (f_.size() == 0) throw std::runtime_error("Model not fitted");
        
        Eigen::VectorXd M(n_);
        M[0] = 0.0;
        M[n_-1] = 0.0;
        M.segment(1, n_-2) = gamma_;
        
        long n_new = x_new.size();
        Eigen::VectorXd y_pred(n_new);
        
        for(long i=0; i<n_new; ++i) {
            double val = x_new[i];
            long k = 0;
            if (val < x_[0]) {
                k = 0;
            } else if (val >= x_[n_-1]) {
                k = n_ - 2;
            } else {
                auto it = std::upper_bound(x_.data(), x_.data() + n_, val);
                k = std::distance(x_.data(), it) - 1;
                if (k < 0) k = 0;
                if (k >= n_ - 1) k = n_ - 2;
            }
            
            if (val < x_[0]) {
                double h = x_[1] - x_[0];
                double first_deriv = (f_[1] - f_[0]) / h - h * M[1] / 6.0;
                if (deriv == 0) {
                    y_pred[i] = f_[0] + first_deriv * (val - x_[0]);
                } else if (deriv == 1) {
                    y_pred[i] = first_deriv;
                } else {
                    y_pred[i] = 0;
                }
            } else if (val > x_[n_-1]) {
                long last_k = n_ - 2;
                double h = x_[last_k+1] - x_[last_k];
                double first_deriv = (f_[last_k+1] - f_[last_k]) / h + M[last_k] * h / 6.0 + M[last_k+1] * h / 3.0;
                if (deriv == 0) {
                    y_pred[i] = f_[n_-1] + first_deriv * (val - x_[n_-1]);
                } else if (deriv == 1) {
                    y_pred[i] = first_deriv;
                } else {
                    y_pred[i] = 0;
                }
            } else {
                double h = x_[k+1] - x_[k];
                if (deriv == 0) {
                    double term1 = (std::pow(x_[k+1] - val, 3) * M[k] + std::pow(val - x_[k], 3) * M[k+1]) / (6.0 * h);
                    double term2 = (f_[k] - h*h * M[k] / 6.0) * (x_[k+1] - val) / h;
                    double term3 = (f_[k+1] - h*h * M[k+1] / 6.0) * (val - x_[k]) / h;
                    y_pred[i] = term1 + term2 + term3;
                } else if (deriv == 1) {
                    double d_term1 = (-3.0 * std::pow(x_[k+1] - val, 2) * M[k] + 3.0 * std::pow(val - x_[k], 2) * M[k+1]) / (6.0 * h);
                    double d_term2 = -(f_[k] - h*h * M[k] / 6.0) / h;
                    double d_term3 = (f_[k+1] - h*h * M[k+1] / 6.0) / h;
                    y_pred[i] = d_term1 + d_term2 + d_term3;
                } else if (deriv == 2) {
                    double d2_term1 = (6.0 * (x_[k+1] - val) * M[k] + 6.0 * (val - x_[k]) * M[k+1]) / (6.0 * h);
                    y_pred[i] = d2_term1;
                } else {
                    y_pred[i] = 0;
                }
            }
        }
        return y_pred;
    }
};

PYBIND11_MODULE(_spline_extension, m) {
    m.doc() = "C++ implementation of SplineFitter core components"; 
    
    m.def("compute_natural_spline_basis", &compute_natural_spline_basis, 
          "Compute the Natural Cubic Spline Basis matrix",
          py::arg("x"), py::arg("knots"), py::arg("extrapolate_linear") = true,
          py::arg("derivative_order") = 0);
          
    m.def("compute_penalty_matrix", &compute_penalty_matrix,
          "Compute the penalty matrix Omega",
          py::arg("knots"));
          
    py::class_<SplineFitterCpp>(m, "SplineFitterCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, 
                      const Eigen::Ref<const Eigen::VectorXd>&,
                      py::object>(),
             py::arg("x"), py::arg("knots"), py::arg("weights") = py::none())
        .def("fit", &SplineFitterCpp::fit, "Solve for spline coefficients",
             py::arg("y"), py::arg("lamval"))
        .def("update_weights", &SplineFitterCpp::update_weights, "Update weights without recomputing basis",
             py::arg("weights"))
        .def("compute_df", &SplineFitterCpp::compute_df, "Compute degrees of freedom for lambda",
             py::arg("lamval"))
        .def("gcv_score", &SplineFitterCpp::gcv_score, "Compute GCV score",
             py::arg("lamval"), py::arg("y"))
        .def("solve_for_df", &SplineFitterCpp::solve_for_df, "Find lambda for target DF",
             py::arg("target_df"))
        .def("solve_gcv", &SplineFitterCpp::solve_gcv, "Solve for GCV optimal lambda",
             py::arg("y"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("predict", &SplineFitterCpp::predict, "Predict at new points",
             py::arg("x_new"), py::arg("deriv")=0)
        .def("get_N", &SplineFitterCpp::get_N)
        .def("get_Omega", &SplineFitterCpp::get_Omega);

    py::class_<SplineFitterReinschCpp>(m, "SplineFitterReinschCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&,
                      py::object>(),
             py::arg("x"), py::arg("weights") = py::none())
        .def("fit", &SplineFitterReinschCpp::fit, "Fit using Reinsch algorithm",
             py::arg("y"), py::arg("lamval"))
        .def("update_weights", &SplineFitterReinschCpp::update_weights, "Update weights",
             py::arg("weights"))
        .def("compute_df", &SplineFitterReinschCpp::compute_df, "Compute DF",
             py::arg("lamval"))
        .def("compute_df_sparse", &SplineFitterReinschCpp::compute_df_sparse, "Compute DF using sparse solve",
             py::arg("lamval"))
        .def("gcv_score", &SplineFitterReinschCpp::gcv_score, "Compute GCV score",
             py::arg("lamval"), py::arg("y"))
        .def("solve_for_df", &SplineFitterReinschCpp::solve_for_df, "Find lambda for target DF",
             py::arg("target_df"))
        .def("solve_gcv", &SplineFitterReinschCpp::solve_gcv, "Solve for GCV optimal lambda",
             py::arg("y"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("predict", &SplineFitterReinschCpp::predict, "Predict at new points",
             py::arg("x_new"), py::arg("deriv")=0);

    py::class_<CubicSplineTraceCpp>(m, "CubicSplineTraceCpp")
        .def(py::init<const Eigen::VectorXd&>(), py::arg("x"))
        .def("compute_trace", &CubicSplineTraceCpp::compute_trace, "Compute trace for a given lambda",
             py::arg("lam"));
}
