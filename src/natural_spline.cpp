#include "natural_spline.h"
#include "utils.h"
#include <iostream>

/**
 * Computes the Natural Cubic Spline basis matrix N.
 * 
 * The natural cubic spline basis functions N_j(x) are defined such that any natural spline f(x)
 * with knots at 'knots' can be written as f(x) = sum_j alpha_j * N_j(x).
 * 
 * We use the standard construction where the second derivatives M at the knots satisfy a 
 * tridiagonal system (A * M = B * alpha). This function effectively inverts this relationship 
 * to express the spline values directly in terms of the coefficients alpha.
 * 
 * If extrapolate_linear is true, the spline is extended linearly beyond the boundary knots,
 * consistent with the definition of a natural spline (zero second derivative implies linear).
 */
Eigen::MatrixXd compute_natural_spline_basis(
    const Eigen::Ref<const Eigen::VectorXd>& x,
    const Eigen::Ref<const Eigen::VectorXd>& knots,
    bool extrapolate_linear,
    int derivative_order
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

/**
 * Computes the roughness penalty matrix Omega for natural cubic splines.
 * 
 * Omega_ij = Integral phi''_i(x) * phi''_j(x) dx
 * 
 * The penalty term in the objective function is lambda * alpha^T * Omega * alpha.
 * This function essentially computes the matrix K in the standard notation (or close to it)
 * by solving R^T Omega R = Q^T Q (conceptually) or using the property that 
 * for natural splines, the integral of squared second derivatives reduces to a form 
 * involving the tridiagonal matrix R and the band matrix Q.
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

NaturalSplineFitter::NaturalSplineFitter(const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& knots, py::object weights_obj) {
    knots_ = knots;
    N_ = compute_natural_spline_basis(x, knots, true, 0);
    Omega_ = compute_penalty_matrix(knots);
    update_weights(weights_obj);
}

void NaturalSplineFitter::update_weights(py::object weights_obj) {
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

Eigen::VectorXd NaturalSplineFitter::fit(const Eigen::Ref<const Eigen::VectorXd>& y, double lamval) {
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

Eigen::VectorXd NaturalSplineFitter::predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv) {
    Eigen::MatrixXd N_new = compute_natural_spline_basis(x_new, knots_, true, deriv);
    return N_new * alpha_;
}

double NaturalSplineFitter::compute_df(double lamval) {
    Eigen::MatrixXd LHS = NTWN_ + lamval * Omega_;
    Eigen::LLT<Eigen::MatrixXd> solver;
    solver.compute(LHS);
    if (solver.info() != Eigen::Success) return 0.0;
    Eigen::MatrixXd X = solver.solve(NTWN_);
    return X.trace();
}

double NaturalSplineFitter::gcv_score(double lamval, const Eigen::Ref<const Eigen::VectorXd>& y) {
    Eigen::VectorXd alpha = fit(y, lamval);
    Eigen::VectorXd f = N_ * alpha;
    double rss = (y - f).squaredNorm(); 
    double df = compute_df(lamval);
    double n = (double)y.size();
    double denom = 1.0 - df / n;
    if (denom < 1e-6) return 1e20;
    return (rss / n) / (denom * denom);
}

double NaturalSplineFitter::solve_for_df(double target_df, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return compute_df(std::pow(10.0, log_lam)) - target_df; };
    return std::pow(10.0, utils::brent_root(func, min_log_lam, max_log_lam));
}

double NaturalSplineFitter::solve_gcv(const Eigen::Ref<const Eigen::VectorXd>& y, double min_log_lam, double max_log_lam) {
    auto func = [&](double log_lam) { return gcv_score(std::pow(10.0, log_lam), y); };
    return std::pow(10.0, utils::brent_min(func, min_log_lam, max_log_lam));
}

Eigen::MatrixXd NaturalSplineFitter::get_N() { return N_; }
Eigen::MatrixXd NaturalSplineFitter::get_Omega() { return Omega_; }
