#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

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
    
public:
    SplineFitterCpp(const Eigen::Ref<const Eigen::VectorXd>& x, 
                    const Eigen::Ref<const Eigen::VectorXd>& knots,
                    py::object weights_obj) {
        
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
             return solver_ldlt.solve(RHS);
        }
        
        return solver.solve(RHS);
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

    Eigen::MatrixXd get_N() { return N_; }
    Eigen::MatrixXd get_Omega() { return Omega_; }
};

/**
 * Reinsch Form Fitter (O(N))
 */
class SplineFitterReinschCpp {
    Eigen::SparseMatrix<double> Q_;
    Eigen::SparseMatrix<double> R_;
    Eigen::SparseMatrix<double> M_; // Q^T W^{-1} Q
    Eigen::VectorXd weights_inv_;
    long n_;
    
public:
    SplineFitterReinschCpp(const Eigen::Ref<const Eigen::VectorXd>& x,
                           py::object weights_obj) {
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
        
        Eigen::VectorXd gamma = solver.solve(QT_y);
        
        // f = y - lam * W^{-1} * Q * gamma
        Eigen::VectorXd Q_gamma = Q_ * gamma;
        Eigen::VectorXd term = weights_inv_.cwiseProduct(Q_gamma);
        return y - lamval * term;
    }
    
    double compute_df(double lamval) {
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
        
        double df = compute_df(lamval);
        double n = (double)y.size();
        double denom = 1.0 - df / n;
        if (denom < 1e-6) return 1e20;
        
        return (rss / n) / (denom * denom);
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
        .def("gcv_score", &SplineFitterReinschCpp::gcv_score, "Compute GCV score",
             py::arg("lamval"), py::arg("y"));
}
