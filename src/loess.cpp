#include "loess.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

LoessSmootherCpp::LoessSmootherCpp(const Eigen::Ref<const Eigen::VectorXd>& x, 
                                   py::object weights_obj, 
                                   double span, 
                                   int degree) 
    : x_(x), span_(span), degree_(degree) {
    n_ = x_.size();
    if (weights_obj.is_none()) {
        w_ = Eigen::VectorXd::Ones(n_);
    } else {
        w_ = weights_obj.cast<Eigen::VectorXd>();
    }
}

void LoessSmootherCpp::fit(const Eigen::Ref<const Eigen::VectorXd>& y) {
    y_ = y;
}

void LoessSmootherCpp::update_weights(py::object weights_obj) {
    if (weights_obj.is_none()) {
        w_ = Eigen::VectorXd::Ones(n_);
    } else {
        w_ = weights_obj.cast<Eigen::VectorXd>();
    }
}

long factorial(int n) {
    long res = 1;
    for (int i = 2; i <= n; ++i) res *= i;
    return res;
}

Eigen::VectorXd LoessSmootherCpp::predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv) {
    int n_new = x_new.size();
    Eigen::VectorXd y_pred(n_new);

    int k = static_cast<int>(std::ceil(span_ * n_));
    if (k < degree_ + 1) k = degree_ + 1;
    if (k > n_) k = n_;

    std::vector<Neighbor> neighbors(n_);
    long fact_deriv = factorial(deriv);

    for (int i = 0; i < n_new; ++i) {
        double val = x_new(i);

        // 1. Compute distances
        for (int j = 0; j < n_; ++j) {
            neighbors[j] = {j, std::abs(x_(j) - val)};
        }

        // 2. Find k nearest neighbors (partial sort)
        std::nth_element(neighbors.begin(), neighbors.begin() + k - 1, neighbors.end(),
                         [](const Neighbor& a, const Neighbor& b) {
                             return a.dist < b.dist;
                         });

        double max_dist = 0;
        for (int j = 0; j < k; ++j) {
            if (neighbors[j].dist > max_dist) max_dist = neighbors[j].dist;
        }

        // 3. Optimized Local Weighted Least Squares for common degrees (0, 1, 2)
        // Solves the Normal Equations: (X^T W X) beta = X^T W y
        // Avoids heap allocation for X and generic QR decomposition
        
        bool solved = false;

        if (degree_ == 0) {
            double sum_w = 0, sum_wy = 0;
            for (int j = 0; j < k; ++j) {
                int idx = neighbors[j].index;
                double u = (max_dist > 1e-14) ? neighbors[j].dist / max_dist : 0.0;
                double tri_w = std::pow(std::max(0.0, 1.0 - std::pow(u, 3)), 3);
                double w_total = tri_w * w_(idx);
                sum_w += w_total;
                sum_wy += w_total * y_(idx);
            }
            if (sum_w < 1e-14) y_pred(i) = std::numeric_limits<double>::quiet_NaN();
            else y_pred(i) = (deriv == 0) ? sum_wy / sum_w : 0.0;
            solved = true;
        } 
        else if (degree_ == 1) {
            double s0 = 0, s1 = 0, s2 = 0;
            double t0 = 0, t1 = 0;
            for (int j = 0; j < k; ++j) {
                int idx = neighbors[j].index;
                double u = (max_dist > 1e-14) ? neighbors[j].dist / max_dist : 0.0;
                double tri_w = std::pow(std::max(0.0, 1.0 - std::pow(u, 3)), 3);
                double w = tri_w * w_(idx);
                
                double diff = x_(idx) - val;
                s0 += w;
                s1 += w * diff;
                s2 += w * diff * diff;
                t0 += w * y_(idx);
                t1 += w * diff * y_(idx);
            }
            
            // Determinant of 2x2 matrix [[s0, s1], [s1, s2]]
            double det = s0 * s2 - s1 * s1;
            
            if (std::abs(det) < 1e-14 || s0 < 1e-14) {
                 y_pred(i) = std::numeric_limits<double>::quiet_NaN();
            } else {
                 if (deriv == 0) {
                     // beta0
                     y_pred(i) = (s2 * t0 - s1 * t1) / det;
                 } else if (deriv == 1) {
                     // beta1 * 1!
                     y_pred(i) = (s0 * t1 - s1 * t0) / det;
                 } else {
                     y_pred(i) = 0.0;
                 }
            }
            solved = true;
        }
        else if (degree_ == 2) {
             // 3x3 System
             Eigen::Matrix3d A; A.setZero();
             Eigen::Vector3d b; b.setZero();
             
             for (int j = 0; j < k; ++j) {
                 int idx = neighbors[j].index;
                 double u = (max_dist > 1e-14) ? neighbors[j].dist / max_dist : 0.0;
                 double tri_w = std::pow(std::max(0.0, 1.0 - std::pow(u, 3)), 3);
                 double w = tri_w * w_(idx);

                 double diff = x_(idx) - val;
                 double diff2 = diff * diff;
                 
                 // Symmetric A accumulation
                 A(0,0) += w; 
                 double wd = w * diff;
                 A(0,1) += wd; 
                 double wd2 = w * diff2;
                 A(0,2) += wd2; // A(1,1) is also sum(w d^2)
                 
                 double wd3 = wd2 * diff;
                 A(1,2) += wd3; 
                 
                 double wd4 = wd3 * diff;
                 A(2,2) += wd4;
                 
                 double wy = w * y_(idx);
                 b(0) += wy;
                 b(1) += wy * diff;
                 b(2) += wy * diff2;
             }
             // Fill symmetric parts
             A(1,0) = A(0,1);
             A(1,1) = A(0,2);
             A(2,0) = A(0,2);
             A(2,1) = A(1,2);
             
             // Solve A x = b using LDLT
             Eigen::LDLT<Eigen::Matrix3d> solver;
             solver.compute(A);
             if (solver.info() != Eigen::Success) {
                 y_pred(i) = std::numeric_limits<double>::quiet_NaN();
             } else {
                 Eigen::Vector3d beta = solver.solve(b);
                 if (deriv == 0) y_pred(i) = beta(0);
                 else if (deriv == 1) y_pred(i) = beta(1);
                 else if (deriv == 2) y_pred(i) = beta(2) * 2.0; // 2!
                 else y_pred(i) = 0.0;
             }
             solved = true;
        }

        if (!solved) {
            // General Fallback for degree >= 3 using QR decomposition
            Eigen::VectorXd local_w(k);
            Eigen::MatrixXd X(k, degree_ + 1);
            Eigen::VectorXd local_y(k);

            for (int j = 0; j < k; ++j) {
                int idx = neighbors[j].index;
                double u = (max_dist > 1e-14) ? neighbors[j].dist / max_dist : 0.0;
                double tri_w = std::pow(std::max(0.0, 1.0 - std::pow(u, 3)), 3);
                double total_w = tri_w * w_(idx);
                
                double sqrt_w = std::sqrt(total_w);
                local_w(j) = total_w;

                double diff = x_(idx) - val;
                double p = 1.0;
                for (int d = 0; d <= degree_; ++d) {
                    X(j, d) = p * sqrt_w;
                    p *= diff;
                }
                local_y(j) = y_(idx) * sqrt_w;
            }
            
            if (local_w.sum() < 1e-14) {
                 y_pred(i) = std::numeric_limits<double>::quiet_NaN();
                 continue;
            }

            Eigen::VectorXd beta = X.colPivHouseholderQr().solve(local_y);
            if (deriv <= degree_) {
                y_pred(i) = beta(deriv) * fact_deriv;
            } else {
                y_pred(i) = 0.0;
            }
        }
    }

    return y_pred;
}
