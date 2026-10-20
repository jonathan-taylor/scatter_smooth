#include "loess.h"
#include "utils.h"
#include <stdexcept>
#include <limits>
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
    if (n_ < 1) throw std::invalid_argument("x is empty");
    if (!(span_ > 0) || !std::isfinite(span_)) throw std::invalid_argument("span should be positive and finite");
    if (degree_ < 0 || degree_ > 3) throw std::invalid_argument("degree should be 0, 1, 2 or 3");
    update_weights(weights_obj);
}

void LoessSmootherCpp::set_y(const Eigen::Ref<const Eigen::VectorXd>& y) {
    utils::check_size(y.size(), n_, "y");
    y_ = y;
}

void LoessSmootherCpp::update_weights(py::object weights_obj) {
    if (weights_obj.is_none()) {
        w_ = Eigen::VectorXd::Ones(n_);
    } else {
        Eigen::VectorXd w = weights_obj.cast<Eigen::VectorXd>();
        utils::check_size(w.size(), n_, "weights");
        w_ = w;
    }
}

namespace {
    long factorial(int n) {
        long res = 1;
        for (int i = 2; i <= n; ++i) res *= i;
        return res;
    }
}

/**
 * The local fit at each point: weighted least squares of y on the powers
 * (x - val) / h, 0..degree, with tricube weights in |x - val| / h, h the
 * distance to the k-th nearest neighbour (k = ceil(span * n); points tied at
 * that distance are all in the neighbourhood); for span > 1
 * all points, with h sqrt(span) times the largest distance (R's rule).
 *
 * The normal equations are solved with columns scaled to unit norm, by a
 * pseudo-inverse when they are (nearly) singular: with tied x, the
 * neighbourhood can hold fewer distinct values than the degree needs (e.g.
 * all the points at the boundary distance, of weight 0), and the fit is then
 * that of the lower-degree terms, as R's loess.
 */
Eigen::VectorXd LoessSmootherCpp::predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv) {
    if (y_.size() != n_) throw std::runtime_error("predict: the smoother is not fitted yet (call smooth first)");
    if (deriv < 0 || deriv > degree_)
        throw std::invalid_argument("deriv should be between 0 and the degree of the local polynomial");
    int n_new = x_new.size();
    Eigen::VectorXd y_pred(n_new);

    int k = static_cast<int>(std::ceil(std::min(span_, 1.0) * n_));
    if (k < degree_ + 1) k = degree_ + 1;
    if (k > n_) k = n_;
    double widen = (span_ > 1.0) ? std::sqrt(span_) : 1.0;

    const int p = degree_ + 1;
    std::vector<Neighbor> neighbors(n_);
    long fact_deriv = factorial(deriv);
    Eigen::MatrixXd A(p, p);
    Eigen::VectorXd b(p), powers(p);

    for (int i = 0; i < n_new; ++i) {
        double val = x_new(i);

        for (int j = 0; j < n_; ++j) {
            neighbors[j] = {j, std::abs(x_(j) - val)};
        }
        std::nth_element(neighbors.begin(), neighbors.begin() + k - 1, neighbors.end(),
                         [](const Neighbor& a, const Neighbor& b) {
                             return a.dist < b.dist;
                         });
        double max_dist = 0;
        for (int j = 0; j < k; ++j) {
            if (neighbors[j].dist > max_dist) max_dist = neighbors[j].dist;
        }
        // the neighbourhood holds every point tied at the boundary distance,
        // so that it doesn't depend on the order of the data
        int k_i = static_cast<int>(std::partition(neighbors.begin() + k, neighbors.end(),
                                                  [max_dist](const Neighbor& a) {
                                                      return a.dist <= max_dist;
                                                  }) - neighbors.begin());
        double h = max_dist * widen;
        // all the neighbours at val: they all get weight 1, and the scale is moot
        double scale = (h > 0) ? h : 1.0;

        // pass 0: tricube weights. If they are all 0 (every neighbour, tied, at
        // the boundary distance h), pass 1: the limit of a bandwidth just
        // above h, in which the neighbours at h get equal weights.
        for (int pass = 0; pass < 2; ++pass) {
            A.setZero(); b.setZero();
            for (int j = 0; j < k_i; ++j) {
                int idx = neighbors[j].index;
                double u = (h > 0) ? neighbors[j].dist / h : 0.0;
                double tri_w = (u < 1.0) ? std::pow(1.0 - u * u * u, 3) : 0.0;
                if (pass == 1) tri_w = (neighbors[j].dist >= max_dist) ? 1.0 : 0.0;
                double w = tri_w * w_(idx);
                if (w <= 0) continue;
                double z = (x_(idx) - val) / scale, pw = 1.0;
                for (int d = 0; d < p; ++d) { powers(d) = pw; pw *= z; }
                A.noalias() += w * powers * powers.transpose();
                b.noalias() += (w * y_(idx)) * powers;
            }
            if (A(0, 0) > 0 || widen > 1.0) break;
        }
        if (A(0, 0) <= 0) {
            // no weight at all in the neighbourhood
            y_pred(i) = std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        // unit column norms (of the weighted design); with fewer distinct x
        // values in the neighbourhood than the degree needs, the local
        // polynomial drops its highest powers (to the rank of the system):
        // e.g. the weighted mean when they are all at one x value
        Eigen::VectorXd cn = A.diagonal().cwiseSqrt();
        for (int d = 0; d < p; ++d) if (cn(d) <= 0) cn(d) = 1.0;
        Eigen::MatrixXd An = cn.cwiseInverse().asDiagonal() * A * cn.cwiseInverse().asDiagonal();
        Eigen::VectorXd bn = b.cwiseQuotient(cn);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(An, Eigen::EigenvaluesOnly);
        Eigen::VectorXd ev = eig.eigenvalues();
        double tol = ev.maxCoeff() * 1e-12;
        int rank = 0;
        for (int d = 0; d < p; ++d) if (ev(d) > tol) ++rank;
        if (rank < 1) rank = 1;
        Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
        coef.head(rank) = An.topLeftCorner(rank, rank).ldlt().solve(bn.head(rank));
        Eigen::VectorXd beta = coef.cwiseQuotient(cn);
        // beta(d) is the coefficient of ((x - val) / scale)^d
        y_pred(i) = beta(deriv) * fact_deriv / std::pow(scale, deriv);
    }

    return y_pred;
}
