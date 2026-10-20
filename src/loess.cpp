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
    order_.resize(n_);
    for (int i = 0; i < n_; ++i) order_[i] = i;
    std::sort(order_.begin(), order_.end(), [this](int a, int b) { return x_(a) < x_(b); });
    xs_.resize(n_);
    for (int i = 0; i < n_; ++i) xs_(i) = x_(order_[i]);
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
int LoessSmootherCpp::neighbourhood_size() const {
    int k = static_cast<int>(std::ceil(std::min(span_, 1.0) * n_));
    if (k < degree_ + 1) k = degree_ + 1;
    if (k > n_) k = n_;
    return k;
}

bool LoessSmootherCpp::local_fit(double val, int k, std::vector<Neighbor>& neighbors,
                                 Eigen::VectorXd& beta, double& scale, double& inv00) const {
    const int p = degree_ + 1;
    double widen = (span_ > 1.0) ? std::sqrt(span_) : 1.0;

    // the k nearest: a window [lo, hi) of the sorted x, grown from val's
    // position towards the nearer side, O(log n + k)
    int hi = static_cast<int>(std::lower_bound(xs_.data(), xs_.data() + n_, val) - xs_.data());
    int lo = hi;
    double max_dist = 0;
    for (int m = 0; m < k; ++m) {
        bool left = (lo > 0) && (hi >= n_ || val - xs_(lo - 1) <= xs_(hi) - val);
        double d = left ? val - xs_(--lo) : xs_(hi++) - val;
        if (d > max_dist) max_dist = d;
    }
    // the neighbourhood holds every point tied at the boundary distance,
    // so that it doesn't depend on the order of the data
    while (lo > 0 && val - xs_(lo - 1) <= max_dist) --lo;
    while (hi < n_ && xs_(hi) - val <= max_dist) ++hi;
    neighbors.clear();
    for (int j = lo; j < hi; ++j) neighbors.push_back({order_[j], std::abs(xs_(j) - val)});
    int k_i = hi - lo;
    double h = max_dist * widen;
    // all the neighbours at val: they all get weight 1, and the scale is moot
    scale = (h > 0) ? h : 1.0;

    // pass 0: tricube weights. If they are all 0 (every neighbour, tied, at
    // the boundary distance h), pass 1: the limit of a bandwidth just above
    // h, in which the neighbours at h get equal weights.
    // The moments sum(w z^m), m < 2p - 1, and sum(w y z^m), m < p, give the
    // normal equations A(r, c) = M(r + c), b(r) = T(r).
    double M[7], T[4];
    for (int pass = 0; pass < 2; ++pass) {
        for (int m = 0; m < 2 * p - 1; ++m) M[m] = 0.0;
        for (int m = 0; m < p; ++m) T[m] = 0.0;
        for (int j = 0; j < k_i; ++j) {
            int idx = neighbors[j].index;
            double tri_w;
            if (pass == 0) {
                double u = (h > 0) ? neighbors[j].dist / h : 0.0;
                double t = 1.0 - u * u * u;
                tri_w = (u < 1.0) ? t * t * t : 0.0;
            } else {
                tri_w = (neighbors[j].dist >= max_dist) ? 1.0 : 0.0;
            }
            double w = tri_w * w_(idx);
            if (w <= 0) continue;
            double z = (x_(idx) - val) / scale, y = y_.size() == n_ ? y_(idx) : 0.0, pz = 1.0;
            for (int m = 0; m < 2 * p - 1; ++m) {
                M[m] += w * pz;
                if (m < p) T[m] += w * y * pz;
                pz *= z;
            }
        }
        if (M[0] > 0 || widen > 1.0) break;
    }
    if (M[0] <= 0) return false;      // no weight at all in the neighbourhood

    Eigen::MatrixXd A(p, p);
    Eigen::VectorXd b(p);
    for (int r = 0; r < p; ++r) {
        b(r) = T[r];
        for (int c = 0; c < p; ++c) A(r, c) = M[r + c];
    }
    // unit column norms (of the weighted design); with fewer distinct x
    // values in the neighbourhood than the degree needs, the local
    // polynomial drops its highest powers (to the rank of the system): e.g.
    // the weighted mean when they are all at one x value
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
    auto ldlt = An.topLeftCorner(rank, rank).ldlt();
    Eigen::VectorXd coef = Eigen::VectorXd::Zero(p);
    coef.head(rank) = ldlt.solve(bn.head(rank));
    // beta(d) is the coefficient of ((x - val) / scale)^d
    beta = coef.cwiseQuotient(cn);
    // (A^{-1})_{00} of the (rank-reduced) system: the weight of an
    // observation at val is its weight times this
    Eigen::VectorXd e0 = Eigen::VectorXd::Zero(rank);
    e0(0) = 1.0;
    inv00 = ldlt.solve(e0)(0) / (cn(0) * cn(0));
    return true;
}

Eigen::VectorXd LoessSmootherCpp::predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv) {
    if (y_.size() != n_) throw std::runtime_error("predict: the smoother is not fitted yet (call smooth first)");
    if (deriv < 0 || deriv > degree_)
        throw std::invalid_argument("deriv should be between 0 and the degree of the local polynomial");
    int n_new = x_new.size(), k = neighbourhood_size();
    Eigen::VectorXd y_pred(n_new), beta;
    std::vector<Neighbor> neighbors;
    neighbors.reserve(n_);
    long fact_deriv = factorial(deriv);
    double scale, inv00;
    for (int i = 0; i < n_new; ++i) {
        if (!local_fit(x_new(i), k, neighbors, beta, scale, inv00))
            y_pred(i) = std::numeric_limits<double>::quiet_NaN();
        else
            y_pred(i) = beta(deriv) * fact_deriv / std::pow(scale, deriv);
    }
    return y_pred;
}

// the diagonal of the smoother matrix: the weight of y_i in the fit at x_i
// (its own tricube weight is 1)
Eigen::VectorXd LoessSmootherCpp::leverage() {
    int k = neighbourhood_size();
    Eigen::VectorXd lev(n_), beta;
    std::vector<Neighbor> neighbors;
    neighbors.reserve(n_);
    double scale, inv00;
    for (int i = 0; i < n_; ++i) {
        lev(i) = (w_(i) > 0 && local_fit(x_(i), k, neighbors, beta, scale, inv00)) ? w_(i) * inv00 : 0.0;
    }
    return lev;
}
