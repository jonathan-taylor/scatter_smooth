#ifndef LOESS_H
#define LOESS_H

#include <Eigen/Dense>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class LoessSmootherCpp {
public:
    LoessSmootherCpp(const Eigen::Ref<const Eigen::VectorXd>& x, 
                     py::object weights_obj, 
                     double span, 
                     int degree);

    void set_y(const Eigen::Ref<const Eigen::VectorXd>& y);
    void update_weights(py::object weights_obj);
    
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv = 0);
    Eigen::VectorXd leverage();

private:
    Eigen::VectorXd x_;
    // x sorted once, with the permutation: neighbourhoods are windows of it
    Eigen::VectorXd xs_;
    std::vector<int> order_;
    Eigen::VectorXd y_;
    Eigen::VectorXd w_;
    double span_;
    int degree_;
    int n_;

    struct Neighbor {
        int index;
        double dist;
    };
    int neighbourhood_size() const;
    // the local fit at val: false if the neighbourhood has no weight
    bool local_fit(double val, int k, std::vector<Neighbor>& neighbors,
                   Eigen::VectorXd& beta, double& scale, double& inv00) const;
};

#endif // LOESS_H
