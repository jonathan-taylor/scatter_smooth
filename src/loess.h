#ifndef LOESS_H
#define LOESS_H

#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class LoessSmootherCpp {
public:
    LoessSmootherCpp(const Eigen::Ref<const Eigen::VectorXd>& x, 
                     py::object weights_obj, 
                     double span, 
                     int degree);

    void fit(const Eigen::Ref<const Eigen::VectorXd>& y);
    void update_weights(py::object weights_obj);
    
    Eigen::VectorXd predict(const Eigen::Ref<const Eigen::VectorXd>& x_new, int deriv = 0);

private:
    Eigen::VectorXd x_;
    Eigen::VectorXd y_;
    Eigen::VectorXd w_;
    double span_;
    int degree_;
    int n_;

    struct Neighbor {
        int index;
        double dist;
    };
};

#endif // LOESS_H
