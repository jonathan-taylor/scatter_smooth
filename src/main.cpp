#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "natural_spline.h"
#include "reinsch.h"
#include "bspline.h"
#include "loess.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_scatter_smooth_extension, m) {
    m.def("trace_takahashi", &utils::trace_takahashi, py::arg("U_banded"), py::arg("B_banded"));
    m.def("takahashi_upper", &utils::takahashi_upper, py::arg("U_banded"));
    m.def("trace_product_banded", &utils::trace_product_banded, py::arg("Z_banded"), py::arg("B_banded"));

    py::class_<LoessSmootherCpp>(m, "LoessSmootherCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object, double, int>(),
             py::arg("x"), py::arg("weights_obj") = py::none(), py::arg("span") = 0.75, py::arg("degree") = 1)
        .def("fit", &LoessSmootherCpp::fit)
        .def("update_weights", &LoessSmootherCpp::update_weights)
        .def("predict", &LoessSmootherCpp::predict, py::arg("x_new"), py::arg("deriv") = 0);

    py::class_<NaturalSplineSmoother>(m, "NaturalSplineSmoother")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("knots"), py::arg("weights_obj") = py::none())
        .def("smooth", &NaturalSplineSmoother::smooth)
        .def("update_weights", &NaturalSplineSmoother::update_weights)
        .def("compute_df", &NaturalSplineSmoother::compute_df)
        .def("gcv_score", &NaturalSplineSmoother::gcv_score)
        .def("solve_for_df", &NaturalSplineSmoother::solve_for_df, py::arg("target_df"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("solve_gcv", &NaturalSplineSmoother::solve_gcv, py::arg("y"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("predict", &NaturalSplineSmoother::predict)
        .def("get_N", &NaturalSplineSmoother::get_N)
        .def("get_Omega", &NaturalSplineSmoother::get_Omega);

    py::class_<ReinschSmoother>(m, "ReinschSmoother")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("weights") = py::none())
        .def("smooth", &ReinschSmoother::smooth)
        .def("update_weights", &ReinschSmoother::update_weights)
        .def("compute_df", &ReinschSmoother::compute_df)
        .def("compute_df_sparse", &ReinschSmoother::compute_df_sparse)
        .def("gcv_score", &ReinschSmoother::gcv_score)
        .def("solve_for_df", &ReinschSmoother::solve_for_df, py::arg("target_df"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("solve_gcv", &ReinschSmoother::solve_gcv, py::arg("y"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("predict", &ReinschSmoother::predict);

    py::class_<CubicSplineTraceCpp>(m, "CubicSplineTraceCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("weights_obj") = py::none())
        .def("compute_trace", &CubicSplineTraceCpp::compute_trace);

    py::class_<BSplineSmoother>(m, "BSplineSmoother")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&, py::object, int>())
        .def("compute_design", &BSplineSmoother::compute_design)
        .def("compute_penalty", &BSplineSmoother::compute_penalty)
        .def("compute_rhs", &BSplineSmoother::compute_rhs)
        .def("set_solution", &BSplineSmoother::set_solution)
        .def("smooth", &BSplineSmoother::smooth)
        .def("compute_df", &BSplineSmoother::compute_df)
        .def("gcv_score", &BSplineSmoother::gcv_score)
        .def("solve_for_df", &BSplineSmoother::solve_for_df, py::arg("target_df"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("solve_gcv", &BSplineSmoother::solve_gcv, py::arg("y"), py::arg("min_log_lam") = -12.0, py::arg("max_log_lam") = 12.0)
        .def("predict", &BSplineSmoother::predict)
        .def("get_NTWN", &BSplineSmoother::get_NTWN)
        .def("get_Omega", &BSplineSmoother::get_Omega)
        .def("get_knots", &BSplineSmoother::get_knots)
        .def("eval_basis", &BSplineSmoother::eval_basis, py::arg("x_val"), py::arg("deriv")=0);
    
    m.def("compute_natural_spline_basis", &compute_natural_spline_basis,
          py::arg("x"), py::arg("knots"), py::arg("extrapolate_linear") = true, py::arg("derivative_order") = 0);
    m.def("compute_penalty_matrix", &compute_penalty_matrix, py::arg("knots"));
}
