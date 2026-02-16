#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "natural_spline.h"
#include "reinsch.h"
#include "bspline.h"

namespace py = pybind11;

PYBIND11_MODULE(_spline_extension, m) {
    py::class_<NaturalSplineFitter>(m, "NaturalSplineFitter")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("knots"), py::arg("weights_obj") = py::none())
        .def("fit", &NaturalSplineFitter::fit)
        .def("update_weights", &NaturalSplineFitter::update_weights)
        .def("compute_df", &NaturalSplineFitter::compute_df)
        .def("gcv_score", &NaturalSplineFitter::gcv_score)
        .def("solve_for_df", &NaturalSplineFitter::solve_for_df)
        .def("solve_gcv", &NaturalSplineFitter::solve_gcv)
        .def("predict", &NaturalSplineFitter::predict)
        .def("get_N", &NaturalSplineFitter::get_N)
        .def("get_Omega", &NaturalSplineFitter::get_Omega);

    py::class_<ReinschFitter>(m, "ReinschFitter")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("weights") = py::none())
        .def("fit", &ReinschFitter::fit)
        .def("update_weights", &ReinschFitter::update_weights)
        .def("compute_df", &ReinschFitter::compute_df)
        .def("compute_df_sparse", &ReinschFitter::compute_df_sparse)
        .def("gcv_score", &ReinschFitter::gcv_score)
        .def("solve_for_df", &ReinschFitter::solve_for_df)
        .def("solve_gcv", &ReinschFitter::solve_gcv)
        .def("predict", &ReinschFitter::predict);

    py::class_<CubicSplineTraceCpp>(m, "CubicSplineTraceCpp")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, py::object>(),
             py::arg("x"), py::arg("weights_obj") = py::none())
        .def("compute_trace", &CubicSplineTraceCpp::compute_trace);

    py::class_<BSplineFitter>(m, "BSplineFitter")
        .def(py::init<const Eigen::Ref<const Eigen::VectorXd>&, const Eigen::Ref<const Eigen::VectorXd>&, py::object, int>())
        .def("compute_system", &BSplineFitter::compute_system)
        .def("set_solution", &BSplineFitter::set_solution)
        .def("fit", &BSplineFitter::fit)
        .def("compute_df", &BSplineFitter::compute_df)
        .def("gcv_score", &BSplineFitter::gcv_score)
        .def("solve_for_df", &BSplineFitter::solve_for_df)
        .def("solve_gcv", &BSplineFitter::solve_gcv)
        .def("predict", &BSplineFitter::predict)
        .def("get_NTWN", &BSplineFitter::get_NTWN)
        .def("get_Omega", &BSplineFitter::get_Omega)
        .def("get_knots", &BSplineFitter::get_knots)
        .def("eval_basis", &BSplineFitter::eval_basis, py::arg("x_val"), py::arg("deriv")=0);
    
    m.def("compute_natural_spline_basis", &compute_natural_spline_basis,
          py::arg("x"), py::arg("knots"), py::arg("extrapolate_linear") = true, py::arg("derivative_order") = 0);
    m.def("compute_penalty_matrix", &compute_penalty_matrix, py::arg("knots"));
}
