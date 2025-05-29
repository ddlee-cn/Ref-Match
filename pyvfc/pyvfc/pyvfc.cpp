#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vfc.h"

namespace py = pybind11;

PYBIND11_MODULE(pyvfc, m) {
    py::class_<VFC>(m, "VFC")
        .def(py::init())
        .def("setData", &VFC::setData)
        .def("setCtrlPts", &VFC::setCtrlPts)
        .def("setCtrlPtsNum", &VFC::setCtrlPtsNum)
        .def("optimize", &VFC::optimize)
        .def("obtainCorrectMatch", &VFC::obtainCorrectMatch)
        .def("obtainP", &VFC::obtainP)
        .def("obtainCoef", &VFC::obtainCoef)
        .def("obtainVecField", &VFC::obtainVecField)
        .def("obtainCtrlPts", &VFC::obtainCtrlPts);
}
