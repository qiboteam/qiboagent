use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use qibo_core::gate::{Gate, GateKind};

#[pyclass(name = "Gate")]
#[derive(Clone)]
pub struct PyGate {
    pub inner: Gate,
}

#[pymethods]
impl PyGate {
    #[new]
    fn new(name: String, targets: Vec<usize>, controls: Vec<usize>, params: Vec<f64>) -> PyResult<Self> {
        // Determine GateKind based on name and params
        let kind = match name.as_str() {
            "H" => GateKind::H,
            "X" => GateKind::X,
            "Y" => GateKind::Y,
            "Z" => GateKind::Z,
            "RX" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("RX gate requires exactly one parameter (theta)"));
                }
                GateKind::RX(params[0])
            },
            "RY" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("RY gate requires exactly one parameter (theta)"));
                }
                GateKind::RY(params[0])
            },
            "RZ" => {
                if params.len() != 1 {
                    return Err(PyValueError::new_err("RZ gate requires exactly one parameter (theta)"));
                }
                GateKind::RZ(params[0])
            },
            "CNOT" => GateKind::CNOT,
            "CZ" => GateKind::CZ,
            _ => return Err(PyValueError::new_err(format!("Unknown gate name: {}", name))),
        };
        Ok(PyGate { inner: Gate::new(kind, targets, controls, params) })
    }

    /// Return a string representation of the gate kind.
    fn kind_str(&self) -> PyResult<String> {
        Ok(format!("{}", self.inner.kind))
    }
}
