use pyo3::prelude::*;
use pyo3::exceptions::PyIndexError;
use std::sync::Mutex;
use qibo_core::circuit::Circuit;

use crate::gate::PyGate;

#[pyclass(name = "Circuit")]
pub struct PyCircuit {
    pub inner: Mutex<Circuit>,
}

#[pymethods]
impl PyCircuit {
    #[new]
    fn new(nqubits: usize) -> Self {
        PyCircuit { inner: Mutex::new(Circuit::new(nqubits)) }
    }

    fn add(&self, gate: &PyGate) {
        let mut circuit = self.inner.lock().unwrap();
        circuit.add_gate(gate.inner.clone());
    }

    fn nqubits(&self) -> PyResult<usize> {
        let circuit = self.inner.lock().unwrap();
        Ok(circuit.nqubits)
    }

    fn gate_count(&self) -> PyResult<usize> {
        let circuit = self.inner.lock().unwrap();
        Ok(circuit.gates.len())
    }

    fn get_gate(&self, idx: usize) -> PyResult<PyGate> {
        let circuit = self.inner.lock().unwrap();
        if idx < circuit.gates.len() {
            Ok(PyGate { inner: circuit.gates[idx].clone() })
        } else {
            Err(PyIndexError::new_err("Gate index out of range"))
        }
    }
}
