use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use std::sync::Mutex;
use qibo_core::circuit::Circuit as CoreCircuit;
use crate::gate::PyGate;

#[pyclass(name = "Circuit", subclass)]
pub struct PyCircuit {
    inner: Mutex<CoreCircuit>,
}

#[pymethods]
impl PyCircuit {
    #[new]
    fn new(nqubits: usize) -> PyResult<Self> {
        Ok(PyCircuit {
            inner: Mutex::new(CoreCircuit::new(nqubits)),
        })
    }

    fn add(&self, gate: &PyGate) -> PyResult<()> {
        let mut circuit = self.inner.lock().map_err(|_| PyRuntimeError::new_err("Mutex poisoned"))?;
        circuit.add_gate(gate.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    #[getter]
    fn nqubits(&self) -> PyResult<usize> {
        let circuit = self.inner.lock().map_err(|_| PyRuntimeError::new_err("Mutex poisoned"))?;
        Ok(circuit.num_qubits)
    }

    #[getter]
    fn queue(&self) -> PyResult<Vec<PyGate>> {
        let circuit = self.inner.lock().map_err(|_| PyRuntimeError::new_err("Mutex poisoned"))?;
        Ok(circuit.gates.iter().cloned().map(|g| PyGate { inner: g }).collect())
    }

    #[getter]
    fn gates(&self) -> PyResult<Vec<PyGate>> {
        // Alias for queue
        self.queue()
    }
}
