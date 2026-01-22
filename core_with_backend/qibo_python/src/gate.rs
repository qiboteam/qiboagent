use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, IntoPyArray, ndarray::Array2};
use num_complex::Complex64;
use qibo_core::gate::{Gate, GateKind};

#[pyclass(name = "Gate", subclass)]
#[derive(Clone)]
pub struct PyGate {
    pub inner: Gate,
}

impl PyGate {
    fn kind_to_string(kind: &GateKind) -> String {
        match kind {
            GateKind::H => "H".to_string(),
            GateKind::X => "X".to_string(),
            GateKind::Y => "Y".to_string(),
            GateKind::Z => "Z".to_string(),
            GateKind::I => "I".to_string(),
            GateKind::RX(_) => "RX".to_string(),
            GateKind::RY(_) => "RY".to_string(),
            GateKind::RZ(_) => "RZ".to_string(),
            GateKind::CNOT => "CNOT".to_string(),
            GateKind::CZ => "CZ".to_string(),
            GateKind::SWAP => "SWAP".to_string(),
        }
    }

    fn string_to_kind(name: &str, param: Option<f64>) -> Result<GateKind, String> {
        match name {
            "H" => Ok(GateKind::H),
            "X" => Ok(GateKind::X),
            "Y" => Ok(GateKind::Y),
            "Z" => Ok(GateKind::Z),
            "I" => Ok(GateKind::I),
            "RX" => param.map(GateKind::RX).ok_or_else(|| "RX requires a parameter".to_string()),
            "RY" => param.map(GateKind::RY).ok_or_else(|| "RY requires a parameter".to_string()),
            "RZ" => param.map(GateKind::RZ).ok_or_else(|| "RZ requires a parameter".to_string()),
            "CNOT" => Ok(GateKind::CNOT),
            "CZ" => Ok(GateKind::CZ),
            "SWAP" => Ok(GateKind::SWAP),
            _ => Err(format!("Unknown gate name: {}", name)),
        }
    }
}

#[pymethods]
impl PyGate {
    #[new]
    fn new(name: String, targets: Vec<usize>, controls: Option<Vec<usize>>, param: Option<f64>) -> PyResult<Self> {
        let kind = Self::string_to_kind(&name, param)
            .map_err(|e| PyValueError::new_err(e))?;
        // Core expects Option<Vec<usize>> for controls
        let gate = Gate::new(kind, targets, controls);
        Ok(PyGate { inner: gate })
    }

    #[getter]
    fn kind(&self) -> PyResult<String> {
        Ok(Self::kind_to_string(&self.inner.kind))
    }

    #[getter]
    fn target_qubits(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.targets.clone())
    }
    
    #[getter]
    fn control_qubits(&self) -> PyResult<Vec<usize>> {
        Ok(self.inner.controls.clone().unwrap_or_default())
    }
    
    #[getter]
    fn is_controlled_by(&self) -> PyResult<bool> {
        Ok(self.inner.controls.as_ref().map_or(false, |c| !c.is_empty()))
    }
    
    #[getter]
    fn qubits(&self) -> PyResult<Vec<usize>> {
        let mut qs = self.control_qubits()?;
        qs.extend(self.target_qubits()?);
        Ok(qs)
    }
    
    fn matrix<'py>(&self, py: Python<'py>, _backend: Option<&PyAny>) -> PyResult<&'py PyArray2<Complex64>> {
        let m = self.inner.matrix.as_ref().ok_or_else(|| PyValueError::new_err("Gate has no matrix"))?;
        let n_targets = self.inner.targets.len();
        let dim = 2_usize.pow(n_targets as u32);
        
        // For controlled gates, return only the target submatrix
        let data: Vec<Complex64> = if self.is_controlled_by()? && m.len() == 16 && dim == 2 {
            vec![m[10], m[11], m[14], m[15]] 
        } else {
            m.clone()
        };

        let mat2d: Vec<Vec<Complex64>> = data.chunks(dim).map(|r| r.to_vec()).collect();
        Ok(PyArray2::from_vec2(py, &mat2d)?)
    }

    fn apply(&self, _py: Python, backend: &PyAny, state: &PyAny, nqubits: usize) -> PyResult<PyObject> {
        backend
            .call_method1("apply_gate", (self.clone(), state, nqubits))
            .map(|obj| obj.into())
    }

    fn apply_density_matrix(&self, _py: Python, backend: &PyAny, state: &PyAny, nqubits: usize) -> PyResult<PyObject> {
        backend
            .call_method1("apply_gate_density_matrix", (self.clone(), state, nqubits))
            .map(|obj| obj.into())
    }

}
