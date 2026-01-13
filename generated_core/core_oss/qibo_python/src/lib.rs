use pyo3::prelude::*;

mod gate;
mod circuit;

use crate::gate::PyGate;
use crate::circuit::PyCircuit;

#[pymodule]
fn qibo_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGate>()?;
    m.add_class::<PyCircuit>()?;
    Ok(())
}
