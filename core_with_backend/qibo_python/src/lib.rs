use pyo3::prelude::*;

mod gate;
mod circuit;

#[pymodule]
fn _qibo_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<gate::PyGate>()?;
    m.add_class::<circuit::PyCircuit>()?;
    Ok(())
}
