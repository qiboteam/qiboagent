use std::fmt;

use crate::gate::Gate;

pub struct Circuit {
    pub nqubits: usize,
    pub gates: Vec<Gate>,
}

impl Circuit {
    pub fn new(nqubits: usize) -> Self {
        Circuit { nqubits, gates: Vec::new() }
    }

    pub fn add_gate(&mut self, gate: Gate) {
        self.gates.push(gate);
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Circuit with {} qubits and {} gates", self.nqubits, self.gates.len())
    }
}
