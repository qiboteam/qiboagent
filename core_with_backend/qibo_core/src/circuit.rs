use crate::gate::Gate;

#[derive(Debug, PartialEq)]
pub struct Circuit {
    pub num_qubits: usize,
    pub gates: Vec<Gate>,
}

impl Circuit {
    /// Create a new circuit with the given number of qubits.
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            num_qubits,
            gates: Vec::new(),
        }
    }

    /// Add a gate to the circuit, performing basic validation.
    pub fn add_gate(&mut self, gate: Gate) -> Result<(), String> {
        // Validate qubit indices are within range
        for &q in gate.targets.iter() {
            if q >= self.num_qubits {
                return Err(format!(
                    "Qubit index {} out of bounds for circuit with {} qubits",
                    q, self.num_qubits
                ));
            }
        }
        if let Some(ctrls) = &gate.controls {
            for &c in ctrls.iter() {
                if c >= self.num_qubits {
                    return Err(format!(
                        "Control qubit index {} out of bounds for circuit with {} qubits",
                        c, self.num_qubits
                    ));
                }
            }
        }
        self.gates.push(gate);
        Ok(())
    }
}
