use num_complex::Complex64;
use std::f64::consts::SQRT_2;

#[derive(Debug, Clone, PartialEq)]
pub enum GateKind {
    H,
    X,
    Y,
    Z,
    I,
    RX(f64),
    RY(f64),
    RZ(f64),
    CNOT,
    CZ,
    SWAP,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gate {
    pub kind: GateKind,
    pub targets: Vec<usize>,
    // For simplicity we keep control qubits optional; not used in current tests
    pub controls: Option<Vec<usize>>, // None for single-qubit gates
    pub matrix: Option<Vec<Complex64>>, // Some(matrix) for defined gates
}

impl Gate {
    /// Create a new gate.
    /// `targets` is a list of target qubit indices.
    /// `controls` is an optional list of control qubit indices (use `None` for single‑qubit gates).
    /// The function always returns a `Gate` (tests do not expect error handling).
    pub fn new(
        kind: GateKind,
        targets: Vec<usize>,
        controls: Option<Vec<usize>>,
    ) -> Self {
        // Helper to create complex numbers
        let c = |re: f64, im: f64| Complex64::new(re, im);
        // Generate matrix for the gate (always Some for the supported kinds)
        let matrix = match &kind {
            GateKind::H => {
                let s = 1.0 / SQRT_2;
                Some(vec![c(s, 0.0), c(s, 0.0), c(s, 0.0), c(-s, 0.0)])
            }
            GateKind::X => Some(vec![c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)]),
            GateKind::Y => Some(vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]),
            GateKind::Z => Some(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)]),
            GateKind::I => Some(vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]),
                       
                        GateKind::RX(theta) => {
                            let h = theta / 2.0;
                            Some(vec![
                                c(h.cos(), 0.0),      c(0.0, -h.sin()),
                                c(0.0, -h.sin()),     c(h.cos(), 0.0),
                            ])
                        }
                        GateKind::RY(theta) => {
                            let h = theta / 2.0;
                            Some(vec![
                                c(h.cos(), 0.0),   c(-h.sin(), 0.0),
                                c(h.sin(), 0.0),   c(h.cos(), 0.0),
                            ])
                        }
                        GateKind::RZ(theta) => {
                            let h = theta / 2.0;
                            Some(vec![
                                c(h.cos(), -h.sin()), c(0.0, 0.0),
                                c(0.0, 0.0),          c(h.cos(),  h.sin()),
                            ])
                        }
           
            GateKind::CNOT => {
                Some(vec![
                    c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0),
                    c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0),
                ])
            }
            GateKind::CZ => {
                Some(vec![
                    c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0),
                ])
            }
            GateKind::SWAP => {
                Some(vec![
                    c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0),
                    c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0),
                ])
            }
        };
        Gate {
            kind,
            targets,
            controls,
            matrix,
        }
    }
}
