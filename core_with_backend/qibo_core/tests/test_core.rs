//! Unit tests for qibo_core library.

use qibo_core::{gate::{Gate, GateKind}, circuit::Circuit};
use num_complex::Complex64;

#[test]
fn test_gate_creation_hadamard() {
    // Create a Hadamard gate on qubit 0.
    let gate = Gate::new(GateKind::H, vec![0], None);
    assert_eq!(gate.kind, GateKind::H);
    assert_eq!(gate.targets, vec![0]);
    // The matrix for H should be 1/sqrt(2) * [[1, 1], [1, -1]]
    let sqrt2_inv = 1.0_f64 / 2_f64.sqrt();
    let expected = vec![
        Complex64::new(sqrt2_inv, 0.0),
        Complex64::new(sqrt2_inv, 0.0),
        Complex64::new(sqrt2_inv, 0.0),
        Complex64::new(-sqrt2_inv, 0.0),
    ];
    assert_eq!(gate.matrix.unwrap(), expected);
}

#[test]
fn test_circuit_add_and_len() {
    let mut circuit = Circuit::new(2);
    let h_gate = Gate::new(GateKind::H, vec![0], None);
    let x_gate = Gate::new(GateKind::X, vec![1], None);
    circuit.add_gate(h_gate.clone());
    circuit.add_gate(x_gate.clone());
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.gates.len(), 2);
    assert_eq!(circuit.gates[0].kind, GateKind::H);
    assert_eq!(circuit.gates[1].kind, GateKind::X);
}

#[test]
fn test_circuit_equality() {
    let mut c1 = Circuit::new(1);
    let mut c2 = Circuit::new(1);
    let x_gate = Gate::new(GateKind::X, vec![0], None);
    c1.add_gate(x_gate.clone());
    c2.add_gate(x_gate);
    assert_eq!(c1, c2);
}
