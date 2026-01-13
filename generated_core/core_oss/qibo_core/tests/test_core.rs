use qibo_core::gate::{Gate, GateKind};
use qibo_core::circuit::Circuit;

#[test]
fn test_gate_creation() {
    let gate = Gate::new(GateKind::H, vec![0], vec![], vec![]);
    assert_eq!(gate.kind, GateKind::H);
    assert_eq!(gate.targets, vec![0]);
    assert!(gate.controls.is_empty());
    assert!(gate.params.is_empty());
}

#[test]
fn test_circuit_add_gate() {
    let mut circuit = Circuit::new(2);
    let gate = Gate::new(GateKind::CNOT, vec![1], vec![0], vec![]);
    circuit.add_gate(gate.clone());
    assert_eq!(circuit.nqubits, 2);
    assert_eq!(circuit.gates.len(), 1);
    assert_eq!(circuit.gates[0], gate);
}
