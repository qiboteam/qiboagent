import pytest
import sys
import os

# Ensure the compiled extension is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "qibo_python")))

import qibo_python

def test_gate_creation():
    # Test creation of a simple H gate
    gate = qibo_python.Gate("H", [0], [], [])
    assert gate.kind_str() == "H"

    # Test creation of a parameterized RX gate
    theta = 0.5
    gate_rx = qibo_python.Gate("RX", [1], [], [theta])
    assert gate_rx.kind_str() == f"RX({theta})"

def test_circuit_add_gate():
    circuit = qibo_python.Circuit(2)
    gate = qibo_python.Gate("CNOT", [1], [0], [])
    circuit.add(gate)
    assert circuit.nqubits() == 2
    assert circuit.gate_count() == 1
    retrieved_gate = circuit.get_gate(0)
    assert retrieved_gate.kind_str() == "CNOT"

def test_integration():
    circuit = qibo_python.Circuit(3)
    h_gate = qibo_python.Gate("H", [0], [], [])
    cx_gate = qibo_python.Gate("CNOT", [2], [0], [])
    rz_gate = qibo_python.Gate("RZ", [1], [], [3.1415])
    circuit.add(h_gate)
    circuit.add(cx_gate)
    circuit.add(rz_gate)
    assert circuit.gate_count() == 3
    assert circuit.get_gate(0).kind_str() == "H"
    assert circuit.get_gate(1).kind_str() == "CNOT"
    assert circuit.get_gate(2).kind_str() == f"RZ({3.1415})"
