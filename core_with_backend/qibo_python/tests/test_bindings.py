import math
import numpy as np
import qibo
from qibo import gates as qibogates
from qibo import Circuit as QiboCircuit
from qibo_python import circuit as py_circuit_mod
from qibo_python import gates as pygates

# Alias for convenience
PyCircuit = py_circuit_mod.Circuit

def test_circuit_comparison():
    backend = qibo.backends.NumpyBackend()

    # Build Qibo circuit using original Qibo API
    qibo_circuit = QiboCircuit(3)
    qibo_circuit.add(qibogates.H(0))
    qibo_circuit.add(qibogates.X(1))
    qibo_circuit.add(qibogates.RY(2, math.pi / 3))
    qibo_circuit.add(qibogates.CNOT(0, 1))
    qibo_circuit.add(qibogates.CZ(1, 2))

    # Build circuit using our Python wrapper over Rust core
    py_circuit = PyCircuit(3)
    py_circuit.add(pygates.H(0))
    py_circuit.add(pygates.X(1))
    py_circuit.add(pygates.RY(2, math.pi / 3))
    py_circuit.add(pygates.CNOT(0, 1))
    py_circuit.add(pygates.CZ(1, 2))

    # Verify that the number of gates matches
    assert len(py_circuit.gates) == len(qibo_circuit.queue)

    # Verify that each gate kind matches
    for i, (qg, pyg) in enumerate(zip(qibo_circuit.queue, py_circuit.gates)):
        assert qg.__class__.__name__ == pyg.kind, (
            f"Gate {i} mismatch: Qibo={qg.__class__.__name__}, PyCircuit={pyg.kind}"
        )

    # Execute both circuits using the same backend and compare final states
    s1 = backend.execute_circuit(qibo_circuit)
    s2 = backend.execute_circuit(py_circuit)

    np.testing.assert_allclose(s1.state(), s2.state(), atol=1e-8)

def test_circuit_comparison2():
    backend = qibo.backends.NumpyBackend()

    # Build Qibo circuit using original Qibo API
    qibo_circuit = QiboCircuit(2)
    qibo_circuit.add(qibogates.H(0))
    qibo_circuit.add(qibogates.H(1))
    qibo_circuit.add(qibogates.CZ(0, 1))
    qibo_circuit.add(qibogates.RY(0, theta=np.pi/3))
    qibo_circuit.add(qibogates.RX(1, theta=np.pi/5))

    # Build circuit using our Python wrapper over Rust core
    py_circuit = PyCircuit(2)
    py_circuit.add(pygates.H(0))
    py_circuit.add(pygates.H(1))
    py_circuit.add(pygates.CZ(0, 1))
    py_circuit.add(pygates.RY(0, theta=np.pi/3))
    py_circuit.add(pygates.RX(1, theta=np.pi/5))

    # Verify that the number of gates matches
    assert len(py_circuit.gates) == len(qibo_circuit.queue)

    # Verify that each gate kind matches
    for i, (qg, pyg) in enumerate(zip(qibo_circuit.queue, py_circuit.gates)):
        assert qg.__class__.__name__ == pyg.kind, (
            f"Gate {i} mismatch: Qibo={qg.__class__.__name__}, PyCircuit={pyg.kind}"
        )

    # Execute both circuits using the same backend and compare final states
    s1 = backend.execute_circuit(qibo_circuit)
    s2 = backend.execute_circuit(py_circuit)

    np.testing.assert_allclose(s1.state(), s2.state(), atol=1e-8)


if __name__ == "__main__":
    test_circuit_comparison()
    test_circuit_comparison2()
    print("All tests passed successfully.")
