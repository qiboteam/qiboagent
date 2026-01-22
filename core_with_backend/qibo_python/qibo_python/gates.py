"""Python wrapper for quantum gates."""
from _qibo_core import Gate as _RustGate

class Gate(_RustGate):
    """Base gate; inherits from Rust binding."""
    def apply(self, backend, state, nqubits: int):
        return backend.apply_gate(self, state, nqubits)

# Factory functions
def H(target: int):
    return _RustGate("H", [target], None, None)

def X(target: int):
    return _RustGate("X", [target], None, None)

def Y(target: int):
    return _RustGate("Y", [target], None, None)

def Z(target: int):
    return _RustGate("Z", [target], None, None)

def I(target: int):
    return _RustGate("I", [target], None, None)

def RX(target: int, theta: float):
    return _RustGate("RX", [target], None, theta)

def RY(target: int, theta: float):
    return _RustGate("RY", [target], None, theta)

def RZ(target: int, theta: float):
    return _RustGate("RZ", [target], None, theta)

def CNOT(control: int, target: int):
    return _RustGate("CNOT", [target], [control], None)

def CZ(control: int, target: int):
    return _RustGate("CZ", [target], [control], None)

def SWAP(q1: int, q2: int):
    return _RustGate("SWAP", [q1, q2], None, None)