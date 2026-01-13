use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub enum GateKind {
    H,
    X,
    Y,
    Z,
    RX(f64),
    RY(f64),
    RZ(f64),
    CNOT,
    CZ,
    // Additional gates can be added here
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub kind: GateKind,
    pub targets: Vec<usize>,
    pub controls: Vec<usize>,
    pub params: Vec<f64>,
}

impl Gate {
    pub fn new(kind: GateKind, targets: Vec<usize>, controls: Vec<usize>, params: Vec<f64>) -> Self {
        Gate { kind, targets, controls, params }
    }
}

impl fmt::Display for GateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateKind::H => write!(f, "H"),
            GateKind::X => write!(f, "X"),
            GateKind::Y => write!(f, "Y"),
            GateKind::Z => write!(f, "Z"),
            GateKind::RX(theta) => write!(f, "RX({})", theta),
            GateKind::RY(theta) => write!(f, "RY({})", theta),
            GateKind::RZ(theta) => write!(f, "RZ({})", theta),
            GateKind::CNOT => write!(f, "CNOT"),
            GateKind::CZ => write!(f, "CZ"),
        }
    }
}
