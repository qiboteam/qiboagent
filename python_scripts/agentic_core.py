"""
Agentic Approach for Qibo-Core module Generation
"""

import os
import sys
import logging
import json
import subprocess, shlex
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from ollama._types import ResponseError

from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# LOG SETTINGS
console = Console()
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

for logger_name in ['httpx', 'httpcore', 'langchain_ollama', 'langgraph']:
    logging.getLogger(logger_name).setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

SETTINGS_PATH = "../settings_json/settings.json"
BASE_DIR = Path("../qibo")

CORE_DIR = Path("../core_gen/core")
RUST_CORE_DIR = CORE_DIR / "qibo_core"
BINDINGS_DIR = CORE_DIR / "qibo_python"


# PROMPT DEFINITIONS
SYSTEM_PROMPT = """You are a software assistant for Qibo, a python framework for quantum computing.
Your Mission is to create qibo_core, A Rust module that implements the core logic of Qibo, and qibo_python, a PyO3 bindings module that exposes qibo_core to Python being compatible with qibo original backends.
You will work only on a single phase at a time, following the instructions given in each phase prompt.

# ABSOLUTELY MOST IMPORTANT PRIMARY RULE:
- YOU MUST WORK ONLY ON THE FILES AND TASK RELATED TO YOUR CURRENT PHASE/LEVEL, AS DETAILED IN THE PHASE PROMPT.
- DO NOT ATTEMPT TO WRITE OR MODIFY FILES OUTSIDE YOUR CURRENT PHASE.

## OVERALL ARCHITECTURE:
1. **LEVEL 1: The Engine (`qibo_core`)**
   - Pure Rust. No Python dependencies.
   - Handles fundamental data structures and logic of qibo: Gate, Circuit.
   - Located in: `qibo_core/`

2. **LEVEL 2: The Bridge (`_qibo_core`)**
   - Rust + PyO3.
   - Wraps Level 1 structs into Python classes.
   - **Crucial:** This module is INTERNAL. It creates a generic interface not a user-friendly API.
   - Located in: `qibo_python/src` (Rust code)

3. **LEVEL 3: The Interface (`qibo_python` package)**
   - Pure Python.
   - Provides the Class Inheritance to match the original API exactly.
   - Inherits from Level 2 classes and overrides `__init__` to simplify user arguments.
   - Located in: `qibo_python/qibo_python/`

# CODING RULES:
   - Use idiomatic Rust (Result<T, E> for errors, Vec for lists, Option for nullables).
   - Add comments where necessary.

# TOOL USAGE:
   - You must use `write_file` to save your code.
   - You must use `read_file` if you need to recall previously generated code.
   - Do not output code blocks in chat; strictly use the tools.

## PROJECT STRUCTURE:
.
├── qibo_core/                  # [LEVEL 1] Pure Rust Logic
│   ├── Cargo.toml              # Dependencies: smallvec, num-complex, serde
│   ├── src/
│   │   ├── lib.rs              # Exports modules
│   │   ├── gate.rs             # Struct Gate (with matrix), Enum GateKind
│   │   └── circuit.rs          # Struct Circuit
│   └── tests/
│       └── test_core.rs        # Unit tests for matrix generation
│
└── qibo_python/                # [LEVEL 2 & 3] Bindings + Python Wrapper
    ├── Cargo.toml              # Package: _qibo_core, crate-type: cdylib
    ├── pyproject.toml          # Config: module-name="_qibo_core", python-source="."
    ├── src/                    # [LEVEL 2] Rust Bindings Code
    │   ├── lib.rs              # #[pymodule] _qibo_core
    │   ├── gate.rs             # #[pyclass] PyGate
    │   └── circuit.rs          # #[pyclass] PyCircuit
    ├── qibo_python/            # [LEVEL 3] Python Wrapper Package (RENAMED from qibo_interface)
    │   ├── __init__.py         # Exports Circuit, gates
    │   ├── circuit.py          # class Circuit(_RustCircuit)
    │   └── gates.py            # class H(_RustGate) - Factory classes
    └── tests/
        └── test_bindings.py    # Pytest checking the high-level API
"""
        
PHASE1_PROMPT_TEMPLATE = """
### PHASE 1: CORE IMPLEMENTATION (PURE RUST)

TASK:
Implement ONLY `qibo_core`, the pure Rust core logic for quantum circuits and gates.

# CONTEXT:
This is the relevant Python source code that defines Quantum gates and Quantum circuits in Qibo:
{source_code}

WORKFLOW (FOLLOW EXACTLY):

Your task is to re-implement the core logic in Rust based on context, following EXACTLY these steps in order:

1. Create **Gate Model (`src/gate.rs`) using `write_file` with:**
    - **Enum:** `pub enum GateKind` with variants: H, X, Y, Z, I, RX(f64), RY(f64), RZ(f64), CNOT, CZ, SWAP.
    - **Struct:** `pub struct Gate` MUST have these exact fields:
        - pub kind: GateKind,
        - pub param: Option<f64>,
        - pub target_qubits: Vec<usize>,
        - pub control_qubits: Vec<usize>,
        - pub matrix: Vec<Complex64>
    - **Logic (`impl Gate`):**
      - derive on Debug, Clone, PartialEq, only
      - Implement `fn new(kind: GateKind, targets: ..., controls: ...) -> Result<Self, String>`.
        - Validate qbit indices for all kind of gates (one qbit, parameters, two qbits).
      - **Matrix Generation:** Inside `new()`, you must automatically generate the matrix based on `kind` and store it in `self.matrix`.
        - Example `H`: `vec![c(1./s2, 0.), c(1./s2, 0.), c(1./s2, 0.), c(-1./s2, 0.)]` where `s2 = sqrt(2)`.
      

2. Create **Circuit Model (`src/circuit.rs`) using `write_file` with:**
    - `pub struct Circuit`: `nqubits`, `queue`.
    - Implement `add(gate)` with validation rules.
    - derive partialeq only.

3. Create Library Entry Point (`src/lib.rs`) related to these files with all the necessary `mod` and `pub use` statements.

4. Generate Cargo.toml for `qibo_core` with all the necessary dependencies

5. Completion: Once all the previous steps are completed and files are saved, this phase is finished. use `task_complete` to signal completion. stop your execution for this phase.

CRITICAL: 
- Work only on level 1 files mentioned above.
"""

PHASE2_PROMPT = """
### PHASE 2: INTERNAL BINDINGS (_qibo_core)

TASK:
Implement the Rust-side bindings that expose the core module to Python.

WORKFLOW (FOLLOW EXACTLY):

1. Read the previously generated `qibo_core` Rust code using `read_file` to gain context on the structures.:
    - `qibo_core/src/gate.rs`
    - `qibo_core/src/circuit.rs`
    - `qibo_core/Cargo.toml`

2. Generate **The Module (`src/lib.rs`):**
   - Define `#[pymodule] fn _qibo_core(...)`.
   - Register classes `PyGate` and `PyCircuit`.

3. Generate **Gate Bindings (`src/gate.rs`):**
   - `#[pyclass(name = "Gate", subclass)]` -> **Add `subclass` argument!** This allows Python to inherit from it.
   - Wrap `qibo_core::gate::Gate`.
   - `#[new]` constructor should be generic: `fn new(name: String, targets, controls, params)`.
   - implement all the following getters:
        - `fn kind(&self) -> String`
        - `fn target_qubits(&self) -> Vec<usize>`
        - `fn control_qubits(&self) -> Vec<usize>`
        - `fn qubits (&self) -> Vec<usize>` // returns all qubits involved (targets + controls)
        
   - **Matrix Method (CRUCIAL):**
     - `fn matrix<'py>(&self, py: Python<'py>, _backend: Option<&PyAny>) -> PyResult<&'py PyArray2<Complex64>>`
     **Logic:**
     - Get the inner matrix from `qibo_core::gate::Gate`
     - Calculate `dim = 2^(number of target qubits)`
     - **IF the gate is controlled (check `is_controlled_by()`) AND the matrix is 4x4 (16 elements) AND dim==2:**
       - Extract the bottom-right 2x2 block: `vec![m[10], m[11], m[14], m[15]]`
       - This is the operation that acts on the target when control=|1⟩
     - **ELSE:**
       - Use the full matrix as-is
     - Convert to 2D numpy array using `PyArray2::from_vec2`
     
     **Explanation:** Qibo's backend applies controlled gates using `_apply_gate_controlled_by()` which expects:
     - `gate.target_qubits` = only targets (for CNOT: 1 qubit)
     - `gate.control_qubits` = only controls (for CNOT: 1 qubit)
     - `gate.matrix()` = 2x2 matrix acting on target when control is |1⟩

   - **Execution Methods (Callbacks):**
     - `fn apply(&self, py: Python, backend: &PyAny, state: &PyAny, nqubits: usize) -> PyResult<PyObject>`
       - Logic: Call `backend.call_method1("apply_gate", (self, state, nqubits))`.
     - `fn apply_density_matrix(...)` wrapping `apply_gate_density_matrix`.
     - `fn is_controlled_by(&self, qubit: usize) -> bool`

4. Generate **Circuit Bindings (`src/circuit.rs`):**
   - `#[pyclass(name = "Circuit", subclass)]`.
   - Wrap `Mutex<qibo_core::circuit::Circuit>`.
   - `fn add(&self, gate: &PyGate)`: Takes the generic PyGate.
   - implement #[getters] methods for all the following:
        - `fn nqubits(&self) -> PyResult<usize>`
        - `fn queue(&self) -> PyResult<Vec<PyGate>>`
        - `fn gates(&self) -> PyResult<Vec<PyGate>>` // returns all gates in the circuit

5. Generate **Cargo Configuration (`qibo_python/Cargo.toml`):**
   - **Package Name:** `_qibo_core`.
   - `crate-type = ["cdylib"]`.
   - Dependencies: `pyo3`, `qibo_core` (path="../qibo_core"), numpy and num-complex.

6. **Completion**: Once all the previous steps are completed and files are saved, this phase is finished. use `task_complete` to signal completion. stop your execution for this phase.

CRITICAL:
- The compiled module MUST be named `_qibo_core`.
- DO NOT implement any high-level Python API here. This is ONLY the Rust bindings.
- Work only on level 2 files mentioned above.
"""

PHASE3_PROMPT = """
### PHASE 3: PYTHON API WRAPPER

TASK:
Create the user-facing Python package that wraps the Rust extension.
the python api must match the original Qibo API exactly. 

QIBO API EXAMPLE
```python
from qibo import Circuit, gates

circuit = Circuit(3)
circuit.add(gates.H(0))
circuit.add(gates.CNOT(0, 2))
circuit.add(gates.CNOT(1, 2))
circuit.add(gates.RX(1, 0.5))
circuit.add(gates.H(2))
```
WORKFLOW:
Structure: `qibo_python/qibo_python/`.

0. Read the previously generated Rust binding code using `read_file` to understand the exposed classes and methods:
    - `src/lib.rs`
    - `src/gate.rs`
    - `src/circuit.rs`
    - `tests/test_bindings.py`

1. **Gates Module (`qibo_python/qibo_python/gates.py`):**
   - Create a Python class Gate(_RustGate) inheriting from Rust with
        - @property
        def is_controlled_by(self) -> bool:
        - def apply(self, backend, state, nqubits: int):
        return backend.apply_gate(self, state, nqubits)
   - From the Rust `PyGate`, create factory functions for each gate:
    example:
    ```python
    def H(target: int):
         return _RustGate("H", [target], None, None)
    
    def CNOT(control: int, target: int):
         return _RustGate("CNOT", [target], [control], None)
    
    ...
    ```
     
   - The classes must match the original Qibo API.

2. **Circuit Module (`qibo_python/qibo_python/circuit.py`):**
   - `from _qibo_core import Circuit as _RustCircuit`
   - `class Circuit(_RustCircuit): ...` (Add docstrings or helpers if needed).
   - add all the following methods for the backend compatibility:
        @property
    def repeated_execution(self) -> bool:
        return False
    
    @property
    def accelerators(self):
        return None
    
    @property
    def density_matrix(self) -> bool:
        return False
    
    @property
    def measurements(self) -> list:
        return []
    
    @property
    def has_collapse(self) -> bool:
        return False
    
    @property
    def has_unitary_channel(self) -> bool:
        

3. **Package Init (`qibo_python/qibo_python/__init__.py`):**
   - Expose `Circuit` and `gates`.

4. **Build Config (`qibo_python/pyproject.toml`):**
   - **CRITICAL:** Use this mapping configuration:
     ```toml
     [build-system]
     requires = ["maturin>=1.0,<2.0"]
     build-backend = "maturin"

     [project]
     name = "qibo_python"
     version = "0.1.0"
     requires-python = ">=3.8"

     [tool.maturin]
     python-source = "."
     module-name = "_qibo_core"
     ```

5. **Completion**: Once all the previous steps are completed and files are saved, this phase is finished. use `task_complete` to signal completion. stop your execution for this phase.

CRITICAL:
- Work only on level 3 files mentioned above.
"""

PHASE4_PROMPT = """
### PHASE 4: TESTS GENERATION

TASK:
Generate ONLY test files to verify the implementation. DO NOT modify any existing source code.

WORKFLOW:
1. generate **Rust Tests:** `qibo_core/tests/test_core.rs` (Unit tests for math/logic).
    - Files in `tests/` are **integration tests** (external to the crate)
    - They MUST import using: `use qibo_core::{Gate, GateKind, Circuit};`
2. **Python Tests:** `qibo_python/tests/test_bindings.py`
   generate this test
   def test_circuit_comparison():

    backend = qibo.backends.NumpyBackend()

    qibo_circuit = QiboCircuit(3)
    qibo_circuit.add(qibogates.H(0))
    qibo_circuit.add(qibogates.X(1))
    qibo_circuit.add(qibogates.RY(2, math.pi / 3))
    qibo_circuit.add(qibogates.CNOT(0, 1))
    qibo_circuit.add(qibogates.CZ(1, 2))

    py_circuit = PyCircuit(3)
    py_circuit.add(pygates.H(0))
    py_circuit.add(pygates.X(1))
    py_circuit.add(pygates.RY(2, math.pi / 3))
    py_circuit.add(pygates.CNOT(0, 1))
    py_circuit.add(pygates.CZ(1, 2))

    assert len(py_circuit.gates) == len(qibo_circuit.queue)

    for i, (qg, pyg) in enumerate(zip(qibo_circuit.queue, py_circuit.gates)):
        assert qg.__class__.__name__ == pyg.kind, f"Gate {i} mismatch: Qibo={qg.__class__.__name__}, PyCircuit={pyg.kind}"
    
    s1 = backend.execute_circuit(qibo_circuit)
    s2 = backend.execute_circuit(py_circuit)

    np.testing.assert_allclose(s1.state(), s2.state(), atol=1e-8)

3. **Completion**: Once all the previous steps are completed and files are saved, this phase is finished. use `task_complete` to signal completion. stop your execution for this phase.

"""


#-------TOOL DEFINITIONS--------#

@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@tool("write_file", description="Writes content to a specified file path.")
def write_file(file_path: str, content: str) -> str:
    path = CORE_DIR / file_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    rprint(f"[green]Wrote {len(content)} bytes to {path}[/green]")
    return f"Wrote {file_path}, NOW PROCEED WITH THE NEXT STEPS."

@tool("task_complete", description="Call this tool ONLY when you have completed all the tasks for the current phase.")
def task_complete(summary: str) -> str:
    """
    Used to signal that the agent has finished the job.
    """
    return "TASK_COMPLETED_SUCCESSFULLY"

@tool("read_file", description="Reads content from a specified file path.")
def read_file(file_path: str) -> str:
    path = CORE_DIR / file_path
    if not path.exists():
        return f"Error: File {file_path} does not exist."
    rprint(f"[blue]Reading file {path}[/blue]")
    return path.read_text(encoding="utf-8")

# @tool("run_shell_command", description="Executes a shell command. Use this for 'cargo', 'maturin', or 'pytest'.")
# def run_shell_command(command: str, working_directory: str = ".") -> str:
#     target_dir = CORE_DIR / working_directory
#     if command.strip() == "rm":
#         return "Error: 'rm' command is not allowed for safety reasons."
#     if not target_dir.exists():
#         return f"Error: Directory {working_directory} does not exist."

#     try:
#         result = subprocess.run(
#             shlex.split(command),
#             cwd=str(target_dir),
#             capture_output=True,
#             text=True,
#             check=False 
#         )
        
#         output = f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
        
#         if result.returncode == 0:
#             return f"SUCCESS:\n{output}"
#         else:
#             return f"FAILURE (Exit Code {result.returncode}):\n{output}"
            
#     except Exception as e:
#         return f"Execution Error: {str(e)}"

def run_command_check(command: str, working_directory: str) -> tuple[bool, str]:
    """Runs a shell command and returns a tuple of (success, output)."""
    target_dir = CORE_DIR / working_directory
    try:
        result = subprocess.run(
            shlex.split(command),
            cwd=str(target_dir),
            capture_output=True,
            text=True
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return (result.returncode == 0), output
    except Exception as e:
        return False, str(e)

def load_settings(path: str) -> dict:
    """Load settings from a JSON file."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}

def get_python_source(base_dir: Path) -> str:
    """Reads the relevant Python source files and concatenates their content for context."""
    files = [
        "src/qibo/models/circuit.py",
        "src/qibo/gates/abstract.py",
        "src/qibo/gates/gates.py",
    ]

    context = []
    for f in files:
        p = base_dir / f
        if p.exists():
            context.append(f"=== {f} ===\n{p.read_text()[:10000]}") 
        else:
            rprint(f"[red]Missing source: {f}[/red]")

    return "\n\n".join(context)

def init_agent(model_name: str, reasoning: bool, checkpointer=None):
    """Initializes the agent with the specified model and tools."""
    model = ChatOllama(
        model=model_name,
        temperature=0.0,
        reasoning=reasoning,
        base_url=["llm"]["base_url"],
        timeout=1000.0,
        num_ctx=32768
    )

    return create_agent(
        model=model,
        tools=[write_file, read_file, task_complete],#run_shell_command],
        checkpointer=InMemorySaver() if checkpointer is None else checkpointer,
        system_prompt=SYSTEM_PROMPT,
        middleware=[handle_tool_errors],
    )

def print_stream(event):
    """Prints the agent's execution log with rich formatting, including tool calls and their content previews."""
    if "messages" not in event:
        return

    for msg in event["messages"]:
        if msg.type == "ai":
            if msg.content:
                preview = msg.content[:300] + "..." if len(msg.content) > 300 else msg.content
                rprint(f"[magenta]AI Response: {preview}[/magenta]")
            
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    file_path = tc['args'].get('file_path', 'unknown')
                    content_size = len(tc['args'].get('content', ''))
                    rprint(Panel(
                        f"File: {file_path}\nSize: {content_size} bytes",
                        title=f"[bold yellow]Tool Call: {tc['name']}[/bold yellow]",
                        border_style="yellow"
                    ))

        if msg.type == "tool":
            rprint(Panel(
                msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                title=f"[bold cyan]Tool Result: {msg.name}[/bold cyan]",
                border_style="cyan"
            ))

def run_phase(agent, prompt, config):
    """Execute a phase with automatic retry on JSON parsing errors."""
    
    max_retries = 10
    current_message = prompt
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                inputs = {"messages": [("user", current_message)]}
            else:
                inputs = {"messages": [("user", current_message)]}

            result = agent.invoke(inputs, config=config)
            
            rprint("\n[bold green]========== AGENT EXECUTION LOG ==========[/bold green]")
            if "messages" in result:
                for msg in result["messages"]:
                    if msg.type == "human":
                        rprint(Panel(msg.content[:300] + "..." if len(msg.content) > 300 else msg.content, 
                                   title="[bold blue]USER[/bold blue]", border_style="blue"))
                    elif msg.type == "ai":
                        if msg.content and '"name": "ResponseFormat"' not in msg.content:
                            content_preview = msg.content[:1000] + "..." if len(msg.content) > 1000 else msg.content
                            rprint(f"\n[bold magenta] AI Thought:[/bold magenta] {content_preview}")
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                file_path = tool_call['args'].get('file_path', 'unknown')
                                content_size = len(tool_call['args'].get('content', ''))
                                rprint(Panel(f"File: {file_path}\nSize: {content_size} bytes", 
                                    title=f"[bold yellow]Tool Call: {tool_call['name']}[/bold yellow]", border_style="yellow"))
                    elif msg.type == "tool":
                        rprint(Panel(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content, 
                            title=f"[bold cyan]Tool Result: {msg.name}[/bold cyan]", border_style="cyan"))
            
            return result

        except Exception as e:
            error_str = str(e)
            if "parsing tool call" in error_str or "invalid character" in error_str or "Unterminated string" in error_str:
                rprint(f"\n[bold red] JSON PARSING ERROR (Attempt {attempt+1}/{max_retries}):[/bold red] {error_str}")
                rprint("[yellow]Asking agent to fix the format...[/yellow]")
              
                current_message = (
                f"CRITICAL JSON ERROR:\n"
                f"{error_str}\n\n"
                f"RULE: The 'content' field in write_file MUST be a plain string, NOT an array.\n"
                f"CORRECT: {{\"file_path\": \"path/to/file\", \"content\": \"code here\"}}\n"
                f"WRONG: {{\"file_path\": \"...\", \"content\": [\"code\"]}}\n"
                f"Fix the syntax and retry ONCE. Do not modify code logic."
                )
            else:
                raise e
    
    raise RuntimeError("Max retries exceeded due to JSON parsing errors.")

def main():
    settings = load_settings(SETTINGS_PATH)
    model = settings.get("llm", {}).get("model_name", "gpt-oss:120b")
    reasoning = settings.get("llm", {}).get("reasoning", False)
    
    # One can set a starting phase to skip already completed phases, and optionally skip cleanup to preserve files between runs. Useful for iterative development and debugging.
    START_FROM_PHASE = settings.get("workflow", {}).get("start_from_phase", 1)
    SKIP_CLEANUP = settings.get("workflow", {}).get("skip_cleanup", False)
    
    checkpointer = None 

    if not SKIP_CLEANUP and CORE_DIR.exists():
        import shutil
        shutil.rmtree(CORE_DIR)
    
    CORE_DIR.mkdir(parents=True, exist_ok=True)
    RUST_CORE_DIR.mkdir(parents=True, exist_ok=True)
    BINDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    rprint(f"[bold cyan]Starting Workflow with Model: {model}[/bold cyan]")
    rprint(f"[bold yellow]Starting from Phase: {START_FROM_PHASE}[/bold yellow]")

    source_code = get_python_source(BASE_DIR) if START_FROM_PHASE == 1 else ""

    # --- PHASE 1: CORE GENERATION ---
    if START_FROM_PHASE <= 1:
        rprint("\n[bold yellow]--- PHASE 1: CORE GENERATION ---[/bold yellow]")
        if not source_code:
            rprint("[bold red]No Python source found[/bold red]")
            return
        
        agent1 = init_agent(model, reasoning, checkpointer)
        config = {"configurable": {"thread_id": "1"}}
        # inject the source code into the prompt for phase 1, which is crucial for the agent to understand the logic 
        # it needs to re-implement in Rust
        prompt_p1 = PHASE1_PROMPT_TEMPLATE.format(source_code=source_code)

        try:
            run_phase(agent1, prompt_p1, config)
        except ResponseError as e:
            rprint(f"[bold red]Phase 1 failed: {str(e)}[/bold red]")
            return
    else:
        rprint("[dim]Skipping Phase 1 (already completed)[/dim]")
    
    # --- PHASE 2: PYTHON BINDINGS ---
    if START_FROM_PHASE <= 2:
        rprint("\n[bold yellow]--- PHASE 2: PYTHON BINDINGS ---[/bold yellow]")
        agent2 = init_agent(model, reasoning, checkpointer)
        config = {"configurable": {"thread_id": "2"}}
        prompt_p2 = PHASE2_PROMPT

        try:
            run_phase(agent2, prompt_p2, config)
        except ResponseError as e:
            rprint(f"[bold red]Phase 2 failed: {str(e)}[/bold red]")
            return
    else:
        rprint("[dim]Skipping Phase 2 (already completed)[/dim]")

    # --- PHASE 3: PYTHON API WRAPPER ---
    if START_FROM_PHASE <= 3:
        rprint("\n[bold yellow]--- PHASE 3: PYTHON API WRAPPER ---[/bold yellow]")
        agent3 = init_agent(model, reasoning, checkpointer)
        config = {"configurable": {"thread_id": "3"}}
        prompt_p3 = PHASE3_PROMPT

        try:
            run_phase(agent3, prompt_p3, config)
        except ResponseError as e:
            rprint(f"[bold red]Phase 3 failed: {str(e)}[/bold red]")
            return
    else:
        rprint("[dim]Skipping Phase 3 (already completed)[/dim]")
    
    # --- PHASE 4: REVIEW AND TEST ---
    if START_FROM_PHASE <= 4:
        rprint("\n[bold yellow]--- PHASE 4: REVIEW AND TEST ---[/bold yellow]")
        agent4 = init_agent(model, reasoning, checkpointer)
        config = {"configurable": {"thread_id": "4"}}
        prompt_p4 = PHASE4_PROMPT

        try:
            run_phase(agent4, prompt_p4, config)
        except ResponseError as e:
            rprint(f"[bold red]Phase 4 failed: {str(e)}[/bold red]")
            return
    else:
        rprint("[dim]Skipping Phase 4 (already completed)[/dim]")

    # --- PHASE 5: DEBUGGING LOOP ---
    if START_FROM_PHASE <= 5:
        rprint("\n[bold yellow]--- PHASE 5: DEBUGGING LOOP ---[/bold yellow]")
        
        agent5 = init_agent(model, reasoning, checkpointer)
        config = {"configurable": {"thread_id": "5"}}
        MAX_RETRIES = 8
        
        initial_prompt = """
        We are entering the debugging phase. 
        I (the system) will run the tests for you. 
        If they fail, I will paste the error log here. 

        YOUR DEBUGGING STRATEGY:
        1. **For Rust test errors (E0432, E0433 "unresolved import"):**
           - The file `tests/test_core.rs` is an INTEGRATION TEST (external to crate)
           - Read the test file with `read_file` and fix the imports

        2. **For Cargo dependency errors:**
           - Use `read_file` to check `Cargo.toml`
           - Add missing dependencies with `write_file`

        3. **For logic/type errors:**
           - Read the relevant `.rs` file
           - Fix the implementation, not the test

        CRITICAL: 
        - If you see the SAME error 3 times in a row, try a DIFFERENT approach
        - After fixing imports, do NOT change them again unless the error changes

        Wait for my input.
        """
        run_phase(agent5, initial_prompt, config)

        for attempt in range(1, MAX_RETRIES + 1):
            rprint(f"\n[bold white on blue] Attempt {attempt}/{MAX_RETRIES} [/bold white on blue]")
            
            rprint("[cyan]Running: cargo test (Core)...[/cyan]")
            success, output = run_command_check("cargo test", "qibo_core")
            
            if not success:
                rprint(Panel(
                    output[:2000] + "..." if len(output) > 2000 else output,
                    title="[bold red]CARGO TEST ERROR[/bold red]",
                    border_style="red"
                ))
                error_msg = f"CORE TEST FAILED:\n{output}\n\nTASK: Analyze the error above. Read the relevant file. Fix the code."
                rprint("[bold red]Core Test Failed. Sending error to Agent...[/bold red]")
                run_phase(agent5, error_msg, config)
                continue
                
            rprint("[green]✓ Core tests passed[/green]")

            rprint("[cyan]Running: maturin develop...[/cyan]")
            success, output = run_command_check("maturin develop", "qibo_python")

            # check for errors not warnings
            has_error = "error:" in output.lower() or not success

            if has_error:
                rprint(Panel(
                    output[:2000] + "..." if len(output) > 2000 else output,
                    title="[bold red]MATURIN BUILD ERROR[/bold red]",
                    border_style="red"
                ))
                error_msg = f"BINDINGS BUILD FAILED:\n{output}\n\nTASK: Analyze the error. Is it a PyO3 type mismatch? Missing Cargo dependency? Fix it."
                rprint("[bold red]Build Failed. Sending error to Agent...[/bold red]")
                run_phase(agent5, error_msg, config)
                continue
            
            # warnings ok, show them but continue
            if "warning:" in output.lower():
                rprint(Panel(
                    output[:1000] + "..." if len(output) > 1000 else output,
                    title="[bold yellow]Build Warnings (non-fatal)[/bold yellow]",
                    border_style="yellow"
                ))

            rprint("[green]✓ Maturin build succeeded[/green]")
            rprint("[cyan]Running: pytest...[/cyan]")
            test_file = "~/core_gen/core/qibo_python/tests/test_bindings.py"

            success, output = run_command_check(
                f"PYTHONPATH=. {sys.executable} -m pytest {test_file} -v",
                "qibo_python"
            )     
            if not success:
                rprint(Panel(
                    output[:2000] + "..." if len(output) > 2000 else output,
                    title="[bold red]PYTEST ERROR[/bold red]",
                    border_style="red",
                    expand=False
                ))
                error_msg = f"PYTHON TESTS FAILED:\n{output}\n\nTASK: The code compiles, but logic is wrong. Fix the Rust implementation to pass these tests."
                rprint("[bold red]Pytest Failed. Sending error to Agent...[/bold red]")
                run_phase(agent5, error_msg, config)
                continue

            rprint("\n[bold green] SUCCESS! All tests passed. The library is ready.[/bold green]")
            break
        else:
            rprint("\n[bold red] Maximum retries reached. Debugging failed.[/bold red]")
    else:
        rprint("[dim]Skipping Phase 5 (debugging loop)[/dim]")

if __name__ == "__main__":
    main()