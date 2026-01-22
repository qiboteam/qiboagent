"""Python wrapper for quantum circuits.

This module provides a :class:`Circuit` class that inherits from the Rust
implementation exposed via the ``_qibo_core`` extension.  It adds a few
properties required by the high‑level Qibo API but otherwise delegates all
functionality to the underlying Rust object.
"""

from _qibo_core import Circuit as _RustCircuit


class Circuit(_RustCircuit):
    """High‑level circuit class compatible with the original Qibo API.

    The heavy lifting (gate storage, validation, etc.) is performed by the
    Rust backend.  This subclass only adds convenience properties that the
    Python‑level API expects.
    """

    # ---------------------------------------------------------------------
    # Compatibility properties – these are read‑only and return default values
    # matching the behaviour of Qibo's reference implementation when no
    # advanced features are used.
    # ---------------------------------------------------------------------
    @property
    def repeated_execution(self) -> bool:
        """Whether the circuit is executed repeatedly.

        The reference Qibo implementation returns ``False`` unless the user
        explicitly enables repeated execution.  For the minimal wrapper we
        always return ``False``.
        """
        return False

    @property
    def accelerators(self):
        """Accelerator information (e.g., GPU devices).

        Not used in this minimal wrapper – return ``None``.
        """
        return None

    @property
    def density_matrix(self) -> bool:
        """Indicates if the circuit works with density matrices.

        The default behaviour is to work with state vectors, so ``False``.
        """
        return False

    @property
    def measurements(self) -> list:
        """List of measurement operations in the circuit.

        This minimal wrapper does not implement measurement gates, therefore an
        empty list is returned.
        """
        return []

    @property
    def has_collapse(self) -> bool:
        """Whether the circuit contains collapse (measurement) operations.

        Always ``False`` for the basic wrapper.
        """
        return False

    @property
    def has_unitary_channel(self) -> bool:
        """Whether the circuit contains unitary channel operations.

        Always ``False`` for the basic wrapper.
        """
        return False

    # Additional helper methods could be added here if needed for higher‑level
    # functionality, but the core API (add, nqubits, queue, etc.) is provided by
    # the Rust base class.
