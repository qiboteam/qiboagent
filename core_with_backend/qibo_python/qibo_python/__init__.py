"""Top‑level package for the Python Qibo wrapper.

Exports the high‑level :class:`Circuit` class and the :mod:`gates` module so
that users can write ``from qibo import Circuit, gates`` just like with the
original library.
"""

from .circuit import Circuit
from . import gates

__all__ = ["Circuit", "gates"]
