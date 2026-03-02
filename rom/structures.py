"""Data structures for reduced-order models."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class ReducedStructure:
    """
    Container for reduced-order model data needed for concatenation.

    Parameters
    ----------
    Ard : ndarray
        Reduced system matrix (r x r)
    Brd : ndarray
        Reduced port basis matrix (r x n_ports * n_modes_per_port)
    ports : list of str
        Port names for this structure
    port_modes : dict
        Port modes dict: {port_name: {mode_idx: mode_data}}
    domain : str
        Domain name
    r : int, optional
        Reduced dimension
    n_full : int, optional
        Original full dimension
    """
    Ard: np.ndarray
    Brd: np.ndarray
    ports: List[str]
    port_modes: Dict[str, Dict[int, Any]]
    domain: str = 'default'
    r: int = None
    n_full: int = None

    def __post_init__(self):
        if self.r is None:
            self.r = self.Ard.shape[0]

        # Compute n_modes_per_port from port_modes dict
        self._n_port_modes = 1  # default
        if self.port_modes and self.ports:
            for port in self.ports:
                if port in self.port_modes:
                    self._n_port_modes = len(self.port_modes[port])
                    break

        # Validate shapes
        if self.Ard.shape[0] != self.Ard.shape[1]:
            raise ValueError(f"Ard must be square, got shape {self.Ard.shape}")
        if self.Brd.shape[0] != self.r:
            raise ValueError(f"Brd rows ({self.Brd.shape[0]}) must match r ({self.r})")

        expected_cols = len(self.ports) * self._n_port_modes
        if self.Brd.shape[1] != expected_cols:
            raise ValueError(
                f"Brd columns ({self.Brd.shape[1]}) must match "
                f"n_ports × n_modes = {len(self.ports)} × {self._n_port_modes} = {expected_cols}"
            )

    @property
    def n_ports(self) -> int:
        """Number of ports."""
        return len(self.ports)

    @property
    def n_port_modes(self) -> int:
        """Number of modes per port."""
        return self._n_port_modes

    @property
    def compression_ratio(self) -> Optional[float]:
        """Compression ratio if n_full is known."""
        if self.n_full is not None and self.n_full > 0:
            return 1 - self.r / self.n_full
        return None

    def get_port_index(self, port_name: str) -> int:
        """Get index of a port by name."""
        try:
            return self.ports.index(port_name)
        except ValueError:
            raise KeyError(f"Port '{port_name}' not found. Available: {self.ports}")

    def get_port_mode_column(self, port_name: str, mode: int = 0) -> int:
        """Get column index in Brd for a specific port-mode combination."""
        port_idx = self.get_port_index(port_name)
        return port_idx * self._n_port_modes + mode

    def copy(self) -> 'ReducedStructure':
        """Create a deep copy."""
        return ReducedStructure(
            Ard=self.Ard.copy(),
            Brd=self.Brd.copy(),
            ports=self.ports.copy(),
            port_modes={p: dict(m) for p, m in self.port_modes.items()},
            domain=self.domain,
            r=self.r,
            n_full=self.n_full
        )

    def __repr__(self) -> str:
        compression = f", compression={100 * self.compression_ratio:.1f}%" if self.compression_ratio else ""
        return (f"ReducedStructure(domain='{self.domain}', r={self.r}, "
                f"ports={self.ports}, modes/port={self._n_port_modes}{compression})")