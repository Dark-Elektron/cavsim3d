"""Data structures for reduced-order models."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ngsolve import HCurl, Mesh


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
    W : ndarray, optional
        POD basis matrix (n_full x r) for field reconstruction
    Q_L_inv : ndarray, optional
        Mass transformation matrix (r x r) for field reconstruction
    fes : HCurl, optional
        Finite element space reference for field reconstruction
    mesh : Mesh, optional
        Mesh reference
    """
    Ard: np.ndarray
    Brd: np.ndarray
    ports: List[str]
    port_modes: Dict[str, Dict[int, Any]]
    domain: str = 'default'
    r: int = None
    n_full: int = None
    is_full_order: bool = False  # True when W=I (FOM wrapped for concatenation)
    
    # Field reconstruction data
    W: Optional[np.ndarray] = None
    Q_L_inv: Optional[np.ndarray] = None
    
    # FEM references
    fes: Optional[Any] = None
    mesh: Optional[Any] = None

    def __post_init__(self):
        if self.r is None:
            self.r = self.Ard.shape[0]

        # Compute modes-per-port from port_modes dict (first port = back-compat
        # scalar; per-port counts may differ — see port_mode_pairs).
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

        expected_cols = len(self.port_mode_pairs)
        if self.Brd.shape[1] != expected_cols:
            raise ValueError(
                f"Brd columns ({self.Brd.shape[1]}) must match the total number "
                f"of port-modes ({expected_cols}) for ports={self.ports}"
            )

    @property
    def n_ports(self) -> int:
        """Number of ports."""
        return len(self.ports)

    @property
    def n_port_modes(self) -> int:
        """Modes on the first port (back-compat scalar; use port_mode_pairs
        for the per-port breakdown when ports have different mode counts)."""
        return self._n_port_modes

    @property
    def port_mode_pairs(self) -> List[Tuple[str, int]]:
        """Ordered ``(port_name, mode_index)`` list aligned with Brd columns.

        Supports a different number of modes per port.  Falls back to a
        uniform layout (``n_ports × n_port_modes``) if per-port mode data is
        unavailable.
        """
        if self.port_modes and self.ports:
            pairs = []
            for p in self.ports:
                if p in self.port_modes:
                    for m in sorted(self.port_modes[p].keys()):
                        pairs.append((p, m))
            if pairs:
                return pairs
        # Uniform fallback.
        return [(p, m) for p in self.ports for m in range(self._n_port_modes)]

    @property
    def n_total_port_modes(self) -> int:
        """Total number of port-modes (sum over ports)."""
        return len(self.port_mode_pairs)

    @property
    def compression_ratio(self) -> Optional[float]:
        """Compression ratio if n_full is known."""
        if self.is_full_order:
            return 0.0
        if self.n_full is not None and self.n_full > 0:
            return 1 - self.r / self.n_full
        return None

    def can_reconstruct(self) -> bool:
        """Check if field reconstruction is possible."""
        if self.is_full_order:
            return True
        return self.W is not None and self.Q_L_inv is not None

    def reconstruct(self, x_r: np.ndarray) -> np.ndarray:
        """Reconstruct full-order solution from reduced coordinates."""
        if self.is_full_order:
            return x_r
        if not self.can_reconstruct():
            raise ValueError(f"Cannot reconstruct: W or Q_L_inv missing for '{self.domain}'")
        return self.W @ (self.Q_L_inv @ x_r)

    def get_port_index(self, port_name: str) -> int:
        """Get index of a port by name."""
        try:
            return self.ports.index(port_name)
        except ValueError:
            raise KeyError(f"Port '{port_name}' not found. Available: {self.ports}")

    def get_port_mode_column(self, port_name: str, mode: int = 0) -> int:
        """Get column index in Brd for a specific port-mode combination."""
        try:
            return self.port_mode_pairs.index((port_name, mode))
        except ValueError:
            # Uniform fallback.
            return self.get_port_index(port_name) * self._n_port_modes + mode

    def copy(self) -> 'ReducedStructure':
        """Create a deep copy."""
        return ReducedStructure(
            Ard=self.Ard.copy(),
            Brd=self.Brd.copy(),
            ports=self.ports.copy(),
            port_modes={p: dict(m) for p, m in self.port_modes.items()},
            domain=self.domain,
            r=self.r,
            n_full=self.n_full,
            is_full_order=self.is_full_order,
            W=self.W.copy() if self.W is not None else None,
            Q_L_inv=self.Q_L_inv.copy() if self.Q_L_inv is not None else None,
            fes=self.fes,
            mesh=self.mesh,
        )

    def __repr__(self) -> str:
        recon_str = ", can_reconstruct=True" if self.can_reconstruct() else ""
        if self.is_full_order:
            return (f"ReducedStructure(domain='{self.domain}', r={self.r}, "
                    f"ports={self.ports}, modes/port={self._n_port_modes}, "
                    f"full-order W=I{recon_str})")
        compression = f", compression={100 * self.compression_ratio:.1f}%" if self.compression_ratio else ""
        return (f"ReducedStructure(domain='{self.domain}', r={self.r}, "
                f"ports={self.ports}, modes/port={self._n_port_modes}{compression}{recon_str})")