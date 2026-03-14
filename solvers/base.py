"""Base classes for electromagnetic solvers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np

from core.constants import Z0


class ParameterConverter:
    """Static utility methods for network parameter conversions."""

    @staticmethod
    def z_to_s(Z: np.ndarray, Z0_ref: Union[complex, np.ndarray]) -> np.ndarray:
        """
        Convert Z-parameters to S-parameters.

        Uses the generalized formula that works with complex reference impedance:
        S = (Z - Z0) @ (Z + Z0)^{-1}

        For diagonal Z0: S_ij = (Z_ij - δ_ij * Z0_i) / (Z_ij + δ_ij * Z0_i)

        Parameters
        ----------
        Z : ndarray
            Z-parameter matrix (n x n) or batch (n_freq x n x n)
        Z0_ref : complex or ndarray
            Reference impedance (scalar, diagonal array, or matrix)

        Returns
        -------
        S : ndarray
            S-parameter matrix
        """
        # Handle batch dimension
        if Z.ndim == 3:
            n_freq = Z.shape[0]
            S = np.zeros_like(Z, dtype=complex)
            for k in range(n_freq):
                Z0_k = Z0_ref[k] if isinstance(Z0_ref, np.ndarray) and Z0_ref.ndim == 3 else Z0_ref
                S[k] = ParameterConverter.z_to_s(Z[k], Z0_k)
            return S

        n = Z.shape[0]

        # Normalize Z0 to matrix form
        if np.isscalar(Z0_ref):
            Z0_mat = Z0_ref * np.eye(n, dtype=complex)
        elif Z0_ref.ndim == 1:
            Z0_mat = np.diag(Z0_ref.astype(complex))
        else:
            Z0_mat = Z0_ref.astype(complex)

        # Generalized S-parameter formula: S = (Z - Z0)(Z + Z0)^{-1}
        # This works for both real and complex reference impedance
        Zn = Z - Z0_mat
        Zd = Z + Z0_mat

        try:
            S = Zn @ np.linalg.inv(Zd)
        except np.linalg.LinAlgError:
            S = Zn @ np.linalg.pinv(Zd)

        return S

    @staticmethod
    def s_to_z(S: np.ndarray, Z0_ref: Union[complex, np.ndarray]) -> np.ndarray:
        """
        Convert S-parameters to Z-parameters.

        Uses the generalized formula: Z = Z0 @ (I + S) @ (I - S)^{-1}
        For scalar Z0: Z = Z0 * (I + S)(I - S)^{-1}

        Parameters
        ----------
        S : ndarray
            S-parameter matrix (n x n) or batch (n_freq x n x n)
        Z0_ref : complex or ndarray
            Reference impedance

        Returns
        -------
        Z : ndarray
            Z-parameter matrix
        """
        # Handle batch dimension
        if S.ndim == 3:
            n_freq = S.shape[0]
            Z = np.zeros_like(S, dtype=complex)
            for k in range(n_freq):
                Z0_k = Z0_ref[k] if isinstance(Z0_ref, np.ndarray) and Z0_ref.ndim == 3 else Z0_ref
                Z[k] = ParameterConverter.s_to_z(S[k], Z0_k)
            return Z

        n = S.shape[0]
        I = np.eye(n, dtype=complex)

        # Normalize Z0 to matrix form
        if np.isscalar(Z0_ref):
            Z0_mat = Z0_ref * np.eye(n, dtype=complex)
        elif Z0_ref.ndim == 1:
            Z0_mat = np.diag(Z0_ref.astype(complex))
        else:
            Z0_mat = Z0_ref.astype(complex)

        # Z = Z0 @ (I + S) @ (I - S)^{-1}
        try:
            Z = Z0_mat @ (I + S) @ np.linalg.inv(I - S)
        except np.linalg.LinAlgError:
            Z = Z0_mat @ (I + S) @ np.linalg.pinv(I - S)

        return Z

    @staticmethod
    def cascade_s_matrices(S1: np.ndarray, S2: np.ndarray) -> np.ndarray:
        """
        Cascade two 2-port S-matrices (generalized for any number of modes).

        Parameters
        ----------
        S1 : ndarray
            First 2-port S-matrix (size 2n x 2n)
        S2 : ndarray
            Second 2-port S-matrix (size 2m x 2m, usually n=m)

        Returns
        -------
        S : ndarray
            Cascaded S-matrix (size 2n x 2n)
        """
        # Dimensions: 2n x 2n. Split into n x n blocks.
        n1 = S1.shape[0] // 2
        n2 = S2.shape[0] // 2

        S11_1 = S1[:n1, :n1]
        S12_1 = S1[:n1, n1:]
        S21_1 = S1[n1:, :n1]
        S22_1 = S1[n1:, n1:]

        S11_2 = S2[:n2, :n2]
        S12_2 = S2[:n2, n2:]
        S21_2 = S2[n2:, :n2]
        S22_2 = S2[n2:, n2:]

        I1 = np.eye(n1, dtype=complex)
        I2 = np.eye(n2, dtype=complex)

        # Generalized cascade formulas
        # S11 = S11_1 + S12_1 @ inv(I - S11_2 @ S22_1) @ S11_2 @ S21_1
        # S12 = S12_1 @ inv(I - S11_2 @ S22_1) @ S12_2
        # S21 = S21_2 @ inv(I - S22_1 @ S11_2) @ S21_1
        # S22 = S22_2 + S21_2 @ inv(I - S22_1 @ S11_2) @ S22_1 @ S12_2

        # Standard matrix cascade (T-matrix approach or direct block)
        # We use the direct block inversion form:
        inv1 = np.linalg.inv(I2 - S11_2 @ S22_1)
        inv2 = np.linalg.inv(I1 - S22_1 @ S11_2)

        S11_c = S11_1 + S12_1 @ inv1 @ S11_2 @ S21_1
        S12_c = S12_1 @ inv1 @ S12_2
        S21_c = S21_2 @ inv2 @ S21_1
        S22_c = S22_2 + S21_2 @ inv2 @ S22_1 @ S12_2

        # Combine into 2x2 block matrix
        S_top = np.hstack([S11_c, S12_c])
        S_bot = np.hstack([S21_c, S22_c])
        return np.vstack([S_top, S_bot])


class BaseEMSolver(ABC):
    """
    Abstract base class for electromagnetic solvers.

    Provides unified interface for:
    - Z/S parameter access (matrix and dict forms)
    - Eigenvalue/resonant frequency computation
    - Plotting and comparison

    Subclasses must implement:
    - n_ports: property returning number of ports
    - ports: property returning list of port names
    - solve(): method to perform frequency sweep
    - get_eigenvalues(): method to compute system eigenvalues
    - _get_port_impedance(): method to get reference impedance for a port
    """

    def __init__(self):
        # State (populated by subclasses after solve)
        self._n_modes_per_port = None
        self.frequencies: np.ndarray = None
        self._Z_matrix: np.ndarray = None  # (n_freq, n_ports, n_ports)
        self._S_matrix: np.ndarray = None

        # Cached dicts (built lazily)
        self._Z_dict: Dict[str, np.ndarray] = None
        self._S_dict: Dict[str, np.ndarray] = None

    # === Abstract properties and methods ===

    @property
    @abstractmethod
    def n_ports(self) -> int:
        """Number of ports in the system."""
        pass

    @property
    @abstractmethod
    def ports(self) -> List[str]:
        """List of port names."""
        pass

    @abstractmethod
    def solve(
            self,
            fmin: float,
            fmax: float,
            nsamples: int = 100,
            **kwargs
    ) -> Dict:
        """
        Solve over frequency range.

        Parameters
        ----------
        fmin : float
            Minimum frequency in GHz
        fmax : float
            Maximum frequency in GHz
        nsamples : int
            Number of frequency samples
        **kwargs
            Subclass-specific options

        Returns
        -------
        dict
            Results dictionary with at least:
            - 'frequencies': frequency array in Hz
            - 'Z': Z-parameter matrix array
            - 'S': S-parameter matrix array
        """
        pass

    @abstractmethod
    def calculate_resonant_modes(self) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Compute eigenvalues and eigenvectors of system matrix.

        Returns
        -------
        modes : tuple or dict
            Tuple of (eigenvalues, eigenvectors).
            Dict mapping domain names to (eigenvalues, eigenvectors) for multi-domain systems.
        """
        raise NotImplementedError

    def get_eigenvalues(self, **kwargs):
        """Deprecated alias for calculate_resonant_modes."""
        import warnings
        warnings.warn("get_eigenvalues() is deprecated. Use calculate_resonant_modes() instead.",
                      DeprecationWarning, stacklevel=2)
        res = self.calculate_resonant_modes(**kwargs)
        if isinstance(res, dict):
            return {k: v[0] for k, v in res.items()}
        return res[0]

    @abstractmethod
    def _get_port_impedance(self, port: str, mode: int, freq: float) -> complex:
        """
        Get reference impedance for a port at given frequency.

        Parameters
        ----------
        port : str
            Port name
        mode : int
            Mode index
        freq : float
            Frequency in Hz

        Returns
        -------
        Z0 : complex
            Reference impedance
        """
        pass

    # === Concrete implementations ===

    def _get_impedance_matrix(self, freq: float) -> np.ndarray:
        """
        Get diagonal reference impedance matrix at given frequency.

        Subclasses with non-standard port indexing (like ConcatenatedSystem)
        should override this method.
        """
        if self._Z_matrix is None:
            raise ValueError("No Z-matrix available. Call solve() first.")

        n_total = self._Z_matrix.shape[1]
        n_modes = self._n_modes_per_port if self._n_modes_per_port is not None else 1
        n_ports = len(self.ports)

        Z0_diag = []

        for idx in range(n_total):
            port_idx = idx // n_modes
            mode_idx = idx % n_modes

            if port_idx < n_ports:
                port_name = self.ports[port_idx]
                Zw = self._get_port_impedance(port_name, mode_idx, freq)
            else:
                # Fallback for edge cases
                from core.constants import Z0
                Zw = Z0

            Z0_diag.append(Zw)

        return np.diag(Z0_diag)

    def _invalidate_cache(self) -> None:
        """Invalidate cached dictionaries (call when matrices change)."""
        self._Z_dict = None
        self._S_dict = None

    def _build_dicts(self) -> None:
        """
        Build Z/S parameter dictionaries from matrices.

        Keys are formatted as 'port_i(mode_i)port_j(mode_j)' where:
        - First part is excitation (column index)
        - Second part is measurement (row index)
        - All indices are 1-based

        Works directly from matrix dimensions, so compatible with all solvers.
        """
        if self._Z_matrix is None:
            return

        n_total = self._Z_matrix.shape[1]  # Matrix dimension
        n_modes = self._n_modes_per_port if self._n_modes_per_port is not None else 1

        self._Z_dict = {'frequencies': self.frequencies}
        self._S_dict = {'frequencies': self.frequencies} if self._S_matrix is not None else None

        for row in range(n_total):
            port_row = row // n_modes + 1
            mode_row = row % n_modes + 1

            for col in range(n_total):
                port_col = col // n_modes + 1
                mode_col = col % n_modes + 1

                key = f'{port_col}({mode_col}){port_row}({mode_row})'
                self._Z_dict[key] = self._Z_matrix[:, row, col]
                if self._S_matrix is not None:
                    self._S_dict[key] = self._S_matrix[:, row, col]

    def _compute_s_from_z(self) -> None:
        """Compute S-parameters from Z-parameters using port impedances."""
        if self._Z_matrix is None:
            return

        n_freq = len(self.frequencies)
        n_ports = self._Z_matrix.shape[1]
        self._S_matrix = np.zeros((n_freq, n_ports, n_ports), dtype=complex)

        for k, freq in enumerate(self.frequencies):
            Z0_mat = self._get_impedance_matrix(freq)
            self._S_matrix[k] = ParameterConverter.z_to_s(self._Z_matrix[k], Z0_mat)

        self._invalidate_cache()

    @property
    def Z_dict(self) -> Dict[str, np.ndarray]:
        """Z-parameters in dictionary format with keys like '1(1)1(1)'."""
        if self._Z_dict is None:
            self._build_dicts()
        return self._Z_dict

    @property
    def S_dict(self) -> Dict[str, np.ndarray]:
        """S-parameters in dictionary format with keys like '1(1)1(1)'."""
        if self._S_dict is None:
            self._build_dicts()
        return self._S_dict

    def get_z_matrix(self, freq_idx: int) -> np.ndarray:
        """
        Get Z-matrix at given frequency index.

        Parameters
        ----------
        freq_idx : int
            Frequency index

        Returns
        -------
        Z : ndarray
            Returns Z-matrix at given frequency index.
            Shape: (n_total_modes, n_total_modes)
        """
        if self._Z_matrix is None:
            raise ValueError("No solution available. Call solve() first.")
        if freq_idx < 0 or freq_idx >= len(self.frequencies):
            raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(self.frequencies) - 1}]")
        return self._Z_matrix[freq_idx]

    def get_s_matrix(self, freq_idx: int) -> np.ndarray:
        """
        Get S-matrix at given frequency index.

        Parameters
        ----------
        freq_idx : int
            Frequency index

        Returns
        -------
        S : ndarray
            Returns S-matrix at given frequency index.
            Shape: (n_total_modes, n_total_modes)
        """
        if self._S_matrix is None:
            raise ValueError("No S-parameters available. Call solve() first.")
        if freq_idx < 0 or freq_idx >= len(self.frequencies):
            raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(self.frequencies) - 1}]")
        return self._S_matrix[freq_idx]

    def get_s_db(self, i: int, j: int) -> np.ndarray:
        """
        Get S-parameter S_{ij} magnitude in dB.

        Parameters
        ----------
        i, j : int
            Port indices (1-based)

        Returns
        -------
        s_db : ndarray
            S-parameter magnitude in dB vs frequency
        """
        if self._S_matrix is None:
            raise ValueError("No S-parameters available. Call solve() first.")
        return 20 * np.log10(np.abs(self._S_matrix[:, i - 1, j - 1]) + 1e-15)

    def get_s_phase(self, i: int, j: int, degrees: bool = True) -> np.ndarray:
        """
        Get S-parameter S_{ij} phase.

        Parameters
        ----------
        i, j : int
            Port indices (1-based)
        degrees : bool
            Return phase in degrees (default) or radians

        Returns
        -------
        phase : ndarray
            S-parameter phase vs frequency
        """
        if self._S_matrix is None:
            raise ValueError("No S-parameters available. Call solve() first.")
        phase = np.angle(self._S_matrix[:, i - 1, j - 1])
        return np.degrees(phase) if degrees else phase

    def get_z_db(self, i: int, j: int) -> np.ndarray:
        """
        Get Z-parameter Z_{ij} magnitude in dB.

        Parameters
        ----------
        i, j : int
            Port indices (1-based)

        Returns
        -------
        z_db : ndarray
            Z-parameter magnitude in dB vs frequency
        """
        if self._Z_matrix is None:
            raise ValueError("No solution available. Call solve() first.")
        return 20 * np.log10(np.abs(self._Z_matrix[:, i - 1, j - 1]) + 1e-15)

    def get_z_phase(self, i: int, j: int, degrees: bool = True) -> np.ndarray:
        """
        Get Z-parameter Z_{ij} phase.

        Parameters
        ----------
        i, j : int
            Port indices (1-based)
        degrees : bool
            Return phase in degrees (default) or radians

        Returns
        -------
        phase : ndarray
            Z-parameter phase vs frequency
        """
        if self._Z_matrix is None:
            raise ValueError("No solution available. Call solve() first.")
        phase = np.angle(self._Z_matrix[:, i - 1, j - 1])
        return np.degrees(phase) if degrees else phase

    def get_param(self, param_type: str, key: str) -> np.ndarray:
        """
        Get parameter by string key.

        Parameters
        ----------
        param_type : str
            'Z' or 'S'
        key : str
            Parameter key (e.g., '1(1)1(1)', '1(1)2(1)')

        Returns
        -------
        param : ndarray
            Complex parameter vs frequency
        """
        param_type = param_type.upper()

        if param_type == 'Z':
            data = self.Z_dict
            if key not in data:
                available = [k for k in data.keys() if k != 'frequencies']
                raise KeyError(f"Z-parameter '{key}' not found. Available: {available}")
            return data[key]
        elif param_type == 'S':
            data = self.S_dict
            if data is None:
                raise ValueError("No S-parameters available.")
            if key not in data:
                available = [k for k in data.keys() if k != 'frequencies']
                raise KeyError(f"S-parameter '{key}' not found. Available: {available}")
            return data[key]
        else:
            raise ValueError(f"Unknown parameter type: {param_type}. Use 'Z' or 'S'.")

    def get_resonant_frequencies(self, return_omega: bool = False) -> np.ndarray:
        """
        Compute resonant frequencies from eigenvalues.

        Parameters
        ----------
        return_omega : bool
            If True, return angular frequencies. If False, return frequencies in Hz.

        Returns
        -------
        freqs : ndarray
            Resonant frequencies in Hz (or rad/s if return_omega=True)
        """
        eigs = self.get_eigenvalues()

        # Handle dict return from multi-domain solvers
        if isinstance(eigs, dict):
            all_eigs = np.concatenate(list(eigs.values()))
        else:
            all_eigs = eigs

        # Filter positive eigenvalues and compute frequencies
        # Eigenvalues are ω² for the system (A - ω²I)x = ωBu
        eigs_pos = all_eigs[all_eigs > 0]
        omega = np.sqrt(eigs_pos)

        if return_omega:
            return np.sort(omega)
        return np.sort(omega / (2 * np.pi))

    def plot_s_parameters(
            self,
            params: Optional[List[str]] = None,
            plot_type: str = 'db',
            fig_ax: Optional[Tuple] = None,
            title: Optional[str] = None,
            **kwargs
    ):
        """
        Plot S-parameters vs frequency.

        Parameters
        ----------
        params : list of str, optional
            S-parameter keys to plot (e.g., ['1(1)1(1)', '1(1)2(1)']).
            If None, plots all.
        plot_type : str
            'db': magnitude in dB
            'mag': linear magnitude
            'phase': phase in degrees
        fig_ax : tuple, optional
            (fig, ax) tuple for existing figure
        title : str, optional
            Custom title for the plot
        **kwargs
            Additional arguments passed to ax.plot()

        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        if self._S_matrix is None:
            raise ValueError("No S-parameters available. Call solve() first.")

        figsize = kwargs.pop('figsize', (10, 6))
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = fig_ax

        freq_ghz = self.frequencies / 1e9

        if params is None:
            params = [k for k in self.S_dict.keys() if k != 'frequencies']

        for param in params:
            s_data = self.get_param('S', param)

            if plot_type == 'db':
                y_data = 20 * np.log10(np.abs(s_data) + 1e-15)
                ylabel = '|S| (dB)'
            elif plot_type == 'mag':
                y_data = np.abs(s_data)
                ylabel = '|S|'
            elif plot_type == 'phase':
                y_data = np.degrees(np.angle(s_data))
                ylabel = 'Phase (degrees)'
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

            label = f'S{param}'
            ax.plot(freq_ghz, y_data, label=label, **kwargs)

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title or f'{self.__class__.__name__} S-Parameters')

        plt.tight_layout()
        return fig, ax

    def plot_z_parameters(
            self,
            params: Optional[List[str]] = None,
            plot_type: str = 'db',
            fig_ax: Optional[Tuple] = None,
            title: Optional[str] = None,
            **kwargs
    ):
        """
        Plot Z-parameters vs frequency.

        Parameters
        ----------
        params : list of str, optional
            Z-parameter keys to plot. If None, plots all.
        plot_type : str
            'db': magnitude in dB
            'mag': linear magnitude
            'phase': phase in degrees
        fig_ax : tuple, optional
            (fig, ax) tuple for existing figure
        title : str, optional
            Custom title
        **kwargs
            Additional arguments passed to ax.plot()

        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        if self._Z_matrix is None:
            raise ValueError("No solution available. Call solve() first.")

        figsize = kwargs.pop('figsize', (10, 6))
        if fig_ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig, ax = fig_ax

        freq_ghz = self.frequencies / 1e9

        if params is None:
            params = [k for k in self.Z_dict.keys() if k != 'frequencies']

        for param in params:
            z_data = self.get_param('Z', param)

            if plot_type == 'db':
                y_data = 20 * np.log10(np.abs(z_data) + 1e-15)
                ylabel = '|Z| (dB)'
            elif plot_type == 'mag':
                y_data = np.abs(z_data)
                ylabel = '|Z|'
            elif plot_type == 'phase':
                y_data = np.degrees(np.angle(z_data))
                ylabel = 'Phase (degrees)'
            else:
                raise ValueError(f"Unknown plot_type: {plot_type}")

            label = f'Z{param}'
            ax.plot(freq_ghz, y_data, label=label, **kwargs)

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title or f'{self.__class__.__name__} Z-Parameters')

        plt.tight_layout()
        return fig, ax

    def compare_with(
            self,
            other: 'BaseEMSolver',
            params: Optional[List[str]] = None,
            plot_type: str = 'db',
            labels: Optional[Tuple[str, str]] = None,
            title: Optional[str] = None,
            **kwargs
    ):
        """
        Compare S-parameters with another solver.

        Parameters
        ----------
        other : BaseEMSolver
            Another solver to compare with
        params : list of str, optional
            Parameters to compare. If None, compares all common parameters.
        plot_type : str
            'db', 'mag', or 'phase'
        labels : tuple of str, optional
            Labels for the two solvers. Defaults to class names.
        title : str, optional
            Custom title
        **kwargs
            Additional arguments passed to plotting

        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        if labels is None:
            labels = (self.__class__.__name__, other.__class__.__name__)

        if params is None:
            # Find common parameters
            self_params = set(k for k in self.S_dict.keys() if k != 'frequencies')
            other_params = set(k for k in other.S_dict.keys() if k != 'frequencies')
            params = sorted(self_params & other_params)
            if not params:
                # Default to diagonal and off-diagonal
                params = ['1(1)1(1)']
                if self.n_ports > 1 and other.n_ports > 1:
                    params.append('1(1)2(1)')

        figsize = kwargs.pop('figsize', (10, 6))
        fig, ax = plt.subplots(figsize=figsize)

        freq_ghz_self = self.frequencies / 1e9
        freq_ghz_other = other.frequencies / 1e9

        colors = plt.cm.tab10.colors

        for i, param in enumerate(params):
            color = colors[i % len(colors)]

            # Plot self
            s_self = self.get_param('S', param)
            if plot_type == 'db':
                y_self = 20 * np.log10(np.abs(s_self) + 1e-15)
                ylabel = '|S| (dB)'
            elif plot_type == 'mag':
                y_self = np.abs(s_self)
                ylabel = '|S|'
            else:
                y_self = np.degrees(np.angle(s_self))
                ylabel = 'Phase (degrees)'

            ax.plot(freq_ghz_self, y_self, '-', color=color,
                    label=f'{labels[0]} S{param}', **kwargs)

            # Plot other
            try:
                s_other = other.get_param('S', param)
                if plot_type == 'db':
                    y_other = 20 * np.log10(np.abs(s_other) + 1e-15)
                elif plot_type == 'mag':
                    y_other = np.abs(s_other)
                else:
                    y_other = np.degrees(np.angle(s_other))

                ax.plot(freq_ghz_other, y_other, '--', color=color,
                        label=f'{labels[1]} S{param}', **kwargs)
            except KeyError:
                pass  # Parameter not available in other solver

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title or 'S-Parameter Comparison')

        plt.tight_layout()
        return fig, ax

    def compute_error(
            self,
            reference: 'BaseEMSolver',
            metric: str = 'max_rel',
            param_type: str = 'S'
    ) -> Dict[str, float]:
        """
        Compute error relative to a reference solver.

        Parameters
        ----------
        reference : BaseEMSolver
            Reference solver (e.g., FOM)
        metric : str
            'max_rel': Maximum relative error
            'rms_rel': RMS relative error
            'max_abs': Maximum absolute error
        param_type : str
            'S' or 'Z'

        Returns
        -------
        errors : dict
            Dictionary with error for each parameter
        """
        if param_type == 'S':
            self_data = self._S_matrix
            ref_data = reference._S_matrix
        else:
            self_data = self._Z_matrix
            ref_data = reference._Z_matrix

        if self_data is None or ref_data is None:
            raise ValueError("Both solvers must have computed results.")

        # Interpolate if frequency grids differ
        if not np.allclose(self.frequencies, reference.frequencies):
            from scipy.interpolate import interp1d
            # Interpolate self to reference frequencies
            interp_real = interp1d(self.frequencies, np.real(self_data), axis=0,
                                   bounds_error=False, fill_value='extrapolate')
            interp_imag = interp1d(self.frequencies, np.imag(self_data), axis=0,
                                   bounds_error=False, fill_value='extrapolate')
            self_data = interp_real(reference.frequencies) + 1j * interp_imag(reference.frequencies)

        errors = {}
        n_ports = min(self_data.shape[1], ref_data.shape[1])

        for i in range(n_ports):
            for j in range(n_ports):
                key = f'{i + 1}(1){j + 1}(1)'

                self_param = self_data[:, i, j]
                ref_param = ref_data[:, i, j]

                if metric == 'max_rel':
                    denom = np.abs(ref_param)
                    denom = np.where(denom < 1e-15, 1e-15, denom)
                    error = np.max(np.abs(self_param - ref_param) / denom)
                elif metric == 'rms_rel':
                    denom = np.abs(ref_param)
                    denom = np.where(denom < 1e-15, 1e-15, denom)
                    error = np.sqrt(np.mean((np.abs(self_param - ref_param) / denom) ** 2))
                elif metric == 'max_abs':
                    error = np.max(np.abs(self_param - ref_param))
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                errors[key] = error

            return errors

    def print_info(self) -> None:
        """Print solver information."""
        print("\n" + "=" * 60)
        print(f"{self.__class__.__name__} Information")
        print("=" * 60)
        print(f"Number of ports: {self.n_ports}")
        print(f"Port names: {self.ports}")
        print(f"Solution available: {self.frequencies is not None}")

        if self.frequencies is not None:
            print(f"Frequency range: {self.frequencies[0] / 1e9:.4f} - {self.frequencies[-1] / 1e9:.4f} GHz")
            print(f"Number of samples: {len(self.frequencies)}")

        print("=" * 60)