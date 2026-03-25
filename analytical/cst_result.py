"""CSTResult — Load and compare CST Studio S-parameter exports."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import re
import glob
from utils.plot_mixin import PlotMixin


class CSTResult(PlotMixin):
    """
    Load S-parameter results exported from CST Studio Suite.

    Expects an Export folder containing files named:
        S-Parameters_S{i}({m}),{j}({n}).txt

    where i,j are port indices and m,n are mode indices.

    Each file contains whitespace-separated columns:
        frequency[GHz]  magnitude[linear]  phase[degrees]

    Parameters
    ----------
    project_folder : str or Path
        Path to the CST project folder (containing the Export subfolder).
    z0 : float
        Reference impedance for S-to-Z conversion (default 50 Ω).

    Attributes
    ----------
    frequencies : np.ndarray
        Frequency points in Hz.
    S_dict : dict
        S-parameters keyed as '{i}({m}){j}({n})' → complex array.
    Z_dict : dict
        Z-parameters computed from S (same key format).

    Examples
    --------
    >>> cst = CSTResult("/path/to/cst_project")
    >>> fig, ax = cst.plot_s(params=['1(1)1(1)', '2(1)1(1)'])
    >>> # Overlay with numerical solver
    >>> fig, ax = fds.fom.plot_s(ax=ax, label='FOM')
    """

    _FILENAME_PATTERN_S = re.compile(
        r"(?:S-Parameters_S|S Matrix_S)(\d+)\((\d+)\),(\d+)\((\d+)\)\.txt$"
    )
    _FILENAME_PATTERN_Z = re.compile(
        r"(?:Z-Parameters_Z|Z Matrix_Z)(\d+)\((\d+)\),(\d+)\((\d+)\)\.txt$"
    )
    _FILENAME_PATTERN_Y = re.compile(
        r"(?:Y-Parameters_Y|Y Matrix_Y)(\d+)\((\d+)\),(\d+)\((\d+)\)\.txt$"
    )

    def __init__(
        self,
        project_folder: Union[str, Path],
        z0: float = 50.0,
    ):
        self.project_folder = Path(project_folder)
        self.z0 = z0

        # Validate folder structure
        self._export_folder = self._validate_folder()

        # Storage
        self._frequencies: Optional[np.ndarray] = None
        self._S_dict: Dict[str, np.ndarray] = {}
        self._Z_dict: Optional[Dict[str, np.ndarray]] = None
        self._Y_dict: Optional[Dict[str, np.ndarray]] = None
        self._S_matrix: Optional[np.ndarray] = None
        self._Z_matrix: Optional[np.ndarray] = None
        self._Y_matrix: Optional[np.ndarray] = None

        # Port/mode info extracted from available files
        self._available_params: List[str] = []
        self._n_ports: int = 0
        self._n_modes_per_port: int = 1

        # Load all available parameters
        self._load_all_s_parameters()
        self._load_all_z_parameters()
        self._load_all_y_parameters()

    def _validate_folder(self) -> Path:
        """Validate project folder and return Export folder path."""
        if not self.project_folder.exists():
            raise FileNotFoundError(
                f"Project folder not found: {self.project_folder}"
            )

        export_folder = self.project_folder / "Export"
        if not export_folder.exists():
            raise FileNotFoundError(
                f"Export folder not found in {self.project_folder}. "
                f"Please export S-parameters from CST first."
            )

        # Check for S-parameter files
        s_param_files = list(export_folder.glob("S-Parameters_S*.txt"))
        if not s_param_files:
            raise FileNotFoundError(
                f"No S-parameter files found in {export_folder}. "
                f"Expected files like 'S-Parameters_S1(1),1(1).txt'."
            )

        return export_folder

    # _parse_filename removed as regex is used directly in loaders

    def _load_s_parameter_file(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single S-parameter file.

        Parameters
        ----------
        filepath : Path
            Path to the S-parameter text file.

        Returns
        -------
        frequencies : np.ndarray
            Frequency in Hz.
        s_complex : np.ndarray
            Complex S-parameter values.
        """
        # Load data (whitespace separated)
        data = np.loadtxt(filepath)

        if data.ndim == 1:
            data = data.reshape(1, -1)

        freq_ghz = data[:, 0]
        magnitude = data[:, 1]
        phase_deg = data[:, 2]

        # Convert to Hz
        freq_hz = freq_ghz * 1e9

        # Convert magnitude and phase to complex
        # S = magnitude * exp(j * phase)
        phase_rad = np.deg2rad(phase_deg)
        s_complex = magnitude * np.exp(1j * phase_rad)

        return freq_hz, s_complex

    def _load_all_s_parameters(self) -> None:
        """Load all available S-parameter files from Export folder."""
        s_param_files = sorted(
            list(self._export_folder.glob("S-Parameters_S*.txt")) +
            list(self._export_folder.glob("S Matrix_S*.txt"))
        )

        if not s_param_files:
            raise FileNotFoundError(
                f"No S-parameter files found in {self._export_folder}"
            )

        # First pass: determine frequency grid from first file
        first_file = s_param_files[0]
        self._frequencies, _ = self._load_s_parameter_file(first_file)
        n_freqs = len(self._frequencies)

        # Track port/mode indices
        port_indices = set()
        mode_indices = set()

        # Load all files
        for filepath in s_param_files:
            match = self._FILENAME_PATTERN_S.search(filepath.name)
            if not match:
                print(f"  Warning: Could not parse filename {filepath.name}, skipping.")
                continue

            indices = tuple(int(x) for x in match.groups())
            port_i, mode_i, port_j, mode_j = indices
            port_indices.update([port_i, port_j])
            mode_indices.update([mode_i, mode_j])

            # Load data
            freq_hz, s_complex = self._load_s_parameter_file(filepath)

            # Validate frequency consistency
            if len(freq_hz) != n_freqs:
                print(f"  Warning: {filepath.name} has {len(freq_hz)} points, "
                      f"expected {n_freqs}. Skipping.")
                continue

            if not np.allclose(freq_hz, self._frequencies, rtol=1e-6):
                print(f"  Warning: {filepath.name} has inconsistent frequencies. Skipping.")
                continue

            # Store with key format matching PlotMixin: '{i}({m}){j}({n})'
            key = f"{port_i}({mode_i}){port_j}({mode_j})"
            self._S_dict[key] = s_complex
            self._available_params.append(key)

        # Store metadata
        self._n_ports = max(port_indices) if port_indices else 0
        self._n_modes_per_port = max(mode_indices) if mode_indices else 1

        print(f"Loaded CST S-parameters from: {self.project_folder.name}")
        print(f"  Frequency range: {self._frequencies[0]/1e9:.4f} - "
              f"{self._frequencies[-1]/1e9:.4f} GHz ({n_freqs} points)")
        print(f"  Ports: {self._n_ports}, Modes per port: {self._n_modes_per_port}")
        print(f"  S-parameters loaded: {len(self._S_dict)}")
        self._build_s_matrix()

    def _build_s_matrix(self) -> None:
        """Build global S-matrix from dictionaries."""
        if not self._S_dict:
            return

        n_freqs = len(self._frequencies)
        n_total = self._n_ports * self._n_modes_per_port
        self._S_matrix = np.zeros((n_freqs, n_total, n_total), dtype=complex)

        for key, s_data in self._S_dict.items():
            match = re.match(r"(\d+)\((\d+)\)(\d+)\((\d+)\)", key)
            if match:
                pi, mi, pj, mj = [int(x) for x in match.groups()]
                row = (pi - 1) * self._n_modes_per_port + (mi - 1)
                col = (pj - 1) * self._n_modes_per_port + (mj - 1)
                if row < n_total and col < n_total:
                    self._S_matrix[:, row, col] = s_data

    def _load_all_z_parameters(self) -> None:
        """Load all available Z-parameter files from Export folder."""
        z_param_files = sorted(
            list(self._export_folder.glob("Z-Parameters_Z*.txt")) +
            list(self._export_folder.glob("Z Matrix_Z*.txt"))
        )

        if not z_param_files:
            # We don't raise error here, just don't populate Z_dict
            self._Z_dict = {}
            return

        # Use same frequency grid as S if available, otherwise determine from first Z file
        if self._frequencies is None:
            first_file = z_param_files[0]
            self._frequencies, _ = self._load_s_parameter_file(first_file)
        
        n_freqs = len(self._frequencies)

        if self._Z_dict is None:
            self._Z_dict = {}

        for filepath in z_param_files:
            match = self._FILENAME_PATTERN_Z.search(filepath.name)
            if not match:
                continue
            
            indices = tuple(int(x) for x in match.groups())
            port_i, mode_i, port_j, mode_j = indices

            # Load data (reuse S-loader as format is identical: freq, mag, phase)
            freq_hz, z_complex = self._load_s_parameter_file(filepath)

            # Validate consistency
            if len(freq_hz) != n_freqs or not np.allclose(freq_hz, self._frequencies, rtol=1e-6):
                print(f"  Warning: {filepath.name} has inconsistent frequencies. Skipping.")
                continue

            key = f"{port_i}({mode_i}){port_j}({mode_j})"
            self._Z_dict[key] = z_complex
            if key not in self._available_params:
                self._available_params.append(key)

        print(f"  Z-parameters loaded: {len(self._Z_dict)}")
        self._build_z_matrix()

    def _build_z_matrix(self) -> None:
        """Build global Z-matrix from dictionaries."""
        if not self._Z_dict:
            return

        n_freqs = len(self._frequencies)
        n_total = self._n_ports * self._n_modes_per_port
        self._Z_matrix = np.zeros((n_freqs, n_total, n_total), dtype=complex)

        for key, z_data in self._Z_dict.items():
            match = re.match(r"(\d+)\((\d+)\)(\d+)\((\d+)\)", key)
            if match:
                pi, mi, pj, mj = [int(x) for x in match.groups()]
                row = (pi - 1) * self._n_modes_per_port + (mi - 1)
                col = (pj - 1) * self._n_modes_per_port + (mj - 1)
                if row < n_total and col < n_total:
                    self._Z_matrix[:, row, col] = z_data

    def _load_all_y_parameters(self) -> None:
        """Load all available Y-parameter files from Export folder."""
        y_param_files = sorted(
            list(self._export_folder.glob("Y-Parameters_Y*.txt")) +
            list(self._export_folder.glob("Y Matrix_Y*.txt"))
        )

        if not y_param_files:
            self._Y_dict = {}
            return

        if self._frequencies is None:
            first_file = y_param_files[0]
            self._frequencies, _ = self._load_s_parameter_file(first_file)
        
        n_freqs = len(self._frequencies)

        if self._Y_dict is None:
            self._Y_dict = {}

        for filepath in y_param_files:
            match = self._FILENAME_PATTERN_Y.search(filepath.name)
            if not match:
                continue
            
            indices = tuple(int(x) for x in match.groups())
            port_i, mode_i, port_j, mode_j = indices

            freq_hz, y_complex = self._load_s_parameter_file(filepath)

            if len(freq_hz) != n_freqs or not np.allclose(freq_hz, self._frequencies, rtol=1e-6):
                print(f"  Warning: {filepath.name} has inconsistent frequencies. Skipping.")
                continue

            key = f"{port_i}({mode_i}){port_j}({mode_j})"
            self._Y_dict[key] = y_complex
            if key not in self._available_params:
                self._available_params.append(key)

        print(f"  Y-parameters loaded: {len(self._Y_dict)}")
        self._build_y_matrix()

    def _build_y_matrix(self) -> None:
        """Build global Y-matrix from dictionaries."""
        if not self._Y_dict:
            return

        n_freqs = len(self._frequencies)
        n_total = self._n_ports * self._n_modes_per_port
        self._Y_matrix = np.zeros((n_freqs, n_total, n_total), dtype=complex)

        for key, y_data in self._Y_dict.items():
            match = re.match(r"(\d+)\((\d+)\)(\d+)\((\d+)\)", key)
            if match:
                pi, mi, pj, mj = [int(x) for x in match.groups()]
                row = (pi - 1) * self._n_modes_per_port + (mi - 1)
                col = (pj - 1) * self._n_modes_per_port + (mj - 1)
                if row < n_total and col < n_total:
                    self._Y_matrix[:, row, col] = y_data

    @property
    def Y_dict(self) -> Dict[str, np.ndarray]:
        """Y-parameters as dict: key → complex array."""
        return self._Y_dict or {}

    @property
    def Y_matrix(self) -> np.ndarray:
        """Y-parameter matrix: (n_freqs, n_ports*n_modes, n_ports*n_modes)."""
        return self._Y_matrix

    def _compute_z_from_s(self) -> None:
        """Compute Z-parameters from S-parameters using reference impedance."""
        if not self._S_dict:
            return

        n_freqs = len(self._frequencies)
        n_total = self._n_ports * self._n_modes_per_port

        # Build S-matrix: (n_freqs, n_total, n_total)
        self._S_matrix = np.zeros((n_freqs, n_total, n_total), dtype=complex)

        for key, s_data in self._S_dict.items():
            # Parse key: '{i}({m}){j}({n})'
            match = re.match(r"(\d+)\((\d+)\)(\d+)\((\d+)\)", key)
            if match:
                pi, mi, pj, mj = [int(x) for x in match.groups()]
                # Convert to 0-indexed matrix position
                row = (pi - 1) * self._n_modes_per_port + (mi - 1)
                col = (pj - 1) * self._n_modes_per_port + (mj - 1)
                if row < n_total and col < n_total:
                    self._S_matrix[:, row, col] = s_data

        # Z = Z0 * (I + S) @ inv(I - S)
        I = np.eye(n_total, dtype=complex)
        self._Z_matrix = np.zeros_like(self._S_matrix)
        self._Z_dict = {}

        for k in range(n_freqs):
            S_k = self._S_matrix[k]
            try:
                self._Z_matrix[k] = self.z0 * (I + S_k) @ np.linalg.inv(I - S_k)
            except np.linalg.LinAlgError:
                # Singular matrix (S ≈ I at some frequency)
                self._Z_matrix[k] = np.full((n_total, n_total), np.nan + 1j*np.nan)

        # Build Z_dict with same keys as S_dict
        for key in self._S_dict.keys():
            match = re.match(r"(\d+)\((\d+)\)(\d+)\((\d+)\)", key)
            if match:
                pi, mi, pj, mj = [int(x) for x in match.groups()]
                row = (pi - 1) * self._n_modes_per_port + (mi - 1)
                col = (pj - 1) * self._n_modes_per_port + (mj - 1)
                if row < n_total and col < n_total:
                    self._Z_dict[key] = self._Z_matrix[:, row, col]

    # ------------------------------------------------------------------
    # Properties for PlotMixin compatibility
    # ------------------------------------------------------------------

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency points in Hz."""
        return self._frequencies

    @property
    def S_dict(self) -> Dict[str, np.ndarray]:
        """S-parameters as dict: key → complex array."""
        return self._S_dict

    @property
    def Z_dict(self) -> Dict[str, np.ndarray]:
        """Z-parameters as dict: key → complex array."""
        if not self._Z_dict:
            print("\n" + "!" * 60)
            print("WARNING: No Z-parameter files found in CST Export folder.")
            print("Z-parameters must be exported specifically from CST as .txt files.")
            print("Automatic conversion from S-parameters is disabled to avoid accuracy issues.")
            print("!" * 60 + "\n")
            return {}
        return self._Z_dict

    @property
    def S_matrix(self) -> np.ndarray:
        """S-parameter matrix: (n_freqs, n_ports*n_modes, n_ports*n_modes)."""
        return self._S_matrix

    @property
    def Z_matrix(self) -> np.ndarray:
        """Z-parameter matrix: (n_freqs, n_ports*n_modes, n_ports*n_modes)."""
        return self._Z_matrix

    @property
    def available_parameters(self) -> List[str]:
        """List of available S-parameter keys."""
        return self._available_params.copy()

    @property
    def n_ports(self) -> int:
        """Number of ports."""
        return self._n_ports

    @property
    def n_modes_per_port(self) -> int:
        """Number of modes per port."""
        return self._n_modes_per_port

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_s_parameter(self, port_i: int, mode_i: int,
                        port_j: int, mode_j: int) -> np.ndarray:
        """
        Get a specific S-parameter by port/mode indices.

        Parameters
        ----------
        port_i, mode_i : int
            Output port and mode (1-indexed).
        port_j, mode_j : int
            Input port and mode (1-indexed).

        Returns
        -------
        np.ndarray
            Complex S-parameter vs frequency.
        """
        key = f"{port_i}({mode_i}){port_j}({mode_j})"
        if key not in self._S_dict:
            raise KeyError(
                f"S-parameter {key} not found. "
                f"Available: {self._available_params}"
            )
        return self._S_dict[key]

    def get_z_parameter(self, port_i: int, mode_i: int,
                        port_j: int, mode_j: int) -> np.ndarray:
        """
        Get a specific Z-parameter by port/mode indices.

        Parameters
        ----------
        port_i, mode_i : int
            Output port and mode (1-indexed).
        port_j, mode_j : int
            Input port and mode (1-indexed).

        Returns
        -------
        np.ndarray
            Complex Z-parameter vs frequency.
        """
        key = f"{port_i}({mode_i}){port_j}({mode_j})"
        if key not in self.Z_dict:
            raise KeyError(
                f"Z-parameter {key} not found. "
                f"Available: {list(self.Z_dict.keys())}"
            )
        return self.Z_dict[key]

    def interpolate_to(
        self,
        target_frequencies: np.ndarray,
    ) -> 'CSTResult':
        """
        Return a new CSTResult interpolated to target frequencies.

        Useful for comparing with numerical solvers on different frequency grids.

        Parameters
        ----------
        target_frequencies : np.ndarray
            Target frequency points in Hz.

        Returns
        -------
        CSTResult
            New instance with interpolated data.
        """
        from scipy.interpolate import interp1d

        # Create a shallow copy
        new_result = CSTResult.__new__(CSTResult)
        new_result.project_folder = self.project_folder
        new_result.z0 = self.z0
        new_result._export_folder = self._export_folder
        new_result._n_ports = self._n_ports
        new_result._n_modes_per_port = self._n_modes_per_port
        new_result._available_params = self._available_params.copy()

        # Interpolate frequencies
        new_result._frequencies = target_frequencies.copy()

        # Interpolate S-parameters (magnitude and phase separately for smoothness)
        new_result._S_dict = {}
        for key, s_data in self._S_dict.items():
            mag_interp = interp1d(
                self._frequencies, np.abs(s_data),
                kind='cubic', fill_value='extrapolate'
            )
            phase_interp = interp1d(
                self._frequencies, np.unwrap(np.angle(s_data)),
                kind='cubic', fill_value='extrapolate'
            )

            new_mag = mag_interp(target_frequencies)
            new_phase = phase_interp(target_frequencies)
            new_result._S_dict[key] = new_mag * np.exp(1j * new_phase)

        # Recompute Z from interpolated S
        new_result._Z_dict = None
        new_result._S_matrix = None
        new_result._Z_matrix = None
        new_result._compute_z_from_s()

        return new_result

    def __repr__(self) -> str:
        return (
            f"CSTResult('{self.project_folder.name}', "
            f"ports={self._n_ports}, modes={self._n_modes_per_port}, "
            f"n_freq={len(self._frequencies) if self._frequencies is not None else 0})"
        )


def load_cst_results(project_folder: Union[str, Path], z0: float = 50.0) -> CSTResult:
    """
    Convenience function to load CST results.

    Parameters
    ----------
    project_folder : str or Path
        Path to CST project folder.
    z0 : float
        Reference impedance (default 50 Ω).

    Returns
    -------
    CSTResult
        Loaded results object.

    Examples
    --------
    >>> cst = load_cst_results("/path/to/cst_project")
    >>> cst.plot_s()
    """
    return CSTResult(project_folder, z0=z0)