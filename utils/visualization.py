"""Visualization utilities for electromagnetic solvers."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional, Dict, List, Tuple, Union, Any
import scipy.sparse as sp


def spy_colored(
        A,
        cmap: str = 'viridis',
        markersize: int = 10,
        ax: Optional[plt.Axes] = None,
        title: str = "Sparse Matrix Pattern"
) -> plt.Axes:
    """Visualize sparse matrix with colors representing magnitude."""
    A = sp.coo_matrix(A)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    values = np.abs(A.data)
    norm = Normalize(vmin=values.min(), vmax=values.max())

    sc = ax.scatter(
        A.col, A.row,
        marker='s', c=values, s=markersize,
        cmap=cmap, norm=norm
    )

    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    plt.colorbar(sc, ax=ax, label="|Value|")

    return ax


class DataExtractor:
    """
    Extract Z/S parameter data from various sources.

    Supports:
    - FrequencyDomainSolver
    - ModelOrderReduction
    - ConcatenatedSystem
    - ReducedConcatenatedSystem
    - RWGAnalytical
    - Dict with appropriate keys
    """

    @staticmethod
    def get_source_type(source: Any) -> str:
        """Identify the type of data source."""
        class_name = source.__class__.__name__

        if class_name in ['FrequencyDomainSolver', 'ModelOrderReduction',
                          'ConcatenatedSystem', 'ReducedConcatenatedSystem',
                          'FOMResult', 'FOMCollection', 'ROMCollection']:
            return 'solver'
        elif class_name in ['RWGAnalytical', 'CWGAnalytical']:
            return 'analytical'
        elif isinstance(source, dict):
            return 'dict'
        else:
            raise TypeError(f"Unknown source type: {class_name}")

    @staticmethod
    def get_label(source: Any, custom_label: str = None) -> str:
        """Get a display label for the source."""
        if custom_label:
            return custom_label

        class_name = source.__class__.__name__
        label_map = {
            'FrequencyDomainSolver': 'FOM',
            'ModelOrderReduction': 'ROM',
            'ConcatenatedSystem': 'Concatenated',
            'ReducedConcatenatedSystem': 'Reduced Concat',
            'RWGAnalytical': 'Analytical',
            # Result wrapper types
            'FOMResult': 'FOM',
            'FOMCollection': 'FOM (per-domain)',
            'ROMCollection': 'ROM (per-domain)',
        }
        return label_map.get(class_name, class_name)

    @staticmethod
    def get_style(source: Any, custom_style: dict = None) -> dict:
        """Get plot style for the source."""
        if custom_style:
            return custom_style

        class_name = source.__class__.__name__

        styles = {
            'RWGAnalytical': {'linestyle': '-', 'marker': '', 'linewidth': 2},
            'CWGAnalytical': {'linestyle': '-', 'marker': '', 'linewidth': 2},
            'FrequencyDomainSolver': {'linestyle': '--', 'marker': '', 'linewidth': 1.5},
            'ModelOrderReduction': {'linestyle': ':', 'marker': '', 'linewidth': 1.5},
            'ConcatenatedSystem': {'linestyle': '-.', 'marker': '', 'linewidth': 1.5},
            'ReducedConcatenatedSystem': {'linestyle': '', 'marker': 'o', 'markersize': 4},
            # Result wrapper types
            'FOMResult': {'linestyle': '--', 'marker': '', 'linewidth': 1.5},
            'FOMCollection': {'linestyle': '--', 'marker': '', 'linewidth': 1.5},
            'ROMCollection': {'linestyle': ':', 'marker': '', 'linewidth': 1.5},
        }

        return styles.get(class_name, {'linestyle': '-', 'marker': '', 'linewidth': 1})

    @staticmethod
    def extract_z_parameters(
            source: Any,
            frequencies: np.ndarray = None,
            port_i: int = 1,
            port_j: int = 1,
            mode_i: int = 1,  # 1-based mode index
            mode_j: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Z_{port_i (mode_i) → port_j (mode_j)} from source.

        Returns:
            frequencies (Hz), complex Z array
        """
        source_type = DataExtractor.get_source_type(source)

        if source_type == 'analytical':
            if frequencies is None:
                raise ValueError("frequencies required for analytical source")

            Z_dict = source.z_parameters(frequencies)

            # Analytical is usually single-mode / fundamental
            if port_i == 1 and port_j == 1:
                key = 'Z11'
            elif port_i == 2 and port_j == 2:
                key = 'Z22'
            elif port_i == 1 and port_j == 2:
                key = 'Z12'
            elif port_i == 2 and port_j == 1:
                key = 'Z21'
            else:
                raise ValueError("Analytical source is 2-port only (ports 1 and 2)")

            return np.asarray(frequencies), np.asarray(Z_dict[key])

        elif source_type == 'solver':
            freqs = source.frequencies

            # Preferred: use dict with explicit mode keys
            if hasattr(source, 'Z_dict') and source.Z_dict:
                key = f'{port_i}({mode_i}){port_j}({mode_j})'
                if key in source.Z_dict:
                    return freqs, source.Z_dict[key]
                else:
                    available = [k for k in source.Z_dict.keys() if k != 'frequencies']
                    raise KeyError(
                        f"Z key not found: '{key}'\n"
                        f"Available Z keys: {available[:10]}{'...' if len(available) > 10 else ''}"
                    )

            # Fallback: direct matrix access with mode-aware indexing
            elif hasattr(source, '_Z_matrix') and source._Z_matrix is not None:
                n_modes = getattr(source, '_n_modes_per_port', 1)
                row = (port_i - 1) * n_modes + (mode_i - 1)
                col = (port_j - 1) * n_modes + (mode_j - 1)

                if row >= source._Z_matrix.shape[1] or col >= source._Z_matrix.shape[2]:
                    raise IndexError(
                        f"Index out of bounds: row={row}, col={col}, "
                        f"matrix shape={source._Z_matrix.shape[1:]} "
                        f"(n_modes={n_modes}, ports={getattr(source, 'n_ports', '?')})"
                    )

                return freqs, source._Z_matrix[:, row, col]

            else:
                raise ValueError("No Z-parameter data available in solver")

        elif source_type == 'dict':
            freqs = source.get('frequencies')
            key = f'{port_i}({mode_i}){port_j}({mode_j})'
            if key in source:
                return freqs, source[key]
            else:
                raise KeyError(f"Cannot find Z key '{key}' in dictionary")

        raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def extract_s_parameters(
            source: Any,
            frequencies: np.ndarray = None,
            port_i: int = 1,
            port_j: int = 1,
            mode_i: int = 1,
            mode_j: int = 1,
            Z0_ref: Union[str, float] = 'ZTE'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract S_{port_i (mode_i) → port_j (mode_j)} from source.
        """
        source_type = DataExtractor.get_source_type(source)

        if source_type == 'analytical':
            if frequencies is None:
                raise ValueError("frequencies required for analytical source")

            S_dict = source.s_parameters(frequencies, Z0_ref=Z0_ref)

            # Same 2-port mapping as Z
            if port_i == 1 and port_j == 1:
                key = 'S11'
            elif port_i == 2 and port_j == 2:
                key = 'S22'
            elif port_i == 1 and port_j == 2:
                key = 'S12'
            elif port_i == 2 and port_j == 1:
                key = 'S21'
            else:
                raise ValueError("Analytical source is 2-port only")

            return np.asarray(frequencies), np.asarray(S_dict[key])

        elif source_type == 'solver':
            freqs = source.frequencies

            # Preferred: dict lookup
            if hasattr(source, 'S_dict') and source.S_dict:
                key = f'{port_i}({mode_i}){port_j}({mode_j})'
                if key in source.S_dict:
                    return freqs, source.S_dict[key]
                else:
                    available = [k for k in source.S_dict.keys() if k != 'frequencies']
                    raise KeyError(
                        f"S key not found: '{key}'\n"
                        f"Available S keys: {available[:10]}{'...' if len(available) > 10 else ''}"
                    )

            # Fallback: matrix
            elif hasattr(source, '_S_matrix') and source._S_matrix is not None:
                n_modes = getattr(source, '_n_modes_per_port', 1)
                row = (port_i - 1) * n_modes + (mode_i - 1)
                col = (port_j - 1) * n_modes + (mode_j - 1)

                if row >= source._S_matrix.shape[1] or col >= source._S_matrix.shape[2]:
                    raise IndexError(
                        f"Index out of bounds: row={row}, col={col}, "
                        f"matrix shape={source._S_matrix.shape[1:]}"
                    )

                return freqs, source._S_matrix[:, row, col]

            else:
                raise ValueError("No S-parameter data available in solver")

        elif source_type == 'dict':
            freqs = source.get('frequencies')
            key = f'{port_i}({mode_i}){port_j}({mode_j})'
            if key in source:
                return freqs, source[key]
            else:
                raise KeyError(f"Cannot find S key '{key}' in dictionary")

        raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def get_n_ports(source: Any) -> int:
        """Get number of ports from source."""
        source_type = DataExtractor.get_source_type(source)

        if source_type == 'analytical':
            return 2
        elif source_type == 'solver':
            return source.n_ports
        elif source_type == 'dict':
            if 'n_ports' in source:
                return source['n_ports']
            elif 'Z' in source and isinstance(source['Z'], np.ndarray):
                return source['Z'].shape[1]
            return 2

    @staticmethod
    def get_frequencies(source: Any) -> np.ndarray:
        """Get frequency array from source."""
        source_type = DataExtractor.get_source_type(source)

        if source_type == 'analytical':
            return None
        elif source_type == 'solver':
            return source.frequencies
        elif source_type == 'dict':
            return source.get('frequencies')

    @staticmethod
    def extract_resonant_frequencies(
            source: Any,
            n_modes: int = None,
            domain: str = None,
            filter_static: bool = True,
            min_eigenvalue: float = 1.0,
            boundary_type: str = 'PEC'
    ) -> Optional[np.ndarray]:
        """
        Extract resonant frequencies from source.

        Parameters
        ----------
        source : solver, analytical, or object with get_resonant_frequencies
        n_modes : int, optional
            Number of modes to return
        domain : str, optional
            Specific domain or 'global'
        filter_static : bool
            If True (default), remove static modes from numerical sources
        min_eigenvalue : float
            Threshold for static mode filtering
        boundary_type : str
            'PEC' or 'PMC' boundary conditions (for analytical sources)

        Returns
        -------
        frequencies : ndarray or None
            Resonant frequencies in Hz
        """
        try:
            source_type = DataExtractor.get_source_type(source)

            # For analytical sources, prefer all_eigenfrequencies with boundary_type
            if source_type == 'analytical':
                if hasattr(source, 'all_eigenfrequencies'):
                    import inspect
                    sig = inspect.signature(source.all_eigenfrequencies)
                    params = sig.parameters

                    kwargs = {
                        'n_modes': n_modes or 100,
                        'return_format': 'array'
                    }
                    if 'boundary_type' in params:
                        kwargs['boundary_type'] = boundary_type

                    freqs = source.all_eigenfrequencies(**kwargs)
                    if freqs is not None and n_modes is not None:
                        freqs = freqs[:n_modes]
                    return freqs

                # Fallback to resonant_frequencies (legacy)
                elif hasattr(source, 'resonant_frequencies'):
                    freqs = source.resonant_frequencies(n_modes or 100)
                    if n_modes is not None and freqs is not None:
                        freqs = freqs[:n_modes]
                    return freqs

            # For solver sources, try get_resonant_frequencies
            if hasattr(source, 'get_resonant_frequencies'):
                import inspect
                sig = inspect.signature(source.get_resonant_frequencies)
                params = list(sig.parameters.keys())

                kwargs = {}
                if 'n_modes' in params:
                    kwargs['n_modes'] = n_modes
                if 'domain' in params:
                    kwargs['domain'] = domain
                if 'filter_static' in params:
                    kwargs['filter_static'] = filter_static
                if 'min_eigenvalue' in params:
                    kwargs['min_eigenvalue'] = min_eigenvalue

                freqs = source.get_resonant_frequencies(**kwargs)

                if freqs is not None and n_modes is not None:
                    freqs = freqs[:n_modes]

                return freqs

            # Fall back to eigenvalues
            if hasattr(source, 'get_eigenvalues'):
                import inspect
                sig = inspect.signature(source.get_eigenvalues)
                params = list(sig.parameters.keys())

                kwargs = {}
                if 'domain' in params:
                    kwargs['domain'] = domain
                if 'filter_static' in params:
                    kwargs['filter_static'] = filter_static
                if 'min_eigenvalue' in params:
                    kwargs['min_eigenvalue'] = min_eigenvalue
                if 'n_modes' in params:
                    kwargs['n_modes'] = n_modes

                eigs = source.get_eigenvalues(**kwargs)

                if isinstance(eigs, dict):
                    if 'global' in eigs:
                        eigs = eigs['global']
                    else:
                        eigs = np.concatenate(list(eigs.values()))

                if eigs is not None and len(eigs) > 0:
                    eigs_pos = eigs[eigs > 0]
                    freqs = np.sqrt(eigs_pos) / (2 * np.pi)
                    freqs = np.sort(freqs)
                    if n_modes is not None:
                        freqs = freqs[:n_modes]
                    return freqs

            return None

        except Exception as e:
            print(f"Warning: Could not extract resonant frequencies from "
                  f"{type(source).__name__}: {e}")
            return None

    @staticmethod
    def extract_resonant_frequencies(
        source: Any,
        n_modes: int = None,
        domain: str = None,
        filter_static: bool = True,
        min_eigenvalue: float = 1.0
    ) -> Optional[np.ndarray]:
        """
        Extract resonant frequencies from source.

        Parameters
        ----------
        source : solver, analytical, or object with get_resonant_frequencies
        n_modes : int, optional
            Number of modes to return
        domain : str, optional
            Specific domain or 'global'
        filter_static : bool
            If True (default), remove static modes from numerical sources
        min_eigenvalue : float
            Threshold for static mode filtering

        Returns
        -------
        frequencies : ndarray or None
            Resonant frequencies in Hz
        """
        try:
            if hasattr(source, 'all_eigenfrequencies'):
                import inspect
                sig = inspect.signature(source.all_eigenfrequencies)
                params = sig.parameters

                kwargs = {
                    'n_modes': n_modes or 100,
                    'return_format': 'array'
                }

                freqs = source.all_eigenfrequencies(**kwargs)
                if freqs is not None and n_modes is not None:
                    freqs = freqs[:n_modes]
                return freqs

            # Try get_resonant_frequencies first
            if hasattr(source, 'get_resonant_frequencies'):
                import inspect
                sig = inspect.signature(source.get_resonant_frequencies)
                params = list(sig.parameters.keys())

                kwargs = {}
                if 'n_modes' in params:
                    kwargs['n_modes'] = n_modes
                if 'domain' in params:
                    kwargs['domain'] = domain
                if 'filter_static' in params:
                    kwargs['filter_static'] = filter_static
                if 'min_eigenvalue' in params:
                    kwargs['min_eigenvalue'] = min_eigenvalue

                freqs = source.get_resonant_frequencies(**kwargs)

                # Ensure we return at most n_modes
                if freqs is not None and n_modes is not None:
                    freqs = freqs[:n_modes]

                return freqs

            # Try resonant_frequencies (analytical)
            if hasattr(source, 'resonant_frequencies'):
                freqs = source.resonant_frequencies(n_modes or 100)
                if n_modes is not None:
                    freqs = freqs[:n_modes]
                return freqs

            # Fall back to eigenvalues
            if hasattr(source, 'get_eigenvalues'):
                import inspect
                sig = inspect.signature(source.get_eigenvalues)
                params = list(sig.parameters.keys())

                kwargs = {}
                if 'domain' in params:
                    kwargs['domain'] = domain
                if 'filter_static' in params:
                    kwargs['filter_static'] = filter_static
                if 'min_eigenvalue' in params:
                    kwargs['min_eigenvalue'] = min_eigenvalue
                if 'n_modes' in params:
                    kwargs['n_modes'] = n_modes

                eigs = source.get_eigenvalues(**kwargs)

                if isinstance(eigs, dict):
                    if 'global' in eigs:
                        eigs = eigs['global']
                    else:
                        eigs = np.concatenate(list(eigs.values()))

                if eigs is not None and len(eigs) > 0:
                    eigs_pos = eigs[eigs > 0]
                    freqs = np.sqrt(eigs_pos) / (2 * np.pi)
                    freqs = np.sort(freqs)
                    if n_modes is not None:
                        freqs = freqs[:n_modes]
                    return freqs

            return None

        except Exception as e:
            print(f"Warning: Could not extract resonant frequencies from "
                  f"{type(source).__name__}: {e}")
            return None


class ParameterPlotter:
    """
    Unified plotter for Z and S parameters.

    Can plot comparisons between multiple solvers and/or analytical solutions.

    Examples
    --------
    >>> plotter = ParameterPlotter()
    >>> plotter.plot_z_comparison([fds, rom, analytical], frequencies=freqs)
    >>> plotter.plot_s_comparison([fds, rom], labels=['FOM', 'ROM'])
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 10)):
        self.figsize = figsize
        self.fig = None
        self.axes = None
        self.extractor = DataExtractor()

    def plot_z_comparison(
            self,
            sources: List[Any],
            frequencies: np.ndarray = None,
            port_pairs: List[Tuple[int, int]] = None,
            mode_pairs: List[Tuple[int, int]] = None,  # NEW
            labels: List[str] = None,
            styles: List[dict] = None,
            title: str = 'Z-Parameter Comparison',
            show: bool = True
    ) -> Tuple[plt.Figure, Dict]:
        if port_pairs is None:
            port_pairs = [(1, 1), (1, 2)]

        if mode_pairs is None:
            mode_pairs = [(1, 1)] * len(port_pairs)  # default to fundamental modes

        if len(mode_pairs) != len(port_pairs):
            raise ValueError("mode_pairs must match length of port_pairs")

        if frequencies is None:
            for src in sources:
                f = self.extractor.get_frequencies(src)
                if f is not None:
                    frequencies = f
                    break
            if frequencies is None:
                raise ValueError("No frequencies found. Provide frequencies parameter.")

        n_pairs = len(port_pairs)
        fig, axes_array = plt.subplots(2, n_pairs, figsize=self.figsize)
        if n_pairs == 1:
            axes_array = axes_array.reshape(2, 1)

        self.axes = {}
        for col, ((pi, pj), (mi, mj)) in enumerate(zip(port_pairs, mode_pairs)):
            self.axes[f'Z{pi}({mi}),{pj}({mj})_mag'] = axes_array[0, col]
            self.axes[f'Z{pi}({mi}),{pj}({mj})_phase'] = axes_array[1, col]

        for idx, source in enumerate(sources):
            label = labels[idx] if labels and idx < len(labels) else self.extractor.get_label(source)
            style = styles[idx] if styles and idx < len(styles) else self.extractor.get_style(source)

            for col, ((pi, pj), (mi, mj)) in enumerate(zip(port_pairs, mode_pairs)):
                freqs, Z = self.extractor.extract_z_parameters(
                    source, frequencies, port_i=pi, port_j=pj, mode_i=mi, mode_j=mj
                )
                freq_ghz = freqs / 1e9

                ax_mag = self.axes[f'Z{pi}({mi}),{pj}({mj})_mag']
                ax_mag.plot(freq_ghz, 20 * np.log10(np.abs(Z) + 1e-15), label=label, **style)

                ax_phase = self.axes[f'Z{pi}({mi}),{pj}({mj})_phase']
                ax_phase.plot(freq_ghz, np.angle(Z, deg=True), label=label, **style)

        for col, ((pi, pj), (mi, mj)) in enumerate(zip(port_pairs, mode_pairs)):
            ax_mag = self.axes[f'Z{pi}({mi}),{pj}({mj})_mag']
            ax_mag.set_xlabel('Frequency (GHz)')
            ax_mag.set_ylabel(f'|Z{pi}({mi}),{pj}({mj})| (dB)')
            ax_mag.set_title(f'Z{pi}({mi}),{pj}({mj}) Magnitude')
            ax_mag.legend()
            ax_mag.grid(True, alpha=0.3)

            ax_phase = self.axes[f'Z{pi}({mi}),{pj}({mj})_phase']
            ax_phase.set_xlabel('Frequency (GHz)')
            ax_phase.set_ylabel(f'∠Z{pi}({mi}),{pj}({mj}) (°)')
            ax_phase.set_title(f'Z{pi}({mi}),{pj}({mj}) Phase')
            ax_phase.legend()
            ax_phase.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if show:
            plt.show()

        return fig, self.axes

    def plot_s_comparison(
            self,
            sources: List[Any],
            frequencies: np.ndarray = None,
            port_pairs: List[Tuple[int, int]] = None,
            mode_pairs: List[Tuple[int, int]] = None,  # NEW
            labels: List[str] = None,
            styles: List[dict] = None,
            Z0_ref: Union[str, float] = 'ZTE',
            title: str = 'S-Parameter Comparison',
            show: bool = True
    ) -> Tuple[plt.Figure, Dict]:
        if port_pairs is None:
            port_pairs = [(1, 1), (1, 2)]

        if mode_pairs is None:
            mode_pairs = [(1, 1)] * len(port_pairs)

        if len(mode_pairs) != len(port_pairs):
            raise ValueError("mode_pairs must match length of port_pairs")

        # ... same frequency detection logic as above ...

        # Create axes
        n_pairs = len(port_pairs)
        fig, axes_array = plt.subplots(2, n_pairs, figsize=self.figsize)
        if n_pairs == 1:
            axes_array = axes_array.reshape(2, 1)

        self.axes = {}
        for col, ((pi, pj), (mi, mj)) in enumerate(zip(port_pairs, mode_pairs)):
            self.axes[f'S{pi}({mi}),{pj}({mj})_mag'] = axes_array[0, col]
            self.axes[f'S{pi}({mi}),{pj}({mj})_phase'] = axes_array[1, col]

        for idx, source in enumerate(sources):
            label = labels[idx] if labels and idx < len(labels) else self.extractor.get_label(source)
            style = styles[idx] if styles and idx < len(styles) else self.extractor.get_style(source)

            for col, ((pi, pj), (mi, mj)) in enumerate(zip(port_pairs, mode_pairs)):
                freqs, S = self.extractor.extract_s_parameters(
                    source, frequencies, port_i=pi, port_j=pj, mode_i=mi, mode_j=mj, Z0_ref=Z0_ref
                )
                freq_ghz = freqs / 1e9

                ax_mag = self.axes[f'S{pi}({mi}),{pj}({mj})_mag']
                ax_mag.plot(freq_ghz, 20 * np.log10(np.abs(S) + 1e-15), label=label, **style)

                ax_phase = self.axes[f'S{pi}({mi}),{pj}({mj})_phase']
                ax_phase.plot(freq_ghz, np.angle(S, deg=True), label=label, **style)

        # Labeling
        for col, ((pi, pj), (mi, mj)) in enumerate(zip(port_pairs, mode_pairs)):
            ax_mag = self.axes[f'S{pi}({mi}),{pj}({mj})_mag']
            ax_mag.set_xlabel('Frequency (GHz)')
            ax_mag.set_ylabel(f'|S{pi}({mi}),{pj}({mj})| (dB)')
            ax_mag.set_title(f'S{pi}({mi}),{pj}({mj}) Magnitude')
            ax_mag.legend()
            ax_mag.grid(True, alpha=0.3)

            ax_phase = self.axes[f'S{pi}({mi}),{pj}({mj})_phase']
            ax_phase.set_xlabel('Frequency (GHz)')
            ax_phase.set_ylabel(f'∠S{pi}({mi}),{pj}({mj}) (°)')
            ax_phase.set_title(f'S{pi}({mi}),{pj}({mj}) Phase')
            ax_phase.legend()
            ax_phase.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if show:
            plt.show()

        return fig, self.axes

    def plot_both(
        self,
        sources: List[Any],
        frequencies: np.ndarray = None,
        labels: List[str] = None,
        styles: List[dict] = None,
        Z0_ref: Union[str, float] = 'ZTE',
        title: str = 'Network Parameter Comparison',
        show: bool = True
    ) -> Tuple[plt.Figure, Dict]:
        """Plot both Z and S parameters in a 2x4 grid."""
        if frequencies is None:
            for src in sources:
                f = self.extractor.get_frequencies(src)
                if f is not None:
                    frequencies = f
                    break

        if frequencies is None:
            raise ValueError("No frequencies found. Provide frequencies parameter.")

        self.fig, axes_array = plt.subplots(2, 4, figsize=(16, 10))

        params = [
            ('Z', 1, 1, 0), ('Z', 1, 2, 1), ('S', 1, 1, 2), ('S', 1, 2, 3)
        ]

        self.axes = {}
        for param_type, pi, pj, col in params:
            self.axes[f'{param_type}{pi}{pj}_mag'] = axes_array[0, col]
            self.axes[f'{param_type}{pi}{pj}_phase'] = axes_array[1, col]

        for idx, source in enumerate(sources):
            label = labels[idx] if labels and idx < len(labels) else self.extractor.get_label(source)
            style = styles[idx] if styles and idx < len(styles) else self.extractor.get_style(source)

            for param_type, pi, pj, col in params:
                if param_type == 'Z':
                    freqs, data = self.extractor.extract_z_parameters(source, frequencies, pi, pj)
                else:
                    freqs, data = self.extractor.extract_s_parameters(source, frequencies, pi, pj, Z0_ref)

                freq_ghz = freqs / 1e9

                ax = self.axes[f'{param_type}{pi}{pj}_mag']
                ax.plot(freq_ghz, 20 * np.log10(np.abs(data) + 1e-15), label=label, **style)

                ax = self.axes[f'{param_type}{pi}{pj}_phase']
                ax.plot(freq_ghz, np.angle(data, deg=True), label=label, **style)

        for param_type, pi, pj, col in params:
            ax = self.axes[f'{param_type}{pi}{pj}_mag']
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel(f'|{param_type}{pi}{pj}| (dB)')
            ax.set_title(f'{param_type}{pi}{pj} Magnitude')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = self.axes[f'{param_type}{pi}{pj}_phase']
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel(f'∠{param_type}{pi}{pj} (°)')
            ax.set_title(f'{param_type}{pi}{pj} Phase')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if show:
            plt.show()

        return self.fig, self.axes


class EigenfrequencyPlotter:
    """
    Plot and compare eigenfrequencies from different sources.

    Default behavior is to use global eigenvalues with static modes filtered.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        self.figsize = figsize

    def plot_comparison(
            self,
            sources: List[Any],
            analytical: Any = None,
            n_modes: int = 20,
            labels: List[str] = None,
            title: str = 'Eigenfrequency Comparison',
            domain: str = None,
            filter_static: bool = True,
            min_eigenvalue: float = 1.0,
            show: bool = False
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot eigenfrequency comparison.

        Parameters
        ----------
        sources : list
            List of solvers with get_eigenvalues() or get_resonant_frequencies()
        analytical : RWGAnalytical, optional
            Analytical solution for reference
        n_modes : int
            Number of modes to plot (applies to ALL sources including analytical)
        labels : list of str, optional
            Labels for each source
        title : str
            Plot title
        domain : str, optional
            Specific domain or 'global'. Default: 'global' (or auto-detect)
        filter_static : bool
            If True (default), remove static modes from numerical sources
        min_eigenvalue : float
            Threshold for static mode filtering
        show : bool
            Whether to call plt.show()
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
        colors = plt.cm.tab10.colors

        # Plot analytical if provided (also limited to n_modes)
        if analytical is not None:
            anal_freqs = DataExtractor.extract_resonant_frequencies(
                analytical, n_modes=n_modes
            )
            if anal_freqs is not None:
                # Limit to n_modes
                anal_freqs = anal_freqs[:n_modes]
                for f in anal_freqs:
                    ax.axhline(f / 1e9, color='red', alpha=0.5, linewidth=0.8)
                    ax.axvline(f / 1e9, color='red', alpha=0.5, linewidth=0.8)
                ax.plot([], [], 'r-', label='Analytical', linewidth=0.8)

        # Plot each source (all limited to n_modes)
        for idx, source in enumerate(sources):
            label = labels[idx] if labels and idx < len(labels) else DataExtractor.get_label(source)
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]

            freqs = DataExtractor.extract_resonant_frequencies(
                source,
                n_modes=n_modes,
                domain=domain,
                filter_static=filter_static,
                min_eigenvalue=min_eigenvalue
            )

            if freqs is not None and len(freqs) > 0:
                # Ensure we don't exceed n_modes
                freqs = freqs[:n_modes]
                ax.scatter(freqs / 1e9, freqs / 1e9,
                    marker=marker, edgecolor=color, label=label, s=70, facecolor='none', zorder=3
                )

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Frequency (GHz)')
        ax.set_title(f'{title} (first {n_modes} modes)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if show:
            plt.show()

        return fig, ax

    def print_comparison(
            self,
            sources: List[Any],
            analytical: Any = None,
            n_modes: int = 10,
            labels: List[str] = None,
            domain: str = None,
            filter_static: bool = True,
            min_eigenvalue: float = 1.0
    ) -> None:
        """Print eigenfrequency comparison table."""
        print("\n" + "=" * 80)
        print(f"Eigenfrequency Comparison (first {n_modes} modes)")
        print("=" * 80)

        # Build header
        header = f"{'Mode':<6}"
        if analytical is not None:
            header += f"{'Analytical':<15}"

        for idx, source in enumerate(sources):
            label = labels[idx] if labels and idx < len(labels) else DataExtractor.get_label(source)
            header += f"{label:<15}"

        if analytical is not None and len(sources) > 0:
            header += f"{'Error (%)':<12}"

        print(header)
        print("-" * 80)

        # Get analytical frequencies (limited to n_modes)
        anal_freqs = None
        if analytical is not None:
            anal_freqs = DataExtractor.extract_resonant_frequencies(
                analytical, n_modes=n_modes
            )
            if anal_freqs is not None:
                anal_freqs = anal_freqs[:n_modes]

        # Get frequencies from each source (limited to n_modes)
        all_freqs = []
        for source in sources:
            freqs = DataExtractor.extract_resonant_frequencies(
                source,
                n_modes=n_modes,
                domain=domain,
                filter_static=filter_static,
                min_eigenvalue=min_eigenvalue
            )
            if freqs is not None:
                freqs = freqs[:n_modes]
            all_freqs.append(freqs if freqs is not None else np.array([]))

        # Print rows
        for i in range(n_modes):
            row = f"{i:<6}"

            if anal_freqs is not None and i < len(anal_freqs):
                row += f"{anal_freqs[i] / 1e9:<15.4f}"
            elif anal_freqs is not None:
                row += f"{'N/A':<15}"

            errors = []
            for freqs in all_freqs:
                if i < len(freqs):
                    row += f"{freqs[i] / 1e9:<15.4f}"
                    if anal_freqs is not None and i < len(anal_freqs):
                        err = abs(freqs[i] - anal_freqs[i]) / anal_freqs[i] * 100
                        errors.append(err)
                else:
                    row += f"{'N/A':<15}"

            if errors:
                row += f"{errors[0]:<12.2f}"

            print(row)

        print("=" * 80)


class WaveImpedancePlotter:
    """Plotting utilities for wave impedance visualization."""

    def __init__(self, figsize: Tuple[int, int] = (10, 5)):
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def setup_figure(self) -> Tuple[plt.Figure, plt.Axes]:
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        return self.fig, self.ax

    def plot_comparison(
        self,
        frequencies: np.ndarray,
        analytical: Any,
        solver: Any = None,
        title: str = 'TE₁₀ Wave Impedance',
        ylim: Tuple[float, float] = (-500, 1500),
        show: bool = True
    ) -> plt.Axes:
        """Plot comparison of analytical and numerical wave impedance."""
        if self.ax is None:
            self.setup_figure()

        freq_ghz = frequencies / 1e9
        ax = self.ax

        Zw_analytical = analytical.wave_impedance(frequencies)
        ax.plot(freq_ghz, np.real(Zw_analytical), 'b-', label='Analytical Re(Zw)', lw=2)
        ax.plot(freq_ghz, np.imag(Zw_analytical), 'b--', label='Analytical Im(Zw)', lw=2)

        if solver is not None and hasattr(solver, 'port_solver'):
            port = solver.ports[0]
            Zw_numerical = np.array([
                solver.port_solver.get_port_wave_impedance(port, 0, f)
                for f in frequencies
            ])
            ax.plot(freq_ghz, np.real(Zw_numerical), 'ro', label='Numerical Re(Zw)', ms=4)
            ax.plot(freq_ghz, np.imag(Zw_numerical), 'gs', label='Numerical Im(Zw)', ms=4)

        ax.axhline(377, color='gray', linestyle=':', label='η₀ = 377 Ω')
        ax.axvline(analytical.fc / 1e9, color='green', linestyle='--',
                   label=f'fc = {analytical.fc / 1e9:.3f} GHz')

        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Wave Impedance (Ω)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(ylim)

        plt.tight_layout()

        if show:
            plt.show()

        return ax


class ConvergencePlotter:
    """Plotting utilities for convergence studies."""

    @staticmethod
    def plot_singular_values(
        source: Any,
        domain: str = None,
        ax: Optional[plt.Axes] = None,
        title: str = 'Singular Value Decay',
        show: bool = True
    ) -> plt.Axes:
        """Plot singular value decay from ROM or concatenated system."""
        if ax is None:
            fig, ax = plt.subplots()

        if hasattr(source, 'singular_values'):
            sv_dict = source.singular_values
            if isinstance(sv_dict, dict):
                if domain is not None:
                    singular_values = sv_dict[domain]
                    title = f'{title} ({domain})'
                else:
                    for d, sv in sv_dict.items():
                        ax.semilogy(sv / sv[0], 'o-', label=d)
                    ax.set_xlabel('Index')
                    ax.set_ylabel('Normalized Singular Value')
                    ax.set_title(title)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    if show:
                        plt.show()
                    return ax
            else:
                singular_values = sv_dict
        elif hasattr(source, '_singular_values'):
            singular_values = source._singular_values
        else:
            raise ValueError("Source has no singular values")

        sv_norm = singular_values / singular_values[0]
        ax.semilogy(sv_norm, 'o-')

        if hasattr(source, '_r'):
            if isinstance(source._r, dict):
                if domain:
                    r = source._r[domain]
                else:
                    r = list(source._r.values())[0]
            else:
                r = source._r
            ax.axvline(r - 0.5, color='r', linestyle='--', label=f'Truncation (r={r})')
            ax.legend()

        ax.set_xlabel('Index')
        ax.set_ylabel('Normalized Singular Value')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if show:
            plt.show()

        return ax


# === Convenience functions ===

def plot_z_comparison(sources, frequencies=None, labels=None, **kwargs):
    """Quick Z-parameter comparison plot."""
    plotter = ParameterPlotter()
    return plotter.plot_z_comparison(sources, frequencies, labels=labels, **kwargs)


def plot_s_comparison(sources, frequencies=None, labels=None, **kwargs):
    """Quick S-parameter comparison plot."""
    plotter = ParameterPlotter()
    return plotter.plot_s_comparison(sources, frequencies, labels=labels, **kwargs)


def plot_all_parameters(sources, frequencies=None, labels=None, **kwargs):
    """Quick combined Z and S parameter comparison plot."""
    plotter = ParameterPlotter()
    return plotter.plot_both(sources, frequencies, labels=labels, **kwargs)


def plot_eigenfrequencies(
    sources,
    analytical=None,
    labels=None,
    n_modes: int = 20,
    domain: str = None,
    filter_static: bool = True,
    min_eigenvalue: float = 1.0,
    **kwargs
):
    """
    Quick eigenfrequency comparison plot.

    Parameters
    ----------
    sources : list
        List of solvers (FOM, ROM, ConcatenatedSystem, etc.)
    analytical : RWGAnalytical, optional
        Analytical reference
    labels : list of str, optional
        Labels for each source
    n_modes : int
        Number of modes to plot (applies to ALL sources)
    domain : str, optional
        Specific domain or 'global' (default: auto-detect, prefers global)
    filter_static : bool
        If True (default), remove static modes from numerical sources
    min_eigenvalue : float
        Threshold for static mode filtering (eigenvalues <= this are removed)
    """
    plotter = EigenfrequencyPlotter()
    return plotter.plot_comparison(
        sources, analytical, n_modes=n_modes, labels=labels,
        domain=domain, filter_static=filter_static,
        min_eigenvalue=min_eigenvalue, **kwargs
    )


def print_eigenfrequency_comparison(
    sources,
    analytical=None,
    labels=None,
    n_modes: int = 10,
    domain: str = None,
    filter_static: bool = True,
    min_eigenvalue: float = 1.0
):
    """Print eigenfrequency comparison table."""
    plotter = EigenfrequencyPlotter()
    plotter.print_comparison(
        sources, analytical, n_modes=n_modes, labels=labels,
        domain=domain, filter_static=filter_static,
        min_eigenvalue=min_eigenvalue
    )
