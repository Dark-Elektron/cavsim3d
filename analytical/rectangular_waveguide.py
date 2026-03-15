"""Analytical solutions for rectangular waveguide."""

import numpy as np
from typing import Dict, Union, List, Tuple, Optional
from core.constants import c0, mu0, eps0, Z0
import itertools

from utils.plot_mixin import PlotMixin


class RWGAnalytical(PlotMixin):
    """
    Analytical Z and S parameters for rectangular waveguide.

    For TE10 mode in a rectangular waveguide of dimensions a x b x L.

    Parameters
    ----------
    a : float
        Width (broad dimension) [m]
    L : float
        Length [m]
    b : float, optional
        Height [m] (default: a/2)
    freq_range : tuple of float, optional
        Frequency range (f_min, f_max) in Hz for auto-sampling.
        If None, defaults to (0.5·fc, 2.5·fc) based on the TE10 cutoff.
    n_samples : int
        Number of frequency samples (default: 1000).
    """

    def __init__(
        self,
        a: float,
        L: float,
        b: float = None,
        freq_range: Optional[Tuple[float, float]] = None,
        n_samples: int = 1000,
    ):
        self.a = a
        self.b = b if b is not None else a / 2
        self.L = L

        # Cutoff quantities for TE10 mode
        self.kc = np.pi / self.a   # Cutoff wavenumber
        self.wc = self.kc * c0     # Cutoff angular frequency
        self.fc = c0 / (2 * self.a)  # Cutoff frequency [Hz]

        # Frequency grid config
        self._freq_range = freq_range  # (f_min, f_max) Hz, or None → physics default
        self._n_samples  = n_samples

        # Cache — populated lazily on first plot call (or manually via compute())
        self._frequencies: Optional[np.ndarray] = None
        self._S_dict: Optional[Dict[str, np.ndarray]] = None
        self._Z_dict: Optional[Dict[str, np.ndarray]] = None

    # =========================================================================
    # Core electromagnetic quantities
    # =========================================================================

    def wave_impedance(self, freq: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute TE10 wave impedance.

        ZTE = s * eta0 / sqrt(s^2 + wc^2)  where s = j*omega

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]

        Returns
        -------
        ZTE : ndarray
            TE10 wave impedance (complex)
        """
        freq = np.atleast_1d(freq).astype(complex)
        omega = 2 * np.pi * freq
        s = 1j * omega
        return s * Z0 / np.sqrt(s**2 + self.wc**2)

    def propagation_constant(self, freq: Union[float, np.ndarray]) -> np.ndarray:
        """
        Compute propagation constant kz for TE10 mode.

        kz = omega * mu0 / ZTE

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]

        Returns
        -------
        kz : ndarray
            Propagation constant (complex)
        """
        freq = np.atleast_1d(freq).astype(complex)
        omega = 2 * np.pi * freq
        return omega * mu0 / self.wave_impedance(freq)

    def z_parameters(self, freq: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute Z-parameters for waveguide section.

        Z11 = -j * ZTE / tan(kz * L)
        Z21 = -j * ZTE / sin(kz * L)

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]

        Returns
        -------
        Z : dict
            Dictionary with 'Z11', 'Z21', 'Z12', 'Z22', 'frequencies'
        """
        freq = np.atleast_1d(freq).astype(complex)
        ZTE = self.wave_impedance(freq)
        kzL = self.propagation_constant(freq) * self.L

        Z11 = -1j * ZTE / np.tan(kzL)
        Z21 = -1j * ZTE / np.sin(kzL)

        return {
            'Z11': Z11,
            'Z22': Z11.copy(),  # Symmetric
            'Z12': Z21.copy(),
            'Z21': Z21,
            'frequencies': freq.real
        }

    def s_parameters(
        self,
        freq: Union[float, np.ndarray],
        Z0_ref: Union[float, np.ndarray, str] = 'ZTE'
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters for waveguide section.

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]
        Z0_ref : float, ndarray, or str
            Reference impedance for S-parameter conversion.
            - 'ZTE': Use frequency-dependent wave impedance (matched case)
            - float: Use fixed reference impedance
            - ndarray: Use frequency-dependent reference impedance

        Returns
        -------
        S : dict
            Dictionary with 'S11', 'S21', 'S12', 'S22', 'frequencies'
        """
        freq = np.atleast_1d(freq).astype(complex)
        n_freq = len(freq)

        Z = self.z_parameters(freq)
        ZTE = self.wave_impedance(freq)

        if isinstance(Z0_ref, str) and Z0_ref.upper() == 'ZTE':
            Z0_array = ZTE
        elif np.isscalar(Z0_ref):
            Z0_array = np.full(n_freq, Z0_ref, dtype=complex)
        else:
            Z0_array = np.atleast_1d(Z0_ref).astype(complex)

        S11 = np.zeros(n_freq, dtype=complex)
        S21 = np.zeros(n_freq, dtype=complex)

        for i in range(n_freq):
            Z_mat = np.array([
                [Z['Z11'][i], Z['Z12'][i]],
                [Z['Z21'][i], Z['Z22'][i]]
            ])
            S_mat = self._z_to_s_matrix(Z_mat, Z0_array[i])
            S11[i] = S_mat[0, 0]
            S21[i] = S_mat[1, 0]

        return {
            'S11': S11,
            'S22': S11.copy(),  # Symmetric
            'S12': S21.copy(),
            'S21': S21,
            'frequencies': freq.real
        }

    def s_parameters_matched(self, freq: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute S-parameters with matched (wave impedance) reference. S11 = 0."""
        return self.s_parameters(freq, Z0_ref='ZTE')

    def s_parameters_with_reference(
        self,
        freq: Union[float, np.ndarray],
        Z0_ref: float = Z0
    ) -> Dict[str, np.ndarray]:
        """Compute S-parameters with fixed reference impedance."""
        return self.s_parameters(freq, Z0_ref=Z0_ref)

    @staticmethod
    def _z_to_s_matrix(Z: np.ndarray, Z0: complex) -> np.ndarray:
        """Convert 2x2 Z-parameter matrix to S-parameter matrix."""
        n = Z.shape[0]
        I = np.eye(n, dtype=complex)
        Zn = Z - Z0 * I
        Zd = Z + Z0 * I
        return Zn @ np.linalg.inv(Zd)

    def abcd_parameters(self, freq: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute ABCD (transmission) parameters."""
        freq = np.atleast_1d(freq).astype(complex)
        ZTE = self.wave_impedance(freq)
        kzL = self.propagation_constant(freq) * self.L

        A = np.cos(kzL)
        return {
            'A': A,
            'B': 1j * ZTE * np.sin(kzL),
            'C': 1j * np.sin(kzL) / ZTE,
            'D': A.copy(),
            'frequencies': freq.real
        }

    # =========================================================================
    # Resonant / cutoff quantities
    # =========================================================================

    def resonant_frequencies(self, n_modes: int = 10) -> np.ndarray:
        """
        Compute resonant frequencies of the waveguide cavity (TE10p modes only).

        For TE10p modes: f = c/(2*pi) * sqrt((pi/a)^2 + (p*pi/L)^2)

        Parameters
        ----------
        n_modes : int
            Number of modes to compute

        Returns
        -------
        freqs : ndarray
            Resonant frequencies [Hz]
        """
        freqs = []
        for p in range(1, n_modes + 1):
            kz = p * np.pi / self.L
            k = np.sqrt(self.kc**2 + kz**2)
            freqs.append(c0 * k / (2 * np.pi))
        return np.array(freqs)

    # =========================================================================
    # Port / waveguide eigenmode methods (2D cross-section)
    # =========================================================================

    def cutoff_wavenumber(self, m: int, n: int) -> float:
        """
        Compute cutoff wavenumber for TE_mn or TM_mn mode.

        kc_mn = sqrt((m*pi/a)^2 + (n*pi/b)^2)
        """
        return np.sqrt((m * np.pi / self.a)**2 + (n * np.pi / self.b)**2)

    def cutoff_frequency(self, m: int, n: int) -> float:
        """Compute cutoff frequency for TE_mn or TM_mn mode [Hz]."""
        return c0 * self.cutoff_wavenumber(m, n) / (2 * np.pi)

    def port_eigenmodes(
        self,
        n_modes: int = 10,
        max_index: int = 10,
        return_format: str = 'list'
    ) -> Union[List[Dict], Dict[str, Dict], np.ndarray]:
        """
        Compute all waveguide port eigenmodes (2D cross-section modes).

        Mode validity rules:
        - TE_mn: m >= 0, n >= 0, (m,n) != (0,0)
        - TM_mn: m >= 1 and n >= 1

        Parameters
        ----------
        n_modes : int
            Number of modes to return
        max_index : int
            Maximum index to search (for m, n)
        return_format : str
            'list', 'dict', or 'array'

        Returns
        -------
        Depending on return_format:
            list: [{'label': 'TE10', 'type': 'TE', 'm': 1, 'n': 0, 'kc': ..., 'fc': ...}, ...]
            dict: {'TE10': {'type': 'TE', 'm': 1, ...}, ...}
            array: ndarray of cutoff frequencies [Hz]
        """
        modes = []

        for m in range(max_index + 1):
            for n in range(max_index + 1):
                if m == 0 and n == 0:
                    continue

                kc = self.cutoff_wavenumber(m, n)
                fc = c0 * kc / (2 * np.pi)

                modes.append({
                    'label': f'TE{m}{n}',
                    'type': 'TE',
                    'm': m, 'n': n,
                    'kc': kc,
                    'fc_Hz': fc,
                    'fc_GHz': fc / 1e9,
                    'lambda_c': c0 / fc if fc > 0 else np.inf
                })

                if m >= 1 and n >= 1:
                    modes.append({
                        'label': f'TM{m}{n}',
                        'type': 'TM',
                        'm': m, 'n': n,
                        'kc': kc,
                        'fc_Hz': fc,
                        'fc_GHz': fc / 1e9,
                        'lambda_c': c0 / fc if fc > 0 else np.inf
                    })

        modes.sort(key=lambda x: (x['fc_Hz'], 0 if x['type'] == 'TE' else 1, x['m'], x['n']))
        modes = modes[:n_modes]

        if return_format == 'array':
            return np.array([m['fc_Hz'] for m in modes])
        elif return_format == 'dict':
            return {m['label']: m for m in modes}
        else:
            return modes

    def port_eigenmodes_by_type(
        self,
        n_modes: int = 10,
        max_index: int = 10
    ) -> Dict[str, List[Dict]]:
        """Compute port eigenmodes grouped by type (TE and TM)."""
        te_modes, tm_modes = [], []

        for m in range(max_index + 1):
            for n in range(max_index + 1):
                if m == 0 and n == 0:
                    continue

                kc = self.cutoff_wavenumber(m, n)
                fc = c0 * kc / (2 * np.pi)
                base = {
                    'm': m, 'n': n, 'kc': kc,
                    'fc_Hz': fc, 'fc_GHz': fc / 1e9,
                    'lambda_c': c0 / fc if fc > 0 else np.inf
                }

                te = {**base, 'label': f'TE{m}{n}', 'type': 'TE'}
                te_modes.append(te)

                if m >= 1 and n >= 1:
                    tm = {**base, 'label': f'TM{m}{n}', 'type': 'TM'}
                    tm_modes.append(tm)

        te_modes.sort(key=lambda x: (x['fc_Hz'], x['m'], x['n']))
        tm_modes.sort(key=lambda x: (x['fc_Hz'], x['m'], x['n']))

        return {'TE': te_modes[:n_modes], 'TM': tm_modes[:n_modes]}

    def get_port_mode_wave_impedance(
        self,
        m: int,
        n: int,
        mode_type: str,
        freq: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Compute wave impedance for a specific port mode at given frequency.

        TE: Z_TE = s * Z0 / sqrt(s^2 + ωc^2)
        TM: Z_TM = Z0 * sqrt(s^2 + ωc^2) / s
        """
        freq = np.atleast_1d(freq).astype(complex)
        wc = self.cutoff_wavenumber(m, n) * c0
        s = 1j * 2 * np.pi * freq
        sqrt_term = np.sqrt(s**2 + wc**2)

        if mode_type.upper() == 'TE':
            return s * Z0 / sqrt_term
        elif mode_type.upper() == 'TM':
            return Z0 * sqrt_term / s
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")

    def get_port_mode_propagation_constant(
        self,
        m: int,
        n: int,
        freq: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Compute propagation constant for a specific port mode.

        β = sqrt((ω/c)^2 - kc^2)  [real above cutoff, imaginary below]
        """
        freq = np.atleast_1d(freq).astype(complex)
        wc = self.cutoff_wavenumber(m, n) * c0
        s = 1j * 2 * np.pi * freq
        return np.sqrt(s**2 + wc**2) / c0

    def print_port_eigenmodes(self, n_modes: int = 20, max_index: int = 10) -> None:
        """Print a formatted table of port eigenmodes."""
        modes = self.port_eigenmodes(n_modes=n_modes, max_index=max_index)

        print("\n" + "=" * 80)
        print("Rectangular Waveguide Port Eigenmodes")
        print(f"Cross-section: a = {self.a*1e3:.2f} mm, b = {self.b*1e3:.2f} mm")
        print("=" * 80)
        print(f"{'Rank':<6} {'Mode':<10} {'Type':<6} {'(m,n)':<10} "
              f"{'fc [GHz]':<12} {'kc [rad/m]':<14} {'λc [mm]':<12}")
        print("-" * 80)

        for i, mode in enumerate(modes, 1):
            lc_mm = mode['lambda_c'] * 1e3 if mode['lambda_c'] < np.inf else np.inf
            print(f"{i:<6} {mode['label']:<10} {mode['type']:<6} "
                  f"({mode['m']},{mode['n']}){'':>5} "
                  f"{mode['fc_GHz']:<12.4f} {mode['kc']:<14.4f} {lc_mm:<12.2f}")

        print("=" * 80)

        from collections import defaultdict
        groups = defaultdict(list)
        for mode in modes:
            groups[round(mode['fc_GHz'], 6)].append(mode['label'])

        print("\nDegenerate mode groups (same cutoff frequency):")
        print("-" * 40)
        for fc, labels in sorted(groups.items()):
            if len(labels) > 1:
                print(f"  fc = {fc:.4f} GHz: {', '.join(labels)}")
        print("=" * 80)

    # =========================================================================
    # Cavity eigenfrequency methods (3D resonant modes)
    # =========================================================================

    def eigenfrequencies(
        self,
        n_modes: int = 10,
        mode_types: List[str] = None
    ) -> Dict[str, float]:
        """
        Compute eigenfrequencies for specific mode families.

        Parameters
        ----------
        n_modes : int
            Number of modes per type
        mode_types : list of str, optional
            Mode types to compute. Default: ['TE10p']
            Options: 'TE10p', 'TE01p', 'TE11p', 'TM11p', 'TE20p', 'TE02p'

        Returns
        -------
        freqs : dict
            {mode_label: frequency_Hz}
        """
        if mode_types is None:
            mode_types = ['TE10p']

        results = {}
        _families = {
            'TE10p': (self.kc,                                             'TE10', 1),
            'TE01p': (np.pi / self.b,                                      'TE01', 1),
            'TE11p': (np.sqrt((np.pi/self.a)**2 + (np.pi/self.b)**2),     'TE11', 0),
            'TM11p': (np.sqrt((np.pi/self.a)**2 + (np.pi/self.b)**2),     'TM11', 1),
            'TE20p': (2 * np.pi / self.a,                                  'TE20', 0),
            'TE02p': (2 * np.pi / self.b,                                  'TE02', 0),
        }

        for mode_type in mode_types:
            if mode_type not in _families:
                continue
            kc_fam, prefix, p_start = _families[mode_type]
            for p in range(p_start, p_start + n_modes):
                kz = p * np.pi / self.L
                k = np.sqrt(kc_fam**2 + kz**2)
                results[f'{prefix}{p}'] = c0 * k / (2 * np.pi)

        return results

    def all_eigenfrequencies(
        self,
        n_modes: int = 20,
        max_index: int = 10,
        return_format: str = 'dict',
        boundary_type: str = 'PEC'
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Compute all physical eigenfrequencies of the rectangular cavity.

        Parameters
        ----------
        n_modes : int
            Number of eigenfrequencies to return
        max_index : int
            Maximum index to search (for m, n, p)
        return_format : str
            'dict', 'list', or 'array'
        boundary_type : str
            'PEC' (standard) or 'PMC'

        Returns
        -------
        Depending on return_format
        """
        pi_over_a_sq = (np.pi / self.a) ** 2
        pi_over_b_sq = (np.pi / self.b) ** 2
        pi_over_L_sq = (np.pi / self.L) ** 2

        eigenvalue_modes = {}

        for m, n, p in itertools.product(range(max_index + 1), repeat=3):
            if m == 0 and n == 0:
                continue

            k_squared = (m**2 * pi_over_a_sq +
                         n**2 * pi_over_b_sq +
                         p**2 * pi_over_L_sq)

            if k_squared == 0:
                continue

            mode_types = []

            if boundary_type.upper() == 'PMC':
                if m > 0 or n > 0:
                    mode_types.append('TE')
                if m >= 1 and n >= 1 and p >= 1:
                    mode_types.append('TM')
            else:  # PEC
                if (m > 0 or n > 0) and p >= 0:
                    mode_types.append('TE')
                if m >= 1 and n >= 1 and p >= 1:
                    mode_types.append('TM')

            if not mode_types:
                continue

            k_sq_r = round(k_squared, 10)
            if k_sq_r not in eigenvalue_modes:
                eigenvalue_modes[k_sq_r] = []
            for mode_type in mode_types:
                eigenvalue_modes[k_sq_r].append((m, n, p, mode_type))

        sorted_eigenvalues = sorted(eigenvalue_modes.keys())

        if return_format == 'array':
            return np.array([
                c0 * np.sqrt(k_sq) / (2 * np.pi)
                for k_sq in sorted_eigenvalues[:n_modes]
            ])

        elif return_format == 'list':
            result_list = []
            for k_sq in sorted_eigenvalues:
                if len(result_list) >= n_modes:
                    break
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                modes = eigenvalue_modes[k_sq]
                mode_string = ", ".join(
                    f"{mt}{m}{n}{p}"
                    for m, n, p, mt in sorted(modes)
                )
                result_list.append((f, k, mode_string, modes))
            return result_list[:n_modes]

        else:  # 'dict'
            result_dict = {}
            count = 0
            for k_sq in sorted_eigenvalues:
                if count >= n_modes:
                    break
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                for m, n, p, mt in eigenvalue_modes[k_sq]:
                    label = f"{mt}{m}{n}{p}"
                    if label not in result_dict:
                        result_dict[label] = f
                        count += 1
                        if count >= n_modes:
                            break
            return result_dict

    def get_mode_info(self, m: int, n: int, p: int, mode_type: str = 'TE') -> Dict:
        """Get detailed information about a specific cavity mode."""
        kx = m * np.pi / self.a
        ky = n * np.pi / self.b
        kz = p * np.pi / self.L
        kc = np.sqrt(kx**2 + ky**2)
        k  = np.sqrt(kx**2 + ky**2 + kz**2)
        f  = c0 * k / (2 * np.pi)
        fc = c0 * kc / (2 * np.pi)

        return {
            'mode_type': mode_type,
            'indices': (m, n, p),
            'label': f"{mode_type}{m}{n}{p}",
            'frequency_Hz': f,
            'frequency_GHz': f / 1e9,
            'cutoff_frequency_Hz': fc,
            'cutoff_frequency_GHz': fc / 1e9,
            'k': k, 'kc': kc,
            'kx': kx, 'ky': ky, 'kz': kz,
            'wavelength_m': c0 / f if f > 0 else np.inf,
            'cutoff_wavelength_m': c0 / fc if fc > 0 else np.inf,
        }

    def get_port_mode_info(self, m: int, n: int, mode_type: str = 'TE') -> Dict:
        """Get detailed information about a specific port (2D) mode."""
        kc = self.cutoff_wavenumber(m, n)
        fc = c0 * kc / (2 * np.pi)
        lc = c0 / fc if fc > 0 else np.inf

        return {
            'mode_type': mode_type,
            'indices': (m, n),
            'label': f"{mode_type}{m}{n}",
            'kc': kc,
            'fc_Hz': fc,
            'fc_GHz': fc / 1e9,
            'lambda_c_m': lc,
            'lambda_c_mm': lc * 1e3,
        }

    def print_eigenfrequencies(self, n_modes: int = 20, max_index: int = 10) -> None:
        """Print a formatted table of cavity eigenfrequencies."""
        results = self.all_eigenfrequencies(n_modes=n_modes, max_index=max_index,
                                            return_format='list')
        print("\n" + "=" * 90)
        print("Rectangular Cavity Eigenfrequencies")
        print(f"Dimensions: a={self.a*1e3:.1f}mm, b={self.b*1e3:.1f}mm, L={self.L*1e3:.1f}mm")
        print("=" * 90)
        print(f"{'Rank':<6} {'Frequency [GHz]':<18} {'k [rad/m]':<15} {'Modes'}")
        print("-" * 90)
        for i, (f, k, mode_string, _) in enumerate(results, 1):
            print(f"{i:<6} {f/1e9:<18.6f} {k:<15.4f} {mode_string}")
        print("=" * 90)

    # =========================================================================
    # PlotMixin interface — lazy compute + required properties
    # =========================================================================

    def _default_freq_range(self) -> Tuple[float, float]:
        """Physics-based default: 0.5·fc → 2.5·fc, spanning below and above cutoff."""
        return (0.5 * self.fc, 2.5 * self.fc)

    def _ensure_computed(self, n_samples: Optional[int] = None) -> None:
        """Compute S/Z on the stored grid if not already done."""
        if self._S_dict is not None:
            return
        f_min, f_max = self._freq_range or self._default_freq_range()
        self.compute(np.linspace(f_min, f_max, n_samples or self._n_samples))

    def compute(
        self,
        freq: Optional[Union[float, np.ndarray]] = None,
        n_samples: Optional[int] = None,
        Z0_ref: Union[float, str] = 'ZTE',
    ) -> 'RWGAnalytical':
        """
        Evaluate and cache S/Z on a frequency grid.

        Parameters
        ----------
        freq : array-like, optional
            Frequency points [Hz]. If None, uses freq_range from __init__
            (or the physics default 0.5·fc → 2.5·fc).
        n_samples : int, optional
            Number of points. Overrides the instance default.
            If freq is an array, resamples it to n_samples evenly-spaced points.
        Z0_ref : float or str
            Reference impedance for S-parameters (default: 'ZTE').

        Returns
        -------
        self
            Allows chaining: rwg.compute(freq).plot_s()
        """
        if freq is None:
            f_min, f_max = self._freq_range or self._default_freq_range()
            freq = np.linspace(f_min, f_max, n_samples or self._n_samples)
        else:
            freq = np.atleast_1d(freq).astype(float)
            if n_samples is not None:
                freq = np.linspace(freq[0], freq[-1], n_samples)

        self._frequencies = freq

        S = self.s_parameters(freq, Z0_ref=Z0_ref)
        Z = self.z_parameters(freq)

        # Key format matches CSTResult / PlotMixin convention: 'i(mi)j(mj)'
        self._S_dict = {
            '1(1)1(1)': S['S11'],
            '2(1)1(1)': S['S21'],
            '1(1)2(1)': S['S12'],
            '2(1)2(1)': S['S22'],
        }
        self._Z_dict = {
            '1(1)1(1)': Z['Z11'],
            '2(1)1(1)': Z['Z21'],
            '1(1)2(1)': Z['Z12'],
            '2(1)2(1)': Z['Z22'],
        }
        return self

    @property
    def frequencies(self) -> np.ndarray:
        """Frequency grid [Hz] set by compute(). Required by PlotMixin."""
        if self._frequencies is None:
            raise AttributeError("Call compute() or a plot method first.")
        return self._frequencies

    @property
    def S_dict(self) -> Dict[str, np.ndarray]:
        """S-parameters keyed as '1(1)1(1)' etc. Required by PlotMixin."""
        if self._S_dict is None:
            raise AttributeError("Call compute() or a plot method first.")
        return self._S_dict

    @property
    def Z_dict(self) -> Dict[str, np.ndarray]:
        """Z-parameters keyed as '1(1)1(1)' etc. Required by PlotMixin."""
        if self._Z_dict is None:
            raise AttributeError("Call compute() or a plot method first.")
        return self._Z_dict

    def get_resonant_frequencies(self) -> np.ndarray:
        """Exposes resonant frequencies so PlotMixin.plot_eigenvalues() works."""
        return self.resonant_frequencies()

    # PlotMixin overrides — auto-compute, then delegate to super()

    def plot_s(self, *args, n_samples: Optional[int] = None, **kwargs):
        """Auto-compute if needed, then delegate to PlotMixin.plot_s()."""
        self._ensure_computed(n_samples)
        return super().plot_s(*args, **kwargs)

    def plot_z(self, *args, n_samples: Optional[int] = None, **kwargs):
        """Auto-compute if needed, then delegate to PlotMixin.plot_z()."""
        self._ensure_computed(n_samples)
        return super().plot_z(*args, **kwargs)


# =============================================================================
# Comparison utilities
# =============================================================================

def compare_eigenfrequencies(
    fem_freqs: np.ndarray,
    a: float,
    L: float,
    b: float = None,
    n_modes: int = 20,
    tolerance: float = 0.05
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Compare FEM eigenfrequencies with analytical solutions.

    Parameters
    ----------
    fem_freqs : ndarray
        Eigenfrequencies from FEM solver [Hz]
    a : float
        Waveguide width [m]
    L : float
        Waveguide length [m]
    b : float, optional
        Waveguide height [m] (default: a/2)
    n_modes : int
        Number of analytical modes to compute
    tolerance : float
        Relative tolerance for matching (default: 5%)

    Returns
    -------
    matches : dict
    errors : ndarray
    unmatched : ndarray
    """
    analytical = RWGAnalytical(a=a, L=L, b=b)
    anal_results = analytical.all_eigenfrequencies(n_modes=n_modes * 2, return_format='list')

    matches = {}
    errors = []
    matched_fem = set()
    matched_anal = set()

    for fem_idx, fem_f in enumerate(fem_freqs):
        best_match, best_error = None, tolerance

        for anal_idx, (anal_f, k, modes_str, modes) in enumerate(anal_results):
            if anal_idx in matched_anal:
                continue
            rel_error = abs(fem_f - anal_f) / anal_f if anal_f > 0 else np.inf
            if rel_error < best_error:
                best_error = rel_error
                best_match = (anal_idx, anal_f, modes_str, modes)

        if best_match is not None:
            anal_idx, anal_f, modes_str, modes = best_match
            matches[fem_idx] = {
                'mode': modes_str, 'fem_freq': fem_f,
                'anal_freq': anal_f, 'error': best_error, 'indices': modes
            }
            errors.append(best_error)
            matched_fem.add(fem_idx)
            matched_anal.add(anal_idx)

    unmatched = np.array([i for i in range(len(fem_freqs)) if i not in matched_fem])
    return matches, np.array(errors), unmatched


def compare_port_eigenmodes(
    fem_cutoffs: np.ndarray,
    a: float,
    b: float = None,
    n_modes: int = 20,
    tolerance: float = 0.05
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Compare FEM port eigenmode cutoff frequencies with analytical solutions.

    Parameters
    ----------
    fem_cutoffs : ndarray
        Cutoff frequencies from FEM solver [Hz]
    a : float
        Waveguide width [m]
    b : float, optional
        Waveguide height [m] (default: a/2)
    n_modes : int
        Number of analytical modes to compute
    tolerance : float
        Relative tolerance for matching (default: 5%)

    Returns
    -------
    matches : dict
    errors : ndarray
    unmatched : ndarray
    """
    analytical = RWGAnalytical(a=a, L=1.0, b=b)  # L irrelevant for port modes
    anal_modes = analytical.port_eigenmodes(n_modes=n_modes * 2)

    matches = {}
    errors = []
    matched_fem = set()
    matched_anal = set()

    for fem_idx, fem_fc in enumerate(fem_cutoffs):
        best_match, best_error = None, tolerance

        for anal_idx, mode in enumerate(anal_modes):
            if anal_idx in matched_anal:
                continue
            anal_fc = mode['fc_Hz']
            rel_error = abs(fem_fc - anal_fc) / anal_fc if anal_fc > 0 else np.inf
            if rel_error < best_error:
                best_error = rel_error
                best_match = (anal_idx, mode)

        if best_match is not None:
            anal_idx, mode = best_match
            matches[fem_idx] = {
                'mode': mode['label'], 'type': mode['type'],
                'm': mode['m'], 'n': mode['n'],
                'fem_fc': fem_fc, 'anal_fc': mode['fc_Hz'], 'error': best_error
            }
            errors.append(best_error)
            matched_fem.add(fem_idx)
            matched_anal.add(anal_idx)

    unmatched = np.array([i for i in range(len(fem_cutoffs)) if i not in matched_fem])
    return matches, np.array(errors), unmatched


def print_eigenfrequency_comparison(
    fem_freqs: np.ndarray,
    a: float,
    L: float,
    b: float = None,
    n_modes: int = 20
) -> None:
    """Print a formatted comparison of FEM and analytical eigenfrequencies."""
    matches, errors, unmatched = compare_eigenfrequencies(fem_freqs, a, L, b, n_modes)
    b_str = f"{b*1e3:.1f}" if b is not None else f"{a*0.5e3:.1f}"

    print("\n" + "=" * 85)
    print("Eigenfrequency Comparison: FEM vs Analytical")
    print(f"Dimensions: a={a*1e3:.1f}mm, b={b_str}mm, L={L*1e3:.1f}mm")
    print("=" * 85)
    print(f"{'Index':<6} {'FEM [GHz]':<14} {'Analytical [GHz]':<18} "
          f"{'Mode(s)':<25} {'Error [%]':<10}")
    print("-" * 85)

    for idx in sorted(matches.keys()):
        m = matches[idx]
        print(f"{idx:<6} {m['fem_freq']/1e9:<14.4f} {m['anal_freq']/1e9:<18.4f} "
              f"{m['mode']:<25} {m['error']*100:<10.2f}")

    if len(unmatched) > 0:
        print("-" * 85)
        print("Unmatched FEM frequencies:")
        for idx in unmatched[:10]:
            print(f"  Index {idx}: {fem_freqs[idx]/1e9:.4f} GHz")

    if len(errors) > 0:
        print("-" * 85)
        print(f"Matched modes: {len(errors)}/{len(fem_freqs)}")
        print(f"Mean error: {np.mean(errors)*100:.2f}%")
        print(f"Max error:  {np.max(errors)*100:.2f}%")
    print("=" * 85)


def print_port_eigenmode_comparison(
    fem_cutoffs: np.ndarray,
    a: float,
    b: float = None,
    n_modes: int = 20
) -> None:
    """Print a formatted comparison of FEM and analytical port eigenmodes."""
    b_val = b if b is not None else a / 2
    matches, errors, unmatched = compare_port_eigenmodes(fem_cutoffs, a, b, n_modes)

    print("\n" + "=" * 85)
    print("Port Eigenmode Comparison: FEM vs Analytical")
    print(f"Cross-section: a={a*1e3:.2f}mm, b={b_val*1e3:.2f}mm")
    print("=" * 85)
    print(f"{'Index':<6} {'FEM fc [GHz]':<14} {'Anal fc [GHz]':<14} "
          f"{'Mode':<10} {'Type':<6} {'(m,n)':<10} {'Error [%]':<10}")
    print("-" * 85)

    for idx in sorted(matches.keys()):
        m = matches[idx]
        print(f"{idx:<6} {m['fem_fc']/1e9:<14.4f} {m['anal_fc']/1e9:<14.4f} "
              f"{m['mode']:<10} {m['type']:<6} ({m['m']},{m['n']}){'':>4} "
              f"{m['error']*100:<10.2f}")

    if len(unmatched) > 0:
        print("-" * 85)
        print("Unmatched FEM cutoff frequencies:")
        for idx in unmatched[:10]:
            print(f"  Index {idx}: {fem_cutoffs[idx]/1e9:.4f} GHz")

    if len(errors) > 0:
        print("-" * 85)
        print(f"Matched modes: {len(errors)}/{len(fem_cutoffs)}")
        print(f"Mean error: {np.mean(errors)*100:.2f}%")
        print(f"Max error:  {np.max(errors)*100:.2f}%")
    print("=" * 85)