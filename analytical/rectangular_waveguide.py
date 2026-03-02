"""Analytical solutions for rectangular waveguide."""

import numpy as np
from typing import Dict, Union, List, Tuple, Optional
from core.constants import c0, mu0, eps0, Z0
import itertools


class RWGAnalytical:
    """
    Analytical Z and S parameters for rectangular waveguide.

    For TE10 mode in a rectangular waveguide of dimensions a x b x L.

    Parameters
    ----------
    a : float
        Width (broad dimension) [m]
    b : float
        Height [m]
    L : float
        Length [m]
    """

    def __init__(self, a: float, L: float, b: float = None):
        self.a = a
        self.b = b if b is not None else a / 2
        self.L = L

        # Cutoff quantities for TE10 mode
        self.kc = np.pi / self.a  # Cutoff wavenumber
        self.wc = self.kc * c0    # Cutoff angular frequency
        self.fc = c0 / (2 * self.a)  # Cutoff frequency [Hz]

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

        ZTE = s * Z0 / np.sqrt(s**2 + self.wc**2)

        return ZTE

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
        ZTE = self.wave_impedance(freq)

        kz = omega * mu0 / ZTE

        return kz

    def z_parameters(
        self,
        freq: Union[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
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
            Dictionary with 'Z11', 'Z21', 'Z12', 'Z22' arrays
        """
        freq = np.atleast_1d(freq).astype(complex)

        ZTE = self.wave_impedance(freq)
        kz = self.propagation_constant(freq)

        kzL = kz * self.L

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
        Z0_ref : float, ndarray, or 'ZTE'
            Reference impedance for S-parameter conversion.
            - 'ZTE': Use frequency-dependent wave impedance (matched case)
            - float: Use fixed reference impedance
            - ndarray: Use frequency-dependent reference impedance

        Returns
        -------
        S : dict
            Dictionary with 'S11', 'S21', 'S12', 'S22' arrays
        """
        freq = np.atleast_1d(freq).astype(complex)
        n_freq = len(freq)

        # Get Z-parameters
        Z = self.z_parameters(freq)
        ZTE = self.wave_impedance(freq)

        # Determine reference impedance
        if isinstance(Z0_ref, str) and Z0_ref.upper() == 'ZTE':
            Z0_array = ZTE  # Use wave impedance (frequency-dependent)
        elif np.isscalar(Z0_ref):
            Z0_array = np.full(n_freq, Z0_ref, dtype=complex)
        else:
            Z0_array = np.atleast_1d(Z0_ref).astype(complex)

        # Compute S-parameters at each frequency
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

    def s_parameters_matched(
        self,
        freq: Union[float, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters with matched (wave impedance) reference.

        This gives S11 = 0 for a lossless matched line.
        """
        return self.s_parameters(freq, Z0_ref='ZTE')

    def s_parameters_with_reference(
        self,
        freq: Union[float, np.ndarray],
        Z0_ref: float = Z0
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters with fixed reference impedance.
        """
        return self.s_parameters(freq, Z0_ref=Z0_ref)

    @staticmethod
    def _z_to_s_matrix(Z: np.ndarray, Z0: complex) -> np.ndarray:
        """Convert 2x2 Z-parameter matrix to S-parameter matrix."""
        n = Z.shape[0]
        I = np.eye(n, dtype=complex)

        Zn = Z - Z0 * I
        Zd = Z + Z0 * I

        S = Zn @ np.linalg.inv(Zd)
        return S

    def abcd_parameters(self, freq: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute ABCD (transmission) parameters."""
        freq = np.atleast_1d(freq).astype(complex)

        ZTE = self.wave_impedance(freq)
        kz = self.propagation_constant(freq)
        kzL = kz * self.L

        A = np.cos(kzL)
        B = 1j * ZTE * np.sin(kzL)
        C = 1j * np.sin(kzL) / ZTE
        D = A.copy()

        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'frequencies': freq.real
        }

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
            f = c0 * k / (2 * np.pi)
            freqs.append(f)

        return np.array(freqs)

    # =========================================================================
    # Port/Waveguide Eigenmode Methods (2D cross-section modes)
    # =========================================================================

    def cutoff_wavenumber(self, m: int, n: int) -> float:
        """
        Compute cutoff wavenumber for TE_mn or TM_mn mode.

        kc_mn = sqrt((m*pi/a)^2 + (n*pi/b)^2)

        Parameters
        ----------
        m, n : int
            Mode indices

        Returns
        -------
        kc : float
            Cutoff wavenumber [rad/m]
        """
        return np.sqrt((m * np.pi / self.a)**2 + (n * np.pi / self.b)**2)

    def cutoff_frequency(self, m: int, n: int) -> float:
        """
        Compute cutoff frequency for TE_mn or TM_mn mode.

        fc_mn = c0 * kc_mn / (2*pi)

        Parameters
        ----------
        m, n : int
            Mode indices

        Returns
        -------
        fc : float
            Cutoff frequency [Hz]
        """
        kc = self.cutoff_wavenumber(m, n)
        return c0 * kc / (2 * np.pi)

    def port_eigenmodes(
        self,
        n_modes: int = 10,
        max_index: int = 10,
        return_format: str = 'list'
    ) -> Union[List[Dict], Dict[str, Dict], np.ndarray]:
        """
        Compute all waveguide port eigenmodes (2D cross-section modes).

        This computes the cutoff frequencies/wavenumbers for all valid TE and TM
        modes of the rectangular waveguide cross-section, sorted by cutoff frequency.

        Mode validity rules:
        - TE_mn: Valid if m >= 0, n >= 0, and (m,n) != (0,0)
                 At least one of m or n must be > 0
        - TM_mn: Valid if m >= 1 and n >= 1
                 Both indices must be positive

        Parameters
        ----------
        n_modes : int
            Number of modes to return
        max_index : int
            Maximum index to search (for m, n)
        return_format : str
            'list': Return list of mode dictionaries sorted by fc
            'dict': Return {mode_label: info_dict}
            'array': Return array of cutoff frequencies

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
                # Skip (0,0) - no valid mode
                if m == 0 and n == 0:
                    continue

                kc = self.cutoff_wavenumber(m, n)
                fc = c0 * kc / (2 * np.pi)

                # TE modes: valid for any (m,n) != (0,0)
                modes.append({
                    'label': f'TE{m}{n}',
                    'type': 'TE',
                    'm': m,
                    'n': n,
                    'kc': kc,
                    'fc_Hz': fc,
                    'fc_GHz': fc / 1e9,
                    'lambda_c': c0 / fc if fc > 0 else np.inf
                })

                # TM modes: valid only for m >= 1 and n >= 1
                if m >= 1 and n >= 1:
                    modes.append({
                        'label': f'TM{m}{n}',
                        'type': 'TM',
                        'm': m,
                        'n': n,
                        'kc': kc,
                        'fc_Hz': fc,
                        'fc_GHz': fc / 1e9,
                        'lambda_c': c0 / fc if fc > 0 else np.inf
                    })

        # Sort by cutoff frequency, then by mode type (TE before TM), then by indices
        modes.sort(key=lambda x: (x['fc_Hz'], 0 if x['type'] == 'TE' else 1, x['m'], x['n']))

        # Limit to requested number
        modes = modes[:n_modes]

        if return_format == 'array':
            return np.array([m['fc_Hz'] for m in modes])

        elif return_format == 'dict':
            return {m['label']: m for m in modes}

        else:  # 'list'
            return modes

    def port_eigenmodes_by_type(
        self,
        n_modes: int = 10,
        max_index: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Compute port eigenmodes grouped by type (TE and TM).

        Parameters
        ----------
        n_modes : int
            Number of modes per type to return
        max_index : int
            Maximum index to search

        Returns
        -------
        modes : dict
            {'TE': [list of TE modes], 'TM': [list of TM modes]}
            Each list is sorted by cutoff frequency
        """
        te_modes = []
        tm_modes = []

        for m in range(max_index + 1):
            for n in range(max_index + 1):
                if m == 0 and n == 0:
                    continue

                kc = self.cutoff_wavenumber(m, n)
                fc = c0 * kc / (2 * np.pi)

                mode_info = {
                    'm': m,
                    'n': n,
                    'kc': kc,
                    'fc_Hz': fc,
                    'fc_GHz': fc / 1e9,
                    'lambda_c': c0 / fc if fc > 0 else np.inf
                }

                # TE mode
                te_mode = mode_info.copy()
                te_mode['label'] = f'TE{m}{n}'
                te_mode['type'] = 'TE'
                te_modes.append(te_mode)

                # TM mode (only if m >= 1 and n >= 1)
                if m >= 1 and n >= 1:
                    tm_mode = mode_info.copy()
                    tm_mode['label'] = f'TM{m}{n}'
                    tm_mode['type'] = 'TM'
                    tm_modes.append(tm_mode)

        # Sort by cutoff frequency
        te_modes.sort(key=lambda x: (x['fc_Hz'], x['m'], x['n']))
        tm_modes.sort(key=lambda x: (x['fc_Hz'], x['m'], x['n']))

        return {
            'TE': te_modes[:n_modes],
            'TM': tm_modes[:n_modes]
        }

    def get_port_mode_wave_impedance(
        self,
        m: int,
        n: int,
        mode_type: str,
        freq: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Compute wave impedance for a specific port mode at given frequency.

        TE modes: Z_TE = ω * μ0 / β = s * Z0 / sqrt(s^2 + ωc^2)
        TM modes: Z_TM = β / (ω * ε0) = Z0 * sqrt(s^2 + ωc^2) / s

        Parameters
        ----------
        m, n : int
            Mode indices
        mode_type : str
            'TE' or 'TM'
        freq : float or ndarray
            Frequency [Hz]

        Returns
        -------
        Z : ndarray
            Wave impedance (complex)
        """
        freq = np.atleast_1d(freq).astype(complex)

        kc = self.cutoff_wavenumber(m, n)
        wc = kc * c0

        s = 1j * 2 * np.pi * freq
        sqrt_term = np.sqrt(s**2 + wc**2)

        if mode_type.upper() == 'TE':
            Z = s * Z0 / sqrt_term
        elif mode_type.upper() == 'TM':
            Z = Z0 * sqrt_term / s
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")

        return Z

    def get_port_mode_propagation_constant(
        self,
        m: int,
        n: int,
        freq: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Compute propagation constant for a specific port mode.

        β = sqrt(k0^2 - kc^2) = sqrt((ω/c)^2 - kc^2)

        In s-domain: β = sqrt(s^2 + ωc^2) / c0

        Parameters
        ----------
        m, n : int
            Mode indices
        freq : float or ndarray
            Frequency [Hz]

        Returns
        -------
        beta : ndarray
            Propagation constant (complex)
            - Real and positive above cutoff (propagating)
            - Imaginary below cutoff (evanescent)
        """
        freq = np.atleast_1d(freq).astype(complex)

        kc = self.cutoff_wavenumber(m, n)
        wc = kc * c0

        s = 1j * 2 * np.pi * freq
        beta = np.sqrt(s**2 + wc**2) / c0

        return beta

    def print_port_eigenmodes(
        self,
        n_modes: int = 20,
        max_index: int = 10
    ) -> None:
        """
        Print a formatted table of port eigenmodes.

        Parameters
        ----------
        n_modes : int
            Number of modes to display
        max_index : int
            Maximum index to search
        """
        modes = self.port_eigenmodes(
            n_modes=n_modes,
            max_index=max_index,
            return_format='list'
        )

        print("\n" + "=" * 80)
        print(f"Rectangular Waveguide Port Eigenmodes")
        print(f"Cross-section: a = {self.a*1e3:.2f} mm, b = {self.b*1e3:.2f} mm")
        print("=" * 80)
        print(f"{'Rank':<6} {'Mode':<10} {'Type':<6} {'(m,n)':<10} "
              f"{'fc [GHz]':<12} {'kc [rad/m]':<14} {'λc [mm]':<12}")
        print("-" * 80)

        for i, mode in enumerate(modes, 1):
            lambda_c_mm = mode['lambda_c'] * 1e3 if mode['lambda_c'] < np.inf else np.inf
            print(f"{i:<6} {mode['label']:<10} {mode['type']:<6} "
                  f"({mode['m']},{mode['n']}){'':>5} "
                  f"{mode['fc_GHz']:<12.4f} {mode['kc']:<14.4f} "
                  f"{lambda_c_mm:<12.2f}")

        print("=" * 80)

        # Also print degenerate mode groups
        print("\nDegenerate mode groups (same cutoff frequency):")
        print("-" * 40)

        # Group modes by cutoff frequency
        from collections import defaultdict
        groups = defaultdict(list)
        for mode in modes:
            fc_rounded = round(mode['fc_GHz'], 6)
            groups[fc_rounded].append(mode['label'])

        for fc, mode_labels in sorted(groups.items()):
            if len(mode_labels) > 1:
                print(f"  fc = {fc:.4f} GHz: {', '.join(mode_labels)}")

        print("=" * 80)

    # =========================================================================
    # Cavity Eigenfrequency Methods (3D resonant modes)
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

        Returns
        -------
        freqs : dict
            Dictionary mapping mode labels to frequencies [Hz]
        """
        if mode_types is None:
            mode_types = ['TE10p']

        results = {}

        for mode_type in mode_types:
            if mode_type == 'TE10p':
                for p in range(1, n_modes + 1):
                    kz = p * np.pi / self.L
                    k = np.sqrt(self.kc**2 + kz**2)
                    f = c0 * k / (2 * np.pi)
                    results[f'TE10{p}'] = f

            elif mode_type == 'TE01p':
                kc_01 = np.pi / self.b
                for p in range(1, n_modes + 1):
                    kz = p * np.pi / self.L
                    k = np.sqrt(kc_01**2 + kz**2)
                    f = c0 * k / (2 * np.pi)
                    results[f'TE01{p}'] = f

            elif mode_type == 'TE11p':
                kc_11 = np.sqrt((np.pi / self.a)**2 + (np.pi / self.b)**2)
                for p in range(0, n_modes):
                    kz = p * np.pi / self.L
                    k = np.sqrt(kc_11**2 + kz**2)
                    f = c0 * k / (2 * np.pi)
                    results[f'TE11{p}'] = f

            elif mode_type == 'TM11p':
                kc_11 = np.sqrt((np.pi / self.a)**2 + (np.pi / self.b)**2)
                for p in range(1, n_modes + 1):
                    kz = p * np.pi / self.L
                    k = np.sqrt(kc_11**2 + kz**2)
                    f = c0 * k / (2 * np.pi)
                    results[f'TM11{p}'] = f

            elif mode_type == 'TE20p':
                kc_20 = 2 * np.pi / self.a
                for p in range(0, n_modes):
                    kz = p * np.pi / self.L
                    k = np.sqrt(kc_20**2 + kz**2)
                    f = c0 * k / (2 * np.pi)
                    results[f'TE20{p}'] = f

            elif mode_type == 'TE02p':
                kc_02 = 2 * np.pi / self.b
                for p in range(0, n_modes):
                    kz = p * np.pi / self.L
                    k = np.sqrt(kc_02**2 + kz**2)
                    f = c0 * k / (2 * np.pi)
                    results[f'TE02{p}'] = f

        return results

    def all_eigenfrequencies(
        self,
        n_modes: int = 20,
        max_index: int = 10,
        return_format: str = 'dict'
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Compute all physical eigenfrequencies of the rectangular cavity.

        This method computes eigenfrequencies for all valid modes (TE and TM)
        by iterating over all (m, n, p) combinations and filtering out
        non-physical modes where m=0 AND n=0.

        Mode classification:
        - TE modes: m or n can be 0 (but not both), p >= 0 (p=0 allowed for some)
        - TM modes: m >= 1, n >= 1, p >= 1

        Parameters
        ----------
        n_modes : int
            Number of eigenfrequencies to return
        max_index : int
            Maximum index to search (for m, n, p)
        return_format : str
            'dict': Return {mode_label: frequency} dict
            'list': Return [(frequency, k, mode_string, indices)] list
            'array': Return sorted frequency array

        Returns
        -------
        Depending on return_format:
            dict: {mode_label: frequency_Hz}
            list: [(freq_Hz, k, mode_string, [(m,n,p), ...])]
            array: ndarray of frequencies
        """
        # Pre-calculate squared constants
        pi_over_a_sq = (np.pi / self.a) ** 2
        pi_over_b_sq = (np.pi / self.b) ** 2
        pi_over_L_sq = (np.pi / self.L) ** 2

        # Store unique eigenvalues with their mode indices
        # Key: eigenvalue (k^2), Value: list of (m, n, p, mode_type) tuples
        eigenvalue_modes = {}

        for m, n, p in itertools.product(range(max_index), repeat=3):
            # Filter: Physical modes must have valid transverse indices
            # TE modes: (m,n) != (0,0), at least one of m,n > 0
            # TM modes: m >= 1, n >= 1

            if m == 0 and n == 0:
                # No valid waveguide mode exists for (0,0) transverse indices
                continue

            # Calculate eigenvalue k^2
            k_squared = (m ** 2 * pi_over_a_sq +
                        n ** 2 * pi_over_b_sq +
                        p ** 2 * pi_over_L_sq)

            # Determine mode types for this (m, n, p)
            mode_types = []

            # TE modes: require m or n > 0 (already satisfied by filter above)
            # For TE_mnp, we need kc^2 = (m*pi/a)^2 + (n*pi/b)^2 > 0
            # p can be 0 for TE modes (gives cutoff mode)
            if m > 0 or n > 0:
                mode_types.append('TE')

            # TM modes: require m >= 1 AND n >= 1 AND p >= 1
            # TM modes have Ez != 0, which requires sin(m*pi*x/a)*sin(n*pi*y/b)*sin(p*pi*z/L)
            if m >= 1 and n >= 1 and p >= 1:
                mode_types.append('TM')

            # Round to handle floating point precision
            k_squared_rounded = round(k_squared, 10)

            if k_squared_rounded not in eigenvalue_modes:
                eigenvalue_modes[k_squared_rounded] = []

            for mode_type in mode_types:
                eigenvalue_modes[k_squared_rounded].append((m, n, p, mode_type))

        # Sort by eigenvalue
        sorted_eigenvalues = sorted(eigenvalue_modes.keys())

        # Build result based on format
        if return_format == 'array':
            frequencies = []
            for k_sq in sorted_eigenvalues[:n_modes]:
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                frequencies.append(f)
            return np.array(frequencies)

        elif return_format == 'list':
            result_list = []
            for k_sq in sorted_eigenvalues:
                if len(result_list) >= n_modes:
                    break
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                modes = eigenvalue_modes[k_sq]

                # Format mode string
                mode_strs = []
                for m, n, p, mode_type in sorted(modes):
                    mode_strs.append(f"{mode_type}{m}{n}{p}")
                mode_string = ", ".join(mode_strs)

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
                modes = eigenvalue_modes[k_sq]

                # Create labels for each mode
                for m, n, p, mode_type in modes:
                    label = f"{mode_type}{m}{n}{p}"
                    if label not in result_dict:
                        result_dict[label] = f
                        count += 1
                        if count >= n_modes:
                            break

            return result_dict

    def get_mode_info(
        self,
        m: int,
        n: int,
        p: int,
        mode_type: str = 'TE'
    ) -> Dict:
        """
        Get detailed information about a specific cavity mode.

        Parameters
        ----------
        m, n, p : int
            Mode indices
        mode_type : str
            'TE' or 'TM'

        Returns
        -------
        info : dict
            Mode information including frequency, wavenumbers, etc.
        """
        kx = m * np.pi / self.a
        ky = n * np.pi / self.b
        kz = p * np.pi / self.L

        kc = np.sqrt(kx**2 + ky**2)  # Cutoff wavenumber
        k = np.sqrt(kx**2 + ky**2 + kz**2)  # Total wavenumber

        f = c0 * k / (2 * np.pi)  # Resonant frequency
        fc = c0 * kc / (2 * np.pi)  # Cutoff frequency

        # Wavelengths
        lambda_0 = c0 / f if f > 0 else np.inf
        lambda_c = c0 / fc if fc > 0 else np.inf

        return {
            'mode_type': mode_type,
            'indices': (m, n, p),
            'label': f"{mode_type}{m}{n}{p}",
            'frequency_Hz': f,
            'frequency_GHz': f / 1e9,
            'cutoff_frequency_Hz': fc,
            'cutoff_frequency_GHz': fc / 1e9,
            'k': k,
            'kc': kc,
            'kx': kx,
            'ky': ky,
            'kz': kz,
            'wavelength_m': lambda_0,
            'cutoff_wavelength_m': lambda_c,
        }

    def get_port_mode_info(
        self,
        m: int,
        n: int,
        mode_type: str = 'TE'
    ) -> Dict:
        """
        Get detailed information about a specific port (2D) mode.

        Parameters
        ----------
        m, n : int
            Mode indices
        mode_type : str
            'TE' or 'TM'

        Returns
        -------
        info : dict
            Mode information including cutoff frequency, wavenumber, etc.
        """
        kc = self.cutoff_wavenumber(m, n)
        fc = c0 * kc / (2 * np.pi)
        lambda_c = c0 / fc if fc > 0 else np.inf

        return {
            'mode_type': mode_type,
            'indices': (m, n),
            'label': f"{mode_type}{m}{n}",
            'kc': kc,
            'fc_Hz': fc,
            'fc_GHz': fc / 1e9,
            'lambda_c_m': lambda_c,
            'lambda_c_mm': lambda_c * 1e3,
        }

    def print_eigenfrequencies(
        self,
        n_modes: int = 20,
        max_index: int = 10
    ) -> None:
        """
        Print a formatted table of cavity eigenfrequencies.

        Parameters
        ----------
        n_modes : int
            Number of modes to display
        max_index : int
            Maximum index to search
        """
        results = self.all_eigenfrequencies(
            n_modes=n_modes,
            max_index=max_index,
            return_format='list'
        )

        print("\n" + "=" * 90)
        print(f"Rectangular Cavity Eigenfrequencies")
        print(f"Dimensions: a={self.a*1e3:.1f}mm, b={self.b*1e3:.1f}mm, L={self.L*1e3:.1f}mm")
        print("=" * 90)
        print(f"{'Rank':<6} {'Frequency [GHz]':<18} {'k [rad/m]':<15} {'Modes'}")
        print("-" * 90)

        for i, (f, k, mode_string, modes) in enumerate(results, 1):
            print(f"{i:<6} {f/1e9:<18.6f} {k:<15.4f} {mode_string}")

        print("=" * 90)


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
        Dictionary mapping FEM frequency indices to analytical mode labels
    errors : ndarray
        Relative errors for matched modes
    unmatched : ndarray
        FEM frequency indices that were not matched
    """
    analytical = RWGAnalytical(a=a, L=L, b=b)

    # Get all eigenfrequencies
    anal_results = analytical.all_eigenfrequencies(
        n_modes=n_modes * 2,  # Get extra to ensure coverage
        return_format='list'
    )

    matches = {}
    errors = []
    matched_fem_indices = set()
    matched_anal_indices = set()

    # Build analytical frequency list with mode info
    anal_freqs = [(f, k, modes_str, modes) for f, k, modes_str, modes in anal_results]

    for fem_idx, fem_f in enumerate(fem_freqs):
        best_match = None
        best_error = tolerance

        for anal_idx, (anal_f, k, modes_str, modes) in enumerate(anal_freqs):
            if anal_idx in matched_anal_indices:
                continue

            rel_error = abs(fem_f - anal_f) / anal_f if anal_f > 0 else np.inf

            if rel_error < best_error:
                best_error = rel_error
                best_match = (anal_idx, anal_f, modes_str, modes)

        if best_match is not None:
            anal_idx, anal_f, modes_str, modes = best_match
            matches[fem_idx] = {
                'mode': modes_str,
                'fem_freq': fem_f,
                'anal_freq': anal_f,
                'error': best_error,
                'indices': modes
            }
            errors.append(best_error)
            matched_fem_indices.add(fem_idx)
            matched_anal_indices.add(anal_idx)

    # Find unmatched FEM frequencies
    unmatched = np.array([i for i in range(len(fem_freqs)) if i not in matched_fem_indices])

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
        Dictionary mapping FEM frequency indices to analytical mode labels
    errors : ndarray
        Relative errors for matched modes
    unmatched : ndarray
        FEM frequency indices that were not matched
    """
    analytical = RWGAnalytical(a=a, L=1.0, b=b)  # L doesn't matter for port modes

    # Get port eigenmodes
    anal_modes = analytical.port_eigenmodes(
        n_modes=n_modes * 2,
        return_format='list'
    )

    matches = {}
    errors = []
    matched_fem_indices = set()
    matched_anal_indices = set()

    for fem_idx, fem_fc in enumerate(fem_cutoffs):
        best_match = None
        best_error = tolerance

        for anal_idx, mode in enumerate(anal_modes):
            if anal_idx in matched_anal_indices:
                continue

            anal_fc = mode['fc_Hz']
            rel_error = abs(fem_fc - anal_fc) / anal_fc if anal_fc > 0 else np.inf

            if rel_error < best_error:
                best_error = rel_error
                best_match = (anal_idx, mode)

        if best_match is not None:
            anal_idx, mode = best_match
            matches[fem_idx] = {
                'mode': mode['label'],
                'type': mode['type'],
                'm': mode['m'],
                'n': mode['n'],
                'fem_fc': fem_fc,
                'anal_fc': mode['fc_Hz'],
                'error': best_error
            }
            errors.append(best_error)
            matched_fem_indices.add(fem_idx)
            matched_anal_indices.add(anal_idx)

    unmatched = np.array([i for i in range(len(fem_cutoffs)) if i not in matched_fem_indices])

    return matches, np.array(errors), unmatched


def print_eigenfrequency_comparison(
    fem_freqs: np.ndarray,
    a: float,
    L: float,
    b: float = None,
    n_modes: int = 20
) -> None:
    """
    Print a formatted comparison of FEM and analytical eigenfrequencies.

    Parameters
    ----------
    fem_freqs : ndarray
        Eigenfrequencies from FEM solver [Hz]
    a, L, b : float
        Waveguide dimensions [m]
    n_modes : int
        Number of modes to compare
    """
    matches, errors, unmatched = compare_eigenfrequencies(
        fem_freqs, a, L, b, n_modes
    )

    print("\n" + "=" * 85)
    print("Eigenfrequency Comparison: FEM vs Analytical")
    print(f"Dimensions: a={a*1e3:.1f}mm, b={b*1e3 if b else a*0.5e3:.1f}mm, L={L*1e3:.1f}mm")
    print("=" * 85)
    print(f"{'Index':<6} {'FEM [GHz]':<14} {'Analytical [GHz]':<18} {'Mode(s)':<25} {'Error [%]':<10}")
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
    """
    Print a formatted comparison of FEM and analytical port eigenmodes.

    Parameters
    ----------
    fem_cutoffs : ndarray
        Cutoff frequencies from FEM solver [Hz]
    a, b : float
        Waveguide cross-section dimensions [m]
    n_modes : int
        Number of modes to compare
    """
    b_val = b if b is not None else a / 2
    matches, errors, unmatched = compare_port_eigenmodes(
        fem_cutoffs, a, b, n_modes
    )

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