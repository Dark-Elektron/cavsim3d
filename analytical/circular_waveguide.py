"""Analytical solutions for circular waveguide."""

from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from scipy.special import jn_zeros, jnp_zeros

from core.constants import mu0, eps0, c0, Z0

# Free space impedance
ETA0 = np.sqrt(mu0 / eps0)  # ~377 Ω


class CWGAnalytical:
    """
    Analytical Z and S parameters for circular waveguide.

    For TE11 mode (fundamental) in a circular waveguide of radius a and length L.

    Parameters
    ----------
    radius : float
        Waveguide radius [m]
    length : float
        Waveguide length [m]

    Notes
    -----
    TE modes: E_z = 0, cutoff from J'_n(p'_nm) = 0
    TM modes: H_z = 0, cutoff from J_n(p_nm) = 0

    Cutoff wavenumber: k_c = p_nm / a  (TM) or p'_nm / a (TE)
    Cutoff frequency: f_c = c * k_c / (2*pi)

    The fundamental mode is TE11 with p'_11 ≈ 1.8412
    """

    # Cache for computed Bessel zeros to avoid repeated scipy calls
    _bessel_zeros_cache: Dict[Tuple[int, int], float] = {}
    _bessel_derivative_zeros_cache: Dict[Tuple[int, int], float] = {}

    def __init__(self, radius: float, length: float):
        self.radius = radius
        self.length = length
        self.a = radius  # Alias for consistency
        self.L = length  # Alias for consistency

        # Cutoff quantities for TE11 mode (fundamental)
        self.p_prime_11 = self.get_bessel_derivative_zero(1, 1)  # ≈ 1.8412
        self.kc = self.p_prime_11 / self.radius  # Cutoff wavenumber
        self.wc = self.kc * c0  # Cutoff angular frequency
        self.fc = c0 * self.kc / (2 * np.pi)  # Cutoff frequency [Hz]

    @classmethod
    def get_bessel_zero(cls, n: int, m: int) -> float:
        """
        Get m-th zero of J_n(x) for TM modes.

        Uses scipy.special.jn_zeros with caching for efficiency.

        Parameters
        ----------
        n : int
            Order of Bessel function (n >= 0)
        m : int
            Which zero to return (m >= 1)

        Returns
        -------
        float
            The m-th zero of J_n(x)
        """
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")

        key = (n, m)
        if key not in cls._bessel_zeros_cache:
            # Compute all zeros up to m for efficiency
            zeros = jn_zeros(n, m)
            for i, z in enumerate(zeros, 1):
                cls._bessel_zeros_cache[(n, i)] = float(z)

        return cls._bessel_zeros_cache[key]

    @classmethod
    def get_bessel_derivative_zero(cls, n: int, m: int) -> float:
        """
        Get m-th zero of J'_n(x) for TE modes.

        Uses scipy.special.jnp_zeros with caching for efficiency.

        Parameters
        ----------
        n : int
            Order of Bessel function (n >= 0)
        m : int
            Which zero to return (m >= 1)

        Returns
        -------
        float
            The m-th zero of J'_n(x)
        """
        if m < 1:
            raise ValueError(f"m must be >= 1, got {m}")
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")

        key = (n, m)
        if key not in cls._bessel_derivative_zeros_cache:
            # Compute all zeros up to m for efficiency
            zeros = jnp_zeros(n, m)
            for i, z in enumerate(zeros, 1):
                cls._bessel_derivative_zeros_cache[(n, i)] = float(z)

        return cls._bessel_derivative_zeros_cache[key]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the Bessel zeros cache."""
        cls._bessel_zeros_cache.clear()
        cls._bessel_derivative_zeros_cache.clear()

    # =========================================================================
    # Cutoff Properties
    # =========================================================================

    def cutoff_frequency_TE(self, n: int, m: int) -> float:
        """
        Cutoff frequency for TE_nm mode [Hz].

        Parameters
        ----------
        n : int
            Azimuthal mode number (n >= 0)
        m : int
            Radial mode number (m >= 1)
        """
        p_prime_nm = self.get_bessel_derivative_zero(n, m)
        return c0 * p_prime_nm / (2 * np.pi * self.radius)

    def cutoff_frequency_TM(self, n: int, m: int) -> float:
        """
        Cutoff frequency for TM_nm mode [Hz].

        Parameters
        ----------
        n : int
            Azimuthal mode number (n >= 0)
        m : int
            Radial mode number (m >= 1)
        """
        p_nm = self.get_bessel_zero(n, m)
        return c0 * p_nm / (2 * np.pi * self.radius)

    def cutoff_wavenumber_TE(self, n: int, m: int) -> float:
        """Cutoff wavenumber for TE_nm mode [rad/m]."""
        p_prime_nm = self.get_bessel_derivative_zero(n, m)
        return p_prime_nm / self.radius

    def cutoff_wavenumber_TM(self, n: int, m: int) -> float:
        """Cutoff wavenumber for TM_nm mode [rad/m]."""
        p_nm = self.get_bessel_zero(n, m)
        return p_nm / self.radius

    # =========================================================================
    # Wave Impedance (for TE11 fundamental mode by default)
    # =========================================================================

    def wave_impedance(
        self,
        freq: Union[float, np.ndarray],
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> np.ndarray:
        """
        Compute wave impedance for specified mode.

        For TE modes: Z_TE = jωμ₀/kz = η₀·k₀/kz
        For TM modes: Z_TM = kz/(jωε₀) = η₀·kz/k₀

        Using Laplace variable s = jω:
            Z_TE = s·η₀ / sqrt(s² + ωc²)

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]
        mode_type : str
            'TE' or 'TM'
        n, m : int
            Mode indices (default: TE11)

        Returns
        -------
        Z : ndarray
            Wave impedance (complex)
        """
        freq = np.atleast_1d(freq).astype(complex)
        omega = 2 * np.pi * freq
        s = 1j * omega

        if mode_type.upper() == 'TE':
            kc = self.cutoff_wavenumber_TE(n, m)
        else:
            kc = self.cutoff_wavenumber_TM(n, m)

        wc = kc * c0

        if mode_type.upper() == 'TE':
            # Z_TE = s·η₀ / sqrt(s² + ωc²)
            Z = s * ETA0 / np.sqrt(s**2 + wc**2)
        else:
            # Z_TM = η₀·sqrt(s² + ωc²) / s
            Z = ETA0 * np.sqrt(s**2 + wc**2) / s

        return Z

    def wave_impedance_TE11(self, freq: Union[float, np.ndarray]) -> np.ndarray:
        """Wave impedance for fundamental TE11 mode."""
        return self.wave_impedance(freq, mode_type='TE', n=1, m=1)

    # =========================================================================
    # Propagation Constant
    # =========================================================================

    def propagation_constant(
        self,
        freq: Union[float, np.ndarray],
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> np.ndarray:
        """
        Compute propagation constant kz.

        kz = sqrt(k₀² - kc²) for propagating modes
        kz = j·sqrt(kc² - k₀²) for evanescent modes

        Or using: kz = ω·μ₀ / Z_TE

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]
        mode_type : str
            'TE' or 'TM'
        n, m : int
            Mode indices

        Returns
        -------
        kz : ndarray
            Propagation constant (complex)
        """
        freq = np.atleast_1d(freq).astype(complex)
        omega = 2 * np.pi * freq
        Z = self.wave_impedance(freq, mode_type, n, m)

        if mode_type.upper() == 'TE':
            kz = omega * mu0 / Z
        else:
            kz = omega * eps0 * Z

        return kz

    def propagation_constant_TE11(self, freq: Union[float, np.ndarray]) -> np.ndarray:
        """Propagation constant for fundamental TE11 mode."""
        return self.propagation_constant(freq, mode_type='TE', n=1, m=1)

    # =========================================================================
    # Z-Parameters
    # =========================================================================

    def z_parameters(
        self,
        freq: Union[float, np.ndarray],
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute Z-parameters for waveguide section.

        For a transmission line:
            Z11 = -j·Z_w / tan(kz·L)
            Z21 = -j·Z_w / sin(kz·L)

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]
        mode_type : str
            'TE' or 'TM'
        n, m : int
            Mode indices (default: TE11)

        Returns
        -------
        Z : dict
            Dictionary with 'Z11', 'Z21', 'Z12', 'Z22', 'frequencies'
        """
        freq = np.atleast_1d(freq).astype(complex)

        Zw = self.wave_impedance(freq, mode_type, n, m)
        kz = self.propagation_constant(freq, mode_type, n, m)

        kzL = kz * self.length

        Z11 = -1j * Zw / np.tan(kzL)
        Z21 = -1j * Zw / np.sin(kzL)

        return {
            'Z11': Z11,
            'Z22': Z11.copy(),  # Symmetric
            'Z12': Z21.copy(),
            'Z21': Z21,
            'frequencies': freq.real,
            'mode': f'{mode_type}{n}{m}'
        }

    def z_parameters_TE11(self, freq: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """Z-parameters for fundamental TE11 mode."""
        return self.z_parameters(freq, mode_type='TE', n=1, m=1)

    # =========================================================================
    # S-Parameters
    # =========================================================================

    def s_parameters(
        self,
        freq: Union[float, np.ndarray],
        Z0_ref: Union[float, np.ndarray, str] = 'Zw',
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters for waveguide section.

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]
        Z0_ref : float, ndarray, or str
            Reference impedance for S-parameter conversion.
            - 'Zw', 'ZTE', 'ZTM', 'wave', 'matched': Use frequency-dependent wave impedance
            - float: Use fixed reference impedance
            - ndarray: Use frequency-dependent reference impedance
        mode_type : str
            'TE' or 'TM'
        n, m : int
            Mode indices (default: TE11)

        Returns
        -------
        S : dict
            Dictionary with 'S11', 'S21', 'S12', 'S22', 'frequencies'
        """
        freq = np.atleast_1d(freq).astype(complex)
        n_freq = len(freq)

        # Get Z-parameters
        Z = self.z_parameters(freq, mode_type, n, m)
        Zw = self.wave_impedance(freq, mode_type, n, m)

        # Determine reference impedance
        if isinstance(Z0_ref, str):
            wave_impedance_options = {'zw', 'zte', 'ztm', 'wave', 'matched'}
            if Z0_ref.lower() in wave_impedance_options:
                Z0_array = Zw
            else:
                raise ValueError(
                    f"Unknown Z0_ref string: '{Z0_ref}'. "
                    f"Valid string options: 'Zw', 'ZTE', 'ZTM', 'wave', 'matched', "
                    f"or provide a numeric value."
                )
        elif np.isscalar(Z0_ref):
            Z0_array = np.full(n_freq, complex(Z0_ref), dtype=complex)
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
            'frequencies': freq.real,
            'mode': f'{mode_type}{n}{m}'
        }

    def s_parameters_TE11(
        self,
        freq: Union[float, np.ndarray],
        Z0_ref: Union[float, np.ndarray, str] = 'Zw'
    ) -> Dict[str, np.ndarray]:
        """S-parameters for fundamental TE11 mode."""
        return self.s_parameters(freq, Z0_ref, mode_type='TE', n=1, m=1)

    def s_parameters_matched(
        self,
        freq: Union[float, np.ndarray],
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters with matched (wave impedance) reference.

        This gives S11 = 0 for a lossless matched line.
        """
        return self.s_parameters(freq, Z0_ref='Zw', mode_type=mode_type, n=n, m=m)

    def s_parameters_with_reference(
        self,
        freq: Union[float, np.ndarray],
        Z0_ref: float = ETA0,
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters with fixed reference impedance.
        """
        return self.s_parameters(freq, Z0_ref=Z0_ref, mode_type=mode_type, n=n, m=m)

    @staticmethod
    def _z_to_s_matrix(Z: np.ndarray, Z0: complex) -> np.ndarray:
        """Convert 2x2 Z-parameter matrix to S-parameter matrix."""
        n = Z.shape[0]
        I = np.eye(n, dtype=complex)

        Zn = Z - Z0 * I
        Zd = Z + Z0 * I

        S = Zn @ np.linalg.inv(Zd)
        return S

    # =========================================================================
    # ABCD Parameters
    # =========================================================================

    def abcd_parameters(
        self,
        freq: Union[float, np.ndarray],
        mode_type: str = 'TE',
        n: int = 1,
        m: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Compute ABCD (transmission) parameters.

        For a transmission line:
            A = cos(kz·L)
            B = j·Zw·sin(kz·L)
            C = j·sin(kz·L)/Zw
            D = cos(kz·L)

        Parameters
        ----------
        freq : float or ndarray
            Frequency [Hz]
        mode_type : str
            'TE' or 'TM'
        n, m : int
            Mode indices

        Returns
        -------
        ABCD : dict
            Dictionary with 'A', 'B', 'C', 'D', 'frequencies'
        """
        freq = np.atleast_1d(freq).astype(complex)

        Zw = self.wave_impedance(freq, mode_type, n, m)
        kz = self.propagation_constant(freq, mode_type, n, m)
        kzL = kz * self.length

        A = np.cos(kzL)
        B = 1j * Zw * np.sin(kzL)
        C = 1j * np.sin(kzL) / Zw
        D = A.copy()

        return {
            'A': A,
            'B': B,
            'C': C,
            'D': D,
            'frequencies': freq.real,
            'mode': f'{mode_type}{n}{m}'
        }

    def abcd_parameters_TE11(self, freq: Union[float, np.ndarray]) -> Dict[str, np.ndarray]:
        """ABCD parameters for fundamental TE11 mode."""
        return self.abcd_parameters(freq, mode_type='TE', n=1, m=1)

    # =========================================================================
    # Eigenfrequencies
    # =========================================================================

    def resonant_frequencies(self, n_modes: int = 10) -> np.ndarray:
        """
        Compute resonant frequencies of the waveguide cavity (TE11p modes only).

        For TE11p modes: f = c/(2π) * sqrt((p'_11/a)² + (p·π/L)²)

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
            kz = p * np.pi / self.length
            k = np.sqrt(self.kc**2 + kz**2)
            f = c0 * k / (2 * np.pi)
            freqs.append(f)

        return np.array(freqs)

    def eigenfrequencies(
        self,
        n_modes: int = 10,
        mode_types: Optional[List[str]] = None,
        n_max: int = 5,
        m_max: int = 5,
        p_max: int = 10,
        include_cutoff_modes: bool = True,
        return_format: str = 'dict'
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Compute eigenfrequencies for circular cavity modes.

        By default, computes ALL valid TE and TM modes sorted by frequency.
        Can be restricted to specific mode families via mode_types parameter.

        Mode classification for circular waveguide cavity:
        - TE_nmp: n >= 0, m >= 1, p >= 0 (p=0 gives cutoff mode)
        - TM_nmp: n >= 0, m >= 1, p >= 1

        Parameters
        ----------
        n_modes : int
            Number of eigenfrequencies to return (default: 10)
        mode_types : list of str, optional
            Mode types to compute. Options:
            - None or ['all']: All TE and TM modes (default)
            - ['TE']: All TE modes
            - ['TM']: All TM modes
            - ['TE11p']: Only TE11p family (fundamental)
            - ['TE01p']: Only TE01p family
            - ['TM01p']: Only TM01p family
            - Or any combination: ['TE11p', 'TM01p', 'TE21p']
        n_max : int
            Maximum azimuthal index to search (default: 5)
        m_max : int
            Maximum radial index to search (default: 5)
        p_max : int
            Maximum longitudinal index to search (default: 10)
        include_cutoff_modes : bool
            If True, include modes with p=0 (at cutoff frequency)
        return_format : str
            Output format:
            - 'dict': Return {mode_label: frequency_Hz} dict (default)
            - 'list': Return [(freq_Hz, k, mode_string, indices)] list
            - 'array': Return sorted frequency array only

        Returns
        -------
        Depending on return_format:
            dict: {mode_label: frequency_Hz}
            list: [(freq_Hz, k, mode_string, [(n,m,p,mode_type), ...])]
            array: ndarray of frequencies [Hz]

        Examples
        --------
        >>> cwg = CWGAnalytical(radius=0.01, length=0.05)
        >>> # Get all modes
        >>> freqs = cwg.eigenfrequencies(n_modes=20)
        >>> # Get only TE modes
        >>> te_freqs = cwg.eigenfrequencies(n_modes=10, mode_types=['TE'])
        >>> # Get specific families
        >>> families = cwg.eigenfrequencies(n_modes=10, mode_types=['TE11p', 'TM01p'])
        >>> # Get as sorted array
        >>> freq_array = cwg.eigenfrequencies(n_modes=10, return_format='array')
        """
        # Determine which modes to compute
        if mode_types is None or mode_types == ['all']:
            compute_all = True
            specific_families = []
        else:
            compute_all = False
            # Check if requesting general TE/TM or specific families
            general_types = [t for t in mode_types if t in ['TE', 'TM']]
            specific_families = [t for t in mode_types if t not in ['TE', 'TM']]

            if 'TE' in general_types and 'TM' in general_types:
                compute_all = True
            elif 'TE' in general_types:
                return self._compute_general_modes(
                    n_modes, n_max, m_max, p_max, include_cutoff_modes,
                    return_format, te_only=True
                )
            elif 'TM' in general_types:
                return self._compute_general_modes(
                    n_modes, n_max, m_max, p_max, include_cutoff_modes,
                    return_format, tm_only=True
                )

        if compute_all:
            return self._compute_general_modes(
                n_modes, n_max, m_max, p_max, include_cutoff_modes, return_format
            )

        # Handle specific mode families
        if specific_families:
            return self._compute_specific_families(
                n_modes, specific_families, p_max, include_cutoff_modes, return_format
            )

        # Default: all modes
        return self._compute_general_modes(
            n_modes, n_max, m_max, p_max, include_cutoff_modes, return_format
        )

    def _compute_general_modes(
        self,
        n_modes: int,
        n_max: int,
        m_max: int,
        p_max: int,
        include_cutoff_modes: bool,
        return_format: str,
        te_only: bool = False,
        tm_only: bool = False
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Compute all physical eigenfrequencies by iterating over (n, m, p).

        Parameters
        ----------
        n_modes : int
            Number of eigenfrequencies to return
        n_max : int
            Maximum azimuthal index
        m_max : int
            Maximum radial index
        p_max : int
            Maximum longitudinal index
        include_cutoff_modes : bool
            Include p=0 modes (at cutoff)
        return_format : str
            'dict', 'list', or 'array'
        te_only : bool
            If True, only return TE modes
        tm_only : bool
            If True, only return TM modes

        Returns
        -------
        Eigenfrequencies in requested format
        """
        # Store unique eigenvalues with their mode info
        # Key: k² (rounded), Value: list of (n, m, p, mode_type) tuples
        eigenvalue_modes = {}

        p_start = 0 if include_cutoff_modes else 1

        for n in range(n_max + 1):
            for m in range(1, m_max + 1):
                for p in range(p_start, p_max + 1):
                    # TE modes: n >= 0, m >= 1, p >= 0
                    if not tm_only:
                        try:
                            kc_te = self.cutoff_wavenumber_TE(n, m)
                            kz = p * np.pi / self.length
                            k_squared = kc_te**2 + kz**2

                            # Skip zero frequency
                            if k_squared < 1e-20:
                                continue

                            k_squared_rounded = round(k_squared, 10)

                            if k_squared_rounded not in eigenvalue_modes:
                                eigenvalue_modes[k_squared_rounded] = []

                            eigenvalue_modes[k_squared_rounded].append((n, m, p, 'TE'))
                        except Exception:
                            pass

                    # TM modes: n >= 0, m >= 1, p >= 1
                    if p >= 1 and not te_only:
                        try:
                            kc_tm = self.cutoff_wavenumber_TM(n, m)
                            kz = p * np.pi / self.length
                            k_squared = kc_tm**2 + kz**2

                            k_squared_rounded = round(k_squared, 10)

                            if k_squared_rounded not in eigenvalue_modes:
                                eigenvalue_modes[k_squared_rounded] = []

                            eigenvalue_modes[k_squared_rounded].append((n, m, p, 'TM'))
                        except Exception:
                            pass

        # Sort by eigenvalue
        sorted_eigenvalues = sorted(eigenvalue_modes.keys())

        return self._format_eigenfrequency_results(
            sorted_eigenvalues, eigenvalue_modes, n_modes, return_format
        )

    def _compute_specific_families(
        self,
        n_modes: int,
        families: List[str],
        p_max: int,
        include_cutoff_modes: bool,
        return_format: str
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Compute eigenfrequencies for specific mode families.

        Parameters
        ----------
        n_modes : int
            Number of modes to return
        families : list of str
            Mode family names like 'TE11p', 'TM01p', etc.
        p_max : int
            Maximum longitudinal index
        include_cutoff_modes : bool
            Include p=0 modes
        return_format : str
            'dict', 'list', or 'array'

        Returns
        -------
        Eigenfrequencies in requested format
        """
        eigenvalue_modes = {}

        for family in families:
            modes = self._get_family_modes(family, p_max, include_cutoff_modes)
            for k_sq, mode_info in modes.items():
                k_sq_rounded = round(k_sq, 10)
                if k_sq_rounded not in eigenvalue_modes:
                    eigenvalue_modes[k_sq_rounded] = []
                eigenvalue_modes[k_sq_rounded].extend(mode_info)

        sorted_eigenvalues = sorted(eigenvalue_modes.keys())

        return self._format_eigenfrequency_results(
            sorted_eigenvalues, eigenvalue_modes, n_modes, return_format
        )

    def _get_family_modes(
        self,
        family: str,
        p_max: int,
        include_cutoff_modes: bool
    ) -> Dict[float, List[Tuple]]:
        """
        Get modes for a specific family like 'TE11p', 'TM01p', etc.

        Parameters
        ----------
        family : str
            Family name (e.g., 'TE11p', 'TM01p', 'TE21p')
        p_max : int
            Maximum longitudinal index
        include_cutoff_modes : bool
            Include p=0 modes

        Returns
        -------
        dict
            {k_squared: [(n, m, p, mode_type), ...]}
        """
        modes = {}

        # Parse family string (e.g., 'TE11p' -> mode_type='TE', n=1, m=1)
        if len(family) < 4:
            return modes

        mode_type = family[:2].upper()  # 'TE' or 'TM'
        indices = family[2:-1]  # e.g., '11', '01', '21'

        if len(indices) < 2:
            return modes

        try:
            n = int(indices[0])
            m = int(indices[1])
        except ValueError:
            return modes

        # Get cutoff wavenumber for this family
        try:
            if mode_type == 'TE':
                kc = self.cutoff_wavenumber_TE(n, m)
            else:
                kc = self.cutoff_wavenumber_TM(n, m)
        except Exception:
            return modes

        # Determine starting p based on mode type
        if mode_type == 'TM':
            p_start = 1  # TM modes require p >= 1
        else:
            p_start = 0 if include_cutoff_modes else 1

        for p in range(p_start, p_max + 1):
            kz = p * np.pi / self.length
            k_sq = kc**2 + kz**2

            if k_sq < 1e-20:
                continue

            k_sq_rounded = round(k_sq, 10)
            if k_sq_rounded not in modes:
                modes[k_sq_rounded] = []
            modes[k_sq_rounded].append((n, m, p, mode_type))

        return modes

    def _format_eigenfrequency_results(
        self,
        sorted_eigenvalues: List[float],
        eigenvalue_modes: Dict[float, List[Tuple]],
        n_modes: int,
        return_format: str
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Format eigenfrequency results based on requested format.

        Parameters
        ----------
        sorted_eigenvalues : list
            Sorted list of k² values
        eigenvalue_modes : dict
            {k_squared: [(n, m, p, mode_type), ...]}
        n_modes : int
            Number of modes to return
        return_format : str
            'dict', 'list', or 'array'

        Returns
        -------
        Formatted results
        """
        if return_format == 'array':
            frequencies = []
            for k_sq in sorted_eigenvalues:
                if len(frequencies) >= n_modes:
                    break
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                frequencies.append(f)
            return np.array(frequencies[:n_modes])

        elif return_format == 'list':
            result_list = []
            for k_sq in sorted_eigenvalues:
                if len(result_list) >= n_modes:
                    break
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                modes = eigenvalue_modes[k_sq]

                # Format mode string - sort by type, then indices
                mode_strs = []
                for n, m, p, mode_type in sorted(modes, key=lambda x: (x[3], x[0], x[1], x[2])):
                    mode_strs.append(f"{mode_type}{n}{m}{p}")
                mode_string = ", ".join(mode_strs)

                result_list.append((f, k, mode_string, modes))

            return result_list[:n_modes]

        else:  # 'dict' (default)
            result_dict = {}
            count = 0
            for k_sq in sorted_eigenvalues:
                if count >= n_modes:
                    break
                k = np.sqrt(k_sq)
                f = c0 * k / (2 * np.pi)
                modes = eigenvalue_modes[k_sq]

                # Create labels for each mode at this frequency
                for n, m, p, mode_type in sorted(modes, key=lambda x: (x[3], x[0], x[1], x[2])):
                    label = f"{mode_type}{n}{m}{p}"
                    if label not in result_dict:
                        result_dict[label] = f
                        count += 1
                        if count >= n_modes:
                            break

            return result_dict

    def all_eigenfrequencies(
        self,
        n_modes: int = 20,
        n_max: int = 5,
        m_max: int = 5,
        p_max: int = 10,
        return_format: str = 'dict'
    ) -> Union[Dict[str, float], List[Tuple], np.ndarray]:
        """
        Compute all physical eigenfrequencies of the circular cavity.

        This is an alias for eigenfrequencies() with mode_types=None (all modes).

        Parameters
        ----------
        n_modes : int
            Number of eigenfrequencies to return
        n_max : int
            Maximum azimuthal index
        m_max : int
            Maximum radial index
        p_max : int
            Maximum longitudinal index
        return_format : str
            'dict': Return {mode_label: frequency} dict
            'list': Return [(frequency, k, mode_string, indices)] list
            'array': Return sorted frequency array

        Returns
        -------
        Depending on return_format:
            dict: {mode_label: frequency_Hz}
            list: [(freq_Hz, k, mode_string, [(n,m,p), ...])]
            array: ndarray of frequencies
        """
        return self.eigenfrequencies(
            n_modes=n_modes,
            mode_types=None,
            n_max=n_max,
            m_max=m_max,
            p_max=p_max,
            include_cutoff_modes=True,
            return_format=return_format
        )

    def get_mode_info(
        self,
        n: int,
        m: int,
        p: int,
        mode_type: str = 'TE'
    ) -> Dict:
        """
        Get detailed information about a specific mode.

        Parameters
        ----------
        n : int
            Azimuthal mode number
        m : int
            Radial mode number
        p : int
            Longitudinal mode number
        mode_type : str
            'TE' or 'TM'

        Returns
        -------
        info : dict
            Mode information including frequency, wavenumbers, etc.
        """
        if mode_type.upper() == 'TE':
            kc = self.cutoff_wavenumber_TE(n, m)
            fc = self.cutoff_frequency_TE(n, m)
        else:
            kc = self.cutoff_wavenumber_TM(n, m)
            fc = self.cutoff_frequency_TM(n, m)

        kz = p * np.pi / self.length
        k = np.sqrt(kc**2 + kz**2)

        f = c0 * k / (2 * np.pi)

        # Wavelengths
        lambda_0 = c0 / f if f > 0 else np.inf
        lambda_c = c0 / fc if fc > 0 else np.inf

        return {
            'mode_type': mode_type,
            'indices': (n, m, p),
            'label': f"{mode_type}{n}{m}{p}",
            'frequency_Hz': f,
            'frequency_GHz': f / 1e9,
            'cutoff_frequency_Hz': fc,
            'cutoff_frequency_GHz': fc / 1e9,
            'k': k,
            'kc': kc,
            'kz': kz,
            'wavelength_m': lambda_0,
            'cutoff_wavelength_m': lambda_c,
        }

    def print_eigenfrequencies(
        self,
        n_modes: int = 20,
        n_max: int = 5,
        m_max: int = 5,
        p_max: int = 10,
        mode_types: Optional[List[str]] = None
    ) -> None:
        """
        Print a formatted table of eigenfrequencies.

        Parameters
        ----------
        n_modes : int
            Number of modes to display
        n_max, m_max, p_max : int
            Maximum indices to search
        mode_types : list of str, optional
            Filter by mode types (None for all modes)
        """
        results = self.eigenfrequencies(
            n_modes=n_modes,
            mode_types=mode_types,
            n_max=n_max,
            m_max=m_max,
            p_max=p_max,
            return_format='list'
        )

        # Build title based on mode types
        if mode_types is None:
            type_str = "All Modes"
        else:
            type_str = ", ".join(mode_types)

        print("\n" + "=" * 90)
        print(f"Circular Cavity Eigenfrequencies ({type_str})")
        print(f"Dimensions: radius={self.radius*1e3:.1f}mm, L={self.length*1e3:.1f}mm")
        print("=" * 90)
        print(f"{'Rank':<6} {'Frequency [GHz]':<18} {'k [rad/m]':<15} {'Modes'}")
        print("-" * 90)

        for i, (f, k, mode_string, modes) in enumerate(results, 1):
            print(f"{i:<6} {f/1e9:<18.6f} {k:<15.4f} {mode_string}")

        print("=" * 90)

    def list_cutoff_frequencies(
        self,
        n_max: int = 5,
        m_max: int = 5
    ) -> List[Dict]:
        """
        Get list of cutoff frequencies sorted by value.

        Parameters
        ----------
        n_max : int
            Maximum azimuthal index
        m_max : int
            Maximum radial index

        Returns
        -------
        list of dict
            Sorted list of mode information dictionaries
        """
        modes = []

        # TE modes
        for n in range(n_max + 1):
            for m in range(1, m_max + 1):
                try:
                    f_c = self.cutoff_frequency_TE(n, m)
                    modes.append({
                        'type': 'TE',
                        'n': n,
                        'm': m,
                        'label': f'TE{n}{m}',
                        'frequency': f_c,
                        'kc': self.cutoff_wavenumber_TE(n, m)
                    })
                except Exception:
                    pass

        # TM modes
        for n in range(n_max + 1):
            for m in range(1, m_max + 1):
                try:
                    f_c = self.cutoff_frequency_TM(n, m)
                    modes.append({
                        'type': 'TM',
                        'n': n,
                        'm': m,
                        'label': f'TM{n}{m}',
                        'frequency': f_c,
                        'kc': self.cutoff_wavenumber_TM(n, m)
                    })
                except Exception:
                    pass

        modes.sort(key=lambda x: x['frequency'])
        return modes

    def print_cutoff_frequencies(self, n_max: int = 5, m_max: int = 5) -> None:
        """
        Print table of cutoff frequencies.

        Parameters
        ----------
        n_max : int
            Maximum azimuthal index
        m_max : int
            Maximum radial index
        """
        print(f"\nCutoff frequencies for circular waveguide (radius = {self.radius*1e3:.1f} mm)")
        print("=" * 60)

        modes = self.list_cutoff_frequencies(n_max, m_max)

        print(f"{'Mode':<10} {'f_c [GHz]':>12} {'λ_c [mm]':>12} {'kc [rad/m]':>15}")
        print("-" * 50)
        for mode in modes[:15]:
            wavelength = c0 / mode['frequency'] * 1e3  # mm
            print(f"{mode['label']:<10} {mode['frequency']/1e9:>12.4f} {wavelength:>12.2f} {mode['kc']:>15.4f}")

        print("=" * 60)
        print(f"Note: TE11 is the fundamental mode with fc = {self.fc/1e9:.4f} GHz")


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_eigenfrequencies(
    fem_freqs: np.ndarray,
    radius: float,
    length: float,
    n_modes: int = 20,
    tolerance: float = 0.05
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Compare FEM eigenfrequencies with analytical solutions.

    Parameters
    ----------
    fem_freqs : ndarray
        Eigenfrequencies from FEM solver [Hz]
    radius : float
        Waveguide radius [m]
    length : float
        Waveguide length [m]
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
    analytical = CWGAnalytical(radius=radius, length=length)

    # Get all eigenfrequencies
    anal_results = analytical.eigenfrequencies(
        n_modes=n_modes * 2,
        return_format='list'
    )

    matches = {}
    errors = []
    matched_fem_indices = set()
    matched_anal_indices = set()

    for fem_idx, fem_f in enumerate(fem_freqs):
        best_match = None
        best_error = tolerance

        for anal_idx, (anal_f, k, modes_str, modes) in enumerate(anal_results):
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

    unmatched = np.array([i for i in range(len(fem_freqs)) if i not in matched_fem_indices])

    return matches, np.array(errors), unmatched


def print_eigenfrequency_comparison(
    fem_freqs: np.ndarray,
    radius: float,
    length: float,
    n_modes: int = 20
) -> None:
    """
    Print a formatted comparison of FEM and analytical eigenfrequencies.

    Parameters
    ----------
    fem_freqs : ndarray
        Eigenfrequencies from FEM solver [Hz]
    radius, length : float
        Waveguide dimensions [m]
    n_modes : int
        Number of modes to compare
    """
    matches, errors, unmatched = compare_eigenfrequencies(
        fem_freqs, radius, length, n_modes
    )

    print("\n" + "=" * 85)
    print("Eigenfrequency Comparison: FEM vs Analytical (Circular Waveguide)")
    print(f"Dimensions: radius={radius*1e3:.1f}mm, L={length*1e3:.1f}mm")
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