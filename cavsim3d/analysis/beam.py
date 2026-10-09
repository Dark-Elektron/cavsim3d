"""Beam current spectra for the HOM power of a structure (``get_hom_power``)."""

from typing import Tuple

import numpy as np


def bunch_train_spectrum(charge: float, spacing: float, sigma_t: float = 0.0,
                         fmax: float = None, fmin: float = 0.0,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Spectral lines of an endless train of equal Gaussian bunches.

    A bunch of ``charge`` [C] every ``spacing`` [s], each of rms length
    ``sigma_t`` [s], is the current

        i(t) = I_0 + sum_p Re{I_p exp(j w_p t)},   w_p = 2 pi p / spacing,
        I_p = 2 (charge / spacing) exp(-(w_p sigma_t)^2 / 2),

    with the average current I_0 = charge / spacing. The lines between
    ``fmin`` and ``fmax`` [GHz] are returned (the average current, at 0 Hz,
    is left out): ``(frequencies [GHz], I_p [A])``, ready for
    ``rom.solve(frequencies=f)`` and ``rom.get_hom_power(I_p)``.
    """
    for name, val in (("charge", charge), ("spacing", spacing)):
        if not np.isfinite(val) or val <= 0:
            raise ValueError(f"{name} must be > 0 (got {val!r}).")
    if fmax is None or not np.isfinite(fmax) or fmax <= max(fmin, 0.0):
        raise ValueError(f"fmax must be a frequency in GHz above fmin (got {fmax!r}).")
    if sigma_t < 0:
        raise ValueError(f"sigma_t must be >= 0 (got {sigma_t!r}).")
    f_line = 1.0 / spacing                                  # Hz
    p = np.arange(max(1, int(np.ceil(fmin * 1e9 / f_line))),
                  int(np.floor(fmax * 1e9 / f_line)) + 1)
    f = p * f_line
    current = 2 * charge / spacing * np.exp(-0.5 * (2 * np.pi * f * sigma_t) ** 2)
    return f / 1e9, current
