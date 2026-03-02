"""Physical constants and default settings."""

import numpy as np

# Physical constants (SI units)
mu0 = 4 * np.pi * 1e-7      # Permeability of free space [H/m]
eps0 = 8.85418782e-12        # Permittivity of free space [F/m]
c0 = 299792458               # Speed of light [m/s]
Z0 = np.sqrt(mu0 / eps0)     # Impedance of free space [Ohm]

# Verify consistency
assert np.isclose(c0, 1 / np.sqrt(mu0 * eps0), rtol=1e-6)