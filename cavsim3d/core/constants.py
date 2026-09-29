"""Physical constants and default settings."""

import numpy as np



# Physical constants (SI units)
mu0 = 4 * np.pi * 1e-7      # Permeability of free space [H/m]
c0 = 299792458               # Speed of light [m/s]
eps0 = 1 / (mu0 * c0 ** 2)   # Permittivity of free space [F/m] (8.854187817e-12)
Z0 = np.sqrt(mu0 / eps0)     # Impedance of free space [Ohm]
SIGMA_COPPER = 5.96e7        # Conductivity of copper [S/m] (the value cavsim2d uses)

# Eigenvalues of the curl-curl pencil (K, M) are omega^2.  Its large null space
# of gradient ("static") fields comes out of a sparse/dense eigensolver at
# roughly machine-precision magnitudes -- far below any physical mode, but far
# above 0.  Everything below this frequency is treated as static.
STATIC_MODE_CUTOFF_HZ = 1e6
MIN_EIGENVALUE = (2 * np.pi * STATIC_MODE_CUTOFF_HZ) ** 2   # omega^2 [rad^2/s^2]
