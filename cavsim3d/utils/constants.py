import os
import numpy as np

SOFTWARE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))  # str(Path().parents[0])
CUSTOM_COLORS = ['#4b8f63', '#fc6d2d', '#6a7bbf', '#e567a7', '#8cd839', '#ff5f00', '#d1a67a', '#a3a3a3']
VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6, 'l': 7}
TUNE_ACCURACY = 1e-4
DIMENSION = 'm'
DIMENSION_FACTOR = {'mm': 1, 'cm': 1e-1, 'm': 1e-3}
BOUNDARY_CONDITIONS_DICT = {'ee': 11, 'em': 13, 'me': 31, 'mm': 33}
LABELS = {'freq [MHz]': r'$f$ [MHz]', 'R/Q [Ohm]': r"$R/Q ~\mathrm{[\Omega]}$",
          "Epk/Eacc []": r"$E_\mathrm{pk}/E_\mathrm{acc} ~[\cdot]$",
          "Bpk/Eacc [mT/MV/m]": r"$B_\mathrm{pk}/E_\mathrm{acc} ~\mathrm{[mT/MV/m]}$",
          "G [Ohm]": r"$G ~\mathrm{[\Omega]}$", "Q []": r'$Q$ []',
          'kcc [%]': r'$k_\mathrm{cc}$ [%]', 'GR/Q [Ohm^2]': r'$G \cdot R/Q \mathrm{[\Omega^2]}$',
          'ff [%]': r'$\eta_ff$ [%]',
          'k_FM [V/pC]': r"$|k_\mathrm{FM}| ~\mathrm{[V/pC]}$",
          '|k_loss| [V/pC]': r"$|k_\parallel| ~\mathrm{[V/pC]}$",
          '|k_kick| [V/pC/m]': r"$|k_\perp| ~\mathrm{[V/pC/m]}$",
          'P_HOM [kW]': r"$P_\mathrm{HOM}/\mathrm{cav} ~\mathrm{[kW]}$",
          'Z_2023': 'Z', 'W_2023': 'W', r'H_2023': 'H', 'ttbar_2023': r'$\mathrm{t \bar t}$',
          'Z_b_2024': r'Z$_\mathrm{b}$', r'W_b_2024': 'W$_\mathrm{b}$',
          'H_b_2024': r'H$_\mathrm{b}$', r'ttbar_b_2024': r'$\mathrm{t \bar t}_\mathrm{b}$',
          'Z_b_2024_FB': r'Z$_\mathrm{b}$[FB]', r'W_b_2024_FB': r'W$_\mathrm{b}$[FB]',
          'H_b_2024_FB': r'H$_\mathrm{b}$[FB]', r'ttbar_b_2024_FB': r'$\mathrm{t \bar t}_\mathrm{b}$[FB]',
          r"Ncav": r"$N_\mathrm{cav}$",
          r"Q0 []": r"$Q_0 ~\mathrm{[]}$",
          r"Pstat/cav [W]": r"$P_\mathrm{stat}$/cav [W]",
          r"Pdyn/cav [W]": r"$P_\mathrm{dyn}$/cav [W]",
          r"Pwp/cav [kW]": r"$P_\mathrm{wp}$/cav [kW]",
          r"Pin/cav [kW]": r"$P_\mathrm{in}$/cav [kW]",
          r"PHOM/cav [kW]": r"$P_\mathrm{HOM}$/cav [kW]"
          }


MROT_DICT = {0: 'longitudinal',
             1: 'transversal'}

EIGENMODE_CONFIG = {
    'processes': 3,
    'rerun': True,
    'boundary_conditions': 'mm', # m: PMC, e: PEC
    'mesh_config': {
        'p': 3, # order
        'h': 1, # global upper bound for mesh size
        'grading': 0.3, # mesh grading how fast the local mesh size can change
        'segmentsperedge': 1, # minimal number of segments per edge
    }
    }

UQ_CONFIG = {
    'variables': ['A', 'B', 'a', 'b'],
    'objectives': ["Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]"],
    'delta': [0.05, 0.05, 0.05, 0.05],
    'processes': 4,
    'distribution': 'gaussian',
    'method': ['Quadrature', 'Stroud3'],
    'cell_type': 'mid-cell',
    'cell complexity': 'multicell'
}
OPERATING_POINTS = {
            "Z": {
                "freq [MHz]": 400.79,  # Operating frequency
                "E [GeV]": 45.6,  # <- Beam energy
                "I0 [mA]": 1280,  # <- Beam current
                "V [GV]": 0.12,  # <- Total voltage
                "Eacc [MV/m]": 5.72,  # <- Accelerating field
                "nu_s []": 0.0370,  # <- Synchrotron oscillation tune
                "alpha_p [1e-5]": 2.85,  # <- Momentum compaction factor
                "tau_z [ms]": 354.91,  # <- Longitudinal damping time
                "tau_xy [ms]": 709.82,  # <- Transverse damping time
                "f_rev [kHz]": 3.07,  # <- Revolution frequency
                "beta_xy [m]": 56,  # <- Beta function
                "N_c []": 56,  # <- Number of cavities
                "T [K]": 4.5,  # <- Operating tempereature
                "sigma_SR [mm]": 4.32,  # <- Bunch length
                "sigma_BS [mm]": 15.2,  # <- Bunch length
                "Nb [1e11]": 2.76  # <- Bunch population
            }
}

WAKEFIELD_CONFIG = {
    'beam_config': {
        'bunch_length': 25
    },
    'wake_config': {
        'wakelength': 50
    },
    'processes': 2,
    'rerun': True
}

TUNE_CONFIG = {
    'freqs': 1300,
    'parameters': 'A',
    'cell_types': 'mid-cell',
    'processes': 1,
    'rerun': True
}

m0 = 9.1093879e-31
q0 = 1.6021773e-19
c0 = 2.99792458e8
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12
Z0 = (mu0/eps0)**0.5