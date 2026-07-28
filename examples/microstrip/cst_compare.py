"""Helpers to read CST ASCII exports and overlay them on cavsim3d results.

CST 1D exports are 3 columns: frequency [GHz], magnitude (linear), phase [deg].
Port-information exports (eps_eff, line impedance) are: freq [GHz], real, imag.
"""
from pathlib import Path
import numpy as np


def read_cst_magphase(path):
    """Read a CST S-parameter style file -> (freq_Hz, complex values)."""
    data = np.loadtxt(path)
    f = data[:, 0] * 1e9
    mag = data[:, 1]
    ph = np.deg2rad(data[:, 2])
    return f, mag * np.exp(1j * ph)


def read_cst_realimag(path):
    """Read a CST real/imag style file (eps_eff, impedance) -> (freq_Hz, complex)."""
    data = np.loadtxt(path)
    f = data[:, 0] * 1e9
    return f, data[:, 1] + 1j * data[:, 2]


def default_cst_dir():
    p = Path(r"C:\Users\Soske\Documents\CEM2\cst\microstrip_line\Export")
    return p if p.exists() else None


def overlay_s_parameters(freqs, S, cst_dir=None, ax=None):
    """Plot |S11|, |S21| (dB) from cavsim3d against CST, if available."""
    import matplotlib.pyplot as plt
    cst_dir = Path(cst_dir) if cst_dir else default_cst_dir()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    f_ghz = np.asarray(freqs) / 1e9

    def db(x):
        return 20 * np.log10(np.maximum(np.abs(x), 1e-12))

    ax.plot(f_ghz, db(S[:, 0, 0]), '-', label='cavsim3d |S11|')
    if S.shape[1] > 1:
        ax.plot(f_ghz, db(S[:, 1, 0]), '-', label='cavsim3d |S21|')

    if cst_dir is not None:
        try:
            f1, s11 = read_cst_magphase(cst_dir / "S-Parameters_S1,1.txt")
            ax.plot(f1 / 1e9, db(s11), '--', label='CST |S11|')
            f2, s21 = read_cst_magphase(cst_dir / "S-Parameters_S2,1.txt")
            ax.plot(f2 / 1e9, db(s21), '--', label='CST |S21|')
        except Exception as e:
            print(f"  (CST S-param overlay skipped: {e})")

    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("|S| [dB]")
    ax.set_title("Microstrip S-parameters: cavsim3d vs CST")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def cst_port_info(cst_dir=None):
    """Return dict with CST eps_eff and line impedance arrays for port 1."""
    cst_dir = Path(cst_dir) if cst_dir else default_cst_dir()
    if cst_dir is None:
        return {}
    out = {}
    try:
        out['eps_eff'] = read_cst_realimag(
            cst_dir / "Port Information_Effective Dielectric Constant_1(1).txt")
        out['line_impedance'] = read_cst_realimag(
            cst_dir / "Port Information_Line Impedance_1(1).txt")
    except Exception as e:
        print(f"  (CST port info skipped: {e})")
    return out
