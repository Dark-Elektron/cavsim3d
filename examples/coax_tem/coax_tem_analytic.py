"""Does the TEM port machinery work? Homogeneous coax vs closed form.

A homogeneous coaxial line is the cleanest TEM test there is:
    Z0   = (eta0 / 2pi) * ln(b/a)          [vacuum]
    beta = k0 = omega/c                     (no cutoff, no dispersion)
    S11  = 0,  S21 = exp(-j*beta*L)
and it takes the ANALYTIC coax path (_generate_coaxial_modes), so the Arnoldi
qTEM solver is never involved. If this passes, TEM ports are sound and any
disagreement in the ceramic case belongs to the qTEM path.
"""
import matplotlib
matplotlib.use("Agg")
import tempfile
import numpy as np

# IMPORT ORDER MATTERS: cavsim3d pulls in pythonocc-core (conda occt 7.9.0).
# netgen ships its own OCCT (pip netgen_occt 7.8.1); if netgen.occ loads first,
# pythonocc then fails with
#   ImportError: DLL load failed while importing _STEPControl
# because the already-loaded OCCT is missing symbols it needs.
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.base import BaseGeometry

from netgen.occ import Cylinder, Axes, Z as OCC_Z

ETA0 = 376.730313668
C0 = 299792458.0

A_IN, B_OUT, LENGTH = 0.001, 0.0023, 0.050      # m  -> ~50 ohm
FMIN, FMAX, NS = 1.0, 10.0, 21                  # GHz (TE11 cuts on ~29 GHz)


class CoaxialLine(BaseGeometry):
    """Air-filled coaxial line, ports on both ends."""

    def __init__(self, r_inner, r_outer, length, maxh=0.0008):
        super().__init__()
        self.r_inner, self.r_outer, self.length = r_inner, r_outer, length
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        outer = Cylinder(Axes((0, 0, 0), OCC_Z), r=self.r_outer, h=self.length)
        inner = Cylinder(Axes((0, 0, 0), OCC_Z), r=self.r_inner, h=self.length)
        self.geo = outer - inner
        self.geo.faces.Min(OCC_Z).name = "port1"
        self.geo.faces.Max(OCC_Z).name = "port2"
        for f in self.geo.faces:
            if f.name not in ("port1", "port2"):
                f.name = "default"
        self.geo.mat("vacuum")
        self.bc = "default"

    @property
    def supports_analytical(self):
        return False


def main():
    z0_exact = ETA0 / (2 * np.pi) * np.log(B_OUT / A_IN)
    print(f"analytic coax Z0 = {z0_exact:.3f} ohm   (b/a = {B_OUT/A_IN:.3f})")

    proj = EMProject(name="coax", base_dir=tempfile.mkdtemp(prefix="coax_"),
                     overwrite=True)
    proj.geometry = CoaxialLine(A_IN, B_OUT, LENGTH)
    print(f"mesh: {proj.geometry.mesh.ne} elements")

    res = proj.fds.solve(config=dict(nportmodes=1, order=2, nsamples=NS,
                                     fmin=FMIN, fmax=FMAX,
                                     solver_type="direct", rerun=True))

    ps = proj.fds.port_solver
    for port in ("port1", "port2"):
        geom = ps.port_geometries[port]
        kc = ps.port_cutoff_kc[port][0]
        mtype = ps.port_mode_types[port][0]
        print(f"  {port}: geometry={geom.type.value}, mode0 type={mtype}, kc={kc:.4e}")
        zline = ps.port_line_impedance.get(port, {}).get(0)
        if zline is not None:
            print(f"      Z_line = {complex(zline).real:.3f} ohm "
                  f"(analytic {z0_exact:.3f}, err {abs(complex(zline).real-z0_exact)/z0_exact*100:.2f}%)")

    f = np.asarray(res["frequencies"]).ravel()
    f_ghz = f / 1e9 if f.max() > 1e6 else f
    S = res["S_dict"]
    s11 = np.asarray(S["1(1)1(1)"]).ravel()
    s21 = np.asarray(S["1(1)2(1)"]).ravel()

    beta = 2 * np.pi * (f_ghz * 1e9) / C0            # TEM: beta = k0
    s21_exact = np.exp(-1j * beta * LENGTH)

    print("\n  f[GHz]   |S11|      |S21|    arg(S21) num/exact [deg]   phase err")
    for i in range(0, len(f_ghz), max(1, len(f_ghz) // 6)):
        pn, pe = np.degrees(np.angle(s21[i])), np.degrees(np.angle(s21_exact[i]))
        d = (pn - pe + 180) % 360 - 180
        print(f"  {f_ghz[i]:6.2f}  {abs(s11[i]):.5f}  {abs(s21[i]):.5f}   "
              f"{pn:8.2f} / {pe:8.2f}        {d:+7.2f}")

    dphi = np.degrees(np.angle(s21) - np.angle(s21_exact))
    dphi = (dphi + 180) % 360 - 180
    print(f"\n  mean |S11| = {np.abs(s11).mean():.3e}   (exact 0)")
    print(f"  mean |S21| = {np.abs(s21).mean():.6f}   (exact 1)")
    print(f"  phase error: mean {dphi.mean():+.3f} deg, max |{np.abs(dphi).max():.3f}| deg")


if __name__ == "__main__":
    main()
