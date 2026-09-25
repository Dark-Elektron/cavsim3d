"""Is cavsim3d's Z the network Z-parameter, or a modal impedance?

Exact lossless line of length L, characteristic impedance Z0:
    Z11 = Z22 = -j Z0 cot(beta L)
    Z21 = Z12 = -j Z0 / sin(beta L)
If cavsim3d's Z_dict matches that, Z is referenced like CST's.
"""
import matplotlib; matplotlib.use("Agg")
import numpy as np, tempfile
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry.base import BaseGeometry
from cavsim3d.solvers.base import ParameterConverter
from netgen.occ import Cylinder, Axes, Z as OCC_Z

ETA0, C0 = 376.730313668, 299792458.0
A_IN, B_OUT, LENGTH = 0.001, 0.0023, 0.050

class Coax(BaseGeometry):
    def __init__(self, maxh=0.0008):
        super().__init__(); self.build(); self.generate_mesh(maxh=maxh)
    def build(self):
        self.geo = (Cylinder(Axes((0,0,0), OCC_Z), r=B_OUT, h=LENGTH)
                    - Cylinder(Axes((0,0,0), OCC_Z), r=A_IN, h=LENGTH))
        self.geo.faces.Min(OCC_Z).name = "port1"; self.geo.faces.Max(OCC_Z).name = "port2"
        for f in self.geo.faces:
            if f.name not in ("port1","port2"): f.name = "default"
        self.geo.mat("vacuum"); self.bc = "default"
    @property
    def supports_analytical(self): return False

proj = EMProject(name="coaxz", base_dir=tempfile.mkdtemp(prefix="coaxz_"), overwrite=True)
proj.geometry = Coax()
res = proj.fds.solve(config=dict(nportmodes=1, order=2, nsamples=11, fmin=1.0, fmax=6.0,
                                 solver_type="direct", rerun=True))
f = np.asarray(res["frequencies"]).ravel(); f = f/1e9 if f.max() > 1e6 else f
Z0 = ETA0/(2*np.pi)*np.log(B_OUT/A_IN)
beta = 2*np.pi*(f*1e9)/C0
z11_ex = -1j*Z0/np.tan(beta*LENGTH)
z21_ex = -1j*Z0/np.sin(beta*LENGTH)

Zc = np.asarray(res["Z_dict"]["1(1)1(1)"]).ravel()
Zt = np.asarray(res["Z_dict"]["1(1)2(1)"]).ravel()
Sc = np.asarray(res["S_dict"]["1(1)1(1)"]).ravel()
St = np.asarray(res["S_dict"]["1(1)2(1)"]).ravel()

zref = complex(proj.fds._get_port_impedance("port1", 0, f[3]*1e9))
print(f"\nZ0 analytic (line)      = {Z0:.3f} ohm")
print(f"reference used for S    = {zref.real:.3f} ohm  <-- _get_port_impedance")
print("\n  f[GHz]   |Z11|cav    |Z11|exact   ratio     |Z21|cav    |Z21|exact   ratio")
for i in range(1, len(f)-1):
    r1 = abs(Zc[i])/abs(z11_ex[i]); r2 = abs(Zt[i])/abs(z21_ex[i])
    print(f"  {f[i]:5.2f}  {abs(Zc[i]):10.3f}  {abs(z11_ex[i]):10.3f}  {r1:7.3f}   "
          f"{abs(Zt[i]):10.3f}  {abs(z21_ex[i]):10.3f}  {r2:7.3f}")

# renormalise cavsim3d's own S back to Z with the analytic line impedance
S2 = np.array([[[Sc[i], St[i]],[St[i], Sc[i]]] for i in range(len(f))])
Zfix = ParameterConverter.s_to_z(S2, Z0)
print("\n  after s_to_z(S_cav, Z0_line):")
print("  f[GHz]   |Z11|renorm  |Z11|exact   rel.err")
for i in range(1, len(f)-1):
    e = abs(abs(Zfix[i,0,0])-abs(z11_ex[i]))/abs(z11_ex[i])
    print(f"  {f[i]:5.2f}  {abs(Zfix[i,0,0]):11.3f}  {abs(z11_ex[i]):10.3f}  {e:8.2%}")
