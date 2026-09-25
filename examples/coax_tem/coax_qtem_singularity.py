"""qTEM singularity: cause, fix, and an analytic check.

Cause      : fesEt = HCurl(mesh, definedon=port_region) lives on the whole 3-D
             mesh; DOFs away from the port face are UNUSED, giving identically
             zero rows in both a and m. FreeDofs() removes only Dirichlet DOFs,
             so those empty rows are handed to the solver and (a - shift*m) is
             structurally singular for ANY shift.
Fix        : intersect FreeDofs() with the DOFs actually on the port region.
Yardstick  : an air-filled coax has an exact TEM solution, so the qTEM solver
             (built for inhomogeneous cross-sections) must reproduce
             eps_eff = 1, beta = k0, Z0 = (eta0/2pi) ln(b/a).
"""
import numpy as np

from cavsim3d.geometry.base import BaseGeometry

from netgen.occ import Cylinder, Axes, Z as OCC_Z
from ngsolve import (HCurl, BilinearForm, GridFunction, TaskManager, ds, grad,
                     curl, CoefficientFunction, ArnoldiSolver)
from pyngcore import BitArray
import scipy.sparse as sp

ETA0, C0 = 376.730313668, 299792458.0
A_IN, B_OUT, LENGTH = 0.001, 0.0023, 0.020
FREQ = 5e9


class Coax(BaseGeometry):
    def __init__(self, maxh=0.0008):
        super().__init__()
        self.build()
        self.generate_mesh(maxh=maxh)

    def build(self):
        outer = Cylinder(Axes((0, 0, 0), OCC_Z), r=B_OUT, h=LENGTH)
        inner = Cylinder(Axes((0, 0, 0), OCC_Z), r=A_IN, h=LENGTH)
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


def main(maxh=0.0008, shift_factor=1.0):
    geo = Coax(maxh=maxh)
    mesh = geo.mesh
    region = mesh.Boundaries("port1")
    order, k0 = 2, 2 * np.pi * FREQ / C0

    fesEt = HCurl(mesh, order=order, definedon=region, complex=True)
    GEt, fesEz = fesEt.CreateGradient()
    fes = fesEt * fesEz
    (Et, p), (Ft, q) = fes.TnT()
    eps = CoefficientFunction(1.0)

    a = BilinearForm(fes)
    a += (curl(Et).Trace() * curl(Ft).Trace()
          - k0**2 * eps * Et.Trace() * Ft.Trace()) * ds("port1")
    a += -grad(p).Trace() * Ft.Trace() * ds("port1")
    a += (grad(p).Trace() * grad(q).Trace()
          - k0**2 * eps * p.Trace() * q.Trace()) * ds("port1")
    m = BilinearForm(fes)
    m += -Et.Trace() * Ft.Trace() * ds("port1")
    m += Et.Trace() * grad(q).Trace() * ds("port1")
    with TaskManager():
        a.Assemble(); m.Assemble()

    A = sp.csr_matrix(a.mat.CSR())
    M = sp.csr_matrix(m.mat.CSR())
    n = A.shape[0]
    print(f"matrix size {A.shape}, fes.ndof={fes.ndof}")

    free_now = fes.FreeDofs()
    empty = (np.diff(A.indptr) == 0) & (np.diff(M.indptr) == 0)

    def count(bits, label):
        idx = np.array([i for i in range(n) if bits[i]])
        bad = int(empty[idx].sum()) if len(idx) else 0
        print(f"  {label}: {len(idx)} dofs, {bad} of them structurally EMPTY")
        return idx

    print("\nDOFs handed to ArnoldiSolver:")
    count(free_now, "FreeDofs() (current code)")

    # ---- the fix: keep only DOFs that live on the port region --------------
    on_region = fes.GetDofs(region)
    free_fixed = BitArray(free_now)
    free_fixed &= on_region
    idx_fixed = count(free_fixed, "FreeDofs() & GetDofs(port)")

    # conditioning of the shifted matrix on each set
    shift = k0**2
    S = (A - shift * M)
    for bits, label in ((free_now, "current"), (free_fixed, "fixed")):
        idx = np.array([i for i in range(n) if bits[i]])
        if len(idx) > 2500:
            print(f"  cond({label}): {len(idx)} dofs, skipped (dense SVD too large)")
            continue
        sub = S[idx][:, idx].toarray()
        sv = np.linalg.svd(sub, compute_uv=False)
        print(f"  cond({label}): n={len(idx)}, sv_min={sv[-1]:.3e}, "
              f"sv_max={sv[0]:.3e}, cond={sv[0]/max(sv[-1],1e-300):.3e}")

    # ---- solve with the fixed freedofs and compare to the exact TEM --------
    n_eig = 12
    evecs = GridFunction(fes, multidim=n_eig)
    sig = k0**2 * shift_factor
    with TaskManager():
        lam = ArnoldiSolver(a.mat, m.mat, free_fixed, list(evecs.vecs),
                            shift=sig, inverse="pardiso")
    lam = np.array([complex(l) for l in lam])
    beta = np.sqrt(lam)
    eps_eff = lam / k0**2
    ok = [(b, e) for b, e in zip(beta, eps_eff)
          if b.real > 1e-3 and abs(b.imag) < 0.3 * abs(b.real) and 0.5 <= e.real <= 1.5]
    ok.sort(key=lambda z: -z[0].real)

    z0_exact = ETA0 / (2 * np.pi) * np.log(B_OUT / A_IN)
    print(f"\nexact TEM: eps_eff = 1.000000, beta = k0 = {k0:.4f} rad/m, Z0 = {z0_exact:.3f} ohm")
    if not ok:
        print("  NO physical mode found")
        print("  all eps_eff:", np.array2string(eps_eff[:8], precision=4))
    for b, e in ok[:3]:
        print(f"  qTEM mode: eps_eff = {e.real:.6f}  (err {abs(e.real-1)*100:.3f}%), "
              f"beta = {b.real:.4f}  (err {abs(b.real-k0)/k0*100:.3f}%)")


if __name__ == "__main__":
    import sys
    for maxh, sf in [(0.0008, 1.0), (0.0008, 1.15), (0.0004, 1.15), (0.0002, 1.15)]:
        print()
        print("=" * 68)
        print(f"maxh={maxh}  shift_factor={sf}")
        print("="*68)
        try:
            main(maxh=maxh, shift_factor=sf)
        except Exception as e:
            print("  FAILED:", type(e).__name__, str(e)[:120])
