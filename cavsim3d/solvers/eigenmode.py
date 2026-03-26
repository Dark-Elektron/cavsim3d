import numpy as np
from ngsolve import  (x, z, ds, dx, pi, Mesh, sin, cos, tan, exp, sqrt, BND, ArnoldiSolver,
                      GridFunction, BilinearForm, LinearForm, InnerProduct, Cross, curl, BoundaryFromVolumeCF, ContactBoundary, BaseVector,
                      Integrate, TaskManager, HCurl, Compress, Preconditioner, solvers, Norm, specialcf, ConvertOperator, IdentityMatrix,
                      CoefficientFunction
                      )
import scipy.sparse as sp
from ngsolve.webgui import Draw
from netgen.occ import X, Y, Z, Rectangle, OCCGeometry, Axis, Glue

class MEVP:
    def __init__(self):
        pass

    def solve(self, a, m, apre, fes):
        with TaskManager():
            a.Assemble()
            m.Assemble()
            acsr = -sp.csr_matrix(a.mat.CSR())
            mcsr = sp.csr_matrix(m.mat.CSR())
            freedofs = fes.FreeDofs()
            print(acsr.shape, mcsr.shape, np.max(acsr), np.max(mcsr))
            evals, evecs = sp.linalg.eigs(acsr[freedofs, :][:, freedofs], k=20, M=mcsr[freedofs, :][:, freedofs],
                                          sigma=240)
            print(np.sort(np.abs(evals)))

            apre.Assemble()

            G, fesh1 = fes.CreateGradient()
            GT = G.CreateTranspose()
            math1 = GT @ m.mat @ G
            #     math1[0, 0] += 1
            invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs())

            proj = IdentityMatrix() - G @ invh1 @ GT @ m.mat
            projpre = proj @ pre.mat

            evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=10, maxit=20,
                                          printrates=False)
        print('pinvit', evals)
        freq_fes = []
        for i, lam in enumerate(evals):
            freq_fes.append(c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6)
        #     print(freq_fes)

        # plot results
        gfu_E = []
        gfu_H = []
        for i in range(len(evecs)):
            w = 2 * pi * freq_fes[i] * 1e6
            gfu = GridFunction(fes)
            gfu.vec.data = evecs[i]

            gfu_E.append(gfu)
            gfu_H.append(1j / (mu0 * w) * curl(gfu))

        gfu_E = BoundaryFromVolumeCF(Norm(gfu_E[0]))
        # normalise by energy to 1J
        Un = Integrate(InnerProduct(gfu_E, gfu_E), mesh)
        gfu_E = gfu_E / np.sqrt(Un)

        # get analytic solution
        gfu_analytic = eig_analytic(GridFunction(fes))
        # normalise by energy to 1J
        Ua = Integrate(InnerProduct(gfu_analytic, gfu_analytic), mesh)
        gfu_analytic = gfu_analytic / np.sqrt(Ua)

        error = gfu_analytic - gfu_E[0]

        return fes.ndof, error, gfu_analytic, Norm(gfu_E[0])

    def port_eigenmodes(self, fesport, mesh, port, nmodes=1):
        u, v = fesport.TnT()
        a = BilinearForm(curl(u.Trace()) * curl(v.Trace()) * ds(port))
        m = BilinearForm(u.Trace() * v.Trace() * ds(port))
        apre = BilinearForm((curl(u).Trace() * curl(v).Trace() + u.Trace() * v.Trace()) * ds(port))
        pre = Preconditioner(apre, type="direct", inverse="sparsecholesky")

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            G, fesh1 = fesport.CreateGradient()
            GT = G.CreateTranspose()
            math1 = GT @ m.mat @ G
            invh1 = math1.Inverse(inverse="sparsecholesky", freedofs=fesh1.FreeDofs())

            proj = IdentityMatrix(fesport.ndof) - G @ invh1 @ GT @ m.mat

            projpre = proj @ pre
            evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=nmodes + 2, maxit=20,
                                          printrates=False)
        filt = np.array(evals) > 1
        evals = np.array(evals)[filt]
        evecs = np.array(evecs)[filt]
        freq_fes = []
        # evals[0] = 1  # <- replace nan with zero
        for i, lam in enumerate(evals):
            freq_fes.append(c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6)
        # print(freq_fes)

        E, B = dict(), dict()
        for mode_num in range(nmodes):
            efield = GridFunction(fesport)
            efield.vec.data = evecs[mode_num]

            U = Integrate(InnerProduct(efield, efield), mesh, BND, definedon=mesh.Boundaries(port))
            efield.vec.data = efield.vec / np.sqrt(U)

            E[mode_num] = efield
            # Draw(efield, settings=settings)

            efield_mass_weighted = GridFunction(fesport)
            efield_mass_weighted.vec.data = sp.csr_matrix(m.mat.CSR()) @ evecs[mode_num]
            # efield_full.Set(efield, definedon=mesh.Boundaries(port))
            # efield_full.Set(-mu0*efield, definedon=mesh.Boundaries(port))

            B[mode_num] = efield_mass_weighted

        return E, B
