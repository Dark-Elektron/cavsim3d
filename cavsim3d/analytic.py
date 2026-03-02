import numpy as np
import matplotlib.pyplot as plt
from cavsim3d.utils.constants import *
from cavsim3d.helpers import z2s
from ngsolve import (x, z, ds, dx, pi, Mesh, sin, cos, tan, exp, sqrt,
                     GridFunction, BilinearForm, LinearForm, InnerProduct,
                     BaseVector, Integrate, TaskManager, HCurl, Preconditioner,
                     solvers, BoundaryFromVolumeCF)


class RWGAnalytic:
    def __init__(self, geo):
        self.Z11, self.Z21 = [], []
        self.geo = geo
        self.mesh = geo.mesh
        self.order = 3
        self.freq = []

    def analytic_field(self, freq):
        fes_analytic = HCurl(self.mesh, order=self.order, complex=True, dirichlet="top|bottom|left|right")

        # rect waveguide dimensions
        a, b, L = self.geo.a, self.geo.b, self.geo.L

        w = 2*pi*float(freq) # have to explicitly cast from numpy object to float
        Z0 = (mu0/eps0)**0.5
        kc = (pi/a)
        s = 1j*w
        wc = kc*c0
        ZTE = s*Z0/((s**2 + wc**2)**0.5)
        kz = w*mu0/ZTE

        # Analytic field
        A0 = (2/(a*b))**0.5
        print('A0=', A0)
        I1, I2 = 0.0, 1.0
        Vp = (1/2) * ZTE * (1 - 1/tan(kz * L)) * I1 - (1/2) * 1j * ZTE * 1/sin(kz * L) * I2
        Vn = (ZTE / (exp(1j * 2 * kz * L) - 1)) * I1 - (1/2) * 1j * ZTE * 1/sin(kz * L) * I2

        E_y = A0 * sin(np.pi * x / a) * (Vp*exp(-1j*kz*z) + Vn*exp(1j*kz*z))
        fes_analytic.Set((0, E_y, 0))
        gfu = BoundaryFromVolumeCF(fes_analytic)

        return gfu

    def analytic_(self, geo, pts):
        xi, yi, zi = pts[:,0], pts[:,1], pts[:,2]
        # rect waveguide dimensions
        a, b, L = geo.a, geo.b, geo.L

        w = 2*pi*float(self.freq) # have to explicitly cast from numpy object to float
        k = w/c0
        eta = (mu0/eps0)**0.5
        Z0 = eta
        kc = (pi/a)
        s = 1j*w
        wc = kc*c0
        ZTE = s*eta/((s**2 + wc**2)**0.5)
        kz = w*mu0/ZTE

        # Analytic field
        A0 = (2/(a*b))**0.5
        I1, I2 = 0.0, 1.0
        Vp = (1/2) * ZTE * (1 - 1/tan(kz * L)) * I1 - (1/2) * 1j * ZTE * 1/sin(kz * L) * I2
        Vn = (ZTE / (exp(1j * 2 * kz * L) - 1)) * I1 - (1/2) * 1j * ZTE * 1/sin(kz * L) * I2

        E_y = A0 * sin(np.pi * xi / a) * (Vp*exp(-1j*kz*zi) + Vn*exp(1j*kz*zi))
        return E_y

    def solve_FD(self, fmin, fmax, nsamples=1000):
        self.freqs = np.linspace(fmin, fmax, nsamples)*1e9
        ZTEs = []
        for kk, freq in enumerate(self.freqs):
            a, b, L = self.geo.a, self.geo.b, self.geo.L
            w = 2*pi*freq
            k = w/c0
            Z0 = (mu0/eps0)**0.5
            kc = (pi/a)
            s = 1j*w
            wc = kc*c0
            ZTE = s*Z0/((s**2 + wc**2)**0.5)
            ZTEs.append(ZTE)
            kz = w*mu0/ZTE

            self.Z11.append(-1j*ZTE/np.tan(kz*L))
            self.Z21.append(-1j*ZTE/np.sin(kz*L))

    def sparameters(self):
        S11a = []
        # allocate space
        for ind in range(0, len(self.freqs)):
            Zmat = np.array([[self.Z11_a[ind], self.Z21_a[ind]], [self.Z21_a[ind], self.Z11_a[ind]]])
            # print(Zmat)
            S = z2s(Zmat, self.ZTEs[ind])
            S11a.append(S[0, 1])

    def plot_analytical(self, ax=None):
        Z11 = self.Z11
        Z21 = self.Z21
        freqs = self.freqs

        if ax is None:
            fig, ax = plt.subplot_mosaic([[1, 2], [3, 4]], layout='constrained', figsize=(10,8))
        ax[1].plot(freqs, 20*np.log10(np.abs(Z11)), marker='o', label=f'|z11| [dB] analytical', mfc='none', lw=0)
        ax[2].plot(freqs, 20*np.log10(np.abs(Z21)), marker='o', label=f'|z12| [dB] analytical', mfc='none', lw=0)
        ax[1].set_ylabel('|z11| [dB]')
        ax[1].set_xlabel('freq [GHz]')
        ax[2].set_xlabel('freq [GHz]')
        ax[2].set_ylabel('|z12| [dB]')
        ax[1].legend()
        ax[2].legend()

        # plot phase
        ax[3].plot(freqs, np.angle(Z11), marker='o', label=fr'$\\angle$ z11 analytical', mfc='none', lw=0)
        ax[4].plot(freqs, np.angle(Z21), marker='o', label=fr'$\\angle$ z12 analytical', mfc='none', lw=0)
        ax[3].set_xlabel('freq [GHz]')
        ax[4].set_ylabel(r'$\\angle$ z11 [deg]')
        ax[4].set_xlabel('freq [GHz]')
        ax[4].set_ylabel(r'$\\angle$ z11 [deg]')
        ax[3].legend()
        ax[4].legend()

        return ax
