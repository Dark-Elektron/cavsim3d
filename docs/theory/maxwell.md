# 1. Maxwell's Equations

In the frequency domain, assuming an $e^{j\omega t}$ time dependence:

$$
\nabla \times \mathbf{E} = -j\omega \mu \mathbf{H}
$$

$$
\nabla \times \mathbf{H} = j\omega \varepsilon \mathbf{E} + \sigma\mathbf{E}
$$

where:

- $\omega = 2\pi f$ is the angular frequency
- $\mu = \mu_0\mu_r$ is the magnetic permeability
- $\varepsilon = \varepsilon_0\varepsilon_r$ is the permittivity
- $\sigma$ is the electrical conductivity of the filling medium (zero for a lossless material)

A dielectric loss tangent $\tan\delta$ and the conductivity are combined into one **complex
permittivity**,

$$
\varepsilon_c = \varepsilon_0\varepsilon_r\,(1 - j\tan\delta) - j\,\frac{\sigma}{\omega},
$$

so that $\nabla \times \mathbf{H} = j\omega\varepsilon_c\mathbf{E}$. The walls themselves are perfect
conductors; $\sigma$ and $\tan\delta$ describe losses in the volume.

## Vector Wave Equation

Taking the curl of the first equation and substituting the second yields the second-order equation for $\mathbf{E}$:

$$
\nabla \times \left( \frac{1}{\mu_r} \nabla \times \mathbf{E} \right) - k_0^2 \frac{\varepsilon_c}{\varepsilon_0} \mathbf{E} = \mathbf{0}
$$

where $k_0 = \omega\sqrt{\mu_0\varepsilon_0}$ is the free-space wavenumber. For a lossless material
$\varepsilon_c/\varepsilon_0 = \varepsilon_r$. This is the core equation solved by **FrequencyDomainSolver**.

---

**Next:** [2. Variational formulation](variational.md)
