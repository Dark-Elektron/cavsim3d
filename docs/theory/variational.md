# 2. Variational Formulation

To solve numerically via FEM, we multiply by a test function $\mathbf{v} \in H(\text{curl})$ and
integrate over the volume $\Omega$[^monk][^jin]. Using 

$$\int_\Omega (\nabla\times\mathbf{A})\cdot\mathbf{v} = \int_\Omega \mathbf{A}\cdot(\nabla\times\mathbf{v})
+ \oint_{\partial\Omega}(\mathbf{n}\times\mathbf{A})\cdot\mathbf{v}$$ 

with the outward normal $\mathbf{n}$, and $\frac{1}{\mu}\nabla\times\mathbf{E} = -j\omega\mathbf{H}$, and dividing by $\mu_0$:

$$
\int_\Omega \frac{1}{\mu_0\mu_r} (\nabla \times \mathbf{E}) \cdot (\nabla \times \mathbf{v}) \, \mathrm{d}\Omega
- \omega^2 \int_\Omega \varepsilon_c \, \mathbf{E} \cdot \mathbf{v} \, \mathrm{d}\Omega
- j\omega \oint_{\partial\Omega} (\mathbf{n} \times \mathbf{H}) \cdot \mathbf{v} \, \mathrm{d}S
= 0
$$

Keeping $1/\mu_0$ in the curl term (rather than absorbing $\mu_0$ into the mass term) is the
scaling used; with it, the port excitation of [§3.2](ports.md#32-building-the-right-hand-side-b) enters without extra factors.

The boundary conditions are applied using the surface integral term:

| Boundary | Condition | Effect |
|----------|-----------|--------|
| **PEC** | $\mathbf{n} \times \mathbf{E} = 0$ | Perfect conductor (default for cavity walls) |
| **PMC** | $\mathbf{n} \times \mathbf{H} = 0$ | Perfect magnetic conductor (any boundary that is not constrained) |
| **Port** | $\mathbf{n} \times \mathbf{H} = \sum_m I_m \mathbf{e}_m$ | Modal current excitation for Z-parameter extraction |

!!! info "Ports are driven by current"
    A port face is a natural boundary on which the tangential magnetic field is prescribed
    by the modal currents $I_m$ ([§4](z_parameters.md)). An undriven port therefore sees $\mathbf{n}\times\mathbf{H}=0$ --
    an open circuit, i.e. a magnetic wall. This is what makes the extracted quantity an
    impedance matrix $\mathbf{Z}$; the scattering matrix is derived from it ([§5](s_parameters.md)). Ports are not
    absorbing boundaries: a port face does not by itself absorb an outgoing wave.

## Discretisation

We expand the electric field in terms of Nédélec (edge) basis functions[^nedelec80][^nedelec86],
here the high-order bases of Schöberl and Zaglmayr[^sz05] as implemented in NGSolve[^ngsolve]:


$$
\mathbf{E} \approx \sum_{i} x_i \mathbf{N}_i
$$


and choose the test function $\mathbf{v} = \mathbf{N}_j$ (Galerkin method) for each degree of freedom $j$.
Substituting into the variational form:


$$
\int_\Omega \frac{1}{\mu_0\mu_r}
  \left(\nabla \times \sum_{i} x_i \mathbf{N}_i\right)
  \cdot (\nabla \times \mathbf{N}_j) \, \mathrm{d}\Omega
\;-\; \omega^2 \int_\Omega \varepsilon_c
  \left(\sum_{i} x_i \mathbf{N}_i\right)
  \cdot \mathbf{N}_j \, \mathrm{d}\Omega
\;-\; j\omega \oint_{\partial\Omega}
  (\mathbf{n} \times \mathbf{H}) \cdot \mathbf{N}_j \, \mathrm{d}S
= 0
$$


We now treat each term separately.

---

### Term 1 — Stiffness (Curl–Curl)

Using linearity of the curl operator, pull the sum and coefficients out of the integral:


$$
\int_\Omega \frac{1}{\mu_0\mu_r}
  \left(\sum_i x_i \,\nabla \times \mathbf{N}_i\right)
  \cdot (\nabla \times \mathbf{N}_j) \, \mathrm{d}\Omega
\;=\;
\sum_i x_i
  \underbrace{
    \int_\Omega \frac{1}{\mu_0\mu_r}
    (\nabla \times \mathbf{N}_i) \cdot (\nabla \times \mathbf{N}_j)
    \, \mathrm{d}\Omega
  }_{K_{ji}}
$$


Define the **stiffness matrix**:


$$
\boxed{
  K_{ji} = \int_\Omega \frac{1}{\mu_0\mu_r}\,
  (\nabla \times \mathbf{N}_i) \cdot (\nabla \times \mathbf{N}_j)
  \, \mathrm{d}\Omega
}
$$

!!! note "The static null space"
    The curl–curl operator vanishes on gradient fields ($\nabla \times \nabla \phi = \mathbf{0}$), so
    $\mathbf{K}$ on its own is singular[^boffi]. The system matrix $\mathbf{K} - \omega^2\mathbf{M}$ is not:
    for $\omega > 0$ the mass term is negative definite on those gradients. No regularisation
    is added, so the frequency sweep, the reduced models and the eigenmode analysis all use
    exactly the same $\mathbf{K}$ and $\mathbf{M}$. The gradients reappear only in the
    eigenvalue problem, as a large cluster of modes at $\omega^2 = 0$ ([§8](resonances.md)), and the sweep must
    start above $f = 0$.


So Term 1 becomes $\displaystyle\sum_i K_{ji}\, x_i$.

---

### Term 2 — Mass and Losses

Split the complex permittivity into its lossless part and its two loss mechanisms,
$\varepsilon_c = \varepsilon_0\varepsilon_r - j\,\varepsilon_0\varepsilon_r\tan\delta - j\sigma/\omega$, and
expand as before:

$$
\omega^2 \int_\Omega \varepsilon_c
  \left(\sum_i x_i \,\mathbf{N}_i\right) \cdot \mathbf{N}_j
  \, \mathrm{d}\Omega
\;=\;
\sum_i x_i \left( \omega^2 M_{ji} - j\omega^2 D_{ji} - j\omega\, C_{ji} \right)
$$

with the **mass matrix** and the two **loss matrices**

$$
\boxed{
  M_{ji} = \varepsilon_0 \int_\Omega \varepsilon_r \, \mathbf{N}_i \cdot \mathbf{N}_j \, \mathrm{d}\Omega,
  \qquad
  D_{ji} = \varepsilon_0 \int_\Omega \varepsilon_r \tan\delta \, \mathbf{N}_i \cdot \mathbf{N}_j \, \mathrm{d}\Omega,
  \qquad
  C_{ji} = \int_\Omega \sigma \, \mathbf{N}_i \cdot \mathbf{N}_j \, \mathrm{d}\Omega
}
$$

All three are real, symmetric and frequency-independent. So Term 2 becomes
$\displaystyle -\omega^2 \sum_i M_{ji}\, x_i + j\omega^2 \sum_i D_{ji}\, x_i + j\omega \sum_i C_{ji}\, x_i$.
For a lossless structure $\mathbf{C} = \mathbf{D} = \mathbf{0}$.

---

### Term 3 — Boundary Conditions

#### PEC Boundary

Starting from the surface integral:


$$
-j\omega \oint_{\partial\Omega}
(\mathbf{n} \times \mathbf{H}) \cdot \mathbf{v}
\,\mathrm{d}S
\;=\;
- j\omega \oint_{\partial\Omega}
\mathbf{H} \cdot (\mathbf{v} \times \mathbf{n})
\,\mathrm{d}S
$$


On PEC walls, $\mathbf{n} \times \mathbf{E} = 0$ is
an **essential (Dirichlet) boundary condition**, enforced
by constraining the edge degrees of freedom. Since the
test functions live in the same constrained space, they
must also satisfy:


$$
\mathbf{n} \times \mathbf{v}\big|_{\partial\Omega_{\text{PEC}}} = 0
$$


Therefore:


$$
-j\omega \oint_{\partial\Omega_{\text{PEC}}}
\mathbf{H} \cdot
\underbrace{(\mathbf{v} \times \mathbf{n})}_{=\,0}
\,\mathrm{d}S = 0
$$

#### PMC Boundary

For PMC, the tangential magnetic field vanishes:
$\mathbf{n} \times \mathbf{H} = 0$.
This is a **natural (Neumann) boundary condition** — the
surface integral vanishes directly without constraining
any degrees of freedom:


$$
-j\omega
\oint_{\partial\Omega_{\text{PMC}}}
\underbrace{(\mathbf{n} \times \mathbf{H})}_{= \,0}
\cdot \mathbf{v}\,\mathrm{d}S \;=\; 0
$$


No special treatment is needed: simply not imposing any
boundary condition on a surface automatically enforces PMC.

#### Source / Port Excitation
On a port face the tangential magnetic field is not unknown: it is set by the port currents,
$\mathbf{n}\times\mathbf{H} = \sum_m I_m\,\mathbf{e}_m$, where $\mathbf{e}_m$ are the port modes of
[§3](ports.md). The surface integral therefore does not depend on the coefficients $x_i$:


$$
-j\omega \oint_{\partial\Omega_\mathrm{port}}
  (\mathbf{n} \times \mathbf{H}) \cdot \mathbf{N}_j \, \mathrm{d}S
\;=\;
-j\omega \sum_m I_m
\underbrace{
  \oint_{\partial\Omega_\mathrm{port}}
    \mathbf{e}_m \cdot \mathbf{N}_j
    \, \mathrm{d}S
}_{b_{j,m}}
$$


Define the **excitation vector** of mode $m$:


$$
\boxed{
  b_{j,m} = \oint_{\partial\Omega_\mathrm{port}}
  \mathbf{e}_m \cdot \mathbf{N}_j
  \, \mathrm{d}S
}
$$


So Term 3 is $-j\omega\, b_j$ for a unit current, which moves to the right-hand side as $+j\omega\, b_j$.

---

### Assembly into Matrix Form

Collecting all three terms for each test function index $j$:


$$
\sum_i K_{ji}\, x_i
\;+\; j\omega \sum_i C_{ji}\, x_i
\;-\; \omega^2 \sum_i \left(M_{ji} - jD_{ji}\right) x_i
\;=\; j\omega\, b_j
$$


Recognising the sums as matrix–vector products:

$$
\boxed{
  \left(\mathbf{K} + j\omega\,\mathbf{C} - \omega^2\,(\mathbf{M} - j\mathbf{D})\right)\mathbf{x}
  = j\omega\,\mathbf{b}
}
$$

which for a lossless structure is $(\mathbf{K} - \omega^2\mathbf{M})\,\mathbf{x} = j\omega\,\mathbf{b}$.
The system matrix is **complex symmetric** (not Hermitian) when there are losses.

where:

| Symbol | Size | Definition |
|--------|------|------------|
| $\mathbf{K}$ | $n \times n$ | Stiffness (curl–curl) matrix, with $1/\mu_0$ |
| $\mathbf{M}$ | $n \times n$ | Mass matrix, with $\varepsilon_0$ |
| $\mathbf{C}$ | $n \times n$ | Conduction loss matrix ($\sigma$) |
| $\mathbf{D}$ | $n \times n$ | Dielectric loss matrix ($\tan\delta$) |
| $\mathbf{x}$ | $n \times 1$ | Unknown edge-element coefficients |
| $\mathbf{b}$ | $n \times 1$ | Port excitation vector |
| $n$           | —    | Number of degrees of freedom |


!!! info "Notation"
    The system is solved one excitation at a time. For each port-mode pair $(p, m)$, the solver constructs a dedicated RHS vector $\mathbf{b}_{p,m}$ assembled in the excitation matrix $\mathbf{B}$ and solves for the corresponding field solution $\mathbf{x}_{p,m}$. The collection of all solutions is assembled into a solution matrix $\mathbf{X} = [\mathbf{x}_{1,1} \mid \mathbf{x}_{1,2} \mid \dots \mid \mathbf{x}_{p,m}]$.

## References

///Footnotes Go Here///

[^monk]: P. Monk, *Finite Element Methods for Maxwell's Equations* (Oxford University Press,
    Oxford, 2003).
[^jin]: J.-M. Jin, *The Finite Element Method in Electromagnetics*, 3rd ed. (Wiley–IEEE Press,
    Hoboken, NJ, 2014).
[^nedelec80]: J.-C. Nédélec, "Mixed finite elements in $\mathbb{R}^3$," *Numer. Math.* **35**,
    315–341 (1980). [doi:10.1007/BF01396415](https://doi.org/10.1007/BF01396415). The first-kind
    elements.
[^nedelec86]: J.-C. Nédélec, "A new family of mixed finite elements in $\mathbb{R}^3$," *Numer.
    Math.* **50**, 57–81 (1986). [doi:10.1007/BF01389668](https://doi.org/10.1007/BF01389668).
    The second-kind elements.
[^sz05]: J. Schöberl and S. Zaglmayr, "High order Nédélec elements with local complete sequence
    properties," *COMPEL* **24**(2), 374–384 (2005).
[^ngsolve]: J. Schöberl, "C++11 implementation of finite elements in NGSolve," ASC Report
    30/2014, Institute for Analysis and Scientific Computing, TU Wien (2014).
[^boffi]: D. Boffi, "Finite element approximation of eigenvalue problems," *Acta Numer.* **19**,
    1–120 (2010).

---

**Next:** [3. Port modes](ports.md)
