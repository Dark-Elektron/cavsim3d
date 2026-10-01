# 3. Port Modal Analysis

At each waveguide port, a 2D eigenvalue problem is solved on the port
cross-section to determine the port modes $\mathbf{e}_m$ -- the transverse
electric field patterns in which the port fields are expanded.

## 3.1 Port Eigenvalue Problems

A homogeneously filled cross-section carries three families of modes. Each is
characterised by its **cutoff wavenumber** $k_{c,m}$, which depends only on the
cross-section's shape; the filling medium enters only later, through the wave
impedance and the cutoff frequency ([§5.1](s_parameters.md#51-characteristic-wave-impedance)).

**TE modes** ($E_z = 0$) solve the transverse curl–curl problem with the
tangential field vanishing on the port outline,

$$
\nabla_t \times (\nabla_t \times \mathbf{e}_m)
= k_{c,m}^2 \, \mathbf{e}_m .
$$

Every transverse gradient $\nabla_t\phi$ with $\phi = 0$ on the outline also
satisfies this equation, with $k_c = 0$. Those gradients are projected out of the
iteration, so the problem returns TE modes only.

**TM modes** ($H_z = 0$) have a transverse field that *is* a gradient, so the
curl–curl problem cannot see them. They come from the scalar Helmholtz problem
for the axial field, and the transverse mode is its gradient:

$$
-\nabla_t^2 E_z = k_{c,m}^2 \, E_z, \quad E_z = 0 \text{ on the outline},
\qquad \mathbf{e}_m = -\nabla_t E_z .
$$

**TEM modes** exist when the cross-section has more than one conductor (e.g. a
coaxial line). Their field is the gradient of a potential that is constant on
each conductor but different from one conductor to the next. Such a potential
does not vanish on the whole outline, so its gradient survives the projection
above and appears in the TE problem as a mode with $k_c = 0$. A cross-section
with $N$ separate conductors has $N - 1$ TEM modes.

**Quasi-TEM modes** of an inhomogeneous cross-section (microstrip: substrate
and air) have no closed form and no frequency-independent cutoff. They are
computed at a reference wavenumber $k_0$ from the coupled problem for the
transverse field and the axial field (a mixed $H(\mathrm{curl}) \times H^1$
formulation), whose eigenvalue is $\beta^2$ directly. The modes are ordered by
decreasing $\mathrm{Re}\,\beta$, so the fundamental quasi-TEM mode comes first.

!!! tip "Mode sources"
    For rectangular, circular and coaxial cross-sections, the code can use
    **analytic mode formulas** (fast, with a phase fixed by the formula). Any
    cross-section can instead be solved **numerically** on the port FE space,
    as described above. Both external and internal ports use the analytic
    formulas by default. Choose per call with
    `solve(mode_source=..., mode_source_internal=...)`, each `'analytic'` or
    `'numeric'`.

!!! info "Port reference frame"
    Each mode is built in a tangent frame of the port plane that does not
    depend on which way the face's outward normal points. The two faces of a
    join -- whose outward normals are opposite -- therefore produce the same
    mode functions, with the same sign, which is what allows two sections to
    be coupled mode by mode ([§7](concatenation.md)).

### Port Eigenmode Expansion

The port eigenmodes are computed on the 2D port surface $\Gamma_p$
using the **same Nédélec basis functions** as the 3D domain, but
restricted (traced) to the port boundary. Specifically, the $m$-th
port eigenmode field is expanded as:


$$
\mathbf{e}_m = \sum_i (\hat{e}_m)_i \, \mathbf{N}_i^{\text{trace}}
$$


where $(\hat{e}_m)_i$ are the expansion coefficients forming the
discrete eigenvector $\hat{\mathbf{e}}_m \in \mathbb{R}^{n_p}$, and $\mathbf{N}_i^{\text{trace}}$ is the tangential trace of the $i$-th Nédélec edge basis function onto the port surface $\partial\Omega_{\text{port}}$. Here $n_p$ is the number of edge
degrees of freedom on the port.

!!! note "Notation"
    Throughout this section we distinguish between:

    - $\mathbf{e}_m$ — the **continuous mode field** (a function on
      the port surface)
    - $\hat{\mathbf{e}}_m$ — the **discrete coefficient vector**
      (the eigenvector in $\mathbb{R}^{n_p}$)

    They are related by
    $\mathbf{e}_m = \sum_i (\hat{e}_m)_i \,
    \mathbf{N}_i^{\text{trace}}$.

The discrete problems below are the Galerkin forms of the continuous problems of
[§3.1](#31-port-eigenvalue-problems). Besides the traced Nédélec functions $\mathbf{N}_i$, they use scalar ($H^1$)
basis functions $L_i$ on the port, one polynomial order higher, so that every
gradient $\nabla_t L_i$ lies in the span of the $\mathbf{N}_i$. On the conductor
edges of the port outline the tangential field and the scalar field are zero.

**TE modes.** Testing the transverse curl–curl equation with each
$\mathbf{N}_j^{\text{trace}}$ gives the 2D eigenvalue problem on
$\partial\Omega_{\text{port}}$:


$$
\int_{\partial\Omega_{\text{port}}}
(\nabla_t \times \mathbf{e}_m) \cdot
(\nabla_t \times \mathbf{N}_j^{\text{trace}})
\,\mathrm{d}S
= k_{c,m}^2
\int_{\partial\Omega_{\text{port}}}
\mathbf{e}_m \cdot \mathbf{N}_j^{\text{trace}}
\,\mathrm{d}S
$$


which in matrix form reads:


$$
\mathbf{K}_{\text{port}}\,\hat{\mathbf{e}}_m
= k_{c,m}^2\,\mathbf{M}_{\text{port}}\,\hat{\mathbf{e}}_m
$$


where:


$$
(K_{\text{port}})_{ji}
= \int_{\partial\Omega_{\text{port}}}
  (\nabla_t \times \mathbf{N}_i^{\text{trace}})
  \cdot
  (\nabla_t \times \mathbf{N}_j^{\text{trace}})
  \,\mathrm{d}S,
\quad
(M_{\text{port}})_{ji}
= \int_{\partial\Omega_{\text{port}}}
  \mathbf{N}_i^{\text{trace}}
  \cdot
  \mathbf{N}_j^{\text{trace}}
  \,\mathrm{d}S
$$


and $k_{c,m}$ is the cutoff wavenumber of the $m$-th mode.

Every discrete gradient also solves this problem, with $k_c = 0$. Let $\mathbf{G}$ be
the matrix that maps the coefficients of a scalar function vanishing on the outline
to the Nédélec coefficients of its gradient. The projector

$$
\mathbf{P} = \mathbf{I} - \mathbf{G}\left(\mathbf{G}^T\mathbf{M}_{\text{port}}\mathbf{G}\right)^{-1}
\mathbf{G}^T\mathbf{M}_{\text{port}}
$$

removes the gradient part of a vector (it is the $\mathbf{M}_{\text{port}}$-orthogonal
projection onto the complement of the range of $\mathbf{G}$). A port face of up to 600 free
unknowns is solved directly, as a dense generalised eigenvalue problem whose null space holds
the gradients. A larger face is solved by preconditioned inverse iteration (PINVIT) with the
preconditioner $\mathbf{P}\,(\mathbf{K}_{\text{port}} + \mathbf{M}_{\text{port}})^{-1}$, so the
iteration never enters the gradient space. Either way, a vector whose gradient-free part
$\mathbf{P}\hat{\mathbf{e}}$ holds less than a quarter of its norm is round-off, not a mode,
and is dropped. A face that resolves fewer modes than requested is an error: the port needs a
finer mesh or a higher element order, or analytic modes.

**TEM modes.** After the projection, $\mathbf{K}_{\text{port}}\hat{\mathbf{e}} = \mathbf{0}$
still holds for curl-free fields that are *not* gradients of functions vanishing on the
outline: the discrete harmonic fields. Such a field is the gradient of a potential that
is constant on each conductor, which is the electrostatic field of the line. There is one
for each conductor beyond the first, i.e. one per hole in the port face. The direct solve
takes them from its null space, as the null vectors $\mathbf{M}_{\text{port}}$-orthogonal to
every gradient; PINVIT returns them with $k_c^2$ at round-off level. An eigenpair is
classified as TEM when

$$
|k_c^2| \le \frac{10^{-6}}{A_p},
$$

where $A_p$ is the port area (the lowest TE mode has $k_c^2 \sim \pi^2/A_p$). The TEM
field therefore needs no separate electrostatic solve. A scalar Laplace problem with the
potential set to zero on *every* conductor could not find it: that problem has no zero
eigenvalue, and its eigenfunctions are the TM modes.

**TM modes.** The scalar problem for the axial field is discretised in the port's $H^1$
space, $E_{z,m} = \sum_i (\hat{u}_m)_i\,L_i$ with $E_{z,m} = 0$ on the outline:

$$
\int_{\partial\Omega_{\text{port}}} \nabla_t E_{z,m} \cdot \nabla_t L_j \,\mathrm{d}S
= k_{c,m}^2 \int_{\partial\Omega_{\text{port}}} E_{z,m}\, L_j \,\mathrm{d}S ,
$$

or in matrix form
$\mathbf{S}_{\text{port}}\,\hat{\mathbf{u}}_m = k_{c,m}^2\,\mathbf{T}_{\text{port}}\,\hat{\mathbf{u}}_m$,
with $(S_{\text{port}})_{ji} = \int \nabla_t L_i \cdot \nabla_t L_j \,\mathrm{d}S$ and
$(T_{\text{port}})_{ji} = \int L_i L_j \,\mathrm{d}S$. No projection is needed, because
with $E_z = 0$ on the outline the problem has no zero eigenvalue. It is solved directly for a
face of up to 600 free unknowns, and otherwise by PINVIT with the preconditioner
$(\mathbf{S}_{\text{port}} + \mathbf{T}_{\text{port}})^{-1}$. The
transverse mode $\mathbf{e}_m = -\nabla_t E_{z,m}$ lies in the Nédélec trace space, since
the scalar space is one order higher, so it is transferred into that space without
approximation.

**Assembling the mode set.** TEM, TE and TM modes are merged and sorted by $k_c$, TEM
first. Modes whose cutoffs agree to a relative $10^{-3}$ form a degenerate group (for
example the two $\mathrm{TE}_{11}$ polarisations of a circular guide), which is rotated to the
requested polarisation angle. Each mode then gets the sign fixed by the port's tangent
frame and is normalised as in [§3.2](#32-building-the-right-hand-side-b), and the lowest $m_p$ modes are kept.

**Quasi-TEM modes.** On an inhomogeneous cross-section the transverse and axial fields
are coupled and have to be solved together. With
$\mathbf{E} = (\mathbf{e}_t + \hat{\mathbf{z}}\,e_z)\,e^{-j\beta z}$ and non-magnetic media
($\mu_r = 1$), the wave equation
$\nabla\times\nabla\times\mathbf{E} - k_0^2\varepsilon_r\mathbf{E} = 0$ splits into

$$
\nabla_t\times\nabla_t\times\mathbf{e}_t - k_0^2\varepsilon_r\,\mathbf{e}_t -
j\beta\,\nabla_t e_z = -\beta^2\,\mathbf{e}_t ,
\qquad
-\nabla_t^2 e_z - k_0^2\varepsilon_r\, e_z - j\beta\,\nabla_t\cdot\mathbf{e}_t = 0 .
$$

Substituting $p = j\beta\,e_z$ makes both equations linear in $\beta^2$. Testing with
$(\mathbf{f}, q)$ from $H(\mathrm{curl}) \times H^1$ on the port gives

$$
\begin{aligned}
&\int_{\partial\Omega_{\text{port}}} \left[(\nabla_t\times\mathbf{e}_t)\cdot(\nabla_t\times\mathbf{f}) -
k_0^2\varepsilon_r\,\mathbf{e}_t\cdot\mathbf{f}\right]\mathrm{d}S -
\int_{\partial\Omega_{\text{port}}} \nabla_t p\cdot\mathbf{f}\,\mathrm{d}S \\
&\qquad = -\beta^2 \int_{\partial\Omega_{\text{port}}} \mathbf{e}_t\cdot\mathbf{f}\,\mathrm{d}S ,
\end{aligned}
$$

$$
\int_{\partial\Omega_{\text{port}}} \left[\nabla_t p\cdot\nabla_t q -
k_0^2\varepsilon_r\,p\,q\right]\mathrm{d}S
= \beta^2 \int_{\partial\Omega_{\text{port}}} \mathbf{e}_t\cdot\nabla_t q\,\mathrm{d}S .
$$

The divergence term was integrated by parts. Its boundary term vanishes on the conductors,
where $q = 0$; on any other part of the outline it is dropped, which makes that part a
magnetic wall. In matrix form this is the generalised eigenvalue problem

$$
\begin{bmatrix} \mathbf{A}_{tt} & \mathbf{A}_{tz} \\ \mathbf{0} & \mathbf{A}_{zz} \end{bmatrix}
\begin{bmatrix} \hat{\mathbf{e}} \\ \hat{\mathbf{p}} \end{bmatrix}
= \beta^2
\begin{bmatrix} -\mathbf{M}_{tt} & \mathbf{0} \\ \mathbf{B}_{zt} & \mathbf{0} \end{bmatrix}
\begin{bmatrix} \hat{\mathbf{e}} \\ \hat{\mathbf{p}} \end{bmatrix} ,
$$

with

$$
\begin{aligned}
(A_{tt})_{ji} &= \int \left[(\nabla_t\times\mathbf{N}_i)\cdot(\nabla_t\times\mathbf{N}_j) -
 k_0^2\varepsilon_r\,\mathbf{N}_i\cdot\mathbf{N}_j\right]\mathrm{d}S, \\
(A_{tz})_{ji} &= -\int \nabla_t L_i\cdot\mathbf{N}_j \,\mathrm{d}S, \\
(A_{zz})_{ji} &= \int \left[\nabla_t L_i\cdot\nabla_t L_j - k_0^2\varepsilon_r\,L_i L_j\right]\mathrm{d}S, \\
(M_{tt})_{ji} &= \int \mathbf{N}_i\cdot\mathbf{N}_j \,\mathrm{d}S, \qquad
(B_{zt})_{ji} = \int \mathbf{N}_i\cdot\nabla_t L_j \,\mathrm{d}S .
\end{aligned}
$$

The eigenvalue is $\beta^2$ itself, and a propagating mode has
$k_0^2 \le \beta^2 \le k_0^2\,\varepsilon_{r,\max}$, between air and the densest filling.
The problem is solved by shift-and-invert Arnoldi with the shift
$1.15\,k_0^2\,\varepsilon_{r,\max}$, just above that range. An eigenpair is kept as a
physical propagating mode when $\mathrm{Re}\,\beta > 0$,
$|\mathrm{Im}\,\beta| < 0.3\,\mathrm{Re}\,\beta$ and
$1 \le \varepsilon_{\text{eff}} = \beta^2/k_0^2 \le 1.15\,\varepsilon_{r,\max}$. The kept modes
are sorted by decreasing $\mathrm{Re}\,\beta$.

The port mode is $\mathbf{e}_t$ alone; $\hat{\mathbf{p}}$ is discarded. The eigenvector is
real up to one global phase. That phase is read from the non-conjugated product
$\int \mathbf{e}_t\cdot\mathbf{e}_t\,\mathrm{d}S$ and removed, and the real field is
normalised like the other modes. The mode is solved once, at
$k_0 = 2\pi f_{\max}/c_0$ (the top of the sweep band), and the same field pattern and
$\varepsilon_{\text{eff}}$ are used at every frequency of the sweep, so the dispersion
of the quasi-TEM mode across the band is neglected. Its reference impedance $Z_{PV}$ is
computed from this field ([§5.2](s_parameters.md#52-reference-impedance-wave-versus-line)).

!!! info "Why the same basis?"
    Using the trace of the 3D Nédélec basis on the port
    — rather than an independent 2D basis — ensures that
    the coefficient vector $\hat{\mathbf{e}}_m$ lives directly
    in the same discrete space as the 3D field $\mathbf{E}$.
    A numerically computed mode therefore slots into the
    corresponding global DOFs on the port face without any
    interpolation. An analytic mode formula is interpolated into
    this space once, when the mode is created.

## 3.2 Building the Right-Hand Side (b)

The right-hand side vector $\mathbf{b}$ encodes the port modal
excitation. For each port $p$ and mode $m$, the excitation is a
boundary integral over the port surface.

Starting from the boundary integral for the $j$-th component of the
right-hand side vector:


$$
b_{j,p,m} = \int_{\partial\Omega_{\text{port}}}
\mathbf{e}_{p,m} \cdot \mathbf{N}_j^{\text{trace}}
\,\mathrm{d}S
$$


Substitute the basis expansion
$\mathbf{e}_{p,m} = \sum_i (\hat{e}_{p,m})_i \,
\mathbf{N}_i^{\text{trace}}$:


$$
b_{j,p,m} = \int_{\partial\Omega_{\text{port}}}
\left(\sum_i (\hat{e}_{p,m})_i \,
\mathbf{N}_i^{\text{trace}}\right)
\cdot \mathbf{N}_j^{\text{trace}}
\,\mathrm{d}S
$$


Pull the sum and coefficients out of the integral:


$$
 b_{j,p,m} = \sum_i (\hat{e}_{p,m})_i
\underbrace{
  \int_{\partial\Omega_{\text{port}}}
  \mathbf{N}_i^{\text{trace}} \cdot \mathbf{N}_j^{\text{trace}}
  \,\mathrm{d}S
}_{(M_{\text{port}})_{ji}}
$$


This reveals the **port boundary mass matrix**:


$$
(M_{\text{port}})_{ji} = \int_{\partial\Omega_{\text{port}}}
\mathbf{N}_i^{\text{trace}} \cdot \mathbf{N}_j^{\text{trace}}
\,\mathrm{d}S
$$


Recognising the sum as a matrix–vector product, the full
right-hand side vector for port $p$, mode $m$ is:


$$
\mathbf{b}_{p,m} = M_{\text{port}} \, \hat{\mathbf{e}}_{p,m}
$$


where $\hat{\mathbf{e}}_{p,m} \in \mathbb{R}^{n_p}$ is the discrete
eigenvector of the port eigenvalue problem, containing the
coefficients $(\hat{e}_{p,m})_i$.

!!! info "Normalisation"
    Each mode is normalised to unit **power-like norm** over the port surface --
    the $L^2$ integral of the field, not the Euclidean length of its coefficient
    vector:

    $$
    \int_{\partial\Omega_\text{port}} |\mathbf{e}_m|^2 \,\mathrm{d}S
    \;=\; \hat{\mathbf{e}}_m^{\,T}\, \mathbf{M}_{\text{port}}\, \hat{\mathbf{e}}_m \;=\; 1,
    \qquad
    \hat{\mathbf{e}}_m \leftarrow
        \frac{\hat{\mathbf{e}}_m}{\sqrt{\hat{\mathbf{e}}_m^{\,T}\mathbf{M}_{\text{port}}\hat{\mathbf{e}}_m}} .
    $$

    Different modes of the same port are orthogonal in this inner product.

The solver assembles these vectors for all port-mode pairs and
collects them column-wise into the **port basis matrix**:

$$
\mathbf{B} = \bigl[\mathbf{b}_{1,1} \mid \mathbf{b}_{1,2} \mid \cdots \mid \mathbf{b}_{P,m_P}\bigr] \in \mathbb{R}^{n \times N_{pm}},
\qquad N_{pm} = \sum_{p=1}^{P} m_p
$$

where $P$ is the number of ports and $m_p$ the number of modes carried by port $p$
(ports may carry different numbers of modes, e.g. one TEM mode on a coaxial port
and several TE/TM modes on a waveguide port). The general equation therefore for
multiple ports and multiple modes per port is:

$$
  \left(\mathbf{K} + j\omega\,\mathbf{C} - \omega^2\,(\mathbf{M} - j\mathbf{D})\right)\mathbf{X}
  = j\omega\,\mathbf{B}
$$

!!! note "Where the $j$ goes"
    The solver factors the constant $j$ out of the right-hand side, solving

    $$
      \left(\mathbf{K} + j\omega\,\mathbf{C} - \omega^2\,(\mathbf{M} - j\mathbf{D})\right)\mathbf{X} = \omega\,\mathbf{B}
    $$

    so the field coefficients are $j\mathbf{X}$, and the $j$ is reinstated during the
    Z-extraction of [Section 4](z_parameters.md). For a lossless structure
    this keeps $\mathbf{K}$, $\mathbf{M}$, $\mathbf{B}$ and $\mathbf{X}$ real, which is what
    makes the POD basis of [Section 6](reduction.md) real as well. Every
    subsequent section carries the $\omega\,\mathbf{B}$ form.

---

**Next:** [4. Z-parameters](z_parameters.md)
