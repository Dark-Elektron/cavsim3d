# 9. Beam Excitation

A charged particle beam travelling through the structure excites it through a source term in
the field equation, next to the port modes. Its response gives the **beam impedance** and the
coupling between the beam and the port modes. Together with the S-matrix they form the
generalised scattering matrix $\tilde{\mathbf{S}}$ of the CSC-BEAM method (T. Flisgen et al.,
*Phys. Rev. Accel. Beams* **23**, 034601 (2020)), through which segments are concatenated. Equation
numbers in parentheses, such as (eq. 4), refer to that paper.

!!! note "Formulation only"
    The solver API does not include the beam excitation; this section derives its formulation.
    The materials are those of [§1](maxwell.md): $\mu = \mu_0\mu_r$ and the complex permittivity
    $\varepsilon_c$ with conductivity and loss tangent. The walls are perfect conductors.

## 9.1 Beam Current

The beam moves with the velocity $v_b = \beta c_0$ along the line $\ell_b$: $x = x_b$, $y = y_b$,
parallel to $z$. A point bunch of charge $q$ that passes $z = 0$ at $t = 0$ crosses the plane $z$
at $t = z/v_b$. Its current through that plane, $q\,\delta(t - z/v_b)$, has the spectrum
$q\,e^{-j\omega z/v_b}$. In the frequency domain the beam is therefore the line current (eq. 4)

$$
\mathbf{J}(\mathbf{r}) = i\;\delta(x - x_b)\,\delta(y - y_b)\;e^{-jk_b z}\;\hat{\mathbf{z}},
\qquad k_b = \frac{\omega}{v_b} = \frac{\omega}{\beta c_0},
$$

with the current amplitude $i$ ($i = q$ for the point bunch). Continuity,
$\nabla\cdot\mathbf{J} + j\omega\varrho = 0$, gives the line charge that travels with it,

$$
\varrho(\mathbf{r}) = \frac{i}{v_b}\;\delta(x - x_b)\,\delta(y - y_b)\;e^{-jk_b z} .
$$

Around the beam line, $\rho = \sqrt{(x - x_b)^2 + (y - y_b)^2}$ is the distance from it and
$\hat{\mathbf{e}}_\rho$, $\hat{\mathbf{e}}_\varphi$ are the radial and azimuthal unit vectors,
$\hat{\mathbf{e}}_\rho\times\hat{\mathbf{e}}_\varphi = \hat{\mathbf{z}}$. The shorthand
$\delta_b = \delta(x - x_b)\,\delta(y - y_b)$ is used below.

## 9.2 Field Equation with the Beam

With the beam, Ampère's law of [§1](maxwell.md) carries the source current,
$\nabla\times\mathbf{H} = j\omega\varepsilon_c\mathbf{E} + \mathbf{J}$, and the vector wave equation becomes

$$
\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}\right) - \omega^2\varepsilon_c\,\mathbf{E} = -j\omega\,\mathbf{J} .
$$

Testing with $\mathbf{v}$ and integrating by parts as in [§2](variational.md) gives (eq. 6)

$$
\begin{aligned}
&\int_\Omega \frac{1}{\mu}(\nabla\times\mathbf{E})\cdot(\nabla\times\mathbf{v})\,\mathrm{d}\Omega
- \omega^2\int_\Omega \varepsilon_c\,\mathbf{E}\cdot\mathbf{v}\,\mathrm{d}\Omega
- j\omega\oint_{\partial\Omega}(\mathbf{n}\times\mathbf{H})\cdot\mathbf{v}\,\mathrm{d}S \\
&\qquad = -j\omega\int_\Omega\mathbf{J}\cdot\mathbf{v}\,\mathrm{d}\Omega
= -j\omega\, i\int_{\ell_b} v_z\,e^{-jk_b z}\,\mathrm{d}z .
\end{aligned}
$$

On the PEC walls $\mathbf{n}\times\mathbf{v} = 0$. On every other boundary $\mathbf{n}\times\mathbf{H}$ is
prescribed. A port face that the beam crosses is open, as for the port excitations, but there
$\mathbf{n}\times\mathbf{H}$ is the tangential magnetic field of the beam's own field in the pipe,
$\mathbf{n}\times\mathbf{H}^{inc}$ ([§9.5](#95-the-beams-field-in-a-port)); on boundaries the beam does
not cross, $\mathbf{H}^{inc} = 0$. With $\mathbf{E} \approx \sum_i x_i\mathbf{N}_i$ the system is

$$
\left(\mathbf{K} + j\omega\,\mathbf{C} - \omega^2(\mathbf{M} - j\mathbf{D})\right)\mathbf{x} = \mathbf{f}(\omega),
$$

with the matrices of [§2](variational.md) and the beam's load vector

$$
f_j(\omega) = -j\omega\, i\int_{z_{\min}}^{z_{\max}} N_{j,z}(x_b, y_b, z)\,e^{-jk_b z}\,\mathrm{d}z
\;+\; j\omega\sum_{p}\oint_{\Gamma_p}(\mathbf{n}\times\mathbf{H}^{inc})\cdot\mathbf{N}_j\,\mathrm{d}S ,
$$

where $z_{\min}$ and $z_{\max}$ are the ends of the beam line in $\Omega$ and the sum runs over the
port faces $\Gamma_p$. The line integral is well defined when the beam line is made of element
edges: $N_{j,z}$ is then the tangential component of $\mathbf{N}_j$ along those edges, which is
continuous. It is evaluated with Gauss points on every edge of the beam line.

!!! warning "The field of a line current is not in $H(\mathrm{curl})$"
    Near the beam line the field of a line current grows like $1/\rho$
    ([§9.4](#94-the-beams-own-field)). Then $\int|\mathbf{E}|^2\,\mathrm{d}\Omega$ diverges
    logarithmically around $\ell_b$: the exact solution is not square-integrable there, and the
    line functional $\mathbf{v}\mapsto\int_{\ell_b}v_z\,\mathrm{d}z$ is not bounded on
    $H(\mathrm{curl})$. The discrete system still has a solution, but it does not converge on the
    beam line, which is where the beam voltage of [§9.3](#93-beam-voltage-and-beam-impedance) is
    read. The beam is therefore solved for the scattered field of
    [§9.6](#96-scattered-field-formulation), from which this singular part is removed.

## 9.3 Beam Voltage and Beam Impedance

The beam picks up the voltage (eq. 5)

$$
v = \int_{z_{\min}}^{z_{\max}} E_z(x_b, y_b, z)\,e^{jk_b z}\,\mathrm{d}z ,
$$

the longitudinal field at its own position with its own phase $e^{-jk_b z}$ removed. The ratio
$v/i$ is the beam impedance. Its value depends on how the ports are terminated: open ports give
$z_{oc}$ ([§9.7](#97-port-voltages-and-currents-with-the-beam)), matched ports give $z_b$
([§9.8](#98-generalised-scattering-matrix)). The longitudinal impedance is its negative,
$Z_\parallel = -v/i$.

In terms of the field coefficients, $v = \mathbf{c}(\omega)^T\mathbf{x}$ with

$$
c_j(\omega) = \int_{z_{\min}}^{z_{\max}} N_{j,z}(x_b, y_b, z)\,e^{jk_b z}\,\mathrm{d}z ,
$$

the line integral of the load vector with the opposite phase.

## 9.4 The Beam's Own Field

Let the beam travel through a homogeneous **reference medium** with complex permittivity
$\varepsilon_b$ and permeability $\mu_b$, the medium around the beam line. In this medium,
unbounded, the beam's field follows from the potentials in the Lorenz gauge,

$$
\mathbf{E} = -j\omega\mathbf{A} - \nabla\Phi, \qquad \mu_b\mathbf{H} = \nabla\times\mathbf{A}, \qquad
\nabla\cdot\mathbf{A} + j\omega\mu_b\varepsilon_b\,\Phi = 0,
$$

which satisfy $(\nabla^2 + \omega^2\mu_b\varepsilon_b)\,\mathbf{A} = -\mu_b\mathbf{J}$ and
$(\nabla^2 + \omega^2\mu_b\varepsilon_b)\,\Phi = -\varrho/\varepsilon_b$. With $\mathbf{A} = A_z\hat{\mathbf{z}}$
and the beam's dependence $e^{-jk_b z}$, both become 2D modified Helmholtz equations,

$$
(\nabla_t^2 - \kappa_b^2)\,A_z = -\mu_b\, i\,\delta_b, \qquad
(\nabla_t^2 - \kappa_b^2)\,\Phi = -\frac{i}{v_b\varepsilon_b}\,\delta_b,
$$

$$
\kappa_b^2 = k_b^2 - \omega^2\mu_b\varepsilon_b = k_b^2\left(1 - v_b^2\mu_b\varepsilon_b\right),
\qquad \mathrm{Re}\,\kappa_b \ge 0 .
$$

The solution of $(\nabla_t^2 - \kappa^2)\,G = -\delta$ in the plane that decays, or radiates
outwards, is $G = K_0(\kappa\rho)/(2\pi)$, so

$$
A_z^{free} = \frac{\mu_b\, i}{2\pi}\,K_0(\kappa_b\rho)\,e^{-jk_b z}, \qquad
\Phi^{free} = \frac{i}{2\pi v_b\varepsilon_b}\,K_0(\kappa_b\rho)\,e^{-jk_b z},
$$

that is, $A_z^{free} = v_b\,\mu_b\varepsilon_b\,\Phi^{free}$. With $K_0' = -K_1$ and
$E_z = -j\omega A_z + jk_b\Phi$, the fields are

$$
\begin{aligned}
\mathbf{E}^{free} &= \frac{i}{2\pi v_b\varepsilon_b}\Bigl[\kappa_b K_1(\kappa_b\rho)\,\hat{\mathbf{e}}_\rho
+ jk_b\bigl(1 - v_b^2\mu_b\varepsilon_b\bigr)K_0(\kappa_b\rho)\,\hat{\mathbf{z}}\Bigr]e^{-jk_b z}, \\
\mathbf{H}^{free} &= \frac{i}{2\pi}\,\kappa_b K_1(\kappa_b\rho)\,\hat{\mathbf{e}}_\varphi\;e^{-jk_b z},
\end{aligned}
$$

related by $\mathbf{H}^{free} = v_b\varepsilon_b\,\hat{\mathbf{z}}\times\mathbf{E}^{free}_t$. By construction
they satisfy, everywhere,

$$
\nabla\times\left(\frac{1}{\mu_b}\nabla\times\mathbf{E}^{free}\right) - \omega^2\varepsilon_b\,\mathbf{E}^{free} = -j\omega\,\mathbf{J},
$$

the field equation of [§9.2](#92-field-equation-with-the-beam) with the same source, written for
the reference medium.

- **Singularity.** As $\kappa\rho\to 0$, $\kappa K_1(\kappa\rho)\to 1/\rho$ and
  $K_0(\kappa\rho)\sim -\ln(\kappa\rho)$: $\mathbf{E}^{free}_t$ and $\mathbf{H}^{free}$ grow like
  $1/\rho$, $E^{free}_z$ like $\ln\rho$. The magnetic field near the line,
  $i/(2\pi\rho)\,\hat{\mathbf{e}}_\varphi$, does not depend on the medium (Ampère's law around the line).
- **$\kappa_b = 0$**, i.e. $v_b^2\mu_b\varepsilon_b = 1$ (for example $\beta = 1$ in vacuum):
  $\mathbf{E}^{free} = \dfrac{i}{2\pi v_b\varepsilon_b}\dfrac{\hat{\mathbf{e}}_\rho}{\rho}\,e^{-jk_b z}$ and
  $\mathbf{H}^{free} = \dfrac{i}{2\pi\rho}\,\hat{\mathbf{e}}_\varphi\,e^{-jk_b z}$, without a longitudinal
  component.
- **Faster than light in the medium**, $v_b^2\mu_b\,\mathrm{Re}\,\varepsilon_b > 1$: $\kappa_b^2$ is
  (nearly) negative, $\kappa_b \approx +j\sqrt{\omega^2\mu_b\varepsilon_b - k_b^2}$, and $K_0$ describes
  the outgoing Cherenkov wave. The root $\mathrm{Re}\,\kappa_b \ge 0$ of a lossy medium selects it.

## 9.5 The Beam's Field in a Port

A port face $\Gamma_p$ that the beam crosses lies in a plane $z = z_p$, with outward normal
$\mathbf{n} = n_z\hat{\mathbf{z}}$, $n_z = \pm 1$. It is the cross-section of a uniform pipe that is
filled homogeneously with $\varepsilon_p$, $\mu_p$ and bounded by perfect conductors. In the
infinitely long pipe the beam's field has the structure of the free field,
$A_z = v_b\,\mu_p\varepsilon_p\,\Phi_p$, with the potential (at unit phase) solving (eq. 9–10)

$$
-\nabla_t^2\Phi_p + \kappa_p^2\,\Phi_p = \frac{i}{v_b\varepsilon_p}\,\delta_b \ \ \text{in }\Gamma_p,
\qquad
\Phi_p = 0 \ \text{on the conductors},
$$

with $\kappa_p^2 = k_b^2 - \omega^2\mu_p\varepsilon_p$, $\mathrm{Re}\,\kappa_p \ge 0$. Since $A_z$ and
$\Phi_p$ both vanish on the conductors, $\mathbf{n}\times\mathbf{E} = 0$ there. The fields are

$$
\begin{aligned}
\mathbf{E}^{inc} &= \Bigl[-\nabla_t\Phi_p + jk_b\bigl(1 - v_b^2\mu_p\varepsilon_p\bigr)\Phi_p\,\hat{\mathbf{z}}\Bigr]e^{-jk_b z}, \\
\mathbf{H}^{inc} &= -v_b\varepsilon_p\,\hat{\mathbf{z}}\times\nabla_t\Phi_p\;e^{-jk_b z} .
\end{aligned}
$$

This is the field that comes in, or goes out, with the beam; the pipe carries it without any
modal amplitude. Its tangential magnetic field on the port face, the boundary data of
[§9.2](#92-field-equation-with-the-beam), is

$$
\mathbf{n}\times\mathbf{H}^{inc} = n_z\,v_b\varepsilon_p\,\nabla_t\Phi_p\;e^{-jk_b z_p}
= -n_z\,v_b\varepsilon_p\,\mathbf{E}^{inc}_t .
$$

The potential carries the point singularity of the beam. It is split off with the free
potential of the port medium,

$$
\Phi_p = \Phi_p^{free} + \Phi_p^{reg}, \qquad
\Phi_p^{free} = \frac{i}{2\pi v_b\varepsilon_p}\,K_0(\kappa_p\rho),
$$

where $\Phi_p^{reg}$ solves the homogeneous problem $-\nabla_t^2\Phi_p^{reg} + \kappa_p^2\Phi_p^{reg} = 0$
in $\Gamma_p$, with $\Phi_p^{reg} = -\Phi_p^{free}$ on the conductors. $\Phi_p^{reg}$ is smooth on the
whole face; its transverse field is $\mathbf{E}^{reg}_t = -\nabla_t\Phi_p^{reg}\,e^{-jk_b z_p}$. For
$\kappa_p = 0$ the problem for $\Phi_p$ is electrostatic and $\Phi_p^{reg}$ is harmonic.

## 9.6 Scattered-Field Formulation

The beam is solved for the **scattered field**

$$
\mathbf{E}_s = \mathbf{E} - \mathbf{E}^{free},
$$

with $\mathbf{E}^{free}$ of [§9.4](#94-the-beams-own-field) taken everywhere in $\Omega$.

### The beam current cancels

The total field and the free field satisfy, in $\Omega$,

$$
\begin{aligned}
\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}\right) - \omega^2\varepsilon_c\,\mathbf{E} &= -j\omega\,\mathbf{J}, \\
\nabla\times\left(\frac{1}{\mu_b}\nabla\times\mathbf{E}^{free}\right) - \omega^2\varepsilon_b\,\mathbf{E}^{free} &= -j\omega\,\mathbf{J} .
\end{aligned}
$$

Insert $\mathbf{E} = \mathbf{E}_s + \mathbf{E}^{free}$ into the first equation. The operator acting on
$\mathbf{E}^{free}$ is split into the operator of the reference medium and the difference of the
media:

$$
\begin{aligned}
&\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}^{free}\right) - \omega^2\varepsilon_c\,\mathbf{E}^{free} \\
&\qquad = \underbrace{\nabla\times\left(\frac{1}{\mu_b}\nabla\times\mathbf{E}^{free}\right) - \omega^2\varepsilon_b\,\mathbf{E}^{free}}_{=\;-j\omega\mathbf{J}} \\
&\qquad\quad + \nabla\times\left(\Bigl(\frac{1}{\mu} - \frac{1}{\mu_b}\Bigr)\nabla\times\mathbf{E}^{free}\right)
- \omega^2(\varepsilon_c - \varepsilon_b)\,\mathbf{E}^{free} .
\end{aligned}
$$

The first equation then reads

$$
\begin{aligned}
&\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}_s\right) - \omega^2\varepsilon_c\,\mathbf{E}_s
\;\underbrace{-\,j\omega\mathbf{J}}_{\text{from }\mathbf{E}^{free}} \\
&\qquad + \nabla\times\left(\Bigl(\frac{1}{\mu} - \frac{1}{\mu_b}\Bigr)\nabla\times\mathbf{E}^{free}\right)
- \omega^2(\varepsilon_c - \varepsilon_b)\,\mathbf{E}^{free}
= -j\omega\,\mathbf{J} .
\end{aligned}
$$

The beam current appears on both sides with the same coefficient and cancels. With
$\nabla\times\mathbf{E}^{free} = -j\omega\mu_b\mathbf{H}^{free}$, what remains is

$$
\boxed{
\nabla\times\left(\frac{1}{\mu}\nabla\times\mathbf{E}_s\right) - \omega^2\varepsilon_c\,\mathbf{E}_s
= j\omega\,\nabla\times\left(\Bigl(\frac{\mu_b}{\mu} - 1\Bigr)\mathbf{H}^{free}\right)
+ \omega^2(\varepsilon_c - \varepsilon_b)\,\mathbf{E}^{free}
}
$$

The right-hand side is a **contrast source**: it is non-zero only where the material differs
from the reference medium. In a structure filled with the reference medium the scattered field
satisfies the source-free wave equation, and the beam reaches it only through the boundary
conditions:

- **PEC walls:** $\mathbf{n}\times\mathbf{E} = 0$ becomes $\mathbf{n}\times\mathbf{E}_s = -\mathbf{n}\times\mathbf{E}^{free}$,
  a prescribed tangential value.
- **Natural boundaries** (open port faces, magnetic walls): $\mathbf{n}\times\mathbf{H}$ is prescribed,
  $\mathbf{n}\times\mathbf{H}^{inc}$ on a port face the beam crosses and zero elsewhere; for
  $\mathbf{E}_s$ the data is the part not already carried by $\mathbf{E}^{free}$, as the weak form
  shows.

### Weak form: the load vector without the line source

Testing both field equations with $\mathbf{v}$ and integrating by parts gives

$$
\begin{aligned}
&\int_\Omega \frac{1}{\mu}(\nabla\times\mathbf{E})\cdot(\nabla\times\mathbf{v})\,\mathrm{d}\Omega
- \omega^2\int_\Omega\varepsilon_c\,\mathbf{E}\cdot\mathbf{v}\,\mathrm{d}\Omega \\
&\qquad - j\omega\oint_{\partial\Omega}(\mathbf{n}\times\mathbf{H})\cdot\mathbf{v}\,\mathrm{d}S
= -j\omega\, i\int_{\ell_b}v_z\,e^{-jk_bz}\,\mathrm{d}z,
\end{aligned}
$$

$$
\begin{aligned}
&\int_\Omega \frac{1}{\mu_b}(\nabla\times\mathbf{E}^{free})\cdot(\nabla\times\mathbf{v})\,\mathrm{d}\Omega
- \omega^2\int_\Omega\varepsilon_b\,\mathbf{E}^{free}\cdot\mathbf{v}\,\mathrm{d}\Omega \\
&\qquad - j\omega\oint_{\partial\Omega}(\mathbf{n}\times\mathbf{H}^{free})\cdot\mathbf{v}\,\mathrm{d}S
= -j\omega\, i\int_{\ell_b}v_z\,e^{-jk_bz}\,\mathrm{d}z .
\end{aligned}
$$

The right-hand sides are identical: the line integral of the beam current, the term that made
the total-field problem singular. Subtracting the second equation from the first, with
$\mathbf{E} = \mathbf{E}_s + \mathbf{E}^{free}$ and
$\nabla\times\mathbf{E}^{free} = -j\omega\mu_b\mathbf{H}^{free}$, removes it:

$$
\begin{aligned}
&\int_\Omega \frac{1}{\mu}(\nabla\times\mathbf{E}_s)\cdot(\nabla\times\mathbf{v})\,\mathrm{d}\Omega
- \omega^2\int_\Omega\varepsilon_c\,\mathbf{E}_s\cdot\mathbf{v}\,\mathrm{d}\Omega \\
&\qquad = j\omega\int_\Omega\Bigl(\frac{\mu_b}{\mu} - 1\Bigr)\mathbf{H}^{free}\cdot(\nabla\times\mathbf{v})\,\mathrm{d}\Omega \\
&\qquad\quad + \omega^2\int_\Omega(\varepsilon_c - \varepsilon_b)\,\mathbf{E}^{free}\cdot\mathbf{v}\,\mathrm{d}\Omega \\
&\qquad\quad + j\omega\oint_{\partial\Omega}\mathbf{n}\times(\mathbf{H} - \mathbf{H}^{free})\cdot\mathbf{v}\,\mathrm{d}S .
\end{aligned}
$$

On the PEC walls $\mathbf{n}\times\mathbf{v} = 0$, and on the natural boundaries $\mathbf{n}\times\mathbf{H}$
is the prescribed $\mathbf{n}\times\mathbf{H}^{inc}$. The load vector of the beam is therefore

$$
\begin{aligned}
f^{\,s}_j(\omega) ={}& j\omega\int_\Omega\Bigl(\frac{\mu_b}{\mu} - 1\Bigr)\mathbf{H}^{free}\cdot(\nabla\times\mathbf{N}_j)\,\mathrm{d}\Omega
+ \omega^2\int_\Omega(\varepsilon_c - \varepsilon_b)\,\mathbf{E}^{free}\cdot\mathbf{N}_j\,\mathrm{d}\Omega \\
&+ j\omega\sum_p\oint_{\Gamma_p}\mathbf{n}\times(\mathbf{H}^{inc} - \mathbf{H}^{free})\cdot\mathbf{N}_j\,\mathrm{d}S ,
\end{aligned}
$$

the sum running over all natural boundaries ($\mathbf{H}^{inc} = 0$ where the beam does not cross).
In a structure filled with the reference medium ($\mu = \mu_b$, $\varepsilon_c = \varepsilon_b$) both
volume integrals vanish, and the only source left is the **boundary excitation**:

$$
f^{\,s}_j(\omega) = j\omega\sum_p\oint_{\Gamma_p}\mathbf{n}\times(\mathbf{H}^{inc} - \mathbf{H}^{free})\cdot\mathbf{N}_j\,\mathrm{d}S
$$

on the natural boundaries, together with the prescribed tangential values

$$
\mathbf{n}\times\mathbf{E}_s = -\mathbf{n}\times\mathbf{E}^{free} \quad\text{on the PEC walls} .
$$

None of these terms is singular. The data $\mathbf{n}\times(\mathbf{H}^{inc} - \mathbf{H}^{free})$ is
bounded at the beam point, because both fields approach $i/(2\pi\rho)\,\hat{\mathbf{e}}_\varphi$
there, whatever the media ([§9.4](#94-the-beams-own-field)). On a port face filled with the
reference medium ($\varepsilon_p = \varepsilon_b$, $\mu_p = \mu_b$, so $\Phi_p^{free} = \Phi^{free}$ on the
face), [§9.5](#95-the-beams-field-in-a-port) gives

$$
\mathbf{n}\times(\mathbf{H}^{inc} - \mathbf{H}^{free}) = n_z\,v_b\varepsilon_b\,\nabla_t\Phi_p^{reg}\,e^{-jk_b z_p}
= -n_z\,v_b\varepsilon_b\,\mathbf{E}^{reg}_t ,
$$

the smooth part of the beam's field in the pipe. The PEC data $-\mathbf{n}\times\mathbf{E}^{free}$ is
smooth because the walls keep away from the beam line. Where a contrast region touches the beam
line, the volume integrands grow like $1/\rho$, which is integrable.

### Matrix form

The coefficients of $\mathbf{E}_s$ are split into the free degrees of freedom (f) and those on the
PEC walls (d). The latter are prescribed: $\mathbf{g}(\omega)$, the coefficients of the interpolant
of $-\mathbf{E}^{free}$ on the walls. With
$\mathbf{A}(\omega) = \mathbf{K} + j\omega\,\mathbf{C} - \omega^2(\mathbf{M} - j\mathbf{D})$ partitioned in
the same way,

$$
\mathbf{A}_{ff}(\omega)\,\mathbf{e}_f = \mathbf{f}^{\,s}_f(\omega) - \mathbf{A}_{fd}(\omega)\,\mathbf{g}(\omega),
\qquad
\mathbf{e}_s = \begin{bmatrix}\mathbf{e}_f\\ \mathbf{g}\end{bmatrix} .
$$

The matrix is that of the port excitations, so one factorisation per frequency serves the ports
and the beam. The total field is $\mathbf{E} = \mathbf{E}_s + \mathbf{E}^{free}$; on the beam line
$E_z = E_{s,z} + E^{free}_z$.

## 9.7 Port Voltages and Currents with the Beam

On a port face the beam crosses, the modal voltages and currents describe the field beyond the
beam's own field in the pipe:

$$
V_m = \int_{\Gamma_p}\bigl(\mathbf{E} - \mathbf{E}^{inc}\bigr)\cdot\mathbf{e}_m\,\mathrm{d}S, \qquad
\mathbf{n}\times\bigl(\mathbf{H} - \mathbf{H}^{inc}\bigr) = \sum_m I_m\,\mathbf{e}_m ;
$$

on the other port faces $\mathbf{E}^{inc} = \mathbf{H}^{inc} = 0$, as in [§4](z_parameters.md). The
structure is linear in the port currents $\mathbf{I}$ and the beam current $i$:

$$
\mathbf{V} = \mathbf{Z}\,\mathbf{I} + \mathbf{k}_Z\, i, \qquad
v = \mathbf{h}_Z\,\mathbf{I} + z_{oc}\, i .
$$

- $\mathbf{Z} = j\,\mathbf{B}^T\mathbf{X}$, from the port excitations ([§4](z_parameters.md)).
- $\mathbf{h}_Z = j\,\mathbf{c}(\omega)^T\mathbf{X}$: the beam voltage of the port excitations, whose
  field coefficients are $j\mathbf{X}$.
- $\mathbf{k}_Z$ and $z_{oc}$ come from the beam column of [§9.6](#96-scattered-field-formulation).
  It has $\mathbf{I} = 0$, since $\mathbf{n}\times\mathbf{H} = \mathbf{n}\times\mathbf{H}^{inc}$ on the port
  faces.

The beam column gives

$$
k_{Z,m} = \frac{1}{i}\int_{\Gamma_p}\bigl(\mathbf{E}_s + \mathbf{E}^{free} - \mathbf{E}^{inc}\bigr)\cdot\mathbf{e}_m\,\mathrm{d}S,
$$

$$
z_{oc} = \frac{1}{i}\int_{z_{\min}}^{z_{\max}} E_{s,z}(x_b, y_b, z)\,e^{jk_b z}\,\mathrm{d}z
= \frac{1}{i}\,\mathbf{c}(\omega)^T\mathbf{e}_s .
$$

On a face filled with the reference medium, $\mathbf{E}^{inc}_t - \mathbf{E}^{free}_t = \mathbf{E}^{reg}_t$,
so $\mathbf{k}_Z = \frac{1}{i}\,\mathbf{B}^T(\mathbf{e}_s - \hat{\mathbf{e}}^{reg})$, with $\hat{\mathbf{e}}^{reg}$ the
coefficients of $\mathbf{E}^{reg}_t$ on the face.

The voltage of $\mathbf{E}^{free}$ itself is left out of $z_{oc}$. It is the voltage of the beam in
the unbounded reference medium, which for a line beam is infinite unless $\kappa_b = 0$
($K_0(0) = \infty$), and zero for $\kappa_b = 0$, where $E^{free}_z \equiv 0$. $z_{oc}$ is therefore the
beam impedance of the structure relative to the unbounded reference medium.

## 9.8 Generalised Scattering Matrix

With the pseudo-waves of [§5.3](s_parameters.md#53-z-to-s-conversion),

$$
\mathbf{a} = \tfrac12\,\mathbf{Z}_{\mathrm{ref}}^{-1/2}\bigl(\mathbf{V} + \mathbf{Z}_{\mathrm{ref}}\mathbf{I}\bigr), \qquad
\mathbf{b} = \tfrac12\,\mathbf{Z}_{\mathrm{ref}}^{-1/2}\bigl(\mathbf{V} - \mathbf{Z}_{\mathrm{ref}}\mathbf{I}\bigr),
$$

so that $\mathbf{V} = \mathbf{Z}_{\mathrm{ref}}^{1/2}(\mathbf{a} + \mathbf{b})$ and
$\mathbf{I} = \mathbf{Z}_{\mathrm{ref}}^{-1/2}(\mathbf{a} - \mathbf{b})$. Inserted into the first relation of
[§9.7](#97-port-voltages-and-currents-with-the-beam), they give

$$
\begin{aligned}
\mathbf{b} &= \mathbf{S}\,\mathbf{a} + \mathbf{k}\, i, \\
\mathbf{S} &= \mathbf{Z}_{\mathrm{ref}}^{1/2}(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}(\mathbf{Z} - \mathbf{Z}_{\mathrm{ref}})\,\mathbf{Z}_{\mathrm{ref}}^{-1/2}, \\
\mathbf{k} &= \mathbf{Z}_{\mathrm{ref}}^{1/2}(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}\mathbf{k}_Z .
\end{aligned}
$$

$\mathbf{S}$ is the matrix of [§5.3](s_parameters.md#53-z-to-s-conversion): both forms equal
$\mathbb{1} - 2\,\mathbf{Z}_{\mathrm{ref}}^{1/2}(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}\mathbf{Z}_{\mathrm{ref}}^{1/2}$,
with $\mathbb{1}$ the identity. In the second relation,
$\mathbf{I} = \mathbf{Z}_{\mathrm{ref}}^{-1/2}\bigl[(\mathbb{1} - \mathbf{S})\,\mathbf{a} - \mathbf{k}\, i\bigr]$ and
$\mathbf{Z}_{\mathrm{ref}}^{-1/2}(\mathbb{1} - \mathbf{S}) = 2\,(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}\mathbf{Z}_{\mathrm{ref}}^{1/2}$,
hence

$$
\begin{aligned}
v &= \mathbf{h}\,\mathbf{a} + z_b\, i, \\
\mathbf{h} &= 2\,\mathbf{h}_Z\,(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}\mathbf{Z}_{\mathrm{ref}}^{1/2}, \\
z_b &= z_{oc} - \mathbf{h}_Z\,(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}\mathbf{k}_Z .
\end{aligned}
$$

Together (eq. 1):

$$
\begin{bmatrix}\mathbf{b}\\ v\end{bmatrix} = \tilde{\mathbf{S}}\begin{bmatrix}\mathbf{a}\\ i\end{bmatrix},
\qquad
\tilde{\mathbf{S}} = \begin{bmatrix}\mathbf{S} & \mathbf{k}\\ \mathbf{h} & z_b\end{bmatrix} .
$$

With no incident waves ($\mathbf{a} = 0$) every port mode in $\mathbf{B}$ is terminated in its
reference impedance, $\mathbf{I} = -(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1}\mathbf{k}_Z\, i$. Then
$\mathbf{b} = \mathbf{k}\,i$ are the waves the beam sends into the ports and $v = z_b\, i$: $z_b$ is the
beam impedance of the matched structure, and $\mathbf{h}$ the beam voltage of an incoming wave. A
mode that is not carried in $\mathbf{B}$ sees an open port, a magnetic wall
([§2](variational.md)). Modes that the beam does not excite have zero entries in $\mathbf{k}_Z$ and
$\mathbf{h}_Z$; for a beam on the axis of a rotationally symmetric structure, only the
azimuthally symmetric modes ($\mathrm{TM}_{0n}$) couple.

## 9.9 Concatenation of Segments

Two segments with matrices $\tilde{\mathbf{S}}_1$ and $\tilde{\mathbf{S}}_2$ are joined at a cut, where
port 2 of segment 1 meets port 1 of segment 2 (eq. 11–36). Both faces carry the same port modes
with the same sign convention ([§3.1](ports.md#31-port-eigenvalue-problems),
[§7](concatenation.md)). Each segment's beam quantities refer to its own coordinate, with $z = 0$
at its entrance; segment 2 begins at $z = z_r$. Unlike [§7](concatenation.md), the segments are
joined through their scattering matrices.

Stacked, the two segments are
$\bigl[\mathbf{b}_{\text{blk}};\, \mathbf{v}_{\text{blk}}\bigr] = \tilde{\mathbf{S}}_R\,\bigl[\mathbf{a}_{\text{blk}};\, \mathbf{i}_{\text{blk}}\bigr]$
with $\tilde{\mathbf{S}}_R = \mathrm{diag}(\tilde{\mathbf{S}}_1, \tilde{\mathbf{S}}_2)$,
$\mathbf{i}_{\text{blk}} = (i_1, i_2)$ and $\mathbf{v}_{\text{blk}} = (v_1, v_2)$. A permutation
$\mathbf{P}$ orders the entries into the internal port modes (the two faces of the cut), the
external port modes and the two beam entries:

$$
\mathbf{G} = \mathbf{P}\,\tilde{\mathbf{S}}_R\,\mathbf{P}^T =
\begin{bmatrix}\mathbf{G}_{11} & \mathbf{G}_{12}\\ \mathbf{G}_{21} & \mathbf{G}_{22}\end{bmatrix},
\qquad
\begin{bmatrix}\mathbf{b}_{\text{int}}\\ \mathbf{b}_r\end{bmatrix} = \mathbf{G}\begin{bmatrix}\mathbf{a}_{\text{int}}\\ \mathbf{a}_r\end{bmatrix},
$$

with $\mathbf{a}_r = [\mathbf{a}_{\text{ext}};\, i_1;\, i_2]$ and
$\mathbf{b}_r = [\mathbf{b}_{\text{ext}};\, v_1;\, v_2]$. At the cut, the wave leaving one face is the
wave entering the other, mode by mode:

$$
\mathbf{a}_{\text{int}} = \mathbf{F}\,\mathbf{b}_{\text{int}}, \qquad
\mathbf{F} = \begin{bmatrix}\mathbf{0} & \mathbb{1}\\ \mathbb{1} & \mathbf{0}\end{bmatrix} = \mathbf{F}^{-1} .
$$

With $\mathbf{F}\,\mathbf{a}_{\text{int}} = \mathbf{b}_{\text{int}} = \mathbf{G}_{11}\mathbf{a}_{\text{int}} + \mathbf{G}_{12}\mathbf{a}_r$,
the internal waves are eliminated:

$$
\mathbf{a}_{\text{int}} = (\mathbf{F} - \mathbf{G}_{11})^{-1}\mathbf{G}_{12}\,\mathbf{a}_r, \qquad
\mathbf{b}_r = \bigl(\mathbf{G}_{22} + \mathbf{G}_{21}(\mathbf{F} - \mathbf{G}_{11})^{-1}\mathbf{G}_{12}\bigr)\,\mathbf{a}_r .
$$

The beam is one current that reaches segment 2 later (eq. 26). In segment 2's coordinate
$z' = z - z_r$ its current is $i\,e^{-jk_b z_r}e^{-jk_b z'}$, and the voltage $v_2$ measured there is
$e^{jk_b z_r}v_2$ in the global coordinate:

$$
\begin{bmatrix} i_1\\ i_2\end{bmatrix} = \mathbf{d}\; i, \qquad
v = v_1 + e^{j\psi}\,v_2 = \mathbf{d}^H\begin{bmatrix} v_1\\ v_2\end{bmatrix}, \qquad
\mathbf{d} = \begin{bmatrix}1\\ e^{-j\psi}\end{bmatrix}, \qquad
\psi = k_b z_r = \frac{\omega z_r}{v_b} .
$$

With $\mathbf{T} = \mathrm{diag}(\mathbb{1}_{\text{ext}}, \mathbf{d})$,
$\mathbf{a}_r = \mathbf{T}\,[\mathbf{a}_{\text{ext}};\, i]$ and
$[\mathbf{b}_{\text{ext}};\, v] = \mathbf{T}^H\mathbf{b}_r$, so the joined structure has

$$
\boxed{
\tilde{\mathbf{S}} = \mathbf{T}^H\bigl(\mathbf{G}_{22} + \mathbf{G}_{21}(\mathbf{F} - \mathbf{G}_{11})^{-1}\mathbf{G}_{12}\bigr)\,\mathbf{T}
}
$$

Without port modes at the cut the coupling term is absent and the beam impedances add,
$z_b = z_{b,1} + z_{b,2}$. The modes at the cut carry the interaction between the segments.

## Symbols

| Symbol | Meaning |
|--------|---------|
| $i$ | beam current amplitude; $i = q$ for a point bunch of charge $q$ |
| $v_b = \beta c_0$, $k_b = \omega/v_b$ | beam velocity and beam wavenumber |
| $(x_b, y_b)$, $\ell_b$, $\rho$ | transverse beam position, beam line, distance from it |
| $\varepsilon_b$, $\mu_b$, $\kappa_b$ | reference medium around the beam and its transverse constant $\kappa_b^2 = k_b^2 - \omega^2\mu_b\varepsilon_b$ |
| $\mathbf{E}^{free}$, $\mathbf{H}^{free}$ | the beam's field in the unbounded reference medium |
| $\Phi_p$, $\mathbf{E}^{inc}$, $\mathbf{H}^{inc}$ | potential and field of the beam in the pipe of port $p$ ($\varepsilon_p$, $\mu_p$, $\kappa_p$) |
| $\Phi_p^{reg}$, $\mathbf{E}^{reg}_t$ | smooth part of the port potential and its transverse field |
| $\mathbf{E}_s$ | scattered field $\mathbf{E} - \mathbf{E}^{free}$ |
| $v$, $Z_\parallel$ | beam voltage, longitudinal impedance $-v/i$ |
| $\mathbf{c}(\omega)$ | beam-voltage functional, $v = \mathbf{c}^T\mathbf{x}$ |
| $\mathbf{k}_Z$, $\mathbf{h}_Z$, $z_{oc}$ | open-port coupling and beam impedance |
| $\mathbf{k}$, $\mathbf{h}$, $z_b$, $\tilde{\mathbf{S}}$ | matched coupling, beam impedance and generalised scattering matrix |
| $\mathbf{G}$, $\mathbf{F}$, $\mathbf{d}$, $\mathbf{T}$ | permuted block matrix, cut connection, beam delay, beam projection (concatenation) |

---

**Back to:** [Mathematical Theory](index.md)
