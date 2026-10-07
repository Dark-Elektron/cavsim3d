# 10. Model Order Reduction with the Beam

The reduction of [§6](reduction.md) extends to the beam column of [§9](beam.md). A reduced model
keeps two properties of §6. It is evaluated at any frequency of its band from small precomputed
matrices, and it can be reloaded without the mesh. The beam column puts both at risk:

1. Its field has prescribed, non-zero values on the PEC walls: an inhomogeneous essential
   boundary condition ([§10.2](#102-boundary-conditions-in-the-reduced-model)).
2. Its load and the functional that reads its output change shape with frequency, through the
   beam's phase $e^{-jk_bz}$ ([§10.4](#104-frequency-dependence-in-affine-form)).

!!! note "What the code implements"
    The code does not reduce the beam column yet. This page gives the formulation, for
    $v_b^2\mu_b\varepsilon_b = 1$ as in [§9](beam.md).

## 10.1 The Full-Order Beam Column

The degrees of freedom are split into the free ones (f) and those on the PEC walls (d). For the
beam current $i$, the beam column of [§9.6](beam.md#96-scattered-field-formulation) solves

$$
\mathbf{A}_{ff}(\omega)\,\mathbf{e}_f = \mathbf{b}(\omega), \qquad
\mathbf{b}(\omega) = \mathbf{f}^{\,s}_f(\omega) - \mathbf{A}_{fd}(\omega)\,\mathbf{g}(\omega), \qquad
\mathbf{e}_s = \begin{bmatrix}\mathbf{e}_f\\ \mathbf{g}(\omega)\end{bmatrix},
$$

with $\mathbf{A}(\omega) = \mathbf{K} + j\omega\,\mathbf{C} - \omega^2(\mathbf{M} - j\mathbf{D})$, the load
$\mathbf{f}^{\,s}(\omega)$ from the port faces and from materials that differ from the reference
medium, and the wall data $\mathbf{g}(\omega)$, the interpolant of $-\mathbf{E}^{free}$ on the wall
degrees of freedom. Its outputs ([§9.7](beam.md#97-port-voltages-and-currents-with-the-beam)) are

$$
z_{oc} = \frac{1}{i}\,\mathbf{c}(\omega)^T\mathbf{e}_s, \qquad
\mathbf{k}_Z = \frac{1}{i}\,\mathbf{B}^T\bigl(\mathbf{e}_s + \mathbf{q}(\omega)\bigr), \qquad
\mathbf{h}_Z = j\,\mathbf{c}(\omega)^T\mathbf{X},
$$

where $\mathbf{c}(\omega) = \sum_k w_k\,e^{jk_bz_k}\,\mathbf{p}_k$ reads the beam voltage at the
Gauss points $z_k$ (weights $w_k$) of the beam line, $(p_k)_j = N_{j,z}(x_b, y_b, z_k)$, and
$\mathbf{q}(\omega)$ holds the coefficients of $(\mathbf{E}^{free} - \mathbf{E}^{inc})_t$ on the port
faces:

$$
\mathbf{q}(\omega) = -\sum_{p \in \mathcal{P}_b} e^{-jk_bz_p}\,\hat{\mathbf{e}}^{reg}_p
+ \sum_{p \notin \mathcal{P}_b} \hat{\mathbf{e}}^{free}_p(\omega) .
$$

$\mathcal{P}_b$ is the set of port faces the beam crosses, $\hat{\mathbf{e}}^{reg}_p$ the coefficients
of $\mathbf{E}^{reg}_t$ on face $p$ at unit phase ([§9.5](beam.md#95-the-beams-field-in-a-port)), and
$\hat{\mathbf{e}}^{free}_p(\omega)$ the coefficients of $\mathbf{E}^{free}_t$ on a face the beam does
not cross, where $\mathbf{E}^{inc} = 0$.

## 10.2 Boundary Conditions in the Reduced Model

!!! warning "Boundary conditions of the beam column"
    **Port faces** carry natural conditions: $\mathbf{n}\times\mathbf{H}_s$ is prescribed, and it
    enters the problem as the load $\mathbf{f}^{\,s}$. In the reduced model it is a projected load,
    as for the port excitations, and it puts no condition on the basis.

    **PEC walls** carry an essential condition, and for the beam column it is inhomogeneous:
    $\mathbf{n}\times\mathbf{E}_s = -\mathbf{n}\times\mathbf{E}^{free}$
    ([§9.6](beam.md#96-scattered-field-formulation)), whereas the port columns have
    $\mathbf{n}\times\mathbf{E} = 0$. The reduced space must satisfy the homogeneous condition,
    and the data are carried by a **lift**[^qmn]:

    $$
    \mathbf{e}_s = \begin{bmatrix}\mathbf{e}_f\\ \mathbf{0}\end{bmatrix}
    + \begin{bmatrix}\mathbf{0}\\ \mathbf{g}(\omega)\end{bmatrix},
    \qquad
    \mathbf{e}_f \approx \mathbf{V}\mathbf{Q}_L^{-1}\mathbf{y}_b .
    $$

    Only the first part is reduced. The lift, which is zero on the free degrees of freedom,
    imposes the wall values exactly at every frequency.

Both halves of the rule are needed:

- **Test functions.** The Galerkin projection tests with the basis vectors. The weak form holds
  only for test functions that vanish on the walls: a wall row of $\mathbf{A}$ contains the
  boundary term $-j\omega\oint(\mathbf{n}\times\mathbf{H})\cdot\mathbf{v}\,\mathrm{d}S$ with the
  unknown wall current, so it is not an equation of the problem. A basis vector with wall values
  would bring such a row into the reduced equations.
- **Trial functions.** The wall values of a combination $\mathbf{V}\mathbf{y}$ are combinations of
  the snapshots' wall values $\mathbf{g}(\omega_i)$. They match $\mathbf{g}(\omega)$ only at the
  snapshot frequencies; in between, the reduced field would violate the wall condition.

The snapshots of the beam column are therefore its free part $\mathbf{e}_f(\omega_i)$, zero on the
walls like the port snapshots, and one basis serves both
([§10.3](#103-snapshots-and-basis)).

The lift then appears in the reduced load as $-\mathbf{V}^T\mathbf{A}_{fd}(\omega)\,\mathbf{g}(\omega)$,
and this is the only place where the walls enter. $\mathbf{g}(\omega)$ carries the beam's phase along
the walls, so it changes shape with frequency. Formed directly at each frequency it needs the
mesh, and the reduced model would not be portable. [§10.4](#104-frequency-dependence-in-affine-form)
replaces it by a short expansion with precomputed reduced vectors.

$\mathbf{g} = 0$ on a round wall centred on the beam line, where $\mathbf{E}^{free}$ is normal to the
wall. Walls perpendicular to the beam (cavity end walls, irises) carry $\mathbf{E}^{free}$
tangentially, and so do tapers and steps: there $\mathbf{g} \ne 0$.

## 10.3 Snapshots and Basis

The beam snapshots are complex (the beam's phase), so their real and imaginary parts enter
separately, as in [§6](reduction.md). The port columns are fields per unit modal current, the beam
column a field per unit beam current, and the truncation $\sigma_i/\sigma_1 > \text{tol}$ would drop
the smaller family entirely. Each family is therefore scaled by its own largest singular value
before the SVD:

$$
\mathbf{X}_s = \bigl[\,\mathbf{X}_{ports}/\sigma_1^{ports} \;\big|\; \mathbf{X}_{beam}/\sigma_1^{beam}\,\bigr],
\qquad
\mathbf{X}_{beam} = \bigl[\,\mathrm{Re}\,\mathbf{e}_f(\omega_1) \mid \mathrm{Im}\,\mathbf{e}_f(\omega_1) \mid \dots\,\bigr] .
$$

Truncation, projection and the mass-weighted transformation follow [§6](reduction.md) and give
$\mathbf{V}$, $\mathbf{Q}_L^{-1}$, $\hat{\mathbf{A}}$, $\hat{\mathbf{B}}$, $\hat{\mathbf{C}}$ and
$\hat{\mathbf{D}}$. A reduced vector is written $\hat{\mathbf{u}} = (\mathbf{Q}_L^{-1})^T\mathbf{V}^T\mathbf{u}$
for any full-order vector $\mathbf{u}$ restricted to the free degrees of freedom.

## 10.4 Frequency Dependence in Affine Form

A reduced model evaluates its load and outputs without the mesh only if every frequency-dependent
vector has the **affine form** $\sum_l \alpha_l(\omega)\,\mathbf{u}_l$: known scalar functions
$\alpha_l$ times fixed vectors $\mathbf{u}_l$, which are projected once.

### Separable terms

Three terms are already of this form, exactly:

- **The load on a port face the beam crosses.** The face is perpendicular to the beam, so the phase
  is the constant $e^{-jk_bz_p}$ on it, and for $v_b^2\mu_b\varepsilon_b = 1$ the profile of
  $\mathbf{n}\times(\mathbf{H}^{inc} - \mathbf{H}^{free})$ does not depend on $\omega$
  ([§9.5](beam.md#95-the-beams-field-in-a-port)):

    $$
    \mathbf{f}^{\,s}_{\mathcal{P}_b}(\omega) = \sum_{p\in\mathcal{P}_b} j\omega\,e^{-jk_bz_p}\,\mathbf{f}_p,
    \qquad
    (f_p)_j = n_z\,v_b\varepsilon_b\oint_{\Gamma_p}\nabla_t\Phi_p^{reg}\cdot\mathbf{N}_j\,\mathrm{d}S .
    $$

    A face the beam does not cross but which is perpendicular to it also has a constant phase, and
    its load $j\omega\oint(-\mathbf{n}\times\mathbf{H}^{free})\cdot\mathbf{N}_j\,\mathrm{d}S$ separates in
    the same way.

- **The beam-voltage functional** $\mathbf{c}(\omega) = \sum_k w_k\,e^{jk_bz_k}\,\mathbf{p}_k$, a finite
  sum with one term per Gauss point.
- **The port-face field** of the crossed faces in $\mathbf{q}(\omega)$,
  $-\sum_p e^{-jk_bz_p}\,\hat{\mathbf{e}}^{reg}_p$.

### Phase integrals

The other terms are **phase integrals**: vectors whose entries are integrals of a fixed profile
times $e^{-jk_bz}$ over a region in which $z$ varies. In discrete form,

$$
\mathbf{u}(\omega) = \mathbf{W}\,\mathbf{\Theta}(\omega), \qquad
\theta_q(\omega) = e^{-j\omega z_q/v_b},
$$

with one entry $\theta_q$ of the phase vector $\mathbf{\Theta}$ per quadrature point $z_q$ of the region and a fixed matrix $\mathbf{W}$:
interpolation onto the walls and assembly of a load are both linear in the values at the
quadrature points. Four terms are phase integrals:

| Term | Region | Enters |
|------|--------|--------|
| wall lift $\mathbf{g}(\omega)$ | PEC walls | the load, as $-\mathbf{A}_{fd}(\omega)\,\mathbf{g}(\omega)$; $\mathbf{k}_Z$ and $z_{oc}$ |
| contrast load | materials that differ from the reference medium | the load |
| load on a port face the beam does not cross, unless perpendicular to it | that face | the load |
| $\hat{\mathbf{e}}^{free}_p(\omega)$ in $\mathbf{q}(\omega)$ | the same faces | $\mathbf{k}_Z$ |

The contrast load of [§9.6](beam.md#96-scattered-field-formulation) is a polynomial in $\omega$
times phase integrals. With $\nabla\times\mathbf{E}^{free} = -j\omega\mu_b\mathbf{H}^{free}$ and the
dielectric part of the permittivity $\varepsilon_d = \varepsilon_0\varepsilon_r(1 - j\tan\delta)$, its
entries are

$$
j\omega\int_{\Omega_m}\Bigl(\frac{\mu_b}{\mu} - 1\Bigr)\mathbf{H}^{free}\cdot(\nabla\times\mathbf{N}_j)\,\mathrm{d}\Omega
+ \omega^2\int_{\Omega_m}(\varepsilon_d - \varepsilon_b)\,\mathbf{E}^{free}\cdot\mathbf{N}_j\,\mathrm{d}\Omega
- j\omega\int_{\Omega_m}\sigma\,\mathbf{E}^{free}\cdot\mathbf{N}_j\,\mathrm{d}\Omega .
$$

An exact separation of a phase integral needs one term per quadrature point, which is as many as
the region has in the mesh. The dependence on $\omega$ is known in closed form, however, and is
replaced by an interpolant in $\omega$.

### Chebyshev interpolation in frequency

Let $[\omega_1, \omega_2]$ be the band in which the reduced model will be evaluated, with centre
$\omega_c$ and half-width $\Delta = (\omega_2 - \omega_1)/2$. Let $L$ be the extent along the beam of
the region of a phase integral (for the lift, of the walls on which
$\mathbf{n}\times\mathbf{E}^{free} \ne 0$) and $z_c$ its centre. The common phase of the centre is
taken out exactly,

$$
\mathbf{u}(\omega) = e^{-jk_bz_c}\,\hat{\mathbf{u}}(\omega), \qquad
\hat{\mathbf{u}}(\omega) = \mathbf{W}\,\hat{\mathbf{\Theta}}(\omega), \qquad
\hat\theta_q(\omega) = e^{-j\omega(z_q - z_c)/v_b},
$$

and $\hat{\mathbf{u}}$ is interpolated at the $m$ Chebyshev points
$\omega_l = \omega_c - \Delta\cos\bigl(\pi(l-1)/(m-1)\bigr)$, $l = 1, \dots, m$:

$$
\boxed{
\mathbf{u}(\omega) \approx e^{-jk_bz_c}\sum_{l=1}^{m}\ell_l(\omega)\,e^{jk_lz_c}\,\mathbf{u}(\omega_l),
\qquad k_l = \frac{\omega_l}{v_b},
}
$$

with the Lagrange polynomials $\ell_l$ of the points $\omega_l$, evaluated with the barycentric
formula[^trefethen]. Each node costs one evaluation of the phase integral on the mesh, with no solve,
and the nodes are independent of the snapshot frequencies. This is the expansion that the
empirical interpolation method[^eim] constructs from samples when the parameter dependence is not
known; here it is known, so the interpolant comes with an a-priori error bound.

**Error bound.** $\mathbf{W}$ is fixed, so interpolating $\hat{\mathbf{u}}$ interpolates each
$\hat\theta_q$ and applies $\mathbf{W}$. With $\xi = (\omega - \omega_c)/\Delta \in [-1, 1]$,
$\hat\theta_q = e^{-j\omega_c\tau_q}\,e^{-jc_q\xi}$, where $\tau_q = (z_q - z_c)/v_b$ and
$|c_q| = \Delta|\tau_q| \le c$, with

$$
c = \frac{\Delta L}{2v_b} = \frac{\pi(f_2 - f_1)L}{2v_b} .
$$

The first factor does not depend on $\omega$. The Chebyshev coefficients of the second are
$2(-j)^nJ_n(c_q)$ (the Jacobi–Anger expansion[^dlmf]), and the interpolant through $m$ Chebyshev
points errs by at most twice the sum of the omitted coefficients[^trefethen]. For $m > c$, where
$J_n$ increases on $[0, c]$ for every $n \ge m$, and with the inequality[^dlmf]
$|J_n(x)| \le (x/2)^n/n!$ for real $x \ge 0$,

$$
\bigl\|\mathbf{u}(\omega) - \mathbf{u}_m(\omega)\bigr\|_\infty \le \|\mathbf{W}\|_\infty\,\epsilon_m,
\qquad
\epsilon_m = 4\sum_{n\ge m}|J_n(c)| \le 4\sum_{n \ge m}\frac{(c/2)^n}{n!},
$$

for every $\omega$ in the band, with $\mathbf{u}_m$ the interpolant. $\|\mathbf{W}\|_\infty$ is the size
of the entries of $\mathbf{u}$ (without cancellation of the phase), so $\epsilon_m$ is a relative
error. It falls faster than exponentially once $m$ exceeds $c$:

| $c$ | 1 | 5 | 10 | 20 | 40 | 80 |
|-----|---|---|----|----|----|----|
| $m$ for $\epsilon_m \le 10^{-12}$ | 13 | 23 | 32 | 47 | 73 | 121 |

For a structure 1 m long, a band from 0 to 3 GHz and $\beta = 1$, $c = 15.7$ and $m = 40$ nodes
reach $10^{-12}$. $m$ is chosen so that $\epsilon_m$ lies well below the reduction tolerance; the lift
then adds no visible error.

!!! warning "Interpolation band"
    Outside $[\omega_1, \omega_2]$ the polynomial interpolant diverges quickly. The band must
    cover every frequency at which the reduced model is evaluated. It can be wider than the band
    of the snapshots: $m$ grows only linearly with its width.

### Reduced load and outputs in affine form

With the interpolant, the projected lift is

$$
(\mathbf{Q}_L^{-1})^T\mathbf{V}^T\mathbf{A}_{fd}(\omega)\,\mathbf{g}(\omega) \approx
e^{-jk_bz_c}\sum_{l=1}^{m}\ell_l(\omega)
\Bigl[\hat{\mathbf{g}}^K_l + j\omega\,\hat{\mathbf{g}}^C_l - \omega^2\bigl(\hat{\mathbf{g}}^M_l - j\,\hat{\mathbf{g}}^D_l\bigr)\Bigr],
$$

with the reduced vectors, computed once from the mesh,

$$
\hat{\mathbf{g}}^K_l = e^{jk_lz_c}\,(\mathbf{Q}_L^{-1})^T\mathbf{V}^T\mathbf{K}_{fd}\,\mathbf{g}(\omega_l)
$$

and likewise $\hat{\mathbf{g}}^C_l$, $\hat{\mathbf{g}}^M_l$, $\hat{\mathbf{g}}^D_l$ with $\mathbf{C}_{fd}$,
$\mathbf{M}_{fd}$, $\mathbf{D}_{fd}$. The contrast load and the loads of the faces the beam does not
cross are projected in the same way, each phase integral with its own $z_c$, and keep their
polynomial factors in $\omega$. The parts of the outputs that involve the walls or those faces become
small matrices: $\mathbf{B}_d^T\mathbf{g}(\omega_l)$ and $\mathbf{B}^T\hat{\mathbf{e}}^{free}_p(\omega_l)$
($N_{pm}\times m$), and $\mathbf{P}_d^T\mathbf{g}(\omega_l)$ ($n_k\times m$), where
$\mathbf{P} = [\mathbf{p}_1 \mid \dots \mid \mathbf{p}_{n_k}]$ and the index $d$ selects the wall rows.

The reduced model then stores, besides the matrices of [§6](reduction.md):

- $\hat{\mathbf{f}}_p$ and $\mathbf{B}^T\hat{\mathbf{e}}^{reg}_p$ for every crossed face, with $z_p$;
- $\hat{\mathbf{c}}_k$, $z_k$ and $w_k$ for the Gauss points of the beam line;
- for every phase integral: $\omega_l$, $z_c$ and the reduced vectors (an $r\times m$ matrix per term
  of its polynomial), and the output matrices above;
- $v_b$ and the beam position $(x_b, y_b)$ it was built for.

None of these needs the mesh.

## 10.5 Reduced Solve and Outputs

The reduced load is

$$
\hat{\mathbf{b}}(\omega) = \sum_{p\in\mathcal{P}_b} j\omega\,e^{-jk_bz_p}\,\hat{\mathbf{f}}_p
+ \bigl[\hat{\mathbf{f}}^{\,s}_{c}\bigr]_m(\omega)
- \bigl[(\mathbf{Q}_L^{-1})^T\mathbf{V}^T\mathbf{A}_{fd}\,\mathbf{g}\bigr]_m(\omega),
$$

where $\hat{\mathbf{f}}^{\,s}_{c}$ is the reduced contrast load together with the loads of the
faces the beam does not cross, and $[\,\cdot\,]_m$ is the interpolant of
[§10.4](#104-frequency-dependence-in-affine-form). The reduced beam column solves

$$
\bigl(\hat{\mathbf{A}} + j\omega\hat{\mathbf{C}} - \omega^2(\mathbf{I} - j\hat{\mathbf{D}})\bigr)\,\mathbf{y}_b = \hat{\mathbf{b}}(\omega),
\qquad
\mathbf{e}_s \approx \begin{bmatrix}\mathbf{V}\mathbf{Q}_L^{-1}\mathbf{y}_b\\ \mathbf{g}(\omega)\end{bmatrix} .
$$

For a lossless structure the eigendecomposition $\hat{\mathbf{A}} = \mathbf{\Phi}\mathbf{\Lambda}\mathbf{\Phi}^T$
of [§6](reduction.md) solves every frequency at once,
$\mathbf{y}_b = \mathbf{\Phi}\,\mathrm{diag}\bigl(1/(\lambda_i - \omega^2)\bigr)\mathbf{\Phi}^T\hat{\mathbf{b}}(\omega)$.
The eigenvalues near zero belong to the quasi-static part of the field, which at low frequency
carries most of the beam's field; they stay in the sum.

With $\hat{\mathbf{c}}_k = (\mathbf{Q}_L^{-1})^T\mathbf{V}^T\mathbf{p}_k$, the outputs are

$$
\begin{aligned}
z_{oc} &= \frac{1}{i}\sum_k w_k\,e^{jk_bz_k}\Bigl(\hat{\mathbf{c}}_k^T\mathbf{y}_b + \mathbf{p}_{k,d}^T\,\mathbf{g}(\omega)\Bigr), \\
\mathbf{h}_Z &= j\sum_k w_k\,e^{jk_bz_k}\,\hat{\mathbf{c}}_k^T\,\mathbf{Y}, \\
\mathbf{k}_Z &= \frac{1}{i}\Bigl(\hat{\mathbf{B}}^T\mathbf{y}_b + \mathbf{B}_d^T\,\mathbf{g}(\omega) + \mathbf{B}^T\mathbf{q}(\omega)\Bigr),
\end{aligned}
$$

with $\mathbf{Y}$ the reduced port columns of [§6](reduction.md) and the wall and face terms taken from
the interpolants. $\mathbf{p}_{k,d}^T\mathbf{g}$ is zero unless an element on the beam line touches a
PEC wall. $\mathbf{B}_d^T\mathbf{g}$ reaches only the wall degrees of freedom on the rims of the port
faces, whose basis functions extend into the face. $\mathbf{Z} = j\hat{\mathbf{B}}^T\mathbf{Y}$ is that
of [§6](reduction.md), and $\tilde{\mathbf{S}}$ follows as in
[§9.8](beam.md#98-generalised-scattering-matrix).

## 10.6 Accuracy and Sampling

- **Rank.** The truncation bounds the error of the field in the norm of the snapshots, which the
  large transverse field near the walls dominates. $z_{oc}$ comes from $E_z$ on the beam line, which
  can be a small part of that field, so the rank has to be judged by $z_{oc}$, not by the singular
  values alone.
- **Sampling.** The beam's phase across a structure of length $L$ repeats every $\Delta f = v_b/L$,
  and the beam column changes with frequency at least that fast. The snapshots need a spacing below
  $v_b/(2L)$, and the basis needs about $2f_{\max}L/v_b$ vectors for the phase alone. Short
  segments ([§9.9](beam.md#99-concatenation-of-segments)) keep both small.
- **Validity.** A reduced model holds for the beam velocity $v_b$ and the beam position
  $(x_b, y_b)$ of its snapshots, and for the interpolation band of its phase integrals.

## 10.7 Joining Reduced Segments

A reduced model gives $\tilde{\mathbf{S}}$ at any frequency of its band, exactly as a full-order model
does ([§9.8](beam.md#98-generalised-scattering-matrix)). Reduced segments are therefore joined
through their generalised scattering matrices as in [§9.9](beam.md#99-concatenation-of-segments),
with the beam delay $\mathbf{d}$ of each segment.

## Symbols

| Symbol | Meaning |
|--------|---------|
| f, d | free and PEC-wall degrees of freedom |
| $\mathbf{e}_f$, $\mathbf{g}(\omega)$ | free part of the beam column, wall lift (interpolant of $-\mathbf{E}^{free}$ on the walls) |
| $\mathbf{b}(\omega)$, $\hat{\mathbf{b}}(\omega)$ | load of the beam column on the free degrees of freedom, and its reduced form |
| $\mathbf{y}_b$ | reduced beam column |
| $\mathcal{P}_b$ | port faces the beam crosses |
| $\mathbf{q}(\omega)$ | coefficients of $(\mathbf{E}^{free} - \mathbf{E}^{inc})_t$ on the port faces |
| $\mathbf{f}_p$, $\hat{\mathbf{f}}_p$ | load of a crossed face at unit phase, and its reduced form |
| $\mathbf{p}_k$, $\hat{\mathbf{c}}_k$, $z_k$, $w_k$ | beam-line values at a Gauss point, their reduced form, the point and its weight |
| $\mathbf{W}$, $\mathbf{\Theta}(\omega)$ | fixed matrix and phase vector ($\theta_q = e^{-j\omega z_q/v_b}$) of a phase integral |
| $\omega_l$, $\ell_l$, $m$ | Chebyshev points, Lagrange polynomials, number of points |
| $z_c$, $L$, $c$ | centre and extent of a phase integral's region, $c = \Delta L/(2v_b)$ |
| $\epsilon_m$ | relative interpolation error bound |
| $\hat{\mathbf{g}}^K_l, \dots, \hat{\mathbf{g}}^D_l$ | reduced lift vectors at the node $\omega_l$ |

## References

///Footnotes Go Here///

[^qmn]: A. Quarteroni, A. Manzoni and F. Negri, *Reduced Basis Methods for Partial Differential
    Equations: An Introduction*, UNITEXT vol. 92 (Springer, Cham, 2016).
    [doi:10.1007/978-3-319-15431-2](https://doi.org/10.1007/978-3-319-15431-2)
[^trefethen]: L. N. Trefethen, *Approximation Theory and Approximation Practice*, extended ed.
    (SIAM, Philadelphia, 2019), ch. 4–5.
    [doi:10.1137/1.9781611975949](https://doi.org/10.1137/1.9781611975949)
[^eim]: M. Barrault, Y. Maday, N. C. Nguyen and A. T. Patera, "An 'empirical interpolation'
    method: application to efficient reduced-basis discretization of partial differential
    equations," *C. R. Math. Acad. Sci. Paris* **339**(9), 667–672 (2004).
    [doi:10.1016/j.crma.2004.08.006](https://doi.org/10.1016/j.crma.2004.08.006)
[^dlmf]: F. W. J. Olver et al. (eds.), *NIST Digital Library of Mathematical Functions*,
    [dlmf.nist.gov](https://dlmf.nist.gov/), §10.12 (Jacobi–Anger expansion) and §10.14
    (inequalities).

---

**Back to:** [Mathematical Theory](index.md)
