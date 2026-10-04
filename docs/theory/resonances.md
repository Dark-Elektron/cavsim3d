# 8. Resonant Modes

The resonances of a model are the non-trivial solutions of the source-free problem, the
generalised eigenvalue problem

$$
\mathbf{K}\,\mathbf{x} = \omega^2\,\mathbf{M}\,\mathbf{x}
$$

(for a reduced model, $\hat{\mathbf{A}}\,\mathbf{y} = \omega^2\,\mathbf{y}$; for a coupled one,
$\mathbf{A}_{\text{coupled}}\,\mathbf{y} = \omega^2\,\mathbf{y}$). Three properties matter when reading
the results:

- **Port faces are magnetic walls.** An undriven port is an open circuit ([§2](variational.md)), so the
  resonances are those of the structure with $\mathbf{n}\times\mathbf{H} = 0$ on every port face,
  not those of a closed metal cavity. A waveguide section of length $L$ with PEC walls
  resonates at $f = \tfrac{c}{2}\sqrt{(m/a)^2 + (n/b)^2 + (p/L)^2}$ with $p \ge 0$ for TE modes and
  $p \ge 1$ for TM modes -- the TE$_{mn0}$ resonance sits exactly at the cutoff frequency. (A fully
  closed PEC box has the opposite rule: TE modes from $p = 1$, TM modes from $p = 0$.)
- **The static null space.** Every gradient field is an eigenvector with $\omega^2 = 0$ ([§2](variational.md)). A
  numerical eigensolver returns these at round-off level rather than exactly zero, so modes
  below 1 MHz are treated as static and removed.
- **Shift-invert.** The full-order problem is large and sparse, so only the modes nearest a
  target $\sigma$ (in $\omega^2$) are computed. Because the static modes sit at distance $\sigma$ from
  it, only physical modes below $2\sigma$ can be found. The default target is the centre of the
  solved band, $\sigma = \tfrac12(\omega_\text{min}^2 + \omega_\text{max}^2)$, which places every
  in-band mode closer than the static ones; before any sweep it is $(2\pi c_0/L)^2$ for the
  model's largest extent $L$. A different target can be passed as `sigma`.
- **Reduced and joined models are trusted near their band.** Far from the band its snapshots
  covered, the projection of a reduced model has spurious eigenvalues. Its resonances are
  therefore listed only within 10 % of the training band's edges (`fmin=` and `fmax=` list
  others), and every method that takes a mode index counts that list.

The eigenvalue analysis uses the lossless operators $\mathbf{K}$ and $\mathbf{M}$; losses
($\mathbf{C}$, $\mathbf{D}$) shift and damp the resonances but are not included in it.

## 8.1 Loaded Resonances

With every port mode terminated in its reference impedance $Z_0$ -- the matched load the
S-parameters assume -- power leaves through the ports, and the resonances of a reduced model
become the eigenvalues of

$$
\left(\hat{\mathbf{A}} + j\omega\,\hat{\mathbf{B}}\,\mathbf{Y}_0\,\hat{\mathbf{B}}^T - \omega^2\,\mathbf{I}\right)\mathbf{y} = \mathbf{0},
\qquad \mathbf{Y}_0 = \mathrm{diag}(1/Z_0) .
$$

For a fixed $\mathbf{Y}_0$ this is a quadratic eigenvalue problem, solved exactly through its
linearisation of size $2r$,

$$
\begin{bmatrix} \mathbf{0} & \mathbf{I} \\ \hat{\mathbf{A}} & j\,\hat{\mathbf{B}}\mathbf{Y}_0\hat{\mathbf{B}}^T \end{bmatrix}
\begin{bmatrix} \mathbf{y} \\ \omega\,\mathbf{y} \end{bmatrix}
= \omega
\begin{bmatrix} \mathbf{y} \\ \omega\,\mathbf{y} \end{bmatrix} .
$$

Its complex eigenvalues are the loaded resonances, with the loaded quality factor
$Q_L = \mathrm{Re}\,\omega / (2\,\mathrm{Im}\,\omega)$. The power a port takes from the mode is
$\mathrm{Re}(Y_{0,k})\,|(\hat{\mathbf{B}}^T\mathbf{y})_k|^2$ summed over the port's modes $k$,
and its external Q splits the damping in that proportion, $Q_{\text{ext},p} = Q_L\,P/P_p$, so
that $1/Q_L = \sum_p 1/Q_{\text{ext},p}$. A port mode below cutoff has an imaginary $Z_0$: it
takes no power, but its reactance still shifts the resonance.

The $Z_0$ of a TE or TM mode depends on frequency, so $\mathbf{Y}_0$ is taken at the resonance
itself. The problem is solved with $\mathbf{Y}_0$ at the closed-problem frequency of the mode,
then again at the loaded frequency found, until that frequency settles. Each loaded resonance
belongs to the closed-problem mode its eigenvector overlaps most, one to one; modes within
0.1 % of each other are matched together, so both members of a degenerate pair keep their own.
A strongly damped mode whose best match is another mode's resonance has none of its own.

Only the external loading enters here. The losses in the walls and the materials give the
unloaded Q of the figures of merit instead.

---

**Next:** [9. Beam excitation](beam.md)
