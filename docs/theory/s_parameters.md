# 5. Reference Impedances and S-Parameters

## 5.1 Characteristic (Wave) Impedance
Each port mode has a frequency-dependent characteristic impedance. With the propagation
constant $\gamma_m = \sqrt{k_{c,m}^2 - \varepsilon_r\mu_r k_0^2}$ (taken with
$\mathrm{Re}\,\gamma_m \ge 0$), the TE, TM and TEM wave impedances are

$$
Z_\mathrm{TE} = \frac{j\omega\mu}{\gamma_m}, \qquad
Z_\mathrm{TM} = \frac{\gamma_m}{j\omega\varepsilon}, \qquad
Z_\mathrm{TEM} = \eta = \sqrt{\mu/\varepsilon} .
$$

Above cutoff $\gamma_m = j\beta_m$ and these reduce to the familiar forms

$$ Z_\mathrm{TE} = \frac{\eta}{\sqrt{1 - \left( \frac{f_{c,m}}{f} \right)^2}} , \qquad
   Z_\mathrm{TM} = \eta \sqrt{1 - \left( \frac{f_{c,m}}{f} \right)^2} . $$

Below cutoff $\gamma_m$ is real and positive (an evanescent, decaying mode for $e^{j\omega t}$),
so $Z_\mathrm{TE}$ is **inductive** ($+j$) and $Z_\mathrm{TM}$ is **capacitive** ($-j$). The square
roots in the familiar forms must then be read on that branch,
$\sqrt{1 - (f_c/f)^2} = -j\sqrt{(f_c/f)^2 - 1}$.

Here $f_{c,m}$ is the cutoff frequency of the $m$-th mode at port $p$. Both $\eta$ and $f_{c,m}$
are taken in the medium that fills the port: $\eta = \eta_0\sqrt{\mu_r/\varepsilon_r}$ and
$f_{c,m} = c_0 k_{c,m} / (2\pi\sqrt{\varepsilon_r\mu_r})$, where $k_{c,m}$ depends only on the
cross-section. A dielectric filling therefore lowers the cutoff as well as the impedance.

With the modes normalised as in [§3.2](ports.md#32-building-the-right-hand-side-b), a single forward-travelling mode has
$I_m = V_m / Z_{w,m}$ and carries the power $\tfrac12 V_m I_m^*$: the modal $V$ and $I$ are
normalised to the **wave impedance** $Z_{w,m}$, which is therefore the natural reference.
The reference impedance matrix is:

$$ \mathbf{Z}_{\mathrm{ref}} = \begin{bmatrix}
Z_{\mathrm{ref},1,1} & 0 & \dots & 0 \\
0 & Z_{\mathrm{ref},1,2} & \dots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \dots & Z_{\mathrm{ref},P,m_P}
\end{bmatrix} $$

where $Z_{\mathrm{ref},p,m}$ is the reference impedance of port $p$, mode $m$. It could be a TE, TM, or TEM mode.

## 5.2 Reference Impedance: Wave versus Line

The impedances of [§5.1](#51-characteristic-wave-impedance) are physical properties of the mode. The impedance used to *normalise*
$\mathbf{S}$ is a separate matter -- a **convention** -- and for TEM and quasi-TEM modes the
two do not coincide.

A TE or TM mode has no usable voltage between conductors. The transverse field of a TE mode
is not a gradient, so $\int \mathbf{E}\cdot\mathrm{d}\mathbf{l}$ between two conductor points
depends on the path taken. The transverse field of a TM mode is the gradient $-\nabla_t E_z$,
but $E_z = 0$ on every conductor, so that integral is zero for any path. The wave impedance is
then the only impedance available, and it is used as the reference.

A TEM mode does possess a unique voltage and current, because its transverse electric field
is the gradient of a potential that takes a *different* constant value on each conductor.
Three further impedances therefore exist,

$$ Z_{PV} = \frac{|V|^2}{2P}, \qquad Z_{PI} = \frac{2P}{|I|^2}, \qquad Z_{VI} = \frac{V}{I} $$

with $P$ the time-averaged power crossing the port. For a lossless, homogeneously filled TEM
line the three agree and reduce to the classical line impedance. For a coaxial cross-section
of inner radius $a$ and outer radius $b$,

$$ Z_0 = \frac{\eta}{2\pi}\ln\frac{b}{a} , \qquad \eta = \eta_0\sqrt{\mu_r/\varepsilon_r} $$

which depends on the **geometry**, whereas the TEM wave impedance $Z_\mathrm{TEM} = \eta$
depends only on the **medium**. For an inhomogeneous cross-section (a quasi-TEM line such as
microstrip) the three definitions separate, and $Z_{PV}$ is used.

In the code, TEM and quasi-TEM modes are referenced to the line impedance and TE/TM modes to
their wave impedance, which is the convention CST uses. The choice is made per mode: the
higher (TE, TM) modes of a coaxial port are referenced to their own wave impedance, not to
the line impedance of its TEM mode. It is selectable through `impedance_reference`, either
`'line'` (default) or `'wave'`.

**The S-matrix cannot reveal the choice.** Let $\mathbf{A} = \mathrm{diag}(\alpha_1, \dots,
\alpha_N)$, $\alpha_i > 0$, collect the per-mode ratio between the two references, so that
changing convention maps

$$ \mathbf{Z} \mapsto \mathbf{A}^{1/2}\mathbf{Z}\,\mathbf{A}^{1/2}, \qquad
   \mathbf{Z}_\mathrm{ref} \mapsto \mathbf{A}\,\mathbf{Z}_\mathrm{ref} . $$

Since $\mathbf{A}$ and $\mathbf{Z}_\mathrm{ref}$ are both diagonal they commute, and
$\mathbf{A}\mathbf{Z}_\mathrm{ref} = \mathbf{A}^{1/2}\mathbf{Z}_\mathrm{ref}\mathbf{A}^{1/2}$.
Substituting into the conversion of [§5.3](#53-z-to-s-conversion) gives

$$ \mathbf{A}^{-1/2}\mathbf{Z}_\mathrm{ref}^{-1/2}
   \cdot \mathbf{A}^{1/2}(\mathbf{Z} - \mathbf{Z}_\mathrm{ref})\mathbf{A}^{1/2}
   \cdot \mathbf{A}^{-1/2}(\mathbf{Z} + \mathbf{Z}_\mathrm{ref})^{-1}\mathbf{A}^{-1/2}
   \cdot \mathbf{A}^{1/2}\mathbf{Z}_\mathrm{ref}^{1/2} = \mathbf{S} , $$

so every factor of $\mathbf{A}$ cancels and $\mathbf{S}$ is **invariant** under the change of
reference. Only $\mathbf{Z}$ moves. A code that normalises to the wave impedance while
reporting line-referenced $Z$ therefore agrees with CST on $S$ and sits at a constant
per-port factor on $Z$ -- a discrepancy that no S-parameter comparison can detect.

## 5.3 Z-to-S Conversion
The S-parameters are obtained from the Z-parameters using the generalised **pseudo-wave**
conversion (Marks & Williams) -- the convention used by CST and HFSS for multimode
S-parameters. Unlike Kurokawa power-waves it does not require $\mathrm{Re}(Z_0) > 0$, so it
stays valid for the purely reactive $Z_0$ of a mode below cutoff:

$$ \mathbf{S} = \mathbf{Z}_{\mathrm{ref}}^{-1/2} (\mathbf{Z} - \mathbf{Z}_{\mathrm{ref}})(\mathbf{Z} + \mathbf{Z}_{\mathrm{ref}})^{-1} \mathbf{Z}_{\mathrm{ref}}^{1/2} $$


## 5.4 The Impedance Matrix (Recovery)
The impedance matrix can be recovered from the S-matrix via:

$$ \mathbf{Z} = \mathbf{Z}_{\mathrm{ref}}^{1/2} (\mathbf{I} + \mathbf{S})(\mathbf{I} - \mathbf{S})^{-1} \mathbf{Z}_{\mathrm{ref}}^{1/2} $$

---

**Next:** [6. Model order reduction](reduction.md)
