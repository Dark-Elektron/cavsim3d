# Ports and port modes

Ports are where the frequency-domain model meets the outside world: the flat faces through
which waves enter and leave. How they are described decides what the S- and Z-parameters
mean. This page explains port modes, how the parameters are normalised and referenced, and
why the port faces behave as magnetic walls when resonances are computed.

## Ports carry modes, not fields

A port is a flat cross-section of a guide: the end face of a waveguide, the annulus of a
coaxial line, the substrate and air of a microstrip. Far enough from any obstacle, the
field on such a face is a sum of the guide's **modes**: fixed field patterns, each with its
own cutoff frequency, above which it propagates. The solver describes the field on a port
face only through a chosen number of these modes (`nportmodes`), and every mode of every
port becomes one row and column of the S- and Z-matrices.

The modes are computed before the sweep. For rectangular, circular and coaxial
cross-sections they are known in closed form; for any other cross-section they come from a
2D eigenvalue problem on the face (`mode_source="numeric"`). A cross-section filled with
several materials, such as a microstrip's substrate and air, has no pure TEM mode; its
quasi-TEM modes come from a mixed 2D problem that also gives their propagation constant
directly.

Modes are ordered by their cutoff frequency (quasi-TEM modes by their propagation
constant, the fundamental first). Degenerate modes, such as the two polarisations of the
TE11 mode of a circular guide (`TE_11 (cos)` and `TE_11 (sin)`), are separate modes with
separate columns.

## How many modes a port needs

A mode that is not carried does not exist for the model: at that port it sees a magnetic
wall and is reflected. At an **external** port this means the model is only valid while no
uncarried mode propagates, or while nothing in the structure converts power into such a
mode. At an **internal** port, where two parts are joined, it means the join can only pass
the carried modes: an obstacle close to a join generally needs several modes there, often
including evanescent ones. The code warns when a join carries fewer modes than propagate in
the band.

## Normalisation and reference impedance

Each mode's transverse electric field is normalised to $\int |E_t|^2 \, dS = 1$ over the
port face. The Z-matrix relates the modal voltages and currents,
$V_n = \sum_m Z_{nm} I_m$, with the time convention $e^{+j\omega t}$.

S-parameters need a reference impedance for each mode:

- **TE and TM modes** are referred to their own wave impedance. It depends on frequency and
  is reactive below cutoff. Two consequences: a uniform guide is matched ($S_{11} = 0$) at
  every frequency, even below cutoff; and below cutoff $|S|$ is no longer bounded by 1, so
  a trapped resonance can show values above 0 dB.
- **TEM modes** are referred to the line impedance by default, as in CST Studio Suite;
  `impedance_reference="wave"` selects the wave impedance. The line impedance belongs to the
  TEM mode only: the higher (TE, TM) modes of a coaxial port are referred to their own wave
  impedance, like any TE or TM mode.
- **Quasi-TEM modes** are referred to their power-voltage line impedance.

When results are exported to a Touchstone file, which allows a single reference only, the
code writes S as solved and lists the true references in the header, or renormalises every
port to a given real impedance.

## Port faces in the resonance calculation

The system matrices contain no condition at the port faces: in the frequency sweep, the port
modes supply it. In the eigenvalue problem $\mathbf{K} x = \omega^2 \mathbf{M} x$, used for
resonant frequencies, the port faces are therefore *natural* boundaries, which act as
magnetic walls (PMC). The resonances of a waveguide section are those of the section closed
by metal walls on its sides and magnetic walls at its ports, and a closed-form check must
use the same boundary (`RWGAnalytical.all_eigenfrequencies` does, by default).

This is also why these resonances appear as peaks of the Z-parameters: an open-circuited
port is exactly a magnetic wall.

## Loaded resonances

A resonance of the closed problem has no external Q: no power leaves through a magnetic
wall. With every port mode terminated in its reference impedance, the load the S-parameters
assume, the resonances become complex:

$$
\left(\mathbf{A} + j\omega\,\mathbf{B}\,\mathbf{Y}_0\,\mathbf{B}^\mathsf{T} - \omega^2\right) x = 0,
\qquad \mathbf{Y}_0 = \operatorname{diag}(1/Z_0),
$$

with $Q_L = \operatorname{Re}\omega / (2\operatorname{Im}\omega)$. The external Q of a port
splits that damping by the power the port takes from the loaded mode. `get_external_q()`
solves this problem on a reduced model; its resonance frequency and $Q_L$ are those of the
peak and 3-dB width of the S-parameters.

The residues of the closed problem at a resonance give its external Q only when nothing else
couples to the port. A feed line between the coupler and the port face, or a strongly
coupled neighbouring mode, adds reactance at the port and can change the external Q by an
order of magnitude. The loaded problem includes it.

No absorbing layer is needed for this: a port terminated in its reference impedance absorbs
every mode it carries without reflection, however the field arrives. A mode the port does
not carry sees the magnetic wall and is reflected, so a port must carry every mode that
propagates at the resonance ([How many modes a port needs](#how-many-modes-a-port-needs)).
The model has no absorbing boundary elsewhere: a structure that radiates into open space
through a face that is not a port reflects that wave instead.

The unloaded Q, from the losses in the walls and the materials, is a separate calculation on
the closed problem's mode; `get_figures_of_merit()` reports it together with R/Q, the
geometry factor and the peak fields.

## Ports at joins

When two parts are joined, the modes of the shared port must correspond one to one on both
sides: same number of modes, and for each mode the same type, indices, cutoff and
polarisation. Matching cross-sections give matching modes; the check turns a mismatch
(a different cross-section, a rotated part) into an error instead of a wrong result. The
mode patterns are built in a frame that does not depend on which way a port face points,
so the two faces of a join see the same mode with the same sign.

## Related

- [How to set up port modes](../how-to/port_modes.md)
- [Parts, joins and netlists](parts_and_joins.md)
- [Find resonances and look at fields](../tutorials/basics/resonances_and_fields.ipynb)
