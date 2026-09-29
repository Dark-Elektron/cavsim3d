# How to model a microstrip line

A microstrip port cross-section holds two materials (substrate and air), so it has no pure
TEM mode: its fundamental mode is quasi-TEM. This page shows how to set up such ports, and
how to read the mode's effective permittivity and line impedance.

## Use the microstrip primitive

`MicrostripLine` builds a PEC strip over a PEC ground plane on a dielectric substrate, in an
air box, with split port faces ready for quasi-TEM ports. All lengths in metres:

```python
from cavsim3d.core.em_project import EMProject
from cavsim3d.geometry import MicrostripLine

proj = EMProject(name="microstrip", base_dir="./simulations", overwrite=True)
proj.geometry = MicrostripLine(L=40e-3, W=20e-3, w=3.1e-3, h=1.6e-3, t=0.5e-3,
                               eps_r=4.3, maxh=1.2e-3)
```

The defaults are those values: a 50 Ω line on FR-4.

## Solve with quasi-TEM ports

```python
proj.fds.solve(fmin=1.0, fmax=6.0, nsamples=11, nportmodes=1,
               qtem_ports=["port1", "port2"], solver_type="direct")
```

Each end face is split by material (`port1_substrate`, `port1_air`); faces sharing a
`portN` prefix are grouped into one logical port, `port1`. A port whose cross-section has
more than one material is solved as quasi-TEM even without `qtem_ports`. The conductor
outlines on the port plane, which the port mode must vanish on, are declared by the
geometry; for your own geometry pass them as
`qtem_conductor_bbnd="microstrip_edges|ground_edges"`.

The quasi-TEM modes are ordered by their propagation constant, the fundamental first, as in
CST Studio Suite, so `nportmodes=1` selects the fundamental.

## Read the effective permittivity and the line impedance

```python
ps = proj.fds.port_solver
eps_eff = ps.port_eps_eff["port1"][0].real           # mode 0 of port1
z_pv = ps.port_line_impedance["port1"][0].real       # power-voltage impedance, ohm
print(f"eps_eff = {eps_eff:.2f}, Z = {z_pv:.1f} ohm")
```

S-parameters of quasi-TEM ports are referred to this power-voltage line impedance.

## Know the limit: open structures

The model has no absorbing boundary. A microstrip line radiates into its air box, and the
box reflects that wave back: the through-line S-parameters show periodic notches that a
measured or CST result does not have. The port mode itself (effective permittivity,
propagation constant, line impedance) is not affected. For S-parameters of open structures,
enclose the line in a closed, shielded box that matches the real package, or compare only
port quantities.

**See also:** [How to set up port modes](port_modes.md);
[Ports and port modes](../explanation/ports.md).
