# How to set up port modes

Each port carries one or more modes, and every mode of every port is one row and column of
the S- and Z-matrices. This page shows how to list the ports, choose how many modes each
carries, compute modes for cross-sections without a closed form, and choose the reference
impedance of TEM ports.

## List the ports first

The port map needs only the mesh, not a solve:

```python
proj.fds.print_port_map()
```

```text
 idx  port      geometry    dims [mm]                 carries       role
   0  port1     rectangular area=5000.0 mm^2          TE/TM         external
   1  port2     rectangular area=5000.0 mm^2          TE/TM         external
```

The order of the rows is the order of the ports in the S-matrix and in list-valued
`nportmodes`. `proj.fds.port_map()` returns the same information as a list of dicts.

## Choose the number of modes

`nportmodes` is a `solve()` option. Three forms are accepted:

```python
proj.fds.solve(fmin=1, fmax=5, nsamples=41, nportmodes=3)                # 3 on every port
proj.fds.solve(fmin=1, fmax=5, nsamples=41, nportmodes=[3, 1])           # per port, map order
proj.fds.solve(fmin=1, fmax=5, nsamples=41,
               nportmodes={"port1": 3, "default": 1})                    # by name
```

A list of the wrong length, or a dict naming a port that does not exist, raises an error
before anything is solved.

Include every mode that propagates in the band at a port, plus one or two evanescent ones
near an obstacle. Modes are ordered by cutoff frequency; the solve log lists them
(`port1 mode 0: TE_10, kc=31.4159`).

The S-matrix grows with the modes: with 3 modes on each of 2 ports it is 6 × 6. Labels name
the excitation first, then the response: `'1(3)2(1)'` is the wave leaving port 2 in mode 1
when port 1 is excited in mode 3.

## Keep a subset of the port modes

To look at a model as if only some modes existed (the rest open-circuited):

```python
from cavsim3d.analysis import keep_port_modes, port_mode_labels

print(port_mode_labels(concat))                        # ['1(1)', '1(2)', '2(1)', ...]
S, Z = keep_port_modes(concat, ["1(1)", "2(1)"])       # arrays [freq, 2, 2]
```

## Compute modes numerically

Rectangular, circular and coaxial cross-sections have closed-form modes (the default,
`mode_source="analytic"`). For any other cross-section, solve a 2D eigenproblem on the
port face:

```python
proj.fds.solve(fmin=1, fmax=5, nsamples=41, mode_source="numeric")
```

`mode_source_internal` does the same for the internal ports between parts. A change of mode
source is a different request, so `solve()` recomputes.

## Choose the reference impedance of TEM ports

For TEM (coaxial) ports, S is referred to the line impedance by default, as in CST Studio
Suite. The older wave-impedance reference is still available:

```python
proj.fds.solve(fmin=1, fmax=5, nsamples=41, impedance_reference="wave")
```

TE and TM modes are always referred to their wave impedance.

## Joining parts

At a join between parts, both sides must carry the same number of modes, with matching
mode types, indices and cutoffs; otherwise joining raises an error. If a join carries fewer
modes than propagate in the band, joining warns and names the port and the count needed.

**See also:** [solve() options](../reference/solve_options.md);
[Ports and port modes](../explanation/ports.md);
[How to model a microstrip line](microstrip.md) for quasi-TEM ports.
