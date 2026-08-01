# Rectangular waveguide — S-parameters

A straight length of rectangular waveguide on its fundamental **TE10** mode: a
matched, single-mode transmission line. Above the TE10 cutoff `fc = c / (2a)`
it transmits perfectly, and the cavsim3d frequency-domain solver reproduces
this exactly.

## Files

| file | what it shows |
|---|---|
| `waveguide_sparameters.py` | `|S21| ≈ 1`, `|S11| ≈ 0` across the single-mode band |

Typical output (a = 100 mm → fc ≈ 1.5 GHz):

```
above-cutoff transmission: mean |S21| = 1.0000 (ideal 1.0), max |S11| = 0.0005
```

This is the canonical sanity check for the FOM and its analytic port modes, and
the reference point for the open-structure caveat discussed in
`../microstrip/` (closed guide → flat |S21|; open microstrip → box resonances).

Run:

```bash
python examples/rectangular_waveguide/waveguide_sparameters.py
```
