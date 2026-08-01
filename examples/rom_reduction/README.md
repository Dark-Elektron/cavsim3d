# FOM → ROM reduction

The cavsim3d pipeline is staged: solve the **full-order model (FOM)** on a few
frequencies, build a **reduced-order model (ROM)** from those snapshots, then
sweep the ROM on a dense grid almost for free.

| file | what it shows |
|---|---|
| `fom_to_rom.py` | FOM (few points) → `fom.reduce(tol)` → dense `rom.solve()`; overlay + speedup |

Typical output (rectangular waveguide):

```
FOM: 6 samples in 1.25s   (209.1 ms/point)
ROM: 400 samples in 0.34s (0.86 ms/point)  -> ~243x cheaper per point
ROM sweep: mean |S21| = 1.0000 over 400 points
```

The ROM passes through the FOM snapshots and fills the band between them at a
fraction of the cost — the core value of the reduced-order stage. For the full
staged pipeline (FOM → ROM → concatenation, netlists, cross-project reuse) see
[`tutorials/core_workflows.py`](../../tutorials/core_workflows.py).

Run:

```bash
python examples/rom_reduction/fom_to_rom.py
```
