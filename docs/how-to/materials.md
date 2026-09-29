# How to assign materials

By default every region of a model is vacuum. This page shows how to give a primitive, an
imported CAD model or a part of a chain its own permittivity, permeability and losses, and
how to turn solids of a CAD model into perfect conductors.

## Fill a primitive

Call `set_materials` on the geometry before adding it to the project. `'*'` matches every
material of the geometry:

```python
from cavsim3d.geometry.primitives import RectangularWaveguide

guide = RectangularWaveguide(a=0.1, b=0.05, L=0.2, maxh=0.015)
guide.set_materials({"*": {"eps_r": 2.25, "tan_delta": 0.005}})
proj.add("guide", guide)
```

The properties are:

| Key | Meaning | Default |
|---|---|---|
| `eps_r` (or `epsilon_r`) | relative permittivity | 1 |
| `mu_r` | relative permeability | 1 |
| `tan_delta` | dielectric loss tangent | 0 |
| `sigma` | conductivity, S/m | 0 |

The solver uses $\varepsilon = \varepsilon_0 \varepsilon_r (1 - j \tan\delta) - j\sigma/\omega$.
Missing keys keep their defaults.

## Assign materials to the solids of a CAD model

Name the solids as the STEP file labels them. The importer prints the labels it found
(`STEP solids found: ...`); `geo.solid_labels` lists them again:

```python
geo = proj.import_geometry("line.step", name="line", unit="mm", auto_build=False)
print(geo.solid_labels)

geo.set_materials({
    "solid1": {"eps_r": 1},           # air
    "solid2": {"eps_r": 10},          # ceramic window
    "solid5": "PEC",                  # inner conductor
})
proj.generate_mesh(maxh=0.005)
```

A key can be the solid's name, its full label, or one part of the label: for the label
`coupler|hook_base`, the keys `hook_base`, `coupler|hook_base` and `coupler` all select it,
and `coupler` selects every solid of that component. `*` is a wildcard (`"hook_*"`).

A value of `"PEC"` removes the solid from the computed region; its surface becomes a
perfectly conducting wall. `"PEC"` is available for imported CAD models only.

## Fill one part of a chain

Assign the material on that part before adding it; the other parts stay vacuum:

```python
slab = RectangularWaveguide(a=0.1, b=0.05, L=0.05, maxh=0.02)
slab.set_materials({"*": {"eps_r": 2.0}})

proj.create_primitive("rwg", name="inlet", a=0.1, b=0.05, L=0.1, maxh=0.02)
proj.add("slab", slab)
proj.create_primitive("rwg", name="outlet", a=0.1, b=0.05, L=0.1, maxh=0.02)
```

## Check the assignment

```python
proj.draw_material_cf("eps")      # colour map of eps_r; "mu" for mu_r
print(guide.get_material("vacuum"))
```

## Change a material after solving

Call `set_materials` again and solve with the same request. `solve()` detects the change
and recomputes (see [How to control recomputation](rerun_and_resume.md)).

## Troubleshooting

- **`Material 'x': unknown properties [...]`**: a misspelt key. Use `eps_r`, `mu_r`,
  `sigma`, `tan_delta`.
- **`PEC regions are only supported for imported CAD geometry`**: `"PEC"` was given to a
  primitive. Model the conductor as part of a CAD model instead.
- **`Material key 'x' does not match any solid`**: the key is not a solid name or label of
  the file. Print `geo.solid_labels`.
- **`... all become the material 'x' in the mesh`**: two solids of different components
  have the same name, so they share one mesh material and one set of properties. Rename
  one of them in the CAD file.

**See also:** [Fill a waveguide with lossy material](../tutorials/models/materials_and_losses.ipynb)
(tutorial); [Ports and port modes](../explanation/ports.md) for materials on port faces.
