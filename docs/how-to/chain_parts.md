# How to chain and arrange parts

This page shows how to build a chain of parts, change it, and control how it is laid out
and meshed.

## Add parts in order

Every call adds a part at the end of the chain:

```python
proj.create_primitive("rwg", name="inlet", a=0.1, b=0.05, L=0.1, maxh=0.02)
proj.import_geometry("cavity.step", name="cavity", unit="mm")
proj.import_project("../earlier/coupler", name="coupler")      # a solved project
proj.add("taper", my_geometry)                                  # any geometry object
print(list(proj.parts))
```

## Repeat a part

```python
proj.import_geometry("cell.step", name="cell", unit="mm", n=9)    # nine copies in a row
```

A repeated part is meshed, solved and reduced once.

## Insert a part elsewhere

```python
proj.create_primitive("rwg", name="spacer", a=0.1, b=0.05, L=0.02, maxh=0.01, after="inlet")
proj.create_primitive("rwg", name="lead", a=0.1, b=0.05, L=0.05, maxh=0.02, before="inlet")
```

## Replace a part

Add a part under an existing name; it takes that part's place in the chain:

```python
proj.create_primitive("rwg", name="inlet", a=0.1, b=0.05, L=0.12, maxh=0.02)   # longer inlet
```

If a mesh or results exist, the code asks before discarding them; pass `force=True` to
replace without asking (in a script).

## Remove a part or start again

```python
proj.geometry.remove("spacer")                 # the chain is an Assembly
proj.create_assembly(force=True)               # empty chain; discards mesh and results
```

Changes made directly on `proj.geometry` (removing a part, the mesh strategy below) are
saved with the next `proj.generate_mesh()` or `solve()`.

## Chain along another axis

```python
proj.main_axis = "X"          # before or after adding parts; Z unless set
```

## Turn a part end for end

```python
proj.import_geometry("taper.step", name="taper", unit="mm", flip=True)
```

`flip=True` works for glued parts only. A coupled (repeated or imported) part must be
solved in the orientation it is used.

## Check the layout before solving

```python
print(proj.geometry.describe_layout())
proj.generate_mesh(maxh=0.01)                 # prints the same layout
proj.fds.print_port_map()                     # external and internal ports
```

## Choose glued or coupled meshing

Plain parts used once are glued into one mesh; repeated or imported parts are coupled
through their port modes. To override:

```python
proj.geometry.set_mesh_strategy("coupled")    # or "glued"
```

**See also:** [Parts, joins and netlists](../explanation/parts_and_joins.md);
[Join parts into one model](../tutorials/multi_part/combine_parts.ipynb) and
[Repeat a section](../tutorials/multi_part/repeated_sections.ipynb) (tutorials).
