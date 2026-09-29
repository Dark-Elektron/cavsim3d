# How to import a CAD file

This page shows how to bring a STEP, IGES or BREP model into a project in the right units,
with its ports on the right faces, and how to cut it into domains.

## Import a file

```python
geo = proj.import_geometry("cavity.step", name="cavity", unit="mm")
```

- `unit` is the length unit **the file was written in**: `"mm"` (the default) or `"m"`.
  The code works in metres and scales the file on import.
- `name` is the part's name in the project (default: the file name). Importing again under
  the same name replaces the part.
- `auto_build=False` imports without preparing the geometry, so you can split it or assign
  materials first; meshing builds it.

Check the size after importing:

```python
lo, hi = geo.get_bounding_box()      # metres
print(lo, hi)
```

A model a thousand times too large or too small was imported with the wrong `unit`.

## Check the ports

The importer takes ports from the file when it defines them (named entities containing
`port` or `Ports|`, as CST exports them). Otherwise it names the two flat faces at the ends
of the model's main axis `port1` and `port2`. Always check the result:

```python
proj.fds.print_port_map()        # port geometry (rectangular, circular, coaxial, ...) and role
geo.show()                       # port faces are drawn red
```

## Assign ports by hand

If the ports are on the wrong faces, list the flat faces and name the ones you want:

```python
geo = proj.import_geometry("model.step", name="model", unit="mm", auto_build=False)
geo.list_planar_faces()                         # index, centre, normal, area of each flat face
geo.assign_ports({"port1": 12, "port2": 57})    # face indices from the list
proj.generate_mesh(maxh=0.005)
```

The faces in the map become the model's only external ports: every other face that had a
port name, from the file or from the automatic naming, becomes a wall again (ports on
cutting planes stay). `assign_ports` drops an existing mesh, so mesh afterwards.

`geo.show_planar_faces()` draws the model with its flat faces highlighted. To name the end
faces along a different axis instead, use `geo.name_faces_by_position(axis="X")`.

## Put the parts along another axis

Parts of a project are chained along `proj.main_axis` (Z unless set). For a model whose
beam axis is x, set it before adding parts:

```python
proj.main_axis = "X"
proj.import_geometry("tesla9cell.step", name="cavity", unit="m", n=3)
```

## Cut a model into domains

Add cutting planes before building, then split:

```python
geo = proj.import_geometry("guide.step", name="guide", unit="m", auto_build=False)
geo.add_splitting_plane_at_z(0.1)          # metres, in the file's coordinates after scaling
geo.add_splitting_plane_at_z(0.2)
geo.split()
print(geo.domains)                         # ['subdomain1', 'subdomain2', 'subdomain3']
proj.generate_mesh(maxh=0.04)
```

`add_splitting_plane_at_x` and `add_splitting_plane_at_y` cut across the other axes. The cut
faces become internal ports, and `solve()` then solves each domain on its own.

## Mesh an imported model

```python
proj.generate_mesh(maxh=0.005)        # metres: 5 mm
```

Curved surfaces are meshed with curved elements (`curve_order=3` by default). Local
refinement and the choice of `maxh` are covered in
[How to choose the mesh size and element order](mesh_and_order.md).

## Troubleshooting

- **Ports on the wrong faces, or `Could not determine normal for port ...`**: list the
  faces and assign the ports by hand (above).
- **`maxh=... m is larger than the geometry`**: `maxh` was given in millimetres.
- **Materials of the file are not applied**: see [How to assign materials](materials.md).

**See also:** [Import a CAD model](../tutorials/models/cad_import.ipynb) and
[Cut a model into domains](../tutorials/models/splitting_cad.ipynb) (tutorials).
