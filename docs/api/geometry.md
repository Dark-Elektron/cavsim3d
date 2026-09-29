# Geometry & Assembly

## Base Geometry

`BaseGeometry` is the abstract base class for all geometric primitives and imported CAD models. It handles meshing, port definition, boundary detection, and persistence.

::: cavsim3d.geometry.base.BaseGeometry
    options:
      members:
        - __init__
        - build
        - generate_mesh
        - set_materials
        - get_material
        - set_local_mesh_refinement
        - define_ports
        - show
        - get_extreme_faces
        - get_physical_bounds
        - get_boundary_normal
        - get_point_on_boundary
        - save_step
        - save_brep
        - print_info
        - get_history
        - save_geometry
        - load_geometry
      show_root_heading: true

### Properties

::: cavsim3d.geometry.base.BaseGeometry
    options:
      members:
        - ports
        - boundaries
        - n_ports
      show_root_heading: false
      show_category_heading: false

---

## Assembly

`Assembly` manages multi-component geometries. Components are added sequentially and auto-aligned along a main axis. The assembly tracks connections (shared interfaces) between components and supports per-domain solving.

::: cavsim3d.geometry.assembly.Assembly
    options:
      members:
        - __init__
        - add
        - replace
        - remove
        - connect
        - set_main_axis
        - set_mesh_strategy
        - resolved_mesh_strategy
        - describe_layout
        - build
        - generate_mesh
        - compute_layout
        - rotate
        - translate
        - show
        - inspect
        - get_port_info
        - get_solid_info
        - get_external_ports
        - get_interface_ports
        - get_assembly_bounds
        - get_identical_components
        - get_components_by_base_name
        - get_solver_optimization_info
        - print_port_info
        - print_info
        - summary
        - save_geometry
      show_root_heading: true

### Properties

::: cavsim3d.geometry.assembly.Assembly
    options:
      members:
        - is_assembly
        - tag
        - size
        - centroid
        - components
      show_root_heading: false
      show_category_heading: false

---

## Primitives

Waveguide and cavity primitives. All lengths are in metres; each primitive builds and
meshes itself on construction. `proj.create_primitive("rwg" | "cwg", name=..., ...)`
creates the first two as parts of a project. For axisymmetric cavities and beam-line
elements, see [Bodies of revolution](#bodies-of-revolution).

::: cavsim3d.geometry.primitives.RectangularWaveguide
    options:
      members:
        - __init__
        - cutoff_frequency_TE10
        - get_analytical_modes
      show_root_heading: true

::: cavsim3d.geometry.primitives.CircularWaveguide
    options:
      members:
        - __init__
        - cutoff_frequency_TE11
        - cutoff_frequency_TM01
      show_root_heading: true

::: cavsim3d.geometry.primitives.Box
    options:
      members:
        - __init__
      show_root_heading: true

::: cavsim3d.geometry.microstrip.MicrostripLine
    options:
      members:
        - __init__
        - qtem_voltage_path
      show_root_heading: true

---

## Bodies of revolution

Axisymmetric cavities and beam-line elements, each built from its meridian contour in the
(z, r) plane and revolved 360 degrees about the Z axis. The classes carry the names and
constructor arguments of the cavsim2d models, so a structure defined for cavsim2d builds
here unchanged.

- **Units.** Unlike the primitives above, dimensions default to **millimetres**, as in
  cavsim2d (the RF gun: metres, angles in radians). `unit="m"` (or `"cm"`, `"um"`)
  selects another unit. `maxh` is always in metres.
- **Config dictionaries.** Every constructor also takes its arguments as one dict,
  `config={...}`; arguments given directly take precedence over the dict.
- **Meshing.** A model is built without a mesh. `generate_mesh(maxh=...)` makes it; if a
  solve starts first, the model is meshed with the `maxh` given to its constructor.
  `generate_mesh` keeps earlier settings for arguments left out.

Every beam aperture is a port: `port1` at the low-z end, `port2` at the high-z end. The
rest of the surface is the PEC wall `'default'`.

`proj.create_primitive(kind, name=..., ...)` creates one as a part of a project. `kind` is
the class name or its snake-case form, ignoring case and underscores:

| `kind` | Class |
|---|---|
| `"elliptical_cavity"` | `EllipticalCavity` |
| `"elliptical_cavity_flattop"`, `"flattop"` | `EllipticalCavityFlatTop` |
| `"rfgun"` | `RFGun` |
| `"pillbox"` | `Pillbox` |
| `"spline_cavity"` | `SplineCavity` |
| `"beampipe"` | `Beampipe` |
| `"bla"` | `BLA` |
| `"bellows"` | `Bellows` |
| `"taper"` | `Taper` |

```python
tesla = [42, 42, 12, 19, 35, 57.7, 103.353]     # A, B, a, b, Ri, L, Req in mm
proj.create_primitive("elliptical_cavity", name="cavity", n_cells=9,
                      mid_cell=tesla, beampipe="both", maxh=0.02)

cfg = {"n_cells": 9, "mid_cell": tesla, "beampipe": "both", "maxh": 0.02}
proj.create_primitive("elliptical_cavity", name="cavity", config=cfg)   # the same part
```

::: cavsim3d.geometry.axisymmetric.EllipticalCavity
    options:
      members:
        - __init__
        - profile
        - half_cells
        - wall_angles
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.EllipticalCavityFlatTop
    options:
      members:
        - __init__
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.RFGun
    options:
      members:
        - __init__
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.Pillbox
    options:
      members:
        - __init__
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.SplineCavity
    options:
      members:
        - __init__
        - control_polygons
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.Beampipe
    options:
      members:
        - __init__
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.BLA
    options:
      members:
        - __init__
        - get_material
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.Bellows
    options:
      members:
        - __init__
        - check_feasible
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.Taper
    options:
      members:
        - __init__
        - half_angle
        - check_feasible
      show_root_heading: true

### Defining a new body of revolution

A subclass of `AxisymmetricGeometry` needs only `profile()`, returning the meridian as a
`Profile`; `revolve` turns it into the named solid.

::: cavsim3d.geometry.axisymmetric.AxisymmetricGeometry
    options:
      members:
        - profile
        - generate_mesh
        - plot_profile
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.Profile
    options:
      members:
        - start
        - line_to
        - arc_to
        - circle_arc_to
        - ellipse_arc_to
        - spline_to
        - close
        - add_region
        - chained
        - apertures
        - contour_points
      show_root_heading: true

::: cavsim3d.geometry.axisymmetric.revolve
    options:
      show_root_heading: true

---

## CAD import

`OCCImporter` reads STEP, IGES and BREP files. `proj.import_geometry(path, name=..., unit=...)`
creates one as a part of a project.

::: cavsim3d.geometry.importers.OCCImporter
    options:
      members:
        - __init__
        - build
        - get_bounding_box
        - list_planar_faces
        - show_planar_faces
        - assign_ports
        - name_faces_by_position
        - name_solids
        - set_materials
        - solid_labels
        - add_splitting_plane_at_x
        - add_splitting_plane_at_y
        - add_splitting_plane_at_z
        - add_splitting_plane
        - split
        - domains
        - internal_ports
        - show
        - print_info
      show_root_heading: true

---

## Drawing fields

::: cavsim3d.utils.visualization.draw
    options:
      show_root_heading: true
