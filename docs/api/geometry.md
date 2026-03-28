# Geometry & Assembly

## Base Geometry

`BaseGeometry` is the abstract base class for all geometric primitives and imported CAD models. It handles meshing, port definition, boundary detection, and persistence.

::: geometry.base.BaseGeometry
    options:
      members:
        - __init__
        - build
        - generate_mesh
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

::: geometry.base.BaseGeometry
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

::: geometry.assembly.Assembly
    options:
      members:
        - __init__
        - add
        - remove
        - connect
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

::: geometry.assembly.Assembly
    options:
      members:
        - is_assembly
        - tag
        - size
        - centroid
        - components
      show_root_heading: false
      show_category_heading: false
