# Project Management

The `EMProject` class is the central entry point for managing simulations, geometry, and results. It orchestrates the full workflow: geometry creation, meshing, solving, reduction, and persistence.

## EMProject

::: core.em_project.EMProject
    options:
      members:
        - __init__
        - create_assembly
        - create_primitive
        - create_importer
        - import_geometry
        - generate_mesh
        - save
        - load
        - has_mesh
        - has_results
        - invalidate_mesh
        - invalidate_results
      show_root_heading: true

### Key Properties

::: core.em_project.EMProject
    options:
      members:
        - geometry
        - mesh
        - fds
        - order
        - n_port_modes
        - geo
      show_root_heading: false
      show_category_heading: false

### Persistence Paths

::: core.em_project.EMProject
    options:
      members:
        - mesh_path
        - fds_path
        - geometry_path
        - fom_path
        - foms_path
        - eigenmode_path
      show_root_heading: false
      show_category_heading: false
