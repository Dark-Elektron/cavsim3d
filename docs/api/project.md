# Project Management

The `EMProject` class is the central entry point for managing simulations, geometry, and results. It orchestrates the full workflow: geometry creation, meshing, solving, reduction, and persistence.

## EMProject

::: cavsim3d.core.em_project.EMProject
    options:
      members:
        - __init__
        - import_geometry
        - create_primitive
        - import_project
        - add
        - parts
        - main_axis
        - localize
        - add_beam
        - add_beam_path
        - add_transverse_beams
        - remove_beam
        - remove_beam_path
        - beams
        - beam_paths
        - create_assembly
        - generate_mesh
        - draw_material_cf
        - timing_summary
        - save
        - load
        - has_mesh
        - has_results
        - invalidate_mesh
        - invalidate_results
      show_root_heading: true

### Key Properties

::: cavsim3d.core.em_project.EMProject
    options:
      members:
        - geometry
        - mesh
        - fds
        - order
        - geo
      show_root_heading: false
      show_category_heading: false

### Persistence Paths

::: cavsim3d.core.em_project.EMProject
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
