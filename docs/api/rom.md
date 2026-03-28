# Model Reduction

`ModelOrderReduction` implements Proper Orthogonal Decomposition (POD) for constructing compact reduced-order models from full-order frequency-domain snapshots. The reduced system is orders of magnitude faster to solve, enabling wideband sweeps with thousands of frequency points.

## ModelOrderReduction

::: rom.reduction.ModelOrderReduction
    options:
      members:
        - __init__
        - reduce
        - solve
        - solve_per_domain
        - concatenate
        - get_reduced_structure
        - get_all_structures
        - save
        - load
      show_root_heading: true

### Eigenvalue Analysis

::: rom.reduction.ModelOrderReduction
    options:
      members:
        - calculate_resonant_modes
        - get_eigenvalues
        - get_resonant_frequencies
        - get_eigenmodes
      show_root_heading: false
      show_category_heading: false

### Field Reconstruction

::: rom.reduction.ModelOrderReduction
    options:
      members:
        - reconstruct_field
        - can_reconstruct
        - get_reconstruction_info
        - plot_field
        - plot_field_at_frequency
      show_root_heading: false
      show_category_heading: false

### Diagnostics & Plotting

::: rom.reduction.ModelOrderReduction
    options:
      members:
        - plot_singular_values
        - plot_eigenfrequencies
        - print_info
        - print_eigenfrequency_comparison
        - clear_results
      show_root_heading: false
      show_category_heading: false

### Properties

::: rom.reduction.ModelOrderReduction
    options:
      members:
        - n_ports
        - ports
        - all_ports
        - reduced_dimensions
        - total_dofs
        - total_reduced_dofs
        - global_reduced_dofs
        - compression_ratio
        - singular_values
        - has_global_system
        - concatenated_system
        - A
        - B
        - W
        - A_global
        - B_global
        - W_global
      show_root_heading: false
      show_category_heading: false
