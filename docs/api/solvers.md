# Solvers

## Frequency Domain Solver

`FrequencyDomainSolver` is the main solver class. It assembles the FEM system matrices, solves the frequency-domain Maxwell equations, and extracts S/Z-parameters. For multi-domain (compound) structures it supports both per-domain and globally-coupled solve strategies.

::: solvers.frequency_domain.FrequencyDomainSolver
    options:
      members:
        - __init__
        - assemble_matrices
        - solve
        - compare_methods
        - reset
        - full_reset
        - save
        - load_from_path
      show_root_heading: true

### Results Access

::: solvers.frequency_domain.FrequencyDomainSolver
    options:
      members:
        - fom
        - foms
        - get_domain_results
        - get_all_domain_results
        - get_cascaded_results
        - get_coupled_results
        - get_s_at_frequency
        - get_z_at_frequency
        - get_frequency_index
        - get_domain_z_matrix
        - get_domain_s_matrix
        - get_rom_data
        - export_touchstone
      show_root_heading: false
      show_category_heading: false

### Eigenvalue Analysis

::: solvers.frequency_domain.FrequencyDomainSolver
    options:
      members:
        - calculate_resonant_modes
        - get_eigenvalues
        - get_resonant_frequencies
      show_root_heading: false
      show_category_heading: false

### Plotting

::: solvers.frequency_domain.FrequencyDomainSolver
    options:
      members:
        - plot_s_parameters
        - plot_z_parameters
        - plot_domain_s_parameters
        - plot_s_parameters_comparison
        - plot_method_comparison
        - plot_port_mode
        - plot_field
      show_root_heading: false
      show_category_heading: false

### Port & Domain Info

::: solvers.frequency_domain.FrequencyDomainSolver
    options:
      members:
        - ports
        - all_ports
        - external_ports
        - internal_ports
        - n_ports
        - fes
        - mesh
        - print_info
        - print_port_info
        - print_domain_info
      show_root_heading: false
      show_category_heading: false

---

## Concatenated System

`ConcatenatedSystem` couples multiple domains (FOM or ROM) at shared internal ports via Kirchhoff constraints. It eliminates internal port DOFs to produce a global system with only external ports.

::: solvers.concatenation.ConcatenatedSystem
    options:
      members:
        - __init__
        - define_connections
        - couple
        - solve
        - reduce
        - save
        - load
      show_root_heading: true

### Results & Reconstruction

::: solvers.concatenation.ConcatenatedSystem
    options:
      members:
        - has_solution
        - has_snapshots
        - can_reconstruct
        - get_reconstruction_info
        - get_coupled_dimensions
        - verify_kirchhoff
        - reconstruct_eigenmode
        - plot_field
        - plot_field_at_frequency
      show_root_heading: false
      show_category_heading: false

### Eigenvalue Analysis

::: solvers.concatenation.ConcatenatedSystem
    options:
      members:
        - calculate_resonant_modes
        - get_eigenvalues
        - get_eigenmodes
        - get_eigenvalues_filtered
        - get_resonant_frequencies
      show_root_heading: false
      show_category_heading: false

### Properties

::: solvers.concatenation.ConcatenatedSystem
    options:
      members:
        - n_ports
        - n_external_ports
        - n_modes_per_port
        - ports
        - rom
        - coupled_dofs
        - print_info
      show_root_heading: false
      show_category_heading: false

---

## FOM Results

### FOMResult

`FOMResult` wraps the solved full-order model for a single domain or global system. It provides access to Z/S-parameter dictionaries and serves as the entry point for model reduction.

::: solvers.results.FOMResult
    options:
      members:
        - solve
        - reduce
        - concatenate
        - clear_rom
        - get_eigenvalues
        - get_resonant_frequencies
        - get_eigenmodes
        - save
        - load
      show_root_heading: true

#### Properties

::: solvers.results.FOMResult
    options:
      members:
        - Z_dict
        - S_dict
        - rom
        - K
        - M
        - B
      show_root_heading: false
      show_category_heading: false

### FOMCollection

`FOMCollection` holds per-domain `FOMResult` objects for multi-domain structures. Supports batch reduction and concatenation.

::: solvers.results.FOMCollection
    options:
      members:
        - solve
        - reduce
        - concatenate
        - clear_roms
        - plot_s
        - plot_z
        - plot_eigenvalues
        - plot_residual
        - save
        - load
      show_root_heading: true

#### Properties

::: solvers.results.FOMCollection
    options:
      members:
        - frequencies
        - Z_dict
        - S_dict
        - roms
        - concat
        - K
        - M
        - B
      show_root_heading: false
      show_category_heading: false

---

## Port Eigenmode Solver

`PortEigenmodeSolver` computes the transverse eigenmodes at each waveguide port. Supports analytic modes (rectangular, circular) and numeric modes (arbitrary cross-section).

::: solvers.ports.PortEigenmodeSolver
    options:
      members:
        - solve
        - get_port_wave_impedance
        - get_port_wave_impedance_matrix
        - get_propagation_constant
        - get_cutoff_frequency
        - get_cutoff_frequencies_dict
        - get_polarization_info
        - get_mode_info
        - get_geometry_info
        - get_num_modes
        - get_total_num_modes
        - get_mode_name
        - get_mode_degeneracy
        - is_mode_degenerate
        - print_mode_summary
        - print_info
        - save_to_file
        - load_from_file
      show_root_heading: true
