simulation_2024-01-15/
│
├── config.json
│
├── fom/                                        # === SINGLE SOLID PATH ===
│   ├── metadata.json
│   ├── matrices.h5                             # K, M, B
│   ├── snapshots.h5                            # snapshots, frequencies
│   ├── eigen.h5                                # eigenvalues, eigenvectors
│   │
│   ├── rom_tol1e-3/
│   │   ├── metadata.json
│   │   ├── matrices.h5                         # A_r, B_r, W
│   │   ├── snapshots.h5
│   │   ├── eigen.h5
│   │   │
│   │   └── rom_tol1e-4/
│   │       ├── metadata.json
│   │       ├── matrices.h5
│   │       ├── snapshots.h5
│   │       ├── eigen.h5
│   │       │
│   │       └── rom_tol1e-5/
│   │           └── ...
│   │
│   └── rom_tol1e-2/
│       ├── metadata.json
│       ├── matrices.h5
│       ├── snapshots.h5
│       └── eigen.h5
│
└── foms/                                       # === MULTI-SOLID PATH ===
    ├── metadata.json
    ├── matrices.h5                             # solid_0/{K,M,B}, solid_1/{K,M,B}, ...
    ├── snapshots.h5                            # solid_0/snapshots, solid_1/snapshots, frequencies
    ├── eigen.h5                                # solid_0/{eigenvalues,eigenvectors}, ...
    │
    ├── roms_tol1e-3/                           # Path A: reduce first, then concatenate
    │   ├── metadata.json
    │   ├── matrices.h5                         # solid_0/{A_r,B_r,W}, solid_1/{A_r,B_r,W}, ...
    │   ├── snapshots.h5
    │   ├── eigen.h5
    │   │
    │   └── concat/
    │       ├── metadata.json
    │       ├── matrices.h5                     # A_coupled, B_coupled, W_coupled, solid_sizes
    │       ├── snapshots.h5
    │       ├── eigen.h5
    │       │
    │       ├── rom_tol1e-4/
    │       │   ├── metadata.json
    │       │   ├── matrices.h5                 # A_r, B_r, W
    │       │   ├── snapshots.h5
    │       │   ├── eigen.h5
    │       │   │
    │       │   └── rom_tol1e-5/
    │       │       ├── metadata.json
    │       │       ├── matrices.h5
    │       │       ├── snapshots.h5
    │       │       └── eigen.h5
    │       │
    │       └── rom_tol1e-3/
    │           ├── metadata.json
    │           ├── matrices.h5
    │           ├── snapshots.h5
    │           └── eigen.h5
    │
    ├── roms_tol1e-4/                           # Path B: different per-solid tolerance
    │   ├── metadata.json
    │   ├── matrices.h5
    │   ├── snapshots.h5
    │   ├── eigen.h5
    │   │
    │   └── concat/
    │       ├── metadata.json
    │       ├── matrices.h5
    │       ├── snapshots.h5
    │       ├── eigen.h5
    │       │
    │       └── rom_tol1e-4/
    │           └── ...
    │
    └── concat/                                 # Path C: concatenate first (W=I), then reduce
        ├── metadata.json
        ├── matrices.h5                         # A_coupled, B_coupled, W_coupled (W=I), solid_sizes
        ├── snapshots.h5
        ├── eigen.h5
        │
        ├── rom_tol1e-3/
        │   ├── metadata.json
        │   ├── matrices.h5
        │   ├── snapshots.h5
        │   ├── eigen.h5
        │   │
        │   └── rom_tol1e-4/
        │       ├── metadata.json
        │       ├── matrices.h5
        │       ├── snapshots.h5
        │       └── eigen.h5
        │
        └── rom_tol1e-2/
            ├── metadata.json
            ├── matrices.h5
            ├── snapshots.h5
            └── eigen.h5

{
  "fom": {
    "matrices.h5": {
      "K": { "shape": "[n_dof, n_dof]", "type": "sparse_csr" },
      "M": { "shape": "[n_dof, n_dof]", "type": "sparse_csr" },
      "B": { "shape": "[n_dof, n_inputs]", "type": "dense" }
    },
    "snapshots.h5": {
      "snapshots": { "shape": "[n_dof, n_freq]", "type": "dense" },
      "frequencies": { "shape": "[n_freq]", "type": "dense" }
    },
    "eigen.h5": {
      "eigenvalues": { "shape": "[n_eig]", "type": "dense" },
      "eigenvectors": { "shape": "[n_dof, n_eig]", "type": "dense" }
    }
  },

  "foms": {
    "matrices.h5": {
      "solid_0/K": { "shape": "[n_dof_0, n_dof_0]", "type": "sparse_csr" },
      "solid_0/M": { "shape": "[n_dof_0, n_dof_0]", "type": "sparse_csr" },
      "solid_0/B": { "shape": "[n_dof_0, n_inputs_0]", "type": "dense" },
      "solid_1/K": { "shape": "[n_dof_1, n_dof_1]", "type": "sparse_csr" },
      "solid_1/M": { "shape": "[n_dof_1, n_dof_1]", "type": "sparse_csr" },
      "solid_1/B": { "shape": "[n_dof_1, n_inputs_1]", "type": "dense" },
      "...": "repeat for each solid"
    },
    "snapshots.h5": {
      "solid_0/snapshots": { "shape": "[n_dof_0, n_freq]", "type": "dense" },
      "solid_1/snapshots": { "shape": "[n_dof_1, n_freq]", "type": "dense" },
      "...": "repeat for each solid",
      "frequencies": { "shape": "[n_freq]", "type": "dense" }
    },
    "eigen.h5": {
      "solid_0/eigenvalues": { "shape": "[n_eig_0]", "type": "dense" },
      "solid_0/eigenvectors": { "shape": "[n_dof_0, n_eig_0]", "type": "dense" },
      "solid_1/eigenvalues": { "shape": "[n_eig_1]", "type": "dense" },
      "solid_1/eigenvectors": { "shape": "[n_dof_1, n_eig_1]", "type": "dense" },
      "...": "repeat for each solid"
    }
  },

  "rom": {
    "matrices.h5": {
      "A_r": { "shape": "[n_r, n_r]", "type": "dense" },
      "B_r": { "shape": "[n_r, n_inputs]", "type": "dense" },
      "W": { "shape": "[n_full, n_r]", "type": "dense" }
    },
    "snapshots.h5": {
      "snapshots": { "shape": "[n_r, n_freq]", "type": "dense" },
      "frequencies": { "shape": "[n_freq]", "type": "dense" }
    },
    "eigen.h5": {
      "eigenvalues": { "shape": "[n_eig]", "type": "dense" },
      "eigenvectors": { "shape": "[n_r, n_eig]", "type": "dense" }
    }
  },

  "roms": {
    "matrices.h5": {
      "solid_0/A_r": { "shape": "[n_r_0, n_r_0]", "type": "dense" },
      "solid_0/B_r": { "shape": "[n_r_0, n_inputs_0]", "type": "dense" },
      "solid_0/W": { "shape": "[n_full_0, n_r_0]", "type": "dense" },
      "solid_1/A_r": { "shape": "[n_r_1, n_r_1]", "type": "dense" },
      "solid_1/B_r": { "shape": "[n_r_1, n_inputs_1]", "type": "dense" },
      "solid_1/W": { "shape": "[n_full_1, n_r_1]", "type": "dense" },
      "...": "repeat for each solid"
    },
    "snapshots.h5": {
      "solid_0/snapshots": { "shape": "[n_r_0, n_freq]", "type": "dense" },
      "solid_1/snapshots": { "shape": "[n_r_1, n_freq]", "type": "dense" },
      "...": "repeat for each solid",
      "frequencies": { "shape": "[n_freq]", "type": "dense" }
    },
    "eigen.h5": {
      "solid_0/eigenvalues": { "shape": "[n_eig_0]", "type": "dense" },
      "solid_0/eigenvectors": { "shape": "[n_r_0, n_eig_0]", "type": "dense" },
      "solid_1/eigenvalues": { "shape": "[n_eig_1]", "type": "dense" },
      "solid_1/eigenvectors": { "shape": "[n_r_1, n_eig_1]", "type": "dense" },
      "...": "repeat for each solid"
    }
  },

  "concat": {
    "matrices.h5": {
      "A_coupled": { "shape": "[n_total, n_total]", "type": "dense" },
      "B_coupled": { "shape": "[n_total, n_inputs]", "type": "dense" },
      "W_coupled": { "shape": "[n_full_total, n_total]", "type": "dense" },
      "solid_sizes": { "shape": "[n_solids]", "type": "dense" }
    },
    "snapshots.h5": {
      "snapshots": { "shape": "[n_total, n_freq]", "type": "dense" },
      "frequencies": { "shape": "[n_freq]", "type": "dense" }
    },
    "eigen.h5": {
      "eigenvalues": { "shape": "[n_eig]", "type": "dense" },
      "eigenvectors": { "shape": "[n_total, n_eig]", "type": "dense" }
    }
  },

  "sparse_csr_format": {
    "description": "For sparse matrices, store CSR components in a group",
    "example": {
      "K/data": { "shape": "[nnz]", "type": "dense", "description": "non-zero values" },
      "K/indices": { "shape": "[nnz]", "type": "dense", "description": "column indices" },
      "K/indptr": { "shape": "[n_rows + 1]", "type": "dense", "description": "row pointers" },
      "K/shape": { "shape": "[2]", "type": "dense", "description": "(n_rows, n_cols)" }
    }
  }
}