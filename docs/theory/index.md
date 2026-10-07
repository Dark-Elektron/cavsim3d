# Mathematical Theory

The code solves the frequency-domain Maxwell's equations with the **Finite Element Method
(FEM)** and speeds up wideband analysis with **Model Order Reduction (MOR)**. These pages
derive the equations behind each step, from the variational form to the joined model and its
resonances. The time convention is $e^{j\omega t}$ throughout.

1. [Maxwell's equations](maxwell.md): the vector wave equation, and the complex
   permittivity of lossy materials.
2. [Variational formulation](variational.md): the weak form, the boundary conditions, and
   the system matrices $\mathbf{K}$, $\mathbf{M}$, $\mathbf{C}$ and $\mathbf{D}$.
3. [Port modes](ports.md): the 2D eigenproblems of TE, TM, TEM and quasi-TEM modes, and the
   port excitation $\mathbf{B}$.
4. [Z-parameters](z_parameters.md): modal voltages and currents, and
   $\mathbf{Z} = j\,\mathbf{B}^T\mathbf{X}$.
5. [Reference impedances and S-parameters](s_parameters.md): wave and line impedances, and
   the conversion from Z to S.
6. [Model order reduction](reduction.md): snapshots, proper orthogonal decomposition and the
   reduced system.
7. [Concatenation](concatenation.md): joining reduced models through their port modes.
8. [Resonant modes](resonances.md): the eigenvalue problems, and the loaded and external Q.
9. [Beam excitation](beam.md): the beam as a source, its own field, the scattered-field
   formulation, and the generalised scattering matrix with the beam.
10. [Model order reduction with the beam](beam_reduction.md): the wall lift in the reduced
    model, and the beam's load and outputs in affine form.

## Summary of the Solve Pipeline

The following table summarises the key mathematical objects and where they appear in the pipeline
($N_{pm}$ is the total number of port-modes):

| Object | Symbol | Size | Description |
|--------|--------|------|-------------|
| Stiffness matrix | $\mathbf{K}$ | $n \times n$ | Curl-curl bilinear form: $\int \frac{1}{\mu_0\mu_r}(\nabla \times \mathbf{N}_i) \cdot (\nabla \times \mathbf{N}_j) \,\mathrm{d}\Omega$ |
| Mass matrix | $\mathbf{M}$ | $n \times n$ | $\varepsilon$-weighted inner product: $\int \varepsilon_0\varepsilon_r \, \mathbf{N}_i \cdot \mathbf{N}_j \,\mathrm{d}\Omega$ |
| Loss matrices | $\mathbf{C}, \mathbf{D}$ | $n \times n$ | $\int \sigma\,\mathbf{N}_i\cdot\mathbf{N}_j$ and $\int \varepsilon_0\varepsilon_r\tan\delta\,\mathbf{N}_i\cdot\mathbf{N}_j$; zero if lossless |
| Port basis matrix | $\mathbf{B}$ | $n \times N_{pm}$ | Boundary mass-weighted port modes (see [Section 3.2](ports.md#32-building-the-right-hand-side-b)) |
| Solution | $\mathbf{X}$ | $n \times N_{pm}$ | Solves $(\mathbf{K} + j\omega\mathbf{C} - \omega^2(\mathbf{M} - j\mathbf{D}))\mathbf{X} = \omega\mathbf{B}$; the field coefficients are $j\mathbf{X}$ |
| Z-parameters | $\mathbf{Z}$ | $N_{pm} \times N_{pm}$ | Impedance matrix: $j\mathbf{B}^T\mathbf{X}$ |
| S-parameters | $\mathbf{S}$ | $N_{pm} \times N_{pm}$ | Scattering matrix: $\mathbf{Z}_\mathrm{ref}^{-1/2}(\mathbf{Z}-\mathbf{Z}_\mathrm{ref})(\mathbf{Z}+\mathbf{Z}_\mathrm{ref})^{-1}\mathbf{Z}_\mathrm{ref}^{1/2}$ |
| POD basis | $\mathbf{V}$ | $n \times r$ | Truncated left singular vectors of the (real) snapshot matrix |
| Reduced system | $\hat{\mathbf{A}}_d, \hat{\mathbf{B}}_d$ | $r_d \times r_d$, $r_d \times N_d$ | Per-domain Galerkin-projected, mass-normalised matrices ($r_d \ll n$) |
| Block-diagonal system | $\mathbf{A}_{\text{blk}}$ | $r_\mathrm{blk} \times r_\mathrm{blk}$ | Block-diagonal assembly of all per-domain $\hat{\mathbf{A}}_d$ ($r_\mathrm{blk} = \sum r_d$) |
| Constraint matrix | $\mathbf{G}$ | $r_\mathrm{blk} \times c$ | Kirchhoff coupling: $\mathbf{G} = \mathbf{B}_{\text{int}} \mathbf{F}$ |
| Coupled system | $\mathbf{A}_{\text{coupled}}, \mathbf{B}_{\text{coupled}}$ | $r_c \times r_c$ | Null-space projected system with internal ports eliminated |
