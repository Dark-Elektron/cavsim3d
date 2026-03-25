# Architecture

**cavsim3d** provides four distinct analysis pathways for electromagnetic cavity simulation. The choice depends on the complexity of the geometry and the desired balance between accuracy and computational efficiency.

## Analysis Pathways Overview

The figure below shows the software architecture with the available analysis options for single-segment and multi-segment assemblies.

```mermaid
graph TB
    subgraph col1["Pathway 1"]
        A1["Single Solid<br/>Model"]:::input --> B1["Frequency Domain<br/>Solver"]:::process
        B1 --> C1["Reduced Order<br/>Model"]:::output
        C1 -.-> C1
    end

    subgraph col2["Pathway 2"]
        A2["Multi Solid<br/>Model"]:::inputGreen --> B2["Frequency Domain<br/>Solver for<br/>Entire Domain"]:::process
        B2 --> C2["Reduced Order<br/>Model"]:::output
        C2 -.-> C2
    end

    subgraph col3["Pathway 3"]
        A3["Multi Solid<br/>Model"]:::inputGreen --> B3["Frequency Domain<br/>Solver for<br/>Individual Solids"]:::processGreen
        B3 --> D3["Concat. of FOM for<br/>Individual Solids"]:::process
        D3 --> C3["Reduced<br/>Order Model"]:::output
        C3 -.-> C3
    end

    subgraph col4["Pathway 4"]
        A4["Multi Solid<br/>Model"]:::inputGreen --> B4["Reduced Order Model<br/>for Individual<br/>System Matrices"]:::processGreen
        B4 --> D4["Concatenation"]:::processGreen
        D4 --> C4["Reduced<br/>Order Model"]:::output
        C4 -.-> C4
    end

    classDef input fill:#ffe0b2,stroke:#e65100,stroke-width:2px,color:#000
    classDef inputGreen fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#000
    classDef process fill:#ffe0b2,stroke:#e65100,stroke-width:1px,color:#000
    classDef processGreen fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px,color:#000
    classDef output fill:#ffe0b2,stroke:#e65100,stroke-width:1px,color:#000
```

---

## Pathway Details

### Pathway 1: Single Solid Model

The simplest pathway. A single geometry is meshed and solved globally. Best for small, simple cavities or waveguides.

| Step | Code | Description |
|------|------|-------------|
| Geometry | `RectangularWaveguide(a, L)` | Create a primitive or import a single CAD file |
| Mesh | `geometry.generate_mesh(maxh)` | Generate the finite element mesh |
| Solve FOM | `fds.solve(fmin, fmax, n)` | Full-order frequency sweep (few sample points) |
| Reduce | `fds.fom.reduce(tol)` | POD-based model order reduction |
| Solve ROM | `rom.solve(fmin, fmax, n)` | Wideband sweep on the reduced model (fast) |

**When to use:** Single cavities, simple waveguides, quick studies.

---

### Pathway 2: Multi-Solid, Global Assembly

Multiple parts are assembled and fused into a single mesh. The solver treats the entire assembly as one domain.

| Step | Code | Description |
|------|------|-------------|
| Assembly | `proj.create_assembly()` | Create an assembly container |
| Add Parts | `assembly.add("name", part)` | Add parts; they are auto-aligned |
| Build | `assembly.build()` | Fuse geometry into a single solid |
| Solve FOM | `fds.solve(fmin, fmax, n)` | Global frequency sweep |
| Reduce | `fds.fom.reduce(tol)` | POD reduction on global matrices |
| Solve ROM | `rom.solve(fmin, fmax, n)` | Wideband sweep |

**When to use:** Small multi-component systems where global coupling is important and the mesh is manageable.

---

### Pathway 3: Per-Domain FOM Concatenation

Each solid is solved independently, producing per-domain Full Order Models (FOMs). These are then concatenated via Kirchhoff coupling (S-matrix cascade) to produce the global response.

| Step | Code | Description |
|------|------|-------------|
| Geometry | Multi-solid or split CAD | Each solid gets its own mesh and domain |
| Solve Per-Domain | `fds.solve(fmin, fmax, n)` | Solves each domain independently |
| Concatenate FOMs | `fds.foms.concatenate()` | Cascade S-matrices at shared interfaces |
| Solve Concat | `cs.solve(fmin, fmax, n)` | Solve the concatenated system |
| Reduce | `cs.reduce(tol)` | POD on the concatenated snapshots |

**When to use:** Large assemblies where global meshing is impractical. Allows reuse of unchanged domain FOMs.

---

### Pathway 4: Per-Domain ROM Concatenation

The most efficient pathway. Each domain is **first reduced** independently, then the ROMs are concatenated. This gives massive speedups for systems with many repeated or similar components.

| Step | Code | Description |
|------|------|-------------|
| Geometry | Multi-solid or split CAD | Each solid gets its own mesh and domain |
| Solve Per-Domain | `fds.solve(fmin, fmax, n)` | Solves each domain |
| Reduce Per-Domain | `fds.foms.reduce(tol)` | POD reduction on each domain |
| Concatenate ROMs | `roms.concatenate()` | Cascade reduced S-matrices |
| Solve Concat | `cs.solve(fmin, fmax, n)` | Solve concatenated ROM system |

**When to use:** Large multi-component systems with many frequency points. If one component changes, only its ROM needs recomputation.

---

## Object Interaction

The following sequence diagram shows the typical interaction flow for a single-solid analysis (Pathway 1):

```mermaid
sequenceDiagram
    participant User
    participant EMProject
    participant Geometry
    participant FDS as FrequencyDomainSolver
    participant MOR as ModelOrderReduction

    User->>EMProject: EMProject(name, base_dir)
    User->>Geometry: RectangularWaveguide(a, L)
    User->>EMProject: proj.geometry = geom
    EMProject->>FDS: creates solver internally
    User->>FDS: fds.solve(fmin, fmax, n)
    FDS->>FDS: assemble K, M, B matrices
    FDS->>FDS: solve at n frequency points
    FDS-->>User: results dict (Z, S matrices)
    User->>FDS: fds.fom.reduce(tol)
    FDS->>MOR: create ROM from snapshots
    MOR->>MOR: SVD truncation
    User->>MOR: rom.solve(fmin, fmax, n_fine)
    MOR-->>User: wideband Z, S (milliseconds)
```

The following shows the multi-solid concatenation flow (Pathway 3/4):

```mermaid
sequenceDiagram
    participant User
    participant FDS as FrequencyDomainSolver
    participant FOMCollection
    participant CS as ConcatenatedSystem
    participant MOR as ModelOrderReduction

    User->>FDS: fds.solve(fmin, fmax, n)
    FDS->>FDS: solve each domain independently
    FDS-->>FOMCollection: fds.foms (per-domain results)

    alt Pathway 3: FOM Concatenation
        User->>FOMCollection: fds.foms.concatenate()
        FOMCollection->>CS: cascade S-matrices (W=I)
        User->>CS: cs.solve(fmin, fmax, n)
        User->>CS: cs.reduce(tol)
        CS->>MOR: POD on concatenated snapshots
    else Pathway 4: ROM Concatenation
        User->>FOMCollection: fds.foms.reduce(tol)
        FOMCollection->>MOR: POD each domain
        User->>MOR: roms.concatenate()
        MOR->>CS: cascade reduced S-matrices
        User->>CS: cs.solve(fmin, fmax, n)
    end

    CS-->>User: global S/Z parameters
```

---

## Computation Graph Summary

The entire computation graph is navigable via attribute access:

```
Single-Solid:
  proj.fds.fom                    → FOMResult
  proj.fds.fom.reduce()           → ModelOrderReduction
  rom.solve(fmin, fmax, n)        → (updates ROM in-place)

Multi-Solid:
  proj.fds.foms                   → FOMCollection (per-domain)
  proj.fds.foms[0]                → FOMResult (first domain)
  proj.fds.foms.concatenate()     → ConcatenatedSystem (FOM-level)
  proj.fds.foms.reduce()          → ROMCollection (per-domain ROMs)
  roms.concatenate()              → ConcatenatedSystem (ROM-level)
  cs.solve(fmin, fmax, n)         → (updates CS in-place)
  cs.reduce()                     → ModelOrderReduction (2nd-level)
```
