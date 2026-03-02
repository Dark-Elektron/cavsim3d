# Frequency-Domain Solver — Finalized Architecture

## Design Decisions (Resolved)

| Decision | Resolution |
|----------|-----------|
| `.rom` vs `.reduce()` | **Explicit `.reduce()` only** — no `.rom` property |
| Multiple `.reduce()` calls | Each call returns a **new** object (no caching) |
| Per-solid FOM reducibility | Each `fds.foms[i]` is independently reducible via `.reduce()` |
| `ConcatenatedSystem.reduce()` | Second-level POD — `.solve()` creates snapshots, then `.reduce()` compresses |
| FOM-level concatenation | **Option (a)** — wrap with W=I as [ReducedStructure](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/structures.py#8-108), warn about matrix size |
| Z-matrix concatenation | **Removed** — concatenation always operates on system matrices (K, M, B), not Z/S |
| [get_eigenvalues()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#130-135) | Universal — FOM uses (K, M), all reduced objects use their A matrix |

---

## Finalized Object Hierarchy

```mermaid
graph TD
    FDS["FrequencyDomainSolver"]
    
    subgraph "Single Solid"
        FDS -->|".fom"| FOM["FOM: K, M, B → eigenvalues from (K,M)"]
        FOM -->|".reduce(tol)"| MOR1["ModelOrderReduction: A_r, B_r, W"]
        MOR1 -->|".solve(f)"| MOR1
        MOR1 -->|".reduce(tol)"| MOR1b["ModelOrderReduction (2nd level)"]
        MOR1b -->|".solve(f)"| MOR1b
        MOR1 -->|".get_eigenvalues()"| EV1["eigenvalues from A_r"]
        MOR1b -->|".get_eigenvalues()"| EV1b["eigenvalues from A_r'"]
    end
    
    subgraph "Multi Solid"
        FDS -->|".foms"| FOMS["FOMCollection: per-solid"]
        FOMS -->|"[i]"| FOMi["FOM (solid i): K_i, M_i, B_i"]
        FOMi -->|".reduce(tol)"| MOR_i["MOR (solid i)"]
        MOR_i -->|".solve(f) / .reduce()"| MOR_i
        
        FOMS -->|".reduce(tol)"| ROMS["ROMCollection (new each call)"]
        FOMS -->|".concatenate()"| CS_FOM["ConcatenatedSystem (W=I, ⚠ large)"]
        
        ROMS -->|".concatenate()"| CS_ROM["ConcatenatedSystem"]
        CS_FOM -->|".solve(f)"| CS_FOM
        CS_ROM -->|".solve(f)"| CS_ROM
        CS_FOM -->|".reduce(tol)"| MOR_CF["MOR (2nd-level POD)"]
        CS_ROM -->|".reduce(tol)"| MOR_CR["MOR (2nd-level POD)"]
    end
```

**Key principle:** [ModelOrderReduction](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/reduction.py#13-1027) is always the terminal object type. Every path eventually reaches it. Once there, `.solve()` can be called with different frequency ranges, and `.reduce()` can be called again for further compression (as long as the matrices have sufficient rank).

---

## Matrix Ownership & Eigenvalue Sources

| Object | Owns | [get_eigenvalues()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#130-135) source | [solve()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/frequency_domain.py#650-797) |
|--------|------|---------------------------|-----------|
| [FOM](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#40-145) (single/per-domain) | K, M, B, snapshots | Generalized eigenproblem on (K, M) | FOM frequency sweep |
| [FOMCollection](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#329-556) | List of FOMs | Per-solid (K_i, M_i) | N/A (delegates to FOMs) |
| [ModelOrderReduction](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/reduction.py#13-1027) | A_r, B_r, W | Standard eigenproblem on A_r | Reduced frequency sweep |
| [ConcatenatedSystem](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/concatenation.py#31-624) | A_coupled, B_coupled, W_coupled | Standard eigenproblem on A_coupled | Coupled frequency sweep |
| [ROMCollection](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#562-708) | List of per-domain MORs | Per-solid A_r_i | N/A (delegates to MORs) |

---

## Changes Required

### 1. Remove
- [ROMResult](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#151-218) class (passive container → users interact with [ModelOrderReduction](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/reduction.py#13-1027) directly)
- [ConcatResult](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#224-323) class (passive container → users interact with [ConcatenatedSystem](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/concatenation.py#31-624) directly)
- `.rom` property on [FOMResult](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#40-145)
- `.concat` property on [FOMCollection](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#329-556) / [ROMCollection](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#562-708)
- [_build_cascaded_matrices()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/frequency_domain.py#1160-1221) and `global_method='cascade'`
- [_build_concatenated_z_matrices()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/frequency_domain.py#2567-2714) (Z-level concatenation)
- [_FDSConcatProxy](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#748-756) class
- [_make_concat_result_from_fds_z()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#714-769) helper

### 2. Add / Modify
- **`FOMResult.reduce(tol, ...)`** → returns new [ModelOrderReduction](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/reduction.py#13-1027)
- **`FOMResult.concatenate()`** → raises warning for single-solid
- **`FOMCollection.reduce(tol, ...)`** → returns new [ROMCollection](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#562-708) (no caching)
- **`FOMCollection.concatenate()`** → returns [ConcatenatedSystem](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/concatenation.py#31-624) using W=I (with size warning)
- **`ROMCollection.concatenate()`** → returns [ConcatenatedSystem](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/concatenation.py#31-624) (as now, but explicit method)
- **`ConcatenatedSystem.reduce(tol, ...)`** → second-level POD, returns new [ModelOrderReduction](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/reduction.py#13-1027)
- **`ConcatenatedSystem.solve()`** → stores snapshots for later `.reduce()`
- **`ModelOrderReduction.reduce(tol, ...)`** → further compression, returns new [ModelOrderReduction](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/reduction.py#13-1027)
- **[ReducedStructure](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/rom/structures.py#8-108)** → add `is_full_order` flag (True when W=I), shown in [__repr__](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#706-708)
- **[get_eigenvalues()](file:///c:/Users/Soske/Documents/git_projects/cavsim3d/solvers/results.py#130-135)** on every object, using appropriate source matrix

### 3. Structural flow

```
FOM.reduce()           → ModelOrderReduction (POD on FOM snapshots)
FOMCollection.reduce() → ROMCollection (one MOR per solid)
FOMCollection.concatenate() → ConcatenatedSystem (W=I, warns if large)
ROMCollection.concatenate() → ConcatenatedSystem (standard)
ConcatenatedSystem.solve()  → stores snapshots
ConcatenatedSystem.reduce() → ModelOrderReduction (2nd-level POD on concat snapshots)
ModelOrderReduction.solve()  → stores snapshots
ModelOrderReduction.reduce() → ModelOrderReduction (further POD)
```
