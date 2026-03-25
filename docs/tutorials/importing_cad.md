# Tutorial: Importing CAD Files

**cavsim3d** supports importing existing CAD geometry from **STEP** and **IGES** files using the `OCCImporter` class.

## Basic Import

```python
from geometry.importers import OCCImporter

# Import a STEP file (dimensions in mm, converted to metres internally)
cavity = OCCImporter('./tesla_cavity.step', unit='mm')

# View the imported geometry
cavity.show('geometry')

# Generate mesh
cavity.generate_mesh(maxh=0.02)
cavity.show('mesh')
```

!!! info "Unit Conversion"
    The `unit` parameter tells the importer what units the CAD file uses. Internally, **cavsim3d** works in **metres**. If your CAD file is in millimetres, set `unit='mm'`.

## Import with Manual Control

For more control over the import process (e.g., when you want to split the geometry):

```python
geom = OCCImporter('./cavity.step', unit='mm', auto_build=False)

# Inspect the geometry before meshing
print(f"Bounding box: {geom.get_original_bounds()}")
print(f"Number of solids: {geom.n_solids}")

# Build the geometry (assigns ports, normals, etc.)
geom.build()
geom.generate_mesh(maxh=0.02)
```

## Splitting Imported Geometry

To create a multi-domain structure for per-domain analysis (Pathways 3 & 4):

```python
geom = OCCImporter('./waveguide.step', unit='mm',
                    auto_build=False, maxh=0.04)

# Add splitting planes
geom.add_splitting_plane_at_z(0.05)   # Split at z = 50 mm
geom.add_splitting_plane_at_z(0.10)   # Split at z = 100 mm

# Execute the split and build
geom.split()
geom.finalize(maxh=0.04)

print(f"Domains: {geom.domains}")
print(f"External ports: {geom.external_ports}")
print(f"Internal ports: {geom.internal_ports}")
```

## Using Imports in Assemblies

Imported files can be used as components in an assembly (Pathway 2):

```python
from core.em_project import EMProject

proj = EMProject(name='imported_assembly', base_dir='./simulations')
assembly = proj.create_assembly(main_axis='Z')

part1 = proj.create_importer('./cell.iges', unit='mm')
part2 = proj.create_importer('./cell.iges', unit='mm')

assembly.add("cell1", part1)
assembly.add("cell2", part2, after="cell1")

assembly.build()
assembly.generate_mesh(maxh=0.02)
```

## Supported Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| STEP | `.step`, `.stp` | Preferred — lossless B-Rep |
| IGES | `.iges`, `.igs` | Legacy — may lose some geometry detail |
