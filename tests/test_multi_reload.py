import os
import shutil
from pathlib import Path
import ngsolve # just to ensure import

# Clean up before test
base_dir = Path(r"C:\Users\Soske\Documents\git_projects\cavsim3d_simulations_test")
if base_dir.exists():
    shutil.rmtree(base_dir)
base_dir.mkdir(exist_ok=True)

import sys
sys.path.append(r"C:\Users\Soske\Documents\git_projects\cavsim3d")

from core.em_project import EMProject

print("=== RUN 1: Create and Solve ===")
proj1 = EMProject(name="test_reload", base_dir=str(base_dir))
assembly = proj1.create_assembly(main_axis='Z')

# Create simple boxes instead of iges for speed
from geometry.solids import Box
b1 = Box(center=(0, 0, 0), sizes=(10, 10, 10), material="vacuum")
b2 = Box(center=(0, 0, 10), sizes=(10, 10, 10), material="vacuum")

assembly.add("cavity1", b1)
assembly.add("cavity2", b2, after="cavity1")
assembly.build()
assembly.generate_mesh(maxh=5)

# Setup ports
from geometry.ports import CircularPort
proj1.geo.add_port("cavity1", CircularPort("p1", center=(0, 0, -5), radius=2, normal=(0, 0, -1)))
proj1.geo.add_port("cavity2", CircularPort("p2", center=(0, 0, 15), radius=2, normal=(0, 0, 1)))

# Solve
print("Solving 1...")
res = proj1.fds.solve(fmin=1e9, fmax=2e9, nsamples=3, rerun=False)
print("Solve 1 done")
proj1.save()

print("\n=== RUN 2: Reload 1 ===")
proj2 = EMProject(name="test_reload", base_dir=str(base_dir))
print("Solving 2 (should skip)...")
res2 = proj2.fds.solve(fmin=1e9, fmax=2e9, nsamples=3, rerun=False)
print("Solve 2 done")
proj2.save() # They might be doing this or the system auto-saves

print("\n=== RUN 3: Reload 2 ===")
proj3 = EMProject(name="test_reload", base_dir=str(base_dir))
print("Solving 3 (should skip)...")
res3 = proj3.fds.solve(fmin=1e9, fmax=2e9, nsamples=3, rerun=False)
print("Solve 3 done")
