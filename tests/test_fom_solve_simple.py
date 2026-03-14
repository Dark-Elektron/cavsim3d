import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.em_project import EMProject
from geometry.primitives import Box

def test_fom_solve_routing():
    project_name = "TestFOMSolveSimple"
    base_dir = Path("test_sims")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()

    try:
        project = EMProject(project_name, base_dir=str(base_dir))
        project.geometry = Box(dimensions=(0.1, 0.05, 0.1))
        
        print("--- Step 1: Initial Solve ---")
        project.fds.solve(fmin=1.0, fmax=2.0, nsamples=2)
        
        print("--- Step 2: Test FOMResult.solve() routing ---")
        fom = project.fds.fom
        # Passing rerun=True to ensure it actually executes
        res = fom.solve(fmin=2.0, fmax=3.0, nsamples=5, rerun=True)
        
        print(f"New frequencies count: {len(project.fds.frequencies)}")
        if len(project.fds.frequencies) == 5:
            print("FOMResult.solve() routing PASSED.")
        else:
            print(f"FOMResult.solve() routing FAILED. Count: {len(project.fds.frequencies)}")
            sys.exit(1)

    finally:
        pass

if __name__ == "__main__":
    test_fom_solve_routing()
