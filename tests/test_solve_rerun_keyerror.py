import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.em_project import EMProject
from geometry.primitives import Box

def test_solve_rerun_keyerror():
    project_name = "TestSolveRerunKeyError"
    base_dir = Path("test_sims")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()

    try:
        project = EMProject(project_name, base_dir=str(base_dir))
        project.geometry = Box(dimensions=(0.1, 0.05, 0.1))
        
        print("--- Step 1: Initial Solve ---")
        project.fds.solve(fmin=1.0, fmax=2.0, nsamples=2, order=3)
        print("Initial solve passed.")
        
        print("--- Step 2: Rerun Solve (same order) ---")
        # This triggered the KeyError before the fix
        project.fds.solve(fmin=1.0, fmax=2.0, nsamples=2, order=3, rerun=True)
        print("Rerun solve (same order) passed.")

        print("--- Step 3: Solve with different order ---")
        project.fds.solve(fmin=1.0, fmax=2.0, nsamples=2, order=2, rerun=True)
        print("Solve with different order passed.")

        print("\n=== SOLVE RERUN KEYERROR VERIFICATION PASSED ===")

    except Exception as e:
        print(f"\nVerification FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pass

if __name__ == "__main__":
    test_solve_rerun_keyerror()
