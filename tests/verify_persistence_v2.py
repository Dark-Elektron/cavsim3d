import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.em_project import EMProject
from geometry.primitives import Box

def verify_persistence_v2():
    project_name = "VerifyProjV2"
    base_dir = Path("test_sims")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()

    try:
        print(f"--- Step 1: Initialize Project ---")
        project = EMProject(project_name, base_dir=str(base_dir))
        
        print(f"--- Step 2: Create Geometry ---")
        geo = Box(dimensions=(0.1, 0.05, 0.2), maxh=0.08)
        project.geometry = geo
        
        print(f"--- Step 3: Solve FOM ---")
        project.fds.solve(fmin=1.0, fmax=2.0, nsamples=5, store_snapshots=True)
        
        print(f"--- Step 3.5: Reduce and Solve ROM before saving ---")
        rom_orig = project.fds.fom.reduce(tol=1e-1)
        rom_orig.solve(fmin=1.0, fmax=2.0, nsamples=5)

        print(f"--- Step 4: Save Project ---")
        project.save()
        
        print(f"--- Step 5: Load Project into new instance ---")
        loaded_project = EMProject(project_name, base_dir=str(base_dir))
        
        print(f"--- Step 6: Verify frequencies and snapshots ---")
        if loaded_project.fds.frequencies is None:
            raise ValueError("Frequencies not loaded!")
        print(f"Loaded frequencies count: {len(loaded_project.fds.frequencies)}")
        
        if 'global' not in loaded_project.fds.snapshots:
             raise ValueError("Snapshots not loaded into solver!")
        print("Snapshots found in solver.")
        
        if loaded_project.fds._fes_global is None:
            raise ValueError("Global FES not loaded!")
        print("Global FES found in solver.")

        print(f"--- Step 7: Verify ROM loading ---")
        try:
            rom = loaded_project.fds.fom.rom
            print("ROM successfully loaded from disk.")
            if rom._Z_matrix is None:
                 raise ValueError("ROM results (_Z_matrix) not restored!")
            print("ROM results restored.")
        except RuntimeError as e:
            print(f"ROM access failed: {e}")
            raise

        print(f"--- Step 8: Verify ROM rerun logic ---")
        # Solve ROM again without rerun (should warn and return cached)
        print("Solving ROM (second time, no rerun)...")
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results2 = rom.solve(fmin=1.0, fmax=2.0, nsamples=5)
            found_warn = False
            for warn in w:
                if "Results already exist" in str(warn.message):
                    found_warn = True
                    break
            if not found_warn:
                raise ValueError("Expected warning for existing results not found!")
        print("Warning caught as expected.")
        
        # Solve ROM again with rerun
        print("Solving ROM (third time, with rerun=True)...")
        results3 = rom.solve(fmin=1.0, fmax=2.0, nsamples=5, rerun=True)
        print("Rerun solve complete.")

        print("\n=== ALL VERIFICATIONS PASSED ===")

    finally:
        # shutil.rmtree(base_dir)
        pass

if __name__ == "__main__":
    verify_persistence_v2()
