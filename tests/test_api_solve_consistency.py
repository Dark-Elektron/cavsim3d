import sys
import os
import shutil
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from core.em_project import EMProject
from geometry.primitives import Box, RectangularWaveguide
from geometry.assembly import Assembly

def test_api_solve_consistency():
    project_name = "TestAPISolveFinal"
    base_dir = Path("test_sims")
    if base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir()

    try:
        print("--- Step 1: Initialize Multi-domain Project ---")
        project = EMProject(project_name, base_dir=str(base_dir))
        
        assembly = Assembly(main_axis='Z')
        # Use two different primitives to ensure they don't fuse easily
        b1 = Box(dimensions=(0.04, 0.04, 0.04))
        # RectangularWaveguide(a, L, b, maxh)
        rw1 = RectangularWaveguide(a=0.04, b=0.04, L=0.04)
        
        assembly.add("box", b1)
        assembly.add("wg", rw1, after="box", gap=0.005)
        assembly.build()
        assembly.generate_mesh(maxh=0.03)
        
        project.geometry = assembly
        
        print(f"Collection domains: {project.fds.domains}")
        
        print("--- Step 2: Initial Solve to populate fom/foms properties ---")
        project.fds.solve(fmin=1.0, fmax=1.1, nsamples=2, per_domain=True)
        print("Initial solve complete.")

        print("--- Step 3: Test FOMCollection.solve() ---")
        # fds.foms should be available now
        foms = project.fds.foms
        print(f"FOMS domains: {[f.domain for f in foms]}")
        res_foms = foms.solve(fmin=1.0, fmax=1.1, nsamples=3)
        if len(project.fds.frequencies) != 3:
            raise ValueError("FOMCollection.solve() failed!")
        print("FOMCollection.solve() passed.")

        print("--- Step 4: Test FOMResult.solve() ---")
        fom1 = foms[0]
        res_fom1 = fom1.solve(fmin=1.0, fmax=1.1, nsamples=4)
        if len(project.fds.frequencies) != 4:
            raise ValueError("FOMResult.solve() failed!")
        print("FOMResult.solve() passed.")

        if len(foms) > 1:
            print("--- Step 5: Test ROMCollection.solve() (Multi-domain) ---")
            roms = foms.reduce(nsamples=5, solver_type='direct')
            res_roms = roms.solve(fmin=1.0, fmax=1.1, nsamples=5)
            if len(roms.frequencies) != 5:
                 raise ValueError("ROMCollection.solve() failed!")
            print("ROMCollection.solve() passed.")
        else:
            print("--- Step 5: Test ModelOrderReduction.solve() (Single-domain fallback) ---")
            rom = project.fds.fom.reduce(nsamples=5, solver_type='direct')
            res_rom = rom.solve(fmin=1.0, fmax=1.1, nsamples=5)
            print("Single-domain ROM solve passed.")

        print("\n=== ALL API CONSISTENCY TESTS PASSED ===")

    finally:
        # shutil.rmtree(base_dir)
        pass

if __name__ == "__main__":
    test_api_solve_consistency()
