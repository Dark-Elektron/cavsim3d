import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from core.em_project import EMProject

def verify():
    project_name = "PillboxProject"
    base_dir = r"C:\Users\Soske\Documents\git_projects\cavsim3d_simulations"
    
    if not os.path.exists(os.path.join(base_dir, project_name)):
        print(f"Project {project_name} not found at {base_dir}. Sleeping...")
        return

    print(f"Loading project '{project_name}'...")
    project = EMProject.load(project_name, base_dir=base_dir)
    
    fds = project.fds
    print(f"FES domains detected: {list(fds._fes.keys())}")
    
    if len(fds._fes) > 0:
        print("SUCCESS: FES objects reconstructed.")
        for domain, fes in fds._fes.items():
            print(f"  Domain '{domain}': FES ndof = {fes.ndof}")
            
        # Check snapshot context
        try:
            snapshot_key, fes, ports = fds._get_snapshot_context(domain='cell_1')
            print(f"Snapshot context for 'cell_1': key={snapshot_key}, fes_ndof={fes.ndof}, ports={ports}")
            print("SUCCESS: Snapshot context resolved correctly.")
        except Exception as e:
            print(f"FAILED to resolve snapshot context: {e}")
    else:
        print("ERROR: _fes is still empty.")

if __name__ == "__main__":
    verify()
