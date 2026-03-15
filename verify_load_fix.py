import os
import sys
from pathlib import Path
import warnings

# Add current directory to path
sys.path.append(os.getcwd())

from core.em_project import EMProject

def verify():
    # We will use the PillboxProject if it exists, otherwise create a dummy
    project_name = "PillboxProject"
    base_dir = r"C:\Users\Soske\Documents\git_projects\cavsim3d_simulations"
    
    if not os.path.exists(os.path.join(base_dir, project_name)):
        print(f"Project {project_name} not found at {base_dir}. Skipping specific test.")
        return

    print(f"Attempting to load project '{project_name}' from {base_dir}...")
    
    try:
        from geometry.base import BaseGeometry
        print(f"BaseGeometry subclasses: {[s.__name__ for s in BaseGeometry.__subclasses__()]}")
        
        # Try importing importers explicitly
        try:
            import geometry.importers
            print("Successfully imported geometry.importers")
            print(f"BaseGeometry subclasses after import: {[s.__name__ for s in BaseGeometry.__subclasses__()]}")
        except Exception as e:
            print(f"Failed to import geometry.importers: {e}")

        # This used to hang/fail in non-interactive mode if source_link was missing
        project = EMProject.load(project_name, base_dir=base_dir)
        
        print(f"Project initialized. project.geometry is: {project.geometry}")
        
        if project.geometry is not None:
            print("SUCCESS: Geometry loaded correctly.")
            if hasattr(project.geometry, 'mesh'):
                 print(f"SUCCESS: Geometry has mesh attribute (type: {type(project.geometry.mesh)})")
            else:
                 print("ERROR: Geometry object missing mesh attribute?")
        else:
            print("ERROR: Geometry is still None. Checking project metadata...")
            metadata_file = os.path.join(base_dir, project_name, "project.json")
            with open(metadata_file, "r") as f:
                import json
                meta = json.load(f)
                print(f"Project metadata: {meta}")
            
    except Exception as e:
        print(f"FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
