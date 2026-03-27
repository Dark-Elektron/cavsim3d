import sys
from pathlib import Path

# Provide the absolute path to a saved simulation project
# Try to load multi_tesla project or create a quick dummy one to debug difference in config

base_dir = Path(r"C:\Users\Soske\Documents\git_projects\cavsim3d_simulations")
project_name = 'multi_tesla'

sys.path.append(r"C:\Users\Soske\Documents\git_projects\cavsim3d")
from core.em_project import EMProject

proj = EMProject(project_name, base_dir=str(base_dir))

if not proj.has_results():
    print("Project not found or has no results.")
    sys.exit(0)

fds = proj.fds
# Re-simulate what solve() does when checking config diffs
current_history = getattr(fds.geometry, '_history', [])
loaded_history = fds._loaded_config.get('geometry_history', [])

print("Current history length:", len(current_history))
print("Loaded history length:", len(loaded_history))

if current_history != loaded_history:
    print("MISMATCH DETECTED!")
    # Find mismatch
    import json
    curr_str = json.dumps(current_history, indent=2, default=str)
    load_str = json.dumps(loaded_history, indent=2, default=str)
    
    if curr_str != load_str:
        print("JSON representations differ!")
        # Print first diff
        for i, (c, l) in enumerate(zip(curr_str.splitlines(), load_str.splitlines())):
            if c != l:
                print(f"Line {i} diff:\nCurr: {c}\nLoad: {l}")
                break
    else:
        print("JSON representations are identically stringified but differ in types (e.g. tuple vs list).")
else:
    print("Histories MATCH exactly.")
