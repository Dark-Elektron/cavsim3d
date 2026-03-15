try:
    import OCC.Core.STEPControl
    print("SUCCESS: OCC.Core.STEPControl imported successfully.")
except Exception as e:
    print(f"FAILED to import OCC.Core.STEPControl: {e}")

try:
    import netgen.occ
    print("SUCCESS: netgen.occ imported successfully.")
except Exception as e:
    print(f"FAILED to import netgen.occ: {e}")
