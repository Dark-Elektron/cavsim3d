import sys

def test_order(order):
    print(f"\nTesting order: {order}")
    # Clear modules to force re-import (won't work for DLLs already in process but good for logic)
    for mod in list(sys.modules.keys()):
        if 'OCC' in mod or 'ngsolve' in mod or 'netgen' in mod:
            del sys.modules[mod]
            
    try:
        if order == 'NG_FIRST':
            import ngsolve
            import netgen.occ
            print("NGSolve imported.")
            import OCC.Core.STEPControl
            print("OCC imported.")
        elif order == 'H5_FIRST':
            import h5py
            print("h5py imported.")
            import OCC.Core.STEPControl
            print("OCC imported.")
        else:
            import OCC.Core.STEPControl
            print("OCC imported.")
            import ngsolve
            import netgen.occ
            print("NGSolve imported.")
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_order(sys.argv[1])
    else:
        test_order('NG_FIRST')
