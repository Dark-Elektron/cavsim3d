
import numpy as np
import json
from pathlib import Path
from core.persistence import ProjectManager, NumpyEncoder

def test_numpy_json_serialization():
    # 1. Prepare data with various NumPy types
    data = {
        "int32": np.int32(10),
        "int64": np.int64(20),
        "float64": np.float64(3.14),
        "array": np.array([1, 2, 3]),
        "nested": {
            "rank": np.int32(5),
            "errors": [np.float64(0.1), np.float64(0.2)]
        }
    }
    
    test_path = Path("./tmp_test_json")
    test_path.mkdir(exist_ok=True)
    filename = "test_numpy.json"
    
    print(f"Testing JSON serialization of NumPy types to {test_path / filename}...")
    
    try:
        # 2. Use ProjectManager.save_json
        ProjectManager.save_json(test_path, data, filename=filename)
        print("Successfully saved JSON with NumPy types.")
        
        # 3. Load and verify
        with open(test_path / filename, "r") as f:
            loaded_data = json.load(f)
            
        print("\nLoaded data:")
        print(json.dumps(loaded_data, indent=2))
        
        # Verify types are converted to standard Python types
        assert isinstance(loaded_data["int32"], int)
        assert isinstance(loaded_data["float64"], float)
        assert isinstance(loaded_data["array"], list)
        assert isinstance(loaded_data["nested"]["rank"], int)
        
        print("\nVerification successful: All NumPy types were correctly converted.")
        
    except Exception as e:
        print(f"\nVerification failed with error: {e}")
        raise
    finally:
        # Cleanup
        if (test_path / filename).exists():
            (test_path / filename).unlink()
        if test_path.exists():
            test_path.rmdir()

if __name__ == "__main__":
    test_numpy_json_serialization()
