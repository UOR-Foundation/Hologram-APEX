#!/usr/bin/env python3
"""
Test Phase 2A: Universal DLPack Import Support

This script specifically tests the new HologramTensor.from_dlpack() functionality.
"""

import sys
import numpy as np

print("=" * 70)
print("Phase 2A: Universal DLPack Import Test")
print("=" * 70)
print()

# Test 1: Import from NumPy (if available)
print("[Test 1] Import from NumPy via DLPack")
print("-" * 70)

try:
    import hologram_ffi as hg

    # Create NumPy array
    np_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    print(f"NumPy array shape: {np_array.shape}")
    print(f"NumPy array:\n{np_array}")
    print()

    # Import to Hologram using from_dlpack()
    print("[Importing to Hologram via from_dlpack()...]")
    hologram_tensor = hg.HologramTensor.from_dlpack(np_array)
    print(f"✅ Import successful!")
    print(f"   Hologram tensor: {hologram_tensor}")
    print()

except Exception as e:
    print(f"❌ Test 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Import from PyTorch (if available)
print("[Test 2] Import from PyTorch via DLPack")
print("-" * 70)

try:
    import torch

    # Create PyTorch tensor
    pt_tensor = torch.randn(4, 5, dtype=torch.float32)
    print(f"PyTorch tensor shape: {pt_tensor.shape}")
    print(f"PyTorch tensor (first 2 rows):\n{pt_tensor[:2]}")
    print()

    # Import to Hologram using from_dlpack()
    print("[Importing to Hologram via from_dlpack()...]")
    hologram_tensor = hg.HologramTensor.from_dlpack(pt_tensor)
    print(f"✅ Import successful!")
    print(f"   Hologram tensor: {hologram_tensor}")
    print()

    # Export back to PyTorch to verify data integrity
    print("[Exporting back to PyTorch to verify data...]")
    pt_tensor_back = torch.from_dlpack(hologram_tensor)
    print(f"PyTorch tensor (after round-trip, first 2 rows):\n{pt_tensor_back[:2]}")
    print()

    # Verify data matches
    if torch.allclose(pt_tensor, pt_tensor_back):
        print("✅ Data integrity verified! Round-trip successful.")
    else:
        print("❌ Data mismatch after round-trip!")
        sys.exit(1)
    print()

except ImportError:
    print("⚠️  PyTorch not available, skipping Test 2")
    print()
except Exception as e:
    print(f"❌ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Verify new executor is created if none provided
print("[Test 3] Auto-create executor when not provided")
print("-" * 70)

try:
    import torch

    pt_tensor = torch.ones(3, 3, dtype=torch.float32) * 42.0
    print(f"PyTorch tensor:\n{pt_tensor}")
    print()

    # Import without providing executor_handle
    print("[Importing without executor_handle (should auto-create)...]")
    hologram_tensor = hg.HologramTensor.from_dlpack(pt_tensor)
    print(f"✅ Auto-creation successful!")
    print(f"   Executor handle: {hologram_tensor.executor_handle}")
    print(f"   Hologram tensor: {hologram_tensor}")
    print()

except Exception as e:
    print(f"❌ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Error handling for non-DLPack objects
print("[Test 4] Error handling for invalid input")
print("-" * 70)

try:
    # Try to import a non-DLPack object (should raise TypeError)
    invalid_input = "not a tensor"
    print(f"Attempting to import invalid object: {type(invalid_input)}")

    try:
        hologram_tensor = hg.HologramTensor.from_dlpack(invalid_input)
        print("❌ Should have raised TypeError!")
        sys.exit(1)
    except TypeError as e:
        print(f"✅ Correctly raised TypeError: {e}")
    print()

except Exception as e:
    print(f"❌ Test 4 failed unexpectedly: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: With provided executor
print("[Test 5] Import with explicit executor handle")
print("-" * 70)

try:
    import torch

    # Create executor explicitly
    exec_handle = hg.new_executor()
    print(f"Created executor: handle={exec_handle}")

    pt_tensor = torch.randn(2, 4, dtype=torch.float32)
    print(f"PyTorch tensor shape: {pt_tensor.shape}")
    print()

    # Import with explicit executor
    print("[Importing with explicit executor_handle...]")
    hologram_tensor = hg.HologramTensor.from_dlpack(pt_tensor, executor_handle=exec_handle)
    print(f"✅ Import successful!")
    print(f"   Executor handle matches: {hologram_tensor.executor_handle == exec_handle}")
    print(f"   Hologram tensor: {hologram_tensor}")
    print()

    # Cleanup
    hg.executor_cleanup(exec_handle)
    print("✅ Executor cleaned up")
    print()

except Exception as e:
    print(f"❌ Test 5 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("✅ All Phase 2A Import Tests Passed!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Import from NumPy")
print("  ✅ Import from PyTorch")
print("  ✅ Round-trip data integrity")
print("  ✅ Auto-create executor")
print("  ✅ Error handling")
print("  ✅ Explicit executor handle")
print()
print("Phase 2A (Universal Import Support) is COMPLETE! 🎉")
