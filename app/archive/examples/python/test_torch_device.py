#!/usr/bin/env python3
"""
Test Hologram PyTorch Device Integration

Tests the hologram device functionality with PyTorch.
"""

import sys
import torch
import hologram_ffi as hg

print("=" * 70)
print("Hologram PyTorch Device Integration Test")
print("=" * 70)
print()

# Test 1: Basic tensor creation and conversion
print("[Test 1] Create Hologram tensor and convert to PyTorch")
print("-" * 70)

try:
    # Create Hologram tensor
    exec_handle = hg.new_executor()
    buffer_handle = hg.executor_allocate_buffer(exec_handle, 12)
    hologram_tensor = hg.HologramTensor(exec_handle, buffer_handle, shape=[3, 4])

    print(f"Hologram tensor: {hologram_tensor}")

    # Convert to PyTorch (zero-copy via DLPack)
    pytorch_tensor = torch.from_dlpack(hologram_tensor)
    print(f"PyTorch tensor: {pytorch_tensor}")
    print(f"PyTorch shape: {pytorch_tensor.shape}")
    print(f"PyTorch dtype: {pytorch_tensor.dtype}")
    print("✅ Test 1 passed")
    print()

except Exception as e:
    print(f"❌ Test 1 failed: {e}")
    sys.exit(1)

# Test 2: PyTorch → Hologram → PyTorch round-trip
print("[Test 2] PyTorch → Hologram → PyTorch round-trip")
print("-" * 70)

try:
    # Create PyTorch tensor
    pt_tensor = torch.randn(2, 3, dtype=torch.float32)
    print(f"Original PyTorch tensor:\n{pt_tensor}")
    print()

    # Import to Hologram
    hg_tensor = hg.HologramTensor.from_dlpack(pt_tensor)
    print(f"Imported to Hologram: {hg_tensor}")

    # Export back to PyTorch
    pt_tensor_back = torch.from_dlpack(hg_tensor)
    print(f"Exported back to PyTorch:\n{pt_tensor_back}")
    print()

    # Verify data matches
    if torch.allclose(pt_tensor, pt_tensor_back):
        print("✅ Test 2 passed - Data preserved through round-trip")
    else:
        print("❌ Test 2 failed - Data mismatch")
        sys.exit(1)
    print()

except Exception as e:
    print(f"❌ Test 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: PyTorch operations on Hologram-backed tensors
print("[Test 3] PyTorch operations on Hologram-backed tensors")
print("-" * 70)

try:
    # Create two Hologram tensors
    exec1 = hg.new_executor()
    buf1 = hg.executor_allocate_buffer(exec1, 6)
    hg_tensor1 = hg.HologramTensor(exec1, buf1, shape=[2, 3])

    exec2 = hg.new_executor()
    buf2 = hg.executor_allocate_buffer(exec2, 6)
    hg_tensor2 = hg.HologramTensor(exec2, buf2, shape=[2, 3])

    # Convert to PyTorch
    pt1 = torch.from_dlpack(hg_tensor1)
    pt2 = torch.from_dlpack(hg_tensor2)

    # Initialize with data
    pt1.fill_(2.0)
    pt2.fill_(3.0)

    print(f"Tensor 1: {pt1}")
    print(f"Tensor 2: {pt2}")

    # PyTorch operations
    result_add = pt1 + pt2
    result_mul = pt1 * pt2

    print(f"Addition result: {result_add}")
    print(f"Multiplication result: {result_mul}")

    # Verify results
    expected_add = torch.full((2, 3), 5.0)
    expected_mul = torch.full((2, 3), 6.0)

    if torch.allclose(result_add, expected_add) and torch.allclose(result_mul, expected_mul):
        print("✅ Test 3 passed - Operations work correctly")
    else:
        print("❌ Test 3 failed - Incorrect results")
        sys.exit(1)
    print()

except Exception as e:
    print(f"❌ Test 3 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 70)
print("✅ All PyTorch Device Integration Tests Passed!")
print("=" * 70)
print()
print("Summary:")
print("  ✅ Hologram ↔ PyTorch conversion")
print("  ✅ Round-trip data integrity")
print("  ✅ PyTorch operations on Hologram tensors")
print()
print("Note: For native torch.device('hologram') support,")
print("      a full PrivateUse1 C++ extension is required.")
print("      Current implementation uses DLPack for seamless interop.")
