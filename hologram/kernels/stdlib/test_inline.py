"""
Test inline function support in Atlas kernel compiler

Tests cover:
1. Simple inlining (square function)
2. Nested inlining (abs within l2_norm)
3. Function composition (sigmoid within swish)
4. Recursion detection (should fail)
"""

from atlas_kernel import DeviceArray, f32, u32, get_global_id, exp, sqrt, inline
from compiler import compile_to_json
import json


# Test 1: Simple Inlining
@inline
def square(x: f32) -> f32:
    return x * x


def test_simple_inline_kernel(a: DeviceArray[f32], b: DeviceArray[f32], n: u32):
    idx = get_global_id()
    if idx < n:
        b[idx] = square(a[idx])


# Test 2: Sigmoid Inlining
@inline
def sigmoid(x: f32) -> f32:
    return 1.0 / (1.0 + exp(-x))


def test_sigmoid_kernel(input: DeviceArray[f32], output: DeviceArray[f32], n: u32):
    idx = get_global_id()
    if idx < n:
        output[idx] = sigmoid(input[idx])


# Test 3: Nested Inlining
@inline
def abs_inline(x: f32) -> f32:
    return x if x > 0.0 else -x


@inline
def l2_norm(x: f32, y: f32) -> f32:
    return sqrt(abs_inline(x*x) + abs_inline(y*y))


def test_nested_inline_kernel(a: DeviceArray[f32], b: DeviceArray[f32], c: DeviceArray[f32], n: u32):
    idx = get_global_id()
    if idx < n:
        c[idx] = l2_norm(a[idx], b[idx])


# Test 4: Function Composition
@inline
def swish(x: f32) -> f32:
    return x * sigmoid(x)


def test_composition_kernel(input: DeviceArray[f32], output: DeviceArray[f32], n: u32):
    idx = get_global_id()
    if idx < n:
        output[idx] = swish(input[idx])


# Test 5: Recursion Detection (should fail compilation)
@inline
def factorial(n: u32) -> u32:
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # ERROR: recursion not allowed


def test_recursion_kernel(a: DeviceArray[u32], b: DeviceArray[u32], n: u32):
    idx = get_global_id()
    if idx < n:
        b[idx] = factorial(a[idx])


def run_tests():
    print("=" * 70)
    print("INLINE FUNCTION TESTS")
    print("=" * 70)

    # Test 1: Simple inlining
    print("\n[TEST 1] Simple Inlining: square(x) = x * x")
    print("-" * 70)
    try:
        json_str = compile_to_json(test_simple_inline_kernel)
        schema = json.loads(json_str)
        print("✅ PASS: Compilation succeeded")

        # Verify no function call in JSON
        body_str = json.dumps(schema["kernel"]["body"])
        if '"function": "square"' in body_str:
            print("❌ FAIL: Function call found (not inlined)")
        else:
            print("✅ PASS: Function inlined (no call found)")

        print("\nGenerated JSON:")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"❌ FAIL: {e}")

    # Test 2: Sigmoid inlining
    print("\n[TEST 2] Sigmoid Inlining: sigmoid(x) = 1 / (1 + exp(-x))")
    print("-" * 70)
    try:
        json_str = compile_to_json(test_sigmoid_kernel)
        schema = json.loads(json_str)
        print("✅ PASS: Compilation succeeded")

        # Verify sigmoid is inlined but exp is called
        body_str = json.dumps(schema["kernel"]["body"])
        if '"function": "sigmoid"' in body_str:
            print("❌ FAIL: sigmoid call found (not inlined)")
        else:
            print("✅ PASS: sigmoid inlined")

        if '"function": "exp"' in body_str:
            print("✅ PASS: exp call preserved (standard library)")
        else:
            print("⚠️  WARNING: exp call not found")

        print("\nGenerated JSON:")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"❌ FAIL: {e}")

    # Test 3: Nested inlining
    print("\n[TEST 3] Nested Inlining: l2_norm calls abs_inline")
    print("-" * 70)
    try:
        json_str = compile_to_json(test_nested_inline_kernel)
        schema = json.loads(json_str)
        print("✅ PASS: Compilation succeeded")

        # Verify both functions are inlined
        body_str = json.dumps(schema["kernel"]["body"])
        if '"function": "l2_norm"' in body_str:
            print("❌ FAIL: l2_norm call found (not inlined)")
        else:
            print("✅ PASS: l2_norm inlined")

        if '"function": "abs_inline"' in body_str:
            print("❌ FAIL: abs_inline call found (not inlined)")
        else:
            print("✅ PASS: abs_inline inlined")

        print("\nGenerated JSON:")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"❌ FAIL: {e}")

    # Test 4: Function composition
    print("\n[TEST 4] Function Composition: swish calls sigmoid")
    print("-" * 70)
    try:
        json_str = compile_to_json(test_composition_kernel)
        schema = json.loads(json_str)
        print("✅ PASS: Compilation succeeded")

        # Verify both swish and sigmoid are inlined
        body_str = json.dumps(schema["kernel"]["body"])
        if '"function": "swish"' in body_str:
            print("❌ FAIL: swish call found (not inlined)")
        else:
            print("✅ PASS: swish inlined")

        if '"function": "sigmoid"' in body_str:
            print("❌ FAIL: sigmoid call found (not inlined)")
        else:
            print("✅ PASS: sigmoid inlined (nested)")

        print("\nGenerated JSON:")
        print(json.dumps(schema, indent=2))
    except Exception as e:
        print(f"❌ FAIL: {e}")

    # Test 5: Recursion detection
    print("\n[TEST 5] Recursion Detection: factorial (should fail)")
    print("-" * 70)
    try:
        json_str = compile_to_json(test_recursion_kernel)
        print("❌ FAIL: Compilation succeeded (should have failed)")
        print(json_str)
    except ValueError as e:
        if "Recursive" in str(e) or "recursion" in str(e):
            print(f"✅ PASS: Recursion detected: {e}")
        else:
            print(f"❌ FAIL: Wrong error: {e}")
    except Exception as e:
        print(f"❌ FAIL: Unexpected error: {e}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
