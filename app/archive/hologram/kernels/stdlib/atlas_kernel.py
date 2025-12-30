"""
Atlas Kernel Primitives

Common functions and types for Atlas kernel schemas.
Kernel schemas should import from this module.

Example usage:
    from atlas_kernel import DeviceArray, f32, u32, get_global_id
    
    def my_kernel(a: DeviceArray[f32], n: u32):
        idx = get_global_id()
        if idx < n:
            # kernel logic
"""

# Type aliases for Atlas kernel interface
class DeviceArray:
    """Device array type annotation for Atlas vGPU memory"""
    def __class_getitem__(cls, item):
        return cls

f32 = float  # 32-bit floating point
i32 = int    # 32-bit signed integer
u32 = int    # 32-bit unsigned integer
f64 = float  # 64-bit floating point
usize = int  # Pointer-sized unsigned integer

def get_global_id() -> int:
    """
    Get global thread ID - replaced during compilation
    
    Returns the unique thread index in the parallel execution.
    This function is replaced by the kernel code generator with actual
    thread ID calculation based on grid/block dimensions.
    """
    pass


def atomic_add_f32(addr, value: f32):
    """
    Atomic addition for reduction operations
    
    Atomically adds value to the memory location pointed to by addr.
    This intrinsic is replaced by the kernel code generator with the
    appropriate atomic instruction (CPU or GPU).
    """
    pass


def atomic_add_u32(addr, value: u32):
    """Atomic add for u32 - replaced during compilation"""
    pass


def atomic_add_i32(addr, value: i32):
    """Atomic add for i32 - replaced during compilation"""
    pass


def atomic_add(addr: DeviceArray[u32], index: u32, value: u32) -> u32:
    """
    Atomic addition with return value

    Atomically adds value to addr[index] and returns the old value.
    This intrinsic is replaced by the kernel code generator with the
    appropriate atomic instruction.
    """
    pass


def atomic_min(addr: DeviceArray[f32], index: u32, value: f32):
    """
    Atomic minimum for optimization operations

    Atomically compares and stores the minimum value at addr[index].
    This intrinsic is replaced by the kernel code generator with the
    appropriate atomic compare-and-swap instruction.
    """
    pass


def expf(x: f32) -> f32:
    """Standard library exponential"""
    pass

def logf(x: f32) -> f32:
    """Standard library logarithm"""
    pass

def sinf(x: f32) -> f32:
    """Standard library sine"""
    pass

def cosf(x: f32) -> f32:
    """Standard library cosine"""
    pass

def sqrtf(x: f32) -> f32:
    """Standard library square root"""
    pass

def sqrt(x: f32) -> f32:
    """Square root function"""
    pass

def rsqrt(x: f32) -> f32:
    """Reciprocal square root: 1/sqrt(x)"""
    pass

def exp(x: f32) -> f32:
    """Exponential function"""
    pass


def inline(func):
    """
    Mark a function for compile-time inlining.

    The function will be expanded at every call site during compilation,
    producing no function call overhead at runtime.

    Restrictions:
    - Cannot be recursive (will cause compilation error)
    - Cannot call non-inline functions
    - Must be pure (no side effects beyond return value)

    Example:
        @inline
        def sigmoid(x: f32) -> f32:
            return 1.0 / (1.0 + exp(-x))

        def kernel(a: DeviceArray[f32], b: DeviceArray[f32], n: u32):
            idx = get_global_id()
            if idx < n:
                b[idx] = sigmoid(a[idx])  # Inlined during compilation
    """
    func._atlas_inline = True
    return func


# Higher-Order Primitives (Phase 2B)
# Backend-optimized parallel operations

def parallel_map_unary(
    input: DeviceArray[f32],
    output: DeviceArray[f32],
    operation: str,
    n: u32
):
    """
    Apply unary operation to all elements in parallel.

    Backend implementations can optimize this with SIMD (CPU) or
    workgroups (GPU) for maximum performance.

    Args:
        input: Input array
        output: Output array (same size as input)
        operation: Operation name from supported set:
                   "abs", "neg", "sqrt", "exp", "log", "sin", "cos",
                   "tan", "sigmoid", "tanh", "relu", "ceil", "floor",
                   "round", "reciprocal", "erf"
        n: Number of elements

    Example:
        # Compute sqrt of all elements
        parallel_map_unary(input, output, "sqrt", n)

        # Apply ReLU activation
        parallel_map_unary(input, output, "relu", n)

    Note: This is a compiler intrinsic. The backend will generate
          optimized code (SIMD on CPU, parallel kernels on GPU).
    """
    pass  # Implemented by compiler


def parallel_map_binary(
    input_a: DeviceArray[f32],
    input_b: DeviceArray[f32],
    output: DeviceArray[f32],
    operation: str,
    n: u32
):
    """
    Apply binary operation to pairs of elements in parallel.

    Backend implementations can optimize this with SIMD (CPU) or
    workgroups (GPU) for maximum performance.

    Args:
        input_a: First input array
        input_b: Second input array
        output: Output array (same size as inputs)
        operation: Operation name from supported set:
                   "add", "sub", "mul", "div", "pow", "min", "max", "atan2"
        n: Number of elements

    Example:
        # Element-wise addition
        parallel_map_binary(a, b, c, "add", n)

        # Element-wise multiplication
        parallel_map_binary(weights, activations, output, "mul", n)

    Note: This is a compiler intrinsic. The backend will generate
          optimized code (SIMD on CPU, parallel kernels on GPU).
    """
    pass  # Implemented by compiler


def parallel_reduce(
    input: DeviceArray[f32],
    output: DeviceArray[f32],
    operation: str,
    n: u32
):
    """
    Reduce array to single value using associative operation.

    Backend implementations use hierarchical reduction with shared
    memory (GPU) or SIMD reduction instructions (CPU).

    Args:
        input: Input array to reduce
        output: Output buffer (size 1) to store result
        operation: Operation name from supported set:
                   "sum", "product", "min", "max"
        n: Number of elements to reduce

    Example:
        # Sum all elements
        parallel_reduce(values, result, "sum", n)

        # Find maximum value
        parallel_reduce(values, result, "max", n)

    Note: This is a compiler intrinsic. The backend will generate
          optimized hierarchical reduction (tree reduction on GPU,
          SIMD horizontal reduction on CPU).
    """
    pass  # Implemented by compiler

