//! WASM SIMD (v128) support infrastructure
//!
//! This module provides infrastructure for WASM SIMD operations using 128-bit vectors.
//! SIMD support is optional and provides performance optimizations for bulk operations.
//!
//! # Architecture
//!
//! WASM SIMD provides 128-bit vector types and operations:
//! - `v128` - 128-bit vector (16 bytes)
//! - Lane types: i8x16, i16x8, i32x4, i64x2, f32x4, f64x2
//!
//! # Usage
//!
//! When compiled for `wasm32` target with SIMD enabled:
//! ```text
//! rustc --target wasm32-unknown-unknown -C target-feature=+simd128
//! ```
//!
//! # Implementation Status
//!
//! - ✅ Infrastructure ready (buffer_as_ptr/buffer_as_mut_ptr methods)
//! - ⏳ SIMD operations (to be implemented with `core::arch::wasm32` intrinsics)
//! - ✅ Scalar fallbacks (current implementation)

use crate::backend::BufferHandle;
use crate::backends::wasm::memory::MemoryManager;
use crate::error::Result;

/// SIMD operation trait for vectorized operations
///
/// This trait defines the interface for SIMD-accelerated operations.
/// Implementations can use WASM SIMD intrinsics when available,
/// or fall back to scalar operations otherwise.
pub trait SimdOps {
    /// Vectorized memory copy (memcpy with SIMD)
    ///
    /// Copies data from source buffer to destination buffer using SIMD instructions
    /// when possible. Falls back to scalar copy otherwise.
    ///
    /// # Arguments
    ///
    /// * `src` - Source buffer handle
    /// * `dst` - Destination buffer handle
    /// * `len` - Number of bytes to copy
    ///
    /// # Performance
    ///
    /// With SIMD: ~16x faster (128-bit vectors = 16 bytes per operation)
    /// Without SIMD: Standard memcpy performance
    fn simd_memcpy(&self, src: BufferHandle, dst: BufferHandle, len: usize) -> Result<()>;

    /// Vectorized buffer fill (memset with SIMD)
    ///
    /// Fills buffer with a byte value using SIMD instructions when possible.
    ///
    /// # Arguments
    ///
    /// * `dst` - Destination buffer handle
    /// * `value` - Byte value to fill
    /// * `len` - Number of bytes to fill
    fn simd_memset(&self, dst: BufferHandle, value: u8, len: usize) -> Result<()>;

    /// Vectorized f32 addition (element-wise)
    ///
    /// Adds two f32 arrays element-wise using SIMD (f32x4 lanes).
    ///
    /// # Arguments
    ///
    /// * `a` - First input buffer (f32 array)
    /// * `b` - Second input buffer (f32 array)
    /// * `dst` - Destination buffer (f32 array)
    /// * `len` - Number of f32 elements (not bytes)
    ///
    /// # Performance
    ///
    /// With SIMD: 4x faster (f32x4 = 4 elements per operation)
    fn simd_f32_add(&self, a: BufferHandle, b: BufferHandle, dst: BufferHandle, len: usize) -> Result<()>;

    /// Vectorized f32 multiplication (element-wise)
    ///
    /// Multiplies two f32 arrays element-wise using SIMD (f32x4 lanes).
    fn simd_f32_mul(&self, a: BufferHandle, b: BufferHandle, dst: BufferHandle, len: usize) -> Result<()>;

    /// Vectorized f32 FMA (fused multiply-add)
    ///
    /// Computes `dst[i] = a[i] * b[i] + c[i]` using SIMD (f32x4 lanes).
    fn simd_f32_fma(
        &self,
        a: BufferHandle,
        b: BufferHandle,
        c: BufferHandle,
        dst: BufferHandle,
        len: usize,
    ) -> Result<()>;
}

/// SIMD operations for WASM backend
///
/// Currently uses scalar implementations. To enable SIMD:
///
/// 1. Compile with SIMD target feature:
///    ```text
///    RUSTFLAGS="-C target-feature=+simd128" cargo build --target wasm32-unknown-unknown
///    ```
///
/// 2. Use WASM SIMD intrinsics from `core::arch::wasm32`:
///    ```rust
///    #[cfg(target_arch = "wasm32")]
///    use core::arch::wasm32::*;
///    ```
///
/// 3. Implement vectorized operations:
///    ```rust
///    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
///    unsafe {
///        let va = v128_load(src_ptr);
///        let vb = v128_load(dst_ptr);
///        let result = f32x4_add(va, vb);
///        v128_store(dst_ptr, result);
///    }
///    ```
impl SimdOps for MemoryManager {
    fn simd_memcpy(&self, src: BufferHandle, dst: BufferHandle, len: usize) -> Result<()> {
        // Scalar implementation (fallback)
        // TODO: Implement with v128_load/v128_store when SIMD is available
        let mut data = vec![0u8; len];
        self.copy_from_buffer(src, &mut data)?;
        self.copy_to_buffer(dst, &data)?;
        Ok(())
    }

    fn simd_memset(&self, dst: BufferHandle, value: u8, len: usize) -> Result<()> {
        // Scalar implementation (fallback)
        // TODO: Implement with v128_store when SIMD is available
        let data = vec![value; len];
        self.copy_to_buffer(dst, &data)?;
        Ok(())
    }

    fn simd_f32_add(&self, a: BufferHandle, b: BufferHandle, dst: BufferHandle, len: usize) -> Result<()> {
        // Scalar implementation (fallback)
        // TODO: Implement with f32x4_add when SIMD is available
        let mut a_data = vec![0.0f32; len];
        let mut b_data = vec![0.0f32; len];
        let a_bytes = bytemuck::cast_slice_mut(&mut a_data);
        let b_bytes = bytemuck::cast_slice_mut(&mut b_data);

        self.copy_from_buffer(a, a_bytes)?;
        self.copy_from_buffer(b, b_bytes)?;

        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect();
        let result_bytes = bytemuck::cast_slice(&result);
        self.copy_to_buffer(dst, result_bytes)?;
        Ok(())
    }

    fn simd_f32_mul(&self, a: BufferHandle, b: BufferHandle, dst: BufferHandle, len: usize) -> Result<()> {
        // Scalar implementation (fallback)
        // TODO: Implement with f32x4_mul when SIMD is available
        let mut a_data = vec![0.0f32; len];
        let mut b_data = vec![0.0f32; len];
        let a_bytes = bytemuck::cast_slice_mut(&mut a_data);
        let b_bytes = bytemuck::cast_slice_mut(&mut b_data);

        self.copy_from_buffer(a, a_bytes)?;
        self.copy_from_buffer(b, b_bytes)?;

        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x * y).collect();
        let result_bytes = bytemuck::cast_slice(&result);
        self.copy_to_buffer(dst, result_bytes)?;
        Ok(())
    }

    fn simd_f32_fma(
        &self,
        a: BufferHandle,
        b: BufferHandle,
        c: BufferHandle,
        dst: BufferHandle,
        len: usize,
    ) -> Result<()> {
        // Scalar implementation (fallback)
        // TODO: Implement with f32x4_mul + f32x4_add when SIMD is available
        let mut a_data = vec![0.0f32; len];
        let mut b_data = vec![0.0f32; len];
        let mut c_data = vec![0.0f32; len];
        let a_bytes = bytemuck::cast_slice_mut(&mut a_data);
        let b_bytes = bytemuck::cast_slice_mut(&mut b_data);
        let c_bytes = bytemuck::cast_slice_mut(&mut c_data);

        self.copy_from_buffer(a, a_bytes)?;
        self.copy_from_buffer(b, b_bytes)?;
        self.copy_from_buffer(c, c_bytes)?;

        let result: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .zip(c_data.iter())
            .map(|((x, y), z)| x * y + z)
            .collect();
        let result_bytes = bytemuck::cast_slice(&result);
        self.copy_to_buffer(dst, result_bytes)?;
        Ok(())
    }
}

/// SIMD availability detection
///
/// Returns true if WASM SIMD (v128) is available at runtime.
///
/// # Example
///
/// ```rust
/// use hologram_backends::backends::wasm::simd::is_simd_available;
///
/// if is_simd_available() {
///     // Use SIMD-accelerated path
/// } else {
///     // Fall back to scalar operations
/// }
/// ```
#[cfg(target_arch = "wasm32")]
pub fn is_simd_available() -> bool {
    #[cfg(target_feature = "simd128")]
    {
        true
    }
    #[cfg(not(target_feature = "simd128"))]
    {
        false
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_simd_available() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_availability() {
        // Should work on all platforms (returns false when not wasm32+simd128)
        let _available = is_simd_available();
    }

    #[test]
    fn test_simd_memcpy() {
        let memory = MemoryManager::new();

        // Allocate buffers
        let src = memory.allocate_buffer(64).unwrap();
        let dst = memory.allocate_buffer(64).unwrap();

        // Write test data
        let test_data = vec![42u8; 64];
        memory.copy_to_buffer(src, &test_data).unwrap();

        // SIMD copy
        memory.simd_memcpy(src, dst, 64).unwrap();

        // Verify
        let mut result = vec![0u8; 64];
        memory.copy_from_buffer(dst, &mut result).unwrap();
        assert_eq!(result, test_data);
    }

    #[test]
    fn test_simd_memset() {
        let memory = MemoryManager::new();
        let dst = memory.allocate_buffer(64).unwrap();

        // SIMD memset
        memory.simd_memset(dst, 0xFF, 64).unwrap();

        // Verify
        let mut result = vec![0u8; 64];
        memory.copy_from_buffer(dst, &mut result).unwrap();
        assert_eq!(result, vec![0xFF; 64]);
    }

    #[test]
    fn test_simd_f32_add() {
        let memory = MemoryManager::new();

        // Allocate buffers
        let a = memory.allocate_buffer(16).unwrap(); // 4 f32s
        let b = memory.allocate_buffer(16).unwrap();
        let dst = memory.allocate_buffer(16).unwrap();

        // Write test data
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        memory.copy_to_buffer(a, bytemuck::cast_slice(&a_data)).unwrap();
        memory.copy_to_buffer(b, bytemuck::cast_slice(&b_data)).unwrap();

        // SIMD add
        memory.simd_f32_add(a, b, dst, 4).unwrap();

        // Verify
        let mut result = vec![0.0f32; 4];
        memory
            .copy_from_buffer(dst, bytemuck::cast_slice_mut(&mut result))
            .unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_simd_f32_mul() {
        let memory = MemoryManager::new();

        let a = memory.allocate_buffer(16).unwrap();
        let b = memory.allocate_buffer(16).unwrap();
        let dst = memory.allocate_buffer(16).unwrap();

        let a_data = vec![2.0f32, 3.0, 4.0, 5.0];
        let b_data = vec![10.0f32, 10.0, 10.0, 10.0];
        memory.copy_to_buffer(a, bytemuck::cast_slice(&a_data)).unwrap();
        memory.copy_to_buffer(b, bytemuck::cast_slice(&b_data)).unwrap();

        memory.simd_f32_mul(a, b, dst, 4).unwrap();

        let mut result = vec![0.0f32; 4];
        memory
            .copy_from_buffer(dst, bytemuck::cast_slice_mut(&mut result))
            .unwrap();
        assert_eq!(result, vec![20.0, 30.0, 40.0, 50.0]);
    }

    #[test]
    fn test_simd_f32_fma() {
        let memory = MemoryManager::new();

        let a = memory.allocate_buffer(16).unwrap();
        let b = memory.allocate_buffer(16).unwrap();
        let c = memory.allocate_buffer(16).unwrap();
        let dst = memory.allocate_buffer(16).unwrap();

        let a_data = vec![2.0f32, 3.0, 4.0, 5.0];
        let b_data = vec![10.0f32, 10.0, 10.0, 10.0];
        let c_data = vec![1.0f32, 2.0, 3.0, 4.0];
        memory.copy_to_buffer(a, bytemuck::cast_slice(&a_data)).unwrap();
        memory.copy_to_buffer(b, bytemuck::cast_slice(&b_data)).unwrap();
        memory.copy_to_buffer(c, bytemuck::cast_slice(&c_data)).unwrap();

        memory.simd_f32_fma(a, b, c, dst, 4).unwrap();

        let mut result = vec![0.0f32; 4];
        memory
            .copy_from_buffer(dst, bytemuck::cast_slice_mut(&mut result))
            .unwrap();
        // 2*10+1=21, 3*10+2=32, 4*10+3=43, 5*10+4=54
        assert_eq!(result, vec![21.0, 32.0, 43.0, 54.0]);
    }
}
