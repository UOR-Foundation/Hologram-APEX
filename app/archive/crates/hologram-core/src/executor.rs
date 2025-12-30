//! Executor for managing backend execution
//!
//! The `Executor` wraps a `hologram-backends::Backend` and provides high-level APIs for
//! buffer allocation and operation execution.
//!
//! ## Architecture
//!
//! ```text
//! hologram-core::Executor
//!   ↓ delegates to
//! hologram-backends::Backend (CpuBackend, GpuBackend, etc.)
//!   ↓ executes
//! ISA Program (precompiled at build-time)
//! ```
//!
//! ## Zero-Copy Design
//!
//! - Buffers are backend-managed (no intermediate copying)
//! - Direct memory access via backend handles
//! - Reference-based APIs to avoid unnecessary allocations
//!
//! ## Performance
//!
//! - Zero runtime compilation (all operations precompiled)
//! - Direct ISA execution via backend (~10-20ns overhead)
//! - Rayon parallelization in backend execution loops
//! - <200ns total overhead per operation

use crate::buffer::{Buffer, MemoryPool};
use crate::class_allocator::{ClassAllocator, SharedClassAllocator};
use crate::error::{Error, Result};
use crate::instrumentation::Instant;
use crate::sync::{lock_mutex, read_lock, write_lock, Mutex, RwLock};
use hologram_backends::backend::{BlockDim, GridDim, SharedMemoryConfig};
use hologram_backends::{Backend, BufferHandle, CpuBackend, LaunchConfig};
use std::collections::HashMap;
use std::sync::Arc;

// On non-WASM targets, Backend must be Send + Sync for thread safety
#[cfg(not(target_arch = "wasm32"))]
type BoxedBackend = Box<dyn Backend + Send + Sync>;

// On WASM, Backend doesn't need Send + Sync (single-threaded environment)
#[cfg(target_arch = "wasm32")]
type BoxedBackend = Box<dyn Backend>;

#[cfg(target_vendor = "apple")]
use hologram_backends::MetalBackend;

#[cfg(feature = "cuda")]
use hologram_backends::CudaBackend;

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
use hologram_backends::WebGpuBackend;

/// Backend type for executor initialization
///
/// Specifies which backend to use for computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU backend (always available)
    Cpu,
    /// Metal backend (Apple Silicon, macOS only)
    Metal,
    /// CUDA backend (NVIDIA GPUs)
    Cuda,
    /// WebGPU backend (WASM GPU acceleration)
    WebGpu,
}

/// Executor for backend execution
///
/// The executor wraps a `hologram-backends::Backend` and provides:
/// - Buffer allocation mapped to backend handles
/// - Zero-copy memory management
/// - Direct ISA program execution
/// - Rayon-parallelized execution loops
///
/// # Example
///
/// ```text
/// use hologram_core::Executor;
///
/// let mut exec = Executor::new()?;
/// let buf = exec.allocate::<f32>(3072)?; // One class worth of f32 elements
///
/// // Operations execute precompiled ISA Programs
/// // No runtime compilation overhead
/// ```
pub struct Executor {
    pub(crate) backend: Arc<RwLock<BoxedBackend>>,
    buffer_mappings: [Option<BufferHandle>; 96], // Class → Backend buffer handle (direct array indexing)
    pub(crate) class_allocator: SharedClassAllocator, // Shared class allocator for automatic deallocation
    is_boundary_pool: [bool; 96],                // Track which classes use boundary pool (PoolHandle(0))
    // Class-free buffer support for large allocations (> 1.125 MB)
    pub(crate) class_free_buffers: Arc<Mutex<HashMap<u64, BufferHandle>>>, // buffer_id → handle
    next_buffer_id: Arc<Mutex<u64>>,                                       // Counter for generating unique buffer IDs
}

impl Executor {
    /// Create a new executor with CPU backend
    ///
    /// This initializes:
    /// - CpuBackend with rayon parallelization
    /// - Empty buffer/pool mappings
    /// - Class allocation counter starting at 0
    ///
    /// This is equivalent to `Executor::new_with_backend(BackendType::Cpu)`.
    #[tracing::instrument]
    pub fn new() -> Result<Self> {
        Self::new_with_backend(BackendType::Cpu)
    }

    /// Create a new executor with specified backend
    ///
    /// # Arguments
    ///
    /// * `backend_type` - The backend to use (Cpu, Metal, or Cuda)
    ///
    /// # Returns
    ///
    /// Returns `Err` if the specified backend is not available or not yet implemented.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hologram_core::{Executor, BackendType};
    ///
    /// // Create CPU executor (always available)
    /// let cpu_exec = Executor::new_with_backend(BackendType::Cpu)?;
    ///
    /// // Create Metal executor (Apple Silicon only)
    /// let metal_exec = Executor::new_with_backend(BackendType::Metal)?;
    ///
    /// // Create CUDA executor (NVIDIA GPUs only)
    /// let cuda_exec = Executor::new_with_backend(BackendType::Cuda)?;
    /// # Ok::<(), hologram_core::Error>(())
    /// ```
    #[tracing::instrument]
    pub fn new_with_backend(backend_type: BackendType) -> Result<Self> {
        let start = Instant::now();

        let backend: BoxedBackend = match backend_type {
            BackendType::Cpu => Box::new(CpuBackend::new()),
            BackendType::Metal => {
                // Metal backend only available on Apple platforms
                #[cfg(target_vendor = "apple")]
                {
                    match MetalBackend::new() {
                        Ok(backend) => Box::new(backend),
                        Err(e) => {
                            return Err(Error::InvalidOperation(format!(
                                "Failed to create Metal backend: {}",
                                e
                            )));
                        }
                    }
                }
                #[cfg(not(target_vendor = "apple"))]
                {
                    return Err(Error::InvalidOperation(
                        "Metal backend only available on Apple platforms".into(),
                    ));
                }
            }
            BackendType::Cuda => {
                // CUDA backend for NVIDIA GPUs (requires 'cuda' feature)
                #[cfg(feature = "cuda")]
                {
                    match CudaBackend::new() {
                        Ok(backend) => Box::new(backend),
                        Err(e) => {
                            return Err(Error::InvalidOperation(format!("Failed to create CUDA backend: {}", e)));
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(Error::InvalidOperation(
                        "CUDA backend requires 'cuda' feature to be enabled".into(),
                    ));
                }
            }
            BackendType::WebGpu => {
                return Err(Error::InvalidOperation(
                    "WebGPU backend requires async initialization. Use Executor::new_with_backend_async() instead"
                        .into(),
                ));
            }
        };

        let duration_us = start.elapsed().as_micros() as u64;
        tracing::debug!(
            duration_us = duration_us,
            backend = ?backend_type,
            "executor_created"
        );

        Ok(Self {
            backend: Arc::new(RwLock::new(backend)),
            buffer_mappings: [None; 96],
            class_allocator: Arc::new(Mutex::new(ClassAllocator::new())),
            is_boundary_pool: [false; 96],
            class_free_buffers: Arc::new(Mutex::new(HashMap::new())),
            next_buffer_id: Arc::new(Mutex::new(1)), // Start from 1 (0 reserved for invalid)
        })
    }

    /// Create a new executor with specified backend (async version for WebGPU)
    ///
    /// This async version is required for WebGPU backend initialization.
    /// For other backends, prefer `new_with_backend()`.
    ///
    /// # Arguments
    ///
    /// * `backend_type` - The backend to use
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    /// # async fn example() -> Result<(), hologram_core::Error> {
    /// use hologram_core::{Executor, BackendType};
    ///
    /// // Create WebGPU executor (WASM only)
    /// let exec = Executor::new_with_backend_async(BackendType::WebGpu).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    pub async fn new_with_backend_async(backend_type: BackendType) -> Result<Self> {
        let start = Instant::now();

        let backend: BoxedBackend = match backend_type {
            BackendType::WebGpu => match WebGpuBackend::new().await {
                Ok(backend) => Box::new(backend),
                Err(e) => {
                    return Err(Error::InvalidOperation(format!(
                        "Failed to create WebGPU backend: {}",
                        e
                    )));
                }
            },
            _ => {
                // For non-WebGPU backends, use the sync version
                return Self::new_with_backend(backend_type);
            }
        };

        let duration_us = start.elapsed().as_micros() as u64;
        tracing::debug!(
            duration_us = duration_us,
            backend = ?backend_type,
            "executor_created_async"
        );

        Ok(Self {
            backend: Arc::new(RwLock::new(backend)),
            buffer_mappings: [None; 96],
            class_allocator: Arc::new(Mutex::new(ClassAllocator::new())),
            is_boundary_pool: [false; 96],
            class_free_buffers: Arc::new(Mutex::new(HashMap::new())),
            next_buffer_id: Arc::new(Mutex::new(1)), // Start from 1 (0 reserved for invalid)
        })
    }

    /// Create a new executor with automatic backend detection
    ///
    /// Automatically selects the best available backend in this order:
    /// 1. Metal (if on Apple Silicon)
    /// 2. CUDA (if NVIDIA GPU available)
    /// 3. CPU (fallback, always available)
    ///
    /// # Example
    ///
    /// ```
    /// use hologram_core::Executor;
    ///
    /// // Automatically select best backend
    /// let exec = Executor::new_auto()?;
    /// # Ok::<(), hologram_core::Error>(())
    /// ```
    #[tracing::instrument]
    pub fn new_auto() -> Result<Self> {
        // Try Metal first (Apple Silicon)
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            if let Ok(exec) = Self::new_with_backend(BackendType::Metal) {
                tracing::info!("Auto-selected Metal backend (Apple Silicon detected)");
                return Ok(exec);
            }
        }

        // Try CUDA (NVIDIA GPU)
        // Will succeed if CUDA feature is enabled and NVIDIA GPU is available
        if let Ok(exec) = Self::new_with_backend(BackendType::Cuda) {
            tracing::info!("Auto-selected CUDA backend (NVIDIA GPU detected)");
            return Ok(exec);
        }

        // Fallback to CPU (always available)
        tracing::info!("Auto-selected CPU backend (fallback)");
        Self::new_with_backend(BackendType::Cpu)
    }

    /// Get shared reference to backend (for read operations)
    pub fn backend(&self) -> Arc<RwLock<BoxedBackend>> {
        Arc::clone(&self.backend)
    }

    /// Allocate a class-free buffer (bypasses the 96-class system)
    ///
    /// This method always uses class-free allocation regardless of buffer size.
    /// Useful for model weights and other long-lived allocations that would
    /// fragment the class system.
    ///
    /// # Type Requirements
    ///
    /// `T` must implement `bytemuck::Pod` for safe zero-copy semantics.
    ///
    /// # Example
    ///
    /// ```text
    /// let weights: Buffer<f32> = exec.allocate_class_free(50000)?;  // Always class-free
    /// ```
    #[tracing::instrument(skip(self), fields(
        len = len,
        elem_size = std::mem::size_of::<T>(),
        type_name = std::any::type_name::<T>()
    ))]
    pub fn allocate_class_free<T: bytemuck::Pod>(&mut self, len: usize) -> Result<Buffer<T>> {
        let start = Instant::now();
        let size_bytes = len * std::mem::size_of::<T>();

        // Allocate buffer via backend (no class allocation)
        let handle = write_lock(&self.backend)
            .allocate_buffer(size_bytes)
            .map_err(|e| Error::InvalidOperation(format!("Backend buffer allocation failed: {}", e)))?;

        // Generate unique buffer ID
        let buffer_id = {
            let mut id_gen = lock_mutex(&self.next_buffer_id);
            let id = *id_gen;
            *id_gen += 1;
            id
        };

        // Store handle in class-free buffers map
        lock_mutex(&self.class_free_buffers).insert(buffer_id, handle);

        // Get cached pointer for zero-overhead SIMD operations (CPU backend only)
        let cached_ptr = {
            let mut backend_guard = write_lock(&self.backend);
            let backend_any = backend_guard.as_any_mut();
            if let Some(cpu_backend) = backend_any.downcast_mut::<hologram_backends::CpuBackend>() {
                cpu_backend.get_buffer_mut_ptr(handle).unwrap_or(std::ptr::null_mut())
            } else {
                std::ptr::null_mut() // Non-CPU backend: no cached pointer
            }
        };

        let duration_us = start.elapsed().as_micros() as u64;
        tracing::debug!(
            duration_us = duration_us,
            buffer_id = buffer_id,
            size_bytes = size_bytes,
            size_kb = size_bytes as f64 / 1024.0,
            size_mb = size_bytes as f64 / (1024.0 * 1024.0),
            pool = "Linear (class-free)",
            "class_free_buffer_allocated"
        );

        // Return class-free buffer (with cached handle + pointer for zero-overhead ops)
        Ok(Buffer::new_class_free(
            handle,
            cached_ptr,
            buffer_id,
            len,
            MemoryPool::Linear,
            Arc::clone(&self.class_allocator),
            Arc::clone(&self.class_free_buffers),
        ))
    }

    /// Allocate a linear buffer of `len` elements
    ///
    /// Allocates a buffer managed by the backend. For large buffers that exceed
    /// a single class capacity (12,288 bytes = 3,072 f32 elements), this automatically
    /// allocates multiple consecutive classes.
    ///
    /// # Type Requirements
    ///
    /// `T` must implement `bytemuck::Pod` for safe zero-copy semantics.
    ///
    /// # Example
    ///
    /// ```text
    /// let buf: Buffer<f32> = exec.allocate(3072)?;   // One class worth of f32s
    /// let large_buf: Buffer<f32> = exec.allocate(10000)?;  // Multi-class buffer (4 classes)
    /// ```
    #[tracing::instrument(skip(self), fields(
        len = len,
        elem_size = std::mem::size_of::<T>(),
        type_name = std::any::type_name::<T>()
    ))]
    pub fn allocate<T: bytemuck::Pod>(&mut self, len: usize) -> Result<Buffer<T>> {
        let start = Instant::now();

        const CLASS_SIZE_BYTES: usize = 12_288; // Size of one class in bytes
        const MAX_CLASS_SYSTEM_BYTES: usize = 96 * CLASS_SIZE_BYTES; // 1,179,648 bytes (~1.125 MB)
        const MAX_CONSECUTIVE_CLASSES: usize = 48; // Max consecutive classes before using class-free (50% of 96)

        let size_bytes = len * std::mem::size_of::<T>();

        // Calculate number of classes needed
        let num_classes = size_bytes.div_ceil(CLASS_SIZE_BYTES);

        // For large buffers that need many consecutive classes or exceed total capacity,
        // use class-free allocation to prevent fragmentation. Buffers needing >48 consecutive
        // classes (>50% of total) are likely to cause allocation failures in fragmented memory.
        if num_classes > 96 || size_bytes > MAX_CLASS_SYSTEM_BYTES || num_classes > MAX_CONSECUTIVE_CLASSES {
            tracing::debug!(
                size_bytes = size_bytes,
                size_mb = size_bytes as f64 / (1024.0 * 1024.0),
                num_classes_needed = num_classes,
                "Allocating class-free buffer (exceeds 96-class system capacity)"
            );

            // Allocate buffer via backend (no class allocation)
            let handle = write_lock(&self.backend)
                .allocate_buffer(size_bytes)
                .map_err(|e| Error::InvalidOperation(format!("Backend buffer allocation failed: {}", e)))?;

            // Generate unique buffer ID
            let buffer_id = {
                let mut id_gen = lock_mutex(&self.next_buffer_id);
                let id = *id_gen;
                *id_gen += 1;
                id
            };

            // Store handle in class-free buffers map
            lock_mutex(&self.class_free_buffers).insert(buffer_id, handle);

            // Get cached pointer for zero-overhead SIMD operations (CPU backend only)
            let cached_ptr = {
                let mut backend_guard = write_lock(&self.backend);
                let backend_any = backend_guard.as_any_mut();
                if let Some(cpu_backend) = backend_any.downcast_mut::<hologram_backends::CpuBackend>() {
                    cpu_backend.get_buffer_mut_ptr(handle).unwrap_or(std::ptr::null_mut())
                } else {
                    std::ptr::null_mut()
                }
            };

            let duration_us = start.elapsed().as_micros() as u64;
            tracing::debug!(
                duration_us = duration_us,
                buffer_id = buffer_id,
                size_bytes = size_bytes,
                size_kb = size_bytes as f64 / 1024.0,
                size_mb = size_bytes as f64 / (1024.0 * 1024.0),
                pool = "Linear (class-free)",
                "class_free_buffer_allocated"
            );

            // Return class-free buffer (with cached handle + pointer)
            return Ok(Buffer::new_class_free(
                handle,
                cached_ptr,
                buffer_id,
                len,
                MemoryPool::Linear,
                Arc::clone(&self.class_allocator),
                Arc::clone(&self.class_free_buffers),
            ));
        }

        // Standard allocation using the class system (for buffers ≤ 1.125 MB)
        let (start_class, num_classes_u8) = if num_classes == 1 {
            // Single class allocation
            let class = lock_mutex(&self.class_allocator).allocate().ok_or_else(|| {
                Error::InvalidOperation("No more classes available for allocation (all 96 classes used)".into())
            })?;
            (class, 1u8)
        } else {
            // Multi-class allocation
            let start_class = lock_mutex(&self.class_allocator)
                .allocate_consecutive(num_classes)
                .ok_or_else(|| {
                    Error::InvalidOperation(format!(
                        "No {} consecutive classes available for allocation (buffer requires {} bytes across {} classes)",
                        num_classes, size_bytes, num_classes
                    ))
                })?;
            (start_class, num_classes as u8)
        };

        // Allocate buffer via backend
        let handle = write_lock(&self.backend)
            .allocate_buffer(size_bytes)
            .map_err(|e| Error::InvalidOperation(format!("Backend buffer allocation failed: {}", e)))?;

        // Map all classes to backend handle
        // For multi-class buffers, all classes point to the same handle
        // The backend manages the single contiguous allocation
        for i in 0..num_classes_u8 {
            self.buffer_mappings[(start_class + i) as usize] = Some(handle);
        }

        // Get cached pointer for zero-overhead SIMD operations (CPU backend only)
        let cached_ptr = {
            let mut backend_guard = write_lock(&self.backend);
            let backend_any = backend_guard.as_any_mut();
            if let Some(cpu_backend) = backend_any.downcast_mut::<hologram_backends::CpuBackend>() {
                cpu_backend.get_buffer_mut_ptr(handle).unwrap_or(std::ptr::null_mut())
            } else {
                std::ptr::null_mut()
            }
        };

        let duration_us = start.elapsed().as_micros() as u64;
        tracing::debug!(
            duration_us = duration_us,
            start_class = start_class,
            num_classes = num_classes_u8,
            size_bytes = size_bytes,
            size_kb = size_bytes as f64 / 1024.0,
            size_mb = size_bytes as f64 / (1024.0 * 1024.0),
            pool = "Linear",
            "buffer_allocated"
        );

        if num_classes_u8 == 1 {
            Ok(Buffer::new(
                handle,
                cached_ptr,
                start_class,
                len,
                MemoryPool::Linear,
                Arc::clone(&self.class_allocator),
            ))
        } else {
            Ok(Buffer::new_multi_class(
                handle,
                cached_ptr,
                start_class,
                num_classes_u8,
                len,
                MemoryPool::Linear,
                Arc::clone(&self.class_allocator),
            ))
        }
    }

    /// Allocate a boundary-addressed buffer
    ///
    /// This creates a buffer that directly maps to a specific class
    /// in the 96-class system.
    ///
    /// # Arguments
    ///
    /// * `class` - Class index [0, 96)
    /// * `width` - Width in pages [0, 48) (for compatibility)
    /// * `height` - Height in bytes per page [0, 256) (for compatibility)
    ///
    /// # Example
    ///
    /// ```text
    /// let buf: Buffer<f32> = exec.allocate_boundary(0, 48, 256)?;
    /// ```
    #[tracing::instrument(skip(self), fields(
        class = class,
        width = width,
        height = height,
        elem_size = std::mem::size_of::<T>(),
        type_name = std::any::type_name::<T>()
    ))]
    pub fn allocate_boundary<T: bytemuck::Pod>(&mut self, class: u8, width: usize, height: usize) -> Result<Buffer<T>> {
        let start = Instant::now();

        // Validate class
        if class >= 96 {
            return Err(Error::InvalidOperation(format!("class {} >= 96", class)));
        }
        if width > 48 {
            return Err(Error::InvalidOperation(format!("width {} > 48", width)));
        }
        if height > 256 {
            return Err(Error::InvalidOperation(format!("height {} > 256", height)));
        }

        // Boundary buffers: width × height bytes per class
        let size_bytes = width * height; // Typically: 48 × 256 = 12,288 bytes
        let len = size_bytes / std::mem::size_of::<T>(); // e.g., 3,072 for f32

        // Mark this class as using boundary pool (PoolHandle(0))
        // No BufferHandle allocation needed - boundary pool uses PoolHandle(0) directly
        // The backend will lazily initialize the actual boundary pool on first access
        self.is_boundary_pool[class as usize] = true;
        self.buffer_mappings[class as usize] = None; // No BufferHandle for boundary pool classes

        let duration_us = start.elapsed().as_micros() as u64;
        tracing::debug!(
            duration_us = duration_us,
            class = class,
            size_bytes = size_bytes,
            size_kb = size_bytes as f64 / 1024.0,
            pool = "Boundary",
            "boundary_buffer_allocated"
        );

        Ok(Buffer::new(
            BufferHandle(0),      // Sentinel for boundary buffers (use pool directly)
            std::ptr::null_mut(), // Boundary buffers don't have cached pointers
            class,
            len,
            MemoryPool::Boundary,
            Arc::clone(&self.class_allocator),
        ))
    }

    /// Write data to a buffer (zero-copy via backend)
    ///
    /// # Arguments
    ///
    /// * `class` - The class index [0, 96)
    /// * `data` - Slice of data to write
    ///
    /// # Zero-Copy Design
    ///
    /// Data is written directly to backend-managed memory without intermediate copies.
    pub(crate) fn write_buffer_data<T: bytemuck::Pod>(&mut self, class: u8, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);

        // Check if this class uses boundary pool (PoolHandle(0))
        if self.is_boundary_pool[class as usize] {
            use hologram_backends::backend::PoolHandle;

            // Calculate offset for this class (class * 12,288 bytes)
            const BYTES_PER_CLASS: usize = 12_288;
            let offset = class as usize * BYTES_PER_CLASS;

            // Write to pool at offset (uses efficient pool storage)
            write_lock(&self.backend)
                .copy_to_pool(PoolHandle::new(0), offset, bytes)
                .map_err(|e| Error::InvalidOperation(format!("Backend write failed: {}", e)))?;
        } else {
            // Regular buffer write
            let handle = self.buffer_mappings[class as usize]
                .ok_or_else(|| Error::InvalidOperation(format!("No buffer mapped to class {}", class)))?;

            write_lock(&self.backend)
                .copy_to_buffer(handle, bytes)
                .map_err(|e| Error::InvalidOperation(format!("Backend write failed: {}", e)))?;
        }

        Ok(())
    }

    /// Read data from a buffer (zero-copy via backend)
    ///
    /// # Arguments
    ///
    /// * `class` - The class index [0, 96)
    /// * `len` - Number of elements to read
    ///
    /// # Zero-Copy Design
    ///
    /// Data is read directly from backend-managed memory without intermediate copies.
    ///
    /// # Note
    ///
    /// On WASM/WebGPU, use async read methods instead
    #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
    pub(crate) fn read_buffer_data<T: bytemuck::Pod>(&self, class: u8, len: usize) -> Result<Vec<T>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<T>()];

        // Check if this class uses boundary pool (PoolHandle(0))
        if self.is_boundary_pool[class as usize] {
            use hologram_backends::backend::PoolHandle;

            // Calculate offset for this class (class * 12,288 bytes)
            const BYTES_PER_CLASS: usize = 12_288;
            let offset = class as usize * BYTES_PER_CLASS;

            // Read from pool at offset (uses efficient pool storage)
            write_lock(&self.backend)
                .copy_from_pool(PoolHandle::new(0), offset, &mut bytes)
                .map_err(|e| Error::InvalidOperation(format!("Backend read failed: {}", e)))?;
        } else {
            // Regular buffer read
            let handle = self.buffer_mappings[class as usize]
                .ok_or_else(|| Error::InvalidOperation(format!("No buffer mapped to class {}", class)))?;

            write_lock(&self.backend)
                .copy_from_buffer(handle, &mut bytes)
                .map_err(|e| Error::InvalidOperation(format!("Backend read failed: {}", e)))?;
        }

        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Async write data to a buffer (zero-copy via backend) - WASM/WebGPU only
    ///
    /// This properly yields to the browser event loop, ensuring writes complete
    /// before subsequent operations.
    ///
    /// # Arguments
    ///
    /// * `class` - The class index [0, 96)
    /// * `data` - Slice of data to write
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    pub(crate) async fn write_buffer_data_async<T: bytemuck::Pod>(&mut self, class: u8, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);

        // Check if this class uses boundary pool (PoolHandle(0))
        if self.is_boundary_pool[class as usize] {
            use hologram_backends::backend::PoolHandle;

            // Calculate offset for this class (class * 12,288 bytes)
            const BYTES_PER_CLASS: usize = 12_288;
            let offset = class as usize * BYTES_PER_CLASS;

            // Write to pool at offset (uses async method for proper synchronization)
            let mut backend = write_lock(&self.backend);
            let webgpu_backend = backend
                .as_any_mut()
                .downcast_mut::<hologram_backends::WebGpuBackend>()
                .ok_or_else(|| Error::InvalidOperation("Expected WebGpuBackend on WASM".to_string()))?;

            webgpu_backend
                .copy_to_pool_async_impl(PoolHandle::new(0), offset, bytes)
                .await
                .map_err(|e| Error::InvalidOperation(format!("Backend pool write failed: {}", e)))?;
        } else {
            // Regular buffer write - use async method on WebGPU backend
            let handle = self.buffer_mappings[class as usize]
                .ok_or_else(|| Error::InvalidOperation(format!("No buffer mapped to class {}", class)))?;

            // Downcast to WebGpuBackend to call async impl method
            let mut backend = write_lock(&self.backend);
            let webgpu_backend = backend
                .as_any_mut()
                .downcast_mut::<hologram_backends::WebGpuBackend>()
                .ok_or_else(|| Error::InvalidOperation("Expected WebGpuBackend on WASM".to_string()))?;

            webgpu_backend
                .copy_to_buffer_async_impl(handle, bytes)
                .await
                .map_err(|e| Error::InvalidOperation(format!("Backend write failed: {}", e)))?;
        }

        Ok(())
    }

    /// Write data to a class-free buffer (for large allocations > 1.125 MB)
    ///
    /// # Arguments
    ///
    /// * `buffer_id` - The unique buffer ID
    /// * `data` - Slice of data to write
    pub(crate) fn write_class_free_buffer_data<T: bytemuck::Pod>(&mut self, buffer_id: u64, data: &[T]) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);

        // Get handle from class-free buffers map
        let handle = lock_mutex(&self.class_free_buffers)
            .get(&buffer_id)
            .copied()
            .ok_or_else(|| Error::InvalidOperation(format!("No buffer found for ID {}", buffer_id)))?;

        write_lock(&self.backend)
            .copy_to_buffer(handle, bytes)
            .map_err(|e| Error::InvalidOperation(format!("Backend write failed: {}", e)))?;

        Ok(())
    }

    /// Async write data to a class-free buffer - WASM/WebGPU only
    ///
    /// This properly yields to the browser event loop, ensuring writes complete
    /// before subsequent operations.
    ///
    /// # Arguments
    ///
    /// * `buffer_id` - The unique buffer ID
    /// * `data` - Slice of data to write
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    pub(crate) async fn write_class_free_buffer_data_async<T: bytemuck::Pod>(
        &mut self,
        buffer_id: u64,
        data: &[T],
    ) -> Result<()> {
        let bytes = bytemuck::cast_slice(data);

        // Get handle from class-free buffers map
        let handle = lock_mutex(&self.class_free_buffers)
            .get(&buffer_id)
            .copied()
            .ok_or_else(|| Error::InvalidOperation(format!("No buffer found for ID {}", buffer_id)))?;

        // Downcast to WebGpuBackend to call async impl method
        let mut backend = write_lock(&self.backend);
        let webgpu_backend = backend
            .as_any_mut()
            .downcast_mut::<hologram_backends::WebGpuBackend>()
            .ok_or_else(|| Error::InvalidOperation("Expected WebGpuBackend on WASM".to_string()))?;

        webgpu_backend
            .copy_to_buffer_async_impl(handle, bytes)
            .await
            .map_err(|e| Error::InvalidOperation(format!("Backend write failed: {}", e)))?;

        Ok(())
    }

    /// Read data from a class-free buffer (for large allocations > 1.125 MB)
    ///
    /// # Arguments
    ///
    /// * `buffer_id` - The unique buffer ID
    /// * `len` - Number of elements to read
    ///
    /// # Note
    ///
    /// On WASM/WebGPU, use `read_class_free_buffer_data_async` instead
    #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
    pub(crate) fn read_class_free_buffer_data<T: bytemuck::Pod>(&self, buffer_id: u64, len: usize) -> Result<Vec<T>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<T>()];

        // Get handle from class-free buffers map
        let handle = lock_mutex(&self.class_free_buffers)
            .get(&buffer_id)
            .copied()
            .ok_or_else(|| Error::InvalidOperation(format!("No buffer found for ID {}", buffer_id)))?;

        write_lock(&self.backend)
            .copy_from_buffer(handle, &mut bytes)
            .map_err(|e| Error::InvalidOperation(format!("Backend read failed: {}", e)))?;

        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Read data from a class-free buffer asynchronously (for WebGPU)
    ///
    /// # Arguments
    ///
    /// * `buffer_id` - The unique buffer ID
    /// * `len` - Number of elements to read
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    pub(crate) async fn read_class_free_buffer_data_async<T: bytemuck::Pod>(
        &self,
        buffer_id: u64,
        len: usize,
    ) -> Result<Vec<T>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<T>()];

        // Get handle from class-free buffers map
        let handle = lock_mutex(&self.class_free_buffers)
            .get(&buffer_id)
            .copied()
            .ok_or_else(|| Error::InvalidOperation(format!("No buffer found for ID {}", buffer_id)))?;

        // Get cloneable Arc references from backend (brief lock)
        let (device, queue, buffer_pool, buffers, _pools) = {
            let backend_arc = self.backend();
            let backend = backend_arc.read();
            let webgpu_backend = backend
                .as_any()
                .downcast_ref::<hologram_backends::WebGpuBackend>()
                .ok_or_else(|| Error::InvalidOperation("Async read requires WebGPU backend".into()))?;

            webgpu_backend.get_async_resources()
        }; // Lock dropped here

        // Use standalone async buffer read - no locks held during await
        hologram_backends::WebGpuBackend::copy_buffer_async_standalone(
            device,
            queue,
            buffer_pool,
            buffers,
            handle,
            &mut bytes,
        )
        .await
        .map_err(|e| Error::Backend(e))?;

        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Read data from a buffer asynchronously (for WebGPU)
    ///
    /// This async version properly yields to the browser event loop during GPU→CPU transfers.
    ///
    /// # Arguments
    ///
    /// * `class` - The class index [0, 96)
    /// * `len` - Number of elements to read
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    pub(crate) async fn read_buffer_data_async<T: bytemuck::Pod>(&self, class: u8, len: usize) -> Result<Vec<T>> {
        let mut bytes = vec![0u8; len * std::mem::size_of::<T>()];

        // Check if this class uses boundary pool (PoolHandle(0))
        if self.is_boundary_pool[class as usize] {
            use hologram_backends::backend::PoolHandle;

            // Calculate offset for this class
            const BYTES_PER_CLASS: usize = 12_288;
            let offset = class as usize * BYTES_PER_CLASS;

            // Get cloneable Arc references from backend (brief lock)
            let (device, queue, buffer_pool, _buffers, pools) = {
                let backend_arc = self.backend();
                let backend = backend_arc.read();
                let webgpu_backend = backend
                    .as_any()
                    .downcast_ref::<hologram_backends::WebGpuBackend>()
                    .ok_or_else(|| Error::InvalidOperation("Async read requires WebGPU backend".into()))?;

                webgpu_backend.get_async_resources()
            }; // Lock dropped here

            // Use standalone async pool read - no locks held during await
            hologram_backends::WebGpuBackend::copy_pool_async_standalone(
                device,
                queue,
                buffer_pool,
                pools,
                PoolHandle::new(0),
                offset,
                &mut bytes,
            )
            .await
            .map_err(|e| Error::Backend(e))?;
        } else {
            // Regular buffer read - get the correct handle
            let handle = self.buffer_mappings[class as usize]
                .ok_or_else(|| Error::InvalidOperation(format!("No buffer mapped to class {}", class)))?;

            // Get cloneable Arc references from backend (brief lock)
            let (device, queue, buffer_pool, buffers, _pools) = {
                let backend_arc = self.backend();
                let backend = backend_arc.read();
                let webgpu_backend = backend
                    .as_any()
                    .downcast_ref::<hologram_backends::WebGpuBackend>()
                    .ok_or_else(|| Error::InvalidOperation("Async read requires WebGPU backend".into()))?;

                webgpu_backend.get_async_resources()
            }; // Lock dropped here

            // Use standalone async buffer read - no locks held during await
            hologram_backends::WebGpuBackend::copy_buffer_async_standalone(
                device,
                queue,
                buffer_pool,
                buffers,
                handle,
                &mut bytes,
            )
            .await
            .map_err(|e| Error::Backend(e))?;
        }

        Ok(bytemuck::cast_slice(&bytes).to_vec())
    }

    /// Get buffer handle from a Buffer (handles both class-based and class-free)
    ///
    /// This is the recommended method for operations to use.
    pub(crate) fn handle_from_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &crate::buffer::Buffer<T>,
    ) -> Result<BufferHandle> {
        // Zero-overhead: handle is cached in buffer struct (no mutex lookups!)
        Ok(buffer.handle())
    }

    /// Get buffer handle for a class (used by operations)
    pub(crate) fn get_buffer_handle(&self, class: u8) -> Result<BufferHandle> {
        // Validate class index range
        if class as usize >= 96 {
            return Err(Error::InvalidOperation(format!(
                "Invalid class index {} (must be < 96)",
                class
            )));
        }

        // Check if this is a boundary pool buffer
        if self.is_boundary_pool[class as usize] {
            // Encode class index in buffer handle ID for boundary pool buffers
            // Use high range (u64::MAX - 95 to u64::MAX) to avoid conflicts with regular buffers
            // Handle ID = BOUNDARY_POOL_HANDLE_BASE + class
            // The backend will recognize these and compute offset = class * 12,288 + element_offset
            // See load_bytes_from_storage() and store_bytes_to_storage() in hologram-backends
            const BOUNDARY_POOL_HANDLE_BASE: u64 = u64::MAX - 95;
            return Ok(BufferHandle::new(BOUNDARY_POOL_HANDLE_BASE + class as u64));
        }

        // Regular buffer - look up handle in mappings
        self.buffer_mappings[class as usize]
            .ok_or_else(|| Error::InvalidOperation(format!("No buffer mapped to class {}", class)))
    }

    /// Get buffer handle ID for ISA program construction
    ///
    /// This is used when building ISA programs that need to reference buffers
    /// via handle IDs (e.g., GEMM, Conv2d).
    ///
    /// # Arguments
    ///
    /// * `buffer` - Buffer to get handle ID for
    ///
    /// # Returns
    ///
    /// Returns the backend BufferHandle ID as u64
    #[allow(dead_code)]
    pub(crate) fn get_buffer_handle_id<T: bytemuck::Pod>(&self, buffer: &Buffer<T>) -> Result<u64> {
        // For class-free buffers, return the buffer ID directly
        if buffer.buffer_id() != 0 {
            let class_free_buffers = lock_mutex(&self.class_free_buffers);
            let handle = class_free_buffers.get(&buffer.buffer_id()).ok_or_else(|| {
                Error::InvalidOperation(format!("Class-free buffer {} not found", buffer.buffer_id()))
            })?;
            return Ok(handle.id());
        }

        // For regular class-based buffers, get handle from mappings
        let handle = self.get_buffer_handle(buffer.class_index())?;
        Ok(handle.id())
    }

    /// Get raw const pointer to buffer memory (for inline SIMD kernels)
    ///
    /// # Safety
    ///
    /// This is intended ONLY for inline SIMD kernel fast paths.
    /// The returned pointer is valid as long as:
    /// - The buffer is valid
    /// - No mutable operations occur on the buffer
    /// - The executor/backend is not dropped
    ///
    /// # Returns
    ///
    /// Returns `Err` if the backend is not CPU or the buffer is invalid.
    pub fn get_buffer_ptr<T: bytemuck::Pod>(&self, buffer: &Buffer<T>) -> Result<*const T> {
        // CPU backend only - get raw pointer from MemoryManager
        let backend = read_lock(&self.backend);
        let cpu_backend = backend
            .as_any()
            .downcast_ref::<CpuBackend>()
            .ok_or_else(|| Error::InvalidOperation("Inline kernels only supported on CPU backend".into()))?;

        let byte_ptr = if self.is_boundary_pool[buffer.class_index() as usize] {
            // Boundary pool buffer - get pointer to class data
            cpu_backend
                .get_boundary_class_ptr(buffer.class_index())
                .map_err(|e| Error::InvalidOperation(format!("Failed to get boundary class pointer: {}", e)))?
        } else {
            // Regular buffer - get pointer via buffer handle
            let handle = self.get_buffer_handle(buffer.class_index())?;
            cpu_backend
                .get_buffer_ptr(handle)
                .map_err(|e| Error::InvalidOperation(format!("Failed to get buffer pointer: {}", e)))?
        };

        // Cast u8 pointer to T pointer (bytemuck::Pod ensures this is safe)
        Ok(byte_ptr as *const T)
    }

    /// Get raw mutable pointer to buffer memory (for inline SIMD kernels)
    ///
    /// # Safety
    ///
    /// This is intended ONLY for inline SIMD kernel fast paths.
    /// The returned pointer is valid as long as:
    /// - The buffer is valid
    /// - No concurrent access occurs
    /// - The executor/backend is not dropped
    ///
    /// # Returns
    ///
    /// Returns `Err` if the backend is not CPU or the buffer is invalid.
    pub fn get_buffer_mut_ptr<T: bytemuck::Pod>(&mut self, buffer: &Buffer<T>) -> Result<*mut T> {
        // CPU backend only - get raw pointer from MemoryManager
        let mut backend = write_lock(&self.backend);
        let cpu_backend = backend
            .as_any_mut()
            .downcast_mut::<CpuBackend>()
            .ok_or_else(|| Error::InvalidOperation("Inline kernels only supported on CPU backend".into()))?;

        let byte_ptr = if self.is_boundary_pool[buffer.class_index() as usize] {
            // Boundary pool buffer - get mutable pointer to class data
            cpu_backend
                .get_boundary_class_mut_ptr(buffer.class_index())
                .map_err(|e| Error::InvalidOperation(format!("Failed to get boundary class mut pointer: {}", e)))?
        } else {
            // Regular buffer - get mutable pointer via buffer handle
            let handle = self.get_buffer_handle(buffer.class_index())?;
            cpu_backend
                .get_buffer_mut_ptr(handle)
                .map_err(|e| Error::InvalidOperation(format!("Failed to get buffer mut pointer: {}", e)))?
        };

        // Cast u8 pointer to T pointer (bytemuck::Pod ensures this is safe)
        Ok(byte_ptr as *mut T)
    }

    // ====================================================================================
    // DLPack Interoperability
    // ====================================================================================
    //
    // DLPack Protocol Support:
    // - Export (zero-copy): Hologram → PyTorch/JAX/TensorFlow/CuPy
    // - Import (copy-based): PyTorch/JAX/TensorFlow/CuPy → Hologram
    //
    // Note: Import is currently copy-based for safety and simplicity with Hologram's
    // class-based memory system. Export is zero-copy (the performance-critical path).

    /// Import tensor from DLPack format (universal framework support)
    ///
    /// Creates a Hologram tensor from any DLPack-compatible framework (PyTorch, JAX,
    /// TensorFlow, CuPy, etc.). The data is copied into Hologram's class-based memory.
    ///
    /// # Arguments
    ///
    /// * `dlpack_ptr` - Pointer to DLManagedTensor (from PyCapsule)
    ///
    /// # Returns
    ///
    /// Returns a Tensor<T> with data copied from the external framework
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Device type is unsupported
    /// - Data type doesn't match T
    /// - Memory allocation fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// // From PyTorch
    /// let pytorch_tensor = ...; // torch.randn(3, 4)
    /// let dlpack_capsule = pytorch_tensor.__dlpack__();
    /// let dlpack_ptr = extract_ptr_from_capsule(dlpack_capsule);
    ///
    /// let hologram_tensor: Tensor<f32> = exec.tensor_from_dlpack(dlpack_ptr)?;
    /// ```
    pub fn tensor_from_dlpack<T>(&mut self, dlpack_ptr: u64) -> Result<crate::tensor::Tensor<T>>
    where
        T: bytemuck::Pod + crate::interop::dlpack::DLPackType,
    {
        use crate::interop::dlpack::{DLDeviceType, DLManagedTensor};

        if dlpack_ptr == 0 {
            return Err(Error::InvalidOperation("Invalid DLPack pointer (null)".to_string()));
        }

        // Extract DLTensor from managed tensor
        let managed_tensor = unsafe { &*(dlpack_ptr as *const DLManagedTensor) };
        let dl_tensor = &managed_tensor.dl_tensor;

        // Validate device compatibility
        match dl_tensor.device.device_type {
            DLDeviceType::CPU | DLDeviceType::CUDA | DLDeviceType::CUDAHost => {
                // Supported devices
            }
            _ => {
                return Err(Error::InvalidOperation(format!(
                    "Unsupported DLPack device type: {:?}",
                    dl_tensor.device.device_type
                )));
            }
        }

        // Validate data type matches T
        let expected_dtype = T::dlpack_dtype();
        if dl_tensor.dtype.code != expected_dtype.code || dl_tensor.dtype.bits != expected_dtype.bits {
            return Err(Error::TypeMismatch {
                expected: format!("{:?} ({}bits)", expected_dtype.code, expected_dtype.bits),
                actual: format!("{:?} ({}bits)", dl_tensor.dtype.code, dl_tensor.dtype.bits),
            });
        }

        // Extract shape
        let shape: Vec<usize> = unsafe {
            std::slice::from_raw_parts(dl_tensor.shape, dl_tensor.ndim as usize)
                .iter()
                .map(|&s| s as usize)
                .collect()
        };

        // Calculate total elements
        let numel: usize = shape.iter().product();

        // Allocate Hologram buffer
        let mut buffer = self.allocate::<T>(numel)?;

        // Copy data from external tensor to Hologram buffer
        // Safety: We've validated the type matches and numel is correct
        unsafe {
            let src_ptr = (dl_tensor.data as *const u8).add(dl_tensor.byte_offset as usize) as *const T;
            let src_slice = std::slice::from_raw_parts(src_ptr, numel);

            // Use buffer.copy_from_slice() for safety
            buffer.copy_from_slice(self, src_slice)?;
        }

        // Create tensor from buffer
        let tensor = crate::tensor::Tensor::from_buffer(buffer, shape)?;

        // Note: We do NOT call the deleter here because:
        // 1. This is a copy-based import (we copied the data)
        // 2. The original tensor should remain valid in the external framework
        // 3. The deleter is for zero-copy scenarios where we're sharing memory

        Ok(tensor)
    }

    /// Export tensor to DLPack format for zero-copy sharing with PyTorch, JAX, etc.
    ///
    /// Creates a DLManagedTensor that shares memory with the Hologram tensor. No data is copied.
    /// The tensor must be contiguous for DLPack export.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to export (must be contiguous)
    ///
    /// # Returns
    ///
    /// Returns a boxed DLManagedTensor that can be passed to other frameworks via PyCapsule.
    /// The deleter function will be called when the consuming framework is done with the data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hologram_core::{Executor, Tensor};
    ///
    /// let mut exec = Executor::new()?;
    /// let buffer = exec.allocate::<f32>(12)?;
    /// let tensor = Tensor::from_buffer(buffer, vec![3, 4])?;
    ///
    /// // Export to DLPack (zero-copy)
    /// let dlpack = exec.tensor_to_dlpack(&tensor)?;
    /// // dlpack can be passed to PyTorch, JAX, etc. via PyCapsule
    /// ```
    pub fn tensor_to_dlpack<T>(
        &self,
        tensor: &crate::tensor::Tensor<T>,
    ) -> Result<Box<crate::interop::dlpack::DLManagedTensor>>
    where
        T: bytemuck::Pod + crate::interop::dlpack::DLPackType,
    {
        use crate::interop::dlpack::{DLManagedTensor, DLTensor};

        // Tensor must be contiguous for DLPack export
        if !tensor.is_contiguous() {
            return Err(Error::InvalidOperation(
                "DLPack export requires contiguous tensor. Call .contiguous() first.".to_string(),
            ));
        }

        // Get buffer pointer
        let buffer = tensor.buffer();
        let data_ptr = self.get_buffer_data_ptr(buffer)?;

        // Get device type
        let device = self.get_device_type();

        // Create shape and strides arrays (need to be heap-allocated for DLPack)
        let shape_vec: Vec<i64> = tensor.shape().iter().map(|&s| s as i64).collect();
        let strides_vec: Vec<i64> = tensor.strides().iter().map(|&s| s as i64).collect();

        let shape_ptr = Box::into_raw(shape_vec.into_boxed_slice()) as *mut i64;
        let strides_ptr = Box::into_raw(strides_vec.into_boxed_slice()) as *mut i64;

        // Create DLTensor
        let dl_tensor = DLTensor {
            data: data_ptr,
            device,
            ndim: tensor.ndim() as i32,
            dtype: T::dlpack_dtype(),
            shape: shape_ptr,
            strides: strides_ptr,
            byte_offset: tensor.offset() as u64 * std::mem::size_of::<T>() as u64,
        };

        // Create manager context
        // For now, we don't need special cleanup beyond freeing the shape/strides arrays
        let manager_ctx = std::ptr::null_mut();

        // Create managed tensor with deleter
        let managed = unsafe { DLManagedTensor::new(dl_tensor, manager_ctx, hologram_dlpack_deleter) };

        Ok(managed)
    }

    /// Get raw pointer to buffer data for DLPack export
    ///
    /// # Safety
    ///
    /// The returned pointer is only valid while the buffer exists.
    /// For CPU backend, returns host pointer. For CUDA backend, returns device pointer.
    fn get_buffer_data_ptr<T: bytemuck::Pod>(
        &self,
        buffer: &crate::buffer::Buffer<T>,
    ) -> Result<*mut std::ffi::c_void> {
        let class_index = buffer.class_index();
        let backend_guard = read_lock(&self.backend);
        let backend_any = &**backend_guard;

        // Try to downcast to CPU backend first
        if let Some(cpu_backend) = backend_any.as_any().downcast_ref::<CpuBackend>() {
            let byte_ptr = if self.is_boundary_pool[class_index as usize] {
                cpu_backend
                    .get_boundary_class_ptr(class_index)
                    .map_err(|e| Error::InvalidOperation(format!("Failed to get boundary class pointer: {}", e)))?
            } else {
                let handle = self.get_buffer_handle(class_index)?;
                cpu_backend
                    .get_buffer_ptr(handle)
                    .map_err(|e| Error::InvalidOperation(format!("Failed to get buffer pointer: {}", e)))?
            };
            return Ok(byte_ptr as *mut std::ffi::c_void);
        }

        // Try CUDA backend
        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_backend) = backend_any.as_any().downcast_ref::<CudaBackend>() {
                let handle = self.get_buffer_handle(class_index)?;
                let device_ptr = cuda_backend
                    .get_device_ptr(handle.id())
                    .map_err(|e| Error::InvalidOperation(format!("Failed to get CUDA device pointer: {}", e)))?;
                return Ok(device_ptr);
            }
        }

        Err(Error::InvalidOperation(
            "DLPack export not supported for this backend type".to_string(),
        ))
    }

    /// Determine the device type for DLPack export
    pub fn get_device_type(&self) -> crate::interop::dlpack::DLDevice {
        use crate::interop::dlpack::DLDevice;

        let backend_guard = read_lock(&self.backend);
        let backend_any = &**backend_guard;

        // Check backend type
        if backend_any.as_any().is::<CpuBackend>() {
            return DLDevice::cpu();
        }

        #[cfg(feature = "cuda")]
        {
            if backend_any.as_any().is::<CudaBackend>() {
                return DLDevice::cuda(0); // TODO: Support multi-GPU
            }
        }

        #[cfg(target_vendor = "apple")]
        {
            if backend_any.as_any().is::<MetalBackend>() {
                // DLPack doesn't have a standard Metal device type yet
                // Use CPU as fallback
                return DLDevice::cpu();
            }
        }

        // Default to CPU
        DLDevice::cpu()
    }

    /// Get default launch configuration for N elements
    ///
    /// Uses rayon-compatible parallelization strategy:
    /// - Grid: 1 block
    /// - Block: N lanes (threads)
    /// - Shared memory: default
    pub fn default_launch_config(n: usize) -> LaunchConfig {
        LaunchConfig {
            grid: GridDim { x: 1, y: 1, z: 1 },
            block: BlockDim {
                x: n as u32,
                y: 1,
                z: 1,
            },
            shared_memory: SharedMemoryConfig::default(),
        }
    }

    /// Execute an ISA program with buffer arguments
    ///
    /// This method provides a public interface for executing precompiled ISA programs
    /// from external crates (like hologram-onnx).
    ///
    /// # Arguments
    ///
    /// * `program` - The ISA program to execute
    /// * `input_buffers` - Input buffer handles (passed in registers R1, R2, R3, ...)
    /// * `output_buffer` - Output buffer (passed in register R3 for binary ops)
    /// * `n` - Number of elements to process
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use hologram_core::precompiled_programs::VECTOR_ADD;
    ///
    /// let mut exec = Executor::new()?;
    /// let a = exec.allocate::<f32>(1024)?;
    /// let b = exec.allocate::<f32>(1024)?;
    /// let mut c = exec.allocate::<f32>(1024)?;
    ///
    /// exec.execute_isa_program(&VECTOR_ADD, &[&a, &b], &c, 1024)?;
    /// ```
    pub fn execute_isa_program<T: bytemuck::Pod>(
        &self,
        program: &hologram_backends::Program,
        input_buffers: &[&Buffer<T>],
        output_buffer: &Buffer<T>,
        n: usize,
    ) -> Result<()> {
        use hologram_backends::{ExecutionParams, Register};
        use std::collections::HashMap;

        // Get buffer handles
        let mut handles = Vec::with_capacity(input_buffers.len() + 1);
        for buf in input_buffers {
            handles.push(self.handle_from_buffer(buf)?);
        }
        handles.push(self.handle_from_buffer(output_buffer)?);

        // Create execution params with buffer handles in registers
        let mut initial_registers = HashMap::new();
        for (i, handle) in handles.iter().enumerate() {
            initial_registers.insert(Register((i + 1) as u8), handle.id());
        }

        // Set element count in R4
        initial_registers.insert(Register(4), n as u64);

        let params = ExecutionParams {
            launch_config: Self::default_launch_config(n),
            initial_registers,
            gauge: None,
        };

        // Execute program
        let mut backend = write_lock(&self.backend);
        backend.execute_program_with_params(program, &params)?;

        Ok(())
    }
}

/// Deleter function for DLPack managed tensors
///
/// This is called by the consuming framework (PyTorch, JAX, etc.) when it's done with the tensor.
/// We need to free the shape and strides arrays that we allocated.
extern "C" fn hologram_dlpack_deleter(tensor: *mut crate::interop::dlpack::DLManagedTensor) {
    if tensor.is_null() {
        return;
    }

    unsafe {
        let managed = &mut *tensor;
        let dl_tensor = &managed.dl_tensor;

        // Free shape array
        if !dl_tensor.shape.is_null() {
            let shape_slice = std::slice::from_raw_parts_mut(dl_tensor.shape, dl_tensor.ndim as usize);
            let _ = Box::from_raw(shape_slice as *mut [i64]);
        }

        // Free strides array
        if !dl_tensor.strides.is_null() {
            let strides_slice = std::slice::from_raw_parts_mut(dl_tensor.strides, dl_tensor.ndim as usize);
            let _ = Box::from_raw(strides_slice as *mut [i64]);
        }

        // Free the managed tensor itself
        drop(Box::from_raw(tensor));
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new().expect("Failed to create default executor")
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        tracing::debug!("Executor dropped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_executor_creation() {
        let exec = Executor::new().unwrap();
        // Verify all 96 classes are initially available
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);
    }

    #[test]
    fn test_allocate_linear() {
        let mut exec = Executor::new().unwrap();
        let buf: Buffer<f32> = exec.allocate(1024).unwrap();
        assert_eq!(buf.len(), 1024);
        assert_eq!(buf.pool(), MemoryPool::Linear);
    }

    #[test]
    fn test_allocate_boundary() {
        let mut exec = Executor::new().unwrap();
        let buf: Buffer<f32> = exec.allocate_boundary(0, 48, 256).unwrap();
        // 48 × 256 bytes = 12,288 bytes / 4 bytes per f32 = 3,072 f32 elements
        assert_eq!(buf.len(), 3072);
        assert_eq!(buf.pool(), MemoryPool::Boundary);
    }

    #[test]
    fn test_allocate_boundary_invalid_class() {
        let mut exec = Executor::new().unwrap();
        let result: Result<Buffer<f32>> = exec.allocate_boundary(96, 48, 256);
        assert!(result.is_err());
    }

    #[test]
    fn test_buffer_read_write() {
        let mut exec = Executor::new().unwrap();
        let mut buf: Buffer<f32> = exec.allocate(10).unwrap();

        // Write data
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        buf.copy_from_slice(&mut exec, &data).unwrap();

        // Read data back
        let result = buf.to_vec(&exec).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut exec = Executor::new().unwrap();

        // Allocate multiple buffers
        let buf1: Buffer<f32> = exec.allocate(100).unwrap();
        let buf2: Buffer<f32> = exec.allocate(200).unwrap();
        let buf3: Buffer<f32> = exec.allocate(300).unwrap();

        // Each should get a different class
        assert_eq!(buf1.class(), 0);
        assert_eq!(buf2.class(), 1);
        assert_eq!(buf3.class(), 2);
    }

    #[test]
    fn test_launch_config_creation() {
        let config = Executor::default_launch_config(1024);
        assert_eq!(config.grid.x, 1);
        assert_eq!(config.block.x, 1024);
    }

    #[test]
    fn test_automatic_class_deallocation() {
        let mut exec = Executor::new().unwrap();

        // Initially all 96 classes are free
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);

        // Allocate 3 buffers in a scope
        {
            let _buf1: Buffer<f32> = exec.allocate(100).unwrap();
            let _buf2: Buffer<f32> = exec.allocate(200).unwrap();
            let _buf3: Buffer<f32> = exec.allocate(300).unwrap();

            // Now 3 classes should be allocated
            assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 93);
            assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 3);
        }
        // Buffers dropped here - classes should be automatically freed

        // After buffers are dropped, all 96 classes should be free again
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);

        // Verify we can allocate again
        let _buf4: Buffer<f32> = exec.allocate(400).unwrap();
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 1);
    }

    #[test]
    fn test_multi_class_buffer_allocation() {
        let mut exec = Executor::new().unwrap();

        // Each class holds 12,288 bytes = 3,072 f32 elements
        // Allocate a buffer that spans 4 classes (10,000 f32 elements)
        let buf: Buffer<f32> = exec.allocate(10000).unwrap();

        // Should use 4 classes: (10000 * 4 bytes) / 12288 bytes per class = 3.26 → 4 classes
        assert_eq!(buf.num_classes(), 4);
        assert!(buf.is_multi_class());
        assert_eq!(buf.len(), 10000);

        // 4 classes should be allocated
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 4);
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 92);

        // Verify end_class is correct
        assert_eq!(buf.end_class(), buf.class_index() + 3);
    }

    #[test]
    fn test_multi_class_buffer_deallocation() {
        let mut exec = Executor::new().unwrap();

        {
            // Allocate large buffer
            let _buf: Buffer<f32> = exec.allocate(20000).unwrap(); // ~6.5 classes → 7 classes
            assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 7);
        }
        // Buffer dropped - all 7 classes should be freed

        // All classes should be free again
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);
    }

    #[test]
    fn test_large_vae_sized_buffer() {
        let mut exec = Executor::new().unwrap();

        // VAE weights: ~9.34 MB = 9,340,000 bytes
        // At 12,288 bytes per class: 9,340,000 / 12,288 = 760.4 classes
        // This exceeds 96 classes, so it should work now with the refactored system

        // Let's test with a smaller but still large buffer: 50,000 f32 elements
        // 50,000 * 4 = 200,000 bytes / 12,288 = 16.3 → 17 classes
        let buf: Buffer<f32> = exec.allocate(50000).unwrap();

        assert_eq!(buf.num_classes(), 17);
        assert_eq!(buf.len(), 50000);
        assert!(buf.is_multi_class());
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 17);
    }

    #[test]
    fn test_single_class_buffer_still_works() {
        let mut exec = Executor::new().unwrap();

        // Small buffer that fits in one class
        let buf: Buffer<f32> = exec.allocate(1000).unwrap();

        assert_eq!(buf.num_classes(), 1);
        assert!(!buf.is_multi_class());
        assert_eq!(buf.class_index(), buf.end_class());
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 1);
    }

    #[test]
    fn test_class_free_buffer_allocation() {
        let mut exec = Executor::new().unwrap();

        // Allocate buffer larger than 96-class system capacity (~1.125 MB)
        // 10 MB buffer: 10,485,760 bytes / 4 bytes per f32 = 2,621,440 elements
        let large_size = 2_621_440; // 10 MB
        let buf: Buffer<f32> = exec.allocate(large_size).unwrap();

        // Should be class-free
        assert!(buf.is_class_free());
        assert_eq!(buf.buffer_id(), 1); // First buffer gets ID 1
        assert_eq!(buf.len(), large_size);
        assert!(!buf.is_multi_class());

        // Class system should remain untouched
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);

        // Verify buffer is tracked in class_free_buffers
        assert_eq!(lock_mutex(&exec.class_free_buffers).len(), 1);
    }

    #[test]
    fn test_class_free_buffer_read_write() {
        let mut exec = Executor::new().unwrap();

        // Allocate 5 MB buffer (VAE-sized)
        let size = 1_310_720; // 5 MB worth of f32 elements
        let mut buf: Buffer<f32> = exec.allocate(size).unwrap();

        assert!(buf.is_class_free());

        // Write test data
        let test_data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32).collect();
        buf.copy_from_slice(&mut exec, &test_data).unwrap();

        // Read it back
        let result = buf.to_vec(&exec).unwrap();

        // Verify data matches
        assert_eq!(result.len(), size);
        assert_eq!(result[0], 0.0);
        assert_eq!(result[999], 999.0);
        assert_eq!(result[1000], 0.0); // wraps around due to modulo
    }

    #[test]
    fn test_class_free_buffer_deallocation() {
        let mut exec = Executor::new().unwrap();

        {
            // Allocate large buffer
            let _buf: Buffer<f32> = exec.allocate(3_000_000).unwrap(); // > 1.125 MB
            assert_eq!(lock_mutex(&exec.class_free_buffers).len(), 1);
        }
        // Buffer dropped - should be removed from class_free_buffers

        // Verify buffer was deallocated
        assert_eq!(lock_mutex(&exec.class_free_buffers).len(), 0);
    }

    #[test]
    fn test_multiple_class_free_buffers() {
        let mut exec = Executor::new().unwrap();

        // Allocate multiple large buffers
        let buf1: Buffer<f32> = exec.allocate(2_000_000).unwrap();
        let buf2: Buffer<f32> = exec.allocate(3_000_000).unwrap();
        let buf3: Buffer<f32> = exec.allocate(4_000_000).unwrap();

        // All should be class-free
        assert!(buf1.is_class_free());
        assert!(buf2.is_class_free());
        assert!(buf3.is_class_free());

        // Each should have unique ID
        assert_eq!(buf1.buffer_id(), 1);
        assert_eq!(buf2.buffer_id(), 2);
        assert_eq!(buf3.buffer_id(), 3);

        // All should be tracked
        assert_eq!(lock_mutex(&exec.class_free_buffers).len(), 3);

        // Class system should remain untouched
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);
    }

    #[test]
    fn test_vae_sized_buffer_works() {
        let mut exec = Executor::new().unwrap();

        // VAE weights: ~9.34 MB = 9,340,000 bytes / 4 = 2,335,000 f32 elements
        let vae_size = 2_335_000;
        let buf: Buffer<f32> = exec.allocate(vae_size).unwrap();

        // Should succeed and be class-free
        assert!(buf.is_class_free());
        assert_eq!(buf.len(), vae_size);
    }

    #[test]
    fn test_allocate_class_free_method() {
        let mut exec = Executor::new().unwrap();

        // Small buffer, but force class-free allocation
        let buf: Buffer<f32> = exec.allocate_class_free(1000).unwrap();

        // Should be class-free even though it's small
        assert!(buf.is_class_free());
        assert_eq!(buf.len(), 1000);
        assert_eq!(buf.buffer_id(), 1);

        // Class system should remain untouched
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);
    }

    #[test]
    fn test_allocate_class_free_prevents_fragmentation() {
        let mut exec = Executor::new().unwrap();

        // Simulate loading multiple weight tensors using class-free allocation
        // This prevents fragmentation of the 96-class system
        let tensors: Vec<Buffer<f32>> = (0..20)
            .map(|_| exec.allocate_class_free(36_864).unwrap()) // 147,456 bytes each
            .collect();

        // All should be class-free
        for (i, tensor) in tensors.iter().enumerate() {
            assert!(tensor.is_class_free());
            assert_eq!(tensor.buffer_id(), (i + 1) as u64);
            assert_eq!(tensor.len(), 36_864);
        }

        // Class system should remain completely untouched
        assert_eq!(lock_mutex(&exec.class_allocator).allocated_count(), 0);
        assert_eq!(lock_mutex(&exec.class_allocator).free_count(), 96);

        // All 20 class-free buffers should be tracked
        assert_eq!(lock_mutex(&exec.class_free_buffers).len(), 20);
    }
}
