//! Compute pipeline cache for WebGPU shaders
//!
//! Manages compilation and caching of WGSL compute shaders to avoid
//! repeated compilation overhead.

use crate::sync::{read_lock, write_lock, RwLock};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{ComputePipeline, Device, ShaderModule};

/// Cache key for compute pipelines
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    /// Shader name/identifier
    pub shader_name: String,
    /// Entry point function name
    pub entry_point: String,
}

impl PipelineKey {
    /// Create a new pipeline key
    pub fn new(shader_name: impl Into<String>, entry_point: impl Into<String>) -> Self {
        Self {
            shader_name: shader_name.into(),
            entry_point: entry_point.into(),
        }
    }
}

/// Pipeline cache for compiled compute shaders
///
/// Stores compiled `ComputePipeline` objects to avoid recompilation overhead.
/// Thread-safe with interior mutability via `RwLock`.
///
/// # Example
///
/// ```rust,no_run
/// use hologram_backends::backends::wasm::webgpu::pipeline::PipelineCache;
/// use std::sync::Arc;
///
/// # async fn example(device: Arc<wgpu::Device>) -> Result<(), String> {
/// let mut cache = PipelineCache::new(device);
///
/// // Compile shader on first use
/// let pipeline = cache.get_or_create(
///     "vector_add",
///     include_str!("kernels/vector_add.wgsl"),
///     "main"
/// )?;
///
/// // Second call returns cached pipeline (no recompilation)
/// let same_pipeline = cache.get_or_create(
///     "vector_add",
///     include_str!("kernels/vector_add.wgsl"),
///     "main"
/// )?;
/// # Ok(())
/// # }
/// ```
pub struct PipelineCache {
    device: Arc<Device>,
    pipelines: RwLock<HashMap<PipelineKey, Arc<ComputePipeline>>>,
    shader_modules: RwLock<HashMap<String, Arc<ShaderModule>>>,
}

impl PipelineCache {
    /// Create a new pipeline cache
    ///
    /// # Arguments
    ///
    /// * `device` - WebGPU device for shader compilation
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            pipelines: RwLock::new(HashMap::new()),
            shader_modules: RwLock::new(HashMap::new()),
        }
    }

    /// Get or create a compute pipeline
    ///
    /// If the pipeline is already cached, returns the cached version.
    /// Otherwise, compiles the shader and caches the pipeline.
    ///
    /// # Arguments
    ///
    /// * `shader_name` - Identifier for the shader (for caching)
    /// * `shader_source` - WGSL shader source code
    /// * `entry_point` - Entry point function name (usually "main")
    ///
    /// # Returns
    ///
    /// Returns an `Arc` to the compiled compute pipeline
    ///
    /// # Errors
    ///
    /// Returns `Err` if shader compilation fails
    pub fn get_or_create(
        &self,
        shader_name: impl Into<String>,
        shader_source: &str,
        entry_point: impl Into<String>,
    ) -> Result<Arc<ComputePipeline>, String> {
        let shader_name = shader_name.into();
        let entry_point_str = entry_point.into();
        let key = PipelineKey::new(shader_name.clone(), entry_point_str.clone());

        // Fast path: check if pipeline already cached (read lock)
        {
            let pipelines = read_lock(&self.pipelines);
            if let Some(pipeline) = pipelines.get(&key) {
                return Ok(Arc::clone(pipeline));
            }
        }

        // Slow path: compile shader and create pipeline (write lock)
        let mut pipelines = write_lock(&self.pipelines);

        // Double-check after acquiring write lock (another thread may have compiled)
        if let Some(pipeline) = pipelines.get(&key) {
            return Ok(Arc::clone(pipeline));
        }

        // Compile shader module (or get from cache)
        let shader_module = self.get_or_create_shader_module(&shader_name, shader_source)?;

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}::{}", shader_name, entry_point_str)),
            layout: None, // Auto-layout from shader
            module: &shader_module,
            entry_point: Some(&entry_point_str),
            compilation_options: Default::default(),
            cache: None,
        });

        let pipeline_arc = Arc::new(pipeline);
        pipelines.insert(key, Arc::clone(&pipeline_arc));

        tracing::debug!(
            "Compiled and cached compute pipeline: {}::{}",
            shader_name,
            entry_point_str
        );

        Ok(pipeline_arc)
    }

    /// Get or create a shader module
    ///
    /// Internal method to compile and cache shader modules separately from pipelines.
    /// This allows reusing shader modules for different entry points.
    fn get_or_create_shader_module(&self, shader_name: &str, shader_source: &str) -> Result<Arc<ShaderModule>, String> {
        // Fast path: check cache (read lock)
        {
            let modules = read_lock(&self.shader_modules);
            if let Some(module) = modules.get(shader_name) {
                return Ok(Arc::clone(module));
            }
        }

        // Slow path: compile shader (write lock)
        let mut modules = write_lock(&self.shader_modules);

        // Double-check after acquiring write lock
        if let Some(module) = modules.get(shader_name) {
            return Ok(Arc::clone(module));
        }

        // Compile shader module
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(shader_name),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let module_arc = Arc::new(shader_module);
        modules.insert(shader_name.to_string(), Arc::clone(&module_arc));

        tracing::debug!("Compiled and cached shader module: {}", shader_name);

        Ok(module_arc)
    }

    /// Clear the pipeline cache
    ///
    /// Removes all cached pipelines and shader modules. Useful for
    /// freeing memory or forcing recompilation.
    pub fn clear(&self) {
        write_lock(&self.pipelines).clear();
        write_lock(&self.shader_modules).clear();
        tracing::debug!("Pipeline cache cleared");
    }

    /// Get cache statistics
    ///
    /// Returns the number of cached pipelines and shader modules
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            num_pipelines: read_lock(&self.pipelines).len(),
            num_shader_modules: read_lock(&self.shader_modules).len(),
        }
    }
}

/// Pipeline cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of cached compute pipelines
    pub num_pipelines: usize,
    /// Number of cached shader modules
    pub num_shader_modules: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PipelineCache: {} pipelines, {} shader modules",
            self.num_pipelines, self.num_shader_modules
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_key_equality() {
        let key1 = PipelineKey::new("vector_add", "main");
        let key2 = PipelineKey::new("vector_add", "main");
        let key3 = PipelineKey::new("vector_mul", "main");

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_cache_stats_display() {
        let stats = CacheStats {
            num_pipelines: 5,
            num_shader_modules: 3,
        };

        let display = format!("{}", stats);
        assert!(display.contains("5 pipelines"));
        assert!(display.contains("3 shader modules"));
    }
}
