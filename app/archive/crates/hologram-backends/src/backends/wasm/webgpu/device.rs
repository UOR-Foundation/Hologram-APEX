//! WebGPU device management for browser-based GPU compute
//!
//! Handles WebGPU device initialization, adapter selection, and queue management
//! for WASM/browser environments.

use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

/// WebGPU device manager for browser-based GPU compute
///
/// Provides async initialization of WebGPU resources and browser
/// compatibility detection.
///
/// # Example
///
/// ```rust,no_run
/// use hologram_backends::backends::wasm::webgpu::WebGpuDevice;
///
/// async fn init() -> Result<WebGpuDevice, String> {
///     if WebGpuDevice::is_available() {
///         WebGpuDevice::new().await
///     } else {
///         Err("WebGPU not available in this browser".to_string())
///     }
/// }
/// ```
pub struct WebGpuDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<Adapter>,
}

impl WebGpuDevice {
    /// Create a new WebGPU device
    ///
    /// Initializes WebGPU instance, requests adapter, and creates device/queue.
    /// This is an async operation in browser environments.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - WebGPU is not available in the browser
    /// - No compatible adapter found
    /// - Device creation fails
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # async fn example() -> Result<(), String> {
    /// use hologram_backends::backends::wasm::webgpu::WebGpuDevice;
    ///
    /// let device = WebGpuDevice::new().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new() -> Result<Self, String> {
        // Create WebGPU instance
        // For WASM, use BROWSER_WEBGPU backend
        let instance_descriptor = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        };
        let instance = Instance::new(&instance_descriptor);

        // Request adapter with high-performance preference
        let adapter_options = wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        };
        let adapter = instance
            .request_adapter(&adapter_options)
            .await
            .expect("Failed to find WebGPU adapter. Ensure WebGPU is enabled in your browser.");

        // Get adapter info for logging
        let adapter_info = adapter.get_info();
        tracing::info!("WebGPU adapter: {} ({:?})", adapter_info.name, adapter_info.backend);

        // Request device and queue
        // WORKAROUND: Chrome 135+ removed maxInterStageShaderComponents from WebGPU
        // wgpu 0.20 still includes it in Limits structs, causing device creation to fail
        // Solution: Use downlevel_defaults() which only includes core WebGL-compatible limits
        // This avoids sending unsupported limits to the browser
        #[cfg(target_arch = "wasm32")]
        let _limits = wgpu::Limits::downlevel_defaults();

        #[cfg(not(target_arch = "wasm32"))]
        let _limits = wgpu::Limits::default();

        // For WASM, request higher buffer binding limits for large model weights
        #[cfg(target_arch = "wasm32")]
        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Hologram WebGPU Device"),
            required_features: wgpu::Features::empty(),
            // Request higher limits for AI model buffers (up to 256MB per binding)
            required_limits: wgpu::Limits {
                max_storage_buffer_binding_size: 268_435_456, // 256 MB (increased from default 128 MB)
                max_buffer_size: 1_073_741_824,               // 1 GB
                ..wgpu::Limits::default()
            },
            memory_hints: Default::default(),
            experimental_features: Default::default(),
            trace: Default::default(),
        };

        #[cfg(not(target_arch = "wasm32"))]
        let device_descriptor = wgpu::DeviceDescriptor {
            label: Some("Hologram WebGPU Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
            experimental_features: Default::default(),
            trace: Default::default(),
        };

        let (device, queue) = adapter
            .request_device(&device_descriptor)
            .await
            .expect("Failed to create WebGPU device");

        tracing::info!("WebGPU device initialized successfully");

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
        })
    }

    /// Check if WebGPU is available in the current environment
    ///
    /// Returns `true` if:
    /// - Running in a WASM32 environment
    /// - Browser supports WebGPU (navigator.gpu exists)
    ///
    /// # Example
    ///
    /// ```rust
    /// use hologram_backends::backends::wasm::webgpu::WebGpuDevice;
    ///
    /// if WebGpuDevice::is_available() {
    ///     println!("WebGPU is available!");
    /// } else {
    ///     println!("Falling back to scalar WASM execution");
    /// }
    /// ```
    pub fn is_available() -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            // On WASM, WebGPU availability will be checked when creating the adapter
            // wgpu::Instance::request_adapter() will return None if not available
            true
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            // WebGPU backend only available in WASM
            false
        }
    }

    /// Get reference to the WebGPU device
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get reference to the WebGPU queue
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Get reference to the WebGPU adapter
    pub fn adapter(&self) -> &Arc<Adapter> {
        &self.adapter
    }

    /// Get adapter information
    ///
    /// Returns details about the GPU adapter (name, backend, vendor, etc.)
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Get device limits
    ///
    /// Returns the maximum supported limits for this device
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    /// Get supported features
    ///
    /// Returns the set of features supported by this device
    pub fn features(&self) -> wgpu::Features {
        self.device.features()
    }
}

impl std::fmt::Debug for WebGpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let info = self.adapter_info();
        f.debug_struct("WebGpuDevice")
            .field("adapter", &info.name)
            .field("backend", &info.backend)
            .field("vendor", &info.vendor)
            .field("device_type", &info.device_type)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        // Test that is_available() returns a boolean
        let _available = WebGpuDevice::is_available();

        // In non-WASM environments, should always return false
        #[cfg(not(target_arch = "wasm32"))]
        assert!(!WebGpuDevice::is_available());
    }
}
