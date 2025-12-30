//! Host-side runtime API for Atlas execution.
//!
//! This module implements a CPU-backed reference runtime that mirrors the
//! Atlas Runtime specification (§§5–7). It provides devices, contexts, queues,
//! bursts, and kernel descriptors, and wires them into the existing
//! `AtlasSpace`, `Validator`, and resonance accounting infrastructure.

use std::sync::Arc;

use atlas_isa::{KernelMetadata, LaunchConfig};
use parking_lot::Mutex;
use tracing::info_span;

use crate::{AtlasError, AtlasSpace, Result, Validator};

/// Enumerate available Atlas devices.
///
/// The current implementation exposes a single CPU-backed virtual device.
pub fn enumerate_devices() -> Vec<Device> {
    vec![Device::cpu()]
}

/// Atlas execution device.
#[derive(Debug, Clone)]
pub struct Device {
    id: usize,
    name: String,
}

impl Device {
    fn cpu() -> Self {
        Self {
            id: 0,
            name: "Atlas CPU Virtual Device".to_string(),
        }
    }

    /// Unique device identifier.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Human-readable device name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a new execution context on this device.
    pub fn create_context(&self, desc: ContextDesc) -> Context {
        Context::new(self.id, desc)
    }
}

/// Context configuration options.
#[derive(Debug, Clone, Copy)]
pub struct ContextDesc {
    /// Enable runtime validation (phase windows, mirrors, neighbors, resonance).
    pub validation: bool,
    /// Enable profiling instrumentation (placeholder).
    pub profiling: bool,
}

impl Default for ContextDesc {
    fn default() -> Self {
        Self {
            validation: true,
            profiling: false,
        }
    }
}

struct ContextInner {
    device_id: usize,
    space: Mutex<AtlasSpace>,
    desc: ContextDesc,
}

/// Execution context owning Atlas space and per-device configuration.
#[derive(Clone)]
pub struct Context {
    inner: Arc<ContextInner>,
}

impl Context {
    fn new(device_id: usize, desc: ContextDesc) -> Self {
        Self {
            inner: Arc::new(ContextInner {
                device_id,
                space: Mutex::new(AtlasSpace::new()),
                desc,
            }),
        }
    }

    /// Device identifier backing this context.
    pub fn device_id(&self) -> usize {
        self.inner.device_id
    }

    /// Context configuration descriptor.
    pub fn desc(&self) -> ContextDesc {
        self.inner.desc
    }

    /// Create a compute queue associated with this context.
    pub fn create_queue(&self, desc: QueueDesc) -> Queue {
        Queue {
            context: self.clone(),
            desc,
        }
    }

    /// Begin a burst for batched kernel submission.
    pub fn begin_burst(&self) -> Burst {
        Burst {
            context: self.clone(),
            kernels: Vec::new(),
            phase_advance: None,
        }
    }

    /// Execute a closure with mutable access to the underlying Atlas space.
    pub fn with_space<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&mut AtlasSpace) -> T,
    {
        let mut space = self.inner.space.lock();
        f(&mut space)
    }
}

/// Queue kind (only compute is currently implemented).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueKind {
    Compute,
}

/// Queue descriptor.
#[derive(Debug, Clone, Copy)]
pub struct QueueDesc {
    pub kind: QueueKind,
}

impl Default for QueueDesc {
    fn default() -> Self {
        Self {
            kind: QueueKind::Compute,
        }
    }
}

/// Execution queue bound to a context.
#[derive(Clone)]
pub struct Queue {
    context: Context,
    desc: QueueDesc,
}

impl Queue {
    /// Queue descriptor.
    pub fn desc(&self) -> QueueDesc {
        self.desc
    }

    /// Owning context.
    pub fn context(&self) -> &Context {
        &self.context
    }

    /// Submit a burst for execution.
    pub fn submit(&self, burst: Burst) -> Result<()> {
        let Burst {
            context: burst_context,
            kernels,
            phase_advance,
        } = burst;

        if !Arc::ptr_eq(&self.context.inner, &burst_context.inner) {
            return Err(AtlasError::InvalidOperation(
                "Burst submitted to queue from different context".to_string(),
            ));
        }

        let validation_enabled = self.context.desc().validation;
        let mut space = self.context.inner.space.lock();

        for entry in kernels {
            if validation_enabled {
                Validator::validate_launch(&space, entry.kernel.metadata())?;
            }

            let span = info_span!("atlas_kernel", name = entry.kernel.metadata().name.as_str());
            let _guard = span.enter();
            let mut invocation = KernelInvocation {
                space: &mut space,
                launch: entry.launch,
            };
            (entry.kernel.inner.callable)(&mut invocation)?;

            if validation_enabled {
                Validator::validate_post_launch(&space, entry.kernel.metadata())?;
            }
        }

        if let Some(advance) = phase_advance {
            space.phase().advance(advance.delta);
        }

        Ok(())
    }
}

struct QueuedKernel {
    kernel: Kernel,
    launch: LaunchConfig,
}

/// Batch of kernels executed with phase-ordered semantics.
pub struct Burst {
    context: Context,
    kernels: Vec<QueuedKernel>,
    phase_advance: Option<PhaseAdvance>,
}

impl Burst {
    /// Enqueue a kernel with its launch configuration.
    pub fn enqueue(&mut self, kernel: Kernel, launch: LaunchConfig) {
        self.kernels.push(QueuedKernel { kernel, launch });
    }

    /// Request the phase counter to advance after the burst completes.
    pub fn set_phase_advance(&mut self, advance: PhaseAdvance) {
        self.phase_advance = Some(advance);
    }

    /// Number of kernels queued in this burst.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether the burst currently holds no kernels.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }
}

/// Phase advance descriptor applied after burst execution.
#[derive(Debug, Clone, Copy)]
pub struct PhaseAdvance {
    pub delta: u16,
}

impl PhaseAdvance {
    pub fn new(delta: u16) -> Self {
        Self { delta }
    }
}

/// Invocation context provided to kernel closures.
pub struct KernelInvocation<'a> {
    space: &'a mut AtlasSpace,
    launch: LaunchConfig,
}

impl<'a> KernelInvocation<'a> {
    /// Access mutable Atlas space for the duration of the kernel.
    pub fn space(&mut self) -> &mut AtlasSpace {
        self.space
    }

    /// Launch configuration associated with this kernel.
    pub fn launch(&self) -> &LaunchConfig {
        &self.launch
    }
}

/// Callable Atlas kernel registered with the runtime.
#[derive(Clone)]
pub struct Kernel {
    inner: Arc<KernelInner>,
}

impl Kernel {
    /// Create a new kernel from metadata and executable closure.
    pub fn new<F>(metadata: KernelMetadata, func: F) -> Result<Self>
    where
        F: Fn(&mut KernelInvocation<'_>) -> Result<()> + Send + Sync + 'static,
    {
        metadata
            .validate()
            .map_err(|err| AtlasError::InvalidMetadata(err.to_string()))?;

        Ok(Self {
            inner: Arc::new(KernelInner {
                metadata,
                callable: Arc::new(func),
            }),
        })
    }

    /// Kernel metadata (validated during construction).
    pub fn metadata(&self) -> &KernelMetadata {
        &self.inner.metadata
    }
}

struct KernelInner {
    metadata: KernelMetadata,
    callable: Arc<KernelCallable>,
}

type KernelCallable = dyn Fn(&mut KernelInvocation<'_>) -> Result<()> + Send + Sync + 'static;
