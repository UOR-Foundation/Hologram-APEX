//! Platform-specific memory allocation helpers
//!
//! Provides an abstraction over OS-specific APIs for allocating cache-resident
//! memory required by Atlas boundary pools. Each supported operating system
//! exposes a type implementing [`PlatformMemory`].

use std::{ffi::c_void, ptr::NonNull};

use crate::error::{BackendError, Result};

/// Abstraction over platform-specific memory management required by Atlas.
pub trait PlatformMemory {
    /// Allocate a cache-resident (memory-locked) region of the requested size.
    ///
    /// The allocated memory will be:
    /// - Locked in physical RAM (preventing swap/page-out)
    /// - Aligned to 64-byte cache lines
    /// - Suitable for cache-resident execution
    fn allocate_locked_huge(&self, size: usize) -> Result<NonNull<u8>>;

    /// Release a previously allocated region.
    fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<()>;

    /// Verify that the supplied region is locked in memory.
    fn verify_locked(&self, ptr: NonNull<u8>, size: usize) -> Result<bool>;
}

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(target_os = "linux")]
pub use linux::LinuxPlatform as CurrentPlatform;
#[cfg(target_os = "macos")]
pub use macos::MacPlatform as CurrentPlatform;
#[cfg(target_os = "windows")]
pub use windows::WindowsPlatform as CurrentPlatform;

/// Obtain the platform memory implementation for the current target OS.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
pub fn current_platform() -> CurrentPlatform {
    CurrentPlatform
}

/// Helper to convert raw pointers returned from system calls into [`NonNull`].
fn ptr_to_non_null(addr: *mut c_void) -> Result<NonNull<u8>> {
    if addr.is_null() {
        Err(BackendError::AllocationFailed(
            "platform allocation returned null".into(),
        ))
    } else {
        // Safety: pointer is non-null by construction and points to valid allocation of `size`.
        Ok(unsafe { NonNull::new_unchecked(addr.cast::<u8>()) })
    }
}

/// Convert an errno-style error code into a [`BackendError`].
fn errno_to_backend_error(prefix: &str) -> BackendError {
    let err = std::io::Error::last_os_error();
    BackendError::AllocationFailed(format!("{prefix}: {err}"))
}
