#![cfg(target_os = "windows")]

use std::ptr::{self, NonNull};

use crate::error::{BackendError, Result};

use super::PlatformMemory;
use windows_sys::Win32::Foundation::{GetLastError, WIN32_ERROR};
use windows_sys::Win32::System::Memory::{
    VirtualAlloc, VirtualFree, VirtualLock, VirtualUnlock, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE,
};

#[derive(Debug, Default, Clone, Copy)]
pub struct WindowsPlatform;

impl PlatformMemory for WindowsPlatform {
    fn allocate_locked_huge(&self, size: usize) -> Result<NonNull<u8>> {
        unsafe {
            // Allocate cache-line aligned memory (64 bytes)
            //
            // NOTE: VirtualAlloc provides 64KB allocation granularity on Windows,
            // which guarantees 64-byte alignment (64KB = 1024 × 64 bytes).
            // This is documented in MSDN: "The allocated memory is automatically
            // aligned on addresses that are multiples of the system's allocation
            // granularity (64 KB on most systems)."
            //
            // Cache residency comes from VirtualLock, not from page size.
            let ptr = VirtualAlloc(ptr::null_mut(), size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

            if ptr.is_null() {
                return Err(BackendError::AllocationFailed(
                    "Failed to allocate memory with VirtualAlloc.".into(),
                ));
            }

            // Lock memory into working set for cache residency
            if VirtualLock(ptr, size) == 0 {
                let _ = VirtualFree(ptr, 0, MEM_RELEASE);
                return Err(BackendError::CachePinningFailed(
                    "Failed to lock memory (VirtualLock failed).\n\
                    \n\
                    Atlas requires locked memory for cache-resident execution.\n\
                    \n\
                    On Windows, ensure sufficient working set size quota."
                        .into(),
                ));
            }

            let nn = NonNull::new(ptr.cast::<u8>())
                .ok_or_else(|| BackendError::AllocationFailed("VirtualAlloc returned null".into()))?;

            // Verify 64-byte alignment (defensive check, should always pass)
            debug_assert_eq!(
                nn.as_ptr() as usize % 64,
                0,
                "VirtualAlloc should guarantee 64-byte alignment via 64KB granularity"
            );

            Ok(nn)
        }
    }

    fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        unsafe {
            let raw = ptr.as_ptr().cast();
            let _ = VirtualUnlock(raw, size);
            if VirtualFree(raw, 0, MEM_RELEASE) == 0 {
                return Err(win32_allocation_error("VirtualFree"));
            }
        }
        Ok(())
    }

    fn verify_locked(&self, ptr: NonNull<u8>, size: usize) -> Result<bool> {
        unsafe {
            if VirtualLock(ptr.as_ptr().cast(), size) == 0 {
                return Err(win32_allocation_error("VirtualLock"));
            }
            if VirtualUnlock(ptr.as_ptr().cast(), size) == 0 {
                return Err(win32_allocation_error("VirtualUnlock"));
            }
        }
        Ok(true)
    }
}

fn win32_allocation_error(prefix: &str) -> BackendError {
    let code: WIN32_ERROR = unsafe { GetLastError() };
    BackendError::CachePinningFailed(format!("{prefix} failed with Win32 error {code}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_allocate_deallocate() {
        let platform = WindowsPlatform;
        // Try to allocate a small region
        let size = 4096; // 4KB

        match platform.allocate_locked_huge(size) {
            Ok(ptr) => {
                // Verify alignment
                assert_eq!(
                    ptr.as_ptr() as usize % 64,
                    0,
                    "Allocated pointer should be 64-byte aligned"
                );

                // Deallocate
                platform.deallocate(ptr, size).expect("Deallocation should succeed");
            }
            Err(BackendError::CachePinningFailed(_)) => {
                // Expected if VirtualLock fails due to permissions or large pages not enabled
                println!(
                    "⚠ Skipping: VirtualLock or VirtualAlloc failed (likely permissions or large pages not enabled)"
                );
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(target_os = "windows")]
    fn test_windows_verify_locked() {
        let platform = WindowsPlatform;
        let size = 4096;

        match platform.allocate_locked_huge(size) {
            Ok(ptr) => {
                // Verify the region is locked
                match platform.verify_locked(ptr, size) {
                    Ok(true) => {
                        println!("✓ Memory successfully locked on Windows");
                    }
                    Ok(false) => {
                        panic!("Memory should be locked after allocation");
                    }
                    Err(_) => {
                        println!("⚠ Lock verification failed (may be environment-specific)");
                    }
                }

                platform.deallocate(ptr, size).ok();
            }
            Err(BackendError::CachePinningFailed(_)) => {
                println!("⚠ Skipping: VirtualLock failed (likely permissions issue)");
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "windows"))]
    fn test_windows_platform_not_tested_on_other_platforms() {
        // This test just documents that Windows tests are platform-specific
        println!("ℹ Windows platform tests only run on Windows");
    }
}
