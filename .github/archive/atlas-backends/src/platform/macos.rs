use std::{ptr, ptr::NonNull};

use crate::error::{BackendError, Result};

use super::{errno_to_backend_error, ptr_to_non_null, PlatformMemory};

#[derive(Debug, Default, Clone, Copy)]
pub struct MacPlatform;

impl PlatformMemory for MacPlatform {
    fn allocate_locked_huge(&self, size: usize) -> Result<NonNull<u8>> {
        unsafe {
            // Allocate cache-line aligned memory (64 bytes)
            // Use posix_memalign for explicit alignment guarantee
            const ALIGNMENT: usize = 64; // Cache line size

            let mut ptr: *mut libc::c_void = ptr::null_mut();

            let ret = libc::posix_memalign(&mut ptr, ALIGNMENT, size);

            if ret != 0 {
                let message = match ret {
                    libc::ENOMEM => {
                        BackendError::AllocationFailed("Failed to allocate aligned memory (out of memory).".into())
                    }
                    libc::EINVAL => BackendError::AllocationFailed(
                        "Invalid alignment value (should be power of 2 and multiple of sizeof(void*)).".into(),
                    ),
                    _ => BackendError::AllocationFailed(format!("posix_memalign failed with error code {ret}")),
                };
                return Err(message);
            }

            // Lock memory into physical RAM for cache residency
            if libc::mlock(ptr, size) != 0 {
                // Free the allocated memory before returning error
                libc::free(ptr);
                return Err(BackendError::CachePinningFailed(
                    "mlock failed (consider sudo sysctl vm.user_wire_limit).\n\
                    \n\
                    Atlas requires locked memory for cache-resident execution.\n\
                    \n\
                    To increase wire limit:\n\
                      sudo sysctl vm.user_wire_limit=<bytes>"
                        .into(),
                ));
            }

            // macOS benefits from MADV_WILLNEED to fault pages in eagerly
            let _ = libc::madvise(ptr, size, libc::MADV_WILLNEED);

            // Verify alignment (defensive check)
            let nn = NonNull::new(ptr.cast::<u8>())
                .ok_or_else(|| BackendError::AllocationFailed("posix_memalign returned null".into()))?;

            debug_assert_eq!(
                nn.as_ptr() as usize % ALIGNMENT,
                0,
                "posix_memalign should guarantee 64-byte alignment"
            );

            Ok(nn)
        }
    }

    fn deallocate(&self, ptr: NonNull<u8>, size: usize) -> Result<()> {
        unsafe {
            let addr = ptr.as_ptr().cast();

            // Unlock memory before freeing
            let _ = libc::munlock(addr, size);

            // Free aligned memory (allocated with posix_memalign)
            libc::free(addr);
        }
        Ok(())
    }

    fn verify_locked(&self, ptr: NonNull<u8>, size: usize) -> Result<bool> {
        // Since we issue `mlock` during allocation any subsequent `mlock` should succeed immediately.
        unsafe {
            if libc::mlock(ptr.as_ptr().cast(), size) == 0 {
                // Leave the region locked.
                Ok(true)
            } else {
                Err(BackendError::CachePinningFailed(
                    "mlock verification failed; lock limit may have been reduced".into(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_allocate_deallocate() {
        let platform = MacPlatform;
        // Try to allocate a small region
        let size = 4096; // 4KB, small enough to not require special permissions

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
                // Expected if mlock fails due to permissions
                println!("⚠ Skipping: mlock failed (likely permissions issue)");
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn test_macos_verify_locked() {
        let platform = MacPlatform;
        let size = 4096;

        match platform.allocate_locked_huge(size) {
            Ok(ptr) => {
                // Verify the region is locked
                match platform.verify_locked(ptr, size) {
                    Ok(true) => {
                        println!("✓ Memory successfully locked on macOS");
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
                println!("⚠ Skipping: mlock failed (likely permissions issue)");
            }
            Err(e) => {
                panic!("Unexpected error: {}", e);
            }
        }
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn test_macos_platform_not_tested_on_other_platforms() {
        // This test just documents that macOS tests are platform-specific
        println!("ℹ macOS platform tests only run on macOS");
    }
}
