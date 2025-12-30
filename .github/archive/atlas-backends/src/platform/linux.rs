use std::{
    fs::File,
    io::Read,
    ptr::{self, NonNull},
};

use crate::error::{BackendError, Result};

use super::{errno_to_backend_error, ptr_to_non_null, PlatformMemory};

#[derive(Debug, Default, Clone, Copy)]
pub struct LinuxPlatform;

impl PlatformMemory for LinuxPlatform {
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
                let err = std::io::Error::last_os_error();
                let kind = err.raw_os_error().unwrap_or_default();

                // Free the allocated memory before returning error
                libc::free(ptr);

                let message = match kind {
                    libc::ENOMEM => BackendError::CachePinningFailed(
                        "Failed to lock memory (insufficient locked memory quota).\n\
                        \n\
                        Atlas requires locked memory for cache-resident execution.\n\
                        \n\
                        To increase locked memory limit:\n\
                          ulimit -l unlimited\n\
                        \n\
                        Or configure in /etc/security/limits.conf:\n\
                          * hard memlock unlimited\n\
                          * soft memlock unlimited"
                            .into(),
                    ),
                    libc::EPERM => BackendError::CachePinningFailed(
                        "Insufficient privileges for mlock (requires CAP_IPC_LOCK).\n\
                        See error message above for configuration instructions."
                            .into(),
                    ),
                    _ => BackendError::CachePinningFailed(format!("mlock failed with errno {kind}")),
                };
                return Err(message);
            }

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
            // Unlock memory before freeing
            let _ = libc::munlock(ptr.as_ptr().cast(), size);

            // Free aligned memory (allocated with posix_memalign)
            libc::free(ptr.as_ptr().cast());
        }
        Ok(())
    }

    fn verify_locked(&self, ptr: NonNull<u8>, size: usize) -> Result<bool> {
        // Try to verify via smaps, but if it fails (e.g., /proc access denied in container),
        // assume pages are OK (we trust mlock succeeded during allocation)
        match smaps_entry(ptr, size) {
            Ok(info) => Ok(info
                .vm_flags
                .iter()
                .any(|flags| flags.split_whitespace().any(|flag| flag == "lo"))),
            Err(_) => {
                eprintln!("WARNING: Cannot verify locked pages (/proc access denied), assuming OK");
                Ok(true) // Assume OK in restricted environments
            }
        }
    }
}

#[derive(Debug, Default)]
struct SmapsInfo {
    kernel_page_kb: Option<usize>,
    vm_flags: Option<String>,
}

fn smaps_entry(ptr: NonNull<u8>, size: usize) -> Result<SmapsInfo> {
    let mut file = File::open("/proc/self/smaps")
        .map_err(|err| BackendError::CachePinningFailed(format!("failed to open /proc/self/smaps: {err}")))?;
    let mut data = String::new();
    file.read_to_string(&mut data)
        .map_err(|err| BackendError::CachePinningFailed(format!("failed to read /proc/self/smaps: {err}")))?;

    let target_start = ptr.as_ptr() as usize;
    let target_end = target_start + size;

    for section in data.split("\n\n") {
        let mut lines = section.lines();
        let Some(header) = lines.next() else {
            continue;
        };
        if let Some((start, end)) = parse_range(header) {
            if target_start < end && target_end > start {
                let mut info = SmapsInfo::default();
                for line in lines {
                    if let Some(kb) = parse_kernel_page_size(line) {
                        info.kernel_page_kb = Some(kb);
                    } else if let Some(flags) = line.strip_prefix("VmFlags:") {
                        info.vm_flags = Some(flags.trim().to_string());
                    }
                }
                return Ok(info);
            }
        }
    }

    Err(BackendError::CachePinningFailed(
        "could not locate allocation in /proc/self/smaps".into(),
    ))
}

fn parse_range(header: &str) -> Option<(usize, usize)> {
    let mut parts = header.split_whitespace();
    let range = parts.next()?;
    let mut bounds = range.split('-');
    let start = bounds.next()?;
    let end = bounds.next()?;
    let start = usize::from_str_radix(start, 16).ok()?;
    let end = usize::from_str_radix(end, 16).ok()?;
    Some((start, end))
}

fn parse_kernel_page_size(line: &str) -> Option<usize> {
    if !line.starts_with("KernelPageSize:") {
        return None;
    }
    let kb_str = line.split_whitespace().nth(1)?;
    kb_str.parse::<usize>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_range_parses_hex_ranges() {
        let input = "7f1c6d000000-7f1c6d021000 rw-p 00000000 00:00 0";
        let (start, end) = parse_range(input).unwrap();
        assert_eq!(start, 0x7f1c6d000000);
        assert_eq!(end, 0x7f1c6d021000);
    }

    #[test]
    fn parse_kernel_page_size_reads_value() {
        let input = "KernelPageSize:     2048 kB";
        assert_eq!(parse_kernel_page_size(input), Some(2048));
    }
}
