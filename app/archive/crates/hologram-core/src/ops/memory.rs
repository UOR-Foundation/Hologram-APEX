//! Memory operations

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::executor::Executor;

/// Copy buffer contents from source to destination
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `src` - Source buffer to copy from
/// * `dst` - Destination buffer to copy to
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let src = exec.allocate::<f32>(1024)?;
/// let mut dst = exec.allocate::<f32>(1024)?;
///
/// ops::memory::copy(&mut exec, &src, &mut dst)?;
/// ```
pub fn copy<T: bytemuck::Pod + Copy>(exec: &mut Executor, src: &Buffer<T>, dst: &mut Buffer<T>) -> Result<()> {
    if src.len() != dst.len() {
        return Err(Error::InvalidOperation(format!(
            "copy: source length {} != destination length {}",
            src.len(),
            dst.len()
        )));
    }

    // Read from source and write to destination
    let data = src.to_vec(exec)?;
    dst.copy_from_slice(exec, &data)?;

    Ok(())
}

/// Fill buffer with a constant value
///
/// # Arguments
///
/// * `exec` - Executor for backend operations
/// * `buffer` - Buffer to fill
/// * `value` - Value to fill buffer with
///
/// # Example
///
/// ```ignore
/// use hologram_core::{Executor, ops};
///
/// let mut exec = Executor::new()?;
/// let mut buf = exec.allocate::<f32>(1024)?;
///
/// // Fill buffer with 2.5
/// ops::memory::fill(&mut exec, &mut buf, 2.5f32)?;
/// ```
pub fn fill<T: bytemuck::Pod + Copy>(exec: &mut Executor, buffer: &mut Buffer<T>, value: T) -> Result<()> {
    // Get the number of elements in the buffer
    let elem_count = buffer.len();

    // Create a vector filled with the value
    let data = vec![value; elem_count];

    // Copy to buffer
    buffer.copy_from_slice(exec, &data)?;

    Ok(())
}
