//! Conditional synchronization primitives
//!
//! This module provides locking primitives that switch between parking_lot
//! and std::sync based on the `threading` feature flag.
//!
//! - With `threading` feature: Uses parking_lot (faster, smaller)
//! - Without `threading` feature: Uses std::sync (WASM-compatible)

#[cfg(feature = "threading")]
pub use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[cfg(not(feature = "threading"))]
pub use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Lock a mutex, handling both parking_lot (infallible) and std::sync (fallible) APIs
#[cfg(feature = "threading")]
#[inline]
pub fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock()
}

#[cfg(not(feature = "threading"))]
#[inline]
pub fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().expect("Mutex poisoned")
}

/// Read lock an RwLock, handling both parking_lot and std::sync APIs
#[cfg(feature = "threading")]
#[inline]
pub fn read_lock<T>(rwlock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    rwlock.read()
}

#[cfg(not(feature = "threading"))]
#[inline]
pub fn read_lock<T>(rwlock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    rwlock.read().expect("RwLock poisoned")
}

/// Write lock an RwLock, handling both parking_lot and std::sync APIs
#[cfg(feature = "threading")]
#[inline]
pub fn write_lock<T>(rwlock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    rwlock.write()
}

#[cfg(not(feature = "threading"))]
#[inline]
pub fn write_lock<T>(rwlock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    rwlock.write().expect("RwLock poisoned")
}
