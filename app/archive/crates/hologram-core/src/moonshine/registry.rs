//! OperationRegistry - Batch loading and management of compiled operations
//!
//! Provides a registry for loading multiple .mshr files from a directory
//! and accessing them by operation name.

use super::operation::CompiledOperation;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// Registry of compiled operations
///
/// Loads and manages multiple CompiledOperation instances,
/// providing O(1) access by operation name.
///
/// # Example
///
/// ```ignore
/// use hologram_core::moonshine::OperationRegistry;
///
/// // Load all operations from directory
/// let registry = OperationRegistry::from_directory("target/moonshine/ops")?;
///
/// // Get operations by name
/// let add = registry.get("vector_add")?;
/// let mul = registry.get("vector_mul")?;
///
/// // Execute operations
/// let result1 = add.execute_f32(&input1)?;
/// let result2 = mul.execute_f32(&input2)?;
/// ```
pub struct OperationRegistry {
    /// Map of operation name → CompiledOperation
    operations: HashMap<String, CompiledOperation>,
}

impl OperationRegistry {
    /// Create empty registry
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
        }
    }

    /// Load all .mshr files from a directory
    ///
    /// Scans directory for files with .mshr extension and loads them.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory containing .mshr files
    ///
    /// # Returns
    ///
    /// Registry with all loaded operations
    ///
    /// # Example
    ///
    /// ```ignore
    /// let registry = OperationRegistry::from_directory("ops")?;
    /// println!("Loaded {} operations", registry.len());
    /// ```
    pub fn from_directory<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut registry = Self::new();

        // Read directory
        let entries = std::fs::read_dir(path.as_ref())
            .map_err(|e| Error::InvalidOperation(format!("Failed to read directory: {}", e)))?;

        // Load each .mshr file
        for entry in entries {
            let entry = entry.map_err(|e| Error::InvalidOperation(format!("Failed to read directory entry: {}", e)))?;

            let path = entry.path();

            // Check for .mshr extension
            if path.extension().and_then(|s| s.to_str()) == Some("mshr") {
                let op = CompiledOperation::load(&path)?;
                let name = op.manifest().operation.clone();
                registry.insert(name, op);
            }
        }

        Ok(registry)
    }

    /// Insert an operation into the registry
    ///
    /// # Arguments
    ///
    /// * `name` - Operation name
    /// * `operation` - Compiled operation
    pub fn insert(&mut self, name: String, operation: CompiledOperation) {
        self.operations.insert(name, operation);
    }

    /// Get an operation by name
    ///
    /// # Arguments
    ///
    /// * `name` - Operation name
    ///
    /// # Returns
    ///
    /// Reference to the compiled operation
    ///
    /// # Errors
    ///
    /// Returns error if operation not found
    pub fn get(&self, name: &str) -> Result<&CompiledOperation> {
        self.operations.get(name).ok_or_else(|| {
            Error::InvalidOperation(format!(
                "Operation '{}' not found in registry (available: {:?})",
                name,
                self.operation_names()
            ))
        })
    }

    /// Check if operation exists in registry
    pub fn contains(&self, name: &str) -> bool {
        self.operations.contains_key(name)
    }

    /// Get number of registered operations
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Get list of all operation names
    pub fn operation_names(&self) -> Vec<String> {
        self.operations.keys().cloned().collect()
    }

    /// Clear all operations from registry
    pub fn clear(&mut self) {
        self.operations.clear();
    }
}

impl Default for OperationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = OperationRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_operations() {
        let registry = OperationRegistry::new();

        assert!(!registry.contains("test_op"));
        assert_eq!(registry.len(), 0);

        // Note: We can't easily test insert/get without creating a real .mshr file
        // These would be integration tests with actual .mshr files
    }

    #[test]
    fn test_operation_names() {
        let registry = OperationRegistry::new();
        let names = registry.operation_names();
        assert_eq!(names.len(), 0);
    }

    #[test]
    fn test_registry_default() {
        let registry = OperationRegistry::default();
        assert!(registry.is_empty());
    }
}
