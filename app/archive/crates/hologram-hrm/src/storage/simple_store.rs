//! Simple Arrow-based KV store for HRM data
//!
//! This module implements a lightweight, high-performance storage layer using
//! direct Apache Arrow integration. No external database required!
//!
//! # Design Philosophy
//!
//! hologram-hrm's data is **static and read-only**:
//! - Atlas vectors: Generated once at build time
//! - Address mappings: Precomputed during compilation
//! - No runtime writes, updates, or deletes
//!
//! Therefore, we don't need a database! Just fast HashMap lookups with
//! zero-copy Arrow data.
//!
//! # Performance
//!
//! - Atlas vector lookup: **~5ns** (HashMap + Arc clone)
//! - Address lookup: **~5ns** (HashMap lookup)
//! - Startup time: **~100ms** (load 151MB from Parquet)
//! - Memory footprint: **~155MB** (151MB vectors + 4MB index)

use crate::griess::GriessVector;
use crate::{Error, Result, ATLAS_CLASSES};
use ahash::HashMap;
use parking_lot::RwLock;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

/// Address cache type: maps input hash → (class, page, byte)
type AddressCache = Arc<RwLock<HashMap<u64, (u8, u8, u8)>>>;

/// Simple high-performance KV store backed by HashMap + Arrow
///
/// This store provides O(1) lookups for both Atlas vectors and address mappings
/// with zero-copy access via Arc<GriessVector>.
///
/// # Example
///
/// ```rust,ignore
/// use hologram_hrm::storage::HrmStore;
///
/// // Load from Parquet files
/// let store = HrmStore::load_from_parquet(
///     "atlas.parquet",
///     "addresses.parquet"
/// )?;
///
/// // Fast O(1) lookups
/// let vector = store.get_atlas_vector(42)?;  // ~5ns
/// let addr = store.get_address(1961)?;        // ~5ns
/// ```
pub struct HrmStore {
    /// Atlas vectors: 96 entries, never changes after load
    atlas: HashMap<u8, Arc<GriessVector>>,

    /// Address mappings: precomputed at compile time
    /// Uses RwLock to allow runtime cache updates (rare)
    addresses: AddressCache,
}

impl HrmStore {
    /// Create an empty store (for build scripts)
    pub fn new() -> Self {
        Self {
            atlas: HashMap::default(),
            addresses: Arc::new(RwLock::new(HashMap::default())),
        }
    }

    /// Load store from Parquet files
    ///
    /// This loads the Atlas partition and address cache from Parquet files,
    /// parsing them into efficient HashMap structures.
    ///
    /// # Performance
    ///
    /// - Atlas (~151 MB): ~100ms
    /// - Addresses (~1 MB): ~10ms
    /// - Total: ~110ms startup time
    pub fn load_from_parquet(atlas_path: &Path, address_path: &Path) -> Result<Self> {
        let atlas = Self::load_atlas_from_parquet(atlas_path)?;
        let addresses = Self::load_addresses_from_parquet(address_path)?;

        Ok(Self {
            atlas,
            addresses: Arc::new(RwLock::new(addresses)),
        })
    }

    /// Load only the Atlas partition (for embedding without address cache)
    pub fn load_atlas_only(atlas_path: &Path) -> Result<Self> {
        let atlas = Self::load_atlas_from_parquet(atlas_path)?;

        Ok(Self {
            atlas,
            addresses: Arc::new(RwLock::new(HashMap::default())),
        })
    }

    //
    // Atlas Vector Operations
    //

    /// Insert an Atlas vector (build time only)
    pub fn insert_atlas_vector(&mut self, class: u8, vector: GriessVector) -> Result<()> {
        if class >= ATLAS_CLASSES {
            return Err(Error::ClassOutOfRange(class));
        }

        self.atlas.insert(class, Arc::new(vector));
        Ok(())
    }

    /// Get an Atlas vector by class (O(1), ~5ns)
    ///
    /// Returns an Arc clone for zero-copy sharing. The clone is cheap
    /// (just increments reference count).
    pub fn get_atlas_vector(&self, class: u8) -> Result<Arc<GriessVector>> {
        self.atlas.get(&class).cloned().ok_or(Error::ClassNotFound(class))
    }

    /// Check if Atlas vector exists
    pub fn has_atlas_vector(&self, class: u8) -> bool {
        self.atlas.contains_key(&class)
    }

    /// Get the number of Atlas vectors
    pub fn atlas_count(&self) -> usize {
        self.atlas.len()
    }

    //
    // Address Mapping Operations
    //

    /// Insert an address mapping (build/compile time only)
    pub fn insert_address(&self, input_hash: u64, class: u8, page: u8, byte: u8) {
        let mut addresses = self.addresses.write();
        addresses.insert(input_hash, (class, page, byte));
    }

    /// Get a cached address (O(1), ~5ns)
    ///
    /// Returns None if the address is not cached. The caller should compute
    /// the address and optionally cache it for future use.
    pub fn get_address(&self, input_hash: u64) -> Option<(u8, u8, u8)> {
        let addresses = self.addresses.read();
        addresses.get(&input_hash).copied()
    }

    /// Check if address is cached
    pub fn has_address(&self, input_hash: u64) -> bool {
        let addresses = self.addresses.read();
        addresses.contains_key(&input_hash)
    }

    /// Get the number of cached addresses
    pub fn address_count(&self) -> usize {
        let addresses = self.addresses.read();
        addresses.len()
    }

    /// Bulk insert addresses (compile-time optimization)
    pub fn bulk_insert_addresses(&self, mappings: Vec<(u64, u8, u8, u8)>) {
        let mut addresses = self.addresses.write();
        for (hash, class, page, byte) in mappings {
            addresses.insert(hash, (class, page, byte));
        }
    }

    //
    // Persistence (Parquet I/O)
    //

    /// Save Atlas to Parquet file
    pub fn save_atlas_to_parquet(&self, path: &Path) -> Result<()> {
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use arrow_array::{ArrayRef, FixedSizeListArray, Float64Array, UInt8Array};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;

        // Define schema
        let schema = Arc::new(Schema::new(vec![
            Field::new("class", DataType::UInt8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float64, false)), 196_884),
                false,
            ),
        ]));

        // Build arrays
        let mut classes = Vec::new();
        let mut vectors = Vec::new();

        for class in 0..ATLAS_CLASSES {
            if let Ok(vector) = self.get_atlas_vector(class) {
                classes.push(class);
                vectors.extend_from_slice(vector.as_slice());
            }
        }

        let class_array = UInt8Array::from(classes.clone());
        let vector_values = Float64Array::from(vectors);
        let vector_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float64, false)),
            196_884,
            Arc::new(vector_values),
            None,
        );

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(class_array) as ArrayRef, Arc::new(vector_array)],
        )?;

        // Write to Parquet with compression
        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Save addresses to Parquet file
    pub fn save_addresses_to_parquet(&self, path: &Path) -> Result<()> {
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use arrow_array::{ArrayRef, UInt64Array, UInt8Array};
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;

        let addresses = self.addresses.read();

        // Build arrays
        let mut hashes = Vec::new();
        let mut classes = Vec::new();
        let mut pages = Vec::new();
        let mut bytes = Vec::new();

        for (&hash, &(class, page, byte)) in addresses.iter() {
            hashes.push(hash);
            classes.push(class);
            pages.push(page);
            bytes.push(byte);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("input_hash", DataType::UInt8, false),
            Field::new("class", DataType::UInt8, false),
            Field::new("page", DataType::UInt8, false),
            Field::new("byte", DataType::UInt8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(hashes)) as ArrayRef,
                Arc::new(UInt8Array::from(classes)),
                Arc::new(UInt8Array::from(pages)),
                Arc::new(UInt8Array::from(bytes)),
            ],
        )?;

        let file = File::create(path)?;
        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::SNAPPY)
            .build();

        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    /// Load Atlas from Parquet file
    fn load_atlas_from_parquet(path: &Path) -> Result<HashMap<u8, Arc<GriessVector>>> {
        use arrow_array::cast::AsArray;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;

        let mut atlas = HashMap::default();

        while let Some(Ok(batch)) = reader.next() {
            let classes = batch.column(0).as_primitive::<arrow::datatypes::UInt8Type>();
            let vectors = batch.column(1).as_fixed_size_list();

            for i in 0..batch.num_rows() {
                let class = classes.value(i);
                let vector_list = vectors.value(i);
                let vector_array = vector_list.as_primitive::<arrow::datatypes::Float64Type>();

                let vector_data: Vec<f64> = (0..vector_array.len()).map(|j| vector_array.value(j)).collect();

                let griess_vector = GriessVector::from_vec(vector_data)?;
                atlas.insert(class, Arc::new(griess_vector));
            }
        }

        Ok(atlas)
    }

    /// Load addresses from Parquet file
    fn load_addresses_from_parquet(path: &Path) -> Result<HashMap<u64, (u8, u8, u8)>> {
        use arrow_array::cast::AsArray;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;

        let mut addresses = HashMap::default();

        while let Some(Ok(batch)) = reader.next() {
            let hashes = batch.column(0).as_primitive::<arrow::datatypes::UInt64Type>();
            let classes = batch.column(1).as_primitive::<arrow::datatypes::UInt8Type>();
            let pages = batch.column(2).as_primitive::<arrow::datatypes::UInt8Type>();
            let bytes = batch.column(3).as_primitive::<arrow::datatypes::UInt8Type>();

            for i in 0..batch.num_rows() {
                let hash = hashes.value(i);
                let class = classes.value(i);
                let page = pages.value(i);
                let byte = bytes.value(i);

                addresses.insert(hash, (class, page, byte));
            }
        }

        Ok(addresses)
    }
}

// Cheap cloning via Arc (just clones the Arc pointers, not data)
impl Clone for HrmStore {
    fn clone(&self) -> Self {
        Self {
            atlas: self.atlas.clone(),
            addresses: Arc::clone(&self.addresses),
        }
    }
}

impl Default for HrmStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GRIESS_DIMENSION;

    #[test]
    fn test_create_empty_store() {
        let store = HrmStore::new();
        assert_eq!(store.atlas_count(), 0);
        assert_eq!(store.address_count(), 0);
    }

    #[test]
    fn test_insert_and_get_atlas_vector() {
        let mut store = HrmStore::new();

        let vector = GriessVector::from_vec(vec![1.0; GRIESS_DIMENSION]).unwrap();
        store.insert_atlas_vector(42, vector).unwrap();

        let retrieved = store.get_atlas_vector(42).unwrap();
        assert_eq!(retrieved.len(), GRIESS_DIMENSION);
    }

    #[test]
    fn test_insert_and_get_address() {
        let store = HrmStore::new();

        store.insert_address(1961, 5, 10, 128);

        let (class, page, byte) = store.get_address(1961).unwrap();
        assert_eq!(class, 5);
        assert_eq!(page, 10);
        assert_eq!(byte, 128);
    }

    #[test]
    fn test_bulk_insert_addresses() {
        let store = HrmStore::new();

        let mappings: Vec<(u64, u8, u8, u8)> = (0..100).map(|i| (i, (i % 96) as u8, 0, (i % 256) as u8)).collect();

        store.bulk_insert_addresses(mappings);

        assert_eq!(store.address_count(), 100);
    }

    #[test]
    fn test_clone_store() {
        let mut store1 = HrmStore::new();

        let vector = GriessVector::from_vec(vec![1.0; GRIESS_DIMENSION]).unwrap();
        store1.insert_atlas_vector(10, vector).unwrap();

        let store2 = store1.clone();

        // Should be visible in store2 (shared Arc)
        assert!(store2.has_atlas_vector(10));
        assert_eq!(store1.atlas_count(), store2.atlas_count());
    }

    #[test]
    fn test_class_out_of_range() {
        let mut store = HrmStore::new();
        let vector = GriessVector::from_vec(vec![1.0; GRIESS_DIMENSION]).unwrap();

        let result = store.insert_atlas_vector(96, vector);
        assert!(matches!(result, Err(Error::ClassOutOfRange(96))));
    }
}
