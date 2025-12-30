//! Conjugacy Classes of the Monster Group

/// Conjugacy class of Monster group
pub struct ConjugacyClass {
    /// Class identifier
    pub id: usize,
    
    /// Order of elements in this class
    pub order: u64,
}

impl ConjugacyClass {
    /// Number of conjugacy classes in Monster
    pub const NUM_CLASSES: usize = 194;
}

// Full implementation deferred to Phase 1
