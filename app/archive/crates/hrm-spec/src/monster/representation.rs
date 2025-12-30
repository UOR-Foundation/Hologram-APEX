//! Monster Group 196,884-Dimensional Representation

/// Monster group representation
pub struct MonsterRepresentation {
    /// Dimension of minimal faithful representation
    pub dimension: usize,
}

impl MonsterRepresentation {
    /// The Monster's minimal faithful representation dimension
    pub const DIMENSION: usize = 196_884;
    
    /// Create new Monster representation
    pub fn new() -> Self {
        Self {
            dimension: Self::DIMENSION,
        }
    }
    
    /// Verify dimension is correct
    pub fn verify_dimension(&self) -> bool {
        self.dimension == Self::DIMENSION
    }
}

impl Default for MonsterRepresentation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dimension() {
        let monster = MonsterRepresentation::new();
        assert_eq!(monster.dimension, 196_884);
        assert!(monster.verify_dimension());
    }
}
