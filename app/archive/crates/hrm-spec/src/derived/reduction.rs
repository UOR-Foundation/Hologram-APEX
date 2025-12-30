//! Reduction Operations
//!
//! Implements reductions (sum, product) as iterated composition of ⊕ and ⊗.

use crate::torus::coordinate::TorusCoordinate;
use crate::algebra::addition::add;
use crate::algebra::multiplication::mul;

/// Sum reduction via ⊕
pub fn reduce_sum(coords: &[TorusCoordinate]) -> TorusCoordinate {
    coords.iter().fold(TorusCoordinate::zero(), |acc, c| add(&acc, c))
}

/// Product reduction via ⊗
pub fn reduce_product(coords: &[TorusCoordinate]) -> TorusCoordinate {
    coords.iter().fold(TorusCoordinate::one(), |acc, c| mul(&acc, c))
}

/// Max reduction (using cell index ordering)
pub fn reduce_max(coords: &[TorusCoordinate]) -> Option<TorusCoordinate> {
    coords.iter()
        .max_by_key(|c| c.cell_index())
        .cloned()
}

/// Min reduction (using cell index ordering)
pub fn reduce_min(coords: &[TorusCoordinate]) -> Option<TorusCoordinate> {
    coords.iter()
        .min_by_key(|c| c.cell_index())
        .cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_sum_empty() {
        let coords: Vec<TorusCoordinate> = vec![];
        let result = reduce_sum(&coords);
        assert_eq!(result, TorusCoordinate::zero());
    }

    #[test]
    fn test_reduce_sum() {
        let coords = vec![
            TorusCoordinate { page: 1, resonance: 2 },
            TorusCoordinate { page: 3, resonance: 4 },
            TorusCoordinate { page: 5, resonance: 6 },
        ];
        let result = reduce_sum(&coords);
        assert_eq!(result.page, 9);
        assert_eq!(result.resonance, 12);
    }

    #[test]
    fn test_reduce_product_empty() {
        let coords: Vec<TorusCoordinate> = vec![];
        let result = reduce_product(&coords);
        assert_eq!(result, TorusCoordinate::one());
    }

    #[test]
    fn test_reduce_product() {
        let coords = vec![
            TorusCoordinate { page: 2, resonance: 3 },
            TorusCoordinate { page: 3, resonance: 5 },
        ];
        let result = reduce_product(&coords);
        assert_eq!(result.page, 6);
        assert_eq!(result.resonance, 15);
    }

    #[test]
    fn test_reduce_max() {
        let coords = vec![
            TorusCoordinate { page: 1, resonance: 2 },
            TorusCoordinate { page: 5, resonance: 10 },
            TorusCoordinate { page: 3, resonance: 6 },
        ];
        let result = reduce_max(&coords).unwrap();
        // Max by cell_index
        assert!(result.cell_index() >= coords[0].cell_index());
        assert!(result.cell_index() >= coords[2].cell_index());
    }

    #[test]
    fn test_reduce_min_empty() {
        let coords: Vec<TorusCoordinate> = vec![];
        assert!(reduce_min(&coords).is_none());
    }
}
