//! Torus Coordinates (page, resonance)

use num_bigint::BigInt;
use num_traits::ToPrimitive;

/// Two-torus coordinate
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TorusCoordinate {
    /// Page: T₁ coordinate [0, 47]
    pub page: u8,
    
    /// Resonance: T₂ coordinate [0, 95]
    pub resonance: u8,
}

impl TorusCoordinate {
    /// Page period (T₁)
    pub const PAGE_PERIOD: u8 = 48;
    
    /// Resonance period (T₂)
    pub const RESONANCE_PERIOD: u8 = 96;
    
    /// Boundary lattice size
    pub const BOUNDARY_SIZE: usize = 12_288;  // 48 × 256
    
    /// Create new coordinate
    pub fn new(page: u8, resonance: u8) -> Self {
        assert!(page < Self::PAGE_PERIOD, "Page must be in [0, 47]");
        assert!(resonance < Self::RESONANCE_PERIOD, "Resonance must be in [0, 95]");
        Self { page, resonance }
    }
    
    /// Project integer to torus coordinates
    pub fn from_integer(n: &BigInt) -> Self {
        let page = (n % Self::PAGE_PERIOD as u32).to_u8().unwrap_or(0);
        let resonance = (n % Self::RESONANCE_PERIOD as u32).to_u8().unwrap_or(0);
        Self { page, resonance }
    }
    
    /// Cell index in boundary lattice [0, 12287]
    pub fn cell_index(&self) -> usize {
        (self.page as usize) * 256 + (self.resonance as usize)
    }
    
    /// From cell index
    pub fn from_cell_index(idx: usize) -> Self {
        assert!(idx < Self::BOUNDARY_SIZE, "Index out of bounds");
        let page = (idx / 256) as u8;
        let resonance = (idx % 256).min(95) as u8;
        Self { page, resonance }
    }
    
    /// Zero element (additive identity)
    pub fn zero() -> Self {
        Self { page: 0, resonance: 0 }
    }
    
    /// One element (multiplicative identity)
    pub fn one() -> Self {
        Self { page: 1, resonance: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_from_integer() {
        let n = BigInt::from(12345);
        let coord = TorusCoordinate::from_integer(&n);
        
        assert_eq!(coord.page, (12345 % 48) as u8);
        assert_eq!(coord.resonance, (12345 % 96) as u8);
    }
    
    #[test]
    fn test_cell_index() {
        let coord = TorusCoordinate::new(5, 7);
        let idx = coord.cell_index();
        
        assert_eq!(idx, 5 * 256 + 7);
        
        let coord2 = TorusCoordinate::from_cell_index(idx);
        assert_eq!(coord, coord2);
    }
    
    #[test]
    fn test_identities() {
        let zero = TorusCoordinate::zero();
        assert_eq!(zero.page, 0);
        assert_eq!(zero.resonance, 0);
        
        let one = TorusCoordinate::one();
        assert_eq!(one.page, 1);
        assert_eq!(one.resonance, 1);
    }
}
