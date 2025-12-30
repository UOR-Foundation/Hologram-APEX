//! Addition Generator (⊕)
//!
//! Defines addition on torus coordinates:
//! (page₁, res₁) ⊕ (page₂, res₂) = (page₁ + page₂ mod 48, res₁ + res₂ mod 96)

use crate::torus::coordinate::TorusCoordinate;

/// Addition on torus
pub fn add(a: &TorusCoordinate, b: &TorusCoordinate) -> TorusCoordinate {
    TorusCoordinate {
        page: ((a.page as u16 + b.page as u16) % TorusCoordinate::PAGE_PERIOD as u16) as u8,
        resonance: ((a.resonance as u16 + b.resonance as u16) % TorusCoordinate::RESONANCE_PERIOD as u16) as u8,
    }
}

/// Addition forms an abelian group on T²
impl crate::foundation::group::AbelianGroup for TorusCoordinate {
    // Inherited from Group trait implementation
}

impl crate::foundation::group::Group for TorusCoordinate {
    fn op(&self, other: &Self) -> Self {
        add(self, other)
    }

    fn identity() -> Self {
        TorusCoordinate::zero()
    }

    fn inverse(&self) -> Self {
        TorusCoordinate {
            page: ((TorusCoordinate::PAGE_PERIOD - self.page) % TorusCoordinate::PAGE_PERIOD) as u8,
            resonance: ((TorusCoordinate::RESONANCE_PERIOD - self.resonance) % TorusCoordinate::RESONANCE_PERIOD) as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::group::Group;

    #[test]
    fn test_addition_identity() {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let result = add(&coord, &TorusCoordinate::zero());
        assert_eq!(result.page, coord.page);
        assert_eq!(result.resonance, coord.resonance);
    }

    #[test]
    fn test_addition_commutativity() {
        let a = TorusCoordinate { page: 3, resonance: 7 };
        let b = TorusCoordinate { page: 11, resonance: 23 };
        let ab = add(&a, &b);
        let ba = add(&b, &a);
        assert_eq!(ab.page, ba.page);
        assert_eq!(ab.resonance, ba.resonance);
    }

    #[test]
    fn test_addition_associativity() {
        let a = TorusCoordinate { page: 3, resonance: 7 };
        let b = TorusCoordinate { page: 11, resonance: 23 };
        let c = TorusCoordinate { page: 17, resonance: 31 };
        
        assert!(a.verify_associativity(&b, &c));
    }

    #[test]
    fn test_addition_modular_wraparound() {
        let a = TorusCoordinate { page: 45, resonance: 90 };
        let b = TorusCoordinate { page: 5, resonance: 10 };
        let result = add(&a, &b);
        assert_eq!(result.page, 2);  // (45 + 5) % 48 = 2
        assert_eq!(result.resonance, 4);  // (90 + 10) % 96 = 4
    }

    #[test]
    fn test_inverse() {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let inv = coord.inverse();
        let result = add(&coord, &inv);
        assert_eq!(result.page, 0);
        assert_eq!(result.resonance, 0);
    }
}
