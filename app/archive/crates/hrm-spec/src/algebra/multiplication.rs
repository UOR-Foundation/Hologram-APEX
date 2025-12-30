//! Multiplication Generator (⊗)
//!
//! Defines multiplication on torus coordinates:
//! (page₁, res₁) ⊗ (page₂, res₂) = (page₁ × page₂ mod 48, res₁ × res₂ mod 96)
//!
//! This is the ROUTING PROTOCOL discovered through factorization:
//! - page_p × page_q ≡ page_n (mod 48)
//! - res_p × res_q ≡ res_n (mod 96)

use crate::torus::coordinate::TorusCoordinate;

/// Multiplication on torus (routing protocol)
pub fn mul(a: &TorusCoordinate, b: &TorusCoordinate) -> TorusCoordinate {
    TorusCoordinate {
        page: ((a.page as u16 * b.page as u16) % TorusCoordinate::PAGE_PERIOD as u16) as u8,
        resonance: ((a.resonance as u16 * b.resonance as u16) % TorusCoordinate::RESONANCE_PERIOD as u16) as u8,
    }
}

/// Multiplicative identity element (1, 1)
pub fn multiplicative_identity() -> TorusCoordinate {
    TorusCoordinate::one()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiplication_identity() {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let result = mul(&coord, &multiplicative_identity());
        assert_eq!(result.page, coord.page);
        assert_eq!(result.resonance, coord.resonance);
    }

    #[test]
    fn test_multiplication_commutativity() {
        let a = TorusCoordinate { page: 3, resonance: 7 };
        let b = TorusCoordinate { page: 5, resonance: 11 };
        let ab = mul(&a, &b);
        let ba = mul(&b, &a);
        assert_eq!(ab.page, ba.page);
        assert_eq!(ab.resonance, ba.resonance);
    }

    #[test]
    fn test_multiplication_associativity() {
        let a = TorusCoordinate { page: 3, resonance: 7 };
        let b = TorusCoordinate { page: 5, resonance: 11 };
        let c = TorusCoordinate { page: 7, resonance: 13 };
        
        let ab_c = mul(&mul(&a, &b), &c);
        let a_bc = mul(&a, &mul(&b, &c));
        
        assert_eq!(ab_c.page, a_bc.page);
        assert_eq!(ab_c.resonance, a_bc.resonance);
    }

    #[test]
    fn test_factorization_routing_constraint() {
        // Test case: 3 × 5 = 15
        // π(3) = (3, 3), π(5) = (5, 5), π(15) = (15, 15)
        let p3 = TorusCoordinate { page: 3, resonance: 3 };
        let p5 = TorusCoordinate { page: 5, resonance: 5 };
        let p15 = TorusCoordinate { page: 15, resonance: 15 };
        
        let result = mul(&p3, &p5);
        assert_eq!(result.page, p15.page);
        assert_eq!(result.resonance, p15.resonance);
    }

    #[test]
    fn test_factorization_routing_constraint_large() {
        // Test case: 7 × 11 = 77
        // π(7) = (7, 7), π(11) = (11, 11), π(77) = (29, 77)
        let p7 = TorusCoordinate { page: 7, resonance: 7 };
        let p11 = TorusCoordinate { page: 11, resonance: 11 };
        let p77 = TorusCoordinate { page: 29, resonance: 77 };
        
        let result = mul(&p7, &p11);
        assert_eq!(result.page, p77.page);  // (7 × 11) % 48 = 77 % 48 = 29
        assert_eq!(result.resonance, p77.resonance);  // (7 × 11) % 96 = 77
    }

    #[test]
    fn test_multiplication_modular_wraparound() {
        let a = TorusCoordinate { page: 10, resonance: 20 };
        let b = TorusCoordinate { page: 15, resonance: 30 };
        let result = mul(&a, &b);
        assert_eq!(result.page, 6);  // (10 × 15) % 48 = 150 % 48 = 6
        assert_eq!(result.resonance, 24);  // (20 × 30) % 96 = 600 % 96 = 24
    }
}
