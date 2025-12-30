//! Ring Axiom Conformance Tests
//!
//! Property-based tests verifying ring axioms for multiplication.

use hrm_spec::prelude::*;
use proptest::prelude::*;

fn torus_coord_strategy() -> impl Strategy<Value = TorusCoordinate> {
    (0u8..48, 0u8..96).prop_map(|(page, resonance)| TorusCoordinate { page, resonance })
}

proptest! {
    #[test]
    fn test_multiplication_closure(a in torus_coord_strategy(), b in torus_coord_strategy()) {
        let result = mul(&a, &b);
        assert!(result.page < 48);
        assert!(result.resonance < 96);
    }

    #[test]
    fn test_multiplication_associativity(
        a in torus_coord_strategy(),
        b in torus_coord_strategy(),
        c in torus_coord_strategy()
    ) {
        let ab_c = mul(&mul(&a, &b), &c);
        let a_bc = mul(&a, &mul(&b, &c));
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_multiplication_identity(a in torus_coord_strategy()) {
        let one = TorusCoordinate::one();
        assert_eq!(mul(&a, &one), a);
        assert_eq!(mul(&one, &a), a);
    }

    #[test]
    fn test_multiplication_commutativity(a in torus_coord_strategy(), b in torus_coord_strategy()) {
        assert_eq!(mul(&a, &b), mul(&b, &a));
    }

    #[test]
    fn test_left_distributivity(
        a in torus_coord_strategy(),
        b in torus_coord_strategy(),
        c in torus_coord_strategy()
    ) {
        // a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
        let lhs = mul(&a, &add(&b, &c));
        let rhs = add(&mul(&a, &b), &mul(&a, &c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_right_distributivity(
        a in torus_coord_strategy(),
        b in torus_coord_strategy(),
        c in torus_coord_strategy()
    ) {
        // (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
        let lhs = mul(&add(&a, &b), &c);
        let rhs = add(&mul(&a, &c), &mul(&b, &c));
        assert_eq!(lhs, rhs);
    }
}

#[test]
fn test_ring_axioms_specific() {
    let coords = vec![
        TorusCoordinate { page: 0, resonance: 0 },
        TorusCoordinate { page: 1, resonance: 1 },
        TorusCoordinate { page: 2, resonance: 3 },
        TorusCoordinate { page: 5, resonance: 7 },
    ];
    
    for a in &coords {
        for b in &coords {
            // Multiplicative identity
            assert_eq!(mul(a, &TorusCoordinate::one()), *a);
            
            // Commutativity
            assert_eq!(mul(a, b), mul(b, a));
            
            for c in &coords {
                // Associativity
                assert_eq!(mul(&mul(a, b), c), mul(a, &mul(b, c)));
                
                // Distributivity
                assert_eq!(mul(a, &add(b, c)), add(&mul(a, b), &mul(a, c)));
                assert_eq!(mul(&add(a, b), c), add(&mul(a, c), &mul(b, c)));
            }
        }
    }
}
