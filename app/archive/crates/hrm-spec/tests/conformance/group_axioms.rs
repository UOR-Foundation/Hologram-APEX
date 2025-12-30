//! Group Axiom Conformance Tests
//!
//! Property-based tests verifying group axioms hold for all implementations.

use hrm_spec::prelude::*;
use proptest::prelude::*;

// Strategy for generating TorusCoordinates
fn torus_coord_strategy() -> impl Strategy<Value = TorusCoordinate> {
    (0u8..48, 0u8..96).prop_map(|(page, resonance)| TorusCoordinate { page, resonance })
}

proptest! {
    #[test]
    fn test_addition_closure(a in torus_coord_strategy(), b in torus_coord_strategy()) {
        let result = add(&a, &b);
        // Result is always valid TorusCoordinate
        assert!(result.page < 48);
        assert!(result.resonance < 96);
    }

    #[test]
    fn test_addition_associativity(
        a in torus_coord_strategy(),
        b in torus_coord_strategy(),
        c in torus_coord_strategy()
    ) {
        let ab_c = add(&add(&a, &b), &c);
        let a_bc = add(&a, &add(&b, &c));
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_addition_identity(a in torus_coord_strategy()) {
        let zero = TorusCoordinate::zero();
        assert_eq!(add(&a, &zero), a);
        assert_eq!(add(&zero, &a), a);
    }

    #[test]
    fn test_addition_inverse(a in torus_coord_strategy()) {
        let inv = a.inverse();
        let result = add(&a, &inv);
        assert_eq!(result, TorusCoordinate::zero());
    }

    #[test]
    fn test_addition_commutativity(a in torus_coord_strategy(), b in torus_coord_strategy()) {
        assert_eq!(add(&a, &b), add(&b, &a));
    }
}

#[test]
fn test_all_group_axioms() {
    // Verify axioms for specific examples
    let coords = vec![
        TorusCoordinate { page: 0, resonance: 0 },
        TorusCoordinate { page: 1, resonance: 1 },
        TorusCoordinate { page: 23, resonance: 47 },
        TorusCoordinate { page: 47, resonance: 95 },
    ];
    
    for a in &coords {
        for b in &coords {
            // Closure
            let sum = add(a, b);
            assert!(sum.page < 48 && sum.resonance < 96);
            
            // Identity
            assert_eq!(add(a, &TorusCoordinate::zero()), *a);
            
            // Inverse
            let inv = a.inverse();
            assert_eq!(add(a, &inv), TorusCoordinate::zero());
            
            // Commutativity
            assert_eq!(add(a, b), add(b, a));
            
            for c in &coords {
                // Associativity
                assert_eq!(add(&add(a, b), c), add(a, &add(b, c)));
            }
        }
    }
}
