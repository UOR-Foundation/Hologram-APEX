//! Homomorphism Conformance Tests
//!
//! Verifies coherence properties: π(a ○ b) = π(a) ○ π(b)

use hrm_spec::prelude::*;
use proptest::prelude::*;

fn bigint_strategy() -> impl Strategy<Value = BigInt> {
    (0i64..1_000_000i64).prop_map(BigInt::from)
}

proptest! {
    #[test]
    fn test_addition_coherence(a in bigint_strategy(), b in bigint_strategy()) {
        let verifier = CoherenceVerifier::new();
        assert!(verifier.verify_addition_coherence(&a, &b));
    }

    #[test]
    fn test_multiplication_coherence(a in bigint_strategy(), b in bigint_strategy()) {
        let verifier = CoherenceVerifier::new();
        assert!(verifier.verify_multiplication_coherence(&a, &b));
    }

    #[test]
    fn test_scalar_coherence(k in bigint_strategy(), a in bigint_strategy()) {
        let verifier = CoherenceVerifier::new();
        assert!(verifier.verify_scalar_coherence(&k, &a));
    }
}

#[test]
fn test_factorization_coherence() {
    let verifier = CoherenceVerifier::new();
    
    // Test factorization routing constraints
    let test_cases = vec![
        (3, 5, 15),
        (7, 11, 77),
        (13, 17, 221),
        (23, 29, 667),
    ];
    
    for (p, q, n) in test_cases {
        let p_int = BigInt::from(p);
        let q_int = BigInt::from(q);
        let n_int = BigInt::from(n);
        
        // Verify π(p × q) = π(p) ⊗ π(q)
        assert!(verifier.verify_multiplication_coherence(&p_int, &q_int));
        
        // Verify projection is consistent
        let p_coord = StandardProjection.project(&p_int);
        let q_coord = StandardProjection.project(&q_int);
        let n_coord = StandardProjection.project(&n_int);
        
        let product_coord = mul(&p_coord, &q_coord);
        assert_eq!(product_coord, n_coord);
    }
}

#[test]
fn test_routing_protocol_coherence() {
    use hrm_spec::routing::protocol::verify_routing_coherence;
    
    let test_pairs = vec![
        (BigInt::from(3), BigInt::from(5)),
        (BigInt::from(7), BigInt::from(11)),
        (BigInt::from(13), BigInt::from(17)),
        (BigInt::from(101), BigInt::from(103)),
    ];
    
    for (a, b) in test_pairs {
        assert!(verify_routing_coherence(&a, &b));
    }
}
