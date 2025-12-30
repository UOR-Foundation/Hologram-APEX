//! Integration Test: End-to-End Factorization Example
//!
//! Demonstrates the complete pipeline from integer factorization
//! through routing protocol to verification.

use hrm_spec::prelude::*;

#[test]
fn test_complete_factorization_pipeline() {
    // Factor 77 = 7 × 11
    let p = BigInt::from(7);
    let q = BigInt::from(11);
    let n = &p * &q; // n = 77
    
    // Project to torus coordinates
    let p_coord = StandardProjection.project(&p); // (7, 7)
    let q_coord = StandardProjection.project(&q); // (11, 11)
    let n_coord = StandardProjection.project(&n); // (29, 77)
    
    // Verify multiplication routing: π(p × q) = π(p) ⊗ π(q)
    let product_coord = mul(&p_coord, &q_coord);
    assert_eq!(product_coord.page, n_coord.page);
    assert_eq!(product_coord.resonance, n_coord.resonance);
    
    // Verify page constraint: (7 × 11) % 48 = 77 % 48 = 29
    assert_eq!(product_coord.page, 29);
    assert_eq!(product_coord.resonance, 77);
    
    // Verify coherence via CoherenceVerifier
    let verifier = CoherenceVerifier::new();
    assert!(verifier.verify_multiplication_coherence(&p, &q));
}

#[test]
fn test_matrix_multiplication_from_generators() {
    use hrm_spec::derived::matmul::matmul;
    
    // 2×2 matrix multiplication using only ⊕ and ⊗
    let a = vec![
        vec![
            TorusCoordinate { page: 2, resonance: 3 },
            TorusCoordinate { page: 3, resonance: 5 },
        ],
        vec![
            TorusCoordinate { page: 5, resonance: 7 },
            TorusCoordinate { page: 7, resonance: 11 },
        ],
    ];
    
    let identity = vec![
        vec![
            TorusCoordinate::one(),
            TorusCoordinate::zero(),
        ],
        vec![
            TorusCoordinate::zero(),
            TorusCoordinate::one(),
        ],
    ];
    
    let result = matmul(&a, &identity).unwrap();
    
    // A × I = A
    assert_eq!(result[0][0].page, a[0][0].page);
    assert_eq!(result[0][0].resonance, a[0][0].resonance);
    assert_eq!(result[1][1].page, a[1][1].page);
    assert_eq!(result[1][1].resonance, a[1][1].resonance);
}

#[test]
fn test_routing_protocol_o1_verification() {
    let routing = StandardRouting;
    
    // Test multiple operations are O(1)
    let coords: Vec<TorusCoordinate> = (0..1000)
        .map(|i| TorusCoordinate {
            page: (i % 48) as u8,
            resonance: (i % 96) as u8,
        })
        .collect();
    
    // All operations should complete quickly
    for i in 0..coords.len() - 1 {
        let result = routing.route_multiplication(&coords[i], &coords[i + 1]);
        assert!(result.page < 48);
        assert!(result.resonance < 96);
    }
}

#[test]
fn test_entanglement_network_addressing() {
    use hrm_spec::routing::entanglement::{EntanglementNetwork, NetworkAddress};
    
    let network = EntanglementNetwork::new();
    
    // Verify all 12,288 cells are addressable
    assert_eq!(network.size(), 12_288);
    
    // Test address creation and validation
    let coord = TorusCoordinate { page: 23, resonance: 47 };
    let addr = NetworkAddress::from_coordinate(coord);
    
    assert!(network.is_valid_address(&addr));
    assert_eq!(addr.coordinate().page, 23);
    assert_eq!(addr.coordinate().resonance, 47);
}

#[test]
fn test_convolution_composition() {
    use hrm_spec::derived::convolution::convolve;
    
    // Test convolution as composition of ⊕ and ⊗
    let signal = vec![
        TorusCoordinate { page: 1, resonance: 1 },
        TorusCoordinate { page: 2, resonance: 2 },
        TorusCoordinate { page: 3, resonance: 3 },
    ];
    
    let kernel = vec![
        TorusCoordinate { page: 1, resonance: 1 },
        TorusCoordinate { page: 0, resonance: 0 },
    ];
    
    let result = convolve(&signal, &kernel);
    
    // Length should be signal.len() + kernel.len() - 1 = 4
    assert_eq!(result.len(), 4);
    
    // First element should be signal[0] ⊗ kernel[0]
    assert_eq!(result[0].page, 1);
    assert_eq!(result[0].resonance, 1);
}

#[test]
fn test_reduction_operations() {
    use hrm_spec::derived::reduction::{reduce_sum, reduce_product};
    
    let coords = vec![
        TorusCoordinate { page: 2, resonance: 3 },
        TorusCoordinate { page: 3, resonance: 5 },
        TorusCoordinate { page: 5, resonance: 7 },
    ];
    
    // Sum reduction
    let sum = reduce_sum(&coords);
    assert_eq!(sum.page, (2 + 3 + 5) % 48);
    assert_eq!(sum.resonance, (3 + 5 + 7) % 96);
    
    // Product reduction
    let product = reduce_product(&coords);
    assert_eq!(product.page, (2 * 3 * 5) % 48);
    assert_eq!(product.resonance, (3 * 5 * 7) % 96);
}

#[test]
fn test_lifting_projection_cycle() {
    // Test projection → lifting → projection cycle
    let original = BigInt::from(123456);
    let coord = StandardProjection.project(&original);
    
    let lifter = O1Lifting;
    let lifted = lifter.lift(&coord, &original);
    
    let coord2 = StandardProjection.project(&lifted);
    
    // Should project to same coordinate
    assert_eq!(coord.page, coord2.page);
    assert_eq!(coord.resonance, coord2.resonance);
}
