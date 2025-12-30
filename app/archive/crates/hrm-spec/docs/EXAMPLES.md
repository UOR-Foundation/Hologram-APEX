# MoonshineHRM Examples

This document provides worked examples demonstrating the MoonshineHRM specification.

## Example 1: Basic Projection

**Problem**: Project integer 42 to torus coordinates.

**Solution**:
```rust
use hrm_spec::prelude::*;

let n = BigInt::from(42);
let coord = StandardProjection.project(&n);

// coord.page = 42 % 48 = 42
// coord.resonance = 42 % 96 = 42
```

**Result**: $(42, 42)$ in $T^2$

## Example 2: Addition Coherence

**Problem**: Verify $\pi(37 + 53) = \pi(37) \oplus \pi(53)$

**Solution**:
```rust
let a = BigInt::from(37);
let b = BigInt::from(53);

// Direct projection of sum
let sum_projected = StandardProjection.project(&(a.clone() + b.clone()));
// sum_projected = (90 % 48, 90 % 96) = (42, 90)

// Project then add
let a_proj = StandardProjection.project(&a);  // (37, 37)
let b_proj = StandardProjection.project(&b);  // (53, 53)
let add_projected = add(&a_proj, &b_proj);
// add_projected = ((37+53) % 48, (37+53) % 96) = (42, 90)

assert_eq!(sum_projected, add_projected);
```

**Verification**: ✓ Coherence holds

## Example 3: Factorization Routing

**Problem**: Factor 15 = 3 × 5 and verify routing protocol.

**Solution**:
```rust
let p = BigInt::from(3);
let q = BigInt::from(5);
let n = BigInt::from(15);

// Project factors
let p_coord = StandardProjection.project(&p);  // (3, 3)
let q_coord = StandardProjection.project(&q);  // (5, 5)

// Multiply in torus space
let product_coord = mul(&p_coord, &q_coord);
// product_coord = ((3×5) % 48, (3×5) % 96) = (15, 15)

// Project product directly
let n_coord = StandardProjection.project(&n);  // (15, 15)

assert_eq!(product_coord, n_coord);
```

**Result**: Page constraint verified: $3 \times 5 \equiv 15 \pmod{48}$ ✓

## Example 4: Large Factorization

**Problem**: Factor 77 = 7 × 11 and verify multiplicative constraint.

**Solution**:
```rust
let p = BigInt::from(7);
let q = BigInt::from(11);

let p_coord = StandardProjection.project(&p);   // (7, 7)
let q_coord = StandardProjection.project(&q);   // (11, 11)
let product = mul(&p_coord, &q_coord);
// product = ((7×11) % 48, (7×11) % 96) = (77 % 48, 77) = (29, 77)

let n_coord = StandardProjection.project(&BigInt::from(77));  // (29, 77)

assert_eq!(product, n_coord);
```

**Result**: 
- Page: $(7 \times 11) \bmod 48 = 77 \bmod 48 = 29$ ✓
- Resonance: $(7 \times 11) \bmod 96 = 77$ ✓

## Example 5: Scalar Multiplication

**Problem**: Compute $7 \odot (3, 5)$

**Solution**:
```rust
let k = BigInt::from(7);
let coord = TorusCoordinate { page: 3, resonance: 5 };

let result = scalar_mul_optimized(&k, &coord);
// result = ((7×3) % 48, (7×5) % 96) = (21, 35)
```

**Verification**:
```rust
// Verify via repeated addition
let mut sum = TorusCoordinate::zero();
for _ in 0..7 {
    sum = add(&sum, &coord);
}
assert_eq!(sum, result);  // (21, 35)
```

## Example 6: Matrix Multiplication

**Problem**: Multiply $2 \times 2$ matrices on torus.

**Solution**:
```rust
use hrm_spec::derived::matmul::matmul;

let a = vec![
    vec![
        TorusCoordinate { page: 1, resonance: 2 },
        TorusCoordinate { page: 3, resonance: 4 },
    ],
    vec![
        TorusCoordinate { page: 5, resonance: 6 },
        TorusCoordinate { page: 7, resonance: 8 },
    ],
];

let b = vec![
    vec![
        TorusCoordinate { page: 2, resonance: 1 },
        TorusCoordinate { page: 4, resonance: 3 },
    ],
    vec![
        TorusCoordinate { page: 6, resonance: 5 },
        TorusCoordinate { page: 8, resonance: 7 },
    ],
];

let result = matmul(&a, &b).unwrap();

// result[0][0] = (a[0][0] ⊗ b[0][0]) ⊕ (a[0][1] ⊗ b[1][0])
//              = ((1×2) % 48, (2×1) % 96) ⊕ ((3×6) % 48, (4×5) % 96)
//              = (2, 2) ⊕ (18, 20)
//              = (20, 22)
```

**Result**: $2 \times 2$ result matrix computed via $\oplus$ and $\otimes$ only

## Example 7: Convolution

**Problem**: Convolve signal $[1, 2, 3]$ with kernel $[1, 0]$

**Solution**:
```rust
use hrm_spec::derived::convolution::convolve;

let signal = vec![
    TorusCoordinate { page: 1, resonance: 1 },
    TorusCoordinate { page: 2, resonance: 2 },
    TorusCoordinate { page: 3, resonance: 3 },
];

let kernel = vec![
    TorusCoordinate::one(),   // (1, 1)
    TorusCoordinate::zero(),  // (0, 0)
];

let result = convolve(&signal, &kernel);
// Length: 3 + 2 - 1 = 4
// result[0] = signal[0] ⊗ kernel[0] = (1, 1)
// result[1] = (signal[1] ⊗ kernel[0]) ⊕ (signal[0] ⊗ kernel[1]) = (2, 2) ⊕ (0, 0) = (2, 2)
// result[2] = (signal[2] ⊗ kernel[0]) ⊕ (signal[1] ⊗ kernel[1]) = (3, 3) ⊕ (0, 0) = (3, 3)
// result[3] = signal[2] ⊗ kernel[1] = (0, 0)
```

## Example 8: Group Inverse

**Problem**: Find inverse of $(5, 10)$

**Solution**:
```rust
let coord = TorusCoordinate { page: 5, resonance: 10 };
let inv = coord.inverse();
// inv.page = (48 - 5) % 48 = 43
// inv.resonance = (96 - 10) % 96 = 86

let sum = add(&coord, &inv);
// sum = ((5+43) % 48, (10+86) % 96) = (0, 0)
```

**Verification**: $(5, 10) \oplus (43, 86) = (0, 0)$ ✓

## Example 9: Lifting

**Problem**: Lift $(29, 77)$ with hint 1000.

**Solution**:
```rust
let coord = TorusCoordinate { page: 29, resonance: 77 };
let hint = BigInt::from(1000);

let lifted = O1Lifting.lift(&coord, &hint);
// hint % 96 = 1000 % 96 = 40
// offset = 40 - 77 = -37 → wrap: 96 + (-37) = 59
// lifted = 1000 - 59 = 941

// Verify: π(941) = (941 % 48, 941 % 96) = (29, 77) ✓
```

## Example 10: Channel Composition

**Problem**: Compose routing channels $A \to B \to C$

**Solution**:
```rust
use hrm_spec::routing::channels::{RoutingChannel, compose_channels};

let a = TorusCoordinate { page: 1, resonance: 2 };
let b = TorusCoordinate { page: 3, resonance: 5 };
let c = TorusCoordinate { page: 7, resonance: 11 };

let ch1 = RoutingChannel::new(a.clone(), b.clone());
let ch2 = RoutingChannel::new(b, c.clone());

let composed = ch1.compose(&ch2).unwrap();
// composed.source = a
// composed.target = c
```

**Result**: Direct channel from $A$ to $C$ via composition ✓

## Example 11: Distributivity

**Problem**: Verify $a \otimes (b \oplus c) = (a \otimes b) \oplus (a \otimes c)$

**Solution**:
```rust
let a = TorusCoordinate { page: 3, resonance: 5 };
let b = TorusCoordinate { page: 7, resonance: 11 };
let c = TorusCoordinate { page: 13, resonance: 17 };

// LHS: a ⊗ (b ⊕ c)
let b_plus_c = add(&b, &c);  // (20, 28)
let lhs = mul(&a, &b_plus_c);  // ((3×20) % 48, (5×28) % 96) = (12, 44)

// RHS: (a ⊗ b) ⊕ (a ⊗ c)
let a_times_b = mul(&a, &b);  // (21, 55)
let a_times_c = mul(&a, &c);  // (39, 85)
let rhs = add(&a_times_b, &a_times_c);  // ((21+39) % 48, (55+85) % 96) = (12, 44)

assert_eq!(lhs, rhs);
```

**Verification**: Distributivity holds ✓

## Example 12: Reduction

**Problem**: Compute sum reduction of $[(1,2), (3,5), (7,11)]$

**Solution**:
```rust
use hrm_spec::derived::reduction::reduce_sum;

let coords = vec![
    TorusCoordinate { page: 1, resonance: 2 },
    TorusCoordinate { page: 3, resonance: 5 },
    TorusCoordinate { page: 7, resonance: 11 },
];

let sum = reduce_sum(&coords);
// sum = (1, 2) ⊕ (3, 5) ⊕ (7, 11)
//     = (4, 7) ⊕ (7, 11)
//     = (11, 18)
```

**Result**: $(11, 18)$

---

**Document Version**: 1.0  
**Date**: 2024  
**Example Count**: 12 worked examples  
**Status**: Complete with verification
