//! Primorial Sequence Generation
//!
//! Primorials are products of consecutive primes:
//! - 2# = 2
//! - 3# = 2 × 3 = 6
//! - 5# = 2 × 3 × 5 = 30
//! - 7# = 2 × 3 × 5 × 7 = 210
//! - 11# = 2 × 3 × 5 × 7 × 11 = 2,310
//! - 13# = 2 × 3 × 5 × 7 × 11 × 13 = 30,030
//!
//! These values determine chunk sizes and gauge structures.

/// Generate primorial sequence up to a limit
pub fn generate_primorial_sequence(max_value: u64) -> Vec<u64> {
    let mut sequence = Vec::new();
    let mut primorial = 1u64;
    let mut prime = 2u8;

    while primorial <= max_value {
        sequence.push(primorial);

        // Compute next primorial by multiplying with current prime
        if let Some(next) = primorial.checked_mul(prime as u64) {
            primorial = next;
        } else {
            // Overflow - stop
            break;
        }

        // Get next prime for the next iteration
        prime = next_prime(prime);
    }

    sequence
}

/// Generate N primorials
pub fn generate_n_primorials(n: usize) -> Vec<u64> {
    let mut sequence = Vec::with_capacity(n);
    let mut primorial = 1u64;
    let mut prime = 2u8;

    for _ in 0..n {
        sequence.push(primorial);

        // Compute next primorial by multiplying with current prime
        if let Some(next) = primorial.checked_mul(prime as u64) {
            primorial = next;
        } else {
            // Overflow - stop
            break;
        }

        // Get next prime for the next iteration
        prime = next_prime(prime);
    }

    sequence
}

/// Extract primes from primorial value
///
/// Given a primorial P#, returns the list of primes up to P.
///
/// # Examples
///
/// ```
/// # use hologram_memory_manager::chunking::primorial::factor_primorial;
/// assert_eq!(factor_primorial(2), vec![2]);
/// assert_eq!(factor_primorial(6), vec![2, 3]);
/// assert_eq!(factor_primorial(30), vec![2, 3, 5]);
/// assert_eq!(factor_primorial(210), vec![2, 3, 5, 7]);
/// ```
pub fn factor_primorial(primorial: u64) -> Vec<u8> {
    // Known primorials for fast lookup
    match primorial {
        1 => vec![], // No primes yet (identity)
        2 => vec![2],
        6 => vec![2, 3],
        30 => vec![2, 3, 5],
        210 => vec![2, 3, 5, 7],
        2310 => vec![2, 3, 5, 7, 11],
        30030 => vec![2, 3, 5, 7, 11, 13],
        510510 => vec![2, 3, 5, 7, 11, 13, 17],
        9699690 => vec![2, 3, 5, 7, 11, 13, 17, 19],
        223092870 => vec![2, 3, 5, 7, 11, 13, 17, 19, 23],
        _ => {
            // Compute by trial division
            compute_prime_factors(primorial)
        }
    }
}

/// Compute prime factors of a number
fn compute_prime_factors(mut n: u64) -> Vec<u8> {
    let mut factors = Vec::new();
    let mut prime = 2u8;

    while n > 1 && prime <= 53 {
        if n.is_multiple_of(prime as u64) {
            factors.push(prime);
            while n.is_multiple_of(prime as u64) {
                n /= prime as u64;
            }
        }
        let next = next_prime(prime);
        if next == prime {
            // Reached the limit of our prime table
            break;
        }
        prime = next;
    }

    factors
}

/// Get next prime after p
///
/// Returns the next prime in the sequence up to 53 (the 16th prime).
/// Beyond 53, returns p to signal we've reached the limit.
fn next_prime(p: u8) -> u8 {
    match p {
        2 => 3,
        3 => 5,
        5 => 7,
        7 => 11,
        11 => 13,
        13 => 17,
        17 => 19,
        19 => 23,
        23 => 29,
        29 => 31,
        31 => 37,
        37 => 41,
        41 => 43,
        43 => 47,
        47 => 53,
        // Beyond 53 (the 16th prime), we've reached our limit
        // Return p to signal no more primes available
        _ => p,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primorial_sequence() {
        let seq = generate_n_primorials(8);
        assert_eq!(seq, vec![1, 2, 6, 30, 210, 2310, 30030, 510510]);
    }

    #[test]
    fn test_factor_primorial() {
        assert_eq!(factor_primorial(1), Vec::<u8>::new());
        assert_eq!(factor_primorial(2), vec![2]);
        assert_eq!(factor_primorial(6), vec![2, 3]);
        assert_eq!(factor_primorial(30), vec![2, 3, 5]);
        assert_eq!(factor_primorial(210), vec![2, 3, 5, 7]);
        assert_eq!(factor_primorial(2310), vec![2, 3, 5, 7, 11]);
    }

    #[test]
    fn test_primorial_limit() {
        let seq = generate_primorial_sequence(10000);
        // Should include: 1, 2, 6, 30, 210, 2310
        assert!(seq.contains(&2310));
        // Should NOT include 30030 (> 10000)
        assert!(!seq.contains(&30030));
    }
}
