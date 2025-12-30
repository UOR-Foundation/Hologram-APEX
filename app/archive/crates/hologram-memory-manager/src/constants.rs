//! Constants and Default Values
//!
//! This module defines constants used throughout the processor crate.
//! All values are chosen based on practical considerations and documented
//! to avoid arbitrary limits.

/// Default maximum primorial levels for chunking
///
/// The primorial sequence is: 1, 2, 6, 30, 210, 2310, 30030, 510510, 9699690, 223092870, ...
///
/// At different levels:
/// - 6 levels: max chunk size = 2,310 bytes (~2 KB)
/// - 7 levels: max chunk size = 30,030 bytes (~29 KB)
/// - 8 levels: max chunk size = 510,510 bytes (~498 KB)
/// - 9 levels: max chunk size = 9,699,690 bytes (~9.2 MB)
/// - 10 levels: max chunk size = 223,092,870 bytes (~212 MB)
///
/// **Why 8 as default?**
/// - Provides reasonable chunk sizes up to ~500 KB
/// - Balances memory usage with gauge construction granularity
/// - Suitable for most data processing workloads
/// - Can be easily configured higher or lower as needed
///
/// **Not a hard limit** - users can configure any value via:
/// - `StreamProcessor::with_chunk_levels(n)`
/// - `CircuitStreamCompiler::with_levels(n)`
/// - `PrimordialChunker::new(n)`
pub const DEFAULT_MAX_CHUNK_LEVELS: usize = 8;

/// Maximum supported prime for primorial factorization
///
/// This determines how far the primorial sequence can extend.
/// Currently supports primes up to 53 (the 16th prime).
///
/// The 16th primorial (53#) = 2 × 3 × 5 × 7 × 11 × 13 × 17 × 19 × 23 × 29 × 31 × 37 × 41 × 43 × 47 × 53
/// = 16,294,579,238,595,022,365,967,380
///
/// This is well beyond u64::MAX, so practical usage is limited by data size,
/// not by prime table size.
pub const MAX_SUPPORTED_PRIME: u8 = 53;
