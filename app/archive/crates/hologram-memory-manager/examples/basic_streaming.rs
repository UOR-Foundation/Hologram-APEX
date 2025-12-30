//! Basic Streaming Example
//!
//! Demonstrates basic stream processing with automatic gauge construction.

use hologram_memory_manager::{Stream, StreamContext, DEFAULT_MAX_CHUNK_LEVELS};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  Basic Stream Processing Example                     ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Step 1: Create input data
    let input: Vec<u8> = (0..500).map(|i| (i % 256) as u8).collect();
    println!("Step 1: Created {} bytes of input data", input.len());

    // Step 2: Create stream
    let stream = Stream::new(input);
    println!("Step 2: Created stream with {} elements", stream.len());

    // Step 3: Chunk with automatic gauge construction
    println!(
        "\nStep 3: Chunking with primorial sequence ({} levels - default)",
        DEFAULT_MAX_CHUNK_LEVELS
    );
    let chunked = stream.chunk(DEFAULT_MAX_CHUNK_LEVELS)?;
    println!("  Created {} chunks", chunked.len());

    // Show chunk details
    for (i, chunk) in chunked.chunks().iter().enumerate().take(5) {
        println!(
            "  Chunk {}: primorial={}, gauge={}, size={}",
            i,
            chunk.primorial,
            chunk.gauge_name(),
            chunk.len()
        );
    }
    if chunked.len() > 5 {
        println!("  ... and {} more chunks", chunked.len() - 5);
    }

    // Step 4: Embed into memory pool
    println!("\nStep 4: Embedding into universal memory pool");
    let context: StreamContext = chunked.embed()?;

    println!("  Total bytes: {}", context.total_bytes);
    println!("  Gauges constructed: {}", context.gauges_count);
    println!("  Blocks in pool: {}", context.pool.len());

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Summary                                              ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!();
    println!("✓ Input data chunked via primorial sequence");
    println!("✓ Gauges constructed automatically from chunk sizes");
    println!("✓ Data embedded into universal memory pool");
    println!();
    println!("Key insight: Chunking IS the gauge generator!");
    println!();

    Ok(())
}
