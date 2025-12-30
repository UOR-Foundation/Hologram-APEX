//! Domain Heads Example
//!
//! Demonstrates using domain heads to extract different modalities
//! from the same universal memory pool.
//!
//! Domain heads act as "mediatypes" (like MIME types) that provide
//! different interpretations of the same embedded data.

use hologram_memory_manager::{Modality, RawDomainHead, StreamProcessor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  Domain Heads as Mediatypes Example                  ║");
    println!("╚═══════════════════════════════════════════════════════╝\n");

    // Step 1: Create stream processor and register domain heads
    println!("Step 1: Setting up stream processor");
    let mut processor = StreamProcessor::new();

    processor.register_domain_head(RawDomainHead);

    let mediatypes = processor.list_mediatypes();
    println!("  Registered {} domain head(s):", mediatypes.len());
    for mediatype in &mediatypes {
        println!("    - {}", mediatype);
    }

    // Step 2: Process input data
    println!("\nStep 2: Processing input data");
    let input: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
    println!("  Input size: {} bytes", input.len());

    let context = processor.process(input.clone())?;
    println!("  Embedded into {} blocks", context.pool.len());
    println!("  Gauges constructed: {}", context.gauges_count);

    // Step 3: Extract modality
    println!("\nStep 3: Extracting modality from embedded pool\n");

    // Extract: Raw data reconstruction
    println!("  Domain Head: application/octet-stream (RawDomainHead)");
    let raw = processor.extract_modality(&context, "application/octet-stream")?;

    match raw {
        Modality::Raw(data) => {
            println!("    Reconstructed: {} bytes", data.len());
            println!(
                "    Data integrity: {}",
                if data == input {
                    "✅ EXACT MATCH"
                } else {
                    "❌ MISMATCH"
                }
            );
            println!("    First 10 bytes: {:?}", &data[0..10.min(data.len())]);
            if data.len() > 10 {
                println!("    Last 10 bytes:  {:?}", &data[data.len() - 10.min(data.len())..]);
            }

            // Verify perfect reconstruction
            assert_eq!(data.len(), input.len(), "Size mismatch");
            assert_eq!(data, input, "Data mismatch");
        }
        _ => panic!("Unexpected modality"),
    }

    // Summary
    println!("\n╔═══════════════════════════════════════════════════════╗");
    println!("║  Summary                                              ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!();
    println!("✅ Single memory pool embedded once");
    println!("✅ RawDomainHead reconstructs original data perfectly");
    println!();
    println!("Domain Head Architecture:");
    println!("  - RawDomainHead: Reconstruction (implemented)");
    println!("  - Future: ShorsDomainHead for semiprime factoring");
    println!("  - Future: FFTDomainHead for frequency spectrum");
    println!("  - Future: CompressionDomainHead for compressed output");
    println!();
    println!("Key insight: Same pool, extensible interpretations!");
    println!("Domain heads act as 'mediatypes' (like MIME types).");
    println!();

    Ok(())
}
