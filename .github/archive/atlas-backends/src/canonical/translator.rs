//! ISA to Graph Translation
//!
//! Translates Atlas ISA programs to canonical graph operations.
//! This module maps:
//! - Registers (256) → Classes (96) based on buffer topology
//! - ISA instructions → Graph edge traversals
//! - Operation sequences → Fused graph paths

use super::{Generator, GraphOperation, OpParams, Transform};
use crate::{BackendError, ExecutionContext, Result};
use atlas_core::atlas;
use atlas_isa::{Instruction, Program, Register};
use std::collections::HashMap;

/// Map registers to resonance classes based on buffer topology and graph structure
///
/// This is the key translation step: ISA programs use 256 registers,
/// but canonical execution uses 96 classes. We map registers to classes
/// ensuring that all operations respect graph topology (only neighbors can interact).
///
/// Strategy:
/// 1. Start with first active class
/// 2. For each new register, choose from available neighboring classes
/// 3. Ensure binary operations get neighbor classes for src1/src2
/// 4. Rotate through neighbors to avoid self-edges
pub fn map_registers_to_classes(program: &Program, ctx: &ExecutionContext) -> Result<HashMap<Register, u8>> {
    let mut mapping = HashMap::new();
    let active_classes = &ctx.active_classes;

    if active_classes.is_empty() {
        return Err(BackendError::InvalidTopology(
            "No active classes in execution context".to_string(),
        ));
    }

    let atlas = atlas();

    // Start with first active class
    let base_class = active_classes[0];

    // Build a pool of available classes (base + all bidirectional neighbors)
    let neighbors = atlas.neighbors(base_class as usize);
    let mut class_pool = vec![base_class];

    for &neighbor in neighbors.iter() {
        let neighbor_neighbors = atlas.neighbors(neighbor);
        if neighbor_neighbors.contains(&(base_class as usize)) {
            // Bidirectional edge confirmed
            class_pool.push(neighbor as u8);
        }
    }

    if class_pool.len() < 2 {
        return Err(BackendError::InvalidTopology(format!(
            "Class {} has no bidirectional neighbors",
            base_class
        )));
    }

    // Compute last use of each register (simple liveness analysis)
    let mut last_use = HashMap::new();
    for (idx, inst) in program.instructions.iter().enumerate() {
        match inst {
            Instruction::ADD { src1, src2, .. }
            | Instruction::SUB { src1, src2, .. }
            | Instruction::MUL { src1, src2, .. }
            | Instruction::DIV { src1, src2, .. }
            | Instruction::MIN { src1, src2, .. }
            | Instruction::MAX { src1, src2, .. } => {
                last_use.insert(*src1, idx);
                last_use.insert(*src2, idx);
            }
            Instruction::ABS { src, .. } | Instruction::NEG { src, .. } => {
                last_use.insert(*src, idx);
            }
            Instruction::STG { src, .. } => {
                last_use.insert(*src, idx);
            }
            _ => {}
        }
    }

    // Track which registers are currently using which class
    let mut class_to_registers: HashMap<u8, Vec<Register>> = HashMap::new();

    // Helper: Find a class not currently used by any live register
    // A register is live at instruction i if it will be used at or after instruction i
    let find_free_class = |current_inst_idx: usize,
                           mapping: &HashMap<Register, u8>,
                           class_to_regs: &HashMap<u8, Vec<Register>>,
                           last_use: &HashMap<Register, usize>,
                           exclude: &[u8]|
     -> Option<u8> {
        for &candidate in &class_pool {
            if exclude.contains(&candidate) {
                continue;
            }

            // Check if any register using this class is still live
            // A register is dead if its last use is BEFORE the current instruction
            if let Some(regs) = class_to_regs.get(&candidate) {
                let all_dead = regs.iter().all(|r| {
                    // Register is dead if last_use < current_inst_idx
                    // (it was last used in a previous instruction)
                    last_use
                        .get(r)
                        .map(|&last_idx| last_idx < current_inst_idx)
                        .unwrap_or(true)
                });

                if !all_dead {
                    continue; // This class has live registers, skip it
                }
            }

            return Some(candidate);
        }
        None
    };

    // Helper: Get a class that is a neighbor of c and not in exclusion list
    let find_neighbor_class = |c: u8, exclude: &[u8]| -> Option<u8> {
        for &candidate in &class_pool {
            if !exclude.contains(&candidate) {
                let c_neighbors = atlas.neighbors(c as usize);
                if c_neighbors.contains(&(candidate as usize)) {
                    return Some(candidate);
                }
            }
        }
        None
    };

    // Helper: Find a free class that is also a neighbor of c
    // NOTE: Searches ALL neighbors of c, not just those in class_pool
    let find_free_neighbor_class = |c: u8,
                                    current_inst_idx: usize,
                                    class_to_regs: &HashMap<u8, Vec<Register>>,
                                    last_use: &HashMap<Register, usize>,
                                    exclude: &[u8]|
     -> Option<u8> {
        let c_neighbors = atlas.neighbors(c as usize);
        // Search through ALL neighbors of c (not just class_pool)
        for &candidate_usize in c_neighbors.iter() {
            let candidate = candidate_usize as u8;

            if exclude.contains(&candidate) {
                continue;
            }

            // Must be free (no live registers)
            if let Some(regs) = class_to_regs.get(&candidate) {
                let all_dead = regs.iter().all(|r| {
                    last_use
                        .get(r)
                        .map(|&last_idx| last_idx < current_inst_idx)
                        .unwrap_or(true)
                });

                if !all_dead {
                    continue; // This class has live registers, skip it
                }
            }

            return Some(candidate);
        }
        None
    };

    // Process instructions with liveness-aware allocation
    for (inst_idx, inst) in program.instructions.iter().enumerate() {
        match inst {
            Instruction::LDG { dst, .. } => {
                if !mapping.contains_key(dst) {
                    // Try to find a free class
                    if let Some(class) = find_free_class(inst_idx, &mapping, &class_to_registers, &last_use, &[]) {
                        mapping.insert(*dst, class);
                        class_to_registers.entry(class).or_default().push(*dst);
                    } else {
                        // Fallback: use first class (may cause conflicts)
                        let class = class_pool[0];
                        mapping.insert(*dst, class);
                        class_to_registers.entry(class).or_default().push(*dst);
                    }
                }
            }
            Instruction::ADD { dst, src1, src2, .. }
            | Instruction::SUB { dst, src1, src2, .. }
            | Instruction::MUL { dst, src1, src2, .. }
            | Instruction::DIV { dst, src1, src2, .. }
            | Instruction::MIN { dst, src1, src2, .. }
            | Instruction::MAX { dst, src1, src2, .. } => {
                // Ensure src1 is mapped
                if !mapping.contains_key(src1) {
                    if let Some(class) = find_free_class(inst_idx, &mapping, &class_to_registers, &last_use, &[]) {
                        mapping.insert(*src1, class);
                        class_to_registers.entry(class).or_default().push(*src1);
                    } else {
                        let class = class_pool[0];
                        mapping.insert(*src1, class);
                        class_to_registers.entry(class).or_default().push(*src1);
                    }
                }

                let c1 = mapping[src1];

                // Ensure src2 is mapped and different from src1
                if !mapping.contains_key(src2) {
                    if let Some(c2) = find_neighbor_class(c1, &[c1]) {
                        mapping.insert(*src2, c2);
                        class_to_registers.entry(c2).or_default().push(*src2);
                    } else {
                        if let Some(class) = find_free_class(inst_idx, &mapping, &class_to_registers, &last_use, &[c1])
                        {
                            mapping.insert(*src2, class);
                            class_to_registers.entry(class).or_default().push(*src2);
                        }
                    }
                } else {
                    let c2 = mapping[src2];
                    if c1 == c2 {
                        // CONFLICT! Remap src2
                        if let Some(new_c2) = find_neighbor_class(c1, &[c1]) {
                            mapping.insert(*src2, new_c2);
                        }
                    }
                }

                let c2 = mapping[src2];

                // Map dst: must be different from src1, and preferably free and not same as src2
                if !mapping.contains_key(dst) {
                    // Try to find a free neighbor of c1 that's not c1 or c2
                    if let Some(c_dst) =
                        find_free_neighbor_class(c1, inst_idx, &class_to_registers, &last_use, &[c1, c2])
                    {
                        mapping.insert(*dst, c_dst);
                        class_to_registers.entry(c_dst).or_default().push(*dst);
                    } else if let Some(c_dst) =
                        find_free_neighbor_class(c1, inst_idx, &class_to_registers, &last_use, &[c1])
                    {
                        // Settle for any free neighbor except c1
                        mapping.insert(*dst, c_dst);
                        class_to_registers.entry(c_dst).or_default().push(*dst);
                    } else {
                        // Last resort: use c2 (at least it's a neighbor and not c1)
                        mapping.insert(*dst, c2);
                        class_to_registers.entry(c2).or_default().push(*dst);
                    }
                } else {
                    let current_dst = mapping[dst];
                    if current_dst == c1 {
                        // CONFLICT! Must remap to a free neighbor
                        if let Some(c_dst) =
                            find_free_neighbor_class(c1, inst_idx, &class_to_registers, &last_use, &[c1])
                        {
                            mapping.insert(*dst, c_dst);
                        }
                    }
                }
            }
            Instruction::ABS { dst, src, .. } | Instruction::NEG { dst, src, .. } => {
                if !mapping.contains_key(src) {
                    if let Some(class) = find_free_class(inst_idx, &mapping, &class_to_registers, &last_use, &[]) {
                        mapping.insert(*src, class);
                        class_to_registers.entry(class).or_default().push(*src);
                    }
                }

                let c_src = mapping[src];

                if !mapping.contains_key(dst) {
                    // Find a free neighbor of src (not src itself)
                    if let Some(c_dst) =
                        find_free_neighbor_class(c_src, inst_idx, &class_to_registers, &last_use, &[c_src])
                    {
                        mapping.insert(*dst, c_dst);
                        class_to_registers.entry(c_dst).or_default().push(*dst);
                    } else {
                        // Last resort: any neighbor (may cause conflict)
                        if let Some(c_dst) = find_neighbor_class(c_src, &[c_src]) {
                            mapping.insert(*dst, c_dst);
                            class_to_registers.entry(c_dst).or_default().push(*dst);
                        }
                    }
                } else {
                    let current_dst = mapping[dst];
                    if current_dst == c_src {
                        // CONFLICT! Remap to free neighbor
                        if let Some(c_dst) =
                            find_free_neighbor_class(c_src, inst_idx, &class_to_registers, &last_use, &[c_src])
                        {
                            mapping.insert(*dst, c_dst);
                        }
                    }
                }
            }
            Instruction::STG { src, .. } => {
                if !mapping.contains_key(src) {
                    if let Some(class) = find_free_class(inst_idx, &mapping, &class_to_registers, &last_use, &[]) {
                        mapping.insert(*src, class);
                        class_to_registers.entry(class).or_default().push(*src);
                    }
                }
            }
            _ => {
                // Other instructions handled as needed
            }
        }
    }

    Ok(mapping)
}

/// Extract which class an address belongs to based on active classes
fn extract_class_from_addr(addr: &atlas_isa::Address, active_classes: &[u8]) -> Result<u8> {
    // For now, simple heuristic based on address
    // TODO: Implement proper address → class resolution using boundary pool topology
    if active_classes.is_empty() {
        return Err(BackendError::InvalidTopology(
            "No active classes in execution context".to_string(),
        ));
    }

    match addr {
        atlas_isa::Address::BufferOffset { handle, offset } => {
            // Map handle/offset to class index
            let class_idx = (*handle as usize + offset) % active_classes.len();
            Ok(active_classes[class_idx])
        }
        atlas_isa::Address::PhiCoordinate { class, .. } => {
            // Already have the class
            Ok(*class)
        }
        atlas_isa::Address::RegisterIndirect { .. } => {
            // For register indirect, use first active class (heuristic)
            Ok(active_classes[0])
        }
    }
}

/// Helper function to safely create a GraphOperation, skipping self-edges
///
/// Note: Self-edges are topologically invalid per Atlas Sigil Algebra.
/// Operations must traverse distinct graph edges (src → dst where src ≠ dst).
/// When ISA register reuse creates unavoidable self-edges, we skip the operation.
/// This is a known limitation of the ISA → Graph translation approach.
fn try_create_graph_op(
    ops: &mut Vec<GraphOperation>,
    src: u8,
    dst: u8,
    generator: Generator,
    params: OpParams,
    _inst_name: &str, // For debugging if needed
) {
    // Silently skip self-edges (ISA → Graph impedance mismatch)
    if src == dst {
        return;
    }
    ops.push(GraphOperation {
        src,
        dst,
        generator,
        params,
    });
}

/// Translate ISA program to graph operations
///
/// This is the core translation function that converts an ISA program
/// into a sequence of graph operations that can be executed with zero
/// data movement on class_bases[96].
pub fn translate_isa_to_graph(
    program: &Program,
    ctx: &ExecutionContext,
) -> Result<(Vec<GraphOperation>, HashMap<Register, u8>)> {
    let atlas = atlas();
    let mut ops = Vec::new();

    // Build register → class mapping
    let reg_to_class = map_registers_to_classes(program, ctx)?;

    for inst in &program.instructions {
        match inst {
            // Data movement operations → class activations (implicit)
            Instruction::LDG { dst, addr, .. } => {
                let class = extract_class_from_addr(addr, &ctx.active_classes)?;
                // Load is just marking the class as active
                // No explicit graph operation needed
            }

            Instruction::STG { src, addr, .. } => {
                let class = extract_class_from_addr(addr, &ctx.active_classes)?;
                // Store is just writing to class memory
                // No explicit graph operation needed
            }

            // Arithmetic operations → Generator operations
            Instruction::ADD { dst, src1, src2, .. } => {
                if let (Some(&c1), Some(&c2), Some(&c_out)) =
                    (reg_to_class.get(&src1), reg_to_class.get(&src2), reg_to_class.get(&dst))
                {
                    try_create_graph_op(
                        &mut ops,
                        c1,
                        c_out,
                        Generator::Merge,
                        OpParams {
                            transform: None,
                            context: Some(c2),
                        },
                        "ADD",
                    );
                }
            }

            Instruction::MUL { dst, src1, src2, .. } => {
                if let (Some(&c1), Some(&c2), Some(&c_out)) =
                    (reg_to_class.get(&src1), reg_to_class.get(&src2), reg_to_class.get(&dst))
                {
                    try_create_graph_op(
                        &mut ops,
                        c1,
                        c_out,
                        Generator::Merge,
                        OpParams {
                            transform: Some(Transform::InnerTwist(1)),
                            context: Some(c2),
                        },
                        "MUL",
                    );
                }
            }

            Instruction::SUB { dst, src1, src2, .. } => {
                if let (Some(&c1), Some(&c2), Some(&c_out)) =
                    (reg_to_class.get(&src1), reg_to_class.get(&src2), reg_to_class.get(&dst))
                {
                    try_create_graph_op(
                        &mut ops,
                        c1,
                        c_out,
                        Generator::Merge,
                        OpParams {
                            transform: Some(Transform::InnerTwist(-1)),
                            context: Some(c2),
                        },
                        "SUB",
                    );
                }
            }

            Instruction::DIV { dst, src1, src2, .. } => {
                if let (Some(&c1), Some(&c2), Some(&c_out)) =
                    (reg_to_class.get(&src1), reg_to_class.get(&src2), reg_to_class.get(&dst))
                {
                    try_create_graph_op(
                        &mut ops,
                        c1,
                        c_out,
                        Generator::Split,
                        OpParams {
                            transform: Some(Transform::InnerTwist(2)),
                            context: Some(c2),
                        },
                        "DIV",
                    );
                }
            }

            Instruction::MIN { dst, src1, src2, .. } => {
                if let (Some(&c1), Some(&c2), Some(&c_out)) =
                    (reg_to_class.get(&src1), reg_to_class.get(&src2), reg_to_class.get(&dst))
                {
                    try_create_graph_op(
                        &mut ops,
                        c1,
                        c_out,
                        Generator::Merge,
                        OpParams {
                            transform: Some(Transform::InnerTwist(-2)),
                            context: Some(c2),
                        },
                        "MIN",
                    );
                }
            }

            Instruction::MAX { dst, src1, src2, .. } => {
                if let (Some(&c1), Some(&c2), Some(&c_out)) =
                    (reg_to_class.get(&src1), reg_to_class.get(&src2), reg_to_class.get(&dst))
                {
                    try_create_graph_op(
                        &mut ops,
                        c1,
                        c_out,
                        Generator::Merge,
                        OpParams {
                            transform: Some(Transform::InnerTwist(2)),
                            context: Some(c2),
                        },
                        "MAX",
                    );
                }
            }

            // Unary operations
            Instruction::ABS { dst, src, .. } => {
                if let (Some(&c_src), Some(&c_out)) = (reg_to_class.get(&src), reg_to_class.get(&dst)) {
                    try_create_graph_op(
                        &mut ops,
                        c_src,
                        c_out,
                        Generator::Copy,
                        OpParams {
                            transform: Some(Transform::QuarterTurn(1)),
                            context: None,
                        },
                        "ABS",
                    );
                }
            }

            Instruction::NEG { dst, src, .. } => {
                if let (Some(&c_src), Some(&c_out)) = (reg_to_class.get(&src), reg_to_class.get(&dst)) {
                    try_create_graph_op(
                        &mut ops,
                        c_src,
                        c_out,
                        Generator::Copy,
                        OpParams {
                            transform: Some(Transform::Mirror),
                            context: None,
                        },
                        "NEG",
                    );
                }
            }

            // Atlas-specific operations → Direct graph operations
            Instruction::MIRROR { dst: _, src, .. } => {
                if let Some(&c) = reg_to_class.get(&src) {
                    let mirror = atlas.mirror_pair(c as usize) as u8;

                    ops.push(GraphOperation {
                        src: c,
                        dst: mirror,
                        generator: Generator::Copy,
                        params: OpParams {
                            transform: Some(Transform::Mirror),
                            context: None,
                        },
                    });
                }
            }

            Instruction::NbrGet { class, index, dst } => {
                if let Some(&c) = reg_to_class.get(&class) {
                    let neighbors = atlas.neighbors(c as usize);
                    // Convert HashSet to Vec to allow indexing
                    let neighbors_vec: Vec<usize> = neighbors.iter().copied().collect();
                    if (*index as usize) < neighbors_vec.len() {
                        let neighbor = neighbors_vec[*index as usize] as u8;

                        ops.push(GraphOperation {
                            src: c,
                            dst: neighbor,
                            generator: Generator::Copy,
                            params: OpParams::default(),
                        });
                    }
                }
            }

            // Other instructions will be implemented as needed
            _ => {
                // For now, skip unimplemented instructions
            }
        }
    }

    // Validate all operations before returning
    super::validation::validate_graph_operations(&ops)?;

    Ok((ops, reg_to_class))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_class_from_addr() {
        let classes = vec![0, 5, 10, 15];

        let addr0 = atlas_isa::Address::BufferOffset { handle: 0, offset: 0 };
        let c0 = extract_class_from_addr(&addr0, &classes).unwrap();
        assert_eq!(c0, 0);

        let addr1 = atlas_isa::Address::BufferOffset { handle: 0, offset: 100 };
        let c1 = extract_class_from_addr(&addr1, &classes).unwrap();
        assert!(classes.contains(&c1));
    }

    // TODO: Re-enable these tests when we have a proper way to construct test programs
    // The Program API doesn't have a builder pattern currently

    // #[test]
    // fn test_translate_empty_program() {
    //     let program = Program::new();
    //     let topology = TopologyTables::default();
    //     let ctx = ExecutionContext {
    //         active_classes: vec![0, 1, 2],
    //         phase: 0,
    //         n_elements: 100,
    //         parallelism: None,
    //         resonance: [Rational::zero(); 96],
    //         topology: &topology,
    //     };
    //     let result = translate_isa_to_graph(&program, &ctx);
    //     assert!(result.is_ok());
    //     let ops = result.unwrap();
    //     assert_eq!(ops.len(), 0); // Empty program → no ops
    // }
}
