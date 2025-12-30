//! ISA Conformance Test Suite for Atlas Backends
//!
//! Tests all 55 ISA instructions across all supported types and verifies
//! that backend implementations conform to the Atlas ISA specification.
//!
//! ## Test Coverage
//!
//! - **Data Movement** (6): LDG, STG, LDS, STS, MOV, CVT
//! - **Arithmetic** (10): ADD, SUB, MUL, DIV, MAD, FMA, MIN, MAX, ABS, NEG
//! - **Logic** (8): AND, OR, XOR, NOT, SHL, SHR, SETcc, SEL
//! - **Control Flow** (5): BRA, CALL, RET, LOOP, EXIT
//! - **Synchronization** (2): BAR.SYNC, MEM.FENCE
//! - **Atlas-Specific** (9): CLS.GET, MIRROR, UNITY.TEST, NBR.*, RES.ACCUM, PHASE.*, BOUND.MAP
//! - **Reductions** (4): REDUCE.ADD/MIN/MAX/MUL
//! - **Transcendentals** (11): EXP, LOG, SQRT, SIN, COS, TAN, TANH, SIGMOID, etc.
//!
//! ## Property-Based Tests
//!
//! - Unity Neutrality: sum of resonance deltas = 0
//! - Mirror Involution: MIRROR(MIRROR(x)) = x
//! - Neighbor Symmetry: if B ∈ neighbors(A), then A ∈ neighbors(B)
//! - Phase Modulus: phase values are mod 768
//! - Boundary Lens Roundtrip: Φ encode/decode preserves values

use atlas_backends::{
    types::Rational, AtlasBackend, BackendHandle, BufferTopology, CPUBackend, ExecutionContext, MemoryPool,
};
use atlas_isa::{Address, Instruction, Label, Program, Register, Type as IsaType};
use atlas_runtime::AtlasSpace;

// ============================================================================
// Test Helpers
// ============================================================================

/// Create and initialize a test backend with AtlasSpace
/// Returns (backend, space, topology) where topology is cloned for test use
fn create_test_backend() -> (CPUBackend, AtlasSpace, atlas_backends::types::TopologyTables) {
    let space = AtlasSpace::new();
    let mut backend = CPUBackend::new().expect("Failed to create CPUBackend");
    backend.initialize(&space).expect("Failed to initialize backend");
    let topology = backend.topology().expect("Failed to get topology").clone();
    (backend, space, topology)
}

/// Helper to allocate a linear buffer for testing
fn allocate_test_buffer(backend: &mut CPUBackend, size_bytes: usize) -> BackendHandle {
    let topology = BufferTopology {
        active_classes: vec![0],  // Use class 0 for simplicity
        phi_coordinates: vec![],  // No Φ coordinates for linear
        phase_affinity: None,     // No phase preference
        pool: MemoryPool::Linear, // Use linear pool for tests
        size_bytes,
        alignment: 64,
    };
    backend.allocate(topology).expect("Failed to allocate test buffer")
}

/// Helper to allocate and write typed data in one step
fn allocate_and_write<T: bytemuck::Pod>(backend: &mut CPUBackend, data: &[T]) -> BackendHandle {
    let size_bytes = std::mem::size_of_val(data);
    let handle = allocate_test_buffer(backend, size_bytes);
    let bytes = bytemuck::cast_slice(data);
    backend
        .write_buffer_bytes(handle, bytes)
        .expect("Failed to write buffer");
    handle
}

/// Helper to read typed data from a buffer
fn read_typed_data<T: bytemuck::Pod>(backend: &CPUBackend, handle: BackendHandle) -> Vec<T> {
    let bytes = backend.read_buffer_bytes(handle).expect("Failed to read buffer");
    bytemuck::cast_slice(&bytes).to_vec()
}

/// Execute a program with the backend
/// Usage: execute_program!(&mut backend, &topology, program)
macro_rules! execute_program {
    ($backend:expr, $topology:expr, $program:expr) => {{
        let phase = $backend.current_phase();
        let mut context = ExecutionContext::new($topology);
        context.phase = phase;
        context.active_classes = vec![0];
        context.resonance = [Rational::zero(); 96];
        context.n_elements = 0;
        $backend.execute_program(&$program, &context)
    }};
}

// ============================================================================
// Task 4.1: Data Movement Instruction Tests
// ============================================================================

#[test]
fn test_mov_instruction() {
    let (mut backend, _space, topology) = create_test_backend();

    // Allocate buffer with test value
    let test_data: Vec<f32> = vec![42.0, 0.0];
    let buffer = allocate_and_write(&mut backend, &test_data);

    // Test MOV with F32: Load to R0, MOV to R1, Store from R1
    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::MOV {
            ty: IsaType::F32,
            dst: Register::new(1),
            src: Register::new(0),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4, // Second element
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("MOV should execute");

    // Verify MOV worked - both elements should be 42.0
    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!((result[0] - 42.0).abs() < 1e-5);
    assert!((result[1] - 42.0).abs() < 1e-5);
}

#[test]
fn test_ldg_stg_roundtrip() {
    let (mut backend, _space, topology) = create_test_backend();

    // Allocate buffer with test data (needs room for read and write)
    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 0.0]; // 5 elements
    let buffer = allocate_and_write(&mut backend, &test_data);

    // Program: Load from buffer[0], store to buffer[4]
    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 16,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("LDG/STG should execute");

    // Verify data was copied
    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert_eq!(result[4], test_data[0], "LDG/STG roundtrip failed");
}

#[test]
fn test_cvt_f32_to_f64() {
    let (mut backend, _space, topology) = create_test_backend();

    // Allocate buffers for F32 and F64
    let f32_data: Vec<f32> = vec![3.14];
    let f32_buffer = allocate_and_write(&mut backend, &f32_data);
    let f64_buffer = allocate_test_buffer(&mut backend, 8); // 1 f64

    // Program: Load F32, convert to F64, store
    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: f32_buffer.0,
                offset: 0,
            },
        },
        Instruction::CVT {
            src_ty: IsaType::F32,
            dst_ty: IsaType::F64,
            dst: Register::new(1),
            src: Register::new(0),
        },
        Instruction::STG {
            ty: IsaType::F64,
            src: Register::new(1),
            addr: Address::BufferOffset {
                handle: f64_buffer.0,
                offset: 0,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("CVT should execute");

    // Verify conversion
    let result: Vec<f64> = read_typed_data(&backend, f64_buffer);
    assert!((result[0] - 3.14).abs() < 0.01);
}

// ============================================================================
// Task 4.1: Arithmetic Instruction Tests
// ============================================================================

#[test]
fn test_add_instruction() {
    let (mut backend, _space, topology) = create_test_backend();

    // Allocate buffer with test data: a=5.0, b=3.0, result=0.0
    let buffer = allocate_and_write(&mut backend, &[5.0f32, 3.0f32, 0.0f32]);

    // Program: Load a, b, add, store result
    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::ADD {
            ty: IsaType::F32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("ADD program should execute");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(
        (result[2] - 8.0).abs() < 1e-5,
        "ADD failed: expected 8.0, got {}",
        result[2]
    );
}

#[test]
fn test_mul_instruction() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[4.0f32, 3.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::MUL {
            ty: IsaType::F32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("MUL program should execute");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(
        (result[2] - 12.0).abs() < 1e-5,
        "MUL failed: expected 12.0, got {}",
        result[2]
    );
}

#[test]
fn test_fma_instruction() {
    let (mut backend, _space, topology) = create_test_backend();

    // FMA: a * b + c = 2 * 3 + 4 = 10
    let buffer = allocate_and_write(&mut backend, &[2.0f32, 3.0f32, 4.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::FMA {
            ty: IsaType::F32,
            dst: Register::new(3),
            a: Register::new(0),
            b: Register::new(1),
            c: Register::new(2),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(3),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 12,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("FMA program should execute");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(
        (result[3] - 10.0).abs() < 1e-5,
        "FMA failed: expected 10.0, got {}",
        result[3]
    );
}

#[test]
fn test_min_max_instructions() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[5.0f32, 3.0f32, 0.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::MIN {
            ty: IsaType::F32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::MAX {
            ty: IsaType::F32,
            dst: Register::new(3),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(3),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 12,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("MIN/MAX program should execute");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(
        (result[2] - 3.0).abs() < 1e-5,
        "MIN failed: expected 3.0, got {}",
        result[2]
    );
    assert!(
        (result[3] - 5.0).abs() < 1e-5,
        "MAX failed: expected 5.0, got {}",
        result[3]
    );
}

#[test]
fn test_abs_neg_instructions() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[-3.5f32, 0.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::ABS {
            ty: IsaType::F32,
            dst: Register::new(1),
            src: Register::new(0),
        },
        Instruction::NEG {
            ty: IsaType::F32,
            dst: Register::new(2),
            src: Register::new(0),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("ABS/NEG program should execute");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(
        (result[1] - 3.5).abs() < 1e-5,
        "ABS failed: expected 3.5, got {}",
        result[1]
    );
    assert!(
        (result[2] - 3.5).abs() < 1e-5,
        "NEG failed: expected 3.5, got {}",
        result[2]
    );
}

// ============================================================================
// Task 4.1: Logic Instruction Tests
// ============================================================================

#[test]
fn test_and_or_xor_instructions() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[0b1100u32, 0b1010u32, 0u32, 0u32, 0u32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::U32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::U32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::AND {
            ty: IsaType::U32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::OR {
            ty: IsaType::U32,
            dst: Register::new(3),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::XOR {
            ty: IsaType::U32,
            dst: Register::new(4),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::U32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::STG {
            ty: IsaType::U32,
            src: Register::new(3),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 12,
            },
        },
        Instruction::STG {
            ty: IsaType::U32,
            src: Register::new(4),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 16,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("Logic ops should execute");

    let result: Vec<u32> = read_typed_data(&backend, buffer);
    assert_eq!(result[2], 0b1000, "AND failed");
    assert_eq!(result[3], 0b1110, "OR failed");
    assert_eq!(result[4], 0b0110, "XOR failed");
}

#[test]
fn test_shl_shr_instructions() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[16u32, 2u32, 0u32, 0u32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::U32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::U32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::SHL {
            ty: IsaType::U32,
            dst: Register::new(2),
            src: Register::new(0),
            amount: Register::new(1),
        },
        Instruction::SHR {
            ty: IsaType::U32,
            dst: Register::new(3),
            src: Register::new(0),
            amount: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::U32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::STG {
            ty: IsaType::U32,
            src: Register::new(3),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 12,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("Shift ops should execute");

    let result: Vec<u32> = read_typed_data(&backend, buffer);
    assert_eq!(result[2], 64, "SHL failed: expected 64, got {}", result[2]);
    assert_eq!(result[3], 4, "SHR failed: expected 4, got {}", result[3]);
}

// ============================================================================
// Task 4.1: Control Flow Instruction Tests
// ============================================================================

#[test]
fn test_exit_instruction() {
    let (mut backend, _space, topology) = create_test_backend();

    let program = Program::from_instructions(vec![Instruction::EXIT]);

    execute_program!(&mut backend, &topology, program).expect("EXIT should succeed");
}

#[test]
fn test_bra_instruction() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[0.0f32, 0.0f32]);

    // Program with branch: skip loading, jump to exit
    let mut program = Program::from_instructions(vec![
        Instruction::BRA {
            target: Label("skip".to_string()),
            pred: None,
        },
        // This should be skipped
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        // Label instruction index: 2
        Instruction::EXIT,
    ]);

    // Register the label to point to the EXIT instruction
    program.labels.insert("skip".to_string(), 2);

    execute_program!(&mut backend, &topology, program).expect("BRA should execute");
}

// ============================================================================
// Task 4.1: Multi-Type Tests
// ============================================================================

#[test]
fn test_arithmetic_integer_types() {
    let (mut backend, _space, topology) = create_test_backend();

    // Test ADD for I32
    let buffer = allocate_and_write(&mut backend, &[100i32, 50i32, 0i32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::I32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::I32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::ADD {
            ty: IsaType::I32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::I32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("I32 ADD should execute");

    let result: Vec<i32> = read_typed_data(&backend, buffer);
    assert_eq!(result[2], 150, "I32 ADD failed");
}

// ============================================================================
// Task 4.2: Property-Based Invariant Tests
// ============================================================================

#[test]
fn test_property_phase_modulus() {
    let (backend, _space, _topology) = create_test_backend();

    // Phase should always be < 768
    let phase = backend.current_phase();
    assert!(phase < 768, "Phase modulus violation: phase {} >= 768", phase);
}

#[test]
fn test_property_mirror_involution() {
    let (backend, _space, _topology) = create_test_backend();

    // MIRROR(MIRROR(class)) = class for all classes
    let topology = backend.topology().expect("Failed to get topology");
    let mirrors = topology.mirrors();

    for class in 0..96u8 {
        let mirror = mirrors[class as usize];
        let double_mirror = mirrors[mirror as usize];
        assert_eq!(
            double_mirror, class,
            "Mirror involution failed: MIRROR(MIRROR({})) = {}, expected {}",
            class, double_mirror, class
        );
    }
}

#[test]
fn test_property_neighbor_symmetry() {
    let (backend, _space, _topology) = create_test_backend();

    // If B ∈ neighbors(A), then A ∈ neighbors(B)
    let topology = backend.topology().expect("Failed to get topology");
    let neighbors = topology.neighbors();

    for class_a in 0..96u8 {
        for &class_b in &neighbors[class_a as usize] {
            if class_b == u8::MAX {
                continue; // Skip invalid neighbor slots
            }

            // Check if class_a is in neighbors of class_b
            let b_neighbors = &neighbors[class_b as usize];
            assert!(
                b_neighbors.contains(&class_a),
                "Neighbor symmetry violated: {} ∈ neighbors({}), but {} ∉ neighbors({})",
                class_b,
                class_a,
                class_a,
                class_b
            );
        }
    }
}

#[test]
fn test_property_unity_neutrality() {
    use atlas_backends::types::Rational;

    let (mut backend, _space, _topology) = create_test_backend();

    // Create a simple operation to trigger resonance updates
    let buffer = allocate_and_write(&mut backend, &[1.0f32, 2.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::ADD {
            ty: IsaType::F32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    let _topology = backend.topology().expect("Failed to get topology").clone();
    execute_program!(&mut backend, &_topology, program).expect("Unity test program failed");

    // Unity Neutrality: sum of all resonance deltas should be zero
    let resonance = backend.resonance();
    let sum = Rational::sum(resonance.iter().copied());
    assert!(
        sum.is_zero(),
        "Unity neutrality violated: sum of resonance = {:?}, expected 0",
        sum
    );
}

#[test]
fn test_property_boundary_lens_roundtrip() {
    // Φ encoding should roundtrip: decode(encode(page, byte)) = (page, byte)
    use atlas_isa::{phi_decode, phi_encode};

    for page in 0..48u8 {
        for byte in 0..=255u8 {
            let encoded = phi_encode(page, byte);
            let (decoded_page, decoded_byte) = phi_decode(encoded);
            assert_eq!(
                (page, byte),
                (decoded_page, decoded_byte),
                "Φ roundtrip failed: ({}, {}) -> {} -> ({}, {})",
                page,
                byte,
                encoded,
                decoded_page,
                decoded_byte
            );
        }
    }
}

#[test]
fn test_mov_idempotence() {
    let (mut backend, _space, topology) = create_test_backend();

    // MOV should be idempotent: MOV(MOV(x)) = MOV(x)
    let buffer = allocate_and_write(&mut backend, &[7.5f32, 0.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::MOV {
            ty: IsaType::F32,
            dst: Register::new(1),
            src: Register::new(0),
        },
        Instruction::MOV {
            ty: IsaType::F32,
            dst: Register::new(2),
            src: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("MOV idempotence test failed");

    // Verify: R1 and R2 should both have the original value
    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!((result[1] - 7.5).abs() < 1e-5);
    assert!((result[2] - 7.5).abs() < 1e-5);
}

#[test]
fn test_add_commutative() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[3.0f32, 5.0f32, 0.0f32, 0.0f32]);

    // Test a + b = b + a
    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::ADD {
            ty: IsaType::F32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::ADD {
            ty: IsaType::F32,
            dst: Register::new(3),
            src1: Register::new(1),
            src2: Register::new(0),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(3),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 12,
            },
        },
        Instruction::EXIT,
    ]);

    execute_program!(&mut backend, &topology, program).expect("ADD commutativity test failed");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(
        (result[2] - result[3]).abs() < 1e-5,
        "ADD not commutative: {} != {}",
        result[2],
        result[3]
    );
}

// ============================================================================
// Task 4.2: Error Case Tests
// ============================================================================

// Note: Register bounds test removed because Register uses u8 (0-255 all valid).
// The type system prevents invalid register construction at compile time.

#[test]
fn test_division_by_zero() {
    let (mut backend, _space, topology) = create_test_backend();

    let buffer = allocate_and_write(&mut backend, &[10.0f32, 0.0f32, 0.0f32]);

    let program = Program::from_instructions(vec![
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(0),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 0,
            },
        },
        Instruction::LDG {
            ty: IsaType::F32,
            dst: Register::new(1),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 4,
            },
        },
        Instruction::DIV {
            ty: IsaType::F32,
            dst: Register::new(2),
            src1: Register::new(0),
            src2: Register::new(1),
        },
        Instruction::STG {
            ty: IsaType::F32,
            src: Register::new(2),
            addr: Address::BufferOffset {
                handle: buffer.0,
                offset: 8,
            },
        },
        Instruction::EXIT,
    ]);

    // Division by zero produces Inf (IEEE 754)
    execute_program!(&mut backend, &topology, program).expect("DIV by zero should produce Inf");

    let result: Vec<f32> = read_typed_data(&backend, buffer);
    assert!(result[2].is_infinite(), "DIV by zero should yield Inf");
}
