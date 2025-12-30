#!/usr/bin/env python3
"""
Simple .mshr file generator for basic operations

Generates .mshr binary files without full ONNX compiler.
Uses direct binary writing with FNV-1a hashing.

Usage:
    python3 generate_mshr.py vector_add output/vector_add.mshr
"""

import struct
import json
import sys
from typing import List, Tuple
from datetime import datetime

# FNV-1a constants
FNV_OFFSET_BASIS = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3

MSHR_MAGIC = b"MSHRFMT\x00"
MSHR_VERSION = 1


def fnv1a_hash_f32(values: List[float]) -> int:
    """Hash a list of f32 values using FNV-1a"""
    h = FNV_OFFSET_BASIS
    for value in values:
        # Convert to f32 bit pattern
        bytes_val = struct.pack('<f', value)
        for byte in bytes_val:
            h ^= byte
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def create_header(manifest_offset: int, manifest_size: int,
                 hash_table_offset: int, hash_table_size: int,
                 result_data_offset: int, result_data_size: int) -> bytes:
    """Create 64-byte .mshr header"""
    header = bytearray(64)

    # Magic (8 bytes)
    header[0:8] = MSHR_MAGIC

    # Version (4 bytes)
    struct.pack_into('<I', header, 8, MSHR_VERSION)

    # Flags (4 bytes, reserved)
    struct.pack_into('<I', header, 12, 0)

    # Manifest offset and size (8 bytes each)
    struct.pack_into('<Q', header, 16, manifest_offset)
    struct.pack_into('<Q', header, 24, manifest_size)

    # Hash table offset and size (8 bytes each)
    struct.pack_into('<Q', header, 32, hash_table_offset)
    struct.pack_into('<Q', header, 40, hash_table_size)

    # Result data offset and size (8 bytes each)
    struct.pack_into('<Q', header, 48, result_data_offset)
    struct.pack_into('<Q', header, 56, result_data_size)

    return bytes(header)


def create_manifest(operation: str, input_patterns: int, output_size: int) -> bytes:
    """Create JSON manifest"""
    manifest = {
        "operation": operation,
        "version": "1.0.0",
        "input_patterns": input_patterns,
        "output_size": output_size,
        "data_type": "F32",
        "hash_function": "fnv1a_64",
        "compilation_date": datetime.now().isoformat(),
        "atlas_version": "phase3.0"
    }
    return json.dumps(manifest, indent=None).encode('utf-8')


def create_hash_table(patterns: List[List[float]]) -> bytes:
    """Create sorted hash table entries (16 bytes each)"""
    entries = []
    for idx, pattern in enumerate(patterns):
        key_hash = fnv1a_hash_f32(pattern)
        entries.append((key_hash, idx))

    # Sort by hash
    entries.sort(key=lambda x: x[0])

    # Serialize (16 bytes per entry: u64 hash + u32 index + u32 padding)
    hash_table = bytearray()
    for key_hash, result_index in entries:
        hash_table.extend(struct.pack('<Q', key_hash))  # 8 bytes
        hash_table.extend(struct.pack('<I', result_index))  # 4 bytes
        hash_table.extend(struct.pack('<I', 0))  # 4 bytes padding

    return bytes(hash_table)


def create_result_data(results: List[List[float]]) -> bytes:
    """Create result data array"""
    result_data = bytearray()
    for result in results:
        for value in result:
            result_data.extend(struct.pack('<f', value))
    return bytes(result_data)


def generate_mshr(operation: str, patterns: List[List[float]],
                 results: List[List[float]], output_path: str):
    """Generate .mshr binary file"""
    if len(patterns) != len(results):
        raise ValueError("patterns and results must have same length")

    if not results:
        raise ValueError("Must have at least one pattern")

    output_size = len(results[0])

    # Build components
    manifest_bytes = create_manifest(operation, len(patterns), output_size)
    hash_table_bytes = create_hash_table(patterns)
    result_data_bytes = create_result_data(results)

    # Calculate offsets
    header_size = 64
    manifest_offset = header_size
    manifest_size = len(manifest_bytes)
    hash_table_offset = manifest_offset + manifest_size
    hash_table_size = len(hash_table_bytes)
    result_data_offset = hash_table_offset + hash_table_size
    result_data_size = len(result_data_bytes)

    # Create header
    header_bytes = create_header(
        manifest_offset, manifest_size,
        hash_table_offset, hash_table_size,
        result_data_offset, result_data_size
    )

    # Write file
    with open(output_path, 'wb') as f:
        f.write(header_bytes)
        f.write(manifest_bytes)
        f.write(hash_table_bytes)
        f.write(result_data_bytes)

    print(f"Generated {output_path}")
    print(f"  Patterns: {len(patterns)}")
    print(f"  Output size: {output_size}")
    print(f"  File size: {len(header_bytes) + len(manifest_bytes) + len(hash_table_bytes) + len(result_data_bytes)} bytes")


# Predefined operations

def generate_vector_add(output_path: str):
    """Generate vector_add.mshr - element-wise addition"""
    patterns = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [0.0, 0.0],
        [-1.0, -2.0],
        [10.0, 20.0],
        [0.5, 1.5],
        [100.0, 200.0],
        [-5.0, 5.0],
        [1.0, 1.0],
    ]

    # For vector_add, we add consecutive pairs: [a, b] -> [a+b]
    results = [
        [3.0],   # 1+2
        [7.0],   # 3+4
        [11.0],  # 5+6
        [0.0],   # 0+0
        [-3.0],  # -1+-2
        [30.0],  # 10+20
        [2.0],   # 0.5+1.5
        [300.0], # 100+200
        [0.0],   # -5+5
        [2.0],   # 1+1
    ]

    generate_mshr("vector_add", patterns, results, output_path)


def generate_vector_mul(output_path: str):
    """Generate vector_mul.mshr - element-wise multiplication"""
    patterns = [
        [2.0, 3.0],
        [4.0, 5.0],
        [1.0, 1.0],
        [0.0, 10.0],
        [-2.0, 3.0],
        [0.5, 4.0],
        [10.0, 10.0],
        [-1.0, -1.0],
        [7.0, 8.0],
        [0.1, 0.1],
    ]

    # For vector_mul, we multiply consecutive pairs: [a, b] -> [a*b]
    results = [
        [6.0],    # 2*3
        [20.0],   # 4*5
        [1.0],    # 1*1
        [0.0],    # 0*10
        [-6.0],   # -2*3
        [2.0],    # 0.5*4
        [100.0],  # 10*10
        [1.0],    # -1*-1
        [56.0],   # 7*8
        [0.01],   # 0.1*0.1
    ]

    generate_mshr("vector_mul", patterns, results, output_path)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 generate_mshr.py <operation> <output_path>")
        print("Operations: vector_add, vector_mul")
        sys.exit(1)

    operation = sys.argv[1]
    output_path = sys.argv[2]

    if operation == "vector_add":
        generate_vector_add(output_path)
    elif operation == "vector_mul":
        generate_vector_mul(output_path)
    else:
        print(f"Unknown operation: {operation}")
        print("Available: vector_add, vector_mul")
        sys.exit(1)


if __name__ == "__main__":
    main()
