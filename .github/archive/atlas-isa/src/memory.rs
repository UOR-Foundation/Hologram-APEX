//! Memory model and addressing for Atlas ISA

use crate::{types::AddressSpace, uor::PhiCoordinate, AtlasError, Result};

/// Memory address in a specific address space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    pub space: AddressSpace,
    pub offset: u64,
}

impl Address {
    pub fn new(space: AddressSpace, offset: u64) -> Self {
        Self { space, offset }
    }

    pub fn global(offset: u64) -> Self {
        Self::new(AddressSpace::Global, offset)
    }

    pub fn shared(offset: u64) -> Self {
        Self::new(AddressSpace::Shared, offset)
    }
}

/// Boundary lens addressing: map (page, byte) to linear address
pub fn boundary_lens_map(coord: PhiCoordinate) -> u64 {
    coord.linear_index() as u64
}

/// Inverse: map linear address back to boundary coordinates
pub fn boundary_lens_unmap(addr: u64) -> Result<PhiCoordinate> {
    let index = addr as usize;
    if index >= crate::constants::TOTAL_ELEMENTS {
        return Err(AtlasError::BoundaryOutOfRange {
            page: (index / 256) as u32,
            byte: (index % 256) as u32,
        });
    }

    let page = (index / crate::constants::BYTES_PER_PAGE as usize) as u8;
    let byte = (index % crate::constants::BYTES_PER_PAGE as usize) as u8;
    PhiCoordinate::new(page, byte)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_creation() {
        let addr = Address::global(0x1000);
        assert_eq!(addr.space, AddressSpace::Global);
        assert_eq!(addr.offset, 0x1000);
    }

    #[test]
    fn test_boundary_lens_roundtrip() {
        let coord = PhiCoordinate::new(10, 128).unwrap();
        let addr = boundary_lens_map(coord);
        let recovered = boundary_lens_unmap(addr).unwrap();
        assert_eq!(coord, recovered);
    }

    #[test]
    fn test_boundary_lens_out_of_range() {
        let result = boundary_lens_unmap(20000);
        assert!(result.is_err());
    }
}
