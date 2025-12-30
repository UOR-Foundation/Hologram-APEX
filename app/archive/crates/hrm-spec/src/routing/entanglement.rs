//! Entanglement Network
//!
//! Defines addressing and navigation through Monster representation entanglement.
//! The 12,288 boundary lattice cells form routing channels.

use crate::torus::coordinate::TorusCoordinate;

/// Entanglement network address
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkAddress {
    /// Cell index in boundary lattice [0, 12287]
    pub cell_index: usize,
    /// Torus coordinate
    pub coordinate: TorusCoordinate,
}

impl NetworkAddress {
    /// Create address from torus coordinate
    pub fn from_coordinate(coord: TorusCoordinate) -> Self {
        Self {
            cell_index: coord.cell_index(),
            coordinate: coord,
        }
    }
    
    /// Get torus coordinate
    pub fn coordinate(&self) -> &TorusCoordinate {
        &self.coordinate
    }
    
    /// Get cell index
    pub fn cell_index(&self) -> usize {
        self.cell_index
    }
}

/// Entanglement network navigation
pub struct EntanglementNetwork {
    /// Total number of cells
    size: usize,
}

impl EntanglementNetwork {
    /// Create network with standard boundary lattice size
    pub fn new() -> Self {
        Self {
            size: TorusCoordinate::BOUNDARY_SIZE,
        }
    }
    
    /// Get network size
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Check if address is valid
    pub fn is_valid_address(&self, addr: &NetworkAddress) -> bool {
        addr.cell_index < self.size
    }
}

impl Default for EntanglementNetwork {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_address_creation() {
        let coord = TorusCoordinate { page: 5, resonance: 10 };
        let addr = NetworkAddress::from_coordinate(coord.clone());
        
        assert_eq!(addr.coordinate, coord);
        assert_eq!(addr.cell_index, coord.cell_index());
    }

    #[test]
    fn test_entanglement_network_size() {
        let network = EntanglementNetwork::new();
        assert_eq!(network.size(), 12_288);
    }

    #[test]
    fn test_valid_address() {
        let network = EntanglementNetwork::new();
        let coord = TorusCoordinate { page: 0, resonance: 0 };
        let addr = NetworkAddress::from_coordinate(coord);
        
        assert!(network.is_valid_address(&addr));
    }

    #[test]
    fn test_all_coordinates_valid() {
        let network = EntanglementNetwork::new();
        
        // Check all 12,288 cells are valid
        for page in 0..48 {
            for res in 0u16..256 {
                let coord = TorusCoordinate {
                    page: page,
                    resonance: (res % 96) as u8,
                };
                let addr = NetworkAddress::from_coordinate(coord);
                assert!(network.is_valid_address(&addr));
            }
        }
    }
}
