//! Routing Channels
//!
//! Defines composition and navigation of routing channels in boundary lattice.

use crate::torus::coordinate::TorusCoordinate;
use crate::routing::entanglement::NetworkAddress;

/// Routing channel in boundary lattice
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingChannel {
    /// Source address
    pub source: NetworkAddress,
    /// Target address
    pub target: NetworkAddress,
}

impl RoutingChannel {
    /// Create channel from source to target
    pub fn new(source: TorusCoordinate, target: TorusCoordinate) -> Self {
        Self {
            source: NetworkAddress::from_coordinate(source),
            target: NetworkAddress::from_coordinate(target),
        }
    }
    
    /// Compose two channels (if compatible)
    pub fn compose(&self, other: &RoutingChannel) -> Option<RoutingChannel> {
        // Channels compose if target of first matches source of second
        if self.target.coordinate == other.source.coordinate {
            Some(RoutingChannel {
                source: self.source.clone(),
                target: other.target.clone(),
            })
        } else {
            None
        }
    }
}

/// Channel composition operator
pub fn compose_channels(channels: &[RoutingChannel]) -> Option<RoutingChannel> {
    if channels.is_empty() {
        return None;
    }
    
    let mut result = channels[0].clone();
    for channel in &channels[1..] {
        result = result.compose(channel)?;
    }
    
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_creation() {
        let source = TorusCoordinate { page: 3, resonance: 5 };
        let target = TorusCoordinate { page: 7, resonance: 11 };
        
        let channel = RoutingChannel::new(source.clone(), target.clone());
        assert_eq!(channel.source.coordinate, source);
        assert_eq!(channel.target.coordinate, target);
    }

    #[test]
    fn test_channel_composition() {
        let a = TorusCoordinate { page: 3, resonance: 5 };
        let b = TorusCoordinate { page: 7, resonance: 11 };
        let c = TorusCoordinate { page: 13, resonance: 17 };
        
        let ch1 = RoutingChannel::new(a, b.clone());
        let ch2 = RoutingChannel::new(b, c.clone());
        
        let composed = ch1.compose(&ch2).unwrap();
        assert_eq!(composed.source.coordinate.page, 3);
        assert_eq!(composed.target.coordinate.page, 13);
    }

    #[test]
    fn test_channel_composition_incompatible() {
        let a = TorusCoordinate { page: 3, resonance: 5 };
        let b = TorusCoordinate { page: 7, resonance: 11 };
        let c = TorusCoordinate { page: 13, resonance: 17 };
        
        let ch1 = RoutingChannel::new(a, b);
        let ch2 = RoutingChannel::new(c.clone(), c);
        
        assert!(ch1.compose(&ch2).is_none());
    }

    #[test]
    fn test_compose_channels_sequence() {
        let coords = vec![
            TorusCoordinate { page: 0, resonance: 0 },
            TorusCoordinate { page: 1, resonance: 1 },
            TorusCoordinate { page: 2, resonance: 2 },
            TorusCoordinate { page: 3, resonance: 3 },
        ];
        
        let channels: Vec<RoutingChannel> = coords.windows(2)
            .map(|w| RoutingChannel::new(w[0].clone(), w[1].clone()))
            .collect();
        
        let composed = compose_channels(&channels).unwrap();
        assert_eq!(composed.source.coordinate, coords[0]);
        assert_eq!(composed.target.coordinate, coords[3]);
    }
}
