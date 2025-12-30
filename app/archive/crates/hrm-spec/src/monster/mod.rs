//! Monster Group Structure
//!
//! The Monster group M is the largest sporadic simple group with:
//! - Order: ~8 × 10^53
//! - Minimal faithful representation: 196,884 dimensions
//! - 194 conjugacy classes
//! - Moonshine correspondence to modular functions

pub mod representation;
pub mod conjugacy;
pub mod character;
pub mod moonshine;

pub use representation::MonsterRepresentation;
pub use conjugacy::ConjugacyClass;
pub use character::CharacterTable;
pub use moonshine::MoonshineCorrespondence;
