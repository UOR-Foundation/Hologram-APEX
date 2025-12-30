//! Circuit → IR Conversion (Builder)
//!
//! This module converts circuit string representations into IR trees.
//! Used during compilation only.

use super::nodes::{Generator, IRNode};

/// Build IR from circuit components
///
/// This is a placeholder for full circuit → IR conversion.
/// In practice, this would parse the circuit string and build an IR tree.
pub struct IRBuilder;

impl IRBuilder {
    /// Build IR from a circuit string
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ir = IRBuilder::build("mark@c05 . copy@c10->c11")?;
    /// ```
    pub fn build(_circuit: &str) -> Result<IRNode, BuildError> {
        // Placeholder: For now, just create a simple mark node
        // Full implementation would parse the circuit string
        Ok(IRNode::atom(Generator::Mark, None, None))
    }

    /// Build IR from parsed AST (when lang module is available)
    pub fn from_ast(_ast: &str) -> Result<IRNode, BuildError> {
        // Placeholder for AST → IR conversion
        Ok(IRNode::atom(Generator::Mark, None, None))
    }
}

/// Build errors
#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Invalid generator: {0}")]
    InvalidGenerator(String),

    #[error("Invalid class: {0}")]
    InvalidClass(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple() {
        let result = IRBuilder::build("mark@c05");
        assert!(result.is_ok());
    }
}
