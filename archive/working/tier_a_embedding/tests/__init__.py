"""
Test suite for Tier-A embedding package.

This test suite provides comprehensive testing for:
- Atlas graph construction and properties
- E8 root system generation and verification
- Embedding search algorithms
- Certificate generation and verification
- Canonicalization and automorphism operations
- End-to-end integration
"""

# Make the package importable from tests
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test markers for organization
UNIT_TEST = "unit"
INTEGRATION_TEST = "integration"
SLOW_TEST = "slow"