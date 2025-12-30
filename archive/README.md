# Hologram

A component-based system defined through JSON Schema specifications with content-addressed storage.

## Structure

```
spec/           # Component definitions
src/            # MCP server and validation tools
Makefile        # Build and maintenance commands
```

## Components

The system contains validated components including:
- `hologram` - Root component
- `hologram.component` - Component model definition
- `hologram.spec` - Specification schema
- `hologram.interface` - Interface conformance
- `hologram.docs` - Documentation conformance
- `hologram.test` - Test conformance
- `hologram.manager` - Lifecycle management conformance
- `hologram.dependency` - Dependency declarations
- `hologram.build` - Build instructions
- `hologram.log` - Logging interface
- `hologram.create` - Create operation
- `hologram.read` - Read operation
- `hologram.update` - Update operation
- `hologram.validate` - Validation operation
- `hologram.index` - Index structure

Each component includes:
- spec - Component specification
- interface - Interface definition
- docs - Documentation
- test - Test specifications
- manager - Lifecycle management
- dependency - Dependency declarations
- build - Build instructions
- log - Logging configuration
- index - File mappings

## Usage

### Build and Test
```bash
make build          # Build TypeScript
make test           # Run tests
make validate       # Validate all components
```

### Maintenance
```bash
make gc             # Analyze spec/ for orphaned files
make gc-clean       # Remove orphaned files
make clean-artifacts # Clear .artifacts staging area
```

### MCP Server

Start the server manually:
```bash
make mcp            # Start MCP server
make mcp-validate   # Validate via MCP
```

Add to Claude Code:
```bash
# From project root
claude mcp add hologram "node" "src/dist/mcp-server.js"

# Or with full path
claude mcp add hologram "node" "/workspaces/Hologram/src/dist/mcp-server.js"
```

## MCP Tools

The Model Context Protocol server provides these tools:

**Core Operations:**
- `validate` - Validate components
- `read` - Read component files
- `create` - Create new components
- `update` - Update existing components
- `delete` - Delete components

**Artifact Management:**
- `submitArtifact` - Submit artifact for validation
- `submitManifest` - Create component from artifacts

**Discovery:**
- `listComponents` - Show all components and status
- `getComponentModel` - Get component requirements
- `getSchema` - Get specific schema
- `getComponentExample` - Get example structures
- `discover` - Discover available schemas

## File Format

All files use content-addressed storage with SHA256 hashes:
- `namespace.index.json` - Component index
- `namespace.hash.json` - Content files

## Requirements

- Node.js 22+
- TypeScript 5+