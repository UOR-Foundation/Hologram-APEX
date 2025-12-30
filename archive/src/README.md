# Hologram Component Manager - MCP Server

Model Context Protocol server for managing Hologram components.

## Structure

```
src/
├── core/               # Schema validator, artifact store
├── operations/         # CRUD+V operations
├── utils/              # Garbage collection
├── __tests__/          # Integration tests
├── mcp-server.ts       # MCP server entry point
├── index.ts            # Main exports
└── types.ts            # TypeScript types
```

## Build and Run

```bash
npm install             # Install dependencies
npm run build           # Build TypeScript
npm run test            # Run tests
npm run mcp            # Start MCP server
```

## Adding to Claude Code

```bash
# Build first
cd src && npm run build

# Add MCP server to Claude Code (from project root)
claude mcp add hologram "node" "src/dist/mcp-server.js"

# List configured servers
claude mcp list

# Remove if needed
claude mcp remove hologram
```

## MCP Tools

- `listComponents` - List all components and validation status
- `validate` - Validate component(s)
- `read` - Read component files
- `create` - Create new component
- `update` - Update existing component
- `delete` - Delete component
- `submitArtifact` - Submit artifact for staging
- `submitManifest` - Create component from artifacts
- `getComponentModel` - Get component requirements
- `getSchema` - Get specific schema
- `getComponentExample` - Get example structures

## Component Files

Each component requires:
1. spec - Component specification
2. interface - Interface conformance
3. docs - Documentation conformance
4. test - Test conformance
5. manager - Manager conformance
6. dependency - Dependency declarations
7. build - Build instructions
8. log - Logging configuration
9. index - File mappings

## Validation

- Strict JSON Schema validation
- Content-addressed storage (SHA256)
- Atomic operations (all-or-nothing)
- Model-driven from hologram.component.json

## Maintenance

```bash
npm run gc              # Analyze for orphaned files
npm run gc:clean        # Remove orphaned files
npm run clean           # Clean build artifacts
```

## Tests

Run tests with `npm test` to verify all functionality.