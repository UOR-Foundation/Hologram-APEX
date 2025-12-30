# OpenAPI Specification

The UOR-MCP API is documented using the OpenAPI 3.1 specification. This provides a machine-readable description of the API that can be used by tools to generate client libraries, documentation, and more.

## Accessing the OpenAPI Specification

The OpenAPI specification is available in YAML format at the root of the repository:

```
/openapi.yaml
```

## Key Endpoints

### MCP JSON-RPC Endpoint

The main endpoint for all MCP JSON-RPC requests is:

```
POST /mcp
```

This endpoint accepts JSON-RPC 2.0 requests and returns JSON-RPC 2.0 responses.

### OAuth Callback Endpoint

The OAuth callback endpoint for GitHub authentication is:

```
GET /auth-callback.html
```

## Authentication

The API uses GitHub OAuth for authentication. The OAuth flow is as follows:

1. The client redirects the user to GitHub's authorization URL
2. The user authorizes the application
3. GitHub redirects back to the callback URL with an authorization code
4. The client exchanges the code for an access token using the token exchange proxy
5. The client uses the access token for subsequent API requests

## JSON-RPC Methods

The MCP API supports the following JSON-RPC methods:

### initialize

Initializes the MCP connection and returns server capabilities.

### tools/list

Lists available tools for the authenticated user.

### resources/list

Lists available resources for the authenticated user.

### tools/call

Executes a tool with the provided parameters.

## Error Handling

The API uses standard JSON-RPC error codes and messages. Common error codes include:

- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error
- `-32000` to `-32099`: Server error

## Using with OpenAPI Tools

The OpenAPI specification can be used with various tools:

- **Swagger UI**: For interactive API documentation
- **OpenAPI Generator**: For generating client libraries
- **Postman**: For testing API endpoints
- **Redoc**: For generating static documentation

## Example Usage

```javascript
// Example JSON-RPC request
const request = {
  jsonrpc: "2.0",
  method: "tools/call",
  params: {
    tool: "createUOR",
    parameters: {
      type: "concept",
      data: {
        name: "Example Concept",
        description: "This is an example concept"
      }
    }
  },
  id: 1
};

// Send the request to the MCP endpoint
fetch("https://68113dd199a34737508b5211--uor-mcp.netlify.app/mcp", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + accessToken
  },
  body: JSON.stringify(request)
})
.then(response => response.json())
.then(data => console.log(data));
```
