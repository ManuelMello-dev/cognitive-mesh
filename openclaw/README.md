# OpenClaw Integration for Cognitive Mesh

This directory contains the OpenClaw integration that provides:

1. **Reverse Proxy Gateway**: OpenClaw serves as the single public entrypoint on Railway
2. **Mesh Tools**: Tools that allow OpenClaw agents to interact with all mesh endpoints
3. **I/O Loop Integration**: OpenClaw can read mesh outputs and feed insights back

## Architecture

```
Railway Public Port ($PORT)
    ↓
OpenClaw Gateway (Node.js)
    ├─ Reverse Proxy → Cognitive Mesh (Python on internal port)
    └─ OpenClaw Tools
        ├─ callMeshEndpoint()
        ├─ ingestToMesh()
        ├─ getMeshMetrics()
        ├─ getMeshState()
        ├─ analyzePatterns()
        ├─ getPredictions()
        └─ getIntrospection()
```

## Tools

### `callMeshEndpoint(options)`

Generic tool to call any mesh endpoint.

**Parameters:**
- `method` (string): HTTP method (GET, POST, etc.)
- `path` (string): Endpoint path (e.g., '/api/metrics')
- `body` (object): Request body for POST requests
- `headers` (object): Custom headers

**Returns:**
- `success` (boolean): Whether the request succeeded
- `response` (object): Full HTTP response with status, headers, data
- `summary` (string): Concise summary of the result

**Example:**
```javascript
const result = await callMeshEndpoint({
  method: 'GET',
  path: '/api/metrics'
});
console.log(result.summary); // "Metrics: PHI=0.85, concepts=42"
```

### `ingestToMesh(options)`

Feed observations back into the mesh.

**Parameters:**
- `observation` (object): The observation to ingest
- `domain` (string): Domain name (defaults to 'openclaw_io')

**Returns:**
- `success` (boolean): Whether ingestion succeeded
- `response` (object): Full HTTP response
- `summary` (string): Concise summary

**Example:**
```javascript
const result = await ingestToMesh({
  observation: {
    type: 'insight',
    content: 'Market sentiment is bullish',
    confidence: 0.85
  },
  domain: 'openclaw_io'
});
```

### Other Tools

- `getMeshMetrics()`: Get current mesh metrics
- `getMeshState()`: Get full mesh state
- `analyzePatterns()`: Trigger pattern analysis
- `getPredictions()`: Get prediction engine state
- `getIntrospection()`: Get full system introspection

## Environment Variables

- `PORT`: Public port for Railway (default: 8080)
- `COGNITIVE_MESH_PORT`: Internal port for Python server (default: 8081)
- `COGNITIVE_MESH_BASE_URL`: Base URL for mesh (default: http://127.0.0.1:8081)

## Usage

The OpenClaw gateway automatically:
1. Starts the Python cognitive-mesh server on an internal port
2. Proxies all HTTP requests transparently
3. Provides tools for agents to interact with the mesh

All existing mesh endpoints are accessible at the same paths through the OpenClaw gateway.
