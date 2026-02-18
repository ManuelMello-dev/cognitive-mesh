/**
 * OpenClaw Tools for Cognitive Mesh Integration
 * 
 * These tools allow OpenClaw agents to interact with the cognitive mesh endpoints
 * and be part of the I/O loop.
 */

import http from 'http';
import https from 'https';

const MESH_BASE_URL = process.env.COGNITIVE_MESH_BASE_URL || 
                      `http://127.0.0.1:${process.env.COGNITIVE_MESH_PORT || 8081}`;

/**
 * Generic HTTP client for making requests to mesh endpoints
 */
function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const client = urlObj.protocol === 'https:' ? https : http;
    
    const reqOptions = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname + urlObj.search,
      method: options.method || 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      }
    };

    const req = client.request(reqOptions, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        try {
          const json = data ? JSON.parse(data) : {};
          resolve({
            status: res.statusCode,
            headers: res.headers,
            data: json,
            raw: data
          });
        } catch (e) {
          resolve({
            status: res.statusCode,
            headers: res.headers,
            data: null,
            raw: data
          });
        }
      });
    });

    req.on('error', (err) => {
      reject(err);
    });

    if (options.body) {
      req.write(JSON.stringify(options.body));
    }
    
    req.end();
  });
}

/**
 * Tool: Call arbitrary mesh endpoint
 * 
 * This tool allows OpenClaw agents to call any mesh endpoint with custom
 * method, path, body, and headers.
 */
export async function callMeshEndpoint({ method = 'GET', path, body = null, headers = {} }) {
  try {
    const url = `${MESH_BASE_URL}${path}`;
    const response = await makeRequest(url, { method, body, headers });
    
    // Generate a concise summary
    const summary = generateSummary(path, response);
    
    return {
      success: true,
      response,
      summary
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to call ${path}: ${error.message}`
    };
  }
}

/**
 * Tool: Ingest observation into mesh
 * 
 * This helper feeds insights back into the mesh by calling POST /api/ingest
 * with the domain "openclaw_io".
 */
export async function ingestToMesh({ observation, domain = 'openclaw_io' }) {
  try {
    const response = await makeRequest(`${MESH_BASE_URL}/api/ingest`, {
      method: 'POST',
      body: { observation, domain }
    });
    
    return {
      success: response.status >= 200 && response.status < 300,
      response,
      summary: `Ingested observation to domain '${domain}': ${truncate(JSON.stringify(observation), 100)}`
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to ingest observation: ${error.message}`
    };
  }
}

/**
 * Tool: Get mesh metrics
 */
export async function getMeshMetrics() {
  try {
    const response = await makeRequest(`${MESH_BASE_URL}/api/metrics`);
    return {
      success: true,
      response,
      summary: `Mesh metrics: PHI=${response.data?.phi || 'N/A'}, concepts=${response.data?.concepts_formed || 0}, rules=${response.data?.rules_learned || 0}`
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to get metrics: ${error.message}`
    };
  }
}

/**
 * Tool: Get mesh state
 */
export async function getMeshState() {
  try {
    const response = await makeRequest(`${MESH_BASE_URL}/api/state`);
    const conceptCount = Object.keys(response.data?.concepts || {}).length;
    const ruleCount = Object.keys(response.data?.rules || {}).length;
    
    return {
      success: true,
      response,
      summary: `Mesh state: ${conceptCount} concepts, ${ruleCount} rules, node_id=${response.data?.node_id || 'N/A'}`
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to get state: ${error.message}`
    };
  }
}

/**
 * Tool: Analyze patterns
 */
export async function analyzePatterns() {
  try {
    const response = await makeRequest(`${MESH_BASE_URL}/api/analyze`);
    return {
      success: true,
      response,
      summary: `Pattern analysis complete: ${Object.keys(response.data || {}).length} patterns found`
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to analyze patterns: ${error.message}`
    };
  }
}

/**
 * Tool: Get predictions
 */
export async function getPredictions() {
  try {
    const response = await makeRequest(`${MESH_BASE_URL}/api/predictions`);
    return {
      success: true,
      response,
      summary: `Predictions retrieved: accuracy=${response.data?.accuracy || 'N/A'}`
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to get predictions: ${error.message}`
    };
  }
}

/**
 * Tool: Get introspection data
 */
export async function getIntrospection() {
  try {
    const response = await makeRequest(`${MESH_BASE_URL}/api/introspection`);
    return {
      success: true,
      response,
      summary: `Introspection complete: ${Object.keys(response.data || {}).length} engines reported`
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      summary: `Failed to get introspection: ${error.message}`
    };
  }
}

/**
 * Helper: Generate concise summary from response
 */
function generateSummary(path, response) {
  if (response.status < 200 || response.status >= 300) {
    return `Request to ${path} failed with status ${response.status}`;
  }
  
  const data = response.data;
  if (!data) {
    return `Request to ${path} succeeded (${response.status})`;
  }
  
  // Generate context-aware summaries based on endpoint
  if (path.includes('/metrics')) {
    return `Metrics: PHI=${data.phi || 'N/A'}, concepts=${data.concepts_formed || 0}`;
  } else if (path.includes('/state')) {
    return `State: ${Object.keys(data.concepts || {}).length} concepts, ${Object.keys(data.rules || {}).length} rules`;
  } else if (path.includes('/health')) {
    return `Health: ${data.status || 'unknown'}`;
  } else {
    // Generic summary
    const keys = Object.keys(data);
    return `Response (${response.status}): ${keys.length} fields [${keys.slice(0, 3).join(', ')}${keys.length > 3 ? '...' : ''}]`;
  }
}

/**
 * Helper: Truncate string
 */
function truncate(str, maxLength) {
  if (str.length <= maxLength) return str;
  return str.substring(0, maxLength) + '...';
}

/**
 * Export all tools
 */
export const meshTools = {
  callMeshEndpoint,
  ingestToMesh,
  getMeshMetrics,
  getMeshState,
  analyzePatterns,
  getPredictions,
  getIntrospection
};

export default meshTools;
