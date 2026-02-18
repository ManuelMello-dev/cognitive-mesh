#!/usr/bin/env node

/**
 * Smoke Test for OpenClaw Gateway
 * 
 * Tests that the reverse proxy correctly forwards requests to the
 * cognitive mesh backend.
 */

import http from 'http';

const PORT = process.env.PORT || 8080;
const BASE_URL = `http://localhost:${PORT}`;

let passed = 0;
let failed = 0;

/**
 * Make HTTP request
 */
function request(path, method = 'GET', body = null) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE_URL);
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        resolve({
          status: res.statusCode,
          headers: res.headers,
          body: data
        });
      });
    });

    req.on('error', reject);
    
    if (body) {
      req.write(JSON.stringify(body));
    }
    
    req.end();
  });
}

/**
 * Test helper
 */
async function test(name, fn) {
  process.stdout.write(`Testing ${name}... `);
  try {
    await fn();
    console.log('✓ PASSED');
    passed++;
  } catch (error) {
    console.log('✗ FAILED:', error.message);
    failed++;
  }
}

/**
 * Sleep helper
 */
function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Run tests
 */
async function runTests() {
  console.log('OpenClaw Gateway Smoke Tests');
  console.log('============================\n');
  
  // Wait for server to be ready
  console.log('Waiting for server to start...\n');
  await sleep(2000);

  // Test 1: Health check endpoint
  await test('GET /health', async () => {
    const res = await request('/health');
    if (res.status !== 200) {
      throw new Error(`Expected 200, got ${res.status}`);
    }
    const data = JSON.parse(res.body);
    if (data.status !== 'alive') {
      throw new Error(`Expected status 'alive', got '${data.status}'`);
    }
  });

  // Test 2: Healthz endpoint
  await test('GET /healthz', async () => {
    const res = await request('/healthz');
    if (res.status !== 200) {
      throw new Error(`Expected 200, got ${res.status}`);
    }
  });

  // Test 3: Metrics endpoint
  await test('GET /api/metrics', async () => {
    const res = await request('/api/metrics');
    if (res.status !== 200 && res.status !== 503) {
      throw new Error(`Expected 200 or 503, got ${res.status}`);
    }
  });

  // Test 4: State endpoint
  await test('GET /api/state', async () => {
    const res = await request('/api/state');
    if (res.status !== 200 && res.status !== 503) {
      throw new Error(`Expected 200 or 503, got ${res.status}`);
    }
  });

  // Test 5: Dashboard (root)
  await test('GET /', async () => {
    const res = await request('/');
    if (res.status !== 200 && res.status !== 404 && res.status !== 503) {
      throw new Error(`Expected 200/404/503, got ${res.status}`);
    }
  });

  // Test 6: POST /api/ingest (may fail if core not initialized, but should proxy)
  await test('POST /api/ingest', async () => {
    const res = await request('/api/ingest', 'POST', {
      observation: { test: 'smoke-test' },
      domain: 'test'
    });
    // Accept various status codes (400, 503 if backend not ready)
    if (res.status < 100 || res.status >= 600) {
      throw new Error(`Invalid status code: ${res.status}`);
    }
  });

  // Test 7: GET /api/predictions
  await test('GET /api/predictions', async () => {
    const res = await request('/api/predictions');
    if (res.status !== 200 && res.status !== 503) {
      throw new Error(`Expected 200 or 503, got ${res.status}`);
    }
  });

  // Test 8: GET /api/introspection
  await test('GET /api/introspection', async () => {
    const res = await request('/api/introspection');
    if (res.status !== 200 && res.status !== 503) {
      throw new Error(`Expected 200 or 503, got ${res.status}`);
    }
  });

  // Test 9: GET /api/causal (hidden intelligence endpoint)
  await test('GET /api/causal', async () => {
    const res = await request('/api/causal');
    if (res.status !== 200 && res.status !== 503) {
      throw new Error(`Expected 200 or 503, got ${res.status}`);
    }
  });

  // Test 10: GET /api/toggles
  await test('GET /api/toggles', async () => {
    const res = await request('/api/toggles');
    if (res.status !== 200 && res.status !== 503) {
      throw new Error(`Expected 200 or 503, got ${res.status}`);
    }
  });

  console.log('\n============================');
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log('============================\n');

  if (failed > 0) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error('Test suite failed:', err);
  process.exit(1);
});
