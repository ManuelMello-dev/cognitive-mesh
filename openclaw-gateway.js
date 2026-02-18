#!/usr/bin/env node

/**
 * OpenClaw Gateway - Reverse Proxy for Cognitive Mesh
 * 
 * This gateway serves as the single public entrypoint on Railway.
 * It:
 * 1. Starts the Python cognitive-mesh server on an internal port
 * 2. Proxies all HTTP requests to the Python server at the same paths
 * 3. Provides OpenClaw tools that can interact with mesh endpoints
 * 4. Allows OpenClaw agents to be in the I/O loop
 */

import http from 'http';
import httpProxy from 'http-proxy';
import { spawn } from 'child_process';
import dotenv from 'dotenv';

dotenv.config();

// Configuration
const PORT = process.env.PORT || 8080;
const COGNITIVE_MESH_PORT = process.env.COGNITIVE_MESH_PORT || 8081;
const COGNITIVE_MESH_BASE_URL = process.env.COGNITIVE_MESH_BASE_URL || `http://127.0.0.1:${COGNITIVE_MESH_PORT}`;

let pythonProcess = null;
let proxyServer = null;
let httpServer = null;

/**
 * Start the Python cognitive-mesh server on internal port
 */
function startPythonServer() {
  console.log(`[OpenClaw Gateway] Starting Python server on port ${COGNITIVE_MESH_PORT}...`);
  
  const env = {
    ...process.env,
    PORT: COGNITIVE_MESH_PORT.toString(),
    PYTHONUNBUFFERED: '1'
  };

  pythonProcess = spawn('python', ['main.py'], {
    env,
    stdio: ['ignore', 'inherit', 'inherit']
  });

  pythonProcess.on('error', (err) => {
    console.error('[OpenClaw Gateway] Failed to start Python server:', err);
    process.exit(1);
  });

  pythonProcess.on('exit', (code, signal) => {
    console.log(`[OpenClaw Gateway] Python server exited with code ${code}, signal ${signal}`);
    if (code !== 0 && code !== null) {
      console.error('[OpenClaw Gateway] Python server crashed, shutting down...');
      process.exit(1);
    }
  });

  // Give Python server time to start
  return new Promise((resolve) => {
    setTimeout(resolve, 3000);
  });
}

/**
 * Create and configure the reverse proxy
 */
function createProxy() {
  console.log(`[OpenClaw Gateway] Creating reverse proxy to ${COGNITIVE_MESH_BASE_URL}...`);
  
  const proxy = httpProxy.createProxyServer({
    target: COGNITIVE_MESH_BASE_URL,
    changeOrigin: false,
    preserveHeaderKeyCase: true,
    xfwd: true
  });

  proxy.on('error', (err, req, res) => {
    console.error('[OpenClaw Gateway] Proxy error:', err.message);
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        error: 'Bad Gateway',
        message: 'The cognitive mesh backend is unavailable',
        details: err.message
      }));
    }
  });

  proxy.on('proxyReq', (proxyReq, req) => {
    // Log proxied requests
    console.log(`[OpenClaw Gateway] ${req.method} ${req.url} -> ${COGNITIVE_MESH_BASE_URL}${req.url}`);
  });

  return proxy;
}

/**
 * Create the HTTP server
 */
function createHttpServer(proxy) {
  const server = http.createServer((req, res) => {
    // All requests are transparently proxied to the Python backend
    proxy.web(req, res);
  });

  server.on('error', (err) => {
    console.error('[OpenClaw Gateway] HTTP server error:', err);
    process.exit(1);
  });

  return server;
}

/**
 * Start the gateway
 */
async function start() {
  console.log('[OpenClaw Gateway] Starting...');
  console.log(`[OpenClaw Gateway] Public PORT: ${PORT}`);
  console.log(`[OpenClaw Gateway] Internal Mesh PORT: ${COGNITIVE_MESH_PORT}`);

  // Start Python server
  await startPythonServer();
  
  // Create proxy
  proxyServer = createProxy();
  
  // Create HTTP server
  httpServer = createHttpServer(proxyServer);
  
  // Start listening
  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`[OpenClaw Gateway] Listening on 0.0.0.0:${PORT}`);
    console.log(`[OpenClaw Gateway] Proxying all requests to ${COGNITIVE_MESH_BASE_URL}`);
    console.log('[OpenClaw Gateway] Ready!');
  });
}

/**
 * Graceful shutdown
 */
function shutdown(signal) {
  console.log(`[OpenClaw Gateway] Received ${signal}, shutting down gracefully...`);
  
  if (httpServer) {
    httpServer.close(() => {
      console.log('[OpenClaw Gateway] HTTP server closed');
    });
  }
  
  if (pythonProcess) {
    console.log('[OpenClaw Gateway] Stopping Python server...');
    pythonProcess.kill('SIGTERM');
    
    // Force kill after 10 seconds
    setTimeout(() => {
      if (pythonProcess && !pythonProcess.killed) {
        console.log('[OpenClaw Gateway] Force killing Python server...');
        pythonProcess.kill('SIGKILL');
      }
      process.exit(0);
    }, 10000);
  } else {
    process.exit(0);
  }
}

// Handle shutdown signals
process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));

// Handle uncaught errors
process.on('uncaughtException', (err) => {
  console.error('[OpenClaw Gateway] Uncaught exception:', err);
  shutdown('uncaughtException');
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('[OpenClaw Gateway] Unhandled rejection at:', promise, 'reason:', reason);
  shutdown('unhandledRejection');
});

// Start the gateway
start().catch((err) => {
  console.error('[OpenClaw Gateway] Failed to start:', err);
  process.exit(1);
});
