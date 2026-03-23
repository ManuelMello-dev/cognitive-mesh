import http from 'http';
import fs from 'fs';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const PORT = process.env.PORT || 8080;
const PYTHON_PORT = process.env.COGNITIVE_MESH_PORT || 3000;
const TARGET = process.env.COGNITIVE_MESH_BASE_URL || `http://localhost:${PYTHON_PORT}`;

console.log(`[OpenClaw] Starting gateway on port ${PORT}...`);
console.log(`[OpenClaw] Proxying to backend at ${TARGET}`);

// Pre-load the dashboard HTML at startup so it is always served instantly.
// This eliminates 502s on GET / during Python startup and under cognitive loop load.
const DASHBOARD_PATH = path.join(__dirname, 'market_consciousness_dashboard.html');
let dashboardHtml = null;
try {
    dashboardHtml = fs.readFileSync(DASHBOARD_PATH, 'utf8');
    console.log(`[OpenClaw] Dashboard HTML pre-loaded (${dashboardHtml.length} bytes)`);
} catch (e) {
    console.warn('[OpenClaw] Could not pre-load dashboard HTML:', e.message);
}

// 1. Start the Python Backend
function startBackend() {
    console.log('[OpenClaw] Spawning Python cognitive-mesh backend...');
    
    const pythonProcess = spawn('python3', ['main.py'], {
        env: { ...process.env, PORT: PYTHON_PORT },
        stdio: 'inherit'
    });

    pythonProcess.on('close', (code) => {
        console.log(`[OpenClaw] Python backend exited with code ${code}`);
        process.exit(code);
    });

    pythonProcess.on('error', (err) => {
        console.error('[OpenClaw] Failed to start Python backend:', err);
    });

    // ── CRITICAL: Forward SIGTERM/SIGINT to Python so save_state runs on Railway redeploy ──
    // Railway sends SIGTERM to the Node process. Without forwarding, Python never receives it,
    // save_state never runs, and the mesh loses ALL memory (concepts, rules, patterns, goals,
    // observation history, short-term memory) on every redeploy or crash.
    // We give Python 25 seconds to flush state to Postgres before Node exits.
    let shuttingDown = false;
    const forwardSignal = (sig) => {
        if (shuttingDown) return;
        shuttingDown = true;
        console.log(`[OpenClaw] Forwarding ${sig} to Python backend (PID ${pythonProcess.pid}) — waiting up to 25s for state save...`);
        try { pythonProcess.kill(sig); } catch (e) { /* process already gone */ }
        // Hard exit after 25s in case Python hangs during shutdown
        setTimeout(() => {
            console.log('[OpenClaw] Shutdown timeout reached — forcing exit.');
            process.exit(0);
        }, 25000).unref();
    };

    process.on('SIGTERM', () => forwardSignal('SIGTERM'));
    process.on('SIGINT',  () => forwardSignal('SIGINT'));

    return pythonProcess;
}

// 2. Create the Proxy Gateway
function createGateway() {
    const proxy = createProxyMiddleware({
        target: TARGET,
        changeOrigin: true,
        ws: true,
        logLevel: 'warn',       // Reduce proxy log noise
        proxyTimeout: 30000,    // 30s proxy timeout
        timeout: 30000,         // 30s socket inactivity timeout
        onError: (err, req, res) => {
            // Only log non-ECONNREFUSED errors (backend still starting is expected)
            if (err.code !== 'ECONNREFUSED') {
                console.error('[OpenClaw] Proxy Error:', err.message);
            }
            if (!res.headersSent) {
                res.writeHead(503, { 'Content-Type': 'application/json' });
            }
            res.end(JSON.stringify({ 
                error: 'Backend service unavailable', 
                details: err.message,
                hint: 'The Python backend might still be starting up. Please wait a few seconds and refresh.'
            }));
        }
    });

    const server = http.createServer((req, res) => {
        // Health check endpoints for Railway — always fast, no proxy needed
        if (req.url === '/health' || req.url === '/healthz') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ 
                status: 'healthy', 
                gateway: 'OpenClaw',
                timestamp: new Date().toISOString()
            }));
            return;
        }

        // Serve dashboard HTML directly from the gateway — never proxy to Python for this.
        // The Python backend's handle_dashboard reads the same file from disk, but doing it
        // here means the dashboard always loads instantly even if Python is busy or starting.
        if (req.url === '/' || req.url === '/index.html') {
            if (dashboardHtml) {
                res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
                res.end(dashboardHtml);
            } else {
                // Fallback: read from disk on demand
                try {
                    const html = fs.readFileSync(DASHBOARD_PATH, 'utf8');
                    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
                    res.end(html);
                } catch (e) {
                    res.writeHead(503, { 'Content-Type': 'text/plain' });
                    res.end('Dashboard not found');
                }
            }
            return;
        }

        // Forward all other requests (all /api/* routes) to Python backend
        proxy(req, res);
    });

    // Set server-level timeout to match proxy timeout
    server.timeout = 35000;
    server.keepAliveTimeout = 65000;  // > Railway's 60s idle timeout

    server.listen(PORT, '0.0.0.0', () => {
        console.log(`[OpenClaw] Gateway is listening on 0.0.0.0:${PORT}`);
    });

    return server;
}

// Execution
startBackend();
createGateway();
