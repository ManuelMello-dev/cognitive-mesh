import http from 'http';
import fs from 'fs';
import { createProxyMiddleware } from 'http-proxy-middleware';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ── Configuration ─────────────────────────────────────────────────────────────
const PORT        = process.env.PORT || 8080;
const PYTHON_PORT = process.env.COGNITIVE_MESH_PORT || 3000;
const TARGET      = process.env.COGNITIVE_MESH_BASE_URL || `http://localhost:${PYTHON_PORT}`;

console.log(`[OpenClaw] Starting gateway on port ${PORT}...`);
console.log(`[OpenClaw] Proxying to backend at ${TARGET}`);

// ── Pre-load dashboard HTML ───────────────────────────────────────────────────
// Served directly from Node so the dashboard always loads instantly even when
// Python is starting up, busy, or restarting after a crash.
const DASHBOARD_PATH = path.join(__dirname, 'market_consciousness_dashboard.html');
let dashboardHtml = null;
try {
    dashboardHtml = fs.readFileSync(DASHBOARD_PATH, 'utf8');
    console.log(`[OpenClaw] Dashboard HTML pre-loaded (${dashboardHtml.length} bytes)`);
} catch (e) {
    console.warn('[OpenClaw] Could not pre-load dashboard HTML:', e.message);
}

// ── Python backend supervisor ─────────────────────────────────────────────────
// The Node gateway NEVER exits due to a Python crash.
// Python is restarted automatically with exponential back-off.
// Only a Railway SIGTERM (intentional shutdown) will stop the gateway.

const MAX_RESTARTS   = 20;        // hard ceiling before giving up
const BASE_DELAY_MS  = 2_000;     // 2 s minimum between restarts
const MAX_DELAY_MS   = 120_000;   // 2 min maximum back-off
const RESET_AFTER_MS = 300_000;   // reset restart counter after 5 min of stability

let pythonProcess  = null;
let restartCount   = 0;
let lastStartTime  = 0;
let shuttingDown   = false;       // true only when Node itself is shutting down

function startPython() {
    if (shuttingDown) return;
    if (restartCount >= MAX_RESTARTS) {
        console.error(`[OpenClaw] Python backend exceeded ${MAX_RESTARTS} restarts — giving up.`);
        return;
    }

    // Reset counter if Python ran stably for RESET_AFTER_MS
    const now = Date.now();
    if (lastStartTime > 0 && (now - lastStartTime) > RESET_AFTER_MS) {
        restartCount = 0;
    }

    const delay = restartCount === 0
        ? 0
        : Math.min(BASE_DELAY_MS * Math.pow(2, restartCount - 1), MAX_DELAY_MS);

    if (delay > 0) {
        console.log(`[OpenClaw] Restarting Python in ${delay / 1000}s (attempt ${restartCount + 1}/${MAX_RESTARTS})...`);
    }

    setTimeout(() => {
        if (shuttingDown) return;

        console.log('[OpenClaw] Spawning Python cognitive-mesh backend...');
        lastStartTime = Date.now();
        restartCount++;

        pythonProcess = spawn('python3', ['main.py'], {
            env:   { ...process.env, PORT: PYTHON_PORT },
            stdio: 'inherit',
            cwd:   __dirname,
        });

        pythonProcess.on('error', (err) => {
            console.error('[OpenClaw] Failed to spawn Python backend:', err.message);
            // 'close' will fire next and trigger restart
        });

        pythonProcess.on('close', (code, signal) => {
            if (shuttingDown) return;   // intentional shutdown — do not restart
            console.log(`[OpenClaw] Python exited (code=${code}, signal=${signal}) — scheduling restart...`);
            startPython();              // recursive restart with back-off
        });

    }, delay);
}

// ── Graceful shutdown ─────────────────────────────────────────────────────────
// Forward SIGTERM/SIGINT to Python so save_state() runs before Node exits.
// Give Python up to 25 s to flush state to Postgres, then hard-exit.
function handleShutdown(sig) {
    if (shuttingDown) return;
    shuttingDown = true;
    console.log(`[OpenClaw] Received ${sig} — forwarding to Python and waiting up to 25s for state save...`);

    if (pythonProcess) {
        try { pythonProcess.kill(sig); } catch (_) { /* already gone */ }
    }

    setTimeout(() => {
        console.log('[OpenClaw] Shutdown timeout reached — forcing exit.');
        process.exit(0);
    }, 25_000).unref();
}

process.on('SIGTERM', () => handleShutdown('SIGTERM'));
process.on('SIGINT',  () => handleShutdown('SIGINT'));

// ── HTTP Gateway ──────────────────────────────────────────────────────────────
function createGateway() {
    const proxy = createProxyMiddleware({
        target:       TARGET,
        changeOrigin: true,
        ws:           true,
        logLevel:     'warn',
        proxyTimeout: 30_000,
        timeout:      30_000,
        onError: (err, req, res) => {
            if (err.code !== 'ECONNREFUSED') {
                console.error('[OpenClaw] Proxy Error:', err.message);
            }
            if (!res.headersSent) {
                res.writeHead(503, { 'Content-Type': 'application/json' });
            }
            res.end(JSON.stringify({
                error:   'Backend service unavailable',
                details: err.message,
                hint:    'The Python backend is starting up or restarting. Please wait a few seconds and refresh.',
            }));
        },
    });

    const server = http.createServer((req, res) => {
        // ── Health check — always instant, never proxied ──────────────────
        if (req.url === '/health' || req.url === '/healthz') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({
                status:    'healthy',
                gateway:   'OpenClaw',
                python_up: pythonProcess !== null && !pythonProcess.killed,
                restarts:  restartCount,
                timestamp: new Date().toISOString(),
            }));
            return;
        }

        // ── Dashboard — served directly from Node, always instant ─────────
        if (req.url === '/' || req.url === '/index.html') {
            if (dashboardHtml) {
                res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
                res.end(dashboardHtml);
            } else {
                try {
                    const html = fs.readFileSync(DASHBOARD_PATH, 'utf8');
                    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
                    res.end(html);
                } catch (_) {
                    res.writeHead(503, { 'Content-Type': 'text/plain' });
                    res.end('Dashboard not available');
                }
            }
            return;
        }

        // ── All /api/* routes → Python backend ───────────────────────────
        proxy(req, res);
    });

    server.timeout          = 35_000;
    server.keepAliveTimeout = 65_000;   // > Railway's 60 s idle timeout

    server.listen(PORT, '0.0.0.0', () => {
        console.log(`[OpenClaw] Gateway is listening on 0.0.0.0:${PORT}`);
    });

    return server;
}

// ── Boot sequence ─────────────────────────────────────────────────────────────
startPython();
createGateway();
