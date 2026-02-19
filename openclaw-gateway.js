import http from 'http';
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

    return pythonProcess;
}

// 2. Create the Proxy Gateway
function createGateway() {
    const proxy = createProxyMiddleware({
        target: TARGET,
        changeOrigin: true,
        ws: true,
        logLevel: 'debug',
        onError: (err, req, res) => {
            console.error('[OpenClaw] Proxy Error:', err.message);
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
        // Health check endpoints for Railway
        if (req.url === '/health' || req.url === '/healthz') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ 
                status: 'healthy', 
                gateway: 'OpenClaw',
                timestamp: new Date().toISOString()
            }));
            return;
        }

        // Forward all other requests to Python backend
        proxy(req, res);
    });

    server.listen(PORT, '0.0.0.0', () => {
        console.log(`[OpenClaw] Gateway is listening on 0.0.0.0:${PORT}`);
    });

    return server;
}

// Execution
startBackend();
createGateway();
