const http = require('http');
const { createProxyMiddleware } = require('http-proxy-middleware');

function createHttpServer() {
    const server = http.createServer((req, res) => {
        // Check for health endpoints
        if (req.url === '/health' || req.url === '/healthz') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'healthy' }));
            return;
        }

        // Proxy middleware for other paths
        createProxyMiddleware({ target: 'http://localhost:3000', changeOrigin: true })(req, res);
    });
    return server;
}

module.exports = { createHttpServer };