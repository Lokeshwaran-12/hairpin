/**
 * Node.js Express Server for Hairpin Bend Traffic Monitoring System.
 * Serves the modern frontend dashboard and orchestrates the Python AI service.
 */
const express = require('express');
const cors = require('cors');
const path = require('path');
const http = require('http');
const fs = require('fs');
const multer = require('multer');
const { spawn } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = 'http://127.0.0.1:8000';

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Ensure upload directory exists
const UPLOAD_DIR = path.join(__dirname, 'data', 'uploaded_videos');
if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Multer storage for uploaded video files
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, UPLOAD_DIR),
    filename: (req, file, cb) => {
        const cleanName = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
        cb(null, `${Date.now()}_${cleanName}`);
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: 300 * 1024 * 1024 }
});

// Serve frontend static assets
app.use(express.static(path.join(__dirname, 'public')));

// Dedicated Video Upload Handler (saves video and tells Python backend to stream with YOLO)
app.post('/api/video/upload', upload.single('video'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: 'No video file provided.' });
    }

    const videoPath = path.join(UPLOAD_DIR, req.file.filename);
    console.log(`[Video] Uploaded: ${req.file.originalname} -> ${videoPath}`);

    const postData = JSON.stringify({
        video_path: videoPath,
        filename: req.file.originalname
    });

    // Resilient relay with automatic retry if Python service is still initializing
    let attempts = 0;
    const maxAttempts = 10;
    const delayMs = 1000;

    while (attempts < maxAttempts) {
        attempts++;
        try {
            const result = await new Promise((resolve, reject) => {
                const options = {
                    hostname: '127.0.0.1',
                    port: 8000,
                    path: '/api/video/set-source',
                    method: 'POST',
                    headers: {
                        'content-type': 'application/json',
                        'content-length': Buffer.byteLength(postData)
                    },
                    timeout: 8000
                };

                const pyReq = http.request(options, (pyRes) => {
                    let responseBody = '';
                    pyRes.on('data', chunk => responseBody += chunk);
                    pyRes.on('end', () => {
                        resolve({ statusCode: pyRes.statusCode, body: responseBody });
                    });
                });

                pyReq.on('error', (err) => reject(err));
                pyReq.on('timeout', () => {
                    pyReq.destroy();
                    reject(new Error('Connection timed out'));
                });

                pyReq.write(postData);
                pyReq.end();
            });

            if (result.statusCode >= 200 && result.statusCode < 300) {
                return res.status(result.statusCode).json({
                    success: true,
                    filename: req.file.originalname,
                    video_path: videoPath,
                    stream_url: '/api/stream/feed'
                });
            } else {
                return res.status(result.statusCode).send(result.body);
            }
        } catch (err) {
            console.log(`[Video Relay] Connecting to Python AI service (attempt ${attempts}/${maxAttempts})...`);
            if (attempts < maxAttempts) {
                await new Promise(r => setTimeout(r, delayMs));
            } else {
                console.error(`[Video Relay Error] Failed after ${maxAttempts} attempts: ${err.message}`);
                return res.status(502).json({
                    error: 'Failed to notify Python AI service of video upload.',
                    details: 'Python AI backend is still initializing or unavailable. Please retry in a few seconds.'
                });
            }
        }
    }
});

// Throttled logging for transient proxy startup errors
let lastProxyErrorTime = 0;

// Forward API requests to Python FastAPI service
app.use('/api', (req, res) => {
    const targetUrl = new URL(req.originalUrl, PYTHON_API_URL);

    const headers = {
        ...req.headers,
        host: `${targetUrl.hostname}:${targetUrl.port}`
    };

    let bodyData = null;
    if (req.body && Object.keys(req.body).length > 0) {
        bodyData = JSON.stringify(req.body);
        headers['content-type'] = 'application/json';
        headers['content-length'] = Buffer.byteLength(bodyData);
    } else if (['GET', 'HEAD'].includes(req.method)) {
        delete headers['content-length'];
    }

    const options = {
        hostname: targetUrl.hostname,
        port: targetUrl.port,
        path: targetUrl.pathname + targetUrl.search,
        method: req.method,
        headers: headers
    };

    const proxyReq = http.request(options, (proxyRes) => {
        res.writeHead(proxyRes.statusCode, proxyRes.headers);
        proxyRes.pipe(res, { end: true });
    });

    proxyReq.on('error', (err) => {
        const now = Date.now();
        if (now - lastProxyErrorTime > 4000) {
            console.warn(`[Node Proxy] Waiting for Python AI backend (${err.code || err.message})...`);
            lastProxyErrorTime = now;
        }
        if (!res.headersSent) {
            res.status(503).json({
                error: 'Python AI backend is initializing.',
                details: err.message
            });
        }
    });

    if (['GET', 'HEAD'].includes(req.method)) {
        proxyReq.end();
    } else if (bodyData) {
        proxyReq.write(bodyData);
        proxyReq.end();
    } else {
        req.pipe(proxyReq, { end: true });
    }
});

// Serve SPA fallback
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Process Management: Auto-spawn Python AI backend if not already running
let pythonProcess = null;
let isPythonReady = false;

function getPythonCommand() {
    if (process.env.PYTHON) return process.env.PYTHON;
    const candidates = [
        'C:\\Program Files\\Python311\\python.exe',
        'python',
        'python3'
    ];
    for (const cand of candidates) {
        if (cand.includes('\\') && fs.existsSync(cand)) return cand;
    }
    return 'python';
}

function checkPythonHealth() {
    return new Promise((resolve) => {
        const req = http.get(`${PYTHON_API_URL}/api/health`, (res) => {
            resolve(res.statusCode === 200);
        });
        req.on('error', () => resolve(false));
        req.setTimeout(1500, () => {
            req.destroy();
            resolve(false);
        });
    });
}

async function ensurePythonBackend() {
    const isRunning = await checkPythonHealth();
    if (isRunning) {
        isPythonReady = true;
        console.log('[System] Python AI Backend is already running on port 8000.');
        return;
    }

    const pythonCmd = getPythonCommand();
    console.log(`[System] Spawning Python AI Backend (${pythonCmd} backend_service.py)...`);

    // Use shell: false to eliminate DEP0190 warning on Node 22+
    pythonProcess = spawn(pythonCmd, ['backend_service.py'], {
        cwd: __dirname,
        stdio: 'inherit',
        shell: false
    });

    pythonProcess.on('error', (err) => {
        console.error('[System] Failed to start Python process:', err);
    });

    pythonProcess.on('exit', (code) => {
        isPythonReady = false;
        console.log(`[System] Python process exited with code ${code}`);
    });

    // Wait until backend becomes responsive
    let attempts = 0;
    while (attempts < 35) {
        await new Promise(r => setTimeout(r, 600));
        attempts++;
        if (await checkPythonHealth()) {
            isPythonReady = true;
            console.log('[System] Python AI Backend successfully connected on port 8000!');
            break;
        }
    }
}

// Graceful cleanup
function cleanup() {
    if (pythonProcess) {
        console.log('\n[System] Shutting down Python backend process...');
        try {
            pythonProcess.kill();
        } catch (e) {
            // Ignore
        }
    }
    process.exit();
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);

// Start Server
app.listen(PORT, async () => {
    console.log('==================================================================');
    console.log('  YOLO Hairpin Bend Traffic Monitoring - Node.js Frontend');
    console.log('==================================================================');
    console.log(`  Frontend Dashboard: http://localhost:${PORT}`);
    console.log(`  Python AI Engine:   ${PYTHON_API_URL}`);
    console.log('==================================================================');
    await ensurePythonBackend();
});
