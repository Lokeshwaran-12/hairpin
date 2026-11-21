---
title: 06 - API Reference & Full-Stack Integration
tags:
  - api
  - fastapi
  - nodejs
  - rest
  - mjpeg
  - dashboard
created: 2026-11-22
updated: 2026-11-22
---

# 🌐 06 - API Reference & Full-Stack Integration

> [!NOTE] Architecture Context
> The backend exposes high-performance REST and MJPEG endpoints via FastAPI on port 8000. Node.js Express on port 3000 provides a unified reverse proxy, static asset delivery, and frontend UI routing.

---

## 1. RESTful API Specification (FastAPI :8000)

### System & Health

| Method | Endpoint | Description | Query / Body Params | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Service health & model status | None | `{"status": "healthy", "model": "yolov8n.pt"}` |
| `GET` | `/api/stats` | Aggregated traffic analytics | None | Historical counts, violations, risk metrics |

### Video Streaming & Telemetry

| Method | Endpoint | Description | Query / Body Params | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/stream/feed` | Real-time live HUD video stream | None | `multipart/x-mixed-replace; boundary=frame` (MJPEG) |
| `GET` | `/api/stream/telemetry` | Real-time vehicle & risk data | None | JSON snapshot of active tracks, signals & TTC |
| `GET` | `/api/video/status` | Current video source info | None | Source name, resolution, FPS, running status |

### Video Source & Trigger Line Control

| Method | Endpoint | Description | Body Schema |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/video/set-source` | Change active stream | `{"source": "synthetic" \| "path/to/video.mp4"}` |
| `POST` | `/api/video/upload` | Upload recorded video file | `multipart/form-data` with `file` |
| `POST` | `/api/video/centric-line` | Adjust trigger line position | `{"enabled": true, "y_ratio": 0.52}` |
| `POST` | `/api/video/centric-line/reset` | Reset crossing counters | None |
| `POST` | `/api/video/left-lane-mode` | Toggle left-lane filtering | `{"left_lane_only": true \| false}` |

### Static Image Detection

| Method | Endpoint | Description | Request Payload | Response |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/detect` | Single-image inference | `{"image": "data:image/jpeg;base64,...", "confidence": 0.25}` | Annotated base64 image + detected vehicle objects |
| `GET` | `/api/sample` | Sample test image detection | None | Annotated sample output with bounding boxes |

### Incidents & Snapshot Archives

| Method | Endpoint | Description | Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/logs` | Fetch recent logged incidents | JSON array of SQLite incident records |
| `GET` | `/api/snapshots` | List archived emergency image files | List of saved `.jpg` snapshot paths |
| `GET` | `/api/snapshots/{filename}` | Retrieve specific incident image | Direct JPEG image file download |

---

## 2. Node.js Express Gateway Integration (`server.js`)

Node.js acts as the user-facing reverse proxy running on port 3000:
```javascript
// Express reverse proxy relaying API calls to Python backend
app.use('/api', createProxyMiddleware({
  target: 'http://127.0.0.1:8000',
  changeOrigin: true,
  ws: true,
  timeout: 30000
}));
```

---

## 3. Frontend Dashboard Architecture

The dashboard (`public/`) is implemented in clean, high-performance vanilla JavaScript (ES6+), HTML5, and CSS3 without heavy frameworks:
* **Live MJPEG Canvas**: Direct streaming feed with zero decoding latency.
* **Telemetry HUD**: Real-time signal indicators (Signal A and Signal B) with pulsing animations.
* **Interactive Centric Line Control**: Slider updating the trigger line $Y$-position in real time via `/api/video/centric-line`.
* **Left-Lane Filter Toggle**: Instant switching between `ALL LANES` and `ONLY LEFT LANE`.

---

## 4. Next Chapter

Proceed to [[07_Verification_Edge_Cases_and_Benchmarks|Chapter 07 - Verification, Edge Cases & Benchmarks]] for automated testing details.
