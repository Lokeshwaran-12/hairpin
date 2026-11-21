---
title: 01 - System Architecture & Design
tags:
  - architecture
  - design-patterns
  - streaming
  - concurrency
  - full-stack
created: 2026-11-22
updated: 2026-11-22
---

# 🏗️ 01 - System Architecture & Design

> [!NOTE] Architecture Objective
> Deliver deterministic, low-latency (<40ms per frame) computer vision inference and collision risk assessment on Edge and Cloud environments while serving a responsive, real-time UI dashboard to traffic operators.

---

## 1. High-Level System Architecture

The system decouples the **Heavy Compute Python AI Engine** from the **User-Facing Node.js Gateway & Web Dashboard**:

```mermaid
flowchart TD
    subgraph EdgeInput [Input & Ingestion]
        A1[CCTV RTSP Camera]
        A2[Recorded MP4/AVI Upload]
        A3[Procedural Synthetic Mountain Feed]
    end

    subgraph PythonEngine [Python FastAPI AI Engine - Port 8000]
        B[VideoStreamSource / OpenCV Capture]
        C[CLAHE Preprocessor]
        D[Dual-Head YOLOv8 Detection Engine]
        E[ByteTrack Multi-Object Tracker]
        F[Centric Virtual Trigger Line Engine]
        G[Spatial Zone Mapper & TTC Risk Predictor]
        H[Emergency Corridor Preemption Lock]
        I[HUD Visualizer & MJPEG Generator]
        J[SQLite & CSV Incident Logger]
    end

    subgraph NodeGateway [Node.js Express Gateway - Port 3000]
        K[Express API Gateway & Reverse Proxy]
        L[Static Dashboard Asset Server]
        M[Upload Multipart Stream Interceptor]
    end

    subgraph ClientUI [Traffic Control Dashboard]
        N[Web Dashboard: Live MJPEG Stream]
        O[Dynamic Telemetry HUD & Incident Tables]
        P[Virtual Centric Line & Lane Controls]
    end

    A1 -->|RTSP Stream| B
    A2 -->|Multipart Upload| M -->|Relay to :8000| B
    A3 -->|Procedural Generator| B

    B --> C --> D --> E --> F --> G --> H
    G --> I
    H --> I
    G --> J
    H --> J

    I -->|MJPEG /api/stream/feed| K
    G -->|JSON Telemetry /api/stream/telemetry| K

    K --> N
    K --> O
    P -->|Control POSTs| K -->|Proxy| F
```

---

## 2. Dual-Engine Service Topology

### Why Decouple FastAPI and Node.js?

1. **GIL Isolation**: Python's Global Interpreter Lock (GIL) limits multi-threaded CPU concurrency. Keeping HTTP asset serving, file uploads, and browser client connections in Node.js prevents heavy client requests from starving the real-time vision inference pipeline.
2. **Edge Scalability**: The Python AI inference engine can be hosted on a dedicated GPU edge device (e.g., NVIDIA Jetson AGX Orin, AWS EC2 G4dn), while the Node.js gateway runs on an inexpensive lightweight server or container.
3. **Resilience**: If the video stream encounters a malformed frame or codec reset, the Node.js gateway handles reconnection gracefully without crashing or dropping browser sessions.

---

## 3. Multi-Threaded Streaming Architecture (`StreamManager`)

The backend implements a dedicated worker thread architecture within `backend_service.py` to decouple video frame acquisition from API request handling:

```mermaid
sequenceDiagram
    autonumber
    participant V as VideoStreamSource
    participant W as StreamManager Worker Thread
    participant Y as YOLOv8 Detector
    participant T as ByteTrack
    participant P as CollisionPredictor
    participant L as Thread Lock
    participant C as Client (Node / Browser)

    loop Every Frame (~28 FPS)
        W->>V: Read next frame
        V-->>W: Raw BGR Frame
        W->>Y: Process CLAHE + Detect Objects
        Y-->>W: Base + Emergency Bounding Boxes
        W->>T: Update Track History & Calibrate Speed
        T-->>W: Persistent Tracked Objects
        W->>P: Evaluate Spatial Zones, Line Crossings & TTC
        P-->>W: Risk Assessment + Signal States
        W->>L: Acquire Lock & Update current_frame / assessment
        L-->>W: Release Lock
    end

    Note over C,W: Asynchronous Client Request
    C->>W: GET /api/stream/feed
    W->>L: Read latest annotated frame
    L-->>W: Encode JPEG buffer
    W-->>C: Stream MJPEG chunk boundary
```

### Thread Safety & Natural Cadence Pacing

- **Mutex Synchronization (`threading.Lock`)**: The `_lock` protects `current_frame`, `current_assessment`, and `current_vehicles`. Readers (MJPEG streaming generator and telemetry polling endpoints) acquire the lock only for minimal memory copies (<0.2ms).
- **Controlled Playback Cadence**:
  $$\Delta t_{\text{sleep}} = \max\left(0.005\text{s},\, \frac{1}{\text{target FPS}} - \Delta t_{\text{processing}}\right)$$
  This guarantees smooth, natural video playback (~28 FPS) without runaway CPU loops or hyper-speed fast-forwarding during recorded file playback.

---

## 4. Hardware and Deployment Requirements

| Component | Minimum Specification | Recommended Production Spec |
| :--- | :--- | :--- |
| **CPU** | Intel Core i5 / AMD Ryzen 5 (4+ cores) | Intel Core i7 13th Gen / AMD Ryzen 7 |
| **GPU** | Integrated Graphics (CPU inference supported) | NVIDIA RTX 3060 / NVIDIA Jetson Orin Nano (4GB+) |
| **RAM** | 8 GB DDR4 | 16 GB DDR4/DDR5 |
| **OS** | Windows 10/11, Ubuntu 20.04/22.04 LTS | Ubuntu 22.04 LTS Server |
| **Network** | 100 Mbps Ethernet | 1 Gbps PoE Ethernet (for high-bitrate RTSP feeds) |

---

## 5. Next Chapter

Continue to [[02_Vision_and_Detection_Pipeline|Chapter 02 - Computer Vision & Detection Pipeline]] for the multi-stage detection architecture and bus protection mechanics.
