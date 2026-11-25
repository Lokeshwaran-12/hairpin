# 🏔️ Complete Setup & Run Guide: Hairpin Bend AI Monitoring System

This document contains everything you need to set up, verify, and run the **AI Hairpin Bend Traffic Monitoring & Collision Prevention System** on your local machine and replicate the exact same setup on any other laptop (macOS, Windows, or Linux).

---

## 📋 System Prerequisites

Ensure you have the following installed before beginning:

1. **Python**: Version `3.10`, `3.11`, or `3.12`
   - Check with: `python3 --version` or `python --version`
2. **Node.js**: Version `18+` (tested on v20–v26) & **npm**
   - Check with: `node --version` and `npm --version`
3. **Git**:
   - Check with: `git --version`

---

## ⚡ Quickstart (One-Click Setup & Launch)

### 1. Clone Repository
```bash
git clone https://github.com/shahiilr06/hairpin_detection-.git
cd hairpin_detection-
```

### 2. Set Up Python Virtual Environment
Creating a virtual environment ensures dependencies do not conflict with system packages:

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install lapx
```

**On Windows (PowerShell or Command Prompt):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install lapx
```

> **Note on `lapx`**: `lapx` is the high-performance linear assignment solver required by Ultralytics ByteTrack for multi-vehicle trajectory tracking across consecutive frames.

### 3. Install Frontend Node.js Dependencies
In the root directory of the repository:
```bash
npm install
```

---

## 🚀 How to Run the Application

The system uses a dual-engine architecture:
- **Port 8000**: FastAPI AI Engine (YOLOv8 vehicle detection, ByteTrack tracking, TTC collision risk calculation, incident logging).
- **Port 3000**: Node.js Express Gateway & Web Operations Dashboard.

### Option A: Automatic Launch (Single Command - Recommended)

**On macOS / Linux:**
```bash
./scripts/start_system.sh
```
*Alternatively, you can run:*
```bash
npm start
```

**On Windows:**
Double-click `scripts\start_system.bat` or run:
```powershell
npm start
```

> The Node server automatically locates the virtual environment (`.venv`), starts `backend_service.py` on port 8000, and serves the dashboard on port 3000!

---

### Option B: Manual Two-Terminal Launch

If you prefer to see dedicated logs for both services in separate terminal windows:

#### Terminal 1: Python AI Backend (Port 8000)
```bash
# Activate your virtual environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

python backend_service.py
```
*Wait until you see `Uvicorn running on http://127.0.0.1:8000`.*

#### Terminal 2: Node.js Frontend Gateway (Port 3000)
```bash
node server.js
```

---

## 🌐 Accessing the Application

Once running, open your web browser:

| Interface | URL | Description |
| :--- | :--- | :--- |
| **Web Operations Dashboard** | [`http://localhost:3000`](http://localhost:3000) | Full interactive telemetry, live HUD video stream, speed metrics, traffic lights, and snapshot incident audit |
| **FastAPI Backend Docs** | [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) | Interactive Swagger REST API documentation and test endpoints |
| **Backend Health Check** | [`http://localhost:3000/api/health`](http://localhost:3000/api/health) | Verifies detector and pipeline operational status |

---

## 🖥️ Running in Command-Line (CLI) Mode

You can also run the computer vision pipeline standalone directly in your terminal:

```bash
# Run simulation feed with live OpenCV display:
python main_cli.py --source synthetic

# Run with a video file:
python main_cli.py --source data/sample_videos/your_video.mp4

# Run with connected webcam (Camera index 0):
python main_cli.py --source 0

# Run in headless mode (no GUI window, logs to database):
python main_cli.py --source synthetic --headless
```

---

## 🧪 Running Automated Tests

The repository includes a test suite covering the dual-head detector, ByteTrack tracking, TTC collision prediction, and event logging:

```bash
# Using pytest via your activated virtual environment:
python -m pytest tests/ -v
```

Expected output:
```
tests/test_backend_api.py ...............                                [ 37%]
tests/test_collision_predictor.py ..........                             [ 62%]
tests/test_detector.py .........                                         [ 85%]
tests/test_event_logger.py ......                                        [100%]
============================== 52 passed in 5.04s ==============================
```

---

## 🔧 Project Configuration

All parameters can be tuned in [`config/config.yaml`](config/config.yaml):
- **Video Source**: Change default input from `synthetic` to a video file path or RTSP CCTV stream.
- **TTC Thresholds**: Adjust critical Time-To-Conflict collision warning windows (`ttc_critical_threshold: 3.0`).
- **Hazard Multipliers**: Adjust safety buffer weights for heavy commercial vehicles (`bus: 1.5`, `truck: 1.5`).
- **Detection Zones**: Configure geometric polygon coordinates for Arm A, Arm B, and Conflict Apex.

---

## 🛠️ Common Troubleshooting

1. **`ModuleNotFoundError: No module named 'lap'` or `'lapx'`**:
   - Solution: Run `pip install lapx` inside your virtual environment.

2. **`zsh: command not found: python` (macOS)**:
   - Solution: On macOS, system Python is called `python3`. Use `python3 -m venv .venv` and `./scripts/start_system.sh`.

3. **Port already in use (`EADDRINUSE 3000` or `8000`)**:
   - On macOS/Linux: `lsof -i :3000` or `lsof -i :8000`, then `kill -9 <PID>`.
   - On Windows: `netstat -ano | findstr :3000`, then `taskkill /PID <PID> /F`.

4. **First launch model download**:
   - On the very first run, Ultralytics will automatically download the base `yolov8n.pt` weights (~6.2 MB) into the repository root. Ensure you have an active internet connection on the first boot.
