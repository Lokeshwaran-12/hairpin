"""
Backend FastAPI Service:
Exposes REST and Streaming endpoints for YOLOv8n Hairpin Bend Traffic Monitoring,
Detection, Tracking, Collision Prediction, and Incident Archival.
"""
import os
import io
import sys
import time
import base64
import tempfile
import threading
from typing import Optional, List
import cv2
import yaml
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# Ensure ultralytics writable config dir
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics_config")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from src.detection.yolo_detector import YOLODetector, Detection
from src.preprocessing.preprocessor import FramePreprocessor
from src.capture.video_stream import VideoStreamSource
from src.tracking.tracker import VehicleTracker, TrackedVehicle
from src.risk_assessment.collision_predictor import CollisionPredictor, RiskLevel
from src.warning.visualizer import HUDVisualizer
from src.logging_storage.event_logger import EventLogger
from src.utils.synthetic_feed import generate_demo_video

app = FastAPI(
    title="Hairpin Bend Accident Prevention API",
    version="1.0.0",
    description="YOLOv8n-based predictive traffic safety service"
)

# Enable CORS for Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config_path = "config/config.yaml"
cfg = {}
if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

# Global models and utilities
detector_model = YOLODetector(weights_path="yolov8n.pt", confidence_threshold=0.20)
preprocessor = FramePreprocessor(enable_clahe=True)
visualizer = HUDVisualizer(show_zones=True, show_trajectories=True)
predictor = CollisionPredictor(zones_config=cfg.get("zones", {}))
event_logger = EventLogger()

# Live Video Stream State
class StreamManager:
    def __init__(self):
        self.source = "synthetic"
        self.video_name = "Simulation Highway Curve"
        self.stream = None
        self.tracker = None
        self.is_running = False
        self.current_frame = None
        self.current_assessment = None
        self.current_vehicles = []
        self._lock = threading.Lock()
        self._thread = None

        # Centric Detection Line (Virtual Trigger Line) State
        self.centric_line_enabled = True
        self.centric_line_y_ratio = 0.52
        self.filter_left_lane_only = False
        self.total_line_crossings = 0
        self.uphill_crossings = 0
        self.downhill_crossings = 0
        self.left_lane_crossings = 0
        self.crossed_vehicle_ids = set()
        self.left_lane_alert_active = False

    def start_if_needed(self):
        """Starts live stream worker only when requested."""
        if not self.is_running:
            self._start_stream()

    def _start_stream(self):
        with self._lock:
            if self.stream:
                try:
                    self.stream.release()
                except Exception:
                    pass

            real_source = self.source
            self.stream = VideoStreamSource(source=real_source, width=1280, height=720, fps=30)
            self.tracker = VehicleTracker(model_path="yolov8n.pt", conf=0.18)
            self.is_running = True

            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._worker, daemon=True)
                self._thread.start()

    def set_source(self, source: str, name: Optional[str] = None):
        self.source = source
        self.video_name = name or (os.path.basename(source) if source != "synthetic" else "Simulation Highway Curve")
        with self._lock:
            self.crossed_vehicle_ids.clear()
            self.total_line_crossings = 0
            self.uphill_crossings = 0
            self.downhill_crossings = 0
            self.left_lane_crossings = 0
            self.left_lane_alert_active = False
        self._start_stream()

    def configure_centric_line(self, enabled: bool, y_ratio: float):
        with self._lock:
            self.centric_line_enabled = enabled
            self.centric_line_y_ratio = max(0.15, min(0.85, y_ratio))

    def _worker(self):
        while True:
            if not self.is_running or not self.stream or not self.stream.is_opened:
                time.sleep(0.05)
                continue

            try:
                ret, frame = self.stream.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

                # Process frame with CLAHE & YOLO ByteTrack
                proc = preprocessor.process(frame)
                tracked_v = self.tracker.track(proc)

                h_f, w_f = frame.shape[:2]
                lane_boundary_x = int(w_f * 0.50)

                # Classify lane for each vehicle (Left lane: cx <= road center)
                for v in tracked_v:
                    cx = v.center[0]
                    v.is_left_lane = (cx <= lane_boundary_x)

                # Mode: Only Left Lane Detection
                if self.filter_left_lane_only:
                    tracked_v = [v for v in tracked_v if getattr(v, "is_left_lane", False)]

                assessment = predictor.evaluate(tracked_v, frame.shape)

                # Centric Detection Line Evaluation & Vehicle Marking
                line_y = int(h_f * self.centric_line_y_ratio)
                is_left_passing_active = False

                for v in tracked_v:
                    y1, y2 = v.box[1], v.box[3]
                    cx, cy = v.center
                    is_touching = (y1 <= line_y <= y2)
                    v.is_touching_line = is_touching

                    crossed_now = is_touching
                    if not crossed_now and len(v.history) >= 2:
                        prev_y = v.history[-2][1]
                        crossed_now = (prev_y < line_y <= cy) or (prev_y > line_y >= cy)

                    if v.track_id not in self.crossed_vehicle_ids:
                        if crossed_now:
                            self.crossed_vehicle_ids.add(v.track_id)
                            self.total_line_crossings += 1

                            if len(v.history) >= 2 and cy != v.history[0][1]:
                                dir_str = "DOWNHILL" if cy > v.history[0][1] else "UPHILL"
                            elif v.velocity_vector[1] != 0:
                                dir_str = "DOWNHILL" if v.velocity_vector[1] > 0 else "UPHILL"
                            else:
                                dir_str = "UPHILL" if v.direction_status == "Approaching" else "DOWNHILL"

                            v.has_crossed_line = True
                            v.cross_direction = dir_str
                            if dir_str == "UPHILL":
                                self.uphill_crossings += 1
                            else:
                                self.downhill_crossings += 1

                            if getattr(v, "is_left_lane", False):
                                self.left_lane_crossings += 1
                    else:
                        v.has_crossed_line = True

                    # Left lane passing identification
                    if getattr(v, "is_left_lane", False) and (v.has_crossed_line or v.is_touching_line):
                        v.is_left_passing = True
                        if v.is_touching_line or (v.has_crossed_line and len(v.history) <= 50):
                            is_left_passing_active = True

                self.left_lane_alert_active = is_left_passing_active

                # Critical Red Alert when vehicle in left lane crosses centric line
                if is_left_passing_active:
                    assessment.overall_level = RiskLevel.CRITICAL
                    assessment.signal_a_state = "RED"
                    assessment.signal_a_message = "RED ALERT - LEFT PASSING HAZARD"
                    assessment.signal_b_state = "RED"
                    assessment.signal_b_message = "RED ALERT - OPPOSING CAR IN LEFT LANE"
                    note = "CRITICAL RED ALERT: Left lane car crossing centric detection line"
                    if note not in assessment.hazard_notes:
                        assessment.hazard_notes.insert(0, note)

                # Draw HUD with visible marked bounding boxes, centric detection line & signals
                annotated = visualizer.render(
                    frame=frame,
                    vehicles=tracked_v,
                    assessment=assessment,
                    scaled_zones=predictor.scaled_zones,
                    centric_line_y=line_y,
                    centric_line_enabled=self.centric_line_enabled,
                    total_crossings=self.total_line_crossings,
                    is_left_lane_alert=is_left_passing_active
                )

                # Log Warning, Critical, or Emergency Preemption incidents
                if assessment.overall_level >= RiskLevel.WARNING or getattr(assessment, "emergency_preemption_active", False):
                    event_logger.log_event(assessment, tracked_v, annotated)

                with self._lock:
                    self.current_frame = annotated
                    self.current_assessment = assessment
                    self.current_vehicles = tracked_v

            except Exception as e:
                print(f"[Worker Error] Exception during frame processing: {e}")

            time.sleep(0.03)  # ~30 FPS throttle

stream_manager = StreamManager()


def encode_image_base64(img: np.ndarray) -> str:
    """Encodes OpenCV BGR image to base64 JPEG string."""
    ret, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        return ""
    return f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "YOLOv8 Hairpin Safety Service",
        "model": "yolov8n.pt",
        "target_classes": ["car", "motorcycle", "bus", "truck"]
    }


class DetectPayload(BaseModel):
    image: Optional[str] = None
    confidence: float = 0.25
    iou: float = 0.45
    classes: str = "car,motorcycle,bus,truck"
    enable_clahe: bool = True
    box_thickness: int = 2
    show_conf: bool = True
    show_centroids: bool = True


@app.post("/api/detect")
def detect_image(payload: Optional[DetectPayload] = None):
    """
    Runs YOLOv8n detection, risk assessment, and bounding box rendering on image.
    Executes in FastAPI worker threadpool to prevent blocking the event loop.
    """
    confidence = payload.confidence if payload else 0.25
    iou = payload.iou if payload else 0.45
    classes = payload.classes if payload else "car,motorcycle,bus,truck"
    enable_clahe = payload.enable_clahe if payload else True
    box_thickness = payload.box_thickness if payload else 2
    show_conf = payload.show_conf if payload else True
    show_centroids = payload.show_centroids if payload else True

    img = None
    if payload and payload.image:
        try:
            b64_image = payload.image
            if b64_image.startswith("data:"):
                _, b64_str = b64_image.split(",", 1)
            else:
                b64_str = b64_image
            img_bytes = base64.b64decode(b64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[API] Error decoding base64 image: {e}")

    # Fallback to sample image if decoding failed or image not supplied
    if img is None:
        sample_path = "data/sample_videos/images.jpeg"
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
        else:
            raise HTTPException(status_code=400, detail="Could not decode image and sample image was not found.")

    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image data.")

    h_orig, w_orig = img.shape[:2]

    # Parse target classes with multi-class mapping for trucks and haulers
    class_name_map = {"car": [2], "motorcycle": [3], "bus": [5], "truck": [6, 7]}
    selected_class_names = [c.strip().lower() for c in classes.split(",") if c.strip().lower() in class_name_map]
    if not selected_class_names:
        selected_class_names = ["car", "motorcycle", "bus", "truck"]
    target_ids = []
    for c in selected_class_names:
        target_ids.extend(class_name_map.get(c, []))
    target_ids = list(set(target_ids))

    # Preprocessing
    custom_preproc = FramePreprocessor(enable_clahe=enable_clahe)
    proc_img = custom_preproc.process(img)

    # Detection (reuse pre-loaded global model)
    detector_model.confidence_threshold = confidence
    detector_model.iou_threshold = iou
    detector_model.target_classes = target_ids
    detections = detector_model.detect(proc_img)

    # Class summary
    summary = {"total": len(detections), "car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "ambulance": 0}
    for d in detections:
        name = d.class_name if d.class_name in summary else "car"
        summary[name] += 1

    # Render clean bounding boxes
    box_img = YOLODetector.draw_bounding_boxes(
        frame=img,
        detections=detections,
        show_conf=show_conf,
        show_centroids=show_centroids,
        line_thickness=box_thickness
    )

    # Run collision risk assessment on standardized resolution
    std_w, std_h = 1280, 720
    frame_std = cv2.resize(img, (std_w, std_h))
    scale_x = std_w / float(w_orig)
    scale_y = std_h / float(h_orig)

    tracked_v = []
    for idx, d in enumerate(detections, 1):
        sx1 = int(d.box[0] * scale_x)
        sy1 = int(d.box[1] * scale_y)
        sx2 = int(d.box[2] * scale_x)
        sy2 = int(d.box[3] * scale_y)
        scx = int((sx1 + sx2) / 2)
        scy = int((sy1 + sy2) / 2)
        line_y = int(std_h * 0.52)
        has_crossed = (scy >= line_y)
        is_touching = (sy1 <= line_y <= sy2)

        is_heavy = (d.class_name in ["truck", "bus"]) or (d.width * d.height >= 18000 and (d.width > 130 or d.height > 110))
        is_left = (scx <= int(std_w * 0.50))
        is_left_passing = is_left and (has_crossed or is_touching)

        tracked_v.append(TrackedVehicle(
            track_id=idx,
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            box=[sx1, sy1, sx2, sy2],
            center=(scx, scy),
            history=[(scx, scy)],
            speed_kmh=35.0,
            has_crossed_line=has_crossed,
            is_touching_line=is_touching,
            cross_direction="FORWARD" if has_crossed else "APPROACHING",
            is_heavy_vehicle=is_heavy,
            is_left_lane=is_left,
            is_left_passing=is_left_passing,
            is_emergency=d.is_emergency,
            emergency_type=d.emergency_type
        ))

    assessment = predictor.evaluate(tracked_v, (std_h, std_w))

    has_left_passing_alert = any(getattr(v, "is_left_passing", False) for v in tracked_v)
    if has_left_passing_alert:
        assessment.overall_level = RiskLevel.CRITICAL
        assessment.signal_a_state = "RED"
        assessment.signal_a_message = "RED ALERT - LEFT PASSING HAZARD"
        assessment.signal_b_state = "RED"
        assessment.signal_b_message = "RED ALERT - OPPOSING CAR IN LEFT LANE"
        assessment.hazard_notes.insert(0, "CRITICAL RED ALERT: Left lane car crossing centric line")

    line_y = int(std_h * 0.52)
    hud_img = visualizer.render(
        frame=frame_std,
        vehicles=tracked_v,
        assessment=assessment,
        scaled_zones=predictor.scaled_zones,
        centric_line_y=line_y,
        centric_line_enabled=True,
        total_crossings=len([v for v in tracked_v if v.has_crossed_line]),
        is_left_lane_alert=has_left_passing_alert
    )

    # Base64 encodings
    b64_boxes = encode_image_base64(box_img)
    b64_hud = encode_image_base64(hud_img)
    b64_orig = encode_image_base64(img)

    # Detections list
    detection_items = []
    for idx, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d.box
        detection_items.append({
            "index": idx,
            "class_name": d.class_name.upper(),
            "confidence": round(d.confidence * 100, 1),
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "width": int(d.width),
            "height": int(d.height),
            "center": [int(d.center[0]), int(d.center[1])]
        })

    level_names = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}

    return {
        "success": True,
        "image_width": w_orig,
        "image_height": h_orig,
        "summary": summary,
        "detections": detection_items,
        "risk_assessment": {
            "level": int(assessment.overall_level),
            "label": level_names.get(int(assessment.overall_level), "UNKNOWN"),
            "signal_a_state": assessment.signal_a_state,
            "signal_a_message": assessment.signal_a_message,
            "signal_b_state": assessment.signal_b_state,
            "signal_b_message": assessment.signal_b_message,
            "active_arm_a": assessment.active_vehicles_arm_a,
            "active_arm_b": assessment.active_vehicles_arm_b,
            "active_apex": assessment.active_vehicles_apex,
            "min_ttc": assessment.min_ttc,
            "emergency_preemption_active": getattr(assessment, "emergency_preemption_active", False),
            "emergency_vehicle_count": getattr(assessment, "emergency_vehicle_count", 0),
            "emergency_signal_state": getattr(assessment, "emergency_signal_state", "NORMAL"),
            "hazard_notes": assessment.hazard_notes
        },
        "images": {
            "boxes_only": b64_boxes,
            "full_hud": b64_hud,
            "original": b64_orig
        }
    }


@app.get("/api/sample")
def get_sample_info():
    """Returns sample image base64 if present."""
    sample_path = "data/sample_videos/images.jpeg"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample image not found.")
    img = cv2.imread(sample_path)
    if img is None:
        raise HTTPException(status_code=500, detail="Could not read sample image.")
    return {
        "filename": "images.jpeg",
        "width": img.shape[1],
        "height": img.shape[0],
        "image": encode_image_base64(img)
    }


@app.get("/api/stream/feed")
def stream_feed():
    """MJPEG live video stream endpoint."""
    stream_manager.start_if_needed()
    def _generator():
        while True:
            frame = stream_manager.current_frame
            if frame is not None:
                ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n\r\n")
            time.sleep(0.035)

    return StreamingResponse(_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/stream/telemetry")
def stream_telemetry():
    """Returns real-time telemetry from active video stream."""
    assessment = stream_manager.current_assessment
    vehicles = stream_manager.current_vehicles

    level_names = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}

    v_data = []
    for v in vehicles:
        v_data.append({
            "id": v.track_id,
            "class_name": v.class_name.upper(),
            "speed_kmh": v.speed_kmh,
            "zone": v.current_zone,
            "direction": v.direction_status,
            "confidence": round(v.confidence * 100, 1),
            "is_emergency": getattr(v, "is_emergency", False),
            "emergency_type": getattr(v, "emergency_type", None)
        })

    centric_data = {
        "enabled": stream_manager.centric_line_enabled,
        "y_ratio": stream_manager.centric_line_y_ratio,
        "filter_left_lane_only": stream_manager.filter_left_lane_only,
        "total_crossings": stream_manager.total_line_crossings,
        "uphill_crossings": stream_manager.uphill_crossings,
        "downhill_crossings": stream_manager.downhill_crossings,
        "left_lane_crossings": stream_manager.left_lane_crossings,
        "active_touching": len([v for v in vehicles if getattr(v, "is_touching_line", False)]),
        "red_alert": stream_manager.left_lane_alert_active,
        "red_alert_reason": "CRITICAL: Left lane vehicle crossing centric line" if stream_manager.left_lane_alert_active else None
    }

    if assessment is None:
        return {
            "source": stream_manager.source,
            "risk_level": 0,
            "risk_label": "SAFE",
            "signal_a": {"state": "GREEN", "message": "BEND CLEAR"},
            "signal_b": {"state": "GREEN", "message": "BEND CLEAR"},
            "vehicles": [],
            "centric_line": centric_data,
            "emergency_preemption": {
                "active": False,
                "count": 0,
                "state": "NORMAL",
                "notes": []
            }
        }

    return {
        "source": stream_manager.source,
        "risk_level": int(assessment.overall_level),
        "risk_label": level_names.get(int(assessment.overall_level), "UNKNOWN"),
        "min_ttc": assessment.min_ttc,
        "signal_a": {
            "state": assessment.signal_a_state,
            "message": assessment.signal_a_message
        },
        "signal_b": {
            "state": assessment.signal_b_state,
            "message": assessment.signal_b_message
        },
        "counts": {
            "arm_a": assessment.active_vehicles_arm_a,
            "arm_b": assessment.active_vehicles_arm_b,
            "apex": assessment.active_vehicles_apex,
            "total": len(vehicles)
        },
        "vehicles": v_data,
        "centric_line": centric_data,
        "emergency_preemption": {
            "active": getattr(assessment, "emergency_preemption_active", False),
            "count": getattr(assessment, "emergency_vehicle_count", 0),
            "state": getattr(assessment, "emergency_signal_state", "NORMAL"),
            "notes": assessment.hazard_notes
        }
    }


class CentricLineConfig(BaseModel):
    enabled: bool = True
    y_ratio: float = 0.52


@app.post("/api/video/centric-line")
def configure_centric_line(req: CentricLineConfig):
    """Configures centric detection line position and state."""
    stream_manager.configure_centric_line(req.enabled, req.y_ratio)
    return {
        "success": True,
        "enabled": stream_manager.centric_line_enabled,
        "y_ratio": stream_manager.centric_line_y_ratio,
        "total_crossings": stream_manager.total_line_crossings
    }


@app.post("/api/video/centric-line/reset")
def reset_centric_line():
    """Resets centric line crossing counters."""
    with stream_manager._lock:
        stream_manager.crossed_vehicle_ids.clear()
        stream_manager.total_line_crossings = 0
        stream_manager.uphill_crossings = 0
        stream_manager.downhill_crossings = 0
        stream_manager.left_lane_crossings = 0
        stream_manager.left_lane_alert_active = False
    return {"success": True, "total_crossings": 0, "left_lane_crossings": 0}


class LeftLaneModeConfig(BaseModel):
    enabled: bool = False


@app.post("/api/video/left-lane-mode")
def configure_left_lane_mode(req: LeftLaneModeConfig):
    """Toggles 'Only Left Lane Detection' filter."""
    with stream_manager._lock:
        stream_manager.filter_left_lane_only = req.enabled
    return {
        "success": True,
        "filter_left_lane_only": stream_manager.filter_left_lane_only
    }


class VideoSetSourceRequest(BaseModel):
    video_path: str
    filename: Optional[str] = None


@app.post("/api/video/set-source")
def set_video_source(req: VideoSetSourceRequest):
    """Sets the active live video stream source to an uploaded video."""
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=404, detail=f"Video file not found at {req.video_path}")
    name = req.filename or os.path.basename(req.video_path)
    stream_manager.set_source(req.video_path, name=name)
    return {
        "success": True,
        "status": "active",
        "video_name": name,
        "video_path": req.video_path
    }


@app.post("/api/video/upload")
async def upload_video_direct(file: UploadFile = File(...)):
    """Direct video upload endpoint for live detection streaming."""
    upload_dir = "data/uploaded_videos"
    os.makedirs(upload_dir, exist_ok=True)
    clean_filename = f"{int(time.time())}_{file.filename}"
    dest_path = os.path.join(upload_dir, clean_filename)

    contents = await file.read()
    with open(dest_path, "wb") as f:
        f.write(contents)

    stream_manager.set_source(dest_path, name=file.filename)
    return {
        "success": True,
        "status": "active",
        "video_name": file.filename,
        "video_path": dest_path
    }


@app.get("/api/video/status")
def get_video_status():
    """Returns current active video stream status."""
    return {
        "is_running": stream_manager.is_running,
        "video_name": stream_manager.video_name,
        "is_custom_video": stream_manager.source != "synthetic",
        "source": stream_manager.source
    }


class StreamSwitchRequest(BaseModel):
    source: str  # "synthetic" or video file path


@app.post("/api/stream/switch")
def switch_stream(req: StreamSwitchRequest):
    """Switches stream source (e.g. back to synthetic curve simulation)."""
    stream_manager.set_source(req.source)
    return {"status": "switched", "source": req.source, "video_name": stream_manager.video_name}


@app.get("/api/logs")
def get_logs(limit: int = Query(50), emergency_only: bool = Query(False)):
    """Retrieves recent event records from SQLite with optional emergency filtering."""
    return event_logger.get_recent_events(limit=limit, emergency_only=emergency_only)


@app.get("/api/logs/emergency-summary")
def get_emergency_logs_summary():
    """Retrieves statistical counts of emergency vehicle and preemption incidents."""
    return event_logger.get_emergency_summary()


@app.get("/api/stats")
def get_stats():
    """Retrieves risk distribution statistics."""
    return event_logger.get_risk_summary()


@app.get("/api/snapshots")
def get_snapshots():
    """Lists saved incident snapshots."""
    snaps_dir = event_logger.snapshots_dir
    if not os.path.exists(snaps_dir):
        return []
    files = [f for f in os.listdir(snaps_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files.sort(reverse=True)
    return files[:24]


@app.get("/api/snapshots/{filename}")
def get_snapshot_file(filename: str):
    """Serves incident snapshot image file."""
    filepath = os.path.join(event_logger.snapshots_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Snapshot file not found.")
    return FileResponse(filepath, media_type="image/jpeg")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend_service:app", host="127.0.0.1", port=8000, reload=False)
