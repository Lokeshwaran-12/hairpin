"""
Vehicle Tracking & Trajectory Estimation Module:
Maintains vehicle identity across frames, tracks trajectory history,
and estimates directional velocity and speed (km/h).
"""
import os
import tempfile
import math
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Ensure Ultralytics config is writable across all environments
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics_config")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO
from src.detection.yolo_detector import YOLODetector, Detection


@dataclass
class TrackedVehicle:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: List[int]  # [x1, y1, x2, y2]
    center: Tuple[int, int]
    history: List[Tuple[int, int]] = field(default_factory=list)
    velocity_vector: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in pixels/sec
    speed_kmh: float = 0.0
    heading_deg: float = 0.0
    current_zone: str = "Unknown"
    direction_status: str = "Approaching"  # "Approaching", "Inside Apex", "Exiting"
    frames_tracked: int = 1
    has_crossed_line: bool = False
    is_touching_line: bool = False
    cross_direction: Optional[str] = None
    is_heavy_vehicle: bool = False
    is_left_lane: bool = False
    is_left_passing: bool = False
    is_emergency: bool = False
    emergency_type: Optional[str] = None


def _get_custom_tracker_config() -> str:
    """
    Creates a sensitive ByteTrack configuration in the temp directory
    with lowered new_track_thresh (0.20) and track_high_thresh (0.20)
    so commercial trucks, buses, and low-confidence vehicles are not discarded.
    """
    import yaml
    tracker_path = os.path.join(tempfile.gettempdir(), "hairpin_bytetrack.yaml")
    cfg = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.20,
        "track_low_thresh": 0.08,
        "new_track_thresh": 0.20,
        "track_buffer": 45,
        "match_thresh": 0.8,
        "fuse_score": True
    }
    try:
        with open(tracker_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f)
        return tracker_path
    except Exception:
        return "bytetrack.yaml"


class VehicleTracker:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_type: Optional[str] = None,
        history_len: int = 30,
        pixels_per_meter: float = 18.0,
        fps: int = 30,
        conf: float = 0.18,
        iou: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "cpu",
        ambulance_weights: Optional[str] = "models/yolov8n_ambulance.pt",
        enable_fine_tuned_ambulance: bool = True,
        imgsz: int = 480
    ):
        self.model = YOLO(model_path)
        self.ambulance_weights = ambulance_weights
        self.enable_fine_tuned_ambulance = enable_fine_tuned_ambulance
        self.imgsz = imgsz
        self.frame_idx = 0
        self._last_amb_boxes: List[Tuple[List[int], str, float]] = []
        self.ambulance_model = None
        self.ambulance_model_loaded = False

        if self.enable_fine_tuned_ambulance and self.ambulance_weights and os.path.exists(self.ambulance_weights):
            try:
                self.ambulance_model = YOLO(self.ambulance_weights)
                self.ambulance_model_loaded = True
            except Exception:
                pass
        if tracker_type is None or tracker_type == "bytetrack.yaml":
            self.tracker_type = _get_custom_tracker_config()
        else:
            self.tracker_type = tracker_type
        self.history_len = history_len
        self.pixels_per_meter = max(1.0, pixels_per_meter)
        self.fps = max(1, fps)
        self.conf = conf
        self.iou = iou
        self.target_classes = target_classes or [2, 3, 5, 6, 7]
        self.device = device

        self.class_names = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            6: "truck",  # Semi-trailers / long freight haulers
            7: "truck"
        }

        # Track history buffers: track_id -> deque of (cx, cy)
        self.track_histories: Dict[int, deque] = {}
        # Track speed buffers for smoothing
        self.speed_histories: Dict[int, deque] = {}
        # Box smoothing buffers: track_id -> deque of [x1, y1, x2, y2]
        self.box_histories: Dict[int, deque] = {}
        # Track active counts
        self.track_counts: Dict[int, int] = {}
        # Fallback track IDs for detections before ByteTrack track confirmation
        self._next_fallback_id = 500
        self._fallback_tracks: Dict[int, Tuple[int, int, int, int]] = {}
        # Emergency vehicle track cache: once confirmed ambulance, persist across frames
        self.emergency_tracks: Dict[int, str] = {}  # track_id -> emergency_type
        self.emergency_hits: Dict[int, int] = {}    # track_id -> confirmation count

    def _match_or_create_fallback_id(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Associates unconfirmed detection with active fallback track or creates a new ID."""
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        best_id = None
        best_dist = 75.0

        for fid, (fx1, fy1, fx2, fy2) in list(self._fallback_tracks.items()):
            fcx = (fx1 + fx2) / 2.0
            fcy = (fy1 + fy2) / 2.0
            dist = math.hypot(cx - fcx, cy - fcy)
            if dist < best_dist:
                best_dist = dist
                best_id = fid

        if best_id is None:
            self._next_fallback_id += 1
            best_id = self._next_fallback_id

        self._fallback_tracks[best_id] = (x1, y1, x2, y2)
        return best_id

    def track(self, frame: np.ndarray) -> List[TrackedVehicle]:
        """
        Runs YOLO tracking on frame using ByteTrack.
        Returns list of TrackedVehicle with IDs, trajectories, and estimated speed.
        """
        if frame is None:
            return []

        self.frame_idx += 1

        # 1. Full-Frame Fine-Tuned Ambulance Detection (cadence optimization for smooth real-time video)
        full_frame_amb_boxes: List[Tuple[List[int], str, float]] = []
        if self.ambulance_model_loaded and self.ambulance_model is not None:
            if (self.frame_idx % 2 == 1) or not self._last_amb_boxes:
                try:
                    amb_res = self.ambulance_model.predict(
                        source=frame,
                        imgsz=self.imgsz,
                        conf=0.50,
                        iou=self.iou,
                        device=self.device,
                        verbose=False
                    )
                    if amb_res and len(amb_res) > 0 and amb_res[0].boxes is not None:
                        boxes_out = []
                        for abox in amb_res[0].boxes:
                            acls_id = int(abox.cls[0].item())
                            aconf = float(abox.conf[0].item())
                            axyxy = abox.xyxy[0].cpu().numpy().astype(int).tolist()
                            alabel = "fire_truck" if acls_id == 1 else "ambulance"
                            boxes_out.append((axyxy, alabel, aconf))
                        self._last_amb_boxes = boxes_out
                except Exception:
                    pass
            full_frame_amb_boxes = list(self._last_amb_boxes)

        results = self.model.track(
            source=frame,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker_type,
            conf=self.conf,
            iou=self.iou,
            classes=self.target_classes,
            device=self.device,
            verbose=False
        )

        tracked_objects: List[TrackedVehicle] = []
        if not results or len(results) == 0:
            return tracked_objects

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return tracked_objects

        current_frame_ids = set()

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            # Ensure truck detections are never dropped if ByteTrack hasn't locked track ID yet
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = self._match_or_create_fallback_id(x1, y1, x2, y2)

            current_frame_ids.add(track_id)

            cls_name = self.class_names.get(cls_id, "car")
            w_box = max(1, x2 - x1)
            h_box = max(1, y2 - y1)
            area_box = w_box * h_box
            is_heavy = (cls_name in ["truck", "bus"])

            # Heavy Vehicle Promotion: Large footprint tall commercial vehicles on mountain curves (not flat sedans)
            if not is_heavy and area_box >= 22000 and h_box >= 115 and (h_box / float(w_box)) >= 0.55:
                cls_name = "truck"
                is_heavy = True

            # Smooth bounding box across recent frames to eliminate flicker
            if track_id not in self.box_histories:
                self.box_histories[track_id] = deque(maxlen=4)
            self.box_histories[track_id].append([x1, y1, x2, y2])
            recent_boxes = list(self.box_histories[track_id])
            sx1 = int(sum(b[0] for b in recent_boxes) / len(recent_boxes))
            sy1 = int(sum(b[1] for b in recent_boxes) / len(recent_boxes))
            sx2 = int(sum(b[2] for b in recent_boxes) / len(recent_boxes))
            sy2 = int(sum(b[3] for b in recent_boxes) / len(recent_boxes))

            cx = int((sx1 + sx2) / 2.0)
            cy = int((sy1 + sy2) / 2.0)

            # Initialize history if new track
            if track_id not in self.track_histories:
                self.track_histories[track_id] = deque(maxlen=self.history_len)
                self.speed_histories[track_id] = deque(maxlen=10)
                self.track_counts[track_id] = 0

            self.track_histories[track_id].append((cx, cy))
            self.track_counts[track_id] += 1
            history = list(self.track_histories[track_id])

            # Calculate velocity vector & speed
            vx, vy = 0.0, 0.0
            heading_deg = 0.0
            instant_speed_kmh = 0.0

            if len(history) >= 2:
                window_size = min(5, len(history) - 1)
                p_old = history[-1 - window_size]
                p_cur = history[-1]

                dx = p_cur[0] - p_old[0]
                dy = p_cur[1] - p_old[1]
                time_span = window_size / float(self.fps)

                vx = dx / time_span
                vy = dy / time_span

                px_disp = math.hypot(dx, dy)
                px_per_sec = px_disp / time_span
                meters_per_sec = px_per_sec / self.pixels_per_meter
                raw_kmh = meters_per_sec * 3.6

                heading_rad = math.atan2(dy, dx)
                heading_deg = math.degrees(heading_rad) % 360

                self.speed_histories[track_id].append(raw_kmh)
                instant_speed_kmh = float(np.mean(self.speed_histories[track_id]))

            # Emergency Vehicle Detection: spatial fusion with full-frame neural boxes or cached track
            is_emergency = False
            emergency_type_val: Optional[str] = None

            if track_id in self.emergency_tracks:
                # Already confirmed emergency in previous frames — persist without re-checking
                is_emergency = True
                emergency_type_val = self.emergency_tracks[track_id]
                cls_name = "ambulance"
            elif cls_name in ("car", "truck", "bus"):
                detected_emerg = False
                e_type = None

                # Match with full-frame neural ambulance detections using spatial IoU
                current_box = [sx1, sy1, sx2, sy2]
                for abox, alabel, aconf in full_frame_amb_boxes:
                    iou_val = YOLODetector._compute_iou(current_box, abox)
                    if iou_val >= 0.35:
                        detected_emerg = True
                        e_type = alabel
                        conf_val = max(conf_val, aconf)
                        break

                # Fallback to visual livery / beacon classifier if not detected by neural head
                if not detected_emerg:
                    fh, fw = frame.shape[:2]
                    rx1, ry1 = max(0, sx1), max(0, sy1)
                    rx2, ry2 = min(fw, sx2), min(fh, sy2)
                    if (rx2 - rx1 >= 24) and (ry2 - ry1 >= 24):
                        roi = frame[ry1:ry2, rx1:rx2]
                        detected_emerg, e_type = YOLODetector.classify_emergency_vehicle(roi, cls_name)

                if detected_emerg:
                    self.emergency_hits[track_id] = self.emergency_hits.get(track_id, 0) + 1
                    # Multi-frame confirmation across 2 frames to prevent transient video flicker
                    if self.emergency_hits[track_id] >= 2:
                        is_emergency = True
                        emergency_type_val = e_type
                        cls_name = "ambulance"
                        self.emergency_tracks[track_id] = e_type or "ambulance"

            vehicle = TrackedVehicle(
                track_id=track_id,
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf_val,
                box=[sx1, sy1, sx2, sy2],
                center=(cx, cy),
                history=history,
                velocity_vector=(vx, vy),
                speed_kmh=round(instant_speed_kmh, 1),
                heading_deg=round(heading_deg, 1),
                frames_tracked=self.track_counts[track_id],
                is_heavy_vehicle=is_heavy,
                is_emergency=is_emergency,
                emergency_type=emergency_type_val
            )
            tracked_objects.append(vehicle)

        # Cleanup tracks that disappeared for > 60 frames
        self._fallback_tracks = {
            fid: b for fid, b in self._fallback_tracks.items()
            if fid in current_frame_ids
        }
        # Also prune emergency_tracks for vanished IDs
        self.emergency_tracks = {
            tid: etype for tid, etype in self.emergency_tracks.items()
            if tid in current_frame_ids
        }
        stale_ids = [tid for tid in self.track_histories.keys() if tid not in current_frame_ids and self.track_counts[tid] > 90]
        for tid in stale_ids:
            self.track_histories.pop(tid, None)
            self.speed_histories.pop(tid, None)
            self.box_histories.pop(tid, None)
            self.track_counts.pop(tid, None)

        return tracked_objects
