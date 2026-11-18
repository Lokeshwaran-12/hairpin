"""
YOLOv8n Vehicle Detector:
Detects motorcycles, cars, buses, and trucks for hairpin bend traffic analysis.
"""
import os
import tempfile
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Ensure Ultralytics config is writable across all environments
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics_config")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from ultralytics import YOLO


@dataclass
class Detection:
    box: List[int]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    center: Tuple[int, int]  # (cx, cy)
    width: int
    height: int
    is_emergency: bool = False
    emergency_type: Optional[str] = None


class YOLODetector:
    """
    YOLOv8n vehicle detector filtered for mountain traffic classes.
    Target COCO classes:
      2: car
      3: motorcycle
      5: bus
      6: truck (train/heavy trailer haulers)
      7: truck
    """
    COCO_VEHICLE_CLASSES = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        6: "truck",
        7: "truck"
    }

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.20,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or list(self.COCO_VEHICLE_CLASSES.keys())
        self.device = device

        # Load YOLO model
        self.model = YOLO(self.weights_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs YOLOv8n inference on frame and returns filtered vehicle detections.
        Applies sensitive thresholding to avoid dropping trucks and heavy vehicles.
        """
        if frame is None:
            return []

        # Use lower sensitivity floor for inference so heavy commercial vehicles aren't discarded prematurely
        inference_conf = min(0.18, self.confidence_threshold)

        results = self.model.predict(
            source=frame,
            conf=inference_conf,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            # Heavy vehicles (truck / bus / hauler) are retained down to inference_conf
            # Standard vehicles must meet self.confidence_threshold
            is_heavy_class = cls_id in (5, 6, 7)
            if not is_heavy_class and conf < self.confidence_threshold:
                continue

            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            w = x2 - x1
            h = y2 - y1

            class_name = self.COCO_VEHICLE_CLASSES.get(cls_id, self.model.names.get(cls_id, f"cls_{cls_id}"))

            # Large footprint promotion for heavy vehicles
            if class_name == "car" and (w * h >= 18000) and (w > 130 or h > 110):
                class_name = "truck"

            # Emergency Vehicle (Ambulance) visual heuristic check
            is_emergency = False
            emergency_type = None

            if class_name in ("ambulance", "police", "firetruck"):
                is_emergency = True
                emergency_type = class_name
            elif class_name in ("car", "truck", "bus"):
                fh, fw = frame.shape[:2]
                rx1, ry1 = max(0, x1), max(0, y1)
                rx2, ry2 = min(fw, x2), min(fh, y2)
                if (rx2 - rx1 >= 24) and (ry2 - ry1 >= 24):
                    roi = frame[ry1:ry2, rx1:rx2]
                    is_emerg, e_type = self.classify_emergency_vehicle(roi, class_name)
                    if is_emerg:
                        is_emergency = True
                        emergency_type = e_type
                        class_name = "ambulance"

            detections.append(Detection(
                box=[x1, y1, x2, y2],
                confidence=conf,
                class_id=cls_id,
                class_name=class_name,
                center=(cx, cy),
                width=w,
                height=h,
                is_emergency=is_emergency,
                emergency_type=emergency_type
            ))

        return detections

    @staticmethod
    def classify_emergency_vehicle(roi: np.ndarray, base_class: str = "car") -> Tuple[bool, Optional[str]]:
        """
        Visual heuristic classifier for ambulances and emergency response vehicles.
        Analyzes:
          1. Roof Lightbar/Beacon: Top 15-35% of the vehicle ROI for red/blue strobe signatures.
          2. Livery & Body Pattern: High white reflectance body with high-contrast emergency red markings.
        """
        if roi is None or roi.size == 0:
            return False, None

        h, w = roi.shape[:2]
        if h < 20 or w < 20:
            return False, None

        # 1. Roof Beacon / Lightbar Inspection (Top 15% - 35% of vehicle)
        roof_h = max(4, int(h * 0.28))
        roof_roi = roi[0:roof_h, :]
        hsv_roof = cv2.cvtColor(roof_roi, cv2.COLOR_BGR2HSV)
        roof_pixels = float(roof_roi.shape[0] * roof_roi.shape[1])

        # Red beacons (hue wraps 0-10 and 165-180)
        red_mask_1 = (hsv_roof[:, :, 0] <= 10) & (hsv_roof[:, :, 1] >= 80) & (hsv_roof[:, :, 2] >= 90)
        red_mask_2 = (hsv_roof[:, :, 0] >= 165) & (hsv_roof[:, :, 1] >= 80) & (hsv_roof[:, :, 2] >= 90)
        red_beacon_px = int(np.count_nonzero(red_mask_1 | red_mask_2))

        # Blue beacons (hue 100-135)
        blue_mask = (hsv_roof[:, :, 0] >= 100) & (hsv_roof[:, :, 0] <= 135) & (hsv_roof[:, :, 1] >= 80) & (hsv_roof[:, :, 2] >= 90)
        blue_beacon_px = int(np.count_nonzero(blue_mask))

        flasher_ratio = (red_beacon_px + blue_beacon_px) / roof_pixels

        # 2. Body Livery & Marking Inspection (Middle/Lower 72% of vehicle)
        body_roi = roi[roof_h:, :]
        body_pixels = float(body_roi.shape[0] * body_roi.shape[1])
        hsv_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)

        # White / High-reflectance body base (low saturation, high value)
        white_body_mask = (hsv_body[:, :, 1] <= 65) & (hsv_body[:, :, 2] >= 140)
        white_ratio = int(np.count_nonzero(white_body_mask)) / body_pixels

        # Emergency red stripe / cross markings on body
        red_body_mask = ((hsv_body[:, :, 0] <= 10) | (hsv_body[:, :, 0] >= 165)) & (hsv_body[:, :, 1] >= 90) & (hsv_body[:, :, 2] >= 80)
        red_body_ratio = int(np.count_nonzero(red_body_mask)) / body_pixels

        # Decision Logic:
        # A. Roof lightbar with red + blue flashers on predominantly white body
        has_dual_beacon = (red_beacon_px >= 3 and blue_beacon_px >= 3 and flasher_ratio >= 0.02)
        # B. Single intense emergency flasher on roof + white body
        has_strong_flasher = (flasher_ratio >= 0.04 and (red_beacon_px >= 6 or blue_beacon_px >= 6))
        # C. Signature ambulance livery: high white body with emergency red markings
        has_ambulance_livery = (white_ratio >= 0.30 and red_body_ratio >= 0.02)

        if (has_dual_beacon and white_ratio >= 0.15) or \
           (has_strong_flasher and white_ratio >= 0.18) or \
           (has_ambulance_livery and (flasher_ratio >= 0.015 or red_beacon_px >= 3 or blue_beacon_px >= 3)):
            return True, "ambulance"

        return False, None

    @staticmethod
    def draw_bounding_boxes(
        frame: np.ndarray,
        detections: List[Detection],
        show_conf: bool = True,
        show_centroids: bool = True,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Renders clear, high-contrast bounding boxes with class labels and confidence percentages.
        """
        annotated = frame.copy()
        colors = {
            "car": (0, 220, 0),          # Bright Green
            "truck": (0, 140, 255),      # Orange
            "bus": (255, 200, 0),        # Cyan / Blue
            "motorcycle": (255, 0, 220), # Magenta
            "ambulance": (0, 0, 255)     # Emergency Red
        }

        for d in detections:
            x1, y1, x2, y2 = d.box
            color = colors.get(d.class_name, (0, 255, 255))
            is_emergency = d.is_emergency or d.class_name == "ambulance"

            # Bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # High-visibility double border for emergency vehicles
            if is_emergency:
                cv2.rectangle(annotated, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)

            # Label text
            if is_emergency:
                label = f"[AMBULANCE] {d.confidence * 100:.0f}%" if show_conf else "[AMBULANCE]"
                badge_bg = (0, 0, 255)
                text_color = (255, 255, 255)
            else:
                label = f"{d.class_name.upper()} {d.confidence * 100:.0f}%" if show_conf else d.class_name.upper()
                badge_bg = color
                text_color = (0, 0, 0)

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            badge_y1 = max(0, y1 - th - 8)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 6, badge_y1 + th + 6), badge_bg, -1)
            cv2.putText(annotated, label, (x1 + 3, badge_y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)

            # Centroid point
            if show_centroids:
                cv2.circle(annotated, d.center, 4, (0, 0, 255), -1)

        return annotated
