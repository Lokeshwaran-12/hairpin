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
        ambulance_weights: Optional[str] = "models/yolov8n_ambulance.pt",
        enable_fine_tuned_ambulance: bool = True,
        confidence_threshold: float = 0.20,
        ambulance_conf: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "cpu"
    ):
        self.weights_path = weights_path
        self.ambulance_weights = ambulance_weights
        self.enable_fine_tuned_ambulance = enable_fine_tuned_ambulance
        self.confidence_threshold = confidence_threshold
        self.ambulance_conf = ambulance_conf
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or list(self.COCO_VEHICLE_CLASSES.keys())
        self.device = device

        # Load Base YOLO model (COCO road traffic)
        self.model = YOLO(self.weights_path)

        # Load Fine-Tuned Ambulance Model if available
        self.ambulance_model = None
        self.ambulance_model_loaded = False
        if self.enable_fine_tuned_ambulance and self.ambulance_weights:
            if os.path.exists(self.ambulance_weights):
                try:
                    self.ambulance_model = YOLO(self.ambulance_weights)
                    self.ambulance_model_loaded = True
                except Exception as e:
                    print(f"[YOLODetector] Warning: Could not load {self.ambulance_weights}: {e}")

    @staticmethod
    def _compute_iou(box1: List[int], box2: List[int]) -> float:
        """Computes Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = float(area1 + area2 - inter_area)
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def _compute_overlap_ratio(box1: List[int], box2: List[int]) -> float:
        """Computes intersection area over min area between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        min_area = min(area1, area2)
        return float(inter_area) / float(min_area) if min_area > 0 else 0.0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Runs YOLOv8n inference on frame and returns filtered vehicle detections.
        Applies sensitive thresholding to avoid dropping trucks and heavy vehicles.
        Integrates fine-tuned neural ambulance detection with spatial IoU fusion.
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

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()

                # Heavy vehicles (truck / bus / hauler) are retained down to inference_conf
                is_heavy_class = cls_id in (5, 6, 7)
                if not is_heavy_class and conf < self.confidence_threshold:
                    continue

                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                w = x2 - x1
                h = y2 - y1

                class_name = self.COCO_VEHICLE_CLASSES.get(cls_id, self.model.names.get(cls_id, f"cls_{cls_id}"))

                # Large footprint promotion for heavy vehicles (tall trucks/buses, not flat wide cars)
                if class_name == "car" and (w * h >= 22000) and (h >= 115) and ((h / float(w)) >= 0.55):
                    class_name = "truck"

                # Direct check if class itself is already emergency
                is_emergency = class_name in ("ambulance", "police", "firetruck")
                emergency_type = class_name if is_emergency else None

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

        # ------------------------------------------------------------------ #
        # Step 2: Fine-Tuned Neural Ambulance Detector & Spatial Fusion      #
        # ------------------------------------------------------------------ #
        if self.ambulance_model_loaded and self.ambulance_model is not None:
            amb_results = self.ambulance_model.predict(
                source=frame,
                imgsz=640,
                conf=self.ambulance_conf,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )

            if amb_results and len(amb_results) > 0 and amb_results[0].boxes is not None:
                amb_boxes = []
                fh, fw = frame.shape[:2]
                for abox in amb_results[0].boxes:
                    acls_id = int(abox.cls[0].item())
                    aconf = float(abox.conf[0].item())
                    axyxy = abox.xyxy[0].cpu().numpy().astype(int).tolist()
                    ax1, ay1, ax2, ay2 = axyxy

                    # False-positive filter: verify genuine emergency features (beacon strobes / emergency livery)
                    roi_candidate = frame[max(0, ay1):min(fh, ay2), max(0, ax1):min(fw, ax2)]
                    is_valid_emerg, _ = self.classify_emergency_vehicle(roi_candidate, "car")
                    if not is_valid_emerg:
                        continue  # Rejects ordinary civilian cars, SUVs, and white vans

                    amb_boxes.append((axyxy, acls_id, aconf))

                # Process candidates in descending order of confidence
                amb_boxes.sort(key=lambda x: x[2], reverse=True)

                for axyxy, acls_id, aconf in amb_boxes:
                    ax1, ay1, ax2, ay2 = axyxy
                    acx = int((ax1 + ax2) / 2.0)
                    acy = int((ay1 + ay2) / 2.0)
                    aw = ax2 - ax1
                    ah = ay2 - ay1
                    e_label = "fire_truck" if acls_id == 1 else "ambulance"

                    # 1. Duplicate Suppression: Skip if another confirmed ambulance already covers this region
                    if any(d.is_emergency and (self._compute_iou(d.box, axyxy) >= 0.15 or self._compute_overlap_ratio(d.box, axyxy) >= 0.25) for d in detections):
                        continue

                    # 2. Bus Protection: NEVER override or convert a bus to an ambulance
                    # Match only with car or truck chassis (light commercial / van chassis)
                    matched_base = False
                    for det in detections:
                        if det.class_name == "bus":
                            continue  # Buses are preserved and never overridden
                        iou = self._compute_iou(det.box, [ax1, ay1, ax2, ay2])
                        if iou >= 0.35:
                            det.is_emergency = True
                            det.emergency_type = e_label
                            det.class_name = "ambulance"
                            det.confidence = max(det.confidence, aconf)
                            matched_base = True
                            break

                    # 3. Add standalone ambulance only if it does not overlap a real bus
                    if not matched_base and aconf >= self.confidence_threshold:
                        overlaps_real_bus = any(
                            d.class_name == "bus" and d.confidence >= 0.50 and (self._compute_iou(d.box, axyxy) >= 0.15 or self._compute_overlap_ratio(d.box, axyxy) >= 0.20)
                            for d in detections
                        )
                        if not overlaps_real_bus:
                            detections.append(Detection(
                                box=[ax1, ay1, ax2, ay2],
                                confidence=aconf,
                                class_id=acls_id,
                                class_name="ambulance",
                                center=(acx, acy),
                                width=aw,
                                height=ah,
                                is_emergency=True,
                                emergency_type=e_label
                            ))

        # ------------------------------------------------------------------ #
        # Step 3: Heuristic Fallback Check (ONLY if neural ambulance weights #
        # are not loaded, to avoid false positives on brake lights)          #
        # ------------------------------------------------------------------ #
        if not self.ambulance_model_loaded:
            for det in detections:
                if not det.is_emergency and det.class_name in ("car", "truck", "bus"):
                    x1, y1, x2, y2 = det.box
                    fh, fw = frame.shape[:2]
                    rx1, ry1 = max(0, x1), max(0, y1)
                    rx2, ry2 = min(fw, x2), min(fh, y2)
                    if (rx2 - rx1 >= 24) and (ry2 - ry1 >= 24):
                        roi = frame[ry1:ry2, rx1:rx2]
                        is_emerg, e_type = self.classify_emergency_vehicle(roi, det.class_name)
                        if is_emerg:
                            det.is_emergency = True
                            det.emergency_type = e_type
                            det.class_name = "ambulance"

        return detections

    @staticmethod
    def classify_emergency_vehicle(roi: np.ndarray, base_class: str = "car") -> Tuple[bool, Optional[str]]:
        """
        Visual emergency classifier for ambulances and emergency response vehicles.
        Rejects ordinary civilian passenger cars (sedans, SUVs, hatchbacks) and requires:
          1. Genuine Red Beacon / Strobe on roof (top 22%): red_beacon_px >= 8
          2. Dual Red & Blue Strobes: red_beacon_px >= 4 and blue_beacon_px >= 10
          3. Prominent Red Emergency Markings / Cross on white body: red_body_ratio >= 0.020 and white_ratio >= 0.30
        """
        if roi is None or roi.size == 0:
            return False, None

        h, w = roi.shape[:2]
        if h < 24 or w < 24:
            return False, None

        # 1. Roof Beacon / Lightbar Inspection (Top 22% of vehicle)
        roof_h = max(5, int(h * 0.22))
        roof_roi = roi[0:roof_h, :]
        hsv_roof = cv2.cvtColor(roof_roi, cv2.COLOR_BGR2HSV)

        # Highly saturated red strobes (S >= 110, V >= 110)
        red_mask_1 = (hsv_roof[:, :, 0] <= 10) & (hsv_roof[:, :, 1] >= 110) & (hsv_roof[:, :, 2] >= 110)
        red_mask_2 = (hsv_roof[:, :, 0] >= 168) & (hsv_roof[:, :, 1] >= 110) & (hsv_roof[:, :, 2] >= 110)
        red_beacon_px = int(np.count_nonzero(red_mask_1 | red_mask_2))

        # Highly saturated blue strobes (hue 102-128, S >= 110, V >= 110)
        blue_mask = (hsv_roof[:, :, 0] >= 102) & (hsv_roof[:, :, 0] <= 128) & (hsv_roof[:, :, 1] >= 110) & (hsv_roof[:, :, 2] >= 110)
        blue_beacon_px = int(np.count_nonzero(blue_mask))

        # 2. Body Livery & Marking Inspection (Middle vehicle body)
        ground_cut = max(roof_h + 1, int(h * 0.88))
        body_roi = roi[roof_h:ground_cut, :]
        body_pixels = float(body_roi.shape[0] * body_roi.shape[1])
        hsv_body = cv2.cvtColor(body_roi, cv2.COLOR_BGR2HSV)

        # White / High-reflectance body base (low saturation S <= 50, high value V >= 160)
        white_body_mask = (hsv_body[:, :, 1] <= 50) & (hsv_body[:, :, 2] >= 160)
        white_ratio = int(np.count_nonzero(white_body_mask)) / body_pixels

        # Emergency red stripe / cross markings on body (S >= 110, V >= 90)
        red_body_mask = ((hsv_body[:, :, 0] <= 10) | (hsv_body[:, :, 0] >= 168)) & (hsv_body[:, :, 1] >= 110) & (hsv_body[:, :, 2] >= 90)
        red_body_ratio = int(np.count_nonzero(red_body_mask)) / body_pixels

        # Genuine emergency criteria:
        # 1. Dedicated red beacon (>= 10px on roof)
        # 2. Dual red + blue lightbar (red >= 10 and blue >= 12)
        # 3. Emergency livery with roof beacon presence (white >= 0.24, red livery >= 0.015, red beacon >= 10)
        has_dedicated_red_beacon = (red_beacon_px >= 10)
        has_dual_beacon = (red_beacon_px >= 10 and blue_beacon_px >= 12)
        has_livery_with_beacon = (white_ratio >= 0.24 and red_body_ratio >= 0.015 and red_beacon_px >= 10)

        if has_dedicated_red_beacon or has_dual_beacon or has_livery_with_beacon:
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
