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

            detections.append(Detection(
                box=[x1, y1, x2, y2],
                confidence=conf,
                class_id=cls_id,
                class_name=class_name,
                center=(cx, cy),
                width=w,
                height=h
            ))

        return detections

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
            "motorcycle": (255, 0, 220)  # Magenta
        }

        for d in detections:
            x1, y1, x2, y2 = d.box
            color = colors.get(d.class_name, (0, 255, 255))

            # Bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # Label text
            if show_conf:
                label = f"{d.class_name.upper()} {d.confidence * 100:.0f}%"
            else:
                label = d.class_name.upper()

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            badge_y1 = max(0, y1 - th - 8)
            cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + 6, badge_y1 + th + 6), color, -1)
            cv2.putText(annotated, label, (x1 + 3, badge_y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

            # Centroid point
            if show_centroids:
                cv2.circle(annotated, d.center, 4, (0, 0, 255), -1)

        return annotated
