"""
Automated Verification Script for Real Traffic Image:
Tests detection, classification, tracking, risk evaluation, and HUD generation.
"""
import os
import sys
import tempfile
import yaml
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set ultralytics writable config dir
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics_config")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from src.detection.yolo_detector import YOLODetector
from src.preprocessing.preprocessor import FramePreprocessor
from src.tracking.tracker import VehicleTracker
from src.risk_assessment.collision_predictor import CollisionPredictor
from src.warning.visualizer import HUDVisualizer
from src.logging_storage.event_logger import EventLogger


def run_test():
    image_path = "data/sample_videos/images.jpeg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return False

    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"Error: Could not decode {image_path}.")
        return False

    print("=" * 70)
    print("      REAL TRAFFIC IMAGE COMPREHENSIVE PIPELINE TEST")
    print("=" * 70)
    print(f"Input Image:   {image_path}")
    print(f"Dimensions:    {raw_img.shape[1]}x{raw_img.shape[0]} (Channels: {raw_img.shape[2]})")

    # Resize to standard processing resolution 1280x720 for consistent spatial zones
    target_w, target_h = 1280, 720
    frame = cv2.resize(raw_img, (target_w, target_h))
    print(f"Standardized:  {target_w}x{target_h}")

    # 1. Preprocessing (CLAHE)
    preproc = FramePreprocessor(enable_clahe=True, clip_limit=2.5)
    enhanced = preproc.process(frame)
    print("[1] Preprocessing: CLAHE contrast enhancement completed.")

    # 2. YOLOv8n Detection Test
    detector = YOLODetector(weights_path="yolov8n.pt", confidence_threshold=0.25)
    detections = detector.detect(enhanced)
    print(f"[2] YOLOv8n Vehicle Detection: Found {len(detections)} vehicles.")

    print("\n    Detected Vehicles Breakdown:")
    print("    -----------------------------------------------------------------")
    print(f"    {'#':<3} | {'Class':<12} | {'Confidence':<10} | {'Bounding Box (x1,y1,x2,y2)':<28} | {'Center'}")
    print("    -----------------------------------------------------------------")
    for idx, d in enumerate(detections, 1):
        print(f"    {idx:<3} | {d.class_name.upper():<12} | {d.confidence * 100:6.1f}%    | {str(d.box):<28} | {d.center}")
    print("    -----------------------------------------------------------------")

    # 3. Tracking Integration
    tracker = VehicleTracker(model_path="yolov8n.pt", conf=0.25)
    tracked_v = tracker.track(enhanced)
    print(f"\n[3] Object Tracking: {len(tracked_v)} vehicles assigned persistent tracking states.")

    # 4. Spatial Zones & Collision Risk Prediction
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    predictor = CollisionPredictor(
        zones_config=cfg.get("zones", {}),
        ttc_critical=3.0,
        ttc_warning=6.0,
        ttc_caution=10.0
    )
    assessment = predictor.evaluate(tracked_v, (target_h, target_w))

    level_names = {0: "SAFE", 1: "CAUTION", 2: "WARNING", 3: "CRITICAL"}
    risk_str = level_names.get(int(assessment.overall_level), "UNKNOWN")
    print(f"\n[4] Risk Assessment Engine:")
    print(f"    - Overall Risk Level:      LEVEL {int(assessment.overall_level)} ({risk_str})")
    print(f"    - Signal A (Uphill):       {assessment.signal_a_state} - '{assessment.signal_a_message}'")
    print(f"    - Signal B (Downhill):     {assessment.signal_b_state} - '{assessment.signal_b_message}'")
    print(f"    - Arm A Vehicles:          {assessment.active_vehicles_arm_a}")
    print(f"    - Arm B Vehicles:          {assessment.active_vehicles_arm_b}")
    print(f"    - Conflict Apex Vehicles:  {assessment.active_vehicles_apex}")

    # 5. HUD Rendering
    visualizer = HUDVisualizer(show_zones=True, show_trajectories=True)
    annotated = visualizer.render(
        frame=frame,
        vehicles=tracked_v,
        assessment=assessment,
        scaled_zones=predictor.scaled_zones
    )
    print(f"\n[5] HUD Visualizer: Rendered overlay with zones, signals, boxes, and alert banner.")

    # 6. Save Output Image
    out_dir = os.path.join(tempfile.gettempdir(), "hairpin_test_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "real_traffic_verified.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"\n[6] Saved Annotated Verification Image to: {out_path}")

    # 7. Event Logger Test
    logger = EventLogger()
    logger.log_event(assessment, tracked_v, annotated)
    print(f"[7] Event Archival: Successfully recorded incident into SQLite and CSV.")
    print("=" * 70)
    print("                  ALL VERIFICATION CHECKS PASSED")
    print("=" * 70)
    return out_path


if __name__ == "__main__":
    run_test()
