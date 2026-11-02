"""
Main CLI Pipeline Runner for Hairpin Bend Traffic Monitoring & Accident Prevention.
"""
import os
import sys
import time
import tempfile
import argparse
import yaml
import cv2

# Set ultralytics writable config dir
os.environ["YOLO_CONFIG_DIR"] = os.path.join(tempfile.gettempdir(), "ultralytics_config")
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

from src.capture.video_stream import VideoStreamSource
from src.preprocessing.preprocessor import FramePreprocessor
from src.tracking.tracker import VehicleTracker
from src.risk_assessment.collision_predictor import CollisionPredictor
from src.warning.alert_system import AlertSystem
from src.warning.visualizer import HUDVisualizer
from src.logging_storage.event_logger import EventLogger


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="YOLO-Based Hairpin Bend Traffic Monitoring System")
    parser.add_argument("--source", type=str, default="synthetic",
                        help="Video source: 'synthetic', webcam index (0), or path to video file")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLOv8 weights file")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--no-zones", action="store_true", help="Hide spatial zones overlay")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio buzzer alarms")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--save-output", type=str, default="", help="Path to save annotated output image/frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Exit after processing N frames (0 for infinite)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    # Resolve settings
    video_cfg = config.get("video", {})
    width = video_cfg.get("width", 1280)
    height = video_cfg.get("height", 720)
    fps = video_cfg.get("fps", 30)

    source = args.source if args.source != "synthetic" else video_cfg.get("source", "synthetic")
    conf_thresh = args.conf or config.get("model", {}).get("confidence", 0.35)
    weights_path = args.weights or config.get("model", {}).get("weights", "yolov8n.pt")

    prep_cfg = config.get("preprocessing", {})
    tracking_cfg = config.get("tracking", {})
    risk_cfg = config.get("risk_assessment", {})
    alert_cfg = config.get("alert_system", {})
    storage_cfg = config.get("storage", {})

    print("==================================================================")
    print("  YOLO Hairpin Bend Traffic Monitoring & Accident Prevention")
    print("==================================================================")
    print(f" Source:            {source}")
    print(f" Model Weights:     {weights_path}")
    print(f" Confidence Thresh: {conf_thresh}")
    print(f" Audio Alerts:      {'Disabled' if args.no_audio else 'Enabled'}")
    print(f" Display Mode:      {'Headless' if args.headless else 'OpenCV Window'}")
    print("==================================================================")

    # Initialize pipeline modules
    stream = VideoStreamSource(source=source, width=width, height=height, fps=fps)
    preprocessor = FramePreprocessor(
        enable_clahe=prep_cfg.get("enable_clahe", True),
        clip_limit=prep_cfg.get("clahe_clip_limit", 2.5)
    )
    tracker = VehicleTracker(
        model_path=weights_path,
        tracker_type=tracking_cfg.get("tracker_type", "bytetrack.yaml"),
        history_len=tracking_cfg.get("track_history_length", 30),
        pixels_per_meter=tracking_cfg.get("pixels_per_meter", 18.0),
        fps=fps,
        conf=conf_thresh,
        iou=config.get("model", {}).get("iou_threshold", 0.45),
        target_classes=config.get("model", {}).get("target_classes", [2, 3, 5, 7]),
        device=config.get("model", {}).get("device", "cpu")
    )
    predictor = CollisionPredictor(
        zones_config=config.get("zones", {}),
        ttc_critical=risk_cfg.get("ttc_critical_threshold", 3.0),
        ttc_warning=risk_cfg.get("ttc_warning_threshold", 6.0),
        ttc_caution=risk_cfg.get("ttc_caution_threshold", 10.0),
        pixels_per_meter=tracking_cfg.get("pixels_per_meter", 18.0),
        hazard_weights=risk_cfg.get("hazard_weights", None)
    )
    alert_system = AlertSystem(
        enable_audio=(not args.no_audio) and alert_cfg.get("audio_alert", True),
        buzzer_freq_hz=alert_cfg.get("buzzer_frequency_hz", 1200),
        buzzer_duration_ms=alert_cfg.get("buzzer_duration_ms", 180),
        cooldown_seconds=alert_cfg.get("audio_cooldown_seconds", 1.5)
    )
    visualizer = HUDVisualizer(
        show_zones=not args.no_zones,
        show_trajectories=True
    )
    logger = EventLogger(
        db_path=storage_cfg.get("database_path", "data/logs/hairpin_events.db"),
        csv_path=storage_cfg.get("csv_log_path", "data/logs/traffic_events.csv"),
        snapshots_dir=storage_cfg.get("snapshots_dir", "data/snapshots"),
        save_snapshots=alert_cfg.get("save_snapshots", True),
        snapshot_min_risk=alert_cfg.get("snapshot_min_risk_level", 2)
    )

    frame_count = 0
    t_start = time.time()

    try:
        while stream.is_opened:
            ret, frame = stream.read()
            if not ret or frame is None:
                print("\n[VideoStream] Stream ended or unavailable.")
                break

            frame_count += 1

            # 1. Image preprocessing
            processed_frame = preprocessor.process(frame)

            # 2. Object detection and multi-object tracking
            tracked_vehicles = tracker.track(processed_frame)

            # 3. Collision risk assessment
            assessment = predictor.evaluate(tracked_vehicles, frame.shape)

            # 4. Trigger alert system
            alert_system.trigger(assessment)

            # 5. Render HUD visualizer
            annotated_frame = visualizer.render(
                frame=frame,
                vehicles=tracked_vehicles,
                assessment=assessment,
                scaled_zones=predictor.scaled_zones
            )

            # 6. Event logging & incident snapshot
            # Log periodically or when risk >= CAUTION
            if assessment.overall_level >= 1 or frame_count % 30 == 0:
                logger.log_event(assessment, tracked_vehicles, annotated_frame)

            # Optional save annotated output
            if args.save_output and (frame_count == 1 or stream.is_image):
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(args.save_output)), exist_ok=True)
                except Exception:
                    pass
                cv2.imwrite(args.save_output, annotated_frame)
                print(f"[Output] Saved annotated image to: {args.save_output}")

            # Display frame
            if not args.headless:
                cv2.imshow("Hairpin Bend Traffic Monitoring - YOLOv8", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\nUser requested exit.")
                    break

            if args.max_frames and frame_count >= args.max_frames:
                print(f"\nReached max frames limit ({args.max_frames}).")
                break

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
    finally:
        stream.release()
        if not args.headless:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        print(f"\nCompleted {frame_count} frames in {elapsed:.2f}s (Avg {fps_actual:.1f} FPS)")
        print(f"Events logged to: {logger.db_path} and {logger.csv_path}")


if __name__ == "__main__":
    main()
