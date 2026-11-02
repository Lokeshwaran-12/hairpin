"""
Tests for YOLOv8n vehicle detector.
"""
import os
import numpy as np
import pytest
from src.detection.yolo_detector import YOLODetector, Detection
from src.utils.synthetic_feed import HairpinSimulationFeed


def test_detector_initialization():
    detector = YOLODetector(weights_path="yolov8n.pt", confidence_threshold=0.35)
    assert detector.model is not None
    assert 2 in detector.target_classes  # car
    assert 3 in detector.target_classes  # motorcycle
    assert 5 in detector.target_classes  # bus
    assert 7 in detector.target_classes  # truck


def test_detector_inference_on_synthetic_frame():
    detector = YOLODetector(weights_path="yolov8n.pt", confidence_threshold=0.25)
    feed = HairpinSimulationFeed(width=1280, height=720)
    ret, frame = feed.read()
    assert ret is True and frame is not None
    detections = detector.detect(frame)
    assert isinstance(detections, list)


def test_detector_inference_on_real_traffic_image():
    detector = YOLODetector(weights_path="yolov8n.pt", confidence_threshold=0.20)
    img_path = "data/sample_videos/images.jpeg"
    if not os.path.exists(img_path):
        pytest.skip(f"{img_path} not found")

    import cv2
    img = cv2.imread(img_path)
    assert img is not None

    detections = detector.detect(img)
    assert len(detections) >= 5, f"Expected at least 5 vehicles, found {len(detections)}"

    # Check vehicle classes - both cars and trucks are reliably captured
    classes = [d.class_name for d in detections]
    assert "car" in classes
    assert "truck" in classes
    for d in detections:
        assert d.confidence >= 0.18
        assert d.width > 0 and d.height > 0
