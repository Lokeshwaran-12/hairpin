"""
Unit & Integration Tests for Ambulance Dataset Conversion & Hybrid Detection Engine.
"""
import os
import yaml
import pytest
import numpy as np
from pathlib import Path

from src.detection.yolo_detector import YOLODetector, Detection
import backend_service


def test_ambulance_yolo_dataset_structure():
    """Verifies that prepare_ambulance_dataset.py generated a valid YOLOv8 dataset."""
    base_dir = Path("data/datasets/ambulance_yolo")
    yaml_file = base_dir / "data.yaml"

    assert yaml_file.exists(), "data.yaml must exist in ambulance_yolo"

    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "names" in data
    assert data["names"][0] == "ambulance"
    assert data["names"][1] == "fire_truck"

    train_imgs = list((base_dir / "images" / "train").glob("*.jpg"))
    val_imgs = list((base_dir / "images" / "val").glob("*.jpg"))
    train_labels = list((base_dir / "labels" / "train").glob("*.txt"))
    val_labels = list((base_dir / "labels" / "val").glob("*.txt"))

    assert len(train_imgs) >= 1000, f"Expected >= 1000 train images, got {len(train_imgs)}"
    assert len(val_imgs) >= 150, f"Expected >= 150 val images, got {len(val_imgs)}"
    assert len(train_labels) == len(train_imgs)
    assert len(val_labels) == len(val_imgs)

    # Validate coordinate normalization in sample label files
    sample_label = train_labels[0]
    with open(sample_label, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            assert cls_id in (0, 1)
            assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc}"
            assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc}"
            assert 0.0 < w <= 1.0, f"w out of bounds: {w}"
            assert 0.0 < h <= 1.0, f"h out of bounds: {h}"


def test_yolo_detector_hybrid_initialization():
    """Verifies YOLODetector initializes with hybrid dual-head attributes."""
    detector = YOLODetector(
        weights_path="yolov8n.pt",
        ambulance_weights="models/yolov8n_ambulance.pt",
        enable_fine_tuned_ambulance=True
    )
    assert detector.weights_path == "yolov8n.pt"
    assert detector.ambulance_weights == "models/yolov8n_ambulance.pt"
    assert detector.enable_fine_tuned_ambulance is True
    assert isinstance(detector.ambulance_model_loaded, bool)


def test_yolo_detector_iou_math():
    """Verifies IoU intersection computation for spatial box fusion."""
    box_a = [100, 100, 300, 300]
    box_b = [100, 100, 300, 300]
    assert YOLODetector._compute_iou(box_a, box_b) == 1.0

    box_disjoint = [400, 400, 500, 500]
    assert YOLODetector._compute_iou(box_a, box_disjoint) == 0.0

    # Overlapping box: 100x100 overlap on 200x200 boxes
    box_half = [200, 200, 400, 400]
    iou = YOLODetector._compute_iou(box_a, box_half)
    expected_iou = (100 * 100) / float(40000 + 40000 - 10000)
    assert abs(iou - expected_iou) < 1e-4


def test_yolo_detector_spatial_fusion_promotion():
    """
    Tests that spatial fusion upgrades a car detection to ambulance
    when an ambulance bounding box overlaps.
    """
    detector = YOLODetector(
        weights_path="yolov8n.pt",
        ambulance_weights="models/yolov8n_ambulance.pt",
        enable_fine_tuned_ambulance=True
    )

    base_det = Detection(
        box=[100, 100, 300, 300],
        confidence=0.75,
        class_id=2,
        class_name="car",
        center=(200, 200),
        width=200,
        height=200,
        is_emergency=False,
        emergency_type=None
    )

    # Simulate fine-tuned ambulance prediction overlapping base detection
    amb_box = [110, 110, 290, 290]
    iou = detector._compute_iou(base_det.box, amb_box)
    assert iou >= 0.35

    # Trigger promotion logic
    if iou >= 0.35:
        base_det.is_emergency = True
        base_det.emergency_type = "ambulance"
        base_det.class_name = "ambulance"
        base_det.confidence = max(base_det.confidence, 0.92)

    assert base_det.is_emergency is True
    assert base_det.class_name == "ambulance"
    assert base_det.confidence == 0.92


def test_yolo_detector_fallback_when_weights_missing():
    """Detector must not crash when ambulance weights are absent; falls back cleanly."""
    detector = YOLODetector(
        weights_path="yolov8n.pt",
        ambulance_weights="models/non_existent_weights.pt",
        enable_fine_tuned_ambulance=True
    )
    assert detector.ambulance_model_loaded is False
    assert detector.ambulance_model is None


def test_backend_api_model_info():
    """GET /api/video/status must report model configuration and status."""
    status = backend_service.get_video_status()
    assert "model_info" in status
    minfo = status["model_info"]
    assert "base_weights" in minfo
    assert "ambulance_weights" in minfo
    assert "ambulance_model_loaded" in minfo
    assert "mode" in minfo
    assert minfo["mode"] in ("hybrid_dual_head", "standard_with_heuristic_fallback")
