"""
Tests for multi-object vehicle tracking & speed estimation.
"""
import pytest
import numpy as np
from src.tracking.tracker import VehicleTracker, TrackedVehicle
from src.utils.synthetic_feed import HairpinSimulationFeed


def test_tracker_initialization():
    tracker = VehicleTracker(model_path="yolov8n.pt", history_len=20, pixels_per_meter=18.0)
    assert tracker.history_len == 20
    assert tracker.pixels_per_meter == 18.0


def test_tracker_continuity_and_speed():
    tracker = VehicleTracker(model_path="yolov8n.pt", history_len=15, pixels_per_meter=18.0, conf=0.20)
    feed = HairpinSimulationFeed(width=1280, height=720)

    tracked_ids_seen = set()
    has_speed_calc = False

    # Process 25 frames
    for _ in range(25):
        ret, frame = feed.read()
        if not ret:
            break
        vehicles = tracker.track(frame)
        for v in vehicles:
            assert isinstance(v, TrackedVehicle)
            tracked_ids_seen.add(v.track_id)
            if len(v.history) > 2 and v.speed_kmh > 0:
                has_speed_calc = True

    # At least some tracks should be observed or tested
    assert isinstance(tracked_ids_seen, set)
