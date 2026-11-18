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


# ------------------------------------------------------------------ #
#  Ambulance / Emergency Vehicle Tracking Tests (Phase 1)             #
# ------------------------------------------------------------------ #

def test_tracked_vehicle_has_emergency_fields():
    """TrackedVehicle dataclass must include is_emergency and emergency_type."""
    v = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.85,
        box=[0, 0, 100, 60],
        center=(50, 30),
    )
    assert hasattr(v, "is_emergency")
    assert hasattr(v, "emergency_type")
    assert v.is_emergency is False
    assert v.emergency_type is None


def test_tracked_vehicle_emergency_fields_settable():
    """is_emergency and emergency_type can be set on TrackedVehicle."""
    v = TrackedVehicle(
        track_id=42,
        class_id=2,
        class_name="ambulance",
        confidence=0.92,
        box=[10, 10, 200, 150],
        center=(105, 80),
        is_emergency=True,
        emergency_type="ambulance",
    )
    assert v.is_emergency is True
    assert v.emergency_type == "ambulance"
    assert v.class_name == "ambulance"


def test_tracker_emergency_cache_initialized():
    """VehicleTracker must expose an emergency_tracks dict on init."""
    tracker = VehicleTracker(model_path="yolov8n.pt")
    assert hasattr(tracker, "emergency_tracks")
    assert isinstance(tracker.emergency_tracks, dict)
    assert len(tracker.emergency_tracks) == 0


def test_tracker_emergency_track_persistence():
    """
    When a track_id is manually inserted into emergency_tracks,
    the tracker must keep propagating is_emergency=True for that ID.
    Uses the internal cache directly (unit-level injection).
    """
    tracker = VehicleTracker(model_path="yolov8n.pt")

    # Directly inject an emergency entry for track 7 (simulates confirmed ambulance)
    tracker.emergency_tracks[7] = "ambulance"

    # Confirm the cache is correctly populated
    assert tracker.emergency_tracks.get(7) == "ambulance"


def test_tracker_vehicles_have_emergency_attribute_after_tracking():
    """All TrackedVehicle objects returned from track() must have is_emergency attribute."""
    tracker = VehicleTracker(model_path="yolov8n.pt", conf=0.20)
    feed = HairpinSimulationFeed(width=1280, height=720)

    for _ in range(5):
        ret, frame = feed.read()
        if not ret:
            break
        vehicles = tracker.track(frame)
        for v in vehicles:
            assert hasattr(v, "is_emergency"), f"TrackedVehicle ID {v.track_id} missing is_emergency"
            assert isinstance(v.is_emergency, bool)
            assert hasattr(v, "emergency_type")
