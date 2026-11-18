"""
Tests for Event Logger and SQLite / CSV persistence.
"""
import os
import tempfile
import pytest
from src.tracking.tracker import TrackedVehicle
from src.risk_assessment.collision_predictor import RiskAssessment, RiskLevel
from src.logging_storage.event_logger import EventLogger


@pytest.fixture
def temp_logger():
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_events.db")
    csv_path = os.path.join(tmp_dir, "test_events.csv")
    snaps_dir = os.path.join(tmp_dir, "test_snaps")

    logger = EventLogger(
        db_path=db_path,
        csv_path=csv_path,
        snapshots_dir=snaps_dir,
        save_snapshots=False
    )
    yield logger


def test_logger_initialization(temp_logger):
    assert os.path.exists(temp_logger.csv_path)
    events = temp_logger.get_recent_events()
    assert len(events) == 0


def test_logger_log_and_retrieve(temp_logger):
    v = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.88,
        box=[100, 100, 200, 200],
        center=(150, 150),
        speed_kmh=42.0
    )

    assessment = RiskAssessment(
        overall_level=RiskLevel.WARNING,
        conflict_pairs=[],
        signal_a_state="YELLOW",
        signal_a_message="SLOW DOWN",
        signal_b_state="RED",
        signal_b_message="STOP - OPPOSING CAR",
        active_vehicles_arm_a=1,
        active_vehicles_arm_b=1,
        active_vehicles_apex=0,
        min_ttc=4.5,
        hazard_notes=["Opposing convergence hazard"]
    )

    temp_logger.log_event(assessment, [v])

    recent = temp_logger.get_recent_events(limit=10)
    assert len(recent) == 1
    record = recent[0]
    assert record["risk_level"] == int(RiskLevel.WARNING)
    assert record["risk_label"] == "WARNING"
    assert record["signal_a_state"] == "YELLOW"
    assert record["signal_b_state"] == "RED"
    assert record["min_ttc"] == 4.5

    summary = temp_logger.get_risk_summary()
    assert summary["WARNING"] == 1


def test_logger_swapped_arguments_defensive(temp_logger):
    """Verifies that accidentally swapped (frame, vehicles, assessment) arguments are safely handled."""
    import numpy as np
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    v = TrackedVehicle(
        track_id=2,
        class_id=7,
        class_name="truck",
        confidence=0.75,
        box=[50, 50, 150, 150],
        center=(100, 100),
        speed_kmh=35.0
    )

    assessment = RiskAssessment(
        overall_level=RiskLevel.CRITICAL,
        conflict_pairs=[],
        signal_a_state="RED",
        signal_a_message="STOP",
        signal_b_state="RED",
        signal_b_message="STOP",
        active_vehicles_arm_a=1,
        active_vehicles_arm_b=1,
        active_vehicles_apex=1,
        min_ttc=2.1,
        hazard_notes=["Imminent apex collision"]
    )

    # Calling with frame as first arg and assessment as third arg must NOT throw AttributeError
    temp_logger.log_event(dummy_frame, [v], assessment)

    recent = temp_logger.get_recent_events(limit=10)
    assert len(recent) == 1
    assert recent[0]["risk_label"] == "CRITICAL"


# ------------------------------------------------------------------ #
#  Phase 3: Emergency Archival & Audit Trail Tests                    #
# ------------------------------------------------------------------ #

def test_logger_emergency_event_archival(temp_logger):
    """Verifies that ambulance events store is_emergency, emergency_type, and emergency_preempted."""
    amb = TrackedVehicle(
        track_id=5,
        class_id=2,
        class_name="ambulance",
        confidence=0.96,
        box=[120, 120, 220, 220],
        center=(170, 170),
        speed_kmh=58.0,
        is_emergency=True,
        emergency_type="ambulance"
    )

    assessment = RiskAssessment(
        overall_level=RiskLevel.WARNING,
        conflict_pairs=[],
        signal_a_state="AMBER",
        signal_a_message="PRIORITY CLEARANCE",
        signal_b_state="RED",
        signal_b_message="STOP — YIELD TO AMBULANCE",
        active_vehicles_arm_a=1,
        active_vehicles_arm_b=0,
        active_vehicles_apex=0,
        min_ttc=3.8,
        hazard_notes=["PRIORITY PREEMPTION: 1 AMBULANCE detected"],
        emergency_preemption_active=True,
        emergency_vehicle_count=1,
        emergency_signal_state="PREEMPTED"
    )

    temp_logger.log_event(assessment, [amb])

    recent = temp_logger.get_recent_events(limit=5)
    assert len(recent) == 1
    row = recent[0]
    assert row["is_emergency"] == 1
    assert row["emergency_type"] == "ambulance"
    assert row["emergency_preempted"] == 1
    assert "AMBULANCE" in row["hazard_description"]


def test_logger_emergency_filtering(temp_logger):
    """Verifies that emergency_only=True filters out routine traffic incidents."""
    # Routine car incident
    car = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.85,
        box=[50, 50, 100, 100],
        center=(75, 75),
        speed_kmh=40.0
    )
    routine_assessment = RiskAssessment(
        overall_level=RiskLevel.WARNING,
        conflict_pairs=[],
        signal_a_state="YELLOW",
        signal_a_message="SLOW DOWN",
        signal_b_state="YELLOW",
        signal_b_message="SLOW DOWN",
        hazard_notes=["Opposing cars approaching"]
    )
    temp_logger.log_event(routine_assessment, [car])

    # Ambulance incident
    amb = TrackedVehicle(
        track_id=2,
        class_id=2,
        class_name="ambulance",
        confidence=0.92,
        box=[200, 200, 300, 300],
        center=(250, 250),
        speed_kmh=60.0,
        is_emergency=True,
        emergency_type="ambulance"
    )
    amb_assessment = RiskAssessment(
        overall_level=RiskLevel.CRITICAL,
        conflict_pairs=[],
        signal_a_state="AMBER",
        signal_a_message="CLEAR BEND",
        signal_b_state="RED",
        signal_b_message="STOP",
        hazard_notes=["AMBULANCE PRIORITY"],
        emergency_preemption_active=True,
        emergency_vehicle_count=1
    )
    temp_logger.log_event(amb_assessment, [amb])

    # All events
    all_events = temp_logger.get_recent_events(limit=10, emergency_only=False)
    assert len(all_events) == 2

    # Emergency events only
    emerg_events = temp_logger.get_recent_events(limit=10, emergency_only=True)
    assert len(emerg_events) == 1
    assert emerg_events[0]["emergency_type"] == "ambulance"

    # Emergency summary
    summary = temp_logger.get_emergency_summary()
    assert summary["total_emergency_events"] == 1
    assert summary["ambulance_count"] == 1
    assert summary["preemptions_triggered"] == 1


def test_logger_emergency_snapshot_naming():
    """Verifies emergency incidents generate a distinct snapshot filename prefix."""
    import numpy as np
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "test_snap.db")
    csv_path = os.path.join(tmp_dir, "test_snap.csv")
    snaps_dir = os.path.join(tmp_dir, "test_snaps")

    logger = EventLogger(
        db_path=db_path,
        csv_path=csv_path,
        snapshots_dir=snaps_dir,
        save_snapshots=True,
        snapshot_min_risk=2,
        snapshot_cooldown_seconds=0.0
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    amb = TrackedVehicle(
        track_id=10,
        class_id=2,
        class_name="ambulance",
        confidence=0.95,
        box=[100, 100, 200, 200],
        center=(150, 150),
        is_emergency=True,
        emergency_type="ambulance"
    )
    assessment = RiskAssessment(
        overall_level=RiskLevel.WARNING,
        conflict_pairs=[],
        signal_a_state="AMBER",
        signal_a_message="AMBULANCE CLEARANCE",
        signal_b_state="RED",
        signal_b_message="STOP",
        emergency_preemption_active=True
    )

    snap_path = logger.log_event(assessment, [amb], dummy_frame)
    assert snap_path is not None
    assert os.path.exists(snap_path)
    assert "emergency_ambulance" in os.path.basename(snap_path)

