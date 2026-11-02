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
