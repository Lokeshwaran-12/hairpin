"""
Tests for Collision Risk Predictor and Bidirectional Signal Logic.
"""
import pytest
from src.tracking.tracker import TrackedVehicle
from src.risk_assessment.collision_predictor import CollisionPredictor, RiskLevel


@pytest.fixture
def sample_zones():
    return {
        "zone_a": {
            "name": "Arm A",
            "polygon": [[0.05, 0.95], [0.38, 0.95], [0.38, 0.45], [0.15, 0.45]]
        },
        "zone_b": {
            "name": "Arm B",
            "polygon": [[0.62, 0.95], [0.95, 0.95], [0.85, 0.45], [0.62, 0.45]]
        },
        "conflict_zone": {
            "name": "Blind Apex",
            "polygon": [[0.18, 0.45], [0.82, 0.45], [0.75, 0.15], [0.25, 0.15]]
        }
    }


def test_predictor_safe_state(sample_zones):
    predictor = CollisionPredictor(zones_config=sample_zones)
    # No vehicles
    assessment = predictor.evaluate([], (720, 1280))
    assert assessment.overall_level == RiskLevel.SAFE
    assert assessment.signal_a_state == "GREEN"
    assert assessment.signal_b_state == "GREEN"


def test_predictor_single_vehicle_caution(sample_zones):
    predictor = CollisionPredictor(zones_config=sample_zones)

    # Single vehicle approaching in Zone A
    # Scaled coordinates: (0.25 * 1280 = 320, 0.70 * 720 = 504)
    v1 = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.85,
        box=[300, 480, 340, 528],
        center=(320, 504),
        history=[(310, 530), (320, 504)],
        speed_kmh=35.0
    )

    assessment = predictor.evaluate([v1], (720, 1280))
    assert assessment.overall_level == RiskLevel.CAUTION
    assert assessment.active_vehicles_arm_a == 1
    assert assessment.signal_a_state == "YELLOW"


def test_predictor_opposing_conflict_critical(sample_zones):
    predictor = CollisionPredictor(zones_config=sample_zones, ttc_critical=3.0, ttc_warning=6.0)

    # Vehicle in Arm A heading close to apex
    v_a = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.9,
        box=[340, 360, 380, 400],
        center=(360, 380),
        history=[(340, 440), (360, 380)],
        speed_kmh=40.0
    )

    # Vehicle in Arm B also heading close to apex (imminent meeting)
    v_b = TrackedVehicle(
        track_id=2,
        class_id=5,
        class_name="bus",
        confidence=0.92,
        box=[780, 360, 840, 410],
        center=(810, 385),
        history=[(840, 440), (810, 385)],
        speed_kmh=35.0
    )

    assessment = predictor.evaluate([v_a, v_b], (720, 1280))
    assert assessment.overall_level in [RiskLevel.WARNING, RiskLevel.CRITICAL]
    # At least one signal should alert STOP (RED) or SLOW (YELLOW)
    assert assessment.signal_a_state in ["RED", "YELLOW"]
    assert assessment.signal_b_state in ["RED", "YELLOW"]


def test_predictor_multiple_vehicles_in_apex_critical(sample_zones):
    predictor = CollisionPredictor(zones_config=sample_zones)

    # Two vehicles inside the blind apex polygon (e.g., around x=640, y=250)
    v1 = TrackedVehicle(
        track_id=1,
        class_id=2,
        class_name="car",
        confidence=0.88,
        box=[600, 220, 640, 260],
        center=(620, 240)
    )
    v2 = TrackedVehicle(
        track_id=2,
        class_id=7,
        class_name="truck",
        confidence=0.85,
        box=[650, 220, 710, 270],
        center=(680, 245)
    )

    assessment = predictor.evaluate([v1, v2], (720, 1280))
    assert assessment.overall_level == RiskLevel.CRITICAL
    assert assessment.active_vehicles_apex == 2


# ------------------------------------------------------------------ #
#  Phase 2: Emergency Vehicle Priority & Preemption Tests             #
# ------------------------------------------------------------------ #

def test_predictor_emergency_hazard_weight(sample_zones):
    """Ambulance must carry highest hazard weight (2.0) for rapid collision preemption."""
    predictor = CollisionPredictor(zones_config=sample_zones)
    assert "ambulance" in predictor.hazard_weights
    assert predictor.hazard_weights["ambulance"] >= 2.0


def test_predictor_emergency_preemption_arm_a(sample_zones):
    """
    Ambulance in Arm A (Uphill approach) must trigger priority preemption:
    - Arm A gets AMBER priority clearance
    - Arm B (opposing) gets RED stop signal
    - emergency_preemption_active is True
    """
    predictor = CollisionPredictor(zones_config=sample_zones)
    amb = TrackedVehicle(
        track_id=10,
        class_id=2,
        class_name="ambulance",
        confidence=0.95,
        box=[300, 480, 340, 528],
        center=(320, 504),
        history=[(310, 530), (320, 504)],
        speed_kmh=55.0,
        is_emergency=True,
        emergency_type="ambulance"
    )

    assessment = predictor.evaluate([amb], (720, 1280))
    assert assessment.emergency_preemption_active is True
    assert assessment.emergency_vehicle_count == 1
    assert assessment.emergency_signal_state == "PREEMPTED"
    assert assessment.signal_a_state == "AMBER"
    assert "AMBULANCE" in assessment.signal_a_message
    assert assessment.signal_b_state == "RED"
    assert "STOP" in assessment.signal_b_message
    assert any("AMBULANCE" in note for note in assessment.hazard_notes)


def test_predictor_emergency_preemption_arm_b(sample_zones):
    """
    Ambulance in Arm B (Downhill approach) must trigger priority preemption:
    - Arm B gets AMBER priority clearance
    - Arm A gets RED stop signal
    """
    predictor = CollisionPredictor(zones_config=sample_zones)
    amb = TrackedVehicle(
        track_id=20,
        class_id=2,
        class_name="ambulance",
        confidence=0.94,
        box=[780, 480, 840, 530],
        center=(810, 505),
        history=[(820, 535), (810, 505)],
        speed_kmh=50.0,
        is_emergency=True,
        emergency_type="ambulance"
    )

    assessment = predictor.evaluate([amb], (720, 1280))
    assert assessment.emergency_preemption_active is True
    assert assessment.signal_b_state == "AMBER"
    assert assessment.signal_a_state == "RED"
    assert "STOP" in assessment.signal_a_message


def test_predictor_emergency_preemption_in_apex(sample_zones):
    """
    Ambulance inside conflict apex halts all opposing arms (both RED).
    """
    predictor = CollisionPredictor(zones_config=sample_zones)
    amb = TrackedVehicle(
        track_id=30,
        class_id=2,
        class_name="ambulance",
        confidence=0.96,
        box=[600, 220, 640, 260],
        center=(620, 240),
        is_emergency=True,
        emergency_type="ambulance"
    )

    assessment = predictor.evaluate([amb], (720, 1280))
    assert assessment.emergency_preemption_active is True
    assert assessment.overall_level == RiskLevel.CRITICAL
    assert assessment.signal_a_state == "RED"
    assert assessment.signal_b_state == "RED"

