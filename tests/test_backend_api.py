"""
Tests for FastAPI Backend Service Endpoints (/api/health, /api/sample, /api/detect, /api/video).
"""
import os
import cv2
import base64
import pytest
import backend_service


def test_api_health_endpoint():
    result = backend_service.health()
    assert result["status"] == "healthy"
    assert result["model"] == "yolov8n.pt"
    assert "car" in result["target_classes"]


def test_api_sample_endpoint():
    sample_path = "data/sample_videos/images.jpeg"
    if not os.path.exists(sample_path):
        pytest.skip(f"{sample_path} not found")

    result = backend_service.get_sample_info()
    assert result["filename"] == "images.jpeg"
    assert result["width"] > 0
    assert result["height"] > 0
    assert result["image"].startswith("data:image/jpeg;base64,")


def test_api_detect_endpoint_json():
    sample_path = "data/sample_videos/images.jpeg"
    if not os.path.exists(sample_path):
        pytest.skip(f"{sample_path} not found")

    with open(sample_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = backend_service.DetectPayload(
        image=f"data:image/jpeg;base64,{b64}",
        confidence=0.25,
        iou=0.45,
        classes="car,motorcycle,bus,truck"
    )

    res = backend_service.detect_image(payload)
    assert res["success"] is True
    assert res["summary"]["total"] >= 5
    assert len(res["detections"]) == res["summary"]["total"]
    assert "boxes_only" in res["images"]
    assert "full_hud" in res["images"]
    assert "original" in res["images"]
    assert res["risk_assessment"]["label"] in ["SAFE", "CAUTION", "WARNING", "CRITICAL"]
    assert res["risk_assessment"]["signal_a_state"] in ["GREEN", "YELLOW", "RED"]
    assert res["risk_assessment"]["signal_b_state"] in ["GREEN", "YELLOW", "RED"]


def test_api_detect_fallback():
    # Test fallback to sample image when payload is None
    res = backend_service.detect_image(None)
    assert res["success"] is True
    assert res["summary"]["total"] >= 5


def test_api_video_status():
    status = backend_service.get_video_status()
    assert "video_name" in status
    assert "is_custom_video" in status


def test_api_video_set_source_synthetic():
    req = backend_service.StreamSwitchRequest(source="synthetic")
    res = backend_service.switch_stream(req)
    assert res["status"] == "switched"
    assert res["source"] == "synthetic"


def test_api_video_set_source_endpoint():
    # Test setting source via POST /api/video/set-source
    req = backend_service.VideoSetSourceRequest(video_path="synthetic", filename="Synthetic Curve")
    res = backend_service.set_video_source(req)
    assert res["success"] is True
    assert res["status"] == "active"
    assert "stream_url" in res


def test_api_centric_line_config_and_reset():
    cfg = backend_service.CentricLineConfig(enabled=True, y_ratio=0.55)
    res = backend_service.configure_centric_line(cfg)
    assert res["success"] is True
    assert res["enabled"] is True
    assert abs(res["y_ratio"] - 0.55) < 1e-4

    res_reset = backend_service.reset_centric_line()
    assert res_reset["success"] is True
    assert res_reset["total_crossings"] == 0
    assert res_reset["left_lane_crossings"] == 0


def test_api_left_lane_mode_config():
    req = backend_service.LeftLaneModeConfig(enabled=True)
    res = backend_service.configure_left_lane_mode(req)
    assert res["success"] is True
    assert res["filter_left_lane_only"] is True

    # Reset back to False
    req_off = backend_service.LeftLaneModeConfig(enabled=False)
    res_off = backend_service.configure_left_lane_mode(req_off)
    assert res_off["filter_left_lane_only"] is False


def test_api_left_lane_passing_telemetry():
    telemetry = backend_service.stream_telemetry()
    assert "centric_line" in telemetry
    cl = telemetry["centric_line"]
    assert "left_lane_crossings" in cl
    assert "filter_left_lane_only" in cl
    assert "red_alert" in cl


# ------------------------------------------------------------------ #
#  Phase 2: Emergency Preemption API Telemetry Tests                  #
# ------------------------------------------------------------------ #

def test_api_stream_telemetry_has_emergency_preemption():
    """stream_telemetry endpoint must always expose emergency_preemption block."""
    telemetry = backend_service.stream_telemetry()
    assert "emergency_preemption" in telemetry
    ep = telemetry["emergency_preemption"]
    assert "active" in ep
    assert "count" in ep
    assert "state" in ep
    assert isinstance(ep["active"], bool)


def test_api_detect_has_emergency_preemption_fields():
    """detect endpoint risk_assessment must include emergency preemption metadata."""
    res = backend_service.detect_image(None)
    assert res["success"] is True
    risk = res["risk_assessment"]
    assert "emergency_preemption_active" in risk
    assert "emergency_vehicle_count" in risk
    assert "emergency_signal_state" in risk
    assert "ambulance" in res["summary"]


# ------------------------------------------------------------------ #
#  Phase 4: Corridor Multi-Camera & Clearance Lock API Tests          #
# ------------------------------------------------------------------ #

def test_api_corridor_cameras_endpoint():
    """GET /api/corridor/cameras returns corridor id and all registered sector cameras."""
    res = backend_service.get_corridor_cameras()
    assert res["corridor_id"] == "CORRIDOR_HAIRPIN_01"
    assert "active_camera_id" in res
    assert "cameras" in res
    cameras = res["cameras"]
    assert "CAM_HAIRPIN_01" in cameras
    assert "CAM_UPHILL_02" in cameras
    assert "CAM_DOWNHILL_03" in cameras


def test_api_corridor_switch_camera_endpoint():
    """POST /api/corridor/switch-camera updates active sector camera in corridor."""
    req = backend_service.CameraSwitchRequest(camera_id="CAM_UPHILL_02")
    res = backend_service.switch_corridor_camera(req)
    assert res["success"] is True
    assert res["active_camera_id"] == "CAM_UPHILL_02"
    assert res["camera_info"]["sector"] == "Arm A (Uphill)"

    # Switch back to apex
    req_back = backend_service.CameraSwitchRequest(camera_id="CAM_HAIRPIN_01")
    res_back = backend_service.switch_corridor_camera(req_back)
    assert res_back["active_camera_id"] == "CAM_HAIRPIN_01"


def test_api_corridor_lock_telemetry():
    """Telemetry contains corridor tracking metadata and clearance lock object."""
    telemetry = backend_service.stream_telemetry()
    assert "corridor_id" in telemetry
    assert "camera_id" in telemetry
    assert "corridor_lock" in telemetry
    clock = telemetry["corridor_lock"]
    assert "locked" in clock
    assert "preempted" in clock
    assert "signal_state" in clock


