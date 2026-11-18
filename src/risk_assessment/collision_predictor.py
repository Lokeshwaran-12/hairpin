"""
Predictive Collision Risk Engine for Hairpin Bends:
Models spatial zones (Approach Arm A, Approach Arm B, Conflict Apex),
estimates Time-to-Conflict (TTC), accounts for heavy vehicle sweep hazards,
and computes bidirectional traffic signals (Signal A and Signal B).
"""
import math
from enum import IntEnum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import cv2
import numpy as np
from src.tracking.tracker import TrackedVehicle


class RiskLevel(IntEnum):
    SAFE = 0       # Level 0: Green - Clear / normal flow
    CAUTION = 1    # Level 1: Yellow - Vehicle approaching / single vehicle in curve
    WARNING = 2    # Level 2: Orange - Opposing vehicles approaching, reduced gap
    CRITICAL = 3   # Level 3: Red - Imminent head-on conflict / simultaneous occupancy


@dataclass
class ConflictPair:
    v1: TrackedVehicle
    v2: TrackedVehicle
    ttc_seconds: float
    risk_level: RiskLevel
    description: str


@dataclass
class RiskAssessment:
    overall_level: RiskLevel
    conflict_pairs: List[ConflictPair]
    signal_a_state: str        # "GREEN", "YELLOW", "RED", "AMBER"
    signal_a_message: str
    signal_b_state: str        # "GREEN", "YELLOW", "RED", "AMBER"
    signal_b_message: str
    active_vehicles_arm_a: int = 0
    active_vehicles_arm_b: int = 0
    active_vehicles_apex: int = 0
    min_ttc: Optional[float] = None
    hazard_notes: List[str] = field(default_factory=list)
    # Phase 2: Emergency Vehicle Priority Preemption
    emergency_preemption_active: bool = False
    emergency_vehicle_count: int = 0
    emergency_signal_state: str = "NORMAL"   # "NORMAL" | "PREEMPTED"


class CollisionPredictor:
    """
    Predicts collision hazards at blind hairpin curves.
    """
    def __init__(
        self,
        zones_config: Dict,
        ttc_critical: float = 3.0,
        ttc_warning: float = 6.0,
        ttc_caution: float = 10.0,
        pixels_per_meter: float = 18.0,
        hazard_weights: Optional[Dict[str, float]] = None
    ):
        self.raw_zones = zones_config
        self.ttc_critical = ttc_critical
        self.ttc_warning = ttc_warning
        self.ttc_caution = ttc_caution
        self.pixels_per_meter = max(1.0, pixels_per_meter)
        self.hazard_weights = hazard_weights or {
            "ambulance": 2.0,
            "bus": 1.5,
            "truck": 1.5,
            "car": 1.0,
            "motorcycle": 1.0
        }

        # Scaled polygons cache
        self.scaled_zones: Dict[str, np.ndarray] = {}
        self.current_frame_shape: Optional[Tuple[int, int]] = None

    def _scale_zones(self, width: int, height: int):
        """Scales normalized zone coordinates to the frame dimensions."""
        if self.current_frame_shape == (width, height) and self.scaled_zones:
            return

        self.current_frame_shape = (width, height)
        self.scaled_zones = {}

        for zone_key, zone_info in self.raw_zones.items():
            pts = zone_info.get("polygon", [])
            scaled_pts = []
            for p in pts:
                sx = int(p[0] * width)
                sy = int(p[1] * height)
                scaled_pts.append([sx, sy])
            self.scaled_zones[zone_key] = np.array(scaled_pts, dtype=np.int32)

    def _get_zone(self, point: Tuple[int, int]) -> str:
        """Determines which zone a point lies within."""
        for zone_key, poly in self.scaled_zones.items():
            if cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), measureDist=False) >= 0:
                return zone_key
        return "none"

    def _get_conflict_apex_centroid(self) -> Tuple[float, float]:
        """Calculates centroid of the conflict zone."""
        poly = self.scaled_zones.get("conflict_zone")
        if poly is not None and len(poly) > 0:
            M = cv2.moments(poly)
            if M["m00"] != 0:
                return (M["m10"] / M["m00"], M["m01"] / M["m00"])
        # Fallback to top center
        w, h = self.current_frame_shape if self.current_frame_shape else (1280, 720)
        return (w * 0.5, h * 0.3)

    def _distance_to_conflict(self, point: Tuple[int, int]) -> float:
        """Distance in meters from a point to the conflict zone contour."""
        poly = self.scaled_zones.get("conflict_zone")
        if poly is None:
            return 50.0

        dist_px = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), measureDist=True)
        if dist_px >= 0:
            return 0.0  # Inside conflict apex
        return abs(dist_px) / self.pixels_per_meter

    def _is_moving_towards_apex(self, vehicle: TrackedVehicle, apex_center: Tuple[float, float]) -> bool:
        """Determines if the vehicle is approaching the blind apex."""
        if len(vehicle.history) < 2:
            return True

        cx, cy = vehicle.center
        old_cx, old_cy = vehicle.history[0]

        curr_dist = math.hypot(cx - apex_center[0], cy - apex_center[1])
        old_dist = math.hypot(old_cx - apex_center[0], old_cy - apex_center[1])

        return curr_dist < old_dist

    def evaluate(self, vehicles: List[TrackedVehicle], frame_shape: Tuple[int, int]) -> RiskAssessment:
        """
        Analyzes tracked vehicles, estimates arrival times at blind apex,
        and generates risk state and bidirectional signal outputs.
        """
        height, width = frame_shape[:2]
        self._scale_zones(width, height)
        apex_center = self._get_conflict_apex_centroid()

        arm_a_vehicles: List[TrackedVehicle] = []
        arm_b_vehicles: List[TrackedVehicle] = []
        apex_vehicles: List[TrackedVehicle] = []

        # Classify vehicle zones & movement
        for v in vehicles:
            zone = self._get_zone(v.center)
            v.current_zone = zone

            if zone == "conflict_zone":
                v.direction_status = "Inside Apex"
                apex_vehicles.append(v)
            elif zone == "zone_a":
                if self._is_moving_towards_apex(v, apex_center):
                    v.direction_status = "Approaching Apex"
                    arm_a_vehicles.append(v)
                else:
                    v.direction_status = "Exiting Bend"
            elif zone == "zone_b":
                if self._is_moving_towards_apex(v, apex_center):
                    v.direction_status = "Approaching Apex"
                    arm_b_vehicles.append(v)
                else:
                    v.direction_status = "Exiting Bend"
            else:
                # Outside defined zones: check proximity
                if self._is_moving_towards_apex(v, apex_center):
                    # Check which side of the frame it is on
                    if v.center[0] < width * 0.5:
                        arm_a_vehicles.append(v)
                    else:
                        arm_b_vehicles.append(v)

        overall_risk = RiskLevel.SAFE
        conflict_pairs: List[ConflictPair] = []
        hazard_notes: List[str] = []
        min_ttc: Optional[float] = None

        # Scenario 1: Multiple vehicles already inside the blind apex simultaneously
        if len(apex_vehicles) >= 2:
            overall_risk = RiskLevel.CRITICAL
            hazard_notes.append("Multiple vehicles inside narrow conflict apex!")
            min_ttc = 0.0
            conflict_pairs.append(ConflictPair(
                v1=apex_vehicles[0],
                v2=apex_vehicles[1],
                ttc_seconds=0.0,
                risk_level=RiskLevel.CRITICAL,
                description="Concurrence inside blind apex"
            ))

        # Scenario 2: One vehicle in apex, and an opposing vehicle approaching
        elif len(apex_vehicles) == 1:
            inside_v = apex_vehicles[0]
            # Check for vehicles approaching from either arm
            approaching = arm_a_vehicles + arm_b_vehicles
            if approaching:
                for opp in approaching:
                    dist_m = self._distance_to_conflict(opp.center)
                    speed_ms = max(opp.speed_kmh / 3.6, 1.5)
                    ttc = dist_m / speed_ms
                    min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)

                    weight = max(self.hazard_weights.get(inside_v.class_name, 1.0),
                                 self.hazard_weights.get(opp.class_name, 1.0))
                    adj_ttc = ttc / weight

                    if adj_ttc <= self.ttc_critical:
                        pair_risk = RiskLevel.CRITICAL
                    elif adj_ttc <= self.ttc_warning:
                        pair_risk = RiskLevel.WARNING
                    else:
                        pair_risk = RiskLevel.CAUTION

                    if pair_risk > overall_risk:
                        overall_risk = pair_risk

                    conflict_pairs.append(ConflictPair(
                        v1=inside_v,
                        v2=opp,
                        ttc_seconds=round(ttc, 1),
                        risk_level=pair_risk,
                        description=f"Apex occupied by {inside_v.class_name}, {opp.class_name} approaching"
                    ))
            else:
                overall_risk = max(overall_risk, RiskLevel.CAUTION)

        # Scenario 3: Opposing vehicles approaching from Arm A and Arm B
        if arm_a_vehicles and arm_b_vehicles:
            for va in arm_a_vehicles:
                dist_a = self._distance_to_conflict(va.center)
                speed_a = max(va.speed_kmh / 3.6, 2.0)
                ttc_a = dist_a / speed_a

                for vb in arm_b_vehicles:
                    dist_b = self._distance_to_conflict(vb.center)
                    speed_b = max(vb.speed_kmh / 3.6, 2.0)
                    ttc_b = dist_b / speed_b

                    # Arrival difference
                    arrival_diff = abs(ttc_a - ttc_b)
                    avg_ttc = (ttc_a + ttc_b) / 2.0
                    min_ttc = min(ttc_a, ttc_b) if min_ttc is None else min(min_ttc, ttc_a, ttc_b)

                    # Heavy vehicle hazard multiplier
                    weight = max(self.hazard_weights.get(va.class_name, 1.0),
                                 self.hazard_weights.get(vb.class_name, 1.0))

                    effective_ttc = avg_ttc / weight

                    # Evaluate conflict severity
                    if effective_ttc <= self.ttc_critical or (avg_ttc < 4.0 and arrival_diff < 3.0):
                        pair_risk = RiskLevel.CRITICAL
                        hazard_notes.append(f"Imminent collision risk: {va.class_name} vs {vb.class_name}")
                    elif effective_ttc <= self.ttc_warning or (avg_ttc < 7.0 and arrival_diff < 4.0):
                        pair_risk = RiskLevel.WARNING
                        hazard_notes.append(f"Opposing approach hazard: {va.class_name} vs {vb.class_name}")
                    elif effective_ttc <= self.ttc_caution:
                        pair_risk = RiskLevel.CAUTION
                    else:
                        pair_risk = RiskLevel.SAFE

                    if pair_risk > overall_risk:
                        overall_risk = pair_risk

                    conflict_pairs.append(ConflictPair(
                        v1=va,
                        v2=vb,
                        ttc_seconds=round(avg_ttc, 1),
                        risk_level=pair_risk,
                        description=f"Arm A ({va.class_name}) vs Arm B ({vb.class_name})"
                    ))
        elif arm_a_vehicles or arm_b_vehicles:
            # Single approach active
            if overall_risk == RiskLevel.SAFE:
                overall_risk = RiskLevel.CAUTION

        # ------------------------------------------------------------------ #
        # Signal State Computation (Signal A for Uphill, Signal B for Downhill)
        # ------------------------------------------------------------------ #

        # Phase 2: Emergency Vehicle Priority Preemption
        # Detect ambulances anywhere in scene and override signals immediately.
        emergency_vehicles = [v for v in vehicles if getattr(v, "is_emergency", False) or v.class_name == "ambulance"]
        emergency_preemption_active = len(emergency_vehicles) > 0
        emergency_signal_state = "PREEMPTED" if emergency_preemption_active else "NORMAL"

        if emergency_preemption_active:
            overall_risk = max(overall_risk, RiskLevel.WARNING)
            arm_a_emerg = any(getattr(v, "is_emergency", False) or v.class_name == "ambulance" for v in arm_a_vehicles)
            arm_b_emerg = any(getattr(v, "is_emergency", False) or v.class_name == "ambulance" for v in arm_b_vehicles)
            apex_emerg = any(getattr(v, "is_emergency", False) or v.class_name == "ambulance" for v in apex_vehicles)

            if arm_a_emerg and not arm_b_emerg and not apex_emerg:
                signal_a_state = "AMBER"
                signal_a_msg   = "PRIORITY CLEARANCE — AMBULANCE ASCENDING"
                signal_b_state = "RED"
                signal_b_msg   = "STOP — YIELD TO EMERGENCY AMBULANCE"
            elif arm_b_emerg and not arm_a_emerg and not apex_emerg:
                signal_a_state = "RED"
                signal_a_msg   = "STOP — YIELD TO EMERGENCY AMBULANCE"
                signal_b_state = "AMBER"
                signal_b_msg   = "PRIORITY CLEARANCE — AMBULANCE DESCENDING"
            elif apex_emerg or (arm_a_emerg and arm_b_emerg):
                signal_a_state = "RED"
                signal_a_msg   = "STOP — AMBULANCE IN APEX, YIELD"
                signal_b_state = "RED"
                signal_b_msg   = "STOP — AMBULANCE IN APEX, YIELD"
                overall_risk   = RiskLevel.CRITICAL
            else:
                signal_a_state = "AMBER"
                signal_a_msg   = "EMERGENCY VEHICLE — YIELD & CLEAR BEND"
                signal_b_state = "AMBER"
                signal_b_msg   = "EMERGENCY VEHICLE — YIELD & CLEAR BEND"

            emerg_note = f"PRIORITY PREEMPTION: {len(emergency_vehicles)} AMBULANCE detected — all traffic yield"
            if emerg_note not in hazard_notes:
                hazard_notes.insert(0, emerg_note)

        elif overall_risk == RiskLevel.CRITICAL:
            # Check heavy vehicle priority: uphill or heavier vehicle gets priority
            has_heavy_a = any(v.class_name in ["bus", "truck"] for v in arm_a_vehicles)
            has_heavy_b = any(v.class_name in ["bus", "truck"] for v in arm_b_vehicles)

            if has_heavy_b and not has_heavy_a:
                signal_a_state = "RED"
                signal_a_msg = "STOP - HEAVY VEHICLE ON BEND"
                signal_b_state = "YELLOW"
                signal_b_msg = "CAUTION - PROCEED SLOWLY"
            elif has_heavy_a and not has_heavy_b:
                signal_a_state = "YELLOW"
                signal_a_msg = "CAUTION - PROCEED SLOWLY"
                signal_b_state = "RED"
                signal_b_msg = "STOP - HEAVY VEHICLE ON BEND"
            else:
                # Standard mountain rule: uphill vehicle has right-of-way, downhill vehicle stops
                signal_a_state = "YELLOW"
                signal_a_msg = "CAUTION - OPPOSING TRAFFIC"
                signal_b_state = "RED"
                signal_b_msg = "STOP - YIELD TO UPHILL TRAFFIC"
        elif overall_risk == RiskLevel.WARNING:
            signal_a_state = "YELLOW"
            signal_a_msg = "SLOW DOWN - VEHICLE APPROACHING"
            signal_b_state = "YELLOW"
            signal_b_msg = "SLOW DOWN - VEHICLE APPROACHING"
        elif overall_risk == RiskLevel.CAUTION:
            signal_a_state = "YELLOW"
            signal_a_msg = "HAIRPIN BEND - SOUND HORN"
            signal_b_state = "YELLOW"
            signal_b_msg = "HAIRPIN BEND - SOUND HORN"
        else:
            signal_a_state = "GREEN"
            signal_a_msg = "BEND CLEAR - DRIVE SAFE"
            signal_b_state = "GREEN"
            signal_b_msg = "BEND CLEAR - DRIVE SAFE"

        return RiskAssessment(
            overall_level=overall_risk,
            conflict_pairs=conflict_pairs,
            signal_a_state=signal_a_state,
            signal_a_message=signal_a_msg,
            signal_b_state=signal_b_state,
            signal_b_message=signal_b_msg,
            active_vehicles_arm_a=len(arm_a_vehicles),
            active_vehicles_arm_b=len(arm_b_vehicles),
            active_vehicles_apex=len(apex_vehicles),
            min_ttc=min_ttc,
            hazard_notes=hazard_notes,
            emergency_preemption_active=emergency_preemption_active,
            emergency_vehicle_count=len(emergency_vehicles),
            emergency_signal_state=emergency_signal_state
        )
