"""
HUD Visualizer:
Overlays zones, vehicle trajectories, velocity vectors, dynamic bounding boxes,
bidirectional virtual traffic signals (Arm A and Arm B), and risk warning banners.
"""
import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.tracking.tracker import TrackedVehicle
from src.risk_assessment.collision_predictor import RiskAssessment, RiskLevel


class HUDVisualizer:
    def __init__(self, show_zones: bool = True, show_trajectories: bool = True):
        self.show_zones = show_zones
        self.show_trajectories = show_trajectories
        self.frame_idx = 0

        # Class colors (BGR)
        self.class_colors = {
            "car": (255, 180, 50),         # Sky blue / cyan
            "motorcycle": (220, 80, 220),   # Magenta
            "bus": (30, 215, 255),          # Bright yellow
            "truck": (50, 100, 240),        # Orange
            "ambulance": (0, 0, 255),       # Emergency Red
        }

        # Risk colors (BGR)
        self.risk_colors = {
            RiskLevel.SAFE: (50, 200, 50),       # Green
            RiskLevel.CAUTION: (30, 215, 255),   # Yellow
            RiskLevel.WARNING: (20, 140, 240),   # Orange
            RiskLevel.CRITICAL: (30, 30, 240)    # Red
        }

    def _draw_zones(self, frame: np.ndarray, scaled_zones: Dict[str, np.ndarray]):
        """Draws semi-transparent polygon overlays for monitored zones."""
        overlay = frame.copy()

        zone_styles = {
            "zone_a": ((240, 160, 40), "ARM A (Uphill Approach)"),
            "zone_b": ((200, 70, 200), "ARM B (Downhill Approach)"),
            "conflict_zone": ((40, 40, 220), "BLIND CONFLICT APEX")
        }

        for key, poly in scaled_zones.items():
            if poly is None or len(poly) < 3:
                continue

            color, label = zone_styles.get(key, ((180, 180, 180), key))
            cv2.fillPoly(overlay, [poly], color)
            cv2.polylines(frame, [poly], isClosed=True, color=color, thickness=2)

            # Zone label at polygon apex
            pts = poly.reshape(-1, 2)
            top_pt = pts[np.argmin(pts[:, 1])]
            cv2.putText(frame, label, (int(top_pt[0]) - 50, int(top_pt[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

    def _draw_vehicles(self, frame: np.ndarray, vehicles: List[TrackedVehicle], assessment: RiskAssessment):
        """Draws bounding boxes, trajectory trails, and status tags."""
        conflict_ids = set()
        for pair in assessment.conflict_pairs:
            conflict_ids.add(pair.v1.track_id)
            conflict_ids.add(pair.v2.track_id)

        for v in vehicles:
            is_in_conflict = v.track_id in conflict_ids
            has_crossed = getattr(v, "has_crossed_line", False)
            is_touching = getattr(v, "is_touching_line", False)
            is_heavy = (v.class_name in ["truck", "bus"] or getattr(v, "is_heavy_vehicle", False))
            is_left_passing = getattr(v, "is_left_passing", False)
            is_emergency = getattr(v, "is_emergency", False) or v.class_name == "ambulance"

            # High-visibility color scheme (priority: emergency > left-passing > conflict > crossed > heavy > class)
            if is_emergency:
                color = (0, 0, 255)  # Emergency Red
            elif is_left_passing:
                color = (0, 0, 255)  # Critical Red Alert for left passing car
            elif is_in_conflict:
                color = (30, 30, 240)
            elif has_crossed or is_touching:
                color = (0, 235, 110)  # Bright verification green
            elif is_heavy:
                color = (0, 140, 255)  # Vibrant Amber/Orange
            else:
                color = self.class_colors.get(v.class_name, (200, 200, 200))

            x1, y1, x2, y2 = v.box

            # Trajectory trail
            if self.show_trajectories and len(v.history) > 1:
                pts = np.array(v.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

            # Direction vector arrow
            if len(v.history) >= 2:
                cx, cy = v.center
                vx, vy = v.velocity_vector
                speed_px = math.hypot(vx, vy)
                if speed_px > 3.0:
                    arrow_len = min(40, max(15, int(speed_px * 0.15)))
                    norm = max(1.0, speed_px)
                    ax = int(cx + (vx / norm) * arrow_len)
                    ay = int(cy + (vy / norm) * arrow_len)
                    cv2.arrowedLine(frame, (cx, cy), (ax, ay), (0, 255, 255), 2, tipLength=0.35)

            # Bounding box (prominently marked with 4px border for emergency / left passing / heavy trucks / line passing)
            thickness = 4 if (is_emergency or is_left_passing or is_heavy or is_in_conflict or has_crossed or is_touching) else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Double-border highlight (priority: emergency > left passing > crossed > heavy)
            if is_emergency:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
                cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 0, 200), 1)
            elif is_left_passing:
                cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 2)
            elif has_crossed or is_touching:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1)
            elif is_heavy:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 200, 255), 1)

            # Label badge
            badge_color = color
            badge_text_color = (0, 0, 0)
            if is_emergency:
                badge_color = (0, 0, 200)
                badge_text_color = (255, 255, 255)
                label_text = f"[AMBULANCE] ID:{v.track_id} {v.speed_kmh}km/h"
            elif is_left_passing:
                dir_txt = f"{v.cross_direction}" if v.cross_direction else "PASSING"
                label_text = f"[RED ALERT | LEFT {dir_txt}] ID:{v.track_id} {v.class_name.upper()} {v.speed_kmh}km/h"
            elif has_crossed:
                dir_txt = f"{v.cross_direction}" if v.cross_direction else "VERIFIED"
                heavy_pfx = "[HEAVY] " if is_heavy else ""
                label_text = f"[PASSED LINE | {dir_txt}] {heavy_pfx}ID:{v.track_id} {v.class_name.upper()} {v.speed_kmh}km/h"
            elif is_touching:
                heavy_pfx = "[HEAVY] " if is_heavy else ""
                label_text = f"[CROSSING LINE] {heavy_pfx}ID:{v.track_id} {v.class_name.upper()}"
            elif is_heavy:
                label_text = f"[HEAVY {v.class_name.upper()}] ID:{v.track_id} {v.speed_kmh}km/h"
            else:
                label_text = f"ID:{v.track_id} {v.class_name.upper()} {v.speed_kmh}km/h"

            (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            badge_y1 = max(18, y1 - th - 8)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 8, badge_y1 + th + 6), badge_color, -1)
            cv2.putText(frame, label_text, (x1 + 4, badge_y1 + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, badge_text_color, 1, cv2.LINE_AA)

            # Direction status tag below box
            status_text = f"{v.direction_status} ({v.current_zone})"
            (sw, sh), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + sw + 6, y2 + sh + 8), (40, 40, 40), -1)
            cv2.putText(frame, status_text, (x1 + 3, y2 + sh + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_centric_line(self, frame: np.ndarray, line_y: int, is_active: bool, total_crossings: int, is_left_alert: bool = False):
        """Draws prominent centric detection line across roadway with crossing counter and red alert state."""
        h, w = frame.shape[:2]

        # Line color:
        # - Left passing alert: Pulsing Urgent Red Alert (0, 0, 255)
        # - Vehicle active on line: Energetic Yellow-Green (50, 255, 120)
        # - Idle line: Bright Cyan (0, 240, 255)
        if is_left_alert:
            base_color = (0, 0, 255)
            glow_color = (0, 0, 220)
            thickness = 4
        elif is_active:
            base_color = (50, 255, 120)
            glow_color = (30, 200, 80)
            thickness = 3
        else:
            base_color = (0, 240, 255)
            glow_color = (0, 160, 200)
            thickness = 2

        # 1. Subtle glow backdrop
        overlay = frame.copy()
        cv2.line(overlay, (0, line_y), (w, line_y), glow_color, 8 if not is_left_alert else 12)
        cv2.addWeighted(overlay, 0.35 if is_left_alert else 0.30, frame, 0.65 if is_left_alert else 0.70, 0, frame)

        # 2. Main dashed centric line
        dash_len = 24
        gap_len = 10
        for x in range(0, w, dash_len + gap_len):
            x_end = min(x + dash_len, w)
            cv2.line(frame, (x, line_y), (x_end, line_y), base_color, thickness, cv2.LINE_AA)

        # 3. Center HUD Badge on Line
        if is_left_alert:
            badge_text = f"=== [!] RED ALERT: LEFT PASSING VEHICLE [ PASSED: {total_crossings} ] ==="
        else:
            badge_text = f"=== CENTRIC DETECTION LINE [ PASSED: {total_crossings} ] ==="

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        bx1 = (w - tw) // 2 - 12
        by1 = line_y - th - 8
        bx2 = bx1 + tw + 24
        by2 = line_y + 10

        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 22, 26), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), base_color, 2)
        cv2.putText(frame, badge_text, (bx1 + 12, line_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, base_color, 1, cv2.LINE_AA)

    def _draw_traffic_signals(self, frame: np.ndarray, assessment: RiskAssessment):
        """Draws virtual bidirectional traffic signal boxes for Arm A and Arm B."""
        h, w = frame.shape[:2]

        # Signal A Box (Top Left)
        self._render_signal_box(
            frame,
            origin=(20, 70),
            title="SIGNAL A (UPHILL)",
            state=assessment.signal_a_state,
            message=assessment.signal_a_message
        )

        # Signal B Box (Top Right)
        self._render_signal_box(
            frame,
            origin=(w - 240, 70),
            title="SIGNAL B (DOWNHILL)",
            state=assessment.signal_b_state,
            message=assessment.signal_b_message
        )

    def _render_signal_box(self, frame: np.ndarray, origin: Tuple[int, int], title: str, state: str, message: str):
        """Renders an individual 3-aspect traffic signal board (supports GREEN/YELLOW/RED/AMBER)."""
        ox, oy = origin
        box_w, box_h = 220, 95

        # Housing background
        cv2.rectangle(frame, (ox, oy), (ox + box_w, oy + box_h), (25, 25, 28), -1)
        cv2.rectangle(frame, (ox, oy), (ox + box_w, oy + box_h), (80, 80, 85), 2)

        # Title
        cv2.putText(frame, title, (ox + 10, oy + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # 3 Aspect Lights + AMBER override
        is_amber = (state == "AMBER")
        r_color = (30, 30, 240) if state == "RED" else (40, 40, 70)
        y_color = (30, 215, 255) if state in ("YELLOW", "AMBER") else (40, 60, 70)
        g_color = (50, 220, 50) if state == "GREEN" else (40, 70, 40)

        # Amber overrides yellow slot with brighter orange
        if is_amber:
            y_color = (0, 160, 255)  # BGR: vivid orange-amber

        cx_r = ox + 35
        cx_y = ox + 80
        cx_g = ox + 125
        cy = oy + 45
        radius = 12

        cv2.circle(frame, (cx_r, cy), radius, r_color, -1)
        cv2.circle(frame, (cx_y, cy), radius, y_color, -1)
        cv2.circle(frame, (cx_g, cy), radius, g_color, -1)

        cv2.circle(frame, (cx_r, cy), radius, (150, 150, 150), 1)
        cv2.circle(frame, (cx_y, cy), radius, (150, 150, 150), 1)
        cv2.circle(frame, (cx_g, cy), radius, (150, 150, 150), 1)

        # Glow ring around active light
        if state == "RED":
            active_cx, active_color = cx_r, r_color
        elif state in ("YELLOW", "AMBER"):
            active_cx, active_color = cx_y, y_color
        else:
            active_cx, active_color = cx_g, g_color

        cv2.circle(frame, (active_cx, cy), radius + 3, active_color, 2)

        # AMBER: draw flashing emergency cross mark above signal housing
        if is_amber:
            cross_cx, cross_cy = ox + 175, oy + 45
            cv2.line(frame, (cross_cx - 10, cross_cy), (cross_cx + 10, cross_cy), (0, 160, 255), 3)
            cv2.line(frame, (cross_cx, cross_cy - 10), (cross_cx, cross_cy + 10), (0, 160, 255), 3)

        # Driver advisory message banner
        cv2.putText(frame, message, (ox + 8, oy + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, active_color, 1, cv2.LINE_AA)

    def _draw_top_banner(self, frame: np.ndarray, assessment: RiskAssessment):
        """Draws top central risk warning banner."""
        h, w = frame.shape[:2]
        banner_w = int(w * 0.52)
        banner_h = 44
        bx1 = (w - banner_w) // 2
        bx2 = bx1 + banner_w
        by1 = 12
        by2 = by1 + banner_h

        risk_color = self.risk_colors.get(assessment.overall_level, (100, 100, 100))

        # Flashing effect for critical alerts
        if assessment.overall_level == RiskLevel.CRITICAL and (self.frame_idx // 6) % 2 == 0:
            risk_color = (0, 0, 255)
            # Full screen perimeter warning border
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)

        # Banner background
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 25), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), risk_color, 2)

        # Text
        level_names = {
            RiskLevel.SAFE: "LEVEL 0: SAFE - BEND CLEAR",
            RiskLevel.CAUTION: "LEVEL 1: CAUTION - VEHICLE APPROACHING",
            RiskLevel.WARNING: "LEVEL 2: WARNING - OPPOSING APPROACH HAZARD",
            RiskLevel.CRITICAL: "LEVEL 3: CRITICAL ALERT - IMMINENT CONFLICT!"
        }
        status_str = level_names.get(assessment.overall_level, "UNKNOWN")

        if assessment.min_ttc is not None:
            status_str += f" | TTC: {assessment.min_ttc:.1f}s"

        (tw, th), _ = cv2.getTextSize(status_str, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        tx = bx1 + (banner_w - tw) // 2
        ty = by1 + (banner_h + th) // 2 - 2

        cv2.putText(frame, status_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.58, risk_color, 2, cv2.LINE_AA)

    def _draw_emergency_preemption_banner(
        self,
        frame: np.ndarray,
        emergency_count: int,
        corridor_lock: bool = False,
        corridor_id: str = "CORRIDOR_HAIRPIN_01"
    ):
        """
        Draws a prominent amber emergency preemption banner when an ambulance is detected
        or during corridor clearance lock.
        """
        h, w = frame.shape[:2]
        amber = (0, 160, 255)   # BGR vivid amber/orange

        banner_h = 46
        by1 = h - banner_h - 10
        by2 = h - 10

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, by1), (w - 10, by2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Amber border
        cv2.rectangle(frame, (10, by1), (w - 10, by2), amber, 3)

        # Ambulance cross icon (manual draw)
        icon_x, icon_cy = 38, (by1 + by2) // 2
        cv2.rectangle(frame, (icon_x - 10, icon_cy - 4), (icon_x + 10, icon_cy + 4), amber, -1)
        cv2.rectangle(frame, (icon_x - 4, icon_cy - 10), (icon_x + 4, icon_cy + 10), amber, -1)

        # Banner text
        if corridor_lock and emergency_count == 0:
            text = f"🚑  CORRIDOR CLEARANCE LOCK ACTIVE [{corridor_id}] — SAFE PASSAGE IN PROGRESS — ALL TRAFFIC YIELD"
        else:
            text = f"🚑  EMERGENCY PRIORITY PREEMPTION ACTIVE — {max(1, emergency_count)} AMBULANCE — ALL TRAFFIC YIELD & CLEAR BEND"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 2)
        tx = (w - tw) // 2
        ty = (by1 + by2 + th) // 2

        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, amber, 2, cv2.LINE_AA)

    def render(
        self,
        frame: np.ndarray,
        vehicles: List[TrackedVehicle],
        assessment: RiskAssessment,
        scaled_zones: Dict[str, np.ndarray],
        centric_line_y: Optional[int] = None,
        centric_line_enabled: bool = True,
        total_crossings: int = 0,
        is_left_lane_alert: bool = False
    ) -> np.ndarray:
        """Renders complete HUD overlay on frame."""
        self.frame_idx += 1
        output_frame = frame.copy()
        h, w = output_frame.shape[:2]

        # 1. Monitored spatial zones
        if self.show_zones and scaled_zones:
            self._draw_zones(output_frame, scaled_zones)

        # 2. Centric Detection Line (Virtual Trigger Line)
        if centric_line_enabled:
            ly = centric_line_y if centric_line_y is not None else int(h * 0.52)
            is_active = any(getattr(v, "is_touching_line", False) for v in vehicles)
            self._draw_centric_line(
                output_frame,
                line_y=ly,
                is_active=(is_active or is_left_lane_alert),
                total_crossings=total_crossings,
                is_left_alert=is_left_lane_alert
            )

        # 3. Tracked vehicles with trajectories, marked bounding boxes & badges
        self._draw_vehicles(output_frame, vehicles, assessment)

        # 4. Virtual traffic signal heads (AMBER state supported for preemption)
        self._draw_traffic_signals(output_frame, assessment)

        # 5. Central alert banner
        self._draw_top_banner(output_frame, assessment)

        # 6. Phase 2 & 4 — Emergency Preemption & Corridor Lock Overlay
        is_preempted = getattr(assessment, "emergency_preemption_active", False)
        is_corridor_locked = getattr(assessment, "corridor_clearance_lock", False)
        emerg_count = getattr(assessment, "emergency_vehicle_count", 0)
        corridor_id = getattr(assessment, "corridor_id", "CORRIDOR_HAIRPIN_01")

        if is_preempted or is_corridor_locked:
            self._draw_emergency_preemption_banner(
                output_frame,
                emerg_count,
                corridor_lock=is_corridor_locked,
                corridor_id=corridor_id
            )
            # Pulsing amber perimeter border (every 5 frames alternates)
            if (self.frame_idx // 5) % 2 == 0:
                cv2.rectangle(output_frame, (0, 0), (w - 1, h - 1), (0, 160, 255), 6)

        # 7. Flashing Red Alert Perimeter for Left Passing Car Hazard
        if is_left_lane_alert and (self.frame_idx // 4) % 2 == 0:
            cv2.rectangle(output_frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

        return output_frame
