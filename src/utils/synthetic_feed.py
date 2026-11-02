"""
Synthetic Hairpin Bend Traffic Generator
Procedurally renders a mountainous hairpin road with blind apex, mountain obstruction,
and vehicles (cars, buses, trucks, motorcycles) driving in opposing directions.
"""
import math
import random
import cv2
import numpy as np


class SyntheticVehicle:
    def __init__(self, v_type: str, direction: str, speed_factor: float = 1.0, start_offset: float = 0.0):
        self.v_type = v_type  # "car", "bus", "truck", "motorcycle"
        self.direction = direction  # "uphill" (Arm A -> Apex -> Arm B) or "downhill" (Arm B -> Apex -> Arm A)
        self.progress = start_offset  # 0.0 to 1.0 along the trajectory
        self.speed = (0.003 + random.uniform(-0.0005, 0.0008)) * speed_factor

        # Dimension specifications (width, height in px at standard 1280x720)
        if v_type == "car":
            self.length = 64
            self.width = 34
            self.color = random.choice([(220, 50, 50), (230, 230, 230), (50, 180, 50), (30, 120, 220)])
        elif v_type == "bus":
            self.length = 110
            self.width = 44
            self.color = (30, 160, 240)  # Bright yellow/orange bus
        elif v_type == "truck":
            self.length = 120
            self.width = 46
            self.color = (40, 90, 180)  # Heavy freight truck
        else:  # motorcycle
            self.length = 36
            self.width = 18
            self.color = (20, 20, 20)

    def update(self):
        self.progress += self.speed
        return self.progress <= 1.0


class HairpinSimulationFeed:
    """
    Simulates a live camera feed capturing a ghat hairpin bend.
    Provides standard OpenCV VideoCapture-like .read() and .release() interface.
    """
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = 0
        self.is_opened = True

        # Precompute background image for rendering efficiency
        self.background = self._create_ghat_background()

        # Road spline trajectories (centerlines for uphill & downhill lanes)
        self.uphill_path = self._generate_lane_path(offset_factor=14.0)
        self.downhill_path = self._generate_lane_path(offset_factor=-14.0)

        # Active vehicles
        self.vehicles = []
        self._init_scenario_timeline()

    def _generate_lane_path(self, offset_factor: float = 0.0):
        """
        Creates smooth spline points for hairpin curve from Arm A (bottom-left)
        to Apex (top-center) to Arm B (bottom-right).
        """
        points = []
        num_steps = 400
        for i in range(num_steps + 1):
            t = i / float(num_steps)
            # Parametric horseshoe / hairpin curve
            # t: 0.0 -> bottom-left, 0.5 -> top apex, 1.0 -> bottom-right
            angle = math.pi * (1.0 - t)
            rx = self.width * 0.34
            ry = self.height * 0.40

            center_x = self.width * 0.50
            center_y = self.height * 0.48

            # Stretch the arms downward towards bottom corners
            base_x = center_x + rx * math.cos(angle)
            base_y = center_y - ry * math.sin(angle)

            if t < 0.25:
                # Arm A descending
                arm_factor = (0.25 - t) / 0.25
                base_x -= arm_factor * 120
                base_y += arm_factor * 260
            elif t > 0.75:
                # Arm B descending
                arm_factor = (t - 0.75) / 0.25
                base_x += arm_factor * 120
                base_y += arm_factor * 260

            # Normal offset for dual-lane separation
            # Calculate tangent
            if i > 0:
                prev_x, prev_y = points[-1][0], points[-1][1]
                dx = base_x - prev_x
                dy = base_y - prev_y
                length = math.hypot(dx, dy) or 1.0
                nx = -dy / length
                ny = dx / length
                cur_x = base_x + nx * offset_factor
                cur_y = base_y + ny * offset_factor
            else:
                cur_x = base_x
                cur_y = base_y

            points.append((float(cur_x), float(cur_y)))
        return points

    def _create_ghat_background(self) -> np.ndarray:
        """Draws realistic ghat road landscape with mountains, road, and guard rails."""
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Forest/mountain gradient background
        for y in range(self.height):
            ratio = y / self.height
            # Forest green gradient into earthy brown
            r = int(25 + 40 * ratio)
            g = int(60 + 35 * ratio)
            b = int(25 + 20 * ratio)
            bg[y, :] = (b, g, r)

        # Draw mountain texture contours
        np.random.seed(42)
        for _ in range(120):
            cx = np.random.randint(0, self.width)
            cy = np.random.randint(0, self.height)
            radius = np.random.randint(15, 60)
            c = (int(np.random.randint(20, 50)), int(np.random.randint(60, 110)), int(np.random.randint(20, 45)))
            cv2.circle(bg, (cx, cy), radius, c, -1)

        # Central mountain blind ridge (the obstruction preventing view between Arm A & Arm B)
        ridge_pts = np.array([
            [int(self.width * 0.40), int(self.height * 0.46)],
            [int(self.width * 0.50), int(self.height * 0.22)],
            [int(self.width * 0.60), int(self.height * 0.46)],
            [int(self.width * 0.63), int(self.height * 0.98)],
            [int(self.width * 0.37), int(self.height * 0.98)]
        ], np.int32)
        # Rocky cliff fill
        cv2.fillPoly(bg, [ridge_pts], (45, 65, 80))
        cv2.polylines(bg, [ridge_pts], isClosed=True, color=(30, 45, 60), thickness=4)

        # Add rocky cliff texturing on the central obstruction
        for _ in range(80):
            rx = np.random.randint(int(self.width * 0.40), int(self.width * 0.60))
            ry = np.random.randint(int(self.height * 0.30), int(self.height * 0.95))
            cv2.line(bg, (rx, ry), (rx + np.random.randint(-15, 15), ry + np.random.randint(5, 20)), (35, 50, 65), 2)

        # Draw road base (dark asphalt)
        # Generate full road contour
        left_shoulder = self._generate_lane_path(offset_factor=-36.0)
        right_shoulder = self._generate_lane_path(offset_factor=36.0)

        road_poly = np.array(left_shoulder + right_shoulder[::-1], dtype=np.int32)
        cv2.fillPoly(bg, [road_poly], (52, 54, 58))  # Dark asphalt grey

        # Road edges / white shoulder lines
        cv2.polylines(bg, [np.array(left_shoulder, dtype=np.int32)], isClosed=False, color=(240, 240, 240), thickness=3)
        cv2.polylines(bg, [np.array(right_shoulder, dtype=np.int32)], isClosed=False, color=(240, 240, 240), thickness=3)

        # Center line (dashed yellow)
        center_line = self._generate_lane_path(offset_factor=0.0)
        for i in range(0, len(center_line) - 4, 8):
            p1 = (int(center_line[i][0]), int(center_line[i][1]))
            p2 = (int(center_line[i + 4][0]), int(center_line[i + 4][1]))
            cv2.line(bg, p1, p2, (30, 215, 255), 2)  # Yellow dashed line

        # Add reflective chevrons / arrows on the apex outer barrier
        for t_val in [0.35, 0.42, 0.50, 0.58, 0.65]:
            idx = int(t_val * len(right_shoulder))
            pos = right_shoulder[idx]
            px, py = int(pos[0]), int(pos[1])
            # Draw yellow-black chevron sign
            cv2.rectangle(bg, (px - 10, py - 14), (px + 10, py + 14), (20, 220, 240), -1)
            cv2.rectangle(bg, (px - 10, py - 14), (px + 10, py + 14), (0, 0, 0), 2)

        # Hairpin road warning signs on approaches
        cv2.putText(bg, "HAIRPIN BEND - NARROW BLIND APEX", (int(self.width * 0.28), 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        return bg

    def _init_scenario_timeline(self):
        """Initializes diverse scripted traffic sequences."""
        self.timeline_events = [
            # frame, type, direction, speed_factor, start_offset
            (0, "car", "uphill", 0.9, 0.05),
            (60, "motorcycle", "downhill", 1.1, 0.05),
            (160, "bus", "uphill", 0.8, 0.02),
            (210, "car", "downhill", 1.0, 0.02),      # Will create near-apex conflict!
            (380, "truck", "downhill", 0.75, 0.02),
            (420, "car", "uphill", 0.95, 0.02),       # High risk conflict with heavy truck!
            (600, "motorcycle", "uphill", 1.2, 0.05),
            (650, "car", "downhill", 0.9, 0.05)
        ]

    def _get_path_coordinate(self, path, progress: float):
        """Interpolates (x, y, angle_rad) for a vehicle along path."""
        progress = max(0.0, min(1.0, progress))
        total_pts = len(path) - 1
        idx_f = progress * total_pts
        idx = int(idx_f)
        frac = idx_f - idx

        p1 = path[idx]
        p2 = path[min(idx + 1, total_pts)]

        x = p1[0] + (p2[0] - p1[0]) * frac
        y = p1[1] + (p2[1] - p1[1]) * frac

        # Heading angle
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.atan2(dy, dx)
        return x, y, angle

    def _draw_vehicle(self, frame: np.ndarray, vehicle: SyntheticVehicle):
        """Renders vehicle with realistic contours and details."""
        if vehicle.direction == "uphill":
            path = self.uphill_path
            prog = vehicle.progress
            heading_mult = 1.0
        else:
            path = self.downhill_path
            # Downhill enters from Arm B (progress 1.0 down to 0.0)
            prog = 1.0 - vehicle.progress
            heading_mult = -1.0

        x, y, angle = self._get_path_coordinate(path, prog)
        if vehicle.direction == "downhill":
            angle += math.pi  # Point in reverse travel direction

        # Vehicle center point
        cx, cy = int(x), int(y)
        w, l = vehicle.width, vehicle.length

        # Rotated bounding rectangle coordinates
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        half_l = l / 2.0
        half_w = w / 2.0

        corners = [
            (-half_l, -half_w),
            (half_l, -half_w),
            (half_l, half_w),
            (-half_l, half_w)
        ]

        rot_corners = []
        for (dx, dy) in corners:
            rx = cx + dx * cos_a - dy * sin_a
            ry = cy + dx * sin_a + dy * cos_a
            rot_corners.append([int(rx), int(ry)])

        pts = np.array(rot_corners, np.int32)

        # Vehicle shadow
        shadow_pts = pts + np.array([4, 6])
        cv2.fillPoly(frame, [shadow_pts], (30, 30, 30))

        # Main vehicle body
        cv2.fillPoly(frame, [pts], vehicle.color)
        cv2.polylines(frame, [pts], isClosed=True, color=(15, 15, 15), thickness=2)

        # Windows / Windshield
        front_dx = half_l * 0.4
        front_x = int(cx + front_dx * cos_a)
        front_y = int(cy + front_dx * sin_a)
        glass_w = max(4, int(w * 0.65))
        cv2.circle(frame, (front_x, front_y), glass_w // 2, (70, 70, 80), -1)

        # Headlights
        head_l = half_l * 0.85
        head_w = half_w * 0.65
        h1_x = int(cx + head_l * cos_a - head_w * sin_a)
        h1_y = int(cy + head_l * sin_a + head_w * cos_a)
        h2_x = int(cx + head_l * cos_a + head_w * sin_a)
        h2_y = int(cy + head_l * sin_a - head_w * cos_a)

        cv2.circle(frame, (h1_x, h1_y), 3, (240, 255, 255), -1)
        cv2.circle(frame, (h2_x, h2_y), 3, (240, 255, 255), -1)

        # Taillights (red)
        tail_l = -half_l * 0.85
        t1_x = int(cx + tail_l * cos_a - head_w * sin_a)
        t1_y = int(cy + tail_l * sin_a + head_w * cos_a)
        t2_x = int(cx + tail_l * cos_a + head_w * sin_a)
        t2_y = int(cy + tail_l * sin_a - head_w * cos_a)

        cv2.circle(frame, (t1_x, t1_y), 3, (20, 20, 220), -1)
        cv2.circle(frame, (t2_x, t2_y), 3, (20, 20, 220), -1)

    def read(self):
        """Standard capture read API. Returns (ret, frame)."""
        if not self.is_opened:
            return False, None

        frame = self.background.copy()

        # Spawn new vehicles according to timeline loop
        cycle_frame = self.frame_count % 800
        for (spawn_frame, v_type, direction, spd, start_offset) in self.timeline_events:
            if cycle_frame == spawn_frame:
                self.vehicles.append(SyntheticVehicle(v_type, direction, spd, start_offset))

        # Update and draw vehicles
        remaining_vehicles = []
        for v in self.vehicles:
            is_active = v.update()
            if is_active:
                self._draw_vehicle(frame, v)
                remaining_vehicles.append(v)
        self.vehicles = remaining_vehicles

        self.frame_count += 1
        return True, frame

    def release(self):
        self.is_opened = False


def generate_demo_video(output_path: str, num_frames: int = 700, fps: int = 30):
    """Generates an MP4 demo file containing a full simulation cycle."""
    generator = HairpinSimulationFeed(width=1280, height=720, fps=fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (1280, 720))

    for _ in range(num_frames):
        ret, frame = generator.read()
        if not ret:
            break
        writer.write(frame)

    writer.release()
    generator.release()
    return output_path
