"""
Event Logging & Incident Archival Module:
Maintains SQLite database, CSV log files, and captures incident snapshot images
during Warning (Level 2) and Critical (Level 3) events.
Includes automatic fallback to temp directory if workspace permissions are restricted.
"""
import os
import csv
import time
import sqlite3
import tempfile
import datetime
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from src.tracking.tracker import TrackedVehicle
from src.risk_assessment.collision_predictor import RiskAssessment, RiskLevel


class EventLogger:
    def __init__(
        self,
        db_path: str = "data/logs/hairpin_events.db",
        csv_path: str = "data/logs/traffic_events.csv",
        snapshots_dir: str = "data/snapshots",
        save_snapshots: bool = True,
        snapshot_min_risk: int = 2,
        snapshot_cooldown_seconds: float = 3.0
    ):
        self.save_snapshots = save_snapshots
        self.snapshot_min_risk = snapshot_min_risk
        self.snapshot_cooldown = snapshot_cooldown_seconds
        self.last_snapshot_time = 0.0

        # Attempt primary paths; fallback to temp directory if restricted
        self.db_path = os.path.abspath(db_path)
        self.csv_path = os.path.abspath(csv_path)
        self.snapshots_dir = os.path.abspath(snapshots_dir)

        self._ensure_writable_storage()
        self._init_sqlite()
        self._init_csv()

    def _ensure_writable_storage(self):
        """Checks if configured directories are writable; falls back to temp if not."""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            os.makedirs(self.snapshots_dir, exist_ok=True)
            # Test file write
            test_file = os.path.join(self.snapshots_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
        except (PermissionError, OSError):
            fallback_dir = os.path.join(tempfile.gettempdir(), "hairpin_detect_data")
            fallback_logs = os.path.join(fallback_dir, "logs")
            fallback_snaps = os.path.join(fallback_dir, "snapshots")
            os.makedirs(fallback_logs, exist_ok=True)
            os.makedirs(fallback_snaps, exist_ok=True)

            self.db_path = os.path.join(fallback_logs, "hairpin_events.db")
            self.csv_path = os.path.join(fallback_logs, "traffic_events.csv")
            self.snapshots_dir = fallback_snaps

    def _init_sqlite(self):
        """Creates SQLite tables if they do not exist and applies schema migrations."""
        schema_sql = """
            CREATE TABLE IF NOT EXISTS risk_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_level INTEGER NOT NULL,
                risk_label TEXT NOT NULL,
                min_ttc REAL,
                vehicles_count INTEGER NOT NULL,
                vehicle_classes TEXT,
                vehicle_ids TEXT,
                signal_a_state TEXT,
                signal_b_state TEXT,
                signal_a_message TEXT,
                signal_b_message TEXT,
                hazard_description TEXT,
                snapshot_path TEXT,
                is_emergency INTEGER DEFAULT 0,
                emergency_type TEXT,
                emergency_preempted INTEGER DEFAULT 0
            )
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                # Auto-migrate columns for existing databases
                for col_name, col_def in [
                    ("is_emergency", "INTEGER DEFAULT 0"),
                    ("emergency_type", "TEXT"),
                    ("emergency_preempted", "INTEGER DEFAULT 0")
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE risk_events ADD COLUMN {col_name} {col_def}")
                    except sqlite3.OperationalError:
                        pass
                conn.commit()
        except sqlite3.OperationalError as e:
            # Fallback to in-memory if disk is inaccessible
            print(f"[EventLogger] Warning: Could not open {self.db_path}: {e}. Using in-memory database.")
            self.db_path = ":memory:"
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(schema_sql)
                conn.commit()

    def _init_csv(self):
        """Initializes CSV header if file is missing."""
        try:
            if not os.path.exists(self.csv_path):
                with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "risk_level",
                        "risk_label",
                        "min_ttc",
                        "vehicles_count",
                        "vehicle_classes",
                        "vehicle_ids",
                        "signal_a_state",
                        "signal_b_state",
                        "signal_a_message",
                        "signal_b_message",
                        "hazard_description",
                        "snapshot_path",
                        "is_emergency",
                        "emergency_type",
                        "emergency_preempted"
                    ])
        except (PermissionError, OSError):
            pass

    def log_event(
        self,
        assessment: RiskAssessment,
        vehicles: List[TrackedVehicle],
        frame: Optional[np.ndarray] = None
    ) -> Optional[str]:
        """
        Logs event to SQLite & CSV.
        Saves snapshot image if risk level meets threshold and cooldown has elapsed.
        Returns snapshot_path if saved, else None.
        """
        # Defensive check for accidentally swapped arguments: log_event(frame, vehicles, assessment)
        if isinstance(assessment, np.ndarray) and hasattr(frame, "overall_level"):
            assessment, frame = frame, assessment

        if not hasattr(assessment, "overall_level"):
            return None

        if assessment.overall_level == RiskLevel.SAFE and len(vehicles) == 0:
            return None

        now = time.time()
        now_dt = datetime.datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        level_labels = {
            RiskLevel.SAFE: "SAFE",
            RiskLevel.CAUTION: "CAUTION",
            RiskLevel.WARNING: "WARNING",
            RiskLevel.CRITICAL: "CRITICAL"
        }
        risk_label = level_labels.get(assessment.overall_level, "UNKNOWN")

        v_classes = ",".join([v.class_name for v in vehicles])
        v_ids = ",".join([str(v.track_id) for v in vehicles])
        hazard_desc = "; ".join(assessment.hazard_notes) or "Normal traffic monitoring"

        # Determine emergency status (Phase 3)
        has_emergency_vehicle = any(getattr(v, "is_emergency", False) or v.class_name == "ambulance" for v in vehicles)
        is_preempted = getattr(assessment, "emergency_preemption_active", False)
        is_emergency = 1 if (has_emergency_vehicle or is_preempted) else 0
        emergency_type = None
        for v in vehicles:
            if getattr(v, "is_emergency", False) or v.class_name == "ambulance":
                emergency_type = getattr(v, "emergency_type", None) or "ambulance"
                break
        if not emergency_type and is_preempted:
            emergency_type = "ambulance"
        emergency_preempted = 1 if is_preempted else 0

        # Check snapshot capture (also captures for emergency vehicle / preemption)
        snapshot_path = None
        should_snap = (
            (self.save_snapshots and frame is not None) and
            (assessment.overall_level >= self.snapshot_min_risk or is_emergency == 1)
        )
        if should_snap:
            if (now - self.last_snapshot_time) >= self.snapshot_cooldown:
                self.last_snapshot_time = now
                prefix = "emergency_ambulance" if is_emergency == 1 else "incident"
                filename = f"{prefix}_{now_dt.strftime('%Y%m%d_%H%M%S')}_lvl{int(assessment.overall_level)}.jpg"
                full_path = os.path.join(self.snapshots_dir, filename)
                try:
                    cv2.imwrite(full_path, frame)
                    snapshot_path = full_path
                except Exception:
                    pass

        # Write to SQLite
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO risk_events (
                        timestamp, risk_level, risk_label, min_ttc, vehicles_count,
                        vehicle_classes, vehicle_ids, signal_a_state, signal_b_state,
                        signal_a_message, signal_b_message, hazard_description, snapshot_path,
                        is_emergency, emergency_type, emergency_preempted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp_str,
                    int(assessment.overall_level),
                    risk_label,
                    assessment.min_ttc,
                    len(vehicles),
                    v_classes,
                    v_ids,
                    assessment.signal_a_state,
                    assessment.signal_b_state,
                    assessment.signal_a_message,
                    assessment.signal_b_message,
                    hazard_desc,
                    snapshot_path,
                    is_emergency,
                    emergency_type,
                    emergency_preempted
                ))
                conn.commit()
        except Exception as e:
            print(f"[EventLogger] SQLite write error: {e}")

        # Write to CSV
        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp_str,
                    int(assessment.overall_level),
                    risk_label,
                    assessment.min_ttc if assessment.min_ttc is not None else "",
                    len(vehicles),
                    v_classes,
                    v_ids,
                    assessment.signal_a_state,
                    assessment.signal_b_state,
                    assessment.signal_a_message,
                    assessment.signal_b_message,
                    hazard_desc,
                    snapshot_path or "",
                    is_emergency,
                    emergency_type or "",
                    emergency_preempted
                ])
        except Exception:
            pass

        return snapshot_path

    def get_recent_events(self, limit: int = 50, emergency_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieves recent event records for the dashboard, with optional emergency filter."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                if emergency_only:
                    cursor.execute("""
                        SELECT * FROM risk_events
                        WHERE is_emergency = 1 OR emergency_preempted = 1 OR hazard_description LIKE '%AMBULANCE%'
                        ORDER BY id DESC
                        LIMIT ?
                    """, (limit,))
                else:
                    cursor.execute("""
                        SELECT * FROM risk_events
                        ORDER BY id DESC
                        LIMIT ?
                    """, (limit,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception:
            return []

    def get_emergency_summary(self) -> Dict[str, Any]:
        """Returns statistical counts of emergency incidents and preemptions."""
        stats = {
            "total_emergency_events": 0,
            "ambulance_count": 0,
            "preemptions_triggered": 0
        }
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*),
                        SUM(CASE WHEN emergency_type = 'ambulance' OR vehicle_classes LIKE '%ambulance%' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN emergency_preempted = 1 THEN 1 ELSE 0 END)
                    FROM risk_events
                    WHERE is_emergency = 1 OR emergency_preempted = 1 OR hazard_description LIKE '%AMBULANCE%'
                """)
                row = cursor.fetchone()
                if row and row[0] is not None:
                    stats["total_emergency_events"] = int(row[0] or 0)
                    stats["ambulance_count"] = int(row[1] or 0)
                    stats["preemptions_triggered"] = int(row[2] or 0)
        except Exception:
            pass
        return stats

    def get_risk_summary(self) -> Dict[str, int]:
        """Returns count of events per risk level."""
        summary = {"SAFE": 0, "CAUTION": 0, "WARNING": 0, "CRITICAL": 0}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT risk_label, COUNT(*) FROM risk_events
                    GROUP BY risk_label
                """)
                for label, count in cursor.fetchall():
                    if label in summary:
                        summary[label] = count
        except Exception:
            pass
        return summary
