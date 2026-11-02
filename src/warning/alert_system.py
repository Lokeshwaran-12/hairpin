"""
Alert System:
Manages audio buzzer triggers, hardware relay / GPIO hooks,
and signal light state orchestration.
"""
import time
import threading
from typing import Optional, Callable
from src.risk_assessment.collision_predictor import RiskLevel, RiskAssessment

# Windows audio alert support
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False


class AlertSystem:
    def __init__(
        self,
        enable_audio: bool = True,
        buzzer_freq_hz: int = 1200,
        buzzer_duration_ms: int = 180,
        cooldown_seconds: float = 1.5,
        hardware_callback: Optional[Callable[[RiskAssessment], None]] = None
    ):
        self.enable_audio = enable_audio
        self.buzzer_freq_hz = buzzer_freq_hz
        self.buzzer_duration_ms = buzzer_duration_ms
        self.cooldown_seconds = cooldown_seconds
        self.hardware_callback = hardware_callback

        self.last_beep_time = 0.0
        self._lock = threading.Lock()

    def _play_buzzer_async(self, freq: int, duration: int):
        """Plays buzzer tone asynchronously without blocking main video loop."""
        if not HAS_WINSOUND:
            return

        def _beep():
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass

        threading.Thread(target=_beep, daemon=True).start()

    def trigger(self, assessment: RiskAssessment):
        """
        Evaluates risk assessment and triggers audio alarms / hardware signals.
        """
        now = time.time()

        # Trigger hardware callback if registered (e.g., Arduino relay / GPIO)
        if self.hardware_callback:
            try:
                self.hardware_callback(assessment)
            except Exception as e:
                print(f"[AlertSystem] Hardware callback error: {e}")

        # Check audio trigger
        if self.enable_audio and assessment.overall_level >= RiskLevel.WARNING:
            with self._lock:
                if (now - self.last_beep_time) >= self.cooldown_seconds:
                    self.last_beep_time = now

                    # Critical risk emits higher frequency / longer pulse
                    if assessment.overall_level == RiskLevel.CRITICAL:
                        self._play_buzzer_async(freq=1600, duration=280)
                    else:
                        self._play_buzzer_async(freq=self.buzzer_freq_hz, duration=self.buzzer_duration_ms)
