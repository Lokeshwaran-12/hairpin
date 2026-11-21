"""
Video Stream Ingestion supporting Video Files, Cameras, RTSP, and Synthetic Feeds.
"""
import os
import time
import cv2
import numpy as np
from typing import Tuple, Union, Optional
from src.utils.synthetic_feed import HairpinSimulationFeed


class VideoStreamSource:
    """
    Unified Video Capture Interface for real cameras, pre-recorded video files,
    RTSP streams, and procedural ghat road simulations.
    """
    def __init__(self, source: Union[str, int] = "synthetic", width: int = 1280, height: int = 720, fps: int = 30):
        self.source = source
        self.target_width = width
        self.target_height = height
        self.fps = fps
        self.is_synthetic = False
        self.cap = None
        self.sim_feed = None
        self._init_stream()

    def _init_stream(self):
        source_str = str(self.source).strip().lower()
        self.is_image = False
        self.static_image = None

        if source_str == "synthetic" or not self.source:
            self.is_synthetic = True
            self.sim_feed = HairpinSimulationFeed(width=self.target_width, height=self.target_height, fps=self.fps)
        elif source_str.isdigit():
            # Webcam index
            self.is_synthetic = False
            self.cap = cv2.VideoCapture(int(source_str))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        else:
            # File or RTSP stream
            if not source_str.startswith("rtsp://") and not os.path.exists(self.source):
                # Fallback to synthetic if file not found
                print(f"[VideoStreamSource] Warning: Source '{self.source}' not found. Falling back to synthetic simulation.")
                self.is_synthetic = True
                self.sim_feed = HairpinSimulationFeed(width=self.target_width, height=self.target_height, fps=self.fps)
            else:
                self.is_synthetic = False
                ext = os.path.splitext(source_str)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                    self.is_image = True
                    img = cv2.imread(self.source)
                    if img is not None:
                        if img.shape[1] != self.target_width or img.shape[0] != self.target_height:
                            img = cv2.resize(img, (self.target_width, self.target_height))
                        self.static_image = img
                else:
                    self.cap = cv2.VideoCapture(self.source)

        self.is_video_file = isinstance(self.source, str) and os.path.isfile(self.source) and not self.is_image
        if self.is_video_file and self.cap and self.cap.isOpened():
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or float(self.fps)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._playback_start_time = time.time()
        else:
            self.video_fps = float(self.fps)
            self.total_frames = 0
            self._playback_start_time = time.time()

    @property
    def is_opened(self) -> bool:
        if self.is_synthetic:
            return self.sim_feed.is_opened if self.sim_feed else False
        if self.is_image:
            return self.static_image is not None
        return self.cap.isOpened() if self.cap else False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads next frame smoothly without teleportation jumps, ensuring continuous tracking accuracy.
        """
        if self.is_synthetic:
            return self.sim_feed.read()

        if self.is_image:
            if self.static_image is not None:
                return True, self.static_image.copy()
            return False, None

        if not self.cap or not self.cap.isOpened():
            return False, None

        # Gentle normalization for ultra-high FPS video (e.g. 60 FPS -> smooth 30 FPS cadence)
        if self.is_video_file and self.cap:
            if getattr(self, "video_fps", 30.0) >= 50.0:
                self.cap.grab()

        ret, frame = self.cap.read()
        if not ret:
            # End of video file: automatically loop back for continuous monitoring
            if self.is_video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()

        if ret and frame is not None:
            # Ensure target resolution consistency
            if frame.shape[1] != self.target_width or frame.shape[0] != self.target_height:
                frame = cv2.resize(frame, (self.target_width, self.target_height))

        return ret, frame

    def reset(self):
        """Resets stream to beginning."""
        self._playback_start_time = time.time()
        if self.is_synthetic:
            self.sim_feed = HairpinSimulationFeed(width=self.target_width, height=self.target_height, fps=self.fps)
        elif self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        """Releases underlying resources."""
        if self.is_synthetic and self.sim_feed:
            self.sim_feed.release()
        elif self.cap:
            self.cap.release()
