"""
Frame preprocessing module:
Applies contrast enhancement (CLAHE) for mountain fog/glare and optional ROI masking.
"""
import cv2
import numpy as np
from typing import Optional, List, Tuple


class FramePreprocessor:
    def __init__(
        self,
        enable_clahe: bool = True,
        clip_limit: float = 2.5,
        grid_size: Tuple[int, int] = (8, 8),
        roi_polygon: Optional[List[Tuple[int, int]]] = None
    ):
        self.enable_clahe = enable_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self.roi_polygon = roi_polygon
        self._mask = None
        self._mask_shape = None

    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhances frame clarity in challenging mountain conditions (fog, shadow, glare)
        using CLAHE on the Lightness (L) channel in LAB color space.
        """
        if not self.enable_clahe:
            return frame

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = self.clahe.apply(l_channel)
        merged = cv2.merge((cl, a_channel, b_channel))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return enhanced

    def apply_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Masks out non-road areas (forests, sky, valley) if ROI polygon is specified.
        """
        if self.roi_polygon is None or len(self.roi_polygon) < 3:
            return frame

        h, w = frame.shape[:2]
        if self._mask is None or self._mask_shape != (h, w):
            self._mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.roi_polygon, dtype=np.int32)
            cv2.fillPoly(self._mask, [pts], 255)
            self._mask_shape = (h, w)

        masked_frame = cv2.bitwise_and(frame, frame, mask=self._mask)
        return masked_frame

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Runs full preprocessing pipeline."""
        if frame is None:
            return frame

        enhanced = self.enhance_contrast(frame)
        if self.roi_polygon is not None:
            enhanced = self.apply_roi_mask(enhanced)

        return enhanced
