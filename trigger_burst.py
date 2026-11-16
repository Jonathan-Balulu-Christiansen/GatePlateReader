"""Motion trigger on a vertical line of a fixed camera, then a burst of frames; the sharpest is kept."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TriggerCfg:
    line_x: float = 0.80          # trigger line, fraction of the frame width
    strip_w: int = 30             # width of the motion strip around the line (px)
    strip_y0: float = 0.40        # the strip starts at this fraction of the frame height
    diff_thresh: int = 30         # |gray - background| that counts as motion
    min_motion: float = 0.03      # floor of the trigger threshold (share of strip pixels)
    trigger_factor: float = 1.80  # fire when motion > baseline * factor ...
    trigger_delta: float = 0.010  # ... and > baseline + delta
    warmup: int = 20              # frames that only learn the baseline
    debounce: int = 3             # consecutive frames above the threshold (rejects one-frame noise at any fps)
    # Tuned at 120 fps; the burst window and the EMA rates are converted to keep the same duration at any fps.
    ref_fps: float = 120.0
    burst: int = 10               # frames captured after the trigger (83 ms at 120 fps)
    bg_alpha: float = 0.10        # background EMA rate per frame
    baseline_alpha: float = 0.10  # baseline EMA rate per frame (while idle)


def sharpness(frame_bgr: np.ndarray) -> float:
    """Variance of the Laplacian: high for a sharp frame, low for a blurred one."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class MotionTrigger:
    """Background-subtracted motion in a thin strip around the trigger line, against an adaptive baseline."""

    def __init__(self, fps: float, cfg: TriggerCfg = TriggerCfg()):
        s = max(1, round(fps)) / cfg.ref_fps  # nominal fps: a 119.99 fps file is a 120 fps camera
        self.cfg, self.burst = cfg, max(1, int(cfg.burst * s + 0.5))
        self.bg_a = 1 - (1 - cfg.bg_alpha) ** (1 / s)
        self.bl_a = 1 - (1 - cfg.baseline_alpha) ** (1 / s)
        self.bg, self.baseline, self.seen, self.hits = None, 0.0, 0, 0

    def strip(self, shape) -> tuple[int, int, int, int]:
        """(x0, y0, x1, line_x) of the motion strip in a frame of this shape."""
        h, w = shape[:2]
        lx = int(round(self.cfg.line_x * w))
        return lx - self.cfg.strip_w // 2, int(round(self.cfg.strip_y0 * h)), lx + self.cfg.strip_w // 2, lx

    def update(self, frame_bgr: np.ndarray) -> bool:
        """Feed one frame; True when the trigger fires."""
        x0, y0, x1, _ = self.strip(frame_bgr.shape)
        roi = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)[y0:, x0:x1].astype(np.float32)
        if self.bg is None:
            self.bg, self.seen = roi, 1
            return False
        ratio = float(np.count_nonzero(np.abs(roi - self.bg) > self.cfg.diff_thresh)) / roi.size
        c = self.cfg
        if self.seen < c.warmup:
            self.baseline = (self.baseline * self.seen + ratio) / (self.seen + 1)
        elif ratio < max(c.min_motion, self.baseline * c.trigger_factor, self.baseline + c.trigger_delta):
            self.baseline = (1 - self.bl_a) * self.baseline + self.bl_a * ratio
            self.hits = 0
        else:
            self.hits += 1
            if self.hits >= c.debounce:
                return True
        self.bg = (1 - self.bg_a) * self.bg + self.bg_a * roi
        self.seen += 1
        return False


def sharpest_of_burst(cap: cv2.VideoCapture, first: np.ndarray, n: int) -> tuple[np.ndarray, int, list[float]]:
    """Read n - 1 more frames after `first`; return (sharpest frame, its index, all scores)."""
    frames = [first]
    while len(frames) < n:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    scores = [sharpness(f) for f in frames]
    i = int(np.argmax(scores))
    return frames[i], i, scores
