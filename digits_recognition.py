"""Digit reader for an Israeli plate: alignment -> binarization -> digit boxes -> hole-constrained template matching."""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]


@dataclass(frozen=True)  # frozen: a default Cfg() is shared between calls, so it must not change
class Cfg:
    char_h: int = 60
    char_w: int = 40
    # Blue strip (IL badge): cropped away when it sits on the left or at the bottom
    blue_hsv_lower: tuple = (90, 60, 40)
    blue_hsv_upper: tuple = (140, 255, 255)
    blue_left_region: float = 0.25
    blue_left_share: float = 0.70
    blue_col_thresh: float = 0.20
    blue_bottom_region: float = 0.30
    blue_bottom_share: float = 0.55
    blue_row_thresh: float = 0.10
    blue_margin: float = 0.02
    # Preprocessing
    clahe_clip: float = 1.0
    clahe_tile: tuple = (8, 8)
    gaussian_ksize: int = 3
    yellow_hsv_lower: tuple = (15, 60, 60)
    yellow_hsv_upper: tuple = (40, 255, 255)
    yellow_clean_strength: float = 0.6
    dark_brightness: float = 110.0
    dark_clahe_clip: float = 2.5
    # Binarization candidates
    blackhat_kernel: int = 17
    blackhat_alphas: tuple = (1.0, 0.85, 0.70, 0.55, 0.40)
    open_kernel: tuple = (3, 3)
    close_kernel: tuple = (1, 3)
    small_cc_ratio: float = 0.004
    clear_border_px: int = 5
    # Digit boxes
    min_h: float = 0.40
    max_h: float = 0.98
    min_w: float = 0.010
    max_w: float = 0.35
    min_area: float = 0.0015
    ar_min: float = 0.12
    ar_max: float = 1.25
    dash_h: float = 0.55
    dash_ar: float = 1.80
    pad: float = 0.12
    wide_ar: float = 0.80
    wide_min_w: float = 0.08
    lengths: tuple = (7, 8)
    # Character cleanup: opening breaks thin bridges; retry larger openings for weak matches
    keep_cc_secondary: float = 0.45
    open_size: int = 3
    min_sub_ratio: float = 0.15
    retry_sizes: tuple = (5, 7, 9)
    retry_below: float = 0.42
    retry_gain: float = 0.02
    # Holes: 2 -> 8/0, 1 high -> 9, 1 low -> 6, 1 centered -> 0, none -> no 8 (and 0/6/9 only if clearly better)
    hole_top: float = 0.40
    hole_center_low: float = 0.43
    hole_center_high: float = 0.57
    hole_bottom: float = 0.60
    no_hole_tolerance: float = 0.05
    # Plate alignment for loose crops (e.g. detector boxes from street video)
    rectify: bool = True             # warp the yellow plate region (minAreaRect) to 400x80
    rectify_left_ext: float = 0.12   # keep the blue strip, which lies outside the yellow region
    deslant: bool = True             # undo the italic shear of plates seen from the side
    max_shear: float = 0.5           # searched in steps of 0.05


@dataclass(frozen=True)
class DigitsResult:
    digits: str
    confs: list = field(default_factory=list)
    binary: np.ndarray | None = None
    boxes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.digits) in (7, 8)

    @property
    def conf(self) -> float:
        return float(np.mean(self.confs)) if self.confs else 0.0


# ── Plate alignment ──────────────────────────────────────────────────

W, H = 400, 80


def rectify(img: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Warp the largest yellow region (minAreaRect, widened left for the blue strip) to 400x80; else resize."""
    m = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), (12, 70, 60), (40, 255, 255))
    cs = cv2.findContours(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)), cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)[0]
    c = max(cs, key=cv2.contourArea) if cs else None
    if not cfg.rectify or c is None or cv2.contourArea(c) < 0.35 * img.shape[0] * img.shape[1]:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_CUBIC)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw < rh:
        rw, rh, ang = rh, rw, ang - 90
    box = cv2.boxPoints(((cx, cy), (rw, rh), ang))
    box = box[np.argsort(box[:, 0])]
    (tl, bl), (tr, br) = sorted(box[:2], key=lambda p: p[1]), sorted(box[2:], key=lambda p: p[1])
    ext = (tr - tl) * cfg.rectify_left_ext  # may reach past a tight crop's edge: the replicated border helps there
    src = np.float32([tl - ext, tr, br, bl - ext])
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def deslant(plate: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Shear the plate so the digit strokes stand upright: the shear whose column profile is sharpest."""
    h = plate.shape[0]
    b = cv2.threshold(cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    def shear(im, s, **kw):
        return cv2.warpAffine(im, np.float32([[1, s, -s * h / 2], [0, 1, 0]]), im.shape[1::-1], **kw)

    steps = np.linspace(-cfg.max_shear, cfg.max_shear, int(round(2 * cfg.max_shear / 0.05)) + 1)
    s = max(steps, key=lambda s: float((shear(b, s, flags=cv2.INTER_NEAREST).sum(0).astype(np.float64) ** 2).sum()))
    return shear(plate, s, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def align(img: np.ndarray, cfg: Cfg, warped: bool = False) -> np.ndarray:
    """Any plate image -> an upright 400x80 plate. A perspective warp from the detector only needs the shear."""
    plate = img if warped else rectify(img, cfg)
    return deslant(plate, cfg) if cfg.deslant else plate


# ── Plate preprocessing ──────────────────────────────────────────────

def crop_blue_strip(plate_bgr: np.ndarray, cfg: Cfg) -> np.ndarray:
    hsv = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.inRange(hsv, np.array(cfg.blue_hsv_lower, np.uint8), np.array(cfg.blue_hsv_upper, np.uint8))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, se), cv2.MORPH_CLOSE, se) > 0
    h, w = mask.shape
    total = int(mask.sum())
    if total == 0:
        return plate_bgr
    left_w, bottom_h = max(1, int(round(cfg.blue_left_region * w))), max(1, int(round(cfg.blue_bottom_region * h)))
    if mask[:, :left_w].sum() / total >= cfg.blue_left_share:
        xs = np.where(mask.sum(0)[:left_w] > cfg.blue_col_thresh * h)[0]
        if xs.size:
            return plate_bgr[:, min(w - 1, int(xs.max()) + int(cfg.blue_margin * w)):]
    if mask[h - bottom_h:].sum() / total >= cfg.blue_bottom_share:
        ys = np.where(mask.sum(1)[h - bottom_h:] > cfg.blue_row_thresh * w)[0]
        if ys.size:
            return plate_bgr[:max(1, int(ys.min()) + h - bottom_h - int(cfg.blue_margin * h)), :]
    return plate_bgr


def preprocess(plate_bgr: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Gray, yellow background pulled to its mean, dark plates illumination-normalized, adaptive CLAHE, blur."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    ym = cv2.inRange(cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2HSV), np.array(cfg.yellow_hsv_lower, np.uint8),
                     np.array(cfg.yellow_hsv_upper, np.uint8))
    if np.count_nonzero(ym) / ym.size >= 0.05:
        m = cv2.erode(ym, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))) > 0
        mean_v, s = float(np.mean(gray[m])) if np.any(m) else 128.0, cfg.yellow_clean_strength
        gray = gray.copy()
        gray[m] = np.clip(gray[m].astype(np.float32) * (1 - s) + mean_v * s, 0, 255).astype(np.uint8)
    w = gray.shape[1]
    if w > 30 and brightness < 60.0:
        bg = cv2.GaussianBlur(gray, (w | 1, 1), 0).astype(np.float32) + 1.0
        gray = np.clip(gray.astype(np.float32) / bg * 128, 0, 255).astype(np.uint8)
    clip = cfg.dark_clahe_clip if brightness < cfg.dark_brightness else cfg.clahe_clip
    pre = cv2.createCLAHE(clipLimit=clip, tileGridSize=cfg.clahe_tile).apply(gray)
    k = cfg.gaussian_ksize | 1
    return cv2.GaussianBlur(pre, (k, k), 0) if k >= 3 else pre


# ── Binarization: several candidates, the one with the most plate-like boxes wins ──

def _remove_small_cc(b: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _clean(b: np.ndarray, cfg: Cfg) -> np.ndarray:
    opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, cfg.open_kernel))
    opened = _remove_small_cc(opened, int(cfg.small_cc_ratio * b.shape[0] * b.shape[1]))
    out = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, cfg.close_kernel))
    p = cfg.clear_border_px
    out[:p], out[-p:], out[:, :p], out[:, -p:] = 0, 0, 0, 0
    return out


def binarization_candidates(pre: np.ndarray, cfg: Cfg) -> list[tuple[str, np.ndarray]]:
    """Inverse Otsu plus black-hat at several thresholds, each cleaned the same way."""
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.blackhat_kernel,) * 2)
    bh = cv2.subtract(cv2.morphologyEx(pre, cv2.MORPH_CLOSE, se), pre)
    t = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    cands = [("inv_otsu", cv2.threshold(pre, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])]
    cands += [(f"blackhat_{a:.2f}", cv2.threshold(bh, int(max(1, min(255, round(a * t)))), 255, cv2.THRESH_BINARY)[1])
              for a in cfg.blackhat_alphas]
    return [(name, _clean(b, cfg)) for name, b in cands]


def boxes_quality(boxes: list[Box], h_img: int) -> float:
    """Lower is better: digits share a top line and a height, are tall, and there are 7 or 8 of them."""
    if not boxes:
        return 1e9
    ys = np.array([b[1] for b in boxes], np.float32) / h_img
    hs = np.array([b[3] for b in boxes], np.float32) / h_img
    n = len(boxes)
    count_pen = 0.0 if n in (7, 8) else 0.6 if n == 6 else 0.4 if n == 9 else 1.5
    return 10 * float(np.std(ys)) + 6 * float(np.std(hs)) + 4 * max(0, 0.65 - float(np.median(hs))) + 3 * count_pen


def best_binarization(pre: np.ndarray, cfg: Cfg) -> tuple[str, np.ndarray]:
    return min(binarization_candidates(pre, cfg), key=lambda c: boxes_quality(extract_boxes(c[1], cfg), pre.shape[0]))


# ── Digit boxes ──────────────────────────────────────────────────────

def extract_boxes(b: np.ndarray, cfg: Cfg) -> list[Box]:
    """Character-shaped external contours, left to right."""
    hi, wi = b.shape
    boxes = []
    for cnt in cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(cnt)
        ar = w / max(h, 1)
        if (w * h < cfg.min_area * hi * wi or not cfg.min_h * hi <= h <= cfg.max_h * hi
                or not cfg.min_w * wi <= w <= cfg.max_w * wi or not cfg.ar_min <= ar <= cfg.ar_max
                or (h < cfg.dash_h * hi and ar > cfg.dash_ar)):
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda bx: bx[0])


def split_wide_boxes(b: np.ndarray, boxes: list[Box], cfg: Cfg) -> list[Box]:
    """Split two touching digits at the weakest column of the middle half."""
    out = []
    for x, y, w, h in boxes:
        if w / max(h, 1) > cfg.wide_ar and w >= cfg.wide_min_w * b.shape[1] * 2:
            proj = b[y:y + h, x:x + w].sum(0).astype(np.float32)
            q = max(1, w // 4)
            if proj.max() >= 1 and len(proj[q:w - q]) >= 3:
                sx = int(np.argmin(proj[q:w - q])) + q
                if proj[sx] <= 0.5 * proj.max() and sx >= 4 and w - sx >= 4:
                    out += [(x, y, sx, h), (x + sx, y, w - sx, h)]
                    continue
        out.append((x, y, w, h))
    return sorted(out, key=lambda bx: bx[0])


def digit_boxes(plate_bgr: np.ndarray, cfg: Cfg) -> tuple[np.ndarray, list[Box]]:
    """(binary plate without the blue strip, digit boxes)."""
    plate = crop_blue_strip(plate_bgr, cfg)
    _, b = best_binarization(preprocess(plate, cfg), cfg)
    return b, split_wide_boxes(b, extract_boxes(b, cfg), cfg)


# ── Characters ───────────────────────────────────────────────────────

def _keep_main_cc(b: np.ndarray, ratio: float) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((b > 0).astype(np.uint8), 8)
    if n <= 1:
        return b
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    keep = labels == order[0] + 1
    if len(order) >= 2 and areas[order[0]] > 0 and areas[order[1]] / areas[order[0]] >= ratio:
        keep |= labels == order[1] + 1
    return np.where(keep, 255, 0).astype(b.dtype)


def _open_and_clean(b: np.ndarray, k: int, min_sub_ratio: float) -> np.ndarray:
    opened = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if cv2.countNonZero(opened) == 0:
        return b
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
    if n <= 2:
        return opened
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1, int(min_sub_ratio * stats[1:, cv2.CC_STAT_AREA].max()))
    return np.where(keep[labels], 255, 0).astype(opened.dtype)


def normalize_char(b: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Tight crop, scaled to fit out_h x out_w with a 4 px margin, centered."""
    ys, xs = np.where(b > 0)
    canvas = np.zeros((out_h, out_w), np.uint8)
    if xs.size == 0:
        return canvas
    crop = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    s = min((out_w - 8) / cw, (out_h - 8) / ch)
    nw, nh = max(1, int(round(cw * s))), max(1, int(round(ch * s)))
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    return canvas


def char_image(b: np.ndarray, box: Box, k: int, cfg: Cfg) -> np.ndarray:
    """One normalized character: padded box, opening of size k, largest components."""
    x, y, w, h = box
    p = int(round(cfg.pad * h))
    roi = b[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
    return normalize_char(_keep_main_cc(_open_and_clean(roi, k, cfg.min_sub_ratio), cfg.keep_cc_secondary),
                          cfg.char_h, cfg.char_w)


# ── Template matching ────────────────────────────────────────────────

def load_templates(folder: Path, cfg: Cfg = Cfg()) -> dict[str, np.ndarray]:
    tpl = {}
    for d in "0123456789":
        im = cv2.imread(str(Path(folder) / f"{d}.png"), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(f"missing template {Path(folder) / f'{d}.png'} (run build_templates.py)")
        if im.shape != (cfg.char_h, cfg.char_w):
            im = cv2.resize(im, (cfg.char_w, cfg.char_h), interpolation=cv2.INTER_NEAREST)
        tpl[d] = im
    return tpl


def match_templates(ch: np.ndarray, templates: dict, allowed: str = "0123456789") -> tuple[str, float]:
    """Best of 0.7 * NCC + 0.3 * IoU over the union of both shapes; confidence mapped to [0, 1]."""
    a = ch > 0
    best, best_s = "?", -1e9
    for d in allowed:
        t = templates[d]
        mask = a | (t > 0)
        if mask.sum() < 50:
            continue
        c, tt = ch[mask].astype(np.float32) / 255.0, t[mask].astype(np.float32) / 255.0
        c0, t0 = c - c.mean(), tt - tt.mean()
        ncc = float(np.sum(c0 * t0) / (np.sqrt(np.sum(c0 ** 2) * np.sum(t0 ** 2)) + 1e-9))
        iou = float(np.logical_and(a, t > 0).sum()) / (float(mask.sum()) + 1e-9)
        s = 0.70 * ncc + 0.30 * iou
        if s > best_s:
            best_s, best = s, d
    return best, max(0.0, min(1.0, 0.5 * (best_s + 1)))


def count_holes(ch: np.ndarray, min_area_ratio: float = 0.01) -> tuple[int, float | None]:
    """Enclosed background regions: (count, mean normalized y of their centers)."""
    h, w = ch.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((ch == 0).astype(np.uint8), 8)
    border = set(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]).tolist())
    ys = [centroids[i][1] / h for i in range(1, n)
          if i not in border and stats[i, cv2.CC_STAT_AREA] >= round(min_area_ratio * h * w)]
    return (len(ys), float(np.mean(ys))) if ys else (0, None)


def match_with_holes(ch: np.ndarray, templates: dict, cfg: Cfg) -> tuple[str, float]:
    holes, hole_y = count_holes(ch)
    if holes >= 2:
        return match_templates(ch, templates, "80")
    if holes == 1:
        allowed = ("9" if hole_y < cfg.hole_top else "90" if hole_y < cfg.hole_center_low else
                   "6" if hole_y > cfg.hole_bottom else "60" if hole_y > cfg.hole_center_high else "0")
        return match_templates(ch, templates, allowed)
    d_nh, c_nh = match_templates(ch, templates, "123457")
    d_all, c_all = match_templates(ch, templates, "012345679")
    return (d_nh, c_nh) if c_nh >= c_all - cfg.no_hole_tolerance else (d_all, c_all)


# ── Reader ───────────────────────────────────────────────────────────

def recognize(plate_bgr: np.ndarray, templates: dict, cfg: Cfg = Cfg()) -> DigitsResult:
    """Read an aligned plate (400x80, blue strip on the left or at the bottom)."""
    b, boxes = digit_boxes(plate_bgr, cfg)
    text, confs = [], []
    for box in boxes:
        d, c = match_with_holes(char_image(b, box, cfg.open_size, cfg), templates, cfg)
        for k in cfg.retry_sizes if c < cfg.retry_below else ():  # a weak match may be two digits joined by a bridge
            d2, c2 = match_with_holes(char_image(b, box, k, cfg), templates, cfg)
            if c2 > c + cfg.retry_gain:
                d, c = d2, c2
        text.append(d)
        confs.append(c)
    return DigitsResult("".join(t for t in text if t.isdigit()), confs, b, boxes)


def read_plate(img: np.ndarray, templates: dict, cfg: Cfg = Cfg(), warped: bool = False) -> DigitsResult:
    """Read a plate image: a loose or hand-made crop, or (warped=True) the detector's 400x80 perspective warp."""
    return recognize(align(img, cfg, warped), templates, cfg)


def digit_chars(img: np.ndarray, cfg: Cfg = Cfg()) -> list[np.ndarray]:
    """The normalized characters of a plate, as the reader sees them (for building templates)."""
    b, boxes = digit_boxes(align(img, cfg), cfg)
    return [char_image(b, box, cfg.open_size, cfg) for box in boxes]
