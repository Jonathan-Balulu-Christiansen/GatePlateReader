"""Hough-line plate detector for a fixed gate camera: the four plate edges -> perspective warp to 400x80."""
from dataclasses import dataclass

import cv2
import numpy as np

ROI = (0.55, 0.35, 0.80, 0.75)        # x0, y0, x1, y1 as fractions of the frame: where plates pass this camera
RECT_W, RECT_H = (150, 600), (30, 260)  # plate size limits in the full frame (px)
HOUGH = dict(rho=0.5, theta=np.pi / 120, threshold=20, minLineLength=20, maxLineGap=6)
CANNY_PASSES = ((50, 150), (40, 120), (30, 90))  # strict first, relaxed if no plate is found
MAX_LINES, H_GAP, V_GAP, H_YTOL, V_XTOL, TOL_PX, BOT_TOL = 50, 15, 8, 5, 2, 30, 65


@dataclass(frozen=True)
class PlateResult:
    corners: np.ndarray              # 4x2 float32, full-frame coordinates
    bbox: tuple[int, int, int, int]  # x, y, w, h
    warped: np.ndarray               # the plate, perspective-corrected to out_w x out_h


def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL."""
    pts = np.asarray(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def four_point_warp(img: np.ndarray, pts: np.ndarray, out_w: int = 400, out_h: int = 80) -> np.ndarray:
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(order_points(pts), dst), (out_w, out_h),
                               flags=cv2.INTER_CUBIC)


def _seg_len(x1, y1, x2, y2):
    return float(np.hypot(x2 - x1, y2 - y1))


def _span(seg, axis):
    """(lo, hi) of a segment along x (axis 'h') or y (axis 'v')."""
    a, b = (seg[0], seg[2]) if axis == 'h' else (seg[1], seg[3])
    return min(a, b), max(a, b)


def _overlap(a1, a2, b1, b2):
    return max(0.0, float(min(max(a1, a2), max(b1, b2)) - max(min(a1, a2), min(b1, b2))))


def _intersect(s, t):
    """Intersection of the infinite lines through segments s and t, or None if parallel."""
    (x1, y1, x2, y2), (x3, y3, x4, y4) = s, t
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    a, b = x1 * y2 - y1 * x2, x3 * y4 - y3 * x4
    return (a * (x3 - x4) - (x1 - x2) * b) / d, (a * (y3 - y4) - (y1 - y2) * b) / d


# ── Yellow plate evidence ────────────────────────────────────────────

def yellow_mask(frame_bgr: np.ndarray, roi_gray: np.ndarray) -> np.ndarray:
    """HSV yellow; the V threshold follows the scene brightness, so it works by day and at night."""
    v = max(40, min(100, int(float(np.median(roi_gray)) * 1.2)))
    return cv2.inRange(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV), (15, 80, v), (40, 255, 255))


def _yellow_weight(ym, seg, band=8, n=9):
    """1.0 for a plate border (yellow on one side only), 0.3 inside the plate, 0.1 for the background."""
    x1, y1, x2, y2 = seg
    H, W = ym.shape
    dx, dy = float(x2 - x1), float(y2 - y1)
    ll = np.hypot(dx, dy)
    if ll < 1:
        return 0.1
    nx, ny = -dy / ll, dx / ll
    left = right = 0
    for i in range(n):
        cx, cy = x1 + (i + 1) / (n + 1) * dx, y1 + (i + 1) / (n + 1) * dy
        lf = rf = False
        for r in range(2, band + 1):
            px, py = int(round(cx + r * nx)), int(round(cy + r * ny))
            lf |= 0 <= px < W and 0 <= py < H and ym[py, px] > 0
            px, py = int(round(cx - r * nx)), int(round(cy - r * ny))
            rf |= 0 <= px < W and 0 <= py < H and ym[py, px] > 0
            if lf and rf:
                break
        left += lf
        right += rf
    lr, rr = left / n, right / n
    prox, s = max(lr, rr), lr + rr
    one_side = abs(lr - rr) / s if s >= 0.1 else 0.0
    if prox < 0.2:
        return 0.1
    if one_side > 0.4 and prox > 0.5:
        return 1.0
    if one_side > 0.25 and prox > 0.3:
        return 0.7
    return 0.3 if prox > 0.5 else 0.2


# ── Line merging and pairing ─────────────────────────────────────────

def _merge(segments, gap, tol, axis):
    """Merge collinear segments: group by position (y for 'h', x for 'v'), join along the line if the gap is small."""
    if not segments:
        return []
    segs = sorted(segments, key=lambda t: t[1])
    groups, grp, gp = [], [segs[0]], float(segs[0][1])
    for item in segs[1:]:
        if abs(item[1] - gp) <= tol:
            grp.append(item)
            gp = (gp * (len(grp) - 1) + item[1]) / len(grp)
        else:
            groups.append(grp)
            grp, gp = [item], float(item[1])
    groups.append(grp)

    def merged(p, e0, e1):
        q, a, b = int(round(p)), int(round(e0)), int(round(e1))
        seg = (a, q, b, q) if axis == 'h' else (q, a, q, b)
        return _seg_len(*seg), float(p), seg

    out = []
    for g in groups:
        g = sorted(g, key=lambda t: _span(t[2], axis)[0])
        cp, (e0, e1) = float(g[0][1]), _span(g[0][2], axis)
        for _, p, seg in g[1:]:
            s0, s1 = _span(seg, axis)
            if abs(p - cp) <= tol and s0 - e1 <= gap:
                e0, e1, cp = min(e0, s0), max(e1, s1), 0.5 * (cp + p)
            else:
                out.append(merged(cp, e0, e1))
                cp, e0, e1 = float(p), s0, s1
        out.append(merged(cp, e0, e1))
    return sorted(out, key=lambda t: -t[0])


def _pairs(segments, min_sep, max_sep, len_tol, min_overlap, keep, axis):
    """Roughly parallel pairs (score, first, second), first = lower position; scored by total length."""
    a, b = (0, 2) if axis == 'h' else (1, 3)
    out = []
    for i, (li, mi, si) in enumerate(segments):
        for lj, mj, sj in segments[i + 1:]:
            if not min_sep <= abs(mi - mj) <= max_sep or abs(li - lj) / (max(li, lj) + 1e-6) > len_tol:
                continue
            shorter = min(abs(si[b] - si[a]), abs(sj[b] - sj[a])) + 1e-6
            if _overlap(si[a], si[b], sj[a], sj[b]) / shorter >= min_overlap:
                out.append((li + lj, si, sj) if mi < mj else (li + lj, sj, si))
    return sorted(out, key=lambda t: -t[0])[:keep]


# ── Detector ─────────────────────────────────────────────────────────

def detect_plate(frame_bgr: np.ndarray, h_max_deg: float = 8.0, v_tol_deg: float = 6.0,
                 out_w: int = 400, out_h: int = 80, dbg: dict | None = None) -> PlateResult | None:
    """Find the plate as the best-scoring rectangle of two horizontal and two vertical Hough lines."""
    fh, fw = frame_bgr.shape[:2]
    rx0, ry0, rx1, ry1 = int(ROI[0] * fw), int(ROI[1] * fh), int(ROI[2] * fw), int(ROI[3] * fh)
    roi_gray = cv2.cvtColor(frame_bgr[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2GRAY)
    pre = cv2.fastNlMeansDenoising(roi_gray, None, h=6, templateWindowSize=7, searchWindowSize=21)
    pre = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(pre)
    pre = cv2.bilateralFilter(pre, d=7, sigmaColor=50, sigmaSpace=50)
    ym = yellow_mask(frame_bgr, roi_gray)
    yellow = np.count_nonzero(ym[ry0:ry1, rx0:rx1]) > 200
    weights = {}  # each line takes part in many candidate rectangles: weigh it once

    def weight(seg):
        if seg not in weights:
            weights[seg] = _yellow_weight(ym, seg)
        return weights[seg]

    if dbg is not None:
        dbg.update(roi_xyxy=(rx0, ry0, rx1, ry1), roi_pre=pre)

    for canny_pass, (lo, hi) in enumerate(CANNY_PASSES, 1):
        edges = cv2.Canny(pre, lo, hi, apertureSize=3, L2gradient=False)
        lines = cv2.HoughLinesP(edges, **HOUGH)
        if dbg is not None:
            dbg.update(edges=edges, lines_raw=lines)
        if lines is None or len(lines) < 4:
            continue
        hor, ver = [], []
        for x1, y1, x2, y2 in lines[:, 0]:
            x1, y1, x2, y2 = x1 + rx0, y1 + ry0, x2 + rx0, y2 + ry0
            ang, length = abs(float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))), _seg_len(x1, y1, x2, y2)
            if ang <= h_max_deg or abs(ang - 180) <= h_max_deg:
                if x2 < x1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                hor.append((length, 0.5 * (y1 + y2), (x1, y1, x2, y2)))
            if abs(ang - 90) <= v_tol_deg:
                if y2 < y1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                ver.append((length, 0.5 * (x1 + x2), (x1, y1, x2, y2)))
        hor, ver = _merge(hor, H_GAP, H_YTOL, 'h')[:MAX_LINES], _merge(ver, V_GAP, V_XTOL, 'v')[:MAX_LINES]
        h_pairs = _pairs(hor, RECT_H[0], RECT_H[1], 0.80, 0.55, 50, 'h')
        v_pairs = _pairs(ver, RECT_W[0], RECT_W[1], 0.55, 0.45, 30, 'v')
        if dbg is not None:
            dbg.update(horizontals=hor, verticals=ver)
        best = None
        for _, top, bottom in h_pairs:
            hx0, hx1 = min(_span(top, 'h')[0], _span(bottom, 'h')[0]), max(_span(top, 'h')[1], _span(bottom, 'h')[1])
            top_y, bot_y = 0.5 * (top[1] + top[3]), 0.5 * (bottom[1] + bottom[3])
            for _, left, right in v_pairs:
                vy0 = min(_span(left, 'v')[0], _span(right, 'v')[0])
                vy1 = max(_span(left, 'v')[1], _span(right, 'v')[1])
                if (0.5 * (left[0] + left[2]) > hx0 + TOL_PX or 0.5 * (right[0] + right[2]) < hx1 - TOL_PX
                        or top_y > vy0 + TOL_PX or bot_y < vy1 - BOT_TOL):
                    continue
                c = [_intersect(top, left), _intersect(top, right), _intersect(bottom, right), _intersect(bottom, left)]
                if any(p is None for p in c):
                    continue
                corners = np.array(c, np.float32)
                if (corners[:, 0].min() < rx0 or corners[:, 0].max() > rx1
                        or corners[:, 1].min() < ry0 or corners[:, 1].max() > ry1):
                    continue
                x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
                if not (RECT_W[0] <= w <= RECT_W[1] and RECT_H[0] <= h <= RECT_H[1]):
                    continue
                aspect = w / (h + 1e-6)
                if not 1.5 <= aspect <= 5.5:
                    continue
                # Full weight for 2.5-5:1 (a plate is ~4.7:1 head-on, less under perspective), less outside
                amult = (0.5 + 0.5 * max(0, aspect - 1.5) if aspect < 2.5
                         else max(0.5, 1.0 - (aspect - 5.0)) if aspect > 5.0 else 1.0)
                segs = (top, bottom, left, right)
                if yellow:
                    cov = np.count_nonzero(ym[y:y + h, x:x + w]) / max(w * h, 1)
                    score = sum(_seg_len(*s) * weight(s) for s in segs) * (1 + cov * 2) * amult
                else:
                    score = sum(_seg_len(*s) for s in segs) * amult
                if best is None or score > best[0]:
                    best = (score, corners, (x, y, w, h), segs)
        if best is None:
            continue
        score, corners, bbox, segs = best
        if dbg is not None:
            dbg.update(score=float(score), canny_pass=canny_pass, lines=segs)
        return PlateResult(corners, bbox, four_point_warp(frame_bgr, corners, out_w, out_h))
    return None
