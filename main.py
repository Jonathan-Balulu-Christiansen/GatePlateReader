"""License-plate reader for a fixed gate camera: motion trigger -> sharpest frame -> plate -> digits.

    python main.py                        # the example clip
    python main.py PATH                   # a gate video (waits for the trigger) or a single frame
    python main.py PATH --save DIR        # no windows: write the pipeline figure to DIR
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import digits_recognition as dr
import plate_detection as pd
import trigger_burst as tb

ROOT = Path(__file__).resolve().parent
TILE_W, TILE_H, BAR = 480, 270, 26
GREEN, ORANGE, RED, YELLOW, WHITE = (80, 200, 80), (0, 165, 255), (40, 40, 230), (0, 220, 255), (235, 235, 235)


def fmt(number: str) -> str:
    """Israeli layout: 123-45-678 or 12-345-67."""
    return f"{number[:3]}-{number[3:5]}-{number[5:]}" if len(number) == 8 else \
        f"{number[:2]}-{number[2:5]}-{number[5:]}" if len(number) == 7 else number or "-"


def open_window(title: str, w: int, h: int) -> None:
    """A resizable window (the figure is 1440x810, larger than some laptop screens)."""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title, w, h)


def window_open(title: str) -> bool:
    return cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1


def wait_for_car(path: Path, show: bool) -> tuple[np.ndarray | None, str]:
    """Play the video until the trigger fires; return the sharpest frame of the burst."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, "cannot be opened"
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    trig, i, every = tb.MotionTrigger(fps), 0, max(1, round(fps / 30))  # show at most 30 frames per second
    t_shown, title = time.perf_counter(), "Gate camera (q: quit)"
    if show:
        open_window(title, 1280, 720)
    while True:
        ok, frame = cap.read()
        if not ok:
            return None, f"no trigger in {i} frames"
        if trig.update(frame):
            best, k, _ = tb.sharpest_of_burst(cap, frame, trig.burst)
            return best, f"trigger at {i / fps:.2f} s, sharpest of {trig.burst} burst frames: #{k + 1}"
        if show and i % every == 0:
            _, y0, _, lx = trig.strip(frame.shape)
            cv2.imshow(title, cv2.line(frame.copy(), (lx, y0), (lx, frame.shape[0] - 1), RED, 3))
            wait_ms = every * 1000 / fps - 1000 * (time.perf_counter() - t_shown)  # play at the clip's own speed
            if cv2.waitKey(max(1, int(wait_ms))) in (27, ord("q")) or not window_open(title):
                return None, "stopped"
            t_shown = time.perf_counter()
        i += 1


def tile(img: np.ndarray, title: str) -> np.ndarray:
    """Letterbox an image into one captioned tile."""
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    s = min(TILE_W / img.shape[1], (TILE_H - BAR) / img.shape[0])
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    out = np.full((TILE_H, TILE_W, 3), 24, np.uint8)
    y, x = BAR + (TILE_H - BAR - img.shape[0]) // 2, (TILE_W - img.shape[1]) // 2
    out[y:y + img.shape[0], x:x + img.shape[1]] = img
    cv2.putText(out, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    return out


def text_tile(lines: list[tuple[str, float, tuple]], title: str) -> np.ndarray:
    out = tile(np.full((TILE_H - BAR, TILE_W, 3), 24, np.uint8), title)
    y = BAR + 20
    for text, scale, color in lines:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y += h + 22
        cv2.putText(out, text, ((TILE_W - w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return out


def pipeline_figure(frame: np.ndarray, note: str, templates: dict) -> tuple[np.ndarray, str]:
    """Run detection and reading on one frame; return a 3x3 figure of every step and the number read."""
    cfg, dbg = dr.Cfg(), {}
    t0 = time.perf_counter()
    det = pd.detect_plate(frame, dbg=dbg)
    t1 = time.perf_counter()
    rx0, ry0, rx1, ry1 = dbg["roi_xyxy"]
    view = frame.copy()
    cv2.rectangle(view, (rx0, ry0), (rx1, ry1), YELLOW, 3)
    tiles = [None, tile(dbg["roi_pre"], "2  ROI: denoise, CLAHE, bilateral"), tile(dbg["edges"], "3  Canny edges")]
    lines = cv2.cvtColor(dbg["edges"] // 3, cv2.COLOR_GRAY2BGR)
    for key, color in (("horizontals", GREEN), ("verticals", ORANGE)):
        for _, _, (x1, y1, x2, y2) in dbg.get(key, []):
            cv2.line(lines, (x1 - rx0, y1 - ry0), (x2 - rx0, y2 - ry0), color, 2)
    tiles.append(tile(lines, "4  Merged Hough lines: horizontal, vertical"))
    if det is None:
        tiles[0] = tile(view, "1  Frame and fixed ROI")
        tiles += [text_tile([("no plate found", 1.2, RED)], "5  Plate")] + [tile(np.zeros((9, 16), np.uint8), "")] * 4
        return np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3, 6)]), ""
    cv2.polylines(view, [det.corners.astype(np.int32)], True, RED, 4)
    tiles[0] = tile(view, "1  Frame, fixed ROI (yellow), plate (red)")
    win = cv2.cvtColor(dbg["edges"] // 3, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in dbg["lines"]:
        cv2.line(win, (x1 - rx0, y1 - ry0), (x2 - rx0, y2 - ry0), RED, 3)
    tiles.append(tile(win, "5  Best rectangle of 2 + 2 lines"))
    aligned = dr.align(det.warped, cfg, warped=True)
    res = dr.recognize(aligned, templates, cfg)
    t3 = time.perf_counter()
    sep = np.full((12, 400, 3), 24, np.uint8)
    tiles.append(tile(np.vstack([det.warped, sep, aligned]), "6  Perspective warp, then shear corrected"))
    boxes = cv2.cvtColor(res.binary, cv2.COLOR_GRAY2BGR)
    for x, y, w, h in res.boxes:
        cv2.rectangle(boxes, (x, y), (x + w, y + h), GREEN, 2)
    tiles.append(tile(boxes, "7  Best binarization, digit boxes"))
    chars = [dr.char_image(res.binary, b, cfg.open_size, cfg) for b in res.boxes]
    if len(chars) == len(res.digits):  # each digit above the template it matched
        chars = [np.vstack([c, np.full((6, 40), 60, np.uint8), templates[d]]) for c, d in zip(chars, res.digits)]
    strip = np.hstack([np.pad(c, ((0, 0), (3, 3)), constant_values=60) for c in chars or [np.zeros((9, 16), np.uint8)]])
    tiles.append(tile(strip, "8  Digits (top) and matched templates (bottom)"))
    tiles.append(text_tile([(fmt(res.digits), 1.6, GREEN if res.ok else RED),
                            (f"mean match {res.conf:.2f}", 0.7, WHITE),
                            (f"detect {1e3 * (t1 - t0):.0f} ms, read {1e3 * (t3 - t1):.0f} ms", 0.6, WHITE),
                            (note, 0.5, WHITE)],
                           "9  Result"))
    return np.vstack([np.hstack(tiles[i:i + 3]) for i in (0, 3, 6)]), res.digits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=ROOT / "examples/gate_example.mp4", type=Path)
    ap.add_argument("--save", type=Path, help="write the figure here instead of opening windows")
    ap.add_argument("--templates", default=ROOT / "templates", type=Path)
    args = ap.parse_args()
    templates = dr.load_templates(args.templates)
    if args.path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        frame, note = wait_for_car(args.path, show=args.save is None)
    else:
        frame, note = cv2.imread(str(args.path)), "single image"
    if frame is None:
        raise SystemExit(f"{args.path}: {note if note != 'single image' else 'cannot be read'}")
    figure, number = pipeline_figure(frame, note, templates)
    print(f"{args.path.name}: {note}; plate {fmt(number) if number else 'not found'}")
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save / "pipeline.png"), figure)
        print(f"figure written to {args.save / 'pipeline.png'}")
        return
    cv2.destroyAllWindows()
    title = f"Plate {fmt(number)} (any key: close)" if number else "No plate (any key: close)"
    open_window(title, *figure.shape[1::-1])
    cv2.imshow(title, figure)
    while cv2.waitKey(100) == -1 and window_open(title):  # waitKey(0) never returns if the window is closed
        pass
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
