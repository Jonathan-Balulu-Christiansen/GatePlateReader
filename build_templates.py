"""Build the 10 digit templates: the per-pixel median of every digit the reader segments in labeled plates.

    python build_templates.py                  # data/reader/plates -> templates/
    python build_templates.py --data DIR --out DIR

A plate image is labeled by its file name: the number, optionally followed by "_anything".
Only plates that segment into exactly as many boxes as their number has digits contribute.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import cv2
import numpy as np

import digits_recognition as dr

EXT = (".png", ".jpg", ".jpeg")


def label_of(path: Path) -> str:
    return path.stem.split("_")[0]


def plate_images(folder) -> list[Path]:
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXT)


def samples_of(path: Path, cfg: dr.Cfg = dr.Cfg()) -> list[np.ndarray] | None:
    """The plate's normalized characters, or None if its segmentation does not match its label."""
    chars = dr.digit_chars(cv2.imread(str(path)), cfg)
    return chars if len(chars) == len(label_of(path)) else None


def all_samples(paths: list[Path], cfg: dr.Cfg = dr.Cfg(), workers: int = 8) -> list[list[np.ndarray] | None]:
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(samples_of, paths, repeat(cfg), chunksize=16))


def median_templates(labels: list[str], samples: list) -> dict[str, np.ndarray]:
    """Per digit: the median of all its samples, thresholded at one half."""
    by_digit = {d: [] for d in "0123456789"}
    for label, chars in zip(labels, samples):
        for d, ch in zip(label, chars or ()):
            by_digit[d].append(ch)
    return {d: (np.median(np.stack(v), axis=0) >= 127.5).astype(np.uint8) * 255 for d, v in by_digit.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/reader/plates", help="labeled plate images, searched recursively")
    ap.add_argument("--out", default="templates")
    args = ap.parse_args()
    paths = plate_images(args.data)
    if not paths:
        raise SystemExit(f"no plate images in {args.data}: the data is not part of the repository (see the README)")
    labels = [label_of(p) for p in paths]
    samples = all_samples(paths)
    tpl = median_templates(labels, samples)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for d, t in tpl.items():
        cv2.imwrite(str(out / f"{d}.png"), t)
    used = sum(s is not None for s in samples)
    counts = {d: sum(label.count(d) for label, s in zip(labels, samples) if s is not None) for d in "0123456789"}
    print(f"{len(paths)} plates, {used} segmented into the right number of digits; samples per digit: {counts}")
    print(f"templates written to {out}/")


if __name__ == "__main__":
    main()
