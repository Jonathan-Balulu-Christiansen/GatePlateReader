"""Evaluate the reader: a plate counts only if every digit is right.

    python evaluate.py                        # every set below that exists under data/
    python evaluate.py --sets test cv

Sets (data is not part of the repository; see the README for the layout):
  street  40 hand-cropped plates, leave-one-out: templates from the training plates without that number
  test    held-out test crops (data/reader/test), templates from all training plates (data/reader/plates)
  cv      5-fold cross-validation by plate on the training plates (all crops of a plate in one fold)
  gate    full pipeline on the gate clips and pictures (data/gate/labels.json: {folder: {file stem: number}})
"""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import digits_recognition as dr
import plate_detection as pd
import trigger_burst as tb
from build_templates import all_samples, label_of, median_templates, plate_images

ROOT = Path(__file__).resolve().parent
DATA = Path("data")
_tpl, _cfg = None, None


def _init(templates, cfg):
    global _tpl, _cfg
    _tpl, _cfg = templates, cfg


def _read(path):
    return dr.read_plate(cv2.imread(str(path)), _tpl, _cfg).digits


def read_all(paths, templates, cfg=dr.Cfg(), workers=8) -> list[str]:
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(templates, cfg)) as ex:
        return list(ex.map(_read, paths, chunksize=16))


def report(name: str, truths: list[str], reads: list[str]) -> dict:
    if not truths:
        print(f"{name:28s} no labeled plates")
        return dict(set=name, plates=0)
    n, exact = len(truths), sum(t == r for t, r in zip(truths, reads))
    same_len = [(t, r) for t, r in zip(truths, reads) if len(t) == len(r)]
    digits = [(a, b) for t, r in same_len for a, b in zip(t, r)]
    conf = Counter((a, b) for a, b in digits if a != b).most_common(6)
    res = dict(set=name, plates=n, exact=exact, exact_pct=round(100 * exact / n, 1),
               per_digit_pct=round(100 * float(np.mean([a == b for a, b in digits])), 1) if digits else None,
               wrong_length=n - len(same_len), confusions=" ".join(f"{a}->{b}:{k}" for (a, b), k in conf))
    print(f"{name:28s} whole plate exact {exact}/{n} = {res['exact_pct']}% | per digit (right-length reads) "
          f"{res['per_digit_pct']}% | wrong length {res['wrong_length']} | top confusions {res['confusions'] or '-'}")
    return res


def cv_folds(labels: list[str], k: int = 5) -> list[set]:
    """Folds by plate, reproducible: plates in file order, seed-0 shuffle, every k-th plate."""
    order = list(dict.fromkeys(labels))
    random.Random(0).shuffle(order)
    return [set(order[i::k]) for i in range(k)]


def gate(templates) -> list[dict]:
    """Trigger -> burst -> sharpest frame -> plate -> digits, on every labeled clip and picture."""
    out = []
    for group, labels in json.loads((DATA / "gate/labels.json").read_text()).items():
        truths, reads = [], []
        for f in sorted((DATA / "gate" / group).glob("*")):
            if f.stem not in labels:
                continue
            if f.suffix == ".mp4":
                cap = cv2.VideoCapture(str(f))
                trig, frame = tb.MotionTrigger(cap.get(cv2.CAP_PROP_FPS)), None
                while True:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if trig.update(fr):
                        frame = tb.sharpest_of_burst(cap, fr, trig.burst)[0]
                        break
            else:
                frame = cv2.imread(str(f))
            det = pd.detect_plate(frame) if frame is not None else None
            truths.append(labels[f.stem])
            reads.append(dr.read_plate(det.warped, templates, warped=True).digits if det else "")
        if truths:
            out.append(report(f"gate {group}", truths, reads))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["street", "test", "cv", "gate"])
    ap.add_argument("--data", default="data", help="the data folder (layout in the README)")
    ap.add_argument("--out", default="outputs/evaluation.json")
    args = ap.parse_args()
    global DATA
    DATA = Path(args.data)
    train = plate_images(DATA / "reader/plates")
    if not train and {"street", "test", "cv"} & set(args.sets):
        raise SystemExit(f"no labeled plates in {DATA / 'reader/plates'}: the data is not in the repository (README)")
    labels = [label_of(p) for p in train]
    samples = all_samples(train) if {"street", "test", "cv"} & set(args.sets) else []
    results = []
    if "street" in args.sets and (DATA / "street_plates").exists():
        street = plate_images(DATA / "street_plates")
        reads = []
        for p in street:  # leave one out: the plate's own number never reaches its templates
            tpl = median_templates(*zip(*[(lab, s) for lab, s in zip(labels, samples) if lab != label_of(p)]))
            reads.append(dr.read_plate(cv2.imread(str(p)), tpl).digits)
        results.append(report("street plates (leave-one-out)", [label_of(p) for p in street], reads))
    if "test" in args.sets:
        for folder in sorted(p for p in (DATA / "reader/test").glob("*") if p.is_dir()):
            test = plate_images(folder)
            results.append(report(f"test: {folder.name}", [label_of(p) for p in test],
                                  read_all(test, median_templates(labels, samples))))
    if "cv" in args.sets:
        truths, reads = [], []
        for k, fold in enumerate(cv_folds(labels)):
            tpl = median_templates(*zip(*[(lab, s) for lab, s in zip(labels, samples) if lab not in fold]))
            test = [p for p, lab in zip(train, labels) if lab in fold]
            r = read_all(test, tpl)
            report(f"  cv fold {k}", [label_of(p) for p in test], r)
            truths += [label_of(p) for p in test]
            reads += r
        results.append(report("cv, 5 folds by plate", truths, reads))
    if "gate" in args.sets and (DATA / "gate/labels.json").exists():
        results += gate(dr.load_templates(ROOT / "templates"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
