#!/usr/bin/env python3
"""DEMO 1 — Dataset treatment (what the pynq-opt grid search taught us).

Takes a dataset in YOLO layout (images/<split>/*.jpg + labels/<split>/*.txt) and:
  1. Applies the grid-search variant filter (px12_mb0_sh0 by default):
     boxes < min_px are dropped; an image losing >50% of its boxes is dropped
     entirely; backgrounds subsampled to 15% with a fixed seed (reproducible).
  2. Letterboxes to the accelerator's resolution (256 slim / 320 lpyolo-yolov3).
  3. Writes uint8 NHWC batches (in_*.npy) ready for the FINN board driver.

COCO and VOC are NOT bundled here (their own licences): download them with their
official tools and pass the path via --src (or use demo_00_download.py). In the
pynq-opt repo the layout is produced by sw/car_dataset/ (COCO vehicles) and
make_voc_dataset.py (VOC).

Usage:
  python demo_01_dataset.py --src <dataset_path> [--split val] [--imgsz 256]
                            [--min-px 12] [--max-boxes 0] [--batch 100] [--out out_dir]
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")  # Windows console: avoids mojibake
import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from common import filter_variant, letterbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root (images/ + labels/)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=256, help="256 slim | 320 lpyolo/yolov3")
    ap.add_argument("--min-px", type=int, default=12)
    ap.add_argument("--max-boxes", type=int, default=0, help="0 = no limit")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--out", default="dataset_out")
    a = ap.parse_args()

    src = Path(a.src)
    idir, ldir = src / "images" / a.split, src / "labels" / a.split
    assert idir.is_dir(), f"missing {idir} (YOLO layout: images/{a.split}/*.jpg)"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    items = filter_variant(idir, ldir, min_px=a.min_px, max_boxes=a.max_boxes)
    n_box = sum(len(k) for _, k in items)
    n_bg = sum(1 for _, k in items if not k)
    total = len(list(idir.glob("*.jpg")))
    print(f"variant px{a.min_px}_mb{a.max_boxes}_sh0 over {src.name}/{a.split}:")
    print(f"  {total} original img -> {len(items)} after filtering "
          f"({len(items)-n_bg} with boxes, {n_bg} backgrounds, {n_box} boxes)")

    metas, batch, bi = [], [], 0
    for ip, keep in items:
        im = cv2.imread(str(ip))
        h, w = im.shape[:2]
        lb, r, (px, py) = letterbox(im, a.imgsz)
        gt = np.array([[(float(v[1]) - float(v[3]) / 2) * w * r + px,
                        (float(v[2]) - float(v[4]) / 2) * h * r + py,
                        (float(v[1]) + float(v[3]) / 2) * w * r + px,
                        (float(v[2]) + float(v[4]) / 2) * h * r + py]
                       for v in (l.split() for l in keep)], np.float32).reshape(-1, 4)
        metas.append(dict(name=ip.name, gt=gt))
        batch.append(lb[:, :, ::-1])  # BGR -> RGB: the network eats RGB (the /255 lives in the graph)
        if len(batch) == a.batch:
            np.save(out / f"in_{bi:03d}.npy", np.stack(batch).astype(np.uint8))
            batch, bi = [], bi + 1
    if batch:
        np.save(out / f"in_{bi:03d}.npy", np.stack(batch).astype(np.uint8))
        bi += 1
    with open(out / "meta.pkl", "wb") as f:
        pickle.dump(metas, f)
    print(f"  {bi} batches of <= {a.batch} ({a.imgsz}x{a.imgsz} uint8 NHWC) + meta.pkl -> {out}/")
    # weights <-> training data correspondence (verified in W&B):
    #   slim 256 (q499evrc)      -> trained on COCO-vehicles px12_mb0_sh0
    #   yolov3_custom (qbpdji23) -> trained on COCO+VOC vehicles px12
    # mAP50 measured on the board: slim 0.337 COCO / 0.594 VOC; yolov3 0.484 / 0.761
    # (VOC scores higher because it is an easier test, not because of the training)
    kind = "COCO" if "voc" not in src.name.lower() else "VOC"
    print(f"  [note] source {kind}: the slim was trained on COCO-vehicles and the "
          f"yolov3_custom on COCO+VOC — both networks are valid for either test.")
    print("  next step: demo_03_board.py --indir", out)


if __name__ == "__main__":
    main()
