#!/usr/bin/env python3
"""DEMO 0 — Download and build the datasets (COCO and VOC -> vehicles).

COCO and Pascal VOC carry their own licences (COCO: CC-BY 4.0 annotations and
images under Flickr terms; VOC: Flickr terms), so they are NOT redistributed
with this repository: this script fetches them from their sources at use time,
as the usual toolkits do (ultralytics, torchvision).

- VOC  (--voc): same procedure as sw/car_dataset/make_voc_dataset.py in the
  pynq-opt project (the notebook's "dataset work"): zips from the ultralytics
  mirrors (~2.8 GB), XML -> YOLO with {car, bus} -> class 0 'vehicle', difficult
  boxes dropped, images without vehicles kept as background (empty txt),
  dataset_voc (07+12 trainval, 80/20 split, seed 0) and dataset_voc_test (07 test).
  Notebook rule: VOC is ALWAYS used in full (demo_01's variant filtering is
  for COCO).
- COCO (--coco): official images.cocodataset.org URLs (val2017 ~1 GB +
  annotations ~241 MB; train2017 ~18 GB optional). instances_*.json -> YOLO with
  {car, bus} -> 0, iscrowd dropped, backgrounds with empty txt. This one DOES
  get the grid-search variant afterwards (demo_01, px12_mb0_sh0).

Every download freezes its MD5 into checksums.json on first fetch and checks it
afterwards (same anti-corruption pattern as the repo).

Usage:
  python demo_00_download.py --voc --out datasets
  python demo_00_download.py --coco --out datasets            # val2017
  python demo_00_download.py --coco --coco-train --out datasets  # + train2017 (18 GB!)
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")  # Windows console: avoids mojibake
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

VEHICLE_VOC = {"car", "bus"}
COCO_VEHICLE = {3: "car", 6: "bus"}   # COCO category_id
VOC_MIRROR = "https://github.com/ultralytics/yolov5/releases/download/v1.0/"
VOC_ZIPS = ["VOCtrainval_06-Nov-2007.zip", "VOCtest_06-Nov-2007.zip",
            "VOCtrainval_11-May-2012.zip"]
COCO_URL = "http://images.cocodataset.org/"
SEED, VAL_FRAC = 0, 0.2


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dst, checks_path):
    """curl with resume + MD5 frozen as a local baseline (repo pattern)."""
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    name = dst.name
    if dst.exists() and (dst.parent / (name + ".ok")).exists():
        got = md5(dst)
        if checks.get(name, got) != got:
            raise RuntimeError(f"{name}: MD5 {got} != baseline {checks[name]} — "
                               f"delete {dst} and retry")
        checks.setdefault(name, got)
        checks_path.write_text(json.dumps(checks, indent=2))
        print(f"  {name}: already downloaded (MD5 ok)")
        return
    print(f"  downloading {name} ...", flush=True)
    subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(dst), url], check=True)
    got = md5(dst)
    if checks.get(name, got) != got:
        raise RuntimeError(f"{name}: MD5 {got} != previous baseline {checks[name]} — "
                           "suspicious download, aborting")
    checks[name] = got
    checks_path.write_text(json.dumps(checks, indent=2))
    (dst.parent / (name + ".ok")).touch()
    print(f"  MD5 {name}: {got} (local baseline)")


def place(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_yaml(root, splits):
    root.joinpath("data.yaml").write_text(
        f"path: {root}\n" + "".join(f"{k}: images/{v}\n" for k, v in splits.items())
        + "nc: 1\nnames: ['vehicle']\n")


# ----------------------------------- VOC ------------------------------------
def build_voc(out):
    dl = out / "voc_download"
    dl.mkdir(parents=True, exist_ok=True)
    devkit = dl / "VOCdevkit"
    for z in VOC_ZIPS:
        fetch(VOC_MIRROR + z, dl / z, dl / "checksums.json")
        # 2007 VOCtrainval and VOCtest share the VOC2007 dir: one marker per zip
        marker = dl / (z + ".extracted")
        if not marker.exists():
            zipfile.ZipFile(dl / z).extractall(dl)
            marker.touch()

    def yolo_lines(xml):
        r = ET.parse(xml).getroot()
        sz = r.find("size")
        w, h = float(sz.find("width").text), float(sz.find("height").text)
        out_l = []
        for o in r.iter("object"):
            if o.find("name").text not in VEHICLE_VOC:
                continue
            if o.find("difficult") is not None and o.find("difficult").text == "1":
                continue
            b = o.find("bndbox")
            x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
            x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
            out_l.append(f"0 {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                         f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
        return out_l

    def ids_of(year, split):
        p = devkit / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
        return [(year, i.strip()) for i in p.read_text().splitlines() if i.strip()]

    def emit(items, root, split):
        oi, ol = root / "images" / split, root / "labels" / split
        oi.mkdir(parents=True, exist_ok=True)
        ol.mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for year, iid in items:
            vdir = devkit / f"VOC{year}"
            jpg = vdir / "JPEGImages" / f"{iid}.jpg"
            lines = yolo_lines(vdir / "Annotations" / f"{iid}.xml")
            place(jpg, oi / f"{year}_{iid}.jpg")
            (ol / f"{year}_{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_img += 1
            n_box += len(lines)
        return dict(imagenes=n_img, cajas=n_box)

    trainval = ids_of(2007, "trainval") + ids_of(2012, "trainval")
    random.Random(SEED).shuffle(trainval)
    n_val = round(len(trainval) * VAL_FRAC)
    tr_root, te_root = out / "dataset_voc", out / "dataset_voc_test"
    counts_tr = {"train": emit(trainval[n_val:], tr_root, "train"),
                 "val": emit(trainval[:n_val], tr_root, "val")}
    counts_te = {"val": emit(ids_of(2007, "test"), te_root, "val")}
    write_yaml(tr_root, {"train": "train", "val": "val"})
    write_yaml(te_root, {"val": "val"})
    (tr_root / "_counts.json").write_text(json.dumps(counts_tr, indent=2))
    (te_root / "_counts.json").write_text(json.dumps(counts_te, indent=2))
    print(f"VOC ready: {counts_tr} / test {counts_te}")
    print("(notebook rule: VOC is used in full, no variant filtering)")


# ----------------------------------- COCO -----------------------------------
def build_coco(out, with_train):
    dl = out / "coco_download"
    dl.mkdir(parents=True, exist_ok=True)
    fetch(COCO_URL + "annotations/annotations_trainval2017.zip",
          dl / "annotations_trainval2017.zip", dl / "checksums.json")
    zipfile.ZipFile(dl / "annotations_trainval2017.zip").extractall(dl)
    splits = ["val2017"] + (["train2017"] if with_train else [])
    root = out / "dataset_coco"
    for sp in splits:
        fetch(COCO_URL + f"zips/{sp}.zip", dl / f"{sp}.zip", dl / "checksums.json")
        if not (dl / sp).exists():
            zipfile.ZipFile(dl / f"{sp}.zip").extractall(dl)
        ann = json.loads((dl / "annotations" / f"instances_{sp}.json").read_text())
        by_img = {}
        for a in ann["annotations"]:
            if a["category_id"] in COCO_VEHICLE and not a.get("iscrowd"):
                by_img.setdefault(a["image_id"], []).append(a["bbox"])
        tag = "train" if sp == "train2017" else "val"
        oi, ol = root / "images" / tag, root / "labels" / tag
        oi.mkdir(parents=True, exist_ok=True)
        ol.mkdir(parents=True, exist_ok=True)
        n_box = 0
        for im in ann["images"]:
            src = dl / sp / im["file_name"]
            place(src, oi / im["file_name"])
            w, h = im["width"], im["height"]
            lines = [f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}"
                     for x, y, bw, bh in by_img.get(im["id"], [])]
            (ol / (Path(im["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            n_box += len(lines)
        print(f"COCO {sp}: {len(ann['images'])} img, {n_box} vehicle boxes -> {tag}/")
    write_yaml(root, {"train": "train", "val": "val"} if with_train else {"val": "val"})
    print(f"COCO ready at {root} — next: demo_01_dataset.py --src {root} "
          "(variant px12_mb0_sh0)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc", action="store_true")
    ap.add_argument("--coco", action="store_true")
    ap.add_argument("--coco-train", action="store_true", help="also train2017 (~18 GB)")
    ap.add_argument("--out", default="datasets")
    a = ap.parse_args()
    if not (a.voc or a.coco):
        sys.exit("choose --voc and/or --coco (see the script header)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print("LICENCE NOTICE: COCO (CC-BY 4.0 annotations, images under Flickr terms)\n"
          "and VOC are downloaded from their sources for your own use; do not\n"
          "redistribute them with this repository.")
    if a.voc:
        build_voc(out)
    if a.coco:
        build_coco(out, a.coco_train)
