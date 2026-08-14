# demo/ — everything learnt, runnable

Four chainable scripts reproducing the project's full flow: download → data
treatment → accelerator → detections. PC requirements: python with numpy, opencv,
pillow (the project's `snn_env` environment works) and `curl`. For the board:
PuTTY (`plink`/`pscp`).

## 0. Downloading COCO and VOC — `demo_00_download.py`

```
python demo_00_download.py --voc  --out datasets     # ~2.8 GB, ultralytics mirrors
python demo_00_download.py --coco --out datasets     # val2017 ~1.2 GB, cocodataset.org
python demo_00_download.py --coco --coco-train ...   # + train2017 (~18 GB)
```

COCO and VOC carry **their own licences** (COCO: CC-BY 4.0 annotations, images under
Flickr terms), which is why they are not redistributed: they are fetched from their
sources at use time, with MD5 checksums frozen in `checksums.json` (the same
anti-corruption pattern as `sw/car_dataset/make_voc_dataset.py` in the pynq-opt repo).

- **VOC**: a replica of the project's builder — XML→YOLO with `{car,bus}`→`vehicle`,
  difficult boxes dropped, backgrounds kept with empty txt, `dataset_voc` (07+12
  trainval, 80/20 split, seed 0) + `dataset_voc_test` (07 test). Notebook rule:
  VOC is always used **in full** (no variant filtering).
- **COCO**: `instances_*.json` → YOLO with the same `{car,bus}`→`vehicle` classes,
  `iscrowd` dropped. This one **does** get the variant filter afterwards (demo_01).

## 1. Datasets — `demo_01_dataset.py`

```
python demo_01_dataset.py --src <dataset> [--split val] [--imgsz 256] \
       [--min-px 12] [--max-boxes 0] [--batch 100] [--out dataset_out]
```

- **COCO and VOC are not bundled here** (their own licences): download them through
  official channels and pass the path via `--src`. The expected layout is YOLO
  (`images/<split>/*.jpg` + `labels/<split>/*.txt` with normalised `cls xc yc w h`).
- Applies the grid-search **variant filter** (the lesson learnt: filtering the
  dataset inflates mAP mechanically, so the variant is defined once and always
  evaluated against the same exam): boxes < `min_px` px (at 256 scale) are dropped;
  an image losing >50% of its boxes is dropped entirely; backgrounds subsampled to
  15% with a fixed seed.
- Letterbox (pad 114, no upscaling) to the accelerator's resolution: **256** for the
  slim builds, **320** for lpyolo/yolov3.
- Output: `in_XXX.npy` uint8 NHWC batches + `meta.pkl` (GT in letterbox canvas).

Verified: on the project's COCO-vehicles test it produces exactly the
894 img / 2013 boxes of the official board validation.

## 2. Offline post-processing — `demo_02_offline.py`

```
python demo_02_offline.py [slim|yolov3|lpyolo320|all]
```

No board needed: uses the real accelerator captures stored in each folder.

- **slim**: `expected_out*.npy` → INT24 unpack → dequant with the export scales
  (`slim_postproc.npz`) → yolov5 decode (sigmoid + anchors + grid) → NMS.
- **yolov3**: `ref_pl_out*.npy` (uint4 from the PL backbone) → **PS tail in pure
  numpy** (`ps_tail.py`: the integer neck+head convolutions) → **exact** comparison
  against `ref_final_out*.npy` (0 diffs) → DFL decode (softmax over 8 bins) → NMS.
- **lpyolo320**: unpack of the 10×10×208 feature map (untrained-weights builds).

Note: the reference inputs are FINN's random verification vectors, so 0 detections
is the correct result — the value lies in the PC numeric chain reproducing the
hardware bit for bit.

## 3. Board — `demo_03_board.py`

```
python demo_03_board.py --impl lpyolo_slim_256_w2a4_125MHz \
       [--host xilinx@pynq] [--password xilinx] [--indir dataset_out] [--batch 100]
```

1. Uploads the deploy (bitfile + driver) over SSH.
2. Functional verification: `driver.py --exec_mode execute` compared against
   `expected_*` (must give 0 diffs).
3. `throughput_test` (configurable batch) and `nw_metrics.txt` dump.
4. With `--indir` (batches from demo_01): batched inference through
   `board_run_all.py`, which stores the **packed** outputs and pulls them to the
   PC — unpacking on the ARM costs ~1 s/img; on the PC it is free
   (`common.unpack_int24`, verified 0 diffs).

Afterwards: decode + mAP on the PC with `common.py` (or the full script
`FINN/slim/16_board_validate.py` in the pynq-opt repo, which computes
P/R/mAP50/mAP50-95).

## Files

- `common.py` — letterbox, variant filter, INT24/UINT4 unpack, v5 decode, NMS.
- `slim_postproc.npz` — per-channel dequantisation scales for the slim outputs.
- `board_run_all.py` — batched board runner (packed outputs, no unpack on the ARM).
