# User manual — finn_implementations

How to use everything in this repository, step by step. For what each folder
*is*, see `README.md`; this manual covers how to *run* it.

## 1. Prerequisites

**PC (Windows or Linux):**
- Python ≥3.8 with `numpy`, `opencv-python`, `pillow` (the pynq-opt project's
  `snn_env` conda environment works as-is).
- `curl` on the PATH (dataset downloads).
- PuTTY's `plink` and `pscp` on the PATH (board demos only).

**Board:** a Zynq-7020 board (Pynq-Z1/Z2 or ZedBoard) with a PYNQ image, reachable
over SSH (default user/password `xilinx`/`xilinx`). All six bitstreams run on
either board — same die; see "Board compatibility" in `README.md`.

## 2. Quick start (no board, no downloads)

Prove the whole numeric chain with the FPGA captures already included:

```
cd demo
python demo_02_offline.py all
```

Expected: the yolov3 PS tail reproduces its references with **0 diffs**, the slim
INT24 outputs unpack/decode cleanly, and the lpyolo320 feature map is dumped.
0 detections is correct — the reference inputs are random verification vectors.

## 3. Getting the datasets

COCO and VOC are licensed datasets: they are downloaded from their sources, never
redistributed.

```
cd demo
python demo_00_download.py --coco --out datasets      # COCO val2017 (~1.2 GB)
python demo_00_download.py --voc  --out datasets      # VOC 07+12 (~2.8 GB)
python demo_00_download.py --coco --coco-train ...    # optional: train2017 (~18 GB)
```

Outputs (YOLO layout, single class `vehicle` = COCO/VOC `{car,bus}`):
- `datasets/dataset_coco/` — apply the variant filter next (step 4).
- `datasets/dataset_voc/` + `datasets/dataset_voc_test/` — used **in full** (project
  rule: no variant filtering for VOC).

Downloads resume if interrupted (`curl -C -`) and are MD5-checked on every rerun.

## 4. Preparing batches for the accelerator

```
python demo_01_dataset.py --src datasets/dataset_coco --imgsz 256 --out coco_px12
```

- `--imgsz 256` for the slim builds, `--imgsz 320` for lpyolo_320/yolov3.
- Applies the px12_mb0_sh0 variant (boxes ≥12 px at 256 scale, 15% backgrounds,
  fixed seed) and letterboxes each image.
- Output: `in_XXX.npy` (uint8 NHWC batches) + `meta.pkl` (ground truth in the
  letterbox canvas) — exactly what `demo_03` and the mAP tooling consume.

## 5. Running on the board

```
python demo_03_board.py --impl lpyolo_slim_256_w2a4_125MHz --host xilinx@<board-ip>
```

What it does, in order:
1. Uploads the deploy (bitfile + driver + weights) to `/home/xilinx/demo_<impl>/`.
2. Functional check: `execute` with the known input, compared against the stored
   reference — must print **0 diffs**.
3. `throughput_test` (batch 100 by default) and prints the metrics.
4. With `--indir coco_px12`: runs your batches, stores the **packed** outputs and
   pulls them back (unpack on the PC via `demo/common.py` — ~1 s/img cheaper than
   unpacking on the ARM).

Pick the implementation with `--impl` (tab-completes from the folder names).
Re-runs can add `--skip-upload`.

If the board is only reachable through an SSH tunnel/jump host, set the tunnel up
externally and point `--host` at it — the script itself stays connection-agnostic.

### Reading the results

- **Latency/throughput**: `nw_metrics*.txt` in the driver directory.
- **Detections / mAP**: unpack the pulled `raw*.npy` with
  `common.unpack_int24` (slim) or `unpack_uint4` (yolov3), dequantise with
  `slim_postproc.npz` scales, decode with `common.decode_v5` + `common.nms`.
  For full P/R/mAP50/mAP50-95 use `FINN/slim/16_board_validate.py` in the
  pynq-opt repo (same maths, plus the metric).

### Which implementation should I use?

| Goal | Use |
|---|---|
| Best accuracy (mAP50 0.48/0.76) | `yolov3_tiny_custom_w4a4` (12 img/s) |
| Best speed with trained weights | `lpyolo_slim_256_w2a4_125MHz` (31 img/s) |
| Toolchain/latency experiments | any `lpyolo_320_*` (untrained weights) |
| — | avoid `lpyolo_slim_256_w2a4_140MHz`: returns all zeros (timing) |

## 6. Troubleshooting

- **`No module named 'pynq'` under sudo** — the sudo prefix in `demo_03` already
  handles both PYNQ image generations; if running by hand, source
  `/etc/profile.d/pynq_venv.sh` + `xrt_setup.sh` (v3 image) or export
  `HOME=/home/xilinx` and add `~xilinx/.local/.../site-packages` to `PYTHONPATH`
  and use `/usr/bin/python3` (v2 image).
- **`Device.devices` IndexError** — XRT env not loaded: source
  `/etc/profile.d/xrt_setup.sh`.
- **Outputs ~100% wrong but stable, INT24 designs on a v2.x image** — the byte
  order of the unpack is reversed; reverse the 3 bytes per element on the host
  (see `README.md`, "Board compatibility").
- **All-zero outputs** — with slim_140 that is the (broken) build; with anything
  else, check weights (`runtime_weights/` must sit next to `driver.py` and be the
  cwd's child) and re-run.
- **Driver never returns / DMA hang** — the PL is stuck: **power-cycle the board;
  do not reload the overlay**.
- **Batch-size mismatch** — `demo_03` refuses batches whose resolution does not
  match the accelerator (256 vs 320); regenerate with the right `--imgsz`.

## 7. Layout recap

```
finn_implementations/
├── README.md            <- what everything is + measured results
├── MANUAL.md            <- this file
├── demo/                <- runnable end-to-end demos (00 download … 03 board)
├── lpyolo_320_*/        <- flat deploys (driver + bitfile in the folder root)
├── lpyolo_slim_256_*/   <- deploy/ = bitfile/ + driver/ (FINN packaging)
└── yolov3_tiny_custom_w4a4/  <- deploy/ + PS tail (ps_tail.py, run_ocupacion.py)
```
