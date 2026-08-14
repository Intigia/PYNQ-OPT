# lpyolo_slim_256 @ 125 MHz — the reference deploy

Slim 2-head network (yolov1-tiny, 2 scales, v5-style anchored, width 0.35),
**i84w4a4o8** — note: the `w2a4` in the folder name is wrong, the weights are w4
(idma0.npy identical to the pynq-opt repo's `zed125` build; checkpoint from W&B run
`intigia/runs_2head/q499evrc`, mAP50 0.4785 on its own val at 320). "it10" build
with the MVAUs in RTL (standalone thresholds) and SIMD=4, WNS +0.011 ns.

- **I/O**: 256×256×3 UINT8 input (idma1); INT24 outputs 8×8×18 (stride 32) and
  16×16×18 (stride 16). External weights over DDR (idma0,
  `runtime_weights/idma0.npy`).
- **Measured performance**: 31.2 img/s (batch 100), full batch sweep in
  `nw_metrics_batch*.txt`; single-frame latency in `latency_metrics_125MHz.txt`
  (scripts `lpyolo_slim_125MHz_256x256i_driver_latency*.py`).
- **On-board accuracy** (same network, measured in the project): mAP50 0.337 on
  COCO-vehicles px12 / 0.594 on VOC.

Board usage: see `deploy/README_BOARD.md` (run from `driver/`, sudo + PYNQ venv).
Board-free demo: `../demo/demo_02_offline.py slim`. Decode: yolov3-tiny anchors in
input pixels + v5 semantics; dequant scales in `../demo/slim_postproc.npz`.

## Verified on the board (2026-08)

On ZedBoard and **also on Pynq-Z1** (same die; FINN uses no PL pins):
execute bit for bit against `expected_out*` (0 diffs — on the PYNQ v2.x image the
INT24 unpack reverses the byte order, fix on the host) and **31.2 img/s** @125 MHz.
