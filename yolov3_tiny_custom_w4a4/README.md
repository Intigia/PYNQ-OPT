# yolov3_tiny_custom_w4a4 — anchor-free DFL, backbone in PL + tail in PS

The "occupancy" network `e_w4a4_ambas035` (ultralytics fork, YOLOv8-style:
anchor-free with DFL reg_max=8, 2 scales, width 0.35, i84w4a4h448 quantisation).
Checkpoint from W&B run `intigia/yolov3-OCUPACIÓN/qbpdji23`, trained on COCO+VOC
vehicles (px12).

**Partitioning** (what makes this deploy distinctive): only the backbone (L0-L10)
runs on the FPGA; it returns two **uint4** feature maps: the L8 tap 20×20×96 and
L10 10×10×184. The rest runs on the ARM: `ps_tail.py` (59 qonnx nodes in pure
numpy: stride-1 pool, the neck+head convolutions as integer Im2Col+MatMul,
upsample, concat) + affine dequant (`postproc.npz`) + DFL decode + NMS
(`run_ocupacion.py`).

- **I/O**: 320×320×3 UINT8 (idma1); external L10 weights (idma0); 2 uint4 odma.
- **Measured performance**: 12.4 img/s (batch 100, `nw_metrics_batch*.txt`);
  latency in `latency_metrics_yolov3_tiny_custom.txt`.
- **On-board accuracy**: mAP50 0.484 COCO-vehicles px12 / 0.761 VOC — the best in
  the collection (in exchange for ⅓ of the slim's throughput).
- **Verification**: `ref_input_u8` → PL vs `ref_pl_out*`; PS tail vs
  `ref_final_out*` — exact (0 diffs, checkable without a board via
  `../demo/demo_02_offline.py yolov3`).

Board usage: `deploy/README_BOARD.md`. Single-photo inference:
`sudo python3 run_ocupacion.py --image photo.jpg` (from `driver/`).

## Verified on the board (2026-08)

On ZedBoard and **also on Pynq-Z1** (same die): backbone execute bit for bit
against `ref_pl_out*` (**0 diffs**, uint4 unaffected by endianness) and
**12.4 img/s** @100 MHz.
