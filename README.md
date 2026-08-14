# finn_implementations — vehicle detection on FPGA (Zynq-7020) with FINN

A collection of accelerators generated with [FINN](https://github.com/Xilinx/finn) for
ZedBoard/Pynq (xc7z020) over the course of the pynq-opt project: from the original
LPYOLO backbone to the project's own networks — `slim` (2-head, v5-style anchored) and
`yolov3_tiny_custom` (anchor-free DFL) — trained with QAT (brevitas) on yolov5/ultralytics
forks and tracked in W&B (`intigia/runs_2head`, `intigia/yolov3-OCUPACIÓN`).

## Implementations

| Folder | Network | Input | Quant. | Clock | img/s (batch 100, measured) | Weights |
|---|---|---|---|---|---|---|
| `lpyolo_320_w4a4_fifo_automatic_100MHz` | LPYOLO backbone (8 convs, 10×10×208 output) | 320×320 INT8 | W4A4 | 100 MHz | **36.0** | untrained¹ |
| `lpyolo_320_w4a4_fifo_depth6_100MHz` | same | 320×320 INT8 | W4A4 | 100 MHz | 22.2 | untrained¹ |
| `lpyolo_320_w4a4_fifo_depth6_200MHz` | same | 320×320 INT8 | W4A4 | 200 MHz | 44.3 | untrained¹ |
| `lpyolo_slim_256_w2a4_125MHz` | slim 2-head v5-anchored (8×8×18 + 16×16×18 outputs) | 256×256 UINT8 | **i84w4a4o8**² | 125 MHz | **31.2** | QAT COCO-vehicles (W&B run `q499evrc`) |
| `lpyolo_slim_256_w2a4_140MHz` | same | 256×256 UINT8 | i84w4a4o8² | 140 MHz | — (build NOT functional³) | same (identical `idma0.npy`) |
| `yolov3_tiny_custom_w4a4` | anchor-free DFL, 2 scales, backbone in PL + tail in PS | 320×320 UINT8 | i84w4a4h448, width 0.35 | 100 MHz | 12.4 | QAT car+voc (W&B run `qbpdji23`) |

¹ Builds from the FINN toolchain bring-up phase: they validate FIFOs, clock and
latency, not detection (the network outputs a saturated feature map).
² The folder name says `w2a4` but the weights are **w4a4** (idma0.npy identical to
the repo's `zed125` build, checkpoint i84**w4**a4o8).
³ Verified on the board: the 140 MHz build returns all zeros (timing; the design's
WNS was already marginal at 125 MHz). Consistent with its folder holding no metrics.

## Board compatibility (verified in hardware)

The three `lpyolo_320_*` builds target `xc7z020clg400` (Pynq-Z1/Z2); the two slim
builds and the yolov3 target `xc7z020clg484` (ZedBoard). **Same die**: since FINN
accelerators use no external PL pins (all I/O goes through the PS AXI ports), the
clg484 bitstreams **also work on the Pynq-Z1** — verified bit for bit on the board.
One caveat: on the PYNQ v2.x image the driver's INT24 unpack delivers the bytes in
reverse order (fixable on the host; uint4 outputs are unaffected).

## Accuracy measured ON THE BOARD (real accelerators, not simulation)

Validation test filtered with the grid-search rules (`px12_mb0_sh0`: boxes ≥12 px
@256, 15% backgrounds), same images for both networks:

| Network | Dataset | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| slim 256 | COCO-vehicles (894 img, 2013 boxes) | 0.467 | 0.347 | **0.337** | 0.157 |
| slim 256 | VOC-vehicles (989 img, 1400 boxes) | 0.729 | 0.549 | **0.594** | 0.321 |
| yolov3 custom 320 | COCO-vehicles | 0.636 | 0.429 | **0.484** | 0.275 |
| yolov3 custom 320 | VOC-vehicles | 0.845 | 0.659 | **0.761** | 0.519 |

Measured trade-off: the anchor-free 320 network gains ~0.15 mAP50 but runs at
12 img/s versus 31 for the slim 256. The PC↔board chain (INT24/UINT4 unpack,
dequant, decode, PS tail) is verified **bit for bit** against the FINN drivers
(0 diffs).

## Demos (`demo/`)

Scripts that reproduce everything learnt, end to end — see `demo/README.md`:

0. `demo_00_download.py` — downloads COCO (cocodataset.org) and VOC (ultralytics
   mirrors) and converts them to YOLO `{car,bus}→vehicle` as in the project.
   **Not redistributed** (their own licences): they are fetched from their sources
   at use time, with frozen MD5 checksums.
1. `demo_01_dataset.py` — dataset treatment: the grid-search variant filter +
   letterbox + uint8 batches for the driver (for COCO; VOC is used in full).
2. `demo_02_offline.py` — full post-processing without a board, using the FPGA
   captures included in each folder (unpack, dequant, v5/DFL decode, exact PS
   tail, NMS).
3. `demo_03_board.py` — everything on the board over SSH: driver functional
   verification, throughput_test and batched inference with unpack on the PC.

## Board operation — the important bits

- Always run the drivers **from `driver/`** (`runtime_weights/` resolves relative
  to the cwd) and as root with the PYNQ venv:
  `sudo bash -c 'source /etc/profile.d/pynq_venv.sh; source /etc/profile.d/xrt_setup.sh; python3 driver.py ...'`
- In designs with external weights, `idma0` = weights and the image enters via `idma1`.
- After a DMA hang **do not reload the overlay: power-cycle** the board.
- The driver's `unpack_output` costs ~1 s/img on the ARM: for large batches, pull
  the *packed* outputs and unpack on the PC (`demo/common.py`, verified 0 diffs).
