# lpyolo_320 W4A4 — automatic FIFOs (rtlsim) @ 100 MHz

LPYOLO backbone (8 convs: 8-8-16-32-32-56-104-208, 320×320 INT8 input, single
10×10×208 UINT4 output) compiled with the full FINN toolchain and the FIFO depths
**sized by rtlsim** (`auto_fifo_depths`). The project's first functional build:
**untrained** weights (internal, baked into the bitstream — `runtime_weights/` is
empty), useful for validating the toolchain, DMA and latency, not for detection.

- Drivers: FINN's standard `driver.py` + a `*_latency_optimization_1.py` variant
  (single-frame latency measurement with less host overhead).
- Vectors: `input_known_nhwc*.npy` → `output_fpga.npy` (a real board capture;
  inspectable without hardware via `../demo/demo_02_offline.py lpyolo320`).
- Compare with the `fifo_depth6` builds: the same design with minimal fixed FIFOs
  (depth 6) — the FIFO study was one of the project's big lessons (sim-sized FIFOs
  are inflated by back-pressure pooling at the bottleneck; capping them barely
  changes latency and saves a lot of LUT/BRAM).

## Verified on the board (2026-08)

`execute` bit for bit against `output_fpga.npy` (0 diffs) and **36.0 img/s**
(batch 100, 100 MHz). The large rtlsim FIFOs sustain the batched pipeline better
than depth6 (22.2 img/s).
