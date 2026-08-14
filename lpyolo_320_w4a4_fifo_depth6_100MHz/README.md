# lpyolo_320 W4A4 — fixed FIFOs depth=6 @ 100 MHz

The same LPYOLO backbone as `fifo_automatic` but with every FIFO at a fixed depth
of 6: it demonstrates that the rtlsim "automatic" sizing was inflated by
back-pressure pooling at the bottleneck (capping does not change latency and saves
resources). Untrained weights (self-contained bitstream, `runtime_weights/` empty).

This folder concentrates the project's **host-side experiments**:

- `benchmark_execute.py` — timing breakdown of the driver's `execute()`.
- `profile_batch100.py` — per-phase profile at batch 100 (`output_batch100*.npy`).
- `test_fast_unpack.py` — origin of the fast unpack: the driver's `unpack_output`
  costs ~1 s/img on the ARM; here unpacking on the PC was validated (later
  generalised in `../demo/common.py`, 0 diffs).
- `driver_latency.py` / `driver_base_latency.py` — single-frame latency.
- Verification vectors: `input_known*.npy`, `output_fpga*.npy`,
  `output_known_pytorch.npy` (model reference), `output_raw_dma.npy`.

## Verified on the board (2026-08)

`execute` bit for bit against `output_fpga.npy` (0 diffs) and **22.2 img/s**
(batch 100, 100 MHz).
