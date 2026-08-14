# lpyolo_320 W4A4 — FIFOs depth=6 @ 200 MHz

The 200 MHz variant of the `fifo_depth6_100MHz` build (same LPYOLO backbone,
untrained weights). It explores the design's frequency ceiling on the xc7z020.

- `convert_input.py` — utility to convert the input to NHWC INT8
  (`input_known_nhwc_int8.npy`).
- `driver_latency_200MHz.py` / `driver_base_latency_200MHz.py` — latency at 200 MHz.
- Vectors: `input_known*.npy` → `output_fpga.npy`.

Housekeeping note: the files `--batchsize`, `--bitfile`, `--exec_mode`,
`--inputfile`, `--outputfile`, `--platform` (0 bytes) are accidental rubbish from a
driver invocation with broken quoting — safe to delete.

## Verified on the board (2026-08)

`execute` bit for bit against `output_fpga.npy` (0 diffs, input
`input_known_nhwc_int8.npy`) and **44.3 img/s** (batch 100, 200 MHz).
