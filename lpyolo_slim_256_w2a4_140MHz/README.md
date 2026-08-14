# lpyolo_slim_256 @ 140 MHz

Same design and weights as `lpyolo_slim_256_w2a4_125MHz` (identical idma0.npy; the
`w2a4` in the name is wrong here too — it is w4a4) but clocked at 140 MHz.
No original `README_BOARD.md` and no stored metrics (the `nw_metrics_batch*.txt`
files are empty): an experimental clock-push build; use the 125 MHz one as the
validated reference.

- I/O identical to the 125 MHz build: 256×256 UINT8 → INT24 8×8×18 + 16×16×18.
- Its own latency scripts: `lpyolo_slim_140MHz_256x256i_driver_latency*.py`.
- To measure it: `../demo/demo_03_board.py --impl lpyolo_slim_256_w2a4_140MHz`.

## Verified on the board (2026-08): NOT FUNCTIONAL

Loads and runs without hanging but returns **all zeros** (tested on Pynq-Z1).
Likely cause: timing — the design closed with WNS +0.011 ns at 125 MHz; at 140 it
does not hold. Consistent with the folder arriving with no recorded metrics at all.
Use the 125 MHz build.
