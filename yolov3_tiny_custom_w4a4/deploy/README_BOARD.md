# occupancy e_w4a4_ambas035 — PL backbone (L0-L10) + PS remainder
- PL: 4 DMAs (idma image, iwdma L10 weights via DDR, 2 odma: L8 tap 96ch + L10 184ch)
- PS: ps_tail.py (pure numpy: L9-12 pool + neck + head) + DFL/NMS decode
- runtime_weights/ relative to the cwd: ALWAYS run from driver/
- Verification: driver execute with ref_input_u8 -> compare vs ref_pl_out*;
  then run_ocupacion.py --selftest-board compares final vs ref_final_out*.
After a DMA hang: do NOT reload the overlay — power-cycle.
