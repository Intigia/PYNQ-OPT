#!/usr/bin/env python3
"""DEMO 2 — Full post-processing WITHOUT a board, from stored FPGA outputs.

Each implementation ships real accelerator captures; this demo reproduces on the
PC the whole chain from those integers to detections:

  slim    : expected_out*.npy (packed INT24) -> unpack -> dequant (export
            scales) -> yolov5 decode (sigmoid+anchors) -> NMS -> boxes.
  yolov3  : ref_pl_out*.npy (UINT4, backbone on the PL) -> PS tail in pure numpy
            (ps_tail: the integer neck+head convolutions) -> EXACT comparison
            against ref_final_out*.npy -> anchor-free DFL decode -> NMS -> boxes.
  lpyolo320: output_fpga.npy (UINT4, LPYOLO backbone) -> unpack -> statistics.
            (a FINN toolchain build with untrained weights: it demonstrates the
            HW pipeline, not detection.)

Usage:  python demo_02_offline.py [slim|yolov3|lpyolo320|all]
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")  # Windows console: avoids mojibake
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from common import decode_v5, nms, unpack_int24, unpack_uint4  # noqa: E402


def demo_slim():
    print("\n=== slim 256 (lpyolo_slim_256_*_125MHz) ===")
    drv = ROOT / "lpyolo_slim_256_w2a4_125MHz" / "deploy" / "driver"
    z = np.load(HERE / "slim_postproc.npz")
    sc = {8: z["scale_global_out"], 16: z["scale_global_out_1"]}
    stride = {8: 32, 16: 16}
    boxes, confs = [], []
    for f in ("expected_out0.npy", "expected_out1.npy"):
        v = np.load(drv / f)               # the driver stores these already unpacked
        if v.shape[-1] == 3:               # in case they were packed: same unpack as the board
            v = unpack_int24(v.astype(np.uint8))
        v = v.reshape((1,) + v.shape[-3:])
        sp = v.shape[1]
        logits = v.astype(np.float32) * sc[sp]
        b, c = decode_v5(logits, stride[sp])
        boxes.append(b[0])
        confs.append(c[0])
        print(f"  {f}: {v.shape[1:]} INT24 -> dequant (scale {sc[sp].mean():.2e}) -> decode")
    allc = np.concatenate(confs)
    det = nms(np.concatenate(boxes), allc)
    print(f"  {len(det)} detections (input.npy is a RANDOM verification vector,")
    print(f"   0 detections is the correct result; max conf {allc.max():.3f}). For")
    print("   real images use demo_01 + demo_03 on the board.")
    for d in det[:6]:
        print("   ", np.round(d, 1))


def demo_yolov3():
    print("\n=== yolov3_tiny_custom_w4a4 (PL backbone + PS tail) ===")
    drv = ROOT / "yolov3_tiny_custom_w4a4" / "deploy" / "driver"
    sys.path.insert(0, str(drv))
    from ps_tail import run_tail

    pl = [np.load(drv / "ref_pl_out0.npy"), np.load(drv / "ref_pl_out1.npy")]
    print(f"  PL outputs (uint4): {[p.shape[1:] for p in pl]}")
    touts = run_tail(np.load(drv / "tail.npz", allow_pickle=True), pl)
    ok = True
    for i, t in enumerate(touts):
        ref = np.load(drv / f"ref_final_out{i}.npy")
        tt = t if t.shape == ref.shape else t.transpose(0, 3, 1, 2)
        d = (tt != ref).sum()
        ok &= d == 0
        print(f"  PS tail vs final reference {i}: {d} diffs {tt.shape}")
    assert ok, "the PS tail does not reproduce the references"
    # DFL decode with the deploy's own runner (same functions as on the board)
    import run_ocupacion as ro
    det = ro.decode(ro.classify_raw(touts, ro.load_postproc()))
    print(f"  DFL+NMS decode: {len(det)} detections (ref_input is RANDOM: 0 is the")
    print("   correct result; the demo's value is the exact PS tail above).")
    for d in det[:6]:
        print("   ", np.round(d, 1))


def demo_lpyolo320():
    print("\n=== lpyolo_320 (LPYOLO backbone, untrained weights) ===")
    d = ROOT / "lpyolo_320_w4a4_fifo_depth6_100MHz"
    raw = np.load(d / "output_fpga.npy")
    v = unpack_uint4(raw[..., None]) if raw.dtype == np.uint8 and raw.ndim == 4 else raw
    v = np.asarray(v)
    print(f"  output_fpga.npy: {v.shape} {raw.dtype} -> uint4 in [{v.min()}, {v.max()}]"
          f", mean {v.mean():.3f}, non-zero {(v != 0).mean() * 100:.1f}%")
    print("  (10x10x208 backbone feature map; without a trained head there are no")
    print("   boxes — these builds document the FINN chain: FIFOs, clock, latency.)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("slim", "all"):
        demo_slim()
    if what in ("yolov3", "all"):
        demo_yolov3()
    if what in ("lpyolo320", "all"):
        demo_lpyolo320()
    print("\nOFFLINE DEMO OK")
