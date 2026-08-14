#!/usr/bin/env python3
"""Runs on the Pynq/Zed: infers every in_*.npy in a directory and stores the
PACKED outputs (raw0_*.npy, raw1_*.npy) without unpacking (the INT24 unpack on
the ARM costs ~1 s/img; it is done on the PC). Run from the driver/ dir as root."""
import argparse
import glob
import os
import time

import numpy as np
from driver import io_shape_dict
from driver_base import FINNExampleOverlay

p = argparse.ArgumentParser()
p.add_argument("--indir", required=True)
p.add_argument("--bitfile", default="../bitfile/finn-accel.bit")
p.add_argument("--batch", type=int, default=200)
a = p.parse_args()

accel = FINNExampleOverlay(
    bitfile_name=a.bitfile, platform="zynq-iodma", io_shape_dict=io_shape_dict,
    batch_size=a.batch, runtime_weight_dir="runtime_weights/",
)

files = sorted(glob.glob(os.path.join(a.indir, "in_*.npy")))
t_total, n_total = 0.0, 0
for f in files:
    tag = os.path.basename(f)[3:-4]
    x = np.load(f)  # (N,256,256,3) uint8
    n = x.shape[0]
    if n < a.batch:  # pad to the fixed batch; the excess is discarded on the PC
        x = np.concatenate([x, np.zeros((a.batch - n,) + x.shape[1:], x.dtype)])
    accel.copy_input_data_to_device(accel.pack_input(accel.fold_input(x)))
    t0 = time.time()
    accel.execute_on_buffers()
    dt = time.time() - t0
    for o in range(2):
        accel.copy_output_data_from_device(accel.obuf_packed[o], o)
        np.save(os.path.join(a.indir, f"raw{o}_{tag}.npy"), accel.obuf_packed[o])
    t_total += dt
    n_total += n
    print(f"{tag}: {n} img, exec {dt*1000:.0f} ms ({a.batch/dt:.1f} img/s)", flush=True)
print(f"TOTAL {n_total} img, exec {t_total:.1f} s")
