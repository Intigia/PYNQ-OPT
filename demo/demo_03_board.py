#!/usr/bin/env python3
"""DEMO 3 — Running on the board (ZedBoard/Pynq with a PYNQ image, from the PC).

Uploads the chosen deploy to the board over SSH, runs the FINN driver's
functional verification (execute with a known input vs expected) and the
throughput_test, and optionally infers the demo_01 batches and pulls the packed
outputs (the INT24 unpack happens on the PC: it costs ~1 s/img on the ARM).

Requirements: plink/pscp (PuTTY) on the PATH and the board powered on.
After a DMA hang the PL is unusable: power-cycle, do not reload the overlay.

Usage:
  python demo_03_board.py --impl lpyolo_slim_256_w2a4_125MHz [--host xilinx@pynq]
                          [--password xilinx] [--indir dataset_out] [--batch 100]
"""
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")  # Windows console: avoids mojibake
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

IMPLS = {d.name: d for d in ROOT.iterdir()
         if d.is_dir() and (d / "deploy" / "driver" / "driver.py").exists()
         or (d / "driver.py").exists()}


def sh(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd[:6]), "...")
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=sorted(IMPLS))
    ap.add_argument("--host", default="xilinx@pynq")
    ap.add_argument("--password", default="xilinx")
    ap.add_argument("--indir", help="in_*.npy batches from demo_01 (optional)")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--skip-upload", action="store_true", help="already uploaded")
    a = ap.parse_args()

    imp = IMPLS[a.impl]
    local = imp / "deploy" if (imp / "deploy").is_dir() else imp
    drv = "deploy/driver" if (imp / "deploy" / "driver").is_dir() else "."
    rdir = f"/home/xilinx/demo_{a.impl}"
    PW = ["-batch", "-pw", a.password]
    plink = lambda cmd: sh(["plink", *PW, a.host, cmd])
    # sudo prefix robust across PYNQ images: on v3 (venv) the venv's python3 ships
    # pynq; on v2 (2020.x) pynq/bitstring live in the xilinx user's HOME.
    # The first python that imports both is selected.
    sudo = (f"echo {a.password} | sudo -S bash -c "
            "'source /etc/profile.d/pynq_venv.sh 2>/dev/null; "
            "source /etc/profile.d/xrt_setup.sh 2>/dev/null; "
            "export HOME=/home/xilinx "
            "PYTHONPATH=/home/xilinx:/home/xilinx/.local/lib/python3.8/site-packages:$PYTHONPATH; "
            "PY=python3; for c in python3 /usr/bin/python3; do "
            "$c -c \"import pynq, bitstring\" 2>/dev/null && PY=$c && break; done; ")

    if not a.skip_upload:
        print(f"[1] uploading {local} -> {rdir} (bitfile ~4 MB + driver)...")
        plink(f"mkdir -p {rdir}")
        sh(["pscp", *PW, "-r", local, f"{a.host}:{rdir}/"])
    if drv != ".":                       # classic FINN deploy (slim/yolov3)
        ddir, bit = f"{rdir}/{drv}", "../bitfile/finn-accel.bit"
        infile, outs = "input.npy", "out0.npy out1.npy"
        # reference: expected_<out> generated at build time
        cmp_py = ("import numpy as np, glob; "
                  "[print(f, (np.load(f)!=np.load('expected_'+f)).sum(), 'diffs') "
                  "for f in ['out0.npy','out1.npy'] if glob.glob('expected_'+f)]")
    else:                                # flat layout (lpyolo_320)
        ddir, bit = f"{rdir}/{local.name}", "finn-accel.bit"
        infile, outs = "input_known_nhwc.npy", "out0.npy"
        # reference: the FPGA capture stored in the folder
        cmp_py = ("import numpy as np; "
                  "print('out0 vs output_fpga:', "
                  "(np.load('out0.npy').squeeze()!=np.load('output_fpga.npy').squeeze()).sum(), 'diffs')")

    print("[2] functional verification (execute vs reference)...")
    plink(sudo + f"cd {ddir} && $PY driver.py --exec_mode execute "
          f"--bitfile {bit} --inputfile {infile} --outputfile {outs}' 2>&1 | tail -2")
    plink(f"cd {ddir} && python3 -c \"{cmp_py}\"")

    print(f"[3] throughput_test batch {a.batch}...")
    plink(sudo + f"cd {ddir} && $PY driver.py --exec_mode throughput_test "
          f"--batchsize {a.batch} --bitfile {bit} --platform zynq-iodma' 2>&1 | tail -2")
    plink(f"cat {ddir}/nw_metrics.txt")

    if a.indir:
        # the batch resolution must match the accelerator input:
        # slim = 256, lpyolo/yolov3 = 320. A mismatch gives no error on the board,
        # it gives rubbish.
        import numpy as np
        expected = 256 if "slim" in a.impl else 320
        first = sorted(Path(a.indir).glob("in_*.npy"))
        assert first, f"no in_*.npy batches in {a.indir} (generate with demo_01)"
        got = np.load(first[0], mmap_mode="r").shape
        assert got[1:3] == (expected, expected), (
            f"batches at {got[1]}x{got[2]} but '{a.impl}' expects {expected}x{expected}: "
            f"regenerate with demo_01_dataset.py --imgsz {expected}")
        print(f"[4] inferring batches {a.indir} ({got[0]} img/batch, "
              f"{expected}x{expected} ok)...")
        sh(["pscp", *PW, str(HERE / "board_run_all.py"),
            *[str(f) for f in sorted(Path(a.indir).glob("in_*.npy"))],
            f"{a.host}:{rdir}/"])
        plink(sudo + f"cp {rdir}/board_run_all.py {ddir}/ 2>/dev/null; cd {ddir} && "
              f"$PY board_run_all.py --indir {rdir} --batch {a.batch}'")
        sh(["pscp", *PW, f"{a.host}:{rdir}/raw*.npy", str(Path(a.indir)) + "/"])
        print("  unpack/decode on the PC: see demo/common.py (unpack_int24 + decode_v5 + nms)")

    print("\nBOARD DEMO OK")


if __name__ == "__main__":
    main()
