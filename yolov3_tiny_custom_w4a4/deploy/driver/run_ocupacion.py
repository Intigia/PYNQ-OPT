#!/usr/bin/env python3
"""Inferencia completa ocupacion: PL (backbone+neck+convs detect, FINN) + PS (decode).

En placa (cwd = driver/):
    sudo python3 run_ocupacion.py --image foto.jpg
    sudo python3 run_ocupacion.py --input input.npy --dump out   # tensores crudos

Selftest en PC (sin placa, usa las referencias torch del 00b):
    python run_ocupacion.py --selftest

Decode = Detect._inference de ultralytics en numpy: afin por rama (postproc.npz)
-> DFL(softmax 8 bins) -> dist2bbox con anchors por celda -> sigmoid(cls) -> NMS.
"""

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REG_MAX = 8
STRIDES = {20: 16, 10: 32}  # celda -> stride (entrada 320)
CONF_TH = 0.25
IOU_TH = 0.45


def load_postproc():
    p = HERE / "postproc.npz"
    if not p.exists():
        p = HERE / "finn" / "postproc.npz"  # en PC; en placa va junto al driver
    z = np.load(p, allow_pickle=True)
    # meta: "nombre|shape|ref|c0|c1"; c0=0 -> box(32), c0=32 -> cls(1)
    entries = []
    for m in z["meta"]:
        name, shape, ref, c0, c1 = str(m).split("|")
        entries.append(dict(name=name, ref=ref, c0=int(c0), c1=int(c1),
                            scale=z[f"scale_{name}"], bias=z[f"bias_{name}"]))
    return entries


def softmax(x, axis):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def decode(outs_float):
    """outs_float: {(sp, 'box'|'cls'): (1,sp,sp,C) float} -> detecciones Nx5."""
    boxes, scores = [], []
    for sp, stride in STRIDES.items():
        box = outs_float[(sp, "box")].reshape(sp * sp, 4, REG_MAX)
        dist = (softmax(box, -1) * np.arange(REG_MAX)).sum(-1)  # (celdas, 4) l,t,r,b
        cy, cx = np.mgrid[0:sp, 0:sp]
        cx = cx.ravel() + 0.5
        cy = cy.ravel() + 0.5
        x1 = (cx - dist[:, 0]) * stride
        y1 = (cy - dist[:, 1]) * stride
        x2 = (cx + dist[:, 2]) * stride
        y2 = (cy + dist[:, 3]) * stride
        boxes.append(np.stack([x1, y1, x2, y2], 1))
        scores.append(1.0 / (1.0 + np.exp(-outs_float[(sp, "cls")].reshape(-1))))
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    keep = scores > CONF_TH
    boxes, scores = boxes[keep], scores[keep]
    # NMS
    order = scores.argsort()[::-1]
    final = []
    while order.size:
        i = order[0]
        final.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= IOU_TH]
    return np.concatenate([boxes[final], scores[final, None]], 1)


def classify_raw(raws, entries):
    """raws: lista de arrays enteros del PL/onnx -> {(sp,'box'|'cls'): float}."""
    outs = {}
    for raw in raws:
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 4 and raw.shape[1] in (32, 1) and raw.shape[1] != raw.shape[-1]:
            raw = raw.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        sp, ch = raw.shape[1], raw.shape[-1]
        # emparejar por (canales, espacial de la ref): ref global_out=20x20, _1=10x10
        cand = [e for e in entries if e["scale"].size == ch
                and {"global_out": 20, "global_out_1": 10}[e["ref"]] == sp]
        assert len(cand) == 1, (raw.shape, [e["name"] for e in cand])
        e = cand[0]
        outs[(sp, "box" if ch == 32 else "cls")] = raw * e["scale"] + e["bias"]
    assert len(outs) == 4, list(outs)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--input")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--verify-pl", action="store_true", dest="verify_pl",
                    help="con ref_input: compara PL y final contra referencias del PC")
    ap.add_argument("--dump")
    a = ap.parse_args()

    entries = load_postproc()

    if a.selftest:
        # referencias torch (00b): ref_out0 = P4 (1,33,20,20), ref_out1 = P5
        outs = {}
        for i, sp in enumerate((20, 10)):
            r = np.load(HERE / f"ref_out{i}.npy").transpose(0, 2, 3, 1)
            outs[(sp, "box")] = r[..., :32]
            outs[(sp, "cls")] = r[..., 32:]
        det = decode(outs)
        print(f"selftest: {len(det)} detecciones del ref aleatorio (decode no casca)")
        print(det[:5])
        return

    if a.image:
        from PIL import Image

        img = Image.open(a.image).convert("RGB").resize((320, 320))
        x = np.asarray(img, dtype=np.uint8)[None]  # (1,320,320,3) NHWC crudo
    else:
        x = np.load(a.input or "input.npy")

    from driver import io_shape_dict
    from driver_base import FINNExampleOverlay
    from ps_tail import run_tail

    accel = FINNExampleOverlay(
        bitfile_name="../bitfile/finn-accel.bit", platform="zynq-iodma",
        io_shape_dict=io_shape_dict, batch_size=1,
        runtime_weight_dir="runtime_weights/",
    )
    raws = accel.execute(x)
    raws = raws if isinstance(raws, (list, tuple)) else [raws]
    if a.dump:
        for i, r in enumerate(raws):
            np.save(f"{a.dump}{i}.npy", r)

    if a.verify_pl:
        refs = [np.load(HERE / "ref_pl_out0.npy"), np.load(HERE / "ref_pl_out1.npy")]
        for r in raws:
            ref = next(v for v in refs if v.shape[-1] == np.asarray(r).shape[-1])
            print(f"  PL {np.asarray(r).shape}: diffs = {(np.asarray(r) != ref).sum()}")

    tail_outs = run_tail(np.load(HERE / "tail.npz", allow_pickle=True), raws)
    if a.verify_pl:
        for i, t in enumerate(tail_outs):
            ref = np.load(HERE / f"ref_final_out{i}.npy")
            tt = t if t.shape == ref.shape else t.transpose(0, 3, 1, 2)
            print(f"  final {i}: diffs = {(tt != ref).sum()}")
    det = decode(classify_raw(tail_outs, entries))
    print(f"{len(det)} detecciones (x1,y1,x2,y2,conf):")
    for d in det:
        print("  ", np.round(d, 1))


if __name__ == "__main__":
    main()
