"""Shared utilities for the demos: exactly the same maths that was verified
bit for bit against the board in the pynq-opt project (0 diffs).

- letterbox: yolov5-val-style image preprocessing (pad 114, no upscaling).
- variant filter: the grid-search rules (min_px, max_boxes, 15% backgrounds).
- INT24/UINT4 unpack: undoes the FINN DMA packing on the PC.
- v5 decode + NMS: from the accelerator's integers to boxes.
"""
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # the offline unpack demos do not need cv2
    cv2 = None

IMGSZ_FILTER = 256   # scale at which the filter's px thresholds are defined (project notebook)

# yolov3-tiny anchors (input pixels) used by the slim builds; yolov5 semantics
ANCHORS = {16: [(10, 14), (23, 27), (37, 58)], 32: [(81, 82), (135, 169), (344, 319)]}


def letterbox(im, new):
    """Resize keeping aspect ratio and pad with grey 114 (yolov5 val)."""
    h, w = im.shape[:2]
    r = min(new / h, new / w, 1.0)
    nw, nh = round(w * r), round(h * r)
    if (nw, nh) != (w, h):
        im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new - nw) / 2, (new - nh) / 2
    t, b = round(dh - 0.1), round(dh + 0.1)
    l, rp = round(dw - 0.1), round(dw + 0.1)
    im = cv2.copyMakeBorder(im, t, b, l, rp, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, (l, t)


def filter_variant(img_dir, lbl_dir, min_px=12, max_boxes=0, bg_frac=0.15,
                   max_frac_drop=0.5, seed=0):
    """The project's grid-search rules (variant pxN_mbN_shN, no sharpness filter).
    Returns [(image_path, [filtered_label_lines])]."""
    import random
    from PIL import Image

    rng = random.Random(seed)
    withbox, backgrounds = [], []
    for ip in sorted(Path(img_dir).glob("*.jpg")):
        lp = Path(lbl_dir) / (ip.stem + ".txt")
        lines = [l for l in lp.read_text().strip().splitlines() if l.strip()] if lp.exists() else []
        if not lines:
            backgrounds.append(ip)
            continue
        w, h = Image.open(ip).size
        sc = IMGSZ_FILTER / max(w, h)
        keep = [l for l in lines
                if (float(l.split()[3]) * w * float(l.split()[4]) * h) ** 0.5 * sc >= min_px]
        if not keep or 1 - len(keep) / len(lines) > max_frac_drop:
            continue
        if max_boxes and len(keep) > max_boxes:
            continue
        withbox.append((ip, keep))
    target = round(len(withbox) * bg_frac / (1 - bg_frac)) if bg_frac else 0
    keep_bg = backgrounds if len(backgrounds) <= target else rng.sample(backgrounds, target)
    return [(ip, k) for ip, k in withbox] + [(ip, []) for ip in keep_bg]


def unpack_int24(raw):
    """(..., 3) little-endian bytes -> signed int32. Verified 0 diffs against the
    FINN driver's unpack on the board."""
    v = raw[..., 0].astype(np.int32) | raw[..., 1].astype(np.int32) << 8 \
        | raw[..., 2].astype(np.int32) << 16
    return v - ((v >> 23) & 1) * (1 << 24)


def unpack_uint4(raw):
    """(..., 1) bytes -> uint4 (one value per byte). Verified against board refs."""
    return raw[..., 0].astype(np.int32)


def decode_v5(logits, stride, anchors=ANCHORS):
    """(N,H,W,18) float logits -> xyxy boxes + conf, yolov5 anchored semantics."""
    n, hh, ww, _ = logits.shape
    p = 1 / (1 + np.exp(-logits.reshape(n, hh, ww, 3, 6)))
    gy, gx = np.meshgrid(np.arange(hh), np.arange(ww), indexing="ij")
    xy = (p[..., 0:2] * 2 - 0.5 + np.stack([gx, gy], -1)[None, :, :, None, :]) * stride
    wh = (p[..., 2:4] * 2) ** 2 * np.array(anchors[stride], np.float32)[None, None, None]
    conf = p[..., 4] * p[..., 5]
    box = np.concatenate([xy - wh / 2, xy + wh / 2], -1)
    return box.reshape(n, -1, 4).astype(np.float32), conf.reshape(n, -1).astype(np.float32)


def nms(boxes, scores, conf_th=0.25, iou_th=0.45, max_det=100):
    """Greedy NMS in pure numpy."""
    keep = scores > conf_th
    boxes, scores = boxes[keep], scores[keep]
    order = scores.argsort()[::-1]
    out = []
    while order.size and len(out) < max_det:
        i = order[0]
        out.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (a[i] + a[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_th]
    return np.concatenate([boxes[out], scores[out, None]], 1) if out else np.zeros((0, 5))
