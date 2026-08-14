#!/usr/bin/env python3
"""Ejecutor numpy puro de la cola PS (tail.npz de ps_extract.py).

Replica exacta de las semanticas qonnx que aparecen en la cola:
Pad, MaxPoolNHWC, Im2Col, MatMul, MultiThreshold, Resize(NN), Concat,
Transpose, Mul, Add. Entrada: enteros uint4 de la PL (float array).
"""

import json

import numpy as np


def im2col(x, kh, kw, sh, sw, pad, pad_val=0.0):
    n, h, w, c = x.shape
    x = np.pad(x, ((0, 0), (pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
               constant_values=pad_val)
    H = (x.shape[1] - kh) // sh + 1
    W = (x.shape[2] - kw) // sw + 1
    out = np.empty((n, H, W, kh * kw * c), x.dtype)
    for oy in range(H):
        for ox in range(W):
            out[:, oy, ox, :] = x[:, oy * sh:oy * sh + kh, ox * sw:ox * sw + kw, :].reshape(n, -1)
    return out


def multithreshold(x, T, out_scale=1.0, out_bias=0.0, layout=None):
    # semantica qonnx: canal = dim 1 en 4D salvo data_layout NHWC (canal ultimo)
    nhwc = layout == "NHWC"
    if x.ndim == 4 and not nhwc:
        x = x.transpose(0, 2, 3, 1)  # NCHW -> NHWC para computar
    ch = x.shape[-1]
    Tb = T if T.shape[0] == ch else np.repeat(T, ch, axis=0)
    y = (x[..., None] >= Tb.reshape((1,) * (x.ndim - 1) + Tb.shape)).sum(-1).astype(np.float32)
    y = out_scale * y + out_bias
    if x.ndim == 4 and not nhwc:
        y = y.transpose(0, 3, 1, 2)  # de vuelta a NCHW
    return y


def maxpool_nhwc(x, k, s):
    n, h, w, c = x.shape
    H = (h - k) // s + 1
    W = (w - k) // s + 1
    out = np.empty((n, H, W, c), x.dtype)
    for oy in range(H):
        for ox in range(W):
            out[:, oy, ox, :] = x[:, oy * s:oy * s + k, ox * s:ox * s + k, :].max((1, 2))
    return out


def run_tail(z, pl_outs):
    """pl_outs: lista con las 2 salidas de la PL (cualquier orden); se asignan
    por forma: 10x10x184 -> corte L10, 20x20x96 -> tap L8."""
    meta = json.loads(str(z["meta"]))
    ctx = {}
    for o in pl_outs:
        o = np.asarray(o, dtype=np.float32)
        ctx[meta["cut"] if o.shape[-1] == 184 else meta["cut2"]] = o
    assert len(ctx) == 2, [np.shape(o) for o in pl_outs]
    get = lambda name: ctx[name] if name in ctx else z["init_" + name].astype(np.float32)

    for nd in meta["nodes"]:
        op, ins, out = nd["op"], nd["inputs"], nd["outputs"][0]
        a = nd["attrs"]
        if op == "Im2Col":
            k = a["kernel_size"]
            s = a["stride"]
            pad = a.get("pad_amount", [0, 0, 0, 0])
            ctx[out] = im2col(get(ins[0]), k[0], k[1], s[0], s[1], pad,
                              float(a.get("pad_value", 0)))
        elif op == "MatMul":
            ctx[out] = get(ins[0]) @ get(ins[1])
        elif op == "MultiThreshold":
            ctx[out] = multithreshold(get(ins[0]), get(ins[1]),
                                      float(a.get("out_scale", 1.0)),
                                      float(a.get("out_bias", 0.0)),
                                      a.get("data_layout"))
        elif op == "MaxPoolNHWC":
            ctx[out] = maxpool_nhwc(get(ins[0]), a["kernel_shape"][0], a["strides"][0])
        elif op == "Pad":
            pads = z["init_" + ins[1]].astype(int)
            n_dims = get(ins[0]).ndim
            pw = [(pads[i], pads[i + n_dims]) for i in range(n_dims)]
            ctx[out] = np.pad(get(ins[0]), pw, constant_values=0)
        elif op == "Resize":
            x = get(ins[0])
            sc = z["init_" + ins[2]] if len(ins) > 2 and ins[2] else z["init_" + ins[-1]]
            fy, fx = int(round(float(sc[1]))), int(round(float(sc[2])))
            ctx[out] = x.repeat(fy, axis=1).repeat(fx, axis=2)
        elif op == "Concat":
            ctx[out] = np.concatenate([get(i) for i in ins], axis=int(a["axis"]))
        elif op == "Transpose":
            ctx[out] = get(ins[0]).transpose(a["perm"])
        elif op == "Mul":
            ctx[out] = get(ins[0]) * get(ins[1])
        elif op == "Add":
            ctx[out] = get(ins[0]) + get(ins[1])
        else:
            raise NotImplementedError(op)
    return [ctx[o] for o in meta["outputs"]]


if __name__ == "__main__":
    # selftest PC: PL simulada = ejecutar bb_tidy con qonnx; comparar la cola
    # numpy contra la ejecucion qonnx del grafo COMPLETO (ints exactos).
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.core.onnx_exec import execute_onnx

    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (1, 3, 320, 320)).astype(np.float32)

    bb = ModelWrapper("finn/bb_tidy.onnx")
    bctx = execute_onnx(bb, {bb.graph.input[0].name: x})
    pl_outs = [bctx[o.name] for o in bb.graph.output]
    for o in pl_outs:
        print("PL sim:", o.shape, "rango", o.min(), o.max())

    z = np.load("finn/tail.npz", allow_pickle=True)
    outs = run_tail(z, pl_outs)

    full = ModelWrapper("finn/ocu_tidy.onnx")
    fctx = execute_onnx(full, {full.graph.input[0].name: x}, return_full_exec_context=True)
    meta = json.loads(str(z["meta"]))
    ok = True
    for o, mine in zip(meta["outputs"], outs):
        refv = fctx[o]
        m = mine
        if m.shape != refv.shape and m.ndim == 4:
            m = m.transpose(0, 3, 1, 2)
        same = m.shape == refv.shape and np.array_equal(m, refv)
        ok &= same
        print(f"  {o}: {'EXACTO' if same else 'DIFIERE'} {m.shape} vs {refv.shape}")
    raise SystemExit(0 if ok else 1)
