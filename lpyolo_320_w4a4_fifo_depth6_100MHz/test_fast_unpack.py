import time
import numpy as np

from driver import io_shape_dict
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device


BITFILE = "finn-accel.bit"
INPUT_FILE = "input_known_nhwc.npy"
BATCH_SIZE = 1


accel = FINNExampleOverlay(
    bitfile_name=BITFILE,
    platform="zynq-iodma",
    io_shape_dict=io_shape_dict,
    batch_size=BATCH_SIZE,
    runtime_weight_dir="runtime_weights/",
    device=Device.devices[0],
)

# ---------------------------------------------------------
# Preparar y ejecutar exactamente como execute()
# ---------------------------------------------------------

input_normal = np.load(INPUT_FILE)

assert input_normal.shape == accel.ishape_normal(), (
    f"Shape incorrecto: {input_normal.shape}, "
    f"esperado: {accel.ishape_normal()}"
)

input_folded = accel.fold_input(input_normal)
input_packed = accel.pack_input(input_folded)
accel.copy_input_data_to_device(input_packed)

print("Ejecutando acelerador...")
accel.execute_on_buffers()

# Copiar salida RAW desde el buffer DMA
accel.copy_output_data_from_device(accel.obuf_packed[0])

raw = accel.obuf_packed[0]

print("Raw shape:", raw.shape)
print("Raw dtype:", raw.dtype)
print("Raw min/max:", raw.min(), raw.max())

# ---------------------------------------------------------
# Método original de FINN
# ---------------------------------------------------------

t0 = time.perf_counter()
original_folded = accel.unpack_output(raw)
t1 = time.perf_counter()

original_normal = accel.unfold_output(original_folded)

# ---------------------------------------------------------
# Método rápido propuesto para UINT4
# Cada salida usa los 4 bits menos significativos del byte
# ---------------------------------------------------------

t2 = time.perf_counter()

fast_normal = np.bitwise_and(raw, 0x0F).reshape(
    accel.oshape_normal()
).astype(np.float32)

t3 = time.perf_counter()

# ---------------------------------------------------------
# Comparación
# ---------------------------------------------------------

same = np.array_equal(original_normal, fast_normal)
max_error = np.max(
    np.abs(
        original_normal.astype(np.int16)
        - fast_normal.astype(np.int16)
    )
)

print()
print("============================================")
print("RESULTADOS")
print("============================================")
print(f"FINN unpack : {(t1 - t0) * 1000:.3f} ms")
print(f"Fast unpack : {(t3 - t2) * 1000:.3f} ms")
print(f"Salidas iguales: {same}")
print(f"Error máximo   : {max_error}")
print("Shape original :", original_normal.shape)
print("Shape rápida   :", fast_normal.shape)
print("Primeros valores originales:")
print(original_normal.reshape(-1)[:32])
print("Primeros valores rápidos:")
print(fast_normal.reshape(-1)[:32])

np.save("output_original_unpack.npy", original_normal)
np.save("output_fast_unpack.npy", fast_normal)
np.save("output_raw_dma.npy", np.asarray(raw))

if not same:
    raise RuntimeError(
        "El desempaquetado rápido no coincide. "
        "No se debe modificar todavía el driver."
    )

print()
print("VALIDACIÓN CORRECTA: puede utilizarse el método rápido.")
