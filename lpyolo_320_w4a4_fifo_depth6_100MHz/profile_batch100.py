import time
import numpy as np

from driver import io_shape_dict
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device


def measure(name, function):
    t0 = time.perf_counter()
    result = function()
    t1 = time.perf_counter()
    print(f"{name:32s}: {(t1 - t0) * 1000:10.3f} ms")
    return result


BATCH_SIZE = 100

print("Inicializando FPGA...")
t0 = time.perf_counter()

accel = FINNExampleOverlay(
    bitfile_name="finn-accel.bit",
    platform="zynq-iodma",
    io_shape_dict=io_shape_dict,
    batch_size=BATCH_SIZE,
    runtime_weight_dir="runtime_weights/",
    device=Device.devices[0],
)

t1 = time.perf_counter()
print(f"{'Inicialización overlay':32s}: {(t1 - t0) * 1000:10.3f} ms")

x = measure(
    "Carga input_batch100.npy",
    lambda: np.load("input_batch100.npy"),
)

print("Input:", x.shape, x.dtype)

folded = measure(
    "fold_input",
    lambda: accel.fold_input(x),
)

packed = measure(
    "pack_input",
    lambda: accel.pack_input(folded),
)

measure(
    "copy_input_data_to_device",
    lambda: accel.copy_input_data_to_device(packed),
)

measure(
    "FPGA + DMA",
    lambda: accel.execute_on_buffers(),
)

measure(
    "copy_output_data_from_device",
    lambda: accel.copy_output_data_from_device(accel.obuf_packed[0]),
)

unpacked = measure(
    "unpack_output rápido",
    lambda: accel.unpack_output(accel.obuf_packed[0]),
)

normal = measure(
    "unfold_output",
    lambda: accel.unfold_output(unpacked),
)

measure(
    "Guardar output",
    lambda: np.save("output_batch100_profile.npy", normal),
)

print("Output:", normal.shape, normal.dtype)
