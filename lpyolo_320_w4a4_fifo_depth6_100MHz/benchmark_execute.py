import time
import numpy as np

from driver import io_shape_dict
from driver_base import FINNExampleOverlay
from pynq.pl_server.device import Device


BATCH_SIZE = 1

print("Cargando bitstream e inicializando driver...")

accel = FINNExampleOverlay(
    bitfile_name="finn-accel.bit",
    platform="zynq-iodma",
    io_shape_dict=io_shape_dict,
    batch_size=BATCH_SIZE,
    runtime_weight_dir="runtime_weights/",
    device=Device.devices[0],
)

x = np.load("input_known_nhwc.npy")

# Primera ejecución de calentamiento
_ = accel.execute(x)

times = []

for i in range(10):
    t0 = time.perf_counter()
    y = accel.execute(x)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000
    times.append(elapsed_ms)
    print(f"Ejecución {i + 1:2d}: {elapsed_ms:.3f} ms")

print()
print(f"Media   : {np.mean(times):.3f} ms")
print(f"Mínimo  : {np.min(times):.3f} ms")
print(f"Máximo  : {np.max(times):.3f} ms")
print(f"FPS E2E : {1000.0 / np.mean(times):.3f}")

np.save("output_fpga_fast_benchmark.npy", y)
