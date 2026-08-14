import numpy as np

x = np.load("input_known.npy")

print("Original:", x.shape, x.dtype)

x = np.transpose(x, (0, 2, 3, 1))
x = x.astype(np.int8)

print("Convertida:", x.shape, x.dtype)

np.save("input_known_nhwc_int8.npy", x)
