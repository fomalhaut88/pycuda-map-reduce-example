import numpy as np
import pycuda.autoinit
from pycuda import gpuarray, elementwise

x = np.arange(0, 1001, dtype=np.uint32)
y = np.zeros(1001, np.uint32)
kernel = elementwise.ElementwiseKernel(
    arguments="unsigned int* x, int* y",
    operation= "y[i] = x[i] * x[i]",
)
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
kernel(x_gpu, y_gpu)

print(y_gpu.get())
