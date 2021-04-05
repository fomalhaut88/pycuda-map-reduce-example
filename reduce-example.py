import numpy as np
import pycuda.autoinit
from pycuda import gpuarray, reduction

x = np.arange(0, 1001, dtype=np.uint32)
kernel = reduction.ReductionKernel(
    dtype_out=np.float32,
    arguments="unsigned int* x",
    map_expr= "(float)x[i] * x[i]",
    reduce_expr="a + b",
    neutral="0.0",
)
x_gpu = gpuarray.to_gpu(x)
result = kernel(x_gpu).get()

print(result)
