import numpy as np
from ctypes import CDLL, c_double, byref
from mlir.runtime import make_nd_memref_descriptor, get_ranked_memref_descriptor, ranked_memref_to_numpy
from nbcc.compiler import compile_shared_lib



libname = "llm_tensor.so"
compile_shared_lib("llm_tensor.spy", libname)


lib = CDLL(libname)
export_function = getattr(lib, "_mlir_ciface_spy_llm_tensor$export_softmax")
print(export_function)

# RUN

DIM0 = 6
DIM1 = 13
memref_2d_f64 = make_nd_memref_descriptor(2, c_double)

A = np.arange(DIM0 * DIM1, dtype=np.float64).reshape(DIM0, DIM1)
# B = np.arange(DIM0 * DIM1, dtype=np.float64).reshape(DIM0, DIM1)

argA = get_ranked_memref_descriptor(A)
# argB = get_ranked_memref_descriptor(B)

out_memref = (memref_2d_f64 * 1)()
# args = [out_memref, byref(argA), byref(argB)]
args = [out_memref, byref(argA)]
export_function(*args)

output = ranked_memref_to_numpy(out_memref)
print(output)

def softmax(A):
    exp_x = np.exp(A - A.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

np.testing.assert_allclose(output, softmax(A))