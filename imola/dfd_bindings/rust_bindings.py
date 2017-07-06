from cffi import FFI
import numpy as np
import os
import platform


ffi = FFI()
ffi.cdef(
    "double dfd(const double *p, const double *q,"
    "size_t len_p, size_t len_q, size_t num_dim);"
    )
current_dir = os.path.dirname(__file__)
if platform.system() == "Linux":
    dylib_path = "{:s}/target/release/libdfd_bindings.so".format(current_dir)
elif platform.system() == "Windows":
    dylib_path = "{:s}/target/release/dfd_bindings.dll".format(current_dir)
else:
    raise ValueError("Unsupported OS")
C = ffi.dlopen(dylib_path)


def dfd(p, q):
    p, q = np.array(p), np.array(q)
    num_dim = p.shape[0]
    p, q = p.ravel(order="F"), q.ravel(order="F")
    p_c = ffi.cast("double *", p.ctypes.data)
    q_c = ffi.cast("double *", q.ctypes.data)
    return C.dfd(p_c, q_c, len(p), len(q), num_dim)
