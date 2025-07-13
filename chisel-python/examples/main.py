import ctypes
import numpy as np
import os

# Load the shared library
lib_path = os.path.join(os.path.dirname(__file__), "libvector_add.so")
lib = ctypes.cdll.LoadLibrary(lib_path)

# Set function signature
lib.launch_vector_add.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]

# Test input
N = 1024
A = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)
C = np.zeros_like(A)

# Launch kernel
lib.launch_vector_add(A, B, C, N)

# Check result
print("Result OK:", np.allclose(C, A + B))
