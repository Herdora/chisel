# examples/hip_vector_add_test.py
import ctypes, os, numpy as np

lib = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), "libvector_add_hip.so"))

lib.launch_vector_add_hip.argtypes = [
    np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
]

N = 1_024
A = np.random.rand(N).astype(np.float32)
B = np.random.rand(N).astype(np.float32)
C = np.empty_like(A)

lib.launch_vector_add_hip(A, B, C, N)
print("Result OK:", np.allclose(C, A + B))
