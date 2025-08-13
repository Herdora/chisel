#!/usr/bin/env python3
import os
import sys
import platform
import ctypes
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIBNAME = "libgpucheck.dylib" if platform.system() == "Darwin" else "libgpucheck.so"
LIBPATH = os.path.join(HERE, "build", LIBNAME)

if not os.path.exists(LIBPATH):
    print(f"❌ Library not found: {LIBPATH}\n   Run: make", flush=True)
    sys.exit(1)

lib = ctypes.CDLL(LIBPATH)
lib.kandc_gpu_available.restype = ctypes.c_int
lib.kandc_gpu_name.restype = ctypes.c_char_p

available = bool(lib.kandc_gpu_available())
name = lib.kandc_gpu_name().decode() if available else ""

print(f"GPU available: {available}", flush=True)
if name:
    print(f"GPU name: {name}", flush=True)

# Emit a few lines to demonstrate streaming with kandc capture
for i in range(5):
    print(f"[check_gpu] step {i+1}/5", flush=True)
    time.sleep(0.3)

sys.exit(0)


