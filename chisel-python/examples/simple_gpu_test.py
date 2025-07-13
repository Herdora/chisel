#!/usr/bin/env python3
"""
Simple test script for ROCm profiling that doesn't require PyTorch.
This helps isolate whether the issue is with rocprofv3 or PyTorch.
"""

import os
import time


def main():
    print("Starting simple GPU test...")
    print(f"Python version: {os.sys.version}")

    # Test basic math operations
    result = 0
    for i in range(1000000):
        result += i * i

    print(f"Computed result: {result}")

    # Check for ROCm environment
    rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
    print(f"ROCm path: {rocm_path}")
    print(f"ROCm exists: {os.path.exists(rocm_path)}")

    # Check for GPU devices
    try:
        devices = os.listdir("/dev/kfd")
        print(f"KFD devices: {devices}")
    except:
        print("No KFD devices found")

    # Simulate some work
    time.sleep(0.1)
    print("Simple test completed successfully!")


if __name__ == "__main__":
    main()
