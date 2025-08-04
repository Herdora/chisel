#!/usr/bin/env python3
"""
Test script to verify cached file restoration is working.
This script will only succeed if large_test_file.npy exists and can be loaded.
"""

import os
import numpy as np
from pathlib import Path



def main():
    """Test if cached file restoration works."""
    print("🧪 Testing Cached File Restoration Fix")
    print("=" * 50)

    # Check current directory
    print(f"📍 Current directory: {os.getcwd()}")
    print(f"📋 Files in directory: {list(os.listdir('.'))}")

    # Check for the specific file we expect
    target_file = "large_test_file.npy"

    if not os.path.exists(target_file):
        print(f"❌ FAIL: {target_file} not found!")
        print("🔍 This means cached file restoration is NOT working")
        exit(1)

    print(f"✅ FOUND: {target_file} exists")

    # Check file size
    file_size = os.path.getsize(target_file)
    size_gb = file_size / (1024 * 1024 * 1024)
    print(f"📏 File size: {file_size:,} bytes ({size_gb:.2f}GB)")

    # Try to load the file
    try:
        print(f"🔄 Attempting to load {target_file}...")
        data = np.load(target_file)
        print(f"✅ SUCCESS: Loaded array with shape {data.shape}")
        print(f"📊 Data type: {data.dtype}")
        print(f"💾 Memory usage: ~{data.nbytes / (1024 * 1024):.1f}MB")

        # Basic validation
        if len(data) > 1000000:  # Should be a large array
            print(f"✅ SUCCESS: File contains expected large dataset")
            print(f"🎯 First few values: {data[:5]}")
            print(f"📈 Mean value: {np.mean(data[:1000]):.6f}")

            print("\n" + "=" * 50)
            print("🎉 CACHED FILE RESTORATION IS WORKING!")
            print("✅ The fix successfully restored the large file from Modal Volume")
            print("=" * 50)
            return True
        else:
            print(f"❌ FAIL: File is too small, might be corrupted")
            return False

    except Exception as e:
        print(f"❌ FAIL: Could not load {target_file}: {e}")
        print("🔍 This means the file exists but is corrupted or incomplete")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
