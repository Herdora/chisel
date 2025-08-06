#!/usr/bin/env python3
"""
Timing test script to measure actual upload times for large files.

This script helps collect real upload timing data for documentation.
Run this with different file sizes to get actual measurements.
"""

import time
import os
import subprocess
import sys
from pathlib import Path


def measure_upload_time(filename, app_name_suffix=""):
    """
    Measure the time it takes to upload a file using chisel.

    Args:
        filename: Path to the file to upload
        app_name_suffix: Suffix for app name to avoid conflicts

    Returns:
        tuple: (success, upload_time_seconds, file_size_bytes)
    """
    if not os.path.exists(filename):
        print(f"❌ File {filename} not found")
        return False, 0, 0

    file_size = os.path.getsize(filename)
    size_gb = file_size / (1024 * 1024 * 1024)
    app_name = f"timing-test-{size_gb:.1f}gb{app_name_suffix}"

    print(f"📊 Testing upload time for {filename}")
    print(f"📏 File size: {file_size:,} bytes ({size_gb:.2f}GB)")
    print(f"🏷️ App name: {app_name}")

    # Measure upload time
    start_time = time.time()

    try:
        # Run chisel command and capture output
        cmd = ["chisel", "python", "test_cached_file_fix.py", "--app-name", app_name, "--gpu", "1"]

        print(f"🚀 Starting upload at {time.strftime('%H:%M:%S')}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

        end_time = time.time()
        upload_time = end_time - start_time

        if result.returncode == 0:
            print(f"✅ Upload completed successfully")
            print(f"⏱️ Upload time: {upload_time:.1f} seconds ({upload_time / 60:.1f} minutes)")
            return True, upload_time, file_size
        else:
            print(f"❌ Upload failed with return code {result.returncode}")
            print(f"Error output: {result.stderr}")
            return False, upload_time, file_size

    except subprocess.TimeoutExpired:
        print(f"❌ Upload timed out after 30 minutes")
        return False, 1800, file_size
    except Exception as e:
        end_time = time.time()
        upload_time = end_time - start_time
        print(f"❌ Upload failed with exception: {e}")
        return False, upload_time, file_size


def run_timing_tests():
    """Run timing tests on available large files."""
    print("🧪 Chisel Upload Timing Tests")
    print("=" * 50)

    # Look for test files
    test_files = []
    for pattern in ["*.npy", "large_*.bin", "massive_*.txt"]:
        test_files.extend(Path(".").glob(pattern))

    if not test_files:
        print("❌ No large test files found")
        print("💡 Create test files with: python create_test_files.py --action dataset")
        return

    results = []

    for test_file in sorted(test_files, key=lambda f: f.stat().st_size):
        print(f"\n" + "=" * 50)

        # Test first upload (caching)
        print(f"🔄 Testing FIRST upload (caching)...")
        success1, time1, size = measure_upload_time(str(test_file), "-first")

        if success1:
            # Test second upload (cached)
            print(f"\n🔄 Testing SECOND upload (cached)...")
            success2, time2, _ = measure_upload_time(str(test_file), "-cached")

            if success2:
                speedup = time1 / time2 if time2 > 0 else float("inf")
                results.append(
                    {
                        "file": test_file.name,
                        "size_gb": size / (1024**3),
                        "first_time": time1,
                        "cached_time": time2,
                        "speedup": speedup,
                    }
                )

        print(f"\n⏸️ Waiting 10 seconds before next test...")
        time.sleep(10)

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"📊 UPLOAD TIMING RESULTS SUMMARY")
    print(f"=" * 60)

    if results:
        print(
            f"{'File':<25} {'Size (GB)':<10} {'First (min)':<12} {'Cached (sec)':<13} {'Speedup':<8}"
        )
        print(f"-" * 70)

        for result in results:
            print(
                f"{result['file']:<25} "
                f"{result['size_gb']:<10.2f} "
                f"{result['first_time'] / 60:<12.1f} "
                f"{result['cached_time']:<13.1f} "
                f"{result['speedup']:<8.1f}x"
            )

        # Calculate averages
        avg_first = sum(r["first_time"] for r in results) / len(results)
        avg_cached = sum(r["cached_time"] for r in results) / len(results)
        avg_speedup = sum(r["speedup"] for r in results) / len(results)

        print(f"-" * 70)
        print(
            f"{'AVERAGE':<25} "
            f"{'N/A':<10} "
            f"{avg_first / 60:<12.1f} "
            f"{avg_cached:<13.1f} "
            f"{avg_speedup:<8.1f}x"
        )
    else:
        print("❌ No successful timing results collected")

    print(f"\n💡 Use these results to update documentation with real timing data!")


if __name__ == "__main__":
    run_timing_tests()
