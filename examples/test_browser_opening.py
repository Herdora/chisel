"""
Test script to verify browser opening functionality.

This is a minimal script that can be used to test the automatic browser opening
feature when a Chisel job is submitted.

Usage:
    chisel run test_browser_opening.py

Expected behavior:
1. Job submits successfully
2. Browser automatically opens to the job's page
3. Success message shows browser opened successfully
"""

import time
from chisel import capture_trace


@capture_trace(trace_name="browser_test")
def test_browser_opening():
    """Simple test function that runs quickly."""
    print("🧪 Testing browser opening functionality")
    print("=" * 50)

    print("✅ Script started successfully")
    print("⏱️  Running for 5 seconds...")

    # Simple computation to keep the job running briefly
    result = 0
    for i in range(1000000):
        result += i * 0.001

    print(f"🔢 Computation result: {result:.2f}")
    print("✅ Script completed successfully")

    return {"test_result": "browser_opening_test_completed", "computation_result": result}


def main():
    """Main function."""
    print("🚀 Browser Opening Test Script")
    print("=" * 60)

    result = test_browser_opening()

    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print("  • This script tests automatic browser opening")
    print("  • After submission, your browser should open automatically")
    print("  • You should see the job running in the Chisel dashboard")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
