#!/usr/bin/env python3
"""
Test script for model profiling functionality.

This script tests the capture_model decorator to ensure it works correctly
both locally and on the Chisel backend.
"""

import torch
import torch.nn as nn
from chisel import capture_model, parse_model_trace
import os
import time


@capture_model(model_name="TestModel")
class TestModel(nn.Module):
    """Simple test model for profiling verification."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(100, 50)
        self.linear2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


def test_model_profiling():
    """Test the model profiling functionality."""
    print("🧪 Testing Model Profiling")
    print("=" * 40)

    # Check environment
    is_backend = os.environ.get("CHISEL_BACKEND_RUN") == "1"
    print(f"🔍 Running on Chisel backend: {is_backend}")

    # Create model and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    model = TestModel().to(device)
    data = torch.randn(32, 100).to(device)

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Input shape: {data.shape}")

    # Test forward pass
    print("\n🔄 Running forward pass...")
    with torch.no_grad():
        output = model(data)

    print(f"✅ Output shape: {output.shape}")
    print(f"📊 Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test multiple forward passes
    print("\n🔄 Running 2 more forward passes...")
    for i in range(2):
        with torch.no_grad():
            output = model(data)
        print(f"  Pass {i + 2}: output shape {output.shape}")

    print("\n✅ Model profiling test completed!")

    # Test trace parsing if we have trace files
    test_trace_parsing()

    return model


def test_trace_parsing():
    """Test the trace parsing functionality."""
    print("\n📊 Testing Trace Parsing")
    print("=" * 40)

    from pathlib import Path

    # Look for trace files
    trace_files = list(Path(".").glob("*.json"))

    if trace_files:
        print(f"🔍 Found {len(trace_files)} trace files:")
        for trace_file in trace_files:
            print(f"  📄 {trace_file}")

            # Test parsing
            try:
                analysis = parse_model_trace(str(trace_file), "TestModel")
                if analysis:
                    layer_count = len(analysis["layer_stats"])
                    print(f"  ✅ Parsed successfully: {layer_count} layers")

                    # Print a quick summary
                    total_time = (
                        sum(stats["total_time_us"] for stats in analysis["layer_stats"].values())
                        / 1000
                    )
                    print(f"  📊 Total model time: {total_time:.2f}ms")
                else:
                    print("  ⚠️  Could not parse trace")
            except Exception as e:
                print(f"  ❌ Error parsing: {e}")
    else:
        print("ℹ️  No trace files found (this is normal for local testing)")


def test_imports():
    """Test that all required imports work."""
    print("🔍 Testing Imports")
    print("=" * 40)

    try:
        from chisel import capture_model

        print("✅ capture_model imported successfully")

        # Test parse_model_trace import
        try:
            from chisel import parse_model_trace

            print("✅ parse_model_trace imported successfully")
        except ImportError:
            print("⚠️  parse_model_trace not available (expected for simplified API)")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    return True


def main():
    """Main test function."""
    print("🚀 Model Profiling Test Suite")
    print("=" * 50)

    # want to print a bunch of stuff and sleep for .5 seconds in between each print
    for i in range(100):
        print(f"🔄 Running test {i + 1}...")
        time.sleep(0.5)

    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return

    # Test model profiling
    try:
        test_model_profiling()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
