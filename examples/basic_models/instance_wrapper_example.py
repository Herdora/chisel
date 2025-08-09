#!/usr/bin/env python3
"""
Simple example demonstrating capture_model_instance functionality.

This shows how to wrap an existing PyTorch model instance (rather than decorating the class)
to automatically profile every forward pass.
"""

import torch
import torch.nn as nn
from chisel import capture_model_instance


def create_simple_model():
    """Create a basic PyTorch model instance."""
    model = nn.Sequential(
        nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
    )
    return model


def main():
    print("🔧 Creating model instance...")

    # Step 1: Create a regular PyTorch model instance
    model = create_simple_model()
    print(f"   Original model: {type(model).__name__}")

    # Step 2: Wrap the instance with capture_model_instance
    # This will profile every forward() call automatically
    model = capture_model_instance(
        model, model_name="SimpleSequential", record_shapes=True, profile_memory=True
    )
    print(f"   Wrapped model: {type(model).__name__}")

    # Step 3: Use the model normally - profiling happens automatically
    print("\n🚀 Running forward passes...")

    # Create some sample data
    batch_sizes = [1, 4, 8]

    for i, batch_size in enumerate(batch_sizes, 1):
        print(f"   Pass {i}: batch_size={batch_size}")

        # Generate random input
        x = torch.randn(batch_size, 10)

        # Forward pass - this will be automatically profiled
        with torch.no_grad():
            output = model(x)

        print(f"   Output shape: {output.shape}")

    print("\n✅ All forward passes completed!")
    print("   Each forward pass was automatically profiled and saved as a trace file.")
    print(
        "   Trace files: SimpleSequential_forward_001.json, SimpleSequential_forward_002.json, etc."
    )


if __name__ == "__main__":
    main()
