#!/usr/bin/env python3
"""
Deeply nested model example.
Tests running models from deeply nested directory structures.
Run from examples/: chisel python nested_directory/deep/deeper/deepest_model.py
"""

import torch
import torch.nn as nn
import os
import sys

# Add the examples directory to the path so we can import from other examples
examples_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, examples_dir)

from chisel import capture_model


@capture_model(model_name="DeepestModel")
class DeepestModel(nn.Module):
    """A model that lives in the deepest directory."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        return self.layers(x)


def show_path_info():
    """Show information about current path and file location."""
    print("📁 Path Information:")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Script file: {__file__}")
    print(f"   Script directory: {os.path.dirname(__file__)}")
    print(f"   Relative to examples: {os.path.relpath(__file__, examples_dir)}")


def main():
    """Test model from deeply nested directory."""
    print("🚀 Deeply Nested Model Test")
    print("=" * 40)

    show_path_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎯 Using device: {device}")

    # Create model
    model = DeepestModel().to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with different batch sizes
    batch_sizes = [1, 4, 16]

    for batch_size in batch_sizes:
        print(f"\n🧪 Testing with batch size: {batch_size}")

        # Create input
        x = torch.randn(batch_size, 50).to(device)

        model.eval()
        with torch.no_grad():
            output = model(x)

        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print("\n✅ Deeply nested model test completed!")
    print("💡 This tests running models from nested directory structures")


if __name__ == "__main__":
    main()
