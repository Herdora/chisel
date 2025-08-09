#!/usr/bin/env python3
"""
Basic model call example - tests the simplest case of using capture_model.
This is the most basic edge case to ensure everything works.
"""

import torch
import torch.nn as nn
from chisel import capture_model_class


@capture_model_class(model_name="BasicModel")
class BasicModel(nn.Module):
    """The simplest possible model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


def main():
    """Test the most basic model call."""
    print("🚀 Basic Model Call Test")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Create model
    model = BasicModel().to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Create simple input
    x = torch.randn(5, 10).to(device)
    print(f"📊 Input shape: {x.shape}")

    # Single forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"✅ Output shape: {output.shape}")
        print(f"📊 Output: {output.squeeze().tolist()}")

    print("✅ Basic model call completed!")


if __name__ == "__main__":
    main()
