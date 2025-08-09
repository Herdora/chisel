#!/usr/bin/env python3
"""
Simple CNN example using capture_model decorator.
This demonstrates basic convolutional neural network profiling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chisel import capture_model_class


@capture_model_class(model_name="SimpleCNN")
class SimpleCNN(nn.Module):
    """Basic CNN for image classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # Calculate the size for the first linear layer
        # For 32x32 input: after 3 pooling operations -> 4x4x128
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def main():
    """Test the CNN model with profiling."""
    print("🚀 Testing Simple CNN Model")
    print("=" * 40)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Create model and move to device
    model = SimpleCNN(num_classes=10).to(device)

    # Create sample input (batch_size=8, channels=3, height=32, width=32)
    batch_size = 8
    sample_input = torch.randn(batch_size, 3, 32, 32).to(device)

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Input shape: {sample_input.shape}")

    # Run inference (this will be profiled by capture_model)
    model.eval()
    with torch.no_grad():
        print("\n🔄 Running forward pass...")
        output = model(sample_input)

        print(f"✅ Output shape: {output.shape}")
        print(f"📊 Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

        # Run a few more passes to test profiling
        print("\n🔄 Running additional forward passes...")
        for i in range(3):
            output = model(sample_input)
            print(f"  Pass {i + 2}: Output shape {output.shape}")

    print("\n✅ Simple CNN example completed!")


if __name__ == "__main__":
    main()
