#!/usr/bin/env python3
"""
Model with command line arguments example.
Tests: kandc python model_with_args.py --model-size large --batch-size 16 --num-layers 5
"""

import argparse
import torch
import torch.nn as nn
from kandc import capture_model_class


@capture_model_class(model_name="ConfigurableModel")
class ConfigurableModel(nn.Module):
    """Model that can be configured via command line arguments."""

    def __init__(self, input_size=100, hidden_size=128, num_layers=3, output_size=10):
        super().__init__()

        layers = []
        prev_size = input_size

        for i in range(num_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.network(x)


def create_model_from_args(args):
    """Create model based on command line arguments."""

    # Map model size to hidden dimensions
    size_map = {"small": 64, "medium": 128, "large": 256, "xl": 512}

    hidden_size = size_map.get(args.model_size, 128)

    print(f"🏗️  Creating model with:")
    print(f"   Model size: {args.model_size} (hidden_size={hidden_size})")
    print(f"   Number of layers: {args.num_layers}")
    print(f"   Batch size: {args.batch_size}")

    model = ConfigurableModel(
        input_size=100, hidden_size=hidden_size, num_layers=args.num_layers, output_size=10
    )

    return model


def run_training_simulation(model, batch_size, num_epochs, device):
    """Simulate training with the configured model."""

    print(f"\n🏃 Running training simulation:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {batch_size}")

    model.eval()  # Just for testing, not actual training

    for epoch in range(num_epochs):
        print(f"\n📊 Epoch {epoch + 1}/{num_epochs}")

        # Simulate multiple batches
        for batch_idx in range(3):  # 3 batches per epoch
            # Generate random batch
            x = torch.randn(batch_size, 100).to(device)

            with torch.no_grad():
                output = model(x)

            # Simulate loss calculation
            fake_loss = torch.randn(1).abs().item()

            print(
                f"   Batch {batch_idx + 1}: Loss = {fake_loss:.4f}, Output shape = {output.shape}"
            )


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Configurable Model Example")

    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["small", "medium", "large", "xl"],
        default="medium",
        help="Size of the model (default: medium)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of hidden layers (default: 3)"
    )

    # Training configuration
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for training (default: 8)"
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default: 2)")

    # Debugging
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print("🚀 Model with Arguments Test")
    print("=" * 40)

    if args.verbose:
        print("🔍 Verbose mode enabled")
        print(f"📋 All arguments: {vars(args)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Create model from arguments
    model = create_model_from_args(args).to(device)

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test single forward pass
    print(f"\n🧪 Testing single forward pass:")
    test_input = torch.randn(args.batch_size, 100).to(device)

    with torch.no_grad():
        output = model(test_input)
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Run training simulation
    run_training_simulation(model, args.batch_size, args.epochs, device)

    print("\n✅ Model with arguments test completed!")

    if args.verbose:
        print(f"🎯 Final model info:")
        print(f"   Hidden size: {model.hidden_size}")
        print(f"   Number of layers: {model.num_layers}")


if __name__ == "__main__":
    main()
