#!/usr/bin/env python3
"""
Simple linear regression model using capture_model decorator.
This demonstrates basic linear model profiling.
"""

import torch
import torch.nn as nn
from chisel import capture_model_class


@capture_model_class(model_name="LinearRegression")
class LinearRegression(nn.Module):
    """Simple linear regression model."""

    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


@capture_model_class(model_name="MultiLayerPerceptron")
class MLP(nn.Module):
    """Multi-layer perceptron for comparison."""

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def generate_synthetic_data(n_samples=1000, n_features=20, device="cpu"):
    """Generate synthetic regression data."""
    print(f"📊 Generating {n_samples} samples with {n_features} features")

    # Generate random features
    X = torch.randn(n_samples, n_features, device=device)

    # Generate random weights for true relationship
    true_weights = torch.randn(n_features, device=device)

    # Generate target with some noise
    y = X @ true_weights + 0.1 * torch.randn(n_samples, device=device)

    return X, y.unsqueeze(1), true_weights


def test_models():
    """Test both linear regression and MLP models."""
    print("🚀 Testing Linear Models")
    print("=" * 40)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # Generate data
    X, y, true_weights = generate_synthetic_data(n_samples=5000, n_features=50, device=device)

    print(f"📊 Data shapes: X={X.shape}, y={y.shape}")

    # Test Linear Regression
    print("\n🔍 Testing Linear Regression Model")
    print("-" * 30)

    lr_model = LinearRegression(input_dim=50).to(device)
    print(f"📊 LR parameters: {sum(p.numel() for p in lr_model.parameters()):,}")

    with torch.no_grad():
        for i in range(3):
            pred = lr_model(X)
            loss = nn.MSELoss()(pred, y)
            print(f"  Forward pass {i + 1}: Loss = {loss.item():.6f}")

    # Test MLP
    print("\n🔍 Testing Multi-Layer Perceptron")
    print("-" * 30)

    mlp_model = MLP(input_dim=50, hidden_dims=[128, 64, 32], output_dim=1).to(device)
    print(f"📊 MLP parameters: {sum(p.numel() for p in mlp_model.parameters()):,}")

    with torch.no_grad():
        for i in range(3):
            pred = mlp_model(X)
            loss = nn.MSELoss()(pred, y)
            print(f"  Forward pass {i + 1}: Loss = {loss.item():.6f}")

    print("\n✅ Linear models testing completed!")


if __name__ == "__main__":
    test_models()
