#!/usr/bin/env python3
"""
Complete kandc Example
======================

This example demonstrates all key features of the kandc library in a single run:
- Model profiling with both @capture_model_class and capture_model_instance
- Timing utilities
- Logging and metrics

Requirements:
    pip install torch kandc

Usage:
    python complete_example.py
"""

import torch
import torch.nn as nn
import kandc
import time


class SimpleNet(nn.Module):
    """A simple neural network for demonstration."""

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.layers(x)


@kandc.capture_model_class(model_name="ProfiledNet", record_shapes=True, profile_memory=True)
class ProfiledNet(nn.Module):
    """A neural network with built-in profiling using @capture_model_class decorator."""

    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.layers(x)


@kandc.timed("data_preparation")
def prepare_data(batch_size=32, input_size=784):
    """Prepare synthetic training data."""
    # Generate synthetic data
    X = torch.randn(batch_size, input_size)
    y = torch.randint(0, 10, (batch_size,))

    return X, y


@kandc.timed("model_inference")
def run_inference(model, X, model_name="model"):
    """Run model inference and return results."""
    with torch.no_grad():
        # Forward pass (this will be profiled)
        outputs = model(X)
        predictions = torch.argmax(outputs, dim=1)

    return outputs, predictions


@kandc.timed("expensive_computation")
def expensive_computation(n=1000):
    """Simulate expensive computation for timing demo."""
    result = 0
    for i in range(n):
        result += i**2
    time.sleep(0.05)  # Simulate work
    return result


def main():
    """Single comprehensive example demonstrating all kandc features."""
    print("🚀 kandc Complete Example")

    run = kandc.init(
        project="kandc-complete-demo",
        name="all-features-demo",
        capture_source=True,
        config={
            "batch_size": 32,
            "hidden_size": 128,
            "input_size": 784,
            "demo_type": "comprehensive",
        },
        tags=["example", "pytorch", "complete-demo", "all-features"],
    )

    try:
        # === 1. Model Profiling with capture_model_instance ===
        print("1️⃣  Testing capture_model_instance")

        model1 = SimpleNet(input_size=784, hidden_size=128, output_size=10)
        wrapped_model1 = kandc.capture_model_instance(
            model1, model_name="SimpleNet", record_shapes=True, profile_memory=True
        )

        # Prepare data (timed)
        X, y = prepare_data(batch_size=32)

        # Run inference (timed and profiled)
        outputs1, predictions1 = run_inference(wrapped_model1, X, "SimpleNet")

        # === 2. Model Profiling with @capture_model_class ===
        print("2️⃣  Testing @capture_model_class decorator")

        # Create model with built-in profiling
        model2 = ProfiledNet(input_size=784, hidden_size=128, output_size=10)

        # Run inference (already profiled due to decorator)
        outputs2, predictions2 = run_inference(model2, X, "ProfiledNet")

        # === 3. Timing Utilities ===
        print("3️⃣  Testing timing utilities")

        # Use timed decorator
        result1 = expensive_computation(500)
        result2 = expensive_computation(1000)

        # Use timed_call for inline timing
        result3 = kandc.timed_call("inline_computation", expensive_computation, 750)

        # === 4. Comprehensive Metrics Logging ===
        print("4️⃣  Logging metrics")

        # Log all metrics in one comprehensive call
        kandc.log(
            {
                # Data metrics
                "batch_size": X.shape[0],
                "input_features": X.shape[1],
                "output_classes": outputs1.shape[1],
                # Model metrics
                "simplenet_parameters": sum(p.numel() for p in model1.parameters()),
                "profilednet_parameters": sum(p.numel() for p in model2.parameters()),
                # Prediction metrics
                "simplenet_predictions": len(predictions1),
                "profilednet_predictions": len(predictions2),
                # Timing metrics summary
                "computations_completed": 3,
                "total_computation_operations": 500 + 1000 + 750,
                # Feature flags
                "used_capture_model_instance": True,
                "used_capture_model_class": True,
                "used_timed_decorator": True,
                "used_timed_call": True,
            }
        )

        print(
            f"✅ Completed! SimpleNet: {len(predictions1)} predictions, ProfiledNet: {len(predictions2)} predictions"
        )

    finally:
        kandc.finish()

    print("🎉 Example completed! Check the dashboard for results.")


if __name__ == "__main__":
    main()
