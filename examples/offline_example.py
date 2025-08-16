#!/usr/bin/env python3
"""
Offline Mode Example
===================

This example demonstrates kandc running in offline mode - no internet required!

Usage:
    python offline_example.py
"""

import torch
import torch.nn as nn
import kandc
import time


def main():
    """Demonstrate kandc offline mode."""

    print("🔌 kandc Offline Mode Demo")
    print("=" * 50)
    print("This example works without internet connection!")
    print("")

    # Initialize in offline mode
    run = kandc.init(
        project="offline-demo",
        name="local-inference",
        mode="offline",  # 🔑 This is the key!
        config={"batch_size": 16, "model_type": "simple"},
        tags=["offline", "demo", "local"],
    )

    print(f"✅ Initialized offline run: {run.name}")
    print(f"📁 Data will be saved to: {run.dir}")
    print("")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.Softmax(dim=1),
    )

    print("🧠 Created model for offline experiment")

    # Generate data function
    def generate_data(batch_size=16):
        """Generate synthetic data."""
        return torch.randn(batch_size, 784)

    def run_inference(model, data):
        """Run model inference."""
        with torch.no_grad():
            return model(data)

    # Run several inference steps
    for step in range(5):
        print(f"🔄 Running inference step {step + 1}/5")

        # Generate data
        data = generate_data(batch_size=16)

        # Run inference
        predictions = run_inference(model, data)

        # Log metrics (saved locally)
        kandc.log(
            {
                "step": step,
                "batch_size": data.shape[0],
                "max_confidence": torch.max(predictions).item(),
                "mean_confidence": torch.mean(torch.max(predictions, dim=1)[0]).item(),
                "predictions_sum": torch.sum(predictions).item(),
            }
        )

        time.sleep(0.1)  # Simulate some processing time

    print("")
    print("📊 Logged metrics for all steps")

    # Finish the run
    kandc.finish()

    print("")
    print("🎉 Offline demo completed!")
    print("📁 Check the local directory for saved data:")
    print(f"   - Traces: {run.dir}/traces/")
    print(f"   - Metrics: {run.dir}/metrics.jsonl")
    print(f"   - Config: {run.dir}/config.json")
    print("")
    print("💡 This all worked without internet! Perfect for:")
    print("   - Development and debugging")
    print("   - Air-gapped environments")
    print("   - CI/CD pipelines")
    print("   - When you just want local profiling")


if __name__ == "__main__":
    main()
