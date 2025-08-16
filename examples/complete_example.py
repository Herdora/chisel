#!/usr/bin/env python3
"""
Simple kandc Example
===================

A minimal example showing how to use kandc for experiment tracking.

Requirements:
    pip install torch kandc

Usage:
    python complete_example.py
"""

import time
import random
import torch
import torch.nn as nn
import kandc


@kandc.capture_model_class(model_name="SimpleNet")
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    print("🚀 Simple kandc Example")

    # Initialize with debug logging
    run = kandc.init(
        project="simple-demo",
        name="basic-example",
    )
    print("✅ Using existing authentication")

    # Create and run model
    model = SimpleNet()
    data = torch.randn(32, 784)

    print("📊 Running model forward pass...")
    output = model(data)
    loss = output.mean()

    @kandc.timed(name="random_wait")
    def random_wait():
        print("⏳ Starting random wait...")
        time.sleep(random.random() * 2)
        print("✅ Random wait complete")

    random_wait()

    # Log some metrics with x values
    print("📈 Logging metrics...")
    for i in range(10):
        # Simulate some training time
        time.sleep(0.1)

        # Use custom x value (could be epoch, iteration, etc.)
        x_value = i * 0.5  # Example: x values will be 0, 0.5, 1.0, 1.5, etc.

        kandc.log({"loss": loss.item(), "accuracy": random.random()}, x=x_value)

    # Verify artifacts / print run id (fallback to local id if backend offline)
    try:
        backend_run_id = (
            getattr(run, "_run_data", {}).get("id") if getattr(run, "_run_data", None) else None
        )
    except Exception:
        backend_run_id = None
    if backend_run_id:
        print(f"✅ Run ID: {backend_run_id}")
        print("✅ API client initialized")
    else:
        # Always available local run id
        print(f"✅ Local Run ID: {run.id}")
        print("⚠️ Backend run not created; operating offline")

    kandc.finish()

    print("✅ Done! Check your dashboard for results.")


if __name__ == "__main__":
    main()
