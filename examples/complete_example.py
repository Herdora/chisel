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

import torch
import torch.nn as nn
import kandc


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

    # Start tracking your experiment
    kandc.init(
        project="simple-demo",
        name="basic-example",
        config={"batch_size": 32, "learning_rate": 0.01},
    )

    # Create model and data
    model = SimpleNet()
    data = torch.randn(32, 784)

    # Run your model
    output = model(data)
    loss = output.mean()

    # Log metrics
    kandc.log({"loss": loss.item(), "accuracy": 0.85})

    # Finish tracking
    kandc.finish()

    print("✅ Done! Check your dashboard for results.")


if __name__ == "__main__":
    main()
