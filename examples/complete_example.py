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


@kandc.capture_model_class(model_name="SimpleTransformer")
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=32, seq_len=16, d_model=64, nhead=4, num_layers=2, num_classes=10):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        # Project input to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = x + self.pos_embedding  # Add positional encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        x = x.mean(dim=1)  # Pool over sequence
        x = self.head(x)   # (batch, num_classes)
        return x


def main():
    print("🚀 Simple kandc Example")

    # Initialize with debug logging
    run = kandc.init(
        project="optimize-transformer",
        name="test-run-1",
    )
    print("✅ Using existing authentication")

    # Create and run model
    model = SimpleTransformer()
    # Simulate a batch of 32 sequences, each of length 16, with 32 features
    data = torch.randn(32, 16, 32)

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
        
        # Log metrics with custom x coordinate
        kandc.log({
            "loss": loss.item() - i * 0.01,  # Simulate decreasing loss
            "accuracy": 0.7 + i * 0.02,     # Simulate increasing accuracy
        }, step=i)

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
