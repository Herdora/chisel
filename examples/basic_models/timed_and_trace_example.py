#!/usr/bin/env python3
"""
Timed calls + tracing example

Demonstrates:
- kandc.timing: @timed decorator and timed_call wrapper
- kandc.trace: capture_model_instance for profiling forward() (when running on backend)

This script runs locally without the Keys & Caches backend. When executed on the
backend (KANDC_BACKEND_RUN_ENV_KEY=1), capture_model_instance will save profiler
traces and embed kandc_io into the trace JSON.
"""

import os
import time
import random
from typing import Tuple

import torch
import torch.nn as nn
from kandc import timed, timed_call, capture_model_instance


# Optional: demonstrate non-standard method profiling by adding a custom method
class WithGenerate(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = TinyMLP()

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        # Pretend this is a generation-like method
        time.sleep(0.01)
        return self.base(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Required for nn.Module
        return self.base(x)


@timed("preprocess_batch")
def preprocess(batch: torch.Tensor) -> torch.Tensor:
    # Simulate non-trivial work
    time.sleep(0.05 + random.random() * 0.05)
    return (batch - batch.mean()) / (batch.std() + 1e-6)


@timed("postprocess_logits")
def postprocess(logits: torch.Tensor) -> torch.Tensor:
    time.sleep(0.02 + random.random() * 0.02)
    return torch.softmax(logits, dim=-1)


class TinyMLP(nn.Module):
    def __init__(self, in_features: int = 16, hidden: int = 32, out_features: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_once(model: nn.Module, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Show both decorator and direct wrapper styles
    x = preprocess(batch)
    # Equivalent one-off timing without decorating:
    x = timed_call("augment_batch", lambda t: t + 0.01 * torch.randn_like(t), x)

    logits = model(x)
    probs = postprocess(logits)
    return logits, probs


def main():
    print("\n🧪 kandc timed + trace example")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use a model that has a non-forward method to show profiling of methods like generate()
    model = WithGenerate().to(device).eval()

    # Enable tracing automatically on backend runs
    if os.getenv("KANDC_BACKEND_RUN_ENV_KEY") == "1":
        # Name traces by model class for clarity
        model = capture_model_instance(
            model, model_name="TinyMLP", record_shapes=True, profile_memory=True
        )
        print("✅ Tracing enabled (running on backend)")
    else:
        print("ℹ️ Tracing disabled (running locally). Set KANDC_BACKEND_RUN_ENV_KEY=1 on backend.")

    with torch.no_grad():
        for i in range(3):
            batch = torch.randn(4, 16).to(device)
            logits, probs = run_once(model, batch)
            print(f"Pass {i + 1}: logits={tuple(logits.shape)}, probs={tuple(probs.shape)}")

        # Exercise a non-standard method to trigger method-specific trace (e.g., TinyMLP_generate_001.json)
        _ = model.generate(torch.randn(2, 16).to(device))

    print(
        "\n✅ Done. If on backend, traces are saved under /volume/<app>/<job>/traces and timings in timings.jsonl"
    )


if __name__ == "__main__":
    main()
