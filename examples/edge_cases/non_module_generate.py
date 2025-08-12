#!/usr/bin/env python3
"""
Example: Profiling a non-nn.Module object that exposes a `generate()` method

This shows how to profile arbitrary code (including non-PyTorch models) by
wrapping a top-level function with `@capture_trace`. When running on the
Keys & Caches backend, the call is profiled and saved as a Chrome trace JSON.

Run locally (no profiling):
  python kandc/examples/edge_cases/non_module_generate.py

Run on backend (profiling enabled):
  kandc capture python kandc/examples/edge_cases/non_module_generate.py
"""

import time
import random
from typing import Any

from kandc import capture_trace, timed


class ToyTextModel:
    """A simple non-nn.Module model with a `generate` method."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def generate(self, prompt: str, max_tokens: int = 8) -> str:
        # Simulate some work
        time.sleep(0.02 + random.random() * 0.03)
        out = [prompt]
        for _ in range(max_tokens):
            out.append(random.choice(self.vocab))
        return " ".join(out)


@timed("prepare_prompt")
def prepare_prompt(user_text: str) -> str:
    time.sleep(0.01)
    return user_text.strip() or "Hello"


@capture_trace(trace_name="toy_generate", record_shapes=False, profile_memory=False)
def call_generate(model: ToyTextModel, prompt: str) -> str:
    """Profiled entrypoint that calls a non-Module model.generate().

    When running on the backend, this function's execution (including all
    Python and Torch ops inside) is captured into a Chrome trace file.
    """
    p = prepare_prompt(prompt)
    return model.generate(p, max_tokens=6)


def main() -> None:
    vocab = ["world", "from", "kandc", "trace", "example", "!", ":)"]
    model = ToyTextModel(vocab)

    print("\n🧪 Non-Module generate() profiling example")
    for i in range(3):
        text = call_generate(model, f"Hello {i}")
        print(f"{i + 1}. {text}")

    print("\n✅ Done. If run via `kandc capture`, a toy_generate.json trace was saved.")


if __name__ == "__main__":
    main()
