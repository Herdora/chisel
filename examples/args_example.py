#!/usr/bin/env python3

import argparse
from chisel import ChiselApp, GPUType

app = ChiselApp("args-example", gpu=GPUType.A100_80GB_1)


@app.capture_trace(trace_name="simple_ops", record_shapes=True)
def simple_operations(iterations: int):
    import torch

    print(f"🔥 Running simple operations for {iterations} iterations...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    for i in range(iterations):
        a = torch.randn(100, 100, device=device)
        b = torch.randn(100, 100, device=device)

        result = torch.mm(a, b)
        result = torch.relu(result)
        result = result.sum()

        print(f"  Iteration {i + 1}/{iterations}: Result = {result.item():.4f}")

    print("✅ Operations completed!")
    return result.cpu().numpy() if hasattr(result, "cpu") else result


def main():
    parser = argparse.ArgumentParser(description="Chisel CLI Args Example")
    parser.add_argument("--iterations", type=int, help="Number of iterations (required)")

    args = parser.parse_args()

    assert args.iterations is not None, "❌ --iterations argument is required!"
    assert args.iterations > 0, "❌ --iterations must be greater than 0!"

    print("🚀 Starting Chisel CLI Args Example")
    print(f"Parameters: iterations={args.iterations}")

    result = simple_operations(args.iterations)
    print(f"Final result: {result}")

    print("✅ Example completed!")


if __name__ == "__main__":
    main()
