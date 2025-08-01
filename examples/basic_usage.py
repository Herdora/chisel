#!/usr/bin/env python3

from chisel import ChiselApp, GPUType

# Example with default requirements.txt file
app = ChiselApp("basic-example", gpu=GPUType.A100_80GB_2)

# Alternative: specify a custom requirements file
# app = ChiselApp("basic-example", gpu=GPUType.A100_80GB_2, requirements_file="requirements.txt")


@app.capture_trace(trace_name="matrix_multiply", record_shapes=True)
def matrix_multiply(size: int = 1000):
    import torch

    print(f"🔥 Running matrix multiplication ({size}x{size})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    result = torch.mm(a, b)

    print(f"✅ Matrix multiplication completed! Result shape: {result.shape}")
    return result.cpu().numpy()


@app.capture_trace(trace_name="simple_computation")
def simple_computation(n: int = 1000000):
    import torch

    print(f"🔥 Running simple computation with {n} elements...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(n, device=device)

    result = x.pow(2).sum()

    print(f"✅ Computation completed! Result: {result.item()}")
    return result.item()


if __name__ == "__main__":
    print("🚀 Starting Chisel CLI Basic Example")

    matrix_result = matrix_multiply(500)
    print(f"Matrix multiply result shape: {matrix_result.shape}")

    computation_result = simple_computation(100000)
    print(f"Simple computation result: {computation_result}")

    print("✅ Example completed!")
