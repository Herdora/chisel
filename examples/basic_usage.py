#!/usr/bin/env python3
"""
Basic Chisel CLI Usage Example

This example shows how to create a ChiselApp and use the tracing decorator
to profile your functions on cloud GPUs.
"""

from chisel import ChiselApp

# Create a Chisel app - authentication happens automatically
app = ChiselApp("basic-example")


@app.capture_trace(trace_name="matrix_multiply", record_shapes=True)
def matrix_multiply(size: int = 1000):
    """Simple matrix multiplication with profiling."""
    import torch

    print(f"🔥 Running matrix multiplication ({size}x{size})...")

    # Create random matrices on GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Perform matrix multiplication
    result = torch.mm(a, b)

    print(f"✅ Matrix multiplication completed! Result shape: {result.shape}")
    return result.cpu().numpy()


@app.capture_trace(trace_name="simple_computation")
def simple_computation(n: int = 1000000):
    """Simple computation example."""
    import torch

    print(f"🔥 Running simple computation with {n} elements...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(n, device=device)

    # Some operations
    result = x.pow(2).sum()

    print(f"✅ Computation completed! Result: {result.item()}")
    return result.item()


if __name__ == "__main__":
    print("🚀 Starting Chisel CLI Basic Example")

    # Run the traced functions
    matrix_result = matrix_multiply(500)
    print(f"Matrix multiply result shape: {matrix_result.shape}")

    computation_result = simple_computation(100000)
    print(f"Simple computation result: {computation_result}")

    print("✅ Example completed!")
