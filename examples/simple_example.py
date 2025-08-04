from chisel import capture_trace


@capture_trace(trace_name="matrix_multiply", record_shapes=True)
def matrix_multiply(size: int = 1000):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    result = torch.mm(a, b)

    print(f"✅ Matrix multiplication completed! Shape: {result.shape}")
    return result.cpu().numpy()


@capture_trace(trace_name="simple_computation")
def simple_computation(n: int = 1000000):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(n, device=device)
    result = x.pow(2).sum()

    print(f"✅ Computation completed! Result: {result.item()}")
    return result.item()


if __name__ == "__main__":
    print("🚀 Starting Chisel example")

    matrix_result = matrix_multiply(500)
    computation_result = simple_computation(100000)

    print("✅ Example completed!")
