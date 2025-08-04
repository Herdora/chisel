#!/usr/bin/env python3

from chisel import capture_trace


# Example external functions to demonstrate inline tracing
def matrix_operations(size: int = 100):
    """External function - matrix operations."""
    import torch

    print(f"🔢 Performing matrix operations with size {size}x{size}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Matrix multiplication
    result = torch.mm(a, b)

    # Some additional operations
    result = torch.relu(result)
    result = result.sum()

    print(f"✅ Matrix operations completed! Result: {result.item():.4f}")
    return result


def data_processing(num_samples: int = 1000):
    """External function - data processing."""
    import torch

    print(f"📊 Processing {num_samples} data samples")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Simulate data processing
    data = torch.randn(num_samples, 128, device=device)

    # Normalization
    normalized = torch.nn.functional.normalize(data, dim=1)

    # Some transformations
    transformed = torch.sigmoid(normalized)
    final_result = transformed.mean(dim=0)

    print(f"✅ Data processing completed! Output shape: {final_result.shape}")
    return final_result.cpu().numpy()


def compute_statistics(data_size: int = 5000):
    """External function - statistical computations."""
    import torch

    print(f"📈 Computing statistics for {data_size} data points")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.randn(data_size, device=device)

    stats = {
        "mean": data.mean().item(),
        "std": data.std().item(),
        "min": data.min().item(),
        "max": data.max().item(),
        "median": data.median().item(),
    }

    print(f"✅ Statistics computed: {stats}")
    return stats


def main():
    print("🚀 Starting Inline Tracing Example")
    print("📋 This demonstrates inline wrapping of external functions with capture_trace")

    # Inline tracing - wrap function calls directly
    print("\n🔧 Inline traced matrix operations:")
    result1 = capture_trace(trace_name="matrix_ops", record_shapes=True, profile_memory=True)(
        matrix_operations
    )(150)

    print("\n🔧 Inline traced data processing:")
    result2 = capture_trace(trace_name="data_processing", record_shapes=True, profile_memory=True)(
        data_processing
    )(2000)

    print("\n🔧 Inline traced statistics:")
    stats = capture_trace(trace_name="statistics", record_shapes=True, profile_memory=True)(
        compute_statistics
    )(5000)

    print("\n📊 All inline traces completed successfully!")
    print(f"Matrix result: {result1.item():.4f}")
    print(f"Data processing result shape: {result2.shape}")
    print(f"Statistics mean: {stats['mean']:.4f}")

    print("✅ Inline tracing example completed!")


if __name__ == "__main__":
    main()
