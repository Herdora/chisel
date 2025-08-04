#!/usr/bin/env python3

from chisel import capture_trace

# # Example 1: Using default requirements.txt (automatically detected)
# app_default = ChiselApp("requirements-default", gpu=GPUType.A100_80GB_1)

# # Example 2: Using custom requirements file path
# app_custom = ChiselApp(
#     "requirements-custom", gpu=GPUType.A100_80GB_1, requirements_file="custom-requirements.txt"
# )

# Example 3: Using requirements in subdirectory
# app_subdir = ChiselApp(
#     "requirements-subdir", gpu=GPUType.A100_80GB_2, requirements_file="requirements/dev.txt"
# )


@capture_trace(trace_name="requirements_demo", record_shapes=True)
def requirements_demo():
    """Demonstrate that packages from requirements.txt are available."""
    import torch
    import numpy as np

    print("🔥 Testing packages from requirements.txt...")

    # Test PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 PyTorch device: {device}")

    x = torch.randn(10, 10, device=device)
    y = torch.randn(10, 10, device=device)
    result = torch.mm(x, y)

    print(f"✅ PyTorch matrix multiplication successful: {result.shape}")

    # Test NumPy
    np_array = np.random.rand(5, 5)
    np_result = np.dot(np_array, np_array.T)

    print(f"✅ NumPy matrix multiplication successful: {np_result.shape}")

    # Test matplotlib (import only, no plotting on backend)
    try:
        import matplotlib

        print(f"✅ Matplotlib available: {matplotlib.__version__}")
    except ImportError:
        print("⚠️ Matplotlib not available")

    return result.cpu().numpy()


if __name__ == "__main__":
    print("🚀 Starting Requirements Example")
    print("📦 This example demonstrates requirements.txt file usage with Chisel CLI")

    result = requirements_demo()
    print(f"Final result shape: {result.shape}")

    print("✅ Requirements example completed!")
