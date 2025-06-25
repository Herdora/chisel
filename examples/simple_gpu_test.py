# simple_gpu_test.py - Quick GPU test for profiling

import torch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Simple matrix multiplication to trigger GPU kernels
    size = 1024
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Do a few iterations to generate some GPU activity
    for i in range(10):
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    print(f"Result shape: {c.shape}")
    print("GPU computation completed!")