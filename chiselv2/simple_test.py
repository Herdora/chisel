#!/usr/bin/env python3

import torch

# Simple tensor operations
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)

# Perform operations that will show up in profiling
z = torch.matmul(x, y)
result = torch.sum(z)

print(f"Result: {result.item()}")
print("✅ Test script completed successfully!") 