#!/usr/bin/env python3
"""
Model Profiling Example for Chisel CLI

This example demonstrates how to use the @capture_model decorator to automatically
profile every forward pass of a PyTorch model, capturing layer-level timing and
shape information.

Features demonstrated:
- Automatic profiling of every model forward pass
- Layer-level timing analysis
- Shape tracking for each layer
- Detailed trace files saved to the backend
- Real-time layer summary during execution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from chisel import capture_model


@capture_model(model_name="SimpleCNN", record_shapes=True, profile_memory=True)
class SimpleCNN(nn.Module):
    """
    A simple CNN model for demonstration.

    This model will be automatically profiled on every forward pass when run
    through Chisel CLI, with detailed layer-level timing and shape information.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate the size after convolutions and pooling
        # Input: 32x32x3 -> 16x16x128 after 3 conv+pool layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Each layer will be individually profiled and timed
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


@capture_model(model_name="TransformerBlock", record_shapes=True, profile_memory=True)
class TransformerBlock(nn.Module):
    """
    A simple transformer block for demonstration.

    This shows how the profiler handles attention mechanisms and
    more complex model architectures.
    """

    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ff_output))

        return x


def create_sample_data(batch_size=32, seq_len=100, d_model=512):
    """Create sample data for the models."""
    print(f"📊 Creating sample data: batch_size={batch_size}, seq_len={seq_len}, d_model={d_model}")

    # Data for CNN
    cnn_data = torch.randn(batch_size, 3, 32, 32)

    # Data for Transformer
    transformer_data = torch.randn(batch_size, seq_len, d_model)

    return cnn_data, transformer_data


def run_cnn_example():
    """Run the CNN model example."""
    print("\n🚀 Running CNN Model Example")
    print("=" * 50)

    # Create model and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    model = SimpleCNN().to(device)
    cnn_data, _ = create_sample_data()
    cnn_data = cnn_data.to(device)

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Input shape: {cnn_data.shape}")

    # Run multiple forward passes to demonstrate profiling
    print(f"\n🔄 Running 3 forward passes...")
    for i in range(3):
        print(f"\n--- Forward Pass {i + 1} ---")
        with torch.no_grad():
            output = model(cnn_data)
        print(f"✅ Output shape: {output.shape}")
        print(f"📊 Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    return model


def run_transformer_example():
    """Run the Transformer model example."""
    print("\n🚀 Running Transformer Model Example")
    print("=" * 50)

    # Create model and data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    model = TransformerBlock().to(device)
    _, transformer_data = create_sample_data()
    transformer_data = transformer_data.to(device)

    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Input shape: {transformer_data.shape}")

    # Run multiple forward passes to demonstrate profiling
    print(f"\n🔄 Running 3 forward passes...")
    for i in range(3):
        print(f"\n--- Forward Pass {i + 1} ---")
        with torch.no_grad():
            output = model(transformer_data)
        print(f"✅ Output shape: {output.shape}")
        print(f"📊 Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    return model


def demonstrate_trace_analysis():
    """Demonstrate how to analyze trace files after they're generated."""
    print("\n📊 Trace Analysis Demonstration")
    print("=" * 50)

    from chisel import parse_model_trace
    import os
    from pathlib import Path

    # Look for trace files in the current directory
    trace_files = list(Path(".").glob("*.json"))

    if trace_files:
        print(f"🔍 Found {len(trace_files)} trace files:")
        for trace_file in trace_files:
            print(f"  📄 {trace_file}")

            # Try to parse the trace file
            analysis = parse_model_trace(str(trace_file), "DemoModel")
            if analysis:
                print(f"  ✅ Successfully parsed {trace_file}")
                layer_count = len(analysis["layer_stats"])
                print(f"  📊 Found {layer_count} layers")
            else:
                print(f"  ⚠️  Could not parse {trace_file}")
    else:
        print("ℹ️  No trace files found in current directory")
        print("💡 Trace files are saved to the backend when running with Chisel CLI")


def main():
    """
    Main function demonstrating model profiling with Chisel.
    """
    import os

    print("🚀 Starting Model Profiling Example")
    print("=" * 60)
    print(f"📍 Current directory: {os.getcwd()}")

    # Check if we're running on Chisel backend
    if os.environ.get("CHISEL_BACKEND_RUN") == "1":
        print("✅ Running on Chisel backend - profiling will be active")
    else:
        print("ℹ️  Running locally - profiling will be simulated")

    # Run examples
    cnn_model = run_cnn_example()
    # transformer_model = run_transformer_example()

    # Demonstrate trace analysis
    demonstrate_trace_analysis()

    print("\n" + "=" * 60)
    print("✅ Model Profiling Example Completed!")
    print("\n📚 What happened:")
    print("  🔄 Each forward pass was automatically profiled")
    print("  📊 Layer-level timing and shapes were captured")
    print("  💾 Trace files were saved to the backend")
    print("  📈 Real-time layer summaries were displayed")
    print("\n💡 To see detailed traces:")
    print("  📁 Check the traces directory in the Chisel frontend")
    print("  🔍 Use Chrome DevTools to view trace files")
    print("  📊 Use parse_model_trace() to analyze saved traces")
    print("=" * 60)


if __name__ == "__main__":
    main()
