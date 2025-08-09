#!/usr/bin/env python3
"""
Large file test example.
This creates and uses a large file to test Chisel's file caching system.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from chisel import capture_model


@capture_model(model_name="LargeDataModel")
class LargeDataModel(nn.Module):
    """Model that processes large data files."""

    def __init__(self, input_size):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1000)  # Reduce to fixed size
        self.fc = nn.Sequential(
            nn.Linear(64 * 1000, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, sequence_length)
        x = torch.relu(self.conv1d(x))  # (batch_size, 64, sequence_length)
        x = self.pool(x)  # (batch_size, 64, 1000)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 64 * 1000)
        x = self.fc(x)
        return x


def create_large_file(filename, size_mb=50):
    """Create a large data file for testing."""

    print(f"📦 Creating large file: {filename} ({size_mb}MB)")

    # Calculate number of float32 values needed for the target size
    bytes_per_float = 4
    total_bytes = size_mb * 1024 * 1024
    num_values = total_bytes // bytes_per_float

    print(f"   Generating {num_values:,} float32 values...")

    # Generate data in chunks to avoid memory issues
    chunk_size = 1_000_000  # 1M values per chunk

    with open(filename, "wb") as f:
        remaining = num_values
        chunk_idx = 0

        while remaining > 0:
            current_chunk_size = min(chunk_size, remaining)

            # Generate random data
            chunk_data = np.random.randn(current_chunk_size).astype(np.float32)

            # Write to file
            chunk_data.tofile(f)

            remaining -= current_chunk_size
            chunk_idx += 1

            if chunk_idx % 10 == 0:  # Progress update every 10 chunks
                progress = (num_values - remaining) / num_values * 100
                print(f"   Progress: {progress:.1f}%")

    # Verify file size
    actual_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"✅ Created file: {actual_size:.1f}MB")

    return filename


def load_large_file(filename, max_samples=100000):
    """Load data from the large file."""

    print(f"📖 Loading data from: {filename}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    file_size = os.path.getsize(filename) / (1024 * 1024)
    print(f"   File size: {file_size:.1f}MB")

    # Load data (limit to max_samples to avoid memory issues)
    data = np.fromfile(filename, dtype=np.float32)

    if len(data) > max_samples:
        print(f"   Limiting to first {max_samples:,} samples")
        data = data[:max_samples]

    print(f"   Loaded {len(data):,} values")
    print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")

    return data


def process_large_data(model, data, batch_size=32, device="cpu"):
    """Process the large data through the model."""

    print(f"🔄 Processing data through model:")
    print(f"   Data shape: {data.shape}")
    print(f"   Batch size: {batch_size}")

    # Convert to tensor
    data_tensor = torch.from_numpy(data).to(device)

    # Create sequences (for demonstration, we'll create overlapping windows)
    sequence_length = 1000
    stride = 500

    sequences = []
    for i in range(0, len(data) - sequence_length + 1, stride):
        seq = data_tensor[i : i + sequence_length]
        sequences.append(seq)

        if len(sequences) >= batch_size * 10:  # Limit number of sequences
            break

    if not sequences:
        print("❌ Not enough data to create sequences")
        return

    # Stack sequences into batches
    sequences_tensor = torch.stack(sequences)
    print(f"   Created {len(sequences)} sequences of length {sequence_length}")
    print(f"   Sequences tensor shape: {sequences_tensor.shape}")

    model.eval()
    results = []

    # Process in batches
    for i in range(0, len(sequences_tensor), batch_size):
        batch = sequences_tensor[i : i + batch_size]

        with torch.no_grad():
            output = model(batch)
            results.append(output)

        batch_num = (i // batch_size) + 1
        total_batches = (len(sequences_tensor) + batch_size - 1) // batch_size
        print(f"   Processed batch {batch_num}/{total_batches}: {batch.shape} -> {output.shape}")

    # Combine results
    all_results = torch.cat(results, dim=0)
    print(f"✅ Processing complete: {all_results.shape}")
    print(f"   Results range: [{all_results.min().item():.4f}, {all_results.max().item():.4f}]")

    return all_results


def main():
    """Test large file handling."""
    print("🚀 Large File Test")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # File configuration
    large_file = "large_test_data.bin"
    file_size_mb = 100  # 100MB file

    # Check if file already exists
    if os.path.exists(large_file):
        print(f"📁 Large file already exists: {large_file}")
        file_size = os.path.getsize(large_file) / (1024 * 1024)
        print(f"   Existing file size: {file_size:.1f}MB")

        if file_size < file_size_mb * 0.9:  # If significantly smaller, recreate
            print("   File too small, recreating...")
            os.remove(large_file)
            create_large_file(large_file, file_size_mb)
    else:
        create_large_file(large_file, file_size_mb)

    # Load data
    try:
        data = load_large_file(large_file, max_samples=500000)  # Limit for demo
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Create model
    print(f"\n🏗️  Creating model...")
    model = LargeDataModel(input_size=1000).to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Process data
    try:
        results = process_large_data(model, data, batch_size=8, device=device)
        print(f"✅ Successfully processed large file!")
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback

        traceback.print_exc()

    # Cleanup option
    print(f"\n🧹 Cleanup:")
    print(f"   Large file: {large_file} ({os.path.getsize(large_file) / (1024 * 1024):.1f}MB)")
    print(f"   To remove: rm {large_file}")

    print("\n✅ Large file test completed!")


if __name__ == "__main__":
    main()
