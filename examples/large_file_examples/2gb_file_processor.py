#!/usr/bin/env python3
"""
Simple 2GB+ file processor example using capture_model_class.

This example demonstrates how to work with very large files (2GB+) using Keys & Caches.
The user can manipulate the data and print results while the model operations are automatically profiled.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from kandc import capture_model_class


@capture_model_class(model_name="LargeFileProcessor")
class LargeFileProcessor(nn.Module):
    """Simple model that processes large data files."""

    def __init__(self):
        super().__init__()
        # Simple linear layers for processing large data
        self.layer1 = nn.Linear(1000, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        # Process data in chunks to handle large files
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.output(x)
        return x


def check_file_exists(filename="large_data_2gb.bin"):
    """Check if the 2GB+ file exists and is the right size."""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        print(f"   Please create a 2GB+ file named '{filename}' before running this script.")
        print(
            f"   You can use any method to create it (e.g., dd, fallocate, or your own data generation)."
        )
        return False

    file_size_gb = os.path.getsize(filename) / (1024 * 1024 * 1024)
    print(f"📁 Found file: {filename}")
    print(f"   File size: {file_size_gb:.2f} GB")

    if file_size_gb < 2.0:
        print(f"   ⚠️  Warning: File is smaller than 2GB ({file_size_gb:.2f} GB)")
        print(f"   This example works best with files 2GB or larger.")

    return True


def load_and_process_data(filename, max_samples=100000):
    """Load data from the large file and allow user manipulation."""
    print(f"📖 Loading data from: {filename}")

    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    file_size_gb = os.path.getsize(filename) / (1024 * 1024 * 1024)
    print(f"   File size: {file_size_gb:.2f} GB")

    # Load data (limit to max_samples to avoid memory issues)
    data = np.fromfile(filename, dtype=np.float32)

    if len(data) > max_samples:
        print(f"   Limiting to first {max_samples:,} samples for processing")
        data = data[:max_samples]

    print(f"   Loaded {len(data):,} values")
    print(f"   Data range: [{data.min():.4f}, {data.max():.4f}]")

    # User can manipulate the data here
    print("\n🔧 Data manipulation examples:")

    # Example 1: Basic statistics
    mean_val = np.mean(data)
    std_val = np.std(data)
    print(f"   Mean: {mean_val:.4f}")
    print(f"   Standard deviation: {std_val:.4f}")

    # Example 2: Data transformation
    normalized_data = (data - mean_val) / std_val
    print(f"   Normalized data range: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")

    # Example 3: Filtering
    positive_values = data[data > 0]
    print(
        f"   Positive values: {len(positive_values):,} ({len(positive_values) / len(data) * 100:.1f}%)"
    )

    # Example 4: Binning
    bins = np.linspace(data.min(), data.max(), 11)
    hist, _ = np.histogram(data, bins=bins)
    print(f"   Histogram bins: {hist}")

    return data, normalized_data


def process_with_model(model, data, batch_size=32):
    """Process data through the model."""
    print(f"\n🚀 Processing data through model:")
    print(f"   Data shape: {data.shape}")
    print(f"   Batch size: {batch_size}")

    # Convert to tensor
    data_tensor = torch.from_numpy(data).float()

    # Create sequences for processing
    sequence_length = 1000
    stride = 500

    sequences = []
    for i in range(0, len(data) - sequence_length + 1, stride):
        seq = data_tensor[i : i + sequence_length]
        sequences.append(seq)

        if len(sequences) >= batch_size * 5:  # Limit for demo
            break

    if not sequences:
        print("❌ Not enough data to create sequences")
        return None

    # Stack sequences
    sequences_tensor = torch.stack(sequences)
    print(f"   Created {len(sequences)} sequences of length {sequence_length}")

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
    """Main function to demonstrate 2GB+ file processing."""
    print("🚀 2GB+ File Processor Example")
    print("=" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🎯 Using device: {device}")

    # File configuration
    large_file = "large_data_2gb.bin"

    # Check if the 2GB+ file exists
    if not check_file_exists(large_file):
        return

    # Load and manipulate data
    try:
        data, normalized_data = load_and_process_data(large_file, max_samples=200000)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # Create model
    print(f"\n🏗️  Creating model...")
    model = LargeFileProcessor().to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Process data through model
    try:
        results = process_with_model(model, normalized_data, batch_size=16)
        print(f"✅ Successfully processed 2GB+ file!")
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback

        traceback.print_exc()

    # Summary
    print(f"\n📋 Summary:")
    print(f"   Large file: {large_file}")
    print(f"   File size: {os.path.getsize(large_file) / (1024 * 1024 * 1024):.2f} GB")
    print(f"   Data processed: {len(data):,} values")
    print(f"   Model operations were automatically profiled")

    print(f"\n🧹 Cleanup:")
    print(f"   To remove large file: rm {large_file}")

    print("\n✅ 2GB+ file processing example completed!")


if __name__ == "__main__":
    main()
