#!/usr/bin/env python3
"""
Large File Caching Example for Chisel CLI

This example demonstrates how Chisel automatically caches large files (>1GB)
to avoid re-uploading them in subsequent jobs. The system uses SHA256 hashing
for deduplication and stores files in Modal Volumes.

Features demonstrated:
- Automatic detection of large files
- Hash-based deduplication
- Placeholder/symlink system
- Seamless restoration during job execution
"""

import os
import numpy as np
from pathlib import Path
from chisel import capture_trace


def create_large_test_file(filename: str = "large_dataset.npy", size_gb: float = 1.5):
    """
    Create a large test file for caching demonstration.

    Args:
        filename: Name of the file to create
        size_gb: Size of the file in GB
    """
    print(f"🔥 Creating large test file: {filename} ({size_gb}GB)")

    # Calculate array size for approximately the desired file size
    # float64 = 8 bytes per element
    bytes_per_gb = 1024 * 1024 * 1024
    target_bytes = int(size_gb * bytes_per_gb)
    elements_needed = target_bytes // 8  # 8 bytes per float64

    # Create a large random array
    print(f"📊 Generating {elements_needed:,} random numbers...")
    large_array = np.random.rand(elements_needed).astype(np.float64)

    # Save to file
    print(f"💾 Saving to {filename}...")
    np.save(filename, large_array)

    # Check actual file size
    file_size = os.path.getsize(filename)
    size_mb = file_size / (1024 * 1024)
    size_gb_actual = file_size / (1024 * 1024 * 1024)

    print(f"✅ Created {filename}: {size_mb:.1f}MB ({size_gb_actual:.2f}GB)")
    print(f"🔍 File path: {os.path.abspath(filename)}")

    return filename, file_size


@capture_trace(trace_name="large_file_processing", record_shapes=True, profile_memory=True)
def process_large_file(filename: str = "large_test_file.npy"):
    """
    Process a large file to demonstrate caching behavior.

    This function will load and process the large file. When run through Chisel:
    1. First run: File is uploaded to cache, job processes normally
    2. Second run: File is found in cache, placeholder is used, job processes normally
    """
    import torch

    print(f"🚀 Starting large file processing example")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"📋 Files in directory: {list(os.listdir('.'))}")

    # Check if our large file exists
    if not os.path.exists(filename):
        print(f"❌ Large file {filename} not found!")
        print(f"📍 Please ensure {filename} exists in the current directory")
        raise FileNotFoundError(f"Required file {filename} not found")

    file_size = os.path.getsize(filename)
    size_mb = file_size / (1024 * 1024)
    size_gb = file_size / (1024 * 1024 * 1024)

    print(f"📊 Processing file: {filename}")
    print(f"📏 File size: {size_mb:.1f}MB ({size_gb:.2f}GB)")

    # Load the large file
    print(f"📂 Loading large dataset from {filename}...")
    try:
        data = np.load(filename)
        print(f"✅ Loaded dataset with shape: {data.shape}")
        print(f"📈 Data type: {data.dtype}")
        print(f"💾 Memory usage: ~{data.nbytes / (1024 * 1024):.1f}MB")

        # Convert to PyTorch tensor for processing
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Using device: {device}")

        # Process in chunks to avoid memory issues
        chunk_size = 1000000  # Process 1M elements at a time
        results = []

        print(f"⚙️ Processing data in chunks of {chunk_size:,} elements...")

        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]

            # Convert to tensor and process
            tensor_chunk = torch.from_numpy(chunk).to(device)

            # Some example operations
            processed = tensor_chunk.pow(2).mean()
            results.append(processed.cpu().item())

            if i // chunk_size < 5:  # Only print first 5 chunks
                print(f"  📊 Chunk {i // chunk_size + 1}: mean of squares = {processed.item():.6f}")

        # Calculate final statistics
        final_result = np.mean(results)
        print(f"🎯 Final result (mean of chunk means): {final_result:.6f}")

        # Some additional PyTorch operations for demonstration
        sample_data = torch.from_numpy(data[:10000]).to(device)  # Use first 10k elements

        # Matrix operations
        if len(sample_data) >= 10000:
            matrix_size = int(np.sqrt(10000))  # 100x100 matrix
            matrix_data = sample_data[: matrix_size * matrix_size].reshape(matrix_size, matrix_size)

            print(f"🔢 Performing matrix operations on {matrix_size}x{matrix_size} sample...")
            matrix_result = torch.mm(matrix_data, matrix_data.T)
            eigenvalues = torch.linalg.eigvals(matrix_result)

            print(f"🧮 Matrix computation completed")
            print(f"📊 Largest eigenvalue: {eigenvalues.real.max().item():.6f}")

        print(f"✅ Large file processing completed successfully!")
        return final_result

    except Exception as e:
        print(f"❌ Error processing large file: {e}")
        raise


@capture_trace(trace_name="file_stats_analysis", record_shapes=True)
def analyze_file_statistics():
    """
    Analyze file statistics to show caching behavior.
    """
    print(f"📋 Analyzing files in current directory...")

    files_info = []
    total_size = 0

    for file_path in Path(".").iterdir():
        if file_path.is_file():
            size = file_path.stat().st_size
            size_mb = size / (1024 * 1024)
            total_size += size

            files_info.append(
                {
                    "name": file_path.name,
                    "size": size,
                    "size_mb": size_mb,
                    "is_large": size > (1024 * 1024 * 1024),  # > 1GB
                }
            )

    # Sort by size
    files_info.sort(key=lambda x: x["size"], reverse=True)

    print(f"📊 File Analysis Results:")
    print(f"📁 Total files: {len(files_info)}")
    print(f"💾 Total size: {total_size / (1024 * 1024):.1f}MB")

    large_files = [f for f in files_info if f["is_large"]]
    if large_files:
        print(f"🔍 Large files (>1GB) found: {len(large_files)}")
        for file_info in large_files:
            print(f"  📄 {file_info['name']}: {file_info['size_mb']:.1f}MB")
    else:
        print(f"ℹ️ No large files (>1GB) found")

    print(f"📋 All files:")
    for file_info in files_info[:10]:  # Show top 10
        print(f"  📄 {file_info['name']}: {file_info['size_mb']:.1f}MB")

    return files_info


def main():
    """
    Main function demonstrating large file caching.
    """
    print("🚀 Starting Large File Caching Example")
    print("=" * 60)
    print(f"📍 Current directory: {os.getcwd()}")

    # Step 1: Use the existing large test file
    large_file = "large_test_file.npy"

    if not os.path.exists(large_file):
        print(f"\n❌ Error: {large_file} not found!")
        print(f"📍 Please ensure {large_file} exists in the current directory")
        print(f"🔧 You can create it by running: python create_test_files.py")
        return
    else:
        file_size = os.path.getsize(large_file)
        size_gb = file_size / (1024 * 1024 * 1024)
        print(f"\n📝 Step 1: Using existing large test file ({size_gb:.2f}GB)")

    print(f"\n📝 Step 2: Analyzing current files...")
    file_stats = analyze_file_statistics()

    print(f"\n📝 Step 3: Processing large file...")
    result = process_large_file(large_file)

    print(f"\n✅ Example completed successfully!")
    print(f"🎯 Final processing result: {result:.6f}")

    print(f"\n" + "=" * 60)
    print(f"📚 About Caching Behavior:")
    print(f"  🔄 First run: Chisel detects large files, uploads to cache")
    print(f"  ⚡ Second run: Chisel uses cached files, much faster upload")
    print(f"  🔍 Files >1GB are automatically cached by SHA256 hash")
    print(f"  📦 Cached files are stored in Modal Volumes")
    print(f"  🔗 Placeholders are created and resolved during job execution")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
