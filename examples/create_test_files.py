#!/usr/bin/env python3
"""
Helper script to create large test files for caching examples.

This script creates various sized files to test the caching functionality.
Run this before testing the large file caching example.
"""

import os
import numpy as np
import argparse
from pathlib import Path


def create_large_file(filename: str, size_gb: float = 1.5, file_type: str = "numpy"):
    """
    Create a large file of specified size and type.
    
    Args:
        filename: Name of the file to create
        size_gb: Size in GB
        file_type: Type of file ('numpy', 'binary', 'text')
    """
    print(f"🔥 Creating {file_type} file: {filename} ({size_gb}GB)")
    
    bytes_per_gb = 1024 * 1024 * 1024
    target_bytes = int(size_gb * bytes_per_gb)
    
    if file_type == "numpy":
        # Create numpy array file
        elements_needed = target_bytes // 8  # 8 bytes per float64
        print(f"📊 Generating {elements_needed:,} random numbers...")
        
        large_array = np.random.rand(elements_needed).astype(np.float64)
        np.save(filename, large_array)
        
    elif file_type == "binary":
        # Create binary file with random data
        print(f"📊 Generating {target_bytes:,} random bytes...")
        
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(filename, 'wb') as f:
            remaining = target_bytes
            while remaining > 0:
                chunk = min(chunk_size, remaining)
                data = np.random.bytes(chunk)
                f.write(data)
                remaining -= chunk
                
                if remaining % (100 * 1024 * 1024) == 0:  # Progress every 100MB
                    progress = (target_bytes - remaining) / target_bytes * 100
                    print(f"  📈 Progress: {progress:.1f}%")
    
    elif file_type == "text":
        # Create text file with repeated content
        print(f"📊 Generating text content...")
        
        # Create a base string to repeat
        base_content = "This is a test line for large file caching demonstration.\n" * 1000
        base_size = len(base_content.encode('utf-8'))
        repeats_needed = target_bytes // base_size + 1
        
        with open(filename, 'w', encoding='utf-8') as f:
            for i in range(repeats_needed):
                f.write(base_content)
                if i % 1000 == 0:
                    progress = i / repeats_needed * 100
                    print(f"  📈 Progress: {progress:.1f}%")
    
    # Check actual file size
    actual_size = os.path.getsize(filename)
    actual_gb = actual_size / (1024 * 1024 * 1024)
    actual_mb = actual_size / (1024 * 1024)
    
    print(f"✅ Created {filename}")
    print(f"📏 Size: {actual_mb:.1f}MB ({actual_gb:.2f}GB)")
    print(f"🔍 Path: {os.path.abspath(filename)}")
    
    return filename, actual_size


def create_test_dataset():
    """Create a variety of test files for comprehensive testing."""
    print("🚀 Creating comprehensive test dataset...")
    
    files_created = []
    
    # Create different types of large files
    test_files = [
        ("model_weights.npy", 1.1, "numpy"),
        ("training_data.npy", 1.3, "numpy"),
        ("large_binary.bin", 1.2, "binary"),
        ("massive_log.txt", 1.1, "text"),
    ]
    
    for filename, size_gb, file_type in test_files:
        try:
            if not os.path.exists(filename):
                created_file, file_size = create_large_file(filename, size_gb, file_type)
                files_created.append((created_file, file_size))
                print()
            else:
                existing_size = os.path.getsize(filename)
                print(f"⏩ {filename} already exists ({existing_size / (1024*1024*1024):.2f}GB)")
                files_created.append((filename, existing_size))
        except Exception as e:
            print(f"❌ Failed to create {filename}: {e}")
    
    # Summary
    print("=" * 60)
    print("📊 Test Dataset Summary:")
    total_size = 0
    for filename, file_size in files_created:
        size_gb = file_size / (1024 * 1024 * 1024)
        total_size += file_size
        print(f"  📄 {filename}: {size_gb:.2f}GB")
    
    total_gb = total_size / (1024 * 1024 * 1024)
    print(f"📏 Total size: {total_gb:.2f}GB")
    print(f"✅ Dataset creation completed!")
    
    return files_created


def cleanup_test_files():
    """Remove all test files."""
    test_patterns = [
        "*.npy",
        "large_*.bin",
        "massive_*.txt",
        "model_*.npy",
        "training_*.npy"
    ]
    
    print("🧹 Cleaning up test files...")
    removed_count = 0
    
    for pattern in test_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                size_gb = file_size / (1024 * 1024 * 1024)
                print(f"🗑️ Removing {file_path.name} ({size_gb:.2f}GB)")
                file_path.unlink()
                removed_count += 1
    
    print(f"✅ Cleanup completed! Removed {removed_count} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Create large test files for Chisel caching examples"
    )
    parser.add_argument(
        "--action", 
        choices=["create", "dataset", "cleanup"], 
        default="create",
        help="Action to perform"
    )
    parser.add_argument(
        "--filename", 
        default="large_test_file.npy",
        help="Name of file to create (for 'create' action)"
    )
    parser.add_argument(
        "--size", 
        type=float, 
        default=1.5,
        help="Size in GB (for 'create' action)"
    )
    parser.add_argument(
        "--type", 
        choices=["numpy", "binary", "text"], 
        default="numpy",
        help="Type of file to create (for 'create' action)"
    )
    
    args = parser.parse_args()
    
    print("🚀 Large File Test Creator")
    print("=" * 50)
    
    if args.action == "create":
        create_large_file(args.filename, args.size, args.type)
    elif args.action == "dataset":
        create_test_dataset()
    elif args.action == "cleanup":
        cleanup_test_files()
    
    print("=" * 50)
    print("✅ Done!")


if __name__ == "__main__":
    main()