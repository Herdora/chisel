"""
Simple test script for Chisel upload limits.

This is the actual script that gets executed by Chisel when testing upload limits.
It demonstrates basic functionality while the upload limit test creates the test files.

Usage:
1. Run: python test_upload_limits.py (creates test files)
2. Run: chisel run --upload-dir upload_limit_test simple_upload_test.py
3. Observe the 5GB limit error for large files
"""

import os
from pathlib import Path
import numpy as np
from chisel import capture_trace


@capture_trace(trace_name="simple_upload_test")
def test_file_access():
    """Test accessing uploaded files."""
    print("🧪 Testing file access in Chisel environment")
    print("=" * 50)

    current_dir = Path(".")
    files_found = []
    total_size = 0

    print(f"📁 Current directory: {current_dir.absolute()}")
    print(f"📋 Files in current directory:")

    for file_path in current_dir.iterdir():
        if file_path.is_file() and file_path.suffix == ".npy":
            try:
                size_bytes = file_path.stat().st_size
                size_gb = size_bytes / (1024 * 1024 * 1024)
                total_size += size_gb

                print(f"  📄 {file_path.name}: {size_gb:.2f}GB")
                files_found.append({"name": file_path.name, "size_gb": size_gb})

                # Try to load the file to verify it works
                print(f"     Loading {file_path.name}...")
                data = np.load(file_path)
                print(f"     ✅ Loaded successfully - shape: {data.shape}")

            except Exception as e:
                print(f"     ❌ Error loading {file_path.name}: {e}")

    print(f"\n📊 Summary:")
    print(f"  Files found: {len(files_found)}")
    print(f"  Total size: {total_size:.2f}GB")

    if not files_found:
        print(f"  ⚠️  No .npy files found. This is expected if large files were rejected.")
        print(f"  💡 Check the Chisel upload output for 5GB limit messages.")

    print(f"\n✅ Test completed successfully!")
    return {"files_processed": files_found, "total_size_gb": total_size}


def demonstrate_large_file_download():
    """Demonstrate downloading a file within the script."""
    print(f"\n🌐 Demonstrating large file download within script...")

    # Simulate downloading a large file
    print(f"📥 Simulating download of large file...")
    print(f"   (In real usage, you would use requests.get() or similar)")

    # Create a small example file to simulate downloaded content
    example_data = np.random.random(1000).astype(np.float32)
    download_path = Path("downloaded_large_file.npy")
    np.save(download_path, example_data)

    size_mb = download_path.stat().st_size / (1024 * 1024)
    print(f"✅ 'Downloaded' file: {download_path.name} ({size_mb:.1f}MB)")
    print(f"💡 In practice, this could be a multi-GB file downloaded from:")
    print(f"   • Cloud storage (S3, GCS, Azure)")
    print(f"   • HTTP/HTTPS URLs")
    print(f"   • FTP servers")
    print(f"   • Generated programmatically")

    return str(download_path)


def main():
    """Main function."""
    print("🚀 Chisel Upload Limit Test Script")
    print("=" * 60)

    # Test file access
    result = test_file_access()

    # Demonstrate downloading large files
    downloaded_file = demonstrate_large_file_download()

    print(f"\n" + "=" * 60)
    print(f"🎯 Key Takeaways:")
    print(f"  • Caching is disabled - upload any folder size")
    print(f"  • Individual files limited to 5GB")
    print(f"  • Large files should be downloaded within scripts")
    print(f"  • This approach is faster and more flexible")
    print(f"=" * 60)

    return {"file_access_result": result, "downloaded_file": downloaded_file}


if __name__ == "__main__":
    main()
