"""
Test script for demonstrating upload limits and disabled caching in Chisel.

This script creates test files of various sizes to demonstrate:
1. Files under 5GB upload successfully (caching disabled)
2. Files over 5GB are rejected with helpful error messages
3. User guidance for handling large files within scripts

Run this script to test the upload limit functionality.
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
from chisel import capture_trace


def create_test_file(file_path: Path, size_gb: float):
    """Create a test file of specified size in GB."""
    print(f"Creating test file: {file_path.name} ({size_gb}GB)")

    # Calculate number of elements needed for the desired size
    # Using float32 (4 bytes per element)
    bytes_per_gb = 1024 * 1024 * 1024
    target_bytes = int(size_gb * bytes_per_gb)
    elements_needed = target_bytes // 4  # 4 bytes per float32

    # Create and save the array
    data = np.random.random(elements_needed).astype(np.float32)
    np.save(file_path, data)

    # Verify the file size
    actual_size = file_path.stat().st_size
    actual_gb = actual_size / bytes_per_gb
    print(f"✅ Created: {file_path.name} - {actual_gb:.2f}GB")

    return actual_gb


@capture_trace(trace_name="upload_limit_demo")
def demonstrate_upload_limits():
    """Demonstrate the upload limit functionality."""
    print("🚀 Chisel Upload Limits Test")
    print("=" * 50)

    # Create a temporary test directory
    test_dir = Path("upload_limit_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    try:
        print("\n📝 Step 1: Creating test files...")

        # Create a small file (should work)
        small_file = test_dir / "small_data.npy"
        create_test_file(small_file, 0.1)  # 100MB

        # Create a medium file (should work)
        medium_file = test_dir / "medium_data.npy"
        create_test_file(medium_file, 2.0)  # 2GB

        # Create a large file that exceeds the limit (should be rejected)
        large_file = test_dir / "large_data.npy"
        create_test_file(large_file, 6.0)  # 6GB - exceeds 5GB limit

        print(f"\n📊 Test Directory Contents:")
        total_size = 0
        for file_path in test_dir.iterdir():
            if file_path.is_file():
                size_bytes = file_path.stat().st_size
                size_gb = size_bytes / (1024 * 1024 * 1024)
                total_size += size_gb
                status = "✅ OK" if size_gb <= 5.0 else "❌ TOO LARGE"
                print(f"  {file_path.name}: {size_gb:.2f}GB {status}")

        print(f"\n📈 Total directory size: {total_size:.2f}GB")

        print(f"\n🧪 Expected Behavior:")
        print(f"  • small_data.npy (0.1GB): ✅ Should upload successfully")
        print(f"  • medium_data.npy (2.0GB): ✅ Should upload successfully")
        print(f"  • large_data.npy (6.0GB): ❌ Should be rejected with error")

        print(f"\n💡 Key Changes Demonstrated:")
        print(f"  🔄 Caching is now DISABLED - no 1GB threshold")
        print(f"  📏 New 5GB per-file upload limit")
        print(f"  📁 Total folder size can exceed 5GB (individual files cannot)")
        print(f"  💬 Helpful error messages for oversized files")

        print(f"\n🔧 To test this with Chisel CLI:")
        print(f"  chisel run --upload-dir {test_dir} your_script.py")
        print(f"  (This will show the 5GB limit error for large_data.npy)")

        return {
            "test_directory": str(test_dir),
            "files_created": [
                {"name": "small_data.npy", "size_gb": 0.1, "should_pass": True},
                {"name": "medium_data.npy", "size_gb": 2.0, "should_pass": True},
                {"name": "large_data.npy", "size_gb": 6.0, "should_pass": False},
            ],
            "total_size_gb": total_size,
        }

    except Exception as e:
        print(f"❌ Error creating test files: {e}")
        return {"error": str(e)}


def demonstrate_large_file_workaround():
    """Show how to handle large files within a script."""
    print(f"\n" + "=" * 50)
    print(f"📚 Large File Workaround Examples")
    print(f"=" * 50)

    print(f"""
🔧 Instead of uploading large files, download them in your script:

Example 1 - Download from URL:
```python
import requests
import numpy as np
from pathlib import Path

def download_large_file():
    url = "https://example.com/large_model.bin"
    local_path = Path("large_model.bin")
    
    print("Downloading large file...")
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return local_path
```

Example 2 - Download from cloud storage:
```python
import boto3
from pathlib import Path

def download_from_s3():
    s3 = boto3.client('s3')
    bucket = 'my-bucket'
    key = 'large-files/model.bin'
    local_path = Path('model.bin')
    
    print("Downloading from S3...")
    s3.download_file(bucket, key, str(local_path))
    
    return local_path
```

Example 3 - Generate large data in script:
```python
import numpy as np

def generate_large_dataset():
    print("Generating large dataset...")
    # Generate 10GB of data
    data = np.random.random((2500000000,)).astype(np.float32)
    np.save('large_dataset.npy', data)
    return data
```

💡 Benefits:
  • No upload time for large files
  • Faster job startup
  • Better resource utilization
  • More flexible data management
""")


def cleanup_test_files():
    """Clean up test files."""
    test_dir = Path("upload_limit_test")
    if test_dir.exists():
        print(f"\n🧹 Cleaning up test files...")
        shutil.rmtree(test_dir)
        print(f"✅ Removed {test_dir}")


def main():
    """Main function to run the upload limits test."""
    print("🚀 Starting Chisel Upload Limits Test")
    print("=" * 60)

    try:
        # Run the demonstration
        result = demonstrate_upload_limits()

        # Show workaround examples
        demonstrate_large_file_workaround()

        print(f"\n" + "=" * 60)
        print(f"✅ Upload Limits Test Completed!")
        print(f"📁 Test directory created: upload_limit_test/")
        print(f"🧪 Try running: chisel run --upload-dir upload_limit_test your_script.py")
        print(f"🧹 Run with --cleanup flag to remove test files")
        print(f"=" * 60)

        return result

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import sys

    # Check for cleanup flag
    if "--cleanup" in sys.argv:
        cleanup_test_files()
    else:
        main()
        print(f"\n💡 Tip: Run 'python test_upload_limits.py --cleanup' to remove test files")
