# Upload Limits Test

This directory contains test scripts to demonstrate Chisel's new upload limit functionality and disabled caching system.

## Overview

Chisel now has the following behavior:
- ✅ **Caching is DISABLED** - No more 1GB minimum file size threshold
- ✅ **5GB per-file upload limit** - Individual files cannot exceed 5GB
- ✅ **Unlimited folder size** - Total folder size can exceed 5GB
- ✅ **Clear error messages** - Helpful guidance for handling large files

## Test Files

### `test_upload_limits.py`
Creates test files to demonstrate the upload limits:
- `small_data.npy` (100MB) - Should upload successfully
- `medium_data.npy` (2GB) - Should upload successfully  
- `large_data.npy` (6GB) - Should be rejected with 5GB limit error

### `simple_upload_test.py`
The actual script that runs in Chisel to test file access and demonstrate large file handling patterns.

## Running the Test

### Step 1: Create Test Files
```bash
cd examples/
python test_upload_limits.py
```

This creates a `upload_limit_test/` directory with test files of various sizes.

### Step 2: Test Upload Limits
```bash
chisel run --upload-dir upload_limit_test simple_upload_test.py
```

**Expected Output:**
- ✅ Small and medium files upload successfully
- ❌ Large file (6GB) is rejected with clear error message:
  ```
  ❌ Error: Found 1 file(s) over 5GB limit
    • large_data.npy: 6.00GB
  📝 Solution: Download these files within your script instead
     Example: Use requests.get(), urllib.request, or similar to download
     the files during script execution rather than uploading them.
  ```

### Step 3: Cleanup (Optional)
```bash
python test_upload_limits.py --cleanup
```

## Large File Handling Examples

Instead of uploading large files, download them within your script:

### Example 1: HTTP Download
```python
import requests
from pathlib import Path

def download_large_model():
    url = "https://huggingface.co/model/large_model.bin"
    local_path = Path("large_model.bin")
    
    print("Downloading large model...")
    response = requests.get(url, stream=True)
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return local_path
```

### Example 2: Cloud Storage
```python
import boto3

def download_from_s3():
    s3 = boto3.client('s3')
    s3.download_file('my-bucket', 'large-files/data.bin', 'data.bin')
    return Path('data.bin')
```

### Example 3: Generate Data
```python
import numpy as np

def create_large_dataset():
    # Generate large dataset programmatically
    data = np.random.random((1000000000,)).astype(np.float32)  # 4GB
    np.save('large_dataset.npy', data)
    return data
```

## Benefits of New System

### ✅ Advantages
- **Faster uploads** - No caching overhead
- **Simpler workflow** - Upload folders directly
- **Better error handling** - Clear messages for oversized files
- **More flexible** - Download large files as needed
- **Resource efficient** - No duplicate storage

### 🔄 Migration from Old System
- **Before**: Files >1GB were automatically cached
- **After**: All files upload directly (up to 5GB per file)
- **Large files**: Download within scripts instead of uploading

## Troubleshooting

### "File over 5GB limit" Error
**Solution**: Download the file within your script instead of uploading it.

**Why**: Individual files over 5GB cause slow uploads and storage issues. It's more efficient to download them during script execution.

### Total Folder Size Concerns
**No problem**: Total folder size can exceed 5GB. Only individual files are limited to 5GB each.

### Missing Caching Benefits
**New approach**: Instead of caching, download files on-demand within scripts. This is often faster and more flexible than the old caching system.

## Testing Different Scenarios

You can modify `test_upload_limits.py` to test different file sizes:

```python
# Create different test scenarios
create_test_file(test_dir / "tiny.npy", 0.01)    # 10MB
create_test_file(test_dir / "small.npy", 0.5)    # 500MB  
create_test_file(test_dir / "medium.npy", 2.5)   # 2.5GB
create_test_file(test_dir / "large.npy", 4.9)    # 4.9GB (just under limit)
create_test_file(test_dir / "too_large.npy", 5.1) # 5.1GB (over limit)
```

This helps verify the exact behavior at the 5GB boundary.
