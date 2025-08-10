# Large File Examples

This directory contains examples demonstrating how to work with very large files (2GB+) using Keys & Caches.

## 2GB+ File Processor Example

**File:** `2gb_file_processor.py`

This example demonstrates:
- Creating a 2GB+ data file
- Loading and manipulating large data
- Processing data through a PyTorch model decorated with `@capture_model_class`
- Automatic profiling of model operations

### What it does:

1. **Expects a 2GB+ file** to already exist (you create it beforehand)
2. **Loads the data** (with memory limits for safety)
3. **Manipulates the data** - calculates statistics, normalizes, filters, creates histograms
4. **Processes through a model** - uses a simple neural network to process the data
5. **Automatically profiles** all model operations using Keys & Caches

### Key Features:

- **Memory efficient**: Processes data in chunks to handle large files
- **User manipulation**: Demonstrates various ways to analyze and transform the data
- **Automatic profiling**: All model operations are automatically captured and profiled
- **Simple model**: Uses a basic PyTorch model with `@capture_model_class` decorator

### Usage:

1. **First, create a 2GB+ file** (you can use any method):
   ```bash
   # Option 1: Using dd (Linux/macOS)
   dd if=/dev/urandom of=large_data_2gb.bin bs=1M count=2100
   
   # Option 2: Using fallocate (Linux)
   fallocate -l 2.1G large_data_2gb.bin
   
   # Option 3: Using your own data generation script
   # (create a file with random float32 data)
   ```

2. **Then run the example**:
   ```bash
   cd kandc/examples/large_file_examples
   python 2gb_file_processor.py
   ```

### What you'll see:

- File size verification and validation
- Data statistics and manipulation examples
- Model processing with automatic profiling
- Summary of operations performed

### Requirements:

- PyTorch
- NumPy
- Keys & Caches (`kandc` package)
- Sufficient disk space for 2GB+ file

### Notes:

- The example limits data loading to 200,000 samples for memory safety
- You need to create the 2GB+ file before running the script
- All model operations are automatically profiled and saved as trace files
- Cleanup instructions are provided to remove the large file when done
