#!/usr/bin/env python3
"""
Test script to verify all examples can be imported and run locally.
This helps ensure examples work before running on cloud GPUs.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def test_import(file_path):
    """Test if a Python file can be imported successfully."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec is None:
            return False, "Could not create module spec"

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "Import successful"
    except Exception as e:
        return False, str(e)


def find_python_files(directory):
    """Find all Python files in the examples directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        dirs[:] = [d for d in dirs if d != "__pycache__"]

        for file in files:
            if file.endswith(".py") and not file.startswith("test_"):
                python_files.append(os.path.join(root, file))

    return sorted(python_files)


def main():
    """Test all example files."""
    print("🧪 Testing All Keys & Caches Examples")
    print("=" * 50)

    # Get examples directory
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Examples directory: {examples_dir}")

    # Find all Python files
    python_files = find_python_files(examples_dir)
    print(f"📊 Found {len(python_files)} Python files to test")

    # Test each file
    results = []

    for file_path in python_files:
        rel_path = os.path.relpath(file_path, examples_dir)
        print(f"\n🔍 Testing: {rel_path}")

        # Test import
        success, message = test_import(file_path)

        if success:
            print(f"   ✅ Import: {message}")
            results.append((rel_path, True, "Import successful"))
        else:
            print(f"   ❌ Import: {message}")
            results.append((rel_path, False, f"Import failed: {message}"))

    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 50)

    successful = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")

    if successful < total:
        print(f"\n❌ Failed Tests:")
        for file_path, success, message in results:
            if not success:
                print(f"   {file_path}: {message}")

    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"   1. Fix any import errors shown above")
    print(f"   2. Test examples locally: python basic_models/simple_cnn.py")
    print(f"   3. Run on cloud GPU: kandc python basic_models/simple_cnn.py")
    print(f"   4. Check requirements: pip install -r requirements.txt")

    # Example commands
    print(f"\n🚀 Example Commands:")
    print(f"   # Basic model")
    print(f"   kandc python basic_models/simple_cnn.py")
    print(f"   ")
    print(f"   # With arguments")
    print(f"   kandc python edge_cases/model_with_args.py --model-size large")
    print(f"   ")
    print(f"   # Custom requirements")
    print(
        f"   kandc python nlp_models/transformer_example.py --requirements requirements_examples/nlp_requirements.txt"
    )

    return successful == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
