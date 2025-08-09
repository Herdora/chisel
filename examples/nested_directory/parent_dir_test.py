#!/usr/bin/env python3
"""
Parent directory test example.
Tests uploading from a parent directory (../examples).
Run from chisel root: kandc python examples/nested_directory/parent_dir_test.py --upload-dir ../examples
"""

import torch
import torch.nn as nn
import os
import sys
import argparse

# Add parent directories to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)

from kandc import capture_model_class


@capture_model_class(model_name="ParentDirModel")
class ParentDirModel(nn.Module):
    """Model for testing parent directory uploads."""

    def __init__(self, config_from_args):
        super().__init__()
        self.config = config_from_args

        # Build model based on config
        layers = []
        input_size = self.config.get("input_size", 64)
        hidden_sizes = self.config.get("hidden_sizes", [128, 64, 32])
        output_size = self.config.get("output_size", 10)

        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.2)])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def load_config_from_file(config_path):
    """Load configuration from a file (if it exists)."""
    if os.path.exists(config_path):
        print(f"📋 Loading config from: {config_path}")
        # Simple config loading (in practice, you might use JSON/YAML)
        config = {"input_size": 64, "hidden_sizes": [256, 128, 64], "output_size": 20}
        return config
    else:
        print(f"⚠️  Config file not found: {config_path}")
        return {"input_size": 64, "hidden_sizes": [128, 64, 32], "output_size": 10}


def test_file_access():
    """Test accessing files from different directory levels."""
    print("📁 Testing file access from different levels:")

    current_dir = os.path.dirname(__file__)

    # Test accessing files at different levels
    test_paths = [
        # Current directory
        os.path.join(current_dir, "parent_dir_test.py"),
        # Parent directory
        os.path.join(current_dir, "..", "requirements.txt"),
        # Sibling directories
        os.path.join(current_dir, "..", "basic_models"),
        # Deep nested
        os.path.join(current_dir, "deep", "deeper", "deepest_model.py"),
    ]

    for path in test_paths:
        abs_path = os.path.abspath(path)
        exists = os.path.exists(abs_path)
        rel_path = os.path.relpath(abs_path, current_dir)

        status = "✅" if exists else "❌"
        print(f"   {status} {rel_path}")
        if exists:
            if os.path.isfile(abs_path):
                size = os.path.getsize(abs_path)
                print(f"      File size: {size} bytes")
            elif os.path.isdir(abs_path):
                try:
                    contents = os.listdir(abs_path)
                    print(f"      Directory contents: {len(contents)} items")
                except PermissionError:
                    print(f"      Directory (permission denied)")


def main():
    """Test parent directory model."""
    parser = argparse.ArgumentParser(description="Parent Directory Test")
    parser.add_argument("--config-path", default="config.json", help="Path to config file")
    parser.add_argument("--test-upload", action="store_true", help="Test upload directory behavior")

    args = parser.parse_args()

    print("🚀 Parent Directory Test")
    print("=" * 35)

    # Show directory information
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Script location: {__file__}")
    print(f"📁 Script directory: {os.path.dirname(__file__)}")

    if args.test_upload:
        test_file_access()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🎯 Using device: {device}")

    # Load configuration
    config = load_config_from_file(args.config_path)
    print(f"📋 Model config: {config}")

    # Create model
    model = ParentDirModel(config).to(device)
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test model
    input_size = config["input_size"]
    batch_size = 8

    print(f"\n🧪 Testing model:")
    x = torch.randn(batch_size, input_size).to(device)

    model.eval()
    with torch.no_grad():
        for i in range(3):
            output = model(x)
            print(f"   Pass {i + 1}: {x.shape} -> {output.shape}")
            print(f"   Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    print(f"\n✅ Parent directory test completed!")
    print(f"💡 This tests:")
    print(f"   - Running from nested directories")
    print(f"   - Accessing parent directory files")
    print(f"   - Upload directory configuration")
    print(f"\n🚀 Example usage:")
    print(f"   kandc python nested_directory/parent_dir_test.py")
    print(f"   kandc python nested_directory/parent_dir_test.py --upload-dir ../examples")


if __name__ == "__main__":
    main()
