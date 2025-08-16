#!/usr/bin/env python3
"""
CLI tool for running kandc sweeps.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .sweep import SweepManager, SweepConfig, sweep_folder, sweep_files


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the sweep CLI."""
    parser = argparse.ArgumentParser(
        description="Run kandc sweeps across multiple configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sweep all configs in a folder
  python -m kandc.cli_sweep sweep-folder ./configs --project "my_experiment"
  
  # Sweep specific config files
  python -m kandc.cli_sweep sweep-files config1.json config2.json --project "test_run"
  
  # Sweep with custom output
  python -m kandc.cli_sweep sweep-folder ./configs --output results.json --device cuda
        """,
    )

    # Common arguments
    parser.add_argument(
        "--project",
        default="sweep_experiment",
        help="Project name for kandc (default: sweep_experiment)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output file for results (default: sweep_results.json)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run on (default: auto)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sweep folder command
    folder_parser = subparsers.add_parser("sweep-folder", help="Sweep all config files in a folder")
    folder_parser.add_argument("folder", type=Path, help="Folder containing JSON config files")

    # Sweep files command
    files_parser = subparsers.add_parser("sweep-files", help="Sweep specific config files")
    files_parser.add_argument("files", nargs="+", type=Path, help="JSON config files to process")

    # List configs command
    list_parser = subparsers.add_parser(
        "list-configs", help="List available configs without running them"
    )
    list_parser.add_argument("path", type=Path, help="Folder or file to list configs from")

    return parser


def list_configs(path: Path, verbose: bool = False):
    """List available configurations without running them."""
    if path.is_file():
        # Single file
        try:
            with open(path, "r") as f:
                config_data = json.load(f)
            name = config_data.get("name", path.stem)
            print(f"📄 {name} ({path})")
            if verbose:
                print(f"   Model size: {config_data.get('model_size', 'N/A')}")
                print(f"   Batch size: {config_data.get('batch_size', 'N/A')}")
                print(f"   Tasks: {config_data.get('tasks', 'N/A')}")
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")

    elif path.is_dir():
        # Directory
        config_files = list(path.glob("*.json"))
        if not config_files:
            print(f"📁 No JSON config files found in {path}")
            return

        print(f"📁 Found {len(config_files)} config files in {path}:")
        for config_file in config_files:
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                name = config_data.get("name", config_file.stem)
                print(f"  📄 {name} ({config_file.name})")
                if verbose:
                    print(f"     Model size: {config_data.get('model_size', 'N/A')}")
                    print(f"     Batch size: {config_data.get('batch_size', 'N/A')}")
                    print(f"     Tasks: {config_data.get('tasks', 'N/A')}")
            except Exception as e:
                print(f"  ❌ Error reading {config_file.name}: {e}")

    else:
        print(f"❌ Path not found: {path}")


def run_sweep_folder(args) -> int:
    """Run sweep on a folder of configs."""
    folder = args.folder

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return 1

    if not folder.is_dir():
        print(f"❌ Path is not a directory: {folder}")
        return 1

    print(f"🔍 Sweeping configs in folder: {folder}")

    try:
        # For now, we'll use a simple model factory
        # In practice, you'd want to provide this via command line or config
        def simple_model_factory(config):
            import torch.nn as nn

            model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
            return model

        def simple_input_factory(config):
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.randn(config.batch_size, 784, device=device)

        results = sweep_folder(
            folder_path=folder,
            project_name=args.project,
            model_factory=simple_model_factory,
            input_factory=simple_input_factory,
            device=args.device,
            output_path=args.output or "sweep_results.json",
        )

        print(f"✅ Sweep completed! Generated {len(results)} results")
        return 0

    except Exception as e:
        print(f"❌ Sweep failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def run_sweep_files(args) -> int:
    """Run sweep on specific config files."""
    files = args.files

    # Check if all files exist
    for file_path in files:
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return 1

    print(f"🔍 Sweeping {len(files)} config files")

    try:
        # Simple model factory (same as above)
        def simple_model_factory(config):
            import torch.nn as nn

            model = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
            return model

        def simple_input_factory(config):
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return torch.randn(config.batch_size, 784, device=device)

        results = sweep_files(
            config_files=files,
            project_name=args.project,
            model_factory=simple_model_factory,
            input_factory=simple_input_factory,
            device=args.device,
            output_path=args.output or "sweep_results.json",
        )

        print(f"✅ Sweep completed! Generated {len(results)} results")
        return 0

    except Exception as e:
        print(f"❌ Sweep failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


def main():
    """Main entry point for the sweep CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO)

    # Handle commands
    if args.command == "sweep-folder":
        return run_sweep_folder(args)
    elif args.command == "sweep-files":
        return run_sweep_files(args)
    elif args.command == "list-configs":
        list_configs(args.path, args.verbose)
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
