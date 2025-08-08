import os
import sys
import subprocess
import tempfile
import tarfile
import requests
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from .auth import _auth_service
from .constants import (
    CHISEL_BACKEND_URL,
    CHISEL_BACKEND_URL_ENV_KEY,
    MINIMUM_PACKAGES,
    GPUType,
)
from .spinner import SimpleSpinner
from .cached_files import process_directory_for_cached_files, scan_directory_for_large_files

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.layout import Layout
    from rich.columns import Columns
    from rich import box
    from rich.live import Live
    from rich.align import Align

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

EXCLUDE_PATTERNS = {
    ".venv",
    "venv",
    ".env",
    "__pycache__",
    ".git",
}


def load_ignore_patterns(directory: Path) -> tuple[set, set]:
    """
    Load patterns from .gitignore and .chiselignore files if they exist.

    Returns:
        tuple: (gitignore_patterns, chiselignore_patterns)
    """
    gitignore_patterns = set()
    chiselignore_patterns = set()

    # Load .gitignore patterns
    gitignore_path = directory / ".gitignore"
    if gitignore_path.exists():
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        # Remove leading slash if present
                        if line.startswith("/"):
                            line = line[1:]
                        # Add pattern to set
                        gitignore_patterns.add(line)
        except (IOError, UnicodeDecodeError):
            # If we can't read .gitignore, just continue
            pass

    # Load .chiselignore patterns
    chiselignore_path = directory / ".chiselignore"
    if chiselignore_path.exists():
        try:
            with open(chiselignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        # Remove leading slash if present
                        if line.startswith("/"):
                            line = line[1:]
                        # Add pattern to set
                        chiselignore_patterns.add(line)
        except (IOError, UnicodeDecodeError):
            # If we can't read .chiselignore, just continue
            pass

    return gitignore_patterns, chiselignore_patterns


def should_exclude(path, gitignore_patterns=None, chiselignore_patterns=None):
    """
    Check if a path should be excluded from upload.

    Args:
        path: Path to check (relative to upload directory)
        gitignore_patterns: Set of gitignore patterns (optional)
        chiselignore_patterns: Set of chiselignore patterns (optional)

    Returns:
        True if the path should be excluded, False otherwise
    """
    path_obj = Path(path)
    path_parts = path_obj.parts

    # Check built-in exclude patterns
    for part in path_parts:
        if part in EXCLUDE_PATTERNS:
            return True

    # Helper function to check patterns
    def matches_patterns(patterns):
        if not patterns:
            return False

        path_str = str(path_obj)
        for pattern in patterns:
            # Simple pattern matching - exact match or directory match
            if path_str == pattern or path_str.startswith(pattern + "/"):
                return True
            # Check if any part of the path matches the pattern
            if pattern in path_parts:
                return True
            # Handle wildcard patterns (basic support)
            if "*" in pattern:
                import fnmatch

                if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_obj.name, pattern):
                    return True
        return False

    # Check chiselignore patterns first (takes precedence)
    if matches_patterns(chiselignore_patterns):
        return True

    # Check gitignore patterns
    if matches_patterns(gitignore_patterns):
        return True

    return False


def tar_filter(tarinfo, gitignore_patterns=None, chiselignore_patterns=None):
    if should_exclude(tarinfo.name, gitignore_patterns, chiselignore_patterns):
        return None
    return tarinfo


def preview_upload_directory(upload_dir: Path, console=None) -> Dict[str, Any]:
    """
    Preview what files will be uploaded vs excluded from the upload directory.

    Args:
        upload_dir: Path to the directory to analyze
        console: Rich console instance for styled output (optional)

    Returns:
        Dict containing included_files, excluded_files, and summary stats
    """
    included_files = []
    excluded_files = []
    total_size = 0
    large_files = []

    # Load ignore patterns
    gitignore_patterns, chiselignore_patterns = load_ignore_patterns(upload_dir)

    # Walk through the directory
    for root, dirs, files in os.walk(upload_dir):
        # Filter out excluded directories to avoid walking them
        dirs[:] = [
            d
            for d in dirs
            if not should_exclude(os.path.join(root, d), gitignore_patterns, chiselignore_patterns)
        ]

        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(upload_dir)

            if should_exclude(str(relative_path), gitignore_patterns, chiselignore_patterns):
                excluded_files.append(str(relative_path))
            else:
                try:
                    file_size = file_path.stat().st_size
                    total_size += file_size

                    # Check if it's a large file (>1GB)
                    if file_size > 1024 * 1024 * 1024:  # 1GB
                        large_files.append(
                            {
                                "path": str(relative_path),
                                "size": file_size,
                                "size_mb": file_size / (1024 * 1024),
                            }
                        )

                    included_files.append({"path": str(relative_path), "size": file_size})
                except (OSError, PermissionError):
                    # Skip files we can't access
                    excluded_files.append(f"{relative_path} (access denied)")

    return {
        "included_files": included_files,
        "excluded_files": excluded_files,
        "large_files": large_files,
        "total_files": len(included_files),
        "excluded_count": len(excluded_files),
        "total_size": total_size,
        "total_size_mb": total_size / (1024 * 1024) if total_size > 0 else 0,
        "gitignore_patterns": gitignore_patterns,
        "chiselignore_patterns": chiselignore_patterns,
    }


def _build_file_tree(files_list, excluded_files=None):
    """
    Build a tree structure from a list of file paths, including both included and excluded files.

    Args:
        files_list: List of file info dicts with 'path' and 'size' keys (included files)
        excluded_files: List of excluded file paths (optional)

    Returns:
        Dict representing the tree structure
    """
    tree = {}

    # Add included files
    for file_info in files_list:
        path_parts = Path(file_info["path"]).parts
        current_level = tree

        # Navigate/create the directory structure
        for i, part in enumerate(path_parts[:-1]):  # All but the last part (filename)
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]

        # Add the file (last part)
        if len(path_parts) > 0:
            filename = path_parts[-1]
            current_level[filename] = {"size": file_info["size"], "included": True}

    # Add excluded files
    if excluded_files:
        for excluded_path in excluded_files:
            # Skip access denied entries (they have additional text)
            if "(access denied)" in excluded_path:
                continue

            path_parts = Path(excluded_path).parts
            current_level = tree

            # Navigate/create the directory structure
            for i, part in enumerate(path_parts[:-1]):  # All but the last part (filename)
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]

            # Add the excluded file
            if len(path_parts) > 0:
                filename = path_parts[-1]
                current_level[filename] = {
                    "size": 0,
                    "included": False,
                }  # Size unknown for excluded files

    return tree


def _display_files_tree_rich(files_list, console, upload_dir="", excluded_files=None, max_files=20):
    """Display files in a tree structure using Rich, showing both included and excluded files."""
    if not files_list and not excluded_files:
        console.print("  [dim]No files found[/dim]")
        return

    tree_dict = _build_file_tree(files_list, excluded_files)
    files_shown = [0]  # Use list to make it mutable in nested function

    # Show the root directory name
    if upload_dir == "." or upload_dir == "":
        root_name = Path.cwd().name
    else:
        root_name = Path(upload_dir).name if upload_dir else "."
    console.print(f"📁 [bold cyan]{root_name}/[/bold cyan]")
    files_shown[0] += 1

    def _print_tree_rich(tree, prefix="", is_last=True, level=0):
        if files_shown[0] >= max_files:
            return

        items = list(tree.items())
        for i, (name, content) in enumerate(items):
            if files_shown[0] >= max_files:
                break

            is_last_item = i == len(items) - 1

            # Choose the right tree characters
            current_prefix = prefix + ("└── " if is_last_item else "├── ")
            next_prefix = prefix + ("    " if is_last_item else "│   ")

            if isinstance(content, dict) and not ("size" in content and "included" in content):
                # It's a directory
                console.print(f"{current_prefix}📁 [bold cyan]{name}/[/bold cyan]")
                files_shown[0] += 1
                _print_tree_rich(content, next_prefix, is_last_item, level + 1)
            else:
                # It's a file (either included or excluded)
                if content.get("included", True):
                    # Included file - show only if we're displaying included files
                    if files_list:  # If files_list is not empty, we're showing included files
                        console.print(f"{current_prefix}📄 [green]{name}[/green]")
                        files_shown[0] += 1
                else:
                    # Excluded file - show only if we're displaying excluded files
                    if (
                        excluded_files and not files_list
                    ):  # If files_list is empty but excluded_files exists
                        console.print(f"{current_prefix}📄 [dim red]{name}[/dim red]")
                        files_shown[0] += 1
                    elif files_list:  # Show excluded files with label when showing both
                        console.print(
                            f"{current_prefix}📄 [dim red]{name}[/dim red] [dim](excluded)[/dim]"
                        )
                        files_shown[0] += 1

    _print_tree_rich(tree_dict, "", True, 1)  # Start with level 1 since we showed root

    # Calculate total files (excluding the root directory from the count)
    total_files = len(files_list) + (len(excluded_files) if excluded_files else 0)
    files_actually_shown = files_shown[0] - 1  # Subtract 1 for the root directory

    if files_actually_shown < total_files:
        remaining = total_files - files_actually_shown
        console.print(f"  [dim]... and {remaining} more files[/dim]")


def _display_files_tree_plain(files_list, upload_dir="", excluded_files=None, max_files=20):
    """Display files in a tree structure using plain text, showing both included and excluded files."""
    if not files_list and not excluded_files:
        print("  No files found")
        return

    tree_dict = _build_file_tree(files_list, excluded_files)
    files_shown = [0]  # Use list to make it mutable in nested function

    # Show the root directory name
    if upload_dir == "." or upload_dir == "":
        root_name = Path.cwd().name
    else:
        root_name = Path(upload_dir).name if upload_dir else "."
    print(f"📁 {root_name}/")
    files_shown[0] += 1

    def _print_tree_plain(tree, prefix="", is_last=True, level=0):
        if files_shown[0] >= max_files:
            return

        items = list(tree.items())
        for i, (name, content) in enumerate(items):
            if files_shown[0] >= max_files:
                break

            is_last_item = i == len(items) - 1

            # Choose the right tree characters
            current_prefix = prefix + ("└── " if is_last_item else "├── ")
            next_prefix = prefix + ("    " if is_last_item else "│   ")

            if isinstance(content, dict) and not ("size" in content and "included" in content):
                # It's a directory
                print(f"{current_prefix}📁 {name}/")
                files_shown[0] += 1
                _print_tree_plain(content, next_prefix, is_last_item, level + 1)
            else:
                # It's a file (either included or excluded)
                if content.get("included", True):
                    # Included file - show only if we're displaying included files
                    if files_list:  # If files_list is not empty, we're showing included files
                        print(f"{current_prefix}📄 {name}")
                        files_shown[0] += 1
                else:
                    # Excluded file - show only if we're displaying excluded files
                    if (
                        excluded_files and not files_list
                    ):  # If files_list is empty but excluded_files exists
                        print(f"{current_prefix}📄 {name}")
                        files_shown[0] += 1
                    elif files_list:  # Show excluded files with label when showing both
                        print(f"{current_prefix}📄 {name} (excluded)")
                        files_shown[0] += 1

    _print_tree_plain(tree_dict, "", True, 1)  # Start with level 1 since we showed root

    # Calculate total files (excluding the root directory from the count)
    total_files = len(files_list) + (len(excluded_files) if excluded_files else 0)
    files_actually_shown = files_shown[0] - 1  # Subtract 1 for the root directory

    if files_actually_shown < total_files:
        remaining = total_files - files_actually_shown
        print(f"  ... and {remaining} more files")


def display_submission_summary(
    preview_data: Dict[str, Any], upload_dir: str, app_name: str, gpu: str, console=None
):
    """
    Display a submission summary with file list and ask for confirmation.

    Args:
        preview_data: Data from preview_upload_directory()
        upload_dir: Path to upload directory (for display)
        app_name: Job name
        gpu: GPU configuration
        console: Rich console instance for styled output (optional)

    Returns:
        bool: True if user confirms submission, False otherwise
    """
    if console and RICH_AVAILABLE:
        # Rich formatted output
        console.print("\n[bold cyan]🚀 Job Submission Summary (Step 5 of 5)[/bold cyan]")
        console.print("─" * 60, style="dim")

        # Job details
        from rich.table import Table

        job_table = Table(box=box.ROUNDED, border_style="green", show_header=False)
        job_table.add_column("Field", style="cyan", width=20)
        job_table.add_column("Value", style="white")

        # Show actual directory name instead of relative paths
        if upload_dir == "." or upload_dir == "":
            display_upload_dir = Path.cwd().name
        elif upload_dir == "..":
            display_upload_dir = Path.cwd().parent.name
        else:
            display_upload_dir = Path(upload_dir).name if Path(upload_dir).name else upload_dir

        job_table.add_row("📝 Job Name", app_name)
        job_table.add_row("🎮 GPU Config", gpu)
        job_table.add_row("📁 Upload Dir", display_upload_dir)

        console.print(job_table)

        # Upload summary
        total_files = preview_data["total_files"]
        excluded_count = preview_data["excluded_count"]
        large_files_count = len(preview_data["large_files"])

        # Only show upload table if there are rows to display
        if excluded_count > 0 or large_files_count > 0:
            upload_table = Table(box=box.ROUNDED, border_style="blue", show_header=False)
            upload_table.add_column("Metric", style="cyan", width=20)
            upload_table.add_column("Value", style="white")

            if excluded_count > 0:
                upload_table.add_row("🚫 Files excluded", f"{excluded_count}")
            if large_files_count > 0:
                upload_table.add_row("🗂️  Large files", f"{large_files_count} (will be cached)")

            console.print(upload_table)

        # Show included files
        console.print(f"\n[bold green]✅ Files to Upload ({total_files}):[/bold green]")
        _display_files_tree_rich(preview_data["included_files"], console, upload_dir, max_files=15)

        # Show excluded files if any
        if excluded_count > 0:
            console.print(f"\n[bold red]🚫 Files Excluded ({excluded_count}):[/bold red]")
            _display_files_tree_rich(
                [], console, upload_dir, preview_data["excluded_files"], max_files=10
            )

        # Show large files if any (compact format)
        if preview_data["large_files"]:
            console.print("\n[yellow]🗂️  Large files (will be cached automatically):[/yellow]")
            for lf in preview_data["large_files"][:3]:  # Show first 3
                console.print(f"  • {lf['path']} ({lf['size_mb']:.1f} MB)")
            if len(preview_data["large_files"]) > 3:
                console.print(f"  ... and {len(preview_data['large_files']) - 3} more")

        # Ask for confirmation
        from rich.prompt import Confirm

        return Confirm.ask(
            "\n[bold yellow]🚀 Submit this job?[/bold yellow]", default=True, console=console
        )

    else:
        # Plain text output
        print("\n🚀 Job Submission Summary (Step 5 of 5)")
        print("─" * 60)

        # Show actual directory name instead of relative paths
        if upload_dir == "." or upload_dir == "":
            display_upload_dir = Path.cwd().name
        elif upload_dir == "..":
            display_upload_dir = Path.cwd().parent.name
        else:
            display_upload_dir = Path(upload_dir).name if Path(upload_dir).name else upload_dir

        print(f"📝 Job Name: {app_name}")
        print(f"🎮 GPU Config: {gpu}")
        print(f"📁 Upload Dir: {display_upload_dir}")

        total_files = preview_data["total_files"]
        excluded_count = preview_data["excluded_count"]
        large_files_count = len(preview_data["large_files"])

        # Only show additional info if there are excluded or large files
        if excluded_count > 0 or large_files_count > 0:
            print()  # Add spacing
            if excluded_count > 0:
                print(f"🚫 Files excluded: {excluded_count}")
            if large_files_count > 0:
                print(f"🗂️  Large files: {large_files_count} (will be cached)")

        # Show included files
        print(f"\n✅ Files to Upload ({total_files}):")
        _display_files_tree_plain(preview_data["included_files"], upload_dir, max_files=15)

        # Show excluded files if any
        if excluded_count > 0:
            print(f"\n🚫 Files Excluded ({excluded_count}):")
            _display_files_tree_plain([], upload_dir, preview_data["excluded_files"], max_files=10)

        # Show large files if any (compact format)
        if preview_data["large_files"]:
            print("\n🗂️  Large files (will be cached automatically):")
            for lf in preview_data["large_files"][:3]:  # Show first 3
                print(f"  • {lf['path']} ({lf['size_mb']:.1f} MB)")
            if len(preview_data["large_files"]) > 3:
                print(f"  ... and {len(preview_data['large_files']) - 3} more")

        # Ask for confirmation
        while True:
            response = input("\n🚀 Submit this job? (y/n, default: y): ").strip().lower()
            if not response or response == "y" or response == "yes":
                return True
            elif response == "n" or response == "no":
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")


def display_upload_preview(preview_data: Dict[str, Any], upload_dir: str, console=None):
    """
    Display a formatted preview of what will be uploaded.

    Args:
        preview_data: Data from preview_upload_directory()
        upload_dir: Path to upload directory (for display)
        console: Rich console instance for styled output (optional)
    """
    if console and RICH_AVAILABLE:
        # Rich formatted output
        console.print(f"\n[bold cyan]📁 Upload Directory Preview: {upload_dir}[/bold cyan]")
        console.print("─" * 60, style="dim")

        # Summary stats
        total_files = preview_data["total_files"]
        excluded_count = preview_data["excluded_count"]
        total_size_mb = preview_data["total_size_mb"]
        large_files_count = len(preview_data["large_files"])

        if total_size_mb < 1.0:
            size_display = f"{preview_data['total_size'] / 1024:.2f} KB"
        else:
            size_display = f"{total_size_mb:.2f} MB"

        # Create summary table
        from rich.table import Table

        summary_table = Table(box=box.ROUNDED, border_style="blue")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("📄 Files to upload", f"{total_files}")
        summary_table.add_row("🚫 Files excluded", f"{excluded_count}")
        summary_table.add_row("📦 Total upload size", size_display)
        if large_files_count > 0:
            summary_table.add_row("🗂️  Large files (>1GB)", f"{large_files_count} (will be cached)")

        console.print(summary_table)

        # Show large files if any
        if preview_data["large_files"]:
            console.print(
                f"\n[bold yellow]🗂️  Large Files (will be automatically cached):[/bold yellow]"
            )
            for lf in preview_data["large_files"]:
                console.print(f"  • {lf['path']} ({lf['size_mb']:.1f} MB)")

        # Show some included files (first 10)
        if preview_data["included_files"]:
            console.print(f"\n[bold green]✅ Files to Upload (showing first 10):[/bold green]")
            for i, file_info in enumerate(preview_data["included_files"][:10]):
                size_kb = file_info["size"] / 1024
                if size_kb < 1024:
                    size_str = f"({size_kb:.1f} KB)"
                else:
                    size_str = f"({size_kb / 1024:.1f} MB)"
                console.print(f"  • {file_info['path']} {size_str}")

            if len(preview_data["included_files"]) > 10:
                console.print(f"  ... and {len(preview_data['included_files']) - 10} more files")

        # Show excluded files if any (first 10)
        if preview_data["excluded_files"]:
            console.print(f"\n[bold red]🚫 Excluded Files/Patterns (showing first 10):[/bold red]")
            for i, excluded in enumerate(preview_data["excluded_files"][:10]):
                console.print(f"  • {excluded}")

            if len(preview_data["excluded_files"]) > 10:
                console.print(f"  ... and {len(preview_data['excluded_files']) - 10} more excluded")

        # Show exclusion patterns
        console.print(f"\n[bold dim]📋 Current Exclusion Patterns:[/bold dim]")
        for pattern in sorted(EXCLUDE_PATTERNS):
            console.print(f"  • {pattern}", style="dim")

        # Show gitignore patterns if any were found
        gitignore_patterns = preview_data.get("gitignore_patterns", set())
        if gitignore_patterns:
            console.print(f"\n[bold dim]📋 .gitignore Patterns:[/bold dim]")
            for pattern in sorted(gitignore_patterns):
                console.print(f"  • {pattern}", style="dim")

        # Show chiselignore patterns if any were found
        chiselignore_patterns = preview_data.get("chiselignore_patterns", set())
        if chiselignore_patterns:
            console.print(f"\n[bold dim]📋 .chiselignore Patterns:[/bold dim]")
            for pattern in sorted(chiselignore_patterns):
                console.print(f"  • {pattern}", style="dim")

    else:
        # Plain text output
        print(f"\n📁 Upload Directory Preview: {upload_dir}")
        print("─" * 60)

        total_files = preview_data["total_files"]
        excluded_count = preview_data["excluded_count"]
        total_size_mb = preview_data["total_size_mb"]
        large_files_count = len(preview_data["large_files"])

        if total_size_mb < 1.0:
            size_display = f"{preview_data['total_size'] / 1024:.2f} KB"
        else:
            size_display = f"{total_size_mb:.2f} MB"

        print(f"📄 Files to upload: {total_files}")
        print(f"🚫 Files excluded: {excluded_count}")
        print(f"📦 Total upload size: {size_display}")
        if large_files_count > 0:
            print(f"🗂️  Large files (>1GB): {large_files_count} (will be cached)")

        # Show large files if any
        if preview_data["large_files"]:
            print(f"\n🗂️  Large Files (will be automatically cached):")
            for lf in preview_data["large_files"]:
                print(f"  • {lf['path']} ({lf['size_mb']:.1f} MB)")

        # Show some included files
        if preview_data["included_files"]:
            print(f"\n✅ Files to Upload (showing first 10):")
            for i, file_info in enumerate(preview_data["included_files"][:10]):
                size_kb = file_info["size"] / 1024
                if size_kb < 1024:
                    size_str = f"({size_kb:.1f} KB)"
                else:
                    size_str = f"({size_kb / 1024:.1f} MB)"
                print(f"  • {file_info['path']} {size_str}")

            if len(preview_data["included_files"]) > 10:
                print(f"  ... and {len(preview_data['included_files']) - 10} more files")

        # Show excluded files if any
        if preview_data["excluded_files"]:
            print(f"\n🚫 Excluded Files/Patterns (showing first 10):")
            for i, excluded in enumerate(preview_data["excluded_files"][:10]):
                print(f"  • {excluded}")

            if len(preview_data["excluded_files"]) > 10:
                print(f"  ... and {len(preview_data['excluded_files']) - 10} more excluded")

        # Show exclusion patterns
        print(f"\n📋 Current Exclusion Patterns:")
        for pattern in sorted(EXCLUDE_PATTERNS):
            print(f"  • {pattern}")

        # Show gitignore patterns if any were found
        gitignore_patterns = preview_data.get("gitignore_patterns", set())
        if gitignore_patterns:
            print(f"\n📋 .gitignore Patterns:")
            for pattern in sorted(gitignore_patterns):
                print(f"  • {pattern}")

        # Show chiselignore patterns if any were found
        chiselignore_patterns = preview_data.get("chiselignore_patterns", set())
        if chiselignore_patterns:
            print(f"\n📋 .chiselignore Patterns:")
            for pattern in sorted(chiselignore_patterns):
                print(f"  • {pattern}")


class ChiselCLI:
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.gpu_options = [
            ("1", "A100-80GB:1", "Single GPU - Development, inference"),
            ("2", "A100-80GB:2", "2x GPUs - Medium training"),
            ("4", "A100-80GB:4", "4x GPUs - Large models"),
            ("8", "A100-80GB:8", "8x GPUs - Massive models"),
        ]
        self.gpu_map = {option: gpu_type for option, gpu_type, _ in self.gpu_options}

    def print_header(self):
        """Print the CLI header with styling."""
        if RICH_AVAILABLE:
            title = Text("🚀 Chisel CLI", style="bold blue")
            subtitle = Text("GPU Job Submission Tool", style="dim")

            header = Panel(
                title, subtitle=subtitle, border_style="blue", box=box.ROUNDED, padding=(1, 2)
            )
            self.console.print(header)
        else:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║                    🚀 Chisel CLI                            ║")
            print("║                GPU Job Submission Tool                      ║")
            print("╚══════════════════════════════════════════════════════════════╝")
            print()

    def print_section_header(self, title: str, step: int = None, total_steps: int = None):
        """Print a section header with optional step indicator."""
        step_indicator = f" (Step {step} of {total_steps})" if step and total_steps else ""
        full_title = f"{title}{step_indicator}"

        if RICH_AVAILABLE:
            self.console.print(f"\n[bold cyan]📋 {full_title}[/bold cyan]")
            self.console.print("─" * (len(full_title) + 4), style="dim")
        else:
            print(f"📋 {full_title}")
            print("─" * (len(full_title) + 4))

    def get_input_with_default(self, prompt: str, default: str = "", required: bool = True) -> str:
        """Get user input with a default value."""
        if RICH_AVAILABLE:
            if default:
                return Prompt.ask(f"{prompt}", default=default, console=self.console)
            else:
                while True:
                    user_input = Prompt.ask(f"{prompt}", console=self.console)
                    if user_input or not required:
                        return user_input
                    self.console.print("❌ This field is required!", style="red")
        else:
            if default:
                user_input = input(f"{prompt} (default: {default}): ").strip()
                return user_input if user_input else default
            else:
                while True:
                    user_input = input(f"{prompt}: ").strip()
                    if user_input or not required:
                        return user_input
                    print("❌ This field is required!")

    def select_gpu(self) -> str:
        """Interactive GPU selection with navigation."""
        if RICH_AVAILABLE:
            # Create a table for better presentation
            table = Table(
                title="Available GPU Configurations",
                box=box.ROUNDED,
                border_style="green",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Option", style="cyan", no_wrap=True, width=8)
            table.add_column("GPU Type", style="yellow", no_wrap=True, width=15)
            table.add_column("Description", style="white")

            for option, gpu_type, description in self.gpu_options:
                table.add_row(option, gpu_type, description)

            self.console.print(table)
            self.console.print()

            # Use a simple but effective selection method
            while True:
                choice = Prompt.ask(
                    "Select GPU configuration",
                    choices=["1", "2", "4", "8"],
                    default="1",
                    console=self.console,
                )

                if choice in self.gpu_map:
                    selected_gpu = self.gpu_map[choice]
                    self.console.print(f"✅ Selected: [bold green]{selected_gpu}[/bold green]")
                    return selected_gpu
                else:
                    self.console.print(
                        "❌ Invalid choice. Please select 1, 2, 4, or 8.", style="red"
                    )
        else:
            for option, gpu_type, description in self.gpu_options:
                print(f"  {option}. {gpu_type}")
                print(f"     {description}")
                print()

            while True:
                choice = input("Select GPU configuration (1-8, default: 1): ").strip()
                if not choice:
                    choice = "1"

                if choice in self.gpu_map:
                    selected_gpu = self.gpu_map[choice]
                    print(f"✅ Selected: {selected_gpu}")
                    return selected_gpu
                else:
                    print("❌ Invalid choice. Please select 1, 2, 4, or 8.")

    def get_user_inputs_interactive(self, script_path: str = "<script.py>") -> Dict[str, Any]:
        """Interactive questionnaire to get job submission parameters."""
        self.print_header()

        # Define total steps for progress tracking
        total_steps = 5

        # Step 1: App name
        self.print_section_header("Job Configuration", step=1, total_steps=total_steps)
        app_name = self.get_input_with_default("📝 App name (for job tracking)")

        # Step 2: Upload directory
        self.print_section_header("Upload Directory", step=2, total_steps=total_steps)
        upload_dir = self.get_input_with_default("📁 Upload directory", default=".", required=False)

        # Step 3: Requirements file
        self.print_section_header("Dependencies", step=3, total_steps=total_steps)
        requirements_file = self.get_input_with_default(
            "📋 Requirements file", default="requirements.txt", required=False
        )

        # Step 4: GPU selection
        self.print_section_header("GPU Configuration", step=4, total_steps=total_steps)
        gpu = self.select_gpu()

        # Show equivalent command for copy/paste
        self.show_equivalent_command(app_name, upload_dir, requirements_file, gpu, script_path)

        return {
            "app_name": app_name,
            "upload_dir": upload_dir,
            "requirements_file": requirements_file,
            "gpu": gpu,
        }

    def show_equivalent_command(
        self, app_name: str, upload_dir: str, requirements_file: str, gpu: str, script_path: str
    ):
        """Show the equivalent command-line command for copy/paste."""
        if RICH_AVAILABLE:
            # Build the command
            cmd_parts = ["chisel", "python", script_path]

            # Add flags
            if app_name:
                cmd_parts.append(f'--app-name "{app_name}"')

            if upload_dir != ".":
                cmd_parts.append(f'--upload-dir "{upload_dir}"')

            if requirements_file != "requirements.txt":
                cmd_parts.append(f'--requirements "{requirements_file}"')

            if gpu != "A100-80GB:1":
                # Convert GPU type back to number
                gpu_number = {v: k for k, v in self.gpu_map.items()}[gpu]
                cmd_parts.append(f"--gpu {gpu_number}")

            command = " ".join(cmd_parts)

            # Create a beautiful panel for the command
            command_text = Text(command, style="bold green")
            panel = Panel(
                command_text,
                title="📋 Equivalent Command (copy/paste for future use)",
                border_style="green",
                box=box.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(panel)
        else:
            print("\n" + "═" * 60)
            print("📋 Equivalent Command (copy/paste for future use):")
            print("═" * 60)

            # Build the command
            cmd_parts = ["chisel", "python", script_path]

            # Add flags
            if app_name:
                cmd_parts.append(f'--app-name "{app_name}"')

            if upload_dir != ".":
                cmd_parts.append(f'--upload-dir "{upload_dir}"')

            if requirements_file != "requirements.txt":
                cmd_parts.append(f'--requirements "{requirements_file}"')

            if gpu != "A100-80GB:1":
                # Convert GPU type back to number
                gpu_number = {v: k for k, v in self.gpu_map.items()}[gpu]
                cmd_parts.append(f"--gpu {gpu_number}")

            command = " ".join(cmd_parts)
            print(f"$ {command}")
            print("═" * 60)
            print()

    def parse_command_line_args(self, args: List[str]) -> Optional[Dict[str, Any]]:
        """Parse command line arguments and return configuration."""
        parser = argparse.ArgumentParser(
            description="Chisel CLI - GPU Job Submission Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  chisel python script.py --app-name my-job --gpu 4
  chisel python train.py --upload-dir ./project --requirements dev.txt
  chisel python inference.py --app-name inference-job --gpu 1
  chisel python script.py --preview  # Preview upload contents without submitting
            """,
        )

        # Script and arguments (positional)
        parser.add_argument(
            "command", nargs="+", help="Python command to run (e.g., python script.py arg1 arg2)"
        )

        # Job configuration
        parser.add_argument("--app-name", "-a", help="App name for job tracking")
        parser.add_argument(
            "--upload-dir",
            "-d",
            default=".",
            help="Directory to upload (default: current directory)",
        )
        parser.add_argument(
            "--requirements",
            "-r",
            default="requirements.txt",
            help="Requirements file (default: requirements.txt)",
        )
        parser.add_argument(
            "--gpu",
            "-g",
            choices=["1", "2", "4", "8"],
            default="1",
            help="GPU configuration: 1, 2, 4, or 8 GPUs (default: 1)",
        )
        parser.add_argument(
            "--interactive",
            "-i",
            action="store_true",
            help="Force interactive mode even when flags are provided",
        )
        parser.add_argument(
            "--preview",
            "-p",
            action="store_true",
            help="Preview upload contents without submitting job",
        )

        try:
            parsed_args = parser.parse_args(args)

            # Extract python command parts
            if parsed_args.command[0] != "python":
                print("❌ Chisel currently only supports 'python' commands!")
                print("Usage: chisel python <script.py> [args...]")
                return None

            script_path = parsed_args.command[1]
            script_args = parsed_args.command[2:] if len(parsed_args.command) > 2 else []

            return {
                "script_path": script_path,
                "script_args": script_args,
                "app_name": parsed_args.app_name,
                "upload_dir": parsed_args.upload_dir,
                "requirements_file": parsed_args.requirements,
                "gpu": self.gpu_map[parsed_args.gpu],
                "interactive": parsed_args.interactive,
                "preview": parsed_args.preview,
            }
        except SystemExit:
            return None

    def submit_job(
        self,
        app_name: str,
        upload_dir: str,
        script_path: str,
        gpu: str,
        requirements_file: str,
        script_args: List[str],
        api_key: str,
    ) -> Dict[str, Any]:
        """Submit job to backend."""
        backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL

        upload_dir = Path(upload_dir)

        # Process directory for cached files
        processed_dir = upload_dir
        cached_files_info = []

        # Check for large files that could be cached
        large_files = scan_directory_for_large_files(upload_dir)
        if large_files:
            if RICH_AVAILABLE:
                self.console.print(
                    f"[yellow]Found {len(large_files)} large file(s) that could be cached[/yellow]"
                )
            else:
                print(f"Found {len(large_files)} large file(s) that could be cached")

            try:
                # Create a temporary directory for processing
                temp_processing_dir = Path(tempfile.mkdtemp())

                # Process the directory to handle cached files
                print(f"Processing directory: {upload_dir}")
                processed_dir, cached_files_info = process_directory_for_cached_files(
                    upload_dir, api_key, temp_processing_dir
                )

                if cached_files_info:
                    if RICH_AVAILABLE:
                        self.console.print(
                            f"[green]Successfully processed {len(cached_files_info)} cached file(s)[/green]"
                        )
                    else:
                        print(f"Successfully processed {len(cached_files_info)} cached file(s)")

            except Exception as e:
                if RICH_AVAILABLE:
                    self.console.print(
                        f"[yellow]Warning: Could not process cached files: {e}[/yellow]"
                    )
                    self.console.print("[yellow]Continuing with original directory...[/yellow]")
                else:
                    print(f"Warning: Could not process cached files: {e}")
                    print("Continuing with original directory...")

                # Fallback to original directory if processing fails
                processed_dir = upload_dir
                cached_files_info = []

        # Load ignore patterns for tar filtering
        gitignore_patterns, chiselignore_patterns = load_ignore_patterns(Path(processed_dir))

        # Create a wrapper function for tar filter with ignore patterns
        def tar_filter_with_ignore_patterns(tarinfo):
            return tar_filter(tarinfo, gitignore_patterns, chiselignore_patterns)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = tmp_file.name

        try:
            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task(
                        f"Creating archive from {processed_dir.name}", total=None
                    )

                    try:
                        with tarfile.open(tar_path, "w:gz") as tar:
                            tar.add(
                                processed_dir, arcname=".", filter=tar_filter_with_ignore_patterns
                            )

                        tar_size = Path(tar_path).stat().st_size
                        size_mb = tar_size / (1024 * 1024)
                        size_kb = tar_size / 1024

                        # Show KB for small archives, MB for larger ones
                        if size_mb < 1.0:
                            progress.update(task, description=f"Archive created: {size_kb:.2f} KB")
                        else:
                            progress.update(task, description=f"Archive created: {size_mb:.3f} MB")
                    except Exception as e:
                        progress.update(task, description="Archive creation failed")
                        raise e
            else:
                spinner = SimpleSpinner(f"Creating archive from {processed_dir.name}")
                spinner.start()

                try:
                    with tarfile.open(tar_path, "w:gz") as tar:
                        tar.add(processed_dir, arcname=".", filter=tar_filter_with_ignore_patterns)

                    tar_size = Path(tar_path).stat().st_size
                    size_mb = tar_size / (1024 * 1024)
                    size_kb = tar_size / 1024

                    # Show KB for small archives, MB for larger ones
                    if size_mb < 1.0:
                        spinner.stop(f"Archive created: {size_kb:.2f} KB")
                    else:
                        spinner.stop(f"Archive created: {size_mb:.3f} MB")
                except Exception as e:
                    spinner.stop("Archive creation failed")
                    raise e

            headers = {"Authorization": f"Bearer {api_key}"}
            files = {"file": ("src.tar.gz", open(tar_path, "rb"), "application/gzip")}
            data = {
                "script_path": script_path,
                "app_name": app_name,
                "pip_packages": ",".join(MINIMUM_PACKAGES),
                "gpu": gpu,
                "script_args": " ".join(script_args) if script_args else "",
                "cached_files": json.dumps(cached_files_info) if cached_files_info else "",
                "requirements_file": requirements_file,
            }

            endpoint = f"{backend_url.rstrip('/')}/api/v1/submit-cachy-job-new"

            if RICH_AVAILABLE:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=self.console,
                ) as progress:
                    task = progress.add_task("Uploading work to backend and running", total=None)

                    try:
                        response = requests.post(
                            endpoint, data=data, files=files, headers=headers, timeout=12 * 60 * 60
                        )
                        response.raise_for_status()

                        result = response.json()
                        job_id = result.get("job_id")
                        message = result.get("message", "Job submitted")
                        visit_url = result.get("visit_url", f"/jobs/{job_id}")

                        progress.update(
                            task, description="Work uploaded successfully! Job submitted"
                        )

                        # Create a success panel
                        success_panel = Panel(
                            f"🔗 Job ID: {job_id}\n🌐 Visit: {visit_url}\n📊 Job is running in the background on cloud GPUs",
                            title="✅ Job Submitted Successfully",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                        self.console.print(success_panel)

                    except Exception as e:
                        progress.update(task, description="Upload failed")
                        raise e
            else:
                upload_spinner = SimpleSpinner("Uploading work to backend and running.")
                upload_spinner.start()

                try:
                    response = requests.post(
                        endpoint, data=data, files=files, headers=headers, timeout=12 * 60 * 60
                    )
                    response.raise_for_status()

                    result = response.json()
                    job_id = result.get("job_id")
                    message = result.get("message", "Job submitted")
                    visit_url = result.get("visit_url", f"/jobs/{job_id}")

                    upload_spinner.stop("Work uploaded successfully! Job submitted")

                    print(f"🔗 Job ID: {job_id}")
                    print(f"🌐 Visit: {visit_url}")
                    print("📊 Job is running in the background on cloud GPUs")

                except Exception as e:
                    upload_spinner.stop("Upload failed")
                    raise e

            return {
                "job_id": job_id,
                "exit_code": 0,
                "logs": [f"{message} (Job ID: {job_id})"],
                "status": "submitted",
                "visit_url": visit_url,
            }
        except Exception as e:
            print(f"🔍 [submit_job] Error creating tar archive: {e}")
            raise
        finally:
            if os.path.exists(tar_path):
                os.unlink(tar_path)

    def run_chisel_command(self, command: List[str]) -> int:
        """Run the chisel command with improved interface."""
        if len(command) < 2:
            if RICH_AVAILABLE:
                self.console.print("❌ No command provided!", style="red")
                self.console.print("\n[bold]Usage:[/bold]")
                self.console.print("  chisel python <script.py> [args...]")
                self.console.print("  chisel python <script.py> --app-name my-job --gpu 4")
                self.console.print("  chisel python <script.py> --interactive")
            else:
                print("❌ No command provided!")
                print("Usage: chisel python <script.py> [args...]")
                print("       chisel python <script.py> --app-name my-job --gpu 4")
                print("       chisel python <script.py> --interactive")
            return 1

        # Try to parse command line arguments first
        parsed_config = self.parse_command_line_args(command)
        if parsed_config is None:
            return 1

        # Handle preview-only mode
        if parsed_config["preview"]:
            upload_dir = Path(parsed_config["upload_dir"]).resolve()
            if not upload_dir.exists():
                if RICH_AVAILABLE:
                    self.console.print(
                        f"❌ Upload directory '{upload_dir}' does not exist", style="red"
                    )
                else:
                    print(f"❌ Upload directory '{upload_dir}' does not exist")
                return 1

            # Show preview and exit
            preview_data = preview_upload_directory(upload_dir, self.console)
            display_upload_preview(preview_data, str(upload_dir), self.console)
            return 0

        # If no app_name provided via flags, we need interactive mode
        if not parsed_config["app_name"] or parsed_config["interactive"]:
            # Get interactive inputs
            interactive_inputs = self.get_user_inputs_interactive(parsed_config["script_path"])

            # Merge with command line args (command line takes precedence)
            final_config = {**interactive_inputs}
            if parsed_config["app_name"]:
                final_config["app_name"] = parsed_config["app_name"]
            if parsed_config["gpu"]:
                final_config["gpu"] = parsed_config["gpu"]
            if parsed_config["upload_dir"]:
                final_config["upload_dir"] = parsed_config["upload_dir"]
            if parsed_config["requirements_file"]:
                final_config["requirements_file"] = parsed_config["requirements_file"]
        else:
            # Use command line configuration
            final_config = {
                "app_name": parsed_config["app_name"],
                "upload_dir": parsed_config["upload_dir"],
                "requirements_file": parsed_config["requirements_file"],
                "gpu": parsed_config["gpu"],
            }

        # Get script information
        script_path = parsed_config["script_path"]
        script_args = parsed_config["script_args"]

        # Get script absolute path
        script_abs_path = Path(script_path).resolve()

        # Validate upload directory contains the script
        upload_dir = Path(final_config["upload_dir"]).resolve()
        try:
            script_relative = script_abs_path.relative_to(upload_dir)
        except ValueError:
            if RICH_AVAILABLE:
                self.console.print(
                    f"❌ Script {script_abs_path} is not inside upload_dir {upload_dir}",
                    style="red",
                )
            else:
                print(f"❌ Script {script_abs_path} is not inside upload_dir {upload_dir}")
            return 1

        script_name = str(script_relative)
        args_display = f" {' '.join(script_args)}" if script_args else ""

        # Show submission summary and get confirmation BEFORE authentication
        preview_data = preview_upload_directory(upload_dir, self.console)
        should_submit = display_submission_summary(
            preview_data,
            final_config["upload_dir"],
            final_config["app_name"],
            final_config["gpu"],
            self.console,
        )

        if not should_submit:
            if RICH_AVAILABLE:
                self.console.print("[yellow]❌ Job submission cancelled by user.[/yellow]")
            else:
                print("❌ Job submission cancelled by user.")
            return 0

        # NOW authenticate after user confirms
        if RICH_AVAILABLE:
            self.console.print("\n🔑 Checking authentication...", style="yellow")
        else:
            print("\n🔑 Checking authentication...")

        backend_url = os.environ.get(CHISEL_BACKEND_URL_ENV_KEY) or CHISEL_BACKEND_URL
        api_key = _auth_service.authenticate(backend_url)

        if not api_key:
            if RICH_AVAILABLE:
                self.console.print("❌ Authentication failed. Please try again.", style="red")
            else:
                print("❌ Authentication failed. Please try again.")
            return 1

        if RICH_AVAILABLE:
            self.console.print("✅ Authentication successful!", style="green")
        else:
            print("✅ Authentication successful!")

        if RICH_AVAILABLE:
            self.console.print(f"\n📦 Submitting job: [bold]{script_name}{args_display}[/bold]")
        else:
            print(f"\n📦 Submitting job: {script_name}{args_display}")

        try:
            result = self.submit_job(
                app_name=final_config["app_name"],
                upload_dir=final_config["upload_dir"],
                script_path=script_name,
                gpu=final_config["gpu"],
                script_args=script_args,
                requirements_file=final_config["requirements_file"],
                api_key=api_key,
            )

            return result["exit_code"]
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"❌ Job submission failed: {e}", style="red")
            else:
                print(f"❌ Job submission failed: {e}")
            return 1


def main():
    """Main CLI entry point."""
    cli = ChiselCLI()

    if len(sys.argv) < 2:
        if cli.console:
            cli.console.print("Chisel CLI is installed and working!", style="bold green")
            cli.console.print("\n[bold]Usage:[/bold]")
            cli.console.print("  chisel python <script.py> [args...]")
            cli.console.print("  chisel python <script.py> --app-name my-job --gpu 4")
            cli.console.print("  chisel python <script.py> --interactive")
            cli.console.print("  chisel python <script.py> --preview")
            cli.console.print("  chisel --logout")
            cli.console.print("  chisel --version")
            cli.console.print("\n[bold]Examples:[/bold]")
            cli.console.print("  chisel python my_script.py")
            cli.console.print("  chisel python train.py --app-name training-job --gpu 4")
            cli.console.print(
                "  chisel python inference.py --upload-dir ./project --requirements dev.txt"
            )
            cli.console.print("  chisel python script.py --preview  # Preview what gets uploaded")
            cli.console.print(
                "\n💡 Tip: Interactive mode shows the equivalent command-line for copy/paste!"
            )
        else:
            print("Chisel CLI is installed and working!")
            print()
            print("Usage:")
            print("  chisel python <script.py> [args...]")
            print("  chisel python <script.py> --app-name my-job --gpu 4")
            print("  chisel python <script.py> --interactive")
            print("  chisel python <script.py> --preview")
            print("  chisel --logout")
            print("  chisel --version")
            print()
            print("Examples:")
            print("  chisel python my_script.py")
            print("  chisel python train.py --app-name training-job --gpu 4")
            print("  chisel python inference.py --upload-dir ./project --requirements dev.txt")
            print("  chisel python script.py --preview  # Preview what gets uploaded")
            print()
            print("💡 Tip: Interactive mode shows the equivalent command-line for copy/paste!")
        return 0

    # Handle version flag
    if sys.argv[1] in ["--version", "-v", "version"]:
        from . import __version__

        print(f"Chisel CLI v{__version__}")
        return 0

    # Handle logout flag
    if sys.argv[1] == "--logout":
        if _auth_service.is_authenticated():
            _auth_service.clear()
            print("✅ Successfully logged out from Chisel CLI")
        else:
            print("ℹ️  No active authentication found")
        return 0

    # Run chisel command
    command = sys.argv[1:]
    return cli.run_chisel_command(command)
