import os
import sys
import subprocess
import tempfile
import tarfile
import requests
import argparse
import json
import webbrowser
import shlex
import re
import select
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
from .auth import _auth_service
from .constants import (
    KANDC_BACKEND_URL,
    KANDC_BACKEND_URL_ENV_KEY,
    MINIMUM_PACKAGES,
    KANDC_BACKEND_RUN_ENV_KEY,
    KANDC_BACKEND_APP_NAME_ENV_KEY,
    KANDC_JOB_ID_ENV_KEY,
    KANDC_TRACE_BASE_DIR_ENV_KEY,
)
from .spinner import SimpleSpinner

MAX_UPLOAD_FILE_SIZE = 5 * 1024 * 1024 * 1024

# Rich imports for beautiful CLI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.prompt import Prompt
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box

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
    Load patterns from .gitignore and .kandcignore files if they exist.

    Returns:
        tuple: (gitignore_patterns, kandcignore_patterns)
    """
    gitignore_patterns = set()
    kandcignore_patterns = set()

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

    # Load .kandcignore patterns
    kandcignore_path = directory / ".kandcignore"
    if kandcignore_path.exists():
        try:
            with open(kandcignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        # Remove leading slash if present
                        if line.startswith("/"):
                            line = line[1:]
                        # Add pattern to set
                        kandcignore_patterns.add(line)
        except (IOError, UnicodeDecodeError):
            # If we can't read .kandcignore, just continue
            pass

    return gitignore_patterns, kandcignore_patterns


def should_exclude(path, gitignore_patterns=None, kandcignore_patterns=None):
    """
    Check if a path should be excluded from upload.

    Args:
        path: Path to check (relative to upload directory)
        gitignore_patterns: Set of gitignore patterns (optional)
        kandcignore_patterns: Set of kandcignore patterns (optional)

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

    # Check kandcignore patterns first (takes precedence)
    if matches_patterns(kandcignore_patterns):
        return True

    # Check gitignore patterns
    if matches_patterns(gitignore_patterns):
        return True

    return False


def tar_filter(tarinfo, gitignore_patterns=None, kandcignore_patterns=None):
    if should_exclude(tarinfo.name, gitignore_patterns, kandcignore_patterns):
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
    gitignore_patterns, kandcignore_patterns = load_ignore_patterns(upload_dir)

    # Walk through the directory
    for root, dirs, files in os.walk(upload_dir):
        # Filter out excluded directories to avoid walking them
        dirs[:] = [
            d
            for d in dirs
            if not should_exclude(os.path.join(root, d), gitignore_patterns, kandcignore_patterns)
        ]

        for file in files:
            file_path = Path(root) / file
            relative_path = file_path.relative_to(upload_dir)

            if should_exclude(str(relative_path), gitignore_patterns, kandcignore_patterns):
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
        "kandcignore_patterns": kandcignore_patterns,
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


def display_upload_preview(
    preview_data: Dict[str, Any], upload_dir: str, console=None, show_patterns: bool = True
):
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
            console.print(f"\n[bold red]🚫 Excluded Files (showing first 10):[/bold red]")
            for i, excluded in enumerate(preview_data["excluded_files"][:10]):
                console.print(f"  • {excluded}")

            if len(preview_data["excluded_files"]) > 10:
                console.print(f"  ... and {len(preview_data['excluded_files']) - 10} more excluded")

        if show_patterns:
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

            # Show kandcignore patterns if any were found
            kandcignore_patterns = preview_data.get("kandcignore_patterns", set())
            if kandcignore_patterns:
                console.print(f"\n[bold dim]📋 .kandcignore Patterns:[/bold dim]")
                for pattern in sorted(kandcignore_patterns):
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
            print(f"\n🚫 Excluded Files (showing first 10):")
            for i, excluded in enumerate(preview_data["excluded_files"][:10]):
                print(f"  • {excluded}")

            if len(preview_data["excluded_files"]) > 10:
                print(f"  ... and {len(preview_data['excluded_files']) - 10} more excluded")

        if show_patterns:
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

            # Show kandcignore patterns if any were found
            kandcignore_patterns = preview_data.get("kandcignore_patterns", set())
            if kandcignore_patterns:
                print(f"\n📋 .kandcignore Patterns:")
                for pattern in sorted(kandcignore_patterns):
                    print(f"  • {pattern}")


def validate_app_name(app_name: str) -> tuple[bool, str]:
    """
    Validate app name according to Modal's requirements.

    Returns:
        tuple: (is_valid, error_message)
    """
    if not app_name:
        return False, "App name cannot be empty"

    if len(app_name) >= 64:
        return False, "App name must be shorter than 64 characters"

    # Check for valid characters: alphanumeric, dashes, periods, underscores
    if not re.match(r"^[a-zA-Z0-9._-]+$", app_name):
        return (
            False,
            "App name may only contain alphanumeric characters, dashes, periods, and underscores",
        )

    # Check for spaces (common mistake)
    if " " in app_name:
        return False, "App name cannot contain spaces (use dashes or underscores instead)"

    return True, ""


def suggest_valid_app_name(invalid_name: str) -> str:
    """Suggest a valid app name based on an invalid one."""
    # Replace spaces with dashes
    suggested = invalid_name.replace(" ", "-")

    # Remove invalid characters
    suggested = re.sub(r"[^a-zA-Z0-9._-]", "", suggested)

    # Truncate if too long
    if len(suggested) >= 64:
        suggested = suggested[:63]

    # Remove trailing special characters
    suggested = suggested.rstrip("._-")

    return suggested or "my-app"


class KandcCLI:
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.gpu_options = [
            # GPU types in alphabetical order
            ("A100:1", "A100:1", "A100-40GB Single GPU - Development, inference"),
            ("A100:2", "A100:2", "A100-40GB 2x GPUs - Medium training"),
            ("A100:4", "A100:4", "A100-40GB 4x GPUs - Large models"),
            ("A100:8", "A100:8", "A100-40GB 8x GPUs - Massive models"),
            ("A100-80GB:1", "A100-80GB:1", "A100-80GB Single GPU - Development, inference"),
            ("A100-80GB:2", "A100-80GB:2", "A100-80GB 2x GPUs - Medium training"),
            ("A100-80GB:4", "A100-80GB:4", "A100-80GB 4x GPUs - Large models"),
            ("A100-80GB:8", "A100-80GB:8", "A100-80GB 8x GPUs - Massive models"),
            ("H100:1", "H100:1", "H100 Single GPU - Latest architecture"),
            ("H100:2", "H100:2", "H100 2x GPUs - Advanced training"),
            ("H100:4", "H100:4", "H100 4x GPUs - High-performance training"),
            ("H100:8", "H100:8", "H100 8x GPUs - Maximum performance"),
            ("L4:1", "L4:1", "L4 Single GPU - Cost-effective inference"),
            ("L4:2", "L4:2", "L4 2x GPUs - Efficient training"),
            ("L4:4", "L4:4", "L4 4x GPUs - Balanced performance"),
            ("L4:8", "L4:8", "L4 8x GPUs - High throughput"),
        ]
        self.gpu_map = {option: gpu_type for option, gpu_type, _ in self.gpu_options}

    def print_header(self):
        """Print the CLI header with styling."""
        if RICH_AVAILABLE:
            title = Text("🚀 Keys & Caches CLI", style="bold blue")
            subtitle = Text("GPU Job Submission Tool", style="dim")

            header = Panel(
                title, subtitle=subtitle, border_style="blue", box=box.ROUNDED, padding=(1, 2)
            )
            self.console.print(header)
        else:
            print("╔══════════════════════════════════════════════════════════════╗")
            print("║                🚀 Keys & Caches CLI                         ║")
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

    def get_valid_app_name(self, prompt: str, default: str = "") -> str:
        """Get a valid app name from user with validation."""
        while True:
            if RICH_AVAILABLE:
                if default:
                    app_name = Prompt.ask(f"{prompt}", default=default, console=self.console)
                else:
                    app_name = Prompt.ask(f"{prompt}", console=self.console)
            else:
                if default:
                    app_name = input(f"{prompt} (default: {default}): ").strip()
                    app_name = app_name if app_name else default
                else:
                    app_name = input(f"{prompt}: ").strip()

            # Validate the app name
            is_valid, error_message = validate_app_name(app_name)

            if is_valid:
                return app_name

            # Show error and suggestion
            if RICH_AVAILABLE:
                self.console.print(f"❌ Invalid app name: {error_message}", style="red")
                suggested = suggest_valid_app_name(app_name)
                if suggested != app_name:
                    self.console.print(f"💡 Suggested: '{suggested}'", style="yellow")
                self.console.print(
                    "💡 Valid names: alphanumeric, dashes, periods, underscores only (no spaces)",
                    style="dim",
                )
            else:
                print(f"❌ Invalid app name: {error_message}")
                suggested = suggest_valid_app_name(app_name)
                if suggested != app_name:
                    print(f"💡 Suggested: '{suggested}'")
                print("💡 Valid names: alphanumeric, dashes, periods, underscores only (no spaces)")

            # Ask if they want to use the suggestion
            if suggested != app_name:
                if RICH_AVAILABLE:
                    use_suggestion = Prompt.ask(
                        f"Use suggested name '{suggested}'? (y/n)",
                        choices=["y", "n"],
                        default="y",
                        console=self.console,
                    )
                else:
                    use_suggestion = (
                        input(f"Use suggested name '{suggested}'? (Y/n): ").strip().lower()
                    )
                    use_suggestion = (
                        "y" if not use_suggestion or use_suggestion in ["y", "yes"] else "n"
                    )

                if use_suggestion == "y":
                    return suggested

    def get_yes_no_input(self, prompt: str, default: bool = True, help_text: str = None) -> bool:
        """Get yes/no input from user with default value."""
        if RICH_AVAILABLE:
            default_str = "Y/n" if default else "y/N"
            full_prompt = f"{prompt} ({default_str})"
            if help_text:
                self.console.print(f"💡 {help_text}", style="dim")

            while True:
                response = Prompt.ask(full_prompt, console=self.console, default="").strip().lower()
                if not response:
                    return default
                elif response in ["y", "yes", "true", "1"]:
                    return True
                elif response in ["n", "no", "false", "0"]:
                    return False
                else:
                    self.console.print("❌ Please enter 'y' for yes or 'n' for no", style="red")
        else:
            default_str = "Y/n" if default else "y/N"
            full_prompt = f"{prompt} ({default_str}): "
            if help_text:
                print(f"💡 {help_text}")

            while True:
                response = input(full_prompt).strip().lower()
                if not response:
                    return default
                elif response in ["y", "yes", "true", "1"]:
                    return True
                elif response in ["n", "no", "false", "0"]:
                    return False
                else:
                    print("❌ Please enter 'y' for yes or 'n' for no")

    def select_gpu(self, default_gpu: str = None) -> str:
        """Interactive GPU selection with navigation.

        Args:
            default_gpu: Pre-filled GPU value to use as default
        """
        # Determine the default choice number from GPU string
        if default_gpu:
            default_choice = None
            for choice, gpu_type in self.gpu_map.items():
                if gpu_type == default_gpu:
                    default_choice = choice
                    break
        else:
            default_choice = "A100-80GB:1"
            default_gpu = self.gpu_map["A100-80GB:1"]

        if RICH_AVAILABLE:
            # Create a numbered table for easy selection
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

            # Create numbered mapping
            numbered_options = {}
            default_number = None

            for i, (option, gpu_type, description) in enumerate(self.gpu_options, 1):
                numbered_options[str(i)] = gpu_type

                # Check if this is the default
                if default_choice and option == default_choice:
                    default_number = str(i)
                    table.add_row(
                        f"[bold green]{i}*[/bold green]",
                        f"[bold green]{gpu_type}[/bold green]",
                        f"[bold green]{description} (default)[/bold green]",
                    )
                else:
                    table.add_row(str(i), gpu_type, description)

            self.console.print(table)
            self.console.print()

            # Use numbered selection
            while True:
                choice = Prompt.ask(
                    f"Select GPU configuration (1-{len(self.gpu_options)})",
                    choices=[str(i) for i in range(1, len(self.gpu_options) + 1)],
                    default=default_number or "5",  # A100-80GB:1 is option 5
                    console=self.console,
                    show_choices=False,
                )

                if choice in numbered_options:
                    selected_gpu = numbered_options[choice]
                    self.console.print(f"✅ Selected: [bold green]{selected_gpu}[/bold green]")
                    return selected_gpu
                else:
                    self.console.print(
                        f"❌ Invalid choice. Please select 1-{len(self.gpu_options)}.", style="red"
                    )
        else:
            print("Available GPU Configurations:")
            print("-" * 80)
            print(f"{'Option':<8} {'GPU Type':<15} {'Description'}")
            print("-" * 80)

            # Create numbered mapping
            numbered_options = {}
            default_number = None

            for i, (option, gpu_type, description) in enumerate(self.gpu_options, 1):
                numbered_options[str(i)] = gpu_type

                # Check if this is the default
                if default_choice and option == default_choice:
                    default_number = str(i)
                    print(f"{i}*{'':<7} {gpu_type:<15} {description} (default)")
                else:
                    print(f"{i}{'':<8} {gpu_type:<15} {description}")

            print("-" * 80)
            print()

            while True:
                prompt_text = f"Select GPU configuration (1-{len(self.gpu_options)}, default: {default_number or '5'}): "
                choice = input(prompt_text).strip()
                if not choice:
                    choice = default_number or "5"

                if choice in numbered_options:
                    selected_gpu = numbered_options[choice]
                    print(f"✅ Selected: {selected_gpu}")
                    return selected_gpu
                else:
                    print(f"❌ Invalid choice. Please select 1-{len(self.gpu_options)}.")

    def get_user_inputs_interactive(self, script_path: str = "<script.py>") -> Dict[str, Any]:
        """Interactive questionnaire to get job submission parameters.

        Args:
            script_path: Path to the script being run
        """
        self.print_header()

        # Define total steps for progress tracking
        total_steps = 6

        # Step 1: App name
        self.print_section_header("Job Configuration", step=1, total_steps=total_steps)
        app_name = self.get_valid_app_name("📝 App name (for job tracking)")

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
        gpu = self.select_gpu(default_gpu=self.gpu_map["A100-80GB:1"])

        # Step 5: Code snapshot option
        self.print_section_header("Code Snapshot", step=5, total_steps=total_steps)
        include_code_snapshot = self.get_yes_no_input(
            "📸 Include code snapshot for debugging?",
            default=True,
            help_text="Uploads a snapshot of your code for viewing in the web interface",
        )

        # Show equivalent command for copy/paste
        self.show_equivalent_command(
            app_name, upload_dir, requirements_file, gpu, script_path, include_code_snapshot
        )

        return {
            "app_name": app_name,
            "upload_dir": upload_dir,
            "requirements_file": requirements_file,
            "gpu": gpu,
            "include_code_snapshot": include_code_snapshot,
        }

    def show_equivalent_command(
        self,
        app_name: str,
        upload_dir: str,
        requirements_file: str,
        gpu: str,
        script_path: str,
        include_code_snapshot: bool = True,
    ):
        """Show the equivalent command-line command for copy/paste."""
        if RICH_AVAILABLE:
            # Build the command using separator format
            kandc_parts = ["kandc"]

            # Add kandc flags first
            if app_name:
                kandc_parts.append(f'--app-name "{app_name}"')

            if upload_dir != ".":
                kandc_parts.append(f'--upload-dir "{upload_dir}"')

            if requirements_file != "requirements.txt":
                kandc_parts.append(f'--requirements "{requirements_file}"')

            if gpu != "A100-80GB:1":
                # Convert GPU type back to option key
                gpu_option = {v: k for k, v in self.gpu_map.items()}[gpu]
                kandc_parts.append(f"--gpu {gpu_option}")

            if not include_code_snapshot:
                kandc_parts.append("--no-code-snapshot")

            # Add separator and python command
            kandc_parts.append("--")
            kandc_parts.append("python")
            kandc_parts.append(script_path)

            command = " ".join(kandc_parts)

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

            # Build the command using separator format
            kandc_parts = ["kandc"]

            # Add kandc flags first
            if app_name:
                kandc_parts.append(f'--app-name "{app_name}"')

            if upload_dir != ".":
                kandc_parts.append(f'--upload-dir "{upload_dir}"')

            if requirements_file != "requirements.txt":
                kandc_parts.append(f'--requirements "{requirements_file}"')

            if gpu != "A100-80GB:1":
                # Convert GPU type back to option key
                gpu_option = {v: k for k, v in self.gpu_map.items()}[gpu]
                kandc_parts.append(f"--gpu {gpu_option}")

            if not include_code_snapshot:
                kandc_parts.append("--no-code-snapshot")

            # Add separator and python command
            kandc_parts.append("--")
            kandc_parts.append("python")
            kandc_parts.append(script_path)

            command = " ".join(kandc_parts)
            print(f"$ {command}")
            print("═" * 60)
            print()

    def parse_command_line_args(self, args: List[str]) -> Optional[Dict[str, Any]]:
        """Parse command line arguments and return configuration.

        Supports two formats:
        1. kandc --kandc-flags -- python script.py --script-args (separator format)
        2. kandc python script.py --script-args (interactive format)
        """
        # Check if we have the -- separator format
        if "--" in args:
            return self._parse_with_separator(args)
        else:
            return self._parse_interactive_format(args)

    def _parse_with_separator(self, args: List[str]) -> Optional[Dict[str, Any]]:
        """Parse arguments with -- separator: kandc --kandc-flags -- python script.py --script-args"""
        try:
            separator_index = args.index("--")
            kandc_args = args[:separator_index]
            command_args = args[separator_index + 1 :]

            # Parse kandc flags
            parser = self._create_kandc_parser()
            # Add dummy command for kandc-only args
            parser.add_argument("dummy", nargs="*", help=argparse.SUPPRESS)

            parsed_kandc = parser.parse_args(kandc_args + ["dummy"])

            # Validate command format
            if len(command_args) < 2 or command_args[0] != "python":
                print("❌ After --, expected: python <script.py> [script-args...]")
                print(
                    "Usage: kandc --app-name my-job --gpu A100-80GB:2 -- python script.py --model-size large"
                )
                return None

            script_path = command_args[1]
            script_args = command_args[2:] if len(command_args) > 2 else []

            return {
                "script_path": script_path,
                "script_args": script_args,
                "app_name": parsed_kandc.app_name,
                "upload_dir": parsed_kandc.upload_dir,
                "requirements_file": parsed_kandc.requirements,
                "gpu": self.gpu_map[parsed_kandc.gpu],
                "interactive": parsed_kandc.interactive,
                "preview": parsed_kandc.preview,
                "no_code_snapshot": parsed_kandc.no_code_snapshot,
            }
        except (ValueError, SystemExit) as e:
            if isinstance(e, ValueError):
                print("❌ Error parsing arguments with -- separator")
            return None

    def _parse_interactive_format(self, args: List[str]) -> Optional[Dict[str, Any]]:
        """Parse interactive format: kandc python script.py --script-args

        This format does NOT accept kandc flags - all configuration must be done interactively.
        If kandc flags are provided without the -- separator, it's an error.
        """
        # Try to find where 'python' starts
        python_index = None
        for i, arg in enumerate(args):
            if arg == "python":
                python_index = i
                break

        if python_index is None:
            print("❌ Error: 'python' command not found")
            print("💡 Use: kandc python script.py --script-args")
            print("💡 Or: kandc --kandc-flags -- python script.py --script-args")
            return None

        # Check if there are any potential kandc flags before 'python'
        potential_kandc_args = args[:python_index]
        if potential_kandc_args:
            print("❌ Error: Keys & Caches flags are not allowed in interactive format")
            print(f"   Found: {' '.join(potential_kandc_args)}")
            print("💡 Use the separator format instead:")
            print(
                f"   kandc {' '.join(potential_kandc_args)} -- python {' '.join(args[python_index + 1 :])}"
            )
            return None

        # Extract python command
        python_command = args[python_index:]

        # Validate python command format
        if len(python_command) < 2:
            print("❌ Error: No script specified after 'python'")
            return None

        script_path = python_command[1]
        script_args = python_command[2:] if len(python_command) > 2 else []

        # Return configuration for pure interactive format
        return {
            "script_path": script_path,
            "script_args": script_args,
            "app_name": None,
            "upload_dir": ".",
            "requirements_file": "requirements.txt",
            "gpu": self.gpu_map["A100-80GB:1"],
            "interactive": True,
            "preview": False,
            "has_prefilled_values": False,
        }

    def _create_kandc_parser(self):
        """Create the argument parser for kandc-specific flags."""
        parser = argparse.ArgumentParser(
            description="Keys & Caches CLI - GPU Job Submission Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Separator format (kandc flags first, then -- separator):
  kandc --app-name my-job --gpu A100-80GB:4 -- python script.py --model-size large
  
  # Interactive mode (script args only, prompts for kandc config):
  kandc python script.py --model-size large --epochs 10
            """,
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
            choices=[option for option, _, _ in self.gpu_options],
            default="A100-80GB:1",
            help="GPU configuration: A100:1-8, A100-80GB:1-8, H100:1-8, L4:1-8 (default: A100-80GB:1)",
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
        parser.add_argument(
            "--no-code-snapshot",
            action="store_true",
            help="Skip uploading code snapshot for debugging (default: include code snapshot)",
        )

        return parser

    def create_code_snapshot(
        self, upload_dir: Path, use_kandcignore_only: bool = False
    ) -> Optional[str]:
        """
        Create a code snapshot archive of the upload directory.
        Returns the path to the created archive, or None if creation failed.
        """
        try:
            # Create temporary file for code snapshot
            snapshot_fd, snapshot_path = tempfile.mkstemp(suffix="-code-snapshot.tar.gz")
            os.close(snapshot_fd)

            files_included = 0
            total_size = 0

            # Load ignore patterns for this upload_dir
            gitignore_patterns, kandcignore_patterns = load_ignore_patterns(upload_dir)

            def _matches_patterns(path_obj: Path, patterns: set[str]) -> bool:
                if not patterns:
                    return False
                path_str = str(path_obj)
                path_parts = path_obj.parts
                for pattern in patterns:
                    if path_str == pattern or path_str.startswith(pattern + "/"):
                        return True
                    if pattern in path_parts:
                        return True
                    if "*" in pattern:
                        import fnmatch

                        if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(
                            path_obj.name, pattern
                        ):
                            return True
                return False

            with tarfile.open(snapshot_path, "w:gz") as tar:
                for root, dirs, files in os.walk(upload_dir):
                    root_path = Path(root)

                    # Filter directories to avoid traversing excluded ones
                    if use_kandcignore_only:
                        dirs[:] = [
                            d
                            for d in dirs
                            if not _matches_patterns(
                                (root_path / d).relative_to(upload_dir), kandcignore_patterns
                            )
                        ]
                    else:
                        dirs[:] = [
                            d
                            for d in dirs
                            if not should_exclude(
                                str((root_path / d).relative_to(upload_dir)),
                                gitignore_patterns,
                                kandcignore_patterns,
                            )
                        ]

                    for file in files:
                        file_path = root_path / file

                        if (
                            not use_kandcignore_only
                            and not should_exclude(
                                str(file_path.relative_to(upload_dir)),
                                gitignore_patterns,
                                kandcignore_patterns,
                            )
                        ) or (
                            use_kandcignore_only
                            and not _matches_patterns(
                                file_path.relative_to(upload_dir), kandcignore_patterns
                            )
                        ):
                            try:
                                # Add file to archive with relative path
                                relative_path = file_path.relative_to(upload_dir)
                                tar.add(file_path, arcname=str(relative_path))
                                files_included += 1
                                total_size += file_path.stat().st_size
                            except (OSError, PermissionError):
                                # Skip files we can't read
                                continue

            if files_included == 0:
                os.unlink(snapshot_path)
                return None

            # Check final archive size
            archive_size = os.path.getsize(snapshot_path)
            if archive_size > 50 * 1024 * 1024:  # 50MB limit for code snapshots
                if RICH_AVAILABLE:
                    self.console.print(
                        f"⚠️  Code snapshot too large ({archive_size / (1024 * 1024):.1f}MB), skipping",
                        style="yellow",
                    )
                else:
                    print(
                        f"⚠️  Code snapshot too large ({archive_size / (1024 * 1024):.1f}MB), skipping"
                    )
                os.unlink(snapshot_path)
                return None

            if RICH_AVAILABLE:
                self.console.print(
                    f"📸 Code snapshot created: {files_included} files, {archive_size / (1024 * 1024):.1f}MB",
                    style="green",
                )
            else:
                print(
                    f"📸 Code snapshot created: {files_included} files, {archive_size / (1024 * 1024):.1f}MB"
                )

            return snapshot_path

        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"⚠️  Failed to create code snapshot: {e}", style="yellow")
            else:
                print(f"⚠️  Failed to create code snapshot: {e}")
            return None

    def upload_code_snapshot(self, job_id: str, snapshot_path: str, api_key: str) -> bool:
        """
        Upload code snapshot for a job.
        Returns True if successful, False otherwise.
        """
        try:
            backend_url = os.environ.get(KANDC_BACKEND_URL_ENV_KEY) or KANDC_BACKEND_URL
            endpoint = f"{backend_url.rstrip('/')}/api/v1/jobs/{job_id}/code-snapshot"

            headers = {"Authorization": f"Bearer {api_key}"}

            with open(snapshot_path, "rb") as f:
                files = {"file": ("code-snapshot.tar.gz", f, "application/gzip")}

                if RICH_AVAILABLE:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=self.console,
                    ) as progress:
                        task = progress.add_task("Uploading code snapshot...", total=None)
                        response = requests.post(
                            endpoint, files=files, headers=headers, timeout=300
                        )
                else:
                    print("📤 Uploading code snapshot...")
                    response = requests.post(endpoint, files=files, headers=headers, timeout=300)

            if response.status_code == 200:
                if RICH_AVAILABLE:
                    self.console.print("✅ Code snapshot uploaded successfully", style="green")
                else:
                    print("✅ Code snapshot uploaded successfully")
                return True
            else:
                if RICH_AVAILABLE:
                    self.console.print(
                        f"⚠️  Code snapshot upload failed: {response.status_code}", style="yellow"
                    )
                else:
                    print(f"⚠️  Code snapshot upload failed: {response.status_code}")
                return False

        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"⚠️  Code snapshot upload error: {e}", style="yellow")
            else:
                print(f"⚠️  Code snapshot upload error: {e}")
            return False

    def submit_job(
        self,
        app_name: str,
        upload_dir: str,
        script_path: str,
        gpu: str,
        requirements_file: str,
        script_args: List[str],
        api_key: str,
        include_code_snapshot: bool = True,
    ) -> Dict[str, Any]:
        """Submit job to backend."""
        backend_url = os.environ.get(KANDC_BACKEND_URL_ENV_KEY) or KANDC_BACKEND_URL

        upload_dir = Path(upload_dir)

        # Check for files over 5GB limit
        oversized_files = []
        for file_path in upload_dir.rglob("*"):
            if file_path.is_file():
                try:
                    file_size = file_path.stat().st_size
                    if file_size > MAX_UPLOAD_FILE_SIZE:
                        oversized_files.append((file_path, file_size))
                except (OSError, IOError):
                    # Skip files we can't access
                    continue

        if oversized_files:
            # Display error for files over 5GB
            if RICH_AVAILABLE:
                self.console.print(
                    f"[red]❌ Error: Found {len(oversized_files)} file(s) over 5GB limit[/red]"
                )
                for file_path, file_size in oversized_files:
                    size_gb = file_size / (1024 * 1024 * 1024)
                    self.console.print(f"[red]  • {file_path.name}: {size_gb:.2f}GB[/red]")
                self.console.print(
                    f"[yellow]📝 Solution: Download these files within your script instead[/yellow]"
                )
                self.console.print(
                    f"[yellow]   Example: Use requests.get(), urllib.request, or similar to download[/yellow]"
                )
                self.console.print(
                    f"[yellow]   the files during script execution rather than uploading them.[/yellow]"
                )
            else:
                print(f"❌ Error: Found {len(oversized_files)} file(s) over 5GB limit")
                for file_path, file_size in oversized_files:
                    size_gb = file_size / (1024 * 1024 * 1024)
                    print(f"  • {file_path.name}: {size_gb:.2f}GB")
                print(f"📝 Solution: Download these files within your script instead")
                print(f"   Example: Use requests.get(), urllib.request, or similar to download")
                print(f"   the files during script execution rather than uploading them.")

            # Return error response
            return {
                "success": False,
                "exit_code": 1,
                "error": f"Files over 5GB limit detected. Please download large files within your script instead of uploading them.",
                "oversized_files": [
                    {"name": fp.name, "size_gb": fs / (1024 * 1024 * 1024)}
                    for fp, fs in oversized_files
                ],
            }

        # Archive upload_dir directly
        processed_dir = upload_dir

        # Load ignore patterns for tar filtering (capture should honor kandcignore primarily)
        gitignore_patterns, kandcignore_patterns = load_ignore_patterns(Path(processed_dir))

        # Create a wrapper function for tar filter with ignore patterns
        def tar_filter_with_ignore_patterns(tarinfo):
            # For run: keep both patterns; for capture we already previewed; continue using kandcignore + gitignore for safety
            return tar_filter(tarinfo, gitignore_patterns, kandcignore_patterns)

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
                "requirements_file": requirements_file,
            }

            endpoint = f"{backend_url.rstrip('/')}/api/v1/submit-cachy-job-new"

            # Variables to store job result info (will be set in the response handling)
            job_id = None
            message = None
            visit_url = None
            full_url = None

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

                        # Construct full URL if visit_url is relative
                        if visit_url.startswith("/"):
                            full_url = f"{backend_url}{visit_url}"
                        else:
                            full_url = visit_url

                        progress.update(
                            task, description="Work uploaded successfully! Job submitted"
                        )

                        # Automatically open the URL in the user's default browser
                        try:
                            webbrowser.open(full_url)
                            browser_status = "✅ Browser opened successfully!"
                        except Exception as e:
                            browser_status = f"⚠️  Could not open browser automatically: {e}\n   Please manually visit: {full_url}"

                        # Create a success panel
                        success_panel = Panel(
                            f"🔗 Job ID: {job_id}\n🌐 Visit: {full_url}\n🌐 Opening in browser...\n{browser_status}\n📊 Job is running in the background on cloud GPUs",
                            title="✅ Job Submitted Successfully",
                            border_style="green",
                            box=box.ROUNDED,
                        )
                        self.console.print(success_panel)

                        # Upload code snapshot if enabled
                        if include_code_snapshot and job_id:
                            try:
                                snapshot_path = self.create_code_snapshot(upload_dir)
                                if snapshot_path:
                                    success = self.upload_code_snapshot(
                                        job_id, snapshot_path, api_key
                                    )
                                    # Clean up snapshot file
                                    try:
                                        os.unlink(snapshot_path)
                                    except OSError:
                                        pass
                            except Exception as e:
                                self.console.print(
                                    f"⚠️  Code snapshot failed (job still running): {e}",
                                    style="yellow",
                                )

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

                    # Construct full URL if visit_url is relative
                    if visit_url.startswith("/"):
                        full_url = f"{backend_url}{visit_url}"
                    else:
                        full_url = visit_url

                    upload_spinner.stop("Work uploaded successfully! Job submitted")

                    print(f"🔗 Job ID: {job_id}")
                    print(f"🌐 Visit: {full_url}")
                    print("🌐 Opening in browser...")

                    # Automatically open the URL in the user's default browser
                    try:
                        webbrowser.open(full_url)
                        print("✅ Browser opened successfully!")
                    except Exception as e:
                        print(f"⚠️  Could not open browser automatically: {e}")
                        print(f"   Please manually visit: {full_url}")

                    print("📊 Job is running in the background on cloud GPUs")

                    # Upload code snapshot if enabled
                    if include_code_snapshot and job_id:
                        try:
                            snapshot_path = self.create_code_snapshot(upload_dir)
                            if snapshot_path:
                                success = self.upload_code_snapshot(job_id, snapshot_path, api_key)
                                # Clean up snapshot file
                                try:
                                    os.unlink(snapshot_path)
                                except OSError:
                                    pass
                        except Exception as e:
                            if RICH_AVAILABLE:
                                self.console.print(
                                    f"⚠️  Code snapshot failed (job still running): {e}",
                                    style="yellow",
                                )
                            else:
                                print(f"⚠️  Code snapshot failed (job still running): {e}")

                except Exception as e:
                    upload_spinner.stop("Upload failed")
                    raise e

            return {
                "job_id": job_id,
                "exit_code": 0,
                "logs": [f"{message} (Job ID: {job_id})"],
                "status": "submitted",
                "visit_url": full_url,
            }
        except Exception as e:
            print(f"🔍 [submit_job] Error creating tar archive: {e}")
            raise
        finally:
            if os.path.exists(tar_path):
                os.unlink(tar_path)

    def run_kandc_command(self, command: List[str]) -> int:
        """Run the kandc command with improved interface."""
        if len(command) < 2:
            if RICH_AVAILABLE:
                self.console.print("❌ No command provided!", style="red")
                self.console.print("\n[bold]Usage:[/bold]")
                self.console.print("  kandc python <script.py> [script-args]")
                self.console.print(
                    "  kandc --app-name my-job --gpu A100-80GB:2 -- python <script.py> [script-args]"
                )
            else:
                print("❌ No command provided!")
                print("Usage: kandc python <script.py> [script-args]")
                print(
                    "       kandc --app-name my-job --gpu A100-80GB:2 -- python <script.py> [script-args]"
                )
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

            # Use interactive inputs as final config
            final_config = interactive_inputs
        else:
            # Use command line configuration
            final_config = {
                "app_name": parsed_config["app_name"],
                "upload_dir": parsed_config["upload_dir"],
                "requirements_file": parsed_config["requirements_file"],
                "gpu": parsed_config["gpu"],
                "include_code_snapshot": not parsed_config.get("no_code_snapshot", False),
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

        backend_url = os.environ.get(KANDC_BACKEND_URL_ENV_KEY) or KANDC_BACKEND_URL
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
                include_code_snapshot=final_config.get("include_code_snapshot", True),
            )
            # Gracefully handle failures that return structured error without exit_code
            if isinstance(result, dict) and "exit_code" in result:
                return result["exit_code"]
            return 1
        except Exception as e:
            if RICH_AVAILABLE:
                self.console.print(f"❌ Job submission failed: {e}", style="red")
            else:
                print(f"❌ Job submission failed: {e}")
            return 1

    def capture(
        self,
        app_name: Optional[str],
        open_browser: bool,
        cmd: List[str],
        include_code_snapshot: bool = True,
        code_snapshot_dir: str = ".",
        auto_confirm: bool = False,
    ) -> int:
        """Run a local command with capture and upload results as a job."""
        if not cmd:
            print("❌ No command provided to run")
            return 1

        # Pre-submit confirmation: show what will be uploaded and get consent
        try:
            display_cmd = " ".join(shlex.quote(part) for part in cmd)
        except Exception:
            display_cmd = " ".join(cmd)

        if include_code_snapshot:
            upload_dir_path = Path(code_snapshot_dir).resolve()
            if not upload_dir_path.exists():
                print(f"❌ Code snapshot directory does not exist: {upload_dir_path}")
                return 1

            # Build and display upload preview (like run)
            # Recompute preview using kandcignore-only exclusion for clarity
            preview_data = preview_upload_directory(upload_dir_path, self.console)
            # Strip out pattern listings from display (only show included/excluded files)
            if self.console and RICH_AVAILABLE:
                from rich.panel import Panel

                self.console.print(
                    Panel.fit(
                        f"📦 Capture job\n📝 App name: {app_name or Path.cwd().name}\n▶️  Command: {display_cmd}\n📁 Upload dir: {upload_dir_path}",
                        title="Submission Preview",
                        border_style="cyan",
                    )
                )
                display_upload_preview(
                    preview_data, str(upload_dir_path), self.console, show_patterns=False
                )
                if auto_confirm:
                    proceed = True
                else:
                    from rich.prompt import Confirm

                    proceed = Confirm.ask(
                        "\n🚀 Submit this capture?", default=True, console=self.console
                    )
            else:
                print("\n📦 Capture job")
                print(f"📝 App name: {app_name or Path.cwd().name}")
                print(f"▶️  Command: {display_cmd}")
                print(f"📁 Upload dir: {upload_dir_path}")
                display_upload_preview(
                    preview_data, str(upload_dir_path), None, show_patterns=False
                )
                if auto_confirm:
                    proceed = True
                else:
                    resp = input("\n🚀 Submit this capture? (Y/n): ").strip().lower()
                    proceed = (not resp) or (resp in ["y", "yes"])

            if not proceed:
                if self.console and RICH_AVAILABLE:
                    self.console.print("[yellow]❌ Submission cancelled by user.[/yellow]")
                else:
                    print("❌ Submission cancelled by user.")
                return 0
        else:
            # No code snapshot; still confirm running the command
            if self.console and RICH_AVAILABLE:
                from rich.prompt import Confirm

                if auto_confirm:
                    proceed = True
                else:
                    proceed = Confirm.ask(
                        f"Run capture without uploading code?\n▶️  {display_cmd}",
                        default=True,
                        console=self.console,
                    )
            else:
                if auto_confirm:
                    proceed = True
                else:
                    resp = (
                        input(f"Run capture without uploading code? (Y/n)\n▶️  {display_cmd}\n> ")
                        .strip()
                        .lower()
                    )
                    proceed = (not resp) or (resp in ["y", "yes"])
            if not proceed:
                print("❌ Submission cancelled by user.")
                return 0

        backend_url = os.environ.get(KANDC_BACKEND_URL_ENV_KEY) or KANDC_BACKEND_URL
        api_key = _auth_service.authenticate(backend_url)
        if not api_key:
            print("❌ Authentication failed. Please try again.")
            return 1

        # Initialize job on backend
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        init_payload = {
            "app_name": app_name or Path.cwd().name,
            "script_cmd": cmd[0],
            "script_args": cmd[1:],
        }
        try:
            resp = requests.post(
                f"{backend_url.rstrip('/')}/api/v1/jobs/local/init",
                headers=headers,
                data=json.dumps(init_payload),
                timeout=30,
            )
            resp.raise_for_status()
            init_data = resp.json()
            job_id = init_data.get("job_id")
            visit_url = init_data.get("visit_url")
            if not job_id:
                print("❌ Failed to initialize job")
                return 1
        except Exception as e:
            print(f"❌ Failed to initialize job: {e}")
            return 1

        # Prepare output dir
        base_runs = Path.home() / ".kandc" / "runs"
        base_runs.mkdir(parents=True, exist_ok=True)
        app_dir_name = app_name or Path.cwd().name
        run_dir = base_runs / app_dir_name / job_id
        traces_dir = run_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"

        # Prepare environment
        env = os.environ.copy()
        env[KANDC_BACKEND_RUN_ENV_KEY] = "1"
        env[KANDC_BACKEND_APP_NAME_ENV_KEY] = app_dir_name
        env[KANDC_JOB_ID_ENV_KEY] = job_id
        env[KANDC_TRACE_BASE_DIR_ENV_KEY] = str(base_runs)
        # Ensure Python subprocesses flush output immediately for real-time streaming
        env["PYTHONUNBUFFERED"] = "1"

        # Show start information
        print(
            f"▶️  Starting: {display_cmd}\n"
            f"📂 CWD: {os.getcwd()}\n"
            f"🧾 Job ID: {job_id}\n"
            f"🗂️  Run dir: {run_dir}\n"
            f"📝 Stdout: {stdout_path}\n"
            f"🛠️  Stderr: {stderr_path}"
        )
        if visit_url:
            print(f"🔗 Live view: {visit_url}")

        # If invoking python directly, force unbuffered with -u
        try:
            exe_name = Path(cmd[0]).name.lower()
            if re.match(r"^python(\d+(\.\d+)?)?$", exe_name):
                if "-u" not in cmd[1:3]:
                    cmd = [cmd[0], "-u"] + cmd[1:]
        except Exception:
            pass

        # Run subprocess streaming stdout/stderr and tee to files
        try:
            with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                assert process.stdout and process.stderr

                out_fd = process.stdout.fileno()
                err_fd = process.stderr.fileno()
                fds_open = {out_fd: (sys.stdout.buffer, out_f), err_fd: (sys.stderr.buffer, err_f)}

                # Read available data from either stream as it arrives
                while fds_open:
                    readable, _, _ = select.select(list(fds_open.keys()), [], [], 0.1)
                    if not readable:
                        # Check for process exit to avoid busy wait
                        if process.poll() is not None and not readable:
                            # Drain any remaining data
                            readable = list(fds_open.keys())
                        else:
                            continue
                    for fd in list(readable):
                        try:
                            chunk = os.read(fd, 8192)
                        except Exception:
                            chunk = b""
                        if chunk:
                            stream, file_handle = fds_open[fd]
                            try:
                                stream.write(chunk)
                                stream.flush()
                            except Exception:
                                # Fallback to text write if needed
                                try:
                                    stream.write(chunk.decode(errors="replace"))
                                    stream.flush()
                                except Exception:
                                    pass
                            try:
                                file_handle.write(chunk)
                                file_handle.flush()
                            except Exception:
                                pass
                        else:
                            # EOF on this fd
                            fds_open.pop(fd, None)

                exit_code = process.poll() or 0
        except Exception as e:
            print(f"❌ Failed to run command: {e}")
            exit_code = 1

        # Create outputs tar.gz
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tar_path = tmp_file.name
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(run_dir, arcname=".")
        except Exception as e:
            print(f"❌ Failed to archive outputs: {e}")
            return 1

        # Upload outputs
        try:
            files = {"file": ("outputs.tar.gz", open(tar_path, "rb"), "application/gzip")}
            up_resp = requests.post(
                f"{backend_url.rstrip('/')}/api/v1/jobs/{job_id}/local/upload",
                headers={"Authorization": f"Bearer {api_key}"},
                files=files,
                timeout=300,
            )
            up_resp.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to upload outputs: {e}")
        finally:
            try:
                os.unlink(tar_path)
            except Exception:
                pass

        # Complete job
        try:
            comp_payload = {"exit_code": exit_code}
            comp_resp = requests.post(
                f"{backend_url.rstrip('/')}/api/v1/jobs/{job_id}/local/complete",
                headers=headers,
                data=json.dumps(comp_payload),
                timeout=30,
            )
            comp_resp.raise_for_status()
        except Exception as e:
            print(f"⚠️  Failed to finalize job: {e}")

        # Upload code snapshot if enabled
        if include_code_snapshot and job_id:
            try:
                snapshot_path = self.create_code_snapshot(Path(code_snapshot_dir))
                if snapshot_path:
                    success = self.upload_code_snapshot(job_id, snapshot_path, api_key)
                    # Clean up snapshot file
                    try:
                        os.unlink(snapshot_path)
                    except OSError:
                        pass
            except Exception as e:
                if RICH_AVAILABLE:
                    self.console.print(
                        f"⚠️  Code snapshot failed (job still running): {e}",
                        style="yellow",
                    )
                else:
                    print(f"⚠️  Code snapshot failed (job still running): {e}")

        # Open URL
        if visit_url and open_browser:
            try:
                webbrowser.open(visit_url)
            except Exception:
                print(f"🔗 Visit: {visit_url}")

        print(f"✅ Capture complete. Job: {job_id}")
        if visit_url:
            print(f"🔗 Visit: {visit_url}")
        return exit_code


def main():
    """Main CLI entry point."""
    cli = KandcCLI()

    if len(sys.argv) < 2:
        if cli.console:
            cli.console.print("Keys & Caches CLI is installed and working!", style="bold green")
            cli.console.print("\n[bold]Usage Formats:[/bold]")
            cli.console.print(
                "  [cyan]Separator format:[/cyan] kandc run [kandc-flags] -- python <script.py> [script-args]"
            )
            cli.console.print(
                "  [cyan]Interactive:[/cyan]      kandc run python <script.py> [script-args]"
            )
            cli.console.print("\n[bold]Keys & Caches Flags:[/bold]")
            cli.console.print("  --app-name, -a     Job name for tracking")
            cli.console.print("  --gpu, -g          GPU count (1,2,4,8)")
            cli.console.print("  --upload-dir, -d   Directory to upload")
            cli.console.print("  --requirements, -r Requirements file")
            cli.console.print("  --interactive, -i  Force interactive mode")
            cli.console.print("  --preview, -p      Preview upload contents")
            cli.console.print("  --logout           Clear authentication")
            cli.console.print("  --version          Show version")
            cli.console.print("\n[bold]Examples:[/bold]")
            cli.console.print("  [green]# Separator format (kandc flags first, then --):[/green]")
            cli.console.print(
                "  kandc run --app-name my-job --gpu A100-80GB:2 -- python train.py --model-size large"
            )
            cli.console.print(
                "  [green]# Interactive mode (script args only, prompts for config):[/green]"
            )
            cli.console.print("  kandc run python train.py --model-size large --epochs 10")
            cli.console.print(
                "\n💡 Tip: Pre-filled mode lets you specify some flags while still confirming interactively!"
            )
        else:
            print("Keys & Caches CLI is installed and working!")
            print()
            print("Usage Formats:")
            print("  Separator format: kandc run [kandc-flags] -- python <script.py> [script-args]")
            print("  Interactive:      kandc run python <script.py> [script-args]")
            print()
            print("Keys & Caches Flags:")
            print("  --app-name, -a     Job name for tracking")
            print("  --gpu, -g          GPU configuration (e.g., A100-80GB:2, H100:4)")
            print("  --upload-dir, -d   Directory to upload")
            print("  --requirements, -r Requirements file")
            print("  --interactive, -i  Force interactive mode")
            print("  --preview, -p      Preview upload contents")
            print("  --logout           Clear authentication")
            print("  --version          Show version")
            print()
            print("Examples:")
            print("  # Separator format (kandc flags first, then --):")
            print(
                "  kandc run --app-name my-job --gpu A100-80GB:2 -- python train.py --model-size large"
            )
            print("  # Interactive mode (script args only, prompts for config):")
            print("  kandc run python train.py --model-size large --epochs 10")
            print()
            print(
                "💡 Tip: Pre-filled mode lets you specify some flags while still confirming interactively!"
            )
        return 0

    # Handle version flag
    if sys.argv[1] in ["--version", "-v", "version"]:
        from . import __version__

        print(f"Keys & Caches CLI v{__version__}")
        return 0

    # Handle logout flag
    if sys.argv[1] == "--logout":
        if _auth_service.is_authenticated():
            _auth_service.clear()
            print("✅ Successfully logged out from Keys & Caches CLI")
        else:
            print("ℹ️  No active authentication found")
        return 0

    # Capture subcommand
    if sys.argv[1] == "capture":
        # Usage A: kandc capture [--app-name NAME] [--no-open] [--no-code-snapshot] -- <command> [args...]
        # Usage B: kandc capture  (then prompt for app name and command)
        try:
            app_name: Optional[str] = None
            open_browser = True
            include_code_snapshot = True
            code_snapshot_dir = "."
            auto_confirm = False
            cmd: List[str] = []

            if "--" in sys.argv:
                sep_index = sys.argv.index("--")
                flags = sys.argv[2:sep_index]
                cmd = sys.argv[sep_index + 1 :]
                i = 0
                while i < len(flags):
                    if flags[i] in ["--app-name", "-a"] and i + 1 < len(flags):
                        app_name = flags[i + 1]
                        i += 2
                    elif flags[i] == "--no-open":
                        open_browser = False
                        i += 1
                    elif flags[i] == "--no-code-snapshot":
                        include_code_snapshot = False
                        i += 1
                    elif flags[i] in ["--code-snapshot-dir", "-d"] and i + 1 < len(flags):
                        code_snapshot_dir = flags[i + 1]
                        include_code_snapshot = True
                        i += 2
                    elif flags[i] == "--auto-confirm":
                        auto_confirm = True
                        i += 1
                    else:
                        print(f"Unknown flag: {flags[i]}")
                        return 1
            else:
                # Parse any flags provided (optional) and then treat remaining tokens as the command
                tokens = sys.argv[2:]
                i = 0
                while i < len(tokens):
                    if tokens[i] in ["--app-name", "-a"] and i + 1 < len(tokens):
                        app_name = tokens[i + 1]
                        i += 2
                    elif tokens[i] == "--no-open":
                        open_browser = False
                        i += 1
                    elif tokens[i] == "--no-code-snapshot":
                        include_code_snapshot = False
                        i += 1
                    elif tokens[i] in ["--code-snapshot-dir", "-d"] and i + 1 < len(tokens):
                        code_snapshot_dir = tokens[i + 1]
                        include_code_snapshot = True
                        i += 2
                    elif tokens[i] == "--auto-confirm":
                        auto_confirm = True
                        i += 1
                    else:
                        break

                # Remaining tokens after flags are considered the command
                if i < len(tokens):
                    cmd = tokens[i:]
                else:
                    # No command supplied; prompt for command and code snapshot option
                    default_app = Path.cwd().name
                    if RICH_AVAILABLE:
                        if not app_name:
                            app_name = cli.get_valid_app_name(
                                "📝 App name (for job tracking)",
                                default=default_app,
                            )
                        # Ask whether to upload repo and which directory
                        include_code_snapshot = cli.get_yes_no_input(
                            "📸 Include code snapshot for debugging?",
                            default=True,
                            help_text="Uploads a snapshot of your code for viewing in the web interface",
                        )
                        code_snapshot_dir = "."
                        if include_code_snapshot:
                            code_snapshot_dir = cli.get_input_with_default(
                                "📁 Code snapshot directory",
                                default=".",
                                required=False,
                            )
                        command_text = Prompt.ask(
                            "▶️  Command to run (example: python script.py --arg val)",
                            console=cli.console,
                        )
                    else:
                        if not app_name:
                            while True:
                                user_input = input(f"App name (default: {default_app}): ").strip()
                                app_name = user_input if user_input else default_app

                                is_valid, error_message = validate_app_name(app_name)
                                if is_valid:
                                    break

                                print(f"❌ Invalid app name: {error_message}")
                                suggested = suggest_valid_app_name(app_name)
                                if suggested != app_name:
                                    print(f"💡 Suggested: '{suggested}'")
                                    use_suggestion = (
                                        input(f"Use suggested name '{suggested}'? (Y/n): ")
                                        .strip()
                                        .lower()
                                    )
                                    if not use_suggestion or use_suggestion in ["y", "yes"]:
                                        app_name = suggested
                                        break
                                print(
                                    "💡 Valid names: alphanumeric, dashes, periods, underscores only (no spaces)"
                                )
                        # Ask upload preferences before command (plain text)
                        while True:
                            response = (
                                input("Include code snapshot for debugging? (Y/n): ")
                                .strip()
                                .lower()
                            )
                            if not response or response in ["y", "yes"]:
                                include_code_snapshot = True
                                break
                            elif response in ["n", "no"]:
                                include_code_snapshot = False
                                break
                            else:
                                print("Please enter 'y' for yes or 'n' for no.")

                        code_snapshot_dir = "."
                        if include_code_snapshot:
                            code_snapshot_dir = (
                                input("Code snapshot directory (default: .): ").strip() or "."
                            )

                        command_text = input(
                            "Command to run (e.g., python script.py --arg val): "
                        ).strip()

                    if not command_text:
                        print("❌ No command provided")
                        return 1
                    try:
                        cmd = shlex.split(command_text)
                    except Exception:
                        cmd = command_text.split()

            # If app name still missing, prompt now
            if not app_name:
                default_app = Path.cwd().name
                if RICH_AVAILABLE:
                    app_name = cli.get_valid_app_name(
                        "📝 App name (for job tracking)", default=default_app
                    )
                else:
                    while True:
                        user_input = input(f"App name (default: {default_app}): ").strip()
                        app_name = user_input if user_input else default_app

                        is_valid, error_message = validate_app_name(app_name)
                        if is_valid:
                            break

                        print(f"❌ Invalid app name: {error_message}")
                        suggested = suggest_valid_app_name(app_name)
                        if suggested != app_name:
                            print(f"💡 Suggested: '{suggested}'")
                            use_suggestion = (
                                input(f"Use suggested name '{suggested}'? (Y/n): ").strip().lower()
                            )
                            if not use_suggestion or use_suggestion in ["y", "yes"]:
                                app_name = suggested
                                break
                        print(
                            "💡 Valid names: alphanumeric, dashes, periods, underscores only (no spaces)"
                        )

            return cli.capture(
                app_name=app_name,
                open_browser=open_browser,
                cmd=cmd,
                include_code_snapshot=include_code_snapshot,
                code_snapshot_dir=code_snapshot_dir,
                auto_confirm=auto_confirm,
            )
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1

    # Run subcommand (cloud submission)
    if sys.argv[1] == "run":
        command = sys.argv[2:]
        return cli.run_kandc_command(command)

    # Sweep subcommand (local capture or cloud run across a folder of configs)
    if sys.argv[1] == "sweep":
        args = sys.argv[2:]
        if not args:
            print("❌ Usage: kandc sweep {capture|run} [options] -- [script-args...]")
            return 1

        mode = args[0]
        if mode not in ("capture", "run"):
            print("❌ Usage: kandc sweep {capture|run} [options] -- [script-args...]")
            return 1

        # Split flags from script args
        if "--" in args:
            sep = args.index("--")
            sweep_flags = args[1:sep]
            script_args = args[sep + 1 :]
        else:
            sweep_flags = args[1:]
            script_args = []

        # Parse sweep flags
        parser = argparse.ArgumentParser(prog=f"kandc sweep {mode}")
        parser.add_argument("--configs-dir", "-c", required=True)
        parser.add_argument("--script", "-s", required=True)
        parser.add_argument("--app-name", "-a", required=True, help="Shared app name for all runs")

        if mode == "capture":
            parser.add_argument("--gpus", default="0", help="Comma-separated GPU ids, e.g. 0,1,2")
            parser.add_argument("--per-run-gpus", type=int, default=1, help="GPUs per run")
            parser.add_argument("--no-open", action="store_true")
            parser.add_argument("--no-code-snapshot", action="store_true")
            parser.add_argument("--code-snapshot-dir", default=".", help="Directory to snapshot for upload")
            parser.add_argument("--auto-confirm", action="store_true", help="Skip all interactive confirmations")
            parser.add_argument("--tmux", action="store_true", help="Launch each job in a tmux window and auto-close when all finish")
        else:
            parser.add_argument("--gpu-type", default="A100-80GB:1")
            parser.add_argument("--upload-dir", default=".")
            parser.add_argument("--requirements", default="requirements.txt")

        ns = parser.parse_args(sweep_flags)
        configs_dir = Path(ns.configs_dir)
        script_path = Path(ns.script)
        shared_app = ns.app_name

        def _list_cfgs(d: Path) -> List[Path]:
            exts = {".yaml", ".yml", ".json"}
            return sorted(p for p in d.glob("**/*") if p.suffix.lower() in exts)

        cfgs = _list_cfgs(configs_dir)
        if not cfgs:
            print(f"❌ No configs found in {configs_dir}")
            return 1

        if mode == "capture":
            # Local capture: schedule runs across GPUs using CUDA_VISIBLE_DEVICES
            try:
                gpu_ids = [int(x.strip()) for x in str(ns.gpus).split(",") if x.strip() != ""]
            except ValueError:
                print("❌ --gpus must be a comma-separated list of integers, e.g. 0,1,2")
                return 1
            per = max(1, int(ns.per_run_gpus))
            if per > len(gpu_ids):
                print("❌ --per-run-gpus cannot exceed number of provided GPUs")
                return 1

            # Optional tmux mode: one window per job, auto-attach and auto-close when all finish
            if getattr(ns, "tmux", False):
                import shutil, time, shlex

                if shutil.which("tmux") is None:
                    print("❌ tmux is not installed or not in PATH")
                    return 1

                # Build commands per config, assign GPUs round-robin
                commands: List[str] = []
                for idx, cfg in enumerate(cfgs):
                    start = (idx * per) % len(gpu_ids)
                    assigned = [str(gpu_ids[(start + j) % len(gpu_ids)]) for j in range(per)]
                    reqs = ",".join(assigned)

                    cmd_list: List[str] = [
                        "kandc",
                        "capture",
                        "--app-name",
                        shared_app,
                    ]
                    if ns.no_open:
                        cmd_list.append("--no-open")
                    if ns.no_code_snapshot:
                        cmd_list.append("--no-code-snapshot")
                    else:
                        cmd_list += ["--code-snapshot-dir", ns.code_snapshot_dir]
                    if ns.auto_confirm:
                        cmd_list += ["--auto-confirm"]
                    cmd_list += ["--", "python", str(script_path), "--config", str(cfg)]
                    if script_args:
                        cmd_list += script_args

                    job_cmd = " ".join(shlex.quote(p) for p in cmd_list)
                    # Wrapper that waits for GPU locks, prints waiting messages, sets CUDA_VISIBLE_DEVICES, then execs the job
                    wrapper = (
                        "bash -lc "
                        + shlex.quote(
                            "LOCK_DIR=\"$HOME/.kandc/gpu_locks\"; "
                            "mkdir -p \"$LOCK_DIR\"; "
                            f"REQS=\"{reqs}\"; "
                            "IFS=',' read -r -a IDS <<< \"$REQS\"; "
                            "acquired=(); "
                            "while :; do "
                            "ok=1; acquired=(); "
                            "for id in \"${IDS[@]}\"; do lf=\"$LOCK_DIR/gpu_${id}.lock\"; if ln -s \"$$\" \"$lf\" 2>/dev/null; then acquired+=(\"$lf\"); else ok=0; fi; done; "
                            "if [ \"$ok\" -eq 1 ]; then break; fi; "
                            "for lf in \"${acquired[@]}\"; do rm -f \"$lf\"; done; "
                            "echo '⏳ waiting for GPUs ['\"$REQS\"'] ...'; sleep 2; "
                            "done; "
                            "trap 'for lf in \"${acquired[@]}\"; do rm -f \"$lf\"; done' EXIT; "
                            "export CUDA_VISIBLE_DEVICES=\"$REQS\"; "
                            + job_cmd
                        )
                    )
                    commands.append(wrapper)

                # Create session name
                import re
                base_name = f"kandc-sweep-{shared_app}"
                session_name = re.sub(r"[^A-Za-z0-9_-]", "-", base_name)[:60]

                # If a previous session exists, kill it to avoid stale layouts
                try:
                    rc = subprocess.call(["tmux", "has-session", "-t", session_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if rc == 0:
                        subprocess.call(["tmux", "kill-session", "-t", session_name])
                except Exception:
                    pass

                # Create tmux session with first command in window 0, pane 0
                first = commands[0]
                try:
                    rc = subprocess.call(["tmux", "new-session", "-d", "-s", session_name, first])
                    if rc != 0:
                        print("❌ Failed to create tmux session")
                        return 1
                except Exception as e:
                    print(f"❌ tmux error: {e}")
                    return 1

                # Create split panes so all jobs are visible simultaneously in a single window
                num_configs = len(commands)
                if num_configs > 1:
                    print(f"🎬 Creating tmux session '{session_name}' with {num_configs} jobs in a tiled pane layout")
                    print("   All jobs will be visible simultaneously side-by-side!")

                    for idx in range(1, num_configs):
                        cmd_str = commands[idx]
                        # Split the current window and start the command in the new pane
                        # Target window 0 to keep all panes in the same window
                        # Alternate split direction for better initial placement
                        split_flag = "-h" if (idx % 2 == 1) else "-v"
                        subprocess.call(["tmux", "split-window", split_flag, "-t", f"{session_name}:0", cmd_str])
                        # After each split, re-tile to keep panes evenly arranged
                        subprocess.call(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"])

                    # Final layout pass
                    subprocess.call(["tmux", "select-layout", "-t", f"{session_name}:0", "tiled"])
                    print(f"✅ Tiled layout created with {num_configs} panes in one window")
                else:
                    print(f"🎬 Created tmux session '{session_name}' with 1 job")

                # Background monitor to kill session when all panes are dead
                def _monitor_tmux():
                    while True:
                        try:
                            # Check if all panes in the session are dead
                            out = subprocess.check_output(
                                ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_dead}"],
                                stderr=subprocess.DEVNULL,
                            ).decode().strip().splitlines()
                            
                            if out and all(x.strip() == "1" for x in out):
                                try:
                                    subprocess.call(["tmux", "kill-session", "-t", session_name])
                                    print(f"✅ All jobs completed, tmux session '{session_name}' closed")
                                except Exception:
                                    pass
                                break
                        except Exception:
                            break
                        time.sleep(1.0)

                t = threading.Thread(target=_monitor_tmux, daemon=True)
                t.start()

                # Attach to the session
                try:
                    print(f"🔗 Attaching to tmux session '{session_name}'...")
                    print("   All jobs are now visible simultaneously in split panes!")
                    print("   Session will auto-close when all jobs finish")
                    subprocess.call(["tmux", "attach-session", "-t", session_name])
                except Exception as e:
                    print(f"⚠️  tmux attach failed: {e}")
                return 0

            class _GpuPool:
                def __init__(self, ids: List[int], per_run: int):
                    self._pool = ids[:]
                    self._per = per_run
                    self._cv = threading.Condition()

                def acquire(self) -> List[int]:
                    with self._cv:
                        while len(self._pool) < self._per:
                            self._cv.wait()
                        out = [self._pool.pop(0) for _ in range(self._per)]
                        return out

                def release(self, ids: List[int]):
                    with self._cv:
                        self._pool.extend(ids)
                        self._cv.notify_all()

            pool = _GpuPool(gpu_ids, per)
            q: "queue.Queue[Path | None]" = queue.Queue()
            for c in cfgs:
                q.put(c)

            def _worker():
                while True:
                    item = q.get()
                    if item is None:
                        break
                    cfg = item
                    g = pool.acquire()
                    try:
                        env = os.environ.copy()
                        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in g)
                        cmd = [
                            "kandc",
                            "capture",
                            "--app-name",
                            shared_app,
                        ]
                        if ns.no_open:
                            cmd.append("--no-open")
                        if ns.no_code_snapshot:
                            cmd.append("--no-code-snapshot")
                        else:
                            # pass snapshot dir through when uploading code
                            cmd += ["--code-snapshot-dir", ns.code_snapshot_dir]
                        if ns.auto_confirm:
                            cmd += ["--auto-confirm"]
                        cmd += ["--", "python", str(script_path), "--config", str(cfg)]
                        if script_args:
                            cmd += script_args
                        try:
                            rc = subprocess.call(cmd, env=env)
                        except Exception:
                            rc = 1
                        status = "ok" if rc == 0 else f"fail({rc})"
                        print(f"[sweep] {cfg.name} on GPUs {g} → {status}")
                    finally:
                        pool.release(g)
                        q.task_done()

            # Start workers based on available capacity
            max_workers = max(1, len(gpu_ids) // per)
            threads: List[threading.Thread] = []
            for _ in range(max_workers):
                t = threading.Thread(target=_worker, daemon=True)
                t.start()
                threads.append(t)

            q.join()
            for _ in threads:
                q.put(None)
            for t in threads:
                t.join()
            return 0

        # mode == "run": cloud submissions
        upload_dir = Path(ns.upload_dir).resolve()
        script_abs = script_path.resolve()
        try:
            script_rel = str(script_abs.relative_to(upload_dir))
        except ValueError:
            print(f"❌ Script {script_abs} is not inside upload_dir {upload_dir}")
            return 1

        failures = 0
        for cfg in cfgs:
            cmd = [
                "kandc",
                "run",
                "--app-name",
                shared_app,
                "--gpu",
                ns.gpu_type,
                "--upload-dir",
                str(upload_dir),
                "--requirements",
                str(ns.requirements),
                "--",
                "python",
                script_rel,
                "--config",
                str(cfg),
            ]
            if script_args:
                cmd += script_args
            try:
                rc = subprocess.call(cmd, env=os.environ.copy())
            except Exception:
                rc = 1
            if rc != 0:
                failures += 1
        print(f"[sweep] run complete. failures={failures}, total={len(cfgs)}")
        return 0 if failures == 0 else 1

    # Friendly guidance for common mistakes
    if sys.argv[1] == "python" or sys.argv[1].endswith(".py"):
        print("❌ Unknown command. Did you mean: kandc run python <script.py> [args] ?")
        return 1

    # Unknown subcommand
    print("❌ Unknown subcommand.")
    print("Usage:")
    print("  kandc run [kandc-flags] -- python <script.py> [script-args]")
    print(
        "  kandc capture [--app-name NAME] [--no-open] [--no-code-snapshot] -- <command> [args...]"
    )
    print("  kandc sweep {capture|run} [options] -- [script-args...]")
    return 1
