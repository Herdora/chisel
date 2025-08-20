"""
Code snapshot capture functionality for Keys & Caches.

This module captures the current state of source code files and uploads them
to the backend for later browsing and analysis.
"""

import os
import json
import fnmatch
import hashlib
import tarfile
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime
import io


def capture_project_source_code(
    project_root: Optional[Union[str, Path]] = None,
    exclude_patterns: Optional[List[str]] = None,
    max_file_size: int = 1024 * 1024,  # 1MB default max file size
    include_extensions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Capture all source code in the project directory.

    Args:
        project_root: Root directory to scan (defaults to current working directory)
        exclude_patterns: List of patterns to exclude (supports wildcards)
        max_file_size: Maximum file size to capture in bytes
        include_extensions: File extensions to include (defaults to common source files)

    Returns:
        Dictionary containing snapshot metadata and file information
    """
    if project_root is None:
        project_root = os.getcwd()

    project_root = Path(project_root).resolve()

    # Default file extensions to capture
    if include_extensions is None:
        include_extensions = [
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".cmd",
            ".yaml",
            ".yml",
            ".json",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            ".md",
            ".rst",
            ".txt",
            ".dockerfile",
            ".gitignore",
            ".env.example",
            ".html",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".vue",
            ".svelte",
        ]

    # Default exclude patterns
    default_exclude = [
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        ".env",
        "env",
        "node_modules",
        "dist",
        "build",
        ".pytest_cache",
        ".ruff_cache",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "*.egg-info",
        ".mypy_cache",
        ".tox",
        "htmlcov",
        ".coverage",
        ".next",
        ".nuxt",
        "target",
        "bin",
        "obj",
        ".gradle",
        ".idea",
        ".vscode",
        "*.min.js",
        "*.min.css",
        "*.map",
        ".git/*",
        "*.log",
        "*.tmp",
        "*.temp",
        "kandc",
        "kandc/*",
        ".kandc",
        ".kandc/*",  # Exclude kandc output directories
    ]

    exclude_patterns = exclude_patterns or []
    exclude_patterns.extend(default_exclude)

    source_files = []
    total_size = 0
    skipped_files = []

    def should_exclude(path: Path) -> bool:
        """Check if a path should be excluded (Git-style patterns)."""
        try:
            path_str = str(path.relative_to(project_root))
        except ValueError:
            path_str = str(path)

        for pattern in exclude_patterns:
            # Handle Git-style directory patterns (ending with /)
            if pattern.endswith('/'):
                # For directory patterns, check if the path starts with the directory
                dir_pattern = pattern[:-1]  # Remove trailing slash
                # Check if path is inside this directory
                if path_str.startswith(dir_pattern + '/') or path_str == dir_pattern:
                    return True
                # Also check if any parent directory matches
                path_parts = path_str.split('/')
                for i in range(len(path_parts)):
                    if '/'.join(path_parts[:i+1]) == dir_pattern:
                        return True
            else:
                # Regular file patterns - use fnmatch for wildcards
                if fnmatch.fnmatch(path.name, pattern) or fnmatch.fnmatch(path_str, pattern):
                    return True
        return False

    def get_file_language(file_path: Path) -> str:
        """Determine the programming language based on file extension."""
        ext = file_path.suffix.lower()
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".toml": "toml",
            ".ini": "ini",
            ".md": "markdown",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
        }
        return language_map.get(ext, "text")

    for root, dirs, files in os.walk(project_root):
        root_path = Path(root)
        dirs[:] = [d for d in dirs if not should_exclude(root_path / d)]

        for file in files:
            file_path = root_path / file
            if should_exclude(file_path):
                continue
            if not any(file.endswith(ext) for ext in include_extensions):
                continue

            try:
                file_size = file_path.stat().st_size
                if file_size > max_file_size:
                    skipped_files.append(
                        {
                            "path": str(file_path.relative_to(project_root)),
                            "reason": "too_large",
                            "size": file_size,
                        }
                    )
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    try:
                        with open(file_path, "r", encoding="latin-1") as f:
                            content = f.read()
                    except:
                        skipped_files.append(
                            {
                                "path": str(file_path.relative_to(project_root)),
                                "reason": "encoding_error",
                            }
                        )
                        continue

                try:
                    relative_path = file_path.relative_to(project_root)
                except ValueError:
                    relative_path = file_path

                file_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

                source_files.append(
                    {
                        "file_path": str(file_path),
                        "relative_path": str(relative_path),
                        "file_name": file,
                        "content": content,
                        "file_size": file_size,
                        "last_modified": file_path.stat().st_mtime,
                        "language": get_file_language(file_path),
                        "hash": file_hash,
                        "lines": len(content.splitlines()),
                    }
                )

                total_size += file_size

            except Exception as e:
                try:
                    rel_path = str(file_path.relative_to(project_root))
                except:
                    rel_path = str(file_path)

                skipped_files.append({"path": rel_path, "reason": "read_error", "error": str(e)})
                continue

    snapshot_metadata = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "total_files": len(source_files),
        "total_size": total_size,
        "skipped_files": len(skipped_files),
        "skipped_details": skipped_files[:10],
        "include_extensions": include_extensions,
        "exclude_patterns": exclude_patterns[:20],
        "languages": list(set(f["language"] for f in source_files)),
    }

    return {"metadata": snapshot_metadata, "files": source_files}


def create_snapshot_archive(snapshot_data: Dict[str, Any]) -> bytes:
    """
    Create a compressed archive of the code snapshot.

    Args:
        snapshot_data: Output from capture_project_source_code()

    Returns:
        Compressed tar.gz archive as bytes
    """
    archive_buffer = io.BytesIO()

    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as tar:
        metadata_json = json.dumps(snapshot_data["metadata"], indent=2)
        metadata_bytes = metadata_json.encode("utf-8")
        metadata_info = tarfile.TarInfo(name="snapshot_metadata.json")
        metadata_info.size = len(metadata_bytes)
        tar.addfile(metadata_info, io.BytesIO(metadata_bytes))

        for file_data in snapshot_data["files"]:
            file_content = file_data["content"]
            file_bytes = file_content.encode("utf-8")
            file_info = tarfile.TarInfo(name=file_data["relative_path"])
            file_info.size = len(file_bytes)
            file_info.mtime = int(file_data["last_modified"])
            tar.addfile(file_info, io.BytesIO(file_bytes))

    archive_buffer.seek(0)
    return archive_buffer.read()
