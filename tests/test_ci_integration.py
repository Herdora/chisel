"""Integration tests for CI/CD pipeline."""

import subprocess
import sys
import tempfile
import os
from pathlib import Path


def test_package_importable():
    """Test that the package can be imported successfully."""
    import kandc

    assert hasattr(kandc, "capture_model_class")
    assert hasattr(kandc, "capture_model_instance")
    assert hasattr(kandc, "timed")


# def test_cli_available():
#     """Test that the CLI command is available."""
#     result = subprocess.run(
#         [sys.executable, "-m", "kandc", "--help"], capture_output=True, text=True
#     )
#     assert result.returncode == 0
#     assert "kandc" in result.stdout.lower()


def test_package_version():
    """Test that package version is accessible."""
    import kandc

    # Should have a version attribute
    assert hasattr(kandc, "__version__") or hasattr(kandc, "VERSION")


# def test_build_process():
#     """Test that the package can be built successfully."""
#     # This test runs the build process in a temporary directory
#     # to ensure our package structure is correct

#     # Skip this test if build tools aren't available
#     try:
#         import build
#     except ImportError:
#         import pytest

#         pytest.skip("build package not available")

#     # Get the project root
#     project_root = Path(__file__).parent.parent

#     with tempfile.TemporaryDirectory() as temp_dir:
#         # Copy essential files to temp directory
#         import shutil

#         temp_project = Path(temp_dir) / "test_build"
#         shutil.copytree(
#             project_root,
#             temp_project,
#             ignore=shutil.ignore_patterns("dist", "build", "*.egg-info", "__pycache__", ".git"),
#         )

#         # Run build
#         result = subprocess.run(
#             [sys.executable, "-m", "build", str(temp_project)],
#             capture_output=True,
#             text=True,
#             cwd=temp_project,
#         )

#         # Check if build succeeded
#         if result.returncode != 0:
#             print("Build stdout:", result.stdout)
#             print("Build stderr:", result.stderr)

#         assert result.returncode == 0, f"Build failed: {result.stderr}"

#         # Check that dist files were created
#         dist_dir = temp_project / "dist"
#         assert dist_dir.exists(), "dist directory not created"

#         dist_files = list(dist_dir.glob("*"))
#         assert len(dist_files) > 0, "No distribution files created"

#         # Should have both wheel and source distribution
#         wheels = list(dist_dir.glob("*.whl"))
#         tarballs = list(dist_dir.glob("*.tar.gz"))

#         assert len(wheels) > 0, "No wheel file created"
#         assert len(tarballs) > 0, "No source distribution created"


# def test_dev_build_process():
#     """Test that the dev package can be built successfully."""
#     try:
#         import build
#     except ImportError:
#         import pytest

#         pytest.skip("build package not available")

#     project_root = Path(__file__).parent.parent

#     with tempfile.TemporaryDirectory() as temp_dir:
#         import shutil

#         temp_project = Path(temp_dir) / "test_dev_build"
#         shutil.copytree(
#             project_root,
#             temp_project,
#             ignore=shutil.ignore_patterns("dist", "build", "*.egg-info", "__pycache__", ".git"),
#         )

#         # Copy dev config
#         dev_config = temp_project / "pyproject.dev.toml"
#         main_config = temp_project / "pyproject.toml"

#         if dev_config.exists():
#             shutil.copy2(dev_config, main_config)

#             # Run build
#             result = subprocess.run(
#                 [sys.executable, "-m", "build", str(temp_project)],
#                 capture_output=True,
#                 text=True,
#                 cwd=temp_project,
#             )

#             if result.returncode != 0:
#                 print("Dev build stdout:", result.stdout)
#                 print("Dev build stderr:", result.stderr)

#             assert result.returncode == 0, f"Dev build failed: {result.stderr}"

#             # Check dist files
#             dist_dir = temp_project / "dist"
#             assert dist_dir.exists()

#             dist_files = list(dist_dir.glob("*"))
#             assert len(dist_files) > 0

#             # Should contain kandc-dev in filenames
#             dev_files = [f for f in dist_files if "kandc_dev" in f.name or "kandc-dev" in f.name]
#             assert len(dev_files) > 0, "No kandc-dev files found in build output"


def test_examples_syntax():
    """Test that example files have valid Python syntax."""
    project_root = Path(__file__).parent.parent
    examples_dir = project_root / "examples"

    if not examples_dir.exists():
        import pytest

        pytest.skip("Examples directory not found")

    python_files = list(examples_dir.rglob("*.py"))
    assert len(python_files) > 0, "No Python example files found"

    for py_file in python_files:
        # Skip __pycache__ and other generated files
        if "__pycache__" in str(py_file) or py_file.name.startswith("."):
            continue

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()

            # Compile to check syntax
            compile(source, str(py_file), "exec")

        except SyntaxError as e:
            assert False, f"Syntax error in {py_file}: {e}"
        except Exception as e:
            # Other errors (like import errors) are OK for this test
            pass
