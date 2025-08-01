__version__ = "0.1.0"

from .core import ChiselApp

App = ChiselApp

__all__ = [
    "ChiselApp",
    "App",
    "__version__",
]


def main():
    import sys

    print("Chisel CLI is installed and working!")
    print(f"Version: {__version__}")
    print("Usage: from chisel import ChiselApp")
    return 0
