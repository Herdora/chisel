__version__ = "0.1.0"

from .core import ChiselApp
from .constants import GPUType

App = ChiselApp

__all__ = [
    "ChiselApp",
    "App",
    "GPUType",
    "__version__",
]


def main():
    import sys
    import os
    import subprocess

    if len(sys.argv) < 2:
        print("Chisel CLI is installed and working!")
        print(f"Version: {__version__}")
        print("Usage: chisel <command>")
        print("Example: chisel python my_script.py")
        return 0

    command = sys.argv[1:]

    env = os.environ.copy()
    env["CHISEL_ACTIVATED"] = "1"

    try:
        result = subprocess.run(command, env=env)
        return result.returncode
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return 1
    except KeyboardInterrupt:
        return 130
