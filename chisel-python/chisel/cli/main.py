"""Chisel - Seamless GPU kernel profiling on cloud infrastructure.

Main entry point for the chisel CLI application.
"""

from chisel.cli.cli import run_cli


def main():
    """Main entry point for the CLI."""
    run_cli()


if __name__ == "__main__":
    main()
