#!/usr/bin/env python3
"""
Command Line Interface for Keys & Caches.
"""

import argparse
import sys
from pathlib import Path

from .auth import get_auth_manager


def logout():
    """Clear stored credentials."""
    try:
        auth_manager = get_auth_manager()
        auth_manager.clear_credentials()
        print("✅ Successfully logged out")
        print("   Credentials cleared from ~/.kandc/settings.json")
        except Exception as e:
        print(f"❌ Logout failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Keys & Caches - GPU profiling and tracing library", prog="kandc"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Logout command
    logout_parser = subparsers.add_parser("logout", help="Clear stored credentials")
    logout_parser.set_defaults(func=logout)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func()


if __name__ == "__main__":
    main()
