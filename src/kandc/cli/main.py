#!/usr/bin/env python3
"""
Command Line Interface for Keys & Caches.
"""

import argparse
import sys

from ..api.auth import get_auth_manager
from ..constants import print_config


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


def config():
    """Show current configuration."""
    try:
        print_config()
    except Exception as e:
        print(f"❌ Failed to show config: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Keys & Caches - GPU profiling and tracing library", prog="kandc"
    )

    # Add flags
    parser.add_argument("--logout", action="store_true", help="Clear stored credentials")
    parser.add_argument("--config", action="store_true", help="Show current configuration")

    # Add subcommands for backward compatibility
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Logout command
    logout_parser = subparsers.add_parser("logout", help="Clear stored credentials")
    logout_parser.set_defaults(func=logout)

    # Config command
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    config_parser.set_defaults(func=config)

    # Parse arguments
    args = parser.parse_args()

    # Handle flags first
    if args.logout:
        logout()
        return

    if args.config:
        config()
        return

    # Handle subcommands
    if args.command:
        args.func()
        return

    # No command or flag provided
    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
