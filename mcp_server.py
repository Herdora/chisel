#!/usr/bin/env python3
"""MCP Server for Chisel - DigitalOcean GPU droplet management."""

import io
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Import chisel modules
from chisel.config import Config
from chisel.do_client import DOClient
from chisel.droplet import DropletManager

# Initialize FastMCP server
mcp = FastMCP("chisel")


class SuppressOutput:
    """Context manager to suppress stdout and stderr."""

    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr


@mcp.tool()
async def configure(token: Optional[str] = None) -> str:
    """Configure Chisel with your DigitalOcean API token.

    Args:
        token: Optional DigitalOcean API token. If not provided, will use existing token if available.
    """
    try:
        config = Config()

        # Check if token already exists
        existing_token = config.token

        if token:
            # Token provided via parameter
            api_token = token
        elif existing_token:
            # Token exists in config/env
            return "✓ DigitalOcean API token is already configured. To update, provide a new token parameter."
        else:
            # No token found
            return """❌ No DigitalOcean API token found.

To configure Chisel:
1. Go to: https://amd.digitalocean.com/account/api/tokens
2. Generate a new token with read and write access
3. Call this tool again with the token parameter

Example: configure with token="your_token_here" """

        # Validate token
        try:
            with SuppressOutput():
                do_client = DOClient(api_token)
                valid, account_info = do_client.validate_token()

            if valid and account_info:
                # Save token to config
                config.token = api_token

                # Get account info
                account_data = account_info.get("account", {})
                email = account_data.get("email", "N/A")
                status = account_data.get("status", "N/A")
                droplet_limit = account_data.get("droplet_limit", "N/A")

                # Get balance info
                with SuppressOutput():
                    balance_info = do_client.get_balance()
                balance_text = ""
                if balance_info:
                    balance_data = balance_info.get("balance", {})
                    account_balance = balance_data.get("account_balance", "N/A")
                    mtd_usage = balance_data.get("month_to_date_usage", "N/A")
                    balance_text = f"\nAccount Balance: ${account_balance}\nMonth-to-date Usage: ${mtd_usage}"

                return f"""✅ Token validated and saved successfully!

Account Information:
- Email: {email}
- Status: {status}
- Droplet Limit: {droplet_limit}{balance_text}

Configuration saved to: {config.config_file}

✅ Chisel is now configured and ready to use!"""

            else:
                return "❌ Invalid API token. Please check your token and try again."

        except Exception as e:
            return f"❌ Error validating token: {e}\n\nPlease ensure you have a valid DigitalOcean API token with read and write permissions."

    except Exception as e:
        return f"❌ Error during configuration: {e}"


@mcp.tool()
async def up() -> str:
    """Create or reuse a GPU droplet for development."""
    try:
        config = Config()

        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""

        # Initialize clients and suppress output
        with SuppressOutput():
            do_client = DOClient(config.token)
            droplet_manager = DropletManager(do_client)

            # Create or find droplet
            droplet = droplet_manager.up()

        # Format response
        name = droplet["name"]
        ip = droplet.get("ip", "N/A")
        region = droplet["region"]["slug"]
        size = droplet["size"]["slug"]

        return f"""✅ Droplet is ready!

Details:
- Name: {name}
- IP: {ip}
- Region: {region}
- Size: {size}

SSH: ssh root@{ip}"""

    except Exception as e:
        return f"❌ Error creating droplet: {e}"


@mcp.tool()
async def down() -> str:
    """Destroy the current droplet to stop billing."""
    try:
        config = Config()

        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""

        # Initialize clients and suppress output
        with SuppressOutput():
            do_client = DOClient(config.token)
            droplet_manager = DropletManager(do_client)

            # Note: In the original CLI, this prompts for confirmation
            # For MCP, we'll proceed directly since Claude will likely ask for confirmation
            success = droplet_manager.down()

        if success:
            return "✅ Droplet destroyed successfully. Billing has stopped."
        else:
            return "❌ Failed to destroy droplet or no droplet found to destroy."

    except Exception as e:
        return f"❌ Error destroying droplet: {e}"


@mcp.tool()
async def status() -> str:
    """Get status of current chisel droplets."""
    try:
        config = Config()

        # Check if configured
        if not config.token:
            return """❌ No API token configured.

Run the 'configure' tool first to set up your DigitalOcean API token."""

        # Initialize clients and suppress output
        with SuppressOutput():
            do_client = DOClient(config.token)
            droplet_manager = DropletManager(do_client)

            # Get droplets
            droplets = droplet_manager.list_droplets()

        if not droplets:
            return "ℹ️ No chisel droplets found"

        # Format droplet info
        result = f"📋 Found {len(droplets)} chisel droplet(s):\n\n"

        for droplet in droplets:
            name = droplet["name"]
            ip = droplet.get("ip", "N/A")
            status_val = droplet["status"]
            region = droplet["region"]["slug"]
            size = droplet["size"]["slug"]
            created = droplet["created_at"][:19].replace("T", " ")

            result += f"• {name}\n"
            result += f"  IP: {ip}\n"
            result += f"  Status: {status_val}\n"
            result += f"  Region: {region}\n"
            result += f"  Size: {size}\n"
            result += f"  Created: {created}\n\n"

        # Show state info
        with SuppressOutput():
            state_info = droplet_manager.state.get_droplet_info()
        if state_info:
            result += f"🎯 Active droplet: {state_info['name']} ({state_info['ip']})"

        return result.strip()

    except Exception as e:
        return f"❌ Error getting status: {e}"


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
