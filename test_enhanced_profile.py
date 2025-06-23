#!/usr/bin/env python3
"""Test script to demonstrate the enhanced chisel profile functionality."""

import asyncio
import sys
import os

# Add the mcp directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp'))

from mcp_server import ensure_droplet_ready, profile

async def test_ensure_droplet_ready():
    """Test the ensure_droplet_ready function."""
    print("Testing ensure_droplet_ready()...")
    ready, message = await ensure_droplet_ready()
    print(f"Ready: {ready}")
    print(f"Message: {message}")
    return ready

async def test_profile_with_auto_setup():
    """Test the profile function with auto_setup enabled."""
    print("\nTesting profile with auto_setup=True...")
    
    # Test with the test kernel we created
    result = await profile(
        file_or_command="test_kernel.hip",
        trace="hip,hsa",
        analyze=True,
        auto_setup=True
    )
    
    print("Profile Result:")
    print(result)

async def test_profile_without_auto_setup():
    """Test the profile function with auto_setup disabled."""
    print("\nTesting profile with auto_setup=False...")
    
    result = await profile(
        file_or_command="test_kernel.hip",
        trace="hip,hsa", 
        analyze=True,
        auto_setup=False
    )
    
    print("Profile Result:")
    print(result)

async def main():
    """Main test function."""
    print("=== Enhanced Chisel Profile Functionality Test ===\n")
    
    # Test 1: Check droplet readiness
    ready = await test_ensure_droplet_ready()
    
    # Test 2: Profile without auto-setup (should show current state)
    await test_profile_without_auto_setup()
    
    # Test 3: If not ready, test with auto-setup
    if not ready:
        print("\nDroplet not ready, testing auto-setup functionality...")
        await test_profile_with_auto_setup()
    else:
        print("\nDroplet already ready, auto-setup not needed!")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(main())