#!/usr/bin/env python3
"""
Test script for the new progress tracking system in the cached files API.
This script tests the upload-with-progress endpoint to ensure it provides
real-time feedback during large file uploads.
"""

import asyncio
import aiohttp
import tempfile
import os
import time
from typing import AsyncGenerator


async def create_test_file(size_mb: int = 10) -> str:
    """Create a test file of specified size in MB."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
    temp_file_path = temp_file.name

    # Write random data to the file
    chunk_size = 1024 * 1024  # 1MB chunks
    remaining = size_mb * 1024 * 1024

    while remaining > 0:
        chunk_size = min(chunk_size, remaining)
        temp_file.write(os.urandom(chunk_size))
        remaining -= chunk_size

    temp_file.close()
    print(f"✅ Created test file: {temp_file_path} ({size_mb}MB)")
    return temp_file_path


async def test_progress_streaming(
    session: aiohttp.ClientSession, file_path: str, auth_token: str = None
) -> None:
    """Test the progress streaming endpoint."""
    url = "http://localhost:8000/api/v1/cached-files/upload-with-progress"

    # Prepare the file upload
    with open(file_path, "rb") as f:
        data = aiohttp.FormData()
        data.add_field("file", f, filename="test_large_file.bin")

        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        print(
            f"🚀 Testing progress streaming for file: {os.path.getsize(file_path) / (1024 * 1024):.1f}MB"
        )
        print("📊 Progress events:")

        async with session.post(url, data=data, headers=headers) as response:
            if response.status != 200:
                print(f"❌ Upload failed with status: {response.status}")
                print(f"Response: {await response.text()}")
                return

            # Read the SSE stream
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        import json

                        event_data = json.loads(line[6:])  # Remove 'data: ' prefix

                        # Format the progress event
                        event_type = event_data.get("type", "unknown")
                        message = event_data.get("message", "")
                        progress = event_data.get("progress", 0)
                        stage = event_data.get("stage", "unknown")

                        # Create a visual progress bar
                        bar_length = 30
                        filled_length = int(bar_length * progress / 100)
                        bar = "█" * filled_length + "░" * (bar_length - filled_length)

                        print(
                            f"📊 [{event_type.upper():<15}] [{bar}] {progress:>6.1f}% | {stage:<15} | {message}"
                        )

                        if event_type == "complete":
                            print("✅ Upload completed successfully!")
                            break
                        elif event_type == "error":
                            print(f"❌ Upload failed: {message}")
                            break

                    except json.JSONDecodeError as e:
                        print(f"⚠️  Failed to parse event: {line} - {e}")
                        continue


async def main():
    """Main test function."""
    print("🧪 Testing Large File Upload Progress Tracking")
    print("=" * 60)

    # Create a test file (10MB for testing)
    test_file_path = await create_test_file(10)

    try:
        async with aiohttp.ClientSession() as session:
            # Test the progress streaming
            await test_progress_streaming(session, test_file_path)

    except Exception as e:
        print(f"❌ Test failed: {e}")
    finally:
        # Clean up test file
        try:
            os.unlink(test_file_path)
            print(f"🧹 Cleaned up test file: {test_file_path}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup test file: {e}")

    print("\n" + "=" * 60)
    print("🏁 Test completed!")


if __name__ == "__main__":
    # Run the test
    asyncio.run(main())
