# Enhanced Chisel Profile MCP Functionality

This document describes the enhanced chisel profile functionality added to the MCP server that provides automatic setup and management.

## Overview

The enhanced profile functionality automatically ensures that a GPU droplet is configured and ready before attempting to profile code. This makes the MCP server more autonomous and user-friendly for Claude.

## Key Features

### 1. Automatic Environment Setup
- **Configuration Check**: Verifies DigitalOcean API token is configured
- **Droplet Detection**: Checks if an active droplet exists
- **Auto-Creation**: Creates a new droplet if needed (when `auto_setup=True`)
- **State Management**: Maintains droplet state across operations

### 2. Enhanced Error Handling
- **Clear Status Messages**: Provides detailed feedback on each step
- **Recovery Guidance**: Suggests next steps when errors occur
- **Graceful Degradation**: Falls back appropriately when auto-setup is disabled

### 3. Intelligent Workflow
- **State Verification**: Checks both local state and remote droplet status
- **Stale State Cleanup**: Removes outdated state when droplets no longer exist
- **Connection Recovery**: Reconnects to existing droplets when possible

## Enhanced Tools

All these tools now support automatic setup:

### `profile(file_or_command, trace="hip,hsa", analyze=True, auto_setup=True)`
- Profiles HIP files or commands with full auto-setup
- Compiles source files automatically
- Provides detailed profiling analysis

### `sync(source, destination=None, auto_setup=True)`
- Syncs files to droplet with auto-setup
- Creates droplet if needed before syncing

### `run(command, auto_setup=True)`
- Executes commands on droplet with auto-setup
- Ensures droplet is ready before execution

### `pull(remote_path, local_path=None, auto_setup=True)`
- Pulls files from droplet with auto-setup
- Creates droplet if needed before pulling

## Usage Examples

### Basic Profiling (with auto-setup)
```python
# Claude can simply call this and everything will be set up automatically
result = await profile("my_kernel.hip")
```

### Profiling without auto-setup
```python
# If you want to check status without creating droplets
result = await profile("my_kernel.hip", auto_setup=False)
```

### Complex Workflow
```python
# All these will auto-setup if needed
await sync("my_project/")           # Syncs project files
await run("make")                   # Builds project  
await profile("./my_binary")        # Profiles the binary
await pull("/tmp/results.csv")      # Gets results
```

## Implementation Details

### `ensure_droplet_ready()` Helper
This function encapsulates the droplet readiness logic:

1. **Configuration Check**: Verifies API token exists
2. **State Validation**: Checks local droplet state
3. **Remote Verification**: Confirms droplet exists and is active on DigitalOcean
4. **State Cleanup**: Removes stale state for non-existent droplets
5. **Discovery**: Finds existing droplets that can be reused

### Auto-Setup Flow
When `auto_setup=True` and no ready droplet is found:

1. **Status Check**: Call `ensure_droplet_ready()`
2. **Configuration Validation**: Ensure API token is configured
3. **Droplet Creation**: Call the `up()` tool to create droplet
4. **Verification**: Re-check that droplet is now ready
5. **Operation**: Proceed with the requested operation

### Error Recovery
The system provides clear guidance for common issues:
- **No API Token**: Directs user to run `configure` tool
- **Droplet Creation Failed**: Shows specific error and suggests fixes
- **SSH Connection Issues**: Provides troubleshooting steps
- **Stale State**: Automatically cleans up and suggests recreation

## Benefits for Claude

1. **Autonomous Operation**: Claude can profile files without manual setup steps
2. **Reduced Friction**: Single command handles the entire workflow
3. **Intelligent Behavior**: System adapts to current environment state
4. **Clear Feedback**: Detailed status updates help Claude understand progress
5. **Consistent Experience**: Same auto-setup behavior across all tools

## Testing

Use the provided test script to verify functionality:

```bash
python test_enhanced_profile.py
```

This will test:
- Droplet readiness detection
- Profile with auto-setup disabled
- Profile with auto-setup enabled (if needed)

## File Structure

```
chisel/
├── mcp/
│   └── mcp_server.py          # Enhanced MCP server with auto-setup
├── test_kernel.hip            # Sample HIP kernel for testing
├── test_enhanced_profile.py   # Test script for new functionality
└── ENHANCED_MCP_PROFILE.md    # This documentation
```

## Configuration

The enhanced functionality uses the same configuration as the standard chisel CLI:
- DigitalOcean API token via environment or config file
- State management in `~/.cache/chisel/state.json`
- Same droplet naming and management conventions