# Chisel MCP Server

MCP (Model Context Protocol) server for the Chisel GPU profiling tool. This server enables LLMs to configure and use Chisel for profiling GPU kernels on cloud infrastructure.

## Installation

```bash
# Clone and install
cd chisel-mcp
uv sync

# Or with pip
pip install .
```

## Usage

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "chisel": {
      "command": "uv",
      "args": ["run", "chisel-mcp"]
    }
  }
}
```

### Available Tools

1. **configure** - Set up DigitalOcean API token
   ```
   configure(token: str) -> str
   ```

2. **profile_nvidia** - Profile kernels on NVIDIA GPUs
   ```
   profile_nvidia(target: str, gpu_type: Optional[str] = None) -> Dict
   ```
   - `target`: File to compile (e.g., kernel.cu) or command to run
   - `gpu_type`: "h100" (default) or "l40s"

3. **profile_amd** - Profile kernels on AMD MI300X
   ```
   profile_amd(target: str, pmc: Optional[str] = None) -> Dict
   ```
   - `target`: File to compile (e.g., kernel.cpp) or command to run
   - `pmc`: Performance counters (e.g., "GRBM_GUI_ACTIVE,SQ_WAVES")

## Example Usage

```python
# Configure API token
configure("your-digitalocean-api-token")

# Profile CUDA kernel on NVIDIA H100
profile_nvidia("kernel.cu")

# Profile on NVIDIA L40S
profile_nvidia("kernel.cu", gpu_type="l40s")

# Profile HIP kernel on AMD MI300X
profile_amd("kernel.cpp")

# Profile with performance counters
profile_amd("kernel.cpp", pmc="GRBM_GUI_ACTIVE,SQ_WAVES")
```

## Requirements

- Python >= 3.10
- Chisel CLI installed
- DigitalOcean API token