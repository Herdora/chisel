<div align="center">
  <img width="300" height="300" src="https://i.imgur.com/KISXGnH.png" alt="Chisel CLI logo" /> 
	<h1>chisel</h1>
</div>

**TL;DR:** A CLI tool for developing and profiling GPU kernels locally. Spins up GPU droplets, syncs code, runs/profiles kernels, and pulls results back locally. Zero GPU hardware required—just write, test, and profile GPU code from your laptop. Supports both AMD MI300X and NVIDIA H100 GPUs with concurrent multi-droplet workflows and context switching for seamless GPU target switching.

## Quick Start

Get up and running in 2 minutes:

```bash
# 1. Install chisel
pip install chisel-cli

# 2. Configure with your DigitalOcean API token
chisel configure

# 3. Set your target GPU and spin up droplet
chisel switch nvidia-h100      # or amd-mi300x
chisel up

# 4. Write, sync, and run GPU code
echo 'int main() { return 0; }' > kernel.cu
chisel sync kernel.cu
chisel run "nvcc kernel.cu && ./a.out"

# 5. Clean up when done
chisel down
```

**That's it!** 🚀 No GPU hardware needed—develop and test GPU kernels from any machine.

> **Need a DigitalOcean API token?** Get one [here](https://amd.digitalocean.com/account/api/tokens) (requires read/write access).

### Setup

**Quick start:**

```bash
pip install chisel-cli
chisel --help
```

**Dev setup:**

_With uv:_

```bash
uv sync
# Note: prefix all chisel commands with 'uv run'
uv run chisel <command>
```

_With venv:_

```bash
pip install -e .
```

### How to use

1. **Configuration**

   - `chisel configure` - Set up your DigitalOcean API credentials

   **Usage:**

   ```bash
   # Interactive configuration (recommended for first-time setup)
   chisel configure

   # Non-interactive with token
   chisel configure --token YOUR_DIGITALOCEAN_TOKEN
   ```

   **Getting your API token:**

   1. Visit [DigitalOcean API Tokens](https://amd.digitalocean.com/account/api/tokens)
   2. Click "Generate New Token"
   3. Give it a name (e.g., "chisel-cli") and ensure it has **read and write** access
   4. Copy the token immediately (you won't be able to see it again)

   For detailed instructions, see the [official guide](https://docs.digitalocean.com/reference/api/create-personal-access-token/).

2. **GPU Context Management**

   - `chisel switch` - Set active GPU context for seamless workflows
   - `chisel context` - Show current active context

   **Usage:**

   ```bash
   # Set active GPU context
   chisel switch nvidia-h100
   chisel switch amd-mi300x

   # Check current context
   chisel context
   ```

3. **Spin up GPU droplet**

   - `chisel up` - Create or reuse a GPU-accelerated droplet

   **Usage:**

   ```bash
   # Using active context (recommended)
   chisel switch nvidia-h100
   chisel up

   # Or specify explicitly
   chisel up --gpu-type amd-mi300x
   ```

   **What it does:**

   - Checks for existing GPU-specific droplet (`chisel-dev-amd` or `chisel-dev-nvidia`)
   - If none exists, creates a new droplet with:
     - **AMD MI300X**: Size `gpu-mi300x1-192gb`, AMD AI/ML Ready image (ROCm pre-installed), ATL1 region
     - **NVIDIA H100**: Size `gpu-h100x1-80gb`, NVIDIA AI/ML Ready image (CUDA pre-installed), NYC2 region
     - SSH keys: Automatically injects all keys from your DO account
   - Waits for droplet to be ready and SSH accessible
   - Displays connection information

4. **List droplets**

   - `chisel list` - Show all chisel droplets

   **Usage:**

   ```bash
   chisel list
   ```

   **Shows:**

   - All active chisel droplets with their GPU type and status
   - IP addresses for SSH access
   - Region, size, cost per hour, and creation time
   - Tracked droplets from local state

5. **Sync code**

   - `chisel sync` - Push local files to droplet (only changed files)

   **Usage:**

   ```bash
   # Using active context (recommended)
   chisel switch amd-mi300x
   chisel sync simple-mm.cpp

   # Sync directory contents
   chisel sync ./src/

   # Sync to custom destination
   chisel sync myfile.cpp --dest /tmp/

   # Override context with explicit GPU type
   chisel sync simple-mm.cpp --gpu-type nvidia-h100
   ```

   **What it does:**

   - Uses rsync for efficient file transfer
   - Only transfers changed files
   - Shows progress during transfer
   - Creates destination directory if needed

6. **Run commands**

   - `chisel run` - Execute commands remotely with live output streaming

   **Usage:**

   ```bash
   # Using active context (recommended)
   chisel switch amd-mi300x
   chisel run "hipcc /root/chisel/simple-mm.cpp -o /tmp/test && /tmp/test"
   chisel run "rocm-smi"

   # Switch context for different GPU
   chisel switch nvidia-h100
   chisel run "nvcc /root/chisel/simple-mm.cu -o /tmp/test && /tmp/test"
   chisel run "nvidia-smi"

   # Override context with explicit GPU type
   chisel run "make && ./bench.sh" --gpu-type amd-mi300x
   ```

   **What it does:**

   - SSH exec with real-time output streaming
   - Returns actual exit codes
   - Handles both stdout and stderr
   - Works with interactive commands

7. **Profile kernels**

   - `chisel profile` - Profile commands or source files with rocprof and pull results locally

   **Usage:**

   ```bash
   # Using active context (recommended)
   chisel switch amd-mi300x
   chisel profile simple-mm.cpp
   chisel profile kernel.cpp --args "-O3 -DNDEBUG"
   chisel profile simple-mm.cpp --trace hip,hsa,roctx --out ./results

   # Switch context for NVIDIA profiling
   chisel switch nvidia-h100
   chisel profile simple-mm.cu
   chisel profile "/tmp/my-binary"

   # Override context with explicit GPU type
   chisel profile kernel.cpp --gpu-type amd-mi300x --open
   ```

   **What it does:**

   - Auto-syncs source files to droplet if needed
   - Compiles source files with hipcc (.cpp, .hip) or nvcc (.cu) based on extension
   - Runs rocprof (AMD) with specified trace options
   - Downloads profile results to local directory
   - Displays summary of top kernel hotspots
   - Optionally opens Chrome trace in Perfetto

8. **Pull artifacts**

   - `chisel pull` - Pull files or directories from the droplet to local machine

   **Usage:**

   ```bash
   # Using active context (recommended)
   chisel switch nvidia-h100
   chisel pull /tmp/results.txt
   chisel pull /root/chisel/my-binary --local ./my-binary

   # Override context with explicit GPU type
   chisel pull /tmp/logs/ --gpu-type amd-mi300x --local ./local_logs/
   ```

   **What it does:**

   - Pulls any file or directory from the remote droplet
   - Automatically handles files vs directories
   - Creates local destination directories as needed
   - Shows progress and confirmation

9. **Clean up old droplets**

   - `chisel sweep` - Clean up droplets that have been running too long

   **Usage:**

   ```bash
   # Clean up droplets older than 6 hours (default)
   chisel sweep

   # Custom time threshold
   chisel sweep --hours 12

   # Auto-confirm without prompting
   chisel sweep --yes
   ```

   **What it does:**

   - Shows all droplets with uptime and estimated costs
   - Identifies droplets running longer than specified threshold
   - Prompts for confirmation before destroying old droplets
   - Clears local state if active droplet was destroyed

10. **Stop billing**

    - `chisel down` - Destroy the droplet to stop charges

    **Usage:**

    ```bash
    # Destroy active context droplet (recommended)
    chisel down

    # Destroy specific GPU droplet
    chisel down --gpu-type nvidia-h100
    ```

    **What it does:**

    - Prompts for confirmation before destroying
    - Completely removes the specified droplet (not just powered off)
    - Clears local state cache for that GPU type
    - Automatically clears active context if destroying the active droplet
    - Stops billing immediately for that droplet

**Context Switching Workflow:**

```bash
# Set context once, then run multiple commands
chisel switch nvidia-h100
chisel up
chisel sync kernel.cu
chisel run "nvcc kernel.cu -o test && ./test"
chisel profile kernel.cu
chisel down

# Switch to different GPU type seamlessly
chisel switch amd-mi300x
chisel up
chisel sync kernel.cpp
chisel run "hipcc kernel.cpp -o test && ./test"
```

**Cost Management:**

- Cost warnings appear when droplets run longer than 12 hours
- Estimated costs: $1.99/hour for AMD MI300X, $4.89/hour for NVIDIA H100
- Use `chisel sweep` to clean up old droplets automatically
- Droplets self-destruct after 15 minutes of inactivity
- **Multi-droplet support**: Run both AMD and NVIDIA droplets simultaneously for cross-platform development
- **Context switching**: Eliminates repetitive `--gpu-type` flags for smoother workflows

### Using chisel from DigitalOcean droplets

If you want to use chisel from one DigitalOcean droplet to manage another (e.g., from a personal GPU droplet to a chisel-dev droplet), you need to set up SSH access:

**Quick setup:**

```bash
# On your DigitalOcean droplet (or local machine)
chisel ssh-setup
```

This command will:

1. Generate an SSH key if you don't have one
2. Show you your public key in a formatted panel
3. **Automatically add it to your DigitalOcean account** via the API
4. Provide instructions for manual setup if the API call fails

**Example output:**

```
Your SSH public key (id_ed25519.pub):
╭──────────────────────────────────────────────────────────────────────────────╮
│ ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL... user@droplet                     │
╰──────────────────────────────────────────────────────────────────────────────╯

Attempting to add SSH key to your DigitalOcean account...
✓ Successfully added SSH key 'chisel-droplet-name-1234567890' to your DigitalOcean account!
Now run 'chisel down' then 'chisel up' to recreate any existing droplets with this key.
```

**After setup:**

```bash
# Set context and recreate droplet to include your new key
chisel switch amd-mi300x
chisel down
chisel up

# Now these commands will work from your droplet
chisel sync myfile.cpp
chisel run "hipcc myfile.cpp && ./a.out"
chisel profile kernel.cpp
```

### MCP Server Integration

**Claude Desktop Integration:**

Chisel includes an MCP (Model Context Protocol) server that lets you manage GPU droplets directly through Claude Desktop.

**Setup:**

1. **Install Claude Desktop** from [claude.ai/download](https://claude.ai/download)

2. **Configure the MCP server** by editing `~/Library/Application Support/Claude/claude_desktop_config.json`:

   ```json
   {
     "mcpServers": {
       "chisel": {
         "command": "/path/to/your/chisel/.venv/bin/python",
         "args": ["/path/to/your/chisel/mcp_server.py"]
       }
     }
   }
   ```

3. **Restart Claude Desktop**

**Available tools:**

- `configure` - Set up DigitalOcean API token
- `switch` - Set active GPU context
- `context` - Show current active context
- `up` - Create or reuse a GPU droplet (uses active context or requires --gpu-type)
- `down` - Destroy the current droplet (uses active context or requires --gpu-type)
- `status` - Check droplet status
- `profile` - Profile HIP/CUDA files or commands with rocprof
- `sync` - Sync files to the droplet
- `run` - Execute commands on the droplet
- `pull` - Pull files from the droplet

**Usage examples:**

- "Configure my chisel setup with token xyz123"
- "Switch to AMD context and start a droplet for development"
- "Switch to H100 context and start a droplet for CUDA development"
- "Profile my matrix_multiply.hip file on the active droplet"
- "Run nvidia-smi on the H100 droplet"
- "Sync my_kernel.cu to the active droplet"
- "Pull the results.csv file from the active droplet"

### Architecture pieces

- AMD's droplets ship with ROCm pre-installed, NVIDIA's with CUDA pre-installed
- Use DigitalOcean's `pydo` to create / destroy nodes

1. **Python CLI skeleton** – `typer` or `argparse`; single `main.py`.
2. **DigitalOcean wrapper**

   ```python
   import pydo
   client = pydo.Client(token=token)
   # AMD GPU droplet
   client.droplets.create(size='gpu-mi300x1-192gb', image='gpu-amd-base', region='atl1', ...)
   # NVIDIA GPU droplet
   client.droplets.create(size='gpu-h100x1-80gb', image='gpu-h100x1-base', region='nyc2', ...)
   ```

3. **SSH/rsync layer** – Use `paramiko` for exec + `rsync`/`scp` shell out (simplest); later swap to async libraries if perf matters.
4. **Cloud-init script** – idempotent bash that:

   - `apt update && apt install -y build-essential rocblas-dev …`
   - Adds a `/etc/profile.d/chisel.sh` that exports ROCm paths.

5. **State cache** – tiny JSON in `~/.cache/chisel/state.json` mapping GPU types → droplet ID & IP so repeated commands target the correct droplet.
6. **Credential handling** – ENV override `CHISEL_DO_TOKEN` > config file, because CI.
7. **Cost guardrails** – warn if droplet has been alive >N hours; `chisel sweep` to nuke zombies.

### Future

- [x] MCP server
- [x] support NVIDIA H100 (basic CUDA compilation)
- [x] concurrent multi-droplet support
- [ ] concurrent runs (non-blocking sync and run)
- [ ] NVIDIA profiling (nsight-compute, nsight-systems)
- [ ] support other cloud backends
- [ ] add Grafana support
