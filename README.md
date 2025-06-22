<div align="center">
  <img width="300" height="300" src="https://i.imgur.com/KISXGnH.png" alt="Chisel CLI logo" /> 
	<h1>chisel</h1>
</div>

**TL;DR:** A CLI tool for developing AMD GPU kernels on DigitalOcean. Spins up GPU droplets, syncs code, runs/profiles HIP kernels with rocprof, and pulls results back locally. Zero GPU hardware required—just write, test, and profile AMD GPU code from your laptop.

### Setup

**From PyPI (recommended):**
```bash
pip install chisel-cli
```

**Development setup:**

*With uv:*
```bash
uv sync
# Note: prefix all chisel commands with 'uv run'
uv run chisel <command>  
```

*With venv:*
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

2. **Spin up GPU droplet**
	- `chisel up` - Create or reuse a GPU-accelerated droplet
	
	**Usage:**
	```bash
	chisel up
	```
	
	**What it does:**
	- Checks for existing 'chisel-dev' droplet
	- If none exists, creates a new droplet with:
	  - Size: `gpu-mi300x1-192gb` (AMD MI300X GPU)
	  - Image: AMD AI/ML Ready (ROCm pre-installed)
	  - Region: ATL1 (where AMD GPUs are available)
	  - SSH keys: Automatically injects all keys from your DO account
	- Waits for droplet to be ready and SSH accessible
	- Displays connection information

3. **List droplets**
	- `chisel list` - Show all chisel droplets
	
	**Usage:**
	```bash
	chisel list
	```
	
	**Shows:**
	- All active chisel droplets with their status
	- IP addresses for SSH access
	- Region, size, and creation time
	- Current active droplet from local state

4. **Sync code**
	- `chisel sync` - Push local files to droplet (only changed files)
	
	**Usage:**
	```bash
	# Sync a file to /root/chisel/ (default)
	chisel sync simple-mm.cpp
	
	# Sync directory contents
	chisel sync ./src/
	
	# Sync to custom destination
	chisel sync myfile.cpp --dest /tmp/
	```
	
	**What it does:**
	- Uses rsync for efficient file transfer
	- Only transfers changed files
	- Shows progress during transfer
	- Creates destination directory if needed

5. **Run commands**
	- `chisel run` - Execute commands remotely with live output streaming
	
	**Usage:**
	```bash
	# Compile and run HIP kernel
	chisel run "hipcc /root/chisel/simple-mm.cpp -o /tmp/test && /tmp/test"
	
	# Run multiple commands
	chisel run "make && ./bench.sh"
	
	# Check GPU status
	chisel run "rocm-smi"
	```
	
	**What it does:**
	- SSH exec with real-time output streaming
	- Returns actual exit codes
	- Handles both stdout and stderr
	- Works with interactive commands

6. **Profile kernels**
	- `chisel profile` - Profile commands or source files with rocprof and pull results locally
	
	**Usage:**
	```bash
	# Profile source file (auto-compiles with hipcc)
	chisel profile simple-mm.cpp
	
	# Profile with custom compiler flags
	chisel profile kernel.cpp --args "-O3 -DNDEBUG"
	
	# Profile existing binary
	chisel profile "/tmp/my-binary"
	
	# Profile any command
	chisel profile "ls -la"
	
	# Custom trace options and output directory
	chisel profile simple-mm.cpp --trace hip,hsa,roctx --out ./results
	
	# Auto-open results in Perfetto
	chisel profile kernel.cpp --open
	```
	
	**What it does:**
	- Auto-syncs source files to droplet if needed
	- Compiles source files with hipcc
	- Runs rocprof with specified trace options
	- Downloads profile results to local directory
	- Displays summary of top kernel hotspots
	- Optionally opens Chrome trace in Perfetto

7. **Pull artifacts**
	- `chisel pull` - Pull files or directories from the droplet to local machine
	
	**Usage:**
	```bash
	# Pull a specific file to current directory
	chisel pull /tmp/results.txt
	
	# Pull to a specific local path
	chisel pull /tmp/logs/ --local ./local_logs/
	
	# Pull compiled binary
	chisel pull /root/chisel/my-binary --local ./my-binary
	```
	
	**What it does:**
	- Pulls any file or directory from the remote droplet
	- Automatically handles files vs directories
	- Creates local destination directories as needed
	- Shows progress and confirmation

8. **Clean up old droplets**
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

9. **Stop billing**
	- `chisel down` - Destroy the droplet to stop charges
	
	**Usage:**
	```bash
	chisel down
	```
	
	**What it does:**
	- Prompts for confirmation before destroying
	- Completely removes the droplet (not just powered off)
	- Clears local state cache
	- Stops all billing immediately

**Cost Management:**
- Cost warnings appear when droplets run longer than 12 hours
- Estimated at $1.99/hour for AMD MI300X GPU droplets
- Use `chisel sweep` to clean up old droplets automatically
- Droplets self-destruct after 15 minutes of inactivity

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
# Recreate any existing droplets to include your new key
chisel down
chisel up

# Now these commands will work from your droplet
chisel sync myfile.cpp
chisel run "hipcc myfile.cpp && ./a.out"
chisel profile kernel.cpp
```

### Architecture pieces

- AMD's droplets ship with ROCm pre-installed, so driver stack is available instantly
- Use DigitalOcean's `pydo` to create / destroy nodes

1. **Python CLI skeleton** – `typer` or `argparse`; single `main.py`.
2. **DigitalOcean wrapper**

   ```python
   import pydo
   client = pydo.Client(token=token)
   client.droplets.create(size='gpu-mi300x1-192gb', image='gpu-amd-base', ...)
   ```
3. **SSH/rsync layer** – Use `paramiko` for exec + `rsync`/`scp` shell out (simplest); later swap to async libraries if perf matters.
4. **Cloud-init script** – idempotent bash that:

   * `apt update && apt install -y build-essential rocblas-dev …`
   * Adds a `/etc/profile.d/chisel.sh` that exports ROCm paths.
5. **State cache** – tiny JSON in `~/.cache/chisel/state.json` mapping project → droplet ID & IP so repeated `chisel run` skips the spin-up step.
6. **Credential handling** – ENV override `CHISEL_DO_TOKEN` > config file, because CI.
7. **Cost guardrails** – warn if droplet has been alive >N hours; `chisel sweep` to nuke zombies.


### TODO list

|TODO | Deliverable                                                 |
| --- | ----------------------------------------------------------- |
| [x] |	`chisel configure` - DO token validation, config storage     |
| [x] | `chisel up` / `down` / `list`, cloud-init basics, state cache. |
| [x] | `sync` + `run` (blocking), colored log streaming.           |
| [x] | `profile` - rocprof integration, result parsing, Perfetto.  |
| [x] | Artifact `pull`, graceful ^C handling, rudimentary tests.   |
| [x] | Cost warnings, README with install script, publish to PyPI. |

### Future

- concurrent runs (non-blocking sync and run) 














