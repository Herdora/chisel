<div align="center">
  <img width="300" height="300" src="https://github.com/user-attachments/assets/75ae4b61-b6a7-40a4-a46b-6b35baba7404" alt="Chisel CLI logo" /> 
	<h5>chisel</h5>
</div>

- AMD's droplets ship with ROCm pre-installed, so driver stack is available instantly
- Use DigitalOcean's `pydo` to create / destroy nodes

### CLI workflow

- chisel's only interactions are `auth / up / sync / run / profile / pull / down`


1. configure
	- `chisel auth --token $DO_TOKEN`
	- get your DO token here: <TODO>
	- this stores token in ~/.config/chisel/config.toml; verify quota.

2. spin-up
	- `chisel up`
	- check for existing 'chisel-dev' droplet
	- if none, call DO API to create `gpu-mi300x1-192gb`, inject user SSH key, boot

3. sync code
	- `chisel sync ./kernels`
	- pushes your local kernel source tree to the droplet (only the files that changed)

4. run test
	- e.g. `chisel run make && ./bench.sh`
	- ssh exec the given command, stream stdoud/sterr, return exit code; etc.

5. grab artifacts
	- `chisel pull out/`
	- SCP back to `./out` locally

6. profile your kernel
	- `chisel profile <cmd> [--trace hip,hsa,roctx] [--out DIR] [--open]`
	- rocprof -d /tmp/chisel_profile --hip-trace --hsa-trace --stats ./bench.sh → generates results.csv, results.stats.csv, and a Chrome-trace JSON.
	- `tar -czf /tmp/chisel_profile.tgz -C /tmp chisel_profile`
	- copies the archive back to local
	- read results sort by total time and prints top N hottest kernels
	- or you can run `chisel profile --open`: same as above plus auto launch w/ perfetto

7. stop billing
	- `chisel down`
	- power-off the droplet

Miscallenous:
	- If no requests in 15 minutes, droplet will destroy itself

### Architecture pieces

1. **Python CLI skeleton** – `typer` or `argparse`; single `main.py`.
2. **DigitalOcean wrapper**

   ```python
   import pydo
   client = pydo.Client(token=token)
   client.droplets.create(size='gpu-mi300x1-192gb', image='ubuntu-22-04-x64', ...)
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
| [ ] | `chisel auth` + DO token validation, skeleton Typer CLI.    |
| [ ] | `chisel up` / `down`, cloud-init basics, state cache.       |
| [ ] | `sync` + `run` (blocking), colored log streaming.           |
| [ ] | `profile` milestone	                                    |
| [ ] | Artifact `pull`, graceful ^C handling, rudimentary tests.   |
| [ ] | Cost warnings, README with install script, publish to PyPI. |














