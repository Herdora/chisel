# Chisel Monorepo

This is the monorepo for the Chisel CLI tool, orchestrator, and GPU worker.

## Setup

1. Install dependencies: `npm install`
2. Build: `npm run build`
3. Run dev: `npm run dev`
4. Lint: `npm run lint`

## Testing the CLI Locally

### Prerequisites
- Node.js installed (version 18 or higher)
- npm or yarn package manager

### Quick Start
1. **Install dependencies** (if not done already):
   ```bash
   npm install
   ```

2. **Build the CLI**:
   ```bash
   npm run build
   ```

3. **Set up PATH** (for macOS with Homebrew):
   ```bash
   export PATH="/opt/homebrew/bin:$PATH"
   ```

4. **Test the CLI commands**:
   ```bash
   # Test help
   node apps/cli/dist/index.js --help
   
   # Test up command (starts orchestrator locally)
   node apps/cli/dist/index.js up
   
   # Test profiler commands
   node apps/cli/dist/index.js ncu --help
   node apps/cli/dist/index.js torchprof --help
   ```

### Development Mode
For development with auto-rebuild:
```bash
# In one terminal - watch and rebuild
npm run dev

# In another terminal - test CLI (with PATH set)
export PATH="/opt/homebrew/bin:$PATH"
node apps/cli/dist/index.js up
```

### Expected Output
When you run `chisel up`, you should see:
```
🚀 Bootstrapping Chisel orchestrator locally...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 Installing dependencies...
✅ Dependencies installed successfully
🔨 Building orchestrator...
✅ Build completed successfully
🚀 Starting orchestrator...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Orchestrator is starting up...
📡 Server will be available at: http://localhost:3001
💡 Press Ctrl+C to stop the server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 Starting Chisel orchestrator...
🔧 Attempting to start server...
✅ Orchestrator running on http://localhost:3001
📡 Ready to accept job requests
💡 Press Ctrl+C to stop the server
```

### Troubleshooting
- **"command not found"**: Make sure you're running from the project root and have set the PATH
- **Build errors**: Run `npm run build` first
- **Port conflicts**: The orchestrator runs on port 3001 by default
- **Node.js not found**: Set PATH with `export PATH="/opt/homebrew/bin:$PATH"`

### One-liner for testing
```bash
export PATH="/opt/homebrew/bin:$PATH" && node apps/cli/dist/index.js up
```

See design doc for more details.

## Bootstrapping the Orchestrator
Run `chisel up` to start the orchestrator locally (default), or `chisel up --cloud` to deploy to DigitalOcean. For cloud deployment, ensure `doctl` is installed and authenticated. 