// apps/cli/src/commands/up.ts
import { execa } from 'execa';
import chalk from 'chalk';

async function getSshKeyId() {
  const { stdout } = await execa('doctl', ['compute', 'ssh-key', 'list', '--format', 'ID,Name']);
  const lines = stdout.trim().split('\n').slice(1); // Skip header
  if (lines.length === 0) {
    throw new Error('No SSH keys found in DigitalOcean. Add one via `doctl compute ssh-key create` or the dashboard.');
  }
  const firstKeyId = lines[0].split(/\s+/)[0]; // Get ID from first column
  console.log(`Using SSH key: ${lines[0]}`);
  return firstKeyId;
}

export async function bootstrapOrchestrator() {
  console.log('Bootstrapping orchestrator on DigitalOcean...');

  // Step 1: Create droplet (assumes doctl is installed and authenticated)
  const dropletName = 'chisel-orch-001';
  const sshKeyId = await getSshKeyId();
  const { stdout: createOutput } = await execa('doctl', [
    'compute', 'droplet', 'create', dropletName,
    '--region', 'nyc3',
    '--image', 'ubuntu-22-04-x64',
    '--size', 's-2vcpu-4gb',
    '--ssh-keys', sshKeyId,
    '--wait'
  ]);
  console.log('Droplet created:', createOutput);

  // Extract public IP from doctl output or query it
  const { stdout: ipOutput } = await execa('doctl', ['compute', 'droplet', 'get', dropletName, '--format', 'PublicIPv4']);
  const ip = ipOutput.trim().split('\n')[1]; // Skip header
  console.log('Droplet IP:', ip);

  // Step 2: Upload repo (using tar and scp for simplicity)
  await execa('tar', ['czf', 'chisel.tar.gz', '.']);
  await execa('scp', ['chisel.tar.gz', `root@${ip}:/root/`]);

  // Step 3: SSH to install Node, unpack, build, and start
  const sshCommands = `
    console.log('Installing Node.js...');
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -;
    apt-get install -y nodejs;
    tar xzf chisel.tar.gz -C /root/chisel/;
    cd /root/chisel/apps/orchestrator;
    npm install --omit=dev;
    npm run build;
    nohup node dist/index.js > out.log 2>&1 &;
  `;
  await execa('ssh', [`root@${ip}`, sshCommands]);

  // Step 4: Optional - Open ports (if using firewall)
  // await execa('doctl', ['compute', 'firewall', 'create', ...]);

  console.log(`✓ Orchestrator online at http://${ip}:3000`); // Assuming port 3000 from skeleton
}

export async function bootstrapOrchestratorLocal() {
  console.log(chalk.blue('🚀 Bootstrapping Chisel orchestrator locally...'));
  console.log(chalk.gray('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

  // Step 1: Install dependencies
  console.log(chalk.yellow('📦 Installing dependencies...'));
  await execa('npm', ['install'], { cwd: './apps/orchestrator' });
  console.log(chalk.green('✅ Dependencies installed successfully'));

  // Step 2: Build the orchestrator
  console.log(chalk.yellow('🔨 Building orchestrator...'));
  await execa('npm', ['run', 'build'], { cwd: './apps/orchestrator' });
  console.log(chalk.green('✅ Build completed successfully'));

  // Step 3: Start the orchestrator
  console.log(chalk.yellow('🚀 Starting orchestrator...'));
  console.log(chalk.gray('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

  const orchestratorUrl = 'http://localhost:3001';
  console.log(chalk.cyan.bold('🎯 Orchestrator is starting up...'));
  console.log(chalk.cyan(`📡 Server will be available at: ${chalk.underline(orchestratorUrl)}`));
  console.log(chalk.gray('💡 Press Ctrl+C to stop the server'));
  console.log(chalk.gray('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));

  await execa('node', ['dist/index.js'], {
    cwd: './apps/orchestrator',
    stdio: 'inherit' // Show output in real-time
  });
} 