import { Command } from 'commander';
import { configureCommand } from './commands/configure';
// import { runNcu } from '@chisel/profiler-hooks'; // TODO: Implement
import { bootstrapOrchestrator, bootstrapOrchestratorLocal } from './commands/up';

const program = new Command();

program
  .name('chisel')
  .description('Chisel CLI for GPU profiling')
  .version('0.1.0');

program
  .command('configure')
  .description('Configure chisel settings (Digital Ocean API key)')
  .option('--show', 'Display current configuration with masked sensitive values')
  .action(async (options) => {
    await configureCommand(options);
  });

program
  .command('ncu')
  .option('--metrics <metrics>', 'NVIDIA metrics to collect')
  .option('--script <path>', 'Python script to run')
  .action((options: { metrics?: string; script?: string; }) => {
    console.log('Running NCU with options:', options);
    // TODO: Implement runNcu
  });

program
  .command('torchprof')
  .option('--trace-output <file>')
  .option('--script <path>')
  .action((options: { traceOutput?: string; script?: string; }) => {
    console.log('Running torchprof with options:', options);
    // TODO: Implement runTorchprof
  });

program
  .command('up')
  .description('Bootstrap the orchestrator (local or DigitalOcean)')
  .option('--local', 'Run orchestrator locally (default)', true)
  .option('--cloud', 'Deploy to DigitalOcean cloud')
  .action(async (options: { local?: boolean; cloud?: boolean; }) => {
    if (options.cloud) {
      await bootstrapOrchestrator();
    } else {
      await bootstrapOrchestratorLocal();
    }
  });

program.parse(); 