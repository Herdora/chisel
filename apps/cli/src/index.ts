import { Command } from 'commander';
// import { runNcu } from '@chisel/profiler-hooks'; // TODO: Implement

const program = new Command();

program
  .command('ncu')
  .option('--metrics <metrics>', 'NVIDIA metrics to collect')
  .option('--script <path>', 'Python script to run')
  .action((options) => {
    console.log('Running NCU with options:', options);
    // TODO: Implement runNcu
  });

program
  .command('torchprof')
  .option('--trace-output <file>')
  .option('--script <path>')
  .action((options) => {
    console.log('Running torchprof with options:', options);
    // TODO: Implement runTorchprof
  });

program.parse(); 