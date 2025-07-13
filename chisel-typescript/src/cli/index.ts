#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import { configure } from './commands/configure';

const program = new Command();

program
  .name('chisel')
  .description('GPU kernel development CLI tool')
  .version('0.1.0');

program
  .command('configure')
  .description('Configure Chisel with your DigitalOcean API token')
  .option('-t, --token <token>', 'API token (will prompt if not provided)')
  .action(async (options) => {
    try {
      const exitCode = await configure(options.token);
      process.exit(exitCode);
    } catch (error) {
      console.error(chalk.red('An unexpected error occurred:'), error);
      process.exit(1);
    }
  });

// Parse command line arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}