import inquirer from 'inquirer';
import chalk from 'chalk';
import ora from 'ora';
import { ChiselConfig } from '@chisel/core';
import { configService } from '../services/config.service';
import { digitalOceanService } from '../services/digitalocean.service';

export async function configureCommand(options: { show?: boolean }) {
  if (options.show) {
    await showConfiguration();
  } else {
    await interactiveConfigure();
  }
}

async function showConfiguration() {
  try {
    const config = await configService.load();
    
    if (!config) {
      console.log(chalk.yellow('No configuration found. Run `chisel configure` to set up.'));
      return;
    }
    
    console.log(chalk.bold('\nChisel Configuration:'));
    console.log(chalk.gray('─'.repeat(40)));
    console.log(`Version: ${config.version}`);
    
    if (config.digitalOcean?.apiKey) {
      const maskedKey = maskApiKey(config.digitalOcean.apiKey);
      console.log(`Digital Ocean API Key: ${maskedKey}`);
    }
    
    console.log(chalk.gray('─'.repeat(40)));
    console.log(chalk.gray(`\nConfig location: ${configService.getConfigPath()}`));
  } catch (error: any) {
    console.error(chalk.red('Error loading configuration:'), error.message);
    process.exit(1);
  }
}

async function interactiveConfigure() {
  console.log(chalk.bold('\nChisel Configuration Setup'));
  console.log(chalk.gray('─'.repeat(50)));
  console.log(chalk.gray('This will configure your Digital Ocean API key for chisel.'));
  console.log(chalk.gray('Your API key will be validated and stored securely.\n'));
  
  // Load existing config if any
  let existingConfig: ChiselConfig | null = null;
  try {
    existingConfig = await configService.load();
  } catch (error) {
    // Ignore errors, we'll create a new config
  }
  
  // Show helpful info for first-time setup
  if (!existingConfig) {
    console.log(chalk.blue('ℹ First time setup detected'));
    console.log(chalk.gray('You can get your API key from: https://cloud.digitalocean.com/account/api/tokens\n'));
  }
  
  // Prompt for Digital Ocean API key
  const answers = await inquirer.prompt([
    {
      type: 'password',
      name: 'doApiKey',
      message: 'Digital Ocean API key (dop_v1_...):',
      mask: '*',
      default: existingConfig?.digitalOcean?.apiKey,
      validate: (input: string) => {
        if (!input || input.trim() === '') {
          return 'API key is required';
        }
        
        const trimmed = input.trim();
        if (!isValidDigitalOceanApiKey(trimmed)) {
          return 'Invalid Digital Ocean API key format. Expected format: dop_v1_...';
        }
        
        return true;
      }
    }
  ]);
  
  const apiKey = answers.doApiKey.trim();
  
  // Validate API key with Digital Ocean
  const spinner = ora('Validating API key with Digital Ocean...').start();
  
  const doService = digitalOceanService.create(apiKey);
  const validationResult = await doService.validateApiKey();
  
  if (!validationResult.valid) {
    spinner.fail('API key validation failed');
    console.error(chalk.red(`\n${validationResult.error}`));
    console.error(chalk.gray('\nPlease check your API key and try again.'));
    console.error(chalk.gray('You can generate a new API key at: https://cloud.digitalocean.com/account/api/tokens'));
    process.exit(1);
  }
  
  // Show account information
  spinner.succeed('API key validated successfully!');
  if (validationResult.account) {
    console.log(chalk.gray(`Account: ${validationResult.account.email}`));
    console.log(chalk.gray(`Status: ${validationResult.account.status}`));
    if (validationResult.account.email_verified) {
      console.log(chalk.green('✓ Email verified'));
    } else {
      console.log(chalk.yellow('⚠ Email not verified'));
    }
  }
  
  // Create new config
  const config: ChiselConfig = {
    version: '1.0',
    digitalOcean: {
      apiKey: apiKey
    }
  };
  
  // Save config
  try {
    await configService.save(config);
    console.log(chalk.green('\n✓ Configuration saved successfully!'));
    console.log(chalk.gray(`Config location: ${configService.getConfigPath()}`));
  } catch (error: any) {
    console.error(chalk.red('\nError saving configuration:'), error.message);
    process.exit(1);
  }
}

function maskApiKey(apiKey: string): string {
  if (apiKey.length <= 8) {
    return '****';
  }
  return `${apiKey.substring(0, 4)}...${apiKey.substring(apiKey.length - 4)}`;
}

function isValidDigitalOceanApiKey(apiKey: string): boolean {
  // Digital Ocean API keys start with 'dop_v1_' followed by 64 hexadecimal characters
  const doApiKeyPattern = /^dop_v1_[a-f0-9]{64}$/i;
  return doApiKeyPattern.test(apiKey);
}