import chalk from 'chalk';
import { ConfigManager } from '../../config';
import { DigitalOceanService } from '../../services/digitalocean';
import { confirmUpdate, promptForApiToken } from '../../utils/prompts';

export async function configure(token?: string): Promise<number> {
  const configManager = new ConfigManager();
  
  try {
    let apiToken: string;
    const existingToken = await configManager.getApiKey();
    
    if (token) {
      apiToken = token;
    } else if (existingToken) {
      console.log(chalk.green('Found existing DigitalOcean API token.'));
      const shouldUpdate = await confirmUpdate();
      
      if (!shouldUpdate) {
        console.log(chalk.green('✓ Keeping existing configuration'));
        return 0;
      }
      
      apiToken = await promptForApiToken();
    } else {
      // No existing token
      console.log(chalk.yellow('No DigitalOcean API token found.'));
      console.log('\nTo get your API token:');
      console.log('1. Go to: https://cloud.digitalocean.com/account/api/tokens');
      console.log('2. Generate a new token with read and write access');
      console.log('3. Copy the token (you won\'t be able to see it again)\n');
      
      apiToken = await promptForApiToken();
    }
    
    // Validate the token
    console.log(chalk.cyan('\nValidating API token...'));
    
    const doService = new DigitalOceanService(apiToken);
    const { valid, accountInfo } = await doService.validateToken();
    
    if (valid && accountInfo) {
      // Save the validated token
      await configManager.setApiKey(apiToken);
      
      console.log(chalk.green('✓ Token validated successfully!'));
      console.log(chalk.green('✓ Configuration saved!'));
      console.log(chalk.green('✓ Chisel is now configured and ready to use!'));
      
      console.log(chalk.cyan('\nAccount Information:'));
      console.log(`  Email: ${accountInfo.email}`);
      console.log(`  Status: ${accountInfo.status}`);
      console.log(`  Droplet Limit: ${accountInfo.droplet_limit}`);
      console.log(`\n`);
      
      return 0;
    } else {
      console.log(chalk.red('\n✗ Invalid API token. Please check your token and try again.'));
      return 1;
    }
    
  } catch (error) {
    console.log(chalk.red(`\nError: ${error instanceof Error ? error.message : 'Unknown error'}`));
    console.log(chalk.yellow('Please ensure you have a valid DigitalOcean API token with read and write permissions.'));
    return 1;
  }
}