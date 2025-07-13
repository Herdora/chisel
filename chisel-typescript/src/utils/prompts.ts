import inquirer from 'inquirer';

export async function confirmUpdate(): Promise<boolean> {
  const { shouldUpdate } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'shouldUpdate',
      message: 'Found existing DigitalOcean API token. Do you want to update it?',
      default: false,
    },
  ]);
  
  return shouldUpdate;
}

export async function promptForApiToken(): Promise<string> {
  const { token } = await inquirer.prompt([
    {
      type: 'password',
      name: 'token',
      message: 'Enter your DigitalOcean API token:',
      mask: '*',
      validate: (input: string) => {
        if (!input || input.trim().length === 0) {
          return 'API token is required';
        }
        return true;
      },
    },
  ]);
  
  return token.trim();
}