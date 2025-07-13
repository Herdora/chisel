import { createApiClient } from 'dots-wrapper';
import { configService } from './config.service';

export interface DigitalOceanAccount {
  uuid: string;
  email: string;
  email_verified: boolean;
  status: string;
  status_message: string;
}

export interface ApiValidationResult {
  valid: boolean;
  account?: DigitalOceanAccount;
  error?: string;
}

export class DigitalOceanService {
  private apiClient: any;

  constructor(apiKey: string) {
    this.apiClient = createApiClient({ token: apiKey });
  }

  async validateApiKey(): Promise<ApiValidationResult> {
    try {
      const response = await this.apiClient.account.getAccount();
      
      if (response.data?.account) {
        return {
          valid: true,
          account: response.data.account
        };
      } else {
        return {
          valid: false,
          error: 'Invalid response format from Digital Ocean API'
        };
      }
    } catch (error: any) {
      let errorMessage = 'Unknown error occurred';
      
      if (error.response?.status === 401) {
        errorMessage = 'Invalid API key - authentication failed';
      } else if (error.response?.status === 403) {
        errorMessage = 'API key does not have required permissions';
      } else if (error.response?.status === 429) {
        errorMessage = 'Rate limit exceeded - please try again later';
      } else if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
        errorMessage = 'Network error - check your internet connection';
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      return {
        valid: false,
        error: errorMessage
      };
    }
  }

  static async createFromConfig(): Promise<DigitalOceanService | null> {
    const config = await configService.load();
    
    if (!config?.digitalOcean?.apiKey) {
      return null;
    }
    
    return new DigitalOceanService(config.digitalOcean.apiKey);
  }
}

export const digitalOceanService = {
  create: (apiKey: string) => new DigitalOceanService(apiKey),
  createFromConfig: () => DigitalOceanService.createFromConfig()
};