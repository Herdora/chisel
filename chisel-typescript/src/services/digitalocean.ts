import { createApiClient } from 'dots-wrapper';

export interface AccountInfo {
  email: string;
  status: string;
  droplet_limit: number;
  floating_ip_limit: number;
  uuid: string;
  email_verified: boolean;
  status_message: string;
  team?: {
    uuid: string;
    name: string;
  };
}

export class DigitalOceanService {
  private client: any;

  constructor(apiToken: string) {
    this.client = createApiClient({ token: apiToken });
  }

  async validateToken(): Promise<{ valid: boolean; accountInfo?: AccountInfo }> {
    try {
      const response = await this.client.account.getAccount();
      
      if (response.data && response.data.account) {
        return {
          valid: true,
          accountInfo: response.data.account as AccountInfo
        };
      }
      
      return { valid: false };
    } catch (error) {
      return { valid: false };
    }
  }

  async getAccountInfo(): Promise<AccountInfo | null> {
    try {
      const response = await this.client.account.getAccount();
      return response.data?.account || null;
    } catch (error) {
      console.error('Error fetching account info:', error);
      return null;
    }
  }
}