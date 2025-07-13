import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { ChiselConfig, ConfigOptions } from './types';

export class ConfigManager {
  private configPath: string;

  constructor(options: ConfigOptions = {}) {
    this.configPath = options.configPath || path.join(os.homedir(), '.chisel');
  }

  async load(): Promise<ChiselConfig> {
    try {
      const data = await fs.readFile(this.configPath, 'utf-8');
      return JSON.parse(data);
    } catch (error) {
      if ((error as any).code === 'ENOENT') {
        return {};
      }
      throw error;
    }
  }

  async save(config: ChiselConfig): Promise<void> {
    const data = JSON.stringify(config, null, 2);
    await fs.writeFile(this.configPath, data, 'utf-8');
  }

  async getApiKey(): Promise<string | undefined> {
    const config = await this.load();
    return config.digitalOcean?.apiKey;
  }

  async setApiKey(apiKey: string): Promise<void> {
    const config = await this.load();
    config.digitalOcean = { apiKey };
    await this.save(config);
  }

  async hasApiKey(): Promise<boolean> {
    const apiKey = await this.getApiKey();
    return !!apiKey;
  }
}