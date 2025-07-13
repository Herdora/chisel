import { promises as fs } from 'fs';
import * as path from 'path';
import * as os from 'os';
import { ChiselConfig, ConfigService } from '@chisel/core';

export class ConfigServiceImpl implements ConfigService {
  private configPath: string;

  constructor() {
    this.configPath = path.join(os.homedir(), '.chisel');
  }

  getConfigPath(): string {
    return this.configPath;
  }

  async load(): Promise<ChiselConfig | null> {
    try {
      const data = await fs.readFile(this.configPath, 'utf-8');
      return JSON.parse(data) as ChiselConfig;
    } catch (error: any) {
      if (error.code === 'ENOENT') {
        // File doesn't exist yet
        return null;
      }
      // Invalid JSON or other error
      throw new Error(`Failed to load config: ${error.message}`);
    }
  }

  async save(config: ChiselConfig): Promise<void> {
    try {
      const data = JSON.stringify(config, null, 2);
      await fs.writeFile(this.configPath, data, {
        encoding: 'utf-8',
        mode: 0o600 // Read/write for owner only
      });
    } catch (error: any) {
      throw new Error(`Failed to save config: ${error.message}`);
    }
  }
}

export const configService = new ConfigServiceImpl();