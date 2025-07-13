export interface ChiselConfig {
  version: string;
  digitalOcean?: {
    apiKey: string;
  };
}

export interface ConfigService {
  load(): Promise<ChiselConfig | null>;
  save(config: ChiselConfig): Promise<void>;
  getConfigPath(): string;
}