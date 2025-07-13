interface ChiselConfig {
    version: string;
    digitalOcean?: {
        apiKey: string;
    };
}
interface ConfigService {
    load(): Promise<ChiselConfig | null>;
    save(config: ChiselConfig): Promise<void>;
    getConfigPath(): string;
}

declare class ConfigServiceImpl implements ConfigService {
    private configPath;
    constructor();
    getConfigPath(): string;
    load(): Promise<ChiselConfig | null>;
    save(config: ChiselConfig): Promise<void>;
}
declare const configService: ConfigServiceImpl;

export { ConfigServiceImpl, configService };
