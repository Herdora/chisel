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

interface JobSpec {
    profiler: string;
    script: string;
}
declare const hello: () => string;

export { type ChiselConfig, type ConfigService, type JobSpec, hello };
