declare enum ProfilerType {
    NCU = "ncu",
    NSYS = "nsys",
    TORCHPROF = "torchprof",
    ROCPROF = "rocprof"
}
declare enum GPUVendor {
    NVIDIA = "nvidia",
    AMD = "amd"
}
declare enum GPUModel {
    H100 = "h100",
    MI300X = "mi300x"
}

interface JobSubmitRequest {
    profiler: ProfilerType;
    command: string;
    gpu: GPUModel;
    flags: Record<string, any>;
    imageTarballPath?: string;
}
interface JobSubmitResponse {
    jobId: string;
    status: 'queued' | 'running' | 'failed';
    message?: string;
}

interface JobDispatchRequest {
    jobId: string;
    profiler: ProfilerType;
    command: string;
    flags: Record<string, any>;
    imageTarballPath?: string;
}
interface JobDispatchResponse {
    acknowledged: boolean;
    message?: string;
}

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

declare const hello: () => string;

export { type ChiselConfig, type ConfigService, GPUModel, GPUVendor, type JobDispatchRequest, type JobDispatchResponse, type JobSubmitRequest, type JobSubmitResponse, ProfilerType, hello };
