import { GPUModel, ProfilerType } from '../enums';


export interface JobSubmitRequest {
  profiler: ProfilerType;
  command: string;
  gpu: GPUModel;
  /* TODO: Add count of GPUs */
  flags: Record<string, any>;
  imageTarballPath?: string;
}

export interface JobSubmitResponse {
  jobId: string;
  status: 'queued' | 'running' | 'failed';
  message?: string;
}
