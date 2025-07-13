import { ProfilerType } from '../enums';

export interface JobDispatchRequest {
  jobId: string;
  profiler: ProfilerType;
  command: string;
  flags: Record<string, any>;
  imageTarballPath?: string;
}

export interface JobDispatchResponse {
  acknowledged: boolean;
  message?: string;
}

