interface JobSpec {
    profiler: string;
    script: string;
}
declare const hello: () => string;

export { type JobSpec, hello };
