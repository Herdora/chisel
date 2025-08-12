#include <dlfcn.h>
#include <fstream>
#include <string>
#include "gpu_check.h"

using cudaGetDeviceCount_t = int (*)(int*);

static bool cuda_available_runtime() {
    void* lib = dlopen("libcuda.so", RTLD_LAZY);
    if (!lib) lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!lib) return false;
    auto fn = (cudaGetDeviceCount_t)dlsym(lib, "cudaGetDeviceCount");
    bool ok = false;
    if (fn) {
        int count = 0;
        ok = (fn(&count) == 0 && count > 0);
    }
    dlclose(lib);
    return ok;
}

int kandc_gpu_available(void) {
#ifdef __APPLE__
    // Use the macOS implementation instead
    return 0;
#else
    if (cuda_available_runtime()) return 1;
    std::ifstream nvidia_ver("/proc/driver/nvidia/version");
    return nvidia_ver.good() ? 1 : 0;
#endif
}

const char* kandc_gpu_name(void) {
#ifdef __APPLE__
    return "";
#else
    static std::string name;
    name = cuda_available_runtime() ? "CUDA-capable GPU" : "";
    return name.c_str();
#endif
}


