#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Returns 1 if a GPU is available, else 0
int kandc_gpu_available(void);

// Returns a static string with the GPU name (may be empty on some platforms)
const char* kandc_gpu_name(void);

#ifdef __cplusplus
}
#endif


