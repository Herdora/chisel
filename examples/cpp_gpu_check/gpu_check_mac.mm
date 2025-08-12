#import <Metal/Metal.h>
#include <string>
#include "gpu_check.h"

int kandc_gpu_available(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device ? 1 : 0;
    }
}

const char* kandc_gpu_name(void) {
    @autoreleasepool {
        static std::string name;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device) {
            name = [[device name] UTF8String];
            return name.c_str();
        }
        return "";
    }
}


