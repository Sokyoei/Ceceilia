#include <iostream>

#include <cuda_runtime.h>
#include <fmt/core.h>

#include "Ahri/Ahri.cuh"

int main(int argc, char const* argv[]) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    fmt::println("Find CUDA Device: {}", device_count);

    if (device_count > 0) {
        int device_id = 0;
        cudaDeviceProp device_prop;
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        fmt::println("NVIDIA GPU name: {}", device_prop.name);

        int attr;
        CUDA_CHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrL2CacheSize, device_id));
        fmt::println("L2 cache size: {}KB", attr / 1024);
    }

    return 0;
}
