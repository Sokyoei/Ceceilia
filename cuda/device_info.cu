#include <iostream>

#include <cuda_runtime.h>

#include "Ahri.cuh"

int main(int argc, char const* argv[]) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "CUDA Device: " << device_count << '\n';

    if (device_count > 0) {
        int device_id = 0;
        cudaDeviceProp device_prop;
        CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
        std::cout << device_prop.name << '\n';

        int attr;
        CUDA_CHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrL2CacheSize, device_id));
        std::cout << "L2 cache size: " << attr / 1024 << " KB" << std::endl;
    }

    return 0;
}
