#ifndef SOKYOEI_CUH
#define SOKYOEI_CUH

#ifdef __NVCC__
#include <stdio.h>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(Callable)                                                                                        \
    {                                                                                                               \
        cudaError_t error = Callable;                                                                               \
        if (error != cudaSuccess) {                                                                                 \
            fprintf(stderr, "[%s %s][%s:%d][%s][CUDA ERROR: %s]", __DATE__, __TIME__, __FILE__, __LINE__, __func__, \
                    error);                                                                                         \
        }                                                                                                           \
    }                                                                                                               \
    while (0)
#endif  // __NVCC__

#endif  // !SOKYOEI_CUH
