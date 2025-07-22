#pragma once
#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

// #ifdef __NVCC__
#include <stdio.h>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(Callable)                                                                                  \
    do {                                                                                                      \
        cudaError_t error = Callable;                                                                         \
        if (error != cudaSuccess) {                                                                           \
            fprintf(stderr, "[%s %s][%s:%d][%s][CUDA ERROR: %s, %s]", __DATE__, __TIME__, __FILE__, __LINE__, \
                    __func__, cudaGetErrorName(error), cudaGetErrorString(error));                            \
        }                                                                                                     \
    } while (0)
// #endif  // __NVCC__

#endif  // !CUDA_UTILS_CUH
