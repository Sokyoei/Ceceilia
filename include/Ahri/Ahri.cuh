#pragma once
#ifndef AHRI_CUH
#define AHRI_CUH

#include <stdio.h>

#include "Ahri.hpp"

#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>

namespace Ahri::CUDA {
#define CUDA_CHECK(Callable)                                                                                  \
    do {                                                                                                      \
        cudaError_t error = Callable;                                                                         \
        if (error != cudaSuccess) {                                                                           \
            fprintf(stderr, "[%s %s][%s:%d][%s][CUDA ERROR: %s, %s]", __DATE__, __TIME__, __FILE__, __LINE__, \
                    __func__, cudaGetErrorName(error), cudaGetErrorString(error));                            \
        }                                                                                                     \
    } while (0)
}  // namespace Ahri::CUDA
#endif  // __has_include(<cuda_runtime.h>)

#endif  // !AHRI_CUH
