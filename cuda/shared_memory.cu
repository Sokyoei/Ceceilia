/**
 * @details
 * CUDA shared memory
 * CUDA 共享内存
 */

#include <iostream>

#include <cuda_runtime.h>

#include "Ahri/Ahri.cuh"

namespace Ahri {
// CUDA核函数：使用共享内存进行向量加法
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    // 定义共享内存数组
    __shared__ float s_a[256];
    __shared__ float s_b[256];

    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据从全局内存加载到共享内存
    if (idx < n) {
        s_a[threadIdx.x] = a[idx];
        s_b[threadIdx.x] = b[idx];
    }

    // 同步线程，确保所有线程都已将数据加载到共享内存
    __syncthreads();

    // 在共享内存中进行加法运算
    if (idx < n) {
        c[idx] = s_a[threadIdx.x] + s_b[threadIdx.x];
    }
}
}  // namespace Ahri

int main(int argc, char const* argv[]) {
    // __shared__ float s_static[256];       // 静态声明
    // extern __shared__ float s_dynamic[];  // 动态声明

    const int n = 1024;
    const int size = n * sizeof(float);

    // 主机端数组
    float *h_a, *h_b, *h_c;
    h_a = new float[n];
    h_b = new float[n];
    h_c = new float[n];

    // 初始化主机端数组
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // 设备端数组
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 定义线程块和网格的大小
    dim3 blockSize(256);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x);

    // 调用CUDA核函数
    Ahri::vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // 检查核函数调用是否出错
    CUDA_CHECK(cudaGetLastError());

    // 将结果从设备复制到主机
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 验证结果
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Error at index " << i << ": expected " << h_a[i] + h_b[i] << ", got " << h_c[i] << std::endl;
            break;
        }
    }
    std::cout << "Vector addition completed successfully." << std::endl;

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // 释放主机内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
